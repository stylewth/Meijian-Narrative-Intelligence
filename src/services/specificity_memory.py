from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas import (
    CandidateSpecificityHistory,
    ConfirmedSpecificityPattern,
    ConfirmedSpecificityPatterns,
    DecisionCallRequestManifest,
    DecisionRunStage,
    NarrativeCandidate,
    SpecificityAuditBatch,
    SpecificityCandidateRevision,
    SpecificityOutcome,
    SpecificityRevisionBatch,
    SpecificityRound,
    SpecificitySession,
    SpecificityAuditType,
)
from src.services.decision_run_store import DecisionRunStore, FrozenCall
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes
from src.services.brand_specificity import apply_specificity_revision


class SpecificityMemoryError(ValueError):
    pass


_MEMORY_SCHEMA_VERSION = "specificity-memory-v1"
_AUDIT_STAGE = DecisionRunStage.AWAITING_SPECIFICITY_AUDIT
_REVISION_STAGE = DecisionRunStage.SPECIFICITY_REVISION


def _strict_json_model(value: Any, model: type[Any]) -> Any:
    try:
        data = value if isinstance(value, (bytes, bytearray, memoryview)) else canonical_json_bytes(value)
        return model.model_validate_json(data, strict=True)
    except (TypeError, ValueError, ValidationError) as exc:
        raise SpecificityMemoryError(f"invalid {model.__name__}") from exc


def _read_canonical(path: Path, label: str) -> bytes:
    if not isinstance(path, Path):
        raise SpecificityMemoryError(f"{label} path must be a Path")
    if path.is_symlink() or not path.is_file():
        raise SpecificityMemoryError(f"{label} must be a regular file")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecificityMemoryError(f"invalid {label} JSON") from exc
    if canonical_json_bytes(value) != data:
        raise SpecificityMemoryError(f"{label} must be canonical JSON")
    return data


def _write_atomic(path: Path, data: bytes, *, allow_replace: bool) -> None:
    if path.is_symlink():
        raise SpecificityMemoryError(f"{path} must not be a symlink")
    if path.exists() and not allow_replace:
        raise SpecificityMemoryError(f"{path} already exists")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise SpecificityMemoryError(f"{path.parent} must be a regular directory")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_calls(store: DecisionRunStore) -> list[FrozenCall]:
    if not isinstance(store, DecisionRunStore):
        raise TypeError("store must be a DecisionRunStore")
    calls_path = store.root / "calls"
    if calls_path.is_symlink() or not calls_path.is_dir():
        raise SpecificityMemoryError("calls must be a regular directory")
    try:
        store._validate_calls(calls_path, store.manifest.decision_run_id)
    except Exception as exc:
        raise SpecificityMemoryError("call ledger validation failed") from exc
    entries = sorted(calls_path.iterdir(), key=lambda path: int(path.name))
    try:
        calls: list[FrozenCall] = []
        for entry in entries:
            request = _strict_json_model(
                (entry / "request_manifest.json").read_bytes(),
                DecisionCallRequestManifest,
            )
            calls.append(store.load_call(request.call_id))
        return calls
    except Exception as exc:
        raise SpecificityMemoryError("call ledger could not be loaded") from exc


def _response_model(call: FrozenCall) -> tuple[SpecificityAuditBatch | SpecificityRevisionBatch, type[Any]]:
    if call.response is None:
        raise SpecificityMemoryError("specificity call is missing a frozen response")
    request = call.manifest
    response = call.response.manifest
    if request.stage is _AUDIT_STAGE:
        model = SpecificityAuditBatch
        expected_name = "SpecificityAuditBatch"
    elif request.stage is _REVISION_STAGE:
        model = SpecificityRevisionBatch
        expected_name = "SpecificityRevisionBatch"
    else:
        raise SpecificityMemoryError("not a specificity call")
    if response.response_model_name != expected_name:
        raise SpecificityMemoryError("response model name does not match stage")
    expected_schema_sha = sha256_bytes(canonical_json_bytes(model.model_json_schema()))
    if request.response_schema_sha256 != expected_schema_sha:
        raise SpecificityMemoryError("response schema hash does not match model")
    value = _strict_json_model(response.response_payload, model)
    if value.model_dump(mode="json") != response.response_payload:
        raise SpecificityMemoryError("response payload is not the exact frozen schema")
    return value, model


def _candidates_from_request(request: DecisionCallRequestManifest) -> dict[str, NarrativeCandidate]:
    try:
        payload = json.loads(request.user_prompt)
    except json.JSONDecodeError as exc:
        raise SpecificityMemoryError("specificity request user_prompt is not JSON") from exc
    if canonical_json_bytes(payload).decode("utf-8") != request.user_prompt:
        raise SpecificityMemoryError("specificity request user_prompt is not canonical JSON")
    if not isinstance(payload, dict):
        raise SpecificityMemoryError("specificity request payload must be an object")
    if isinstance(payload.get("payload"), dict):
        candidates_payload = payload["payload"].get("candidates")
    else:
        candidates_payload = payload.get("candidates")
    if not isinstance(candidates_payload, list):
        raise SpecificityMemoryError("specificity request does not contain candidates")
    candidates: dict[str, NarrativeCandidate] = {}
    for item in candidates_payload:
        candidate = _strict_json_model(item, NarrativeCandidate)
        if candidate.candidate_id in candidates:
            raise SpecificityMemoryError("duplicate candidate in specificity request")
        candidates[candidate.candidate_id] = candidate
    if not candidates:
        raise SpecificityMemoryError("specificity request contains no candidates")
    return candidates


def _assemble_histories(
    store: DecisionRunStore,
    audit_calls: dict[int, tuple[FrozenCall, SpecificityAuditBatch]],
    revision_calls: dict[int, tuple[FrozenCall, SpecificityRevisionBatch]],
) -> tuple[list[CandidateSpecificityHistory], list[str], dict[str, str], dict[str, str]]:
    audit_indices = sorted(audit_calls)
    if audit_indices != list(range(1, len(audit_indices) + 1)) or len(audit_indices) > 3:
        raise SpecificityMemoryError("specificity audit rounds must be contiguous and at most three")
    if not audit_indices:
        raise SpecificityMemoryError("specificity session has no audit rounds")
    if any(index not in audit_calls for index in revision_calls):
        raise SpecificityMemoryError("revision exists without a matching audit")

    initial_candidates: dict[str, NarrativeCandidate] = {}
    all_candidate_ids: set[str] = set()
    for _, (call, audit) in audit_calls.items():
        if audit.run_id != store.manifest.decision_run_id:
            raise SpecificityMemoryError("audit run_id does not match store")
        request_candidates = _candidates_from_request(call.manifest)
        for candidate_id, candidate in request_candidates.items():
            initial_candidates.setdefault(candidate_id, candidate)
        candidate_ids = [candidate_audit.candidate_id for candidate_audit in audit.audits]
        if audit.candidate_ids != sorted(audit.candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
            raise SpecificityMemoryError("audit candidate IDs are not deterministic and unique")
        all_candidate_ids.update(candidate_ids)

    prompt_hashes: dict[str, str] = {}
    schema_hashes: dict[str, str] = {}
    for round_index in audit_indices:
        for call, _ in (audit_calls[round_index],):
            key = f"audit:{call.manifest.prompt_name}"
            if key in prompt_hashes and prompt_hashes[key] != call.manifest.prompt_sha256:
                raise SpecificityMemoryError("audit prompt hash changed across calls")
            prompt_hashes[key] = call.manifest.prompt_sha256
            schema_hashes["SpecificityAuditBatch"] = call.manifest.response_schema_sha256
    for round_index, (call, _) in revision_calls.items():
        key = f"revision:{call.manifest.prompt_name}"
        if key in prompt_hashes and prompt_hashes[key] != call.manifest.prompt_sha256:
            raise SpecificityMemoryError("revision prompt hash changed across calls")
        prompt_hashes[key] = call.manifest.prompt_sha256
        schema_hashes["SpecificityRevisionBatch"] = call.manifest.response_schema_sha256

    rounds_by_index: dict[int, SpecificityRound] = {}
    for round_index in audit_indices:
        audit_call, audit = audit_calls[round_index]
        revision_call, revision = revision_calls.get(round_index, (None, None))
        if revision_call is not None and revision is not None:
            if revision.run_id != store.manifest.decision_run_id or revision.round_index != round_index:
                raise SpecificityMemoryError("revision identity does not match audit round")
            audit_candidate_ids = set(audit.candidate_ids)
            revision_candidate_ids = {item.candidate_id for item in revision.revisions}
            if not revision_candidate_ids.issubset(audit_candidate_ids):
                raise SpecificityMemoryError(
                    "revision contains a candidate outside the audit candidate set"
                )
            rounds_by_index[round_index] = SpecificityRound(
                round_index=round_index,
                audit_call_sequence=audit_call.manifest.sequence,
                audit_response_sha256=audit_call.response.manifest.response_sha256,
                audit=audit,
                revision_call_sequence=revision_call.manifest.sequence,
                revision_response_sha256=revision_call.response.manifest.response_sha256,
                revision=revision,
            )
        else:
            rounds_by_index[round_index] = SpecificityRound(
                round_index=round_index,
                audit_call_sequence=audit_call.manifest.sequence,
                audit_response_sha256=audit_call.response.manifest.response_sha256,
                audit=audit,
            )

    histories: list[CandidateSpecificityHistory] = []
    final_candidate_ids = sorted(audit_calls[audit_indices[-1]][1].candidate_ids)
    for candidate_id in sorted(all_candidate_ids):
        if candidate_id not in initial_candidates:
            raise SpecificityMemoryError("candidate has no initial frozen snapshot")
        current_candidate = initial_candidates[candidate_id]
        current_version = 1
        history_rounds: list[SpecificityRound] = []
        unresolved: list[str] = []
        seen_candidate = False
        for round_index in audit_indices:
            round_record = rounds_by_index[round_index]
            audit_entry = next(
                (item for item in round_record.audit.audits if item.candidate_id == candidate_id),
                None,
            )
            if audit_entry is None:
                if seen_candidate:
                    break
                continue
            seen_candidate = True
            if round_record.audit.candidate_versions[candidate_id] != current_version:
                raise SpecificityMemoryError("audit candidate version does not follow prior revision")
            history_rounds.append(round_record)
            if round_record.revision is not None:
                revision = next(
                    (item for item in round_record.revision.revisions if item.candidate_id == candidate_id),
                    None,
                )
                if revision is None:
                    raise SpecificityMemoryError("audit candidate is missing its revision entry")
                if revision.from_version != current_version:
                    raise SpecificityMemoryError("revision from_version does not match current version")
                substantive_ids = {
                    finding.finding_id
                    for finding in audit_entry.findings
                    if finding.severity.value != "NOTE"
                }
                response_ids = {response.finding_id for response in revision.responses}
                if response_ids != substantive_ids:
                    raise SpecificityMemoryError("revision does not respond to every substantive finding")
                try:
                    current_candidate = apply_specificity_revision(current_candidate, revision)
                except Exception as exc:
                    raise SpecificityMemoryError("revision contract validation failed") from exc
                current_version = revision.to_version
                unresolved = list(revision.unresolved_finding_ids)
            else:
                unresolved = [
                    finding.finding_id
                    for finding in audit_entry.findings
                    if finding.severity.value != "NOTE"
                ]
        if not history_rounds:
            raise SpecificityMemoryError("candidate has no audit history")
        outcome = SpecificityOutcome.HUMAN_REVIEW if unresolved else SpecificityOutcome.READY
        histories.append(
            CandidateSpecificityHistory(
                candidate_id=candidate_id,
                initial_candidate=initial_candidates[candidate_id],
                rounds=history_rounds,
                current_candidate=current_candidate,
                current_version=current_version,
                outcome=outcome,
                unresolved_finding_ids=unresolved,
            )
        )
    return histories, final_candidate_ids, prompt_hashes, schema_hashes


def assemble_specificity_session(
    store: DecisionRunStore,
    confirmed_pattern_ids: list[str],
) -> SpecificitySession:
    if not isinstance(store, DecisionRunStore):
        raise TypeError("store must be a DecisionRunStore")
    if not isinstance(confirmed_pattern_ids, list) or any(
        not isinstance(pattern_id, str) or not pattern_id for pattern_id in confirmed_pattern_ids
    ):
        raise TypeError("confirmed_pattern_ids must be a list of non-empty strings")
    if len(confirmed_pattern_ids) != len(set(confirmed_pattern_ids)):
        raise SpecificityMemoryError("confirmed pattern IDs must be unique")

    audit_calls: dict[int, tuple[FrozenCall, SpecificityAuditBatch]] = {}
    revision_calls: dict[int, tuple[FrozenCall, SpecificityRevisionBatch]] = {}
    for call in _load_calls(store):
        if call.manifest.stage not in {_AUDIT_STAGE, _REVISION_STAGE}:
            continue
        parsed, _ = _response_model(call)
        round_index = parsed.round_index
        target = audit_calls if call.manifest.stage is _AUDIT_STAGE else revision_calls
        if round_index in target:
            raise SpecificityMemoryError("duplicate specificity call for one round")
        target[round_index] = (call, parsed)

    histories, final_candidate_ids, prompt_hashes, schema_hashes = _assemble_histories(
        store, audit_calls, revision_calls
    )
    frozen_calls = [call for call, _ in audit_calls.values()] + [
        call for call, _ in revision_calls.values()
    ]
    finalized_at = max(
        call.response.manifest.completed_at
        for call in frozen_calls
        if call.response is not None
    )
    session = SpecificitySession(
        run_id=store.manifest.decision_run_id,
        decision_model=store.manifest.decision_model,
        specificity_model=store.manifest.specificity_model,
        prompt_hashes=prompt_hashes,
        schema_hashes=schema_hashes,
        confirmed_pattern_ids=sorted(confirmed_pattern_ids),
        candidate_histories=histories,
        final_candidate_ids=final_candidate_ids,
        finalized_at=finalized_at,
    )
    session_path = store.root / "stages" / "specificity_session.json"
    if session_path.exists() or session_path.is_symlink():
        existing = _strict_json_model(_read_canonical(session_path, "specificity session"), SpecificitySession)
        if existing.model_dump(mode="json") != session.model_dump(mode="json"):
            raise SpecificityMemoryError("specificity session already exists with different content")
        return existing
    _write_atomic(session_path, canonical_json_bytes(session), allow_replace=False)
    return session


def load_confirmed_patterns(path: Path) -> ConfirmedSpecificityPatterns:
    data = _read_canonical(path, "confirmed patterns")
    memory = _strict_json_model(data, ConfirmedSpecificityPatterns)
    if any(pattern.confirmation_type not in {"HUMAN", "REAL_VALIDATION"} for pattern in memory.patterns):
        raise SpecificityMemoryError("confirmed memory accepts only HUMAN or REAL_VALIDATION")
    return memory


def select_confirmed_patterns(
    memory: ConfirmedSpecificityPatterns,
    audit_types: set[SpecificityAuditType],
    scope_tags: set[str],
) -> list[ConfirmedSpecificityPattern]:
    if not isinstance(memory, ConfirmedSpecificityPatterns):
        raise TypeError("memory must be ConfirmedSpecificityPatterns")
    if not isinstance(audit_types, set) or any(not isinstance(item, SpecificityAuditType) for item in audit_types):
        raise TypeError("audit_types must be a set of SpecificityAuditType")
    if not isinstance(scope_tags, set) or any(not isinstance(item, str) or not item for item in scope_tags):
        raise TypeError("scope_tags must be a set of non-empty strings")
    selected = [
        pattern
        for pattern in memory.patterns
        if pattern.audit_type in audit_types and scope_tags.issubset(set(pattern.scope_tags))
    ]
    return sorted(selected, key=lambda pattern: pattern.pattern_id)


def append_confirmed_pattern(
    path: Path,
    pattern: ConfirmedSpecificityPattern,
) -> str:
    if not isinstance(pattern, ConfirmedSpecificityPattern):
        raise TypeError("pattern must be a ConfirmedSpecificityPattern")
    if pattern.confirmation_type not in {"HUMAN", "REAL_VALIDATION"}:
        raise SpecificityMemoryError("MODEL_ONLY patterns cannot enter confirmed memory")
    if pattern.source_sha256 != pattern.source_sha256.lower() or len(pattern.source_sha256) != 64:
        raise SpecificityMemoryError("source_sha256 must be a lowercase SHA-256")
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise SpecificityMemoryError("confirmed pattern file must not be a symlink")
        memory = load_confirmed_patterns(path)
    else:
        memory = ConfirmedSpecificityPatterns(schema_version=_MEMORY_SCHEMA_VERSION, patterns=[])
    if any(existing.pattern_id == pattern.pattern_id for existing in memory.patterns):
        raise SpecificityMemoryError("pattern_id already exists")
    updated = ConfirmedSpecificityPatterns(
        schema_version=memory.schema_version,
        patterns=[*memory.patterns, pattern],
    )
    data = canonical_json_bytes(updated)
    _write_atomic(path, data, allow_replace=path.exists())
    return sha256_bytes(data)
