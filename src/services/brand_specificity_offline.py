from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas import (
    BrandFact,
    ConfirmedSpecificityPattern,
    DecisionModelProfile,
    DecisionRunStage,
    EvidenceAtom,
    ModelRuntime,
    NarrativeCandidate,
    OfflineSpecificityResult,
    OfflineSpecificityTask,
    FiveCandidateSharedInputV2,
    PublicNarrativeEvidence,
    SpecificityAuditBatch,
    SpecificityImportReceipt,
    SpecificityRound,
)
from src.services.decision_run_store import DecisionRunStore
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes


LUNA_SPECIFICITY_PROFILE = DecisionModelProfile(
    profile_id="codex-luna-specificity-max-v1",
    provider="codex",
    model_id="gpt-5.6-luna",
    runtime=ModelRuntime.CLOUD,
    response_format="json_schema",
    prompt_version="brand-specificity-v1",
    endpoint_profile="codex-offline",
    thinking_enabled=True,
    reasoning_effort="max",
    max_retries=0,
)

LUNA_SPECIFICITY_V2_PROFILE = DecisionModelProfile(
    profile_id="codex-luna-specificity-max-v2",
    provider="codex",
    model_id="gpt-5.6-luna",
    runtime=ModelRuntime.CLOUD,
    response_format="json_schema",
    prompt_version="brand-specificity-v2",
    endpoint_profile="codex-offline",
    thinking_enabled=True,
    reasoning_effort="max",
    max_retries=0,
)


class SpecificityImportError(ValueError):
    pass


_PAYLOAD_KEYS = frozenset(
    {
        "run_id",
        "round_index",
        "candidate_ids",
        "candidates",
        "evidence_atoms",
        "brand_facts",
        "brand_asset_status",
        "prior_rounds",
        "confirmed_patterns",
    }
)
_FORBIDDEN_PAYLOAD = re.compile(r"holdout|blind|checkpoint-|future[ _-]?delta", re.IGNORECASE)
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "brand_specificity_audit.md"
_V2_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "brand_specificity_audit_v2.md"
_V2_PAYLOAD_KEYS = frozenset(
    {
        "candidate",
        "comment_evidence",
        "public_evidence",
        "confirmed_memory",
        "prior_rounds",
        "round",
        "shared_input_sha256",
    }
)


def _profile_is_exact(profile: DecisionModelProfile) -> None:
    if profile.model_dump(mode="json") != LUNA_SPECIFICITY_PROFILE.model_dump(mode="json"):
        raise ValueError("Luna specificity profile is not exact")


def _prompt_bytes() -> bytes:
    if _PROMPT_PATH.is_symlink() or not _PROMPT_PATH.is_file():
        raise ValueError("specificity audit prompt must be a regular file")
    data = _PROMPT_PATH.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("specificity audit prompt must not contain a UTF-8 BOM")
    data.decode("utf-8")
    return data


def _v2_prompt_bytes() -> bytes:
    if _V2_PROMPT_PATH.is_symlink() or not _V2_PROMPT_PATH.is_file():
        raise ValueError("specificity v2 audit prompt must be a regular file")
    data = _V2_PROMPT_PATH.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError("specificity v2 audit prompt must not contain a UTF-8 BOM")
    data.decode("utf-8")
    return data


def _schema_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(SpecificityAuditBatch.model_json_schema()))


def _reject_invisible_payload(value: Any) -> None:
    serialized = canonical_json_bytes(value).decode("utf-8")
    if _FORBIDDEN_PAYLOAD.search(serialized):
        raise ValueError("offline specificity task/result contains forbidden blind or future data")


def _strict_json_model(value: Any, model: type[Any]) -> Any:
    return model.model_validate_json(canonical_json_bytes(value), strict=True)


def _candidate_version_map(
    candidate_ids: list[str], prior_rounds: list[SpecificityRound]
) -> dict[str, int]:
    versions = {candidate_id: 1 for candidate_id in candidate_ids}
    for specificity_round in prior_rounds:
        for candidate_id, version in specificity_round.audit.candidate_versions.items():
            if candidate_id in versions:
                versions[candidate_id] = version
        if specificity_round.revision is not None:
            for revision in specificity_round.revision.revisions:
                if revision.candidate_id in versions:
                    versions[revision.candidate_id] = revision.to_version
    return versions


def _validate_export_inputs(
    store: DecisionRunStore,
    round_index: int,
    candidates: list[NarrativeCandidate],
    evidence_atoms: list[EvidenceAtom],
    brand_facts: list[BrandFact],
    prior_rounds: list[SpecificityRound],
    confirmed_patterns: list[ConfirmedSpecificityPattern],
) -> None:
    if not isinstance(store, DecisionRunStore):
        raise TypeError("store must be a DecisionRunStore")
    if round_index not in {1, 2, 3}:
        raise ValueError("round_index must be 1..3")
    for value, model, label in (
        (candidates, NarrativeCandidate, "candidate"),
        (evidence_atoms, EvidenceAtom, "evidence atom"),
        (brand_facts, BrandFact, "brand fact"),
        (prior_rounds, SpecificityRound, "prior round"),
        (confirmed_patterns, ConfirmedSpecificityPattern, "confirmed pattern"),
    ):
        if not isinstance(value, list) or any(not isinstance(item, model) for item in value):
            raise TypeError(f"{label} inputs must be typed model lists")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidates must contain unique non-empty IDs")
    evidence_ids = [atom.evidence_id for atom in evidence_atoms]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence IDs must be unique")
    fact_ids = [fact.fact_id for fact in brand_facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise ValueError("brand fact IDs must be unique")
    expected_prior_rounds = list(range(1, round_index))
    if [item.round_index for item in prior_rounds] != expected_prior_rounds:
        raise ValueError("prior rounds must be the contiguous rounds before the requested round")


def _task_payload(
    store: DecisionRunStore,
    round_index: int,
    candidates: list[NarrativeCandidate],
    evidence_atoms: list[EvidenceAtom],
    brand_facts: list[BrandFact],
    prior_rounds: list[SpecificityRound],
    confirmed_patterns: list[ConfirmedSpecificityPattern],
) -> tuple[dict[str, Any], dict[str, int]]:
    ordered_candidates = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    candidate_ids = [candidate.candidate_id for candidate in ordered_candidates]
    candidate_versions = _candidate_version_map(candidate_ids, prior_rounds)
    payload: dict[str, Any] = {
        "run_id": store.manifest.decision_run_id,
        "round_index": round_index,
        "candidate_ids": candidate_ids,
        "candidates": [candidate.model_dump(mode="json") for candidate in ordered_candidates],
        "evidence_atoms": [atom.model_dump(mode="json") for atom in sorted(evidence_atoms, key=lambda item: item.evidence_id)],
        "brand_facts": [fact.model_dump(mode="json") for fact in sorted(brand_facts, key=lambda item: item.fact_id)],
        "brand_asset_status": (
            "VERIFIED_BRAND_ASSETS_PRESENT" if brand_facts else "NO_VERIFIED_BRAND_ASSETS"
        ),
        "prior_rounds": [item.model_dump(mode="json") for item in prior_rounds],
        "confirmed_patterns": [
            item.model_dump(mode="json")
            for item in sorted(confirmed_patterns, key=lambda item: item.pattern_id)
        ],
    }
    _reject_invisible_payload(payload)
    return payload, candidate_versions


def _request_user_prompt(task: OfflineSpecificityTask) -> str:
    return canonical_json_bytes(
        {
            "task_id": task.task_id,
            "run_id": task.run_id,
            "round_index": task.round_index,
            "input_sha256": task.input_sha256,
            "candidate_versions": task.candidate_versions,
            "payload": task.payload,
        }
    ).decode("utf-8")


def export_specificity_task(
    store: DecisionRunStore,
    round_index: int,
    candidates: list[NarrativeCandidate],
    evidence_atoms: list[EvidenceAtom],
    brand_facts: list[BrandFact],
    prior_rounds: list[SpecificityRound],
    confirmed_patterns: list[ConfirmedSpecificityPattern],
) -> OfflineSpecificityTask:
    _validate_export_inputs(
        store,
        round_index,
        candidates,
        evidence_atoms,
        brand_facts,
        prior_rounds,
        confirmed_patterns,
    )
    if store.manifest.specificity_model.model_dump(mode="json") != LUNA_SPECIFICITY_PROFILE.model_dump(mode="json"):
        raise ValueError("run manifest specificity model is not the Luna Max profile")
    payload, candidate_versions = _task_payload(
        store,
        round_index,
        candidates,
        evidence_atoms,
        brand_facts,
        prior_rounds,
        confirmed_patterns,
    )
    input_sha256 = sha256_bytes(canonical_json_bytes(payload))
    task = OfflineSpecificityTask(
        task_id=f"specificity-{store.manifest.decision_run_id}-{round_index}-{input_sha256[:16]}",
        run_id=store.manifest.decision_run_id,
        round_index=round_index,
        model_profile=LUNA_SPECIFICITY_PROFILE,
        prompt_sha256=sha256_bytes(_prompt_bytes()),
        schema_sha256=_schema_sha256(),
        input_sha256=input_sha256,
        candidate_versions=candidate_versions,
        payload=payload,
    )
    store.begin_offline_specificity_call(
        input_artifact_sha256=task.input_sha256,
        prompt_sha256=task.prompt_sha256,
        system_prompt=_prompt_bytes().decode("utf-8"),
        user_prompt=_request_user_prompt(task),
        response_schema_sha256=task.schema_sha256,
    )
    return task


def _write_new_atomic(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists")
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
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


def write_specificity_task(path: Path, task: OfflineSpecificityTask) -> None:
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not isinstance(task, OfflineSpecificityTask):
        raise TypeError("task must be an OfflineSpecificityTask")
    _write_new_atomic(path, canonical_json_bytes(task))


def _read_canonical_json(path: Path, label: str) -> bytes:
    if not isinstance(path, Path):
        raise SpecificityImportError(f"{label} path must be a Path")
    if path.is_symlink() or not path.is_file():
        raise SpecificityImportError(f"{label} must be a regular file")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpecificityImportError(f"invalid {label} JSON") from exc
    if canonical_json_bytes(value) != data:
        raise SpecificityImportError(f"{label} must be canonical JSON")
    return data


def _validate_task_payload(task: OfflineSpecificityTask) -> tuple[set[str], set[str]]:
    payload = task.payload
    if set(payload) != _PAYLOAD_KEYS:
        raise ValueError("specificity task payload fields are not exact")
    if payload["run_id"] != task.run_id or payload["round_index"] != task.round_index:
        raise ValueError("specificity task payload identity does not match task")
    candidate_ids = payload["candidate_ids"]
    if not isinstance(candidate_ids, list) or candidate_ids != sorted(candidate_ids):
        raise ValueError("candidate_ids must be sorted")
    if candidate_ids != list(task.candidate_versions):
        raise ValueError("candidate_ids must match candidate_versions")
    for candidate in payload["candidates"]:
        _strict_json_model(candidate, NarrativeCandidate)
    actual_candidate_ids = [candidate["candidate_id"] for candidate in payload["candidates"]]
    if actual_candidate_ids != candidate_ids:
        raise ValueError("candidate payloads must match candidate_ids")
    for atom in payload["evidence_atoms"]:
        _strict_json_model(atom, EvidenceAtom)
    for fact in payload["brand_facts"]:
        _strict_json_model(fact, BrandFact)
    for specificity_round in payload["prior_rounds"]:
        _strict_json_model(specificity_round, SpecificityRound)
    for pattern in payload["confirmed_patterns"]:
        _strict_json_model(pattern, ConfirmedSpecificityPattern)
    expected_status = "VERIFIED_BRAND_ASSETS_PRESENT" if payload["brand_facts"] else "NO_VERIFIED_BRAND_ASSETS"
    if payload["brand_asset_status"] != expected_status:
        raise ValueError("brand asset status does not describe the supplied facts")
    _reject_invisible_payload(payload)
    return (
        {atom["evidence_id"] for atom in payload["evidence_atoms"]},
        {fact["fact_id"] for fact in payload["brand_facts"]},
    )


def _validate_output_refs(
    output: SpecificityAuditBatch,
    task: OfflineSpecificityTask,
    evidence_ids: set[str],
    brand_fact_ids: set[str],
) -> None:
    expected_ids = list(task.candidate_versions)
    if output.run_id != task.run_id or output.round_index != task.round_index:
        raise ValueError("output run_id or round_index does not match task")
    if output.candidate_ids != expected_ids or output.candidate_versions != task.candidate_versions:
        raise ValueError("output candidates or versions do not match task")
    finding_ids: set[str] = set()
    for audit in output.audits:
        for finding in audit.findings:
            if finding.finding_id in finding_ids:
                raise ValueError("finding_id must be globally unique")
            finding_ids.add(finding.finding_id)
            if not set(finding.evidence_ids).issubset(evidence_ids):
                raise ValueError("finding references an unknown evidence ID")
            if not set(finding.brand_fact_ids).issubset(brand_fact_ids):
                raise ValueError("finding references an unknown BrandFact ID")
            if finding.suggested_patch is not None:
                if not set(finding.suggested_patch.evidence_ids).issubset(evidence_ids):
                    raise ValueError("patch references an unknown evidence ID")
                if not set(finding.suggested_patch.brand_fact_ids).issubset(brand_fact_ids):
                    raise ValueError("patch references an unknown BrandFact ID")
        if not set(audit.consumer_evidence).issubset(evidence_ids):
            raise ValueError("audit references an unknown consumer evidence ID")
        if not set(audit.brand_assets).issubset(brand_fact_ids):
            raise ValueError("audit references an unknown BrandFact ID")


def import_specificity_result(
    store: DecisionRunStore,
    task: OfflineSpecificityTask,
    result_path: Path,
) -> SpecificityImportReceipt:
    try:
        if not isinstance(store, DecisionRunStore):
            raise TypeError("store must be a DecisionRunStore")
        if not isinstance(task, OfflineSpecificityTask):
            raise TypeError("task must be an OfflineSpecificityTask")
        _profile_is_exact(task.model_profile)
        if store.manifest.specificity_model != LUNA_SPECIFICITY_PROFILE:
            raise ValueError("run manifest specificity model is not Luna Max")
        evidence_ids, brand_fact_ids = _validate_task_payload(task)
        prompt_sha256 = sha256_bytes(_prompt_bytes())
        schema_sha256 = _schema_sha256()
        if task.prompt_sha256 != prompt_sha256 or task.schema_sha256 != schema_sha256:
            raise ValueError("task prompt or schema hash does not match current contract")
        if sha256_bytes(canonical_json_bytes(task.payload)) != task.input_sha256:
            raise ValueError("task input SHA does not match payload")

        result_data = _read_canonical_json(result_path, "specificity result")
        try:
            result = OfflineSpecificityResult.model_validate_json(result_data, strict=True)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError("specificity result envelope is invalid") from exc
        if result.model_dump(mode="json") != json.loads(result_data.decode("utf-8")):
            raise ValueError("specificity result envelope is not canonical")
        if result.model_dump(mode="json")["task_id"] != task.task_id:
            raise ValueError("result task_id does not match task")
        for field in ("run_id", "round_index", "prompt_sha256", "schema_sha256", "input_sha256", "candidate_versions"):
            if getattr(result, field) != getattr(task, field):
                raise ValueError(f"result {field} does not match task")
        _profile_is_exact(result.model_profile)
        _reject_invisible_payload(result.output)
        try:
            output = _strict_json_model(result.output, SpecificityAuditBatch)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError("specificity output does not match SpecificityAuditBatch") from exc
        if output.model_dump(mode="json") != result.output:
            raise ValueError("specificity output must use the exact JSON Schema shape")
        _validate_output_refs(output, task, evidence_ids, brand_fact_ids)

        call = store.find_offline_specificity_call(task.task_id)
        if call.response is not None:
            raise ValueError("specificity call is already finished")
        request = call.manifest
        if request.stage is not DecisionRunStage.AWAITING_SPECIFICITY_AUDIT:
            raise ValueError("specificity call stage does not match")
        if request.input_artifact_sha256 != task.input_sha256:
            raise ValueError("request input SHA does not match task")
        if request.decision_model != task.model_profile:
            raise ValueError("request model profile does not match task")
        if request.prompt_version != task.model_profile.prompt_version:
            raise ValueError("request prompt version does not match task")
        if request.prompt_sha256 != task.prompt_sha256 or request.response_schema_sha256 != task.schema_sha256:
            raise ValueError("request prompt or schema hash does not match task")
        if request.user_prompt != _request_user_prompt(task):
            raise ValueError("request user payload does not match task")
        response_payload = output.model_dump(mode="json")
        response_sha256 = sha256_bytes(canonical_json_bytes(response_payload))
        from src.schemas import DecisionCallResponseManifest

        response = DecisionCallResponseManifest(
            call_id=request.call_id,
            request_sha256=sha256_bytes(canonical_json_bytes(request)),
            response_model_name="SpecificityAuditBatch",
            response_payload=response_payload,
            response_sha256=response_sha256,
            completed_at=datetime.now(timezone.utc),
        )
        store.finish_call(request.call_id, response)
        return SpecificityImportReceipt(
            task_id=task.task_id,
            call_sequence=request.sequence,
            response_sha256=response_sha256,
            imported_at=datetime.now(timezone.utc),
        )
    except SpecificityImportError:
        raise
    except Exception as exc:
        raise SpecificityImportError(str(exc)) from exc


def _profile_is_exact_v2(profile: DecisionModelProfile) -> None:
    if profile.model_dump(mode="json") != LUNA_SPECIFICITY_V2_PROFILE.model_dump(mode="json"):
        raise ValueError("Luna specificity v2 profile is not exact")


def _candidate_history_payload(
    candidate_id: str,
    prior_rounds: list[SpecificityRound],
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for specificity_round in prior_rounds:
        audits = [
            audit
            for audit in specificity_round.audit.audits
            if audit.candidate_id == candidate_id
        ]
        if len(audits) != 1:
            raise ValueError("each prior round must contain the candidate audit exactly once")
        revisions = []
        if specificity_round.revision is not None:
            revisions = [
                revision
                for revision in specificity_round.revision.revisions
                if revision.candidate_id == candidate_id
            ]
        if len(revisions) > 1:
            raise ValueError("each prior round may contain at most one candidate revision")
        history.append(
            {
                "round_index": specificity_round.round_index,
                "audit": audits[0].model_dump(mode="json"),
                "revision": revisions[0].model_dump(mode="json") if revisions else None,
            }
        )
    return history


def _validate_v2_export_inputs(
    *,
    run_id: str,
    candidate: NarrativeCandidate,
    comment_evidence: list[EvidenceAtom],
    shared_input: FiveCandidateSharedInputV2,
    confirmed_memory: list[ConfirmedSpecificityPattern],
    prior_rounds: list[SpecificityRound],
    round_index: int,
) -> None:
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be non-empty")
    if not isinstance(candidate, NarrativeCandidate):
        raise TypeError("candidate must be a NarrativeCandidate")
    if not isinstance(shared_input, FiveCandidateSharedInputV2):
        raise TypeError("shared_input must be a FiveCandidateSharedInputV2")
    for items, model, label in (
        (comment_evidence, EvidenceAtom, "comment_evidence"),
        (confirmed_memory, ConfirmedSpecificityPattern, "confirmed_memory"),
        (prior_rounds, SpecificityRound, "prior_rounds"),
    ):
        if not isinstance(items, list) or any(not isinstance(item, model) for item in items):
            raise TypeError(f"{label} must be a typed model list")
    if round_index not in {1, 2, 3}:
        raise ValueError("round_index must be 1..3")
    if candidate.candidate_id not in shared_input.candidate_ids:
        raise ValueError("candidate must belong to the frozen shared input")
    comment_ids = [item.evidence_id for item in comment_evidence]
    if comment_ids != shared_input.comment_evidence_ids:
        raise ValueError("comment evidence must exactly match the frozen shared input")
    memory_ids = [item.pattern_id for item in confirmed_memory]
    if memory_ids != shared_input.confirmed_pattern_ids:
        raise ValueError("confirmed memory must exactly match the frozen shared input")
    expected_rounds = list(range(1, round_index))
    if [item.round_index for item in prior_rounds] != expected_rounds:
        raise ValueError("prior rounds must be contiguous before the requested round")


def _validate_v2_task_payload(
    task: OfflineSpecificityTask,
    shared_input: FiveCandidateSharedInputV2,
) -> tuple[set[str], int]:
    _profile_is_exact_v2(task.model_profile)
    if set(task.payload) != _V2_PAYLOAD_KEYS:
        raise ValueError("specificity v2 task payload fields are not exact")
    payload = task.payload
    if payload["round"] != task.round_index:
        raise ValueError("task round does not match envelope")
    candidate = _strict_json_model(payload["candidate"], NarrativeCandidate)
    if task.candidate_versions != {candidate.candidate_id: task.candidate_versions.get(candidate.candidate_id)}:
        raise ValueError("task must bind exactly one candidate version")
    version = task.candidate_versions[candidate.candidate_id]
    if not isinstance(version, int) or version < 1:
        raise ValueError("candidate version must be a positive integer")
    comments = [_strict_json_model(item, EvidenceAtom) for item in payload["comment_evidence"]]
    public_evidence = [
        _strict_json_model(item, PublicNarrativeEvidence)
        for item in payload["public_evidence"]
    ]
    memory = [
        _strict_json_model(item, ConfirmedSpecificityPattern)
        for item in payload["confirmed_memory"]
    ]
    if shared_input.candidate_ids.count(candidate.candidate_id) != 1:
        raise ValueError("candidate must belong to the supplied shared input")
    if [item.evidence_id for item in comments] != shared_input.comment_evidence_ids:
        raise ValueError("task comment evidence does not match shared input")
    if [item.evidence_id for item in public_evidence] != [
        item.evidence_id for item in shared_input.public_evidence
    ]:
        raise ValueError("task public evidence does not match shared input")
    if [item.pattern_id for item in memory] != shared_input.confirmed_pattern_ids:
        raise ValueError("task confirmed memory does not match shared input")
    expected_shared_sha = sha256_bytes(canonical_json_bytes(shared_input))
    if payload["shared_input_sha256"] != expected_shared_sha:
        raise ValueError("task shared input SHA does not match shared input")
    if sha256_bytes(canonical_json_bytes(payload)) != task.input_sha256:
        raise ValueError("task input SHA does not match payload")
    return {item.evidence_id for item in comments}, version


def export_specificity_task_v2(
    *,
    run_id: str,
    candidate: NarrativeCandidate,
    comment_evidence: list[EvidenceAtom],
    shared_input: FiveCandidateSharedInputV2,
    confirmed_memory: list[ConfirmedSpecificityPattern],
    prior_rounds: list[SpecificityRound],
    round_index: int,
) -> OfflineSpecificityTask:
    """Freeze one candidate-only Luna v2 task without invoking any model or ledger."""
    _validate_v2_export_inputs(
        run_id=run_id,
        candidate=candidate,
        comment_evidence=comment_evidence,
        shared_input=shared_input,
        confirmed_memory=confirmed_memory,
        prior_rounds=prior_rounds,
        round_index=round_index,
    )
    candidate_history = _candidate_history_payload(candidate.candidate_id, prior_rounds)
    candidate_version = _candidate_version_map([candidate.candidate_id], prior_rounds)[
        candidate.candidate_id
    ]
    payload: dict[str, Any] = {
        "candidate": candidate.model_dump(mode="json"),
        "comment_evidence": [item.model_dump(mode="json") for item in comment_evidence],
        "public_evidence": [
            item.model_dump(mode="json") for item in shared_input.public_evidence
        ],
        "confirmed_memory": [item.model_dump(mode="json") for item in confirmed_memory],
        "prior_rounds": candidate_history,
        "round": round_index,
        "shared_input_sha256": sha256_bytes(canonical_json_bytes(shared_input)),
    }
    _reject_invisible_payload(payload)
    input_sha256 = sha256_bytes(canonical_json_bytes(payload))
    return OfflineSpecificityTask(
        task_id=(
            f"specificity-v2-{run_id}-{candidate.candidate_id}-{round_index}-"
            f"{input_sha256[:16]}"
        ),
        run_id=run_id,
        round_index=round_index,
        model_profile=LUNA_SPECIFICITY_V2_PROFILE,
        prompt_sha256=sha256_bytes(_v2_prompt_bytes()),
        schema_sha256=_schema_sha256(),
        input_sha256=input_sha256,
        candidate_versions={candidate.candidate_id: candidate_version},
        payload=payload,
    )


def import_specificity_result_v2(
    task: OfflineSpecificityTask,
    result_path: Path,
    shared_input: FiveCandidateSharedInputV2 | None = None,
) -> SpecificityAuditBatch:
    """Validate a Luna v2 result file; it deliberately has no online fallback or writes."""
    try:
        if not isinstance(task, OfflineSpecificityTask):
            raise TypeError("task must be an OfflineSpecificityTask")
        if not isinstance(shared_input, FiveCandidateSharedInputV2):
            raise TypeError("shared_input must be a FiveCandidateSharedInputV2")
        evidence_ids, _ = _validate_v2_task_payload(task, shared_input)
        if task.prompt_sha256 != sha256_bytes(_v2_prompt_bytes()):
            raise ValueError("task prompt SHA does not match the v2 contract")
        if task.schema_sha256 != _schema_sha256():
            raise ValueError("task schema SHA does not match the v2 contract")
        result_data = _read_canonical_json(result_path, "specificity v2 result")
        result = OfflineSpecificityResult.model_validate_json(result_data, strict=True)
        if result.model_dump(mode="json") != json.loads(result_data.decode("utf-8")):
            raise ValueError("specificity v2 result envelope is not canonical")
        _profile_is_exact_v2(result.model_profile)
        for field in (
            "task_id",
            "run_id",
            "round_index",
            "prompt_sha256",
            "schema_sha256",
            "input_sha256",
            "candidate_versions",
        ):
            if getattr(result, field) != getattr(task, field):
                raise ValueError(f"result {field} does not match task")
        _reject_invisible_payload(result.output)
        output = _strict_json_model(result.output, SpecificityAuditBatch)
        if output.model_dump(mode="json") != result.output:
            raise ValueError("specificity v2 output must use the exact JSON Schema shape")
        _validate_output_refs(output, task, evidence_ids, set())
        return output
    except SpecificityImportError:
        raise
    except Exception as exc:
        raise SpecificityImportError(str(exc)) from exc
