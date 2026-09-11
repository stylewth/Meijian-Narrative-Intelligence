"""Recoverable parent/child state machine for five-candidate dialogue validation.

The module deliberately owns orchestration only.  Luna and DeepSeek are injected
as callbacks so this layer can be tested without a model and can later be called
by a CLI that performs the real offline import/API exchange.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.schemas import (
    CandidateSpecificityAudit,
    FiveCandidateSharedInputV2,
    NarrativeCandidate,
    SpecificityAuditBatch,
    SpecificityDisposition,
    SpecificityOutcome,
    SpecificityRevisionBatch,
    StrictBaseModel,
)
from src.services.five_candidate_validation_inputs import FiveCandidateValidationInputs
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes
from src.services.specificity_memory import SpecificityMemoryError
from src.services.brand_specificity import (
    SpecificityCandidateRevisionV2,
    SpecificityFindingResponseV2,
    SpecificityNewBrandFactV2,
    SpecificityRevisionBatchV2,
    _validate_v2_revision_contract,
)


MAX_LUNA_ROUNDS = 3
MAX_DEEPSEEK_REVISIONS = 2
TERMINAL_STATUSES = frozenset({"READY", "HUMAN_REVIEW", "BLOCKED"})
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SHARED_COUNTS = {
    "candidates": 5,
    "comment_evidence": 309,
    "public_evidence": 54,
    "confirmed_patterns": 6,
}

LunaCallback = Callable[[NarrativeCandidate, int, list[dict[str, Any]]], SpecificityAuditBatch]
DeepSeekCallback = Callable[
    [NarrativeCandidate, SpecificityAuditBatch, int, list[dict[str, Any]]], Any
]
DeterministicGate = Callable[[SpecificityAuditBatch], str | bool | None]


class ValidationStateError(SpecificityMemoryError):
    """The persisted state or requested transition violates the contract."""


class ParentNotReadyError(ValidationStateError):
    """The parent still has a non-terminal child stream."""


class ChildState(StrictBaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    candidate_slug: str = Field(min_length=1)
    shared_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["PENDING", "AWAITING_LUNA", "AWAITING_DEEPSEEK", "FAILED", "READY", "HUMAN_REVIEW", "BLOCKED"]
    next_step: Literal["LUNA", "DEEPSEEK", "TERMINAL"]
    current_candidate: NarrativeCandidate
    current_version: int = Field(ge=1)
    rounds: list[dict[str, Any]] = Field(default_factory=list)
    blocking_gate_id: str | None = Field(default=None, min_length=1)
    failure: str | None = None


class _V2FieldDiffArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    before: str | list[str]
    after: str | list[str]
    source_finding_ids: list[str] = Field(min_length=1)
    comment_evidence_ids: list[str] = Field(default_factory=list)
    public_evidence_ids: list[str] = Field(default_factory=list)
    evidence_gap_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)


class _V2RevisionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    from_version: int = Field(ge=1)
    to_version: int = Field(ge=1)
    responses: list[SpecificityFindingResponseV2]
    revised_candidate: NarrativeCandidate
    field_diffs: list[_V2FieldDiffArtifact]
    unresolved_finding_ids: list[str]
    new_brand_facts: list[SpecificityNewBrandFactV2]


class _V2BatchArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=3)
    revisions: list[_V2RevisionArtifact]


@dataclass(frozen=True, slots=True)
class ParentAttempt:
    root: Path
    attempt_id: str
    candidate_ids: tuple[str, ...]
    shared_input_sha256: str
    candidate_slugs: dict[str, str]

    def slug_for(self, candidate_id: str) -> str:
        try:
            return self.candidate_slugs[candidate_id]
        except KeyError as exc:
            raise KeyError(f"unknown candidate: {candidate_id}") from exc

    def child_root(self, candidate_id: str) -> Path:
        return self.root / "candidates" / self.slug_for(candidate_id)

    def load_child_state(self, candidate_id: str) -> ChildState:
        path = self.child_root(candidate_id) / "state.json"
        return _load_model(path, ChildState, "child state")


def _write_json(path: Path, value: Any, *, replace: bool) -> None:
    data = canonical_json_bytes(value)
    if path.is_symlink():
        raise ValidationStateError(f"artifact must not be a symlink: {path}")
    if path.exists() and not replace:
        if path.is_file() and path.read_bytes() == data:
            return
        raise ValidationStateError(f"refusing to replace different artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_model(path: Path, model: type[BaseModel], label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValidationStateError(f"{label} must be a regular file: {path}")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
        if canonical_json_bytes(value) != data:
            raise ValueError("non-canonical JSON")
        return model.model_validate_json(data, strict=True)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        raise ValidationStateError(f"invalid {label}: {path}") from exc


def _validate_inputs(
    inputs: FiveCandidateValidationInputs,
    candidates: list[NarrativeCandidate],
) -> FiveCandidateSharedInputV2:
    if not isinstance(inputs, FiveCandidateValidationInputs):
        raise TypeError("inputs must be FiveCandidateValidationInputs")
    if not isinstance(candidates, list) or any(not isinstance(item, NarrativeCandidate) for item in candidates):
        raise TypeError("candidates must be a list of NarrativeCandidate")
    if len(candidates) != 5 or len({item.candidate_id for item in candidates}) != 5:
        raise ValidationStateError("parent attempt requires exactly five unique candidates")
    shared = inputs.shared_input
    counts = {
        "candidates": len(shared.candidate_ids),
        "comment_evidence": len(shared.comment_evidence_ids),
        "public_evidence": len(shared.public_evidence),
        "confirmed_patterns": len(shared.confirmed_pattern_ids),
    }
    if counts != _SHARED_COUNTS:
        raise ValidationStateError("shared input counts must be exactly 5/309/54/6")
    declared_canonical_sha = inputs.manifest.get("canonical_input_sha256")
    actual_canonical_sha = sha256_bytes(inputs.canonical_input_json)
    if declared_canonical_sha != actual_canonical_sha:
        raise ValidationStateError("canonical input SHA does not match the shared input manifest")
    if sha256_bytes(inputs.shared_input_json) != sha256_bytes(canonical_json_bytes(shared.model_dump(mode="json"))):
        raise ValidationStateError("shared input JSON is not canonical")
    if sha256_bytes(inputs.manifest_json) != sha256_bytes(canonical_json_bytes(inputs.manifest)):
        raise ValidationStateError("input manifest hash does not match manifest JSON")
    if set(shared.candidate_ids) != {item.candidate_id for item in candidates}:
        raise ValidationStateError("candidate IDs do not match shared input")
    return shared


def _candidate_slug(candidate_id: str, used: set[str]) -> str:
    slug = _SLUG_RE.sub("-", candidate_id).strip("-._") or "candidate"
    if slug in used:
        slug = f"{slug}-{sha256_bytes(candidate_id.encode('utf-8'))[:8]}"
    if slug in used:
        raise ValidationStateError("candidate slug collision")
    used.add(slug)
    return slug


def _manifest(attempt_id: str, candidate_ids: list[str], slugs: dict[str, str], shared_sha: str, inputs: FiveCandidateValidationInputs) -> dict[str, Any]:
    return {
        "schema_version": "five-candidate-validation-v1",
        "attempt_id": attempt_id,
        "candidate_ids": candidate_ids,
        "candidate_slugs": slugs,
        "shared_input_sha256": shared_sha,
        "canonical_input_sha256": sha256_bytes(inputs.canonical_input_json),
        "source_manifest_sha256": sha256_bytes(inputs.manifest_json),
        "counts": dict(_SHARED_COUNTS),
        "limits": {
            "max_luna_rounds": MAX_LUNA_ROUNDS,
            "max_deepseek_revisions": MAX_DEEPSEEK_REVISIONS,
        },
    }


def create_parent_attempt(
    root: Path,
    attempt_id: str,
    inputs: FiveCandidateValidationInputs,
    candidates: list[NarrativeCandidate],
) -> ParentAttempt:
    """Create an immutable shared-input parent and five isolated child streams."""

    if not isinstance(attempt_id, str) or not attempt_id or "/" in attempt_id or "\\" in attempt_id:
        raise ValueError("attempt_id must be one non-empty path component")
    shared = _validate_inputs(inputs, candidates)
    parent = Path(root) / attempt_id
    shared_json = canonical_json_bytes(shared.model_dump(mode="json"))
    shared_sha = sha256_bytes(shared_json)
    slugs: dict[str, str] = {}
    used: set[str] = set()
    for candidate in candidates:
        slugs[candidate.candidate_id] = _candidate_slug(candidate.candidate_id, used)
    manifest = _manifest(attempt_id, list(shared.candidate_ids), slugs, shared_sha, inputs)

    if parent.exists() or parent.is_symlink():
        existing = load_parent_attempt(parent)
        if existing.shared_input_sha256 != shared_sha or existing.candidate_ids != tuple(shared.candidate_ids):
            raise ValidationStateError("attempt already exists with different shared input")
        return existing

    parent.mkdir(parents=True)
    try:
        _write_json(parent / "shared_input.json", shared.model_dump(mode="json"), replace=False)
        _write_json(parent / "manifest.json", manifest, replace=False)
        _write_json(parent / "canonical_input.json", json.loads(inputs.canonical_input_json), replace=False)
        _write_json(parent / "source_manifest.json", json.loads(inputs.manifest_json), replace=False)
        for candidate in candidates:
            child = parent / "candidates" / slugs[candidate.candidate_id]
            state = ChildState(
                candidate_id=candidate.candidate_id,
                candidate_slug=slugs[candidate.candidate_id],
                shared_input_sha256=shared_sha,
                status="PENDING",
                next_step="LUNA",
                current_candidate=candidate,
                current_version=1,
            )
            _write_json(child / "state.json", state.model_dump(mode="json"), replace=False)
            (child / "rounds").mkdir(parents=True, exist_ok=True)
    except Exception:
        # Do not reinterpret a partial attempt as usable; the caller can inspect
        # and explicitly remove it if creation failed.
        raise
    return ParentAttempt(parent, attempt_id, tuple(shared.candidate_ids), shared_sha, slugs)


def load_parent_attempt(root: Path) -> ParentAttempt:
    """Load and validate the parent manifest and its five child directories."""

    parent = Path(root)
    manifest_path = parent / "manifest.json"
    manifest = _load_model(manifest_path, _ManifestModel, "parent manifest")
    if manifest.attempt_id != parent.name or len(manifest.candidate_ids) != 5:
        raise ValidationStateError("invalid five-candidate parent manifest")
    shared_path = parent / "shared_input.json"
    shared = _load_model(shared_path, FiveCandidateSharedInputV2, "shared input")
    shared_sha = sha256_bytes(shared_path.read_bytes())
    if shared_sha != manifest.shared_input_sha256:
        raise ValidationStateError("shared input SHA does not match parent manifest")
    if tuple(shared.candidate_ids) != tuple(manifest.candidate_ids):
        raise ValidationStateError("shared input candidates do not match parent manifest")
    counts = {
        "candidates": len(shared.candidate_ids),
        "comment_evidence": len(shared.comment_evidence_ids),
        "public_evidence": len(shared.public_evidence),
        "confirmed_patterns": len(shared.confirmed_pattern_ids),
    }
    if counts != _SHARED_COUNTS or manifest.counts != _SHARED_COUNTS:
        raise ValidationStateError("shared input counts must be exactly 5/309/54/6")
    canonical_path = parent / "canonical_input.json"
    source_manifest_path = parent / "source_manifest.json"
    canonical_data = _canonical_artifact(canonical_path, "canonical input")
    source_manifest_data = _canonical_artifact(source_manifest_path, "source manifest")
    if sha256_bytes(canonical_data) != manifest.canonical_input_sha256:
        raise ValidationStateError("canonical input SHA does not match parent manifest")
    if sha256_bytes(source_manifest_data) != manifest.source_manifest_sha256:
        raise ValidationStateError("source manifest hash does not match parent manifest")
    if set(manifest.candidate_slugs) != set(manifest.candidate_ids):
        raise ValidationStateError("parent candidate slug map is incomplete")
    for candidate_id in manifest.candidate_ids:
        child = parent / "candidates" / manifest.candidate_slugs[candidate_id]
        state = _load_model(child / "state.json", ChildState, "child state")
        if state.candidate_id != candidate_id or state.shared_input_sha256 != shared_sha:
            raise ValidationStateError("child state is not bound to the parent input")
    return ParentAttempt(parent, manifest.attempt_id, tuple(manifest.candidate_ids), shared_sha, dict(manifest.candidate_slugs))


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    attempt_id: str
    candidate_ids: list[str]
    candidate_slugs: dict[str, str]
    shared_input_sha256: str
    canonical_input_sha256: str
    source_manifest_sha256: str
    counts: dict[str, int]
    limits: dict[str, int]


def _canonical_artifact(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValidationStateError(f"{label} must be a regular file: {path}")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationStateError(f"invalid {label}: {path}") from exc
    canonical = canonical_json_bytes(value)
    if canonical != data:
        raise ValidationStateError(f"non-canonical {label}: {path}")
    return data


def _persist_state(attempt: ParentAttempt, state: ChildState) -> None:
    _write_json(attempt.child_root(state.candidate_id) / "state.json", state.model_dump(mode="json"), replace=True)


def _artifact_path(attempt: ParentAttempt, candidate_id: str, round_index: int, name: str) -> Path:
    return attempt.child_root(candidate_id) / "rounds" / f"round-{round_index:02d}" / name


def _load_artifact(path: Path) -> Any | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise ValidationStateError(f"artifact must be a regular file: {path}")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"))
        if canonical_json_bytes(value) != data:
            raise ValueError("non-canonical JSON")
        return value
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValidationStateError(f"invalid artifact: {path}") from exc


def _validate_audit(audit: Any, attempt: ParentAttempt, candidate_id: str, round_index: int, version: int) -> SpecificityAuditBatch:
    if not isinstance(audit, SpecificityAuditBatch):
        raise ValidationStateError("Luna must return SpecificityAuditBatch")
    if audit.run_id != attempt.attempt_id or audit.round_index != round_index:
        raise ValidationStateError("Luna audit identity does not match child stream")
    if audit.candidate_ids != [candidate_id] or audit.candidate_versions != {candidate_id: version}:
        raise ValidationStateError("Luna audit candidate/version does not match child state")
    return audit


def _adapt_v2_candidate(candidate: NarrativeCandidate) -> Any:
    payload = candidate.model_dump(mode="json")
    for field in ("main_scenes", "risks"):
        payload[field] = canonical_json_bytes(payload[field]).decode("utf-8")
    return SimpleNamespace(candidate_id=candidate.candidate_id, model_dump=lambda mode="json": payload)


def _adapt_v2_revision(revision: _V2BatchArtifact) -> Any:
    adapted_revisions = []
    for item in revision.revisions:
        revised_candidate = _adapt_v2_candidate(item.revised_candidate)
        diffs = []
        for diff in item.field_diffs:
            before = diff.before
            after = diff.after
            if isinstance(before, list):
                before = canonical_json_bytes(before).decode("utf-8")
            if isinstance(after, list):
                after = canonical_json_bytes(after).decode("utf-8")
            diffs.append(SimpleNamespace(
                field=diff.field,
                before=before,
                after=after,
                source_finding_ids=diff.source_finding_ids,
                comment_evidence_ids=diff.comment_evidence_ids,
                public_evidence_ids=diff.public_evidence_ids,
                evidence_gap_ids=diff.evidence_gap_ids,
            ))
        adapted_revisions.append(SimpleNamespace(
            candidate_id=item.candidate_id,
            from_version=item.from_version,
            to_version=item.to_version,
            responses=item.responses,
            revised_candidate=revised_candidate,
            field_diffs=diffs,
            unresolved_finding_ids=item.unresolved_finding_ids,
            new_brand_facts=item.new_brand_facts,
        ))
    return SimpleNamespace(
        run_id=revision.run_id,
        round_index=revision.round_index,
        revisions=adapted_revisions,
    )


def _parse_v2_revision(value: Any) -> SpecificityRevisionBatchV2 | _V2BatchArtifact:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    if isinstance(value, SpecificityRevisionBatchV2):
        return value
    try:
        typed = SpecificityRevisionBatchV2.model_validate(payload, strict=True)
    except (TypeError, ValueError, ValidationError):
        try:
            return _V2BatchArtifact.model_validate(payload, strict=True)
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValidationStateError("invalid SpecificityRevisionBatchV2 output") from exc
    return typed


def _validate_v2_revision_bindings(
    revision: SpecificityRevisionBatchV2 | _V2BatchArtifact,
    audit: SpecificityAuditBatch,
    candidate: NarrativeCandidate,
    attempt: ParentAttempt,
) -> None:
    shared = _load_model(attempt.root / "shared_input.json", FiveCandidateSharedInputV2, "shared input")
    canonical = json.loads(_canonical_artifact(attempt.root / "canonical_input.json", "canonical input"))
    comment_atoms = canonical.get("comment_evidence_atoms")
    if not isinstance(comment_atoms, list) or len(comment_atoms) != _SHARED_COUNTS["comment_evidence"]:
        raise ValidationStateError("canonical input must contain exactly 309 comment evidence atoms")
    comment_ids = [
        atom.get("comment_id") if isinstance(atom, dict) else None
        for atom in comment_atoms
    ]
    if any(not isinstance(comment_id, str) or not comment_id for comment_id in comment_ids):
        raise ValidationStateError("canonical comment evidence atoms must contain comment IDs")
    if len(comment_ids) != len(set(comment_ids)):
        raise ValidationStateError("canonical comment evidence atom comment IDs must be unique")
    try:
        if isinstance(revision, SpecificityRevisionBatchV2):
            _validate_v2_revision_contract(
                audit,
                revision,
                {candidate.candidate_id: candidate},
                set(comment_ids),
                {item.evidence_id for item in shared.public_evidence},
            )
            return
        adapted_candidate = _adapt_v2_candidate(candidate)
        _validate_v2_revision_contract(
            audit,
            _adapt_v2_revision(revision),
            {candidate.candidate_id: adapted_candidate},
            set(comment_ids),
            {item.evidence_id for item in shared.public_evidence},
        )
    except (TypeError, ValueError) as exc:
        raise ValidationStateError(str(exc)) from exc


def _validate_revision(
    revision: Any,
    audit: SpecificityAuditBatch,
    candidate: NarrativeCandidate,
    current_version: int,
    round_index: int,
    attempt: ParentAttempt,
) -> Any:
    if isinstance(revision, SpecificityRevisionBatchV2) or (
        isinstance(revision, dict) and "revisions" in revision
    ):
        revision = _parse_v2_revision(revision)
    if not isinstance(revision, (SpecificityRevisionBatch, SpecificityRevisionBatchV2, _V2BatchArtifact)):
        raise ValidationStateError("DeepSeek must return a specificity revision batch")
    if revision.run_id != audit.run_id or revision.round_index != round_index:
        raise ValidationStateError("DeepSeek revision identity does not match audit")
    if len(revision.revisions) != 1:
        raise ValidationStateError("each child stream accepts exactly one revision")
    item = revision.revisions[0]
    if item.candidate_id != candidate.candidate_id:
        raise ValidationStateError("revision candidate does not match child stream")
    if item.from_version != current_version:
        raise ValidationStateError("revision from_version does not match current candidate version")
    if item.to_version != item.from_version + 1:
        raise ValidationStateError("revision version must increase by one")
    if item.revised_candidate.candidate_id != candidate.candidate_id:
        raise ValidationStateError("revision cannot change candidate identity")
    if isinstance(revision, SpecificityRevisionBatch):
        _validate_v1_revision_bindings(item, audit, candidate)
    else:
        _validate_v2_revision_bindings(revision, audit, candidate, attempt)
    return revision


def _validate_v1_revision_bindings(
    revision: Any,
    audit: SpecificityAuditBatch,
    candidate: NarrativeCandidate,
) -> None:
    findings = {finding.finding_id: finding for item in audit.audits for finding in item.findings}
    substantive_ids = {
        finding.finding_id
        for finding in findings.values()
        if finding.severity.value != "NOTE"
    }
    responses = {response.finding_id: response for response in revision.responses}
    if set(responses) != substantive_ids:
        raise ValidationStateError("revision must respond to every substantive finding exactly once")
    accepted_ids = {
        finding_id
        for finding_id, response in responses.items()
        if response.disposition is SpecificityDisposition.ACCEPT
    }
    unresolved_ids = {
        finding_id
        for finding_id, response in responses.items()
        if response.disposition is not SpecificityDisposition.ACCEPT
    }
    if set(revision.unresolved_finding_ids) != unresolved_ids:
        raise ValidationStateError("revision unresolved findings do not match dispositions")
    allowed_fields = {
        "title", "target_audience", "user_conflict", "brand_opportunity",
        "why_brand", "why_meijian", "competitor_difference", "brand_role",
        "draft_proposition", "main_scenes", "content_theme", "risks",
    }
    original = candidate.model_dump(mode="json")
    revised = revision.revised_candidate.model_dump(mode="json")
    changed_fields = {field for field in original if original[field] != revised[field]}
    if not changed_fields.issubset(allowed_fields):
        raise ValidationStateError("revision changed a non-narrative candidate field")
    patch_fields: set[str] = set()
    for finding_id in accepted_ids:
        patch = responses[finding_id].applied_patch
        if patch is None or patch.field_name not in allowed_fields:
            raise ValidationStateError("ACCEPT findings require an allowed applied patch")
        if patch.before != getattr(candidate, patch.field_name) or patch.after != getattr(revision.revised_candidate, patch.field_name):
            raise ValidationStateError("applied patch is not bound to the candidate diff")
        patch_fields.add(patch.field_name)
    if changed_fields != patch_fields:
        raise ValidationStateError("candidate changes do not match ACCEPT patches")
    diff_fields: set[str] = set()
    for diff in revision.field_diffs:
        if diff.field_name in diff_fields or diff.field_name not in allowed_fields:
            raise ValidationStateError("revision field diffs must be unique and allowed")
        if diff.before != getattr(candidate, diff.field_name) or diff.after != getattr(revision.revised_candidate, diff.field_name):
            raise ValidationStateError("field diff is not bound to the candidate text")
        if not set(diff.source_finding_ids).issubset(accepted_ids):
            raise ValidationStateError("field diff must cite an ACCEPT finding")
        diff_fields.add(diff.field_name)
    if diff_fields != changed_fields:
        raise ValidationStateError("field diffs do not cover candidate changes")


def _revision_candidate(revision: Any) -> NarrativeCandidate:
    revised = revision.revisions[0].revised_candidate
    if not isinstance(revised, NarrativeCandidate):
        raise ValidationStateError("revision must contain a NarrativeCandidate")
    return revised


def _history_for_callback(state: ChildState) -> list[dict[str, Any]]:
    return [dict(item) for item in state.rounds]


def _outcome_for_audit(audit: SpecificityAuditBatch, round_index: int, gate: DeterministicGate | None) -> tuple[str | None, str | None]:
    blocking_gate = gate(audit) if gate is not None else None
    if blocking_gate is True:
        blocking_gate = "deterministic-gate"
    if blocking_gate:
        return "BLOCKED", str(blocking_gate)
    if all(finding.severity.value == "NOTE" for item in audit.audits for finding in item.findings):
        return "READY", None
    if round_index >= MAX_LUNA_ROUNDS:
        return "HUMAN_REVIEW", None
    return None, None


def run_candidate(
    attempt: ParentAttempt | Path,
    candidate_id: str,
    *,
    luna: LunaCallback,
    deepseek: DeepSeekCallback,
    deterministic_gate: DeterministicGate | None = None,
) -> ChildState:
    """Advance exactly one child stream until it reaches a terminal outcome."""

    parent = attempt if isinstance(attempt, ParentAttempt) else load_parent_attempt(Path(attempt))
    if not callable(luna) or not callable(deepseek):
        raise TypeError("luna and deepseek must be callable")
    state = parent.load_child_state(candidate_id)
    if state.status in TERMINAL_STATUSES:
        return state

    while state.status not in TERMINAL_STATUSES:
        round_index = len(state.rounds) + 1
        if round_index > MAX_LUNA_ROUNDS:
            raise ValidationStateError("Luna round limit exceeded")
        candidate = state.current_candidate
        round_dir = _artifact_path(parent, candidate_id, round_index, "luna_task.json").parent
        round_dir.mkdir(parents=True, exist_ok=True)

        if state.next_step == "LUNA":
            task_path = round_dir / "luna_task.json"
            task = {
                "attempt_id": parent.attempt_id,
                "candidate_id": candidate_id,
                "round_index": round_index,
                "candidate_version": state.current_version,
                "shared_input_sha256": parent.shared_input_sha256,
                "shared_input_counts": dict(_SHARED_COUNTS),
                "candidate": candidate.model_dump(mode="json"),
                "history": _history_for_callback(state),
            }
            _write_json(task_path, task, replace=False)
            result_path = round_dir / "luna_result.json"
            try:
                raw_audit = _load_artifact(result_path)
                audit = (
                    SpecificityAuditBatch.model_validate_json(canonical_json_bytes(raw_audit), strict=True)
                    if raw_audit is not None
                    else luna(candidate, round_index, _history_for_callback(state))
                )
                audit = _validate_audit(audit, parent, candidate_id, round_index, state.current_version)
                _write_json(result_path, audit.model_dump(mode="json"), replace=False)
            except Exception as exc:
                state = state.model_copy(update={"status": "FAILED", "next_step": "LUNA", "failure": str(exc)})
                _persist_state(parent, state)
                raise

            outcome, blocking_gate = _outcome_for_audit(audit, round_index, deterministic_gate)
            round_record = {
                "candidate_id": candidate_id,
                "round_index": round_index,
                "candidate_version": state.current_version,
                "luna_result_sha256": sha256_bytes(result_path.read_bytes()),
                "audit": audit.model_dump(mode="json"),
                "revision": None,
            }
            if state.rounds and state.rounds[-1].get("round_index") == round_index:
                rounds = [*state.rounds[:-1], round_record]
            else:
                rounds = [*state.rounds, round_record]
            if outcome is not None:
                state = state.model_copy(
                    update={
                        "rounds": rounds,
                        "status": outcome,
                        "next_step": "TERMINAL",
                        "blocking_gate_id": blocking_gate,
                        "failure": None,
                    }
                )
                _persist_state(parent, state)
                _write_json(parent.child_root(candidate_id) / "final_candidate.json", {"candidate": candidate.model_dump(mode="json"), "status": outcome, "version": state.current_version, "blocking_gate_id": blocking_gate}, replace=False)
                return state
            state = state.model_copy(update={"rounds": rounds, "status": "AWAITING_DEEPSEEK", "next_step": "DEEPSEEK", "failure": None})
            _persist_state(parent, state)

        if state.next_step == "DEEPSEEK":
            if len(state.rounds) > MAX_DEEPSEEK_REVISIONS:
                raise ValidationStateError("DeepSeek revision limit exceeded")
            round_index = len(state.rounds)
            current_round = state.rounds[-1]
            audit = SpecificityAuditBatch.model_validate_json(
                canonical_json_bytes(current_round["audit"]), strict=True
            )
            call_path = _artifact_path(parent, candidate_id, round_index, "deepseek_call.json")
            call = {
                "attempt_id": parent.attempt_id,
                "candidate_id": candidate_id,
                "round_index": round_index,
                "shared_input_sha256": parent.shared_input_sha256,
                "candidate_version": state.current_version,
                "shared_input_counts": dict(_SHARED_COUNTS),
                "candidate": state.current_candidate.model_dump(mode="json"),
                "audit": audit.model_dump(mode="json"),
                "history": _history_for_callback(state),
            }
            _write_json(call_path, call, replace=False)
            revision_path = call_path.parent / "revision.json"
            try:
                raw_revision = _load_artifact(revision_path)
                revision = (
                    _parse_v2_revision(raw_revision)
                    if raw_revision is not None
                    else deepseek(state.current_candidate, audit, round_index, _history_for_callback(state))
                )
                revision = _validate_revision(
                    revision,
                    audit,
                    state.current_candidate,
                    state.current_version,
                    round_index,
                    parent,
                )
                _write_json(revision_path, revision.model_dump(mode="json"), replace=False)
            except Exception as exc:
                state = state.model_copy(update={"status": "FAILED", "next_step": "DEEPSEEK", "failure": str(exc)})
                _persist_state(parent, state)
                raise
            revised_candidate = _revision_candidate(revision)
            updated_round = {**current_round, "revision": revision.model_dump(mode="json"), "revision_sha256": sha256_bytes(revision_path.read_bytes())}
            state = state.model_copy(
                update={
                    "rounds": [*state.rounds[:-1], updated_round],
                    "current_candidate": revised_candidate,
                    "current_version": state.current_version + 1,
                    "status": "AWAITING_LUNA",
                    "next_step": "LUNA",
                    "failure": None,
                }
            )
            _persist_state(parent, state)
            continue
        raise ValidationStateError(f"unknown child next_step: {state.next_step}")
    return state


def prepare_unified_evaluation(attempt: ParentAttempt | Path) -> dict[str, Any]:
    """Return a frozen five-candidate evaluation input only after all children finish."""

    parent = attempt if isinstance(attempt, ParentAttempt) else load_parent_attempt(Path(attempt))
    states = [parent.load_child_state(candidate_id) for candidate_id in parent.candidate_ids]
    pending = [state.candidate_id for state in states if state.status not in TERMINAL_STATUSES]
    if pending:
        raise ParentNotReadyError(f"unified evaluation requires terminal children: {pending}")
    return {
        "attempt_id": parent.attempt_id,
        "shared_input_sha256": parent.shared_input_sha256,
        "candidate_ids": list(parent.candidate_ids),
        "candidates": [
            {
                "candidate_id": state.candidate_id,
                "candidate_version": state.current_version,
                "status": state.status,
                "candidate": state.current_candidate.model_dump(mode="json"),
                "rounds": state.rounds,
            }
            for state in states
        ],
    }


__all__ = [
    "MAX_LUNA_ROUNDS",
    "MAX_DEEPSEEK_REVISIONS",
    "ChildState",
    "ParentAttempt",
    "ParentNotReadyError",
    "ValidationStateError",
    "create_parent_attempt",
    "load_parent_attempt",
    "prepare_unified_evaluation",
    "run_candidate",
]
