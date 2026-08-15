from __future__ import annotations

import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from src.schemas import (
    BrandFact,
    CandidateSpecificityAudit,
    CandidateSpecificityHistory,
    ConfirmedSpecificityPattern,
    DecisionRunStage,
    EvidenceAtom,
    NarrativeCandidate,
    PublicNarrativeEvidence,
    SpecificityEvidenceBindingV2,
    SpecificityAuditBatch,
    SpecificityCandidateRevision,
    SpecificityDisposition,
    SpecificityFinalRankResult,
    SpecificityOutcome,
    SpecificityRevisionBatch,
    StrictBaseModel,
)
from src.services.decision_run_store import AuditedLLMClient
from src.services.prepared_corpus import canonical_json_bytes


ALLOWED_NARRATIVE_PATCH_FIELDS = frozenset(
    {
        "title",
        "target_audience",
        "user_conflict",
        "brand_opportunity",
        "why_meijian",
        "competitor_difference",
        "brand_role",
        "draft_proposition",
        "main_scenes",
        "content_theme",
        "risks",
    }
)


class SpecificityContractError(ValueError):
    pass


_FORBIDDEN_INPUT = re.compile(r"holdout|blind|checkpoint-|future[ _-]?delta", re.IGNORECASE)
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "brand_specificity_revision.md"
_V2_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "brand_specificity_revision_v2.md"
)
SPECIFICITY_FINAL_RANK_PROMPT = (
    "Return only SpecificityFinalRankResult JSON. Rank the supplied candidate IDs "
    "without changing candidate text, evidence, BrandFacts, scores, or human selection. "
    "candidate_ids, candidate_versions, and input_sha256 are identity bindings; copy them "
    "exactly. ranked_candidate_ids must be a permutation of candidate_ids."
)
SPECIFICITY_FINAL_RANK_PROMPT_SHA256 = hashlib.sha256(
    SPECIFICITY_FINAL_RANK_PROMPT.encode("utf-8")
).hexdigest()


class SpecificityFindingResponseV2(StrictBaseModel):
    finding_id: str
    disposition: Literal["ACCEPT", "REJECT", "DEFER"]
    reason: str


class SpecificityNewBrandFactV2(StrictBaseModel):
    fact_id: str
    statement: str
    public_evidence_ids: list[str]
    reason: str


class SpecificityCandidateRevisionV2(StrictBaseModel):
    candidate_id: str
    from_version: int
    to_version: int
    responses: list[SpecificityFindingResponseV2]
    revised_candidate: NarrativeCandidate
    field_diffs: list[SpecificityEvidenceBindingV2]
    unresolved_finding_ids: list[str]
    new_brand_facts: list[SpecificityNewBrandFactV2]


class SpecificityRevisionBatchV2(StrictBaseModel):
    run_id: str
    round_index: int
    revisions: list[SpecificityCandidateRevisionV2]


def _reject_forbidden(value: Any) -> None:
    serialized = canonical_json_bytes(value).decode("utf-8")
    if _FORBIDDEN_INPUT.search(serialized):
        raise SpecificityContractError(
            "specificity revision input cannot contain holdout, blind, checkpoint, or future delta data"
        )


def _strict_json_model(value: Any, model: type[Any]) -> Any:
    return model.model_validate_json(canonical_json_bytes(value), strict=True)


def _prompt_text() -> str:
    if _PROMPT_PATH.is_symlink() or not _PROMPT_PATH.is_file():
        raise SpecificityContractError("specificity revision prompt must be a regular file")
    data = _PROMPT_PATH.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise SpecificityContractError("specificity revision prompt must not contain a UTF-8 BOM")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpecificityContractError("specificity revision prompt must be UTF-8") from exc


def _v2_prompt_text() -> str:
    if _V2_PROMPT_PATH.is_symlink() or not _V2_PROMPT_PATH.is_file():
        raise SpecificityContractError("specificity revision v2 prompt must be a regular file")
    data = _V2_PROMPT_PATH.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise SpecificityContractError("specificity revision v2 prompt must not contain a UTF-8 BOM")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SpecificityContractError("specificity revision v2 prompt must be UTF-8") from exc


def _typed_list(values: list[Any], model: type[Any], label: str) -> None:
    if not isinstance(values, list) or any(not isinstance(value, model) for value in values):
        raise TypeError(f"{label} must be a typed list")


def build_specificity_audit_input(
    candidates: list[NarrativeCandidate],
    evidence_atoms: list[EvidenceAtom],
    brand_facts: list[BrandFact],
    histories: list[CandidateSpecificityHistory],
    confirmed_patterns: list[ConfirmedSpecificityPattern],
) -> dict[str, object]:
    _typed_list(candidates, NarrativeCandidate, "candidates")
    _typed_list(evidence_atoms, EvidenceAtom, "evidence_atoms")
    _typed_list(brand_facts, BrandFact, "brand_facts")
    _typed_list(histories, CandidateSpecificityHistory, "histories")
    _typed_list(confirmed_patterns, ConfirmedSpecificityPattern, "confirmed_patterns")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise SpecificityContractError("candidates must have unique IDs")
    evidence_ids = [atom.evidence_id for atom in evidence_atoms]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise SpecificityContractError("evidence IDs must be unique")
    fact_ids = [fact.fact_id for fact in brand_facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise SpecificityContractError("BrandFact IDs must be unique")
    payload: dict[str, object] = {
        "candidates": [
            candidate.model_dump(mode="json")
            for candidate in sorted(candidates, key=lambda item: item.candidate_id)
        ],
        "evidence_atoms": [
            atom.model_dump(mode="json")
            for atom in sorted(evidence_atoms, key=lambda item: item.evidence_id)
        ],
        "brand_facts": [
            fact.model_dump(mode="json")
            for fact in sorted(brand_facts, key=lambda item: item.fact_id)
        ],
        "histories": [
            history.model_dump(mode="json")
            for history in sorted(histories, key=lambda item: item.candidate_id)
        ],
        "confirmed_patterns": [
            pattern.model_dump(mode="json")
            for pattern in sorted(confirmed_patterns, key=lambda item: item.pattern_id)
        ],
    }
    _reject_forbidden(payload)
    return payload


def _audit_findings(audit: SpecificityAuditBatch) -> dict[str, Any]:
    return {
        finding.finding_id: finding
        for candidate_audit in audit.audits
        for finding in candidate_audit.findings
    }


def _validate_revision_contract(
    audit: SpecificityAuditBatch,
    revision_batch: SpecificityRevisionBatch,
    candidates: dict[str, NarrativeCandidate],
) -> None:
    if revision_batch.run_id != audit.run_id or revision_batch.round_index != audit.round_index:
        raise SpecificityContractError("revision batch identity does not match audit")
    findings = _audit_findings(audit)
    substantive_by_candidate: dict[str, set[str]] = {}
    for candidate_audit in audit.audits:
        substantive_by_candidate[candidate_audit.candidate_id] = {
            finding.finding_id
            for finding in candidate_audit.findings
            if finding.severity.value != "NOTE"
        }
    revision_ids = [revision.candidate_id for revision in revision_batch.revisions]
    if len(revision_ids) != len(set(revision_ids)):
        raise SpecificityContractError("each candidate can have only one revision")
    if not set(revision_ids).issubset(candidates):
        raise SpecificityContractError("revision contains an unknown candidate")

    for revision in revision_batch.revisions:
        candidate = candidates[revision.candidate_id]
        expected_version = audit.candidate_versions.get(revision.candidate_id)
        if expected_version is None:
            raise SpecificityContractError("revision candidate is missing from audit candidate_versions")
        if revision.from_version != expected_version:
            raise SpecificityContractError(
                "revision from_version does not match the audit candidate_version"
            )
        expected_substantive = substantive_by_candidate.get(revision.candidate_id, set())
        response_ids = [response.finding_id for response in revision.responses]
        if len(response_ids) != len(set(response_ids)):
            raise SpecificityContractError("each finding can have only one response")
        if set(response_ids) != expected_substantive:
            raise SpecificityContractError(
                "every substantive finding must have exactly one response"
            )
        _validate_single_revision(revision, candidate, findings, expected_substantive)


def _validate_single_revision(
    revision: SpecificityCandidateRevision,
    candidate: NarrativeCandidate,
    findings: dict[str, Any] | None = None,
    expected_substantive: set[str] | None = None,
) -> None:
    if revision.candidate_id != candidate.candidate_id:
        raise SpecificityContractError("revision candidate_id does not match candidate")
    if revision.revised_candidate.candidate_id != candidate.candidate_id:
        raise SpecificityContractError("revised candidate_id does not match candidate")
    if revision.from_version < 1 or revision.to_version != revision.from_version + 1:
        raise SpecificityContractError("revision versions must increase by exactly one")
    response_ids = {response.finding_id for response in revision.responses}
    accepted_ids: set[str] = set()
    nonaccepted_ids: set[str] = set()
    patch_fields: set[str] = set()
    for response in revision.responses:
        if response.disposition is SpecificityDisposition.ACCEPT:
            if response.applied_patch is None:
                raise SpecificityContractError("ACCEPT requires an applied patch")
            patch = response.applied_patch
            if patch.field_name not in ALLOWED_NARRATIVE_PATCH_FIELDS:
                raise SpecificityContractError("patch field is not an allowed narrative field")
            accepted_ids.add(response.finding_id)
            patch_fields.add(patch.field_name)
            if patch.before != getattr(candidate, patch.field_name):
                raise SpecificityContractError("patch before does not match candidate")
            if patch.after != getattr(revision.revised_candidate, patch.field_name):
                raise SpecificityContractError("patch after does not match revised candidate")
        else:
            if response.applied_patch is not None:
                raise SpecificityContractError("REJECT or DEFER cannot carry a patch")
            nonaccepted_ids.add(response.finding_id)
    if set(revision.unresolved_finding_ids) != nonaccepted_ids:
        raise SpecificityContractError("REJECT and DEFER findings must remain unresolved")
    if expected_substantive is not None and response_ids != expected_substantive:
        raise SpecificityContractError("response IDs do not match substantive findings")
    if findings is not None and any(finding_id not in findings for finding_id in response_ids):
        raise SpecificityContractError("response references an unknown finding")

    original = candidate.model_dump(mode="json")
    revised = revision.revised_candidate.model_dump(mode="json")
    changed_fields = {field for field in original if original[field] != revised[field]}
    if not changed_fields.issubset(ALLOWED_NARRATIVE_PATCH_FIELDS):
        raise SpecificityContractError("revision changed evidence, facts, scores, or human selection")
    if changed_fields != patch_fields:
        raise SpecificityContractError("changed fields do not match ACCEPT patches")
    diffs_by_field: dict[str, Any] = {}
    for diff in revision.field_diffs:
        if diff.field_name in diffs_by_field:
            raise SpecificityContractError("field diff fields must be unique")
        if diff.field_name not in ALLOWED_NARRATIVE_PATCH_FIELDS:
            raise SpecificityContractError("field diff contains a disallowed field")
        if diff.before != getattr(candidate, diff.field_name):
            raise SpecificityContractError("field diff before does not match candidate")
        if diff.after != getattr(revision.revised_candidate, diff.field_name):
            raise SpecificityContractError("field diff after does not match revised candidate")
        if not set(diff.source_finding_ids).issubset(accepted_ids):
            raise SpecificityContractError("field diff source must be an ACCEPT finding")
        diffs_by_field[diff.field_name] = diff
    if set(diffs_by_field) != changed_fields:
        raise SpecificityContractError("field diffs do not match changed fields")
    if not accepted_ids and changed_fields:
        raise SpecificityContractError("only ACCEPT findings may change a candidate")


def revise_from_specificity_audit(
    client: AuditedLLMClient,
    audit: SpecificityAuditBatch,
    candidates: list[NarrativeCandidate],
) -> SpecificityRevisionBatch:
    if not isinstance(client, AuditedLLMClient):
        raise TypeError("client must be an AuditedLLMClient")
    if not isinstance(audit, SpecificityAuditBatch):
        raise TypeError("audit must be a SpecificityAuditBatch")
    _typed_list(candidates, NarrativeCandidate, "candidates")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)) or set(candidate_ids) != set(audit.candidate_ids):
        raise SpecificityContractError("candidates must exactly match audit candidates")
    store = getattr(client, "_store", None)
    profile = getattr(getattr(store, "manifest", None), "decision_model", None)
    if profile is None or profile.provider != "deepseek":
        raise SpecificityContractError("specificity revision must use the DeepSeek decision profile")
    request_payload = {
        "audit": audit.model_dump(mode="json"),
        "candidate_versions": dict(sorted(audit.candidate_versions.items())),
        "candidates": [
            candidate.model_dump(mode="json")
            for candidate in sorted(candidates, key=lambda item: item.candidate_id)
        ],
    }
    _reject_forbidden(request_payload)
    result = None
    with client.stage(DecisionRunStage.SPECIFICITY_REVISION):
        result = client.generate_json(
            system_prompt=_prompt_text(),
            user_prompt=canonical_json_bytes(request_payload).decode("utf-8"),
            response_model=SpecificityRevisionBatch,
        )
    if not isinstance(result, SpecificityRevisionBatch):
        raise SpecificityContractError("DeepSeek did not return SpecificityRevisionBatch")
    try:
        validated = _strict_json_model(result.model_dump(mode="json"), SpecificityRevisionBatch)
    except (TypeError, ValueError, ValidationError) as exc:
        raise SpecificityContractError("invalid SpecificityRevisionBatch output") from exc
    _validate_revision_contract(
        audit,
        validated,
        {candidate.candidate_id: candidate for candidate in candidates},
    )
    return validated


def _validate_v2_deepseek_profile(client: AuditedLLMClient) -> None:
    store = getattr(client, "_store", None)
    profile = getattr(getattr(store, "manifest", None), "decision_model", None)
    expected = {
        "profile_id": "decision-profile",
        "provider": "deepseek",
        "model_id": "deepseek-v4-flash",
        "runtime": "cloud",
        "response_format": "json_object",
        "prompt_version": "decision-v1",
        "endpoint_profile": "deepseek-official",
        "thinking_enabled": True,
        "reasoning_effort": "max",
        "max_retries": 0,
    }
    if profile is None:
        raise SpecificityContractError("specificity revision v2 requires the fixed DeepSeek profile")
    actual = profile.model_dump(mode="json")
    if actual != expected:
        raise SpecificityContractError("specificity revision v2 requires the fixed DeepSeek profile")


def _is_narrowing(before: str, after: str) -> bool:
    return not after.strip() or after.strip() in before


def _v2_patch_binding_text(before: Any, after: Any) -> tuple[str, str]:
    if isinstance(before, str) and isinstance(after, str):
        return before, after
    if (
        isinstance(before, list)
        and isinstance(after, list)
        and all(isinstance(item, str) for item in before)
        and all(isinstance(item, str) for item in after)
    ):
        return (
            canonical_json_bytes(before).decode("utf-8"),
            canonical_json_bytes(after).decode("utf-8"),
        )
    raise SpecificityContractError(
        "revision v2 field diffs can only modify str or list[str] fields without type drift"
    )


def _record_v2_failure(client: AuditedLLMClient, error: Exception) -> None:
    store = getattr(client, "_store", None)
    if store is None:
        raise error
    calls_path = store.root / "calls"
    call_dirs = [
        path
        for path in calls_path.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    if not call_dirs:
        raise error
    call_dir = max(call_dirs, key=lambda path: int(path.name))
    request_path = call_dir / "request_manifest.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))

    failure_dir = store.root / "failures"
    failure_files = [
        path
        for path in failure_dir.iterdir()
        if path.is_file() and path.stem.isdigit()
    ]
    sequence = max((int(path.stem) for path in failure_files), default=0) + 1
    record = {
        "run_id": store.manifest.decision_run_id,
        "sequence": sequence,
        "action": "specificity_revision_v2",
        "stage": DecisionRunStage.SPECIFICITY_REVISION.value,
        "call_id": request["call_id"],
        "error_type": type(error).__name__,
        "error_message": str(error) or type(error).__name__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (failure_dir / f"{sequence}.json").write_bytes(canonical_json_bytes(record))


def _validate_v2_response_reason(response: SpecificityFindingResponseV2) -> None:
    reason = response.reason.strip()
    if not reason:
        raise SpecificityContractError("revision response reason must not be blank")
    if response.disposition == "REJECT" and not any(
        marker in reason for marker in ("证据", "事实", "边界")
    ):
        raise SpecificityContractError("REJECT reason must state its evidence, fact, or boundary")
    if response.disposition == "DEFER" and (
        "验证" not in reason or not any(marker in reason for marker in ("?", "？"))
    ):
        raise SpecificityContractError("DEFER reason must state a validation question")


def _validate_v2_revision_contract(
    audit: SpecificityAuditBatch,
    revision_batch: SpecificityRevisionBatchV2,
    candidates: dict[str, NarrativeCandidate],
    comment_ids: set[str],
    public_evidence_ids: set[str],
) -> None:
    if audit.round_index > 2:
        raise SpecificityContractError("specificity revision v2 allows at most two revision rounds")
    if revision_batch.run_id != audit.run_id or revision_batch.round_index != audit.round_index:
        raise SpecificityContractError("revision v2 batch identity does not match audit")

    substantive_by_candidate = {
        candidate_audit.candidate_id: {
            finding.finding_id
            for finding in candidate_audit.findings
            if finding.severity.value != "NOTE"
        }
        for candidate_audit in audit.audits
    }
    required_candidates = {
        candidate_id
        for candidate_id, finding_ids in substantive_by_candidate.items()
        if finding_ids
    }
    revision_ids = [revision.candidate_id for revision in revision_batch.revisions]
    if len(revision_ids) != len(set(revision_ids)):
        raise SpecificityContractError("each candidate can have only one v2 revision")
    if set(revision_ids) != required_candidates:
        raise SpecificityContractError("v2 revisions must cover every candidate with a substantive finding")

    all_findings = _audit_findings(audit)
    for revision in revision_batch.revisions:
        candidate = candidates.get(revision.candidate_id)
        if candidate is None:
            raise SpecificityContractError("revision v2 contains an unknown candidate")
        expected_version = audit.candidate_versions.get(revision.candidate_id)
        if revision.from_version != expected_version or revision.to_version != revision.from_version + 1:
            raise SpecificityContractError("revision v2 versions do not match the audit")

        expected_finding_ids = substantive_by_candidate[revision.candidate_id]
        response_ids = [response.finding_id for response in revision.responses]
        if len(response_ids) != len(set(response_ids)) or set(response_ids) != expected_finding_ids:
            raise SpecificityContractError("every substantive finding requires exactly one response")
        if any(finding_id not in all_findings for finding_id in response_ids):
            raise SpecificityContractError("revision v2 response references an unknown finding")
        for response in revision.responses:
            _validate_v2_response_reason(response)

        accepted_ids = {
            response.finding_id
            for response in revision.responses
            if response.disposition == "ACCEPT"
        }
        unresolved_ids = {
            response.finding_id
            for response in revision.responses
            if response.disposition in {"REJECT", "DEFER"}
        }
        if len(revision.unresolved_finding_ids) != len(set(revision.unresolved_finding_ids)):
            raise SpecificityContractError("unresolved finding IDs must be unique")
        if set(revision.unresolved_finding_ids) != unresolved_ids:
            raise SpecificityContractError("REJECT and DEFER findings must remain unresolved")

        original = candidate.model_dump(mode="json")
        revised = revision.revised_candidate.model_dump(mode="json")
        if revision.revised_candidate.candidate_id != revision.candidate_id:
            raise SpecificityContractError("revision v2 candidate identity does not match")
        changed_fields = {field for field in original if original[field] != revised[field]}
        if not changed_fields.issubset(ALLOWED_NARRATIVE_PATCH_FIELDS):
            raise SpecificityContractError("revision v2 changed a protected candidate field")
        if accepted_ids and not changed_fields:
            raise SpecificityContractError("every ACCEPT response must actually change candidate text")

        diff_by_field: dict[str, SpecificityEvidenceBindingV2] = {}
        accepted_bound_ids: set[str] = set()
        for diff in revision.field_diffs:
            if diff.field in diff_by_field:
                raise SpecificityContractError("revision v2 field diffs must be unique")
            if diff.field not in ALLOWED_NARRATIVE_PATCH_FIELDS:
                raise SpecificityContractError("revision v2 field diff is not allowed")
            expected_before, expected_after = _v2_patch_binding_text(
                original[diff.field], revised[diff.field]
            )
            if diff.before != expected_before or diff.after != expected_after:
                raise SpecificityContractError("revision v2 field diff does not match candidate text")
            source_ids = set(diff.source_finding_ids)
            if not source_ids or not source_ids.issubset(accepted_ids):
                raise SpecificityContractError("field diff must bind an ACCEPT finding")
            if not diff.comment_evidence_ids and (
                not diff.evidence_gap_ids or not _is_narrowing(diff.before, diff.after)
            ):
                raise SpecificityContractError("every field diff must bind a comment ID")
            if not set(diff.comment_evidence_ids).issubset(comment_ids):
                raise SpecificityContractError("field diff references an unknown comment ID")
            if not set(diff.public_evidence_ids).issubset(public_evidence_ids):
                raise SpecificityContractError("field diff references unknown public evidence")
            if diff.evidence_gap_ids and not _is_narrowing(diff.before, diff.after):
                raise SpecificityContractError("evidence gaps may only delete or narrow a claim")
            diff_by_field[diff.field] = diff
            accepted_bound_ids.update(source_ids)
        if set(diff_by_field) != changed_fields:
            raise SpecificityContractError("field diffs must cover every changed field")
        if accepted_ids != accepted_bound_ids:
            raise SpecificityContractError("every ACCEPT response must actually change candidate text")

        fact_ids = [fact.fact_id for fact in revision.new_brand_facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise SpecificityContractError("new brand fact IDs must be unique")
        for fact in revision.new_brand_facts:
            if not fact.statement.strip() or not fact.reason.strip():
                raise SpecificityContractError("new brand facts must contain text and reason")
            if not fact.public_evidence_ids:
                raise SpecificityContractError("new brand facts require public evidence")
            if not set(fact.public_evidence_ids).issubset(public_evidence_ids):
                raise SpecificityContractError("new brand fact references unknown public evidence")


def revise_from_specificity_audit_v2(
    client: AuditedLLMClient,
    audit: SpecificityAuditBatch,
    candidates: list[NarrativeCandidate],
    comment_evidence: list[EvidenceAtom],
    public_evidence: list[PublicNarrativeEvidence],
) -> SpecificityRevisionBatchV2:
    """Apply at most two strictly evidence-bound DeepSeek revision rounds."""

    if not isinstance(client, AuditedLLMClient):
        raise TypeError("client must be an AuditedLLMClient")
    if not isinstance(audit, SpecificityAuditBatch):
        raise TypeError("audit must be a SpecificityAuditBatch")
    _typed_list(candidates, NarrativeCandidate, "candidates")
    _typed_list(comment_evidence, EvidenceAtom, "comment_evidence")
    _typed_list(public_evidence, PublicNarrativeEvidence, "public_evidence")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)) or set(candidate_ids) != set(audit.candidate_ids):
        raise SpecificityContractError("candidates must exactly match audit candidates")
    comment_ids = [evidence.comment_id for evidence in comment_evidence]
    if len(comment_ids) != len(set(comment_ids)):
        raise SpecificityContractError("comment evidence must have unique comment IDs")
    public_ids = [evidence.evidence_id for evidence in public_evidence]
    if len(public_ids) != len(set(public_ids)):
        raise SpecificityContractError("public evidence IDs must be unique")
    if audit.round_index > 2:
        raise SpecificityContractError("specificity revision v2 allows at most two revision rounds")
    _validate_v2_deepseek_profile(client)

    request_payload = {
        "audit": audit.model_dump(mode="json"),
        "candidate_versions": dict(sorted(audit.candidate_versions.items())),
        "candidates": [
            candidate.model_dump(mode="json")
            for candidate in sorted(candidates, key=lambda item: item.candidate_id)
        ],
        "comment_evidence": [
            evidence.model_dump(mode="json")
            for evidence in sorted(comment_evidence, key=lambda item: item.comment_id)
        ],
        "public_evidence": [
            evidence.model_dump(mode="json")
            for evidence in sorted(public_evidence, key=lambda item: item.evidence_id)
        ],
    }
    _reject_forbidden(request_payload)
    try:
        with client.stage(DecisionRunStage.SPECIFICITY_REVISION):
            result = client.generate_json(
                system_prompt=_v2_prompt_text(),
                user_prompt=canonical_json_bytes(request_payload).decode("utf-8"),
                response_model=SpecificityRevisionBatchV2,
            )
        if not isinstance(result, SpecificityRevisionBatchV2):
            raise SpecificityContractError("DeepSeek did not return SpecificityRevisionBatchV2")
        validated = _strict_json_model(result.model_dump(mode="json"), SpecificityRevisionBatchV2)
        _validate_v2_revision_contract(
            audit,
            validated,
            {candidate.candidate_id: candidate for candidate in candidates},
            set(comment_ids),
            set(public_ids),
        )
        return validated
    except SpecificityContractError as exc:
        _record_v2_failure(client, exc)
        raise
    except (TypeError, ValueError, ValidationError) as exc:
        wrapped = SpecificityContractError("invalid SpecificityRevisionBatchV2 output")
        _record_v2_failure(client, wrapped)
        raise wrapped from exc
    except Exception as exc:
        _record_v2_failure(client, exc)
        raise


def run_specificity_final_rank(
    client: AuditedLLMClient,
    *,
    run_id: str,
    candidates: list[NarrativeCandidate],
    candidate_versions: dict[str, int],
    input_sha256: str,
) -> SpecificityFinalRankResult:
    if not isinstance(client, AuditedLLMClient):
        raise TypeError("client must be an AuditedLLMClient")
    if not isinstance(run_id, str) or not run_id:
        raise SpecificityContractError("run_id must be a non-empty string")
    _typed_list(candidates, NarrativeCandidate, "candidates")
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise SpecificityContractError("final rank candidates must have unique IDs")
    if set(candidate_versions) != set(candidate_ids):
        raise SpecificityContractError("final rank versions must match final candidates")
    if any(not isinstance(version, int) or version < 1 for version in candidate_versions.values()):
        raise SpecificityContractError("final rank candidate versions must be positive integers")
    store = getattr(client, "_store", None)
    profile = getattr(getattr(store, "manifest", None), "decision_model", None)
    if profile is None or profile.provider != "deepseek":
        raise SpecificityContractError("specificity final rank must use the DeepSeek decision profile")
    request_payload = {
        "run_id": run_id,
        "candidate_ids": candidate_ids,
        "candidate_versions": dict(candidate_versions),
        "input_sha256": input_sha256,
    }
    with client.stage(DecisionRunStage.SPECIFICITY_FINAL_RANK):
        result = client.generate_json(
            system_prompt=SPECIFICITY_FINAL_RANK_PROMPT,
            user_prompt=canonical_json_bytes(request_payload).decode("utf-8"),
            response_model=SpecificityFinalRankResult,
        )
    if not isinstance(result, SpecificityFinalRankResult):
        raise SpecificityContractError("DeepSeek did not return SpecificityFinalRankResult")
    try:
        validated = _strict_json_model(result.model_dump(mode="json"), SpecificityFinalRankResult)
    except (TypeError, ValueError, ValidationError) as exc:
        raise SpecificityContractError("invalid SpecificityFinalRankResult output") from exc
    if validated.run_id != run_id or validated.input_sha256 != input_sha256:
        raise SpecificityContractError("final rank identity does not match its input")
    if validated.candidate_ids != candidate_ids:
        raise SpecificityContractError("final rank candidate order does not match its input")
    if validated.candidate_versions != candidate_versions:
        raise SpecificityContractError("final rank candidate versions do not match its input")
    return validated


def apply_specificity_revision(
    candidate: NarrativeCandidate,
    revision: SpecificityCandidateRevision,
) -> NarrativeCandidate:
    if not isinstance(candidate, NarrativeCandidate):
        raise TypeError("candidate must be a NarrativeCandidate")
    if not isinstance(revision, SpecificityCandidateRevision):
        raise TypeError("revision must be a SpecificityCandidateRevision")
    _validate_single_revision(revision, candidate)
    return revision.revised_candidate


def can_rebut_finding(
    history: CandidateSpecificityHistory,
    finding_id: str,
    has_new_evidence: bool,
) -> bool:
    if not isinstance(history, CandidateSpecificityHistory):
        raise TypeError("history must be a CandidateSpecificityHistory")
    if not has_new_evidence:
        return False
    reject_count = 0
    for specificity_round in history.rounds:
        if specificity_round.revision is None:
            continue
        for revision in specificity_round.revision.revisions:
            reject_count += sum(
                response.finding_id == finding_id
                and response.disposition is SpecificityDisposition.REJECT
                for response in revision.responses
            )
    return reject_count == 1


def derive_specificity_outcome(
    audit: CandidateSpecificityAudit,
    blocking_gate_id: str | None,
) -> SpecificityOutcome:
    if not isinstance(audit, CandidateSpecificityAudit):
        raise TypeError("audit must be a CandidateSpecificityAudit")
    if blocking_gate_id is not None:
        if not isinstance(blocking_gate_id, str) or not blocking_gate_id.strip():
            raise SpecificityContractError("blocking_gate_id must be a non-empty deterministic gate ID")
        return SpecificityOutcome.BLOCKED
    if any(finding.severity.value != "NOTE" for finding in audit.findings):
        return SpecificityOutcome.HUMAN_REVIEW
    return SpecificityOutcome.READY


def should_continue_specificity(
    audit: SpecificityAuditBatch,
    revision: SpecificityRevisionBatch | None,
    round_index: int,
) -> bool:
    if not isinstance(audit, SpecificityAuditBatch):
        raise TypeError("audit must be a SpecificityAuditBatch")
    if round_index not in {1, 2, 3}:
        raise ValueError("round_index must be 1..3")
    substantive_ids = {
        finding.finding_id
        for candidate_audit in audit.audits
        for finding in candidate_audit.findings
        if finding.severity.value != "NOTE"
    }
    if not substantive_ids or round_index >= 3:
        return False
    if revision is None:
        return True
    unresolved = {
        finding_id
        for candidate_revision in revision.revisions
        for finding_id in candidate_revision.unresolved_finding_ids
    }
    return bool(substantive_ids & unresolved)
