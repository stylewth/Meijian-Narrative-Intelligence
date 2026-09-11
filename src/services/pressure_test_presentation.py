"""Read-only projection of the frozen five-candidate pressure test."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from src.schemas import (
    FiveCandidateHoldoutResultV2,
    FiveCandidateSharedInputV2,
    FiveCandidateUnifiedEvaluationV2,
    SpecificityAuditBatch,
)
from src.services.brand_specificity import SpecificityRevisionBatchV2
from src.services.five_candidate_validation import load_parent_attempt


CHECK_LABELS = {
    "EVIDENCE_COVERAGE": "证据覆盖",
    "COUNTER_EVIDENCE": "反证压力",
    "COMPETITOR_SUBSTITUTION": "竞品替代",
    "PRODUCT_GROUNDING": "产品锚定",
    "ROBUSTNESS": "鲁棒性",
}
_CHECK_AUDIT_TYPES = {
    "COMPETITOR_SUBSTITUTION": "COMPETITOR_REPLACEMENT",
    "PRODUCT_GROUNDING": "BRAND_ASSET",
    "ROBUSTNESS": "FAILURE_PREMORTEM",
}
_METHOD = "AI 推荐 → 真人盲评验证 → 团队确认"
_ROUND_PATTERN = re.compile(r"round-(\d{2})$")
_FIELD_LABELS = {
    "title": "机会标题",
    "target_audience": "目标人群",
    "user_conflict": "用户矛盾",
    "brand_opportunity": "品牌机会",
    "why_brand": "品牌理由",
    "why_meijian": "梅见理由",
    "competitor_difference": "竞品差异",
    "brand_role": "品牌角色",
    "draft_proposition": "提案表述",
    "main_scenes": "落地场景",
    "content_theme": "内容主题",
    "risks": "风险边界",
}


class PressurePresentationError(ValueError):
    """A frozen pressure-test artifact cannot be safely displayed."""


@dataclass(frozen=True, slots=True)
class PressureCheckView:
    check_type: str
    label: str
    status: str
    rationale: str
    reference_ids: tuple[str, ...]
    source_path: Path


@dataclass(frozen=True, slots=True)
class EvidenceExcerptView:
    evidence_id: str
    comment_id: str
    raw_content: str
    source_platform: str
    source_type: str


@dataclass(frozen=True, slots=True)
class AgentFieldDiffView:
    field_label: str
    before: str
    after: str
    reason: str


@dataclass(frozen=True, slots=True)
class AgentMessageView:
    round_index: int
    agent: str
    kind: str
    headline: str
    summary: str
    body: str
    highlights: tuple[str, ...]
    field_diffs: tuple[AgentFieldDiffView, ...]
    source_path: Path


@dataclass(frozen=True, slots=True)
class PressureCandidateView:
    candidate_id: str
    title: str
    candidate_json: str
    checks: tuple[PressureCheckView, ...]
    dialogue: tuple[AgentMessageView, ...]
    rank: int
    overall_assessment: str


@dataclass(frozen=True, slots=True)
class HoldoutCandidateView:
    candidate_id: str
    support_count: int
    challenge_count: int
    neutral_count: int
    blocking_findings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HoldoutView:
    result_path: Path
    candidate_ids: tuple[str, ...]
    candidates: tuple[HoldoutCandidateView, ...]
    result_json: str


@dataclass(frozen=True, slots=True)
class BlindSelectionView:
    selection_path: Path
    method: str
    selected_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BlindResponseView:
    response_id: str
    text: str


@dataclass(frozen=True, slots=True)
class BlindSelectionMetricView:
    opportunity_id: str
    label: str
    n: int
    selected: int
    share: float


@dataclass(frozen=True, slots=True)
class BlindReviewView:
    summary_path: Path
    survey_count: int
    open_response_count: int
    open_responses: tuple[BlindResponseView, ...]
    selection_metrics: tuple[BlindSelectionMetricView, ...]


@dataclass(frozen=True, slots=True)
class PressureRunView:
    validation_root: Path
    selection_path: Path
    shared_input: FiveCandidateSharedInputV2
    evaluation: FiveCandidateUnifiedEvaluationV2
    candidates: tuple[PressureCandidateView, ...]
    holdout: HoldoutView
    selection: BlindSelectionView
    blind_review: BlindReviewView
    evidence_catalog: Mapping[str, EvidenceExcerptView]


def load_pressure_test_run(
    validation_root: str | Path,
    selection_path: str | Path,
) -> PressureRunView:
    """Load only the two explicitly supplied frozen artifact locations."""

    root = Path(validation_root)
    selection = Path(selection_path)
    shared_path = root / "shared_input.json"
    shared_input = _load_formal_json(shared_path, FiveCandidateSharedInputV2, "shared input")
    candidate_hint = ", ".join(shared_input.candidate_ids)

    try:
        parent = load_parent_attempt(root)
    except Exception as exc:
        raise PressurePresentationError(
            f"invalid validation parent {root}; candidates [{candidate_hint}]: {exc}"
        ) from exc

    evaluation_path = root / "evaluation" / "result.json"
    evaluation = _load_formal_json(
        evaluation_path,
        FiveCandidateUnifiedEvaluationV2,
        "unified evaluation",
        candidate_hint=candidate_hint,
    )
    holdout_path = root / "holdout" / "result.json"
    holdout_result = _load_formal_json(
        holdout_path,
        FiveCandidateHoldoutResultV2,
        "HOLDOUT result",
        candidate_hint=candidate_hint,
    )
    _require_same_candidates(
        shared_input.candidate_ids,
        evaluation.candidate_ids,
        evaluation_path,
        "unified evaluation",
    )
    _require_same_candidates(
        shared_input.candidate_ids,
        holdout_result.candidate_ids,
        holdout_path,
        "HOLDOUT result",
    )

    candidate_views = tuple(
        _candidate_view(
            parent=parent,
            candidate_id=candidate_id,
            evaluation=evaluation,
        )
        for candidate_id in shared_input.candidate_ids
    )
    holdout = _holdout_view(holdout_path, holdout_result)
    selection_view = _selection_view(selection, shared_input.candidate_ids)
    blind_review = _blind_review_view(
        selection.parent / "blind_summary.json",
        expected_run_id=selection.parent.parent.name,
    )
    displayed_evidence_ids = sorted(
        {
            reference
            for candidate in candidate_views
            for check in candidate.checks
            if check.check_type == "EVIDENCE_COVERAGE"
            for reference in check.reference_ids
        }
    )
    evidence_catalog = _load_evidence_catalog(displayed_evidence_ids)
    return PressureRunView(
        validation_root=root,
        selection_path=selection,
        shared_input=shared_input,
        evaluation=evaluation,
        candidates=candidate_views,
        holdout=holdout,
        selection=selection_view,
        blind_review=blind_review,
        evidence_catalog=evidence_catalog,
    )


def _load_evidence_catalog(
    expected_evidence_ids: list[str],
) -> Mapping[str, EvidenceExcerptView]:
    catalog_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "submission"
        / "evidence_catalog.json"
    )
    catalog: dict[str, EvidenceExcerptView] = {}
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"schema_version", "evidence"}:
            raise ValueError("submission evidence catalog 顶层结构无效")
        if payload["schema_version"] != "submission_evidence_catalog_v1":
            raise ValueError("submission evidence catalog 版本无效")
        entries = payload["evidence"]
        if not isinstance(entries, list):
            raise ValueError("submission evidence catalog evidence 必须是列表")
        required_keys = {
            "evidence_id",
            "comment_id",
            "raw_content",
            "source_platform",
            "source_type",
        }
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != required_keys:
                raise ValueError("submission evidence catalog 条目结构无效")
            if any(not isinstance(entry[key], str) or not entry[key] for key in required_keys):
                raise ValueError("submission evidence catalog 条目必须全部为非空字符串")
            evidence_id = entry["evidence_id"]
            if evidence_id in catalog:
                raise ValueError(f"重复 evidence_id：{evidence_id}")
            catalog[evidence_id] = EvidenceExcerptView(**entry)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise PressurePresentationError(
            f"invalid submission evidence catalog: {catalog_path}; {exc}"
        ) from exc
    expected = set(expected_evidence_ids)
    actual = set(catalog)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise PressurePresentationError(
            "submission evidence catalog 与展示引用不一致："
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return MappingProxyType(catalog)


def _load_formal_json(
    path: Path,
    model: type[Any],
    label: str,
    *,
    candidate_hint: str = "",
) -> Any:
    if path.is_symlink() or not path.is_file():
        raise PressurePresentationError(
            f"missing {label}: {path}; candidate(s): {candidate_hint}"
        )
    try:
        data = path.read_bytes()
        return model.model_validate_json(data, strict=True)
    except (OSError, UnicodeDecodeError, ValueError, TypeError, ValidationError) as exc:
        raise PressurePresentationError(
            f"invalid {label}: {path}; candidate(s): {candidate_hint}"
        ) from exc


def _require_same_candidates(
    expected: list[str],
    actual: list[str],
    path: Path,
    label: str,
) -> None:
    if tuple(expected) != tuple(actual):
        hint = ", ".join(expected)
        raise PressurePresentationError(
            f"{label} candidate IDs disagree at {path}; candidate(s): {hint}"
        )


def _candidate_view(
    *,
    parent: Any,
    candidate_id: str,
    evaluation: FiveCandidateUnifiedEvaluationV2,
) -> PressureCandidateView:
    try:
        candidate_root = parent.child_root(candidate_id)
        state = parent.load_child_state(candidate_id)
    except Exception as exc:
        raise PressurePresentationError(
            f"cannot load candidate {candidate_id} at {parent.root}; {exc}"
        ) from exc

    rounds = _round_directories(candidate_root / "rounds", candidate_id)
    dialogue: list[AgentMessageView] = []
    latest_luna_payload: dict[str, Any] | None = None
    latest_luna_path: Path | None = None
    expected_version = 1
    for round_index, round_dir in rounds:
        deepseek_paths = _display_deepseek_paths(round_dir)
        revision_path = round_dir / "revision.json"
        luna_path = round_dir / "luna_result.json"
        has_revision = revision_path.is_file() and not revision_path.is_symlink()
        if len(deepseek_paths) > 1:
            _candidate_error(
                candidate_id,
                round_dir,
                "multiple deepseek_*result.json files",
            )
        luna = _load_formal_json(
            luna_path,
            SpecificityAuditBatch,
            "Luna dialogue",
            candidate_hint=candidate_id,
        )
        _validate_audit_binding(
            luna,
            luna_path,
            candidate_id,
            parent.attempt_id,
            round_index,
            expected_version,
        )

        if deepseek_paths:
            deepseek_path = deepseek_paths[0]
            deepseek = _load_formal_json(
                deepseek_path,
                SpecificityRevisionBatchV2,
                "DeepSeek decision",
                candidate_hint=candidate_id,
            )
            _validate_revision_binding(
                deepseek,
                deepseek_path,
                candidate_id,
                parent.attempt_id,
                round_index,
                expected_version,
            )
        if has_revision:
            revision = _load_formal_json(
                revision_path,
                SpecificityRevisionBatchV2,
                "DeepSeek revision",
                candidate_hint=candidate_id,
            )
            _validate_revision_binding(
                revision,
                revision_path,
                candidate_id,
                parent.attempt_id,
                round_index,
                expected_version,
            )

        dialogue.append(
            _review_message_view(
                luna,
                round_index=round_index,
                source_path=luna_path,
            )
        )
        if deepseek_paths:
            dialogue.append(
                _revision_message_view(
                    deepseek,
                    round_index=round_index,
                    kind="decision",
                    source_path=deepseek_path,
                )
            )
        if has_revision:
            dialogue.append(
                _revision_message_view(
                    revision,
                    round_index=round_index,
                    kind="revision",
                    source_path=revision_path,
                )
            )
            expected_version += 1
        latest_luna_payload = luna.model_dump(mode="json")
        latest_luna_path = luna_path

    if latest_luna_payload is None or latest_luna_path is None:
        _candidate_error(candidate_id, candidate_root / "rounds", "no Luna review")
    if expected_version != state.current_version:
        _candidate_error(
            candidate_id,
            candidate_root / "state.json",
            "dialogue version does not match child state",
        )
    checks = _checks_for_candidate(
        candidate_id,
        state.current_candidate,
        latest_luna_payload,
        latest_luna_path,
    )
    evaluation_item = next(
        item for item in evaluation.evaluations if item.candidate_id == candidate_id
    )
    rank = evaluation.ranked_candidate_ids.index(candidate_id) + 1
    return PressureCandidateView(
        candidate_id=candidate_id,
        title=state.current_candidate.title,
        candidate_json=state.current_candidate.model_dump_json(),
        checks=checks,
        dialogue=tuple(dialogue),
        rank=rank,
        overall_assessment=evaluation_item.overall_assessment,
    )


def _round_directories(rounds_root: Path, candidate_id: str) -> list[tuple[int, Path]]:
    if rounds_root.is_symlink() or not rounds_root.is_dir():
        _candidate_error(candidate_id, rounds_root, "missing rounds directory")
    directories = sorted(
        (path for path in rounds_root.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.name,
    )
    parsed: list[tuple[int, Path]] = []
    for directory in directories:
        match = _ROUND_PATTERN.fullmatch(directory.name)
        if match is None:
            _candidate_error(candidate_id, directory, "invalid round directory")
        parsed.append((int(match.group(1)), directory))
    numbers = [number for number, _ in parsed]
    if not parsed or numbers != list(range(1, len(parsed) + 1)):
        _candidate_error(candidate_id, rounds_root, "dialogue rounds are not contiguous")
    return parsed


def _display_deepseek_paths(round_dir: Path) -> list[Path]:
    """Select only the frozen DeepSeek response meant for presentation."""

    for filename in ("deepseek_demo_repaired.json", "deepseek_model_result.json"):
        path = round_dir / filename
        if path.is_file() and not path.is_symlink():
            return [path]
    return []


def _read_object(path: Path, candidate_id: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        _candidate_error(candidate_id, path, "missing frozen JSON")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PressurePresentationError(
            f"invalid frozen JSON at {path}; candidate {candidate_id}"
        ) from exc
    if not isinstance(value, dict):
        _candidate_error(candidate_id, path, "frozen JSON must be an object")
    return value


def _read_message(path: Path, candidate_id: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PressurePresentationError(
            f"cannot read frozen message at {path}; candidate {candidate_id}"
        ) from exc


def _as_display_text(value: object) -> str:
    if isinstance(value, list):
        return "；".join(str(item) for item in value)
    return str(value)


def _revision_message_view(
    batch: SpecificityRevisionBatchV2,
    *,
    round_index: int,
    kind: str,
    source_path: Path,
) -> AgentMessageView:
    item = batch.revisions[0]
    diffs = tuple(
        AgentFieldDiffView(
            field_label=_FIELD_LABELS.get(diff.field, diff.field),
            before=_as_display_text(diff.before),
            after=_as_display_text(diff.after),
            reason=diff.reason,
        )
        for diff in item.field_diffs
    )
    accepted = sum(response.disposition == "ACCEPT" for response in item.responses)
    deferred = sum(response.disposition != "ACCEPT" for response in item.responses)
    if kind == "decision":
        headline = "决策 Agent 提交本轮修订方案"
        summary = (
            f"回应 {len(item.responses)} 项审查意见，接受 {accepted} 项；"
            f"计划调整 {len(diffs)} 个叙事字段。"
        )
    else:
        headline = "决策 Agent 写入经审查的版本"
        summary = (
            f"候选由 V{item.from_version} 更新至 V{item.to_version}，"
            f"落实 {len(diffs)} 处修改，保留 {deferred} 项未采纳意见。"
        )
    highlights = tuple(response.reason for response in item.responses[:3])
    return AgentMessageView(
        round_index=round_index,
        agent="DeepSeek",
        kind=kind,
        headline=headline,
        summary=summary,
        body=f"{headline}。{summary}",
        highlights=highlights,
        field_diffs=() if kind == "decision" else diffs,
        source_path=source_path,
    )


def _review_message_view(
    batch: SpecificityAuditBatch,
    *,
    round_index: int,
    source_path: Path,
) -> AgentMessageView:
    audit = batch.audits[0]
    material = [
        finding for finding in audit.findings if finding.severity.value == "MATERIAL_RISK"
    ]
    headline = (
        f"审查 Agent 发现 {len(material)} 项实质风险"
        if material
        else "审查 Agent 确认风险已收敛"
    )
    summary = audit.next_validation_question
    highlights = tuple(
        finding.claim for finding in (material or audit.findings)[:3]
    )
    return AgentMessageView(
        round_index=round_index,
        agent="Luna",
        kind="review",
        headline=headline,
        summary=summary,
        body=f"{headline}。下一步验证：{summary}",
        highlights=highlights,
        field_diffs=(),
        source_path=source_path,
    )


def _validate_audit_binding(
    audit: SpecificityAuditBatch,
    path: Path,
    candidate_id: str,
    run_id: str,
    round_index: int,
    expected_version: int,
) -> None:
    if audit.run_id != run_id:
        _candidate_error(candidate_id, path, "Luna run_id does not match validation run")
    if audit.round_index != round_index:
        _candidate_error(candidate_id, path, "Luna round_index does not match round directory")
    if audit.candidate_ids != [candidate_id]:
        _candidate_error(candidate_id, path, "Luna candidate IDs do not match candidate directory")
    if len(audit.audits) != 1 or audit.audits[0].candidate_id != candidate_id:
        _candidate_error(candidate_id, path, "Luna audit candidate does not match candidate directory")
    if audit.candidate_versions.get(candidate_id) != expected_version:
        _candidate_error(candidate_id, path, "Luna candidate version does not follow dialogue rounds")
    if audit.audits[0].candidate_version != expected_version:
        _candidate_error(candidate_id, path, "Luna audit version does not follow dialogue rounds")


def _validate_revision_binding(
    revision: SpecificityRevisionBatchV2,
    path: Path,
    candidate_id: str,
    run_id: str,
    round_index: int,
    expected_version: int,
) -> None:
    if revision.run_id != run_id:
        _candidate_error(candidate_id, path, "DeepSeek run_id does not match validation run")
    if revision.round_index != round_index:
        _candidate_error(candidate_id, path, "DeepSeek round_index does not match round directory")
    if len(revision.revisions) != 1:
        _candidate_error(candidate_id, path, "DeepSeek revision must contain exactly one candidate")
    item = revision.revisions[0]
    if item.candidate_id != candidate_id:
        _candidate_error(candidate_id, path, "DeepSeek revision candidate does not match candidate directory")
    if item.from_version != expected_version or item.to_version != expected_version + 1:
        _candidate_error(candidate_id, path, "DeepSeek revision versions do not follow dialogue rounds")


def _checks_for_candidate(
    candidate_id: str,
    candidate: Any,
    luna_payload: dict[str, Any],
    luna_path: Path,
) -> tuple[PressureCheckView, ...]:
    audits = luna_payload.get("audits")
    if not isinstance(audits, list):
        _candidate_error(candidate_id, luna_path, "Luna result has no audits")
    audit = next(
        (item for item in audits if isinstance(item, dict) and item.get("candidate_id") == candidate_id),
        None,
    )
    if audit is None:
        _candidate_error(candidate_id, luna_path, "Luna result has no candidate audit")
    findings = audit.get("findings")
    if not isinstance(findings, list):
        _candidate_error(candidate_id, luna_path, "candidate audit has no findings")

    views: list[PressureCheckView] = []
    for check_type, label in CHECK_LABELS.items():
        if check_type == "EVIDENCE_COVERAGE":
            references = tuple(str(item) for item in audit.get("consumer_evidence", []) if isinstance(item, str))
            rationale = "、".join(references) or "未记录消费者证据"
            status = "RECORDED" if references else "MISSING"
        elif check_type == "COUNTER_EVIDENCE":
            references = tuple(item.comment_id for item in candidate.counter_evidence)
            rationale = candidate.counter_evidence_note or "未记录反证说明"
            status = "RECORDED" if references or candidate.counter_evidence_note else "MISSING"
        else:
            audit_type = _CHECK_AUDIT_TYPES[check_type]
            matching = [
                item for item in findings
                if isinstance(item, dict) and item.get("audit_type") == audit_type
            ]
            if not matching:
                _candidate_error(candidate_id, luna_path, f"missing audit type {audit_type}")
            references = tuple(
                str(reference)
                for item in matching
                for reference in item.get("evidence_ids", [])
                if isinstance(reference, str)
            )
            rationale = "\n".join(
                str(item.get("claim") or item.get("failure_mode") or "")
                for item in matching
            ).strip()
            status = (
                "MATERIAL_RISK"
                if any(item.get("severity") == "MATERIAL_RISK" for item in matching)
                else "NOTE"
            )
        views.append(
            PressureCheckView(
                check_type=check_type,
                label=label,
                status=status,
                rationale=rationale,
                reference_ids=tuple(dict.fromkeys(references)),
                source_path=luna_path,
            )
        )
    return tuple(views)


def _holdout_view(path: Path, result: FiveCandidateHoldoutResultV2) -> HoldoutView:
    try:
        result_json = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PressurePresentationError(f"cannot read HOLDOUT result: {path}") from exc
    candidates = []
    for candidate_id in result.candidate_ids:
        impacts = [impact for impact in result.impacts if impact.candidate_id == candidate_id]
        blocking = tuple(impact.rationale for impact in impacts if impact.is_blocking)
        candidates.append(
            HoldoutCandidateView(
                candidate_id=candidate_id,
                support_count=sum(impact.impact == "SUPPORT" for impact in impacts),
                challenge_count=sum(impact.impact == "CHALLENGE" for impact in impacts),
                neutral_count=sum(impact.impact == "NEUTRAL" for impact in impacts),
                blocking_findings=blocking,
            )
        )
    return HoldoutView(
        result_path=path,
        candidate_ids=tuple(result.candidate_ids),
        candidates=tuple(candidates),
        result_json=result_json,
    )


def _selection_view(path: Path, candidate_ids: list[str]) -> BlindSelectionView:
    payload = _read_json_object(path, "blind selection")
    if path.parent.name != "blind" or path.parent.parent.name == "":
        raise PressurePresentationError(
            f"blind selection path must be <run>/blind/selection_confirmation.json: {path}"
        )
    expected_run_id = path.parent.parent.name
    if payload.get("run_id") != expected_run_id:
        raise PressurePresentationError(
            f"blind selection run_id does not match selection run at {path}; "
            f"candidate(s): {', '.join(candidate_ids)}"
        )
    if payload.get("status") != "CONFIRMED":
        raise PressurePresentationError(
            f"blind selection status must be CONFIRMED: {path}; "
            f"candidate(s): {', '.join(candidate_ids)}"
        )
    selected = payload.get("selected_candidate_ids")
    if not isinstance(selected, list) or len(selected) != 3 or len(set(selected)) != 3:
        raise PressurePresentationError(
            f"blind selection must contain three unique candidate IDs: {path}; candidate(s): {', '.join(candidate_ids)}"
        )
    if not set(selected).issubset(set(candidate_ids)):
        raise PressurePresentationError(
            f"blind selection contains unknown candidate at {path}; candidate(s): {', '.join(candidate_ids)}"
        )
    selections = payload.get("selections")
    if not isinstance(selections, list):
        raise PressurePresentationError(
            f"blind selection must contain selection records: {path}; candidate(s): {', '.join(candidate_ids)}"
        )
    selection_ids = [
        item.get("candidate_id")
        for item in selections
        if isinstance(item, dict)
    ]
    if (
        len(selection_ids) != len(selections)
        or len(selection_ids) != len(set(selection_ids))
        or set(selection_ids) != set(selected)
    ):
        raise PressurePresentationError(
            f"blind selection records do not match selected candidate IDs: {path}; "
            f"candidate(s): {', '.join(candidate_ids)}"
        )
    return BlindSelectionView(path, _METHOD, tuple(selected))


def _blind_review_view(path: Path, *, expected_run_id: str) -> BlindReviewView:
    payload = _read_json_object(path, "blind summary")
    if payload.get("run_id") != expected_run_id:
        raise PressurePresentationError(
            f"blind summary run_id does not match selection run: {path}"
        )

    sample_counts = payload.get("sample_counts")
    q1_8 = payload.get("q1_8")
    q9_12 = payload.get("q9_12")
    if not all(isinstance(item, dict) for item in (sample_counts, q1_8, q9_12)):
        raise PressurePresentationError(f"blind summary sections are invalid: {path}")
    survey_count = sample_counts.get("q1_8")
    open_response_count = sample_counts.get("q9_12")
    if (
        not isinstance(survey_count, int)
        or isinstance(survey_count, bool)
        or survey_count < 1
        or not isinstance(open_response_count, int)
        or isinstance(open_response_count, bool)
        or open_response_count < 1
    ):
        raise PressurePresentationError(f"blind summary sample counts are invalid: {path}")

    raw_responses = q9_12.get("open_responses")
    if not isinstance(raw_responses, list) or len(raw_responses) != open_response_count:
        raise PressurePresentationError(f"blind open response count is invalid: {path}")
    responses: list[BlindResponseView] = []
    response_ids: set[str] = set()
    for index, item in enumerate(raw_responses):
        if not isinstance(item, dict) or set(item) != {"id", "text"}:
            raise PressurePresentationError(
                f"blind open response {index} is invalid: {path}"
            )
        response_id = item["id"]
        text = item["text"]
        if (
            not isinstance(response_id, str)
            or not response_id
            or response_id in response_ids
            or not isinstance(text, str)
            or not text
        ):
            raise PressurePresentationError(
                f"blind open response {index} is invalid: {path}"
            )
        response_ids.add(response_id)
        responses.append(BlindResponseView(response_id=response_id, text=text))

    raw_metrics = q1_8.get("q6")
    if not isinstance(raw_metrics, list) or len(raw_metrics) != 5:
        raise PressurePresentationError(f"blind q6 metrics are invalid: {path}")
    metrics: list[BlindSelectionMetricView] = []
    metric_ids: set[str] = set()
    for index, item in enumerate(raw_metrics):
        if not isinstance(item, dict):
            raise PressurePresentationError(f"blind q6 metric {index} is invalid: {path}")
        opportunity_id = item.get("opportunity_id")
        label = item.get("label")
        n = item.get("n")
        selected = item.get("selected")
        share = item.get("share")
        if (
            not isinstance(opportunity_id, str)
            or not opportunity_id
            or opportunity_id in metric_ids
            or not isinstance(label, str)
            or not label
            or not isinstance(n, int)
            or isinstance(n, bool)
            or n != survey_count
            or not isinstance(selected, int)
            or isinstance(selected, bool)
            or not 0 <= selected <= n
            or not isinstance(share, (int, float))
            or isinstance(share, bool)
            or abs(float(share) - selected / n) > 1e-12
        ):
            raise PressurePresentationError(f"blind q6 metric {index} is invalid: {path}")
        metric_ids.add(opportunity_id)
        metrics.append(
            BlindSelectionMetricView(
                opportunity_id=opportunity_id,
                label=label,
                n=n,
                selected=selected,
                share=float(share),
            )
        )

    return BlindReviewView(
        summary_path=path,
        survey_count=survey_count,
        open_response_count=open_response_count,
        open_responses=tuple(responses),
        selection_metrics=tuple(metrics),
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PressurePresentationError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PressurePresentationError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PressurePresentationError(f"{label} must be a JSON object: {path}")
    return value


def _candidate_error(candidate_id: str, path: Path, message: str) -> None:
    raise PressurePresentationError(f"{message} at {path}; candidate {candidate_id}")


__all__ = [
    "AgentMessageView",
    "BlindResponseView",
    "BlindReviewView",
    "BlindSelectionView",
    "BlindSelectionMetricView",
    "CHECK_LABELS",
    "HoldoutView",
    "PressureCandidateView",
    "EvidenceExcerptView",
    "PressureCheckView",
    "PressurePresentationError",
    "PressureRunView",
    "load_pressure_test_run",
]
