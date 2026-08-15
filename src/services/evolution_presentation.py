from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.schemas import CandidateScores
from src.services.candidate_ranking import calculate_weighted_score


class EvolutionPresentationError(ValueError):
    """Raised when the frozen evolution artifacts cannot form one read-only view."""


class FrozenDict(dict[str, Any]):
    """A dict-compatible recursive leaf that rejects every normal mutation."""

    def __init__(self, values: dict[str, Any] | tuple[tuple[str, Any], ...]) -> None:
        dict.__init__(self, values)

    def _readonly(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("frozen view mappings are read-only")

    __setitem__ = _readonly
    __delitem__ = _readonly
    clear = _readonly
    pop = _readonly
    popitem = _readonly
    setdefault = _readonly
    update = _readonly
    __ior__ = _readonly


class StrictViewModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        arbitrary_types_allowed=True,
    )


class ScoreValue(StrictViewModel):
    score: float = Field(ge=0, le=100)
    rationale: str = Field(min_length=1)


class CandidatePoint(StrictViewModel):
    candidate_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    presentation_text: str = Field(min_length=1)
    weighted_score: float = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    scores: FrozenDict
    score_change: float
    patches: tuple[FrozenDict, ...]
    risks: tuple[str, ...]


class NumericPoint(StrictViewModel):
    checkpoint_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    new_evidence_ids: tuple[str, ...]
    candidates: tuple[CandidatePoint, ...]
    scenes: tuple[FrozenDict, ...]
    decision_brief: str


class DeltaRecord(StrictViewModel):
    raw_id: str = Field(min_length=1)
    raw_content: str = Field(min_length=1)
    source_platform: str = Field(min_length=1)
    source_ref: str = Field(min_length=1)
    original_url: str | None = None


class DeltaBatch(StrictViewModel):
    batch_id: str = Field(min_length=1)
    records: tuple[DeltaRecord, ...]


class FinalSynthesis(StrictViewModel):
    source_checkpoint_id: str = Field(min_length=1)
    core_narrative: str = Field(min_length=1)
    pillars: tuple[FrozenDict, ...]
    scenes: tuple[FrozenDict, ...]
    remaining_risks: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    decision_brief: str = Field(min_length=1)


class EvolutionRunView(StrictViewModel):
    run_id: str = Field(min_length=1)
    selected_candidate_ids: tuple[str, ...]
    candidates: tuple[FrozenDict, ...]
    numeric_points: tuple[NumericPoint, ...]
    delta_batches: tuple[DeltaBatch, ...]
    final_synthesis: FinalSynthesis


_CHECKPOINT_IDS = tuple(f"checkpoint-{index:02d}" for index in range(5))
_DELTA_BATCH_IDS = tuple(f"delta-{index:02d}" for index in range(1, 5))
_FORMAL_STAGES = (*_CHECKPOINT_IDS, "final-synthesis")
_PATCH_KEYS = frozenset(
    {"field_name", "before", "after", "trigger_evidence_ids", "reason"}
)
_PILLAR_KEYS = frozenset({"candidate_id", "name", "role", "evidence_refs"})
_SCENE_KEYS = frozenset({"scene_id", "tier", "label", "action", "evidence_refs"})
_CHECKPOINT_SCENE_KEYS = frozenset(
    {"scene_id", "tier", "label", "support_summary", "evidence_refs"}
)
_SCORE_FIELDS = (
    "evidence_strength",
    "emotional_tension",
    "meijian_fit_and_exclusivity",
    "competitor_difference",
    "scene_conversion",
)
_ARTIFACTS = (
    "run_manifest.json",
    "artifact_audit.json",
    "inputs/selected_opportunities.json",
    "inputs/delta_batches.json",
    *(f"results/checkpoint-{index:02d}.json" for index in range(5)),
    "results/final-synthesis.json",
)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict(tuple((key, _freeze_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise EvolutionPresentationError(f"正式演化产物必须是普通文件: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    _regular_file(path)
    try:
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvolutionPresentationError(f"无法读取正式演化产物 {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvolutionPresentationError(f"正式演化产物顶层必须是 object: {path}")
    return payload


def _required(payload: dict[str, Any], key: str, path: Path) -> Any:
    if key not in payload:
        raise EvolutionPresentationError(f"正式演化产物缺少字段 {key}: {path}")
    return payload[key]


def _non_empty_string(value: Any, label: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise EvolutionPresentationError(f"{label} 必须是非空字符串: {path}")
    return value


def _strict_string_list(
    value: Any,
    label: str,
    path: Path,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise EvolutionPresentationError(f"{label} 必须是字符串列表: {path}")
    if any(not isinstance(item, str) or not item for item in value):
        raise EvolutionPresentationError(f"{label} 必须只包含非空字符串: {path}")
    if len(set(value)) != len(value):
        raise EvolutionPresentationError(f"{label} 不得重复: {path}")
    return tuple(value)


def _patches(value: Any, path: Path) -> tuple[FrozenDict, ...]:
    if not isinstance(value, list):
        raise EvolutionPresentationError(f"patches 必须是对象列表: {path}")
    result: list[FrozenDict] = []
    for index, patch in enumerate(value):
        item_label = f"patches[{index}]"
        if not isinstance(patch, dict) or set(patch) != _PATCH_KEYS:
            raise EvolutionPresentationError(f"{item_label} 结构无效: {path}")
        _non_empty_string(patch["field_name"], f"{item_label}.field_name", path)
        _non_empty_string(patch["before"], f"{item_label}.before", path)
        _non_empty_string(patch["after"], f"{item_label}.after", path)
        _non_empty_string(patch["reason"], f"{item_label}.reason", path)
        _strict_string_list(
            patch["trigger_evidence_ids"],
            f"{item_label}.trigger_evidence_ids",
            path,
            allow_empty=False,
        )
        result.append(_freeze_value(patch))
    return tuple(result)


def _final_objects(
    value: Any,
    label: str,
    path: Path,
    required_keys: frozenset[str],
) -> tuple[FrozenDict, ...]:
    if not isinstance(value, list) or not value:
        raise EvolutionPresentationError(f"{label} 必须是非空对象列表: {path}")
    result: list[FrozenDict] = []
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if not isinstance(item, dict) or set(item) != required_keys:
            raise EvolutionPresentationError(f"{item_label} 结构无效: {path}")
        for key in required_keys - {"evidence_refs"}:
            _non_empty_string(item[key], f"{item_label}.{key}", path)
        _strict_string_list(
            item["evidence_refs"],
            f"{item_label}.evidence_refs",
            path,
            allow_empty=False,
        )
        result.append(_freeze_value(item))
    return tuple(result)


def _checkpoint_scenes(value: Any, path: Path) -> tuple[FrozenDict, ...]:
    return _final_objects(value, "scenes", path, _CHECKPOINT_SCENE_KEYS)


def _validate_manifest(payload: dict[str, Any], path: Path) -> str:
    run_id = _run_id(payload, path)
    status = _required(payload, "status", path)
    if status != "COMPLETE":
        raise EvolutionPresentationError(f"run_manifest status 必须为 COMPLETE: {path}")
    stages = _strict_string_list(_required(payload, "stages", path), "manifest stages", path)
    completed = _strict_string_list(
        _required(payload, "completed_stages", path),
        "manifest completed_stages",
        path,
    )
    if stages != _FORMAL_STAGES or completed != _FORMAL_STAGES:
        raise EvolutionPresentationError(f"run_manifest 阶段不完整或顺序无效: {path}")
    return run_id


def _run_id(payload: dict[str, Any], path: Path) -> str:
    value = _required(payload, "run_id", path)
    if not isinstance(value, str) or not value:
        raise EvolutionPresentationError(f"run_id 无效: {path}")
    return value


def _assert_run_id(payload: dict[str, Any], expected: str, path: Path) -> None:
    actual = _run_id(payload, path)
    if actual != expected:
        raise EvolutionPresentationError(
            f"run_id 不一致，期望 {expected}，实际 {actual}: {path}"
        )


def _candidate_scores(raw_scores: Any, path: Path) -> CandidateScores:
    try:
        return CandidateScores.model_validate(raw_scores)
    except ValidationError as exc:
        raise EvolutionPresentationError(f"original_scores 无效: {path}: {exc}") from exc


def _score_values(scores: CandidateScores) -> dict[str, ScoreValue]:
    return FrozenDict(
        tuple(
            (
                field_name,
                ScoreValue.model_validate(getattr(scores, field_name).model_dump()),
            )
            for field_name in _SCORE_FIELDS
        )
    )


def _candidate_point(
    raw: dict[str, Any],
    path: Path,
    *,
    original: bool = False,
) -> CandidatePoint:
    candidate_id = _required(raw, "candidate_id", path)
    if not isinstance(candidate_id, str) or not candidate_id:
        raise EvolutionPresentationError(f"candidate_id 无效: {path}")

    if original:
        raw_scores = _required(raw, "original_scores", path)
        scores = _candidate_scores(raw_scores, path)
        title = _required(raw, "title", path)
        presentation_text = _required(raw, "draft_proposition", path)
        rank = _required(raw, "ai_original_rank", path)
        patches: tuple[FrozenDict, ...] = ()
        risks = _strict_string_list(_required(raw, "risks", path), "risks", path)
        score_change = 0.0
    else:
        raw_scores = _required(raw, "scores", path)
        scores = _candidate_scores(raw_scores, path)
        title = _required(raw, "title", path)
        presentation_text = _required(raw, "presentation_text", path)
        rank = _required(raw, "rank", path)
        patches = _patches(_required(raw, "patches", path), path)
        risks = _strict_string_list(_required(raw, "risks", path), "risks", path)
        score_change = _required(raw, "score_change", path)

    weighted_score = calculate_weighted_score(scores)
    if not isinstance(title, str) or not title:
        raise EvolutionPresentationError(f"candidate title 无效: {path}")
    if not isinstance(presentation_text, str) or not presentation_text:
        raise EvolutionPresentationError(f"candidate presentation text 无效: {path}")
    if not isinstance(rank, int) or rank < 1:
        raise EvolutionPresentationError(f"candidate rank 无效: {path}")
    if not isinstance(score_change, (int, float)):
        raise EvolutionPresentationError(f"candidate score_change 无效: {path}")

    if not original:
        declared_weighted_score = _required(raw, "weighted_score", path)
        if declared_weighted_score != weighted_score:
            raise EvolutionPresentationError(
                f"weighted_score 与现有评分权重不一致: {path}"
            )

    try:
        return CandidatePoint(
            candidate_id=candidate_id,
            title=title,
            presentation_text=presentation_text,
            weighted_score=weighted_score,
            rank=rank,
            scores=_score_values(scores),
            score_change=float(score_change),
            patches=patches,
            risks=risks,
        )
    except ValidationError as exc:
        raise EvolutionPresentationError(f"candidate view 无效: {path}: {exc}") from exc


def _validate_ids(ids: Any, label: str, path: Path) -> tuple[str, ...]:
    if not isinstance(ids, list) or not ids or any(
        not isinstance(item, str) or not item for item in ids
    ):
        raise EvolutionPresentationError(f"{label} 无效: {path}")
    if len(set(ids)) != len(ids):
        raise EvolutionPresentationError(f"{label} 必须唯一: {path}")
    return tuple(ids)


def _build_delta_batches(payload: dict[str, Any], path: Path, run_id: str) -> tuple[DeltaBatch, ...]:
    _assert_run_id(payload, run_id, path)
    batches = _required(payload, "batches", path)
    if not isinstance(batches, list) or len(batches) != 4:
        raise EvolutionPresentationError(f"delta_batches 必须包含四个批次: {path}")
    seen_raw_ids: set[str] = set()
    result: list[DeltaBatch] = []
    for index, raw_batch in enumerate(batches):
        if not isinstance(raw_batch, dict):
            raise EvolutionPresentationError(f"delta batch 必须是 object: {path}")
        batch_id = _required(raw_batch, "batch_id", path)
        expected_batch_id = _DELTA_BATCH_IDS[index]
        if batch_id != expected_batch_id:
            raise EvolutionPresentationError(
                f"delta batch 顺序或 ID 不一致，期望 {expected_batch_id}: {path}"
            )
        records = _required(raw_batch, "records", path)
        if not isinstance(records, list) or len(records) != 5:
            raise EvolutionPresentationError(f"每个 delta batch 必须包含五条记录: {path}")
        view_records: list[DeltaRecord] = []
        for raw_record in records:
            if not isinstance(raw_record, dict):
                raise EvolutionPresentationError(f"delta record 必须是 object: {path}")
            raw_id = _required(raw_record, "raw_id", path)
            if raw_id in seen_raw_ids:
                raise EvolutionPresentationError(f"delta raw_id 重复: {path}")
            seen_raw_ids.add(raw_id)
            try:
                view_records.append(
                    DeltaRecord(
                        raw_id=raw_id,
                        raw_content=_required(raw_record, "raw_content", path),
                        source_platform=_required(raw_record, "source_platform", path),
                        source_ref=_required(raw_record, "source_ref", path),
                        original_url=raw_record.get("original_url"),
                    )
                )
            except ValidationError as exc:
                raise EvolutionPresentationError(f"delta record 无效: {path}: {exc}") from exc
        result.append(DeltaBatch(batch_id=batch_id, records=tuple(view_records)))
    record_count = _required(payload, "record_count", path)
    if record_count != len(seen_raw_ids):
        raise EvolutionPresentationError(f"delta record_count 不一致: {path}")
    return tuple(result)


def load_official_evolution_run(root: Path) -> EvolutionRunView:
    """Load only the frozen official evolution artifacts into an immutable view."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise EvolutionPresentationError(f"正式演化目录必须是普通目录: {root}")
    paths = {relative: root / relative for relative in _ARTIFACTS}
    manifest_path = paths["run_manifest.json"]
    manifest = _read_json(manifest_path)
    run_id = _validate_manifest(manifest, manifest_path)
    payloads = {relative: _read_json(path) for relative, path in paths.items()}
    audit_path = paths["artifact_audit.json"]
    selected_path = paths["inputs/selected_opportunities.json"]
    delta_path = paths["inputs/delta_batches.json"]
    audit = payloads["artifact_audit.json"]
    selected = payloads["inputs/selected_opportunities.json"]
    _assert_run_id(audit, run_id, audit_path)
    _assert_run_id(selected, run_id, selected_path)

    selected_ids = _validate_ids(
        _required(selected, "selected_candidate_ids", selected_path),
        "selected_candidate_ids",
        selected_path,
    )
    audit_ids = _validate_ids(
        _required(audit, "selected_candidate_ids", audit_path),
        "audit selected_candidate_ids",
        audit_path,
    )
    if selected_ids != audit_ids:
        raise EvolutionPresentationError(
            f"selected candidate ids disagree: {selected_path}; {audit_path}"
        )
    if len(selected_ids) != 3:
        raise EvolutionPresentationError(f"正式演化必须有三个 selected candidates: {selected_path}")

    opportunities = _required(selected, "opportunities", selected_path)
    if not isinstance(opportunities, list) or len(opportunities) != len(selected_ids):
        raise EvolutionPresentationError(f"opportunities 与 selected_candidate_ids 不一致: {selected_path}")
    opportunity_ids: list[str] = []
    original_candidates: list[CandidatePoint] = []
    candidate_views: list[dict[str, Any]] = []
    for opportunity in opportunities:
        if not isinstance(opportunity, dict):
            raise EvolutionPresentationError(f"opportunity 必须是 object: {selected_path}")
        candidate = _required(opportunity, "candidate", selected_path)
        if not isinstance(candidate, dict):
            raise EvolutionPresentationError(f"opportunity.candidate 必须是 object: {selected_path}")
        opportunity_id = _required(opportunity, "candidate_id", selected_path)
        candidate_id = _required(candidate, "candidate_id", selected_path)
        if opportunity_id != candidate_id:
            raise EvolutionPresentationError(f"selected opportunity candidate id 不一致: {selected_path}")
        opportunity_ids.append(candidate_id)
        original_candidates.append(_candidate_point({**opportunity, **candidate}, selected_path, original=True))
        candidate_views.append(_freeze_value(candidate))
    if tuple(opportunity_ids) != selected_ids:
        raise EvolutionPresentationError(f"selected opportunity candidate ids disagree: {selected_path}")

    delta_batches = _build_delta_batches(payloads["inputs/delta_batches.json"], delta_path, run_id)

    checkpoints: list[NumericPoint] = []
    for checkpoint_id in _CHECKPOINT_IDS:
        relative = f"results/{checkpoint_id}.json"
        path = paths[relative]
        payload = payloads[relative]
        _assert_run_id(payload, run_id, path)
        if _required(payload, "checkpoint_id", path) != checkpoint_id:
            raise EvolutionPresentationError(f"checkpoint_id 与文件名不一致: {path}")
        candidates = _required(payload, "candidates", path)
        if not isinstance(candidates, list) or len(candidates) != len(selected_ids):
            raise EvolutionPresentationError(f"checkpoint candidate 数量不一致: {path}")
        points = tuple(_candidate_point(item, path) for item in candidates if isinstance(item, dict))
        if len(points) != len(candidates):
            raise EvolutionPresentationError(f"checkpoint candidate 必须是 object: {path}")
        point_ids = tuple(point.candidate_id for point in points)
        if set(point_ids) != set(selected_ids):
            raise EvolutionPresentationError(
                f"checkpoint candidate ids disagree: {path}; {selected_path}"
            )
        checkpoints.append(
            NumericPoint(
                checkpoint_id=checkpoint_id,
                label=checkpoint_id,
                new_evidence_ids=_strict_string_list(
                    _required(payload, "new_evidence_ids", path),
                    "new_evidence_ids",
                    path,
                    allow_empty=False,
                ),
                candidates=points,
                scenes=_checkpoint_scenes(_required(payload, "scenes", path), path),
                decision_brief=_non_empty_string(
                    _required(payload, "decision_brief", path),
                    "decision_brief",
                    path,
                ),
            )
        )

    final_path = paths["results/final-synthesis.json"]
    final_payload = payloads["results/final-synthesis.json"]
    _assert_run_id(final_payload, run_id, final_path)
    final_source = _required(final_payload, "source_checkpoint_id", final_path)
    if final_source != "checkpoint-04":
        raise EvolutionPresentationError(f"final synthesis must follow checkpoint-04: {final_path}")
    pillars = _final_objects(
        _required(final_payload, "pillars", final_path),
        "pillars",
        final_path,
        _PILLAR_KEYS,
    )
    pillar_ids = tuple(item["candidate_id"] for item in pillars)
    if len(pillar_ids) != len(selected_ids) or set(pillar_ids) != set(selected_ids):
        raise EvolutionPresentationError(
            f"final synthesis candidate ids disagree: {final_path}; {selected_path}"
        )
    scenes = _final_objects(
        _required(final_payload, "scenes", final_path),
        "scenes",
        final_path,
        _SCENE_KEYS,
    )
    try:
        final_synthesis = FinalSynthesis(
            source_checkpoint_id=final_source,
            core_narrative=_required(final_payload, "core_narrative", final_path),
            pillars=pillars,
            scenes=scenes,
            remaining_risks=_strict_string_list(
                _required(final_payload, "remaining_risks", final_path),
                "remaining_risks",
                final_path,
            ),
            forbidden_claims=_strict_string_list(
                _required(final_payload, "forbidden_claims", final_path),
                "forbidden_claims",
                final_path,
            ),
            decision_brief=_required(final_payload, "decision_brief", final_path),
        )
    except (TypeError, ValidationError) as exc:
        raise EvolutionPresentationError(f"final synthesis view 无效: {final_path}: {exc}") from exc

    return EvolutionRunView(
        run_id=run_id,
        selected_candidate_ids=selected_ids,
        candidates=tuple(candidate_views),
        numeric_points=(
            NumericPoint(
                checkpoint_id="ai-original",
                label="AI 原始版",
                new_evidence_ids=(),
                candidates=tuple(original_candidates),
                scenes=(),
                decision_brief="AI original checkpoint has no formal checkpoint attribution.",
            ),
            *checkpoints,
        ),
        delta_batches=delta_batches,
        final_synthesis=final_synthesis,
    )


__all__ = [
    "CandidatePoint",
    "DeltaBatch",
    "DeltaRecord",
    "EvolutionPresentationError",
    "EvolutionRunView",
    "FinalSynthesis",
    "NumericPoint",
    "ScoreValue",
    "load_official_evolution_run",
]
