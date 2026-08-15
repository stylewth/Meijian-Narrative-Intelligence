"""Pure, bounded projections for the eight-node Feishu demo timeline."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, field_validator

from src.schemas import StrictBaseModel
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes
from src.string_enum import StringEnum


class DemoNotificationNode(StringEnum):
    PREPROCESSING_COMPLETE = "PREPROCESSING_COMPLETE"
    PRESSURE_TEST_COMPLETE = "PRESSURE_TEST_COMPLETE"
    BLIND_SELECTION_COMPLETE = "BLIND_SELECTION_COMPLETE"
    DELTA_01_COMPLETE = "DELTA_01_COMPLETE"
    DELTA_02_COMPLETE = "DELTA_02_COMPLETE"
    DELTA_03_COMPLETE = "DELTA_03_COMPLETE"
    DELTA_04_COMPLETE = "DELTA_04_COMPLETE"
    FINAL_SYNTHESIS_COMPLETE = "FINAL_SYNTHESIS_COMPLETE"


class _ImmutableStrictModel(StrictBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NotificationMetric(_ImmutableStrictModel):
    label: str = Field(min_length=1, max_length=24)
    value: str = Field(min_length=1, max_length=80)


class NotificationCandidate(_ImmutableStrictModel):
    candidate_id: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=40)
    score: float | None = None
    score_delta: float | None = None
    rank: int | None = Field(default=None, ge=1)
    narrative_change: str = Field(default="", max_length=180)


class DemoNotificationSnapshot(_ImmutableStrictModel):
    node: DemoNotificationNode
    ordinal: int = Field(ge=1, le=8)
    title: str = Field(min_length=1, max_length=48)
    conclusion: str = Field(min_length=1, max_length=300)
    metrics: tuple[NotificationMetric, ...] = Field(default_factory=tuple, max_length=8)
    candidates: tuple[NotificationCandidate, ...] = Field(default_factory=tuple, max_length=5)
    primary_risk: str = Field(default="", max_length=200)
    source_run_id: str = Field(min_length=1)
    source_checkpoint_id: str | None = Field(default=None, min_length=1)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("metrics", "candidates", mode="before")
    @classmethod
    def _freeze_collections(cls, value: Any) -> Any:
        return tuple(value) if isinstance(value, (list, tuple)) else value


_DELTA_NODES = (
    (DemoNotificationNode.DELTA_01_COMPLETE, "checkpoint-01", 2),
    (DemoNotificationNode.DELTA_02_COMPLETE, "checkpoint-02", 3),
    (DemoNotificationNode.DELTA_03_COMPLETE, "checkpoint-03", 4),
    (DemoNotificationNode.DELTA_04_COMPLETE, "checkpoint-04", 5),
)


def _compact(value: str, maximum: int) -> str:
    if len(value) <= maximum:
        return value
    return value[: maximum - 1].rstrip() + "…"


def _metric(label: str, value: object) -> NotificationMetric:
    return NotificationMetric(label=label, value=str(value))


def _snapshot(
    *,
    node: DemoNotificationNode,
    ordinal: int,
    title: str,
    conclusion: str,
    metrics: list[NotificationMetric],
    candidates: list[NotificationCandidate],
    primary_risk: str,
    source_run_id: str,
    source_checkpoint_id: str | None = None,
) -> DemoNotificationSnapshot:
    payload: dict[str, Any] = {
        "node": node,
        "ordinal": ordinal,
        "title": title,
        "conclusion": conclusion,
        "metrics": metrics,
        "candidates": candidates,
        "primary_risk": primary_risk,
        "source_run_id": source_run_id,
        "source_checkpoint_id": source_checkpoint_id,
    }
    source_sha256 = sha256_bytes(canonical_json_bytes(payload))
    return DemoNotificationSnapshot(**payload, source_sha256=source_sha256)


def _preprocessing_snapshot(preprocessing: Any, source_run_id: str) -> DemoNotificationSnapshot:
    stages = {stage.key: dict(stage.metrics) for stage in preprocessing.stages}
    metrics = [
        _metric("原始语料", stages["validate"]["原始语料"]),
        _metric(
            "有效语料",
            (
                f"{stages['clean']['有效语料']}（排除 {stages['clean']['排除']}，"
                f"待复核 {stages['clean']['待复核']}）"
            ),
        ),
        *[
            _metric(label, stages["split"][label])
            for label in ("ANALYSIS", "GOLD", "CHALLENGE_POOL", "HOLDOUT")
        ],
        _metric(
            "正式标注",
            f"{stages['annotate']['正式标注']}（批次 {stages['annotate']['批次']}）",
        ),
        _metric("正式包", stages["freeze"]["正式包"]),
    ]
    return _snapshot(
        node=DemoNotificationNode.PREPROCESSING_COMPLETE,
        ordinal=1,
        title="预处理结果已冻结",
        conclusion="五个预处理阶段已完成，正式数据包与证据边界可以进入后续演示。",
        metrics=metrics,
        candidates=[],
        primary_risk="",
        source_run_id=source_run_id,
    )


def _pressure_snapshot(pressure: Any, source_run_id: str) -> DemoNotificationSnapshot:
    support = sum(item.support_count for item in pressure.holdout.candidates)
    challenge = sum(item.challenge_count for item in pressure.holdout.candidates)
    neutral = sum(item.neutral_count for item in pressure.holdout.candidates)
    check_count = sum(len(candidate.checks) for candidate in pressure.candidates)
    dialogue_count = sum(len(candidate.dialogue) for candidate in pressure.candidates)
    blocking_count = sum(
        len(candidate.blocking_findings) for candidate in pressure.holdout.candidates
    )
    return _snapshot(
        node=DemoNotificationNode.PRESSURE_TEST_COMPLETE,
        ordinal=2,
        title="压力测试结果已完成",
        conclusion=(
            f"{len(pressure.evaluation.candidate_ids)} 个候选完成检查、互审与 HOLDOUT；"
            "盲选结果留到下一节点确认。"
        ),
        metrics=[
            _metric("候选数量", len(pressure.evaluation.candidate_ids)),
            _metric("检查数量", check_count),
            _metric("互审消息", dialogue_count),
            _metric("HOLDOUT 支持", support),
            _metric("HOLDOUT 挑战", challenge),
            _metric("HOLDOUT 中性", neutral),
            _metric("阻断发现", blocking_count),
        ],
        candidates=[],
        primary_risk=(
            "HOLDOUT 尚未形成阻断发现。"
            if blocking_count == 0
            else f"HOLDOUT 记录 {blocking_count} 条阻断发现，需人工复核。"
        ),
        source_run_id=source_run_id,
    )


def _blind_selection_snapshot(pressure: Any, source_run_id: str) -> DemoNotificationSnapshot:
    candidates = [
        NotificationCandidate(
            candidate_id=candidate_id,
            title="团队确认候选",
        )
        for candidate_id in pressure.selection.selected_candidate_ids
    ]
    selected_count = len(pressure.selection.selected_candidate_ids)
    return _snapshot(
        node=DemoNotificationNode.BLIND_SELECTION_COMPLETE,
        ordinal=3,
        title="盲选确认结果已完成",
        conclusion=(
            f"真人盲评收到 {pressure.blind_review.survey_count} 份问卷，"
            f"团队确认 {selected_count} 个候选进入演化。"
        ),
        metrics=[
            _metric("盲评问卷", pressure.blind_review.survey_count),
            _metric("开放回答", pressure.blind_review.open_response_count),
            _metric("确认候选", selected_count),
            _metric("盲评指标", len(pressure.blind_review.selection_metrics)),
        ],
        candidates=candidates,
        primary_risk="盲评结论仍需结合后续新增证据持续验证。",
        source_run_id=source_run_id,
    )


def _delta_snapshot(
    evolution: Any,
    source_run_id: str,
    ordinal: int,
    node: DemoNotificationNode,
    checkpoint_id: str,
    point_index: int,
) -> DemoNotificationSnapshot:
    previous = evolution.numeric_points[point_index - 1]
    current = evolution.numeric_points[point_index]
    previous_by_id = {candidate.candidate_id: candidate for candidate in previous.candidates}
    current_candidates = []
    changed_narratives = 0
    for candidate in current.candidates:
        before = previous_by_id[candidate.candidate_id]
        narrative_changed = candidate.presentation_text != before.presentation_text
        changed_narratives += narrative_changed
        current_candidates.append(
            NotificationCandidate(
                candidate_id=candidate.candidate_id,
                title=_compact(candidate.title, 40),
                score=candidate.weighted_score,
                score_delta=round(candidate.weighted_score - before.weighted_score, 10),
                rank=candidate.rank,
                narrative_change=(
                    "叙事已更新"
                    if narrative_changed
                    else "叙事未变化"
                ),
            )
        )
    batch = evolution.delta_batches[point_index - 2]
    top_candidate = min(current.candidates, key=lambda item: item.rank)
    return _snapshot(
        node=node,
        ordinal=ordinal,
        title=f"{checkpoint_id} 证据增量已完成",
        conclusion=(
            f"{checkpoint_id} 已完成 {len(batch.records)} 条新增证据的归因、"
            f"分数与叙事变化展示。"
        ),
        metrics=[
            _metric("新增证据", len(batch.records)),
            _metric("候选数量", len(current.candidates)),
            _metric("叙事变化", changed_narratives),
            _metric("最高分", top_candidate.weighted_score),
        ],
        candidates=current_candidates,
        primary_risk="新增证据带来的业务边界仍需人工复核。",
        source_run_id=source_run_id,
        source_checkpoint_id=checkpoint_id,
    )


def _final_snapshot(evolution: Any, source_run_id: str) -> DemoNotificationSnapshot:
    final = evolution.final_synthesis
    candidates = [
        NotificationCandidate(
            candidate_id=pillar["candidate_id"],
            title=_compact(pillar["name"], 40),
        )
        for pillar in final.pillars
    ]
    return _snapshot(
        node=DemoNotificationNode.FINAL_SYNTHESIS_COMPLETE,
        ordinal=8,
        title="最终叙事综合已完成",
        conclusion=_compact(final.core_narrative, 300),
        metrics=[
            _metric("核心支柱", len(final.pillars)),
            _metric("落地场景", len(final.scenes)),
            _metric("剩余风险", len(final.remaining_risks)),
            _metric("禁止主张", len(final.forbidden_claims)),
        ],
        candidates=candidates,
        primary_risk="最终综合仍保留业务边界，需人工确认后执行。",
        source_run_id=source_run_id,
        source_checkpoint_id=final.source_checkpoint_id,
    )


def build_demo_notification_snapshots(
    preprocessing: Any,
    pressure: Any,
    evolution: Any,
) -> tuple[DemoNotificationSnapshot, ...]:
    """Build the immutable eight-node notification contract without side effects."""

    source_run_id = evolution.run_id
    snapshots = [
        _preprocessing_snapshot(preprocessing, source_run_id),
        _pressure_snapshot(pressure, source_run_id),
        _blind_selection_snapshot(pressure, source_run_id),
    ]
    snapshots.extend(
        _delta_snapshot(evolution, source_run_id, ordinal, node, checkpoint_id, point_index)
        for ordinal, (node, checkpoint_id, point_index) in enumerate(_DELTA_NODES, start=4)
    )
    snapshots.append(_final_snapshot(evolution, source_run_id))
    if tuple(item.node for item in snapshots) != tuple(DemoNotificationNode):
        raise ValueError("demo notification nodes must remain in the fixed eight-node order")
    return tuple(snapshots)


__all__ = [
    "DemoNotificationNode",
    "DemoNotificationSnapshot",
    "NotificationCandidate",
    "NotificationMetric",
    "build_demo_notification_snapshots",
]
