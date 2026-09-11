"""Project frozen decision artifacts into Bitable write-back field records.

输入必须是已经过 schema 校验的冻结产物模型；本层只做投影，
不改写、不补全、不猜测。任何结构不符直接抛错。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from src.schemas import (
    CandidateDecisionSnapshot,
    DecisionEvolutionCheckpoint,
    DecisionFoundationState,
    FinalCandidateSelection,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_log_fields(
    run_id: str,
    *,
    action: str,
    actor_open_id: str,
    summary: str,
    demo_run_id: str | None = None,
    demo_started_at: str | None = None,
    created_at: str | None = None,
) -> dict[str, str]:
    """一条运行/决策日志的字段记录。"""

    for name, value in (
        ("run_id", run_id),
        ("action", action),
        ("actor_open_id", actor_open_id),
        ("summary", summary),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be non-empty")
    if created_at is not None and (
        not isinstance(created_at, str) or not created_at.strip()
    ):
        raise ValueError("created_at must be non-empty")
    return add_demo_context(
        {
            "run_id": run_id,
            "action": action,
            "actor_open_id": actor_open_id,
            "summary": summary,
            "created_at": created_at if created_at is not None else _utc_now_iso(),
        },
        demo_run_id=demo_run_id,
        demo_started_at=demo_started_at,
    )


def add_demo_context(
    fields: Mapping[str, Any],
    *,
    demo_run_id: str | None,
    demo_started_at: str | None,
) -> dict[str, Any]:
    """为本轮动态表增加固定 session 标识；未传时保持旧字段契约。"""

    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("fields must be a non-empty mapping")
    if (demo_run_id is None) != (demo_started_at is None):
        raise ValueError("demo_run_id and demo_started_at must be provided together")
    result = dict(fields)
    if demo_run_id is not None:
        if not isinstance(demo_run_id, str) or not demo_run_id.strip():
            raise ValueError("demo_run_id must be non-empty")
        if not isinstance(demo_started_at, str) or not demo_started_at.strip():
            raise ValueError("demo_started_at must be non-empty")
        result["demo_run_id"] = demo_run_id
        result["demo_started_at"] = demo_started_at
    return result


def milestone_row(
    snapshot: Any,
    *,
    demo_run_id: str,
    demo_started_at: str,
    created_at: datetime,
) -> dict[str, Any]:
    """把一个阶段节点快照投影为结果表中的一行。"""

    source_run_id = getattr(snapshot, "source_run_id", None)
    node = getattr(getattr(snapshot, "node", None), "value", getattr(snapshot, "node", None))
    title = getattr(snapshot, "title", None)
    conclusion = getattr(snapshot, "conclusion", None)
    if not all(isinstance(value, str) and value.strip() for value in (source_run_id, node, title)):
        raise ValueError("snapshot must contain source_run_id, node and title")
    if not isinstance(conclusion, str):
        raise ValueError("snapshot conclusion must be text")
    metrics = getattr(snapshot, "metrics", ())
    candidates = getattr(snapshot, "candidates", ())
    metric_text = "；".join(
        f"{getattr(metric, 'label', '')}：{getattr(metric, 'value', '')}"
        for metric in metrics
    )
    candidate_text = "、".join(
        str(getattr(candidate, "candidate_id", candidate))
        for candidate in candidates
    )
    content_lines = [title, conclusion]
    if metric_text:
        content_lines.append(f"指标：{metric_text}")
    if candidate_text:
        content_lines.append(f"候选：{candidate_text}")
    return add_demo_context(
        {
            "run_id": source_run_id,
            "record_kind": "MILESTONE",
            "record_key": node,
            "content": "\n".join(content_lines),
            "created_at": created_at.astimezone(timezone.utc).isoformat(),
        },
        demo_run_id=demo_run_id,
        demo_started_at=demo_started_at,
    )


def candidate_rows(
    foundation: DecisionFoundationState,
    *,
    created_at: datetime,
) -> list[dict[str, str]]:
    """把 foundation 冻结态投影为每候选一行。"""

    if not isinstance(foundation, DecisionFoundationState):
        raise TypeError("foundation must be DecisionFoundationState")
    rows: list[dict[str, str]] = []
    ranked = {
        item.candidate.candidate_id: item for item in foundation.ranking.ranked_candidates
    }
    for candidate in foundation.candidates:
        item = ranked.get(candidate.candidate_id)
        if item is None:
            raise ValueError(f"候选 {candidate.candidate_id} 缺少排名记录")
        scores = item.evaluation.scores
        content = "\n".join(
            (
                f"标题：{candidate.title}",
                f"主张：{candidate.draft_proposition}",
                f"用户冲突：{candidate.user_conflict}",
                f"品牌机会：{candidate.brand_opportunity}",
                f"加权分：{item.weighted_score:.2f}｜排名：{item.rank}"
                f"{'｜推荐' if item.is_recommended else ''}",
                "五维："
                + "，".join(
                    (
                        f"证据 {scores.evidence_strength.score:.1f}",
                        f"情绪 {scores.emotional_tension.score:.1f}",
                        f"专属 {scores.brand_fit_and_exclusivity.score:.1f}",
                        f"差异 {scores.competitor_difference.score:.1f}",
                        f"场景 {scores.scene_conversion.score:.1f}",
                    )
                ),
                f"主要场景：{'；'.join(candidate.main_scenes)}",
            )
        )
        rows.append(
            {
                "run_id": foundation.run_id,
                "record_kind": "CANDIDATE",
                "record_key": candidate.candidate_id,
                "content": content,
                "created_at": created_at.astimezone(timezone.utc).isoformat(),
            }
        )
    return rows


def checkpoint_rows(
    checkpoint: DecisionEvolutionCheckpoint,
    *,
    created_at: datetime,
) -> list[dict[str, str]]:
    """把演化检查点投影为每候选一行。"""

    if not isinstance(checkpoint, DecisionEvolutionCheckpoint):
        raise TypeError("checkpoint must be DecisionEvolutionCheckpoint")
    patch_by_candidate: dict[str, list[str]] = {}
    for patch in checkpoint.patches:
        patch_by_candidate.setdefault(patch.candidate_id, []).append(patch.reason)
    rows: list[dict[str, str]] = []
    for snapshot in checkpoint.candidates:
        candidate = snapshot.ranked_narrative.candidate
        lines = [
            f"候选：{candidate.title}",
            f"主张：{candidate.draft_proposition}",
            f"加权分：{snapshot.ranked_narrative.weighted_score:.2f}"
            f"｜排名：{snapshot.ranked_narrative.rank}"
            f"｜状态：{snapshot.business_status.value}",
            f"证据：支持 {snapshot.supporting_count}／反例 {snapshot.counter_count}"
            f"／风险 {snapshot.risk_count}",
            f"可见证据：{len(checkpoint.visible_evidence_ids)} 条",
        ]
        patches = patch_by_candidate.get(candidate.candidate_id)
        if patches:
            lines.append(f"修订：{'；'.join(patches)}")
        if (
            checkpoint.switch_suggestion is not None
            and checkpoint.switch_suggestion.to_candidate_id == candidate.candidate_id
        ):
            lines.append("换线建议指向该候选")
        rows.append(
            {
                "run_id": checkpoint.run_id,
                "record_kind": "CHECKPOINT",
                "record_key": f"{checkpoint.checkpoint_id}:{candidate.candidate_id}",
                "content": "\n".join(lines),
                "created_at": created_at.astimezone(timezone.utc).isoformat(),
            }
        )
    return rows


def final_selection_row(
    selection: FinalCandidateSelection,
    *,
    created_at: datetime,
) -> dict[str, str]:
    """把终选记录投影为一行。"""

    if not isinstance(selection, FinalCandidateSelection):
        raise TypeError("selection must be FinalCandidateSelection")
    content = "\n".join(
        (
            f"最终选择：{selection.selected_candidate_id}",
            f"入围候选：{'，'.join(selection.shortlisted_candidate_ids)}",
            f"推荐候选：{selection.recommended_candidate_id}",
            f"确认人：{selection.selected_by}",
            f"时间：{selection.selected_at.isoformat()}",
        )
        + (
            (f"理由：{selection.reason}",)
            if (selection.reason or "").strip()
            else ()
        )
    )
    return {
        "run_id": selection.run_id,
        "record_kind": "FINAL_SELECTION",
        "record_key": selection.selected_candidate_id,
        "content": content,
        "created_at": created_at.astimezone(timezone.utc).isoformat(),
    }


def snapshot_row(
    snapshot: CandidateDecisionSnapshot,
    *,
    run_id: str,
    record_key: str,
    created_at: datetime,
) -> dict[str, str]:
    """把单个候选决策快照投影为一行（选线呈现/演化回放用）。"""

    if not isinstance(snapshot, CandidateDecisionSnapshot):
        raise TypeError("snapshot must be CandidateDecisionSnapshot")
    candidate = snapshot.ranked_narrative.candidate
    content = "\n".join(
        (
            f"候选：{candidate.title}",
            f"主张：{candidate.draft_proposition}",
            f"加权分：{snapshot.ranked_narrative.weighted_score:.2f}"
            f"｜排名：{snapshot.ranked_narrative.rank}"
            f"｜状态：{snapshot.business_status.value}",
            f"证据：支持 {snapshot.supporting_count}／反例 {snapshot.counter_count}"
            f"／风险 {snapshot.risk_count}",
        )
    )
    return {
        "run_id": run_id,
        "record_kind": "CANDIDATE",
        "record_key": record_key,
        "content": content,
        "created_at": created_at.astimezone(timezone.utc).isoformat(),
    }


def _require_utc(value: Any, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


__all__ = [
    "add_demo_context",
    "candidate_rows",
    "checkpoint_rows",
    "final_selection_row",
    "milestone_row",
    "run_log_fields",
    "snapshot_row",
]
