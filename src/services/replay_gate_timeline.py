"""Frozen-replay decision gate timeline for the Feishu bot.

把 official-20260814 演化 run 的冻结产物映射为一列零模型决策门：机器人
现场按门时读取冻结产物、投影写回行并推进阶段。本层不调用任何模型，
不改写冻结文件；所有写回 summary 必须携带「冻结回放」标记，禁止把
回放表述成实时推理。

阶段名与正式 coordinator 状态机保持一致，复用门卡片与校验合同；
时间线只包含演化段，专属度三扇门在回放模式不提供。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

REPLAY_SOURCE_MARK = "冻结回放"

REPLAY_STAGES: tuple[str, ...] = (
    "AWAITING_SELECTION",
    "HOLDOUT",
    "READY_REASSESS",
    "CHECKPOINT_00",
    "CHECKPOINT_01",
    "CHECKPOINT_02",
    "CHECKPOINT_03",
    "CHECKPOINT_04",
    "COMPLETE",
)

_REPLAY_STAGE_ACTIONS: dict[str, str] = {
    "AWAITING_SELECTION": "SUBMIT_SELECTION",
    "HOLDOUT": "RUN_HOLDOUT",
    "READY_REASSESS": "RUN_BLIND_REASSESSMENT",
    "CHECKPOINT_00": "RUN_NEXT_RELEASE",
    "CHECKPOINT_01": "RUN_NEXT_RELEASE",
    "CHECKPOINT_02": "RUN_NEXT_RELEASE",
    "CHECKPOINT_03": "RUN_NEXT_RELEASE",
    "CHECKPOINT_04": "SUBMIT_FINAL_SELECTION",
    "COMPLETE": "",
}

_RESULTS_FIELDS = ("run_id", "record_kind", "record_key", "content", "created_at")

NODE_LABELS = {
    1: "数据预处理",
    2: "叙事压力测试",
    3: "盲选与选线",
    4: "增量批次1",
    5: "增量批次2",
    6: "增量批次3",
    7: "增量批次4",
    8: "终幕合成",
}

# 静态门 -> (需要演示推进到节点, 驱动网页到节点|None)；增量释放按阶段动态计算。
_STATIC_GATE_BINDING: dict[str, tuple[int, int | None]] = {
    "SUBMIT_SELECTION": (2, 3),
    "RUN_HOLDOUT": (3, None),
    "RUN_BLIND_REASSESSMENT": (3, None),
    "SUBMIT_FINAL_SELECTION": (7, 8),
}


def gate_milestone_binding(gate_action: str, stage: str | None = None) -> tuple[int, int | None]:
    """返回 (该门解锁所需演示节点, 按门后网页应推进到的节点)。"""

    if gate_action == "RUN_NEXT_RELEASE":
        if not isinstance(stage, str) or not stage.startswith("CHECKPOINT_"):
            raise ReplayTimelineError(f"释放增量的阶段无效：{stage!r}")
        index = int(stage[-2:]) + 1  # CHECKPOINT_00 -> 批次1
        return (3 + index - 1, 3 + index)
    binding = _STATIC_GATE_BINDING.get(gate_action)
    if binding is None:
        raise ReplayTimelineError(f"未定义里程碑绑定的门：{gate_action}")
    return binding


class ReplayTimelineError(ValueError):
    """冻结产物缺失、不一致或门请求与回放时间线冲突。"""


@dataclass(frozen=True, slots=True)
class ReplayTimeline:
    """经过结构审计的演化 run 冻结产物集合。"""

    run_id: str
    run_root: Path
    selection: dict[str, Any]
    blind_summary: dict[str, Any]
    checkpoints: dict[int, dict[str, Any]]
    final_synthesis: dict[str, Any]
    holdout_result: dict[str, Any]
    holdout_source: str

    @property
    def selected_candidate_ids(self) -> list[str]:
        return list(self.selection["selected_candidate_ids"])

    def candidates_sorted_by_rank(self, checkpoint_index: int) -> list[dict[str, Any]]:
        candidates = list(self.checkpoints[checkpoint_index]["candidates"])
        return sorted(candidates, key=lambda item: item["rank"])

    def recommended_candidate_id(self) -> str:
        ranked = self.candidates_sorted_by_rank(4)
        top = ranked[0]
        if int(top["rank"]) != 1:
            raise ReplayTimelineError("checkpoint-04 缺少 rank=1 的 AI 推荐候选")
        return str(top["candidate_id"])


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReplayTimelineError(f"冻结产物缺失：{path.name}") from exc
    except json.JSONDecodeError as exc:
        raise ReplayTimelineError(f"冻结产物不是合法 JSON：{path.name}") from exc
    if not isinstance(payload, dict):
        raise ReplayTimelineError(f"冻结产物必须是对象：{path.name}")
    return payload


def _require_run_id(payload: Mapping[str, Any], run_id: str, name: str) -> None:
    if payload.get("run_id") != run_id:
        raise ReplayTimelineError(f"{name} 的 run_id 与 run_manifest 不一致")


def load_replay_timeline(run_root: str | Path) -> ReplayTimeline:
    """只读加载并审计演化 run 冻结产物；任何不一致直接抛错。"""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise ReplayTimelineError(f"回放 run 目录不存在：{root}")
    manifest = _load_json(root / "run_manifest.json")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ReplayTimelineError("run_manifest.json 缺少 run_id")
    if manifest.get("status") != "COMPLETE":
        raise ReplayTimelineError("run_manifest.json 状态不是 COMPLETE，不能作为回放基线")
    completed = manifest.get("completed_stages")
    expected_stages = {f"checkpoint-{index:02d}" for index in range(5)} | {
        "final-synthesis"
    }
    if not isinstance(completed, list) or not expected_stages.issubset(set(completed)):
        raise ReplayTimelineError("run_manifest.json 的 completed_stages 不完整")

    selection = _load_json(root / "blind" / "selection_confirmation.json")
    _require_run_id(selection, run_id, "selection_confirmation.json")
    selected_ids = selection.get("selected_candidate_ids")
    if (
        not isinstance(selected_ids, list)
        or not 1 <= len(selected_ids) <= 3
        or not all(isinstance(item, str) and item.strip() for item in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise ReplayTimelineError("selection_confirmation.json 的选线集合无效")

    blind_summary = _load_json(root / "blind" / "blind_summary.json")
    _require_run_id(blind_summary, run_id, "blind_summary.json")

    checkpoints: dict[int, dict[str, Any]] = {}
    for index in range(5):
        checkpoint = _load_json(root / "results" / f"checkpoint-{index:02d}.json")
        _require_run_id(checkpoint, run_id, f"checkpoint-{index:02d}.json")
        if checkpoint.get("checkpoint_id") != f"checkpoint-{index:02d}":
            raise ReplayTimelineError(f"checkpoint-{index:02d}.json 的 checkpoint_id 不符")
        if index >= 1 and checkpoint.get("previous_checkpoint_id") != f"checkpoint-{index - 1:02d}":
            raise ReplayTimelineError(f"checkpoint-{index:02d} 的链式前驱不符")
        if index >= 1 and not checkpoint.get("new_evidence_ids"):
            raise ReplayTimelineError(f"checkpoint-{index:02d} 缺少新增证据")
        candidates = checkpoint.get("candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not all(
                isinstance(item, dict)
                and isinstance(item.get("candidate_id"), str)
                and isinstance(item.get("rank"), int)
                for item in candidates
            )
        ):
            raise ReplayTimelineError(f"checkpoint-{index:02d} 的候选结构无效")
        checkpoints[index] = checkpoint

    checkpoint_ids = {
        item["candidate_id"] for item in checkpoints[4]["candidates"]
    }
    if set(selected_ids) != checkpoint_ids:
        raise ReplayTimelineError("选线集合与 checkpoint-04 候选集合不一致")

    final_synthesis = _load_json(root / "results" / "final-synthesis.json")
    _require_run_id(final_synthesis, run_id, "final-synthesis.json")
    if final_synthesis.get("source_checkpoint_id") != "checkpoint-04":
        raise ReplayTimelineError("final-synthesis 必须锚定 checkpoint-04")

    holdout_path = _locate_holdout_result(root)
    holdout_result = _load_json(holdout_path)
    impacts = holdout_result.get("impacts")
    if (
        not isinstance(impacts, list)
        or not impacts
        or not all(
            isinstance(item, dict)
            and isinstance(item.get("candidate_id"), str)
            and isinstance(item.get("impact"), str)
            for item in impacts
        )
    ):
        raise ReplayTimelineError("HOLDOUT 冻结结果的 impacts 结构无效")
    holdout_source = str(
        holdout_path.relative_to(root.parent.parent.parent)
        if _within(holdout_path, root.parent.parent.parent)
        else holdout_path
    )

    return ReplayTimeline(
        run_id=run_id,
        run_root=root,
        selection=selection,
        blind_summary=blind_summary,
        checkpoints=checkpoints,
        final_synthesis=final_synthesis,
        holdout_result=holdout_result,
        holdout_source=holdout_source,
    )


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _locate_holdout_result(run_root: Path) -> Path:
    """HOLDOUT 冻结结果位于 five_candidate_validation 官方包内，向上定位。"""

    current = run_root.parent
    while current != current.parent:
        candidate_base = current / "five_candidate_validation"
        if candidate_base.is_dir():
            matches = sorted(candidate_base.glob("*/holdout/result.json"))
            if len(matches) != 1:
                raise ReplayTimelineError(
                    f"five_candidate_validation 下必须恰有一个 holdout/result.json，"
                    f"实际 {len(matches)} 个"
                )
            return matches[0]
        current = current.parent
    raise ReplayTimelineError("未能在上级目录定位 five_candidate_validation 的 HOLDOUT 结果")


def initial_replay_state(run_id: str) -> dict[str, Any]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be non-empty")
    return {"mode": "replay", "run_id": run_id, "stage": "AWAITING_SELECTION", "history": []}


def read_replay_state(state_path: str | Path, expected_run_id: str) -> dict[str, Any] | None:
    path = Path(state_path)
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayTimelineError(f"回放状态文件损坏：{path}") from exc
    if not isinstance(state, dict) or state.get("mode") != "replay":
        raise ReplayTimelineError(f"回放状态文件格式无效：{path}")
    if state.get("run_id") != expected_run_id:
        raise ReplayTimelineError(
            f"回放状态属于 run {state.get('run_id')!r}，与请求的 {expected_run_id!r} 不一致"
        )
    if state.get("stage") not in REPLAY_STAGES:
        raise ReplayTimelineError(f"回放状态阶段无效：{state.get('stage')!r}")
    return state


def save_replay_state(state_path: str | Path, state: Mapping[str, Any]) -> None:
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(state), ensure_ascii=False, indent=2, sort_keys=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_action(
    timeline: ReplayTimeline,
    state: Mapping[str, Any],
    gate_action: str,
    payload: Mapping[str, Any] | None,
    *,
    requested_by: str,
    now: datetime,
    milestone_cursor: int | None = None,
) -> "ReplayOutcome":
    """在回放时间线上执行一扇门；返回新状态与写回行投影。"""

    if not isinstance(requested_by, str) or not requested_by.strip():
        raise ReplayTimelineError("操作者 open_id 不能为空")
    if now.tzinfo is None:
        raise ReplayTimelineError("时间戳必须带时区")
    if milestone_cursor is None:
        raise ReplayTimelineError(
            "当前没有活动演示会话：请先在网页端「开始新的案例会话」，再推进决策门。"
        )
    if not isinstance(milestone_cursor, int) or isinstance(milestone_cursor, bool) or not 0 <= milestone_cursor <= 8:
        raise ReplayTimelineError("演示进度游标无效")
    stage = state.get("stage")
    if state.get("run_id") != timeline.run_id:
        raise ReplayTimelineError("回放状态与冻结产物不属于同一个 run")
    expected_action = _REPLAY_STAGE_ACTIONS.get(stage)
    if expected_action is None:
        raise ReplayTimelineError(f"未知回放阶段：{stage!r}")
    if gate_action != expected_action:
        raise ReplayTimelineError(
            f"当前阶段 {stage} 只支持 {expected_action or '无'}，收到 {gate_action}"
        )
    required_node, drives_node = gate_milestone_binding(gate_action, stage)
    if milestone_cursor < required_node:
        raise ReplayTimelineError(
            f"演示尚未推进到该阶段（当前 {milestone_cursor}/8，需 {required_node}："
            f"{NODE_LABELS.get(required_node, required_node)}）；"
            "请先在网页推进，或用叙事卡片「下一步」远端推进。"
        )
    payload = dict(payload) if payload is not None else {}

    if gate_action == "SUBMIT_SELECTION":
        new_stage, detail, rows = _apply_selection(timeline, payload)
    elif gate_action == "RUN_HOLDOUT":
        new_stage, detail, rows = _apply_holdout(timeline)
    elif gate_action == "RUN_BLIND_REASSESSMENT":
        new_stage, detail, rows = _apply_blind_reassessment(timeline)
    elif gate_action == "RUN_NEXT_RELEASE":
        new_stage, detail, rows = _apply_next_release(timeline, stage)
    elif gate_action == "SUBMIT_FINAL_SELECTION":
        new_stage, detail, rows = _apply_final_selection(timeline, payload)
    else:
        raise ReplayTimelineError(f"回放时间线未接线的门：{gate_action}")

    created_at = now.astimezone(timezone.utc).isoformat()
    for row in rows:
        row["created_at"] = created_at
        for name in _RESULTS_FIELDS:
            if name not in row or not isinstance(row[name], str) or not row[name].strip():
                raise ReplayTimelineError(f"回放写回行缺少字段 {name}")
    history = list(state.get("history") or [])
    history.append(
        {
            "gate_action": gate_action,
            "requested_by": requested_by,
            "at": created_at,
            "detail": detail,
        }
    )
    new_state = {
        "mode": "replay",
        "run_id": timeline.run_id,
        "stage": new_stage,
        "history": history,
    }
    return ReplayOutcome(
        state=new_state,
        new_stage=new_stage,
        detail=detail,
        result_rows=rows,
        created_at=created_at,
        drives_node=drives_node,
    )


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    state: dict[str, Any]
    new_stage: str
    detail: str
    result_rows: list[dict[str, str]]
    created_at: str
    drives_node: int | None = None


def _result_row(
    run_id: str,
    *,
    record_kind: str,
    record_key: str,
    content: Mapping[str, Any],
) -> dict[str, str]:
    """写回行骨架；created_at 由 apply_action 统一盖戳。"""

    return {
        "run_id": run_id,
        "record_kind": record_kind,
        "record_key": record_key,
        "content": json.dumps(dict(content), ensure_ascii=False, sort_keys=True),
    }


def _apply_selection(
    timeline: ReplayTimeline, payload: Mapping[str, Any]
) -> tuple[str, str, list[dict[str, str]]]:
    selected = payload.get("selected_candidate_ids")
    primary = payload.get("primary_candidate_id")
    frozen_ids = timeline.selected_candidate_ids
    if (
        not isinstance(selected, list)
        or sorted(str(item) for item in selected) != sorted(frozen_ids)
    ):
        raise ReplayTimelineError(
            f"回放选线必须与冻结人工选线一致：{'、'.join(frozen_ids)}"
        )
    if primary not in frozen_ids:
        raise ReplayTimelineError(f"主叙事候选必须是选线集合之一：{primary!r}")
    row = _result_row(
        timeline.run_id,
        record_kind="SELECTION",
        record_key=f"{timeline.run_id}:selection",
        content={
            "selected_candidate_ids": frozen_ids,
            "primary_candidate_id": primary,
            "source": "blind/selection_confirmation.json",
            "mode": REPLAY_SOURCE_MARK,
        },
    )
    detail = f"选线 {'、'.join(frozen_ids)}（主叙事 {primary}）"
    return "HOLDOUT", detail, [row]


def _impact_counts(timeline: ReplayTimeline) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {
        candidate_id: {"SUPPORT": 0, "CHALLENGE": 0, "NEUTRAL": 0, "BLOCKING": 0, "TOTAL": 0}
        for candidate_id in timeline.selected_candidate_ids
    }
    for item in timeline.holdout_result["impacts"]:
        candidate_id = item["candidate_id"]
        bucket = counts.get(candidate_id)
        if bucket is None:
            continue
        impact = item["impact"]
        if impact in ("SUPPORT", "CHALLENGE", "NEUTRAL"):
            bucket[impact] += 1
        bucket["TOTAL"] += 1
        if item.get("is_blocking") is True:
            bucket["BLOCKING"] += 1
    return counts


def _apply_holdout(timeline: ReplayTimeline) -> tuple[str, str, list[dict[str, str]]]:
    counts = _impact_counts(timeline)
    evidence_ids = timeline.holdout_result.get("holdout_evidence_ids") or []
    rows = [
        _result_row(
            timeline.run_id,
            record_kind="HOLDOUT",
            record_key=f"{timeline.run_id}:holdout:{candidate_id}",
            content={
                "candidate_id": candidate_id,
                **counts[candidate_id],
                "source": timeline.holdout_source,
                "mode": REPLAY_SOURCE_MARK,
            },
        )
        for candidate_id in timeline.selected_candidate_ids
    ]
    total = sum(counts[item]["TOTAL"] for item in counts)
    blocking = sum(counts[item]["BLOCKING"] for item in counts)
    detail = (
        f"HOLDOUT 冻结证据 {len(evidence_ids)} 条，入围候选逐条判定 {total} 条，"
        f"阻断 {blocking} 条（来源 {timeline.holdout_source}）"
    )
    return "READY_REASSESS", detail, rows


def _apply_blind_reassessment(
    timeline: ReplayTimeline,
) -> tuple[str, str, list[dict[str, str]]]:
    sample_counts = timeline.blind_summary.get("sample_counts")
    if not isinstance(sample_counts, dict):
        raise ReplayTimelineError("blind_summary 缺少 sample_counts")
    row = _result_row(
        timeline.run_id,
        record_kind="BLIND_REASSESS",
        record_key=f"{timeline.run_id}:blind_reassess",
        content={
            "sample_counts": sample_counts,
            "source": "blind/blind_summary.json",
            "mode": REPLAY_SOURCE_MARK,
        },
    )
    detail = (
        f"盲评样本 q1-8 共 {sample_counts.get('q1_8')} 人、q9-12 共 "
        f"{sample_counts.get('q9_12')} 人，按冻结汇总重评"
    )
    return "CHECKPOINT_00", detail, [row]


def _apply_next_release(
    timeline: ReplayTimeline, stage: str
) -> tuple[str, str, list[dict[str, str]]]:
    if not stage.startswith("CHECKPOINT_"):
        raise ReplayTimelineError(f"释放增量必须在检查点阶段，当前 {stage}")
    next_index = int(stage[-2:]) + 1
    checkpoint = timeline.checkpoints[next_index]
    rows = []
    for candidate in sorted(checkpoint["candidates"], key=lambda item: item["rank"]):
        rows.append(
            _result_row(
                timeline.run_id,
                record_kind="CHECKPOINT",
                record_key=(
                    f"{timeline.run_id}:checkpoint-{next_index:02d}:"
                    f"{candidate['candidate_id']}"
                ),
                content={
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "candidate_id": candidate["candidate_id"],
                    "title": candidate.get("title"),
                    "rank": candidate["rank"],
                    "weighted_score": candidate.get("weighted_score"),
                    "score_change": candidate.get("score_change"),
                    "presentation_text": candidate.get("presentation_text"),
                    "new_evidence_ids": list(checkpoint.get("new_evidence_ids") or []),
                    "source": f"results/checkpoint-{next_index:02d}.json",
                    "mode": REPLAY_SOURCE_MARK,
                },
                )
        )
    detail = (
        f"释放第 {next_index} 批增量：{len(checkpoint.get('new_evidence_ids') or [])} 条新证据，"
        f"{len(checkpoint['candidates'])} 个候选重排"
    )
    return f"CHECKPOINT_{next_index:02d}", detail, rows


def _apply_final_selection(
    timeline: ReplayTimeline, payload: Mapping[str, Any]
) -> tuple[str, str, list[dict[str, str]]]:
    shortlisted = payload.get("shortlisted_candidate_ids")
    selected = payload.get("selected_candidate_id")
    recommended = payload.get("recommended_candidate_id")
    checkpoint_ids = sorted(
        item["candidate_id"] for item in timeline.checkpoints[4]["candidates"]
    )
    if (
        not isinstance(shortlisted, list)
        or sorted(str(item) for item in shortlisted) != checkpoint_ids
    ):
        raise ReplayTimelineError(
            f"终选入围必须与 checkpoint-04 候选一致：{'、'.join(checkpoint_ids)}"
        )
    ai_recommended = timeline.recommended_candidate_id()
    if recommended != ai_recommended:
        raise ReplayTimelineError(f"终选推荐候选必须与 checkpoint-04 AI 推荐一致：{ai_recommended}")
    if selected not in checkpoint_ids:
        raise ReplayTimelineError(f"终选候选必须位于入围集合内：{selected!r}")
    pillars = timeline.final_synthesis.get("pillars") or []
    row = _result_row(
        timeline.run_id,
        record_kind="FINAL_SELECTION",
        record_key=f"{timeline.run_id}:final_selection",
        content={
            "shortlisted_candidate_ids": checkpoint_ids,
            "selected_candidate_id": selected,
            "recommended_candidate_id": ai_recommended,
            "core_narrative": timeline.final_synthesis.get("core_narrative"),
            "source": "results/final-synthesis.json",
            "mode": REPLAY_SOURCE_MARK,
        },
    )
    detail = f"终选 {selected}；合成叙事 {len(pillars)} 大支柱（锚定 checkpoint-04）"
    return "COMPLETE", detail, [row]


def run_log_summary(label: str, new_stage: str, detail: str) -> str:
    """运行日志 summary；必须携带冻结回放标记。"""

    return f"【{REPLAY_SOURCE_MARK}】{label}完成，新阶段 {new_stage}；{detail}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "NODE_LABELS",
    "REPLAY_SOURCE_MARK",
    "REPLAY_STAGES",
    "gate_milestone_binding",
    "ReplayOutcome",
    "ReplayTimeline",
    "ReplayTimelineError",
    "apply_action",
    "initial_replay_state",
    "load_replay_timeline",
    "read_replay_state",
    "run_log_summary",
    "save_replay_state",
]
