"""Pure validation and status projection for Feishu decision gates.

机器人只呈现与排队人工确认过的门操作；本层不执行任何 LLM 调用，
不替人工选线（选线门必须携带 operator 显式确认的候选 ID）。
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .notification_bot import CallbackResult

if TYPE_CHECKING:
    from .decision_gate_store import DecisionGateStore


_STAGE_GATES: dict[str, tuple[str, ...]] = {
    "PREPARED": (),
    "AWAITING_SPECIFICITY_AUDIT": (
        "EXPORT_SPECIFICITY_AUDIT",
        "IMPORT_SPECIFICITY_AUDIT",
    ),
    "SPECIFICITY_REVISION": ("RUN_SPECIFICITY_REVISION",),
    "SPECIFICITY_FINAL_RANK": ("FINALIZE_SPECIFICITY",),
    "AWAITING_SELECTION": ("SUBMIT_SELECTION",),
    "HOLDOUT": ("RUN_HOLDOUT",),
    "AWAITING_PRIMARY_RESELECTION": ("RESELECT_PRIMARY",),
    # FREEZE_BLIND 需要盲测素材文件，仅本地 CLI 可执行，机器人不排队。
    "AWAITING_BLIND_EVIDENCE": (),
    "READY_REASSESS": ("RUN_BLIND_REASSESSMENT",),
    "CHECKPOINT_00": ("RUN_NEXT_RELEASE",),
    "CHECKPOINT_01": ("RUN_NEXT_RELEASE",),
    "CHECKPOINT_02": ("RUN_NEXT_RELEASE",),
    "CHECKPOINT_03": ("RUN_NEXT_RELEASE",),
    "CHECKPOINT_04": ("SUBMIT_FINAL_SELECTION",),
    "COMPLETE": (),
    "FAILED": (),
}

ALL_GATE_ACTIONS = frozenset(
    action for actions in _STAGE_GATES.values() for action in actions
)

_GATE_LABELS = {
    "EXPORT_SPECIFICITY_AUDIT": "导出专属度审查任务",
    "IMPORT_SPECIFICITY_AUDIT": "导入专属度审查结果",
    "RUN_SPECIFICITY_REVISION": "执行专属度修订",
    "FINALIZE_SPECIFICITY": "冻结专属度会话",
    "SUBMIT_SELECTION": "提交人工选线",
    "RUN_HOLDOUT": "执行 HOLDOUT 验证",
    "RESELECT_PRIMARY": "更换主叙事",
    "RUN_BLIND_REASSESSMENT": "执行盲测后重评",
    "RUN_NEXT_RELEASE": "释放下一批增量",
    "SUBMIT_FINAL_SELECTION": "提交最终选择",
}


class GateRejection(ValueError):
    """决策门请求被拒绝；消息面向操作者，可直接作为 toast。"""


def available_gates(stage: str) -> tuple[str, ...]:
    if not isinstance(stage, str) or stage not in _STAGE_GATES:
        raise GateRejection(f"未知决策阶段：{stage!r}")
    return _STAGE_GATES[stage]


def validate_gate_request(
    *,
    stage: str,
    gate_action: str,
    payload: Mapping[str, Any] | None,
    operator_open_id: str,
    allowed_operator_ids: Collection[str],
) -> None:
    """校验一个决策门请求；不合法即抛 GateRejection，绝不静默修正。"""

    if not isinstance(operator_open_id, str) or operator_open_id not in set(
        allowed_operator_ids
    ):
        raise GateRejection("你没有权限推进决策。")

    if gate_action not in ALL_GATE_ACTIONS:
        raise GateRejection(f"未知决策操作：{gate_action}")

    gates = available_gates(stage)
    if gate_action not in gates:
        label = "、".join(_GATE_LABELS.get(item, item) for item in gates) or "无"
        raise GateRejection(
            f"当前阶段 {stage} 不支持该操作；可用操作：{label}。"
        )

    _validate_payload(gate_action, payload)


def _validate_payload(gate_action: str, payload: Mapping[str, Any] | None) -> None:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise GateRejection("决策操作参数必须是对象。")

    if gate_action == "SUBMIT_SELECTION":
        selected = payload.get("selected_candidate_ids")
        primary = payload.get("primary_candidate_id")
        if (
            not isinstance(selected, list)
            or not selected
            or len(selected) > 3
            or not all(isinstance(item, str) and item.strip() for item in selected)
            or len(set(selected)) != len(selected)
        ):
            raise GateRejection("选线必须携带 1-3 个不重复的候选 ID。")
        if primary not in selected:
            raise GateRejection("主叙事候选必须位于选线集合内。")

    elif gate_action == "SUBMIT_FINAL_SELECTION":
        shortlisted = payload.get("shortlisted_candidate_ids")
        selected = payload.get("selected_candidate_id")
        recommended = payload.get("recommended_candidate_id")
        if (
            not isinstance(shortlisted, list)
            or not shortlisted
            or len(shortlisted) > 3
            or not all(isinstance(item, str) and item.strip() for item in shortlisted)
            or len(set(shortlisted)) != len(shortlisted)
        ):
            raise GateRejection("最终选择必须携带 1-3 个不重复的入围候选 ID。")
        if recommended not in shortlisted:
            raise GateRejection("推荐候选必须位于入围集合内。")
        if selected not in shortlisted:
            raise GateRejection("最终选择候选必须位于入围集合内。")

    elif gate_action == "RESELECT_PRIMARY":
        primary = payload.get("primary_candidate_id")
        if not isinstance(primary, str) or not primary.strip():
            raise GateRejection("更换主叙事必须携带候选 ID。")

    elif gate_action == "IMPORT_SPECIFICITY_AUDIT":
        result_path = payload.get("result_path")
        if not isinstance(result_path, str) or not result_path.strip():
            raise GateRejection("导入审查结果必须携带结果文件路径。")


def stage_status_text(state: Mapping[str, Any]) -> str:
    """把 coordinator state 投影为“决策进度”回复文本。"""

    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    run_id = state.get("run_id")
    stage = state.get("stage")
    for name, value in (("run_id", run_id), ("stage", stage)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"state.{name} must be non-empty")

    lines = [f"决策 run：{run_id}", f"当前阶段：{stage}"]
    if stage == "FAILED":
        resume = state.get("resume_stage")
        lines.append(
            "该 run 处于失败态；请在本机排查 failures 记录，"
            + (f"失败前阶段为 {resume}。" if isinstance(resume, str) and resume else "")
        )
        return "\n".join(lines)

    gates = available_gates(stage)
    if gates:
        lines.append("可推进操作：")
        lines.extend(f"- {item}（{_GATE_LABELS.get(item, item)}）" for item in gates)
    else:
        lines.append("当前阶段没有可由机器人推进的操作。")
    return "\n".join(lines)


def resolve_result_path(*, run_root: str | Path, result_path: str) -> Path:
    """导入路径必须位于 run 目录内，防越界引用。"""

    root = Path(run_root).resolve()
    candidate = (root / result_path).resolve() if not Path(result_path).is_absolute() else Path(result_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise GateRejection("审查结果路径必须位于该 run 目录内。") from exc
    return candidate


DECISION_GATE_ACTION_TYPE = "DECISION_GATE"


def handle_decision_gate_action(
    payload: Mapping[str, Any] | None,
    *,
    operator_open_id: str,
    allowed_operator_ids: Collection[str],
    load_state: Callable[[str], Mapping[str, Any] | None],
    gate_store: "DecisionGateStore",
    gate_enabled: bool,
) -> CallbackResult:
    """校验并排对一个决策门请求；不在回调线程里执行任何门动作。"""

    if not gate_enabled:
        return CallbackResult(
            toast_type="warning",
            toast_content="决策门执行未启用：请先在本机配置 FEISHU_DECISION_RUN_ROOT。",
        )
    if not isinstance(payload, Mapping):
        return CallbackResult(toast_type="error", toast_content="决策门请求格式无效。")

    run_id = payload.get("run_id")
    gate_action = payload.get("gate_action")
    inner_payload = payload.get("payload")
    if not isinstance(run_id, str) or not run_id.strip():
        return CallbackResult(toast_type="error", toast_content="决策门请求缺少 run_id。")
    if not isinstance(gate_action, str) or not gate_action.strip():
        return CallbackResult(toast_type="error", toast_content="决策门请求缺少操作名。")
    if inner_payload is not None and not isinstance(inner_payload, Mapping):
        return CallbackResult(toast_type="error", toast_content="决策门参数必须是对象。")

    state = load_state(run_id)
    if state is None:
        return CallbackResult(toast_type="error", toast_content=f"未找到决策 run：{run_id}")
    stage = state.get("stage")
    try:
        validate_gate_request(
            stage=stage,
            gate_action=gate_action,
            payload=inner_payload,
            operator_open_id=operator_open_id,
            allowed_operator_ids=allowed_operator_ids,
        )
    except GateRejection as exc:
        return CallbackResult(toast_type="warning", toast_content=str(exc))

    try:
        gate_store.enqueue_gate(
            run_id,
            gate_action=gate_action,
            payload=dict(inner_payload) if inner_payload is not None else {},
            requested_by=operator_open_id,
        )
    except ValueError as exc:
        return CallbackResult(toast_type="warning", toast_content=str(exc))

    return CallbackResult(
        toast_type="success",
        toast_content=f"已排队：{_GATE_LABELS.get(gate_action, gate_action)}。执行结果将以卡片推送。",
    )


__all__ = [
    "ALL_GATE_ACTIONS",
    "DECISION_GATE_ACTION_TYPE",
    "GateRejection",
    "available_gates",
    "handle_decision_gate_action",
    "resolve_result_path",
    "stage_status_text",
    "validate_gate_request",
]
