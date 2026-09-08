"""Local presenter controls for the Feishu demo notification session."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from src.integrations.feishu.notification_store import NotificationStore
from src.services.demo_notifications import DemoNotificationNode
from src.state import initialize_state
from src.ui.pressure_test_workspace import PRESSURE_STAGES
from src.ui.realtime_decision_dashboard import PlaybackPhase
from src.ui.workspace_shell import Workspace, complete_workspace


_PLAYBACK_PREFIXES = ("preprocessing_replay_", "pressure_", "evolution_")
_CONFIRM_KEY = "demo_notification_confirm_new_session"
_SESSION_KEY = "demo_notification_session_id"
_DELTA_NODE_BY_CHECKPOINT_INDEX = {
    2: DemoNotificationNode.DELTA_01_COMPLETE,
    3: DemoNotificationNode.DELTA_02_COMPLETE,
    4: DemoNotificationNode.DELTA_03_COMPLETE,
    5: DemoNotificationNode.DELTA_04_COMPLETE,
}


_PREPROCESSING_STAGE_LABELS = ("结构校验", "清洗", "拆分", "标注", "冻结")
_PHASE_LABELS = {
    "comments": "增量评论进入",
    "score": "分数与排名更新",
    "narrative": "叙事文案更新",
}


def _advance_preprocessing_step(state: MutableMapping[str, Any]) -> str:
    completed = list(state.get("preprocessing_replay_completed_steps") or [])
    current = int(state.get("preprocessing_replay_step", 0) or 0)
    if len(completed) >= 5:
        complete_workspace(state, Workspace.PREPROCESSING)
        state["active_workspace"] = Workspace.STRESS_TEST.value
        return "数据预处理已完成，进入叙事压力测试"
    if current not in completed:
        completed.append(current)
    completed.sort()
    state["preprocessing_replay_completed_steps"] = completed
    state["preprocessing_replay_running"] = False
    label = _PREPROCESSING_STAGE_LABELS[current]
    if current >= 4:
        complete_workspace(state, Workspace.PREPROCESSING)
        state["active_workspace"] = Workspace.STRESS_TEST.value
        return f"数据预处理 · 「{label}」完成（5/5），进入叙事压力测试"
    state["preprocessing_replay_step"] = current + 1
    return f"数据预处理 · 「{label}」完成（{current + 1}/5）"


def _advance_pressure_step(state: MutableMapping[str, Any]) -> str:
    stage = state.get("pressure_stage") or PRESSURE_STAGES[0]
    index = PRESSURE_STAGES.index(stage)
    if index < len(PRESSURE_STAGES) - 1:
        nxt = PRESSURE_STAGES[index + 1]
        state["pressure_stage"] = nxt
        return f"叙事压力测试 · 进入「{nxt}」（{index + 2}/5）"
    if not state.get("pressure_team_confirmed"):
        state["pressure_team_confirmed"] = True
        complete_workspace(state, Workspace.STRESS_TEST)
        state["active_workspace"] = Workspace.REALTIME_DECISION.value
        return "真人盲评 5→3 已团队确认；实时决策看板已解锁"
    state["active_workspace"] = Workspace.REALTIME_DECISION.value
    return "已在实时决策看板"


def _advance_evolution_step(state: MutableMapping[str, Any]) -> str:
    from src.ui.realtime_decision_dashboard import (
        PlaybackAction,
        PlaybackPhase,
        apply_playback_action,
    )

    if state.get("evolution_show_finale"):
        act = int(state.get("evolution_finale_act", 1) or 1)
        if act < 2:
            state["evolution_finale_act"] = 2
            return "终幕 · 第二幕已展开（三支柱、场景与边界）"
        return "演示已到终幕结尾"
    before_finale = False
    previous = (
        int(state.get("evolution_checkpoint_index", 0) or 0),
        str(state.get("evolution_phase") or PlaybackPhase.SCORE.value),
    )
    apply_playback_action(state, PlaybackAction.NEXT_CHANGE)
    if state.get("evolution_show_finale"):
        return "五个检查点全部释放完毕，进入终幕 · 第一幕"
    checkpoint_index = int(state.get("evolution_checkpoint_index", 0) or 0)
    phase = str(state.get("evolution_phase") or PlaybackPhase.SCORE.value)
    if (checkpoint_index, phase) == previous:
        return "当前无可推进的变化"
    phase_label = _PHASE_LABELS.get(phase, phase)
    return f"实时决策看板 · 检查点 {checkpoint_index}：{phase_label}"


def advance_demo_one_step(state: MutableMapping[str, Any]) -> str:
    """把演示推进恰好一个小步；返回本步反馈文本。"""

    initialize_state(state)
    workspace = state.get("active_workspace")
    if workspace == Workspace.REALTIME_DECISION.value or state.get(
        "pressure_team_confirmed"
    ):
        return _advance_evolution_step(state)
    if workspace == Workspace.STRESS_TEST.value:
        return _advance_pressure_step(state)
    return _advance_preprocessing_step(state)


def current_demo_progress_text(state: MutableMapping[str, Any]) -> str:
    """把当前演示状态投影为一行进度文本。"""

    initialize_state(state)
    workspace = state.get("active_workspace")
    if state.get("evolution_show_finale"):
        act = int(state.get("evolution_finale_act", 1) or 1)
        return f"实时决策看板 · 终幕 第{act}/2幕"
    if state.get("pressure_team_confirmed") or workspace == Workspace.REALTIME_DECISION.value:
        checkpoint_index = int(state.get("evolution_checkpoint_index", 0) or 0)
        phase = str(state.get("evolution_phase") or "score")
        phase_label = _PHASE_LABELS.get(phase, phase)
        return f"实时决策看板 · 检查点 {checkpoint_index}/5 · {phase_label}"
    if workspace == Workspace.STRESS_TEST.value:
        stage = state.get("pressure_stage") or PRESSURE_STAGES[0]
        index = PRESSURE_STAGES.index(stage)
        return f"叙事压力测试 · 「{stage}」（{index + 1}/5）"
    completed = list(state.get("preprocessing_replay_completed_steps") or [])
    if len(completed) >= 5:
        return "数据预处理 · 已完成（5/5）"
    current = int(state.get("preprocessing_replay_step", 0) or 0)
    label = _PREPROCESSING_STAGE_LABELS[current]
    return f"数据预处理 · 「{label}」（{len(completed)}/5）"


def apply_milestone_state(state: MutableMapping[str, Any], node: int) -> None:
    """把演示状态直接推进到里程碑 node（1-8）的完成点；幂等。"""

    if not isinstance(node, int) or isinstance(node, bool) or not 1 <= node <= 8:
        raise ValueError("node must be between 1 and 8")
    initialize_state(state)
    if node >= 1:
        state["preprocessing_replay_completed_steps"] = list(range(5))
        complete_workspace(state, Workspace.PREPROCESSING)
    if node >= 2:
        state["pressure_stage"] = PRESSURE_STAGES[4]
    if node >= 3:
        state["pressure_team_confirmed"] = True
        complete_workspace(state, Workspace.STRESS_TEST)
    if 4 <= node <= 7:
        state["evolution_phase"] = PlaybackPhase.NARRATIVE.value
        state["evolution_checkpoint_index"] = node - 2
    if node >= 8:
        state["evolution_show_finale"] = True
        state["evolution_finale_act"] = 2
    if node >= 3:
        state["active_workspace"] = Workspace.REALTIME_DECISION.value
    elif node >= 1:
        state["active_workspace"] = Workspace.STRESS_TEST.value


def _consume_remote_commands(
    state: MutableMapping[str, Any], store: "NotificationStore", st: Any
) -> None:
    session = store.active_session()
    if session is None:
        return
    applied = False
    for command in store.pending_demo_commands(session.session_id):
        if command.command == "ADVANCE_TO_NODE":
            from src.services.replay_gate_timeline import NODE_LABELS

            apply_milestone_state(state, int(command.target_node))
            store.consume_demo_command(
                command.command_id,
                result_text=f"已跳转到里程碑：{NODE_LABELS.get(command.target_node)}（节点 {command.target_node}/8）",
            )
            applied = True
        elif command.command == "ADVANCE_STEP":
            feedback = advance_demo_one_step(state)
            store.consume_demo_command(command.command_id, result_text=feedback)
            applied = True
    store.upsert_demo_progress(session.session_id, current_demo_progress_text(state))
    if applied:
        try:
            st.rerun(scope="app")
        except TypeError:
            st.rerun()


def render_remote_command_listener(
    state: MutableMapping[str, Any],
    store: "NotificationStore | None",
    *,
    streamlit_module: Any = None,
) -> None:
    """轮询远端推进指令并驱动本页演示状态；有 st.fragment 时自动刷新。"""

    st = _get_streamlit(streamlit_module)
    if store is None:
        return
    fragment = getattr(st, "fragment", None)
    if fragment is not None:
        @fragment(run_every="2s")
        def _poll_remote_commands() -> None:
            _consume_remote_commands(state, store, st)

        _poll_remote_commands()
    else:
        _consume_remote_commands(state, store, st)


def reset_demo_playback_state(state: MutableMapping[str, Any]) -> None:
    """Reset only the playback and ordered-workspace state owned by the demo."""

    for key in list(state):
        if key.startswith(_PLAYBACK_PREFIXES):
            del state[key]
    state["active_workspace"] = Workspace.PREPROCESSING.value
    state["completed_workspaces"] = []


def _get_streamlit(streamlit_module: Any | None) -> Any:
    if streamlit_module is not None:
        return streamlit_module
    import streamlit as st

    return st


def _enqueue_ready_snapshots(
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
    nodes: tuple[DemoNotificationNode, ...],
) -> None:
    if store is None:
        return
    session = store.active_session()
    if session is None:
        return
    snapshots_by_node = {snapshot.node: snapshot for snapshot in snapshots}
    for node in nodes:
        snapshot = snapshots_by_node[node]
        try:
            store.enqueue_snapshot(session.session_id, snapshot)
        except Exception as exc:
            _get_streamlit(None).error(f"飞书推送入队失败：{exc}")


def sync_preprocessing_notification(
    state: MutableMapping[str, Any],
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
) -> None:
    """Enqueue node 1 only after the official five-step replay is complete."""

    completed = state.get("preprocessing_replay_completed_steps")
    if not isinstance(completed, (list, tuple, set)):
        return
    if len(completed) != 5 or any(type(step) is not int for step in completed):
        return
    if set(completed) != set(range(5)):
        return
    _enqueue_ready_snapshots(
        store,
        snapshots,
        (DemoNotificationNode.PREPROCESSING_COMPLETE,),
    )


def sync_pressure_notifications(
    state: MutableMapping[str, Any],
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
) -> None:
    """Enqueue pressure-test nodes at their visible completion boundaries."""

    ready: list[DemoNotificationNode] = []
    if state.get("pressure_stage") == PRESSURE_STAGES[4]:
        ready.append(DemoNotificationNode.PRESSURE_TEST_COMPLETE)
    if state.get("pressure_team_confirmed") is True:
        ready.append(DemoNotificationNode.BLIND_SELECTION_COMPLETE)
    _enqueue_ready_snapshots(store, snapshots, tuple(ready))


def sync_evolution_notifications(
    state: MutableMapping[str, Any],
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
) -> None:
    """Enqueue narrative deltas and the finale only at their rendered acts."""

    ready: list[DemoNotificationNode] = []
    if state.get("evolution_phase") == PlaybackPhase.NARRATIVE.value:
        checkpoint_index = state.get("evolution_checkpoint_index")
        if type(checkpoint_index) is int:
            node = _DELTA_NODE_BY_CHECKPOINT_INDEX.get(checkpoint_index)
            if node is not None:
                ready.append(node)
    if (
        state.get("evolution_show_finale") is True
        and state.get("evolution_finale_act") == 2
    ):
        ready.append(DemoNotificationNode.FINAL_SYNTHESIS_COMPLETE)
    _enqueue_ready_snapshots(store, snapshots, tuple(ready))


def _missing_configuration(store: NotificationStore | None, source_run_id: str) -> list[str]:
    missing: list[str] = []
    if store is None:
        missing.append("FEISHU_NOTIFICATION_DB")
    if not isinstance(source_run_id, str) or not source_run_id.strip():
        missing.append("source_run_id")
    return missing


def render_demo_notification_control(
    state: MutableMapping[str, Any],
    *,
    store: NotificationStore | None,
    source_run_id: str,
    configuration: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    streamlit_module: Any = None,
) -> None:
    """Render a compact, local-only session and delivery-status control."""

    st = _get_streamlit(streamlit_module)
    st.markdown("### 飞书机器人助手")
    st.caption("连接当前系统会话并同步阶段通知；不提供用户认证，也不显示密钥。")

    if configuration is not None:
        try:
            from tools.run_feishu_bot import REQUIRED_CONFIG_KEYS, load_bot_config

            missing_keys = [
                key
                for key in REQUIRED_CONFIG_KEYS
                if not isinstance(configuration.get(key), str)
                or not configuration.get(key, "").strip()
            ]
            if missing_keys:
                st.info("飞书仅在本地配置后触发；公开站不会连接团队飞书。")
                st.button("开始新的案例会话", disabled=True)
                return
            load_bot_config(configuration, workspace_root=workspace_root)
        except (TypeError, ValueError) as exc:
            st.error(f"通知控制不可用：{str(exc).splitlines()[0]}")
            return
        if store is None:
            st.error("通知控制不可用：FEISHU_NOTIFICATION_DB")
            return
        missing = []
    else:
        missing = _missing_configuration(store, source_run_id)
    if (
        not isinstance(source_run_id, str)
        or not source_run_id.strip()
    ) and "source_run_id" not in missing:
        missing.append("source_run_id")
    if missing:
        st.error(f"通知控制不可用：缺少 {'、'.join(missing)}。")
        return

    confirming = bool(state.get(_CONFIRM_KEY, False))
    if st.button("开始新的案例会话", key="demo-notification-new-session") and not confirming:
        state[_CONFIRM_KEY] = True
        confirming = True

    if confirming:
        st.warning("这会关闭当前通知会话并重置案例播放进度。请再次确认。")
        if st.button("确认开始新的案例会话", key="demo-notification-confirm-session"):
            session = store.create_session(source_run_id, created_by="streamlit-local")
            reset_demo_playback_state(state)
            state[_SESSION_KEY] = session.session_id
            state[_CONFIRM_KEY] = False
            st.info(f"已创建本地案例通知会话：{session.session_id}")

__all__ = [
    "advance_demo_one_step",
    "apply_milestone_state",
    "current_demo_progress_text",
    "render_demo_notification_control",
    "render_remote_command_listener",
    "reset_demo_playback_state",
    "sync_evolution_notifications",
    "sync_preprocessing_notification",
    "sync_pressure_notifications",
]
