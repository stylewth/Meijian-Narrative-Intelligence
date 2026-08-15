"""Local presenter controls for the Feishu demo notification session."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from src.integrations.feishu.notification_store import NotificationStore
from src.services.demo_notifications import DemoNotificationNode
from src.ui.pressure_test_workspace import PRESSURE_STAGES
from src.ui.realtime_decision_dashboard import PlaybackPhase
from src.ui.workspace_shell import Workspace


_PLAYBACK_PREFIXES = ("preprocessing_replay_", "pressure_", "evolution_")
_CONFIRM_KEY = "demo_notification_confirm_new_session"
_SESSION_KEY = "demo_notification_session_id"
_DELTA_NODE_BY_CHECKPOINT_INDEX = {
    2: DemoNotificationNode.DELTA_01_COMPLETE,
    3: DemoNotificationNode.DELTA_02_COMPLETE,
    4: DemoNotificationNode.DELTA_03_COMPLETE,
    5: DemoNotificationNode.DELTA_04_COMPLETE,
}


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
    "render_demo_notification_control",
    "reset_demo_playback_state",
    "sync_evolution_notifications",
    "sync_preprocessing_notification",
    "sync_pressure_notifications",
]
