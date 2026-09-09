"""Local presenter controls for the Feishu demo notification session."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from threading import Thread
from typing import Any
import uuid

from src.integrations.feishu.notification_store import NotificationStore
from src.integrations.feishu.client import FeishuBitableClient
from src.integrations.feishu.writeback_store import WritebackStore
from src.integrations.feishu.writeback_targets import (
    WritebackTarget,
    provision_demo_writeback_targets,
)
from src.integrations.feishu.result_projection import milestone_row, run_log_fields
from src.services.demo_notifications import DemoNotificationNode
from src.state import initialize_state
from src.ui.pressure_test_workspace import PRESSURE_STAGES
from src.ui.realtime_decision_dashboard import PlaybackPhase
from src.ui.workspace_shell import Workspace, complete_workspace


_PLAYBACK_PREFIXES = ("preprocessing_replay_", "pressure_", "evolution_")
_CONFIRM_KEY = "demo_notification_confirm_new_session"
_SESSION_KEY = "demo_notification_session_id"
_PAGE_CONNECTION_INITIALIZED_KEY = "feishu_page_connection_initialized"
_CONNECTION_TASK_KEY = "feishu_connection_task"
_CONNECTION_NOTICE_KEY = "feishu_connection_notice"
_CURRENT_PROGRESS_EVENT_KEY = "progress:current:log"
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


@dataclass(frozen=True, slots=True)
class _ConnectionTask:
    future: Future[dict[str, WritebackTarget]]
    source_run_id: str
    demo_run_id: str
    started_at: str


def _provision_connection_targets(
    *,
    app_id: str,
    app_secret: str,
    run_log_url: str | None,
    results_url: str | None,
    demo_run_id: str,
    started_at: str,
) -> dict[str, WritebackTarget]:
    bitable = FeishuBitableClient(app_id, app_secret)
    return provision_demo_writeback_targets(
        bitable,
        run_log_url=run_log_url,
        results_url=results_url,
        demo_run_id=demo_run_id,
        started_at=started_at,
    )


def _start_connection_task(
    *,
    app_id: str,
    app_secret: str,
    run_log_url: str | None,
    results_url: str | None,
    source_run_id: str,
    demo_run_id: str,
    started_at: str,
) -> _ConnectionTask:
    future: Future[dict[str, WritebackTarget]] = Future()

    def worker() -> None:
        try:
            future.set_result(
                _provision_connection_targets(
                    app_id=app_id,
                    app_secret=app_secret,
                    run_log_url=run_log_url,
                    results_url=results_url,
                    demo_run_id=demo_run_id,
                    started_at=started_at,
                )
            )
        except Exception as exc:
            future.set_exception(exc)

    Thread(
        target=worker,
        name="feishu-connection-provisioner",
        daemon=True,
    ).start()
    return _ConnectionTask(
        future=future,
        source_run_id=source_run_id,
        demo_run_id=demo_run_id,
        started_at=started_at,
    )


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
        return "已到终幕结尾"
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
    """把当前进度推进恰好一个小步；返回本步反馈文本。"""

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
    """把当前进度投影为一行状态文本。"""

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
    """把当前状态直接推进到里程碑 node（1-8）的完成点；幂等。"""

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
    """轮询远端推进指令并驱动本页状态；有 st.fragment 时自动刷新。"""

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


def _ready_nodes_for_state(
    state: Mapping[str, Any],
) -> tuple[DemoNotificationNode, ...]:
    """Return every milestone already visible in the current page state."""

    ready: list[DemoNotificationNode] = []
    completed = state.get("preprocessing_replay_completed_steps")
    if (
        isinstance(completed, (list, tuple, set))
        and all(type(step) is int for step in completed)
        and set(completed) == set(range(5))
    ):
        ready.append(DemoNotificationNode.PREPROCESSING_COMPLETE)

    if state.get("pressure_stage") == PRESSURE_STAGES[-1]:
        ready.append(DemoNotificationNode.PRESSURE_TEST_COMPLETE)
    if state.get("pressure_team_confirmed") is True:
        ready.append(DemoNotificationNode.BLIND_SELECTION_COMPLETE)

    evolution_ready = state.get("evolution_show_finale") is True or state.get(
        "evolution_phase"
    ) == PlaybackPhase.NARRATIVE.value
    checkpoint_index = state.get("evolution_checkpoint_index")
    if evolution_ready and type(checkpoint_index) is int:
        for index, node in _DELTA_NODE_BY_CHECKPOINT_INDEX.items():
            if checkpoint_index >= index:
                ready.append(node)
    if state.get("evolution_show_finale") is True and state.get(
        "evolution_finale_act"
    ) == 2:
        ready.append(DemoNotificationNode.FINAL_SYNTHESIS_COMPLETE)
    return tuple(ready)


def _get_streamlit(streamlit_module: Any | None) -> Any:
    if streamlit_module is not None:
        return streamlit_module
    import streamlit as st

    return st


def _sync_current_progress(
    state: MutableMapping[str, Any],
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
    *,
    writeback_store: WritebackStore | None = None,
) -> int:
    """Keep the local progress row and the dynamic write-back tables in sync."""

    if store is None:
        return 0
    session = store.active_session()
    if session is None:
        return 0

    progress_text = current_demo_progress_text(state)
    store.upsert_demo_progress(session.session_id, progress_text)
    ready_nodes = _ready_nodes_for_state(state)
    if ready_nodes:
        if not snapshots:
            raise ValueError("current progress snapshots are required for an active session")
        _enqueue_ready_snapshots(
            store,
            snapshots,
            ready_nodes,
            enqueue_notifications=False,
            writeback_store=writeback_store,
        )

    if writeback_store is not None:
        targets = store.writeback_targets_for_session(session.session_id)
        if set(targets) != {"RUN_LOG", "RESULTS"}:
            raise ValueError("当前连接缺少两张写回表映射")
        progress_fields = run_log_fields(
            session.source_run_id,
            action="CURRENT_PROGRESS",
            actor_open_id="streamlit-local",
            summary=progress_text,
            demo_run_id=session.session_id,
            demo_started_at=session.created_at,
            created_at=session.created_at,
        )
        writeback_store.enqueue(
            session.source_run_id,
            table_key="RUN_LOG",
            record_kind="RUN_LOG",
            fields=progress_fields,
            demo_run_id=session.session_id,
            event_key=_CURRENT_PROGRESS_EVENT_KEY,
            target=targets["RUN_LOG"],
        )
    return len(ready_nodes)


def _enqueue_ready_snapshots(
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
    nodes: tuple[DemoNotificationNode, ...],
    *,
    enqueue_notifications: bool = True,
    writeback_store: WritebackStore | None = None,
) -> None:
    if store is None:
        return
    session = store.active_session()
    if session is None:
        return
    snapshots_by_node = {snapshot.node: snapshot for snapshot in snapshots}
    targets = (
        store.writeback_targets_for_session(session.session_id)
        if writeback_store is not None
        else {}
    )
    event_created_at = datetime.fromisoformat(session.created_at)
    for node in nodes:
        snapshot = snapshots_by_node[node]
        try:
            if enqueue_notifications:
                store.enqueue_snapshot(session.session_id, snapshot)
            if writeback_store is not None:
                if set(targets) != {"RUN_LOG", "RESULTS"}:
                    raise ValueError("当前连接缺少两张写回表映射")
                event_prefix = f"milestone:{node.value}"
                log_fields = run_log_fields(
                    session.source_run_id,
                    action=event_prefix,
                    actor_open_id="streamlit-local",
                    summary=snapshot.title,
                    demo_run_id=session.session_id,
                    demo_started_at=session.created_at,
                    created_at=session.created_at,
                )
                writeback_store.enqueue(
                    session.source_run_id,
                    table_key="RUN_LOG",
                    record_kind="RUN_LOG",
                    fields=log_fields,
                    demo_run_id=session.session_id,
                    event_key=f"{event_prefix}:log",
                    target=targets["RUN_LOG"],
                )
                result_fields = milestone_row(
                    snapshot,
                    demo_run_id=session.session_id,
                    demo_started_at=session.created_at,
                    created_at=event_created_at,
                )
                writeback_store.enqueue(
                    session.source_run_id,
                    table_key="RESULTS",
                    record_kind="MILESTONE",
                    fields=result_fields,
                    demo_run_id=session.session_id,
                    event_key=f"{event_prefix}:result",
                    target=targets["RESULTS"],
                )
        except Exception as exc:
            _get_streamlit(None).error(f"飞书推送入队失败：{exc}")


def sync_preprocessing_notification(
    state: MutableMapping[str, Any],
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
    *,
    writeback_store: WritebackStore | None = None,
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
        writeback_store=writeback_store,
    )


def sync_pressure_notifications(
    state: MutableMapping[str, Any],
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
    *,
    writeback_store: WritebackStore | None = None,
) -> None:
    """Enqueue pressure-test nodes at their visible completion boundaries."""

    ready: list[DemoNotificationNode] = []
    if state.get("pressure_stage") == PRESSURE_STAGES[4]:
        ready.append(DemoNotificationNode.PRESSURE_TEST_COMPLETE)
    if state.get("pressure_team_confirmed") is True:
        ready.append(DemoNotificationNode.BLIND_SELECTION_COMPLETE)
    _enqueue_ready_snapshots(
        store, snapshots, tuple(ready), writeback_store=writeback_store
    )


def sync_evolution_notifications(
    state: MutableMapping[str, Any],
    store: NotificationStore | None,
    snapshots: tuple[Any, ...],
    *,
    writeback_store: WritebackStore | None = None,
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
    _enqueue_ready_snapshots(
        store, snapshots, tuple(ready), writeback_store=writeback_store
    )


def _missing_configuration(store: NotificationStore | None, source_run_id: str) -> list[str]:
    missing: list[str] = []
    if store is None:
        missing.append("FEISHU_NOTIFICATION_DB")
    if not isinstance(source_run_id, str) or not source_run_id.strip():
        missing.append("source_run_id")
    return missing


def initialize_feishu_page_connection(
    state: MutableMapping[str, Any], store: NotificationStore | None
) -> None:
    """Start a fresh page-level connection scope after a browser refresh."""

    if state.get(_PAGE_CONNECTION_INITIALIZED_KEY) is True:
        return
    state[_PAGE_CONNECTION_INITIALIZED_KEY] = True
    state.pop(_SESSION_KEY, None)
    state.pop(_CONNECTION_TASK_KEY, None)
    state.pop(_CONNECTION_NOTICE_KEY, None)
    if store is not None:
        store.complete_active_session()


def _beijing_time_text(value: str) -> str:
    timestamp = datetime.fromisoformat(value)
    return timestamp.astimezone(timezone(timedelta(hours=8))).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _connection_duration_seconds(
    started_at: str, connected_at: str | None
) -> float | None:
    if connected_at is None:
        return None
    duration = (
        datetime.fromisoformat(connected_at) - datetime.fromisoformat(started_at)
    ).total_seconds()
    return max(0.0, duration)


def _render_active_connection(
    st: Any,
    store: NotificationStore,
    *,
    state: MutableMapping[str, Any],
) -> None:
    session = store.active_session()
    if session is None or state.get(_SESSION_KEY) != session.session_id:
        return
    targets = store.writeback_targets_for_session(session.session_id)
    if set(targets) != {"RUN_LOG", "RESULTS"}:
        return
    caption = f"飞书启动时间：{_beijing_time_text(session.created_at)}"
    if session.connected_at is not None:
        duration = _connection_duration_seconds(
            session.created_at, session.connected_at
        )
        caption += (
            f"｜连接完成时间：{_beijing_time_text(session.connected_at)}"
            f"｜连接耗时：{duration:.1f} 秒"
        )
    caption += f"｜demo_run_id：{session.session_id}"
    st.caption(caption)
    st.markdown(
        f"[打开日志表]({targets['RUN_LOG'].table_url})  ·  "
        f"[打开结果表]({targets['RESULTS'].table_url})"
    )


def _connection_failure_detail(
    exc: Exception, targets: dict[str, WritebackTarget] | None
) -> str:
    detail = str(exc).splitlines()[0] or exc.__class__.__name__
    if targets:
        urls = [
            target.table_url
            for key in ("RUN_LOG", "RESULTS")
            if (target := targets.get(key)) is not None and target.table_url
        ]
        if urls:
            detail += " 已创建但尚未激活：" + "、".join(urls)
    partial_url = getattr(exc, "partial_table_url", None)
    if partial_url:
        detail += f" 已创建但尚未激活：{partial_url}"
    return detail


def _render_connection_notice(
    st: Any, state: MutableMapping[str, Any]
) -> None:
    notice = state.pop(_CONNECTION_NOTICE_KEY, None)
    if not isinstance(notice, tuple) or len(notice) != 2:
        return
    kind, message = notice
    if kind == "success":
        st.success(message)
    elif kind == "error":
        st.error(message)


def _render_connection_task_once(
    st: Any,
    state: MutableMapping[str, Any],
    store: NotificationStore,
    *,
    snapshots: tuple[Any, ...] = (),
    writeback_store: WritebackStore | None = None,
) -> None:
    task = state.get(_CONNECTION_TASK_KEY)
    if not isinstance(task, _ConnectionTask):
        return
    if not task.future.done():
        st.info(
            "正在连接飞书并创建两张空白写回表，请稍候。"
            "页面不会被锁定，当前网页进度保持不变，连接完成后会同步当前进度。"
        )
        return

    targets: dict[str, WritebackTarget] | None = None
    session = None
    try:
        targets = task.future.result()
        connected_at = datetime.now(timezone.utc).isoformat()
        session = store.create_session(
            task.source_run_id,
            created_by="streamlit-local",
            session_id=task.demo_run_id,
            created_at=task.started_at,
            connected_at=connected_at,
            writeback_targets=targets,
        )
    except Exception as exc:
        state.pop(_CONNECTION_TASK_KEY, None)
        state[_CONFIRM_KEY] = False
        state[_CONNECTION_NOTICE_KEY] = (
            "error",
            f"连接助手失败：{_connection_failure_detail(exc, targets)}",
        )
    else:
        state.pop(_CONNECTION_TASK_KEY, None)
        state[_SESSION_KEY] = session.session_id
        state[_CONFIRM_KEY] = False
        try:
            reached_count = _sync_current_progress(
                state,
                store,
                snapshots,
                writeback_store=writeback_store,
            )
        except Exception as exc:
            state[_CONNECTION_NOTICE_KEY] = (
                "error",
                f"已连接，但当前进度同步失败：{str(exc).splitlines()[0]}",
            )
        else:
            duration = _connection_duration_seconds(
                session.created_at, session.connected_at
            )
            state[_CONNECTION_NOTICE_KEY] = (
                "success",
                "已连接，网页当前进度已保留并加入飞书同步队列；"
                f"已同步 {reached_count} 个已完成节点。"
                f"连接耗时：{duration:.1f} 秒。",
            )
    _render_connection_notice(st, state)
    if session is not None:
        _render_active_connection(st, store, state=state)


def _render_connection_task(
    st: Any,
    state: MutableMapping[str, Any],
    store: NotificationStore,
    *,
    snapshots: tuple[Any, ...] = (),
    writeback_store: WritebackStore | None = None,
    use_fragment: bool = True,
) -> bool:
    if not isinstance(state.get(_CONNECTION_TASK_KEY), _ConnectionTask):
        return False
    fragment = getattr(st, "fragment", None)
    if use_fragment and callable(fragment):
        @fragment(run_every="1s")
        def _poll_connection_task() -> None:
            _render_connection_task_once(
                st,
                state,
                store,
                snapshots=snapshots,
                writeback_store=writeback_store,
            )

        _poll_connection_task()
    else:
        _render_connection_task_once(
            st,
            state,
            store,
            snapshots=snapshots,
            writeback_store=writeback_store,
        )
    return True


def render_demo_notification_control(
    state: MutableMapping[str, Any],
    *,
    store: NotificationStore | None,
    source_run_id: str,
    configuration: Mapping[str, Any] | None = None,
    workspace_root: str | Path | None = None,
    writeback_store: WritebackStore | None = None,
    snapshots: tuple[Any, ...] = (),
    streamlit_module: Any = None,
    use_connection_fragment: bool = True,
) -> None:
    """Render a compact, local-only session and delivery-status control."""

    st = _get_streamlit(streamlit_module)
    st.markdown("### 飞书助手")
    st.caption("连接当前系统并同步阶段通知；不提供用户认证，也不显示密钥。")

    loaded_config = None
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
                st.button("连接助手", disabled=True)
                return
            loaded_config = load_bot_config(configuration, workspace_root=workspace_root)
        except (TypeError, ValueError) as exc:
            st.error(f"通知控制不可用：{str(exc).splitlines()[0]}")
            return
        if store is None:
            st.error("通知控制不可用：FEISHU_NOTIFICATION_DB")
            return
        if (
            loaded_config.writeback_run_log_url is None
            or loaded_config.writeback_results_url is None
        ):
            st.error(
                "连接助手不可用：请同时配置 FEISHU_RUNLOG_URL 和 FEISHU_RESULTS_URL。"
            )
            st.button("连接助手", disabled=True)
            return
        if writeback_store is None:
            writeback_store = WritebackStore(loaded_config.writeback_db)
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

    _render_connection_notice(st, state)
    if _render_connection_task(
        st,
        state,
        store,
        snapshots=snapshots,
        writeback_store=writeback_store,
        use_fragment=use_connection_fragment,
    ):
        return

    confirming = bool(state.get(_CONFIRM_KEY, False))
    if st.button("连接助手", key="demo-notification-new-session") and not confirming:
        state[_CONFIRM_KEY] = True
        confirming = True

    if confirming:
        st.warning(
            "连接后将创建两张新的空白写回表，并同步当前网页进度；旧写回表保留。请再次确认。"
        )
        if st.button("确认连接助手", key="demo-notification-confirm-session"):
            if loaded_config is None:
                st.error("连接助手需要完整的本地飞书配置。")
                return
            started_at = datetime.now(timezone.utc)
            demo_run_id = str(uuid.uuid4())
            state[_CONNECTION_TASK_KEY] = _start_connection_task(
                app_id=loaded_config.app_id,
                app_secret=loaded_config.app_secret,
                run_log_url=loaded_config.writeback_run_log_url,
                results_url=loaded_config.writeback_results_url,
                source_run_id=source_run_id,
                demo_run_id=demo_run_id,
                started_at=started_at.isoformat(),
            )
            state[_CONFIRM_KEY] = False
            _render_connection_task(
                st,
                state,
                store,
                snapshots=snapshots,
                writeback_store=writeback_store,
                use_fragment=use_connection_fragment,
            )
            return
    if not confirming:
        _render_active_connection(st, store, state=state)

__all__ = [
    "advance_demo_one_step",
    "apply_milestone_state",
    "current_demo_progress_text",
    "initialize_feishu_page_connection",
    "render_demo_notification_control",
    "render_remote_command_listener",
    "reset_demo_playback_state",
    "sync_evolution_notifications",
    "sync_preprocessing_notification",
    "sync_pressure_notifications",
]
