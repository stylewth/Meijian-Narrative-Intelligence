"""Shared navigation shell for the three ordered workspaces."""

from __future__ import annotations

from collections.abc import MutableMapping
from enum import Enum
from typing import Any

from src.state import initialize_state


class Workspace(str, Enum):
    """The user-facing workspaces, in their unlock order."""

    PREPROCESSING = "数据预处理"
    STRESS_TEST = "叙事压力测试"
    REALTIME_DECISION = "实时决策看板"

    # Descriptive aliases keep call sites readable without creating new values.
    PRESSURE_TEST = STRESS_TEST
    NARRATIVE_STRESS_TEST = STRESS_TEST
    REALTIME_DECISION_DASHBOARD = REALTIME_DECISION


_WORKSPACES = tuple(Workspace)
_LOCKED_MESSAGE = "完成上一阶段后解锁"
_REALTIME_SESSION_KEYS = (
    "evolution_manual_initialized",
    "evolution_candidate_id",
    "evolution_checkpoint_index",
    "evolution_phase",
    "evolution_playing",
    "evolution_show_finale",
    "evolution_finale_act",
    "evolution_finale_auto_reveal",
)


def _as_workspace(value: Workspace | str) -> Workspace:
    if isinstance(value, Workspace):
        return value
    try:
        return Workspace(value)
    except ValueError as exc:
        raise ValueError(f"未知工作区：{value}") from exc


def _completed_workspaces(state: MutableMapping[str, Any]) -> tuple[Workspace, ...]:
    initialize_state(state)
    raw_values = state["completed_workspaces"]
    if not isinstance(raw_values, (list, tuple)):
        raise TypeError("completed_workspaces 必须是字符串列表或元组")

    completed = tuple(_as_workspace(value) for value in raw_values)
    if len(set(completed)) != len(completed):
        raise ValueError("completed_workspaces 不得包含重复工作区")
    expected_prefix = _WORKSPACES[: len(completed)]
    if completed != expected_prefix:
        raise ValueError("completed_workspaces 必须按工作区顺序记录")
    return completed


def allowed_workspaces(state: MutableMapping[str, Any]) -> tuple[Workspace, ...]:
    """Return all workspaces currently reachable by the user."""

    completed = _completed_workspaces(state)
    return _WORKSPACES[: min(len(completed) + 1, len(_WORKSPACES))]


def complete_workspace(
    state: MutableMapping[str, Any], workspace: Workspace | str
) -> tuple[Workspace, ...]:
    """Record one completed workspace and unlock exactly the next one.

    Completion is append-only and idempotent. A later workspace cannot be
    completed before its predecessor, and this function never changes the
    active workspace.
    """

    target = _as_workspace(workspace)
    completed = _completed_workspaces(state)
    if target in completed:
        return completed
    if target not in allowed_workspaces(state):
        raise ValueError(f"工作区必须按顺序完成：{target.value}")

    state["completed_workspaces"] = [item.value for item in (*completed, target)]
    return (*completed, target)


def activate_workspace(
    state: MutableMapping[str, Any], workspace: Workspace | str
) -> Workspace:
    """Activate one workspace and start a fresh realtime presentation on re-entry."""

    initialize_state(state)
    target = _as_workspace(workspace)
    current = _as_workspace(state["active_workspace"])
    if target is Workspace.REALTIME_DECISION and current is not target:
        for key in _REALTIME_SESSION_KEYS:
            state.pop(key, None)
    state["active_workspace"] = target.value
    return target


def _get_streamlit() -> Any:
    import streamlit as st

    return st


def render_workspace_navigator(state: MutableMapping[str, Any]) -> None:
    """Render navigation controls and apply only valid navigation clicks."""

    st = _get_streamlit()
    initialize_state(state)
    completed = _completed_workspaces(state)
    completed_values = {item.value for item in completed}
    active = _as_workspace(state["active_workspace"])
    allowed = set(allowed_workspaces(state))

    st.markdown(
        "<div class='mj-workspace-nav'><span>系统模块</span>"
        "<strong>品牌叙事决策模块</strong></div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(len(_WORKSPACES))
    for column, workspace in zip(columns, _WORKSPACES, strict=True):
        is_completed = workspace.value in completed_values
        is_current = workspace is active
        is_locked = workspace not in allowed

        label = f"{workspace.value} · 未解锁" if is_locked else workspace.value

        with column:
            if not st.button(
                label,
                key=f"workspace-nav-{workspace.name.lower()}",
                use_container_width=True,
            ):
                continue
            if is_locked:
                st.info(_LOCKED_MESSAGE)
                continue
            activate_workspace(state, workspace)


__all__ = [
    "Workspace",
    "activate_workspace",
    "allowed_workspaces",
    "complete_workspace",
    "render_workspace_navigator",
]
