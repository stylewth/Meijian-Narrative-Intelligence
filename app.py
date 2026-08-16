from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import dotenv_values

from src.integrations.feishu.notification_store import NotificationStore
from src.services.demo_notifications import build_demo_notification_snapshots
from src.services.evolution_presentation import load_official_evolution_run
from src.services.preprocessing_demo_presentation import load_preprocessing_demo
from src.services.pressure_test_presentation import load_pressure_test_run
from src.state import initialize_state
from src.ui import (
    DEMO_MODE,
    REAL_MODE,
    SystemEntry,
    render_preprocessing_workspace,
    render_system_gateway,
)
from src.ui.demo_notification_control import (
    render_demo_notification_control,
    sync_evolution_notifications,
    sync_preprocessing_notification,
    sync_pressure_notifications,
)
from src.ui.custom_decision_workspace import render_custom_decision_workspace
from src.ui.pressure_test_workspace import render_pressure_test_workspace
from src.ui.realtime_decision_dashboard import render_realtime_decision_dashboard
from src.ui.system_gateway import (
    build_system_header_html,
    feishu_connection_status,
    reset_system_session,
    resolve_runtime_capabilities,
)
from src.ui.ui_theme import build_theme_css
from src.ui.workspace_shell import (
    Workspace,
    complete_workspace,
    render_workspace_navigator,
)
from tools.run_feishu_bot import REQUIRED_CONFIG_KEYS, load_bot_config


ROOT = Path(__file__).resolve().parent
DEMO_NOTIFICATION_SOURCE_RUN_ID = "official-20260814-three-opportunity-evolution-001"
VALIDATION_ROOT = (
    ROOT
    / "data"
    / "competition"
    / "five_candidate_validation"
    / "official-20260813-five-candidate-validation-002"
)
EVOLUTION_ROOT = (
    ROOT
    / "data"
    / "competition"
    / "evolution_runs"
    / DEMO_NOTIFICATION_SOURCE_RUN_ID
)
RUNTIME_CONFIGURATION_KEYS = tuple(
    dict.fromkeys(
        (
            *REQUIRED_CONFIG_KEYS,
            "STREAMLIT_PUBLIC_URL",
            "FEISHU_WIKI_URL",
            "LLM_API_KEY",
            "LLM_MODEL",
        )
    )
)
STREAMLIT_SECRETS_AVAILABLE = any(
    path.is_file()
    for path in (
        ROOT / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    )
)


st.set_page_config(
    page_title="梅见品牌叙事智能决策系统",
    page_icon="梅",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _secret_or_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if STREAMLIT_SECRETS_AVAILABLE:
        secret = st.secrets.get(name)
        if secret:
            return str(secret)
    value = dotenv_values(ROOT / ".env").get(name)
    return str(value) if value else None


def _runtime_configuration() -> dict[str, str]:
    return {key: _secret_or_env(key) or "" for key in RUNTIME_CONFIGURATION_KEYS}


@st.cache_resource
def _demo_notification_store() -> NotificationStore | None:
    values = _runtime_configuration()
    try:
        config = load_bot_config(values, workspace_root=ROOT)
    except (TypeError, ValueError):
        return None
    return NotificationStore(config.notification_db)


@st.cache_data
def _demo_notification_snapshots() -> tuple[Any, ...]:
    return build_demo_notification_snapshots(
        load_preprocessing_demo(
            ROOT / "data" / "competition" / "screened_v2",
            VALIDATION_ROOT,
        ),
        load_pressure_test_run(
            VALIDATION_ROOT,
            EVOLUTION_ROOT / "blind" / "selection_confirmation.json",
        ),
        load_official_evolution_run(EVOLUTION_ROOT),
    )


initialize_state(st.session_state)
st.markdown(f"<style>{build_theme_css()}</style>", unsafe_allow_html=True)

notification_store = _demo_notification_store()
entry = render_system_gateway(st.session_state, streamlit_module=st)
if entry is None:
    st.stop()

configuration = _runtime_configuration()
capabilities = resolve_runtime_capabilities(configuration)
data_status = (
    "正式案例已载入"
    if entry is SystemEntry.CASE
    else (
        f"已导入 {len(st.session_state.get('comments') or [])} 条"
        if st.session_state.get("comments")
        else "等待数据导入"
    )
)
st.markdown(
    build_system_header_html(
        entry=entry,
        data_status=data_status,
        feishu_status=feishu_connection_status(configuration, notification_store),
    ),
    unsafe_allow_html=True,
)
if entry is SystemEntry.CUSTOM and not capabilities.online_ai_enabled:
    st.info("在线 AI 仅在本地配置后可用；公开站不会使用团队模型密钥。")

with st.expander("飞书机器人助手", expanded=False):
    render_demo_notification_control(
        st.session_state,
        store=notification_store,
        source_run_id=DEMO_NOTIFICATION_SOURCE_RUN_ID,
        configuration=configuration,
        workspace_root=ROOT,
    )

if st.button("切换入口", key="system-entry-reset"):
    reset_system_session(st.session_state)
    st.rerun()

render_workspace_navigator(st.session_state)
active_workspace = Workspace(st.session_state["active_workspace"])

if active_workspace is Workspace.PREPROCESSING:
    render_preprocessing_workspace(
        split_manifest_path=ROOT / "data" / "competition" / "screened_v2" / "split_manifest.json",
        prepared_root=ROOT / "data" / "prepared_corpora",
        mode=DEMO_MODE if entry is SystemEntry.CASE else REAL_MODE,
    )
    sync_preprocessing_notification(
        st.session_state,
        notification_store,
        _demo_notification_snapshots(),
    )
    st.stop()

if active_workspace is Workspace.STRESS_TEST:
    render_pressure_test_workspace(
        VALIDATION_ROOT,
        EVOLUTION_ROOT / "blind" / "selection_confirmation.json",
    )
    sync_pressure_notifications(
        st.session_state,
        notification_store,
        _demo_notification_snapshots(),
    )
    st.stop()

if active_workspace is Workspace.REALTIME_DECISION:
    render_realtime_decision_dashboard(EVOLUTION_ROOT)
    sync_evolution_notifications(
        st.session_state,
        notification_store,
        _demo_notification_snapshots(),
    )
    completed = [Workspace(value) for value in st.session_state.get("completed_workspaces", [])]
    if Workspace.REALTIME_DECISION not in completed:
        if st.button("完成回放 · 解锁自由决策实验", key="realtime_complete_replay"):
            complete_workspace(st.session_state, Workspace.REALTIME_DECISION)
            st.rerun()
    st.stop()

if active_workspace is Workspace.FREE_DECISION:
    render_custom_decision_workspace(
        prepared_root=ROOT / "data" / "prepared_corpora",
        entry=entry,
        online_enabled=capabilities.online_ai_enabled,
        dotenv_path=ROOT / ".env",
    )
    st.stop()
