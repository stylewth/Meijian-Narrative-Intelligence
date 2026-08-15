"""System-level entry and shared status presentation."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from enum import Enum
from html import escape
from typing import Any


class SystemEntry(str, Enum):
    CUSTOM = "自定义使用"
    CASE = "梅见案例展示"


FEISHU_STATUS_KEYS = (
    "FEISHU_APP_ID",
    "FEISHU_APP_SECRET",
    "FEISHU_DEMO_CHAT_ID",
    "FEISHU_BOT_OPEN_ID",
    "FEISHU_OPERATOR_OPEN_IDS",
    "FEISHU_NOTIFICATION_DB",
)
LLM_STATUS_KEYS = ("LLM_API_KEY", "LLM_MODEL")


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    online_ai_enabled: bool
    feishu_enabled: bool
    online_ai_status: str
    feishu_status: str


def _configured(configuration: Mapping[str, object], keys: tuple[str, ...]) -> bool:
    return all(
        isinstance(configuration.get(key), str)
        and str(configuration.get(key)).strip()
        for key in keys
    )


def resolve_runtime_capabilities(
    configuration: Mapping[str, object],
) -> RuntimeCapabilities:
    """Describe credential-dependent features without exposing credential values."""

    online_ai_enabled = _configured(configuration, LLM_STATUS_KEYS)
    feishu_enabled = _configured(configuration, FEISHU_STATUS_KEYS)
    return RuntimeCapabilities(
        online_ai_enabled=online_ai_enabled,
        feishu_enabled=feishu_enabled,
        online_ai_status=(
            "本地配置已就绪" if online_ai_enabled else "仅本地配置后可用"
        ),
        feishu_status=("本地配置已就绪" if feishu_enabled else "仅本地配置后触发"),
    )

_PRESENTATION_SESSION_KEYS = (
    "preprocessing_replay_step",
    "preprocessing_replay_completed_steps",
    "preprocessing_replay_running",
    "pressure_candidate_id",
    "pressure_stage",
    "pressure_blind_revealed",
    "pressure_team_confirmed",
    "evolution_manual_initialized",
    "evolution_candidate_id",
    "evolution_checkpoint_index",
    "evolution_phase",
    "evolution_playing",
    "evolution_show_finale",
    "evolution_finale_act",
    "evolution_finale_auto_reveal",
)
_PRESENTATION_SESSION_PREFIXES = ("pressure_replay_", "pressure_check_open_")


SYSTEM_GATEWAY_CSS = """
.mj-system-gateway{min-height:84vh;display:grid;align-content:center;gap:2rem;padding:4vh 3vw;color:#352F2B;background:radial-gradient(circle at 85% 16%,rgba(143,47,77,.09),transparent 27rem),linear-gradient(135deg,#F7F1E8,#ECE4D8)}
.mj-system-gateway__head{max-width:58rem}.mj-system-gateway__head span{color:#8F2F4D;font-size:.72rem;font-weight:750;letter-spacing:.18em}.mj-system-gateway__head h1{margin:.55rem 0 .7rem!important;font-family:STZhongsong,"华文中宋",serif!important;font-size:clamp(2.5rem,5vw,4.8rem)!important;font-weight:500;line-height:1.08!important}.mj-system-gateway__head p{max-width:43rem;margin:0;color:#6B625B;font-size:1rem;line-height:1.8}
.mj-system-entry-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}.mj-system-entry-card{position:relative;min-height:13rem;padding:1.5rem;border:1px solid #D5C8BA;background:rgba(255,252,247,.78);overflow:hidden}.mj-system-entry-card::after{content:"";position:absolute;right:-2rem;bottom:-4rem;width:11rem;height:11rem;border:1px solid rgba(143,47,77,.14);border-radius:50%}.mj-system-entry-card small{color:#8F2F4D;font-size:.66rem;font-weight:750;letter-spacing:.14em}.mj-system-entry-card h2{margin:.7rem 0 .55rem!important;color:#352F2B!important;font-family:STZhongsong,"华文中宋",serif!important;font-size:1.75rem!important}.mj-system-entry-card p{max-width:30rem;margin:0;color:#70665F;font-size:.82rem;line-height:1.75}.mj-system-entry-card footer{position:absolute;left:1.5rem;bottom:1.25rem;color:#365B4B;font-size:.68rem;font-weight:700;letter-spacing:.08em}
.mj-system-header{display:grid;grid-template-columns:minmax(18rem,1fr) auto;align-items:center;gap:1rem;margin:0 0 .45rem;padding:.75rem 1rem;border-bottom:1px solid #D8CCBE;background:#F7F1E8}.mj-system-header__brand span{color:#8F2F4D;font-size:.6rem;font-weight:750;letter-spacing:.17em}.mj-system-header__brand strong{display:block;margin-top:.15rem;color:#352F2B;font-family:STZhongsong,"华文中宋",serif;font-size:1.1rem}.mj-system-status{display:flex;align-items:center;gap:.55rem}.mj-system-status i{width:.42rem;height:.42rem;border-radius:50%;background:#365B4B;box-shadow:0 0 0 4px rgba(54,91,75,.1)}.mj-system-status article{padding:0 .7rem;border-left:1px solid #D8CCBE}.mj-system-status small{display:block;color:#8A7E75;font-size:.56rem}.mj-system-status b{color:#4F4741;font-size:.68rem;font-weight:650}
@media(max-width:850px){.mj-system-entry-grid{grid-template-columns:1fr}.mj-system-header{grid-template-columns:1fr}.mj-system-status{flex-wrap:wrap}}
""".strip()


def _coerce_entry(value: SystemEntry | str) -> SystemEntry:
    if isinstance(value, SystemEntry):
        return value
    return SystemEntry(value)


def feishu_connection_status(
    configuration: Mapping[str, object], store: object | None
) -> str:
    configured = _configured(configuration, FEISHU_STATUS_KEYS)
    return "已连接" if configured and store is not None else "未连接"


def build_gateway_html() -> str:
    return (
        f"<style>{SYSTEM_GATEWAY_CSS}</style>"
        '<section class="mj-system-gateway">'
        '<header class="mj-system-gateway__head"><span>MEIJIAN · NARRATIVE INTELLIGENCE</span>'
        "<h1>梅见品牌叙事智能决策系统</h1>"
        "<p>连接真实市场意见、品牌证据与多智能体验证，让每一次叙事选择都有来源、有边界、有结果。</p></header>"
        '<div class="mj-system-entry-grid">'
        '<article class="mj-system-entry-card"><small>WORKSPACE 01</small><h2>自定义使用</h2>'
        "<p>导入 XLSX、CSV 或飞书多维表格，从新数据开始运行完整决策链。</p><footer>接入新数据 →</footer></article>"
        '<article class="mj-system-entry-card"><small>WORKSPACE 02</small><h2>梅见案例展示</h2>'
        "<p>查看梅见正式案例的冻结过程、智能体互审与叙事演化结果。</p><footer>进入正式案例 →</footer></article>"
        "</div><p>飞书机器人助手 · 系统公共能力</p></section>"
    )


def build_system_header_html(
    *, entry: SystemEntry | str, data_status: str, feishu_status: str
) -> str:
    selected = _coerce_entry(entry)
    return (
        f"<style>{SYSTEM_GATEWAY_CSS}</style>"
        '<header class="mj-system-header"><div class="mj-system-header__brand">'
        "<span>梅见 · 叙事决策系统</span><strong>梅见品牌叙事智能决策系统</strong></div>"
        '<div class="mj-system-status"><i></i>'
        f"<article><small>当前入口</small><b>{escape(selected.value)}</b></article>"
        f"<article><small>数据状态</small><b>{escape(data_status)}</b></article>"
        f"<article><small>飞书机器人助手</small><b>{escape(feishu_status)}</b></article>"
        "</div></header>"
    )


def reset_system_session(state: MutableMapping[str, Any]) -> None:
    """End the current presentation session without deleting imported user data."""

    for key in tuple(state):
        if key in _PRESENTATION_SESSION_KEYS or key.startswith(
            _PRESENTATION_SESSION_PREFIXES
        ):
            state.pop(key, None)
    state["system_entry"] = None
    state["active_workspace"] = "数据预处理"
    state["completed_workspaces"] = []


def render_system_gateway(
    state: MutableMapping[str, Any], *, streamlit_module: Any
) -> SystemEntry | None:
    st = streamlit_module
    current = state.get("system_entry")
    if current is not None:
        try:
            return _coerce_entry(current)
        except ValueError:
            state["system_entry"] = None
    st.markdown(build_gateway_html(), unsafe_allow_html=True)
    columns = st.columns(2)
    for column, entry in zip(columns, SystemEntry, strict=True):
        with column:
            if st.button(
                f"进入 · {entry.value}",
                key=f"system-entry-{entry.name.lower()}",
                use_container_width=True,
                type="primary" if entry is SystemEntry.CASE else "secondary",
            ):
                state["system_entry"] = entry.value
                st.rerun()
                return entry
    return None


__all__ = [
    "FEISHU_STATUS_KEYS",
    "LLM_STATUS_KEYS",
    "RuntimeCapabilities",
    "SYSTEM_GATEWAY_CSS",
    "SystemEntry",
    "build_gateway_html",
    "build_system_header_html",
    "feishu_connection_status",
    "resolve_runtime_capabilities",
    "render_system_gateway",
    "reset_system_session",
]
