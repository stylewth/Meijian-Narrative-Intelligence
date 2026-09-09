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
.mj-system-gateway{position:relative;padding:2.25rem 0 .75rem;color:#352F2B}
.mj-system-gateway__head>span{color:#8F2F4D;font-size:.7rem;font-weight:750;letter-spacing:.16em}
.mj-system-gateway__head h1{position:relative;padding:0!important;margin:1.4rem 0 1.3rem!important;font-family:STZhongsong,"华文中宋",serif!important;font-size:clamp(2.6rem,4.2vw,4rem)!important;font-weight:500;line-height:1.3!important;letter-spacing:.025em!important}
.mj-system-gateway__head h1 .mj-gateway-title-line{display:block;white-space:nowrap}
.mj-system-gateway__head h1 > [data-testid="stHeaderActionElements"]{position:absolute;top:0;right:0}
.mj-system-gateway__head p{max-width:30rem;margin:0;color:#6B625B;font-size:1.04rem;line-height:1.95}
.mj-gateway-entry-note{margin:0 0 .65rem;color:#766D67;font-size:.86rem;line-height:1.7;max-width:30rem}
.mj-gateway-entry-note strong{display:block;margin-bottom:.2rem;color:#51433F;font-size:.92rem;font-weight:600}
.mj-gateway-art{position:relative;height:33rem;overflow:hidden;pointer-events:none}
.mj-gateway-art svg{inset:0;width:100%;height:100%;position:absolute;overflow:visible}
@keyframes mjGatewayBloomFloat{0%,100%{transform:translateY(4px) rotate(-3deg)}50%{transform:translateY(-4px) rotate(3deg)}}
@keyframes mjGatewayArcTurn{to{transform:rotate(360deg)}}
.mj-gateway-art__bloom{transform-box:view-box;transform-origin:0 0;animation:mjGatewayBloomFloat 10s ease-in-out infinite}
.mj-gateway-art__arcs{transform-box:view-box;transform-origin:338px 286px;animation:mjGatewayArcTurn 42s linear infinite}
.mj-gateway-art__halo{animation:mjGatewayHaloBreath 8s ease-in-out infinite}
.mj-gateway-art__petals{transform-box:view-box;transform-origin:338px 286px}
.mj-gateway-art__core{filter:drop-shadow(0 2px 3px rgba(91,48,55,.16))}
@keyframes mjGatewayHaloBreath{0%,100%{opacity:.58}50%{opacity:.8}}
@media(prefers-reduced-motion:reduce){.mj-gateway-art *{animation:none!important}}
.mj-gateway-art__caption{position:absolute;bottom:.45rem;left:18%;color:#8F2F4D;font-size:.7rem;letter-spacing:.24em}
.mj-gateway-flow{display:flex;align-items:center;gap:1.25rem;margin-top:1rem;padding:1rem 0;border-top:1px solid #DDD6D0;color:#756C65;font-size:.78rem;letter-spacing:.12em}
.mj-gateway-flow i{width:2.8rem;height:1px;background:#C7B1AC;position:relative}
.mj-gateway-flow i:after{content:"";position:absolute;right:0;top:-2px;width:5px;height:5px;border-top:1px solid #C7B1AC;border-right:1px solid #C7B1AC;transform:rotate(45deg)}
.mj-system-header{height:60px;display:flex;align-items:center;gap:1.2rem;white-space:nowrap}
.mj-system-header__brand{color:#8F2F4D;padding:.35rem .65rem;font-family:STZhongsong,"华文中宋",serif;font-size:1.02rem;letter-spacing:.03em}
.mj-system-status{display:flex;align-items:center;gap:.9rem;min-width:0}
.mj-system-status article{display:flex;align-items:center;gap:.35rem}
.mj-system-status small{color:#8A7E75;font-size:.65rem}
.mj-system-status b{color:#4F4741;font-size:.73rem;font-weight:500}
@media(max-width:1100px){.mj-system-header{gap:.6rem}.mj-system-status{gap:.5rem}.mj-system-status small{display:none}}
@media(max-width:700px){.mj-system-gateway{padding-top:1rem}.mj-system-gateway__head h1{font-size:2.25rem!important}.mj-gateway-art{height:22rem}.mj-gateway-flow{gap:.7rem;font-size:.7rem}.mj-gateway-flow i{width:1rem}.mj-system-header{height:auto;min-height:56px;flex-wrap:wrap;gap:.4rem}.mj-system-status{flex-wrap:wrap}}
""".strip()


def _coerce_entry(value: SystemEntry | str) -> SystemEntry:
    if isinstance(value, SystemEntry):
        return value
    return SystemEntry(value)


def feishu_connection_status(
    configuration: Mapping[str, object],
    store: object | None,
    *,
    page_connected: bool = False,
) -> str:
    configured = _configured(configuration, FEISHU_STATUS_KEYS)
    return "已连接" if configured and store is not None and page_connected else "未连接"


def build_gateway_html() -> str:
    return (
        f"<style>{SYSTEM_GATEWAY_CSS}</style>"
        '<section class="mj-system-gateway">'
        '<header class="mj-system-gateway__head"><span>MEIJIAN · NARRATIVE INTELLIGENCE</span>'
        '<h1 aria-label="梅见品牌叙事智能决策系统"><span class="mj-gateway-title-line">梅见品牌叙事</span><span class="mj-gateway-title-line">智能决策系统</span></h1>'
        '<p>连接真实市场意见、品牌证据与多智能体验证，<br>让每一次叙事选择都有来源、有边界、有结果。</p>'
        '</header></section>'
    )


def _build_gateway_art_html() -> str:
    return (
        '<div class="mj-gateway-art" aria-hidden="true">'
        '<svg viewBox="0 0 600 560" xmlns="http://www.w3.org/2000/svg">'
        '<defs><radialGradient id="mj-window"><stop stop-color="#FCFAF7"/>'
        '<stop offset=".72" stop-color="#EAE0D9"/><stop offset="1" stop-color="#D4C0BD"/></radialGradient>'
        '<linearGradient id="mj-petal" x2="1" y2="1"><stop stop-color="#BA7C8B" stop-opacity=".8"/>'
        '<stop offset="1" stop-color="#722C47" stop-opacity=".95"/></linearGradient>'
        '<radialGradient id="mj-halo"><stop stop-color="#FFF8EF" stop-opacity=".9"/><stop offset="1" stop-color="#C9A86A" stop-opacity="0"/></radialGradient>'
        '<linearGradient id="mj-core" x2="0" y2="1"><stop stop-color="#F7EAD7"/><stop offset="1" stop-color="#C49B71"/></linearGradient>'
        '<clipPath id="mj-window-clip"><circle cx="338" cy="286" r="238"/></clipPath></defs>'
        '<circle cx="338" cy="286" r="238" fill="url(#mj-window)"/>'
        '<circle class="mj-gateway-art__halo" cx="345" cy="260" r="130" fill="url(#mj-halo)"/>'
        '<g clip-path="url(#mj-window-clip)" fill="none" stroke="#BCA5A0" stroke-width="1">'
        '<g class="mj-gateway-art__arcs"><circle cx="400" cy="360" r="190"/><circle cx="400" cy="360" r="160"/><circle cx="400" cy="360" r="130"/></g></g>'
        '<g class="mj-gateway-art__petals" transform="translate(345 260)"><g class="mj-gateway-art__bloom">'
        '<path d="M0,-18 C-42,-58 -38,-125 0,-148 C38,-125 42,-58 0,-18Z" fill="url(#mj-petal)" stroke="#8F2F4D" stroke-width="1.2"/>'
        '<path d="M0,-18 C-42,-58 -38,-125 0,-148 C38,-125 42,-58 0,-18Z" transform="rotate(72)" fill="url(#mj-petal)" stroke="#8F2F4D" stroke-width="1.2"/>'
        '<path d="M0,-18 C-42,-58 -38,-125 0,-148 C38,-125 42,-58 0,-18Z" transform="rotate(144)" fill="url(#mj-petal)" stroke="#8F2F4D" stroke-width="1.2"/>'
        '<path d="M0,-18 C-42,-58 -38,-125 0,-148 C38,-125 42,-58 0,-18Z" transform="rotate(216)" fill="url(#mj-petal)" stroke="#8F2F4D" stroke-width="1.2"/>'
        '<path d="M0,-18 C-42,-58 -38,-125 0,-148 C38,-125 42,-58 0,-18Z" transform="rotate(288)" fill="url(#mj-petal)" stroke="#8F2F4D" stroke-width="1.2"/>'
        '<circle class="mj-gateway-art__core" r="32" fill="url(#mj-core)" stroke="#A97879" stroke-width="1.2"/><circle r="12" fill="#BD9B70"/><circle r="4" fill="#FFF8EF"/>'
        '<path d="M-122,92 Q10,72 132,-92" fill="none" stroke="#F6EFE7" stroke-width="2" opacity=".72"/></g></g>'
        '<path d="M92,452 Q300,476 508,452" fill="none" stroke="#C5A66D" stroke-width="2"/>'
        '</svg><span class="mj-gateway-art__caption">从真实意见，看见品牌的下一步</span></div>'
    )

def build_system_header_html(
    *, entry: SystemEntry | str, data_status: str, feishu_status: str
) -> str:
    selected = _coerce_entry(entry)
    return (
        f"<style>{SYSTEM_GATEWAY_CSS}</style>"
        '<header class="mj-system-header"><div class="mj-system-header__brand">'
        '梅见 · 叙事决策</div><div class="mj-system-status">'
        f"<article><b>{escape(selected.value)}</b></article>"
        f"<article><small>数据</small><b>{escape(data_status)}</b></article>"
        f"<article><small>飞书</small><b>{escape(feishu_status)}</b></article>"
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
    columns = st.columns([50, 50])
    with columns[0]:
        st.markdown(build_gateway_html(), unsafe_allow_html=True)
        for entry in (SystemEntry.CASE, SystemEntry.CUSTOM):
            description = (
                "查看正式案例的冻结过程、智能体互审与叙事演化结果。"
                if entry is SystemEntry.CASE
                else "导入 XLSX、CSV 或飞书多维表格，运行完整决策链。"
            )
            st.markdown(
                f'<p class="mj-gateway-entry-note">{description}</p>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"进入 · {entry.value}",
                key=f"system-entry-{entry.name.lower()}",
                use_container_width=True,
                type="primary" if entry is SystemEntry.CASE else "secondary",
            ):
                state["system_entry"] = entry.value
                st.rerun()
                return entry
    with columns[1]:
        st.markdown(_build_gateway_art_html(), unsafe_allow_html=True)
    st.markdown(
        '<div class="mj-gateway-flow"><span>数据整理</span><i></i>'
        '<span>叙事验证</span><i></i><span>演化决策</span></div>',
        unsafe_allow_html=True,
    )
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
