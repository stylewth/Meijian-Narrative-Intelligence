"""只读的五候选验证流水线剧场。"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
import re
import time
from typing import Any, Callable

import streamlit as st

from src.services.candidate_ranking import calculate_weighted_score
from src.services.pressure_test_presentation import (
    AgentFieldDiffView,
    AgentMessageView,
    EvidenceExcerptView,
    PressureRunView,
    load_pressure_test_run,
)
from src.ui.scroll_continuity import render_scroll_continuity
from src.ui.ui_theme import build_theme_css
from src.ui.workspace_shell import Workspace, activate_workspace, complete_workspace


PRESSURE_STAGES = (
    "AI 原始五项",
    "基础压力检查",
    "双 Agent 互审",
    "统一复评与 HOLDOUT",
    "真人盲评 5→3",
)
HOLDOUT_NOTE = "验证稳定性，不改写候选文案和评分"
FINAL_METHOD = "AI 推荐 → 真人盲评验证 → 团队确认"
_RAW_EVIDENCE_ID = re.compile(r"(?:MJ-RAW|XHS-SOC|DY-SOC|TB-\d{8}|JD-\d{8})-\d+")


@dataclass(frozen=True, slots=True)
class CandidateCard:
    candidate_id: str
    title: str
    display_title: str
    original_score: float
    ai_rank: int
    status: str
    selected: bool
    overall_assessment: str


@dataclass(frozen=True, slots=True)
class DialogueMessage:
    round_index: int
    agent: str
    kind: str
    headline: str
    summary: str
    body: str
    highlights: tuple[str, ...]
    field_diffs: tuple[AgentFieldDiffView, ...]
    source_path: Path


@dataclass(frozen=True, slots=True)
class HoldoutDistribution:
    candidate_id: str
    support_count: int
    challenge_count: int
    neutral_count: int
    blocking_count: int

    @property
    def total(self) -> int:
        return self.support_count + self.challenge_count + self.neutral_count


@dataclass(frozen=True, slots=True)
class TerminalSummary:
    method: str
    selected: tuple[CandidateCard, ...]
    not_selected: tuple[CandidateCard, ...]


def build_candidate_cards(run: PressureRunView) -> tuple[CandidateCard, ...]:
    """Project all five frozen candidates without dropping rejected history."""

    selected_ids = set(run.selection.selected_candidate_ids)
    evaluations = {item.candidate_id: item for item in run.evaluation.evaluations}
    ranks = {candidate_id: index for index, candidate_id in enumerate(run.evaluation.ranked_candidate_ids, 1)}
    cards: list[CandidateCard] = []
    for candidate in run.candidates:
        evaluation = evaluations[candidate.candidate_id]
        cards.append(
            CandidateCard(
                candidate_id=candidate.candidate_id,
                title=candidate.title,
                display_title=candidate.candidate_id.rsplit("-", 1)[-1],
                original_score=float(calculate_weighted_score(evaluation.scores)),
                ai_rank=ranks[candidate.candidate_id],
                status="入选" if candidate.candidate_id in selected_ids else "未入选",
                selected=candidate.candidate_id in selected_ids,
                overall_assessment=candidate.overall_assessment,
            )
        )
    return tuple(cards)


def dialogue_for_candidate(
    run: PressureRunView, candidate_id: str
) -> tuple[DialogueMessage, ...]:
    """Return a deterministic review/decision replay from frozen round files."""

    candidate = next(
        (item for item in run.candidates if item.candidate_id == candidate_id),
        None,
    )
    if candidate is None:
        raise ValueError(f"未知候选：{candidate_id}")

    order = {"review": 0, "decision": 1, "revision": 2}
    messages: list[DialogueMessage] = []
    for message in sorted(
        candidate.dialogue,
        key=lambda item: (item.round_index, order.get(item.kind, 99)),
    ):
        messages.append(_dialogue_message(message))
    return tuple(messages)


def advance_dialogue_replay(
    state: MutableMapping[str, Any], *, message_count: int
) -> bool:
    """Advance one frozen message after an explicit user action."""

    if message_count < 1:
        raise ValueError("message_count 必须为正整数")
    if state.get("completed", False):
        return False
    index = int(state.get("index", 0) or 0)
    if not 0 <= index < message_count:
        raise ValueError("dialogue replay index 不在消息范围内")
    next_index = min(index + 1, message_count - 1)
    state["index"] = next_index
    if next_index == message_count - 1:
        state["completed"] = True
    return next_index != index


def holdout_distribution(run: PressureRunView) -> tuple[HoldoutDistribution, ...]:
    """Project the 5 × 60 HOLDOUT counts, including blocking findings."""

    return tuple(
        HoldoutDistribution(
            candidate_id=candidate.candidate_id,
            support_count=candidate.support_count,
            challenge_count=candidate.challenge_count,
            neutral_count=candidate.neutral_count,
            blocking_count=len(candidate.blocking_findings),
        )
        for candidate in run.holdout.candidates
    )


def terminal_summary(run: PressureRunView) -> TerminalSummary:
    """Return the fixed 5→3 conclusion while retaining both outcome groups."""

    cards = build_candidate_cards(run)
    selected = tuple(card for card in cards if card.selected)
    not_selected = tuple(card for card in cards if not card.selected)
    return TerminalSummary(FINAL_METHOD, selected, not_selected)


def active_pressure_section(stage: str) -> str:
    """Map a stage label to the only content section allowed on stage."""

    sections = dict(
        zip(
            PRESSURE_STAGES,
            ("original", "checks", "dialogue", "holdout", "terminal"),
            strict=True,
        )
    )
    try:
        return sections[stage]
    except KeyError as exc:
        raise ValueError(f"未知压力测试舞台：{stage}") from exc


def move_pressure_stage(stage: str, step: int) -> str:
    """Move one stage backward or forward without allowing direct jumps."""

    if stage not in PRESSURE_STAGES:
        raise ValueError(f"未知压力测试舞台：{stage}")
    if step not in {-1, 1}:
        raise ValueError("压力测试阶段只能前进或后退一步")
    index = PRESSURE_STAGES.index(stage)
    return PRESSURE_STAGES[max(0, min(len(PRESSURE_STAGES) - 1, index + step))]


def select_pressure_candidate(
    state: MutableMapping[str, Any],
    candidate_id: str,
    candidate_ids: tuple[str, ...],
) -> bool:
    """Select one known candidate and restart only its frozen dialogue."""

    if candidate_id not in candidate_ids:
        raise ValueError(f"未知候选：{candidate_id}")
    if state.get("pressure_candidate_id") == candidate_id:
        return False
    state["pressure_candidate_id"] = candidate_id
    state[f"pressure_replay_{candidate_id}"] = {
        "index": 0,
        "running": True,
        "completed": False,
    }
    return True


def render_pressure_test_workspace(
    validation_root: str | Path,
    selection_path: str | Path,
    *,
    on_milestone: Callable[[], None] | None = None,
) -> None:
    """Render a frozen pressure-test run; interaction changes session state only."""

    st = _get_streamlit()
    run = load_pressure_test_run(validation_root, selection_path)
    cards = build_candidate_cards(run)
    st.markdown(f"<style>{build_theme_css()}{_pressure_test_css()}</style>", unsafe_allow_html=True)
    st.markdown(
        "<section class='pressure-hero'><span>MEIJIAN · VALIDATION THEATRE</span>"
        "<h1>叙事压力测试</h1><p>五条候选同台接受证据、反证、竞品、产品与鲁棒性检查。"
        "双 Agent 对话、HOLDOUT 与真人盲评均回放冻结结果。</p>"
        "<small>官方冻结案例 · 只读回放</small></section>",
        unsafe_allow_html=True,
    )
    _render_pressure_flow(run, cards, on_milestone)


@st.fragment
def _render_pressure_flow(
    run: Any,
    cards: Any,
    on_milestone: Callable[[], None] | None = None,
) -> None:
    """候选/阶段/回放交互体：局部重跑，hero 与页面外壳保持不动。"""

    st = _get_streamlit()
    candidate_id = _selected_candidate_id(st, cards)
    left, right = st.columns((1.1, 4))
    with left:
        candidate_id = _render_candidate_cards(st, cards, candidate_id)
    with right:
        current_stage = _render_stage_navigation(st, on_milestone)
        candidate = next(item for item in run.candidates if item.candidate_id == candidate_id)
        card = next(item for item in cards if item.candidate_id == candidate_id)
        section = active_pressure_section(current_stage)
        if section in {"original", "checks", "dialogue"}:
            _render_current_candidate(st, card)
        if section == "original":
            _render_original_stage(st, card, cards)
        elif section == "checks":
            _render_checks(st, candidate, run.evidence_catalog)
        elif section == "dialogue":
            _render_dialogue(st, run, candidate_id)
        elif section == "holdout":
            _render_holdout(st, run, cards)
        elif section == "terminal":
            _render_terminal(st, run, on_milestone=on_milestone)
    render_scroll_continuity(f"pressure:{current_stage}")


def _dialogue_message(message: AgentMessageView) -> DialogueMessage:
    agent = "审查 Agent" if message.agent == "Luna" else "决策 Agent"
    return DialogueMessage(
        round_index=message.round_index,
        agent=agent,
        kind=message.kind,
        headline=message.headline,
        summary=message.summary,
        body=message.body,
        highlights=message.highlights,
        field_diffs=message.field_diffs,
        source_path=message.source_path,
    )


def _render_current_candidate(st: Any, card: CandidateCard) -> None:
    st.markdown(build_current_candidate_html(card), unsafe_allow_html=True)
    with st.expander("审计详情 · 英文原始评语", expanded=False):
        st.write(card.overall_assessment)


def build_current_candidate_html(card: CandidateCard) -> str:
    """Build the compact Chinese stage header without exposing model prose."""

    return (
        "<section class='pressure-current'><span>当前候选</span>"
        f"<strong class='pressure-current-title'>{_escape_html(card.display_title)}</strong>"
        f"<p>当前舞台仅展示冻结结果；详细依据随验证阶段展开。</p>"
        f"<div><b>{card.original_score:.1f}</b><small>AI 原始分</small>"
        f"<b>#{card.ai_rank}</b><small>AI 排名</small></div></section>"
    )


def _render_original_stage(st: Any, card: CandidateCard, cards: tuple[CandidateCard, ...]) -> None:
    st.markdown(
        "<section class='pressure-original-stage'><span>阶段 1 · AI 原始五项</span>"
        f"<h2>{_escape_html(card.title)}</h2>"
        f"<p>当前查看：{_escape_html(card.display_title)}。原始得分 {card.original_score:.1f}，"
        f"AI 排名第 {card.ai_rank}。此处保持盲评前版本，不提前展示修订结论。</p>"
        f"{build_original_score_dotplot_html(cards, card.candidate_id)}</section>",
        unsafe_allow_html=True,
    )


def _selected_candidate_id(st: Any, cards: tuple[CandidateCard, ...]) -> str:
    ids = tuple(card.candidate_id for card in cards)
    current = st.session_state.get("pressure_candidate_id")
    if current not in ids:
        st.session_state["pressure_candidate_id"] = ids[0]
        current = ids[0]
    return current


def _render_candidate_cards(st: Any, cards: tuple[CandidateCard, ...], current_id: str) -> str:
    ids = tuple(card.candidate_id for card in cards)
    st.markdown("<nav class='pressure-candidate-nav' aria-label='候选导航'><span>五条候选</span></nav>", unsafe_allow_html=True)
    reveal = bool(st.session_state.get("pressure_blind_revealed", False))
    for card in cards:
        if st.button(
            card.display_title,
            key=f"pressure_candidate_button_{card.candidate_id}",
            use_container_width=True,
        ):
            select_pressure_candidate(st.session_state, card.candidate_id, ids)
            current_id = card.candidate_id
        st.markdown(build_candidate_meta_html((card,), current_id, reveal=reveal), unsafe_allow_html=True)
    return current_id


def build_candidate_meta_html(
    cards: tuple[CandidateCard, ...], current_id: str, *, reveal: bool
) -> str:
    items = []
    for card in cards:
        selected_class = " is-selected" if card.candidate_id == current_id else ""
        outcome_class = (
            " is-finalist"
            if reveal and card.selected
            else " is-rejected"
            if reveal
            else ""
        )
        status = card.status if reveal else "待盲评"
        items.append(
            f"<article class='pressure-candidate-meta{selected_class}{outcome_class}'>"
            f"<span>AI 初始 #{card.ai_rank}</span><span>{_escape_html(status)}</span></article>"
        )
    return '<section class="pressure-candidate-meta-grid">' + "".join(items) + "</section>"


def build_original_score_dotplot_html(
    cards: tuple[CandidateCard, ...], current_id: str
) -> str:
    """Render all frozen original scores on one shared 0–100 axis."""

    dots = "".join(
        "<li class='pressure-score-dot"
        + (" is-selected" if item.candidate_id == current_id else "")
        + f"' style='--score:{item.original_score:.2f}'>"
        + f"<b>{_escape_html(item.display_title)}</b><i></i><span>{item.original_score:.1f}</span></li>"
        for item in cards
    )
    return (
        "<section class='pressure-score-dotplot'><header><span>AI 原始五项</span>"
        "<small>同一 0–100 刻度；高亮为当前候选</small></header>"
        "<div class='pressure-score-axis'><i>0</i><i>50</i><i>100</i></div>"
        f"<ol>{dots}</ol></section>"
    )


def _render_stage_navigation(
    st: Any,
    on_milestone: Callable[[], None] | None = None,
) -> str:
    current_stage = st.session_state.get("pressure_stage", PRESSURE_STAGES[0])
    if current_stage not in PRESSURE_STAGES:
        current_stage = PRESSURE_STAGES[0]
        st.session_state["pressure_stage"] = current_stage
    st.markdown(
        _build_stage_progress_html(current_stage),
        unsafe_allow_html=True,
    )
    index = PRESSURE_STAGES.index(current_stage)
    controls = st.columns([1, 3, 1])
    with controls[0]:
        if st.button(
            "上一阶段",
            key="pressure_previous_stage",
            disabled=index == 0,
            use_container_width=True,
        ):
            current_stage = move_pressure_stage(current_stage, -1)
            st.session_state["pressure_stage"] = current_stage
            if on_milestone is not None:
                on_milestone()
            st.rerun()
    with controls[1]:
        st.markdown(
            f"<div class='pressure-stage-now'>阶段 {PRESSURE_STAGES.index(current_stage) + 1} / {len(PRESSURE_STAGES)} · {_escape_html(current_stage)}</div>",
            unsafe_allow_html=True,
        )
    with controls[2]:
        if st.button(
            "下一阶段",
            key="pressure_next_stage",
            disabled=index == len(PRESSURE_STAGES) - 1,
            use_container_width=True,
        ):
            current_stage = move_pressure_stage(current_stage, 1)
            st.session_state["pressure_stage"] = current_stage
            if on_milestone is not None:
                on_milestone()
            st.rerun()
    return current_stage


def _build_stage_progress_html(current_stage: str) -> str:
    current_index = PRESSURE_STAGES.index(current_stage)
    nodes = "".join(
        "<span class='pressure-stage-node"
        + (" is-complete" if index < current_index else "")
        + (" is-current" if index == current_index else "")
        + f"'><i>{index + 1}</i><b>{_escape_html(stage)}</b></span>"
        for index, stage in enumerate(PRESSURE_STAGES)
    )
    return f"<div class='pressure-stage-bar'><small>五阶段验证进度</small><div>{nodes}</div></div>"


def build_dialogue_message_html(message: DialogueMessage, *, is_current: bool = False) -> str:
    """Build one compact chat message from the structured frozen projection."""

    avatar_class = "decision-avatar" if message.agent == "决策 Agent" else "review-avatar"
    highlights = "".join(
        f"<li>{_escape_html(_display_evidence_text(item))}</li>" for item in message.highlights
    )
    highlights_markup = f"<ul>{highlights}</ul>" if highlights else ""
    audit_markup = _build_pressure_audit_detail(message.body, *message.highlights)
    detail = (
        "<details><summary>本轮详细理由与操作</summary>"
        f"<p class='dialogue-detail-copy'>{_escape_html(_display_evidence_text(message.body))}</p>"
        f"{highlights_markup}{audit_markup}</details>"
    )
    return (
        f'<article class="dialogue-bubble {avatar_class}-bubble" data-current="{"true" if is_current else "false"}">'
        f"<span class='agent-avatar {avatar_class}'></span>"
        "<div class='dialogue-bubble__body'>"
        f"<header><strong>{_escape_html(message.agent)}</strong>"
        f"<span>第 {message.round_index} 轮</span></header>"
        f"<h3>{_escape_html(_display_evidence_text(message.headline))}</h3>"
        f"<p>{_escape_html(_display_evidence_text(message.summary))}</p>{detail}"
        "</div></article>"
    )


def build_dialogue_stage_html(messages: tuple[DialogueMessage, ...]) -> str:
    """Build a self-playing chat frame: bubbles are revealed client-side.

    回放节奏由 iframe 内的原生 JS 驱动，服务端在回放期间零参与——
    不再占用 Streamlit 运行窗口，避免交互被遮罩/丢弃。
    """

    if not messages:
        raise ValueError("对话舞台至少需要一条消息")
    timeline = "".join(
        build_dialogue_message_html(message, is_current=False).replace(
            '<article class="dialogue-bubble', f'<article data-revision-index="{index}" class="dialogue-bubble'
        )
        for index, message in enumerate(messages)
    )
    revisions = "".join(
        build_dialogue_revision_html(message, index) for index, message in enumerate(messages)
    )
    total = len(messages)
    interval_ms = 4200
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        + _dialogue_frame_css()
        + f"""
.stage-controls{{position:absolute;left:0;right:0;bottom:0;display:flex;align-items:center;gap:.45rem;padding:.3rem .8rem .34rem;background:linear-gradient(180deg,rgba(239,233,224,0),#EFE9E0 45%)}}
.stage-controls button{{height:1.5rem;padding:0 .7rem;border:1px solid #C9B8A6;border-radius:999px;background:rgba(255,253,249,.85);color:#6B625B;font-family:inherit;font-size:.64rem;letter-spacing:.08em;cursor:pointer}}
.stage-controls button:hover{{border-color:#8F2F4D;color:#8F2F4D}}
.stage-controls .stage-count{{margin-left:auto;color:#8A8078;font-size:.6rem;letter-spacing:.1em;font-variant-numeric:tabular-nums}}
</style></head><body>
<section class='dialogue-layout'>
  <section class='dialogue-stage dialogue-review-axis' id='dialogue-stage'>{timeline}</section>
  <aside class='dialogue-revision-panel' id='mj-d-revisions'>{revisions}</aside>
</section>
<div class='stage-controls'>
  <button id='mj-d-toggle' type='button'>暂停</button>
  <button id='mj-d-replay' type='button'>重播</button>
  <span class='stage-count' id='mj-d-count'></span>
</div>
<script>
(function () {{
  var INTERVAL = {interval_ms};
  var stage = document.getElementById('dialogue-stage');
  var bubbles = Array.prototype.slice.call(document.querySelectorAll('.dialogue-bubble'));
  var countEl = document.getElementById('mj-d-count');
  var toggleBtn = document.getElementById('mj-d-toggle');
  var replayBtn = document.getElementById('mj-d-replay');
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var idx = 0, playing = true, timer = null, chip = null;

  function scrollCurrent() {{
    var current = bubbles[idx - 1];
    if (!current) return;
    var top = current.offsetTop - (stage.clientHeight - current.offsetHeight) / 2;
    stage.scrollTo({{ top: Math.max(0, top), behavior: 'smooth' }});
  }}
  function updateCount() {{ countEl.textContent = idx + ' / ' + bubbles.length; }}
  function showChip(next) {{
    clearChip();
    var agentName = (next.querySelector('header strong') || {{}}).textContent || 'Agent';
    chip = document.createElement('article');
    chip.className = 'dialogue-thinking ' + (next.classList.contains('review-avatar-bubble') ? 'review-avatar-bubble' : 'decision-avatar-bubble');
    chip.innerHTML = '<span class="agent-avatar ' + (chip.className.indexOf('review') > -1 ? 'review-avatar' : 'decision-avatar') + '"></span><div><strong>' + agentName + '</strong><p><b>Thinking</b><span class="thinking-dots"><i></i><i></i><i></i></span></p></div>';
    chip.style.display = 'flex';
    chip.style.visibility = 'visible';
    chip.style.alignSelf = chip.className.indexOf('review') > -1 ? 'flex-end' : 'flex-start';
    stage.appendChild(chip);
    stage.scrollTo({{ top: stage.scrollHeight, behavior: 'smooth' }});
  }}
  function clearChip() {{ if (chip) {{ chip.remove(); chip = null; }} }}
  function show(i) {{
    bubbles[i].classList.add('shown');
    Array.prototype.forEach.call(document.querySelectorAll('.dialogue-revision'), function (item) {{ item.classList.toggle('is-current', item.getAttribute('data-revision-index') === String(i)); }});
  }}
  function reveal() {{
    if (idx >= bubbles.length) {{ clearChip(); playing = false; toggleBtn.textContent = '重播'; replayBtn.style.display = 'none'; updateCount(); return; }}
    clearChip();
    show(idx);
    idx += 1;
    updateCount();
    scrollCurrent();
    if (idx < bubbles.length) schedule();
    else {{ playing = false; toggleBtn.textContent = '重播'; replayBtn.style.display = 'none'; }}
  }}
  function schedule() {{
    if (reduced) {{ while (idx < bubbles.length) reveal(); return; }}
    timer = window.setTimeout(function () {{
      showChip(bubbles[idx]);
      timer = window.setTimeout(reveal, 1600);
    }}, INTERVAL - 1600);
  }}
  function stop() {{ if (timer) {{ window.clearTimeout(timer); timer = null; }} clearChip(); }}
  function start() {{
    if (reduced) {{ bubbles.forEach(function (b, i) {{ b.classList.add('shown'); show(i); }}); idx = bubbles.length; updateCount(); toggleBtn.textContent = '重播'; replayBtn.style.display = 'none'; return; }}
    playing = true;
    toggleBtn.textContent = '暂停';
    replayBtn.style.display = '';
    reveal();
  }}
  function restart() {{
    stop();
    idx = 0;
    bubbles.forEach(function (b) {{ b.classList.remove('shown'); }});
    updateCount();
    start();
  }}
  toggleBtn.addEventListener('click', function () {{
    if (playing) {{ stop(); toggleBtn.textContent = '继续'; replayBtn.style.display = ''; }}
    else if (idx >= bubbles.length) {{ replayBtn.style.display = ''; restart(); }}
    else {{ start(); toggleBtn.textContent = '暂停'; }}
  }});
  replayBtn.addEventListener('click', function () {{ replayBtn.style.display = ''; restart(); }});
  start();
}})();
</script></body></html>
    """
    )


def _render_checks(st: Any, candidate: Any, evidence_catalog: Any) -> None:
    st.markdown("## 阶段 2 · 基础压力检查")
    check_types = tuple(check.check_type for check in candidate.checks)
    active = st.session_state.get(
        f"pressure_check_open_{candidate.candidate_id}", check_types[0]
    )
    index, reading = st.columns((1, 2.4))
    status_labels = {"RECORDED": "已记录", "MISSING": "缺失", "MATERIAL_RISK": "实质风险", "NOTE": "备注"}
    with index:
        st.markdown("<section class='pressure-check-index'><span>检查索引</span><small>选择一项阅读证据</small></section>", unsafe_allow_html=True)
        active = st.radio(
            "展开检查项",
            check_types,
            index=check_types.index(active) if active in check_types else 0,
            format_func=lambda check_type: (
                lambda check: f"{check.label} · {status_labels.get(check.status, check.status)}"
            )(next(check for check in candidate.checks if check.check_type == check_type)),
            key=f"pressure_check_open_{candidate.candidate_id}",
            horizontal=False,
            label_visibility="collapsed",
        )
    with reading:
        st.markdown(
            build_pressure_checks_html(
                candidate.checks,
                active_check_type=active,
                evidence_catalog=evidence_catalog,
            ),
            unsafe_allow_html=True,
        )


def build_pressure_checks_html(
    checks: tuple[Any, ...],
    *,
    active_check_type: str,
    evidence_catalog: Any | None = None,
) -> str:
    """Render the selected check as one evidence-reading axis."""

    if len(checks) != 5:
        raise ValueError("基础压力检查必须恰好包含五项")
    active = next(
        (check for check in checks if check.check_type == active_check_type), None
    )
    if active is None:
        raise ValueError(f"未知压力检查项：{active_check_type}")
    status_labels = {
        "RECORDED": "已记录",
        "MISSING": "缺失",
        "MATERIAL_RISK": "实质风险",
        "NOTE": "备注",
    }
    references = " · ".join(active.reference_ids) or "本项未记录额外引用"
    active_copy = active.rationale
    if active.check_type == "EVIDENCE_COVERAGE":
        active_copy = f"共关联 {len(active.reference_ids)} 条真实市场评论；优先展示三条代表性原文，其余证据保留在审计索引中。"
    audit_markup = _build_pressure_audit_detail(active.rationale, references)
    detail = (
        '<section class="pressure-check-detail">'
        f'<header><span>{_escape_html(status_labels.get(active.status, active.status))}</span>'
        f'<h3>{_escape_html(active.label)}</h3></header>'
        f'<p>{_escape_html(_display_evidence_text(active_copy))}</p>'
        f'{audit_markup}</section>'
    )
    if active.check_type == "EVIDENCE_COVERAGE" and evidence_catalog is not None:
        detail += build_evidence_excerpt_cards_html(active.reference_ids, evidence_catalog, active_copy)
    return detail


def build_evidence_excerpt_cards_html(
    reference_ids: tuple[str, ...], evidence_catalog: Any, rationale: str = ""
) -> str:
    missing = [reference for reference in reference_ids if reference not in evidence_catalog]
    if missing:
        raise ValueError("证据原文映射缺失：" + "、".join(missing))
    readings = []
    for reference in reference_ids:
        item: EvidenceExcerptView = evidence_catalog[reference]
        platform = item.source_platform or "用户评论"
        readings.append(
            "<article class='pressure-evidence-reading'>"
            "<section><small>原文摘录 · " + _escape_html(platform) + "</small>"
            f"<blockquote>“{_escape_html(_display_evidence_text(item.raw_content))}”</blockquote></section>"
            f"<section><small>检查说明</small><p>{_escape_html(_display_evidence_text(rationale))}</p></section>"
            "<section><small>当前判断</small><p>"
            "该原文已纳入本项冻结核查；完整证据编号保留在审计索引。"
            "</p></section></article>"
        )
    if not readings:
        return '<section class="pressure-evidence-reading is-empty">本项未记录消费者评论</section>'
    return '<section class="pressure-evidence-readings">' + "".join(readings) + "</section>"


def _render_dialogue(st: Any, run: PressureRunView, candidate_id: str) -> None:
    """对话剧场：客户端自动播放，回放期间服务端零参与。"""

    st.markdown(
        "<div class='pressure-dialogue-title'>阶段 3 · 双 Agent 互审</div>",
        unsafe_allow_html=True,
    )
    messages = dialogue_for_candidate(run, candidate_id)
    from streamlit.components.v1 import html as components_html

    components_html(
        build_dialogue_stage_html(messages),
        height=356,
        scrolling=False,
    )


def _render_holdout(st: Any, run: PressureRunView, cards: tuple[CandidateCard, ...]) -> None:
    st.markdown("## 阶段 4 · 统一复评与 HOLDOUT")
    st.warning(HOLDOUT_NOTE)
    st.caption("HOLDOUT：5 个候选 × 60 条留出证据；支持 / 挑战 / 中性 / 阻断")
    titles = {card.candidate_id: card.display_title for card in cards}
    rows = []
    for item in holdout_distribution(run):
        support_width = item.support_count / item.total * 100
        challenge_width = item.challenge_count / item.total * 100
        neutral_width = item.neutral_count / item.total * 100
        if round(support_width + challenge_width + neutral_width, 6) != 100:
            raise ValueError("HOLDOUT 三类分布宽度必须守恒")
        rows.append(
            "<article class='pressure-holdout-row'>"
            f"<strong>{_escape_html(titles[item.candidate_id])}</strong>"
            "<div class='pressure-holdout-bar'>"
            f"<i class='is-support' style='width:{support_width:.2f}%'></i>"
            f"<i class='is-challenge' style='width:{challenge_width:.2f}%'></i>"
            f"<i class='is-neutral' style='width:{neutral_width:.2f}%'></i></div>"
            "<span class='pressure-holdout-counts'>"
            f"支持 {item.support_count} · 挑战 {item.challenge_count} · 中性 {item.neutral_count}"
            "</span>"
            f"<span class='pressure-holdout-blocking'>阻断发现 {item.blocking_count} 项（可与分类重叠）</span>"
            "</article>"
        )
    st.markdown("<section class='pressure-holdout-grid'>" + "".join(rows) + "</section>", unsafe_allow_html=True)


def _render_terminal(
    st: Any,
    run: PressureRunView,
    *,
    on_milestone: Callable[[], None] | None = None,
) -> None:
    summary = terminal_summary(run)
    st.markdown("## 阶段 5 · 真人盲评 5→3")
    st.markdown(build_blind_review_html(run), unsafe_allow_html=True)
    revealed = bool(st.session_state.get("pressure_blind_revealed", False))
    if not revealed:
        if st.button("揭晓 5→3", key="pressure_blind_reveal", type="primary"):
            st.session_state["pressure_blind_revealed"] = True
            st.rerun()
        return
    st.markdown(build_terminal_result_html(summary), unsafe_allow_html=True)

    confirmed = bool(st.session_state.get("pressure_team_confirmed", False))
    st.markdown(
        "<section class='pressure-terminal-handoff'><span>团队交接</span><strong>"
        + ("已确认，可进入实时决策看板" if confirmed else "等待团队确认最终名单")
        + "</strong></section>",
        unsafe_allow_html=True,
    )
    if st.button("团队确认 5→3", key="pressure_team_confirm"):
        st.session_state["pressure_team_confirmed"] = True
        complete_workspace(st.session_state, Workspace.PRESSURE_TEST)
        if on_milestone is not None:
            on_milestone()
        activate_workspace(st.session_state, Workspace.REALTIME_DECISION)
        st.rerun()
    if confirmed:
        st.success("团队已确认；实时决策看板已解锁。")


def build_dialogue_revision_html(message: DialogueMessage, index: int) -> str:
    diffs = "".join(
        "<section class='dialogue-diff'><strong>" + _escape_html(diff.field_label) + "</strong>"
        f"<del>{_escape_html(_display_evidence_text(diff.before))}</del><ins>{_escape_html(_display_evidence_text(diff.after))}</ins>"
        f"<small>{_escape_html(_display_evidence_text(diff.reason))}</small></section>"
        for diff in message.field_diffs
    ) or "<p>本轮未产生字段修订；保留为审查判断。</p>"
    audit_markup = _build_pressure_audit_detail(
        *(text for diff in message.field_diffs for text in (diff.before, diff.after, diff.reason))
    )
    return (
        f"<article class='dialogue-revision' data-revision-index='{index}'><span>当前轮实际修订</span>"
        f"<strong>第 {message.round_index} 轮 · {_escape_html(message.agent)}</strong>{diffs}{audit_markup}</article>"
    )


def build_blind_review_html(run: PressureRunView) -> str:
    """Render frozen anonymous blind-review responses without inventing platforms."""

    review = run.blind_review
    metric_markup = "".join(
        "<article><strong>"
        + _escape_html(metric.label)
        + f"</strong><span>{metric.selected} 人选择 · {metric.share * 100:.1f}%</span></article>"
        for metric in review.selection_metrics
    )
    response_markup = "".join(
        f"<li class='blind-review-quote'><span>{index + 1:02d}</span>{_escape_html(item.text)}</li>"
        for index, item in enumerate(review.open_responses)
    )
    return (
        "<section class='blind-review-stage'>"
        "<header><span>真人盲评 · 冻结回放</span><strong>问卷选项偏好</strong>"
        f"<small>多选口径 · {review.survey_count} 份有效问卷为分母</small></header>"
        "<div class='blind-review-layout'><section class='blind-review-metrics'>"
        f"{metric_markup}</section><aside class='blind-review-quotes'><h3>完整匿名原话</h3>"
        f"<ol>{response_markup}</ol></aside></div>"
        f"<footer>{review.open_response_count} 条开放回答，按冻结问卷原顺序保留</footer></section>"
    )


def build_terminal_result_html(summary: TerminalSummary) -> str:
    """Render the factual 5→3 relationship without implying vote magnitude."""

    source_nodes = "".join(
        "<li class='pressure-terminal-source "
        + ("is-selected" if card.selected else "is-rejected")
        + "'><span class='pressure-terminal-relationship'></span><b>"
        + _escape_html(card.display_title)
        + "</b><small>"
        + ("→ 入选方向" if card.selected else "→ 历史保留")
        + "</small></li>"
        for card in (*summary.selected, *summary.not_selected)
    )
    selected = "".join(
        "<li><b>入选方向</b><strong>" + _escape_html(card.display_title) + "</strong></li>"
        for card in summary.selected
    )
    rejected = "".join(
        "<li>" + _escape_html(card.display_title) + "</li>" for card in summary.not_selected
    )
    return (
        "<section class='pressure-terminal pressure-terminal-map'><header><span>最终选择路径</span>"
        f"<h3>{_escape_html(summary.method)}</h3></header><div class='pressure-terminal-sources'><ol>{source_nodes}</ol></div>"
        f"<section class='pressure-terminal-selected'><span>三条入选</span><ol>{selected}</ol></section>"
        f"<aside class='pressure-terminal-register'><span>两条未入选 · 历史保留</span><ol>{rejected}</ol></aside></section>"
    )


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _display_evidence_text(value: str) -> str:
    """Keep frozen evidence prose readable while removing internal identifiers."""

    return _RAW_EVIDENCE_ID.sub("对应原始评论", value)


def _build_pressure_audit_detail(*raw_values: str) -> str:
    """Keep original frozen wording and IDs behind an explicit disclosure control."""

    raw_text = "\n\n".join(value for value in raw_values if value)
    return (
        "<details class='pressure-audit-detail'><summary>审计详情 · 原始文本与编号</summary>"
        f"<p>{_escape_html(raw_text)}</p></details>"
    )


def _dialogue_frame_css() -> str:
    return """
html,body{margin:0;background:#EFE9E0;color:#292521;font-family:"Microsoft YaHei","PingFang SC",sans-serif;position:relative;}
.dialogue-stage{position:relative;display:flex;flex-direction:column;gap:.65rem;height:302px;padding:.9rem .95rem 2.6rem;overflow-y:auto;box-sizing:border-box;scroll-behavior:smooth;border:1px solid #D8CCBE;background:linear-gradient(180deg,#F3EDE4,#ECE4D8);}
.dialogue-bubble{display:none;gap:.8rem;align-items:flex-start;width:fit-content;max-width:min(80%,40rem);padding:.9rem 1rem;box-sizing:border-box;border-left:3px solid #8F2F4D;background:#FAF7F1;box-shadow:0 .45rem 1.2rem rgba(63,48,54,.06);}
.dialogue-bubble.shown{display:flex;animation:dialogueInLeft .45s cubic-bezier(.16,1,.3,1) both;}
.dialogue-bubble.review-avatar-bubble{align-self:flex-end;border-left-color:#C9A86A;background:#FBF5E8;animation-name:dialogueInRight;}
.dialogue-thinking{display:flex;align-items:center;gap:.7rem;width:min(42%,24rem);padding:.7rem .85rem;box-sizing:border-box;border:1px dashed #BDAFA3;background:rgba(255,253,249,.58);animation:dialogueInLeft .35s ease-out both;}
.dialogue-thinking.review-avatar-bubble{align-self:flex-end;border-color:#C9A86A;animation:dialogueInRight .35s ease-out both}.dialogue-thinking.decision-avatar-bubble{align-self:flex-start;border-color:#BC8798}
.dialogue-thinking>div{display:grid;grid-template-columns:auto 1fr;gap:.12rem .55rem;align-items:center}.dialogue-thinking strong{color:#5E5550;font-size:.7rem}.dialogue-thinking p{display:flex;align-items:center;gap:.35rem;margin:0;color:#8F2F4D;font-size:.68rem}.dialogue-thinking small{grid-column:1/-1;color:#8A8078;font-size:.6rem}
.thinking-dots{display:inline-flex;gap:.16rem}.thinking-dots i{width:.25rem;height:.25rem;border-radius:50%;background:#8F2F4D;animation:thinkingDot 1.15s ease-in-out infinite}.thinking-dots i:nth-child(2){animation-delay:.16s}.thinking-dots i:nth-child(3){animation-delay:.32s}
.dialogue-bubble__body{min-width:0;flex:1;}
.dialogue-bubble header{display:flex;justify-content:space-between;gap:1rem;color:#8F2F4D;font-size:.7rem;}
.dialogue-bubble.review-avatar-bubble header{color:#8A6B33;}
.dialogue-bubble h3{margin:.35rem 0;color:#302B28;font-family:STZhongsong,"华文中宋",serif;font-size:1.02rem;font-weight:650;}
.dialogue-bubble p,.dialogue-bubble li{color:#655D57;font-size:.76rem;line-height:1.65;}
.dialogue-bubble details{margin-top:.55rem;border-top:1px solid #DED3C8;padding-top:.5rem;}
.dialogue-bubble summary{cursor:pointer;color:#746A62;font-size:.7rem;}
.dialogue-detail-copy{font-family:KaiTi,"楷体",serif;}
.dialogue-bubble ul{margin:.55rem 0 0;padding-left:1.1rem;}
.dialogue-diffs{display:grid;gap:.55rem;margin-top:.65rem;}
.dialogue-diff{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;padding:.65rem;border:1px solid #DED3C8;background:#FFFDF9;}
.dialogue-diff>strong,.dialogue-diff>em{grid-column:1/-1;}.dialogue-diff>strong{color:#8F2F4D;font-size:.72rem;}.dialogue-diff div{padding:.5rem;background:#F3EDE6;}.dialogue-diff div:nth-of-type(2){background:#F7F0DE;}.dialogue-diff small{color:#8A8078;font-size:.62rem;}.dialogue-diff p{margin:.2rem 0 0;}.dialogue-diff em{color:#81776F;font-size:.68rem;font-style:normal;}
.agent-avatar{display:inline-block;flex:0 0 2.25rem;width:2.25rem;height:2.25rem;position:relative;}.decision-avatar{border:2px solid #8F2F4D;border-radius:50%;background:conic-gradient(from 45deg,transparent 0 20%,#8F2F4D 20% 25%,transparent 25% 45%,#8F2F4D 45% 50%,transparent 50% 70%,#8F2F4D 70% 75%,transparent 75%);}.decision-avatar::after{content:"";position:absolute;inset:35%;border-radius:50%;background:#8F2F4D;}.review-avatar{border:2px solid #C9A86A;border-radius:35% 35% 48% 48%;background:#C9A86A;clip-path:polygon(50% 0,92% 20%,84% 72%,50% 100%,16% 72%,8% 20%);}.review-avatar::after{content:"";position:absolute;inset:35%;border:2px solid #8F2F4D;transform:rotate(45deg);}
@keyframes dialogueInLeft{from{opacity:0;transform:translate(-14px,8px)}to{opacity:1;transform:translate(0,0)}}
@keyframes dialogueInRight{from{opacity:0;transform:translate(14px,8px)}to{opacity:1;transform:translate(0,0)}}
@keyframes thinkingDot{0%,70%,100%{opacity:.25;transform:translateY(0)}35%{opacity:1;transform:translateY(-3px)}}
.dialogue-layout{display:grid;grid-template-columns:65fr 35fr;height:302px;border:1px solid #D8CCBE;background:#EFE9E0}.dialogue-review-axis{height:302px;border:0;border-right:1px solid #D8CCBE}.dialogue-revision-panel{overflow-y:auto;padding:.8rem;background:#F8F3EB}.dialogue-revision{display:none}.dialogue-revision.is-current{display:grid;gap:.45rem}.dialogue-revision>span{color:#8F2F4D;font-size:.62rem;font-weight:700;letter-spacing:.1em}.dialogue-revision>strong{color:#443C37;font-size:.75rem}.dialogue-revision .dialogue-diff{grid-template-columns:1fr;padding:.5rem;border:0;border-left:2px solid #C9A86A}.dialogue-revision del{padding:.45rem;background:#F3E5E1;color:#8A5D58;text-decoration-color:#8F2F4D}.dialogue-revision ins{padding:.45rem;background:#EAF1E9;color:#365B4B;text-decoration-color:#365B4B}.dialogue-revision small{color:#746A62;font-size:.66rem;line-height:1.5}@media (prefers-reduced-motion: reduce){*{animation:none!important;scroll-behavior:auto!important}.dialogue-thinking{display:none!important}}@media(max-width:720px){.dialogue-layout{grid-template-columns:1fr;height:auto}.dialogue-review-axis{height:18rem;border-right:0;border-bottom:1px solid #D8CCBE}.dialogue-revision-panel{max-height:14rem}}
""".strip()


def _pressure_test_css() -> str:
    base_css = """
[data-testid="stAppViewContainer"] { background:#F4EFE7; color:#292521; }
.block-container,[data-testid="stMainBlockContainer"] { width:100%; max-width:94rem; padding:.65rem clamp(1rem,2.2vw,2.2rem) 2rem; }
[data-testid="stButton"]>button,.stButton>button { min-height:2.5rem; height:2.5rem; padding:.35rem .55rem; font-size:.76rem; line-height:1.2; }
.pressure-hero { position:relative; min-height:7.25rem; margin:.05rem 0 .25rem; padding:.65rem 1rem .7rem; overflow:hidden; border:1px solid #D8CCBE; background:#ECE5DC; }
.pressure-hero::after { content:"验"; position:absolute; right:2.5rem; top:-2.2rem; color:rgba(143,47,77,.07); font-family:STZhongsong,"华文中宋",serif; font-size:11rem; }
.pressure-hero > span,.pressure-current > span,.pressure-stage-bar span { color:#8F2F4D; font-size:.7rem; font-weight:750; letter-spacing:.16em; }
.pressure-hero h1 { margin:.06rem 0 .08rem; color:#352F2B; font-size:clamp(1.8rem,2.6vw,2.5rem); font-weight:500; line-height:1.12; }
.pressure-hero p { max-width:60rem; margin:0; color:#655D57; font-size:.88rem; line-height:1.5; }
.pressure-hero small { display:block; margin-top:.18rem; color:#8A8078; font-size:.62rem; }
.pressure-candidate-meta { display:flex; justify-content:space-between; gap:.4rem; margin-top:-.3rem; padding:.3rem .55rem; border:1px solid #CBBCAF; border-top:0; color:#6F665F; background:#FAF7F1; font-size:.64rem; }
.pressure-candidate-meta-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:1rem; }
.pressure-candidate-meta.is-selected { border-color:#8F2F4D; color:#8F2F4D; box-shadow:0 .5rem 1.2rem rgba(143,47,77,.08); }
.pressure-candidate-meta.is-rejected { opacity:.65; border-color:#AEB7AF; }
.pressure-stage-bar { margin:.42rem 0 .16rem; padding-top:.34rem; border-top:1px solid #D8CCBE; }
.pressure-stage-bar>small { color:#8F2F4D; font-size:.65rem; font-weight:750; letter-spacing:.14em; }
.pressure-stage-bar>div { display:grid; grid-template-columns:repeat(5,1fr); gap:.35rem; margin-top:.28rem; }
.pressure-stage-node { display:flex; align-items:center; gap:.38rem; min-width:0; color:#948A82; font-size:.66rem; }
.pressure-stage-node::after { content:""; height:1px; flex:1; background:#D8CCBE; }
.pressure-stage-node:last-child::after { display:none; }
.pressure-stage-node i { display:grid; place-items:center; flex:0 0 1.35rem; height:1.35rem; border:1px solid #BEB2A7; border-radius:50%; font-style:normal; }
.pressure-stage-node b { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-weight:600; }
.pressure-stage-node.is-complete,.pressure-stage-node.is-current { color:#8F2F4D; }
.pressure-stage-node.is-complete i,.pressure-stage-node.is-current i { border-color:#8F2F4D; background:#8F2F4D; color:#FFF8EF; }
.pressure-stage-now { padding:.32rem; text-align:center; color:#5F5751; font-size:.68rem; letter-spacing:.06em; }
.pressure-current { display:grid; grid-template-columns:1fr auto; gap:.15rem 1.4rem; margin:.3rem 0 .25rem; padding:.5rem .85rem; border-left:4px solid #8F2F4D; background:#FAF7F1; }
.pressure-current > span,.pressure-current-title,.pressure-current p { grid-column:1; }
.pressure-current-title { margin:0; color:#352F2B; font-family:STZhongsong,"华文中宋",serif; font-size:1.72rem; }
.pressure-current p { margin:.2rem 0 0; color:#655D57; font-size:.84rem; line-height:1.55; }
.pressure-current div { grid-column:2; grid-row:1 / 4; display:grid; grid-template-columns:auto auto; gap:.2rem .65rem; align-content:center; min-width:13rem; }
.pressure-current div b { color:#8F2F4D; font-family:Georgia,serif; font-size:1.05rem; }
.pressure-current div small { color:#81776F; }
.pressure-original-stage { min-height:18rem; padding:2rem; border:1px solid #D8CCBE; background:linear-gradient(135deg,#FAF7F1,#ECE5DC); }
.pressure-original-stage > span { color:#8F2F4D; font-size:.7rem; font-weight:700; letter-spacing:.12em; }
.pressure-original-stage h2 { max-width:44rem; margin:.8rem 0; color:#352F2B; font-family:STZhongsong,"华文中宋",serif; font-size:2rem; }
.pressure-original-stage p { max-width:50rem; color:#655D57; line-height:1.8; }
.pressure-check-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.7rem; margin:1rem 0; }
.pressure-check-card { min-height:8.5rem; padding:1rem; border-top:3px solid #89968D; background:#FAF7F1; }
.pressure-check-card.is-active { border-color:#8F2F4D; background:#FFF9F5; box-shadow:0 .6rem 1.5rem rgba(63,48,54,.06); }
.pressure-check-card span,.pressure-check-detail span { color:#8F2F4D; font-size:.65rem; font-weight:700; letter-spacing:.1em; }
.pressure-check-card strong { display:block; margin:.5rem 0; color:#39332F; }
.pressure-check-card p { display:-webkit-box; margin:0; overflow:hidden; color:#6B625B; font-size:.72rem; line-height:1.55; -webkit-box-orient:vertical; -webkit-line-clamp:2; }
.pressure-check-detail { min-height:10rem; padding:1.2rem; border-left:4px solid #8F2F4D; background:#FAF7F1; }
.pressure-check-detail h3 { margin:.35rem 0; color:#39332F; }
.pressure-check-detail p { color:#5F5751; line-height:1.7; }
.pressure-check-detail small { color:#81776F; }
.pressure-evidence-gallery { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.7rem; margin:.75rem 0; }
.pressure-evidence-audit { margin-top:.55rem; color:#81776F; font-size:.64rem; }
.pressure-evidence-audit summary { cursor:pointer; color:#6F665F; font-weight:650; letter-spacing:.04em; }
.pressure-evidence-audit small { display:block; margin-top:.45rem; line-height:1.6; word-break:break-all; }
.pressure-evidence-card { position:relative; min-height:10rem; padding:1rem; border:1px solid #D8CCBE; background:linear-gradient(150deg,#FFFDF9,#F1E9DF); box-shadow:0 .5rem 1.25rem rgba(63,48,54,.05); }
.pressure-evidence-card header { display:flex; align-items:center; gap:.42rem; }
.pressure-evidence-card header>span { display:grid; place-items:center; width:1.45rem; height:1.45rem; border-radius:50%; color:#FFF8EF; background:#8F2F4D; font-size:.66rem; }
.pressure-evidence-card header b { color:#4F4741; font-size:.72rem; }
.pressure-evidence-card header small { margin-left:auto; color:#8F2F4D; font-size:.58rem; letter-spacing:.08em; }
.pressure-evidence-card blockquote { margin:.85rem 0; color:#443D38; font-family:KaiTi,"楷体",serif; font-size:1rem; line-height:1.65; }
.pressure-evidence-card details { color:#756B64; font-size:.65rem; }
.pressure-evidence-card details p { font-size:.72rem; line-height:1.65; }
.pressure-evidence-id { margin-top:.65rem; color:#A0968E; font-size:.55rem; letter-spacing:.03em; }
.pressure-evidence-gallery>p { grid-column:1/-1; margin:0; color:#81776F; font-size:.62rem; text-align:right; }
.pressure-dialogue-title { margin:.55rem 0 .3rem; color:#352F2B; font-family:STZhongsong,"华文中宋",serif; font-size:1.35rem; font-weight:700; }
.dialogue-stage { display:grid; gap:.75rem; height:14rem; margin:.2rem 0; padding:.7rem; overflow-y:auto; border:1px solid #D8CCBE; background:#E8E1D8; scroll-behavior:smooth; }
.dialogue-bubble { display:flex; gap:.75rem; align-items:flex-start; width:min(88%,68rem); padding:.8rem; border-left:3px solid #8f2f4d; background:#FAF7F1; box-shadow:0 .5rem 1.5rem rgba(63,48,54,.05); animation:dialogueBubbleIn .42s ease-out both; }
.dialogue-bubble.review-avatar-bubble { justify-self:end; border-left-color:#365b4b; background:#F1F3EE; }
.dialogue-bubble__body { min-width:0; flex:1; }
.dialogue-bubble header { display:flex; justify-content:space-between; gap:1rem; color:#8F2F4D; font-size:.7rem; }
.dialogue-bubble.review-avatar-bubble header { color:#365B4B; }
.dialogue-bubble h3 { margin:.4rem 0; color:#39332F; font-size:1rem; }
.dialogue-bubble p,.dialogue-bubble li { color:#655D57; font-size:.76rem; line-height:1.65; }
.dialogue-bubble ul { margin:.6rem 0 0; padding-left:1.15rem; }
.dialogue-diffs { display:grid; gap:.65rem; margin-top:.8rem; }
.dialogue-diff { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; padding:.75rem; border:1px solid #DED3C8; background:#FFFDF9; }
.dialogue-diff > strong,.dialogue-diff > em { grid-column:1 / -1; }
.dialogue-diff > strong { color:#8F2F4D; font-size:.74rem; }
.dialogue-diff div { padding:.55rem; background:#F3EDE6; }
.dialogue-diff div:nth-of-type(2) { background:#EDF2ED; }
.dialogue-diff small { color:#8A8078; font-size:.62rem; }
.dialogue-diff p { margin:.25rem 0 0; }
.dialogue-diff em { color:#81776F; font-size:.68rem; font-style:normal; }
.agent-avatar { display:inline-block; flex:0 0 2.4rem; width:2.4rem; height:2.4rem; position:relative; }
.decision-avatar { border:2px solid #8f2f4d; border-radius:50%; background:conic-gradient(from 45deg, transparent 0 20%, #8f2f4d 20% 25%, transparent 25% 45%, #8f2f4d 45% 50%, transparent 50% 70%, #8f2f4d 70% 75%, transparent 75%); }
.decision-avatar::after { content:""; position:absolute; inset:35%; border-radius:50%; background:#8f2f4d; }
.review-avatar { border:2px solid #365b4b; border-radius:35% 35% 48% 48%; background:#365b4b; clip-path:polygon(50% 0, 92% 20%, 84% 72%, 50% 100%, 16% 72%, 8% 20%); }
.review-avatar::after { content:""; position:absolute; inset:35%; border:2px solid #e5c47d; transform:rotate(45deg); }
@keyframes dialogueBubbleIn { from { opacity:0; transform:translateY(1rem); } to { opacity:1; transform:translateY(0); } }
.pressure-holdout-grid { display:grid; gap:.55rem; margin:1rem 0; }
.pressure-holdout-row { display:grid; grid-template-columns:minmax(14rem,1.15fr) minmax(12rem,1fr) repeat(4,auto); gap:.7rem; align-items:center; padding:.75rem 1rem; border:1px solid #D8CCBE; background:#FAF7F1; }
.pressure-holdout-row strong { font-size:.78rem; }
.pressure-holdout-row span { color:#766D66; font-size:.7rem; }
.pressure-holdout-bar { display:flex; height:7px; overflow:hidden; background:#D8D9D2; }
.pressure-holdout-bar i.is-support { background:#496A5A; }
.pressure-holdout-bar i.is-challenge { background:#8F2F4D; }
.pressure-terminal { display:grid; grid-template-columns:3fr 2fr; gap:.8rem; margin:1rem 0; }
.pressure-terminal header { grid-column:1 / -1; padding:1.2rem 1.4rem; background:#3F3036; color:#FFF8EF; }
.pressure-terminal header span { color:#D9C69A; font-size:.68rem; letter-spacing:.12em; }
.pressure-terminal header h3 { margin:.35rem 0 0; color:#FFF8EF; }
.pressure-terminal > div { padding:1.2rem; border:1px solid #D8CCBE; background:#FAF7F1; }
.pressure-terminal > div.is-selected { border-top:3px solid #8F2F4D; }
.pressure-terminal > div.is-rejected { border-top:3px solid #89968D; opacity:.78; }
.pressure-terminal ol { margin:.7rem 0 0; padding-left:1.2rem; }
.pressure-terminal li { margin:.65rem 0; }
.pressure-terminal li b,.pressure-terminal li span { display:block; }
.pressure-terminal li span { color:#81776F; font-size:.7rem; }
.blind-review-stage { margin:.55rem 0 .7rem; padding:.85rem 1rem; border:1px solid #D8CCBE; background:#EEE6DC; overflow:hidden; }
.blind-review-stage header { display:flex; align-items:baseline; gap:1rem; }
.blind-review-stage header span { margin-right:auto; color:#8F2F4D; font-size:.68rem; letter-spacing:.12em; }
.blind-review-stage header strong { color:#403935; font-family:STZhongsong,"华文中宋",serif; font-size:1rem; }
.blind-review-metrics { display:grid; grid-template-columns:repeat(5,1fr); gap:.45rem; margin:.65rem 0; }
.blind-review-metrics article { display:grid; gap:.2rem; padding:.48rem .6rem; background:#FAF7F1; }
.blind-review-metrics strong { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#4A423D; font-size:.68rem; }
.blind-review-metrics span { color:#8F2F4D; font-size:.65rem; }
.blind-review-stage footer { margin-top:.45rem; color:#81776F; font-size:.63rem; text-align:right; }
@media (max-width:900px) { .block-container,[data-testid="stMainBlockContainer"] { padding:.55rem .7rem 1.5rem; } .pressure-current,.pressure-terminal,.pressure-candidate-meta-grid,.pressure-evidence-gallery { grid-template-columns:1fr; } .pressure-current div { grid-column:1; grid-row:auto; min-width:0; } .pressure-check-grid { grid-template-columns:1fr 1fr; } .pressure-holdout-row { grid-template-columns:1fr 1fr; } .dialogue-bubble { width:100%; } .dialogue-diff { grid-template-columns:1fr; } .dialogue-diff > strong,.dialogue-diff > em { grid-column:auto; } }
""".strip()
    redesign_css = """
.pressure-candidate-nav{margin:.35rem 0 .15rem;padding:.5rem 0;border-bottom:2px solid #8F2F4D;color:#8F2F4D;font-size:.7rem;font-weight:750;letter-spacing:.14em}.pressure-candidate-meta-grid{display:block;margin:-.12rem 0 .45rem}.pressure-candidate-meta{margin:0;padding:.22rem .35rem;border:0;border-left:2px solid #D8CCBE;background:transparent}.pressure-candidate-nav~[data-testid="stButton"]>button{height:auto;min-height:3.15rem;text-align:left;justify-content:flex-start;border:0;border-bottom:1px solid #E0D5C8;border-radius:0;background:transparent;font-size:.76rem;white-space:normal}.pressure-candidate-nav~[data-testid="stButton"]>button:hover{background:#FFF8F3}.pressure-score-dotplot{margin:1rem 0 0}.pressure-score-dotplot header{display:flex;justify-content:space-between;gap:1rem;color:#8F2F4D;font-size:.72rem}.pressure-score-axis{display:flex;justify-content:space-between;margin:.7rem 0 .25rem;border-top:1px solid #BDAFA3;color:#8A8078;font-size:.6rem}.pressure-score-axis i{font-style:normal;transform:translateY(-.1rem)}.pressure-score-dotplot ol{display:grid;gap:.28rem;margin:0;padding:0;list-style:none}.pressure-score-dot{display:grid;grid-template-columns:minmax(9rem,1fr) minmax(4rem,2.3fr) 3rem;gap:.6rem;align-items:center;font-size:.72rem}.pressure-score-dot i{height:8px;background:linear-gradient(90deg,#E7DDD2 calc(var(--score) * 1%),transparent 0);position:relative}.pressure-score-dot i::after{content:"";position:absolute;left:calc(var(--score) * 1% - 4px);top:-3px;width:8px;height:8px;border-radius:50%;background:#8A8078}.pressure-score-dot.is-selected{color:#8F2F4D;font-weight:750}.pressure-score-dot.is-selected i::after{width:12px;height:12px;top:-5px;left:calc(var(--score) * 1% - 6px);background:#8F2F4D}.pressure-score-dot span{text-align:right;font-variant-numeric:tabular-nums}.pressure-check-index{display:grid;gap:.25rem;margin-top:.55rem;padding:.45rem 0;border-top:2px solid #8F2F4D;color:#8F2F4D}.pressure-check-index small{color:#81776F}.pressure-check-detail{min-height:0}.pressure-evidence-readings{display:grid;gap:.65rem;margin-top:.8rem}.pressure-evidence-reading{display:grid;grid-template-columns:1.3fr 1fr 1fr;gap:.7rem;padding:.75rem 0;border-top:1px solid #DED3C8}.pressure-evidence-reading small{color:#8F2F4D;font-weight:700;letter-spacing:.08em}.pressure-evidence-reading blockquote{margin:.45rem 0 0;color:#403935;font-family:KaiTi,"楷体",serif;line-height:1.65}.pressure-evidence-reading p{margin:.45rem 0 0;font-size:.75rem;line-height:1.6}.pressure-holdout-row{grid-template-columns:minmax(10rem,1fr) minmax(12rem,2fr) minmax(10rem,1fr) minmax(10rem,1fr)}.pressure-holdout-bar i.is-neutral{background:#AEB7AF}.pressure-holdout-counts,.pressure-holdout-blocking{text-align:right;font-variant-numeric:tabular-nums}.pressure-holdout-blocking{color:#8F2F4D!important}.blind-review-stage header{display:grid;gap:.2rem}.blind-review-stage header small{color:#81776F}.blind-review-layout{display:grid;grid-template-columns:65fr 35fr;gap:1rem;margin-top:.8rem}.blind-review-metrics{display:grid;grid-template-columns:1fr;gap:.5rem;margin:0}.blind-review-metrics article{grid-template-columns:1fr auto;align-items:center;padding:.55rem 0;border-bottom:1px solid #DED3C8;background:transparent}.blind-review-metrics span{text-align:right;font-variant-numeric:tabular-nums}.blind-review-quotes{max-height:18rem;overflow-y:auto;padding-left:.8rem;border-left:1px solid #D8CCBE}.blind-review-quotes h3{margin:0 0 .4rem;font-size:.78rem}.blind-review-quotes ol{display:grid;gap:.4rem;margin:0;padding:0;list-style:none}.blind-review-quote{display:grid;grid-template-columns:2rem 1fr;gap:.4rem;color:#514741;font-size:.72rem;line-height:1.55}.blind-review-quote span{color:#8F2F4D;font-variant-numeric:tabular-nums}.pressure-terminal-map{grid-template-columns:1fr 1.8fr;align-items:start}.pressure-terminal-map header{grid-column:1/-1}.pressure-terminal-sources{position:relative;padding-right:2rem}.pressure-terminal-sources ol,.pressure-terminal-selected ol,.pressure-terminal-register ol{display:grid;gap:.55rem;margin:.5rem 0;padding:0;list-style:none}.pressure-terminal-source{display:flex;gap:.45rem;align-items:center;font-size:.75rem}.pressure-terminal-source span{width:.55rem;height:.55rem;border:1px solid #8F2F4D;border-radius:50%}.pressure-terminal-connector{position:absolute;right:0;top:.7rem;bottom:.7rem;width:1px;background:#8F2F4D}.pressure-terminal-selected{padding-left:1rem;border-left:2px solid #8F2F4D}.pressure-terminal-selected li{padding:.5rem 0;border-bottom:1px solid #DED3C8}.pressure-terminal-selected b,.pressure-terminal-selected strong{display:block}.pressure-terminal-register{grid-column:1/-1;margin-top:.3rem;padding-top:.6rem;border-top:1px solid #DED3C8}.pressure-terminal-register ol{grid-template-columns:repeat(2,minmax(0,1fr))}.pressure-terminal-handoff{display:flex;align-items:baseline;gap:.6rem;margin-top:.4rem;padding:.7rem 0;border-top:2px solid #8F2F4D}.pressure-terminal-handoff span{color:#8F2F4D;font-size:.68rem;font-weight:700;letter-spacing:.1em}@media(prefers-reduced-motion:reduce){.pressure-stage-node.is-current i,.pressure-holdout-bar i,.pressure-terminal,.pressure-terminal li{animation:none!important}}@media(max-width:900px){.pressure-evidence-reading,.blind-review-layout,.pressure-terminal-map{grid-template-columns:1fr}.pressure-terminal-register{grid-column:auto}.pressure-terminal-register ol{grid-template-columns:1fr}.pressure-holdout-row{grid-template-columns:1fr}.pressure-holdout-counts,.pressure-holdout-blocking{text-align:left}}
"""
    connection_css = """
.pressure-terminal-source{display:grid;grid-template-columns:2.4rem 1fr;column-gap:.45rem;align-items:center}.pressure-terminal-relationship{position:relative;display:block!important;width:2.4rem!important;height:1px!important;border:0!important;border-radius:0!important;background:#8F2F4D}.pressure-terminal-relationship::after{content:"";position:absolute;right:0;top:-2px;border-width:3px 0 3px 5px;border-style:solid;border-color:transparent transparent transparent #8F2F4D}.pressure-terminal-source b{font-weight:650}.pressure-terminal-source small{grid-column:2;color:#81776F}.pressure-terminal-source.is-rejected .pressure-terminal-relationship{background:#AEB7AF}.pressure-terminal-source.is-rejected .pressure-terminal-relationship::after{border-left-color:#AEB7AF}
"""
    return f"{base_css}\n{redesign_css}\n{connection_css}"


def _get_streamlit() -> Any:
    import streamlit as st

    return st


__all__ = [
    "CandidateCard",
    "DialogueMessage",
    "FINAL_METHOD",
    "HOLDOUT_NOTE",
    "HoldoutDistribution",
    "PRESSURE_STAGES",
    "TerminalSummary",
    "build_candidate_cards",
    "build_candidate_meta_html",
    "build_blind_review_html",
    "build_current_candidate_html",
    "active_pressure_section",
    "advance_dialogue_replay",
    "build_dialogue_message_html",
    "build_dialogue_stage_html",
    "build_pressure_checks_html",
    "build_evidence_excerpt_cards_html",
    "dialogue_for_candidate",
    "holdout_distribution",
    "move_pressure_stage",
    "render_pressure_test_workspace",
    "select_pressure_candidate",
    "terminal_summary",
]
