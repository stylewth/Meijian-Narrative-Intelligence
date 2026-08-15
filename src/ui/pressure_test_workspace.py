"""只读的五候选验证流水线剧场。"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from src.services.candidate_ranking import calculate_weighted_score
from src.services.pressure_test_presentation import (
    AgentFieldDiffView,
    AgentMessageView,
    EvidenceExcerptView,
    PressureRunView,
    load_pressure_test_run,
)
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
        f"<small>正式运行 · {_escape_html(run.validation_root.name)}</small></section>",
        unsafe_allow_html=True,
    )

    candidate_id = _selected_candidate_id(st, cards)
    candidate_id = _render_candidate_cards(st, cards, candidate_id)
    current_stage = _render_stage_navigation(st)

    candidate = next(item for item in run.candidates if item.candidate_id == candidate_id)
    card = next(item for item in cards if item.candidate_id == candidate_id)
    section = active_pressure_section(current_stage)
    if section in {"original", "checks", "dialogue"}:
        _render_current_candidate(st, card)
    if section == "original":
        _render_original_stage(st, card)
    elif section == "checks":
        _render_checks(st, candidate, run.evidence_catalog)
    elif section == "dialogue":
        _render_dialogue(st, run, candidate_id)
    elif section == "holdout":
        _render_holdout(st, run)
    elif section == "terminal":
        _render_terminal(st, run)


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


def _render_original_stage(st: Any, card: CandidateCard) -> None:
    st.markdown(
        "<section class='pressure-original-stage'><span>阶段 1 · AI 原始五项</span>"
        "<h2>先保留原始判断，再看压力测试如何改变选择</h2>"
        f"<p>当前查看：{_escape_html(card.display_title)}。原始得分 {card.original_score:.1f}，"
        f"AI 排名第 {card.ai_rank}。此处保持盲评前版本，不提前展示修订结论。</p></section>",
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
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        with column:
            if st.button(
                card.display_title,
                key=f"pressure_candidate_button_{card.candidate_id}",
                use_container_width=True,
            ):
                select_pressure_candidate(st.session_state, card.candidate_id, ids)
                current_id = card.candidate_id
    st.markdown(
        build_candidate_meta_html(
            cards,
            current_id,
            reveal=bool(st.session_state.get("pressure_blind_revealed", False)),
        ),
        unsafe_allow_html=True,
    )
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
            f"<span>AI #{card.ai_rank}</span><span>{_escape_html(status)}</span></article>"
        )
    return '<section class="pressure-candidate-meta-grid">' + "".join(items) + "</section>"


def _render_stage_navigation(st: Any) -> str:
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
        f"<li>{_escape_html(item)}</li>" for item in message.highlights
    )
    diffs = "".join(
        "<section class='dialogue-diff'>"
        f"<strong>{_escape_html(diff.field_label)}</strong>"
        f"<div><small>修改前</small><p>{_escape_html(diff.before)}</p></div>"
        f"<div><small>修改后</small><p>{_escape_html(diff.after)}</p></div>"
        f"<em>{_escape_html(diff.reason)}</em></section>"
        for diff in message.field_diffs
    )
    highlights_markup = f"<ul>{highlights}</ul>" if highlights else ""
    diff_markup = f"<div class='dialogue-diffs'>{diffs}</div>" if diffs else ""
    detail = (
        "<details><summary>本轮详细理由与操作</summary>"
        f"<p class='dialogue-detail-copy'>{_escape_html(message.body)}</p>"
        f"{highlights_markup}{diff_markup}</details>"
    )
    return (
        f'<article class="dialogue-bubble {avatar_class}-bubble" data-current="{"true" if is_current else "false"}">'
        f"<span class='agent-avatar {avatar_class}'></span>"
        "<div class='dialogue-bubble__body'>"
        f"<header><strong>{_escape_html(message.agent)}</strong>"
        f"<span>第 {message.round_index} 轮</span></header>"
        f"<h3>{_escape_html(message.headline)}</h3>"
        f"<p>{_escape_html(message.summary)}</p>{detail}"
        "</div></article>"
    )


def build_dialogue_stage_html(
    messages: tuple[DialogueMessage, ...],
    *,
    thinking_agent: str | None = None,
) -> str:
    """Build a fixed-height chat frame that centers the newest visible message."""

    if not messages:
        raise ValueError("对话舞台至少需要一条消息")
    if thinking_agent is not None and thinking_agent not in {"审查 Agent", "决策 Agent"}:
        raise ValueError(f"未知 Thinking Agent：{thinking_agent}")
    timeline = "".join(
        build_dialogue_message_html(
            message,
            is_current=thinking_agent is None and index == len(messages) - 1,
        )
        for index, message in enumerate(messages)
    )
    if thinking_agent is not None:
        avatar_class = "decision-avatar" if thinking_agent == "决策 Agent" else "review-avatar"
        timeline += (
            f'<article class="dialogue-thinking {avatar_class}-bubble" data-current="true">'
            f'<span class="agent-avatar {avatar_class}"></span><div>'
            f"<strong>{_escape_html(thinking_agent)}</strong>"
            '<p><b>Thinking</b><span class="thinking-dots"><i></i><i></i><i></i></span></p>'
            "<small>正在分析证据</small></div></article>"
        )
    stage_class = (
        "dialogue-stage is-single"
        if len(messages) == 1 and thinking_agent is None
        else "dialogue-stage"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        + _dialogue_frame_css()
        + f"</style></head><body><section class='{stage_class}' id='dialogue-stage'>"
        + timeline
        + "</section><script>requestAnimationFrame(() => { const stage = document.getElementById('dialogue-stage'); const current = document.querySelector('[data-current=\"true\"]'); if (stage && current) { const top = current.offsetTop - (stage.clientHeight - current.offsetHeight) / 2; stage.scrollTo({top: Math.max(0, top), behavior: 'smooth'}); } });</script></body></html>"
    )


def _render_checks(st: Any, candidate: Any, evidence_catalog: Any) -> None:
    st.markdown("## 阶段 2 · 基础压力检查")
    check_types = tuple(check.check_type for check in candidate.checks)
    active = st.session_state.get(
        f"pressure_check_open_{candidate.candidate_id}", check_types[0]
    )
    active = st.radio(
        "展开检查项",
        check_types,
        index=check_types.index(active) if active in check_types else 0,
        format_func=lambda check_type: next(
            check.label for check in candidate.checks if check.check_type == check_type
        ),
        key=f"pressure_check_open_{candidate.candidate_id}",
        horizontal=True,
    )
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
    """Render five compact cards and exactly one readable expanded check."""

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
    card_markup: list[str] = []
    for check in checks:
        summary = check.rationale
        if check.check_type == "EVIDENCE_COVERAGE":
            summary = f"已关联 {len(check.reference_ids)} 条真实市场评论，展开查看代表性原文。"
        card_markup.append(
            f'<article class="pressure-check-card{" is-active" if check.check_type == active_check_type else ""}">'
            f'<span>{_escape_html(status_labels.get(check.status, check.status))}</span>'
            f'<strong>{_escape_html(check.label)}</strong>'
            f'<p>{_escape_html(summary[:72])}{"…" if len(summary) > 72 else ""}</p></article>'
        )
    cards = "".join(card_markup)
    references = " · ".join(active.reference_ids) or "本项未记录额外引用"
    active_copy = active.rationale
    reference_markup = f'<small>{_escape_html(references)}</small>'
    if active.check_type == "EVIDENCE_COVERAGE":
        active_copy = f"共关联 {len(active.reference_ids)} 条真实市场评论；优先展示三条代表性原文，其余证据保留在审计索引中。"
        reference_markup = (
            '<details class="pressure-evidence-audit"><summary>'
            f'审计索引 · {len(active.reference_ids)} 条</summary>'
            f'<small>{_escape_html(references)}</small></details>'
        )
    detail = (
        f'<section class="pressure-check-grid">{cards}</section>'
        '<section class="pressure-check-detail">'
        f'<header><span>{_escape_html(status_labels.get(active.status, active.status))}</span>'
        f'<h3>{_escape_html(active.label)}</h3></header>'
        f'<p>{_escape_html(active_copy)}</p>'
        f'{reference_markup}</section>'
    )
    if active.check_type == "EVIDENCE_COVERAGE" and evidence_catalog is not None:
        detail += build_evidence_excerpt_cards_html(active.reference_ids, evidence_catalog)
    return detail


def build_evidence_excerpt_cards_html(
    reference_ids: tuple[str, ...], evidence_catalog: Any
) -> str:
    missing = [reference for reference in reference_ids if reference not in evidence_catalog]
    if missing:
        raise ValueError("证据原文映射缺失：" + "、".join(missing))
    selected: list[EvidenceExcerptView] = [
        evidence_catalog[reference] for reference in reference_ids[:3]
    ]
    cards = []
    for item in selected:
        excerpt = item.raw_content[:78] + ("…" if len(item.raw_content) > 78 else "")
        platform = item.source_platform or "用户评论"
        cards.append(
            '<article class="pressure-evidence-card">'
            f'<header><span>{_escape_html(platform[:1])}</span><b>{_escape_html(platform)}</b>'
            '<small>真实市场意见</small></header>'
            f'<blockquote>“{_escape_html(excerpt)}”</blockquote>'
            '<details><summary>完整原文</summary>'
            f'<p>{_escape_html(item.raw_content)}</p></details>'
            f'<footer class="pressure-evidence-id">{_escape_html(item.evidence_id)}</footer>'
            "</article>"
        )
    if not cards:
        return '<section class="pressure-evidence-gallery is-empty">本项未记录消费者评论</section>'
    remaining = max(0, len(reference_ids) - len(selected))
    tail = f"<p>另有 {remaining} 条证据保留在审计记录中</p>" if remaining else ""
    return '<section class="pressure-evidence-gallery">' + "".join(cards) + tail + "</section>"


def _render_dialogue(st: Any, run: PressureRunView, candidate_id: str) -> None:
    st.markdown(
        "<div class='pressure-dialogue-title'>阶段 3 · 双 Agent 互审</div>",
        unsafe_allow_html=True,
    )
    messages = dialogue_for_candidate(run, candidate_id)
    replay_key = f"pressure_replay_{candidate_id}"
    state = st.session_state.setdefault(
        replay_key,
        {"index": 0, "running": True, "completed": False},
    )
    if not isinstance(state, dict) or not 0 <= int(state.get("index", 0)) < len(messages):
        state = {"index": 0, "running": True, "completed": False}
        st.session_state[replay_key] = state

    state.setdefault("running", not bool(state.get("completed", False)))
    controls = st.columns(2)
    with controls[0]:
        toggle_label = "暂停" if state["running"] else "继续"
        if st.button(toggle_label, key=f"{replay_key}_toggle", use_container_width=True):
            state["running"] = not state["running"]
            st.rerun()
    with controls[1]:
        if st.button("重播", key=f"{replay_key}_replay", use_container_width=True):
            state.update(index=0, running=True, completed=False)
            st.rerun()

    visible = messages[: state["index"] + 1]
    from streamlit.components.v1 import html as components_html

    thinking_agent = None
    if state["running"] and not state["completed"] and state["index"] + 1 < len(messages):
        thinking_agent = messages[state["index"] + 1].agent
    components_html(
        build_dialogue_stage_html(visible, thinking_agent=thinking_agent),
        height=250,
        scrolling=False,
    )
    if state["running"] and not state["completed"]:
        time.sleep(5)
        advance_dialogue_replay(state, message_count=len(messages))
        if state["completed"]:
            state["running"] = False
        st.rerun()


def _render_holdout(st: Any, run: PressureRunView) -> None:
    st.markdown("## 阶段 4 · 统一复评与 HOLDOUT")
    st.warning(HOLDOUT_NOTE)
    st.caption("HOLDOUT：5 个候选 × 60 条留出证据；支持 / 挑战 / 中性 / 阻断")
    rows = []
    for item in holdout_distribution(run):
        support_width = item.support_count / item.total * 100
        challenge_width = item.challenge_count / item.total * 100
        rows.append(
            "<article class='pressure-holdout-row'>"
            f"<strong>{_escape_html(item.candidate_id)}</strong>"
            "<div class='pressure-holdout-bar'>"
            f"<i class='is-support' style='width:{support_width:.2f}%'></i>"
            f"<i class='is-challenge' style='width:{challenge_width:.2f}%'></i></div>"
            f"<span>支持 {item.support_count}</span><span>挑战 {item.challenge_count}</span>"
            f"<span>中性 {item.neutral_count}</span><span>阻断 {item.blocking_count}</span>"
            "</article>"
        )
    st.markdown("<section class='pressure-holdout-grid'>" + "".join(rows) + "</section>", unsafe_allow_html=True)


def _render_terminal(st: Any, run: PressureRunView) -> None:
    summary = terminal_summary(run)
    st.markdown("## 阶段 5 · 真人盲评 5→3")
    st.markdown(build_blind_review_html(run), unsafe_allow_html=True)
    revealed = bool(st.session_state.get("pressure_blind_revealed", False))
    if not revealed:
        if st.button("揭晓 5→3", key="pressure_blind_reveal", type="primary"):
            st.session_state["pressure_blind_revealed"] = True
            st.rerun()
        return
    selected_markup = "".join(
        f"<li><b>{_escape_html(card.display_title)}</b><span>{_escape_html(card.candidate_id)}</span></li>"
        for card in summary.selected
    )
    rejected_markup = "".join(
        f"<li><b>{_escape_html(card.display_title)}</b><span>{_escape_html(card.candidate_id)} · 历史保留</span></li>"
        for card in summary.not_selected
    )
    st.markdown(
        "<section class='pressure-terminal'>"
        f"<header><span>最终选择路径</span><h3>{_escape_html(summary.method)}</h3></header>"
        f"<div class='is-selected'><strong>三条入选</strong><ol>{selected_markup}</ol></div>"
        f"<div class='is-rejected'><strong>两条未入选</strong><ol>{rejected_markup}</ol></div>"
        "</section>",
        unsafe_allow_html=True,
    )

    confirmed = bool(st.session_state.get("pressure_team_confirmed", False))
    if st.button("团队确认 5→3", key="pressure_team_confirm"):
        st.session_state["pressure_team_confirmed"] = True
        complete_workspace(st.session_state, Workspace.PRESSURE_TEST)
        confirmed = True
    if confirmed:
        st.success("团队已确认；实时决策看板已解锁。")
        if st.button("进入实时决策看板", key="pressure_enter_realtime", type="primary"):
            activate_workspace(st.session_state, Workspace.REALTIME_DECISION)
            st.rerun()


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
        f'<div class="blind-review-item is-item-{index}">'
        "<span class='blind-review-avatar'>匿</span>"
        f"<p><b>匿名受访者</b>{_escape_html(item.text)}</p></div>"
        for index, item in enumerate(review.open_responses)
    )
    return (
        "<section class='blind-review-stage'>"
        "<header><span>真人盲评 · 冻结回放</span>"
        f"<strong>{review.survey_count} 份有效问卷</strong>"
        f"<strong>{review.open_response_count} 条开放回答</strong></header>"
        f"<div class='blind-review-metrics'>{metric_markup}</div>"
        f"<div class='blind-review-track' data-animation='blindReviewGlide'>{response_markup}</div>"
        "<footer>本批匿名评价已进入 · 内容来自冻结盲评问卷</footer></section>"
    )


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _dialogue_frame_css() -> str:
    return """
html,body{margin:0;background:#E8E1D8;color:#292521;font-family:"Microsoft YaHei","PingFang SC",sans-serif;}
.dialogue-stage{display:flex;flex-direction:column;gap:.7rem;height:248px;padding:.75rem;overflow-y:auto;box-sizing:border-box;scroll-behavior:smooth;border:1px solid #D8CCBE;background:#E8E1D8;}
.dialogue-stage.is-single{justify-content:center;}
.dialogue-bubble{display:flex;gap:.8rem;align-items:flex-start;width:min(82%,48rem);padding:.9rem 1rem;box-sizing:border-box;border-left:3px solid #8F2F4D;background:#FAF7F1;box-shadow:0 .45rem 1.2rem rgba(63,48,54,.05);animation:dialogueBubbleIn .4s ease-out both;}
.dialogue-bubble.review-avatar-bubble{align-self:flex-end;border-left-color:#365B4B;background:#F1F3EE;}
.dialogue-thinking{display:flex;align-items:center;gap:.7rem;width:min(42%,24rem);padding:.7rem .85rem;box-sizing:border-box;border:1px dashed #BDAFA3;background:rgba(255,253,249,.58);animation:dialogueBubbleIn .35s ease-out both;}
.dialogue-thinking.review-avatar-bubble{align-self:flex-end;border-color:#8CA093}.dialogue-thinking.decision-avatar-bubble{align-self:flex-start;border-color:#BC8798}
.dialogue-thinking>div{display:grid;grid-template-columns:auto 1fr;gap:.12rem .55rem;align-items:center}.dialogue-thinking strong{color:#5E5550;font-size:.7rem}.dialogue-thinking p{display:flex;align-items:center;gap:.35rem;margin:0;color:#8F2F4D;font-size:.68rem}.dialogue-thinking small{grid-column:1/-1;color:#8A8078;font-size:.6rem}
.thinking-dots{display:inline-flex;gap:.16rem}.thinking-dots i{width:.25rem;height:.25rem;border-radius:50%;background:#8F2F4D;animation:thinkingDot 1.15s ease-in-out infinite}.thinking-dots i:nth-child(2){animation-delay:.16s}.thinking-dots i:nth-child(3){animation-delay:.32s}
.dialogue-bubble__body{min-width:0;flex:1;}
.dialogue-bubble header{display:flex;justify-content:space-between;gap:1rem;color:#8F2F4D;font-size:.7rem;}
.dialogue-bubble.review-avatar-bubble header{color:#365B4B;}
.dialogue-bubble h3{margin:.35rem 0;color:#302B28;font-family:STZhongsong,"华文中宋",serif;font-size:1.02rem;font-weight:650;}
.dialogue-bubble p,.dialogue-bubble li{color:#655D57;font-size:.76rem;line-height:1.65;}
.dialogue-bubble details{margin-top:.55rem;border-top:1px solid #DED3C8;padding-top:.5rem;}
.dialogue-bubble summary{cursor:pointer;color:#746A62;font-size:.7rem;}
.dialogue-detail-copy{font-family:KaiTi,"楷体",serif;}
.dialogue-bubble ul{margin:.55rem 0 0;padding-left:1.1rem;}
.dialogue-diffs{display:grid;gap:.55rem;margin-top:.65rem;}
.dialogue-diff{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;padding:.65rem;border:1px solid #DED3C8;background:#FFFDF9;}
.dialogue-diff>strong,.dialogue-diff>em{grid-column:1/-1;}.dialogue-diff>strong{color:#8F2F4D;font-size:.72rem;}.dialogue-diff div{padding:.5rem;background:#F3EDE6;}.dialogue-diff div:nth-of-type(2){background:#EDF2ED;}.dialogue-diff small{color:#8A8078;font-size:.62rem;}.dialogue-diff p{margin:.2rem 0 0;}.dialogue-diff em{color:#81776F;font-size:.68rem;font-style:normal;}
.agent-avatar{display:inline-block;flex:0 0 2.25rem;width:2.25rem;height:2.25rem;position:relative;}.decision-avatar{border:2px solid #8F2F4D;border-radius:50%;background:conic-gradient(from 45deg,transparent 0 20%,#8F2F4D 20% 25%,transparent 25% 45%,#8F2F4D 45% 50%,transparent 50% 70%,#8F2F4D 70% 75%,transparent 75%);}.decision-avatar::after{content:"";position:absolute;inset:35%;border-radius:50%;background:#8F2F4D;}.review-avatar{border:2px solid #365B4B;border-radius:35% 35% 48% 48%;background:#365B4B;clip-path:polygon(50% 0,92% 20%,84% 72%,50% 100%,16% 72%,8% 20%);}.review-avatar::after{content:"";position:absolute;inset:35%;border:2px solid #E5C47D;transform:rotate(45deg);}
@keyframes dialogueBubbleIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes thinkingDot{0%,70%,100%{opacity:.25;transform:translateY(0)}35%{opacity:1;transform:translateY(-3px)}}
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
.blind-review-track { position:relative; height:7.2rem; overflow:hidden; border-top:1px solid #D8CCBE; border-bottom:1px solid #D8CCBE; }
.blind-review-item { position:absolute; left:100%; top:calc((var(--lane, 0)) * 2.25rem + .25rem); display:flex; align-items:center; gap:.5rem; width:max-content; max-width:42rem; animation:blindReviewGlide 28s linear 0s 1 forwards; }
.blind-review-item:nth-child(3n+1) { --lane:0; }.blind-review-item:nth-child(3n+2) { --lane:1; }.blind-review-item:nth-child(3n) { --lane:2; }
.blind-review-avatar { display:grid; place-items:center; flex:0 0 1.65rem; height:1.65rem; border-radius:50%; color:#FFF8EF; background:#8F2F4D; font-size:.65rem; }
.blind-review-item p { display:flex; gap:.55rem; margin:0; padding:.42rem .7rem; border:1px solid #D7C9BC; border-radius:1.2rem; background:#FFF9F3; color:#4F4741; font-size:.68rem; white-space:nowrap; }
.blind-review-item p b { color:#8F2F4D; }
.blind-review-track:hover .blind-review-item,.blind-review-item:hover { animation-play-state:paused; }
.blind-review-stage footer { margin-top:.45rem; color:#81776F; font-size:.63rem; text-align:right; }
@keyframes blindReviewGlide { from { transform:translateX(0); } to { transform:translateX(calc(-100vw - 100%)); } }
@media (max-width:900px) { .block-container,[data-testid="stMainBlockContainer"] { padding:.55rem .7rem 1.5rem; } .pressure-current,.pressure-terminal,.pressure-candidate-meta-grid,.pressure-evidence-gallery { grid-template-columns:1fr; } .pressure-current div { grid-column:1; grid-row:auto; min-width:0; } .pressure-check-grid { grid-template-columns:1fr 1fr; } .pressure-holdout-row { grid-template-columns:1fr 1fr; } .dialogue-bubble { width:100%; } .dialogue-diff { grid-template-columns:1fr; } .dialogue-diff > strong,.dialogue-diff > em { grid-column:auto; } }
""".strip()
    return f"{base_css}\n{_blind_review_schedule_css()}"


def _blind_review_schedule_css() -> str:
    """Keep the 51-item playback schedule outside sanitized Markdown markup."""

    rules: list[str] = []
    for index in range(51):
        duration = 25 + (index % 5) * 2
        group = index // 3
        lane_offset = (index % 3) * 0.8
        delay = (-8 + (index % 3) * 3) if group == 0 else (group - 1) * 4.2 + lane_offset + 2
        rules.append(
            f".blind-review-item.is-item-{index}{{animation-duration:{duration}s;"
            f"animation-delay:{delay:.1f}s}}"
        )
    return "".join(rules)


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
