from __future__ import annotations

import inspect
from pathlib import Path

from src.integrations.feishu.notification_store import NotificationStore
from src.services.demo_notifications import DemoNotificationNode, build_demo_notification_snapshots
from src.services.evolution_presentation import load_official_evolution_run
from src.services.preprocessing_demo_presentation import load_preprocessing_demo
from src.services.pressure_test_presentation import load_pressure_test_run
from src.ui.demo_notification_control import sync_pressure_notifications
from src.ui.pressure_test_workspace import (
    _pressure_test_css,
    build_blind_review_html,
    build_candidate_cards,
    build_dialogue_stage_html,
    build_dialogue_message_html,
    build_dialogue_revision_html,
    build_original_score_dotplot_html,
    build_pressure_checks_html,
    build_terminal_result_html,
    terminal_summary,
)


ROOT = Path(__file__).resolve().parents[1]
SCREENED_ROOT = ROOT / "data/competition/screened_v2"
VALIDATION_ROOT = (
    ROOT
    / "data/competition/five_candidate_validation/official-20260813-five-candidate-validation-002"
)
EVOLUTION_ROOT = (
    ROOT
    / "data/competition/evolution_runs/official-20260814-three-opportunity-evolution-001"
)
SELECTION_PATH = EVOLUTION_ROOT / "blind/selection_confirmation.json"


def _run():
    return load_pressure_test_run(VALIDATION_ROOT, SELECTION_PATH)


class _TerminalStreamlitStub:
    def __init__(self) -> None:
        self.session_state = {
            "pressure_blind_revealed": True,
            "completed_workspaces": ["数据预处理"],
            "active_workspace": "叙事压力测试",
        }
        self.button_calls: list[str] = []
        self.rerun_calls: list[tuple[object, ...]] = []

    def markdown(self, *_args: object, **_kwargs: object) -> None:
        return None

    def button(self, label: str, **_kwargs: object) -> bool:
        self.button_calls.append(label)
        return label == "团队确认 5→3"

    def success(self, *_args: object, **_kwargs: object) -> None:
        return None

    def rerun(self, *args: object, **_kwargs: object) -> None:
        self.rerun_calls.append(args)


def test_pressure_workspace_uses_real_five_row_candidate_navigation_and_score_dotplot() -> None:
    source = inspect.getsource(__import__("src.ui.pressure_test_workspace", fromlist=["*"]))
    css = _pressure_test_css()

    assert "st.columns((1.1, 4))" in source
    assert "pressure_candidate_button_{card.candidate_id}" in source
    assert "pressure-score-dotplot" in source
    assert ".pressure-candidate-nav" in css
    assert ".pressure-score-dot" in css
    assert "grid-template-columns:1fr" in css
    assert tuple(card.display_title for card in build_candidate_cards(_run())) == (
        "兑饮比例卡", "火锅局自主饮酒", "双容量双剧本", "口味图鉴", "梅见溯源记"
    )
    assert 'f"<h2>{_escape_html(card.title)}</h2>"' in source
    assert build_original_score_dotplot_html(build_candidate_cards(_run()), _run().candidates[0].candidate_id).count("<li class='pressure-score-dot") == 5


def test_pressure_workspace_renders_five_check_rows_and_one_evidence_reading_axis() -> None:
    run = _run()
    html = build_pressure_checks_html(
        run.candidates[0].checks,
        active_check_type=run.candidates[0].checks[0].check_type,
        evidence_catalog=run.evidence_catalog,
    )

    source = inspect.getsource(__import__("src.ui.pressure_test_workspace", fromlist=["*"]))
    assert "horizontal=False" in source and "len(checks) != 5" in source
    assert "pressure-evidence-reading" in html
    assert "原文摘录" in html and "检查说明" in html and "当前判断" in html
    assert "pressure-evidence-card" not in html


def test_holdout_uses_human_titles_and_conserves_three_category_widths() -> None:
    source = inspect.getsource(__import__("src.ui.pressure_test_workspace", fromlist=["*"]))

    assert "support_width + challenge_width + neutral_width" in source
    assert "is-neutral" in source
    assert "blocking_count" in source
    assert "titles[item.candidate_id]" in source
    assert "candidate_id)}</strong>" not in source


def test_blind_review_keeps_all_51_quotes_in_a_static_reading_panel() -> None:
    html = build_blind_review_html(_run())
    source = inspect.getsource(__import__("src.ui.pressure_test_workspace", fromlist=["*"]))

    assert html.count("<li class='blind-review-quote'>") == 51
    assert "问卷选项偏好" in html
    assert "多选口径" in html and "56" in html
    assert "blindReviewGlide" not in html
    assert "blindReviewGlide" not in source
    assert "_blind_review_schedule_css" not in source


def test_dialogue_frame_keeps_controls_and_uses_65_35_review_revision_split() -> None:
    run = _run()
    module = __import__("src.ui.pressure_test_workspace", fromlist=["dialogue_for_candidate"])
    html = build_dialogue_stage_html(module.dialogue_for_candidate(run, run.candidates[0].candidate_id))

    assert "grid-template-columns:65fr 35fr" in html
    assert "dialogue-review-axis" in html and "dialogue-revision-panel" in html
    assert "var INTERVAL = 4200" in html
    assert 'id=\'mj-d-toggle\'' in html and 'id=\'mj-d-replay\'' in html
    assert "prefers-reduced-motion: reduce" in html


def test_pressure_stage_hides_raw_evidence_ids_from_reading_and_revision_surfaces() -> None:
    run = _run()
    raw_ids = ("MJ-RAW-0006", "XHS-SOC-0019", "XHS-SOC-0185")
    checks_html = tuple(
        build_pressure_checks_html(
            candidate.checks,
            active_check_type=check.check_type,
            evidence_catalog=run.evidence_catalog,
        )
        for candidate in run.candidates
        for check in candidate.checks
    )
    dialogue_messages = tuple(
        message
        for candidate in run.candidates
        for message in __import__("src.ui.pressure_test_workspace", fromlist=["dialogue_for_candidate"]).dialogue_for_candidate(run, candidate.candidate_id)
    )
    dialogue_html = tuple(build_dialogue_message_html(message) for message in dialogue_messages)
    revision_html = tuple(
        build_dialogue_revision_html(message, index)
        for index, message in enumerate(dialogue_messages)
    )

    audit_html = ""
    for html in (*checks_html, *dialogue_html, *revision_html):
        assert "pressure-audit-detail" in html
        visible, audit = html.split("pressure-audit-detail", 1)
        assert all(raw_id not in visible for raw_id in raw_ids)
        audit_html += audit
    assert all(raw_id in audit_html for raw_id in raw_ids)


def test_terminal_connects_five_to_three_and_preserves_action_keys() -> None:
    module = __import__("src.ui.pressure_test_workspace", fromlist=["*"])
    source = inspect.getsource(module)
    terminal_source = inspect.getsource(module._render_terminal)
    revision_source = inspect.getsource(module.build_dialogue_revision_html)
    html = build_terminal_result_html(terminal_summary(_run()))
    css = _pressure_test_css()

    assert "pressure-terminal-map" in source
    assert "pressure-terminal-selected" in source
    assert "pressure-terminal-register" in source
    assert 'key="pressure_blind_reveal"' in source
    assert 'key="pressure_team_confirm"' in terminal_source
    assert 'key="pressure_enter_realtime"' not in terminal_source
    assert 'key="pressure_team_confirm"' not in revision_source
    assert 'key="pressure_enter_realtime"' not in revision_source
    assert html.count("pressure-terminal-source ") == 5
    assert html.count("<li><b>入选方向") == 3
    assert html.split("pressure-terminal-register", 1)[1].count("<li>") == 2
    assert all(card.candidate_id not in html for card in build_candidate_cards(_run()))
    assert ".pressure-terminal-relationship::after" in css


def test_team_confirmation_completes_pressure_test_and_activates_realtime_in_one_click() -> None:
    module = __import__("src.ui.pressure_test_workspace", fromlist=["*"])
    stub = _TerminalStreamlitStub()

    module._render_terminal(stub, _run())

    assert stub.session_state["pressure_team_confirmed"] is True
    assert stub.session_state["completed_workspaces"] == ["数据预处理", "叙事压力测试"]
    assert stub.session_state["active_workspace"] == "实时决策看板"
    assert stub.rerun_calls == [()]
    assert "进入实时决策看板" not in stub.button_calls


def test_team_confirmation_enqueues_node_3_via_milestone_callback(tmp_path: Path) -> None:
    module = __import__("src.ui.pressure_test_workspace", fromlist=["*"])
    store = NotificationStore(tmp_path / "notifications.sqlite3")
    session = store.create_session(
        "official-20260814-three-opportunity-evolution-001",
        created_by="streamlit-local",
    )
    snapshots = build_demo_notification_snapshots(
        load_preprocessing_demo(SCREENED_ROOT, VALIDATION_ROOT),
        _run(),
        load_official_evolution_run(EVOLUTION_ROOT),
    )
    stub = _TerminalStreamlitStub()

    module._render_terminal(
        stub,
        _run(),
        on_milestone=lambda: sync_pressure_notifications(
            stub.session_state,
            store,
            snapshots,
        ),
    )

    jobs = store.jobs_for_session(session.session_id)
    assert [job.node_key for job in jobs] == [
        DemoNotificationNode.BLIND_SELECTION_COMPLETE.value
    ]
