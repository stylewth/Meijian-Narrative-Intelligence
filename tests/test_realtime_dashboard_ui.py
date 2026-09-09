from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from src.integrations.feishu.notification_store import NotificationStore
from src.services.demo_notifications import DemoNotificationNode, build_demo_notification_snapshots
from src.services.evolution_presentation import load_official_evolution_run
from src.services.preprocessing_demo_presentation import load_preprocessing_demo
from src.services.pressure_test_presentation import load_pressure_test_run
from src.ui.demo_notification_control import sync_evolution_notifications
from src.ui.realtime_decision_dashboard import (
    LATEST_SEGMENT_ANIMATION_CSS,
    LIVE_PAGE_CSS,
    PlaybackAction,
    PlaybackPhase,
    PlaybackState,
    apply_playback_action,
    build_full_width_detail_html,
    build_kpi_band_html,
    build_progress_strip_html,
    build_score_figure,
    build_ticker_html,
    render_realtime_decision_dashboard,
)


ROOT = Path(__file__).resolve().parents[1]
EVOLUTION_ROOT = (
    ROOT
    / "data/competition/evolution_runs/official-20260814-three-opportunity-evolution-001"
)
SCREENED_ROOT = ROOT / "data/competition/screened_v2"
VALIDATION_ROOT = (
    ROOT
    / "data/competition/five_candidate_validation/official-20260813-five-candidate-validation-002"
)


def test_kpi_band_reports_the_real_checkpoint_instead_of_fake_live_status() -> None:
    html = build_kpi_band_html(
        {
            "weighted_score": 61.7,
            "score_change": 0.0,
            "rank": 1,
            "checkpoint_label": "初始状态",
        }
    )

    assert "当前节点" in html
    assert "初始状态" in html
    assert "LIVE" not in html


def test_comment_batch_uses_five_live_bullets_and_keeps_static_records() -> None:
    batch = SimpleNamespace(
        batch_id="batch-1",
        records=tuple(
            SimpleNamespace(
                raw_id=f"record-{index}",
                raw_content=f"评论内容 {index}",
                source_platform="小红书",
                source_ref=f"source-{index}",
            )
            for index in range(5)
        ),
    )

    html = build_ticker_html(batch)

    assert html.count('class="mj-comment-bullet ') == 5
    assert html.count('class="mj-comment-lane"') == 3
    assert html.count('class="mj-evidence-ticker__record"') == 5
    assert "data-duration-seconds" in html
    assert "新增评论实时进入" in html
    assert "@keyframes mjCommentGlide" in LIVE_PAGE_CSS


def test_dashboard_uses_real_candidate_buttons_and_a_two_to_one_main_stage() -> None:
    source = inspect.getsource(render_realtime_decision_dashboard)

    assert 'candidate_columns = st.columns((1, 1, 1), gap="small")' in source
    assert 'stage_columns = st.columns((2.15, 1), gap="large")' in source
    assert "build_opportunity_selector_html(run, selected_candidate_id)" not in source
    assert 'control_columns = st.columns((1, 1, 1, 5), gap="small")' in source
    assert source.rindex("control_columns =") < source.rindex(
        "build_progress_strip_html"
    )


def test_narrow_dashboard_stacks_the_real_streamlit_column_elements() -> None:
    assert ':has(.mj-live-panel)>[data-testid="column"]' in LIVE_PAGE_CSS


def test_five_dimensions_share_one_scale_instead_of_five_cards() -> None:
    detail = {
        "checkpoint_label": "阶段 2",
        "dimensions": [
            {"name": f"维度 {index}", "score": 50 + index, "reason": f"理由 {index}"}
            for index in range(5)
        ],
        "risks": [],
        "scenes": [],
    }

    html = build_full_width_detail_html(detail)

    assert html.count('class="mj-dimension-row"') == 5
    assert "mj-dimension-card" not in html


def test_score_figure_is_prominent_without_pushing_controls_below_the_fold() -> None:
    run = load_official_evolution_run(EVOLUTION_ROOT)
    figure = build_score_figure(
        run,
        run.selected_candidate_ids[0],
        visible_checkpoint_index=2,
    )

    assert 300 <= figure.layout.height <= 340
    latest_traces = [trace for trace in figure.data if trace.meta == "latest-segment"]
    assert len(latest_traces) == 3
    assert all("text" in trace.mode for trace in latest_traces)
    assert all(trace.text[-1] for trace in latest_traces)
    assert all("left" not in trace.textposition for trace in figure.data)
    assert [trace.textposition for trace in figure.data[:3]] == [
        "top right",
        "middle right",
        "bottom right",
    ]
    assert list(figure.layout.xaxis.ticktext) == [
        "初始状态",
        "阶段 1",
        "阶段 2",
        "下一变化待定",
    ]
    assert "阶段 5" not in figure.layout.xaxis.ticktext


def test_latest_score_segments_use_a_stable_forward_trace_selector() -> None:
    assert ".scatterlayer > .trace:nth-of-type(n+4) path.js-line" in (
        LATEST_SEGMENT_ANIMATION_CSS
    )
    assert "nth-last-child" not in LATEST_SEGMENT_ANIMATION_CSS
    assert "animation:mjLineGrow" in LATEST_SEGMENT_ANIMATION_CSS
    assert "2.6s" in LATEST_SEGMENT_ANIMATION_CSS


def test_blind_confirmation_keeps_its_content_height() -> None:
    assert ".mj-comment-idle,.mj-blind-confirmation{height:3.1rem" not in LIVE_PAGE_CSS
    assert ".mj-blind-confirmation{height:auto;min-height:8.5rem" in LIVE_PAGE_CSS


def test_progress_strip_reveals_history_without_exposing_a_fixed_endpoint() -> None:
    nodes = [{"label": "初始状态"}] + [
        {"label": f"阶段 {index}"} for index in range(1, 6)
    ]

    initial_html = build_progress_strip_html(
        nodes,
        PlaybackState(0, PlaybackPhase.SCORE, False, False),
    )
    receiving_html = build_progress_strip_html(
        nodes,
        PlaybackState(2, PlaybackPhase.COMMENTS, False, False),
    )

    assert initial_html.count('class="mj-progress-node') == 2
    assert "下一变化待定 · 可随时收敛" in initial_html
    assert "阶段 1" not in initial_html
    assert "阶段 5" not in initial_html
    assert receiving_html.count('class="mj-progress-node') == 3
    assert "阶段 2" in receiving_html
    assert "阶段 3" not in receiving_html
    assert "is-receiving" in receiving_html


def test_first_realtime_delta_enqueues_node_4_via_milestone_callback(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path / "notifications.sqlite3")
    session = store.create_session(
        "official-20260814-three-opportunity-evolution-001",
        created_by="streamlit-local",
    )
    snapshots = build_demo_notification_snapshots(
        load_preprocessing_demo(SCREENED_ROOT, VALIDATION_ROOT),
        load_pressure_test_run(
            VALIDATION_ROOT,
            EVOLUTION_ROOT / "blind/selection_confirmation.json",
        ),
        load_official_evolution_run(EVOLUTION_ROOT),
    )
    state = {
        "evolution_checkpoint_index": 2,
        "evolution_phase": PlaybackPhase.SCORE.value,
        "evolution_playing": False,
        "evolution_show_finale": False,
        "evolution_finale_act": 1,
        "evolution_finale_auto_reveal": False,
    }

    apply_playback_action(
        state,
        PlaybackAction.NEXT_CHANGE,
        on_milestone=lambda: sync_evolution_notifications(
            state,
            store,
            snapshots,
        ),
    )

    jobs = store.jobs_for_session(session.session_id)
    assert [job.node_key for job in jobs] == [
        DemoNotificationNode.DELTA_01_COMPLETE.value
    ]
