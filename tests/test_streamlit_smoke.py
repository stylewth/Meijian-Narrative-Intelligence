from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.integrations.feishu.notification_store import NotificationStore
from src.services.demo_notifications import DemoNotificationNode, build_demo_notification_snapshots
from src.services.evolution_presentation import load_official_evolution_run
from src.services.preprocessing_demo_presentation import load_preprocessing_demo
from src.services.pressure_test_presentation import load_pressure_test_run
from src.ui import (
    preprocessing_workspace,
    pressure_test_workspace,
    realtime_decision_dashboard,
)
from src.ui.preprocessing_workspace import (
    ProcessingStageView,
    apply_official_replay_action,
    _first_count,
    _official_replay_stage_flow,
    build_case_data_intake_html,
    build_custom_preprocessing_status_html,
    build_foundation_opportunities_html,
    build_official_replay_stages,
    build_official_stage_track_html,
    build_stage_detail_html,
)
from src.ui.demo_notification_control import sync_preprocessing_notification
from src.ui.ui_polish import build_polish_css
from src.ui.system_gateway import (
    SYSTEM_GATEWAY_CSS,
    _build_gateway_art_html,
    build_system_header_html,
    render_system_gateway,
)


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "app.py")
APP_SOURCE = (ROOT / "app.py").read_text(encoding="utf-8")
SCREENED_ROOT = ROOT / "data/competition/screened_v2"
VALIDATION_ROOT = (
    ROOT
    / "data/competition/five_candidate_validation/official-20260813-five-candidate-validation-002"
)
EVOLUTION_ROOT = (
    ROOT
    / "data/competition/evolution_runs/official-20260814-three-opportunity-evolution-001"
)


@pytest.fixture(autouse=True)
def _public_no_key_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """冒烟测试验证的是公开无密钥行为，必须与本机 .env 隔离。"""

    for key in (
        "LLM_API_KEY",
        "LLM_MODEL",
        "FEISHU_APP_ID",
        "FEISHU_APP_SECRET",
        "FEISHU_DEMO_CHAT_ID",
        "FEISHU_BOT_OPEN_ID",
        "FEISHU_OPERATOR_OPEN_IDS",
        "FEISHU_NOTIFICATION_DB",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *args, **kwargs: {})


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def visible_text(app: AppTest) -> str:
    groups = (app.markdown, app.caption, app.info, app.warning, app.success, app.title)
    return "\n".join(
        str(item.value)
        for group in groups
        for item in group
        if getattr(item, "value", None)
    )


def test_public_app_starts_without_secrets_and_enters_the_official_case() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=10)

    assert not app.exception
    assert not app.error
    assert "梅见品牌叙事智能决策系统" in visible_text(app)
    assert button(app, "进入 · 自定义使用")
    button(app, "进入 · 梅见案例展示").click().run(timeout=10)

    assert not app.exception
    assert "只读官方运行回放" in visible_text(app)
    assert not any(item.label == "运行入口" for item in app.selectbox)


def test_public_custom_entry_exposes_local_only_credentials_state() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=10)
    button(app, "进入 · 自定义使用").click().run(timeout=10)

    assert not app.exception
    assert "在线 AI 仅在本地配置后可用" in visible_text(app)
    local_feishu = button(app, "开始新的案例会话")
    assert local_feishu.disabled is True


def test_official_pressure_and_evolution_workspaces_render_from_frozen_data() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=10)
    app.session_state.system_entry = "梅见案例展示"
    app.session_state.active_workspace = "叙事压力测试"
    app.session_state.completed_workspaces = ["数据预处理"]
    app.run(timeout=10)

    assert not app.exception
    assert "叙事压力测试" in visible_text(app)

    app.session_state.active_workspace = "实时决策看板"
    app.session_state.completed_workspaces = ["数据预处理", "叙事压力测试"]
    app.run(timeout=10)

    assert not app.exception
    assert "实时叙事演化看板" in visible_text(app)


def test_preprocessing_next_step_advances_inside_fragment_without_exception() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=10)
    app.session_state.system_entry = "梅见案例展示"
    app.session_state.active_workspace = "数据预处理"
    app.run(timeout=10)

    button(app, "下一步").click().run(timeout=10)

    assert not app.exception
    assert app.session_state["preprocessing_replay_step"] == 1
    assert app.session_state["preprocessing_replay_completed_steps"] == [0]


def test_pressure_next_stage_advances_inside_fragment_without_exception() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=10)
    app.session_state.system_entry = "梅见案例展示"
    app.session_state.active_workspace = "叙事压力测试"
    app.session_state.completed_workspaces = ["数据预处理"]
    app.run(timeout=10)

    button(app, "下一阶段").click().run(timeout=10)

    assert not app.exception
    assert app.session_state["pressure_stage"] == "基础压力检查"


def test_preprocessing_final_step_enqueues_node_1_via_milestone_callback(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path / "notifications.sqlite3")
    session = store.create_session(
        "official-20260814-three-opportunity-evolution-001",
        created_by="streamlit-local",
    )
    snapshots = build_demo_notification_snapshots(
        load_preprocessing_demo(SCREENED_ROOT, VALIDATION_ROOT),
        load_pressure_test_run(VALIDATION_ROOT, EVOLUTION_ROOT / "blind/selection_confirmation.json"),
        load_official_evolution_run(EVOLUTION_ROOT),
    )
    state = {
        "preprocessing_replay_step": 4,
        "preprocessing_replay_completed_steps": [0, 1, 2, 3],
        "completed_workspaces": [],
    }

    apply_official_replay_action(
        state,
        "skip",
        on_milestone=lambda: sync_preprocessing_notification(state, store, snapshots),
    )

    jobs = store.jobs_for_session(session.session_id)
    assert [job.node_key for job in jobs] == [DemoNotificationNode.PREPROCESSING_COMPLETE.value]


def test_preprocessing_html_uses_the_five_stage_processing_language() -> None:
    """冻结产物必须投影为来源条、轨道及五种不同的数据关系图。"""

    stages = build_official_replay_stages(SCREENED_ROOT)
    demo = load_preprocessing_demo(SCREENED_ROOT, VALIDATION_ROOT)
    source_html = build_case_data_intake_html(stages)
    track_html = build_official_stage_track_html(
        stages,
        current_index=4,
        completed_keys={stage.key for stage in stages},
    )
    details = {stage.key: build_stage_detail_html(stage) for stage in demo.stages}

    assert 'class="mj-preprocess-sourcebar"' in source_html
    assert "484 条原始语料" in source_html
    assert 'class="mj-stage-track"' in track_html
    assert track_html.count('class="mj-stage-node') == 5
    assert track_html.count("mj-stage-seal") == 1

    assert 'class="mj-validate-flow"' in details["validate"]
    assert "输入" in details["validate"] and "检查" in details["validate"] and "输出" in details["validate"]
    assert 'class="mj-clean-flow"' in details["clean"]
    assert details["clean"].count('data-flow="out"') == 3
    assert all(value in details["clean"] for value in ("484", "419", "54", "11"))
    assert details["split"].count('class="mj-split-lane"') == 4
    assert 'class="mj-annotation-transform"' in details["annotate"]
    assert "字段结构" in details["annotate"]
    assert 'class="mj-freeze-convergence"' in details["freeze"]
    assert details["freeze"].count('class="mj-package-line"') == 3


def test_shared_feishu_control_uses_a_dismissible_popover() -> None:
    """飞书助手应使用原生可再次点击、点外与 Esc 关闭的 popover。"""

    polish_css = build_polish_css()

    assert 'with st.popover("飞书机器人助手"' in APP_SOURCE
    assert 'with st.expander("飞书机器人助手"' not in APP_SOURCE
    assert '[data-testid="stPopover"]' in polish_css
    assert '[data-testid="stPopoverBody"]' in polish_css
    assert '.mj-system-header) [data-testid="stExpander"]' not in polish_css


def test_gateway_home_and_header_fit_the_safe_visual_contract() -> None:
    """首页主视觉与工作台顶栏必须在安全区内呼吸，不挤压或裁切。"""

    art_html = _build_gateway_art_html()
    header_html = build_system_header_html(
        entry="梅见案例展示",
        data_status="官方回放",
        feishu_status="未连接",
    )
    polish_css = build_polish_css()

    assert 'class="mj-gateway-art__bloom"' in art_html
    assert 'class="mj-gateway-art__arcs"' in art_html
    assert 'class="mj-gateway-art__petals"' in art_html
    assert 'class="mj-gateway-art__core"' in art_html
    assert 'class="mj-gateway-art__halo"' in art_html
    assert 'viewBox="0 0 560 580"' not in art_html
    assert 'viewBox="0 0 600 560"' in art_html
    assert '.mj-gateway-art svg{inset:0;width:100%;height:100%' in SYSTEM_GATEWAY_CSS
    assert '.mj-gateway-art{position:relative;height:33rem' in SYSTEM_GATEWAY_CSS
    assert '.mj-gateway-flow{display:flex;align-items:center;gap:1.25rem;margin-top:1rem' in SYSTEM_GATEWAY_CSS
    assert 'mj-system-header__brand{color:#8F2F4D;padding:.35rem .65rem' in SYSTEM_GATEWAY_CSS
    assert 'mj-system-rail::after' in polish_css
    assert 'linear-gradient(180deg,rgba(201,168,106,.55),rgba(201,168,106,.12))' in polish_css

    assert 'max-width:90rem' in polish_css
    assert 'padding-left:3vw;padding-right:3vw' in polish_css
    assert 'font-size:clamp(2.6rem,4.2vw,4rem)' in SYSTEM_GATEWAY_CSS
    assert '.mj-gateway-art{position:relative;height:33rem' in SYSTEM_GATEWAY_CSS


def test_finale_header_polish_is_scoped_to_the_finale_shell() -> None:
    """终幕顶栏的酒红/金线样式不得泄漏到普通工作台。"""

    polish_css = build_polish_css()

    assert 'body:has(.mj-finale-shell) .mj-system-header' in polish_css
    assert 'body:has(.mj-finale-shell) .mj-system-header__brand' in polish_css
    assert 'body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header)' in polish_css
    finale_header_scope = 'body:has(.mj-finale-shell) [data-testid="stHorizontalBlock"]:has(.mj-system-header)'
    assert f'{finale_header_scope} [data-testid="stPopover"] button[data-testid="baseButton-secondary"]' in polish_css
    assert 'stPopoverButton' not in polish_css
    assert f'{finale_header_scope} [data-testid="stButton"]>button' in polish_css
    assert f'{finale_header_scope} [data-testid="stPopover"]>button' not in polish_css
    assert 'body:has(.mj-finale-shell) [data-testid="stButton"]>button{' not in polish_css
    assert 'background:rgba(107,48,66,.72)!important' in polish_css
    assert 'border-color:rgba(217,166,168,.7)!important' in polish_css
    assert 'color:#F7D9C9!important' in polish_css
    assert f'{finale_header_scope} [data-testid="stPopover"] button[data-testid="baseButton-secondary"] p{{color:inherit!important' in polish_css
    assert 'body:has(.mj-finale-shell) [data-testid="stPopoverBody"]' in polish_css
    assert 'background:rgba(59,23,36,.98)!important' in polish_css
    assert 'border-color:#C9A86A!important' in polish_css
    assert 'background:rgba(124,58,77,.84)!important' in polish_css
    assert 'border-color:#D9A6A8!important' in polish_css
    assert 'color:#FFF1E9!important' in polish_css
    assert 'linear-gradient(110deg,#3B1724,#6B3042 58%,#35131F)' in polish_css
    assert '#D9A6A8' in polish_css


def test_gateway_main_columns_keep_the_visual_center_compact() -> None:
    """首页左右主内容均衡分栏，避免主视觉与入口之间出现大空档。"""

    assert "st.columns([50, 50])" in inspect.getsource(render_system_gateway)


def test_preprocessing_track_separates_rail_from_labels_and_reflects_running_state() -> None:
    """粗轨道独立于文字层，且运行态必须来自真实回放状态。"""

    stages = build_official_replay_stages(SCREENED_ROOT)
    track_html = build_official_stage_track_html(
        stages,
        current_index=2,
        completed_keys={stages[0].key, stages[1].key},
        running=True,
    )

    assert 'class="mj-stage-track is-running"' in track_html
    assert track_html.count('class="mj-stage-connector') == 4
    assert track_html.index('class="mj-stage-rail"') < track_html.index("<ol>")
    assert 'class="mj-stage-node is-current is-running"' in track_html
    assert "处理中" in track_html


def test_preprocessing_track_stacks_labels_below_the_rail() -> None:
    """轨道与节点文字必须分行，避免 rail 穿过阶段名称。"""

    polish_css = build_polish_css()

    assert "grid-template-columns:1fr" in polish_css
    assert "grid-template-rows:1.2rem auto auto" in polish_css
    assert "justify-items:center" in polish_css
    assert "row-gap:" in polish_css
    assert ".mj-stage-dot{position:relative;z-index:2;grid-row:1" in polish_css
    assert ".mj-stage-node strong{grid-row:2" in polish_css
    assert ".mj-stage-node small{grid-row:3" in polish_css
    assert ".mj-stage-node:not(:last-child)::after" not in polish_css


def test_preprocessing_motion_is_stage_specific_and_reduced_motion_safe() -> None:
    """五个阶段分别有克制动效，并完整服从减少动态效果偏好。"""

    polish_css = build_polish_css()

    for animation_name in (
        "mjValidateCheck",
        "mjCleanBranch",
        "mjSplitLane",
        "mjAnnotateField",
        "mjFreezeMerge",
    ):
        assert f"@keyframes {animation_name}" in polish_css
    assert "@keyframes mjStageTrackFill" in polish_css
    assert "@keyframes mjStageTrackFlow" in polish_css
    assert '.mj-stage-connector{' in polish_css
    assert "height:5px" in polish_css
    assert "prefers-reduced-motion: reduce" in polish_css
    assert ".mj-stage-track *" in polish_css
    assert ".mj-stage-workbench *" in polish_css


def test_preprocessing_freeze_opportunities_and_controls_are_compact_rows() -> None:
    """冻结结果应为机会名录，原生回放按钮保持一组紧凑控制。"""

    demo = load_preprocessing_demo(SCREENED_ROOT, VALIDATION_ROOT)
    opportunities_html = build_foundation_opportunities_html(demo.opportunities)
    polish_css = build_polish_css()

    assert 'class="mj-opportunity-ledger"' in opportunities_html
    assert opportunities_html.count('class="mj-opportunity-row"') == 5
    assert "原始分" in opportunities_html and "核心洞察" in opportunities_html
    assert ".mj-replay-controls{" in polish_css
    assert "st.columns((1, 1, 1, 1, 6))" in inspect.getsource(_official_replay_stage_flow)
    assert 'st-key-official_replay_"]) {flex-wrap:wrap!important' in polish_css


def test_preprocessing_replay_keeps_one_reset_and_uses_next_step_language() -> None:
    """重播是唯一重置入口，主流程不再显示审计干扰块。"""

    source = inspect.getsource(_official_replay_stage_flow)

    assert 'st.button("下一步", key="official_replay_skip"' in source
    assert "下一变化" not in source
    assert 'st.button("重播", key="official_replay_replay"' in source
    assert "高级操作" not in source
    assert "技术追溯" not in source
    assert "清除回放进度" not in source
    assert 'running=bool(state.get("preprocessing_replay_running", False))' in source


def test_preprocessing_count_parser_rejects_non_contract_text() -> None:
    """清洗流量只能从既定的原始语料摘要格式读取数量。"""

    assert _first_count("484 条结构化原始语料") == 484
    with pytest.raises(ValueError, match="原始语料数量格式无效"):
        _first_count("语料共 484 条")
    with pytest.raises(ValueError, match="原始语料数量格式无效"):
        _first_count("gpt-5.6-luna")


def test_custom_preprocessing_status_reuses_the_five_stage_track() -> None:
    """自定义数据在未运行模型时也要显示真实的当前阻断状态。"""

    stage = ProcessingStageView(
        key="annotate",
        status="waiting",
        ready=False,
        can_complete=False,
        label="标注",
        reason="等待本地模型在线标注",
    )

    html = build_custom_preprocessing_status_html(
        stage,
        entry="Excel/CSV",
        source_name="custom-comments.xlsx",
        record_count=12,
    )

    assert 'class="mj-preprocess-sourcebar mj-preprocess-sourcebar--custom"' in html
    assert html.count('class="mj-stage-node') == 5
    assert 'data-stage-key="annotate"' in html
    assert "等待本地模型在线标注" in html


def test_all_stage_fragments_mount_scroll_continuity() -> None:
    """三个会触发局部重跑的舞台都必须挂载滚动连续性桥接器。"""

    expected_tokens = {
        preprocessing_workspace: 'render_scroll_continuity(f"preprocessing:{current}")',
        pressure_test_workspace: 'render_scroll_continuity(f"pressure:{current_stage}")',
        realtime_decision_dashboard: 'render_scroll_continuity(\n        f"realtime:{state.checkpoint_index}:{state.phase.value}"\n    )',
    }

    for module, call in expected_tokens.items():
        source = inspect.getsource(module)
        assert call in source
        assert "render_scroll_continuity()" not in source


def test_scroll_continuity_mounts_after_each_fragment_content() -> None:
    """恢复桥必须等待当前 fragment 的完整内容进入 DOM 后再挂载。"""

    preprocessing_source = inspect.getsource(
        preprocessing_workspace._official_replay_stage_flow
    )
    pressure_source = inspect.getsource(pressure_test_workspace._render_pressure_flow)
    realtime_source = inspect.getsource(
        realtime_decision_dashboard.render_realtime_decision_dashboard
    )

    assert preprocessing_source.rindex("build_stage_detail_html") < preprocessing_source.rindex(
        "render_scroll_continuity"
    )
    assert pressure_source.rindex("_render_terminal") < pressure_source.rindex(
        "render_scroll_continuity"
    )
    assert realtime_source.rindex("build_full_width_detail_html") < realtime_source.rindex(
        "render_scroll_continuity"
    )
