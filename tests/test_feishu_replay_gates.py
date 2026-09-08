"""冻结回放决策门链路测试：时间线审计、门执行、写回投影与配置合同。"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.replay_gate_timeline import (  # noqa: E402
    REPLAY_SOURCE_MARK,
    ReplayTimelineError,
    apply_action,
    initial_replay_state,
    load_replay_timeline,
    read_replay_state,
    run_log_summary,
    save_replay_state,
)

RUN_ID = "replay-test-001"
NOW = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)
OPERATOR = "ou_operator_1"


@pytest.fixture()
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "evolution_runs" / RUN_ID
    (root / "results").mkdir(parents=True)
    (root / "blind").mkdir(parents=True)
    validation = tmp_path / "five_candidate_validation" / "fc-001"
    (validation / "holdout").mkdir(parents=True)
    candidates = [
        ("EC-01", 3),
        ("EC-02", 2),
        ("EC-03", 1),
    ]
    (validation / "holdout" / "result.json").write_text(
        json.dumps(
            {
                "candidate_ids": [item[0] for item in candidates],
                "holdout_evidence_ids": [f"ev-{index}" for index in range(4)],
                "impacts": [
                    {"candidate_id": "EC-01", "evidence_id": "ev-0", "impact": "SUPPORT", "is_blocking": False, "rationale": "ok"},
                    {"candidate_id": "EC-01", "evidence_id": "ev-1", "impact": "CHALLENGE", "is_blocking": False, "rationale": "ok"},
                    {"candidate_id": "EC-02", "evidence_id": "ev-2", "impact": "NEUTRAL", "is_blocking": False, "rationale": "ok"},
                    {"candidate_id": "EC-03", "evidence_id": "ev-3", "impact": "SUPPORT", "is_blocking": True, "rationale": "ok"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "status": "COMPLETE",
                "completed_stages": [f"checkpoint-{index:02d}" for index in range(5)]
                + ["final-synthesis"],
            }
        ),
        encoding="utf-8",
    )
    selected = ["EC-01", "EC-02", "EC-03"]
    (root / "blind" / "selection_confirmation.json").write_text(
        json.dumps({"run_id": RUN_ID, "selected_candidate_ids": selected, "selections": []}),
        encoding="utf-8",
    )
    (root / "blind" / "blind_summary.json").write_text(
        json.dumps({"run_id": RUN_ID, "sample_counts": {"q1_8": 56, "q9_12": 51}}),
        encoding="utf-8",
    )
    for index in range(5):
        (root / "results" / f"checkpoint-{index:02d}.json").write_text(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "checkpoint_id": f"checkpoint-{index:02d}",
                    "previous_checkpoint_id": f"checkpoint-{index - 1:02d}" if index else None,
                    "new_evidence_ids": [f"delta-{index}"] if index else [],
                    "candidates": [
                        {
                            "candidate_id": candidate_id,
                            "title": f"{candidate_id} 标题",
                            "rank": rank,
                            "weighted_score": 80 - rank,
                            "score_change": 0,
                            "presentation_text": f"{candidate_id} 叙事",
                        }
                        for candidate_id, rank in candidates
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    (root / "results" / "final-synthesis.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "source_checkpoint_id": "checkpoint-04",
                "core_narrative": "核心叙事",
                "pillars": [{"candidate_id": "EC-01", "name": "支柱", "role": "角色", "evidence_refs": []}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return root


def _walk(timeline, primary="EC-03"):
    state = initial_replay_state(timeline.run_id)
    steps = [
        ("SUBMIT_SELECTION", {"selected_candidate_ids": timeline.selected_candidate_ids, "primary_candidate_id": primary}),
        ("RUN_HOLDOUT", {}),
        ("RUN_BLIND_REASSESSMENT", {}),
        ("RUN_NEXT_RELEASE", {}),
        ("RUN_NEXT_RELEASE", {}),
        ("RUN_NEXT_RELEASE", {}),
        ("RUN_NEXT_RELEASE", {}),
        (
            "SUBMIT_FINAL_SELECTION",
            {
                "shortlisted_candidate_ids": timeline.selected_candidate_ids,
                "selected_candidate_id": primary,
                "recommended_candidate_id": timeline.recommended_candidate_id(),
            },
        ),
    ]
    outcomes = []
    for action, payload in steps:
        outcome = apply_action(
            timeline, state, action, payload,
            requested_by=OPERATOR, now=NOW, milestone_cursor=8,
        )
        state = outcome.state
        outcomes.append(outcome)
    return state, outcomes


def test_load_rejects_incomplete_run(run_root: Path) -> None:
    (run_root / "results" / "checkpoint-03.json").unlink()
    with pytest.raises(ReplayTimelineError, match="冻结产物缺失"):
        load_replay_timeline(run_root)


def test_load_rejects_not_complete_status(run_root: Path) -> None:
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "RUNNING"
    (run_root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ReplayTimelineError, match="COMPLETE"):
        load_replay_timeline(run_root)


def test_load_rejects_selection_checkpoint_mismatch(run_root: Path) -> None:
    selection = json.loads(
        (run_root / "blind" / "selection_confirmation.json").read_text(encoding="utf-8")
    )
    selection["selected_candidate_ids"] = ["EC-01"]
    (root := run_root / "blind" / "selection_confirmation.json").write_text(
        json.dumps(selection), encoding="utf-8"
    )
    with pytest.raises(ReplayTimelineError, match="不一致"):
        load_replay_timeline(run_root)


def test_full_gate_walk_and_rows(run_root: Path) -> None:
    timeline = load_replay_timeline(run_root)
    state, outcomes = _walk(timeline)
    assert state["stage"] == "COMPLETE"
    assert len(state["history"]) == 8
    total_rows = sum(len(outcome.result_rows) for outcome in outcomes)
    assert total_rows == 1 + 3 + 1 + 3 * 4 + 1
    kinds = {row["record_kind"] for outcome in outcomes for row in outcome.result_rows}
    assert kinds == {"SELECTION", "HOLDOUT", "BLIND_REASSESS", "CHECKPOINT", "FINAL_SELECTION"}
    for outcome in outcomes:
        for row in outcome.result_rows:
            assert set(row) == {"run_id", "record_kind", "record_key", "content", "created_at"}
            assert row["run_id"] == RUN_ID
            assert row["created_at"] == NOW.isoformat()


def test_selection_gate_rejects_wrong_ids(run_root: Path) -> None:
    timeline = load_replay_timeline(run_root)
    state = initial_replay_state(RUN_ID)
    with pytest.raises(ReplayTimelineError, match="选线必须与冻结"):
        apply_action(
            timeline,
            state,
            "SUBMIT_SELECTION",
            {"selected_candidate_ids": ["EC-01"], "primary_candidate_id": "EC-01"},
            requested_by=OPERATOR,
            now=NOW,
            milestone_cursor=8,
        )


def test_final_selection_rejects_wrong_recommendation(run_root: Path) -> None:
    timeline = load_replay_timeline(run_root)
    state, _ = _walk(timeline)
    replay_state = dict(state)
    replay_state["stage"] = "CHECKPOINT_04"
    with pytest.raises(ReplayTimelineError, match="AI 推荐"):
        apply_action(
            timeline,
            replay_state,
            "SUBMIT_FINAL_SELECTION",
            {
                "shortlisted_candidate_ids": timeline.selected_candidate_ids,
                "selected_candidate_id": "EC-01",
                "recommended_candidate_id": "EC-01",
            },
            requested_by=OPERATOR,
            now=NOW,
            milestone_cursor=8,
        )


def test_gate_order_enforced(run_root: Path) -> None:
    timeline = load_replay_timeline(run_root)
    state = initial_replay_state(RUN_ID)
    with pytest.raises(ReplayTimelineError, match="只支持"):
        apply_action(timeline, state, "RUN_HOLDOUT", {}, requested_by=OPERATOR, now=NOW, milestone_cursor=8)


def test_holdout_counts_block(run_root: Path) -> None:
    timeline = load_replay_timeline(run_root)
    state = initial_replay_state(RUN_ID)
    outcome = apply_action(
        timeline,
        state,
        "SUBMIT_SELECTION",
        {"selected_candidate_ids": timeline.selected_candidate_ids, "primary_candidate_id": "EC-03"},
        requested_by=OPERATOR,
        now=NOW,
        milestone_cursor=8,
    )
    outcome = apply_action(timeline, outcome.state, "RUN_HOLDOUT", {}, requested_by=OPERATOR, now=NOW, milestone_cursor=8)
    contents = {json.loads(row["content"])["candidate_id"]: json.loads(row["content"]) for row in outcome.result_rows}
    assert contents["EC-01"]["SUPPORT"] == 1
    assert contents["EC-01"]["CHALLENGE"] == 1
    assert contents["EC-03"]["BLOCKING"] == 1
    assert contents["EC-02"]["TOTAL"] == 1


def test_state_roundtrip_and_run_guard(run_root: Path, tmp_path: Path) -> None:
    timeline = load_replay_timeline(run_root)
    state, _ = _walk(timeline)
    state_path = tmp_path / "replay_state.json"
    save_replay_state(state_path, state)
    loaded = read_replay_state(state_path, RUN_ID)
    assert loaded["stage"] == "COMPLETE"
    with pytest.raises(ReplayTimelineError, match="不一致"):
        read_replay_state(state_path, "other-run")


def test_run_log_summary_carries_replay_mark(run_root: Path) -> None:
    summary = run_log_summary("提交人工选线", "HOLDOUT", "选线完成")
    assert summary.startswith(f"【{REPLAY_SOURCE_MARK}】")
    assert "提交人工选线" in summary


def test_writeback_rows_content_is_json(run_root: Path) -> None:
    timeline = load_replay_timeline(run_root)
    state, outcomes = _walk(timeline)
    for outcome in outcomes:
        for row in outcome.result_rows:
            payload = json.loads(row["content"])
            assert isinstance(payload, dict)
            assert payload.get("mode") == REPLAY_SOURCE_MARK


def test_bot_config_replay_requires_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.run_feishu_bot import load_bot_config

    empty_root = tmp_path / "runs"
    empty_root.mkdir()
    base = {
        "FEISHU_APP_ID": "cli_x",
        "FEISHU_APP_SECRET": "secret",
        "FEISHU_DEMO_CHAT_ID": "oc_x",
        "FEISHU_BOT_OPEN_ID": "ou_bot",
        "FEISHU_OPERATOR_OPEN_IDS": "ou_op",
        "FEISHU_NOTIFICATION_DB": "outputs/notifications.sqlite3",
    }
    replay = {**base, "FEISHU_GATE_MODE": "replay", "FEISHU_DECISION_RUN_ROOT": str(empty_root)}
    with pytest.raises(ValueError, match="run_manifest.json"):
        load_bot_config(replay, workspace_root=tmp_path)

    replay_ok = {**replay, "FEISHU_GATE_MODE": "coordinator"}
    with pytest.raises(ValueError, match="active.json"):
        load_bot_config(replay_ok, workspace_root=tmp_path)

    active_root = tmp_path / "runs_active"
    active_root.mkdir()
    (active_root / "active.json").write_text("{}", encoding="utf-8")
    config = load_bot_config(
        {**base, "FEISHU_DECISION_RUN_ROOT": str(active_root)},
        workspace_root=tmp_path,
    )
    assert config.gate_mode == "coordinator"

    bad_mode = {**base, "FEISHU_GATE_MODE": "hybrid"}
    with pytest.raises(ValueError, match="coordinator or replay"):
        load_bot_config(bad_mode, workspace_root=tmp_path)


def test_gate_requires_milestone(run_root: Path) -> None:
    timeline = load_replay_timeline(run_root)
    state = initial_replay_state(RUN_ID)
    with pytest.raises(ReplayTimelineError, match="没有活动演示会话"):
        apply_action(timeline, state, "SUBMIT_SELECTION", {}, requested_by=OPERATOR, now=NOW)
    with pytest.raises(ReplayTimelineError, match="演示尚未推进"):
        apply_action(
            timeline,
            state,
            "SUBMIT_SELECTION",
            {"selected_candidate_ids": timeline.selected_candidate_ids, "primary_candidate_id": "EC-03"},
            requested_by=OPERATOR,
            now=NOW,
            milestone_cursor=1,
        )
    outcome = apply_action(
        timeline,
        state,
        "SUBMIT_SELECTION",
        {"selected_candidate_ids": timeline.selected_candidate_ids, "primary_candidate_id": "EC-03"},
        requested_by=OPERATOR,
        now=NOW,
        milestone_cursor=2,
    )
    assert outcome.state["stage"] == "HOLDOUT"
    assert outcome.drives_node == 3


def test_gate_milestone_binding_map() -> None:
    from src.services.replay_gate_timeline import gate_milestone_binding

    assert gate_milestone_binding("SUBMIT_SELECTION") == (2, 3)
    assert gate_milestone_binding("RUN_HOLDOUT") == (3, None)
    assert gate_milestone_binding("SUBMIT_FINAL_SELECTION") == (7, 8)
    assert gate_milestone_binding("RUN_NEXT_RELEASE", "CHECKPOINT_00") == (3, 4)
    assert gate_milestone_binding("RUN_NEXT_RELEASE", "CHECKPOINT_03") == (6, 7)


def test_demo_command_channel_roundtrip(tmp_path: Path) -> None:
    from src.integrations.feishu.notification_store import NotificationStore

    store = NotificationStore(tmp_path / "notify.sqlite3")
    session = store.create_session("run-x", created_by="streamlit-local")
    first = store.enqueue_demo_command(
        session.session_id, command="ADVANCE_TO_NODE", target_node=4, requested_by="ou_op"
    )
    again = store.enqueue_demo_command(
        session.session_id, command="ADVANCE_TO_NODE", target_node=4, requested_by="ou_op"
    )
    assert first.command_id == again.command_id
    pending = store.pending_demo_commands(session.session_id)
    assert [item.target_node for item in pending] == [4]
    consumed = store.consume_demo_command(first.command_id)
    assert consumed.status == "CONSUMED"
    assert store.pending_demo_commands(session.session_id) == []
    with pytest.raises(ValueError):
        store.consume_demo_command(first.command_id)


def test_apply_milestone_state_mutations() -> None:
    from src.ui.demo_notification_control import apply_milestone_state
    from src.ui.workspace_shell import Workspace

    state: dict = {}
    apply_milestone_state(state, 1)
    assert state["preprocessing_replay_completed_steps"] == [0, 1, 2, 3, 4]
    assert state["active_workspace"] == Workspace.STRESS_TEST.value
    apply_milestone_state(state, 4)
    assert state["pressure_team_confirmed"] is True
    assert state["evolution_checkpoint_index"] == 2
    assert state["active_workspace"] == Workspace.REALTIME_DECISION.value
    apply_milestone_state(state, 8)
    assert state["evolution_show_finale"] is True
    assert state["evolution_finale_act"] == 2


def test_replay_next_missing_snapshot_sends_advance_command(tmp_path: Path) -> None:
    from src.integrations.feishu.notification_bot import handle_replay_action
    from src.integrations.feishu.notification_store import NotificationStore
    from src.services.demo_notifications import DemoNotificationNode, DemoNotificationSnapshot

    store = NotificationStore(tmp_path / "notify.sqlite3")
    session = store.create_session("run-y", created_by="streamlit-local")
    snapshot = DemoNotificationSnapshot(
        node=DemoNotificationNode.PREPROCESSING_COMPLETE,
        ordinal=2,
        title="阶段完成",
        conclusion="压力测试完成",
        source_run_id="run-y",
        source_sha256="0" * 64,
    )
    store.enqueue_snapshot(session.session_id, snapshot)
    card = store.reserve_replay_card(
        session.session_id,
        source_message_id="om_source_1",
        ordinal=2,
        reply_uuid="uuid-1",
    )
    result = handle_replay_action(
        {"action": "REPLAY_NEXT", "replay_id": card.replay_id, "target_ordinal": 3},
        operator_open_id="ou_op",
        allowed_operator_ids={"ou_op"},
        store=store,
    )
    assert result.toast_type == "success"
    assert "单步推进指令" in result.toast_content
    pending = store.pending_demo_commands(session.session_id)
    assert [item.command for item in pending] == ["ADVANCE_STEP"]
