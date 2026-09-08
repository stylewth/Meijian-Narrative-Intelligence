"""产物投影、决策门验证/队列与机器人接线的合同测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.schemas import (
    BrandFeedbackComparison,
    BusinessDecisionStatus,
    CandidateDecisionSnapshot,
    CandidateEvaluation,
    CandidateScores,
    CandidateStressResult,
    CommentRecord,
    CorpusAnalysisResult,
    DecisionEntryMode,
    DecisionEvolutionCheckpoint,
    DecisionFoundationState,
    EmotionalConflict,
    EvidenceAtom,
    EvidenceGrade,
    EvidenceQuote,
    EvidenceRoute,
    EvolutionCheckpointRole,
    ExperienceScope,
    FinalCandidateSelection,
    NarrativeCandidate,
    NarrativePatch,
    RankedNarrative,
    RankingResult,
    SampleType,
    ScoreItem,
    SourceReference,
    SourceType,
    StressCheckResult,
    StressCheckType,
    StressExecutionStatus,
)
from src.integrations.feishu.decision_gate_store import (
    DecisionGateRunner,
    DecisionGateStore,
)
from src.integrations.feishu.decision_gates import (
    GateRejection,
    available_gates,
    handle_decision_gate_action,
    resolve_result_path,
    stage_status_text,
    validate_gate_request,
)
from src.integrations.feishu.notification_bot import CallbackResult
from src.integrations.feishu.result_projection import (
    candidate_rows,
    checkpoint_rows,
    final_selection_row,
    run_log_fields,
    snapshot_row,
)
from tools.run_feishu_bot import (
    REQUIRED_CONFIG_KEYS,
    _is_decision_status_command,
    load_bot_config,
    read_active_decision_state,
)


CREATED_AT = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)


def _quote(comment_id: str = "c1") -> EvidenceQuote:
    return EvidenceQuote(comment_id=comment_id, quote="微醺好喝")


def _source() -> SourceReference:
    return SourceReference(
        source_id="s1",
        source_type=SourceType.USER_COMMENT,
        source_ref="screened.xlsx",
    )


def _atom(evidence_id: str = "e1") -> EvidenceAtom:
    return EvidenceAtom(
        evidence_id=evidence_id,
        comment_id="c1",
        route=EvidenceRoute.BRAND,
        experience_scope=ExperienceScope.ACTUAL_USE,
        evidence_grade=EvidenceGrade.A,
        source=_source(),
    )


def _record() -> CommentRecord:
    return CommentRecord(
        comment_id="c1",
        sample_type=SampleType.MEIJIAN_FEEDBACK,
        raw_content="微醺好喝",
    )


def _corpus() -> CorpusAnalysisResult:
    conflict = EmotionalConflict(
        conflict_id="cf1",
        title="想要轻松微醺",
        user_need="放松",
        desired_state="微醺",
        rejected_state="宿醉",
        identity_need="懂生活",
        main_concerns=["太甜"],
        main_scenes=["夜宵"],
        supporting_evidence=[_quote()],
        counter_evidence_note="未发现反面证据",
        implication_for_meijian="低度微醺",
    )
    return CorpusAnalysisResult(
        emotional_conflicts=[conflict],
        feedback_comparison=BrandFeedbackComparison(),
        main_scenes=["夜宵"],
        data_limitations=[],
    )


def _scores() -> CandidateScores:
    item = ScoreItem(score=80.0, rationale="证据充分")
    return CandidateScores(
        evidence_strength=item,
        emotional_tension=item,
        meijian_fit_and_exclusivity=item,
        competitor_difference=item,
        scene_conversion=item,
    )


def _candidate(candidate_id: str) -> NarrativeCandidate:
    return NarrativeCandidate(
        candidate_id=candidate_id,
        primary_conflict_id="cf1",
        supporting_conflict_ids=["cf1"],
        title=f"微醺夜宵叙事{candidate_id}",
        target_audience="都市年轻人",
        user_conflict="想放松又怕宿醉",
        brand_opportunity="低度梅酒",
        why_meijian="青梅原料",
        competitor_difference="更低的度数",
        brand_role="陪伴者",
        draft_proposition="梅见微醺，夜宵刚刚好",
        main_scenes=["夜宵摊"],
        content_theme="轻松夜生活",
        supporting_evidence=[_quote()],
        counter_evidence_note="未发现",
        risks=["人群偏窄"],
    )


def _ranked(candidate_id: str, rank: int, recommended: bool = False) -> RankedNarrative:
    return RankedNarrative(
        candidate=_candidate(candidate_id),
        evaluation=CandidateEvaluation(
            candidate_id=candidate_id,
            scores=_scores(),
            overall_assessment="整体稳健",
        ),
        weighted_score=88.5,
        rank=rank,
        is_recommended=recommended,
    )


def _stress(candidate_id: str) -> CandidateStressResult:
    checks = [
        StressCheckResult(
            check_type=check_type,
            execution_status=StressExecutionStatus.PENDING,
            reference_ids=["e1"],
            rationale="待执行",
        )
        for check_type in StressCheckType
    ]
    return CandidateStressResult(candidate_id=candidate_id, checks=checks)


def _foundation() -> DecisionFoundationState:
    return DecisionFoundationState(
        run_id="run-1",
        baseline_records=[_record()],
        baseline_atoms=[_atom()],
        visible_atoms=[_atom()],
        corpus=_corpus(),
        candidates=[_candidate("C1"), _candidate("C2")],
        ranking=RankingResult(
            ranked_candidates=[_ranked("C1", 1, True), _ranked("C2", 2)],
            recommended_candidate_id="C1",
            recommendation_reason="证据最充分",
        ),
        stress_results=[_stress("C1"), _stress("C2")],
    )


def _snapshot(candidate_id: str, rank: int) -> CandidateDecisionSnapshot:
    return CandidateDecisionSnapshot(
        ranked_narrative=_ranked(candidate_id, rank),
        stress_result=_stress(candidate_id),
        business_status=BusinessDecisionStatus.READY_FOR_VALIDATION,
        supporting_count=3,
        counter_count=0,
        risk_count=1,
    )


def _checkpoint() -> DecisionEvolutionCheckpoint:
    return DecisionEvolutionCheckpoint(
        run_id="run-1",
        checkpoint_id="checkpoint-01",
        entry_mode=DecisionEntryMode.DEMO,
        role=EvolutionCheckpointRole.DELTA,
        release_index=1,
        visible_evidence_ids=["e1", "e2"],
        route_distribution={},
        experience_distribution={},
        grade_distribution={},
        candidates=[_snapshot("C1", 1), _snapshot("C2", 2)],
        selected_candidate_id="C1",
        patches=[
            NarrativePatch(
                candidate_id="C1",
                field_name="draft_proposition",
                before="梅见微醺，夜宵刚刚好",
                after="梅见微醺，夜宵刚刚更松弛",
                trigger_evidence_ids=["e2"],
                reason="新增夜宵场景证据",
            )
        ],
    )


# ---------- S4 产物投影 ----------


def test_candidate_rows_projection() -> None:
    rows = candidate_rows(_foundation(), created_at=CREATED_AT)

    assert [row["record_key"] for row in rows] == ["C1", "C2"]
    assert all(row["record_kind"] == "CANDIDATE" for row in rows)
    assert all(row["run_id"] == "run-1" for row in rows)
    first = rows[0]
    assert "标题：微醺夜宵叙事C1" in first["content"]
    assert "加权分：88.50" in first["content"]
    assert "推荐" in first["content"]
    assert "排名：2" in rows[1]["content"]


def test_candidate_rows_rejects_wrong_type() -> None:
    with pytest.raises(TypeError):
        candidate_rows({"candidates": []}, created_at=CREATED_AT)


def test_checkpoint_rows_projection() -> None:
    rows = checkpoint_rows(_checkpoint(), created_at=CREATED_AT)

    assert [row["record_key"] for row in rows] == [
        "checkpoint-01:C1",
        "checkpoint-01:C2",
    ]
    assert rows[0]["record_kind"] == "CHECKPOINT"
    assert "修订：新增夜宵场景证据" in rows[0]["content"]
    assert "状态：READY_FOR_VALIDATION" in rows[0]["content"]
    assert "支持 3／反例 0" in rows[0]["content"]


def test_final_selection_row_projection() -> None:
    selection = FinalCandidateSelection(
        run_id="run-1",
        checkpoint_04_sha256="a" * 64,
        shortlisted_candidate_ids=["C1", "C2"],
        selected_candidate_id="C2",
        recommended_candidate_id="C1",
        selected_by="ou_operator",
        selected_at=CREATED_AT,
        reason="更贴近夜宵场景",
    )
    row = final_selection_row(selection, created_at=CREATED_AT)

    assert row["record_kind"] == "FINAL_SELECTION"
    assert row["record_key"] == "C2"
    assert "最终选择：C2" in row["content"]
    assert "理由：更贴近夜宵场景" in row["content"]


def test_snapshot_row_projection() -> None:
    row = snapshot_row(
        _snapshot("C1", 1), run_id="run-1", record_key="C1", created_at=CREATED_AT
    )
    assert row["record_kind"] == "CANDIDATE"
    assert "加权分" in row["content"]


def test_run_log_fields_contract() -> None:
    fields = run_log_fields(
        "run-1",
        action="SUBMIT_SELECTION",
        actor_open_id="ou_op",
        summary="人工确认 C1",
    )
    assert set(fields) == {
        "run_id",
        "action",
        "actor_open_id",
        "summary",
        "created_at",
    }
    with pytest.raises(ValueError):
        run_log_fields("run-1", action=" ", actor_open_id="ou_op", summary="s")


# ---------- S5 决策门 ----------


def test_available_gates_known_and_unknown_stage() -> None:
    assert available_gates("AWAITING_SELECTION") == ("SUBMIT_SELECTION",)
    assert available_gates("CHECKPOINT_04") == ("SUBMIT_FINAL_SELECTION",)
    with pytest.raises(GateRejection):
        available_gates("NOT_A_STAGE")


def test_validate_gate_request_operator_stage_payload() -> None:
    operators = {"ou_op"}

    with pytest.raises(GateRejection, match="权限"):
        validate_gate_request(
            stage="AWAITING_SELECTION",
            gate_action="SUBMIT_SELECTION",
            payload=None,
            operator_open_id="ou_stranger",
            allowed_operator_ids=operators,
        )

    with pytest.raises(GateRejection, match="不支持"):
        validate_gate_request(
            stage="HOLDOUT",
            gate_action="SUBMIT_SELECTION",
            payload=None,
            operator_open_id="ou_op",
            allowed_operator_ids=operators,
        )

    with pytest.raises(GateRejection, match="主叙事"):
        validate_gate_request(
            stage="AWAITING_SELECTION",
            gate_action="SUBMIT_SELECTION",
            payload={"selected_candidate_ids": ["C1"], "primary_candidate_id": "C2"},
            operator_open_id="ou_op",
            allowed_operator_ids=operators,
        )

    validate_gate_request(
        stage="AWAITING_SELECTION",
        gate_action="SUBMIT_SELECTION",
        payload={"selected_candidate_ids": ["C1"], "primary_candidate_id": "C1"},
        operator_open_id="ou_op",
        allowed_operator_ids=operators,
    )


def test_stage_status_text_normal_and_failed() -> None:
    text = stage_status_text({"run_id": "run-1", "stage": "AWAITING_SELECTION"})
    assert "run-1" in text
    assert "提交人工选线" in text

    failed = stage_status_text(
        {"run_id": "run-1", "stage": "FAILED", "resume_stage": "HOLDOUT"}
    )
    assert "失败态" in failed
    assert "HOLDOUT" in failed

    with pytest.raises(ValueError):
        stage_status_text({"run_id": "run-1"})


def test_resolve_result_path_must_stay_inside_run_root(tmp_path: Path) -> None:
    inside = resolve_result_path(run_root=tmp_path, result_path="results/r1.json")
    assert inside == (tmp_path / "results/r1.json").resolve()

    with pytest.raises(GateRejection):
        resolve_result_path(run_root=tmp_path, result_path="../outside.json")


def test_gate_store_single_flight_and_idempotency(tmp_path: Path) -> None:
    store = DecisionGateStore(tmp_path / "gates.sqlite3")

    first = store.enqueue_gate(
        "run-1",
        gate_action="RUN_HOLDOUT",
        payload={},
        requested_by="ou_op",
    )
    again = store.enqueue_gate(
        "run-1",
        gate_action="RUN_HOLDOUT",
        payload={},
        requested_by="ou_op",
    )
    assert first.job_id == again.job_id

    with pytest.raises(ValueError, match="排队或执行"):
        store.enqueue_gate(
            "run-1",
            gate_action="RUN_NEXT_RELEASE",
            payload={},
            requested_by="ou_op",
        )

    claimed = store.claim_next_gate(CREATED_AT)
    assert claimed is not None and claimed.status == "IN_FLIGHT"
    store.mark_done(claimed.job_id, finished_at=CREATED_AT)

    with pytest.raises(ValueError, match="已执行过"):
        store.enqueue_gate(
            "run-1",
            gate_action="RUN_HOLDOUT",
            payload={},
            requested_by="ou_op",
        )


def test_gate_runner_success_and_failure(tmp_path: Path) -> None:
    store = DecisionGateStore(tmp_path / "gates.sqlite3")
    store.enqueue_gate(
        "run-1", gate_action="RUN_HOLDOUT", payload={}, requested_by="ou_op"
    )
    executed: list[str] = []
    runner = DecisionGateRunner(
        store,
        lambda job: executed.append(job.gate_action),
        clock=lambda: CREATED_AT,
    )
    assert runner.run_once() is True
    assert executed == ["RUN_HOLDOUT"]
    assert store.gate_jobs_for_run("run-1")[0].status == "DONE"

    store.enqueue_gate(
        "run-2", gate_action="RUN_HOLDOUT", payload={}, requested_by="ou_op"
    )
    failing = DecisionGateRunner(
        store,
        lambda job: (_ for _ in ()).throw(RuntimeError("LLM 网关超时")),
        clock=lambda: CREATED_AT,
    )
    assert failing.run_once() is True
    failed_job = store.gate_jobs_for_run("run-2")[0]
    assert failed_job.status == "FAILED"
    assert "LLM 网关超时" in failed_job.last_error


# ---------- S6 机器人接线 ----------


def test_handle_decision_gate_action_flows(tmp_path: Path) -> None:
    store = DecisionGateStore(tmp_path / "gates.sqlite3")
    state = {"run_id": "run-1", "stage": "AWAITING_SELECTION"}
    load_state = lambda run_id: state if run_id == "run-1" else None

    disabled = handle_decision_gate_action(
        {"type": "DECISION_GATE", "run_id": "run-1", "gate_action": "SUBMIT_SELECTION"},
        operator_open_id="ou_op",
        allowed_operator_ids={"ou_op"},
        load_state=load_state,
        gate_store=store,
        gate_enabled=False,
    )
    assert disabled.toast_type == "warning"

    ok = handle_decision_gate_action(
        {
            "type": "DECISION_GATE",
            "run_id": "run-1",
            "gate_action": "SUBMIT_SELECTION",
            "payload": {"selected_candidate_ids": ["C1"], "primary_candidate_id": "C1"},
        },
        operator_open_id="ou_op",
        allowed_operator_ids={"ou_op"},
        load_state=load_state,
        gate_store=store,
        gate_enabled=True,
    )
    assert ok.toast_type == "success"
    assert store.gate_jobs_for_run("run-1")[0].status == "PENDING"

    mismatch = handle_decision_gate_action(
        {"type": "DECISION_GATE", "run_id": "run-1", "gate_action": "RUN_HOLDOUT"},
        operator_open_id="ou_op",
        allowed_operator_ids={"ou_op"},
        load_state=load_state,
        gate_store=store,
        gate_enabled=True,
    )
    assert mismatch.toast_type == "warning"

    stranger = handle_decision_gate_action(
        {"type": "DECISION_GATE", "run_id": "run-1", "gate_action": "SUBMIT_SELECTION"},
        operator_open_id="ou_stranger",
        allowed_operator_ids={"ou_op"},
        load_state=load_state,
        gate_store=store,
        gate_enabled=True,
    )
    assert stranger.toast_type == "warning"

    missing = handle_decision_gate_action(
        {"type": "DECISION_GATE", "run_id": "run-404", "gate_action": "SUBMIT_SELECTION"},
        operator_open_id="ou_op",
        allowed_operator_ids={"ou_op"},
        load_state=load_state,
        gate_store=store,
        gate_enabled=True,
    )
    assert missing.toast_type == "error"


def test_read_active_decision_state(tmp_path: Path) -> None:
    assert read_active_decision_state(tmp_path) is None

    run_dir = tmp_path / "run-1" / "coordinator"
    run_dir.mkdir(parents=True)
    (run_dir / "state.json").write_text(
        '{"run_id": "run-1", "stage": "AWAITING_SELECTION"}', encoding="utf-8"
    )
    (tmp_path / "active.json").write_text(
        '{"run_id": "run-1", "run_manifest_sha256": "x"}', encoding="utf-8"
    )

    state = read_active_decision_state(tmp_path)
    assert state == {"run_id": "run-1", "stage": "AWAITING_SELECTION"}

    (tmp_path / "active.json").write_text(
        '{"run_id": "../evil", "run_manifest_sha256": "x"}', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        read_active_decision_state(tmp_path)


def test_load_bot_config_optional_writeback_settings(tmp_path: Path) -> None:
    mapping = {key: "value" for key in REQUIRED_CONFIG_KEYS}
    mapping["FEISHU_NOTIFICATION_DB"] = "outputs/notifications.sqlite3"

    config = load_bot_config(mapping, workspace_root=tmp_path)
    assert config.writeback_run_log_url is None
    assert config.writeback_results_url is None
    assert config.writeback_db == (tmp_path / "outputs/feishu_writeback.sqlite3")
    assert config.gate_db == (tmp_path / "outputs/feishu_gate_jobs.sqlite3")
    assert config.decision_run_root is None

    mapping.update(
        {
            "FEISHU_RUNLOG_URL": "https://demo.feishu.cn/base/app1?table=tbl1",
            "FEISHU_RESULTS_URL": "https://demo.feishu.cn/base/app2?table=tbl2",
            "FEISHU_DECISION_RUN_ROOT": "runs",
        }
    )
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "active.json").write_text("{}", encoding="utf-8")
    config = load_bot_config(mapping, workspace_root=tmp_path)
    assert config.writeback_run_log_url is not None
    assert config.decision_run_root == (tmp_path / "runs").resolve()

    mapping["FEISHU_DECISION_RUN_ROOT"] = "missing_dir"
    with pytest.raises(ValueError, match="FEISHU_DECISION_RUN_ROOT"):
        load_bot_config(mapping, workspace_root=tmp_path)


def test_is_decision_status_command_text_extraction() -> None:
    p2p = {
        "message": {"chat_type": "p2p", "content": {"text": "决策进度"}}
    }
    assert _is_decision_status_command(p2p, "ou_bot") is True

    group_no_mention = {
        "message": {"chat_type": "group", "content": {"text": "决策进度"}}
    }
    assert _is_decision_status_command(group_no_mention, "ou_bot") is False

    group_mention = {
        "message": {
            "chat_type": "group",
            "content": {"text": "@_user_1 决策进度"},
            "mentions": [{"key": "@_user_1", "id": "ou_bot"}],
        }
    }
    assert _is_decision_status_command(group_mention, "ou_bot") is True

    other = {"message": {"chat_type": "p2p", "content": {"text": "你好"}}}
    assert _is_decision_status_command(other, "ou_bot") is False


def test_callback_result_shapes() -> None:
    assert CallbackResult(
        toast_type="success", toast_content="ok"
    ).toast_type in {"success", "warning", "error"}
