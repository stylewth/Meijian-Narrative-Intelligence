"""决策门执行器（S7）与完成事件写回（S8）的合同测试。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.integrations.feishu.decision_gate_store import DecisionGateStore
from src.integrations.feishu.decision_gates import GateRejection, validate_gate_request
from src.integrations.feishu.gate_executor import (
    DecisionGateExecutor,
    GateExecutionError,
)
from src.integrations.feishu.writeback_store import WritebackStore
from src.schemas import CandidateSelection, FinalCandidateSelection
from src.services.prepared_corpus import sha256_bytes

from test_feishu_decision_gates import CREATED_AT, _checkpoint, _foundation


class FakeCoordinator:
    def __init__(self, *, stage: str = "AWAITING_SELECTION", fail: bool = False):
        self.stage = stage
        self.fail = fail
        self.calls: list[tuple] = []

    @property
    def current_stage(self) -> str:
        return self.stage

    def _maybe_fail(self) -> None:
        if self.fail:
            raise RuntimeError("coordinator 拒绝：阶段不符")

    def submit_selection(self, run_id, selected_ids, primary_id):
        self.calls.append(("submit_selection", tuple(selected_ids), primary_id))
        self._maybe_fail()

    def run_next_release(self):
        self.calls.append(("run_next_release",))
        self._maybe_fail()
        self.stage = "CHECKPOINT_01"

    def finalize_specificity(self, run_id):
        self.calls.append(("finalize_specificity", run_id))
        self._maybe_fail()
        self.stage = "HOLDOUT"

    def reselect_primary(self, selection, session_sha):
        self.calls.append(("reselect_primary", selection.primary_candidate_id, session_sha))
        self._maybe_fail()
        self.stage = "AWAITING_BLIND_EVIDENCE"

    def submit_final_selection(self, selection):
        self.calls.append(("submit_final_selection", selection))
        self._maybe_fail()
        self.stage = "COMPLETE"


class FakeStageStore:
    def __init__(self, root: Path, stages: dict):
        self.root = root
        self._stages = stages

    def load_stage(self, name, model):
        value = self._stages.get(name)
        if value is None:
            raise KeyError(f"stage {name} not frozen")
        if not isinstance(value, model):
            raise TypeError(f"stage {name} model mismatch")
        return value


class FakeMessageClient:
    def __init__(self):
        self.cards: list[tuple[str, dict, str]] = []

    def send_card(self, chat_id, card, *, uuid):
        self.cards.append((chat_id, card, uuid))
        return "om_gate_result"


def _selection_fixture() -> CandidateSelection:
    return CandidateSelection(
        run_id="run-1",
        pending_selection_sha256="a" * 64,
        selected_candidate_ids=["C1", "C2"],
        primary_candidate_id="C1",
        selected_by="cli",
        selected_at=CREATED_AT,
    )


def _final_fixture() -> FinalCandidateSelection:
    return FinalCandidateSelection(
        run_id="run-1",
        checkpoint_04_sha256="b" * 64,
        shortlisted_candidate_ids=["C1", "C2"],
        selected_candidate_id="C1",
        recommended_candidate_id="C1",
        selected_by="ou_op",
        selected_at=CREATED_AT,
    )


def _make_job(tmp_path: Path, *, action: str, payload: dict, run_id: str = "run-1") -> tuple:
    gate_store = DecisionGateStore(tmp_path / "gates.sqlite3")
    job = gate_store.enqueue_gate(
        run_id, gate_action=action, payload=payload, requested_by="ou_op"
    )
    return job


def _executor(
    tmp_path: Path,
    coordinator,
    store,
    writeback_store,
) -> tuple[DecisionGateExecutor, FakeMessageClient]:
    client = FakeMessageClient()
    executor = DecisionGateExecutor(
        run_root=tmp_path,
        message_client=client,
        chat_id="oc_demo",
        writeback_store=writeback_store,
        coordinator_factory=lambda root, run_id: (coordinator, store),
    )
    return executor, client


def test_submit_selection_success_enqueues_log_and_sends_card(tmp_path):
    job = _make_job(
        tmp_path,
        action="SUBMIT_SELECTION",
        payload={"selected_candidate_ids": ["C1"], "primary_candidate_id": "C1"},
    )
    coordinator = FakeCoordinator(stage="AWAITING_SELECTION")
    store = FakeStageStore(tmp_path, {})
    writeback = WritebackStore(tmp_path / "wb.sqlite3")
    executor, client = _executor(tmp_path, coordinator, store, writeback)

    executor(job)

    assert coordinator.calls == [("submit_selection", ("C1",), "C1")]
    jobs = writeback.jobs_for_run("run-1")
    assert len(jobs) == 1 and jobs[0].record_kind == "RUN_LOG"
    assert "选线 C1" in jobs[0].payload_json
    assert len(client.cards) == 1
    _chat_id, card, _uuid = client.cards[0]
    assert "完成" in card["header"]["title"]["content"]


def test_run_next_release_projects_checkpoint_rows(tmp_path):
    job = _make_job(tmp_path, action="RUN_NEXT_RELEASE", payload={})
    coordinator = FakeCoordinator(stage="CHECKPOINT_00")
    store = FakeStageStore(tmp_path, {"checkpoint_01": _checkpoint()})
    writeback = WritebackStore(tmp_path / "wb.sqlite3")
    executor, client = _executor(tmp_path, coordinator, store, writeback)

    executor(job)

    jobs = writeback.jobs_for_run("run-1")
    kinds = [item.record_kind for item in jobs]
    assert kinds.count("RUN_LOG") == 1
    assert kinds.count("CHECKPOINT") == 2
    assert any("checkpoint-01:C1" in item.payload_json for item in jobs)
    assert len(client.cards) == 1


def test_finalize_specificity_projects_candidate_rows(tmp_path):
    job = _make_job(tmp_path, action="FINALIZE_SPECIFICITY", payload={})
    coordinator = FakeCoordinator(stage="SPECIFICITY_FINAL_RANK")
    store = FakeStageStore(tmp_path, {"specificity_final_foundation": _foundation()})
    writeback = WritebackStore(tmp_path / "wb.sqlite3")
    executor, _client = _executor(tmp_path, coordinator, store, writeback)

    executor(job)

    kinds = [item.record_kind for item in writeback.jobs_for_run("run-1")]
    assert kinds.count("RUN_LOG") == 1
    assert kinds.count("CANDIDATE") == 2


def test_gate_failure_sends_failure_card_and_reraises(tmp_path):
    job = _make_job(
        tmp_path,
        action="SUBMIT_SELECTION",
        payload={"selected_candidate_ids": ["C1"], "primary_candidate_id": "C1"},
    )
    coordinator = FakeCoordinator(stage="AWAITING_SELECTION", fail=True)
    store = FakeStageStore(tmp_path, {})
    writeback = WritebackStore(tmp_path / "wb.sqlite3")
    executor, client = _executor(tmp_path, coordinator, store, writeback)

    with pytest.raises(RuntimeError, match="阶段不符"):
        executor(job)

    assert len(client.cards) == 1
    assert "失败" in client.cards[0][1]["header"]["title"]["content"]
    assert writeback.jobs_for_run("run-1") == []


def test_post_gate_failure_reports_gate_already_executed(tmp_path):
    job = _make_job(tmp_path, action="RUN_NEXT_RELEASE", payload={})
    coordinator = FakeCoordinator(stage="CHECKPOINT_00")
    store = FakeStageStore(tmp_path, {})  # 缺 checkpoint_01，投影必须失败
    writeback = WritebackStore(tmp_path / "wb.sqlite3")
    executor, _client = _executor(tmp_path, coordinator, store, writeback)

    with pytest.raises(GateExecutionError, match="门已执行"):
        executor(job)

    assert coordinator.calls == [("run_next_release",)]


def test_reselect_primary_membership_validation(tmp_path):
    stages = {"selection": _selection_fixture()}
    job_bad = _make_job(
        tmp_path, action="RESELECT_PRIMARY", payload={"primary_candidate_id": "C404"}
    )
    coordinator = FakeCoordinator(stage="AWAITING_PRIMARY_RESELECTION")
    executor, client = _executor(
        tmp_path, coordinator, FakeStageStore(tmp_path, stages), None
    )
    with pytest.raises(ValueError, match="不在已选线集合"):
        executor(job_bad)
    assert "失败" in client.cards[0][1]["header"]["title"]["content"]

    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    (stages_dir / "specificity_session.json").write_bytes(b"session-bytes")
    job_ok = _make_job(
        tmp_path,
        action="RESELECT_PRIMARY",
        payload={"primary_candidate_id": "C2"},
        run_id="run-2",
    )
    coordinator_ok = FakeCoordinator(stage="AWAITING_PRIMARY_RESELECTION")
    executor_ok, client_ok = _executor(
        tmp_path, coordinator_ok, FakeStageStore(tmp_path, stages), None
    )
    executor_ok(job_ok)
    assert coordinator_ok.calls == [
        ("reselect_primary", "C2", sha256_bytes(b"session-bytes"))
    ]


def test_submit_final_selection_assembly(tmp_path):
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    checkpoint_bytes = b"checkpoint-04-bytes"
    (stages_dir / "checkpoint_04.json").write_bytes(checkpoint_bytes)
    job = _make_job(
        tmp_path,
        action="SUBMIT_FINAL_SELECTION",
        payload={
            "shortlisted_candidate_ids": ["C1", "C2"],
            "selected_candidate_id": "C1",
            "recommended_candidate_id": "C1",
        },
    )
    coordinator = FakeCoordinator(stage="CHECKPOINT_04")
    store = FakeStageStore(tmp_path, {"final_selection": _final_fixture()})
    writeback = WritebackStore(tmp_path / "wb.sqlite3")
    executor, _client = _executor(tmp_path, coordinator, store, writeback)

    executor(job)

    recorded = coordinator.calls[0][1]
    assert isinstance(recorded, FinalCandidateSelection)
    assert recorded.checkpoint_04_sha256 == sha256_bytes(checkpoint_bytes)
    assert recorded.selected_by == "ou_op"
    kinds = [item.record_kind for item in writeback.jobs_for_run("run-1")]
    assert kinds.count("RUN_LOG") == 1
    assert kinds.count("FINAL_SELECTION") == 1


def test_no_writeback_config_still_sends_card(tmp_path):
    job = _make_job(
        tmp_path,
        action="SUBMIT_SELECTION",
        payload={"selected_candidate_ids": ["C1"], "primary_candidate_id": "C1"},
    )
    coordinator = FakeCoordinator(stage="AWAITING_SELECTION")
    executor, client = _executor(
        tmp_path, coordinator, FakeStageStore(tmp_path, {}), None
    )

    executor(job)

    assert len(client.cards) == 1
    assert all("写回入队" not in str(card) for _c, card, _u in client.cards)


def test_result_card_carries_next_gate_buttons(tmp_path):
    from src.integrations.feishu.gate_executor import next_gate_elements

    job = _make_job(tmp_path, action="RUN_NEXT_RELEASE", payload={})
    coordinator = FakeCoordinator(stage="CHECKPOINT_00")
    store = FakeStageStore(tmp_path, {"checkpoint_01": _checkpoint()})
    executor, client = _executor(
        tmp_path, coordinator, store, WritebackStore(tmp_path / "wb.sqlite3")
    )

    executor(job)

    card = client.cards[0][1]
    card_json = str(card)
    assert "释放下一批增量" in card_json
    assert "DECISION_GATE" in card_json
    value = next_gate_elements("run-1", "CHECKPOINT_01", prefix="t")
    assert value and "RUN_NEXT_RELEASE" in str(value)
    parameter_gates = next_gate_elements("run-1", "CHECKPOINT_04", prefix="t")
    assert all("behaviors" not in str(item) or "需参数" not in str(item) for item in parameter_gates)
    assert any("需参数" in str(item) for item in parameter_gates)


def test_final_selection_payload_validation_extended() -> None:
    with pytest.raises(GateRejection, match="入围候选"):
        validate_gate_request(
            stage="CHECKPOINT_04",
            gate_action="SUBMIT_FINAL_SELECTION",
            payload={"selected_candidate_id": "C1"},
            operator_open_id="ou_op",
            allowed_operator_ids={"ou_op"},
        )
    with pytest.raises(GateRejection, match="推荐候选"):
        validate_gate_request(
            stage="CHECKPOINT_04",
            gate_action="SUBMIT_FINAL_SELECTION",
            payload={
                "shortlisted_candidate_ids": ["C1", "C2"],
                "selected_candidate_id": "C1",
                "recommended_candidate_id": "C404",
            },
            operator_open_id="ou_op",
            allowed_operator_ids={"ou_op"},
        )
    validate_gate_request(
        stage="CHECKPOINT_04",
        gate_action="SUBMIT_FINAL_SELECTION",
        payload={
            "shortlisted_candidate_ids": ["C1", "C2"],
            "selected_candidate_id": "C1",
            "recommended_candidate_id": "C1",
        },
        operator_open_id="ou_op",
        allowed_operator_ids={"ou_op"},
    )
