"""DecisionGateRunner.run_forever 接线回归：门队列必须能被后台线程持续消费。"""

from __future__ import annotations

from pathlib import Path
import sys
from threading import Event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.integrations.feishu.decision_gate_store import (  # noqa: E402
    DecisionGateRunner,
    DecisionGateStore,
)


def test_run_forever_consumes_queued_job_and_stops(tmp_path: Path) -> None:
    store = DecisionGateStore(tmp_path / "gates.sqlite3")
    stop = Event()
    executed: list[str] = []

    def executor(job: object) -> None:
        executed.append(str(getattr(job, "gate_action")))
        stop.set()

    store.enqueue_gate(
        "run-1",
        gate_action="RUN_NEXT_RELEASE",
        payload={},
        requested_by="ou_op",
    )
    runner = DecisionGateRunner(store, executor)
    runner.run_forever(stop)
    assert executed == ["RUN_NEXT_RELEASE"]
    jobs = store.gate_jobs_for_run("run-1")
    assert jobs[0].status == "DONE"


def test_run_forever_rejects_non_event(tmp_path: Path) -> None:
    store = DecisionGateStore(tmp_path / "gates.sqlite3")
    runner = DecisionGateRunner(store, lambda job: None)
    try:
        runner.run_forever(object())  # type: ignore[arg-type]
    except TypeError:
        pass
    else:
        raise AssertionError("run_forever must reject non-Event stop objects")
