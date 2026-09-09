"""写回队列、表结构盘点与写回工人的合同测试。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock
import time
from types import SimpleNamespace

import pytest

from src.integrations.feishu.writeback_store import WritebackStore, writeback_uuid
from src.integrations.feishu.writeback_targets import (
    DYNAMIC_RESULTS_FIELDS,
    DYNAMIC_RUN_LOG_FIELDS,
    WritebackProvisioningError,
    provision_demo_writeback_targets,
    inventory_writeback_target,
    inventory_writeback_targets,
)
from src.integrations.feishu.writeback_worker import WritebackWorker


RUN_LOG_FIELDS = [
    {"field_name": name}
    for name in ("run_id", "action", "actor_open_id", "summary", "created_at")
]
RESULTS_FIELDS = [
    {"field_name": name}
    for name in ("run_id", "record_kind", "record_key", "content", "created_at")
]


class FakeBitableClient:
    def __init__(self, *, fields=None, fail_calls=0, fail_table_call=None):
        self.field_map = fields or {}
        self.fail_calls = fail_calls
        self.fail_table_call = fail_table_call
        self.created: list[tuple[str, str, dict]] = []
        self.updated: list[tuple[str, str, str, dict]] = []
        self.created_tables: list[tuple[str, str, list[dict]]] = []
        self.table_attempts = 0
        self.resolve_calls = 0
        self._next = 1

    def get_fields(self, app_token: str, table_id: str):
        return self.field_map[(app_token, table_id)]

    def resolve_wiki_app_token(self, wiki_token: str) -> str:
        self.resolve_calls += 1
        return "app_from_wiki"

    def create_record(self, app_token: str, table_id: str, fields: dict) -> str:
        if self.fail_calls > 0:
            self.fail_calls -= 1
            raise RuntimeError("feishu down")
        self.created.append((app_token, table_id, dict(fields)))
        record_id = f"rec_{self._next}"
        self._next += 1
        return record_id

    def update_record(
        self, app_token: str, table_id: str, record_id: str, fields: dict
    ) -> str:
        self.updated.append((app_token, table_id, record_id, dict(fields)))
        return record_id

    def create_table(self, app_token: str, name: str, fields: list[dict]) -> str:
        self.table_attempts += 1
        if self.table_attempts == self.fail_table_call:
            raise RuntimeError("table creation failed")
        self.created_tables.append((app_token, name, list(fields)))
        return f"tbl_new_{len(self.created_tables)}"


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def __call__(self) -> datetime:
        return self.now


def _sample_fields(run_id: str = "run-1") -> dict:
    return {
        "run_id": run_id,
        "action": "SUBMIT_SELECTION",
        "actor_open_id": "ou_demo",
        "summary": "人工确认候选 C1",
        "created_at": "2026-08-16T12:00:00+00:00",
    }


# ---------- WritebackStore ----------


def test_enqueue_is_idempotent_for_identical_payload(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")

    first = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )
    second = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )

    assert first.job_id == second.job_id
    assert second.status == "PENDING"
    assert len(store.jobs_for_run("run-1")) == 1


def test_different_payloads_get_separate_jobs(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")

    store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )
    store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields=_sample_fields(run_id="run-2"),
    )

    assert len(store.jobs_for_run("run-1")) == 2


def test_enqueue_rejects_invalid_arguments(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")

    with pytest.raises(ValueError):
        store.enqueue(
            "run-1", table_key="NOPE", record_kind="RUN_LOG", fields=_sample_fields()
        )
    with pytest.raises(ValueError):
        store.enqueue(
            "run-1", table_key="RUN_LOG", record_kind="NOPE", fields=_sample_fields()
        )
    with pytest.raises(ValueError):
        store.enqueue("run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields={})


def test_claim_marks_in_flight_and_counts_attempts(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    now = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    job = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )

    claimed = store.claim_next_writeback(now)

    assert claimed is not None
    assert claimed.job_id == job.job_id
    assert claimed.status == "IN_FLIGHT"
    assert claimed.attempt_count == 1
    assert store.claim_next_writeback(now) is None


def test_mark_wrote_finalizes_and_prevents_reclaim(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    now = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    job = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )
    store.claim_next_writeback(now)

    store.mark_wrote(job.job_id, record_id="rec_9", written_at=now)

    wrote = store.get_job(job.job_id)
    assert wrote.status == "WROTE"
    assert wrote.record_id == "rec_9"
    assert store.claim_next_writeback(now + timedelta(seconds=5)) is None


def test_retry_fails_job_after_three_attempts(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    now = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    job = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )

    for attempt in range(3):
        store.claim_next_writeback(now + timedelta(seconds=attempt * 10))
        store.mark_retry(
            job.job_id,
            error=f"boom-{attempt}",
            next_attempt_at=now + timedelta(seconds=attempt * 10 + 1),
        )

    failed = store.get_job(job.job_id)
    assert failed.status == "FAILED"
    assert failed.last_error == "boom-2"
    assert store.retry_failed(job.job_id).status == "PENDING"
    assert store.get_job(job.job_id).attempt_count == 0


def test_ambiguity_window_marks_stale_in_flight_unknown(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    start = datetime(2026, 8, 16, 12, 0, 0, tzinfo=timezone.utc)
    job = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )
    store.claim_next_writeback(start)

    store.claim_next_writeback(start + timedelta(hours=2))

    assert store.get_job(job.job_id).status == "UNKNOWN"


def test_writeback_uuid_is_deterministic_and_short():
    payload_sha = "a" * 64
    first = writeback_uuid("run-1", "RUN_LOG", payload_sha)
    assert first == writeback_uuid("run-1", "RUN_LOG", payload_sha)
    assert first != writeback_uuid("run-2", "RUN_LOG", payload_sha)
    assert len(first) <= 50


# ---------- writeback_targets ----------


def _base_url() -> str:
    return "https://demo.feishu.cn/base/app_demo?table=tbl_demo"


def _wiki_url() -> str:
    return "https://demo.feishu.cn/wiki/wiki_demo?table=tbl_demo"


def test_inventory_accepts_base_url_with_all_fields():
    client = FakeBitableClient(fields={("app_demo", "tbl_demo"): RUN_LOG_FIELDS})

    target = inventory_writeback_target(
        client, table_key="RUN_LOG", url=_base_url()
    )

    assert target.app_token == "app_demo"
    assert target.table_id == "tbl_demo"
    assert set(RUN_LOG_FIELDS[0].values()) <= set(target.field_names)


def test_inventory_resolves_wiki_url():
    client = FakeBitableClient(fields={("app_from_wiki", "tbl_demo"): RESULTS_FIELDS})

    target = inventory_writeback_target(
        client, table_key="RESULTS", url=_wiki_url()
    )

    assert target.app_token == "app_from_wiki"


def test_inventory_missing_fields_are_listed_loudly():
    client = FakeBitableClient(
        fields={
            ("app_demo", "tbl_demo"): [
                {"field_name": name}
                for name in ("action", "actor_open_id", "summary", "created_at")
            ]
        }
    )

    with pytest.raises(ValueError) as excinfo:
        inventory_writeback_target(client, table_key="RUN_LOG", url=_base_url())

    message = str(excinfo.value)
    assert "run_id" in message


def test_inventory_targets_skips_unconfigured_urls():
    client = FakeBitableClient()

    assert inventory_writeback_targets(
        client, run_log_url=None, results_url=None
    ) == {}


def test_writeback_store_accepts_reload_like_target(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    target = SimpleNamespace(
        table_key="RUN_LOG",
        app_token="app_demo",
        table_id="tbl_demo",
        field_names=("run_id", "demo_run_id"),
        table_url="https://demo.feishu.cn/base/app_demo?table=tbl_demo",
    )

    job = store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields={"run_id": "run-1", "demo_run_id": "demo-a"},
        demo_run_id="demo-a",
        event_key="progress:current:log",
        target=target,
    )

    assert job.target_table_id == "tbl_demo"


# ---------- WritebackWorker ----------


def _targets():
    from src.integrations.feishu.writeback_targets import WritebackTarget

    return {
        "RUN_LOG": WritebackTarget(
            table_key="RUN_LOG",
            app_token="app_demo",
            table_id="tbl_demo",
            field_names=tuple(field["field_name"] for field in RUN_LOG_FIELDS),
        )
    }


def test_worker_writes_record_and_marks_wrote(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    client = FakeBitableClient()
    clock = FakeClock()
    worker = WritebackWorker(store, client, _targets(), clock=clock)
    job = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )

    assert worker.run_once() is True

    wrote = store.get_job(job.job_id)
    assert wrote.status == "WROTE"
    assert wrote.record_id == "rec_1"
    assert client.created == [("app_demo", "tbl_demo", _sample_fields())]


def test_worker_retries_then_succeeds(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    client = FakeBitableClient(fail_calls=1)
    clock = FakeClock()
    worker = WritebackWorker(store, client, _targets(), clock=clock)
    job = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )

    assert worker.run_once() is True
    pending = store.get_job(job.job_id)
    assert pending.status == "PENDING"
    assert "feishu down" in (pending.last_error or "")

    clock.advance(10)
    assert worker.run_once() is True
    assert store.get_job(job.job_id).status == "WROTE"
    assert store.get_job(job.job_id).attempt_count == 2


def test_worker_fails_job_after_three_attempts(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    client = FakeBitableClient(fail_calls=99)
    clock = FakeClock()
    worker = WritebackWorker(store, client, _targets(), clock=clock)
    job = store.enqueue(
        "run-1", table_key="RUN_LOG", record_kind="RUN_LOG", fields=_sample_fields()
    )

    for _ in range(3):
        assert worker.run_once() is True
        clock.advance(10)

    failed = store.get_job(job.job_id)
    assert failed.status == "FAILED"
    assert "feishu down" in (failed.last_error or "")


def test_worker_missing_target_fails_loud(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    client = FakeBitableClient()
    worker = WritebackWorker(store, client, _targets(), clock=FakeClock())
    job = store.enqueue(
        "run-1",
        table_key="RESULTS",
        record_kind="CANDIDATE",
        fields={"run_id": "run-1", "record_kind": "CANDIDATE"},
    )

    with pytest.raises(LookupError):
        worker.run_once()

    failed = store.get_job(job.job_id)
    assert failed.status == "FAILED"
    assert "RESULTS" in (failed.last_error or "")


def test_worker_requires_at_least_one_target(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")

    with pytest.raises(ValueError):
        WritebackWorker(store, FakeBitableClient(), {}, clock=FakeClock())


def test_dynamic_connection_creates_two_tables_in_the_same_base():
    client = FakeBitableClient()

    targets = provision_demo_writeback_targets(
        client,
        run_log_url=_base_url(),
        results_url="https://demo.feishu.cn/base/app_demo?table=tbl_results&view=vew_old",
        demo_run_id="demo-12345678",
        started_at="2026-09-09T08:30:00+00:00",
    )

    assert set(targets) == {"RUN_LOG", "RESULTS"}
    assert [item[0] for item in client.created_tables] == ["app_demo", "app_demo"]
    assert client.created_tables[0][1].endswith("-日志")
    assert client.created_tables[1][1].endswith("-结果")
    assert {field["field_name"] for field in client.created_tables[0][2]} == set(
        DYNAMIC_RUN_LOG_FIELDS
    )
    assert {field["field_name"] for field in client.created_tables[1][2]} == set(
        DYNAMIC_RESULTS_FIELDS
    )
    assert targets["RUN_LOG"].table_id == "tbl_new_1"
    assert targets["RUN_LOG"].table_url.endswith("table=tbl_new_1")
    assert targets["RESULTS"].table_url.endswith("table=tbl_new_2")
    assert "view=" not in targets["RUN_LOG"].table_url


def test_dynamic_connection_creates_both_tables_concurrently():
    class TimedClient(FakeBitableClient):
        def __init__(self):
            super().__init__()
            self._active = 0
            self._lock = Lock()
            self.max_active = 0

        def create_table(self, app_token: str, name: str, fields: list[dict]) -> str:
            with self._lock:
                self._active += 1
                self.max_active = max(self.max_active, self._active)
            try:
                time.sleep(0.05)
                return super().create_table(app_token, name, fields)
            finally:
                with self._lock:
                    self._active -= 1

    client = TimedClient()

    provision_demo_writeback_targets(
        client,
        run_log_url=_base_url(),
        results_url="https://demo.feishu.cn/base/app_demo?table=tbl_results",
        demo_run_id="demo-12345678",
        started_at="2026-09-09T08:30:00+00:00",
    )

    assert client.max_active == 2


def test_dynamic_connection_resolves_a_shared_wiki_base_once():
    client = FakeBitableClient()

    provision_demo_writeback_targets(
        client,
        run_log_url=_wiki_url(),
        results_url="https://demo.feishu.cn/wiki/wiki_demo?table=tbl_results",
        demo_run_id="demo-12345678",
        started_at="2026-09-09T08:30:00+00:00",
    )

    assert client.resolve_calls == 1


def test_dynamic_connection_rejects_different_bases():
    client = FakeBitableClient()

    with pytest.raises(ValueError, match="同一个飞书 Base"):
        provision_demo_writeback_targets(
            client,
            run_log_url=_base_url(),
            results_url="https://demo.feishu.cn/base/app_other?table=tbl_results",
            demo_run_id="demo-12345678",
            started_at="2026-09-09T08:30:00+00:00",
        )


def test_dynamic_connection_reports_partial_table_without_activating_it():
    client = FakeBitableClient(fail_table_call=2)

    with pytest.raises(WritebackProvisioningError, match="未激活"):
        provision_demo_writeback_targets(
            client,
            run_log_url=_base_url(),
            results_url="https://demo.feishu.cn/base/app_demo?table=tbl_results",
            demo_run_id="demo-12345678",
            started_at="2026-09-09T08:30:00+00:00",
        )

    assert len(client.created_tables) == 1


def test_dynamic_jobs_bind_session_target_and_event_key(tmp_path):
    from src.integrations.feishu.writeback_targets import WritebackTarget

    store = WritebackStore(tmp_path / "wb.sqlite3")
    target_a = WritebackTarget(
        table_key="RUN_LOG",
        app_token="app_a",
        table_id="tbl_a",
        field_names=tuple(DYNAMIC_RUN_LOG_FIELDS),
        table_url="https://demo.feishu.cn/base/app_a?table=tbl_a",
    )
    target_b = WritebackTarget(
        table_key="RUN_LOG",
        app_token="app_b",
        table_id="tbl_b",
        field_names=tuple(DYNAMIC_RUN_LOG_FIELDS),
        table_url="https://demo.feishu.cn/base/app_b?table=tbl_b",
    )
    fields_a = {
        "run_id": "run-1",
        "demo_run_id": "demo-a",
        "action": "MILESTONE",
        "actor_open_id": "streamlit-local",
        "summary": "节点 1",
        "demo_started_at": "2026-09-09T08:30:00+00:00",
        "created_at": "2026-09-09T08:30:01+00:00",
    }
    fields_b = dict(fields_a, demo_run_id="demo-b")

    first = store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields=fields_a,
        demo_run_id="demo-a",
        event_key="milestone:1:log",
        target=target_a,
    )
    duplicate = store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields=fields_a,
        demo_run_id="demo-a",
        event_key="milestone:1:log",
        target=target_a,
    )
    second = store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields=fields_b,
        demo_run_id="demo-b",
        event_key="milestone:1:log",
        target=target_b,
    )

    assert duplicate.job_id == first.job_id
    assert second.job_id != first.job_id
    assert first.target_table_id == "tbl_a"
    assert second.target_table_id == "tbl_b"


def test_dynamic_event_overwrite_reuses_the_existing_feishu_record(tmp_path):
    store = WritebackStore(tmp_path / "wb.sqlite3")
    client = FakeBitableClient()
    target = _targets()["RUN_LOG"]
    first_fields = dict(
        _sample_fields(),
        demo_run_id="demo-a",
        demo_started_at="2026-09-09T08:30:00+00:00",
    )
    first = store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields=first_fields,
        demo_run_id="demo-a",
        event_key="progress:current:log",
        target=target,
    )
    worker = WritebackWorker(store, client, _targets(), clock=FakeClock())

    assert worker.run_once() is True
    wrote = store.get_job(first.job_id)
    assert wrote.status == "WROTE"
    assert wrote.record_id == "rec_1"

    latest = store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields=dict(first_fields, summary="当前进度已更新"),
        demo_run_id="demo-a",
        event_key="progress:current:log",
        target=target,
    )

    assert latest.job_id == first.job_id
    assert latest.status == "PENDING"
    assert latest.record_id == "rec_1"
    assert worker.run_once() is True
    assert client.updated == [
        (
            "app_demo",
            "tbl_demo",
            "rec_1",
            dict(first_fields, summary="当前进度已更新"),
        )
    ]


def test_worker_records_real_completion_feedback_for_dynamic_job(tmp_path):
    from src.integrations.feishu.writeback_targets import WritebackTarget

    store = WritebackStore(tmp_path / "wb.sqlite3")
    target = WritebackTarget(
        table_key="RUN_LOG",
        app_token="app_demo",
        table_id="tbl_demo",
        field_names=tuple(DYNAMIC_RUN_LOG_FIELDS),
        table_url="https://demo.feishu.cn/base/app_demo?table=tbl_demo",
    )
    worker = WritebackWorker(
        store,
        FakeBitableClient(),
        {"RUN_LOG": target},
        clock=FakeClock(),
    )
    job = store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields={
            "run_id": "run-1",
            "demo_run_id": "demo-a",
            "action": "MILESTONE",
            "actor_open_id": "streamlit-local",
            "summary": "节点 1",
            "demo_started_at": "2026-09-09T08:30:00+00:00",
            "created_at": "2026-09-09T08:30:01+00:00",
        },
        demo_run_id="demo-a",
        event_key="milestone:1:log",
        target=target,
    )

    assert worker.run_once() is True
    assert [item.job_id for item in store.pending_feedback_jobs()] == [job.job_id]
    store.mark_feedback_sent(job.job_id)
    assert store.pending_feedback_jobs() == []


def test_feedback_loop_reports_actual_write_and_can_retry_failure(tmp_path):
    from tools.run_feishu_bot import (
        WritebackFeedbackLoop,
        handle_writeback_retry_action,
    )
    from src.integrations.feishu.writeback_targets import WritebackTarget

    store = WritebackStore(tmp_path / "wb.sqlite3")
    target = WritebackTarget(
        table_key="RUN_LOG",
        app_token="app_demo",
        table_id="tbl_demo",
        field_names=tuple(DYNAMIC_RUN_LOG_FIELDS),
        table_url="https://demo.feishu.cn/base/app_demo?table=tbl_demo",
    )
    job = store.enqueue(
        "run-1",
        table_key="RUN_LOG",
        record_kind="RUN_LOG",
        fields={
            "run_id": "run-1",
            "demo_run_id": "demo-a",
            "action": "MILESTONE",
            "actor_open_id": "streamlit-local",
            "summary": "节点 1",
            "demo_started_at": "2026-09-09T08:30:00+00:00",
            "created_at": "2026-09-09T08:30:01+00:00",
        },
        demo_run_id="demo-a",
        event_key="milestone:1:log",
        target=target,
    )
    store.claim_next_writeback(datetime(2026, 9, 9, 8, 30, tzinfo=timezone.utc))
    store.mark_failed(job.job_id, error="连接超时")
    store.enqueue_terminal_feedback(job.job_id)

    class FakeCardClient:
        def __init__(self) -> None:
            self.cards = []

        def send_card(self, chat_id, card, *, uuid):
            self.cards.append((chat_id, card, uuid))
            return "message-1"

    client = FakeCardClient()
    WritebackFeedbackLoop(store, client, "oc_demo")._tick()

    assert len(client.cards) == 1
    card = client.cards[0][1]
    assert "0/1" in str(card)
    assert "tbl_demo" in str(card)
    assert "WRITEBACK_RETRY" in str(card)
    retry = handle_writeback_retry_action(
        {"type": "WRITEBACK_RETRY", "job_id": job.job_id},
        operator_open_id="ou_operator",
        allowed_operator_ids={"ou_operator"},
        store=store,
    )
    assert retry.toast_type == "success"
    assert store.get_job(job.job_id).status == "PENDING"
