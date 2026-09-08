"""SQLite queue for decision gate jobs requested from Feishu."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from threading import Event
from typing import Any, Mapping

from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes


GATE_JOB_STATUSES = ("PENDING", "IN_FLIGHT", "DONE", "FAILED")


def _utc_iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _now_iso() -> str:
    return _utc_iso(datetime.now(timezone.utc))


@dataclass(frozen=True)
class GateJob:
    job_id: int
    run_id: str
    gate_action: str
    payload_json: str
    payload_sha256: str
    requested_by: str
    status: str
    last_error: str | None
    created_at: str
    finished_at: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS gate_jobs (
  job_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  gate_action TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  requested_by TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('PENDING','IN_FLIGHT','DONE','FAILED')),
  last_error TEXT,
  created_at TEXT NOT NULL,
  finished_at TEXT,
  UNIQUE(run_id, gate_action, payload_sha256)
) STRICT;
"""


class DecisionGateStore:
    """Gate request queue; one non-terminal job per run at a time."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._operation() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.database_path),
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @contextmanager
    def _operation(self):
        connection = self._connect()
        try:
            yield connection
            if connection.in_transaction:
                connection.commit()
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _begin_immediate(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _job(row: sqlite3.Row) -> GateJob:
        return GateJob(
            job_id=row["job_id"],
            run_id=row["run_id"],
            gate_action=row["gate_action"],
            payload_json=row["payload_json"],
            payload_sha256=row["payload_sha256"],
            requested_by=row["requested_by"],
            status=row["status"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
        )

    def enqueue_gate(
        self,
        run_id: str,
        *,
        gate_action: str,
        payload: Mapping[str, Any],
        requested_by: str,
    ) -> GateJob:
        for name, value in (
            ("run_id", run_id),
            ("gate_action", gate_action),
            ("requested_by", requested_by),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(payload, Mapping):
            raise ValueError("payload must be a mapping")
        payload_bytes = canonical_json_bytes(dict(payload))
        payload_json = payload_bytes.decode("utf-8")
        payload_sha256 = sha256_bytes(payload_bytes)
        created_at = _now_iso()
        with self._operation() as connection:
            self._begin_immediate(connection)
            existing = connection.execute(
                "SELECT * FROM gate_jobs WHERE run_id = ? AND gate_action = ? "
                "AND payload_sha256 = ?",
                (run_id, gate_action, payload_sha256),
            ).fetchone()
            if existing is not None:
                if existing["status"] in ("PENDING", "IN_FLIGHT"):
                    return self._job(existing)
                raise ValueError("同一门请求已执行过；请确认结果后再发起新请求")
            busy = connection.execute(
                "SELECT job_id FROM gate_jobs WHERE run_id = ? "
                "AND status IN ('PENDING','IN_FLIGHT') LIMIT 1",
                (run_id,),
            ).fetchone()
            if busy is not None:
                raise ValueError(f"该决策 run 已有门操作在排队或执行（job {busy['job_id']}）")
            connection.execute(
                "INSERT INTO gate_jobs("
                "run_id, gate_action, payload_json, payload_sha256, requested_by, "
                "status, created_at) VALUES (?, ?, ?, ?, ?, 'PENDING', ?)",
                (
                    run_id,
                    gate_action,
                    payload_json,
                    payload_sha256,
                    requested_by,
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM gate_jobs WHERE run_id = ? AND gate_action = ? "
                "AND payload_sha256 = ?",
                (run_id, gate_action, payload_sha256),
            ).fetchone()
            assert row is not None
            return self._job(row)

    def get_job(self, job_id: int) -> GateJob:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM gate_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown gate job: {job_id}")
            return self._job(row)

    def gate_jobs_for_run(self, run_id: str) -> list[GateJob]:
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT * FROM gate_jobs WHERE run_id = ? ORDER BY job_id",
                (run_id,),
            ).fetchall()
            return [self._job(row) for row in rows]

    def claim_next_gate(self, now: datetime) -> GateJob | None:
        now_iso = _utc_iso(now)
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT * FROM gate_jobs WHERE status = 'PENDING' "
                "ORDER BY job_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE gate_jobs SET status = 'IN_FLIGHT' WHERE job_id = ? "
                "AND status = 'PENDING'",
                (row["job_id"],),
            )
            claimed = connection.execute(
                "SELECT * FROM gate_jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            assert claimed is not None
            return self._job(claimed)

    def mark_done(self, job_id: int, *, finished_at: datetime) -> None:
        finished_at_iso = _utc_iso(finished_at)
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE gate_jobs SET status = 'DONE', last_error = NULL, "
                "finished_at = ? WHERE job_id = ? AND status = 'IN_FLIGHT'",
                (finished_at_iso, job_id),
            )
            if result.rowcount != 1:
                raise ValueError("gate job is not IN_FLIGHT")

    def mark_gate_failed(self, job_id: int, *, error: str) -> None:
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be non-empty")
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE gate_jobs SET status = 'FAILED', last_error = ?, "
                "finished_at = ? WHERE job_id = ? AND status = 'IN_FLIGHT'",
                (error, _now_iso(), job_id),
            )
            if result.rowcount != 1:
                raise ValueError("gate job is not IN_FLIGHT")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DecisionGateRunner:
    """Execute at most one claimed gate job per ``run_once`` via the injected executor.

    门执行失败即终态 FAILED（LLM 门不自动重试），错误全文保留；
    排查后由人工重新发起门请求。
    """

    def __init__(
        self,
        store: DecisionGateStore,
        executor: Callable[[GateJob], None],
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not callable(executor):
            raise TypeError("executor must be callable")
        self._store = store
        self._executor = executor
        self._clock = clock

    def run_once(self) -> bool:
        now = self._now()
        job = self._store.claim_next_gate(now)
        if job is None:
            return False
        try:
            self._executor(job)
        except Exception as exc:
            self._store.mark_gate_failed(job.job_id, error=str(exc))
            return True
        self._store.mark_done(job.job_id, finished_at=now)
        return True

    def run_forever(self, stop_event: Any) -> None:
        """持续消费门队列；空闲时短暂等待，stop_event 置位即退出。"""

        from threading import Event

        if not isinstance(stop_event, Event):
            raise TypeError("stop_event must be a threading.Event")
        while not stop_event.is_set():
            if not self.run_once():
                stop_event.wait(0.5)

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


__all__ = [
    "DecisionGateRunner",
    "DecisionGateStore",
    "GateJob",
    "GATE_JOB_STATUSES",
]
