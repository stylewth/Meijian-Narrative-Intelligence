"""Transactional local queue for Feishu Bitable write-back."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from typing import Any, Mapping

from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes
from .writeback_targets import WritebackTarget


TABLE_KEYS = ("RUN_LOG", "RESULTS")
RECORD_KINDS = (
    "RUN_LOG",
    "CANDIDATE",
    "SELECTION",
    "HOLDOUT",
    "BLIND_REASSESS",
    "CHECKPOINT",
    "FINAL_SELECTION",
    "MILESTONE",
)
_JOB_STATUSES = ("PENDING", "IN_FLIGHT", "WROTE", "FAILED", "UNKNOWN")
_AMBIGUITY_WINDOW = timedelta(hours=1)
_MAX_ATTEMPTS = 3


def _utc_iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _now_iso() -> str:
    return _utc_iso(datetime.now(timezone.utc))


def _validate_writeback_target(target: Any, table_key: str) -> None:
    """Validate a concrete target by its data contract, not class identity."""

    try:
        target_key = target.table_key
        app_token = target.app_token
        table_id = target.table_id
        field_names = target.field_names
    except AttributeError as exc:
        raise TypeError("target must be WritebackTarget") from exc
    if target_key != table_key:
        raise ValueError("target table_key differs from job table_key")
    if not isinstance(app_token, str) or not app_token.strip():
        raise TypeError("target must include app_token")
    if not isinstance(table_id, str) or not table_id.strip():
        raise TypeError("target must include table_id")
    if not isinstance(field_names, (tuple, list)) or not all(
        isinstance(name, str) and name.strip() for name in field_names
    ):
        raise TypeError("target must include field_names")


def writeback_uuid(run_id: str, record_kind: str, payload_sha256: str) -> str:
    raw = f"{run_id}:{record_kind}:{payload_sha256}".encode("utf-8")
    result = "mj-wb-" + hashlib.sha256(raw).hexdigest()[:42]
    if len(result) > 50:
        raise AssertionError("writeback UUID exceeds Feishu's UUID limit")
    return result


@dataclass(frozen=True)
class WritebackJob:
    job_id: int
    run_id: str
    table_key: str
    record_kind: str
    payload_json: str
    payload_sha256: str
    delivery_uuid: str
    status: str
    attempt_count: int
    first_attempt_at: str | None
    next_attempt_at: str | None
    record_id: str | None
    last_error: str | None
    created_at: str
    written_at: str | None
    demo_run_id: str | None = None
    event_key: str | None = None
    target_app_token: str | None = None
    target_table_id: str | None = None
    target_table_url: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS writeback_jobs (
  job_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  table_key TEXT NOT NULL CHECK (table_key IN ('RUN_LOG','RESULTS')),
  record_kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  delivery_uuid TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL CHECK (status IN ('PENDING','IN_FLIGHT','WROTE','FAILED','UNKNOWN')),
  attempt_count INTEGER NOT NULL DEFAULT 0,
  first_attempt_at TEXT,
  next_attempt_at TEXT,
  record_id TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  written_at TEXT,
  demo_run_id TEXT,
  event_key TEXT,
  target_app_token TEXT,
  target_table_id TEXT,
  target_table_url TEXT,
  UNIQUE(run_id, record_kind, payload_sha256)
) STRICT;

CREATE TABLE IF NOT EXISTS writeback_feedback (
  job_id INTEGER PRIMARY KEY REFERENCES writeback_jobs(job_id),
  status TEXT NOT NULL CHECK (status IN ('PENDING','SENT')),
  created_at TEXT NOT NULL,
  sent_at TEXT
) STRICT;
"""


class WritebackStore:
    """A small SQLite queue for Bitable write-back jobs, one connection per operation."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

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

    def _initialize(self) -> None:
        with self._operation() as connection:
            connection.executescript(_SCHEMA)
            existing_columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(writeback_jobs)")
            }
            for column, column_type in (
                ("demo_run_id", "TEXT"),
                ("event_key", "TEXT"),
                ("target_app_token", "TEXT"),
                ("target_table_id", "TEXT"),
                ("target_table_url", "TEXT"),
            ):
                if column not in existing_columns:
                    connection.execute(
                        f"ALTER TABLE writeback_jobs ADD COLUMN {column} {column_type}"
                    )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS writeback_event_identity "
                "ON writeback_jobs(demo_run_id, event_key, table_key, record_kind) "
                "WHERE demo_run_id IS NOT NULL AND event_key IS NOT NULL"
            )

    @staticmethod
    def _begin_immediate(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _job(row: sqlite3.Row) -> WritebackJob:
        return WritebackJob(
            job_id=row["job_id"],
            run_id=row["run_id"],
            table_key=row["table_key"],
            record_kind=row["record_kind"],
            payload_json=row["payload_json"],
            payload_sha256=row["payload_sha256"],
            delivery_uuid=row["delivery_uuid"],
            status=row["status"],
            attempt_count=row["attempt_count"],
            first_attempt_at=row["first_attempt_at"],
            next_attempt_at=row["next_attempt_at"],
            record_id=row["record_id"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            written_at=row["written_at"],
            demo_run_id=row["demo_run_id"],
            event_key=row["event_key"],
            target_app_token=row["target_app_token"],
            target_table_id=row["target_table_id"],
            target_table_url=row["target_table_url"],
        )

    def enqueue(
        self,
        run_id: str,
        *,
        table_key: str,
        record_kind: str,
        fields: Mapping[str, Any],
        demo_run_id: str | None = None,
        event_key: str | None = None,
        target: WritebackTarget | None = None,
    ) -> WritebackJob:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty")
        if table_key not in TABLE_KEYS:
            raise ValueError(f"table_key must be one of {TABLE_KEYS}")
        if record_kind not in RECORD_KINDS:
            raise ValueError(f"record_kind must be one of {RECORD_KINDS}")
        if not isinstance(fields, Mapping) or not fields:
            raise ValueError("fields must be a non-empty mapping")
        if (demo_run_id is None) != (event_key is None):
            raise ValueError("demo_run_id and event_key must be provided together")
        if demo_run_id is not None:
            if not isinstance(demo_run_id, str) or not demo_run_id.strip():
                raise ValueError("demo_run_id must be non-empty")
            if not isinstance(event_key, str) or not event_key.strip():
                raise ValueError("event_key must be non-empty")
            if target is None:
                raise ValueError("dynamic writeback jobs require a concrete target")
        if target is not None:
            _validate_writeback_target(target, table_key)
        payload_bytes = canonical_json_bytes(dict(fields))
        payload_json = payload_bytes.decode("utf-8")
        payload_sha256 = sha256_bytes(payload_bytes)
        identity = demo_run_id or run_id
        stable_uuid = writeback_uuid(
            identity,
            f"{record_kind}:{event_key or ''}",
            payload_sha256,
        )
        created_at = _now_iso()
        with self._operation() as connection:
            self._begin_immediate(connection)
            if demo_run_id is not None:
                existing = connection.execute(
                    "SELECT * FROM writeback_jobs WHERE demo_run_id = ? "
                    "AND event_key = ? AND table_key = ? AND record_kind = ?",
                    (demo_run_id, event_key, table_key, record_kind),
                ).fetchone()
            else:
                existing = connection.execute(
                    "SELECT * FROM writeback_jobs WHERE run_id = ? AND record_kind = ? "
                    "AND payload_sha256 = ?",
                    (run_id, record_kind, payload_sha256),
                ).fetchone()
            if existing is not None:
                if existing["table_key"] != table_key:
                    raise ValueError("writeback table_key differs for the existing job")
                if demo_run_id is not None and (
                    existing["target_app_token"] != target.app_token
                    or existing["target_table_id"] != target.table_id
                ):
                    raise ValueError("writeback target differs for the existing event")
                if existing["payload_sha256"] == payload_sha256:
                    return self._job(existing)
                if demo_run_id is None:
                    raise ValueError("writeback payload differs for the existing event")
                if existing["status"] in ("IN_FLIGHT", "UNKNOWN"):
                    raise ValueError("writeback event is currently in flight")
                connection.execute(
                    "UPDATE writeback_jobs SET payload_json = ?, payload_sha256 = ?, "
                    "delivery_uuid = ?, status = 'PENDING', attempt_count = 0, "
                    "first_attempt_at = NULL, next_attempt_at = NULL, last_error = NULL, "
                    "written_at = NULL WHERE job_id = ?",
                    (payload_json, payload_sha256, stable_uuid, existing["job_id"]),
                )
                connection.execute(
                    "UPDATE writeback_feedback SET status = 'PENDING', sent_at = NULL "
                    "WHERE job_id = ?",
                    (existing["job_id"],),
                )
                updated = connection.execute(
                    "SELECT * FROM writeback_jobs WHERE job_id = ?",
                    (existing["job_id"],),
                ).fetchone()
                assert updated is not None
                return self._job(updated)
            connection.execute(
                "INSERT INTO writeback_jobs("
                "run_id, table_key, record_kind, payload_json, payload_sha256, "
                "delivery_uuid, status, created_at, demo_run_id, event_key, "
                "target_app_token, target_table_id, target_table_url) "
                "VALUES (?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    table_key,
                    record_kind,
                    payload_json,
                    payload_sha256,
                    stable_uuid,
                    created_at,
                    demo_run_id,
                    event_key,
                    target.app_token if target is not None else None,
                    target.table_id if target is not None else None,
                    target.table_url if target is not None else None,
                ),
            )
            if demo_run_id is not None:
                row = connection.execute(
                    "SELECT * FROM writeback_jobs WHERE demo_run_id = ? "
                    "AND event_key = ? AND table_key = ? AND record_kind = ?",
                    (demo_run_id, event_key, table_key, record_kind),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT * FROM writeback_jobs WHERE run_id = ? AND record_kind = ? "
                    "AND payload_sha256 = ?",
                    (run_id, record_kind, payload_sha256),
                ).fetchone()
            assert row is not None
            return self._job(row)

    def get_job(self, job_id: int) -> WritebackJob:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM writeback_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown writeback job: {job_id}")
            return self._job(row)

    def jobs_for_run(self, run_id: str) -> list[WritebackJob]:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be non-empty")
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT * FROM writeback_jobs WHERE run_id = ? ORDER BY job_id",
                (run_id,),
            ).fetchall()
            return [self._job(row) for row in rows]

    def jobs_for_demo_run(self, demo_run_id: str) -> list[WritebackJob]:
        if not isinstance(demo_run_id, str) or not demo_run_id.strip():
            raise ValueError("demo_run_id must be non-empty")
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT * FROM writeback_jobs WHERE demo_run_id = ? ORDER BY job_id",
                (demo_run_id,),
            ).fetchall()
            return [self._job(row) for row in rows]

    def enqueue_terminal_feedback(self, job_id: int) -> bool:
        """为动态任务登记一次真实完成/最终失败反馈。"""

        if not isinstance(job_id, int) or isinstance(job_id, bool):
            raise ValueError("job_id must be an integer")
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT status, demo_run_id FROM writeback_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown writeback job: {job_id}")
            if row["demo_run_id"] is None or row["status"] not in ("WROTE", "FAILED"):
                return False
            now = _now_iso()
            connection.execute(
                "INSERT INTO writeback_feedback(job_id, status, created_at, sent_at) "
                "VALUES (?, 'PENDING', ?, NULL) "
                "ON CONFLICT(job_id) DO UPDATE SET status = 'PENDING', sent_at = NULL",
                (job_id, now),
            )
            return True

    def pending_feedback_jobs(self) -> list[WritebackJob]:
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT j.* FROM writeback_jobs AS j "
                "JOIN writeback_feedback AS f ON f.job_id = j.job_id "
                "WHERE f.status = 'PENDING' ORDER BY j.job_id"
            ).fetchall()
            return [self._job(row) for row in rows]

    def mark_feedback_sent(self, job_id: int, *, sent_at: datetime | None = None) -> None:
        if not isinstance(job_id, int) or isinstance(job_id, bool):
            raise ValueError("job_id must be an integer")
        timestamp = _now_iso() if sent_at is None else _utc_iso(sent_at)
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE writeback_feedback SET status = 'SENT', sent_at = ? "
                "WHERE job_id = ? AND status = 'PENDING'",
                (timestamp, job_id),
            )
            if result.rowcount != 1:
                raise ValueError("writeback feedback is not PENDING")

    def claim_next_writeback(self, now: datetime) -> WritebackJob | None:
        now_iso = _utc_iso(now)
        ambiguity_cutoff = now.astimezone(timezone.utc) - _AMBIGUITY_WINDOW
        with self._operation() as connection:
            self._begin_immediate(connection)
            inflight = connection.execute(
                "SELECT job_id, first_attempt_at FROM writeback_jobs "
                "WHERE status = 'IN_FLIGHT' AND first_attempt_at IS NOT NULL"
            ).fetchall()
            for row in inflight:
                first_attempt_at = datetime.fromisoformat(row["first_attempt_at"])
                if first_attempt_at < ambiguity_cutoff:
                    connection.execute(
                        "UPDATE writeback_jobs SET status = 'UNKNOWN', "
                        "last_error = ?, next_attempt_at = NULL WHERE job_id = ?",
                        ("write ambiguity window expired", row["job_id"]),
                    )
            row = connection.execute(
                "SELECT * FROM writeback_jobs WHERE status = 'PENDING' "
                "AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
                "ORDER BY job_id LIMIT 1",
                (now_iso,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE writeback_jobs SET status = 'IN_FLIGHT', "
                "attempt_count = attempt_count + 1, "
                "first_attempt_at = COALESCE(first_attempt_at, ?), next_attempt_at = NULL "
                "WHERE job_id = ?",
                (now_iso, row["job_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM writeback_jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            assert claimed is not None
            return self._job(claimed)

    def mark_wrote(self, job_id: int, *, record_id: str, written_at: datetime) -> None:
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError("record_id must be non-empty")
        written_at_iso = _utc_iso(written_at)
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE writeback_jobs SET status = 'WROTE', record_id = ?, "
                "written_at = ?, next_attempt_at = NULL, last_error = NULL "
                "WHERE job_id = ? AND status = 'IN_FLIGHT'",
                (record_id, written_at_iso, job_id),
            )
            if result.rowcount != 1:
                raise ValueError("job is not IN_FLIGHT")

    def mark_retry(self, job_id: int, *, error: str, next_attempt_at: datetime) -> None:
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be non-empty")
        next_attempt_iso = _utc_iso(next_attempt_at)
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT status, attempt_count FROM writeback_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None or row["status"] != "IN_FLIGHT":
                raise ValueError("job is not IN_FLIGHT")
            if row["attempt_count"] >= _MAX_ATTEMPTS:
                connection.execute(
                    "UPDATE writeback_jobs SET status = 'FAILED', last_error = ?, "
                    "next_attempt_at = NULL WHERE job_id = ?",
                    (error, job_id),
                )
            else:
                connection.execute(
                    "UPDATE writeback_jobs SET status = 'PENDING', last_error = ?, "
                    "next_attempt_at = ? WHERE job_id = ?",
                    (error, next_attempt_iso, job_id),
                )

    def mark_failed(self, job_id: int, *, error: str) -> None:
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be non-empty")
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE writeback_jobs SET status = 'FAILED', last_error = ?, "
                "next_attempt_at = NULL WHERE job_id = ? "
                "AND status IN ('IN_FLIGHT','PENDING')",
                (error, job_id),
            )
            if result.rowcount != 1:
                raise ValueError("job is not retryable")

    def retry_failed(self, job_id: int) -> WritebackJob:
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT * FROM writeback_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown writeback job: {job_id}")
            if row["status"] != "FAILED":
                raise ValueError("only FAILED jobs can be retried")
            connection.execute(
                "UPDATE writeback_jobs SET status = 'PENDING', attempt_count = 0, "
                "first_attempt_at = NULL, next_attempt_at = NULL, "
                "last_error = NULL, written_at = NULL WHERE job_id = ?",
                (job_id,),
            )
            connection.execute(
                "UPDATE writeback_feedback SET status = 'PENDING', sent_at = NULL "
                "WHERE job_id = ?",
                (job_id,),
            )
            retried = connection.execute(
                "SELECT * FROM writeback_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            assert retried is not None
            return self._job(retried)


__all__ = [
    "RECORD_KINDS",
    "TABLE_KEYS",
    "WritebackJob",
    "WritebackStore",
    "writeback_uuid",
]
