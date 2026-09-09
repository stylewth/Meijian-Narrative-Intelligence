"""Transactional local state for the Feishu demo notification workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Any, Mapping, TYPE_CHECKING

from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes
from .writeback_targets import WritebackTarget

if TYPE_CHECKING:
    from src.services.demo_notifications import DemoNotificationNode, DemoNotificationSnapshot


_JOB_STATUSES = ("PENDING", "IN_FLIGHT", "SENT", "FAILED", "UNKNOWN")
_REPLAY_STATUSES = (
    "PENDING_CARD",
    "CARD_CREATE_IN_FLIGHT",
    "AWAITING_REPLY",
    "IDLE",
    "PENDING",
    "IN_FLIGHT",
    "FAILED",
    "EXPIRED",
)


def _utc_iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _now_iso() -> str:
    return _utc_iso(datetime.now(timezone.utc))


def _validate_writeback_target(target: Any, table_key: str) -> None:
    """Validate the target data contract without relying on module identity.

    Streamlit can reload one integration module while another still holds the
    previous ``WritebackTarget`` class object.  The target is a data boundary,
    so validate its fields rather than rejecting a valid object solely because
    its class identity came from the previous module generation.
    """

    try:
        target_key = target.table_key
        app_token = target.app_token
        table_id = target.table_id
        field_names = target.field_names
        table_url = target.table_url
    except AttributeError as exc:
        raise TypeError(f"{table_key} target must be WritebackTarget") from exc
    if target_key != table_key:
        raise ValueError(f"{table_key} target key differs")
    if not isinstance(app_token, str) or not app_token.strip():
        raise TypeError(f"{table_key} target must include app_token")
    if not isinstance(table_id, str) or not table_id.strip():
        raise TypeError(f"{table_key} target must include table_id")
    if not isinstance(field_names, (tuple, list)) or not all(
        isinstance(name, str) and name.strip() for name in field_names
    ):
        raise TypeError(f"{table_key} target must include field_names")
    if not isinstance(table_url, str) or not table_url.strip():
        raise ValueError(f"{table_key} target must include table_url")


def _node_key(node: Any) -> str:
    value = getattr(node, "value", node)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("node must be a non-empty string")
    return value


def delivery_uuid(session_id: str, node_key: str, payload_sha256: str) -> str:
    raw = f"{session_id}:{node_key}:{payload_sha256}".encode("utf-8")
    result = "mj-" + hashlib.sha256(raw).hexdigest()[:47]
    if len(result) > 50:
        raise AssertionError("delivery UUID exceeds Feishu's UUID limit")
    return result


@dataclass(frozen=True)
class DemoSession:
    session_id: str
    source_run_id: str
    created_by: str
    created_at: str
    status: str
    connected_at: str | None = None


@dataclass(frozen=True)
class NotificationJob:
    job_id: int
    session_id: str
    node_key: str
    ordinal: int
    payload_json: str
    payload_sha256: str
    delivery_uuid: str
    test_mode: bool
    status: str
    attempt_count: int
    first_attempt_at: str | None
    next_attempt_at: str | None
    message_id: str | None
    last_error: str | None
    created_at: str
    sent_at: str | None


@dataclass(frozen=True)
class ReplayCard:
    replay_id: str
    session_id: str
    source_message_id: str
    reply_uuid: str
    card_id: str | None
    message_id: str | None
    current_ordinal: int
    requested_ordinal: int | None
    sequence: int
    update_status: str
    update_uuid: str | None
    last_error: str | None
    updated_at: str


@dataclass(frozen=True)
class DemoCommand:
    command_id: int
    session_id: str
    command: str
    target_node: int
    requested_by: str
    status: str
    created_at: str
    consumed_at: str | None
    result_text: str | None = None
    feedback_status: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS demo_commands (
  command_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
  command TEXT NOT NULL CHECK (command IN ('ADVANCE_TO_NODE','ADVANCE_STEP')),
  result_text TEXT,
  feedback_status TEXT CHECK (feedback_status IN ('PENDING','SENT') OR feedback_status IS NULL),
  target_node INTEGER NOT NULL CHECK (target_node BETWEEN 1 AND 8),
  requested_by TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('PENDING','CONSUMED')),
  created_at TEXT NOT NULL,
  consumed_at TEXT,
  session_node_key TEXT NOT NULL,
  UNIQUE(session_node_key)
) STRICT;

CREATE TABLE IF NOT EXISTS demo_progress (
  session_id TEXT PRIMARY KEY REFERENCES demo_sessions(session_id),
  progress_text TEXT NOT NULL,
  updated_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS demo_sessions (
  session_id TEXT PRIMARY KEY,
  source_run_id TEXT NOT NULL,
  created_by TEXT NOT NULL CHECK (created_by = 'streamlit-local'),
  created_at TEXT NOT NULL,
  connected_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('ACTIVE','COMPLETE'))
) STRICT;
CREATE UNIQUE INDEX IF NOT EXISTS one_active_session
ON demo_sessions(status) WHERE status = 'ACTIVE';

CREATE TABLE IF NOT EXISTS demo_writeback_targets (
  session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
  table_key TEXT NOT NULL CHECK (table_key IN ('RUN_LOG','RESULTS')),
  app_token TEXT NOT NULL,
  table_id TEXT NOT NULL,
  table_url TEXT NOT NULL,
  field_names_json TEXT NOT NULL,
  PRIMARY KEY(session_id, table_key)
) STRICT;

CREATE TABLE IF NOT EXISTS notification_jobs (
  job_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
  node_key TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal BETWEEN 1 AND 8),
  payload_json TEXT NOT NULL,
  payload_sha256 TEXT NOT NULL,
  delivery_uuid TEXT NOT NULL UNIQUE,
  test_mode INTEGER NOT NULL DEFAULT 0 CHECK (test_mode IN (0,1)),
  status TEXT NOT NULL CHECK (status IN ('PENDING','IN_FLIGHT','SENT','FAILED','UNKNOWN')),
  attempt_count INTEGER NOT NULL DEFAULT 0,
  first_attempt_at TEXT,
  next_attempt_at TEXT,
  message_id TEXT,
  last_error TEXT,
  created_at TEXT NOT NULL,
  sent_at TEXT,
  UNIQUE(session_id, node_key)
) STRICT;

CREATE TABLE IF NOT EXISTS replay_cards (
  replay_id TEXT PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES demo_sessions(session_id),
  source_message_id TEXT NOT NULL UNIQUE,
  reply_uuid TEXT NOT NULL UNIQUE,
  card_id TEXT UNIQUE,
  message_id TEXT UNIQUE,
  current_ordinal INTEGER NOT NULL CHECK (current_ordinal BETWEEN 1 AND 8),
  requested_ordinal INTEGER CHECK (requested_ordinal BETWEEN 1 AND 8),
  sequence INTEGER NOT NULL DEFAULT 0 CHECK (sequence >= 0),
  update_status TEXT NOT NULL CHECK (update_status IN ('PENDING_CARD','CARD_CREATE_IN_FLIGHT','AWAITING_REPLY','IDLE','PENDING','IN_FLIGHT','FAILED','EXPIRED')),
  update_uuid TEXT,
  last_error TEXT,
  updated_at TEXT NOT NULL
) STRICT;

CREATE TABLE IF NOT EXISTS processed_events (
  event_key TEXT PRIMARY KEY,
  processed_at TEXT NOT NULL
) STRICT;
"""


class NotificationStore:
    """A small SQLite repository whose public operations each own one connection."""

    def __init__(
        self,
        database_path: str | Path,
        *,
        replay_update_stale_after: timedelta = timedelta(minutes=5),
    ) -> None:
        self.database_path = Path(database_path)
        if not isinstance(replay_update_stale_after, timedelta) or replay_update_stale_after <= timedelta(0):
            raise ValueError("replay_update_stale_after must be positive")
        self.replay_update_stale_after = replay_update_stale_after
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
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='demo_commands'"
            ).fetchone()
            if row is not None and "ADVANCE_STEP" not in str(row["sql"]):
                connection.execute("DROP TABLE demo_commands")
            connection.executescript(_SCHEMA)
            columns = {
                item["name"]
                for item in connection.execute(
                    "PRAGMA table_info(demo_sessions)"
                ).fetchall()
            }
            if "connected_at" not in columns:
                connection.execute(
                    "ALTER TABLE demo_sessions ADD COLUMN connected_at TEXT"
                )

    @staticmethod
    def _begin_immediate(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _session(row: sqlite3.Row) -> DemoSession:
        return DemoSession(
            session_id=row["session_id"],
            source_run_id=row["source_run_id"],
            created_by=row["created_by"],
            created_at=row["created_at"],
            status=row["status"],
            connected_at=row["connected_at"],
        )

    @staticmethod
    def _job(row: sqlite3.Row) -> NotificationJob:
        return NotificationJob(
            job_id=row["job_id"],
            session_id=row["session_id"],
            node_key=row["node_key"],
            ordinal=row["ordinal"],
            payload_json=row["payload_json"],
            payload_sha256=row["payload_sha256"],
            delivery_uuid=row["delivery_uuid"],
            test_mode=bool(row["test_mode"]),
            status=row["status"],
            attempt_count=row["attempt_count"],
            first_attempt_at=row["first_attempt_at"],
            next_attempt_at=row["next_attempt_at"],
            message_id=row["message_id"],
            last_error=row["last_error"],
            created_at=row["created_at"],
            sent_at=row["sent_at"],
        )

    @staticmethod
    def _replay(row: sqlite3.Row) -> ReplayCard:
        return ReplayCard(
            replay_id=row["replay_id"],
            session_id=row["session_id"],
            source_message_id=row["source_message_id"],
            reply_uuid=row["reply_uuid"],
            card_id=row["card_id"],
            message_id=row["message_id"],
            current_ordinal=row["current_ordinal"],
            requested_ordinal=row["requested_ordinal"],
            sequence=row["sequence"],
            update_status=row["update_status"],
            update_uuid=row["update_uuid"],
            last_error=row["last_error"],
            updated_at=row["updated_at"],
        )


    def enqueue_demo_command(
        self,
        session_id: str,
        *,
        command: str,
        target_node: int,
        requested_by: str,
    ) -> "DemoCommand":
        """登记一条远端推进指令；同会话同节点的未消费指令幂等返回。"""

        if command not in ("ADVANCE_TO_NODE", "ADVANCE_STEP"):
            raise ValueError("command must be ADVANCE_TO_NODE or ADVANCE_STEP")
        if not isinstance(target_node, int) or isinstance(target_node, bool) or not 1 <= target_node <= 8:
            raise ValueError("target_node must be between 1 and 8")
        for name, value in (("session_id", session_id), ("requested_by", requested_by)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if command == "ADVANCE_STEP":
            import uuid as _uuid

            session_node_key = f"{session_id}:ADVANCE_STEP:{_uuid.uuid4().hex[:12]}"
        else:
            session_node_key = f"{session_id}:ADVANCE_TO_NODE:{target_node}:PENDING"
        created_at = _now_iso()
        with self._operation() as connection:
            self._begin_immediate(connection)
            existing = connection.execute(
                "SELECT * FROM demo_commands WHERE session_node_key = ?",
                (session_node_key,),
            ).fetchone()
            if existing is not None:
                return self._demo_command(existing)
            connection.execute(
                "INSERT INTO demo_commands(session_id, command, target_node, "
                "requested_by, status, created_at, session_node_key) "
                "VALUES (?, ?, ?, ?, 'PENDING', ?, ?)",
                (session_id, command, target_node, requested_by, created_at, session_node_key),
            )
            row = connection.execute(
                "SELECT * FROM demo_commands WHERE session_node_key = ?",
                (session_node_key,),
            ).fetchone()
            assert row is not None
            return self._demo_command(row)

    def pending_demo_commands(self, session_id: str) -> list["DemoCommand"]:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT * FROM demo_commands WHERE session_id = ? AND status = 'PENDING' "
                "ORDER BY target_node, command_id",
                (session_id,),
            ).fetchall()
            return [self._demo_command(row) for row in rows]

    def consume_demo_command(
        self, command_id: int, *, result_text: str | None = None
    ) -> "DemoCommand":
        if not isinstance(command_id, int) or isinstance(command_id, bool):
            raise ValueError("command_id must be an integer")
        if result_text is not None and (
            not isinstance(result_text, str) or not result_text.strip()
        ):
            raise ValueError("result_text must be non-empty text")
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT * FROM demo_commands WHERE command_id = ?", (command_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown demo command: {command_id}")
            if row["status"] != "PENDING":
                raise ValueError(f"demo command {command_id} is not PENDING")
            connection.execute(
                "UPDATE demo_commands SET status = 'CONSUMED', consumed_at = ?, "
                "result_text = ?, feedback_status = 'PENDING', "
                "session_node_key = session_node_key || ':done' WHERE command_id = ?",
                (_now_iso(), result_text, command_id),
            )
            updated = connection.execute(
                "SELECT * FROM demo_commands WHERE command_id = ?", (command_id,)
            ).fetchone()
            assert updated is not None
            return self._demo_command(updated)

    def commands_pending_feedback(self, session_id: str) -> list["DemoCommand"]:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT * FROM demo_commands WHERE session_id = ? "
                "AND status = 'CONSUMED' AND feedback_status = 'PENDING' "
                "ORDER BY command_id",
                (session_id,),
            ).fetchall()
            return [self._demo_command(row) for row in rows]

    def mark_feedback_sent(self, command_id: int) -> None:
        if not isinstance(command_id, int) or isinstance(command_id, bool):
            raise ValueError("command_id must be an integer")
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE demo_commands SET feedback_status = 'SENT' "
                "WHERE command_id = ? AND feedback_status = 'PENDING'",
                (command_id,),
            )
            if result.rowcount != 1:
                raise ValueError(f"demo command {command_id} has no pending feedback")

    def upsert_demo_progress(self, session_id: str, progress_text: str) -> None:
        for name, value in (("session_id", session_id), ("progress_text", progress_text)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        with self._operation() as connection:
            connection.execute(
                "INSERT INTO demo_progress(session_id, progress_text, updated_at) "
                "VALUES (?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET "
                "progress_text = excluded.progress_text, updated_at = excluded.updated_at",
                (session_id, progress_text, _now_iso()),
            )

    def get_demo_progress(self, session_id: str) -> str | None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        with self._operation() as connection:
            row = connection.execute(
                "SELECT progress_text FROM demo_progress WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return None if row is None else str(row["progress_text"])

    def replay_cards_for_session(self, session_id: str) -> list["ReplayCard"]:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT * FROM replay_cards WHERE session_id = ? ORDER BY current_ordinal DESC",
                (session_id,),
            ).fetchall()
            return [self._replay(row) for row in rows]

    @staticmethod
    def _demo_command(row: sqlite3.Row) -> "DemoCommand":
        return DemoCommand(
            command_id=row["command_id"],
            session_id=row["session_id"],
            command=row["command"],
            target_node=row["target_node"],
            requested_by=row["requested_by"],
            status=row["status"],
            created_at=row["created_at"],
            consumed_at=row["consumed_at"],
            result_text=row["result_text"],
            feedback_status=row["feedback_status"],
        )

    def create_session(
        self,
        source_run_id: str,
        *,
        created_by: str,
        session_id: str | None = None,
        created_at: str | None = None,
        connected_at: str | None = None,
        writeback_targets: Mapping[str, WritebackTarget] | None = None,
    ) -> DemoSession:
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            raise ValueError("source_run_id must be non-empty")
        if created_by != "streamlit-local":
            raise ValueError("created_by must be streamlit-local")
        if session_id is None:
            session_id = str(uuid.uuid4())
        elif not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        if created_at is None:
            normalized_created_at = _now_iso()
        else:
            if not isinstance(created_at, str) or not created_at.strip():
                raise ValueError("created_at must be a non-empty ISO timestamp")
            try:
                normalized_created_at = _utc_iso(datetime.fromisoformat(created_at))
            except (TypeError, ValueError) as exc:
                raise ValueError("created_at must include a valid timezone") from exc
        if connected_at is None:
            normalized_connected_at = normalized_created_at
        else:
            if not isinstance(connected_at, str) or not connected_at.strip():
                raise ValueError("connected_at must be a non-empty ISO timestamp")
            try:
                normalized_connected_at = _utc_iso(datetime.fromisoformat(connected_at))
            except (TypeError, ValueError) as exc:
                raise ValueError("connected_at must include a valid timezone") from exc
        if writeback_targets is not None:
            if set(writeback_targets) != {"RUN_LOG", "RESULTS"}:
                raise ValueError("writeback_targets must contain RUN_LOG and RESULTS")
            for table_key, target in writeback_targets.items():
                _validate_writeback_target(target, table_key)
        with self._operation() as connection:
            self._begin_immediate(connection)
            connection.execute("UPDATE demo_sessions SET status = 'COMPLETE' WHERE status = 'ACTIVE'")
            connection.execute(
                "INSERT INTO demo_sessions("
                "session_id, source_run_id, created_by, created_at, connected_at, status) "
                "VALUES (?, ?, ?, ?, ?, 'ACTIVE')",
                (
                    session_id,
                    source_run_id,
                    created_by,
                    normalized_created_at,
                    normalized_connected_at,
                ),
            )
            if writeback_targets is not None:
                for table_key in ("RUN_LOG", "RESULTS"):
                    target = writeback_targets[table_key]
                    connection.execute(
                        "INSERT INTO demo_writeback_targets("
                        "session_id, table_key, app_token, table_id, table_url, field_names_json) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            session_id,
                            table_key,
                            target.app_token,
                            target.table_id,
                            target.table_url,
                            json.dumps(
                                list(target.field_names),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        ),
                    )
            row = connection.execute(
                "SELECT * FROM demo_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            assert row is not None
            return self._session(row)

    def active_session(self) -> DemoSession | None:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM demo_sessions WHERE status = 'ACTIVE'"
            ).fetchone()
            return None if row is None else self._session(row)

    def complete_active_session(self) -> int:
        """Close the persisted connection when a new page session starts."""

        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE demo_sessions SET status = 'COMPLETE' WHERE status = 'ACTIVE'"
            )
            return result.rowcount

    def get_session(self, session_id: str) -> DemoSession:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM demo_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown session: {session_id}")
            return self._session(row)

    def writeback_targets_for_session(
        self, session_id: str
    ) -> dict[str, WritebackTarget]:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be non-empty")
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT * FROM demo_writeback_targets WHERE session_id = ? "
                "ORDER BY table_key",
                (session_id,),
            ).fetchall()
            targets: dict[str, WritebackTarget] = {}
            for row in rows:
                field_names = json.loads(row["field_names_json"])
                if not isinstance(field_names, list) or not all(
                    isinstance(name, str) and name for name in field_names
                ):
                    raise ValueError(
                        f"session {session_id} has invalid field_names_json"
                    )
                targets[row["table_key"]] = WritebackTarget(
                    table_key=row["table_key"],
                    app_token=row["app_token"],
                    table_id=row["table_id"],
                    field_names=tuple(field_names),
                    table_url=row["table_url"],
                )
            return targets

    def active_writeback_targets(self) -> dict[str, WritebackTarget]:
        session = self.active_session()
        if session is None:
            return {}
        return self.writeback_targets_for_session(session.session_id)

    def enqueue_snapshot(
        self,
        session_id: str,
        snapshot: "DemoNotificationSnapshot",
        *,
        test_mode: bool = False,
    ) -> NotificationJob:
        if not isinstance(test_mode, bool):
            raise ValueError("test_mode must be bool")
        if not hasattr(snapshot, "model_dump"):
            raise TypeError("snapshot must provide model_dump()")
        payload = snapshot.model_dump(mode="json")
        if not isinstance(payload, dict):
            raise ValueError("snapshot model_dump must return an object")
        payload_bytes = canonical_json_bytes(payload)
        payload_json = payload_bytes.decode("utf-8")
        payload_sha256 = sha256_bytes(payload_bytes)
        node_key = _node_key(getattr(snapshot, "node", payload.get("node")))
        ordinal = getattr(snapshot, "ordinal", payload.get("ordinal"))
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 1 <= ordinal <= 8:
            raise ValueError("snapshot ordinal must be between 1 and 8")
        stable_uuid = delivery_uuid(session_id, node_key, payload_sha256)
        created_at = _now_iso()
        with self._operation() as connection:
            self._begin_immediate(connection)
            session = connection.execute(
                "SELECT status FROM demo_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise KeyError(f"unknown session: {session_id}")
            if session["status"] != "ACTIVE":
                raise ValueError("cannot enqueue into a complete session")
            existing = connection.execute(
                "SELECT * FROM notification_jobs WHERE session_id = ? AND node_key = ?",
                (session_id, node_key),
            ).fetchone()
            if existing is not None:
                if existing["payload_sha256"] != payload_sha256:
                    raise ValueError("snapshot hash differs for the existing job")
                return self._job(existing)
            connection.execute(
                "INSERT INTO notification_jobs(" 
                "session_id, node_key, ordinal, payload_json, payload_sha256, delivery_uuid, "
                "test_mode, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)",
                (
                    session_id,
                    node_key,
                    ordinal,
                    payload_json,
                    payload_sha256,
                    stable_uuid,
                    int(test_mode),
                    created_at,
                ),
            )
            row = connection.execute(
                "SELECT * FROM notification_jobs WHERE session_id = ? AND node_key = ?",
                (session_id, node_key),
            ).fetchone()
            assert row is not None
            return self._job(row)

    def get_job(self, session_id: str, node: Any) -> NotificationJob | None:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM notification_jobs WHERE session_id = ? AND node_key = ?",
                (session_id, _node_key(node)),
            ).fetchone()
            return None if row is None else self._job(row)

    def jobs_for_session(self, session_id: str) -> list[NotificationJob]:
        with self._operation() as connection:
            rows = connection.execute(
                "SELECT * FROM notification_jobs WHERE session_id = ? ORDER BY ordinal, job_id",
                (session_id,),
            ).fetchall()
            return [self._job(row) for row in rows]

    def claim_next_delivery(self, now: datetime) -> NotificationJob | None:
        now_iso = _utc_iso(now)
        ambiguity_cutoff = now.astimezone(timezone.utc) - timedelta(hours=1)
        with self._operation() as connection:
            self._begin_immediate(connection)
            inflight = connection.execute(
                "SELECT job_id, first_attempt_at FROM notification_jobs "
                "WHERE status = 'IN_FLIGHT' AND first_attempt_at IS NOT NULL"
            ).fetchall()
            for row in inflight:
                first_attempt_at = datetime.fromisoformat(row["first_attempt_at"])
                if first_attempt_at < ambiguity_cutoff:
                    connection.execute(
                        "UPDATE notification_jobs SET status = 'UNKNOWN', "
                        "last_error = ?, next_attempt_at = NULL WHERE job_id = ?",
                        ("delivery ambiguity window expired", row["job_id"]),
                    )
            row = connection.execute(
                "SELECT * FROM notification_jobs WHERE status = 'PENDING' "
                "AND (next_attempt_at IS NULL OR next_attempt_at <= ?) "
                "ORDER BY job_id LIMIT 1",
                (now_iso,),
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE notification_jobs SET status = 'IN_FLIGHT', "
                "attempt_count = attempt_count + 1, "
                "first_attempt_at = COALESCE(first_attempt_at, ?), next_attempt_at = NULL "
                "WHERE job_id = ?",
                (now_iso, row["job_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM notification_jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            assert claimed is not None
            return self._job(claimed)

    def mark_sent(self, job_id: int, *, message_id: str, sent_at: datetime) -> None:
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("message_id must be non-empty")
        sent_at_iso = _utc_iso(sent_at)
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE notification_jobs SET status = 'SENT', message_id = ?, "
                "sent_at = ?, next_attempt_at = NULL, last_error = NULL WHERE job_id = ? "
                "AND status = 'IN_FLIGHT'",
                (message_id, sent_at_iso, job_id),
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
                "SELECT status, attempt_count FROM notification_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None or row["status"] != "IN_FLIGHT":
                raise ValueError("job is not IN_FLIGHT")
            if row["attempt_count"] >= 3:
                connection.execute(
                    "UPDATE notification_jobs SET status = 'FAILED', last_error = ?, "
                    "next_attempt_at = NULL WHERE job_id = ?",
                    (error, job_id),
                )
            else:
                connection.execute(
                    "UPDATE notification_jobs SET status = 'PENDING', last_error = ?, "
                    "next_attempt_at = ? WHERE job_id = ?",
                    (error, next_attempt_iso, job_id),
                )

    def mark_failed(self, job_id: int, *, error: str) -> None:
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be non-empty")
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE notification_jobs SET status = 'FAILED', last_error = ?, "
                "next_attempt_at = NULL WHERE job_id = ? AND status IN ('IN_FLIGHT','PENDING')",
                (error, job_id),
            )
            if result.rowcount != 1:
                raise ValueError("job is not retryable")

    def mark_unknown(self, job_id: int, *, error: str) -> None:
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be non-empty")
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE notification_jobs SET status = 'UNKNOWN', last_error = ?, "
                "next_attempt_at = NULL WHERE job_id = ? AND status = 'IN_FLIGHT'",
                (error, job_id),
            )
            if result.rowcount != 1:
                raise ValueError("job is not IN_FLIGHT")

    def retry_failed(self, session_id: str, node: "DemoNotificationNode") -> NotificationJob:
        node_key = _node_key(node)
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT * FROM notification_jobs WHERE session_id = ? AND node_key = ?",
                (session_id, node_key),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown job: {session_id}/{node_key}")
            if row["status"] != "FAILED":
                raise ValueError("only FAILED jobs can be retried")
            connection.execute(
                "UPDATE notification_jobs SET status = 'PENDING', attempt_count = 0, "
                "first_attempt_at = NULL, next_attempt_at = NULL, message_id = NULL, "
                "last_error = NULL, sent_at = NULL WHERE job_id = ?",
                (row["job_id"],),
            )
            retried = connection.execute(
                "SELECT * FROM notification_jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            assert retried is not None
            return self._job(retried)

    def record_processed_event(self, event_key: str, processed_at: datetime) -> bool:
        if not isinstance(event_key, str) or not event_key.strip():
            raise ValueError("event_key must be non-empty")
        processed_at_iso = _utc_iso(processed_at)
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "INSERT OR IGNORE INTO processed_events(event_key, processed_at) VALUES (?, ?)",
                (event_key, processed_at_iso),
            )
            return result.rowcount == 1

    def create_replay_card(
        self,
        session_id: str,
        *,
        source_message_id: str,
        card_id: str,
        reply_uuid: str,
        ordinal: int,
    ) -> ReplayCard:
        for name, value in (("source_message_id", source_message_id), ("card_id", card_id), ("reply_uuid", reply_uuid)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 1 <= ordinal <= 8:
            raise ValueError("ordinal must be between 1 and 8")
        reserved = self.reserve_replay_card(
            session_id,
            source_message_id=source_message_id,
            ordinal=ordinal,
            reply_uuid=reply_uuid,
        )
        return self.bind_replay_card(reserved.replay_id, card_id=card_id)

    def reserve_replay_card(
        self,
        session_id: str,
        *,
        source_message_id: str,
        ordinal: int,
        reply_uuid: str | None = None,
    ) -> ReplayCard:
        for name, value in (("source_message_id", source_message_id),):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or not 1 <= ordinal <= 8:
            raise ValueError("ordinal must be between 1 and 8")
        updated_at = _now_iso()
        identity = f"{session_id}:{source_message_id}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()[:32]
        replay_id = "replay-" + digest
        stable_reply_uuid = "mj-reply-" + digest if reply_uuid is None else reply_uuid
        if not isinstance(stable_reply_uuid, str) or not stable_reply_uuid.strip():
            raise ValueError("reply_uuid must be non-empty")
        with self._operation() as connection:
            self._begin_immediate(connection)
            existing = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ? OR source_message_id = ?",
                (replay_id, source_message_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["session_id"] != session_id
                    or existing["reply_uuid"] != stable_reply_uuid
                    or existing["current_ordinal"] != ordinal
                ):
                    raise ValueError("replay card identity differs")
                return self._replay(existing)
            connection.execute(
                "INSERT INTO replay_cards(" 
                "replay_id, session_id, source_message_id, reply_uuid, card_id, current_ordinal, "
                "sequence, update_status, updated_at) VALUES (?, ?, ?, ?, NULL, ?, 0, 'PENDING_CARD', ?)",
                (replay_id, session_id, source_message_id, stable_reply_uuid, ordinal, updated_at),
            )
            row = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            assert row is not None
            return self._replay(row)

    def bind_replay_card(self, replay_id: str, *, card_id: str) -> ReplayCard:
        if not isinstance(card_id, str) or not card_id.strip():
            raise ValueError("card_id must be non-empty")
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown replay card: {replay_id}")
            if row["card_id"] == card_id and row["update_status"] == "AWAITING_REPLY":
                return self._replay(row)
            if row["card_id"] is not None or row["update_status"] not in (
                "PENDING_CARD",
                "CARD_CREATE_IN_FLIGHT",
            ):
                raise ValueError("replay card is already bound")
            connection.execute(
                "UPDATE replay_cards SET card_id = ?, update_status = 'AWAITING_REPLY', updated_at = ? "
                "WHERE replay_id = ? AND card_id IS NULL AND update_status IN "
                "('PENDING_CARD', 'CARD_CREATE_IN_FLIGHT')",
                (card_id, _now_iso(), replay_id),
            )
            bound = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            assert bound is not None
            return self._replay(bound)

    def claim_replay_card_creation(self, replay_id: str) -> ReplayCard | None:
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE replay_cards SET update_status = 'CARD_CREATE_IN_FLIGHT', updated_at = ? "
                "WHERE replay_id = ? AND card_id IS NULL AND update_status = 'PENDING_CARD'",
                (_now_iso(), replay_id),
            )
            if result.rowcount != 1:
                return None
            row = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            assert row is not None
            return self._replay(row)

    def get_replay_card(self, replay_id: str) -> ReplayCard:
        with self._operation() as connection:
            row = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown replay card: {replay_id}")
            return self._replay(row)

    def mark_replay_replied(
        self,
        replay_id: str,
        *,
        message_id: str,
        replied_at: datetime,
    ) -> ReplayCard:
        if not isinstance(message_id, str) or not message_id.strip():
            raise ValueError("message_id must be non-empty")
        replied_at_iso = _utc_iso(replied_at)
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown replay card: {replay_id}")
            if row["update_status"] == "IDLE" and row["message_id"] == message_id:
                return self._replay(row)
            if row["update_status"] != "AWAITING_REPLY":
                raise ValueError("replay card is not awaiting reply")
            connection.execute(
                "UPDATE replay_cards SET message_id = ?, update_status = 'IDLE', "
                "updated_at = ?, last_error = NULL WHERE replay_id = ?",
                (message_id, replied_at_iso, replay_id),
            )
            updated = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            assert updated is not None
            return self._replay(updated)

    def request_replay_update(
        self,
        replay_id: str,
        *,
        target_ordinal: int,
        update_uuid: str,
    ) -> ReplayCard:
        if not isinstance(target_ordinal, int) or isinstance(target_ordinal, bool) or not 1 <= target_ordinal <= 8:
            raise ValueError("target_ordinal must be between 1 and 8")
        if not isinstance(update_uuid, str) or not update_uuid.strip():
            raise ValueError("update_uuid must be non-empty")
        updated_at = _now_iso()
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown replay card: {replay_id}")
            status = row["update_status"]
            if status == "AWAITING_REPLY":
                raise ValueError("replay card is awaiting reply")
            if status in ("PENDING", "IN_FLIGHT"):
                raise ValueError(f"replay update is {status}")
            if status == "EXPIRED":
                raise ValueError("replay card is expired")
            connection.execute(
                "UPDATE replay_cards SET requested_ordinal = ?, update_uuid = ?, "
                "update_status = 'PENDING', updated_at = ?, last_error = NULL WHERE replay_id = ?",
                (target_ordinal, update_uuid, updated_at, replay_id),
            )
            requested = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            assert requested is not None
            return self._replay(requested)

    def claim_next_replay_update(
        self,
        now: datetime | None = None,
        *,
        stale_after: timedelta | None = None,
    ) -> ReplayCard | None:
        current = datetime.now(timezone.utc) if now is None else now
        now_iso = _utc_iso(current)
        window = self.replay_update_stale_after if stale_after is None else stale_after
        if not isinstance(window, timedelta) or window <= timedelta(0):
            raise ValueError("stale_after must be positive")
        cutoff = current.astimezone(timezone.utc) - window
        with self._operation() as connection:
            self._begin_immediate(connection)
            stale_rows = connection.execute(
                "SELECT replay_id, updated_at FROM replay_cards WHERE update_status = 'IN_FLIGHT'"
            ).fetchall()
            for stale in stale_rows:
                if datetime.fromisoformat(stale["updated_at"]) < cutoff:
                    connection.execute(
                        "UPDATE replay_cards SET update_status = 'PENDING', updated_at = ? "
                        "WHERE replay_id = ? AND update_status = 'IN_FLIGHT'",
                        (now_iso, stale["replay_id"]),
                    )
            row = connection.execute(
                "SELECT * FROM replay_cards WHERE update_status = 'PENDING' "
                "ORDER BY updated_at, replay_id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                "UPDATE replay_cards SET update_status = 'IN_FLIGHT', updated_at = ? "
                "WHERE replay_id = ? AND update_status = 'PENDING'",
                (now_iso, row["replay_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (row["replay_id"],)
            ).fetchone()
            assert claimed is not None
            return self._replay(claimed)

    def mark_replay_updated(
        self,
        replay_id: str,
        *,
        target_ordinal: int,
        sequence: int,
        updated_at: datetime,
    ) -> None:
        if not isinstance(target_ordinal, int) or isinstance(target_ordinal, bool) or not 1 <= target_ordinal <= 8:
            raise ValueError("target_ordinal must be between 1 and 8")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            raise ValueError("sequence must be an integer")
        updated_at_iso = _utc_iso(updated_at)
        with self._operation() as connection:
            self._begin_immediate(connection)
            row = connection.execute(
                "SELECT * FROM replay_cards WHERE replay_id = ?", (replay_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown replay card: {replay_id}")
            if row["update_status"] != "IN_FLIGHT":
                raise ValueError("replay update is not IN_FLIGHT")
            if row["requested_ordinal"] != target_ordinal:
                raise ValueError("target ordinal differs from requested ordinal")
            if sequence != row["sequence"] + 1:
                raise ValueError("sequence must increase by one")
            connection.execute(
                "UPDATE replay_cards SET current_ordinal = ?, sequence = ?, "
                "requested_ordinal = NULL, update_uuid = NULL, update_status = 'IDLE', "
                "last_error = NULL, updated_at = ? WHERE replay_id = ?",
                (target_ordinal, sequence, updated_at_iso, replay_id),
            )

    def mark_replay_failed(self, replay_id: str, *, error: str, expired: bool = False) -> None:
        if not isinstance(error, str) or not error.strip():
            raise ValueError("error must be non-empty")
        status = "EXPIRED" if expired else "FAILED"
        with self._operation() as connection:
            self._begin_immediate(connection)
            result = connection.execute(
                "UPDATE replay_cards SET update_status = ?, requested_ordinal = NULL, "
                "update_uuid = NULL, last_error = ?, updated_at = ? "
                "WHERE replay_id = ? AND update_status IN ('PENDING','IN_FLIGHT')",
                (status, error, _now_iso(), replay_id),
            )
            if result.rowcount != 1:
                raise ValueError("replay card is not pending or in flight")
