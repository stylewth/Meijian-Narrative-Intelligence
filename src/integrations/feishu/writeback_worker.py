"""Offline-testable worker that drains the Bitable write-back queue."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
import json
import time
from typing import Any

from .writeback_store import WritebackStore, WritebackJob
from .writeback_targets import WritebackTarget


RETRY_DELAYS_SECONDS = (1.0, 5.0)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WritebackWorker:
    """Claim and process at most one write-back job per ``run_once`` invocation."""

    def __init__(
        self,
        store: WritebackStore,
        client: Any,
        targets: dict[str, WritebackTarget],
        *,
        clock: Callable[[], datetime] = _utc_now,
        wait_fn: Callable[[float], None] | None = None,
    ) -> None:
        if not targets:
            raise ValueError("writeback worker requires at least one target table")
        self._store = store
        self._client = client
        self._targets = dict(targets)
        self._clock = clock
        self._wait_fn = time.sleep if wait_fn is None else wait_fn

    def run_once(self) -> bool:
        now = self._now()
        job = self._store.claim_next_writeback(now)
        if job is None:
            return False
        self._process(job, now)
        return True

    def run_forever(
        self,
        stop_event: Any,
        *,
        poll_seconds: float = 0.25,
        wait_fn: Callable[[float], None] | None = None,
    ) -> None:
        if not isinstance(poll_seconds, (int, float)) or isinstance(poll_seconds, bool):
            raise ValueError("poll_seconds must be a non-negative number")
        if poll_seconds < 0:
            raise ValueError("poll_seconds must be a non-negative number")
        waiter = self._wait_fn if wait_fn is None else wait_fn
        while not stop_event.is_set():
            if not self.run_once():
                waiter(poll_seconds)

    def _process(self, job: WritebackJob, now: datetime) -> None:
        target = self._targets.get(job.table_key)
        if target is None:
            self._store.mark_failed(
                job.job_id,
                error=f"writeback target not configured: {job.table_key}",
            )
            raise LookupError(f"writeback target not configured: {job.table_key}")

        try:
            fields = json.loads(job.payload_json)
            if not isinstance(fields, dict):
                raise ValueError("writeback payload must decode to an object")
            record_id = self._client.create_record(
                target.app_token, target.table_id, fields
            )
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError("create_record must return a non-empty record_id")
        except Exception as exc:
            self._record_failure(job, error=str(exc), now=now)
            return

        self._store.mark_wrote(job.job_id, record_id=record_id, written_at=now)

    def _record_failure(
        self,
        job: WritebackJob,
        *,
        error: str,
        now: datetime,
    ) -> None:
        delay_index = job.attempt_count - 1
        delay = RETRY_DELAYS_SECONDS[delay_index] if delay_index < len(RETRY_DELAYS_SECONDS) else 0
        self._store.mark_retry(
            job.job_id,
            error=error,
            next_attempt_at=now + timedelta(seconds=delay),
        )

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


__all__ = ["RETRY_DELAYS_SECONDS", "WritebackWorker"]
