"""Offline-testable worker for notification delivery and queued CardKit updates."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import time
from typing import Any

from src.services.demo_notifications import DemoNotificationSnapshot

from .message_client import FeishuMessageError
from .notification_cards import (
    build_notification_card,
    build_replay_card,
    validate_card,
)
from .result_links import resolve_result_url
from .notification_store import NotificationJob, NotificationStore, ReplayCard


RETRY_DELAYS_SECONDS = (1.0, 5.0)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class NotificationWorker:
    """Claim and process at most one local job per ``run_once`` invocation."""

    def __init__(
        self,
        store: NotificationStore,
        client: Any,
        *,
        chat_id: str,
        web_url: str,
        feishu_wiki_url: str | None = None,
        clock: Callable[[], datetime] | Any = _utc_now,
        wait_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._store = store
        self._client = client
        self._chat_id = chat_id
        self._web_url = web_url
        self._feishu_wiki_url = feishu_wiki_url
        self._clock = clock
        self._wait_fn = time.sleep if wait_fn is None else wait_fn

    def run_once(self) -> bool:
        """Process one claimed notification or replay update, without waiting."""

        now = self._now()
        job = self._store.claim_next_delivery(now)
        if job is not None:
            self._process_delivery(job, now)
            return True

        replay = self._store.claim_next_replay_update()
        if replay is not None:
            self._process_replay_update(replay, now)
            return True
        return False

    def run_forever(
        self,
        stop_event: Any,
        *,
        poll_seconds: float = 0.25,
        wait_fn: Callable[[float], None] | None = None,
    ) -> None:
        """Run the non-blocking state machine and own all polling waits."""

        if not isinstance(poll_seconds, (int, float)) or isinstance(poll_seconds, bool):
            raise ValueError("poll_seconds must be a non-negative number")
        if poll_seconds < 0:
            raise ValueError("poll_seconds must be a non-negative number")
        waiter = self._wait_fn if wait_fn is None else wait_fn
        while not stop_event.is_set():
            if not self.run_once():
                waiter(poll_seconds)

    def _process_delivery(self, job: NotificationJob, now: datetime) -> None:
        try:
            snapshot = DemoNotificationSnapshot.model_validate_json(
                job.payload_json,
                strict=True,
            )
            result_url = resolve_result_url(
                snapshot.node,
                wiki_url=self._feishu_wiki_url,
                web_url=self._web_url,
            )
            card = build_notification_card(
                snapshot,
                session_short_id=job.session_id[:8],
                web_url=self._web_url,
                result_url=result_url,
            )
            if job.test_mode:
                card = self._with_test_title(card, snapshot.title)
            validate_card(card)
        except Exception as exc:
            self._store.mark_failed(job.job_id, error=str(exc))
            return

        try:
            message_id = self._client.send_card(
                self._chat_id,
                card,
                uuid=job.delivery_uuid,
            )
            if not isinstance(message_id, str) or not message_id.strip():
                raise ValueError("message_id must be non-empty")
        except Exception as exc:
            self._record_delivery_failure(job, error=str(exc), now=now)
            return

        self._store.mark_sent(job.job_id, message_id=message_id, sent_at=now)

    def _record_delivery_failure(
        self,
        job: NotificationJob,
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

    def _process_replay_update(self, replay: ReplayCard, now: datetime) -> None:
        try:
            snapshot = self._load_snapshot(replay)
            target_ordinal = replay.requested_ordinal
            if target_ordinal is None:
                raise ValueError("replay update is missing target ordinal")
            card = build_replay_card(
                snapshot,
                replay_id=replay.replay_id,
                current_ordinal=target_ordinal,
                web_url=self._web_url,
            )
            validate_card(card)
        except Exception as exc:
            self._store.mark_replay_failed(replay.replay_id, error=str(exc))
            return

        try:
            self._client.update_card_entity(
                replay.card_id,
                sequence=replay.sequence + 1,
                uuid=replay.update_uuid or "",
                card=card,
            )
        except Exception as exc:
            error = str(exc)
            self._store.mark_replay_failed(
                replay.replay_id,
                error=error,
                expired=self._is_expired_card_error(exc),
            )
            return

        target_ordinal = replay.requested_ordinal
        if target_ordinal is None:
            raise ValueError("replay update is missing target ordinal")
        self._store.mark_replay_updated(
            replay.replay_id,
            target_ordinal=target_ordinal,
            sequence=replay.sequence + 1,
            updated_at=now,
        )

    def _load_snapshot(self, replay: ReplayCard) -> DemoNotificationSnapshot:
        jobs = self._store.jobs_for_session(replay.session_id)
        target_ordinal = replay.requested_ordinal
        for job in jobs:
            if job.ordinal == target_ordinal:
                return DemoNotificationSnapshot.model_validate_json(
                    job.payload_json,
                    strict=True,
                )
        raise KeyError(
            f"missing notification snapshot: {replay.session_id}/{target_ordinal}"
        )

    @staticmethod
    def _with_test_title(card: dict[str, Any], title: str) -> dict[str, Any]:
        result = deepcopy(card)
        test_title = f"[TEST] {title}"
        result["config"]["summary"]["content"] = test_title
        result["header"]["title"]["content"] = test_title
        return result

    @staticmethod
    def _is_expired_card_error(exc: Exception) -> bool:
        if not isinstance(exc, FeishuMessageError):
            return False
        code = str(exc.code).lower() if exc.code is not None else ""
        message = str(exc).lower()
        text = f"{code} {message}"
        return any(
            marker in text
            for marker in (
                "card_expired",
                "card expired",
                "card_not_found",
                "card not found",
                "card does not exist",
                "card not exist",
            )
        )

    def _now(self) -> datetime:
        value = self._clock() if callable(self._clock) else self._clock.now()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(timezone.utc)


__all__ = ["NotificationWorker", "RETRY_DELAYS_SECONDS"]
