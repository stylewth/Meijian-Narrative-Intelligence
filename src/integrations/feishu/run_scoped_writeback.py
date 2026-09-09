"""Resolve the active connection's concrete Feishu write-back targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .writeback_targets import WritebackTarget


@dataclass(frozen=True, slots=True)
class ActiveWritebackContext:
    demo_run_id: str
    demo_started_at: str
    targets: dict[str, WritebackTarget]


def active_writeback_context(store: Any) -> ActiveWritebackContext | None:
    """Return the active session's two targets, or None for legacy sessions."""

    if store is None:
        return None
    session = store.active_session()
    if session is None:
        return None
    targets = store.writeback_targets_for_session(session.session_id)
    if not targets:
        return None
    if set(targets) != {"RUN_LOG", "RESULTS"}:
        raise ValueError("活动连接的写回表映射不完整")
    return ActiveWritebackContext(
        demo_run_id=session.session_id,
        demo_started_at=session.created_at,
        targets=targets,
    )


__all__ = ["ActiveWritebackContext", "active_writeback_context"]
