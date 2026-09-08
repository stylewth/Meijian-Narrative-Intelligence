"""Pure Feishu notification queries and replay callback decisions."""

from __future__ import annotations

from collections.abc import Collection, Mapping
import hashlib
from typing import Any, Literal

from pydantic import Field

from src.schemas import StrictBaseModel
from src.string_enum import StringEnum

from .notification_store import NotificationStore


class QueryCommand(StringEnum):
    """The only query command understood by the notification bot."""

    CURRENT = "CURRENT"


class QueryReply(StrictBaseModel):
    session_id: str
    ordinal: int = Field(ge=1, le=8)
    source_message_id: str


class CallbackResult(StrictBaseModel):
    toast_type: Literal["success", "warning", "error"]
    toast_content: str


_COMMANDS = frozenset({"远程操控", "查看叙事", "查看叙事变化"})
_REPLAY_ACTIONS = frozenset({"REPLAY_PREVIOUS", "REPLAY_NEXT"})


def parse_query(event: Mapping[str, Any], *, bot_open_id: str) -> QueryCommand | None:
    """Parse one already-normalized message event without inferring intent."""

    if not isinstance(event, Mapping) or not isinstance(bot_open_id, str):
        return None

    message = _message_mapping(event)
    if message is None:
        return None
    if _sender_open_id(event) == bot_open_id:
        return None

    content = message.get("content")
    if not isinstance(content, Mapping) or not isinstance(content.get("text"), str):
        return None
    text = content["text"]

    mentions = message.get("mentions", ())
    if not isinstance(mentions, list):
        mentions = ()
    bot_mentioned = False
    for mention in mentions:
        if _mention_open_id(mention) != bot_open_id:
            continue
        bot_mentioned = True
        key = mention.get("key") if isinstance(mention, Mapping) else None
        if isinstance(key, str) and key:
            text = text.replace(key, "", 1)

    chat_type = message.get("chat_type")
    if chat_type == "group" and not bot_mentioned:
        return None
    if chat_type not in {"p2p", "group"}:
        return None

    if text.strip() not in _COMMANDS:
        return None
    return QueryCommand.CURRENT


def handle_query(
    event: Mapping[str, Any],
    *,
    bot_open_id: str,
    store: NotificationStore,
) -> QueryReply | CallbackResult | None:
    """Return a query-card instruction or a local status toast.

    This function only reads the local store. Card creation and message delivery
    belong to the worker after the event callback has returned.
    """

    if parse_query(event, bot_open_id=bot_open_id) is None:
        return None

    session = store.active_session()
    if session is None:
        return CallbackResult(
            toast_type="warning",
            toast_content="当前没有活动演示会话。",
        )

    jobs = store.jobs_for_session(session.session_id)
    if not jobs:
        return CallbackResult(
            toast_type="warning",
            toast_content="当前会话还没有已入队的叙事节点。",
        )

    latest = max(jobs, key=lambda job: (job.ordinal, job.job_id))
    source_message_id = _source_message_id(event)
    if source_message_id is None:
        raise ValueError("query event message_id is required")
    return QueryReply(
        session_id=session.session_id,
        ordinal=latest.ordinal,
        source_message_id=source_message_id,
    )


def handle_replay_action(
    payload: Mapping[str, Any],
    *,
    operator_open_id: str,
    allowed_operator_ids: Collection[str],
    store: NotificationStore,
) -> CallbackResult:
    """Validate and persist one replay request; never call Feishu inline."""

    if not isinstance(operator_open_id, str) or operator_open_id not in set(
        allowed_operator_ids
    ):
        return CallbackResult(
            toast_type="warning",
            toast_content="你没有权限操作叙事回放。",
        )

    if not isinstance(payload, Mapping):
        return CallbackResult(toast_type="error", toast_content="回放请求格式无效。")

    action = payload.get("action")
    replay_id = payload.get("replay_id")
    target_ordinal = payload.get("target_ordinal")
    if action not in _REPLAY_ACTIONS:
        return CallbackResult(toast_type="error", toast_content="不支持的回放操作。")
    if not isinstance(replay_id, str) or not replay_id.strip():
        return CallbackResult(toast_type="error", toast_content="回放卡片标识无效。")
    if (
        not isinstance(target_ordinal, int)
        or isinstance(target_ordinal, bool)
        or not 1 <= target_ordinal <= 8
    ):
        return CallbackResult(toast_type="error", toast_content="回放目标节点无效。")

    try:
        replay = store.get_replay_card(replay_id)
    except KeyError:
        return CallbackResult(toast_type="error", toast_content="回放卡片不存在。")

    expected_target = (
        replay.current_ordinal - 1
        if action == "REPLAY_PREVIOUS"
        else replay.current_ordinal + 1
    )
    if not 1 <= expected_target <= 8 or target_ordinal != expected_target:
        return CallbackResult(toast_type="error", toast_content="回放目标不是相邻节点。")

    available_ordinals = {
        job.ordinal for job in store.jobs_for_session(replay.session_id)
    }
    try:
        store.enqueue_demo_command(
            replay.session_id,
            command="ADVANCE_STEP",
            target_node=expected_target,
            requested_by=operator_open_id,
        )
    except ValueError as exc:
        return CallbackResult(toast_type="warning", toast_content=str(exc))
    return CallbackResult(
        toast_type="success",
        toast_content="已发送单步推进指令；网页执行后会推送本步反馈卡片。",
    )

    update_uuid = _replay_update_uuid(
        replay_id, replay.sequence, target_ordinal
    )
    try:
        store.request_replay_update(
            replay_id,
            target_ordinal=target_ordinal,
            update_uuid=update_uuid,
        )
    except ValueError as exc:
        detail = str(exc)
        if any(
            marker in detail.lower()
            for marker in ("pending", "in_flight", "awaiting", "expired")
        ):
            return CallbackResult(
                toast_type="warning",
                toast_content="该回放卡片已有更新在处理中。",
            )
        return CallbackResult(toast_type="error", toast_content="回放请求未能保存。")

    return CallbackResult(toast_type="success", toast_content="回放更新已排队。")


def decide_query(
    event: Mapping[str, Any],
    *,
    bot_open_id: str,
    store: NotificationStore,
) -> QueryReply | CallbackResult | None:
    """Compatibility name for callers that treat query handling as a decision."""

    return handle_query(event, bot_open_id=bot_open_id, store=store)


def build_query_reply(
    event: Mapping[str, Any],
    *,
    bot_open_id: str,
    store: NotificationStore,
) -> QueryReply | CallbackResult | None:
    """Compatibility name for the pure query-card instruction builder."""

    return handle_query(event, bot_open_id=bot_open_id, store=store)


def _message_mapping(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    message = event.get("message")
    if isinstance(message, Mapping):
        return message
    nested = event.get("event")
    if isinstance(nested, Mapping) and isinstance(nested.get("message"), Mapping):
        return nested["message"]
    return None


def _sender_open_id(event: Mapping[str, Any]) -> str | None:
    sender = event.get("sender")
    if not isinstance(sender, Mapping):
        nested = event.get("event")
        sender = nested.get("sender") if isinstance(nested, Mapping) else None
    if not isinstance(sender, Mapping):
        return None
    sender_id = sender.get("sender_id")
    if not isinstance(sender_id, Mapping):
        return None
    value = sender_id.get("open_id")
    return value if isinstance(value, str) else None


def _mention_open_id(mention: Any) -> str | None:
    if not isinstance(mention, Mapping):
        return None
    item_id = mention.get("id")
    if isinstance(item_id, Mapping) and isinstance(item_id.get("open_id"), str):
        return item_id["open_id"]
    if isinstance(mention.get("open_id"), str):
        return mention["open_id"]
    if isinstance(item_id, str):
        return item_id
    return None


def _source_message_id(event: Mapping[str, Any]) -> str | None:
    message = _message_mapping(event)
    if message is None:
        return None
    value = message.get("message_id")
    return value if isinstance(value, str) and value.strip() else None


def _replay_update_uuid(replay_id: str, sequence: int, target_ordinal: int) -> str:
    raw = f"{replay_id}:{sequence}:{target_ordinal}".encode("utf-8")
    return "mj-replay-" + hashlib.sha256(raw).hexdigest()[:32]


__all__ = [
    "CallbackResult",
    "QueryCommand",
    "QueryReply",
    "build_query_reply",
    "decide_query",
    "handle_query",
    "handle_replay_action",
    "parse_query",
]
