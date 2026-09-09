"""Long-connection runtime for the local Feishu notification demo."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import traceback
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from src.integrations.feishu.client import FeishuBitableClient
from src.integrations.feishu.decision_gate_store import DecisionGateStore
from src.integrations.feishu.decision_gates import (
    DECISION_GATE_ACTION_TYPE,
    handle_decision_gate_action,
    stage_status_text,
)
from src.integrations.feishu.message_client import FeishuMessageClient
from src.integrations.feishu.notification_bot import (
    CallbackResult,
    QueryReply,
    handle_query,
    handle_replay_action,
)
from src.integrations.feishu.notification_cards import _button, validate_card
from src.integrations.feishu.notification_store import NotificationStore
from src.integrations.feishu.notification_worker import NotificationWorker
from src.integrations.feishu.streamlit_url import (
    AUTO_STREAMLIT_URL,
    resolve_streamlit_public_url,
)
from src.integrations.feishu.writeback_store import WritebackStore
from src.integrations.feishu.writeback_targets import inventory_writeback_targets
from src.integrations.feishu.writeback_worker import WritebackWorker
from src.services.demo_notifications import DemoNotificationSnapshot


REQUIRED_CONFIG_KEYS = (
    "FEISHU_APP_ID",
    "FEISHU_APP_SECRET",
    "FEISHU_DEMO_CHAT_ID",
    "FEISHU_BOT_OPEN_ID",
    "FEISHU_OPERATOR_OPEN_IDS",
    "FEISHU_NOTIFICATION_DB",
)


@dataclass(frozen=True, slots=True)
class BotConfig:
    app_id: str
    app_secret: str = field(repr=False)
    demo_chat_id: str
    streamlit_public_url: str
    bot_open_id: str
    operator_open_ids: frozenset[str]
    notification_db: Path
    feishu_wiki_url: str | None = None
    writeback_run_log_url: str | None = None
    writeback_results_url: str | None = None
    writeback_db: Path | None = None
    gate_db: Path | None = None
    decision_run_root: Path | None = None
    gate_mode: str = "coordinator"
    replay_state_path: Path | None = None


def load_bot_config(
    mapping: Mapping[str, Any], *, workspace_root: str | Path | None = None
) -> BotConfig:
    """Parse only notification settings and keep the database in the workspace."""

    if not isinstance(mapping, Mapping):
        raise TypeError("configuration must be a mapping")

    values: dict[str, str] = {}
    for key in REQUIRED_CONFIG_KEYS:
        value = mapping.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} is required")
        values[key] = value.strip()

    raw_streamlit_url = mapping.get("STREAMLIT_PUBLIC_URL")
    if raw_streamlit_url is None:
        streamlit_public_url = AUTO_STREAMLIT_URL
    elif not isinstance(raw_streamlit_url, str):
        raise ValueError("STREAMLIT_PUBLIC_URL must be text or auto")
    else:
        streamlit_public_url = raw_streamlit_url.strip() or AUTO_STREAMLIT_URL

    raw_wiki_url = mapping.get("FEISHU_WIKI_URL")
    if raw_wiki_url is None:
        feishu_wiki_url = None
    elif not isinstance(raw_wiki_url, str):
        raise ValueError("FEISHU_WIKI_URL must be a text URL")
    else:
        feishu_wiki_url = raw_wiki_url.strip() or None

    raw_operator_ids = values["FEISHU_OPERATOR_OPEN_IDS"]
    operator_parts = [part.strip() for part in raw_operator_ids.split(",")]
    if any(not part for part in operator_parts):
        raise ValueError("FEISHU_OPERATOR_OPEN_IDS contains an empty item")
    operator_ids = frozenset(operator_parts)
    if len(operator_ids) != len(operator_parts):
        raise ValueError("FEISHU_OPERATOR_OPEN_IDS contains duplicates")

    workspace = (PROJECT_ROOT if workspace_root is None else Path(workspace_root)).resolve()
    configured_db = Path(values["FEISHU_NOTIFICATION_DB"])
    database_path = (workspace / configured_db if not configured_db.is_absolute() else configured_db).resolve()
    try:
        database_path.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("FEISHU_NOTIFICATION_DB must stay inside the workspace") from exc
    if database_path == workspace:
        raise ValueError("FEISHU_NOTIFICATION_DB must be a file path")
    if database_path.exists() and database_path.is_dir():
        raise ValueError("FEISHU_NOTIFICATION_DB must be a file path")
    parent = database_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError("FEISHU_NOTIFICATION_DB 父目录无法创建") from exc
    if not parent.is_dir():
        raise ValueError("FEISHU_NOTIFICATION_DB parent directory must be a directory")
    if not os.access(parent, os.W_OK):
        raise ValueError("FEISHU_NOTIFICATION_DB parent directory is not writable")
    if database_path.exists() and not os.access(database_path, os.W_OK):
        raise ValueError("FEISHU_NOTIFICATION_DB is not writable")

    def optional_text(key: str) -> str | None:
        value = mapping.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{key} must be text")
        return value.strip() or None

    def workspace_db_path(key: str, default_name: str) -> Path:
        configured = optional_text(key)
        if configured is None:
            return database_path.parent / default_name
        candidate = Path(configured)
        path = (workspace / candidate if not candidate.is_absolute() else candidate).resolve()
        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"{key} must stay inside the workspace") from exc
        if path == workspace:
            raise ValueError(f"{key} must be a file path")
        return path

    writeback_run_log_url = optional_text("FEISHU_RUNLOG_URL")
    writeback_results_url = optional_text("FEISHU_RESULTS_URL")
    writeback_db = workspace_db_path(
        "FEISHU_WRITEBACK_DB", "feishu_writeback.sqlite3"
    )
    gate_db = workspace_db_path("FEISHU_GATE_DB", "feishu_gate_jobs.sqlite3")

    decision_run_root = None
    raw_decision_root = optional_text("FEISHU_DECISION_RUN_ROOT")
    if raw_decision_root is not None:
        root_candidate = Path(raw_decision_root)
        decision_run_root = (
            workspace / root_candidate
            if not root_candidate.is_absolute()
            else root_candidate
        ).resolve()
        if not decision_run_root.is_dir():
            raise ValueError("FEISHU_DECISION_RUN_ROOT must be an existing directory")

    gate_mode = (optional_text("FEISHU_GATE_MODE") or "coordinator").strip().lower()
    if gate_mode not in ("coordinator", "replay"):
        raise ValueError("FEISHU_GATE_MODE must be coordinator or replay")
    if gate_mode == "replay":
        if decision_run_root is None:
            raise ValueError(
                "FEISHU_GATE_MODE=replay 需要同时配置 FEISHU_DECISION_RUN_ROOT "
                "指向演化 run 冻结目录（含 run_manifest.json）"
            )
        if not (decision_run_root / "run_manifest.json").is_file():
            raise ValueError(
                "FEISHU_GATE_MODE=replay 的 run 目录缺少 run_manifest.json；"
                "请指向 official 演化 run 冻结目录"
            )
    elif decision_run_root is not None and not (
        decision_run_root / "active.json"
    ).is_file():
        raise ValueError(
            "FEISHU_GATE_MODE=coordinator 需要 run 根目录包含 active.json "
            "（活动 run）；没有活动 run 时请移除 FEISHU_DECISION_RUN_ROOT 或改用 replay"
        )

    replay_state_path: Path | None = None
    if gate_mode == "replay":
        configured_state = optional_text("FEISHU_REPLAY_STATE_PATH")
        candidate = Path(configured_state) if configured_state else Path(
            "outputs/feishu_replay_state.json"
        )
        replay_state_path = (
            workspace / candidate if not candidate.is_absolute() else candidate
        ).resolve()
        try:
            replay_state_path.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(
                "FEISHU_REPLAY_STATE_PATH must stay inside the workspace"
            ) from exc
        if replay_state_path == workspace:
            raise ValueError("FEISHU_REPLAY_STATE_PATH must be a file path")

    return BotConfig(
        app_id=values["FEISHU_APP_ID"],
        app_secret=values["FEISHU_APP_SECRET"],
        demo_chat_id=values["FEISHU_DEMO_CHAT_ID"],
        streamlit_public_url=streamlit_public_url,
        bot_open_id=values["FEISHU_BOT_OPEN_ID"],
        operator_open_ids=operator_ids,
        notification_db=database_path,
        feishu_wiki_url=feishu_wiki_url,
        writeback_run_log_url=writeback_run_log_url,
        writeback_results_url=writeback_results_url,
        writeback_db=writeback_db,
        gate_db=gate_db,
        decision_run_root=decision_run_root,
        gate_mode=gate_mode,
        replay_state_path=replay_state_path,
    )


DECISION_STATUS_COMMANDS = frozenset({"决策进度"})
WRITEBACK_RETRY_ACTION_TYPE = "WRITEBACK_RETRY"
_RUN_ID_PATTERN = re.compile(r"^[^/\\]+$")


def read_active_decision_state(decision_run_root: Path) -> dict[str, Any] | None:
    """读取活动 run 的 coordinator 状态；没有活动 run 时返回 None。"""

    active_path = decision_run_root / "active.json"
    if not active_path.is_file():
        return None
    active = json.loads(active_path.read_text(encoding="utf-8"))
    run_id = active.get("run_id") if isinstance(active, dict) else None
    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.match(run_id):
        raise ValueError("active.json 中的 run_id 无效")
    state_path = decision_run_root / run_id / "coordinator" / "state.json"
    if not state_path.is_file():
        raise ValueError(f"活动 run 缺少 coordinator/state.json：{run_id}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"coordinator/state.json 必须是对象：{run_id}")
    return state


def read_decision_run_state(decision_run_root: Path, run_id: str) -> dict[str, Any] | None:
    """读取指定 run 的 coordinator 状态。"""

    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.match(run_id):
        return None
    state_path = decision_run_root / run_id / "coordinator" / "state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError(f"coordinator/state.json 必须是对象：{run_id}")
    return state


def normalize_message_event(raw_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an SDK event into the mapping consumed by notification_bot."""

    if not isinstance(raw_dict, Mapping):
        raise ValueError("raw_dict must be a mapping")
    payload = raw_dict.get("event")
    if not isinstance(payload, Mapping):
        payload = raw_dict

    message = payload.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("message is required")
    content_raw = message.get("content")
    if not isinstance(content_raw, str) or not content_raw.strip():
        raise ValueError("message.content must be a JSON string")
    try:
        content = json.loads(content_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("message.content must be valid JSON") from exc
    if not isinstance(content, dict):
        raise ValueError("message.content must decode to an object")
    text = content.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("message.content.text is required")

    chat_type = message.get("chat_type", payload.get("chat_type"))
    if not isinstance(chat_type, str) or not chat_type.strip():
        raise ValueError("message.chat_type is required")
    sender = payload.get("sender", raw_dict.get("sender"))
    sender_open_id = _extract_sender_open_id(sender)
    mentions = message.get("mentions", payload.get("mentions", []))
    normalized_mentions = _normalize_mentions(mentions)

    normalized_message = dict(message)
    normalized_message["content"] = content
    normalized_message["chat_type"] = chat_type
    normalized_message["mentions"] = normalized_mentions
    normalized = dict(payload)
    normalized["message"] = normalized_message
    normalized["chat_type"] = chat_type
    normalized["mentions"] = normalized_mentions
    normalized["sender"] = {
        "open_id": sender_open_id,
        "sender_id": {"open_id": sender_open_id},
    }
    normalized["text"] = text
    return normalized


def build_event_handler(
    config: BotConfig,
    *,
    lark: Any | None = None,
    store: NotificationStore | None = None,
    client: Any | None = None,
    worker: NotificationWorker | None = None,
    gate_store: DecisionGateStore | None = None,
    gate_enabled: bool = False,
    replay_state_loader: Any | None = None,
) -> Any:
    """Build the SDK dispatcher while keeping all business decisions pure."""

    lark = _import_lark_oapi() if lark is None else lark
    store = NotificationStore(config.notification_db) if store is None else store
    client = (
        FeishuMessageClient(config.app_id, config.app_secret)
        if client is None
        else client
    )
    if worker is None:
        worker = NotificationWorker(
            store,
            client,
            chat_id=config.demo_chat_id,
            web_url=config.streamlit_public_url,
            feishu_wiki_url=config.feishu_wiki_url,
        )
    if gate_store is None:
        gate_store = DecisionGateStore(config.gate_db or (config.notification_db.parent / "feishu_gate_jobs.sqlite3"))
    if replay_state_loader is None:
        replay_state_loader = _default_replay_state_loader(config)

    def on_message(event: Any) -> None:
        try:
            _on_message_inner(event)
        except Exception:
            traceback.print_exc()
            print("[handler] on_message failed; exception printed above", flush=True)

    def _on_message_inner(event: Any) -> None:
        event_dict = _event_to_mapping(event, lark)
        normalized = normalize_message_event(event_dict)
        message = normalized["message"]
        message_id = _require_text(message.get("message_id"), "message.message_id")

        if _sender_open_id(normalized) == config.bot_open_id:
            return

        # This lookup intentionally precedes parsing and every network call.  It
        # closes the crash window between CardKit creation and message reply.
        replay = _find_replay(store, message_id)
        if replay is not None:
            if replay.message_id is not None:
                return
            if replay.card_id is None:
                return
            _reply_existing_replay(
                store,
                client,
                replay,
                source_message_id=message_id,
            )
            return

        decision = handle_query(
            normalized,
            bot_open_id=config.bot_open_id,
            store=store,
        )
        if isinstance(decision, QueryReply):
            _create_and_reply_replay(
                store,
                client,
                decision,
                web_url=config.streamlit_public_url,
            )
            return

        if _is_decision_status_command(normalized, config.bot_open_id):
            _reply_decision_status(config, client, message_id, message.get("chat_id"))
            return

        if _help_is_allowed(normalized, config.bot_open_id):
            content = decision.toast_content if isinstance(decision, CallbackResult) else None
            _reply_help(client, message_id, message.get("chat_id"), content=content)

    def on_card_action(event: Any) -> Any:
        try:
            return _on_card_action_inner(event)
        except Exception:
            traceback.print_exc()
            print("[handler] on_card_action failed; exception printed above", flush=True)
            return _adapt_card_action_response(
                lark,
                CallbackResult(
                    toast_type="error",
                    toast_content="机器人内部错误，请查看本机日志。",
                ),
            )

    def _on_card_action_inner(event: Any) -> Any:
        event_dict = _event_to_mapping(event, lark)
        payload = _extract_card_action_payload(event_dict)
        if isinstance(payload, Mapping) and payload.get("type") == WRITEBACK_RETRY_ACTION_TYPE:
            result = handle_writeback_retry_action(
                payload,
                operator_open_id=_extract_operator_open_id(event_dict),
                allowed_operator_ids=config.operator_open_ids,
                store=(
                    WritebackStore(config.writeback_db)
                    if config.writeback_db is not None
                    else None
                ),
            )
        elif isinstance(payload, Mapping) and payload.get("type") == DECISION_GATE_ACTION_TYPE:
            result = handle_decision_gate_action(
                payload,
                operator_open_id=_extract_operator_open_id(event_dict),
                allowed_operator_ids=config.operator_open_ids,
                load_state=replay_state_loader
                if config.gate_mode == "replay"
                else lambda run_id: (
                    read_decision_run_state(config.decision_run_root, run_id)
                    if config.decision_run_root is not None
                    else None
                ),
                gate_store=gate_store,
                gate_enabled=gate_enabled,
            )
        else:
            result = handle_replay_action(
                payload,
                operator_open_id=_extract_operator_open_id(event_dict),
                allowed_operator_ids=config.operator_open_ids,
                store=store,
            )
        return _adapt_card_action_response(lark, result)

    return (
        lark.EventDispatcherHandler.builder(
            getattr(lark, "ENCRYPT_KEY", "") or "",
            getattr(lark, "VERIFICATION_TOKEN", "") or "",
            lark.LogLevel.DEBUG,
        )
        .register_p2_im_message_receive_v1(on_message)
        .register_p2_card_action_trigger(on_card_action)
        .build()
    )


def _build_runtime(
    config: BotConfig, *, lark: Any | None = None
) -> tuple[Any, NotificationWorker, WritebackWorker | None, Any | None, DecisionGateStore | None]:
    lark = _import_lark_oapi() if lark is None else lark
    store = NotificationStore(config.notification_db)
    client = FeishuMessageClient(config.app_id, config.app_secret)
    worker = NotificationWorker(
        store,
        client,
        chat_id=config.demo_chat_id,
        web_url=config.streamlit_public_url,
        feishu_wiki_url=config.feishu_wiki_url,
    )
    writeback_worker = _build_writeback_worker(config)
    gate_store = (
        DecisionGateStore(config.gate_db)
        if config.gate_db is not None
        else None
    )
    if config.gate_mode == "replay":
        from src.integrations.feishu.replay_gate_executor import ReplayGateExecutor

        writeback_store_for_replay = (
            WritebackStore(config.writeback_db)
            if config.writeback_db is not None and writeback_worker is not None
            else None
        )
        gate_executor = ReplayGateExecutor(
            run_root=config.decision_run_root,
            state_path=config.replay_state_path,
            message_client=client,
            chat_id=config.demo_chat_id,
            writeback_store=writeback_store_for_replay,
            notification_store=store,
        )
    else:
        writeback_store_for_gates = (
            WritebackStore(config.writeback_db)
            if config.writeback_db is not None and writeback_worker is not None
            else None
        )
        gate_executor = _build_gate_executor(
            config,
            client,
            writeback_store_for_gates,
            notification_store=store,
        )
    handler = build_event_handler(
        config,
        lark=lark,
        store=store,
        client=client,
        worker=worker,
        gate_store=gate_store,
        gate_enabled=gate_executor is not None,
    )
    return handler, worker, writeback_worker, gate_executor, gate_store


def _build_gate_executor(
    config: BotConfig,
    client: Any,
    writeback_store: Any,
    *,
    notification_store: NotificationStore | None = None,
) -> Any | None:
    """决策门执行器仅在配置了 run 根目录后启用；LLM 栈延迟导入。"""

    if config.decision_run_root is None:
        return None
    from src.integrations.feishu.gate_executor import DecisionGateExecutor
    from tools.run_official_decision import build_gate_coordinator

    return DecisionGateExecutor(
        run_root=config.decision_run_root,
        message_client=client,
        chat_id=config.demo_chat_id,
        writeback_store=writeback_store,
        coordinator_factory=build_gate_coordinator,
        notification_store=notification_store,
    )


def _build_writeback_worker(config: BotConfig) -> WritebackWorker | None:
    """配置了任一写回 URL 才启用；盘点失败直接抛错，不降级。"""

    if config.writeback_run_log_url is None and config.writeback_results_url is None:
        return None
    if config.writeback_db is None:
        raise ValueError("writeback database path is required")
    bitable = FeishuBitableClient(config.app_id, config.app_secret)
    targets = inventory_writeback_targets(
        bitable,
        run_log_url=config.writeback_run_log_url,
        results_url=config.writeback_results_url,
    )
    if not targets:
        raise RuntimeError("写回已配置但没有任何可用目标表")
    store = WritebackStore(config.writeback_db)
    return WritebackWorker(store, bitable, targets)


class ReplayFeedbackLoop:
    """推送远端推进反馈、门入口卡，并自动前翻已解锁的叙事回放卡。"""

    def __init__(self, config: BotConfig, store: NotificationStore, client: Any) -> None:
        self._config = config
        self._store = store
        self._client = client

    def run_forever(self, stop_event: Any) -> None:
        from threading import Event

        if not isinstance(stop_event, Event):
            raise TypeError("stop_event must be a threading.Event")
        while not stop_event.is_set():
            try:
                self._tick()
            except Exception:
                traceback.print_exc()
                print("[worker] feishu-replay-feedback tick failed; printed above", flush=True)
            stop_event.wait(0.6)

    def _tick(self) -> None:
        session = self._store.active_session()
        if session is None:
            return
        for command in self._store.commands_pending_feedback(session.session_id):
            self._send_plain_card(
                title="系统推进",
                content=str(command.result_text),
                summary="系统推进",
                uuid=self._uuid("mj-stepfb", command.command_id),
            )
            self._store.mark_feedback_sent(command.command_id)

        ordinals = [
            job.ordinal
            for job in self._store.jobs_for_session(session.session_id)
        ]
        available = set(ordinals)
        cursor = max(ordinals) if ordinals else 0
        if cursor >= 3:
            self._maybe_send_gate_entry(session.session_id)

        from src.integrations.feishu.notification_bot import _replay_update_uuid

        for card in self._store.replay_cards_for_session(session.session_id):
            target = card.current_ordinal + 1
            if card.update_status in ("IDLE", "FAILED") and target in available:
                try:
                    self._store.request_replay_update(
                        card.replay_id,
                        target_ordinal=target,
                        update_uuid=_replay_update_uuid(card.replay_id, card.sequence, target),
                    )
                except ValueError:
                    pass

    def _maybe_send_gate_entry(self, session_id: str) -> None:
        from src.services.replay_gate_timeline import (
            initial_replay_state,
            read_replay_state,
        )

        timeline = _load_replay_timeline_cached(self._config)
        state = read_replay_state(self._config.replay_state_path, timeline.run_id)
        if state is None:
            state = initial_replay_state(timeline.run_id)
        if state.get("stage") != "AWAITING_SELECTION":
            return
        if not self._store.record_processed_event(
            f"gate_entry:{session_id}", _now()
        ):
            return
        from src.integrations.feishu.replay_gate_executor import replay_gate_elements

        elements = replay_gate_elements(
            timeline, timeline.run_id, "AWAITING_SELECTION", prefix="gentry"
        )
        card = {
            "schema": "2.0",
            "config": {"update_multi": True, "summary": {"content": "决策门已解锁"}},
            "header": {
                "title": {"tag": "plain_text", "content": "决策门已解锁：提交人工选线"},
                "template": "turquoise",
            },
            "body": {"elements": elements},
        }
        validate_card(card)
        self._client.send_card(self._config.demo_chat_id, card, uuid=self._uuid("mj-gentry", session_id))

    def _send_plain_card(self, *, title: str, content: str, summary: str, uuid: str) -> None:
        card = {
            "schema": "2.0",
            "config": {"update_multi": True, "summary": {"content": summary}},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "turquoise",
            },
            "body": {
                "elements": [
                    {
                        "tag": "div",
                        "element_id": "feedback_content",
                        "text": {"tag": "lark_md", "content": content},
                    }
                ]
            },
        }
        validate_card(card)
        self._client.send_card(self._config.demo_chat_id, card, uuid=uuid)

    @staticmethod
    def _uuid(prefix: str, identifier: Any) -> str:
        raw = f"{prefix}:{identifier}".encode("utf-8")
        return ("mj-fb-" + hashlib.sha256(raw).hexdigest())[:48]


class WritebackFeedbackLoop:
    """把真实写入结果送回飞书，并为最终失败任务提供重试按钮。"""

    def __init__(self, store: WritebackStore, client: Any, chat_id: str) -> None:
        self._store = store
        self._client = client
        self._chat_id = chat_id

    def run_forever(self, stop_event: Any) -> None:
        from threading import Event

        if not isinstance(stop_event, Event):
            raise TypeError("stop_event must be a threading.Event")
        while not stop_event.is_set():
            try:
                self._tick()
            except Exception:
                traceback.print_exc()
                print("[worker] feishu-writeback-feedback tick failed; printed above", flush=True)
            stop_event.wait(0.6)

    def _tick(self) -> None:
        for job in self._store.pending_feedback_jobs():
            jobs = self._store.jobs_for_demo_run(job.demo_run_id or "")
            wrote = sum(item.status == "WROTE" for item in jobs)
            total = len(jobs)
            table_label = "日志表" if job.table_key == "RUN_LOG" else "结果表"
            if job.status == "WROTE":
                title = "飞书写回完成"
                template = "turquoise"
                status_line = f"本轮实际已写入：{wrote}/{total} 条"
            else:
                title = "飞书写回失败"
                template = "red"
                status_line = f"本轮实际已写入：{wrote}/{total} 条；当前记录失败"
            content = (
                f"demo_run_id：{job.demo_run_id}\n"
                f"当前记录：{table_label} · {job.record_kind}\n"
                f"{status_line}"
            )
            if job.last_error:
                content += f"\n原因：{job.last_error}"
            if job.target_table_url:
                content += f"\n[打开{table_label}]({job.target_table_url})"
            elements: list[dict[str, Any]] = [
                {
                    "tag": "div",
                    "element_id": f"wb_result_{job.job_id}",
                    "text": {"tag": "lark_md", "content": content},
                }
            ]
            if job.status == "FAILED":
                elements.append(
                    _button(
                        f"wb_retry_{job.job_id}",
                        "重试写回",
                        {"type": WRITEBACK_RETRY_ACTION_TYPE, "job_id": job.job_id},
                        button_type="primary",
                    )
                )
            card = {
                "schema": "2.0",
                "config": {
                    "update_multi": True,
                    "summary": {"content": title},
                },
                "header": {
                    "title": {"tag": "plain_text", "content": title},
                    "template": template,
                },
                "body": {"elements": elements},
            }
            validate_card(card)
            self._client.send_card(
                self._chat_id,
                card,
                uuid=self._uuid(job.job_id, job.status),
            )
            self._store.mark_feedback_sent(job.job_id)

    @staticmethod
    def _uuid(job_id: int, status: str) -> str:
        raw = f"writeback:{job_id}:{status}".encode("utf-8")
        return ("mj-wb-fb-" + hashlib.sha256(raw).hexdigest())[:48]


def handle_writeback_retry_action(
    payload: Mapping[str, Any],
    *,
    operator_open_id: str,
    allowed_operator_ids: set[str] | frozenset[str],
    store: WritebackStore | None,
) -> CallbackResult:
    if operator_open_id not in allowed_operator_ids:
        return CallbackResult(toast_type="warning", toast_content="你没有权限重试写回。")
    job_id = payload.get("job_id") if isinstance(payload, Mapping) else None
    if not isinstance(job_id, int) or isinstance(job_id, bool):
        return CallbackResult(toast_type="error", toast_content="写回任务编号无效。")
    if store is None:
        return CallbackResult(toast_type="error", toast_content="写回未启用。")
    try:
        store.retry_failed(job_id)
    except (KeyError, ValueError) as exc:
        return CallbackResult(toast_type="warning", toast_content=str(exc))
    return CallbackResult(toast_type="success", toast_content="已重新排队写回任务。")


def run_runtime(config: BotConfig, *, lark: Any | None = None) -> None:
    lark = _import_lark_oapi() if lark is None else lark
    event_handler, worker, writeback_worker, gate_executor, gate_store = _build_runtime(config, lark=lark)
    stop_event = threading.Event()
    worker_errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            worker.run_forever(stop_event)
        except BaseException as exc:
            print(f"[worker] notification worker crashed: {exc!r}", flush=True)
            worker_errors.append(exc)
        else:
            if not stop_event.is_set():
                print("[worker] notification worker exited unexpectedly", flush=True)
                worker_errors.append(RuntimeError("notification worker exited unexpectedly"))
        finally:
            stop_event.set()

    thread = threading.Thread(
        target=run_worker,
        name="feishu-notification-worker",
        daemon=True,
    )
    thread.start()

    extra_threads: list[tuple[str, Any]] = []
    if writeback_worker is not None:
        extra_threads.append(("feishu-writeback-worker", writeback_worker))
        extra_threads.append(
            (
                "feishu-writeback-feedback",
                WritebackFeedbackLoop(
                    WritebackStore(config.writeback_db),
                    FeishuMessageClient(config.app_id, config.app_secret),
                    config.demo_chat_id,
                ),
            )
        )
    if config.gate_mode == "replay":
        extra_threads.append(
            (
                "feishu-replay-feedback",
                ReplayFeedbackLoop(
                    config,
                    NotificationStore(config.notification_db),
                    FeishuMessageClient(config.app_id, config.app_secret),
                ),
            )
        )
    if gate_executor is not None and gate_store is not None:
        from src.integrations.feishu.decision_gate_store import DecisionGateRunner

        extra_threads.append(
            ("feishu-gate-runner", DecisionGateRunner(gate_store, gate_executor))
        )

    for name, runnable in extra_threads:
        def run_extra(runnable=runnable) -> None:
            try:
                runnable.run_forever(stop_event)
            except BaseException as exc:
                print(f"[worker] {name} crashed: {exc!r}", flush=True)
                worker_errors.append(exc)
            finally:
                stop_event.set()

        threading.Thread(target=run_extra, name=name, daemon=True).start()

    try:
        lark.ws.Client(
            config.app_id,
            config.app_secret,
            event_handler=event_handler,
            log_level=lark.LogLevel.DEBUG,
        ).start()
    finally:
        stop_event.set()
        thread.join(timeout=5)

    live_extras = False
    for name, _runnable in extra_threads:
        for item in threading.enumerate():
            if item.name == name and item.is_alive():
                live_extras = True
    if thread.is_alive() or live_extras:
        raise RuntimeError("background worker did not stop")
    if worker_errors:
        raise RuntimeError(
            f"background worker exited unexpectedly: {worker_errors[0]}"
        ) from worker_errors[0]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    config = load_bot_config(os.environ)
    config = replace(
        config,
        streamlit_public_url=resolve_streamlit_public_url(
            config.streamlit_public_url,
            environ=os.environ,
        ),
    )
    run_runtime(config)


def _import_lark_oapi() -> Any:
    import lark_oapi

    return lark_oapi


def _event_to_mapping(value: Any, lark: Any = None) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {key: _event_to_mapping(item, lark) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_event_to_mapping(item, lark) for item in value]

    json_api = getattr(lark, "JSON", None) if lark is not None else None
    marshal = getattr(json_api, "marshal", None)
    if callable(marshal):
        marshaled = marshal(value)
        if not isinstance(marshaled, str):
            raise ValueError("lark.JSON.marshal must return a JSON string")
        parsed = json.loads(marshaled)
        if not isinstance(parsed, Mapping):
            raise ValueError("lark.JSON.marshal must return an event object")
        return _event_to_mapping(parsed, lark)

    attributes = getattr(value, "__dict__", None)
    if not isinstance(attributes, dict):
        raise ValueError("event must be a mapping or SDK event model")
    return {
        key: _event_to_mapping(item, lark)
        for key, item in attributes.items()
        if not key.startswith("_")
    }


def _normalize_mentions(mentions: Any) -> list[dict[str, str]]:
    if mentions is None:
        return []
    if not isinstance(mentions, list):
        raise ValueError("mentions must be a list")
    normalized: list[dict[str, str]] = []
    for item in mentions:
        if not isinstance(item, Mapping):
            raise ValueError("mentions items must be objects")
        key = item.get("key")
        identifier: str | None = None
        mention_id = item.get("id")
        if isinstance(mention_id, Mapping):
            nested = mention_id.get("open_id")
            if isinstance(nested, str) and nested.strip():
                identifier = nested
        elif isinstance(mention_id, str) and mention_id.strip():
            identifier = mention_id
        elif isinstance(item.get("open_id"), str) and item["open_id"].strip():
            identifier = item["open_id"]
        if identifier is None:
            raise ValueError("mentions items must contain id or open_id")
        result = {"id": identifier}
        if isinstance(key, str) and key.strip():
            result["key"] = key
        normalized.append(result)
    return normalized


def _extract_sender_open_id(sender: Any) -> str:
    if not isinstance(sender, Mapping):
        raise ValueError("sender is required")
    sender_id = sender.get("sender_id")
    open_id = sender_id.get("open_id") if isinstance(sender_id, Mapping) else sender.get("open_id")
    return _require_text(open_id, "sender.sender_id.open_id")


def _sender_open_id(event: Mapping[str, Any]) -> str | None:
    sender = event.get("sender")
    if not isinstance(sender, Mapping):
        nested = event.get("event")
        sender = nested.get("sender") if isinstance(nested, Mapping) else None
    if not isinstance(sender, Mapping):
        return None
    sender_id = sender.get("sender_id")
    value = sender_id.get("open_id") if isinstance(sender_id, Mapping) else sender.get("open_id")
    return value if isinstance(value, str) else None


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _extract_card_action_payload(event: Any) -> dict[str, Any]:
    event_dict = _event_to_mapping(event)
    if not isinstance(event_dict, Mapping):
        raise ValueError("card action event must be a mapping")
    payload = event_dict.get("event")
    payload = payload if isinstance(payload, Mapping) else event_dict
    action = payload.get("action")
    if isinstance(action, Mapping) and isinstance(action.get("value"), Mapping):
        return dict(action["value"])
    raise ValueError("card action payload is invalid")


def _extract_operator_open_id(event: Any) -> str:
    event_dict = _event_to_mapping(event)
    if not isinstance(event_dict, Mapping):
        raise ValueError("card action event must be a mapping")
    payload = event_dict.get("event")
    payload = payload if isinstance(payload, Mapping) else event_dict
    operator = payload.get("operator")
    if not isinstance(operator, Mapping):
        raise ValueError("card action operator is required")
    return _require_text(operator.get("open_id"), "card action operator.open_id")


def _find_replay(store: NotificationStore, source_message_id: str) -> Any | None:
    session = store.active_session()
    if session is None:
        return None
    try:
        replay = store.get_replay_card(_replay_id(session.session_id, source_message_id))
    except KeyError:
        return None
    if replay.source_message_id != source_message_id:
        return None
    return replay


def _create_and_reply_replay(
    store: NotificationStore,
    client: Any,
    decision: QueryReply,
    *,
    web_url: str,
) -> None:
    jobs = store.jobs_for_session(decision.session_id)
    job = next((item for item in jobs if item.ordinal == decision.ordinal), None)
    if job is None:
        raise ValueError("query result has no matching notification job")
    snapshot = DemoNotificationSnapshot.model_validate_json(job.payload_json, strict=True)
    replay_id = _replay_id(decision.session_id, decision.source_message_id)
    reserved = store.reserve_replay_card(
        decision.session_id,
        source_message_id=decision.source_message_id,
        ordinal=decision.ordinal,
        reply_uuid=_reply_uuid(decision.session_id, decision.source_message_id),
    )
    if reserved.message_id is not None or reserved.card_id is not None:
        return
    if store.claim_replay_card_creation(replay_id) is None:
        return
    from src.integrations.feishu.notification_cards import build_replay_card

    card = build_replay_card(
        snapshot,
        replay_id=replay_id,
        current_ordinal=decision.ordinal,
        web_url=web_url,
    )
    card_id = client.create_card_entity(card)
    replay = store.bind_replay_card(replay_id, card_id=card_id)
    reply_message_id = client.reply_card_entity(
        decision.source_message_id,
        replay.card_id,
        uuid=replay.reply_uuid,
    )
    store.mark_replay_replied(
        replay.replay_id,
        message_id=reply_message_id,
        replied_at=datetime.now(timezone.utc),
    )
    store.record_processed_event(
        f"message:{decision.source_message_id}",
        datetime.now(timezone.utc),
    )


def _reply_existing_replay(
    store: NotificationStore,
    client: Any,
    replay: Any,
    *,
    source_message_id: str,
) -> None:
    reply_message_id = client.reply_card_entity(
        source_message_id,
        replay.card_id,
        uuid=replay.reply_uuid,
    )
    store.mark_replay_replied(
        replay.replay_id,
        message_id=reply_message_id,
        replied_at=datetime.now(timezone.utc),
    )
    store.record_processed_event(
        f"message:{source_message_id}",
        datetime.now(timezone.utc),
    )


def _reply_help(client: Any, source_message_id: str, chat_id: Any, *, content: str | None) -> None:
    _reply_plain_card(
        client,
        source_message_id,
        chat_id,
        title="梅见助手",
        content=content or "支持的命令：远程操控（远端单步推进网页）、决策进度",
        summary="帮助",
    )


def _default_replay_state_loader(config: BotConfig) -> Any:
    """replay 模式的 load_state：读回放状态文件，校验 run 一致。"""

    from src.services.replay_gate_timeline import (
        initial_replay_state,
        load_replay_timeline,
        read_replay_state,
    )

    timeline = load_replay_timeline(config.decision_run_root)

    def load_state(run_id: str) -> dict[str, Any] | None:
        if run_id != timeline.run_id:
            return None
        state = read_replay_state(config.replay_state_path, timeline.run_id)
        if state is None:
            return initial_replay_state(timeline.run_id)
        return state

    return load_state


def _reply_decision_status(
    config: BotConfig, client: Any, source_message_id: str, chat_id: Any
) -> None:
    gate_elements: list[dict[str, Any]] = []
    gate_elements: list[dict[str, Any]] = []
    gate_elements: list[dict[str, Any]] = []
    if config.gate_mode == "replay":
        timeline = _load_replay_timeline_cached(config)
        state = _default_replay_state_loader(config)(timeline.run_id)
        stage = state.get("stage") if isinstance(state, dict) else None
        history = state.get("history") if isinstance(state, dict) else []
        done = len(history) if isinstance(history, list) else 0
        progress_store = NotificationStore(config.notification_db)
        session = progress_store.active_session()
        if session is None:
            content = (
                "当前没有活动连接：请先在网页端「连接助手」。\n"
                "连接后，可用叙事卡片「下一步」远端单步推进网页，"
                "每步都会推送反馈卡片。"
            )
        else:
            progress = progress_store.get_demo_progress(session.session_id)
            content = (
                f"决策 run：{timeline.run_id}\n"
                f"模式：冻结回放（读取官方演化 run 冻结产物，零模型调用）\n"
                f"系统当前进度：{progress or '尚未同步（等待网页首次推进）'}\n"
                f"决策阶段：{stage}（已推进 {done} 个门）\n"
                "提示：叙事卡片「下一步」= 远端单步推进网页；"
                "选线门卡片会在盲选里程碑自动推送。"
            )
    elif config.decision_run_root is None:
        content = "决策推进未启用：本机未配置 FEISHU_DECISION_RUN_ROOT。"
    else:
        state = read_active_decision_state(config.decision_run_root)
        if state is None:
            content = "当前没有活动决策 run。"
        else:
            content = stage_status_text(state)
            from src.integrations.feishu.gate_executor import next_gate_elements

            gate_elements = next_gate_elements(
                state["run_id"],
                state["stage"],
                prefix="status",
                run_root=config.decision_run_root,
            )
    _reply_plain_card(
        client,
        source_message_id,
        chat_id,
        title="决策进度",
        content=content,
        summary="决策进度",
        extra_elements=gate_elements,
    )


def _load_replay_timeline_cached(config: BotConfig) -> Any:
    from src.services.replay_gate_timeline import load_replay_timeline

    return load_replay_timeline(config.decision_run_root)


def _reply_plain_card(
    client: Any,
    source_message_id: str,
    chat_id: Any,
    *,
    title: str,
    content: str,
    summary: str,
    extra_elements: list[dict[str, Any]] | None = None,
) -> None:
    if not isinstance(chat_id, str) or not chat_id.strip():
        raise ValueError("message.chat_id is required")
    elements = [
        {
            "tag": "div",
            "element_id": "plain_content",
            "text": {"tag": "lark_md", "content": content},
        }
    ]
    elements.extend(extra_elements or [])
    card = {
        "schema": "2.0",
        "config": {"update_multi": True, "summary": {"content": summary}},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": "turquoise",
        },
        "body": {"elements": elements},
    }
    validate_card(card)
    card_id = client.create_card_entity(card)
    client.reply_card_entity(
        source_message_id,
        card_id,
        uuid=_reply_uuid(summary, source_message_id),
    )


def _is_decision_status_command(event: Mapping[str, Any], bot_open_id: str) -> bool:
    if not _help_is_allowed(event, bot_open_id):
        return False
    text = _plain_command_text(event, bot_open_id)
    return text in DECISION_STATUS_COMMANDS


def _plain_command_text(event: Mapping[str, Any], bot_open_id: str) -> str | None:
    message = event.get("message")
    if not isinstance(message, Mapping):
        return None
    content = message.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except ValueError:
            return None
    if not isinstance(content, Mapping) or not isinstance(content.get("text"), str):
        return None
    text = content["text"]
    mentions = message.get("mentions")
    if isinstance(mentions, list):
        for mention in mentions:
            if _mention_open_id(mention) == bot_open_id:
                key = mention.get("key") if isinstance(mention, Mapping) else None
                if isinstance(key, str) and key:
                    text = text.replace(key, "", 1)
    return text.strip() or None


def _help_is_allowed(event: Mapping[str, Any], bot_open_id: str) -> bool:
    message = event.get("message")
    if not isinstance(message, Mapping):
        return False
    if message.get("chat_type") == "p2p":
        return True
    if message.get("chat_type") != "group":
        return False
    mentions = message.get("mentions", [])
    if not isinstance(mentions, list):
        return False
    return any(_mention_open_id(item) == bot_open_id for item in mentions)


def _mention_open_id(mention: Any) -> str | None:
    if not isinstance(mention, Mapping):
        return None
    value = mention.get("id")
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and isinstance(value.get("open_id"), str):
        return value["open_id"]
    if isinstance(mention.get("open_id"), str):
        return mention["open_id"]
    return None


def _replay_id(session_id: str, source_message_id: str) -> str:
    return "replay-" + hashlib.sha256(
        f"{session_id}:{source_message_id}".encode("utf-8")
    ).hexdigest()[:32]


def _reply_uuid(session_id: str, source_message_id: str) -> str:
    return "mj-reply-" + hashlib.sha256(
        f"{session_id}:{source_message_id}".encode("utf-8")
    ).hexdigest()[:32]


def _adapt_card_action_response(lark: Any, result: CallbackResult) -> Any:
    payload = {
        "toast": {
            "type": result.toast_type,
            "content": result.toast_content,
        }
    }
    response_type = getattr(lark, "P2CardActionTriggerResponse", None)
    if response_type is None:
        class LocalResponse:
            def __init__(self, value: Mapping[str, Any]):
                self.payload = dict(value)

        response_type = LocalResponse
    return response_type(payload)


def _now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    main()
