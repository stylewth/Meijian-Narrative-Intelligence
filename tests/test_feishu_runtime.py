"""飞书集成运行时合同测试：token 刷新、失效重试、目录自建与配置透传。"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

from src.config import LLMResponseFormat, Settings
from src.integrations.feishu.auth import (
    AUTH_URL,
    FeishuAuthError,
    TenantAccessTokenProvider,
    get_tenant_access_token_details,
)
from src.integrations.feishu.client import FeishuAPIError, FeishuBitableClient
from src.integrations.feishu.message_client import (
    REQUEST_TIMEOUT,
    FeishuMessageClient,
)
from src.integrations.feishu.notification_store import NotificationStore
from src.integrations.feishu.writeback_targets import WritebackTarget
from src.integrations.feishu.notification_worker import RETRY_DELAYS_SECONDS
from src.llm_client import LLMClient
from src.prompt_loader import (
    PROMPT_NAMES,
    REPLAY_V2_PROMPT_VERSIONS,
    load_prompt_metadata,
)
from tools.run_feishu_bot import REQUIRED_CONFIG_KEYS, load_bot_config


TOKEN_INVALID_CODE = 99991663


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeSession:
    """按顺序返回预置响应，并记录每次请求的 URL。"""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.requested_urls: list[str] = []
        self.requests: list[tuple[str, str, dict]] = []

    def request(self, method: str, url: str, **kwargs) -> FakeResponse:
        self.requested_urls.append(url)
        self.requests.append((method, url, kwargs))
        return FakeResponse(self._responses.pop(0))

    @property
    def auth_call_count(self) -> int:
        return sum(1 for url in self.requested_urls if url == AUTH_URL)


def _auth_payload(token: str, expire: int = 7200) -> dict:
    return {"code": 0, "tenant_access_token": token, "expire": expire}


def _token_payload(**extra: object) -> dict:
    payload = {"code": 0, "tenant_access_token": "t-1"}
    payload.update(extra)
    return payload


def test_token_provider_caches_until_expiry() -> None:
    session = FakeSession([_auth_payload("t-1", expire=100)])
    now = [0.0]
    provider = TenantAccessTokenProvider(
        "app-id", "app-secret", session=session, clock=lambda: now[0]
    )

    assert provider.get_token() == "t-1"
    now[0] = 0.5
    assert provider.get_token() == "t-1"
    assert session.auth_call_count == 1

    now[0] = 2.0
    session._responses.append(_auth_payload("t-2", expire=100))
    assert provider.get_token() == "t-2"
    assert session.auth_call_count == 2


def test_token_provider_force_refresh_bypasses_cache() -> None:
    session = FakeSession([_auth_payload("t-1"), _auth_payload("t-2")])
    provider = TenantAccessTokenProvider(
        "app-id", "app-secret", session=session, clock=lambda: 0.0
    )

    assert provider.get_token() == "t-1"
    assert provider.get_token(force_refresh=True) == "t-2"
    assert session.auth_call_count == 2


def test_token_details_missing_expire_is_rejected() -> None:
    session = FakeSession([_token_payload()])
    with pytest.raises(FeishuAuthError, match="expire"):
        get_tenant_access_token_details("app-id", "app-secret", session=session)


def test_bitable_client_refreshes_token_and_retries_once() -> None:
    session = FakeSession(
        [
            _auth_payload("t-1"),
            {"code": TOKEN_INVALID_CODE, "msg": "invalid access token"},
            _auth_payload("t-2"),
            {"code": 0, "data": {"items": [], "has_more": False}},
        ]
    )
    client = FeishuBitableClient("app-id", "app-secret", session=session)

    assert client.get_tables("app-token") == []
    assert session.auth_call_count == 2


def test_bitable_client_raises_when_token_still_invalid() -> None:
    session = FakeSession(
        [
            _auth_payload("t-1"),
            {"code": TOKEN_INVALID_CODE, "msg": "invalid access token"},
            _auth_payload("t-2"),
            {"code": TOKEN_INVALID_CODE, "msg": "invalid access token"},
        ]
    )
    client = FeishuBitableClient("app-id", "app-secret", session=session)

    with pytest.raises(FeishuAPIError, match=str(TOKEN_INVALID_CODE)):
        client.get_tables("app-token")


def test_bitable_client_creates_table_with_text_fields() -> None:
    session = FakeSession(
        [
            _auth_payload("t-1"),
            {"code": 0, "data": {"table_id": "tbl_new"}},
        ]
    )
    client = FeishuBitableClient("app-id", "app-secret", session=session)

    table_id = client.create_table(
        "app-token",
        "梅见-连接-20260909-0830-ABCD1234-日志",
        [{"field_name": "run_id", "type": 1}],
    )

    assert table_id == "tbl_new"
    method, url, kwargs = session.requests[-1]
    assert method == "POST"
    assert url.endswith("/bitable/v1/apps/app-token/tables")
    assert kwargs["json"]["table"]["fields"] == [{"field_name": "run_id", "type": 1}]


def test_bitable_client_updates_an_existing_record() -> None:
    session = FakeSession(
        [
            _auth_payload("t-1"),
            {"code": 0, "data": {"record": {"record_id": "rec_existing"}}},
        ]
    )
    client = FeishuBitableClient("app-id", "app-secret", session=session)

    record_id = client.update_record(
        "app-token",
        "tbl-demo",
        "rec_existing",
        {"summary": "当前进度已更新"},
    )

    assert record_id == "rec_existing"
    method, url, kwargs = session.requests[-1]
    assert method == "PUT"
    assert url.endswith(
        "/bitable/v1/apps/app-token/tables/tbl-demo/records/rec_existing"
    )
    assert kwargs["json"] == {"fields": {"summary": "当前进度已更新"}}


def test_message_client_refreshes_token_and_retries_once() -> None:
    session = FakeSession(
        [
            _auth_payload("t-1"),
            {"code": TOKEN_INVALID_CODE, "msg": "invalid access token"},
            _auth_payload("t-2"),
            {"code": 0, "data": {"message_id": "om_new"}},
        ]
    )
    client = FeishuMessageClient("app-id", "app-secret", session=session)

    message_id = client.reply_card_entity("om_x", "card_x", uuid="u-1")
    assert message_id == "om_new"
    assert session.auth_call_count == 2


def test_network_timeouts_keep_demo_safe_margins() -> None:
    assert REQUEST_TIMEOUT[0] >= 3.0
    assert REQUEST_TIMEOUT[1] >= 10.0
    assert min(RETRY_DELAYS_SECONDS) >= 1.0


def test_load_bot_config_creates_missing_parent_directory(tmp_path: Path) -> None:
    mapping = {key: "value" for key in REQUIRED_CONFIG_KEYS}
    mapping["FEISHU_NOTIFICATION_DB"] = "outputs/sub/notifications.sqlite3"

    config = load_bot_config(mapping, workspace_root=tmp_path)

    assert config.notification_db.parent.is_dir()
    assert config.notification_db == (tmp_path / "outputs/sub/notifications.sqlite3").resolve()


def test_notification_store_persists_run_scoped_writeback_targets(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path / "notifications.sqlite3")
    targets = {
        key: WritebackTarget(
            table_key=key,
            app_token="app_demo",
            table_id=f"tbl_{key.lower()}",
            field_names=("run_id", "demo_run_id"),
            table_url=f"https://demo.feishu.cn/base/app_demo?table=tbl_{key.lower()}",
        )
        for key in ("RUN_LOG", "RESULTS")
    }

    session = store.create_session(
        "run-source",
        created_by="streamlit-local",
        session_id="demo-1",
        created_at="2026-09-09T08:30:00+00:00",
        connected_at="2026-09-09T08:30:03+00:00",
        writeback_targets=targets,
    )

    assert session.session_id == "demo-1"
    assert session.created_at == "2026-09-09T08:30:00+00:00"
    assert session.connected_at == "2026-09-09T08:30:03+00:00"
    assert store.writeback_targets_for_session("demo-1") == targets
    assert store.active_writeback_targets() == targets


def test_notification_store_normalizes_reload_like_writeback_targets(
    tmp_path: Path,
) -> None:
    store = NotificationStore(tmp_path / "notifications.sqlite3")
    targets = {
        key: SimpleNamespace(
            table_key=key,
            app_token="app_demo",
            table_id=f"tbl_{key.lower()}",
            field_names=("run_id", "demo_run_id"),
            table_url=f"https://demo.feishu.cn/base/app_demo?table=tbl_{key.lower()}",
        )
        for key in ("RUN_LOG", "RESULTS")
    }

    store.create_session(
        "run-source",
        created_by="streamlit-local",
        session_id="reload-like",
        writeback_targets=targets,
    )

    persisted = store.writeback_targets_for_session("reload-like")
    assert all(isinstance(target, WritebackTarget) for target in persisted.values())
    assert persisted["RUN_LOG"].table_id == "tbl_run_log"


def test_notification_store_migrates_connected_at_column(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy-notifications.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute(
        "CREATE TABLE demo_sessions ("
        "session_id TEXT PRIMARY KEY,"
        "source_run_id TEXT NOT NULL,"
        "created_by TEXT NOT NULL,"
        "created_at TEXT NOT NULL,"
        "status TEXT NOT NULL)"
    )
    connection.commit()
    connection.close()

    store = NotificationStore(database_path)
    with sqlite3.connect(database_path) as verify:
        columns = {
            row[1]
            for row in verify.execute("PRAGMA table_info(demo_sessions)")
        }

    assert "connected_at" in columns
    session = store.create_session(
        "run-source",
        created_by="streamlit-local",
        created_at="2026-09-09T08:30:00+00:00",
        connected_at="2026-09-09T08:30:03+00:00",
    )
    assert session.connected_at == "2026-09-09T08:30:03+00:00"


class _DemoOutput(BaseModel):
    label: str
    model_config = ConfigDict(extra="forbid")


def _capturing_llm_client(content: str) -> SimpleNamespace:
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), captured=captured)


def test_llm_client_respects_configured_reasoning_effort() -> None:
    fake = _capturing_llm_client('{"label":"ok"}')
    settings = Settings(
        llm_api_key="key",
        llm_model="model",
        llm_response_format=LLMResponseFormat.JSON_SCHEMA,
        llm_thinking_enabled=True,
        llm_reasoning_effort="high",
    )

    result = LLMClient(settings, client=fake).generate_json(
        system_prompt="s", user_prompt="u", response_model=_DemoOutput
    )

    assert result.label == "ok"
    assert fake.captured["reasoning_effort"] == "high"


def test_llm_client_defaults_reasoning_effort_to_max() -> None:
    fake = _capturing_llm_client('{"label":"ok"}')
    settings = Settings(
        llm_api_key="key",
        llm_model="model",
        llm_thinking_enabled=True,
        llm_reasoning_effort=None,
    )

    LLMClient(settings, client=fake).generate_json(
        system_prompt="s", user_prompt="u", response_model=_DemoOutput
    )

    assert fake.captured["reasoning_effort"] == "max"


def test_removed_prompt_names_are_rejected_and_history_kept() -> None:
    assert not PROMPT_NAMES & {"single_comment", "template_check", "final_refinement"}
    for name in ("single_comment", "template_check", "final_refinement"):
        with pytest.raises(ValueError, match="未知 Prompt"):
            load_prompt_metadata(name)

    assert REPLAY_V2_PROMPT_VERSIONS["candidate_generation"] == "v3"
    assert REPLAY_V2_PROMPT_VERSIONS["template_check"] == "v2"
