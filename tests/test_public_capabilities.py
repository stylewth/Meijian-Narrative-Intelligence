from __future__ import annotations

from pathlib import Path
from typing import Any

from src.ui import demo_notification_control, system_gateway


class FakeStreamlit:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.buttons: list[tuple[str, dict[str, Any]]] = []

    def markdown(self, _: str) -> None:
        return None

    def caption(self, _: str) -> None:
        return None

    def info(self, message: str) -> None:
        self.infos.append(message)

    def error(self, message: str) -> None:
        raise AssertionError(f"missing local configuration must not render as an error: {message}")

    def button(self, label: str, **kwargs: Any) -> bool:
        self.buttons.append((label, kwargs))
        return False


def test_missing_credentials_are_reported_as_local_only_capabilities() -> None:
    capabilities = system_gateway.resolve_runtime_capabilities({})

    assert capabilities.online_ai_enabled is False
    assert capabilities.feishu_enabled is False
    assert capabilities.online_ai_status == "仅本地配置后可用"
    assert capabilities.feishu_status == "仅本地配置后触发"


def test_local_credentials_enable_their_matching_capabilities() -> None:
    capabilities = system_gateway.resolve_runtime_capabilities(
        {
            "LLM_API_KEY": "local-test-key",
            "LLM_MODEL": "local-test-model",
            "FEISHU_APP_ID": "local-app-id",
            "FEISHU_APP_SECRET": "local-app-secret",
            "FEISHU_DEMO_CHAT_ID": "local-chat-id",
            "FEISHU_BOT_OPEN_ID": "local-bot-id",
            "FEISHU_OPERATOR_OPEN_IDS": "local-operator-id",
            "FEISHU_NOTIFICATION_DB": "runtime/notifications.sqlite3",
        }
    )

    assert capabilities.online_ai_enabled is True
    assert capabilities.feishu_enabled is True
    assert capabilities.online_ai_status == "本地配置已就绪"
    assert capabilities.feishu_status == "本地配置已就绪"


def test_missing_feishu_configuration_renders_a_disabled_local_only_control() -> None:
    fake = FakeStreamlit()

    demo_notification_control.render_demo_notification_control(
        {},
        store=None,
        source_run_id="official-release",
        configuration={},
        workspace_root=Path.cwd(),
        streamlit_module=fake,
    )

    assert fake.infos == ["飞书仅在本地配置后触发；公开站不会连接团队飞书。"]
    assert fake.buttons == [("开始新的案例会话", {"disabled": True})]
