from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = str(ROOT / "app.py")


@pytest.fixture(autouse=True)
def _public_no_key_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """冒烟测试验证的是公开无密钥行为，必须与本机 .env 隔离。"""

    for key in (
        "LLM_API_KEY",
        "LLM_MODEL",
        "FEISHU_APP_ID",
        "FEISHU_APP_SECRET",
        "FEISHU_DEMO_CHAT_ID",
        "FEISHU_BOT_OPEN_ID",
        "FEISHU_OPERATOR_OPEN_IDS",
        "FEISHU_NOTIFICATION_DB",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("dotenv.dotenv_values", lambda *args, **kwargs: {})


def button(app: AppTest, label: str):
    return next(item for item in app.button if item.label == label)


def visible_text(app: AppTest) -> str:
    groups = (app.markdown, app.caption, app.info, app.warning, app.success, app.title)
    return "\n".join(
        str(item.value)
        for group in groups
        for item in group
        if getattr(item, "value", None)
    )


def test_public_app_starts_without_secrets_and_enters_the_official_case() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=10)

    assert not app.exception
    assert not app.error
    assert "梅见品牌叙事智能决策系统" in visible_text(app)
    assert button(app, "进入 · 自定义使用")
    button(app, "进入 · 梅见案例展示").click().run(timeout=10)

    assert not app.exception
    assert "只读官方运行回放" in visible_text(app)
    assert not any(item.label == "运行入口" for item in app.selectbox)


def test_public_custom_entry_exposes_local_only_credentials_state() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=10)
    button(app, "进入 · 自定义使用").click().run(timeout=10)

    assert not app.exception
    assert "在线 AI 仅在本地配置后可用" in visible_text(app)
    local_feishu = button(app, "开始新的案例会话")
    assert local_feishu.disabled is True


def test_official_pressure_and_evolution_workspaces_render_from_frozen_data() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=10)
    app.session_state.system_entry = "梅见案例展示"
    app.session_state.active_workspace = "叙事压力测试"
    app.session_state.completed_workspaces = ["数据预处理"]
    app.run(timeout=10)

    assert not app.exception
    assert "叙事压力测试" in visible_text(app)

    app.session_state.active_workspace = "实时决策看板"
    app.session_state.completed_workspaces = ["数据预处理", "叙事压力测试"]
    app.run(timeout=10)

    assert not app.exception
    assert "实时叙事演化看板" in visible_text(app)
