from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from threading import Event
from typing import Any

from src.ui import demo_notification_control, system_gateway
from src.integrations.feishu.writeback_targets import WritebackTarget
from tools.run_feishu_bot import REQUIRED_CONFIG_KEYS


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


class ConfirmConnectionStreamlit:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def markdown(self, message: str) -> None:
        self.events.append(("markdown", message))

    def caption(self, message: str) -> None:
        self.events.append(("caption", message))

    def warning(self, message: str) -> None:
        self.events.append(("warning", message))

    def info(self, message: str) -> None:
        self.events.append(("info", message))

    def success(self, message: str) -> None:
        self.events.append(("success", message))

    def error(self, message: str) -> None:
        raise AssertionError(f"connection should not fail: {message}")

    def button(self, label: str, **kwargs: Any) -> bool:
        self.events.append(("button", label))
        return label in {"连接助手", "确认连接助手"}


class ParentFragmentStreamlit(ConfirmConnectionStreamlit):
    def __init__(self) -> None:
        super().__init__()
        self.fragment_calls = 0

    def fragment(self, **kwargs: Any):
        self.fragment_calls += 1

        def decorate(function):
            return function

        return decorate


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


def test_feishu_header_status_requires_the_current_page_connection() -> None:
    configuration = {key: "local-value" for key in REQUIRED_CONFIG_KEYS}

    assert (
        system_gateway.feishu_connection_status(
            configuration,
            object(),
            page_connected=False,
        )
        == "未连接"
    )
    assert (
        system_gateway.feishu_connection_status(
            configuration,
            object(),
            page_connected=True,
        )
        == "已连接"
    )


def test_new_page_connection_state_completes_persisted_connection(tmp_path: Path) -> None:
    store = demo_notification_control.NotificationStore(
        tmp_path / "notifications.sqlite3"
    )
    store.create_session(
        "official-release",
        created_by="streamlit-local",
        session_id="old-page",
    )
    state: dict[str, Any] = {}

    demo_notification_control.initialize_feishu_page_connection(state, store)

    assert state[demo_notification_control._PAGE_CONNECTION_INITIALIZED_KEY] is True
    assert store.active_session() is None

    store.create_session(
        "official-release",
        created_by="streamlit-local",
        session_id="new-page",
    )
    demo_notification_control.initialize_feishu_page_connection(state, store)

    assert store.active_session().session_id == "new-page"


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
    assert fake.buttons == [("连接助手", {"disabled": True})]


def test_confirm_connection_surfaces_non_blocking_progress(
    tmp_path: Path, monkeypatch
) -> None:
    fake = ConfirmConnectionStreamlit()
    store = demo_notification_control.NotificationStore(
        tmp_path / "notifications.sqlite3"
    )
    configuration = {key: "local-value" for key in REQUIRED_CONFIG_KEYS}
    configuration.update(
        {
            "FEISHU_NOTIFICATION_DB": "notifications.sqlite3",
            "FEISHU_RUNLOG_URL": "https://demo.feishu.cn/base/app_demo?table=tbl_log",
            "FEISHU_RESULTS_URL": "https://demo.feishu.cn/base/app_demo?table=tbl_results",
        }
    )
    pending: Future[dict[str, WritebackTarget]] = Future()
    task = demo_notification_control._ConnectionTask(
        future=pending,
        source_run_id="official-release",
        demo_run_id="run-1",
        started_at="2026-09-09T08:30:00+00:00",
    )
    monkeypatch.setattr(
        demo_notification_control,
        "_start_connection_task",
        lambda **kwargs: task,
    )

    demo_notification_control.render_demo_notification_control(
        {},
        store=store,
        source_run_id="official-release",
        configuration=configuration,
        workspace_root=tmp_path,
        streamlit_module=fake,
    )

    assert any(
        name == "info" and "页面不会被锁定" in message
        for name, message in fake.events
    )


def test_connection_control_can_use_parent_fragment_polling(tmp_path: Path) -> None:
    fake = ParentFragmentStreamlit()
    store = demo_notification_control.NotificationStore(
        tmp_path / "notifications.sqlite3"
    )
    configuration = {key: "local-value" for key in REQUIRED_CONFIG_KEYS}
    configuration.update(
        {
            "FEISHU_NOTIFICATION_DB": "notifications.sqlite3",
            "FEISHU_RUNLOG_URL": "https://demo.feishu.cn/base/app_demo?table=tbl_log",
            "FEISHU_RESULTS_URL": "https://demo.feishu.cn/base/app_demo?table=tbl_results",
        }
    )
    pending: Future[dict[str, WritebackTarget]] = Future()
    state = {
        "feishu_connection_task": demo_notification_control._ConnectionTask(
            future=pending,
            source_run_id="official-release",
            demo_run_id="run-1",
            started_at="2026-09-09T08:30:00+00:00",
        )
    }

    demo_notification_control.render_demo_notification_control(
        state,
        store=store,
        source_run_id="official-release",
        configuration=configuration,
        workspace_root=tmp_path,
        streamlit_module=fake,
        use_connection_fragment=False,
    )

    assert fake.fragment_calls == 0
    assert any(name == "info" for name, _ in fake.events)


def test_connection_provisioning_runs_off_the_ui_thread(monkeypatch) -> None:
    started = Event()
    release = Event()

    def fake_provision(*args: Any, **kwargs: Any) -> dict[str, WritebackTarget]:
        started.set()
        release.wait(timeout=1)
        return {}

    monkeypatch.setattr(
        demo_notification_control,
        "FeishuBitableClient",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        demo_notification_control,
        "provision_demo_writeback_targets",
        fake_provision,
    )

    task = demo_notification_control._start_connection_task(
        app_id="app_demo",
        app_secret="secret",
        run_log_url="https://demo.feishu.cn/base/app_demo?table=tbl_log",
        results_url="https://demo.feishu.cn/base/app_demo?table=tbl_results",
        source_run_id="official-release",
        demo_run_id="run-1",
        started_at="2026-09-09T08:30:00+00:00",
    )

    assert started.wait(timeout=1)
    assert task.future.done() is False
    release.set()
    assert task.future.result(timeout=1) == {}


def test_connection_completion_keeps_current_progress_and_records_it(tmp_path) -> None:
    store = demo_notification_control.NotificationStore(
        tmp_path / "notifications.sqlite3"
    )
    targets = {
        key: WritebackTarget(
            table_key=key,
            app_token="app_demo",
            table_id=f"tbl_{key.lower()}",
            field_names=("run_id",),
            table_url=f"https://demo.feishu.cn/base/app_demo?table=tbl_{key.lower()}",
        )
        for key in ("RUN_LOG", "RESULTS")
    }
    pending: Future[dict[str, WritebackTarget]] = Future()
    pending.set_result(targets)
    state = {
        "preprocessing_replay_step": 2,
        "preprocessing_replay_completed_steps": [0],
        "active_workspace": "数据预处理",
        "feishu_connection_task": demo_notification_control._ConnectionTask(
            future=pending,
            source_run_id="official-release",
            demo_run_id="run-1",
            started_at="2026-09-09T08:30:00+00:00",
        ),
    }
    fake = ConfirmConnectionStreamlit()

    demo_notification_control._render_connection_task_once(
        fake,
        state,
        store,
        snapshots=(),
    )

    assert state["preprocessing_replay_completed_steps"] == [0]
    assert state["preprocessing_replay_step"] == 2
    assert state[demo_notification_control._SESSION_KEY] == "run-1"
    assert store.get_demo_progress("run-1") == "数据预处理 · 「拆分」（1/5）"
    assert not any(name == "rerun" for name, _ in fake.events)


def test_active_connection_shows_completion_time_and_duration(tmp_path) -> None:
    store = demo_notification_control.NotificationStore(
        tmp_path / 'notifications.sqlite3'
    )
    targets = {
        key: WritebackTarget(
            table_key=key,
            app_token='app_demo',
            table_id=f'tbl_{key.lower()}',
            field_names=('run_id',),
            table_url=f'https://demo.feishu.cn/base/app_demo?table=tbl_{key.lower()}',
        )
        for key in ('RUN_LOG', 'RESULTS')
    }
    session = store.create_session(
        'official-release',
        created_by='streamlit-local',
        session_id='run-1',
        created_at='2026-09-09T08:30:00+00:00',
        connected_at='2026-09-09T08:30:03+00:00',
        writeback_targets=targets,
    )
    state = {demo_notification_control._SESSION_KEY: session.session_id}
    fake = ConfirmConnectionStreamlit()

    demo_notification_control._render_active_connection(fake, store, state=state)

    captions = [message for name, message in fake.events if name == 'caption']
    assert len(captions) == 1
    assert '飞书启动时间：2026-09-09 16:30:00' in captions[0]
    assert '连接完成时间：2026-09-09 16:30:03' in captions[0]
    assert '连接耗时：3.0 秒' in captions[0]
