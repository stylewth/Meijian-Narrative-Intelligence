"""Read-only inventory of Feishu Bitable write-back targets.

按 2026-08-15 教训：写入前必须只读盘点目标表结构；字段缺失即拒绝，
绝不按旧映射猜测或静默降级。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .client import FeishuBitableClient
from .url_parser import parse_bitable_url


EXPECTED_RUN_LOG_FIELDS = (
    "run_id",
    "action",
    "actor_open_id",
    "summary",
    "created_at",
)
EXPECTED_RESULTS_FIELDS = (
    "run_id",
    "record_kind",
    "record_key",
    "content",
    "created_at",
)
DYNAMIC_RUN_LOG_FIELDS = (
    "run_id",
    "demo_run_id",
    "action",
    "actor_open_id",
    "summary",
    "demo_started_at",
    "created_at",
)
DYNAMIC_RESULTS_FIELDS = (
    "run_id",
    "demo_run_id",
    "record_kind",
    "record_key",
    "content",
    "demo_started_at",
    "created_at",
)


@dataclass(frozen=True, slots=True)
class WritebackTarget:
    table_key: str
    app_token: str
    table_id: str
    field_names: tuple[str, ...]
    table_url: str | None = None


class WritebackProvisioningError(RuntimeError):
    """创建两张本轮表时发生失败，且可能已有一张表被创建。"""

    def __init__(self, message: str, *, partial_table_url: str | None = None) -> None:
        super().__init__(message)
        self.partial_table_url = partial_table_url


def inventory_writeback_target(
    client: FeishuBitableClient,
    *,
    table_key: str,
    url: str,
) -> WritebackTarget:
    """解析链接并只读校验目标表字段；缺字段直接抛错并列出缺失清单。"""

    expected = (
        EXPECTED_RUN_LOG_FIELDS if table_key == "RUN_LOG" else EXPECTED_RESULTS_FIELDS
    )
    location = parse_bitable_url(url)
    if location.app_token is not None:
        app_token = location.app_token
    else:
        app_token = client.resolve_wiki_app_token(location.wiki_token or "")

    fields = client.get_fields(app_token, location.table_id)
    actual = {field.get("field_name") for field in fields if isinstance(field, dict)}
    missing = [name for name in expected if name not in actual]
    if missing:
        raise ValueError(
            f"写回目标表 {table_key} 缺少字段：{', '.join(missing)}；"
            "请先按设计文档补齐表结构后再启用写回"
        )
    return WritebackTarget(
        table_key=table_key,
        app_token=app_token,
        table_id=location.table_id,
        field_names=tuple(actual),
        table_url=url,
    )


def provision_demo_writeback_targets(
    client: FeishuBitableClient,
    *,
    run_log_url: str | None,
    results_url: str | None,
    demo_run_id: str,
    started_at: str,
) -> dict[str, WritebackTarget]:
    """为一次连接创建两张空白表，并返回它们的具体写回目标。"""

    if not isinstance(run_log_url, str) or not run_log_url.strip():
        raise ValueError("FEISHU_RUNLOG_URL 必须配置")
    if not isinstance(results_url, str) or not results_url.strip():
        raise ValueError("FEISHU_RESULTS_URL 必须配置")
    if not isinstance(demo_run_id, str) or not demo_run_id.strip():
        raise ValueError("demo_run_id must be non-empty")
    started = _parse_started_at(started_at)

    run_log_location = parse_bitable_url(run_log_url)
    results_location = parse_bitable_url(results_url)
    if (
        run_log_location.wiki_token is not None
        and run_log_location.wiki_token == results_location.wiki_token
    ):
        shared_app = client.resolve_wiki_app_token(run_log_location.wiki_token)
        run_log_app = shared_app
        results_app = shared_app
    else:
        run_log_app = _resolve_app_token(client, run_log_location)
        results_app = _resolve_app_token(client, results_location)
    if run_log_app != results_app:
        raise ValueError("日志表和结果表必须属于同一个飞书 Base")

    short_id = "".join(ch for ch in demo_run_id if ch.isalnum())[:8].upper()
    if not short_id:
        raise ValueError("demo_run_id 无法生成表名短标识")
    timestamp = started.astimezone(_BEIJING).strftime("%Y%m%d-%H%M")
    name_prefix = f"梅见-连接-{timestamp}-{short_id}"

    ensure_token = getattr(client, "ensure_token", None)
    if callable(ensure_token):
        ensure_token()
    table_specs = {
        "RUN_LOG": (
            f"{name_prefix}-日志",
            _text_fields(DYNAMIC_RUN_LOG_FIELDS),
        ),
        "RESULTS": (
            f"{name_prefix}-结果",
            _text_fields(DYNAMIC_RESULTS_FIELDS),
        ),
    }
    created_ids: dict[str, str] = {}
    failures: dict[str, Exception] = {}
    with ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="feishu-table-provisioner",
    ) as executor:
        futures = {
            executor.submit(
                client.create_table,
                run_log_app,
                table_name,
                fields,
            ): table_key
            for table_key, (table_name, fields) in table_specs.items()
        }
        for future in as_completed(futures):
            table_key = futures[future]
            try:
                created_ids[table_key] = future.result()
            except Exception as exc:
                failures[table_key] = exc

    if failures:
        partial_urls = [
            _table_url(
                run_log_url if table_key == "RUN_LOG" else results_url,
                created_ids[table_key],
            )
            for table_key in ("RUN_LOG", "RESULTS")
            if table_key in created_ids
        ]
        failed_labels = "、".join(
            "日志表" if table_key == "RUN_LOG" else "结果表"
            for table_key in ("RUN_LOG", "RESULTS")
            if table_key in failures
        )
        message = f"{failed_labels}创建失败"
        if partial_urls:
            message += f"；已创建但未激活：{'、'.join(partial_urls)}"
        first_failure = next(iter(failures.values()))
        raise WritebackProvisioningError(
            message,
            partial_table_url=partial_urls[0] if partial_urls else None,
        ) from first_failure

    run_log_id = created_ids["RUN_LOG"]
    results_id = created_ids["RESULTS"]
    return {
        "RUN_LOG": WritebackTarget(
            table_key="RUN_LOG",
            app_token=run_log_app,
            table_id=run_log_id,
            field_names=DYNAMIC_RUN_LOG_FIELDS,
            table_url=_table_url(run_log_url, run_log_id),
        ),
        "RESULTS": WritebackTarget(
            table_key="RESULTS",
            app_token=run_log_app,
            table_id=results_id,
            field_names=DYNAMIC_RESULTS_FIELDS,
            table_url=_table_url(results_url, results_id),
        ),
    }


_BEIJING = timezone(timedelta(hours=8))


def _parse_started_at(value: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("started_at must be a non-empty ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("started_at must be a valid ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("started_at must include a timezone")
    return parsed


def _resolve_app_token(
    client: FeishuBitableClient, location: Any
) -> str:
    if location.app_token is not None:
        return location.app_token
    return client.resolve_wiki_app_token(location.wiki_token or "")


def _text_fields(field_names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"field_name": name, "type": 1} for name in field_names]


def _table_url(template_url: str, table_id: str) -> str:
    parsed = urlsplit(template_url.strip())
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"table", "view"}
    ]
    query.append(("table", table_id))
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query), "")
    )


def inventory_writeback_targets(
    client: Any,
    *,
    run_log_url: str | None,
    results_url: str | None,
) -> dict[str, WritebackTarget]:
    """盘点全部已配置的写回目标；未配置的目标不返回也不报错。"""

    targets: dict[str, WritebackTarget] = {}
    if run_log_url:
        targets["RUN_LOG"] = inventory_writeback_target(
            client, table_key="RUN_LOG", url=run_log_url
        )
    if results_url:
        targets["RESULTS"] = inventory_writeback_target(
            client, table_key="RESULTS", url=results_url
        )
    return targets


__all__ = [
    "DYNAMIC_RESULTS_FIELDS",
    "DYNAMIC_RUN_LOG_FIELDS",
    "EXPECTED_RESULTS_FIELDS",
    "EXPECTED_RUN_LOG_FIELDS",
    "WritebackTarget",
    "WritebackProvisioningError",
    "inventory_writeback_target",
    "inventory_writeback_targets",
    "provision_demo_writeback_targets",
]
