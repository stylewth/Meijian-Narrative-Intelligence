"""Read-only inventory of Feishu Bitable write-back targets.

按 2026-08-15 教训：写入前必须只读盘点目标表结构；字段缺失即拒绝，
绝不按旧映射猜测或静默降级。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True, slots=True)
class WritebackTarget:
    table_key: str
    app_token: str
    table_id: str
    field_names: tuple[str, ...]


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
    "EXPECTED_RESULTS_FIELDS",
    "EXPECTED_RUN_LOG_FIELDS",
    "WritebackTarget",
    "inventory_writeback_target",
    "inventory_writeback_targets",
]
