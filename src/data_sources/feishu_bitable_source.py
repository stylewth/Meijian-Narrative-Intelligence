"""将飞书多维表格记录转换到既有统一样本校验流程。"""

from typing import Any

import pandas as pd

from src.data_validator import ValidatedDataset, validate_dataframe
from src.field_mapper import map_fields
from src.integrations.feishu.client import FeishuAPIError, FeishuBitableClient
from src.integrations.feishu.url_parser import parse_bitable_url
from src.schemas import SampleType


def load_feishu_bitable(
    url: str,
    *,
    app_id: str | None = None,
    app_secret: str | None = None,
    client: FeishuBitableClient | None = None,
    default_sample_type: SampleType | str | None = None,
    max_records: int = 500,
) -> ValidatedDataset:
    """只读加载一张飞书多维表格，并返回统一校验结果。"""

    location = parse_bitable_url(url)
    api = client or FeishuBitableClient(app_id or "", app_secret or "")
    app_token = location.app_token
    if app_token is None:
        if location.wiki_token is None:
            raise FeishuAPIError("飞书链接缺少 app_token 或 wiki_token")
        app_token = api.resolve_wiki_app_token(location.wiki_token)

    fields = api.get_fields(app_token, location.table_id)
    field_names = _field_names(fields)
    mapping = map_fields(field_names)

    records = api.get_records(app_token, location.table_id)
    rows = [_record_fields(record) for record in records]
    frame = pd.DataFrame(rows, columns=field_names)
    return validate_dataframe(
        frame,
        mapping=mapping,
        default_sample_type=default_sample_type,
        max_records=max_records,
    )


def _field_names(fields: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for position, field in enumerate(fields, start=1):
        name = field.get("field_name")
        if not isinstance(name, str) or not name.strip():
            raise FeishuAPIError(f"飞书第 {position} 个字段缺少 field_name")
        names.append(name)
    return names


def _record_fields(record: dict[str, Any]) -> dict[str, Any]:
    fields = record.get("fields")
    if not isinstance(fields, dict):
        record_id = record.get("record_id", "未知记录")
        raise FeishuAPIError(f"飞书记录 {record_id} 缺少 fields")
    return fields
