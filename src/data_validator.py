from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from src.field_mapper import FieldMapping, map_fields
from src.schemas import (
    CommentRecord,
    DataHealthResult,
    DataHealthStatus,
    DataVersion,
    HumanAnnotation,
    SampleType,
    ScreeningStatus,
    SourceReference,
    SourceType,
)


@dataclass(frozen=True, slots=True)
class ValidatedDataset:
    comments: list[CommentRecord]
    human_annotations: list[HumanAnnotation]
    field_mapping: FieldMapping
    data_health: DataHealthResult


def _scalar(value: object) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    missing = pd.isna(value)
    if isinstance(missing, bool) and missing:
        return None
    text = str(value).strip()
    return text or None


def _list_value(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        text = _scalar(value)
        if text is None:
            return []
        parts = re.split(r"[,，]", text)
    return [item for part in parts if (item := _scalar(part)) is not None]


def _row_has_content(row: pd.Series) -> bool:
    return any(_scalar(value) is not None for value in row.tolist())


CONTRACT_FIELDS = frozenset(
    {
        "raw_id",
        "collected_at",
        "screening_status",
        "source_id",
        "source_type",
        "source_ref",
        "dataset_version",
        "routing_prompt_version",
        "validation_round",
    }
)


def _required_contract_value(row: pd.Series, mapping: FieldMapping, field_name: str, source_row: int) -> str:
    column = mapping.record_fields.get(field_name)
    value = _scalar(row.get(column)) if column else None
    if value is None:
        raise ValueError(f"第 {source_row} 行缺少必填合同字段 {field_name}")
    return value


def _data_health(comments: list[CommentRecord]) -> DataHealthResult:
    effective_comments = [
        record
        for record in comments
        if record.screening_status is ScreeningStatus.KEEP
        and record.source is not None
        and record.source.source_type is SourceType.USER_COMMENT
    ]
    platform_counts: dict[str, int] = {}
    for record in effective_comments:
        if record.source_platform is not None:
            platform_counts[record.source_platform] = platform_counts.get(record.source_platform, 0) + 1
    largest_platform = max(platform_counts, key=platform_counts.get) if platform_counts else None
    effective_count = len(effective_comments)
    largest_share = (
        platform_counts[largest_platform] / effective_count if largest_platform is not None and effective_count else 0.0
    )
    kept_records = [record for record in comments if record.screening_status is ScreeningStatus.KEEP]
    traceable_rate = (
        sum(record.source is not None for record in kept_records) / len(kept_records) if kept_records else 0.0
    )
    platform_url_coverage = (
        sum(record.platform_url_available is True for record in effective_comments) / effective_count
        if effective_count
        else 0.0
    )
    simulated_record_count = sum(
        record.source is not None and record.source.source_type is SourceType.SIMULATED for record in comments
    )
    actual_use_ratio = (
        sum(record.actual_use is True for record in effective_comments) / effective_count
        if effective_count
        else None
    )
    ready = (
        effective_count > 0
        and len(platform_counts) >= 4
        and largest_share <= 0.5
        and traceable_rate == 1.0
    )
    return DataHealthResult(
        status=DataHealthStatus.READY if ready else DataHealthStatus.NEEDS_REBALANCE,
        effective_user_comment_count=effective_count,
        platform_count=len(platform_counts),
        largest_platform=largest_platform,
        largest_platform_share=largest_share,
        traceable_rate=traceable_rate,
        platform_url_coverage=platform_url_coverage,
        actual_use_ratio=actual_use_ratio,
        simulated_record_count=simulated_record_count,
        can_finalize_snapshot=ready,
    )


def _actual_use(value: object, source_row: int) -> bool | None:
    text = _scalar(value)
    if text is None:
        return None
    if text in {"是", "TRUE", "true", "1"}:
        return True
    if text in {"否", "FALSE", "false", "0"}:
        return False
    raise ValueError(f"第 {source_row} 行实际使用必须为 是/否、TRUE/FALSE 或 1/0")


def _platform_url_available(value: object, source_row: int) -> bool | None:
    text = _scalar(value)
    if text is None:
        return None
    if text in {"是", "TRUE", "true", "1"}:
        return True
    if text in {"否", "FALSE", "false", "0"}:
        return False
    raise ValueError(f"第 {source_row} 行平台链接可用必须为 是/否、TRUE/FALSE 或 1/0")


def validate_dataframe(
    frame: pd.DataFrame,
    *,
    mapping: FieldMapping | None = None,
    default_sample_type: SampleType | str | None = None,
    max_records: int = 500,
) -> ValidatedDataset:
    if frame is None or len(frame.columns) == 0:
        raise ValueError("数据表为空")
    mapping = mapping or map_fields(frame.columns)
    rows = [
        (source_row, row)
        for source_row, (_, row) in enumerate(frame.iterrows(), start=2)
        if _row_has_content(row)
    ]
    if not rows:
        raise ValueError("数据表没有有效记录")
    if len(rows) > max_records:
        raise ValueError(f"有效记录数 {len(rows)} 超过配置上限 {max_records}")

    content_column = mapping.record_fields["raw_content"]
    type_column = mapping.record_fields.get("sample_type")
    id_column = mapping.record_fields.get("comment_id")
    comments: list[CommentRecord] = []
    annotation_values: list[dict[str, str | None]] = []
    contract_mode = bool(CONTRACT_FIELDS & mapping.record_fields.keys())

    for import_order, (source_row, row) in enumerate(rows, start=1):
        raw_content = _scalar(row.get(content_column))
        if raw_content is None:
            raise ValueError(f"第 {source_row} 行 raw_content 为空")
        sample_type_value = _scalar(row.get(type_column)) if type_column else None
        if sample_type_value is None:
            if default_sample_type is None:
                raise ValueError("缺少 sample_type，必须由用户明确指定默认样本类型")
            sample_type_value = str(default_sample_type)
        if contract_mode:
            raw_id = _required_contract_value(row, mapping, "raw_id", source_row)
            original_url = _scalar(row.get(mapping.record_fields.get("original_url")))
            collected_at = _required_contract_value(row, mapping, "collected_at", source_row)
            screening_status = ScreeningStatus(
                _required_contract_value(row, mapping, "screening_status", source_row)
            )
            screening_reason = _scalar(row.get(mapping.record_fields.get("screening_reason")))
            duplicate_group = _scalar(row.get(mapping.record_fields.get("duplicate_group")))
            if screening_status is not ScreeningStatus.KEEP and screening_reason is None:
                raise ValueError(f"第 {source_row} 行非 KEEP 记录必须填写筛选原因")
            if screening_status is ScreeningStatus.DUPLICATE:
                if duplicate_group is None or re.fullmatch(r"DUP-\d+", duplicate_group) is None:
                    raise ValueError(f"第 {source_row} 行重复记录必须填写可读重复组，例如 DUP-03")
            source_type = SourceType(_required_contract_value(row, mapping, "source_type", source_row))
            source = SourceReference(
                source_id=_required_contract_value(row, mapping, "source_id", source_row),
                source_type=source_type,
                source_ref=_required_contract_value(row, mapping, "source_ref", source_row),
            )
            if source_type is SourceType.USER_COMMENT and source.source_id != raw_id:
                raise ValueError(f"第 {source_row} 行 USER_COMMENT 的 source_id 必须等于 raw_id")
            versions = DataVersion(
                dataset_version=_required_contract_value(row, mapping, "dataset_version", source_row),
                routing_prompt_version=_required_contract_value(
                    row, mapping, "routing_prompt_version", source_row
                ),
                validation_round=_required_contract_value(row, mapping, "validation_round", source_row),
            )
            comment_id = raw_id
        else:
            raw_id = None
            original_url = None
            collected_at = None
            screening_status = None
            screening_reason = None
            duplicate_group = None
            source = None
            versions = None
            comment_id = _scalar(row.get(id_column)) if id_column else None
            comment_id = comment_id or f"C{import_order:03d}"
        comments.append(
            CommentRecord(
                comment_id=comment_id,
                sample_type=sample_type_value,
                raw_sample_type=_scalar(row.get(mapping.record_fields.get("raw_sample_type"))),
                raw_content=raw_content,
                context_content=_scalar(row.get(mapping.record_fields.get("context_content"))),
                source_note=_scalar(row.get(mapping.record_fields.get("source_note"))),
                source_platform=_scalar(row.get(mapping.record_fields.get("source_platform"))),
                drinking_scene=_list_value(row.get(mapping.record_fields.get("drinking_scene"))),
                emotion_keywords=_list_value(row.get(mapping.record_fields.get("emotion_keywords"))),
                source_row=source_row,
                raw_id=raw_id,
                original_url=original_url,
                platform_url_available=_platform_url_available(
                    row.get(mapping.record_fields.get("platform_url_available")), source_row
                ),
                collected_at=collected_at,
                screening_status=screening_status,
                screening_reason=screening_reason,
                duplicate_group=duplicate_group,
                source=source,
                actual_use=_actual_use(row.get(mapping.record_fields.get("actual_use")), source_row),
                versions=versions,
            )
        )
        annotation_values.append(
            {
                canonical: _scalar(row.get(source_column))
                for canonical, source_column in mapping.annotation_fields.items()
            }
        )

    ids = [record.comment_id for record in comments]
    if len(set(ids)) != len(ids):
        if contract_mode:
            raise ValueError("raw_id 必须唯一")
        raise ValueError("comment_id 必须唯一")
    content_counts: dict[str, int] = {}
    for record in comments:
        key = record.raw_content.strip()
        content_counts[key] = content_counts.get(key, 0) + 1
    for record in comments:
        record.possible_duplicate = content_counts[record.raw_content.strip()] > 1

    annotations = [
        HumanAnnotation(comment_id=record.comment_id, **values)
        for record, values in zip(comments, annotation_values, strict=True)
        if any(value is not None for value in values.values())
    ]
    return ValidatedDataset(
        comments=comments,
        human_annotations=annotations,
        field_mapping=mapping,
        data_health=_data_health(comments),
    )
