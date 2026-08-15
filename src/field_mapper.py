from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


RECORD_ALIASES: dict[str, tuple[str, ...]] = {
    "raw_sample_type": ("原始样本类型",),
    "context_content": ("上下文内容",),
    "source_note": ("来源说明",),
    "platform_url_available": ("平台链接可用",),
    "raw_id": ("原始编号",),
    "raw_content": ("原始内容", "用户真实评论", "用户评论", "评论内容"),
    "comment_id": ("编号", "评论编号", "ID"),
    "sample_type": ("样本类型", "语料类型", "反馈类型"),
    "source_platform": ("来源平台", "平台", "来源"),
    "original_url": ("原始链接",),
    "collected_at": ("采集时间",),
    "screening_status": ("筛选状态",),
    "screening_reason": ("筛选原因",),
    "duplicate_group": ("重复组",),
    "source_id": ("来源编号",),
    "source_type": ("来源类型",),
    "source_ref": ("来源引用",),
    "dataset_version": ("数据集版本",),
    "routing_prompt_version": ("路由提示词版本",),
    "validation_round": ("验证轮次",),
    "actual_use": ("实际使用",),
    "drinking_scene": ("用户场景", "饮酒场景", "场景"),
    "emotion_keywords": ("情绪关键词", "情绪标签"),
}

ANNOTATION_ALIASES: dict[str, tuple[str, ...]] = {
    "surface_need": ("表层需求",),
    "deep_emotion": ("深层情绪",),
    "identity_need": ("身份需求",),
    "drinking_attitude": ("饮酒态度",),
    "brand_feedback": ("品牌反馈",),
    "narrative_opportunity": ("叙事机会", "可转化品牌洞察"),
}


@dataclass(frozen=True, slots=True)
class FieldMapping:
    record_fields: dict[str, str]
    annotation_fields: dict[str, str]


def _resolve(columns: dict[str, str], aliases: dict[str, tuple[str, ...]]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for canonical_name, accepted_names in aliases.items():
        matches = [columns[name] for name in accepted_names if name in columns]
        if len(matches) > 1:
            raise ValueError(f"字段 {canonical_name} 匹配到多个列: {matches}")
        if matches:
            resolved[canonical_name] = matches[0]
    return resolved


def map_fields(columns: Iterable[object]) -> FieldMapping:
    normalized: dict[str, str] = {}
    for column in columns:
        original = str(column)
        clean = original.strip()
        if clean in normalized:
            raise ValueError(f"存在重复表头: {clean}")
        normalized[clean] = original
    record_fields = _resolve(normalized, RECORD_ALIASES)
    if "raw_content" not in record_fields:
        raise ValueError("缺少必需字段 raw_content（原始内容/用户真实评论/用户评论/评论内容）")
    return FieldMapping(
        record_fields=record_fields,
        annotation_fields=_resolve(normalized, ANNOTATION_ALIASES),
    )
