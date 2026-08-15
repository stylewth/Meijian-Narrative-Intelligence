from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable

from src.schemas import CommentRecord


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def _normalize_list(values: list[str]) -> list[str]:
    return sorted({_normalize_text(value) for value in values if _normalize_text(value)})  # type: ignore[type-var]


def normalized_dataset_payload(records: Iterable[CommentRecord]) -> list[dict[str, object]]:
    payload = [
        {
            "comment_id": _normalize_text(record.comment_id),
            "sample_type": record.sample_type.value,
            "raw_content": _normalize_text(record.raw_content),
            "source_platform": _normalize_text(record.source_platform),
            "drinking_scene": _normalize_list(record.drinking_scene),
            "emotion_keywords": _normalize_list(record.emotion_keywords),
        }
        for record in records
    ]
    if not payload:
        raise ValueError("无法为无记录的数据集计算指纹")
    comment_ids = [item["comment_id"] for item in payload]
    if len(set(comment_ids)) != len(comment_ids):
        raise ValueError("规范化后的 comment_id 必须唯一")
    return sorted(payload, key=lambda item: str(item["comment_id"]))


def normalized_runtime_dataset_payload(records: Iterable[CommentRecord]) -> list[dict[str, object]]:
    """Fingerprint all routing-governance inputs for a mutable online run."""

    record_list = list(records)
    payload = normalized_dataset_payload(record_list)
    records_by_id = {
        _normalize_text(record.comment_id): record
        for record in record_list
    }
    for item in payload:
        record = records_by_id[item["comment_id"]]
        item.update(
            {
                "screening_status": record.screening_status.value if record.screening_status else None,
                "source": (
                    {
                        "source_id": _normalize_text(record.source.source_id),
                        "source_type": record.source.source_type.value,
                        "source_ref": _normalize_text(record.source.source_ref),
                    }
                    if record.source is not None else None
                ),
                "duplicate_group": _normalize_text(record.duplicate_group),
                "actual_use": record.actual_use,
                "versions": (
                    {
                        "dataset_version": _normalize_text(record.versions.dataset_version),
                        "routing_prompt_version": _normalize_text(record.versions.routing_prompt_version),
                        "validation_round": _normalize_text(record.versions.validation_round),
                    }
                    if record.versions is not None else None
                ),
            }
        )
    return payload


def calculate_dataset_fingerprint(records: Iterable[CommentRecord]) -> str:
    canonical_json = json.dumps(
        normalized_dataset_payload(records),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def calculate_runtime_dataset_fingerprint(records: Iterable[CommentRecord]) -> str:
    canonical_json = json.dumps(
        normalized_runtime_dataset_payload(records),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


dataset_fingerprint = calculate_dataset_fingerprint
