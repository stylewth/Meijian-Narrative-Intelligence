from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field

from src.schemas import AnnotationRunManifest, StrictBaseModel
from src.prompt_loader import PromptMetadata


MODEL_ID = "gpt-5.6-sol"
REASONING_EFFORT = "high"
APPROVED_MODEL_EFFORT_PAIRS = {
    ("gpt-5.6-sol", "high"),
    ("gpt-5.6-luna", "max"),
}
MIN_BATCH_SIZE = 10
MAX_BATCH_SIZE = 20

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BATCH_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_IMMUTABLE_FIELDS = (
    "raw_id",
    "raw_content",
    "source_platform",
    "source_id",
    "source_type",
    "source_ref",
)
_SOURCE_TYPES = Literal[
    "USER_COMMENT",
    "OFFICIAL_CONTENT",
    "MEDIA_CONTENT",
    "BRAND_DOC",
    "PRODUCT_FACT",
    "COMPETITOR_DOC",
    "VALIDATION",
    "SIMULATED",
]


class OfflineTaskItem(StrictBaseModel):
    raw_id: str = Field(min_length=1)
    raw_content: str = Field(min_length=1)
    source_platform: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_type: _SOURCE_TYPES
    source_ref: str = Field(min_length=1)


class OfflineAnnotation(OfflineTaskItem):
    route: Literal["BRAND", "PRODUCT", "SERVICE", "SCENE", "COMPETITOR", "OTHER"]
    experience_scope: Literal[
        "ACTUAL_USE", "PURCHASE_ONLY", "NON_USE", "UNKNOWN"
    ]
    evidence_grade: Literal["A", "B", "C"]
    ai_confidence: Literal[0.5, 0.7, 0.9]
    explanation: str = Field(
        pattern=r"^核心对象：[^；。\r\n]+；体验：[^；。\r\n]+；证据强度：[^；。\r\n]+。$"
    )


class OfflineBatchResult(StrictBaseModel):
    batch_id: str = Field(min_length=1)
    input: list[OfflineTaskItem] = Field(min_length=1)
    annotations: list[OfflineAnnotation] = Field(min_length=1)


@dataclass(frozen=True)
class OfflineAnnotationImportResult:
    annotations: list[OfflineAnnotation]
    input_ids: list[str]
    batch_ids: list[str]
    input_count: int
    routed_count: int
    result_sha256: str


def _validate_sha256(value: str, field_name: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase 64-character SHA-256")


def canonical_input_sha256(items: Iterable[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        list(items), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_prompt_identity(
    *, prompt_version: str | None, prompt_sha256: str | None
) -> None:
    if (prompt_version is None) != (prompt_sha256 is None):
        raise ValueError("prompt_version and prompt_sha256 must be provided together")
    if prompt_sha256 is not None:
        _validate_sha256(prompt_sha256, "prompt_sha256")


def _validate_explicit_prompt_identity(
    *,
    prompt_metadata: PromptMetadata,
    prompt_version: str | None,
    prompt_sha256: str | None,
) -> None:
    if (prompt_version is None) != (prompt_sha256 is None):
        raise ValueError("explicit prompt_version and prompt_sha256 must be provided together")
    if prompt_version is not None and prompt_version != prompt_metadata.version:
        raise ValueError("explicit prompt_version does not match prompt metadata")
    if prompt_sha256 is not None and prompt_sha256 != prompt_metadata.sha256:
        raise ValueError("explicit prompt SHA-256 does not match prompt metadata")


def validate_annotation_run_manifest(
    manifest: AnnotationRunManifest | Mapping[str, Any],
    *,
    dataset_sha256: str | None = None,
    prompt_sha256: str | None = None,
    batch_ids: Iterable[str] | None = None,
    result_sha256: str | None = None,
    dataset_version: str | None = None,
    prompt_version: str | None = None,
) -> AnnotationRunManifest:
    """Validate the immutable contract for one offline annotation run."""

    _validate_prompt_identity(
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
    )
    if isinstance(manifest, AnnotationRunManifest):
        validated = manifest
    else:
        try:
            validated = AnnotationRunManifest.model_validate_json(
                json.dumps(dict(manifest), ensure_ascii=False),
                strict=True,
            )
        except TypeError as exc:
            raise ValueError("annotation manifest must be a JSON object") from exc

    if (validated.model_id, validated.reasoning_effort) not in APPROVED_MODEL_EFFORT_PAIRS:
        approved = ", ".join(
            f"{model_id}/{effort}"
            for model_id, effort in sorted(APPROVED_MODEL_EFFORT_PAIRS)
        )
        raise ValueError(
            "model_id/reasoning_effort must be an approved annotation pair: "
            f"{approved}"
        )
    if len(validated.batch_ids) != len(set(validated.batch_ids)):
        raise ValueError("manifest batch_ids must be unique")
    if dataset_sha256 is not None and validated.dataset_sha256 != dataset_sha256:
        raise ValueError("dataset SHA-256 does not match")
    if prompt_sha256 is not None and validated.prompt_sha256 != prompt_sha256:
        raise ValueError("prompt SHA-256 does not match")
    if result_sha256 is not None and validated.result_sha256 != result_sha256:
        raise ValueError("result SHA-256 does not match")
    if dataset_version is not None and validated.dataset_version != dataset_version:
        raise ValueError("dataset_version does not match")
    if prompt_version is not None and validated.prompt_version != prompt_version:
        raise ValueError("prompt_version does not match")
    if batch_ids is not None:
        expected = list(batch_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("expected batch_ids must be unique")
        if validated.batch_ids != expected:
            raise ValueError("manifest batch_ids do not match expected batch_ids")
    return validated


def _record_value(record: object, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _as_task_item(record: object) -> dict[str, str]:
    if hasattr(record, "model_dump"):
        values = record.model_dump(mode="json")
    elif isinstance(record, Mapping):
        values = dict(record)
    else:
        values = {
            key: getattr(record, key, None)
            for key in (
                "raw_id",
                "comment_id",
                "raw_content",
                "source_platform",
                "source",
                "source_id",
                "source_type",
                "source_ref",
            )
        }

    forbidden = {
        "ai_route",
        "ai_evidence_grade",
        "ai_experience_scope",
        "ai_confidence",
    }
    if forbidden.intersection(values):
        raise ValueError("offline task input must not contain AI labels")

    source = values.get("source")
    if hasattr(source, "model_dump"):
        source = source.model_dump(mode="json")
    if source is None:
        source = {}

    raw_id = values.get("raw_id") or values.get("comment_id")
    item = {
        "raw_id": raw_id,
        "raw_content": values.get("raw_content"),
        "source_platform": values.get("source_platform"),
        "source_id": values.get("source_id") or source.get("source_id"),
        "source_type": values.get("source_type") or source.get("source_type"),
        "source_ref": values.get("source_ref") or source.get("source_ref"),
    }
    if hasattr(item["source_type"], "value"):
        item["source_type"] = item["source_type"].value
    missing = [key for key, value in item.items() if not isinstance(value, str) or not value]
    if missing:
        raise ValueError(f"offline task input missing required fields: {', '.join(missing)}")
    return OfflineTaskItem.model_validate(item, strict=True).model_dump(mode="json")


def build_annotation_batches(
    records: Iterable[object],
    *,
    dataset_sha256: str,
    batch_prefix: str,
    batch_size: int = MAX_BATCH_SIZE,
) -> list[dict[str, object]]:
    """Build deterministic, source-preserving offline task batches."""

    _validate_sha256(dataset_sha256, "dataset_sha256")
    if not _BATCH_PREFIX.fullmatch(batch_prefix):
        raise ValueError("batch_prefix must contain only letters, digits, '_' or '-'")
    if not MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError("batch_size must be between 10 and 20")

    items = [_as_task_item(record) for record in records]
    ids = [item["raw_id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("input raw_id values must be unique")
    if len(items) < MIN_BATCH_SIZE:
        raise ValueError("offline annotation requires at least 10 records")

    batch_count = (len(items) + batch_size - 1) // batch_size
    sizes = [batch_size] * (batch_count - 1)
    sizes.append(len(items) - sum(sizes))
    if sizes[-1] < MIN_BATCH_SIZE and len(sizes) > 1:
        deficit = MIN_BATCH_SIZE - sizes[-1]
        sizes[-2] -= deficit
        sizes[-1] += deficit
    if min(sizes) < MIN_BATCH_SIZE:
        raise ValueError("cannot form batches with at least 10 records each")

    batches: list[dict[str, object]] = []
    offset = 0
    for index, size in enumerate(sizes):
        batches.append(
            {
                "batch_id": f"{batch_prefix}-{index + 1:04d}",
                "dataset_sha256": dataset_sha256,
                "items": items[offset : offset + size],
                "input_sha256": canonical_input_sha256(items[offset : offset + size]),
            }
        )
        offset += size
    return batches


def _load_payload(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "read"):
        loaded = json.load(value)
    else:
        loaded = json.loads(Path(value).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("offline annotation result must be a JSON object")
    return loaded


def _canonical_result_hash(payloads: Iterable[dict[str, Any]]) -> str:
    canonical = json.dumps(
        sorted(payloads, key=lambda payload: str(payload["batch_id"])),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_batch(
    batch: OfflineBatchResult,
    *,
    expected_batch_id: str | None,
    seen_ids: set[str],
) -> tuple[list[str], list[OfflineAnnotation]]:
    if expected_batch_id is not None and batch.batch_id != expected_batch_id:
        raise ValueError(
            f"batch_id mismatch: expected {expected_batch_id}, got {batch.batch_id}"
        )
    if not MIN_BATCH_SIZE <= len(batch.input) <= MAX_BATCH_SIZE:
        raise ValueError(f"batch {batch.batch_id} must contain 10-20 input records")
    if len(batch.annotations) != len(batch.input):
        raise ValueError(f"batch {batch.batch_id} input/output count mismatch")

    input_ids = [item.raw_id for item in batch.input]
    output_ids = [item.raw_id for item in batch.annotations]
    if len(input_ids) != len(set(input_ids)):
        raise ValueError(f"batch {batch.batch_id} input ID values must be unique")
    if len(output_ids) != len(set(output_ids)):
        raise ValueError(f"batch {batch.batch_id} output ID values must be unique")
    if input_ids != output_ids and set(input_ids) != set(output_ids):
        raise ValueError(f"batch {batch.batch_id} input/output ID sets do not match")
    if set(input_ids) != set(output_ids):
        raise ValueError(f"batch {batch.batch_id} input/output ID sets do not match")
    if seen_ids.intersection(input_ids):
        raise ValueError(f"duplicate raw_id across batches: {sorted(seen_ids.intersection(input_ids))[0]}")

    input_by_id = {item.raw_id: item for item in batch.input}
    for annotation in batch.annotations:
        source = input_by_id[annotation.raw_id]
        for field in _IMMUTABLE_FIELDS:
            if getattr(annotation, field) != getattr(source, field):
                raise ValueError(
                    f"batch {batch.batch_id} source mismatch for {annotation.raw_id}: {field}"
                )
    return input_ids, batch.annotations


def import_offline_annotations(
    manifest: AnnotationRunManifest | Mapping[str, Any],
    result_files: Iterable[object] | object,
    *,
    dataset_sha256: str | None = None,
    prompt_sha256: str | None = None,
    dataset_version: str | None = None,
    prompt_version: str | None = None,
    prompt_metadata: PromptMetadata | None = None,
    trusted_batch_inputs: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    trusted_batch_input_hashes: Mapping[str, str] | None = None,
) -> OfflineAnnotationImportResult:
    """Strictly import validated offline JSON batches without calling an LLM."""

    if prompt_metadata is None:
        raise ValueError("trusted Prompt metadata is required")
    if trusted_batch_inputs is None and trusted_batch_input_hashes is None:
        raise ValueError("trusted batch input binding is required")
    if trusted_batch_inputs is not None and trusted_batch_input_hashes is not None:
        raise ValueError("provide trusted batch inputs or trusted input hashes, not both")
    _validate_explicit_prompt_identity(
        prompt_metadata=prompt_metadata,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
    )
    trusted_hashes = dict(trusted_batch_input_hashes or {})
    if trusted_batch_inputs is not None:
        trusted_hashes = {
            batch_id: canonical_input_sha256(items)
            for batch_id, items in trusted_batch_inputs.items()
        }
    if set(trusted_hashes) != set(
        manifest.batch_ids if isinstance(manifest, AnnotationRunManifest) else manifest.get("batch_ids", [])
    ):
        raise ValueError("trusted batch input binding must cover every batch")

    validated_manifest = validate_annotation_run_manifest(
        manifest,
        dataset_sha256=dataset_sha256,
        prompt_sha256=prompt_metadata.sha256,
        dataset_version=dataset_version,
        prompt_version=prompt_metadata.version,
    )
    if isinstance(result_files, (str, Path, Mapping)) or hasattr(result_files, "read"):
        values = [result_files]
    else:
        values = list(result_files)
    if len(values) != len(validated_manifest.batch_ids):
        raise ValueError("result file count does not match manifest batch_ids")

    raw_payloads = [_load_payload(value) for value in values]
    parsed_batches: list[OfflineBatchResult] = []
    batch_ids: list[str] = []
    for payload in raw_payloads:
        batch = OfflineBatchResult.model_validate(payload, strict=True)
        if batch.batch_id in batch_ids:
            raise ValueError(f"duplicate batch_id: {batch.batch_id}")
        if batch.batch_id not in validated_manifest.batch_ids:
            raise ValueError(f"unexpected batch_id: {batch.batch_id}")
        parsed_batches.append(batch)
        batch_ids.append(batch.batch_id)

    if set(batch_ids) != set(validated_manifest.batch_ids):
        raise ValueError("result batch_ids do not exactly match manifest batch_ids")
    ordered = [
        parsed_batches[batch_ids.index(batch_id)]
        for batch_id in validated_manifest.batch_ids
    ]
    canonical_payloads = [
        batch.model_dump(mode="json")
        for batch in ordered
    ]
    actual_result_sha256 = _canonical_result_hash(canonical_payloads)
    if actual_result_sha256 != validated_manifest.result_sha256:
        raise ValueError("result SHA-256 does not match manifest")

    seen_ids: set[str] = set()
    all_input_ids: list[str] = []
    all_annotations: list[OfflineAnnotation] = []
    for batch in ordered:
        expected_input_sha256 = trusted_hashes.get(batch.batch_id)
        if expected_input_sha256 is None:
            raise ValueError(f"missing trusted batch input for {batch.batch_id}")
        actual_input_sha256 = canonical_input_sha256(
            [item.model_dump(mode="json") for item in batch.input]
        )
        if actual_input_sha256 != expected_input_sha256:
            raise ValueError(f"batch {batch.batch_id} input SHA-256 does not match trusted input")
        input_ids, annotations = _validate_batch(
            batch,
            expected_batch_id=batch.batch_id,
            seen_ids=seen_ids,
        )
        seen_ids.update(input_ids)
        all_input_ids.extend(input_ids)
        all_annotations.extend(annotations)

    return OfflineAnnotationImportResult(
        annotations=all_annotations,
        input_ids=all_input_ids,
        batch_ids=[batch.batch_id for batch in ordered],
        input_count=len(all_input_ids),
        routed_count=len(all_annotations),
        result_sha256=actual_result_sha256,
    )
