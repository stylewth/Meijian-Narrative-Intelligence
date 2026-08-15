from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from src.schemas import (
    AnnotationRunManifest,
    CommentRecord,
    DatasetSplitManifest,
    EvidenceAtom,
    PreparedCorpusManifest,
    PreparedCorpusPackage,
)


_DATA_ARTIFACT_NAMES = (
    "records.json",
    "evidence_atoms.json",
    "split_manifest.json",
    "annotation_run_manifests.json",
)
_FILE_NAMES = frozenset((*_DATA_ARTIFACT_NAMES, "manifest.json"))


@dataclass(frozen=True)
class PublishedPreparedCorpus:
    manifest: PreparedCorpusManifest
    package: PreparedCorpusPackage
    path: Path


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes | bytearray | memoryview) -> str:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError("sha256_bytes requires bytes-like input")
    return hashlib.sha256(bytes(value)).hexdigest()


def package_digest(artifact_sha256: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json_bytes(artifact_sha256))


def _validate_package_id(package_id: str) -> None:
    if not isinstance(package_id, str) or not package_id.strip():
        raise ValueError("package_id must be a non-empty name")
    if package_id in {".", ".."} or "/" in package_id or "\\" in package_id:
        raise ValueError("package_id must not contain path traversal")
    candidate = Path(package_id)
    if candidate.is_absolute() or candidate.name != package_id or "\x00" in package_id:
        raise ValueError("package_id must be a relative directory name")


def _root_path(root: str | os.PathLike[str]) -> Path:
    root_path = Path(root)
    if root_path.exists() and not root_path.is_dir():
        raise ValueError("prepared corpus root must be a directory")
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path.resolve()


def _package_path(root: Path, package_id: str) -> Path:
    _validate_package_id(package_id)
    destination = (root / package_id).resolve(strict=False)
    if destination.parent != root:
        raise ValueError("package path must remain inside root")
    return destination


def _artifact_bytes(package: PreparedCorpusPackage) -> dict[str, bytes]:
    return {
        "records.json": canonical_json_bytes(package.records),
        "evidence_atoms.json": canonical_json_bytes(package.evidence_atoms),
        "split_manifest.json": canonical_json_bytes(package.split_manifest),
        "annotation_run_manifests.json": canonical_json_bytes(
            package.annotation_run_manifests
        ),
    }


def validate_prepared_corpus_package(package: PreparedCorpusPackage) -> None:
    """Strictly validate a prepared corpus package's semantic contract."""
    if not isinstance(package, PreparedCorpusPackage):
        raise TypeError("package must be a PreparedCorpusPackage")

    manifest = package.manifest
    record_ids = [record.comment_id for record in package.records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("records comment_id must be unique")
    raw_ids = [record.raw_id for record in package.records if record.raw_id is not None]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("records raw_id must be unique")
    stable_record_ids = [record.raw_id or record.comment_id for record in package.records]
    if len(stable_record_ids) != len(set(stable_record_ids)):
        raise ValueError("records stable IDs must be unique")
    if manifest.record_count != len(package.records):
        raise ValueError("manifest record_count does not match records")
    if manifest.evidence_count != len(package.evidence_atoms):
        raise ValueError("manifest evidence_count does not match evidence_atoms")

    if (
        manifest.dataset_version != package.split_manifest.dataset_version
        or manifest.dataset_sha256 != package.split_manifest.dataset_sha256
    ):
        raise ValueError("manifest dataset identity must match split manifest")
    assignment_ids = [assignment.raw_id for assignment in package.split_manifest.assignments]
    if len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("split manifest assignment raw_id must be unique")
    if set(stable_record_ids) != set(assignment_ids):
        raise ValueError("records stable IDs must match split manifest assignments")

    records_by_stable_id = {
        record.raw_id or record.comment_id: record for record in package.records
    }
    for assignment in package.split_manifest.assignments:
        record = records_by_stable_id[assignment.raw_id]
        if (
            assignment.source_platform != record.source_platform
            or assignment.raw_sample_type != record.raw_sample_type
            or assignment.platform_url_available != record.platform_url_available
        ):
            raise ValueError(
                "split assignment fields must match the record for its stable ID"
            )

    batch_ids = [
        batch_id
        for annotation_manifest in package.annotation_run_manifests
        for batch_id in annotation_manifest.batch_ids
    ]
    if len(batch_ids) != len(set(batch_ids)):
        raise ValueError("annotation run batch_id values must be unique")
    for annotation_manifest in package.annotation_run_manifests:
        if (
            annotation_manifest.dataset_version != manifest.dataset_version
            or annotation_manifest.dataset_sha256 != manifest.dataset_sha256
            or annotation_manifest.model_id != manifest.annotation_model_id
            or annotation_manifest.reasoning_effort
            != manifest.annotation_reasoning_effort
            or annotation_manifest.annotation_version != manifest.annotation_version
            or annotation_manifest.prompt_version != manifest.annotation_prompt_version
            or annotation_manifest.prompt_sha256 != manifest.annotation_prompt_sha256
        ):
            raise ValueError("annotation run manifest identity must match package manifest")

    records_by_id = {record.comment_id: record for record in package.records}
    evidence_ids = [atom.evidence_id for atom in package.evidence_atoms]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence_id must be unique")
    for atom in package.evidence_atoms:
        record = records_by_id.get(atom.comment_id)
        if record is None:
            raise ValueError("evidence comment_id must reference a record")
        if atom.comment_id != record.comment_id:
            raise ValueError("evidence comment_id must match the record")
        if record.source is None or atom.source != record.source:
            raise ValueError("evidence source must match the record source")
        if (
            atom.source_platform != record.source_platform
            or atom.duplicate_group != record.duplicate_group
            or atom.actual_use != record.actual_use
        ):
            raise ValueError("evidence source mirror fields must match the record")


def _final_package(package: PreparedCorpusPackage) -> PreparedCorpusPackage:
    artifact_bytes = _artifact_bytes(package)
    artifact_sha256 = {
        name: sha256_bytes(content) for name, content in artifact_bytes.items()
    }
    manifest = package.manifest.model_copy(
        update={
            "record_count": len(package.records),
            "evidence_count": len(package.evidence_atoms),
            "artifact_sha256": artifact_sha256,
            "package_sha256": package_digest(artifact_sha256),
        }
    )
    final_package = package.model_copy(deep=True, update={"manifest": manifest})
    validate_prepared_corpus_package(final_package)
    return final_package


def _read_directory(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("published package path must be a real directory")
    entries = {entry.name: entry for entry in directory.iterdir()}
    missing = _FILE_NAMES - entries.keys()
    extra = entries.keys() - _FILE_NAMES
    if missing:
        raise ValueError(f"missing artifact: {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"unexpected artifact: {sorted(extra)[0]}")
    for name, entry in entries.items():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"artifact must be a regular file: {name}")
        if entry.resolve(strict=True).parent != directory.resolve(strict=True):
            raise ValueError(f"artifact path escapes package: {name}")
    return {name: entries[name].read_bytes() for name in _FILE_NAMES}


def _decode_json(data: bytes, name: str) -> Any:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {name}") from exc
    if canonical_json_bytes(payload) != data:
        raise ValueError(f"artifact is not canonical JSON: {name}")
    return payload


def _parse_package(artifacts: dict[str, bytes], package_id: str, path: Path) -> PublishedPreparedCorpus:
    try:
        manifest_payload = _decode_json(artifacts["manifest.json"], "manifest.json")
        manifest = PreparedCorpusManifest.model_validate_json(
            artifacts["manifest.json"], strict=True
        )
        records_payload = _decode_json(artifacts["records.json"], "records.json")
        evidence_payload = _decode_json(
            artifacts["evidence_atoms.json"], "evidence_atoms.json"
        )
        split_payload = _decode_json(artifacts["split_manifest.json"], "split_manifest.json")
        annotation_payload = _decode_json(
            artifacts["annotation_run_manifests.json"],
            "annotation_run_manifests.json",
        )
        records = [
            CommentRecord.model_validate_json(canonical_json_bytes(item), strict=True)
            for item in records_payload
        ]
        evidence_atoms = [
            EvidenceAtom.model_validate_json(canonical_json_bytes(item), strict=True)
            for item in evidence_payload
        ]
        split_manifest = DatasetSplitManifest.model_validate_json(
            canonical_json_bytes(split_payload), strict=True
        )
        annotation_run_manifests = [
            AnnotationRunManifest.model_validate_json(
                canonical_json_bytes(item), strict=True
            )
            for item in annotation_payload
        ]
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("invalid JSON"):
            raise
        raise ValueError("invalid prepared corpus artifact") from exc

    if manifest.package_id != package_id:
        raise ValueError("manifest package_id does not match requested package_id")
    actual_sha256 = {
        name: sha256_bytes(artifacts[name]) for name in _DATA_ARTIFACT_NAMES
    }
    if manifest.artifact_sha256 != actual_sha256:
        raise ValueError("artifact SHA-256 does not match manifest")
    if manifest.package_sha256 != package_digest(actual_sha256):
        raise ValueError("package SHA-256 does not match manifest")

    package = PreparedCorpusPackage(
        manifest=manifest,
        records=records,
        evidence_atoms=evidence_atoms,
        split_manifest=split_manifest,
        annotation_run_manifests=annotation_run_manifests,
    )
    validate_prepared_corpus_package(package)
    return PublishedPreparedCorpus(manifest=manifest, package=package, path=path)


def publish_prepared_corpus(
    root: str | os.PathLike[str], package: PreparedCorpusPackage
) -> PublishedPreparedCorpus:
    if not isinstance(package, PreparedCorpusPackage):
        raise TypeError("package must be a PreparedCorpusPackage")
    root_path = _root_path(root)
    package_id = package.manifest.package_id
    destination = _package_path(root_path, package_id)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"package_id already exists: {package_id}")

    final_package = _final_package(package)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{package_id}.", dir=str(root_path))
    )
    try:
        if temp_path.resolve(strict=True).parent != root_path:
            raise ValueError("temporary package path must remain inside root")
        artifacts = _artifact_bytes(final_package)
        artifacts["manifest.json"] = canonical_json_bytes(final_package.manifest)
        for name, data in artifacts.items():
            (temp_path / name).write_bytes(data)
        _parse_package(_read_directory(temp_path), package_id, temp_path)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"package_id already exists: {package_id}")
        os.rename(temp_path, destination)
        temp_path = Path()
    except Exception:
        if temp_path != Path() and temp_path.exists():
            shutil.rmtree(temp_path)
        raise
    return load_prepared_corpus(root_path, package_id)


def load_prepared_corpus(
    root: str | os.PathLike[str], package_id: str
) -> PublishedPreparedCorpus:
    root_path = _root_path(root)
    package_path = _package_path(root_path, package_id)
    if not package_path.exists():
        raise ValueError(f"published package does not exist: {package_id}")
    if package_path.parent != root_path:
        raise ValueError("package path must remain inside root")
    return _parse_package(_read_directory(package_path), package_id, package_path)


def list_prepared_corpora(
    root: str | os.PathLike[str],
) -> list[PublishedPreparedCorpus]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    root_path = _root_path(root_path)
    result: list[PublishedPreparedCorpus] = []
    for entry in sorted(root_path.iterdir(), key=lambda item: item.name):
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        result.append(load_prepared_corpus(root_path, entry.name))
    return result
