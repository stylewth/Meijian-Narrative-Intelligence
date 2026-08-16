from __future__ import annotations

import datetime as _datetime
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from src.schemas import (
    BlindEvidenceItem,
    BlindEvidencePackage,
    BlindSourceManifest,
    DecisionOriginalSnapshot,
)
from src.services.decision_run_store import DecisionRunStore
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True, slots=True)
class BlindSource:
    path: Path
    relative_name: str


_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _safe_filename(name: str) -> None:
    if not isinstance(name, str) or not name or name in {".", ".."}:
        raise ValueError("relative_name must be a safe filename")
    if name != name.strip() or _UNSAFE_FILENAME_CHARS.search(name):
        raise ValueError("relative_name must be a safe filename")
    if Path(name).is_absolute() or Path(name).name != name:
        raise ValueError("relative_name must be a safe filename")
    if name.endswith((".", " ")):
        raise ValueError("relative_name must be a safe filename")
    if name.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES:
        raise ValueError("relative_name must be a safe filename")


def _regular_nonempty_source(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source must be a regular file: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"source must be non-empty: {path}")


def _write_new_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _validated_items(items: Iterable[BlindEvidenceItem]) -> list[BlindEvidenceItem]:
    result: list[BlindEvidenceItem] = []
    for item in items:
        if isinstance(item, BlindEvidenceItem):
            result.append(item)
        elif isinstance(item, BaseModel):
            raise TypeError("items must contain BlindEvidenceItem values")
        else:
            result.append(BlindEvidenceItem.model_validate(item, strict=True))
    if not result:
        raise ValueError("items must not be empty")
    return result


def freeze_blind_evidence(
    *,
    run_store: DecisionRunStore,
    original_snapshot: DecisionOriginalSnapshot,
    sources: Iterable[BlindSource],
    items: Iterable[BlindEvidenceItem],
    confirmed_by: str,
) -> BlindEvidencePackage:
    if not isinstance(run_store, DecisionRunStore):
        raise TypeError("run_store must be a DecisionRunStore")
    if not isinstance(original_snapshot, DecisionOriginalSnapshot):
        raise TypeError("original_snapshot must be a DecisionOriginalSnapshot")
    if original_snapshot.run_id != run_store.manifest.decision_run_id:
        raise ValueError("original_snapshot run_id does not match run_store")

    stored_snapshot = run_store.load_stage("original_snapshot", DecisionOriginalSnapshot)
    snapshot_bytes = canonical_json_bytes(original_snapshot)
    if stored_snapshot != original_snapshot:
        raise ValueError("original_snapshot does not match frozen original snapshot")
    stored_snapshot_bytes = canonical_json_bytes(stored_snapshot)
    if snapshot_bytes != stored_snapshot_bytes:
        raise ValueError("original_snapshot bytes do not match frozen original snapshot")
    original_snapshot_sha256 = sha256_bytes(stored_snapshot_bytes)

    source_list = list(sources)
    if not source_list:
        raise ValueError("sources must not be empty")
    names: set[str] = set()
    for source in source_list:
        if not isinstance(source, BlindSource):
            raise TypeError("sources must contain BlindSource values")
        _safe_filename(source.relative_name)
        if source.relative_name in names:
            raise ValueError("source relative_name must be unique")
        names.add(source.relative_name)
        _regular_nonempty_source(Path(source.path))

    validated_items = _validated_items(items)
    candidate_ids = {
        candidate.ranked_narrative.candidate.candidate_id
        for candidate in original_snapshot.candidates
    }
    for item in validated_items:
        if item.candidate_id not in candidate_ids:
            raise ValueError("item candidate_id must belong to original snapshot candidates")
        if any(name not in names for name in item.source_artifacts):
            raise ValueError("item source_artifacts must be registered sources")

    stages_dir = run_store.root / "stages"
    if stages_dir.is_symlink() or not stages_dir.is_dir():
        raise ValueError("run stages must be a directory")
    blind_dir = stages_dir / "blind"
    if blind_dir.exists() or blind_dir.is_symlink():
        raise FileExistsError("blind evidence directory already exists")

    temp_blind = Path(tempfile.mkdtemp(prefix=".blind-", dir=stages_dir))
    try:
        temp_sources = temp_blind / "sources"
        temp_sources.mkdir()
        artifact_sha256: dict[str, str] = {}
        for source in source_list:
            destination = temp_sources / source.relative_name
            shutil.copyfile(Path(source.path), destination)
            copied_bytes = destination.read_bytes()
            if not copied_bytes:
                raise ValueError(f"copied source must be non-empty: {source.relative_name}")
            artifact_sha256[source.relative_name] = sha256_bytes(copied_bytes)

        source_manifest = BlindSourceManifest(
            run_id=run_store.manifest.decision_run_id,
            original_snapshot_sha256=original_snapshot_sha256,
            artifact_sha256=artifact_sha256,
            confirmed_by=confirmed_by,
            confirmed_at=_datetime.datetime.now(_datetime.timezone.utc),
        )
        source_manifest_bytes = canonical_json_bytes(source_manifest)
        package = BlindEvidencePackage(
            run_id=run_store.manifest.decision_run_id,
            source_manifest_sha256=sha256_bytes(source_manifest_bytes),
            original_snapshot_sha256=original_snapshot_sha256,
            items=validated_items,
        )
        package_bytes = canonical_json_bytes(package)

        checked_manifest = BlindSourceManifest.model_validate_json(
            source_manifest_bytes, strict=True
        )
        checked_package = BlindEvidencePackage.model_validate_json(
            package_bytes, strict=True
        )
        if canonical_json_bytes(checked_manifest) != source_manifest_bytes:
            raise ValueError("source manifest self-check failed")
        if canonical_json_bytes(checked_package) != package_bytes:
            raise ValueError("blind evidence self-check failed")
        if sha256_bytes(source_manifest_bytes) != checked_package.source_manifest_sha256:
            raise ValueError("blind evidence source manifest binding failed")
        if checked_manifest.original_snapshot_sha256 != original_snapshot_sha256:
            raise ValueError("source manifest original snapshot binding failed")
        for source in source_list:
            copied_path = temp_sources / source.relative_name
            if not copied_path.is_file() or copied_path.is_symlink():
                raise ValueError("copied source self-check failed")
            if sha256_bytes(copied_path.read_bytes()) != artifact_sha256[source.relative_name]:
                raise ValueError("copied source hash self-check failed")

        _write_new_bytes(temp_blind / "source_manifest.json", source_manifest_bytes)
        _write_new_bytes(temp_blind / "blind_evidence.json", package_bytes)
        if blind_dir.exists() or blind_dir.is_symlink():
            raise FileExistsError("blind evidence directory already exists")
        os.rename(temp_blind, blind_dir)
        return checked_package
    finally:
        if temp_blind.exists():
            shutil.rmtree(temp_blind)

