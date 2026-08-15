from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from typing import Any

from src.prompt_loader import PromptMetadata
from src.schemas import (
    AnnotationRunManifest,
    CommentRecord,
    DatasetSplitAssignment,
    DatasetSplitManifest,
    DatasetSplitRole,
    EvidenceAtom,
    FormalAnnotationIdentity,
    FormalPreparedCorpusGroupManifest,
    PackageValidationStatus,
    PreparedCorpusManifest,
    PreparedCorpusPackage,
    SourceReference,
)
from src.services.offline_annotation_import import (
    OfflineAnnotation,
    import_offline_annotations,
)
from src.services.prepared_corpus import (
    PublishedPreparedCorpus,
    canonical_json_bytes,
    load_prepared_corpus,
    package_digest,
    publish_prepared_corpus,
    sha256_bytes,
    validate_prepared_corpus_package,
)


_GROUPS = (
    ("analysis_baseline", DatasetSplitRole.ANALYSIS, "ANALYSIS"),
    ("analysis_demo_delta", DatasetSplitRole.ANALYSIS, "ANALYSIS"),
    ("challenge_pool", DatasetSplitRole.CHALLENGE_POOL, "CHALLENGE"),
    ("holdout", DatasetSplitRole.HOLDOUT, "HOLDOUT"),
)
_PACKAGE_IDS = {
    DatasetSplitRole.ANALYSIS: "formal-analysis",
    DatasetSplitRole.CHALLENGE_POOL: "formal-challenge",
    DatasetSplitRole.HOLDOUT: "formal-holdout",
}


@dataclass(frozen=True)
class FormalPreparedCorpora:
    analysis: PreparedCorpusPackage
    challenge: PreparedCorpusPackage
    holdout: PreparedCorpusPackage

    def __iter__(self):
        return iter((self.analysis, self.challenge, self.holdout))

    def __getitem__(self, index: int) -> PreparedCorpusPackage:
        return (self.analysis, self.challenge, self.holdout)[index]


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path}") from exc


def _read_object(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _resolve_source_manifest_path(source_root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("formal manifest source_manifests.normalized_csv is required")
    path = Path(value)
    if path.is_absolute():
        return path
    source_relative = source_root / path
    if source_relative.exists():
        return source_relative
    return Path.cwd() / path


def _bool_value(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"source record {field} must be boolean")


def _source_reference(values: Mapping[str, object]) -> SourceReference:
    return SourceReference(
        source_id=str(values["source_id"]),
        source_type=str(values["source_type"]),
        source_ref=str(values["source_ref"]),
    )


def _record_from_csv(row: Mapping[str, str]) -> CommentRecord:
    source = _source_reference(row)
    payload: dict[str, object] = {
        "comment_id": row.get("comment_id") or row.get("raw_id"),
        "sample_type": row.get("sample_type"),
        "raw_sample_type": row.get("raw_sample_type"),
        "raw_content": row.get("raw_content"),
        "context_content": row.get("context_content") or None,
        "source_platform": row.get("source_platform"),
        "raw_id": row.get("raw_id") or row.get("comment_id"),
        "original_url": row.get("original_url") or None,
        "platform_url_available": _bool_value(
            row.get("platform_url_available"), "platform_url_available"
        ),
        "collected_at": row.get("collected_at") or None,
        "screening_status": row.get("screening_status") or None,
        "screening_reason": row.get("screening_reason") or None,
        "source": source,
    }
    return CommentRecord.model_validate_json(canonical_json_bytes(payload), strict=True)


def _load_source_records(path: Path) -> dict[str, CommentRecord]:
    if not path.is_file():
        raise ValueError(f"formal source records do not exist: {path}")
    records: list[CommentRecord] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("formal source CSV must have a header")
            records = [_record_from_csv(row) for row in reader]
    else:
        payload = _read_json(path)
        if isinstance(payload, dict):
            payload = payload.get("records")
        if not isinstance(payload, list):
            raise ValueError("formal source records must be a JSON list")
        records = [
            CommentRecord.model_validate_json(canonical_json_bytes(item), strict=True)
            for item in payload
        ]

    result: dict[str, CommentRecord] = {}
    for record in records:
        stable_id = record.raw_id or record.comment_id
        if stable_id in result:
            raise ValueError(f"formal source records contain duplicate stable ID: {stable_id}")
        result[stable_id] = record
    return result


def _load_split_manifest(path: Path) -> DatasetSplitManifest:
    return DatasetSplitManifest.model_validate_json(path.read_bytes(), strict=True)


def _load_selection_ids(path: Path) -> tuple[str, ...]:
    payload = _read_object(path)
    selected = payload.get("selected_ids")
    if not isinstance(selected, list) or len(selected) != 20:
        raise ValueError("selection_manifest.selected_ids must contain exactly 20 IDs")
    if not all(isinstance(value, str) and value for value in selected):
        raise ValueError("selection_manifest.selected_ids must contain strings")
    if len(set(selected)) != 20:
        raise ValueError("selection_manifest.selected_ids must be unique")
    return tuple(selected)


def _formal_groups(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    groups = payload.get("groups")
    if not isinstance(groups, list):
        raise ValueError("formal manifest groups are required")
    result: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("group_id"), str):
            raise ValueError("formal manifest group is invalid")
        group_id = group["group_id"]
        if group_id in result:
            raise ValueError(f"duplicate formal group: {group_id}")
        result[group_id] = group
    expected = {item[0] for item in _GROUPS}
    if set(result) != expected:
        raise ValueError("formal manifest must contain baseline, release, challenge and holdout groups")
    return result


def _group_batch_entries(
    groups: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
    ordered_batch_ids: list[str] = []
    input_hashes: dict[str, str] = {}
    ids_by_group: dict[str, list[str]] = {}
    for group_id, _, _ in _GROUPS:
        group = groups[group_id]
        batches = group.get("batches")
        batch_ids = group.get("batch_ids")
        if not isinstance(batches, list) or not isinstance(batch_ids, list):
            raise ValueError(f"formal group {group_id} batches are required")
        if batch_ids != [item.get("batch_id") for item in batches]:
            raise ValueError(f"formal group {group_id} batch_ids are not ordered")
        group_ids: list[str] = []
        for batch in batches:
            if not isinstance(batch, Mapping):
                raise ValueError(f"formal group {group_id} batch is invalid")
            batch_id = batch.get("batch_id")
            input_sha256 = batch.get("input_sha256")
            input_ids = batch.get("input_ids")
            if (
                not isinstance(batch_id, str)
                or not isinstance(input_sha256, str)
                or not isinstance(input_ids, list)
                or not all(isinstance(value, str) for value in input_ids)
            ):
                raise ValueError(f"formal group {group_id} batch identity is invalid")
            if batch_id in input_hashes:
                raise ValueError(f"duplicate formal batch_id: {batch_id}")
            ordered_batch_ids.append(batch_id)
            input_hashes[batch_id] = input_sha256
            group_ids.extend(input_ids)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError(f"formal group {group_id} contains duplicate stable IDs")
        ids_by_group[group_id] = group_ids
    return ordered_batch_ids, input_hashes, ids_by_group


def _result_path(source_root: Path, group_id: str, batch_id: str, formal_manifest: Mapping[str, Any]) -> Path:
    attempt_id = formal_manifest.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("formal manifest attempt_id is required")
    attempt_dir = attempt_id.removeprefix("formal-")
    return source_root / "results" / group_id / attempt_dir / f"{batch_id}.json"


def _annotation_manifest_identity(manifest: AnnotationRunManifest) -> FormalAnnotationIdentity:
    return FormalAnnotationIdentity(
        annotation_version=manifest.annotation_version,
        model_id=manifest.model_id,
        reasoning_effort=manifest.reasoning_effort,
        dataset_version=manifest.dataset_version,
        dataset_sha256=manifest.dataset_sha256,
        prompt_version=manifest.prompt_version,
        prompt_sha256=manifest.prompt_sha256,
        result_sha256=manifest.result_sha256,
    )


def _make_evidence_atoms(
    records: Mapping[str, CommentRecord], annotations: Iterable[OfflineAnnotation]
) -> tuple[dict[str, CommentRecord], dict[str, EvidenceAtom]]:
    selected_records: dict[str, CommentRecord] = {}
    atoms: dict[str, EvidenceAtom] = {}
    for annotation in annotations:
        record = records.get(annotation.raw_id)
        if record is None:
            raise ValueError(f"annotation references unknown stable ID: {annotation.raw_id}")
        if annotation.raw_id in selected_records:
            raise ValueError(f"annotation stable IDs must be unique: {annotation.raw_id}")
        selected_records[annotation.raw_id] = record
        atoms[annotation.raw_id] = EvidenceAtom(
            evidence_id=f"formal-evidence-{annotation.raw_id}",
            comment_id=record.comment_id,
            route=annotation.route,
            experience_scope=annotation.experience_scope,
            evidence_grade=annotation.evidence_grade,
            ai_confidence=annotation.ai_confidence,
            explanation=annotation.explanation,
            source=record.source,  # type: ignore[arg-type]
            source_platform=record.source_platform,
            duplicate_group=record.duplicate_group,
            actual_use=record.actual_use,
        )
    return selected_records, atoms


def _package(
    *,
    role: DatasetSplitRole,
    stable_ids: list[str],
    records: Mapping[str, CommentRecord],
    atoms: Mapping[str, EvidenceAtom],
    split_manifest: DatasetSplitManifest,
    annotation_manifest: AnnotationRunManifest,
) -> PreparedCorpusPackage:
    package_records = [records[stable_id] for stable_id in stable_ids]
    package_atoms = [atoms[stable_id] for stable_id in stable_ids]
    assignments_by_id = {
        assignment.raw_id: assignment for assignment in split_manifest.assignments
    }
    try:
        assignments = [assignments_by_id[stable_id] for stable_id in stable_ids]
    except KeyError as exc:
        raise ValueError(f"{role.value} package contains an unknown split ID") from exc
    package_split = split_manifest.model_copy(update={"assignments": assignments})
    artifact_values = {
        "records.json": canonical_json_bytes(package_records),
        "evidence_atoms.json": canonical_json_bytes(package_atoms),
        "split_manifest.json": canonical_json_bytes(package_split),
        "annotation_run_manifests.json": canonical_json_bytes([annotation_manifest]),
    }
    artifact_sha256 = {
        name: sha256_bytes(value) for name, value in artifact_values.items()
    }
    manifest = PreparedCorpusManifest(
        package_id=_PACKAGE_IDS[role],
        dataset_version=annotation_manifest.dataset_version,
        dataset_sha256=annotation_manifest.dataset_sha256,
        annotation_version=annotation_manifest.annotation_version,
        annotation_model_id=annotation_manifest.model_id,
        annotation_reasoning_effort=annotation_manifest.reasoning_effort,
        annotation_prompt_version=annotation_manifest.prompt_version,
        annotation_prompt_sha256=annotation_manifest.prompt_sha256,
        validation_status=PackageValidationStatus.PIPELINE_VALIDATED_ONLY,
        record_count=len(package_records),
        evidence_count=len(package_atoms),
        artifact_sha256=artifact_sha256,
        package_sha256=package_digest(artifact_sha256),
    )
    package = PreparedCorpusPackage(
        manifest=manifest,
        records=package_records,
        evidence_atoms=package_atoms,
        split_manifest=package_split,
        annotation_run_manifests=[annotation_manifest],
    )
    validate_prepared_corpus_package(package)
    return package


def build_formal_prepared_corpora(
    *,
    source_root: Path,
    split_manifest_path: Path,
    selection_manifest_path: Path,
) -> FormalPreparedCorpora:
    source_root = Path(source_root).resolve()
    formal_manifest = _read_object(source_root / "formal_task_manifest.json")
    annotation_manifest = AnnotationRunManifest.model_validate_json(
        (source_root / "annotation_run_manifest.json").read_bytes(), strict=True
    )
    if formal_manifest.get("dataset_version") != annotation_manifest.dataset_version:
        raise ValueError("formal manifest dataset_version does not match annotation manifest")
    if formal_manifest.get("dataset_sha256") != annotation_manifest.dataset_sha256:
        raise ValueError("formal manifest dataset_sha256 does not match annotation manifest")
    if formal_manifest.get("prompt_version") != annotation_manifest.prompt_version:
        raise ValueError("formal manifest prompt_version does not match annotation manifest")
    if formal_manifest.get("prompt_sha256") != annotation_manifest.prompt_sha256:
        raise ValueError("formal manifest prompt_sha256 does not match annotation manifest")
    if formal_manifest.get("model_id") != annotation_manifest.model_id:
        raise ValueError("formal manifest model_id does not match annotation manifest")
    if formal_manifest.get("reasoning_effort") != annotation_manifest.reasoning_effort:
        raise ValueError("formal manifest reasoning_effort does not match annotation manifest")

    source_manifests = formal_manifest.get("source_manifests")
    if not isinstance(source_manifests, Mapping):
        raise ValueError("formal manifest source_manifests are required")
    records_path = _resolve_source_manifest_path(
        source_root, source_manifests.get("normalized_csv")
    )
    records = _load_source_records(records_path)
    split_manifest = _load_split_manifest(Path(split_manifest_path))
    if (
        split_manifest.dataset_version != annotation_manifest.dataset_version
        or split_manifest.dataset_sha256 != annotation_manifest.dataset_sha256
    ):
        raise ValueError("split manifest dataset identity does not match annotation manifest")
    selected_ids = _load_selection_ids(Path(selection_manifest_path))
    assignments_by_role: dict[DatasetSplitRole, list[DatasetSplitAssignment]] = {
        role: [] for role in DatasetSplitRole
    }
    assignment_by_id: dict[str, DatasetSplitAssignment] = {}
    for assignment in split_manifest.assignments:
        if assignment.raw_id in assignment_by_id:
            raise ValueError(f"split manifest contains duplicate stable ID: {assignment.raw_id}")
        assignment_by_id[assignment.raw_id] = assignment
        assignments_by_role[assignment.split_role].append(assignment)

    analysis_ids = [item.raw_id for item in assignments_by_role[DatasetSplitRole.ANALYSIS]]
    challenge_ids = [
        item.raw_id for item in assignments_by_role[DatasetSplitRole.CHALLENGE_POOL]
    ]
    holdout_ids = [item.raw_id for item in assignments_by_role[DatasetSplitRole.HOLDOUT]
    ]
    if [len(analysis_ids), len(challenge_ids), len(holdout_ids)] != [249, 60, 60]:
        raise ValueError("formal split must contain ANALYSIS 249, CHALLENGE_POOL 60 and HOLDOUT 60")
    if not set(selected_ids).issubset(analysis_ids):
        raise ValueError("selection manifest IDs must belong to ANALYSIS")
    baseline_ids = [stable_id for stable_id in analysis_ids if stable_id not in selected_ids]
    if len(baseline_ids) != 229:
        raise ValueError("ANALYSIS baseline/release boundary must be 229/20")
    all_ids = [*analysis_ids, *challenge_ids, *holdout_ids]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("formal split stable IDs must be globally unique")
    if not set(all_ids).issubset(records):
        missing = sorted(set(all_ids) - set(records))[0]
        raise ValueError(f"formal source records missing stable ID: {missing}")

    groups = _formal_groups(formal_manifest)
    ordered_batch_ids, input_hashes, ids_by_group = _group_batch_entries(groups)
    if ordered_batch_ids != annotation_manifest.batch_ids:
        raise ValueError("formal batch IDs do not match annotation run manifest")
    expected_group_ids = {
        "analysis_baseline": baseline_ids,
        "analysis_demo_delta": list(selected_ids),
        "challenge_pool": challenge_ids,
        "holdout": holdout_ids,
    }
    if (
        ids_by_group["analysis_demo_delta"] != expected_group_ids["analysis_demo_delta"]
        or any(
            set(ids_by_group[group_id]) != set(expected_group_ids[group_id])
            for group_id in ("analysis_baseline", "challenge_pool", "holdout")
        )
    ):
        raise ValueError("formal task groups do not match split and selection manifests")

    result_files = [
        _result_path(source_root, group_id, batch_id, formal_manifest)
        for group_id, _, _ in _GROUPS
        for batch_id in groups[group_id]["batch_ids"]
    ]
    if any(not path.is_file() for path in result_files):
        missing = next(path for path in result_files if not path.is_file())
        raise ValueError(f"formal annotation result does not exist: {missing}")
    imported = import_offline_annotations(
        annotation_manifest,
        result_files,
        dataset_sha256=split_manifest.dataset_sha256,
        prompt_sha256=annotation_manifest.prompt_sha256,
        dataset_version=split_manifest.dataset_version,
        prompt_version=annotation_manifest.prompt_version,
        prompt_metadata=PromptMetadata(
            name="formal-routing",
            version=annotation_manifest.prompt_version,
            sha256=annotation_manifest.prompt_sha256,
            content="",
            raw_bytes=b"",
        ),
        trusted_batch_input_hashes=input_hashes,
    )
    if imported.input_ids != [stable_id for group_id, _, _ in _GROUPS for stable_id in ids_by_group[group_id]]:
        raise ValueError("formal annotation input IDs do not preserve task order")
    selected_records, atoms = _make_evidence_atoms(records, imported.annotations)
    if set(selected_records) != set(all_ids):
        raise ValueError("formal annotation IDs do not exactly cover the three official packages")

    return FormalPreparedCorpora(
        analysis=_package(
            role=DatasetSplitRole.ANALYSIS,
            stable_ids=ids_by_group["analysis_baseline"] + list(selected_ids),
            records=selected_records,
            atoms=atoms,
            split_manifest=split_manifest,
            annotation_manifest=annotation_manifest,
        ),
        challenge=_package(
            role=DatasetSplitRole.CHALLENGE_POOL,
            stable_ids=ids_by_group["challenge_pool"],
            records=selected_records,
            atoms=atoms,
            split_manifest=split_manifest,
            annotation_manifest=annotation_manifest,
        ),
        holdout=_package(
            role=DatasetSplitRole.HOLDOUT,
            stable_ids=ids_by_group["holdout"],
            records=selected_records,
            atoms=atoms,
            split_manifest=split_manifest,
            annotation_manifest=annotation_manifest,
        ),
    )


def _stable_id(record: CommentRecord) -> str:
    return record.raw_id or record.comment_id


def _stable_ids_sha256(ids: Iterable[str]) -> str:
    return sha256_bytes(canonical_json_bytes(sorted(ids)))


def _group_sha256(manifest: FormalPreparedCorpusGroupManifest) -> str:
    payload = manifest.model_dump(mode="json", exclude={"group_sha256"})
    return sha256_bytes(canonical_json_bytes(payload))


def _validate_group_id(group_id: str) -> None:
    if not isinstance(group_id, str) or not group_id or group_id in {".", ".."}:
        raise ValueError("group_id must be a non-empty directory name")
    if Path(group_id).is_absolute() or "/" in group_id or "\\" in group_id or "\x00" in group_id:
        raise ValueError("group_id must not contain path traversal")


def _make_group_manifest(
    *, group_id: str, published: tuple[PublishedPreparedCorpus, ...], corpora: FormalPreparedCorpora
) -> FormalPreparedCorpusGroupManifest:
    packages = {item.manifest.package_id: item for item in published}
    expected = {
        "ANALYSIS": corpora.analysis,
        "CHALLENGE": corpora.challenge,
        "HOLDOUT": corpora.holdout,
    }
    package_ids = {role: expected_package.manifest.package_id for role, expected_package in expected.items()}
    package_sha256 = {
        role: packages[package_id].manifest.package_sha256
        for role, package_id in package_ids.items()
    }
    analysis_ids = [_stable_id(record) for record in corpora.analysis.records]
    if len(analysis_ids) != 249:
        raise ValueError("ANALYSIS package must contain exactly 249 records")
    release_ids = analysis_ids[-20:]
    baseline_ids = [stable_id for stable_id in analysis_ids if stable_id not in set(release_ids)]
    all_ids = [
        *analysis_ids,
        *[_stable_id(record) for record in corpora.challenge.records],
        *[_stable_id(record) for record in corpora.holdout.records],
    ]
    annotation_manifest = corpora.analysis.annotation_run_manifests[0]
    manifest = FormalPreparedCorpusGroupManifest(
        group_id=group_id,
        package_ids=package_ids,
        package_sha256=package_sha256,
        analysis_baseline_ids=baseline_ids,
        analysis_release_ids=release_ids,
        all_stable_ids_sha256=_stable_ids_sha256(all_ids),
        annotation_identity=_annotation_manifest_identity(annotation_manifest),
        group_sha256="0" * 64,
    )
    return manifest.model_copy(update={"group_sha256": _group_sha256(manifest)})


def _validate_group_manifest_and_packages(
    manifest: FormalPreparedCorpusGroupManifest,
    packages: tuple[PublishedPreparedCorpus, ...],
) -> None:
    by_role = dict(zip(("ANALYSIS", "CHALLENGE", "HOLDOUT"), packages, strict=True))
    expected_counts = {"ANALYSIS": 249, "CHALLENGE": 60, "HOLDOUT": 60}
    all_ids: list[str] = []
    annotation_identity: FormalAnnotationIdentity | None = None
    for role, published in by_role.items():
        package = published.package
        if package.manifest.package_id != manifest.package_ids[role]:
            raise ValueError("formal group package ID does not match package manifest")
        if package.manifest.package_sha256 != manifest.package_sha256[role]:
            raise ValueError("formal group package SHA-256 does not match package manifest")
        if package.manifest.record_count != expected_counts[role]:
            raise ValueError("formal group package count does not match role boundary")
        package_ids = [_stable_id(record) for record in package.records]
        if len(package_ids) != len(set(package_ids)):
            raise ValueError("formal group package stable IDs must be unique")
        expected_role = {
            "ANALYSIS": DatasetSplitRole.ANALYSIS,
            "CHALLENGE": DatasetSplitRole.CHALLENGE_POOL,
            "HOLDOUT": DatasetSplitRole.HOLDOUT,
        }[role]
        if {assignment.split_role for assignment in package.split_manifest.assignments} != {
            expected_role
        }:
            raise ValueError("formal group package split role does not match package role")
        all_ids.extend(package_ids)
        if len(package.annotation_run_manifests) != 1:
            raise ValueError("formal group packages must contain one annotation run manifest")
        current_identity = _annotation_manifest_identity(
            package.annotation_run_manifests[0]
        )
        if annotation_identity is None:
            annotation_identity = current_identity
        elif current_identity != annotation_identity:
            raise ValueError("formal group annotation identity is not shared")
    if annotation_identity != manifest.annotation_identity:
        raise ValueError("formal group annotation identity does not match packages")
    if len(all_ids) != 369 or len(set(all_ids)) != 369:
        raise ValueError("formal group must contain exactly 369 unique stable IDs")
    if manifest.all_stable_ids_sha256 != _stable_ids_sha256(all_ids):
        raise ValueError("formal group all_stable_ids_sha256 does not match packages")
    analysis_ids = set(_stable_id(record) for record in by_role["ANALYSIS"].package.records)
    baseline_ids = set(manifest.analysis_baseline_ids)
    release_ids = set(manifest.analysis_release_ids)
    if baseline_ids | release_ids != analysis_ids:
        raise ValueError("formal group baseline/release IDs do not cover ANALYSIS")
    if baseline_ids.intersection(release_ids):
        raise ValueError("formal group baseline/release IDs overlap")


def publish_formal_prepared_corpora(
    *,
    prepared_root: Path,
    group_id: str,
    corpora: FormalPreparedCorpora,
) -> tuple[PublishedPreparedCorpus, PublishedPreparedCorpus, PublishedPreparedCorpus]:
    if not isinstance(corpora, FormalPreparedCorpora):
        raise TypeError("corpora must be FormalPreparedCorpora")
    _validate_group_id(group_id)
    root = Path(prepared_root)
    if root.exists() and not root.is_dir():
        raise ValueError("prepared_root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / group_id
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"formal group already exists: {group_id}")

    temp_group = Path(tempfile.mkdtemp(prefix=f".{group_id}.", dir=str(root)))
    packages_root = temp_group / "packages"
    published: tuple[PublishedPreparedCorpus, ...] | None = None
    try:
        published = tuple(
            publish_prepared_corpus(packages_root, package)
            for package in (corpora.analysis, corpora.challenge, corpora.holdout)
        )
        group_manifest = _make_group_manifest(
            group_id=group_id,
            published=published,
            corpora=corpora,
        )
        _write_json(temp_group / "manifest.json", group_manifest)
        reloaded_manifest = FormalPreparedCorpusGroupManifest.model_validate_json(
            (temp_group / "manifest.json").read_bytes(), strict=True
        )
        if reloaded_manifest.group_sha256 != _group_sha256(reloaded_manifest):
            raise ValueError("formal group SHA-256 does not match manifest")
        reloaded = tuple(
            load_prepared_corpus(packages_root, reloaded_manifest.package_ids[role])
            for role in ("ANALYSIS", "CHALLENGE", "HOLDOUT")
        )
        _validate_group_manifest_and_packages(reloaded_manifest, reloaded)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"formal group already exists: {group_id}")
        os.rename(temp_group, destination)
        temp_group = Path()
    except Exception:
        if temp_group != Path() and temp_group.exists():
            shutil.rmtree(temp_group)
        raise

    return tuple(
        load_prepared_corpus(
            destination / "packages",
            package_id,
        )
        for package_id in (
            corpora.analysis.manifest.package_id,
            corpora.challenge.manifest.package_id,
            corpora.holdout.manifest.package_id,
        )
    )  # type: ignore[return-value]
