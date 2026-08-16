from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

from src.schemas import (
    CommentRecord,
    DatasetSplitRole,
    EvidenceAtom,
    PreparedCorpusPackage,
)
from src.services.gold_decision_input import GoldDecisionInput
from src.services.prepared_corpus import validate_prepared_corpus_package


@dataclass(frozen=True)
class DecisionInputBundle:
    baseline_records: tuple[CommentRecord, ...]
    baseline_atoms: tuple[EvidenceAtom, ...]
    challenge_atoms: tuple[EvidenceAtom, ...]
    holdout_atoms: tuple[EvidenceAtom, ...]
    release_batches: tuple[tuple[EvidenceAtom, ...], ...]
    analysis_package_sha256: str | None = None
    challenge_package_sha256: str | None = None
    holdout_package_sha256: str | None = None
    all_stable_ids_sha256: str | None = None
    gold_input: GoldDecisionInput | None = None
    gold_records: tuple[CommentRecord, ...] = ()
    gold_atoms: tuple[EvidenceAtom, ...] = ()
    gold_calibration_sha256: str | None = None
    gold_result_sha256: str | None = None
    gold_stable_ids_sha256: str | None = None
    all_decision_stable_ids_sha256: str | None = None


_EXPECTED_COUNTS = {
    DatasetSplitRole.ANALYSIS: 249,
    DatasetSplitRole.CHALLENGE_POOL: 60,
    DatasetSplitRole.HOLDOUT: 60,
}


def _require_package(package: object, name: str) -> PreparedCorpusPackage:
    if not isinstance(package, PreparedCorpusPackage):
        raise ValueError(f"{name} package is required")
    return package


def _stable_id(record: CommentRecord) -> str:
    return record.raw_id or record.comment_id


def _split_contract_identity(package: PreparedCorpusPackage) -> tuple[object, ...]:
    split = package.split_manifest
    return (
        split.split_algorithm_version,
        tuple(sorted(split.holdout_stratum_targets.items())),
        split.holdout_stratum_tolerance,
    )


def _package_identity(package: PreparedCorpusPackage) -> tuple[object, ...]:
    manifest = package.manifest
    return (
        manifest.dataset_version,
        manifest.dataset_sha256,
        manifest.annotation_version,
        manifest.annotation_model_id,
        manifest.annotation_reasoning_effort,
        manifest.annotation_prompt_version,
        manifest.annotation_prompt_sha256,
        tuple(
            (
                annotation.annotation_version,
                annotation.model_id,
                annotation.reasoning_effort,
                annotation.dataset_version,
                annotation.dataset_sha256,
                annotation.prompt_version,
                annotation.prompt_sha256,
                annotation.result_sha256,
                tuple(annotation.batch_ids),
            )
            for annotation in package.annotation_run_manifests
        ),
        _split_contract_identity(package),
    )


def _require_matching_package_identities(
    packages: tuple[PreparedCorpusPackage, ...],
) -> None:
    expected = _package_identity(packages[0])
    if any(_package_identity(package) != expected for package in packages[1:]):
        raise ValueError("official package identity mismatch")


def _package_indexes(
    package: PreparedCorpusPackage, name: str
) -> tuple[dict[str, CommentRecord], dict[str, DatasetSplitRole]]:
    records_by_comment_id: dict[str, CommentRecord] = {}
    roles_by_stable_id: dict[str, DatasetSplitRole] = {}
    for assignment in package.split_manifest.assignments:
        if assignment.raw_id in roles_by_stable_id:
            raise ValueError(f"{name} split manifest contains duplicate raw_id")
        roles_by_stable_id[assignment.raw_id] = assignment.split_role

    for record in package.records:
        if record.comment_id in records_by_comment_id:
            raise ValueError(f"{name} records contain duplicate comment_id")
        stable_id = _stable_id(record)
        if stable_id in {item.raw_id or item.comment_id for item in records_by_comment_id.values()}:
            raise ValueError(f"{name} records contain duplicate stable ID")
        role = roles_by_stable_id.get(stable_id)
        if role is None:
            raise ValueError(f"{name} record ID does not match split manifest")
        records_by_comment_id[record.comment_id] = record
    return records_by_comment_id, roles_by_stable_id


def _role_atoms(
    package: PreparedCorpusPackage,
    *,
    name: str,
    role: DatasetSplitRole,
) -> tuple[tuple[str, CommentRecord, EvidenceAtom], ...]:
    records_by_comment_id, roles_by_stable_id = _package_indexes(package, name)
    items: list[tuple[str, CommentRecord, EvidenceAtom]] = []
    seen_stable_ids: set[str] = set()
    seen_evidence_ids: set[str] = set()
    for atom in package.evidence_atoms:
        record = records_by_comment_id.get(atom.comment_id)
        if record is None:
            raise ValueError(f"{name} evidence atom has no corresponding record")
        stable_id = _stable_id(record)
        if roles_by_stable_id[stable_id] is not role:
            raise ValueError(f"{name} evidence atom has a mismatched split role")
        if stable_id in seen_stable_ids:
            raise ValueError(f"{name} evidence atoms contain duplicate stable ID")
        if atom.evidence_id in seen_evidence_ids:
            raise ValueError(f"{name} evidence atoms contain duplicate evidence_id")
        seen_stable_ids.add(stable_id)
        seen_evidence_ids.add(atom.evidence_id)
        items.append((stable_id, record, atom))
    expected_count = _EXPECTED_COUNTS[role]
    if len(items) != expected_count:
        raise ValueError(f"{name} must contain exactly {expected_count} evidence atoms")
    return tuple(items)


def _ordered_selected_ids(selection_manifest: Mapping[str, object]) -> tuple[str, ...]:
    if not isinstance(selection_manifest, Mapping):
        raise ValueError("selection_manifest is required")
    selected_value = selection_manifest.get("selected_ids")
    if not isinstance(selected_value, (list, tuple)):
        raise ValueError("selection_manifest.selected_ids must be a sequence")
    if len(selected_value) != 20 or not all(
        isinstance(value, str) and value for value in selected_value
    ):
        raise ValueError("selection_manifest.selected_ids must contain exactly 20 IDs")
    selected_ids = tuple(selected_value)
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selection_manifest.selected_ids must be unique")
    return selected_ids


def _validate_gold_input(
    gold_input: GoldDecisionInput,
    *,
    official_ids: set[str],
) -> tuple[tuple[CommentRecord, ...], tuple[EvidenceAtom, ...]]:
    if not isinstance(gold_input, GoldDecisionInput):
        raise ValueError("gold_input must be a Human Gold decision input")
    gold_input.validate()
    records = tuple(gold_input.records)
    atoms = tuple(gold_input.atoms)
    gold_ids = {_stable_id(record) for record in records}
    if gold_ids & official_ids:
        raise ValueError("Human Gold IDs must be disjoint from official ANALYSIS/CHALLENGE/HOLDOUT IDs")
    records_by_comment_id = {record.comment_id: record for record in records}
    for atom in atoms:
        record = records_by_comment_id.get(atom.comment_id)
        if record is None or _stable_id(record) not in gold_ids:
            raise ValueError("Human Gold atom must reference a Human Gold record")
        if atom.label_source != "HUMAN_GOLD":
            raise ValueError("Human Gold atoms must have label_source=HUMAN_GOLD")
    return records, atoms


def assemble_official_demo_inputs(
    *,
    analysis: PreparedCorpusPackage,
    challenge: PreparedCorpusPackage,
    holdout: PreparedCorpusPackage,
    selection_manifest: Mapping[str, object],
    gold_input: GoldDecisionInput | None = None,
) -> DecisionInputBundle:
    analysis_package = _require_package(analysis, "analysis")
    challenge_package = _require_package(challenge, "challenge")
    holdout_package = _require_package(holdout, "holdout")
    packages = (analysis_package, challenge_package, holdout_package)
    for package in packages:
        validate_prepared_corpus_package(package)
    _require_matching_package_identities(packages)
    selected_ids = _ordered_selected_ids(selection_manifest)

    analysis_items = _role_atoms(
        analysis_package,
        name="analysis",
        role=DatasetSplitRole.ANALYSIS,
    )
    challenge_items = _role_atoms(
        challenge_package,
        name="challenge",
        role=DatasetSplitRole.CHALLENGE_POOL,
    )
    holdout_items = _role_atoms(
        holdout_package,
        name="holdout",
        role=DatasetSplitRole.HOLDOUT,
    )

    analysis_ids = {stable_id for stable_id, _, _ in analysis_items}
    challenge_ids = {stable_id for stable_id, _, _ in challenge_items}
    holdout_ids = {stable_id for stable_id, _, _ in holdout_items}
    analysis_record_ids = {_stable_id(record) for record in analysis_package.records}
    challenge_record_ids = {_stable_id(record) for record in challenge_package.records}
    holdout_record_ids = {_stable_id(record) for record in holdout_package.records}
    if (
        analysis_record_ids & challenge_record_ids
        or analysis_record_ids & holdout_record_ids
        or challenge_record_ids & holdout_record_ids
    ):
        raise ValueError("official split IDs must be pairwise disjoint")
    official_ids = analysis_record_ids | challenge_record_ids | holdout_record_ids
    gold_records: tuple[CommentRecord, ...] = ()
    gold_atoms: tuple[EvidenceAtom, ...] = ()
    if gold_input is not None:
        gold_records, gold_atoms = _validate_gold_input(
            gold_input,
            official_ids=official_ids,
        )

    atoms_by_id = {stable_id: (record, atom) for stable_id, record, atom in analysis_items}
    unknown_ids = set(selected_ids) - analysis_ids
    if unknown_ids:
        raise ValueError("selection_manifest contains an unknown ANALYSIS ID")

    selected_id_set = set(selected_ids)
    baseline_records = tuple(
        record
        for record in analysis_package.records
        if _stable_id(record) in analysis_ids
        and _stable_id(record) not in selected_id_set
    )
    baseline_atoms = tuple(
        atoms_by_id[_stable_id(record)][1] for record in baseline_records
    )
    release_atoms = tuple(atoms_by_id[stable_id][1] for stable_id in selected_ids)
    release_batches = tuple(
        tuple(release_atoms[offset : offset + 5]) for offset in range(0, 20, 5)
    )

    baseline_ids = {_stable_id(record) for record in baseline_records}
    release_ids = set(selected_ids)
    if len(baseline_records) != 229 or baseline_ids & release_ids:
        raise ValueError("ANALYSIS baseline and release batches are invalid")
    if baseline_ids | release_ids != analysis_ids:
        raise ValueError("ANALYSIS baseline and release batches do not cover exactly 249 IDs")

    baseline_records = (*baseline_records, *gold_records)
    baseline_atoms = (*baseline_atoms, *gold_atoms)

    all_stable_ids = [
        *analysis_ids,
        *challenge_ids,
        *holdout_ids,
    ]
    if len(all_stable_ids) != 369 or len(set(all_stable_ids)) != 369:
        raise ValueError("official packages must contain exactly 369 unique stable IDs")
    annotation_manifests = [
        annotation_manifest
        for package in packages
        for annotation_manifest in package.annotation_run_manifests
    ]
    if not annotation_manifests or any(
        annotation_manifest.result_sha256 != annotation_manifests[0].result_sha256
        for annotation_manifest in annotation_manifests[1:]
    ):
        raise ValueError("official annotation result identity mismatch")
    all_stable_ids_sha256 = hashlib.sha256(
        json.dumps(
            sorted(all_stable_ids),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        ).hexdigest()

    all_decision_ids = [*all_stable_ids, *(_stable_id(record) for record in gold_records)]
    all_decision_stable_ids_sha256 = (
        hashlib.sha256(
            json.dumps(
                sorted(all_decision_ids),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if gold_input is not None
        else None
    )

    return DecisionInputBundle(
        baseline_records=baseline_records,
        baseline_atoms=baseline_atoms,
        challenge_atoms=tuple(atom for _, _, atom in challenge_items),
        holdout_atoms=tuple(atom for _, _, atom in holdout_items),
        release_batches=release_batches,
        analysis_package_sha256=analysis_package.manifest.package_sha256,
        challenge_package_sha256=challenge_package.manifest.package_sha256,
        holdout_package_sha256=holdout_package.manifest.package_sha256,
        all_stable_ids_sha256=all_stable_ids_sha256,
        gold_input=gold_input,
        gold_records=gold_records,
        gold_atoms=gold_atoms,
        gold_calibration_sha256=(gold_input.calibration_sha256 if gold_input else None),
        gold_result_sha256=(gold_input.result_sha256 if gold_input else None),
        gold_stable_ids_sha256=(gold_input.stable_ids_sha256 if gold_input else None),
        all_decision_stable_ids_sha256=all_decision_stable_ids_sha256,
    )
