from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.schemas import (
    EvidenceAtom,
    EvidenceGrade,
    EvidenceImpact,
    EvidenceImpactRecord,
    EvidenceRoute,
)


def eligible_brand_atoms(atoms: Iterable[EvidenceAtom]) -> list[EvidenceAtom]:
    """Return independent, non-C BRAND evidence only when at least two exist."""

    eligible = [
        atom
        for atom in atoms
        if atom.route is EvidenceRoute.BRAND and atom.evidence_grade is not EvidenceGrade.C
    ]
    independent: list[EvidenceAtom] = []
    independent_groups: set[str] = set()
    for atom in eligible:
        group = atom.duplicate_group or atom.evidence_id
        if group not in independent_groups:
            independent.append(atom)
            independent_groups.add(group)
    return independent if len(independent) >= 2 else []


def build_evidence_impacts(
    *,
    atoms: Iterable[EvidenceAtom],
    proposed_impacts: Mapping[str, tuple[EvidenceImpact, str | None]],
    conflict_ids: set[str],
    candidate_ids: set[str],
) -> list[EvidenceImpactRecord]:
    """Attach impacts only to already existing conflicts or candidates."""

    atom_list = list(atoms)
    atom_ids = {atom.evidence_id for atom in atom_list}
    if set(proposed_impacts) - atom_ids:
        raise ValueError("Evidence Impact 引用了不存在的 evidence_id")
    target_ids = conflict_ids | candidate_ids
    records: list[EvidenceImpactRecord] = []
    for atom in atom_list:
        proposal = proposed_impacts.get(atom.evidence_id)
        if proposal is None:
            records.append(
                EvidenceImpactRecord(
                    evidence_id=atom.evidence_id,
                    impact=EvidenceImpact.NEW_SIGNAL,
                )
            )
            continue
        impact, target_id = proposal
        if impact in {EvidenceImpact.SUPPORT, EvidenceImpact.CHALLENGE}:
            if target_id is None or target_id not in target_ids:
                raise ValueError("SUPPORT / CHALLENGE 必须绑定已有目标 ID")
        records.append(
            EvidenceImpactRecord(
                evidence_id=atom.evidence_id,
                impact=impact,
                target_id=target_id,
            )
        )
    return records
