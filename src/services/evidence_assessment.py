from __future__ import annotations

from collections.abc import Iterable

from pydantic import Field, model_validator

from src.llm_client import LLMClient
from src.schemas import (
    EvidenceAtom,
    EvidenceImpact,
    EvidenceImpactRecord,
    HoldoutEvidenceImpact,
    StrictBaseModel,
)
from src.services import _build_system_prompt, _serialize_prompt_input
from src.services.evidence_impact import build_evidence_impacts


class EvidenceAssessmentProposal(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    impact: EvidenceImpact
    target_id: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validates_target_contract(self) -> "EvidenceAssessmentProposal":
        if self.impact in {EvidenceImpact.SUPPORT, EvidenceImpact.CHALLENGE}:
            if self.target_id is None:
                raise ValueError("SUPPORT / CHALLENGE 必须绑定目标 ID")
        elif self.target_id is not None:
            raise ValueError("NEW_SIGNAL / NEUTRAL 不得绑定目标 ID")
        return self


class _EvidenceAssessmentResult(StrictBaseModel):
    proposals: list[EvidenceAssessmentProposal]


class _HoldoutAssessmentResult(StrictBaseModel):
    impacts: list[HoldoutEvidenceImpact]


def _require_exact_evidence_ids(
    *, expected_atoms: list[EvidenceAtom], received_ids: list[str]
) -> None:
    expected_ids = [atom.evidence_id for atom in expected_atoms]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("输入 evidence_id 必须唯一")
    if len(received_ids) != len(set(received_ids)) or set(received_ids) != set(expected_ids):
        raise ValueError("每条新增证据必须且只能有一条影响判定")


def assess_incremental_evidence(
    *,
    client: LLMClient,
    atoms: Iterable[EvidenceAtom],
    conflict_ids: set[str],
    candidate_ids: set[str],
) -> list[EvidenceImpactRecord]:
    atom_list = list(atoms)
    response = client.generate_json(
        system_prompt=_build_system_prompt("evidence_assessment"),
        user_prompt=_serialize_prompt_input(
            {
                "mode": "INCREMENTAL",
                "evidence_atoms": [atom.model_dump(mode="json") for atom in atom_list],
                "conflict_ids": sorted(conflict_ids),
                "candidate_ids": sorted(candidate_ids),
            }
        ),
        response_model=_EvidenceAssessmentResult,
    )
    _require_exact_evidence_ids(
        expected_atoms=atom_list,
        received_ids=[proposal.evidence_id for proposal in response.proposals],
    )
    return build_evidence_impacts(
        atoms=atom_list,
        proposed_impacts={
            proposal.evidence_id: (proposal.impact, proposal.target_id)
            for proposal in response.proposals
        },
        conflict_ids=conflict_ids,
        candidate_ids=candidate_ids,
    )


def assess_holdout_evidence(
    *,
    client: LLMClient,
    atoms: Iterable[EvidenceAtom],
    frozen_candidate_ids: set[str],
) -> list[HoldoutEvidenceImpact]:
    atom_list = list(atoms)
    response = client.generate_json(
        system_prompt=_build_system_prompt("evidence_assessment"),
        user_prompt=_serialize_prompt_input(
            {
                "mode": "HOLDOUT",
                "evidence_atoms": [atom.model_dump(mode="json") for atom in atom_list],
                "frozen_candidate_ids": sorted(frozen_candidate_ids),
            }
        ),
        response_model=_HoldoutAssessmentResult,
    )
    _require_exact_evidence_ids(
        expected_atoms=atom_list,
        received_ids=[item.evidence_id for item in response.impacts],
    )
    if any(item.candidate_id not in frozen_candidate_ids for item in response.impacts):
        raise ValueError("HOLDOUT 影响只能绑定已冻结候选")
    return response.impacts
