"""Blind-test-only reassessment of the frozen admitted narratives."""

from __future__ import annotations

from src.llm_client import LLMClient
from src.schemas import (
    BlindEvidencePackage,
    BlindReassessmentResult,
    DecisionOriginalSnapshot,
)
from src.services import _build_system_prompt, _serialize_prompt_input
from src.services.candidate_ranking import calculate_weighted_score
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes


def _require_exact_candidate_ids(
    *, expected_ids: list[str], received_ids: list[str]
) -> None:
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("original selected candidate IDs must be unique")
    if len(received_ids) != len(set(received_ids)) or set(received_ids) != set(expected_ids):
        raise ValueError("blind reassessment candidate IDs must exactly match selected candidates")


def reassess_from_blind_evidence(
    *,
    client: LLMClient,
    original: DecisionOriginalSnapshot,
    blind: BlindEvidencePackage,
) -> BlindReassessmentResult:
    """Re-score selected candidates from the confirmed blind manuscript only."""

    if blind.run_id != original.run_id:
        raise ValueError("blind evidence run_id does not match original snapshot")
    original_snapshot_sha256 = sha256_bytes(canonical_json_bytes(original))
    if blind.original_snapshot_sha256 != original_snapshot_sha256:
        raise ValueError("blind evidence is not bound to the original snapshot")
    blind_evidence_sha256 = sha256_bytes(canonical_json_bytes(blind))

    selected_ids = [
        snapshot.ranked_narrative.candidate.candidate_id
        for snapshot in original.candidates
    ]
    _require_exact_candidate_ids(
        expected_ids=selected_ids,
        received_ids=[item.candidate_id for item in blind.items],
    )

    response = client.generate_json(
        system_prompt=_build_system_prompt("blind_reassessment"),
        user_prompt=_serialize_prompt_input(
            {
                "run_id": original.run_id,
                "original_snapshot_sha256": original_snapshot_sha256,
                "blind_evidence_sha256": blind_evidence_sha256,
                "selected_candidate_ids": selected_ids,
                "confirmed_blind_manuscripts": [
                    {
                        "candidate_id": item.candidate_id,
                        "content": item.content,
                    }
                    for item in blind.items
                ],
            }
        ),
        response_model=BlindReassessmentResult,
    )
    if not isinstance(response, BlindReassessmentResult):
        raise TypeError("blind reassessment client must return BlindReassessmentResult")
    if response.run_id != original.run_id:
        raise ValueError("blind reassessment run_id does not match original snapshot")
    if response.original_snapshot_sha256 != original_snapshot_sha256:
        raise ValueError("blind reassessment is not bound to the original snapshot")
    if response.blind_evidence_sha256 != blind_evidence_sha256:
        raise ValueError("blind evidence SHA does not match frozen package")

    _require_exact_candidate_ids(
        expected_ids=selected_ids,
        received_ids=[item.candidate_id for item in response.candidates],
    )
    ranks = [item.rank for item in response.candidates]
    if sorted(ranks) != list(range(1, len(response.candidates) + 1)):
        raise ValueError("blind reassessment rank must be contiguous 1..N")
    for item in response.candidates:
        expected_score = calculate_weighted_score(item.evaluation.scores)
        if item.weighted_score != expected_score:
            raise ValueError(
                f"weighted_score does not match five-dimension contract for {item.candidate_id}"
            )
    return BlindReassessmentResult.model_validate(
        response.model_dump(), strict=True
    )
