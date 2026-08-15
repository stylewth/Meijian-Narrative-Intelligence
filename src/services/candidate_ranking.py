"""候选五维加权、可信校验、排序和人工决策。"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from typing import Callable, TypeVar

from pydantic import Field

from src.evidence import validate_evidence_quotes
from src.llm_client import LLMClient
from src.schemas import (
    CandidateEvaluation,
    CandidateScores,
    CandidateStressResult,
    CommentRecord,
    EmotionalConflict,
    NarrativeCandidate,
    NarrativeDecision,
    RankedNarrative,
    RankingResult,
    StrictBaseModel,
    StressDecision,
    StressExecutionStatus,
)
from src.services import _build_system_prompt, _serialize_prompt_input


SCORE_WEIGHTS: dict[str, float] = {
    "evidence_strength": 0.25,
    "emotional_tension": 0.20,
    "meijian_fit_and_exclusivity": 0.25,
    "competitor_difference": 0.15,
    "scene_conversion": 0.15,
}

T = TypeVar("T")


class _CandidateScoringResult(StrictBaseModel):
    evaluations: list[CandidateEvaluation]
    recommendation_reason: str = Field(min_length=1)


def calculate_weighted_score(scores: CandidateScores) -> float:
    return round(
        scores.evidence_strength.score * SCORE_WEIGHTS["evidence_strength"]
        + scores.emotional_tension.score * SCORE_WEIGHTS["emotional_tension"]
        + scores.meijian_fit_and_exclusivity.score
        * SCORE_WEIGHTS["meijian_fit_and_exclusivity"]
        + scores.competitor_difference.score
        * SCORE_WEIGHTS["competitor_difference"]
        + scores.scene_conversion.score * SCORE_WEIGHTS["scene_conversion"],
        2,
    )


def select_candidate_for_validation(
    ranking: RankingResult,
    stress_results: list[CandidateStressResult],
) -> NarrativeCandidate | None:
    """Return the highest soft-ranked candidate whose five checks all passed."""

    stress_by_candidate: dict[str, CandidateStressResult] = {}
    for result in stress_results:
        if result.candidate_id in stress_by_candidate:
            raise ValueError("每个候选只能提供一份当前压力测试结果")
        stress_by_candidate[result.candidate_id] = result

    ranked_ids = {
        item.candidate.candidate_id for item in ranking.ranked_candidates
    }
    if not set(stress_by_candidate).issubset(ranked_ids):
        raise ValueError("压力测试包含不在当前软排名中的候选")

    for item in sorted(ranking.ranked_candidates, key=lambda ranked: ranked.rank):
        stress = stress_by_candidate.get(item.candidate.candidate_id)
        if stress is None:
            continue
        if all(
            check.execution_status is StressExecutionStatus.COMPLETED
            and check.decision is StressDecision.PASS
            for check in stress.checks
        ):
            return item.candidate
    return None


def evaluate_and_rank_candidates(
    *,
    client: LLMClient,
    candidates: list[NarrativeCandidate],
    comments: list[CommentRecord],
    conflicts: list[EmotionalConflict],
    no_candidate_reason: str | None = None,
) -> RankingResult:
    if not candidates:
        if not (no_candidate_reason or "").strip():
            raise ValueError("零候选时必须提供 no_candidate_reason")
        return rank_candidates(
            candidates=[],
            evaluations=[],
            comments=comments,
            conflicts=conflicts,
            recommendation_reason=no_candidate_reason,
        )

    response = client.generate_json(
        system_prompt=_build_system_prompt("candidate_scoring"),
        user_prompt=_serialize_prompt_input(
            {
                "candidates": [
                    candidate.model_dump(mode="json") for candidate in candidates
                ],
                "emotional_conflicts": [
                    conflict.model_dump(mode="json") for conflict in conflicts
                ],
                "original_evidence": [
                    {
                        "comment_id": comment.comment_id,
                        "raw_content": comment.raw_content,
                    }
                    for comment in comments
                ],
            }
        ),
        response_model=_CandidateScoringResult,
    )
    return rank_candidates(
        candidates=candidates,
        evaluations=response.evaluations,
        comments=comments,
        conflicts=conflicts,
        recommendation_reason=response.recommendation_reason,
    )


def rank_candidates(
    *,
    candidates: list[NarrativeCandidate],
    evaluations: list[CandidateEvaluation],
    comments: list[CommentRecord],
    conflicts: list[EmotionalConflict],
    recommendation_reason: str,
) -> RankingResult:
    evaluation_index = _evaluation_index_for_candidates(candidates, evaluations)
    conflict_index = _unique_index(
        conflicts, key=lambda item: item.conflict_id, label="conflict_id"
    )
    _validate_conflicts(conflicts, comments)
    _validate_candidates(candidates, comments, set(conflict_index))
    return _assemble_ranking_result(
        candidates=candidates,
        evaluation_index=evaluation_index,
        recommendation_reason=recommendation_reason,
    )


def assemble_ranking_result(
    *,
    candidates: list[NarrativeCandidate],
    evaluations: list[CandidateEvaluation],
    recommendation_reason: str,
) -> RankingResult:
    evaluation_index = _evaluation_index_for_candidates(candidates, evaluations)
    return _assemble_ranking_result(
        candidates=candidates,
        evaluation_index=evaluation_index,
        recommendation_reason=recommendation_reason,
    )


def _evaluation_index_for_candidates(
    candidates: list[NarrativeCandidate], evaluations: list[CandidateEvaluation]
) -> dict[str, CandidateEvaluation]:
    candidate_index = _unique_index(
        candidates, key=lambda item: item.candidate_id, label="candidate_id"
    )
    evaluation_index = _unique_index(
        evaluations, key=lambda item: item.candidate_id, label="评价 candidate_id"
    )
    if set(candidate_index) != set(evaluation_index):
        raise ValueError("候选与评价 candidate_id 集合必须完全一致")
    return evaluation_index


def _assemble_ranking_result(
    *,
    candidates: list[NarrativeCandidate],
    evaluation_index: dict[str, CandidateEvaluation],
    recommendation_reason: str,
) -> RankingResult:
    sortable: list[tuple[int, NarrativeCandidate, CandidateEvaluation, float]] = []
    for generation_order, candidate in enumerate(candidates):
        evaluation = evaluation_index[candidate.candidate_id]
        sortable.append(
            (
                generation_order,
                candidate,
                evaluation,
                calculate_weighted_score(evaluation.scores),
            )
        )
    sortable.sort(key=_ranking_key)

    ranked = [
        RankedNarrative(
            candidate=candidate,
            evaluation=evaluation,
            weighted_score=weighted_score,
            rank=rank,
            is_recommended=rank == 1,
        )
        for rank, (_, candidate, evaluation, weighted_score) in enumerate(
            sortable, start=1
        )
    ]
    recommended_id = ranked[0].candidate.candidate_id if ranked else None
    return RankingResult(
        ranked_candidates=ranked,
        recommended_candidate_id=recommended_id,
        recommendation_reason=recommendation_reason,
    )


def create_narrative_decision(
    *,
    ranking: RankingResult,
    selected_candidate_id: str,
    edited_brand_role: str,
    edited_proposition: str,
    edited_main_scenes: list[str],
) -> NarrativeDecision:
    if ranking.recommended_candidate_id is None or not ranking.ranked_candidates:
        raise ValueError("无候选或无 AI 推荐时不能创建 NarrativeDecision")
    candidates = {
        item.candidate.candidate_id: item.candidate for item in ranking.ranked_candidates
    }
    selected = candidates.get(selected_candidate_id)
    if selected is None:
        raise ValueError("selected_candidate_id 不在当前排名候选中")
    return NarrativeDecision(
        recommended_candidate_id=ranking.recommended_candidate_id,
        selected_candidate_id=selected_candidate_id,
        selection_changed_by_user=selected_candidate_id
        != ranking.recommended_candidate_id,
        original_brand_role=selected.brand_role,
        original_proposition=selected.draft_proposition,
        original_main_scenes=list(selected.main_scenes),
        edited_brand_role=edited_brand_role,
        edited_proposition=edited_proposition,
        edited_main_scenes=list(edited_main_scenes),
    )


def _unique_index(
    items: Iterable[T], *, key: Callable[[T], str], label: str
) -> dict[str, T]:
    index: dict[str, T] = {}
    for item in items:
        value = key(item)
        if value in index:
            raise ValueError(f"{label} 必须唯一: {value}")
        index[value] = item
    return index


def _validate_conflicts(
    conflicts: list[EmotionalConflict], comments: list[CommentRecord]
) -> None:
    for conflict in conflicts:
        validate_evidence_quotes(conflict.supporting_evidence, comments)
        validate_evidence_quotes(conflict.counter_evidence, comments)


def _validate_candidates(
    candidates: list[NarrativeCandidate],
    comments: list[CommentRecord],
    conflict_ids: set[str],
) -> None:
    primary_ids: set[str] = set()
    normalized_candidates: set[tuple[str, str]] = set()
    for candidate in candidates:
        unknown_ids = set(candidate.supporting_conflict_ids) - conflict_ids
        if candidate.primary_conflict_id not in conflict_ids or unknown_ids:
            invalid = sorted(unknown_ids | {candidate.primary_conflict_id} - conflict_ids)
            raise ValueError(f"候选引用不存在的冲突 ID: {invalid}")
        if candidate.primary_conflict_id in primary_ids:
            raise ValueError("不同候选的 primary_conflict_id 不得重复")
        primary_ids.add(candidate.primary_conflict_id)

        normalized = (
            _normalize_candidate_text(candidate.user_conflict),
            _normalize_candidate_text(candidate.brand_opportunity),
        )
        if normalized in normalized_candidates:
            raise ValueError("存在用户冲突和品牌机会相同的规范化重复候选")
        normalized_candidates.add(normalized)

        validate_evidence_quotes(candidate.supporting_evidence, comments)
        validate_evidence_quotes(candidate.counter_evidence, comments)


def _normalize_candidate_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z"))
        and not character.isspace()
    )


def _ranking_key(
    item: tuple[int, NarrativeCandidate, CandidateEvaluation, float],
) -> tuple[float, float, float, float, int]:
    generation_order, _, evaluation, weighted_score = item
    scores = evaluation.scores
    return (
        -weighted_score,
        -scores.evidence_strength.score,
        -scores.meijian_fit_and_exclusivity.score,
        -scores.emotional_tension.score,
        generation_order,
    )
