from __future__ import annotations

import unicodedata
from collections.abc import Iterable

from pydantic import AliasChoices, Field

from src.evidence import validate_evidence_quotes
from src.llm_client import LLMClient
from src.schemas import (
    CommentRecord,
    CorpusAnalysisResult,
    DiversityAssessment,
    EvidenceQuote,
    NarrativeCandidate,
    StrictBaseModel,
)
from src.services import _build_system_prompt, _serialize_prompt_input


class _CandidateDraft(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    supporting_conflict_ids: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    target_audience: str = Field(min_length=1)
    user_conflict: str = Field(min_length=1)
    brand_opportunity: str = Field(min_length=1)
    why_brand: str = Field(
        min_length=1,
        validation_alias=AliasChoices("why_brand", "why_meijian"),
    )
    competitor_difference: str = Field(min_length=1)
    brand_role: str = Field(min_length=1)
    draft_proposition: str = Field(min_length=1)
    main_scenes: list[str] = Field(min_length=1)
    content_theme: str = Field(min_length=1)
    supporting_evidence: list[EvidenceQuote] = Field(min_length=1)
    counter_evidence: list[EvidenceQuote] = Field(default_factory=list)
    counter_evidence_note: str = Field(min_length=1, pattern=r".*\S.*")
    risks: list[str] = Field(min_length=1)

    def to_candidate(self) -> NarrativeCandidate:
        return NarrativeCandidate(
            primary_conflict_id=self.supporting_conflict_ids[0],
            **self.model_dump(),
        )


class _CandidateGenerationResult(StrictBaseModel):
    candidates: list[_CandidateDraft]


def _normalize_candidate_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if not unicodedata.category(character).startswith(("P", "Z"))
        and not character.isspace()
    )


def generate_candidates(
    *,
    client: LLMClient,
    corpus_analysis: CorpusAnalysisResult,
    diversity_assessment: DiversityAssessment,
    comments: Iterable[CommentRecord],
) -> list[NarrativeCandidate]:
    comment_list = list(comments)
    response = client.generate_json(
        system_prompt=_build_system_prompt("candidate_generation"),
        user_prompt=_serialize_prompt_input(
            {
                "emotional_conflicts": [
                    conflict.model_dump(mode="json")
                    for conflict in corpus_analysis.emotional_conflicts
                ],
                "original_evidence": [
                    {
                        "comment_id": comment.comment_id,
                        "sample_type": comment.sample_type.value,
                        "raw_content": comment.raw_content,
                    }
                    for comment in comment_list
                ],
                "diversity_assessment": diversity_assessment.model_dump(mode="json"),
            }
        ),
        response_model=_CandidateGenerationResult,
        evidence_catalog={
            comment.comment_id: comment.raw_content for comment in comment_list
        },
    )
    candidates = [draft.to_candidate() for draft in response.candidates]

    if len(candidates) != diversity_assessment.recommended_candidate_count:
        raise ValueError("候选数量必须与 DiversityAssessment 的建议数量一致")

    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate_id 重复")

    conflict_ids = {
        conflict.conflict_id for conflict in corpus_analysis.emotional_conflicts
    }
    primary_conflict_ids: set[str] = set()
    candidate_signatures: set[tuple[str, str]] = set()
    for candidate in candidates:
        if not set(candidate.supporting_conflict_ids).issubset(conflict_ids):
            raise ValueError("候选引用的冲突 ID 不存在")
        if candidate.primary_conflict_id in primary_conflict_ids:
            raise ValueError("候选主冲突 ID 重复")
        primary_conflict_ids.add(candidate.primary_conflict_id)

        signature = (
            _normalize_candidate_text(candidate.user_conflict),
            _normalize_candidate_text(candidate.brand_opportunity),
        )
        if signature in candidate_signatures:
            raise ValueError("用户冲突和品牌机会规范化后形成重复候选")
        candidate_signatures.add(signature)

        validate_evidence_quotes(
            [*candidate.supporting_evidence, *candidate.counter_evidence],
            comment_list,
        )
    return candidates
