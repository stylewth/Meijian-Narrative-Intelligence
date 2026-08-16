from __future__ import annotations

from src.llm_client import LLMClient
from src.schemas import CorpusAnalysisResult, DiversityAssessment
from src.services import _build_system_prompt, _serialize_prompt_input
from src.services.corpus_analysis import _collect_evidence


def assess_diversity(
    *,
    client: LLMClient,
    corpus_analysis: CorpusAnalysisResult,
) -> DiversityAssessment:
    seen_evidence: set[tuple[str, str]] = set()
    evidence_summary: list[dict[str, str]] = []
    for evidence in _collect_evidence(corpus_analysis):
        key = (evidence.comment_id, evidence.quote)
        if key not in seen_evidence:
            seen_evidence.add(key)
            evidence_summary.append(evidence.model_dump(mode="json"))

    result = client.generate_json(
        system_prompt=_build_system_prompt("diversity_assessment"),
        user_prompt=_serialize_prompt_input(
            {
                "corpus_analysis": corpus_analysis.model_dump(mode="json"),
                "evidence_summary": evidence_summary,
            }
        ),
        response_model=DiversityAssessment,
    )

    conflict_count = len(
        {conflict.conflict_id for conflict in corpus_analysis.emotional_conflicts}
    )
    if result.distinct_conflict_count > conflict_count:
        raise ValueError("独立冲突数不得超过全量分析中的冲突数")
    return result
