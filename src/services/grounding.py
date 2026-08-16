"""Ground candidate claims only in manually verified BrandFact records."""

from __future__ import annotations

from src.schemas import (
    BrandFact,
    GroundingCategory,
    GroundingFactCoverage,
    GroundingResult,
    StressDecision,
    StressExecutionStatus,
)


def evaluate_brand_grounding(*, brand_facts: list[BrandFact]) -> GroundingResult:
    """Check five required delivery areas without adding product common knowledge.

    This deliberately has one input. Future retrieval may produce ``BrandFact[]``, but
    retrieval itself is not a dependency of the stress lab.
    """

    if not brand_facts:
        return GroundingResult(
            execution_status=StressExecutionStatus.PENDING,
            decision=None,
            rationale="尚未提供人工核验的 BrandFact，不能判断产品落地。",
        )

    facts_by_category: dict[GroundingCategory, list[BrandFact]] = {
        category: [] for category in GroundingCategory
    }
    for fact in brand_facts:
        facts_by_category[fact.fact_type].append(fact)

    missing = [
        category.value for category, facts in facts_by_category.items() if not facts
    ]
    fact_ids = [fact.fact_id for fact in brand_facts]
    source_ids = [fact.source.source_id for fact in brand_facts]
    if missing:
        return GroundingResult(
            execution_status=StressExecutionStatus.COMPLETED,
            decision=StressDecision.BLOCK,
            fact_ids=fact_ids,
            source_ids=source_ids,
            missing_categories=missing,
            rationale="缺少完整的产品落地类别，不能以常识补全。",
        )

    partial_facts = [
        fact
        for fact in brand_facts
        if fact.grounding_coverage is GroundingFactCoverage.PARTIAL
    ]
    if partial_facts:
        return GroundingResult(
            execution_status=StressExecutionStatus.COMPLETED,
            decision=StressDecision.REVISE,
            fact_ids=fact_ids,
            source_ids=source_ids,
            rationale="五类事实均已出现，但部分 BrandFact 仍是局部核验，需补齐后再判断。",
        )

    return GroundingResult(
        execution_status=StressExecutionStatus.COMPLETED,
        decision=StressDecision.PASS,
        fact_ids=fact_ids,
        source_ids=source_ids,
        rationale="产品、包装、喝法、销售场景和内容动作均有人工核验事实。",
    )
