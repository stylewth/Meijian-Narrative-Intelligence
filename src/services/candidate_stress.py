"""Deterministic candidate stress checks and cached robustness scenarios."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from src.llm_client import LLMClient
from src.schemas import (
    BrandFact,
    CandidateStressResult,
    EvidenceAtom,
    EvidenceCoverage,
    EvidenceGrade,
    EvidenceImpact,
    EvidenceRoute,
    HoldoutEvidenceImpact,
    NextBestEvidence,
    NarrativeCandidate,
    RobustnessCandidateOutcome,
    RobustnessScenario,
    RobustnessScenarioResult,
    StressCheckResult,
    StressCheckType,
    StressDecision,
    StressExecutionStatus,
    TemplateCheckResult,
)
from src.services import _build_system_prompt, _serialize_prompt_input
from src.services.grounding import evaluate_brand_grounding


def _independent_atoms(evidence: Iterable[EvidenceAtom]) -> list[EvidenceAtom]:
    selected: list[EvidenceAtom] = []
    seen_groups: set[str] = set()
    for atom in evidence:
        group = atom.duplicate_group or atom.evidence_id
        if group not in seen_groups:
            selected.append(atom)
            seen_groups.add(group)
    return selected


def calculate_evidence_coverage(*, evidence: list[EvidenceAtom]) -> EvidenceCoverage:
    """Count independent evidence, never treating duplicate groups as new support."""

    independent = _independent_atoms(evidence)
    grade_counts = Counter(atom.evidence_grade.value for atom in independent)
    non_c_brand_count = sum(
        atom.route is EvidenceRoute.BRAND and atom.evidence_grade is not EvidenceGrade.C
        for atom in independent
    )
    return EvidenceCoverage(
        independent_evidence_count=len(independent),
        platform_count=len({atom.source_platform for atom in independent if atom.source_platform}),
        scene_count=sum(atom.route is EvidenceRoute.SCENE for atom in independent),
        grade_counts={grade.value: grade_counts[grade.value] for grade in EvidenceGrade},
        independent_brand_evidence_count=non_c_brand_count,
        can_support_brand_candidate=non_c_brand_count >= 2,
        reference_ids=[atom.evidence_id for atom in independent],
    )


def _candidate_reference_ids(
    candidate: NarrativeCandidate, evidence: list[EvidenceAtom]
) -> list[str]:
    available = {atom.evidence_id for atom in evidence}
    ids = [quote.comment_id for quote in candidate.supporting_evidence if quote.comment_id in available]
    if ids:
        return list(dict.fromkeys(ids))
    return [atom.evidence_id for atom in evidence]


def _check_from_attack(
    *,
    client: LLMClient,
    candidate: NarrativeCandidate,
    check_type: StressCheckType,
    attack_atoms: list[EvidenceAtom],
    fallback_reference_ids: list[str],
) -> StressCheckResult:
    if not attack_atoms:
        return StressCheckResult(
            check_type=check_type,
            execution_status=StressExecutionStatus.COMPLETED,
            decision=StressDecision.REVISE,
            reference_ids=fallback_reference_ids,
            rationale="当前证据集中没有可核验的攻击证据，需补充后再判断。",
            revision_boundary="补充与该候选主张直接相关的原始反例或竞品证据。",
        )

    available_ids = [atom.evidence_id for atom in attack_atoms]
    result = client.generate_json(
        system_prompt=_build_system_prompt("candidate_stress"),
        user_prompt=_serialize_prompt_input(
            {
                "candidate": candidate.model_dump(mode="json"),
                "check_type": check_type.value,
                "attack_evidence": [atom.model_dump(mode="json") for atom in attack_atoms],
            }
        ),
        response_model=StressCheckResult,
    )
    if not isinstance(result, StressCheckResult):
        raise TypeError("语义攻击必须返回 StressCheckResult")
    if result.check_type is not check_type:
        raise ValueError("语义攻击返回了错误的压力检查类型")
    if not set(result.reference_ids).issubset(available_ids):
        raise ValueError("语义攻击引用了不存在的攻击证据")
    if result.execution_status is not StressExecutionStatus.COMPLETED:
        raise ValueError("已有攻击证据时语义攻击不得保持未执行")
    return result


def evaluate_robustness_scenarios(
    *,
    client: LLMClient,
    candidates: list[NarrativeCandidate],
    evidence: list[EvidenceAtom],
    scenarios: list[RobustnessScenario],
    holdout_evidence: list[EvidenceAtom] | None = None,
    holdout_impacts: list[HoldoutEvidenceImpact] | None = None,
) -> list[RobustnessScenarioResult]:
    """Precompute deterministic scenario results; ``client`` is intentionally unused."""

    del client
    actual_platforms = sorted({atom.source_platform for atom in evidence if atom.source_platform})
    baseline_outcomes = _rank_outcomes(candidates, evidence)
    baseline_ranks = {
        outcome.candidate_id: outcome.rank for outcome in baseline_outcomes
    }
    results: list[RobustnessScenarioResult] = []
    for scenario in scenarios:
        filtered: list[EvidenceAtom]
        impacts: list[HoldoutEvidenceImpact] = []
        applied_filter: str
        if scenario is RobustnessScenario.EXCLUDE_PLATFORM:
            if len(actual_platforms) < 2:
                results.append(
                    RobustnessScenarioResult(
                        scenario=scenario,
                        execution_status=StressExecutionStatus.PENDING,
                        decision=None,
                        applied_filter="实际平台不足两个，无法形成可比较扰动。",
                        reference_ids=[atom.evidence_id for atom in evidence],
                        rationale="排除平台场景至少需要两个实际平台。",
                    )
                )
                continue
            excluded = actual_platforms[0]
            filtered = [atom for atom in evidence if atom.source_platform != excluded]
            applied_filter = f"排除实际平台：{excluded}"
        elif scenario is RobustnessScenario.ACTUAL_USE_ONLY:
            if not any(atom.actual_use is not None for atom in evidence):
                results.append(
                    RobustnessScenarioResult(
                        scenario=scenario,
                        execution_status=StressExecutionStatus.PENDING,
                        decision=None,
                        applied_filter="仅保留 ACTUAL_USE=true",
                        reference_ids=[atom.evidence_id for atom in evidence],
                        rationale="未提供 ACTUAL_USE 标注，不能伪造筛选结果。",
                    )
                )
                continue
            filtered = [atom for atom in evidence if atom.actual_use is True]
            applied_filter = "仅保留 ACTUAL_USE=true"
        else:
            if not holdout_evidence:
                results.append(
                    RobustnessScenarioResult(
                        scenario=scenario,
                        execution_status=StressExecutionStatus.PENDING,
                        decision=None,
                        applied_filter="使用隔离 holdout 证据",
                        reference_ids=[atom.evidence_id for atom in evidence],
                        rationale="未提供 holdout 证据，不能运行留出集检验。",
                    )
                )
                continue
            canonical_holdout = _independent_atoms(holdout_evidence)
            holdout_ids = {atom.evidence_id for atom in canonical_holdout}
            provided_impacts = holdout_impacts or []
            canonical_impacts = [
                item for item in provided_impacts if item.evidence_id in holdout_ids
            ]
            mapped_ids = {item.evidence_id for item in canonical_impacts}
            if mapped_ids != holdout_ids:
                results.append(
                    RobustnessScenarioResult(
                        scenario=scenario,
                        execution_status=StressExecutionStatus.PENDING,
                        decision=None,
                        applied_filter="在基线证据上加入隔离 holdout 证据",
                        reference_ids=[
                            *[atom.evidence_id for atom in evidence],
                            *[atom.evidence_id for atom in canonical_holdout],
                        ],
                        rationale="holdout 证据缺少逐条候选 SUPPORT / CHALLENGE 映射，不能推断影响。",
                    )
                )
                continue
            if len(mapped_ids) != len(canonical_impacts):
                raise ValueError("同一 holdout evidence_id 只能有一条影响映射")
            candidate_ids = {candidate.candidate_id for candidate in candidates}
            if any(item.candidate_id not in candidate_ids for item in canonical_impacts):
                raise ValueError("HOLDOUT 影响映射引用了不存在的 candidate_id")
            filtered = [*evidence, *canonical_holdout]
            impacts = canonical_impacts
            applied_filter = "在基线证据上加入隔离 holdout 证据"
        canonical_filtered = _independent_atoms(filtered)
        canonical_ids = {atom.evidence_id for atom in canonical_filtered}
        effective_impacts = [
            impact for impact in impacts if impact.evidence_id in canonical_ids
        ]
        outcomes = _rank_outcomes(
            candidates,
            canonical_filtered,
            baseline_ranks=baseline_ranks,
            impacts=effective_impacts,
        )
        rank_changed = any(item.rank != item.baseline_rank for item in outcomes)
        has_challenge = any(
            item.impact is EvidenceImpact.CHALLENGE for item in effective_impacts
        )
        results.append(
            RobustnessScenarioResult(
                scenario=scenario,
                execution_status=StressExecutionStatus.COMPLETED,
                decision=(
                    StressDecision.PASS
                    if all(item.supporting_evidence_ids for item in outcomes)
                    and not rank_changed
                    and not has_challenge
                    else StressDecision.REVISE
                ),
                applied_filter=applied_filter,
                reference_ids=[atom.evidence_id for atom in canonical_filtered],
                candidate_outcomes=outcomes,
                rationale="结果由已缓存的证据过滤和候选支持关系确定性计算。",
            )
        )
    return results


def _rank_outcomes(
    candidates: list[NarrativeCandidate], evidence: list[EvidenceAtom], *, baseline_ranks: dict[str, int] | None = None,
    impacts: list[HoldoutEvidenceImpact] | None = None,
) -> list[RobustnessCandidateOutcome]:
    evidence_ids = {atom.evidence_id for atom in _independent_atoms(evidence)}
    impacts_by_candidate: dict[str, list[HoldoutEvidenceImpact]] = {}
    for impact in impacts or []:
        if impact.evidence_id in evidence_ids:
            impacts_by_candidate.setdefault(impact.candidate_id, []).append(impact)
    preliminary = [
        (
            index,
            candidate,
            list(dict.fromkeys([
                quote.comment_id
                for quote in candidate.supporting_evidence
                if quote.comment_id in evidence_ids
            ])),
            [
                impact.evidence_id
                for impact in impacts_by_candidate.get(candidate.candidate_id, [])
                if impact.impact is EvidenceImpact.SUPPORT
            ],
            [
                impact.evidence_id
                for impact in impacts_by_candidate.get(candidate.candidate_id, [])
                if impact.impact is EvidenceImpact.CHALLENGE
            ],
        )
        for index, candidate in enumerate(candidates)
    ]
    preliminary = [
        (index, candidate, list(dict.fromkeys(support_ids + extra_support_ids)), challenge_ids)
        for index, candidate, support_ids, extra_support_ids, challenge_ids in preliminary
    ]
    preliminary.sort(key=lambda item: (-(len(item[2]) - len(item[3])), item[0]))
    return [
        RobustnessCandidateOutcome(
            candidate_id=candidate.candidate_id,
            baseline_rank=(baseline_ranks or {}).get(candidate.candidate_id, rank),
            rank=rank,
            supporting_evidence_ids=support_ids,
            conflict_ids=candidate.supporting_conflict_ids,
            new_risks=(
                challenge_ids
                or ([] if support_ids else ["该场景中没有保留候选的支持证据。"])
            ),
        )
        for rank, (index, candidate, support_ids, challenge_ids) in enumerate(preliminary, start=1)
    ]


def run_candidate_stress(
    *,
    client: LLMClient,
    candidate: NarrativeCandidate,
    evidence: list[EvidenceAtom],
    brand_facts: list[BrandFact],
    robustness_results: list[RobustnessScenarioResult],
    template_check: TemplateCheckResult | None,
) -> CandidateStressResult:
    """Attack one candidate and return five separately auditable checks."""

    coverage = calculate_evidence_coverage(evidence=evidence)
    fallback_ids = _candidate_reference_ids(candidate, evidence)
    if not fallback_ids:
        raise ValueError("Candidate Stress 需要至少一条可引用的 EvidenceAtom")
    coverage_refs = coverage.reference_ids or fallback_ids
    coverage_check = StressCheckResult(
        check_type=StressCheckType.EVIDENCE_COVERAGE,
        execution_status=StressExecutionStatus.COMPLETED,
        decision=(StressDecision.PASS if coverage.can_support_brand_candidate else StressDecision.REVISE),
        reference_ids=coverage_refs,
        rationale=(
            "存在至少两条独立、非 C 级 BRAND 证据。"
            if coverage.can_support_brand_candidate
            else "独立非 C 级 BRAND 证据不足两条，不能单独满足品牌候选门槛。"
        ),
        revision_boundary=(None if coverage.can_support_brand_candidate else "补充独立的非 C 级 BRAND 证据。"),
    )
    counter_ids = {quote.comment_id for quote in candidate.counter_evidence}
    counter_check = _check_from_attack(
        client=client,
        candidate=candidate,
        check_type=StressCheckType.COUNTER_EVIDENCE,
        attack_atoms=[atom for atom in evidence if atom.evidence_id in counter_ids],
        fallback_reference_ids=fallback_ids,
    )
    competitor_check = _check_from_attack(
        client=client,
        candidate=candidate,
        check_type=StressCheckType.COMPETITOR_SUBSTITUTION,
        attack_atoms=[atom for atom in evidence if atom.route is EvidenceRoute.COMPETITOR],
        fallback_reference_ids=fallback_ids,
    )
    grounding = evaluate_brand_grounding(brand_facts=brand_facts)
    grounding_refs = grounding.fact_ids or grounding.source_ids or fallback_ids
    grounding_check = StressCheckResult(
        check_type=StressCheckType.PRODUCT_GROUNDING,
        execution_status=grounding.execution_status,
        decision=grounding.decision,
        reference_ids=grounding_refs,
        rationale=grounding.rationale,
        revision_boundary=(
            "补齐人工核验的 BrandFact 类别。"
            if grounding.decision in {StressDecision.REVISE, StressDecision.BLOCK}
            else None
        ),
    )
    pending_robustness = [
        item for item in robustness_results if item.execution_status is not StressExecutionStatus.COMPLETED
    ]
    robustness_refs = list(
        dict.fromkeys(reference for item in robustness_results for reference in item.reference_ids)
    ) or fallback_ids
    robustness_check = StressCheckResult(
        check_type=StressCheckType.ROBUSTNESS,
        execution_status=(
            StressExecutionStatus.PENDING if pending_robustness else StressExecutionStatus.COMPLETED
        ),
        decision=(
            None
            if pending_robustness
            else (
                StressDecision.PASS
                if robustness_results and all(item.decision is StressDecision.PASS for item in robustness_results)
                else StressDecision.REVISE
            )
        ),
        reference_ids=robustness_refs,
        rationale=(
            "至少一类样本扰动缺少真实数据，鲁棒性暂不下结论。"
            if pending_robustness
            else "鲁棒性结论仅来自预计算缓存的过滤场景。"
        ),
        revision_boundary=(
            "补齐缺失平台、ACTUAL_USE 或 holdout 数据。" if pending_robustness else None
        ),
    )
    del template_check
    return CandidateStressResult(
        candidate_id=candidate.candidate_id,
        checks=[
            coverage_check,
            counter_check,
            competitor_check,
            grounding_check,
            robustness_check,
        ],
    )


def derive_next_best_evidence(
    *, candidate: NarrativeCandidate, stress_result: CandidateStressResult
) -> NextBestEvidence:
    """Describe the largest unresolved check without inventing a numerical benefit."""

    priority = {
        StressCheckType.PRODUCT_GROUNDING: 0,
        StressCheckType.ROBUSTNESS: 1,
        StressCheckType.COUNTER_EVIDENCE: 2,
        StressCheckType.COMPETITOR_SUBSTITUTION: 3,
        StressCheckType.EVIDENCE_COVERAGE: 4,
    }
    unresolved = [
        check
        for check in stress_result.checks
        if check.execution_status is not StressExecutionStatus.COMPLETED
        or check.decision in {StressDecision.REVISE, StressDecision.BLOCK}
    ]
    if not unresolved:
        unresolved = stress_result.checks
    selected = min(unresolved, key=lambda check: priority[check.check_type])
    return NextBestEvidence(
        target_audience=candidate.target_audience,
        scenario=selected.check_type.value,
        suggested_source="人工核验的原始评论、品牌资料或隔离留出集",
        core_question=selected.rationale,
        reference_ids=selected.reference_ids,
        possible_decision_change="该信息可用于确认、修订或阻止当前候选，不预先承诺方向。",
    )
