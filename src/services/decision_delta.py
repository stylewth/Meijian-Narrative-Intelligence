"""Pure decision-state derivation for the Market Decision Center."""

from __future__ import annotations

from src.schemas import (
    BusinessDecisionStatus,
    CandidateStressResult,
    DecisionCheckpoint,
    DecisionDelta,
    StressDecision,
    StressExecutionStatus,
)


def derive_business_status(stress_result: CandidateStressResult) -> BusinessDecisionStatus:
    """Map all five internal checks to one deliberately small business status."""

    checks = stress_result.checks
    if any(check.execution_status is StressExecutionStatus.PENDING for check in checks):
        return BusinessDecisionStatus.ANALYSIS_IN_PROGRESS
    if any(check.execution_status is StressExecutionStatus.ERROR for check in checks):
        return BusinessDecisionStatus.NEEDS_REVISION
    decisions = {check.decision for check in checks}
    if StressDecision.BLOCK in decisions:
        return BusinessDecisionStatus.BLOCKED
    if StressDecision.REVISE in decisions:
        return BusinessDecisionStatus.NEEDS_REVISION
    return BusinessDecisionStatus.READY_FOR_VALIDATION


def _added(current: list[str], previous: list[str]) -> list[str]:
    previous_ids = set(previous)
    return [item for item in current if item not in previous_ids]


def build_decision_delta(
    previous: DecisionCheckpoint | None,
    current: DecisionCheckpoint,
) -> DecisionDelta | None:
    """Compare adjacent real checkpoints; the first checkpoint has no made-up delta."""

    if previous is None:
        return None
    ranks_present = previous.candidate_rank is not None and current.candidate_rank is not None
    scores_present = previous.weighted_score is not None and current.weighted_score is not None
    return DecisionDelta(
        previous_checkpoint_id=previous.checkpoint_id,
        checkpoint_id=current.checkpoint_id,
        trigger=current.trigger,
        versions=current.versions,
        added_evidence_ids=_added(current.evidence_ids, previous.evidence_ids),
        added_fact_ids=_added(current.fact_ids, previous.fact_ids),
        added_supporting_evidence_ids=_added(current.supporting_evidence_ids, previous.supporting_evidence_ids),
        added_counter_evidence_ids=_added(current.counter_evidence_ids, previous.counter_evidence_ids),
        previous_rank=previous.candidate_rank,
        current_rank=current.candidate_rank,
        rank_change=(current.candidate_rank - previous.candidate_rank) if ranks_present else None,
        previous_weighted_score=previous.weighted_score,
        current_weighted_score=current.weighted_score,
        weighted_score_change=round(current.weighted_score - previous.weighted_score, 10) if scores_present else None,
        previous_status=previous.business_status,
        current_status=current.business_status,
        robustness_changed=previous.robustness_summary != current.robustness_summary,
        added_risks=_added(current.primary_risks, previous.primary_risks),
        removed_risks=_added(previous.primary_risks, current.primary_risks),
    )
