from __future__ import annotations

from copy import deepcopy
from typing import Any, MutableMapping

from src.schemas import (
    AlternativeConsidered,
    BrandFinalDecision,
    CandidateAttemptRecord,
    DecisionCheckpoint,
    DecisionTimelineEvent,
    HumanValidationRecord,
    StressAttemptRecord,
    StressDecision,
)


SESSION_DEFAULTS: dict[str, Any] = {
    "system_entry": None,
    "app_mode": None,
    "active_workspace": "数据预处理",
    "completed_workspaces": [],
    "preprocessing_replay": False,
    "stress_test": False,
    "evolution_playback_key": None,
    "dataset_hash": None,
    "decision_hash": None,
    "comments": [],
    "data_health": None,
    "evidence_atoms": [],
    "evidence_impacts": [],
    "brand_facts": [],
    "robustness_results": [],
    "next_best_evidence": [],
    "human_annotations": [],
    "selected_comment_id": None,
    "single_comment_analyses": {},
    "corpus_analysis": None,
    "diversity_assessment": None,
    "narrative_candidates": [],
    "ranking_result": None,
    "selected_candidate_id": None,
    "narrative_decision": None,
    "template_check": None,
    "final_report": None,
    "candidate_attempts": [],
    "stress_attempts": [],
    "current_generation_id": 0,
    "current_stress_attempt_ids": [],
    "validation_records": [],
    "alternatives_considered": [],
    "brand_final_decisions": [],
    "active_validation_cards": [],
    "active_validation_batch": None,
    "validation_summary": [],
    "brand_final_decision": None,
    "decision_checkpoints": [],
    "decision_delta": None,
    "decision_timeline": [],
    "feishu_latest_action": None,
    "feishu_evolution_view": None,
    "feishu_evolution_error": None,
    "decision_workspace_entry": "自定义使用",
    "evolution_checkpoint_index": 0,
    "evolution_candidate_id": None,
    "evolution_accept_switch": False,
}

_DATASET_DEPENDENT_KEYS = (
    "data_health",
    "evidence_atoms",
    "evidence_impacts",
    "brand_facts",
    "robustness_results",
    "next_best_evidence",
    "decision_hash",
    "selected_comment_id",
    "single_comment_analyses",
    "corpus_analysis",
    "diversity_assessment",
    "narrative_candidates",
    "ranking_result",
    "selected_candidate_id",
    "narrative_decision",
    "template_check",
    "final_report",
)
_CORPUS_DEPENDENT_KEYS = (
    "decision_hash",
    "diversity_assessment",
    "narrative_candidates",
    "ranking_result",
    "selected_candidate_id",
    "narrative_decision",
    "template_check",
    "final_report",
)
_CANDIDATE_DEPENDENT_KEYS = (
    "decision_hash",
    "ranking_result",
    "selected_candidate_id",
    "narrative_decision",
    "template_check",
    "final_report",
    "robustness_results",
    "next_best_evidence",
    "active_validation_cards",
    "active_validation_batch",
    "validation_summary",
    "brand_final_decision",
    "decision_checkpoints",
    "decision_delta",
    "decision_timeline",
    "current_stress_attempt_ids",
    "current_generation_id",
)


def initialize_state(state: MutableMapping[str, Any]) -> None:
    for key, default in SESSION_DEFAULTS.items():
        if key not in state:
            state[key] = deepcopy(default)


def _reset_keys(state: MutableMapping[str, Any], keys: tuple[str, ...]) -> None:
    initialize_state(state)
    for key in keys:
        state[key] = deepcopy(SESSION_DEFAULTS[key])


def invalidate_for_dataset_change(state: MutableMapping[str, Any]) -> None:
    _reset_keys(state, _DATASET_DEPENDENT_KEYS)


def invalidate_after_corpus_analysis(state: MutableMapping[str, Any]) -> None:
    _reset_keys(state, _CORPUS_DEPENDENT_KEYS)


def invalidate_after_candidate_generation(state: MutableMapping[str, Any]) -> None:
    initialize_state(state)
    previous_generation_id = state["current_generation_id"]
    if not isinstance(previous_generation_id, int) or isinstance(previous_generation_id, bool):
        raise ValueError("current_generation_id 必须为非负整数")
    if previous_generation_id < 0:
        raise ValueError("current_generation_id 必须为非负整数")
    _reset_keys(state, _CANDIDATE_DEPENDENT_KEYS)
    state["current_generation_id"] = previous_generation_id + 1


def invalidate_after_decision_change(
    state: MutableMapping[str, Any], *, decision_hash: str
) -> None:
    initialize_state(state)
    state["decision_hash"] = decision_hash
    _reset_keys(state, ("template_check", "final_report"))


def invalidate_after_template_check(state: MutableMapping[str, Any]) -> None:
    _reset_keys(state, ("final_report",))


def invalidate_validation_for_dataset_change(state: MutableMapping[str, Any]) -> None:
    _reset_keys(
        state,
        ("active_validation_cards", "active_validation_batch", "validation_summary", "brand_final_decision"),
    )


def invalidate_validation_for_candidate_change(state: MutableMapping[str, Any]) -> None:
    _reset_keys(
        state,
        ("active_validation_cards", "active_validation_batch", "validation_summary", "brand_final_decision"),
    )


def invalidate_validation_for_feedback_change(state: MutableMapping[str, Any]) -> None:
    _reset_keys(state, ("validation_summary", "brand_final_decision"))


def append_candidate_attempt(
    state: MutableMapping[str, Any], record: CandidateAttemptRecord
) -> None:
    initialize_state(state)
    state["candidate_attempts"].append(record)


def current_stress_attempts(state: MutableMapping[str, Any]) -> list[StressAttemptRecord]:
    """Expose only stress attempts created by the active candidate generation."""

    initialize_state(state)
    current_ids = set(state["current_stress_attempt_ids"])
    return [
        attempt
        for attempt in state["stress_attempts"]
        if attempt.attempt_id in current_ids
    ]


def append_stress_attempt(
    state: MutableMapping[str, Any], record: StressAttemptRecord
) -> None:
    initialize_state(state)
    attempts: list[StressAttemptRecord] = state["stress_attempts"]
    if any(item.attempt_id == record.attempt_id for item in attempts):
        raise ValueError("stress attempt_id 必须唯一")
    prior = [item for item in attempts if item.candidate_id == record.candidate_id]
    if prior:
        previous = max(prior, key=lambda item: item.attempt_number)
        if record.attempt_number != previous.attempt_number + 1:
            raise ValueError("压力测试 attempt_number 必须连续追加")
        if previous.next_candidate_version is not None and (
            record.candidate_version != previous.next_candidate_version
        ):
            raise ValueError("REVISE 后必须使用记录的新版本重测")
    attempts.append(record)
    if record.generation_id == state["current_generation_id"]:
        state["current_stress_attempt_ids"].append(record.attempt_id)

    unresolved = [
        check
        for check in record.stress_result.checks
        if check.decision in {StressDecision.REVISE, StressDecision.BLOCK}
    ]
    if unresolved or record.human_edits:
        outcome = (
            StressDecision.BLOCK
            if any(check.decision is StressDecision.BLOCK for check in unresolved)
            else (
                StressDecision.REVISE
                if unresolved
                else StressDecision.PASS
            )
        )
        state["alternatives_considered"].append(
            AlternativeConsidered(
                candidate_id=record.candidate_id,
                candidate_version=record.candidate_version,
                stress_attempt_id=record.attempt_id,
                outcome=outcome,
                reasons=[check.rationale for check in unresolved]
                or ["候选经过人工修改后保留原版本记录。"],
                human_edits=list(record.human_edits),
                reference_ids=list(
                    dict.fromkeys(
                        reference
                        for check in record.stress_result.checks
                        for reference in check.reference_ids
                    )
                ),
            )
        )


def append_validation_record(
    state: MutableMapping[str, Any], record: HumanValidationRecord
) -> None:
    initialize_state(state)
    state["validation_records"].append(record)
    invalidate_validation_for_feedback_change(state)


def append_brand_final_decision(
    state: MutableMapping[str, Any], decision: BrandFinalDecision
) -> None:
    initialize_state(state)
    state["brand_final_decisions"].append(decision)
    state["brand_final_decision"] = decision


def append_decision_checkpoint(
    state: MutableMapping[str, Any], checkpoint: DecisionCheckpoint
) -> None:
    """Append one real checkpoint without overwriting a prior run event."""

    initialize_state(state)
    checkpoints: list[DecisionCheckpoint] = state["decision_checkpoints"]
    if any(item.checkpoint_id == checkpoint.checkpoint_id for item in checkpoints):
        raise ValueError("decision checkpoint_id 必须唯一")
    checkpoints.append(checkpoint)


def append_decision_timeline_event(
    state: MutableMapping[str, Any], event: DecisionTimelineEvent
) -> None:
    """Timeline writes are append-only so BLOCK/REVISE history cannot disappear."""

    initialize_state(state)
    timeline: list[DecisionTimelineEvent] = state["decision_timeline"]
    if any(item.event_id == event.event_id for item in timeline):
        raise ValueError("decision timeline event_id 必须唯一")
    timeline.append(event)


def switch_mode(state: MutableMapping[str, Any], mode: str | None) -> bool:
    initialize_state(state)
    if state["app_mode"] == mode:
        return False
    for key, default in SESSION_DEFAULTS.items():
        state[key] = deepcopy(default)
    state["app_mode"] = mode
    return True
