"""Explicit orchestration for the five-checkpoint decision demo."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from src.schemas import (
    BrandFact,
    BusinessDecisionStatus,
    CandidateDecisionSnapshot,
    CandidateStressResult,
    BlindReassessmentResult,
    DecisionFoundationState,
    DecisionEntryMode,
    DecisionEvolutionCheckpoint,
    DecisionOriginalSnapshot,
    EvidenceAtom,
    EvidenceImpact,
    EvidenceImpactRecord,
    EvolutionCheckpointRole,
    HoldoutEvidenceImpact,
    NarrativeCandidate,
    NarrativePatch,
    RankingResult,
    RobustnessScenario,
    RobustnessScenarioResult,
)
from src.services.candidate_generation import generate_candidates
from src.services.candidate_ranking import evaluate_and_rank_candidates
from src.services.candidate_stress import run_candidate_stress
from src.services.candidate_stress import evaluate_robustness_scenarios
from src.services.corpus_analysis import analyze_corpus
from src.services.decision_delta import derive_business_status
from src.services.diversity import assess_diversity
from src.services.evidence_assessment import (
    assess_holdout_evidence,
    assess_incremental_evidence,
)
from src.services.narrative_revision import revise_candidate_from_evidence
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes


@dataclass(frozen=True)
class DecisionServices:
    analyze: Callable[..., Any]
    diversity: Callable[..., Any]
    generate: Callable[..., list[NarrativeCandidate]]
    rank: Callable[..., RankingResult]
    stress: Callable[..., CandidateStressResult]
    assess_incremental: Callable[..., list[EvidenceImpactRecord]]
    revise: Callable[..., tuple[NarrativeCandidate, list[NarrativePatch]]]
    assess_holdout: Callable[..., list[HoldoutEvidenceImpact]]
    robustness: Callable[..., list[RobustnessScenarioResult]]

    @classmethod
    def from_object(cls, services: object) -> "DecisionServices":
        return cls(
            analyze=getattr(services, "analyze"),
            diversity=getattr(services, "diversity"),
            generate=getattr(services, "generate"),
            rank=getattr(services, "rank"),
            stress=getattr(services, "stress"),
            assess_incremental=getattr(services, "assess_incremental"),
            revise=getattr(services, "revise"),
            assess_holdout=getattr(services, "assess_holdout"),
            robustness=getattr(services, "robustness"),
        )

    @classmethod
    def production(cls) -> "DecisionServices":
        return cls(
            analyze=lambda client, records, atoms: analyze_corpus(
                client=client, comments=records, evidence_atoms=atoms
            ),
            diversity=lambda client, corpus: assess_diversity(
                client=client, corpus_analysis=corpus
            ),
            generate=lambda client, corpus, diversity, records: generate_candidates(
                client=client,
                corpus_analysis=corpus,
                diversity_assessment=diversity,
                comments=records,
            ),
            rank=lambda client, candidates, records, conflicts: evaluate_and_rank_candidates(
                client=client,
                candidates=candidates,
                comments=records,
                conflicts=conflicts,
            ),
            stress=lambda client, candidate, evidence, brand_facts, robustness_results: run_candidate_stress(
                client=client,
                candidate=candidate,
                evidence=evidence,
                brand_facts=brand_facts,
                robustness_results=robustness_results,
                template_check=None,
            ),
            assess_incremental=lambda client, atoms, conflict_ids, candidate_ids: assess_incremental_evidence(
                client=client,
                atoms=atoms,
                conflict_ids=conflict_ids,
                candidate_ids=candidate_ids,
            ),
            revise=lambda client, candidate, atoms: revise_candidate_from_evidence(
                client=client, candidate=candidate, incremental_atoms=atoms
            ),
            assess_holdout=lambda client, atoms, frozen_candidate_ids: assess_holdout_evidence(
                client=client,
                atoms=atoms,
                frozen_candidate_ids=frozen_candidate_ids,
            ),
            robustness=lambda client, candidates, evidence, holdout_evidence, holdout_impacts: evaluate_robustness_scenarios(
                client=client,
                candidates=candidates,
                evidence=evidence,
                scenarios=list(RobustnessScenario),
                holdout_evidence=holdout_evidence or None,
                holdout_impacts=holdout_impacts or None,
            ),
        )


def _distribution(atoms: Iterable[EvidenceAtom], field_name: str) -> dict[str, int]:
    values = [getattr(atom, field_name).value for atom in atoms]
    return {value: values.count(value) for value in sorted(set(values))}


class DecisionRunner:
    def __init__(
        self,
        *,
        client: Any,
        brand_facts: list[BrandFact],
        services: DecisionServices | None = None,
        run_id: str | None = None,
    ) -> None:
        self._client = client
        self._brand_facts = brand_facts
        self._services = services or DecisionServices.production()
        self._run_id = run_id or uuid4().hex
        self._checkpoints: list[DecisionEvolutionCheckpoint] = []
        self._released_batches: set[int] = set()
        self._release_state: DecisionFoundationState | None = None

    @classmethod
    def restore(
        cls,
        client: Any,
        brand_facts: list[BrandFact],
        state: DecisionFoundationState,
        services: DecisionServices | object | None = None,
        checkpoints: Iterable[DecisionEvolutionCheckpoint] = (),
    ) -> "DecisionRunner":
        restored_state = (
            state
            if isinstance(state, DecisionFoundationState)
            else DecisionFoundationState.model_validate(state)
        )
        runner = cls(
            client=client,
            brand_facts=brand_facts,
            services=(
                services
                if isinstance(services, DecisionServices)
                else DecisionServices.from_object(services)
                if services is not None
                else None
            ),
            run_id=restored_state.run_id,
        )
        runner._release_state = restored_state
        runner._checkpoints = list(checkpoints)
        runner._released_batches = {
            checkpoint.release_index
            for checkpoint in runner._checkpoints
            if checkpoint.release_index > 0
        }
        return runner

    @property
    def checkpoints(self) -> tuple[DecisionEvolutionCheckpoint, ...]:
        return tuple(self._checkpoints)

    def build_baseline(self, inputs: Any) -> DecisionFoundationState:
        records = list(inputs.baseline_records)
        atoms = list(inputs.baseline_atoms)
        if not atoms:
            raise ValueError("baseline 必须包含证据")
        corpus = self._services.analyze(self._client, list(records), list(atoms))
        diversity = self._services.diversity(self._client, corpus)
        candidates = tuple(self._services.generate(self._client, corpus, diversity, list(records)))
        if not candidates:
            raise ValueError("baseline 未生成候选")
        ranking, stress_results, robustness_results = self._rank_and_stress(
            candidates=tuple(candidates),
            records=tuple(records),
            conflicts=corpus.emotional_conflicts,
            evidence=tuple(atoms),
            holdout_atoms=(),
            holdout_impacts=(),
        )
        self._checkpoints.clear()
        self._released_batches.clear()
        self._release_state = None
        return DecisionFoundationState(
            run_id=self._run_id,
            baseline_records=records,
            baseline_atoms=atoms,
            visible_atoms=atoms,
            corpus=corpus,
            candidates=list(candidates),
            ranking=ranking,
            stress_results=list(stress_results),
            robustness_results=list(robustness_results),
        )

    def apply_challenge(
        self, foundation: DecisionFoundationState, atoms: Iterable[EvidenceAtom]
    ) -> DecisionFoundationState:
        if foundation.challenge_applied:
            raise ValueError("challenge 不能重复执行")
        challenge_atoms = tuple(atoms)
        if not challenge_atoms:
            raise ValueError("challenge 必须包含证据")
        impacts = self._services.assess_incremental(
            self._client,
            list(challenge_atoms),
            {item.conflict_id for item in foundation.corpus.emotional_conflicts},
            {candidate.candidate_id for candidate in foundation.candidates},
        )
        candidates, _ = self._revise_candidates(
            foundation.candidates,
            challenge_atoms,
            impacts,
        )
        evidence = (*foundation.visible_atoms, *challenge_atoms)
        ranking, stress_results, robustness_results = self._rank_and_stress(
            candidates=candidates,
            records=foundation.baseline_records,
            conflicts=foundation.corpus.emotional_conflicts,
            evidence=evidence,
            holdout_atoms=(),
            holdout_impacts=(),
        )
        return foundation.model_copy(
            update={
                "visible_atoms": list(evidence),
                "candidates": list(candidates),
                "ranking": ranking,
                "stress_results": list(stress_results),
                "robustness_results": list(robustness_results),
                "impacts": list(impacts),
                "challenge_applied": True,
            }
        )

    def select_narratives(
        self,
        foundation: DecisionFoundationState,
        selected_ids: Iterable[str],
        *,
        primary_candidate_id: str,
    ) -> DecisionFoundationState:
        if not foundation.challenge_applied:
            raise ValueError("完成 challenge 前不能选择候选")
        selected = list(selected_ids)
        candidate_ids = {candidate.candidate_id for candidate in foundation.candidates}
        if not 1 <= len(selected) <= 3 or len(set(selected)) != len(selected):
            raise ValueError("必须选择 1 至 3 个不重复候选")
        if not set(selected).issubset(candidate_ids):
            raise ValueError("选择了不存在的候选")
        if primary_candidate_id not in selected:
            raise ValueError("primary_candidate_id 必须存在于 selected_ids")
        return foundation.model_copy(
            update={
                "selected_ids": selected,
                "primary_candidate_id": primary_candidate_id,
            }
        )

    def apply_specificity_candidates(
        self,
        foundation: DecisionFoundationState,
        candidates: Iterable[NarrativeCandidate],
    ) -> DecisionFoundationState:
        """Merge frozen specificity revisions and recompute decision-derived fields."""

        revised = tuple(candidates)
        if not revised:
            raise ValueError("specificity final candidates are required")
        candidate_ids = [candidate.candidate_id for candidate in revised]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("specificity final candidates must be unique")
        if set(candidate_ids) != {candidate.candidate_id for candidate in foundation.candidates}:
            raise ValueError("specificity final candidates must cover the foundation")
        ranking, stress_results, robustness_results = self._rank_and_stress(
            candidates=revised,
            records=tuple(foundation.baseline_records),
            conflicts=foundation.corpus.emotional_conflicts,
            evidence=tuple(foundation.visible_atoms),
            holdout_atoms=(),
            holdout_impacts=(),
        )
        if foundation.selected_ids and not set(foundation.selected_ids).issubset(set(candidate_ids)):
            raise ValueError("selection contains a candidate outside the specificity final set")
        return foundation.model_copy(
            update={
                "candidates": list(revised),
                "ranking": ranking,
                "stress_results": list(stress_results),
                "robustness_results": list(robustness_results),
                "holdout_impacts": [],
                "holdout_atoms": [],
                "holdout_validated": False,
                "frozen": False,
            }
        )

    def reselect_primary(
        self,
        foundation: DecisionFoundationState,
        *,
        primary_candidate_id: str,
    ) -> DecisionFoundationState:
        if not foundation.selected_ids:
            raise ValueError("未选择候选前不能重选主候选")
        if not foundation.holdout_validated:
            raise ValueError("未通过 holdout 验证不能重选主候选")
        if primary_candidate_id not in foundation.selected_ids:
            raise ValueError("primary_candidate_id 必须存在于 selected_ids")
        stress = next(
            result
            for result in foundation.stress_results
            if result.candidate_id == primary_candidate_id
        )
        if derive_business_status(stress) is BusinessDecisionStatus.BLOCKED:
            raise ValueError("不能选择 BLOCKED 候选作为主候选")
        return foundation.model_copy(
            update={"primary_candidate_id": primary_candidate_id}
        )

    def validate_holdout(
        self, foundation: DecisionFoundationState, atoms: Iterable[EvidenceAtom]
    ) -> DecisionFoundationState:
        if not foundation.selected_ids:
            raise ValueError("未选择候选前不能运行 holdout")
        if foundation.holdout_validated:
            raise ValueError("holdout 不能重复执行")
        holdout_atoms = tuple(atoms)
        if not holdout_atoms:
            raise ValueError("holdout 必须包含证据")
        impacts = self._services.assess_holdout(
            self._client, list(holdout_atoms), set(foundation.selected_ids)
        )
        selected_candidates = [
            candidate
            for candidate in foundation.candidates
            if candidate.candidate_id in foundation.selected_ids
        ]
        robustness_results = self._services.robustness(
            self._client,
            selected_candidates,
            list(foundation.visible_atoms),
            list(holdout_atoms),
            impacts,
        )
        selected_stress_results = self._stress_candidates(
            candidates=tuple(selected_candidates),
            evidence=foundation.visible_atoms,
            robustness_results=tuple(robustness_results),
        )
        stress_by_id = {
            result.candidate_id: result for result in selected_stress_results
        }
        stress_results = [
            stress_by_id.get(candidate.candidate_id)
            or next(
                result
                for result in foundation.stress_results
                if result.candidate_id == candidate.candidate_id
            )
            for candidate in foundation.candidates
        ]
        return foundation.model_copy(
            update={
                "holdout_atoms": list(holdout_atoms),
                "holdout_impacts": list(impacts),
                "robustness_results": list(robustness_results),
                "stress_results": stress_results,
                "holdout_validated": True,
            }
        )

    def freeze_original_snapshot(
        self, foundation: DecisionFoundationState
    ) -> DecisionFoundationState:
        if not foundation.holdout_validated:
            raise ValueError("未通过 holdout 验证不能冻结")
        if foundation.primary_candidate_id is None:
            raise ValueError("未指定 primary_candidate_id 不能冻结")
        if foundation.frozen:
            raise ValueError("foundation 已冻结")
        primary_stress = next(
            result
            for result in foundation.stress_results
            if result.candidate_id == foundation.primary_candidate_id
        )
        if derive_business_status(primary_stress) is BusinessDecisionStatus.BLOCKED:
            raise ValueError("主候选 BLOCKED，不能冻结 original")
        frozen = foundation.model_copy(update={"frozen": True})
        self._release_state = frozen
        return frozen

    def freeze_foundation(
        self, foundation: DecisionFoundationState
    ) -> DecisionFoundationState:
        return self.freeze_original_snapshot(foundation)

    def apply_blind_reassessment(
        self,
        original: DecisionOriginalSnapshot,
        reassessment: BlindReassessmentResult,
        *,
        foundation: DecisionFoundationState | None = None,
    ) -> DecisionEvolutionCheckpoint:
        """Project blind-only scores onto the frozen release state."""

        current = foundation or self._release_state
        if current is None:
            raise ValueError("release state is required before blind reassessment")
        if not current.holdout_validated:
            raise ValueError("holdout must be validated before blind reassessment")
        if original.run_id != self._run_id or reassessment.run_id != original.run_id:
            raise ValueError("blind reassessment run_id does not match this runner")
        original_snapshot_sha256 = sha256_bytes(canonical_json_bytes(original))
        if reassessment.original_snapshot_sha256 != original_snapshot_sha256:
            raise ValueError("blind reassessment is not bound to the original snapshot")

        original_ids = [
            snapshot.ranked_narrative.candidate.candidate_id
            for snapshot in original.candidates
        ]
        reassessment_by_id = {
            item.candidate_id: item for item in reassessment.candidates
        }
        if len(reassessment_by_id) != len(reassessment.candidates) or set(reassessment_by_id) != set(original_ids):
            raise ValueError("blind reassessment IDs must exactly match selected snapshots")

        current_ids = {candidate.candidate_id for candidate in current.candidates}
        if not set(original_ids).issubset(current_ids):
            raise ValueError("original selected snapshots must come from release state")

        projected: list[CandidateDecisionSnapshot] = []
        projected_by_id: dict[str, Any] = {}
        for original_snapshot in original.candidates:
            candidate_id = original_snapshot.ranked_narrative.candidate.candidate_id
            reassessed = reassessment_by_id[candidate_id]
            ranked = original_snapshot.ranked_narrative.model_copy(
                update={
                    "evaluation": reassessed.evaluation,
                    "weighted_score": reassessed.weighted_score,
                    "rank": reassessed.rank,
                    "is_recommended": reassessed.rank == 1,
                }
            )
            snapshot = original_snapshot.model_copy(
                update={
                    "ranked_narrative": ranked,
                    "business_status": reassessed.business_status,
                }
            )
            projected.append(snapshot)
            projected_by_id[candidate_id] = ranked

        ranking_items = [
            projected_by_id.get(item.candidate.candidate_id, item)
            for item in current.ranking.ranked_candidates
        ]
        recommended_id = next(
            (
                item.candidate_id
                for item in reassessment.candidates
                if item.rank == 1
            ),
            current.ranking.recommended_candidate_id,
        )
        updated_ranking = current.ranking.model_copy(
            update={
                "ranked_candidates": ranking_items,
                "recommended_candidate_id": recommended_id,
            }
        )
        self._release_state = current.model_copy(
            update={"ranking": updated_ranking, "frozen": True}
        )

        checkpoint = DecisionEvolutionCheckpoint(
            run_id=original.run_id,
            checkpoint_id="checkpoint-00",
            entry_mode=DecisionEntryMode.DEMO,
            role=EvolutionCheckpointRole.BLIND_REASSESSMENT,
            release_index=0,
            visible_evidence_ids=[atom.evidence_id for atom in current.visible_atoms],
            route_distribution=_distribution(current.visible_atoms, "route"),
            experience_distribution=_distribution(current.visible_atoms, "experience_scope"),
            grade_distribution=_distribution(current.visible_atoms, "evidence_grade"),
            candidates=projected,
            selected_candidate_id=original.primary_candidate_id,
            patches=[],
            switch_suggestion=None,
        )
        if any(item.checkpoint_id == checkpoint.checkpoint_id for item in self._checkpoints):
            raise ValueError("checkpoint-00 cannot be created twice")
        self._checkpoints.append(checkpoint)
        return checkpoint

    def release_batch(
        self,
        foundation: DecisionFoundationState,
        *,
        batch_index: int,
        atoms: Iterable[EvidenceAtom],
    ) -> DecisionEvolutionCheckpoint:
        current = self._release_state or foundation
        if not current.frozen:
            raise ValueError("foundation 未冻结，不能释放增量")
        if batch_index not in {1, 2, 3, 4}:
            raise ValueError("release batch_index 必须在 1 到 4 之间")
        if batch_index in self._released_batches:
            raise ValueError("同一 release 不能重复执行")
        if batch_index != len(self._released_batches) + 1:
            raise ValueError("release 必须按顺序执行")
        batch_atoms = tuple(atoms)
        if not batch_atoms:
            raise ValueError("release 必须包含证据")
        impacts = self._services.assess_incremental(
            self._client,
            list(batch_atoms),
            {item.conflict_id for item in current.corpus.emotional_conflicts},
            {candidate.candidate_id for candidate in current.candidates},
        )
        candidates, patches = self._revise_candidates(
            current.candidates,
            batch_atoms,
            impacts,
        )
        ranking, stress_results, robustness_results = self._rank_and_stress(
            candidates=candidates,
            records=current.baseline_records,
            conflicts=current.corpus.emotional_conflicts,
            evidence=(*current.visible_atoms, *batch_atoms),
            holdout_atoms=current.holdout_atoms,
            holdout_impacts=current.holdout_impacts,
        )
        released = current.model_copy(
            update={
                "visible_atoms": [*current.visible_atoms, *batch_atoms],
                "candidates": list(candidates),
                "ranking": ranking,
                "stress_results": list(stress_results),
                "robustness_results": list(robustness_results),
                "impacts": [*current.impacts, *impacts],
            }
        )
        checkpoint = self._checkpoint(
            released,
            release_index=batch_index,
            patches=patches,
            latest_impacts=impacts,
        )
        self._released_batches.add(batch_index)
        self._checkpoints.append(checkpoint)
        self._release_state = released
        return checkpoint

    def _rank_and_stress(
        self,
        *,
        candidates: tuple[NarrativeCandidate, ...],
        records: tuple[Any, ...],
        conflicts: Any,
        evidence: tuple[EvidenceAtom, ...],
        holdout_atoms: tuple[EvidenceAtom, ...],
        holdout_impacts: tuple[HoldoutEvidenceImpact, ...],
    ) -> tuple[
        RankingResult,
        tuple[CandidateStressResult, ...],
        tuple[RobustnessScenarioResult, ...],
    ]:
        ranking = self._services.rank(
            self._client, list(candidates), list(records), list(conflicts)
        )
        ranked_ids = {item.candidate.candidate_id for item in ranking.ranked_candidates}
        if ranked_ids != {candidate.candidate_id for candidate in candidates}:
            raise ValueError("排名服务必须返回全部当前候选")
        robustness_results = self._services.robustness(
            self._client,
            list(candidates),
            list(evidence),
            list(holdout_atoms),
            list(holdout_impacts),
        )
        stress_results = self._stress_candidates(
            candidates=candidates,
            evidence=evidence,
            robustness_results=tuple(robustness_results),
        )
        return ranking, stress_results, tuple(robustness_results)

    def _stress_candidates(
        self,
        *,
        candidates: tuple[NarrativeCandidate, ...],
        evidence: tuple[EvidenceAtom, ...],
        robustness_results: tuple[RobustnessScenarioResult, ...],
    ) -> tuple[CandidateStressResult, ...]:
        stress_results = tuple(
            self._services.stress(
                self._client,
                candidate,
                list(evidence),
                self._brand_facts,
                list(robustness_results),
            )
            for candidate in candidates
        )
        if {item.candidate_id for item in stress_results} != {
            candidate.candidate_id for candidate in candidates
        }:
            raise ValueError("压力测试服务必须返回全部当前候选")
        return stress_results

    def _revise_candidates(
        self,
        candidates: tuple[NarrativeCandidate, ...],
        atoms: tuple[EvidenceAtom, ...],
        impacts: list[EvidenceImpactRecord],
    ) -> tuple[tuple[NarrativeCandidate, ...], list[NarrativePatch]]:
        atoms_by_id = {atom.evidence_id: atom for atom in atoms}
        revised: list[NarrativeCandidate] = []
        patches: list[NarrativePatch] = []
        for candidate in candidates:
            target_ids = {candidate.candidate_id, *candidate.supporting_conflict_ids}
            relevant_atoms = [
                atoms_by_id[impact.evidence_id]
                for impact in impacts
                if impact.target_id in target_ids and impact.evidence_id in atoms_by_id
            ]
            if not relevant_atoms:
                revised.append(candidate)
                continue
            updated, candidate_patches = self._services.revise(
                self._client, candidate, relevant_atoms
            )
            if updated.candidate_id != candidate.candidate_id:
                raise ValueError("局部修订不得改变 candidate_id")
            revised.append(updated)
            patches.extend(candidate_patches)
        return tuple(revised), patches

    def _checkpoint(
        self,
        foundation: DecisionFoundationState,
        *,
        release_index: int,
        patches: list[NarrativePatch],
        latest_impacts: Iterable[EvidenceImpactRecord] = (),
    ) -> DecisionEvolutionCheckpoint:
        ranking_by_id = {
            item.candidate.candidate_id: item for item in foundation.ranking.ranked_candidates
        }
        stress_by_id = {item.candidate_id: item for item in foundation.stress_results}
        snapshots: list[CandidateDecisionSnapshot] = []
        snapshot_ids = list(foundation.selected_ids)
        for ranked in foundation.ranking.ranked_candidates:
            candidate_id = ranked.candidate.candidate_id
            if candidate_id not in snapshot_ids and len(snapshot_ids) < 3:
                snapshot_ids.append(candidate_id)
        for candidate_id in snapshot_ids:
            ranked = ranking_by_id[candidate_id]
            stress = stress_by_id[candidate_id]
            support_count = sum(
                item.impact is EvidenceImpact.SUPPORT and item.target_id == candidate_id
                for item in foundation.impacts
            )
            counter_count = sum(
                item.impact is EvidenceImpact.CHALLENGE and item.target_id == candidate_id
                for item in foundation.impacts
            )
            snapshots.append(
                CandidateDecisionSnapshot(
                    ranked_narrative=ranked,
                    stress_result=stress,
                    business_status=derive_business_status(stress),
                    supporting_count=support_count,
                    counter_count=counter_count,
                    risk_count=len(ranked.candidate.risks) + counter_count,
                )
            )
        if not snapshots:
            raise ValueError("冻结检查点必须包含已选择候选")
        selected_id = foundation.primary_candidate_id
        if selected_id is None:
            raise ValueError("检查点必须包含 primary_candidate_id")
        selected_snapshot = next(
            item
            for item in snapshots
            if item.ranked_narrative.candidate.candidate_id == selected_id
        )
        latest = list(latest_impacts)
        switch_suggestion = self._switch_suggestion(
            selected_snapshot=selected_snapshot,
            snapshots=snapshots,
            latest_impacts=latest,
        )
        return DecisionEvolutionCheckpoint(
            run_id=foundation.run_id,
            checkpoint_id=f"checkpoint-{release_index:02d}",
            entry_mode=DecisionEntryMode.DEMO,
            role=(
                EvolutionCheckpointRole.BASELINE
                if release_index == 0
                else EvolutionCheckpointRole.DELTA
            ),
            release_index=release_index,
            visible_evidence_ids=[atom.evidence_id for atom in foundation.visible_atoms],
            route_distribution=_distribution(foundation.visible_atoms, "route"),
            experience_distribution=_distribution(foundation.visible_atoms, "experience_scope"),
            grade_distribution=_distribution(foundation.visible_atoms, "evidence_grade"),
            candidates=snapshots,
            selected_candidate_id=selected_id,
            patches=patches,
            switch_suggestion=switch_suggestion,
        )

    @staticmethod
    def _switch_suggestion(
        *,
        selected_snapshot: CandidateDecisionSnapshot,
        snapshots: list[CandidateDecisionSnapshot],
        latest_impacts: list[EvidenceImpactRecord],
    ) -> Any:
        selected_id = selected_snapshot.ranked_narrative.candidate.candidate_id
        selected_targets = {
            selected_id,
            *selected_snapshot.ranked_narrative.candidate.supporting_conflict_ids,
        }
        selected_challenged = any(
            item.impact is EvidenceImpact.CHALLENGE and item.target_id in selected_targets
            for item in latest_impacts
        )
        status_rank = {
            BusinessDecisionStatus.READY_FOR_VALIDATION: 0,
            BusinessDecisionStatus.NEEDS_REVISION: 1,
            BusinessDecisionStatus.ANALYSIS_IN_PROGRESS: 2,
            BusinessDecisionStatus.BLOCKED: 3,
        }
        alternatives = []
        for snapshot in snapshots:
            candidate_id = snapshot.ranked_narrative.candidate.candidate_id
            target_ids = {
                candidate_id,
                *snapshot.ranked_narrative.candidate.supporting_conflict_ids,
            }
            support_ids = [
                item.evidence_id
                for item in latest_impacts
                if item.impact is EvidenceImpact.SUPPORT and item.target_id in target_ids
            ]
            if (
                candidate_id == selected_id
                or not support_ids
                or snapshot.business_status is BusinessDecisionStatus.BLOCKED
            ):
                continue
            ranked_higher = (
                snapshot.ranked_narrative.rank
                < selected_snapshot.ranked_narrative.rank
            )
            risk_better = (
                status_rank[snapshot.business_status]
                < status_rank[selected_snapshot.business_status]
            )
            if selected_challenged or ranked_higher or risk_better:
                alternatives.append((snapshot, support_ids))
        if not alternatives:
            return None
        target, support_ids = min(
            alternatives,
            key=lambda item: item[0].ranked_narrative.rank,
        )
        from src.schemas import CandidateSwitchSuggestion

        trigger_reasons: list[str] = []
        if selected_challenged:
            trigger_reasons.append("新增证据挑战当前候选")
        if target.ranked_narrative.rank < selected_snapshot.ranked_narrative.rank:
            trigger_reasons.append("备选排名更高")
        if (
            status_rank[target.business_status]
            < status_rank[selected_snapshot.business_status]
        ):
            trigger_reasons.append("备选风险状态更好")

        return CandidateSwitchSuggestion(
            from_candidate_id=selected_id,
            to_candidate_id=target.ranked_narrative.candidate.candidate_id,
            target_business_status=target.business_status,
            trigger_evidence_ids=support_ids,
            reason=(
                f"{'；'.join(trigger_reasons)}；备选获得当批新增支持；仅建议换线。"
            ),
        )
