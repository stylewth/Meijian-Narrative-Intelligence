"""Strict, resumable orchestration for a decision run.

The run store deliberately has a small immutable stage surface.  Coordinator
runtime state therefore lives beside a run, in a directory keyed by run id;
the immutable business artifacts remain owned by ``DecisionRunStore``.
"""

from __future__ import annotations

import datetime as _datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from pydantic import BaseModel, Field

from src.schemas import (
    BlindEvidenceItem,
    BlindEvidencePackage,
    BlindSourceManifest,
    CandidateDecisionSnapshot,
    CandidateSelection,
    DecisionDisplayEvent,
    DecisionEvolutionCheckpoint,
    DecisionFoundationState,
    DecisionOriginalSnapshot,
    DecisionRunStage,
    SpecificityAuditBatch,
    CandidateSpecificityHistory,
    SpecificityCandidateRevision,
    SpecificityOutcome,
    SpecificityRevisionBatch,
    SpecificityRound,
    SpecificitySession,
    FinalCandidateSelection,
    PendingSelectionPackage,
    OfflineSpecificityTask,
    NarrativeCandidate,
)
from src.services.brand_specificity import (
    apply_specificity_revision,
    revise_from_specificity_audit,
    run_specificity_final_rank,
    should_continue_specificity,
    SPECIFICITY_FINAL_RANK_PROMPT_SHA256,
)
from src.services.brand_specificity_offline import (
    export_specificity_task,
    import_specificity_result,
    write_specificity_task,
)
from src.services.blind_evidence import BlindSource, freeze_blind_evidence
from src.services.blind_reassessment import reassess_from_blind_evidence
from src.services.decision_inputs import DecisionInputBundle
from src.services.decision_delta import derive_business_status
from src.services.decision_run_store import AuditedLLMClient, DecisionRunStore
from src.services.decision_runner import DecisionRunner
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes


_HOLDOUT = DecisionRunStage.HOLDOUT.value
_READY_REASSESS = DecisionRunStage.READY_REASSESS.value
_FAILED = DecisionRunStage.FAILED.value


class DecisionRunStateError(ValueError):
    """Raised when a persisted run cannot legally advance."""


class _CoordinatorState(BaseModel):
    model_config = {"extra": "forbid"}

    run_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    resume_stage: str | None = None
    foundation_sha256: str | None = None
    pending_selection_sha256: str | None = None
    selection_sha256: str | None = None
    specificity_round_index: int = Field(ge=1, le=3)
    pending_luna_task_sha256: str | None = None
    specificity_candidate_ids: list[str]
    specificity_revision_count: int = Field(ge=0)
    specificity_outcome: str | None = None
    specificity_session_sha256: str | None = None
    original_snapshot_sha256: str | None = None
    blind_evidence_sha256: str | None = None
    checkpoint_sha256: dict[str, str] = Field(default_factory=dict)
    final_selection_sha256: str | None = None


class _FailureRecord(BaseModel):
    model_config = {"extra": "forbid"}

    run_id: str = Field(min_length=1)
    sequence: int = Field(gt=0)
    action: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    error_type: str = Field(min_length=1)
    error_message: str = Field(min_length=1)
    retry_call_id: str | None = None
    created_at: _datetime.datetime


def _sha(value: BaseModel) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_model(path: Path, model: type[BaseModel]) -> BaseModel:
    data = path.read_bytes()
    value = json.loads(data.decode("utf-8"))
    if canonical_json_bytes(value) != data:
        raise ValueError(f"{path.name} is not canonical JSON")
    return model.model_validate_json(data, strict=True)


def _read_model_from_payload(payload: Any, model: type[BaseModel]) -> BaseModel:
    data = canonical_json_bytes(payload)
    return model.model_validate_json(data, strict=True)


class DecisionRunCoordinator:
    """Advance exactly one legal run action at a time."""

    def __init__(
        self,
        *,
        run_store: DecisionRunStore | None = None,
        store: DecisionRunStore | None = None,
        runner: DecisionRunner,
        audited_client: AuditedLLMClient | None = None,
        client: Any | None = None,
        inputs: DecisionInputBundle | None = None,
    ) -> None:
        self._store = run_store or store
        if not isinstance(self._store, DecisionRunStore):
            raise TypeError("run_store must be a DecisionRunStore")
        if not isinstance(runner, DecisionRunner):
            raise TypeError("runner must be a DecisionRunner")
        self._runner = runner
        self._audited = audited_client or client or getattr(runner, "_client", None)
        if not isinstance(self._audited, AuditedLLMClient):
            raise TypeError("audited_client must be an AuditedLLMClient")
        self._coord_dir = self._store.root / "coordinator"
        self._state_path = self._coord_dir / "state.json"
        self._runtime_path = self._coord_dir / "runtime.json"
        self._inputs_path = self._coord_dir / "inputs.json"
        self._state = self._load_or_infer_state()
        self.inputs = inputs
        if self.inputs is None and self._inputs_path.exists():
            self.inputs = self._load_inputs()

    @property
    def current_stage(self) -> DecisionRunStage | str:
        try:
            return DecisionRunStage(self._state.stage)
        except ValueError:
            return self._state.stage

    def prepare(self, inputs: DecisionInputBundle, *, retry_call_id: str | None = None) -> DecisionRunStage:
        if not isinstance(inputs, DecisionInputBundle):
            raise TypeError("inputs must be a DecisionInputBundle")
        self._require_stage(DecisionRunStage.PREPARED, retry_call_id)

        def action() -> DecisionRunStage:
            with self._audited.stage(DecisionRunStage.PREPARED, retry_call_id):
                baseline = self._runner.build_baseline(inputs)
                foundation = self._runner.apply_challenge(baseline, inputs.challenge_atoms)
            artifact = self._store.freeze_stage("foundation", foundation)
            self.inputs = inputs
            self._write_inputs(inputs)
            self._write_runtime(foundation)
            self._state = self._state.model_copy(
                update={"foundation_sha256": _sha(artifact.manifest)}
            )
            task_path = self._export_specificity_task(
                round_index=1,
                candidate_ids=[candidate.candidate_id for candidate in foundation.candidates],
                candidates=list(foundation.candidates),
                prior_rounds=[],
            )
            self._state = self._state.model_copy(
                update={
                    "stage": DecisionRunStage.AWAITING_SPECIFICITY_AUDIT.value,
                    "resume_stage": None,
                    "foundation_sha256": _sha(artifact.manifest),
                    "specificity_round_index": 1,
                    "pending_luna_task_sha256": sha256_bytes(task_path.read_bytes()),
                    "specificity_candidate_ids": [
                        candidate.candidate_id for candidate in foundation.candidates
                    ],
                    "specificity_revision_count": 0,
                    "specificity_outcome": None,
                }
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage.AWAITING_SPECIFICITY_AUDIT,
                snapshot_id=task_path.stem,
                snapshot_sha256=self._state.pending_luna_task_sha256,
            )
            return DecisionRunStage.AWAITING_SPECIFICITY_AUDIT

        return self._execute("prepare", action, retry_call_id)

    def export_specificity_audit(self, run_id: str) -> Path:
        self._assert_run_id(run_id)
        self._require_stage(DecisionRunStage.AWAITING_SPECIFICITY_AUDIT, None)
        task_path = self._specificity_task_path(self._state.specificity_round_index)
        if task_path.is_file() and not task_path.is_symlink():
            task = _read_model(task_path, OfflineSpecificityTask)
            if task.run_id != self._run_id or task.round_index != self._state.specificity_round_index:
                raise DecisionRunStateError("pending Luna task identity does not match coordinator state")
            self._assert_task_hash(task_path)
            return task_path
        if task_path.exists() or task_path.is_symlink():
            raise DecisionRunStateError("pending Luna task must be a regular file")
        foundation = self._load_stage("foundation", DecisionFoundationState, self._state.foundation_sha256)
        candidate_ids = list(self._state.specificity_candidate_ids)
        if not candidate_ids:
            raise DecisionRunStateError("specificity candidate set is missing")
        candidates = [
            candidate for candidate in foundation.candidates if candidate.candidate_id in set(candidate_ids)
        ]
        if {candidate.candidate_id for candidate in candidates} != set(candidate_ids):
            raise DecisionRunStateError("specificity candidate set is not from the frozen foundation")
        prior_rounds = self._specificity_rounds_before(self._state.specificity_round_index)
        task_path = self._export_specificity_task(
            round_index=self._state.specificity_round_index,
            candidate_ids=candidate_ids,
            candidates=candidates,
            prior_rounds=prior_rounds,
        )
        self._state = self._state.model_copy(
            update={"pending_luna_task_sha256": sha256_bytes(task_path.read_bytes())}
        )
        self._save_state()
        return task_path

    def import_specificity_audit(self, run_id: str, result_path: Path) -> DecisionRunStage:
        self._assert_run_id(run_id)
        if not isinstance(result_path, Path):
            raise TypeError("result_path must be a Path")
        self._require_stage(DecisionRunStage.AWAITING_SPECIFICITY_AUDIT, None)

        def action() -> DecisionRunStage:
            task_path = self._specificity_task_path(self._state.specificity_round_index)
            task = _read_model(task_path, OfflineSpecificityTask)
            self._assert_task_hash(task_path)
            import_specificity_result(self._store, task, result_path)
            call = self._store.find_offline_specificity_call(task.task_id)
            if call.response is None:
                raise DecisionRunStateError("specificity import did not freeze a response")
            audit = _read_model_from_payload(
                call.response.manifest.response_payload, SpecificityAuditBatch
            )
            if audit.run_id != self._run_id or audit.round_index != task.round_index:
                raise DecisionRunStateError("specificity audit identity does not match task")
            foundation = self._load_stage("foundation", DecisionFoundationState, self._state.foundation_sha256)
            if self._state.specificity_round_index == 1:
                snapshots = self._snapshots_for_candidate_ids(
                    foundation, list(foundation.candidates)
                )
                package = PendingSelectionPackage(
                    run_id=self._run_id,
                    foundation_sha256=_sha(foundation),
                    candidates=snapshots,
                    recommended_candidate_id=foundation.ranking.recommended_candidate_id,
                    specificity_audit=audit,
                    created_at=_datetime.datetime.now(_datetime.timezone.utc),
                )
                artifact = self._store.freeze_stage("pending_selection", package)
                self._state = self._state.model_copy(
                    update={
                        "stage": DecisionRunStage.AWAITING_SELECTION.value,
                        "pending_selection_sha256": _sha(artifact.manifest),
                    }
                )
                self._save_state()
                self._display(
                    stage=DecisionRunStage.AWAITING_SELECTION,
                    snapshot_id="pending-selection",
                    snapshot_sha256=self._state.pending_selection_sha256,
                )
                return DecisionRunStage.AWAITING_SELECTION

            substantive = any(
                finding.severity.value != "NOTE"
                for audit_item in audit.audits
                for finding in audit_item.findings
            )
            next_stage = (
                DecisionRunStage.SPECIFICITY_REVISION
                if substantive and self._state.specificity_round_index < 3
                else DecisionRunStage.SPECIFICITY_FINAL_RANK
            )
            self._state = self._state.model_copy(
                update={
                    "stage": next_stage.value,
                    "specificity_outcome": (
                        SpecificityOutcome.HUMAN_REVIEW.value
                        if substantive and self._state.specificity_round_index >= 3
                        else (None if substantive else SpecificityOutcome.READY.value)
                    ),
                }
            )
            self._save_state()
            self._display(
                stage=next_stage,
                snapshot_id=task_path.stem,
                snapshot_sha256=sha256_bytes(call.response.path.read_bytes()),
            )
            return next_stage

        return self._execute("import_specificity_audit", action, None)

    def submit_selection(
        self,
        run_id: str,
        selected_ids: list[str],
        primary_id: str,
    ) -> DecisionRunStage:
        self._assert_run_id(run_id)
        self._require_stage(DecisionRunStage.AWAITING_SELECTION, None)
        if not isinstance(selected_ids, list) or not 1 <= len(selected_ids) <= 3:
            raise ValueError("selected_ids must contain one to three candidates")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected_ids must be unique")
        if not isinstance(primary_id, str) or primary_id not in selected_ids:
            raise ValueError("primary_id must be one selected candidate")

        def action() -> DecisionRunStage:
            pending = self._load_stage("pending_selection", PendingSelectionPackage, self._state.pending_selection_sha256)
            foundation = self._load_stage("foundation", DecisionFoundationState, self._state.foundation_sha256)
            pending_ids = {
                item.ranked_narrative.candidate.candidate_id for item in pending.candidates
            }
            if not set(selected_ids).issubset(pending_ids):
                raise ValueError("selection contains an unknown candidate")
            selection = CandidateSelection(
                run_id=self._run_id,
                pending_selection_sha256=_sha(pending),
                selected_candidate_ids=selected_ids,
                primary_candidate_id=primary_id,
                selected_by="human",
                selected_at=_datetime.datetime.now(_datetime.timezone.utc),
            )
            selected_state = self._runner.select_narratives(
                foundation, selected_ids, primary_candidate_id=primary_id
            )
            artifact = self._store.freeze_stage("selection", selection)
            self._write_runtime(selected_state)
            substantive = any(
                finding.severity.value != "NOTE"
                for finding in pending.specificity_audit.audits
                for finding in finding.findings
            )
            next_stage = (
                DecisionRunStage.SPECIFICITY_REVISION
                if substantive
                else DecisionRunStage.SPECIFICITY_FINAL_RANK
            )
            self._state = self._state.model_copy(
                update={
                    "stage": next_stage.value,
                    "selection_sha256": _sha(artifact.manifest),
                    "specificity_candidate_ids": list(selected_ids),
                }
            )
            self._save_state()
            self._display(
                stage=next_stage,
                snapshot_id="selection",
                snapshot_sha256=self._state.selection_sha256,
            )
            return next_stage

        return self._execute("submit_selection", action, None)

    def run_specificity_revision(self, run_id: str) -> DecisionRunStage:
        self._assert_run_id(run_id)
        self._require_stage(DecisionRunStage.SPECIFICITY_REVISION, None)

        def action() -> DecisionRunStage:
            audit = self._latest_specificity_audit()
            selected_ids = list(self._state.specificity_candidate_ids)
            if set(selected_ids) != set(audit.candidate_ids) and self._state.specificity_round_index == 1:
                audit = audit.model_copy(
                    update={
                        "candidate_ids": selected_ids,
                        "candidate_versions": {
                            candidate_id: audit.candidate_versions[candidate_id]
                            for candidate_id in selected_ids
                        },
                        "audits": [
                            item for item in audit.audits if item.candidate_id in set(selected_ids)
                        ],
                    }
                )
            foundation = self._load_stage("foundation", DecisionFoundationState, self._state.foundation_sha256)
            candidates = [
                candidate
                for candidate in self._current_specificity_candidates()
                if candidate.candidate_id in set(audit.candidate_ids)
            ]
            revision = revise_from_specificity_audit(self._audited, audit, candidates)
            substantive = any(
                finding.severity.value != "NOTE"
                for audit_item in audit.audits
                for finding in audit_item.findings
            )
            continue_round = should_continue_specificity(
                audit, revision, self._state.specificity_round_index
            )
            self._state = self._state.model_copy(
                update={
                    "specificity_revision_count": self._state.specificity_revision_count + 1,
                    "specificity_outcome": (
                        SpecificityOutcome.HUMAN_REVIEW.value
                        if continue_round or any(
                            revision_item.unresolved_finding_ids
                            for revision_item in revision.revisions
                        )
                        else SpecificityOutcome.READY.value
                    ),
                }
            )
            if continue_round:
                next_round = self._state.specificity_round_index + 1
                current_candidates = self._current_specificity_candidates()
                self._state = self._state.model_copy(
                    update={
                        "stage": DecisionRunStage.AWAITING_SPECIFICITY_AUDIT.value,
                        "specificity_round_index": next_round,
                        "pending_luna_task_sha256": None,
                    }
                )
                task_path = self._export_specificity_task(
                    round_index=next_round,
                    candidate_ids=selected_ids,
                    candidates=current_candidates,
                    prior_rounds=self._specificity_rounds_before(next_round),
                )
                self._state = self._state.model_copy(
                    update={"pending_luna_task_sha256": sha256_bytes(task_path.read_bytes())}
                )
                self._save_state()
                self._display(
                    stage=DecisionRunStage.AWAITING_SPECIFICITY_AUDIT,
                    snapshot_id=task_path.stem,
                    snapshot_sha256=self._state.pending_luna_task_sha256,
                )
                return DecisionRunStage.AWAITING_SPECIFICITY_AUDIT
            self._state = self._state.model_copy(
                update={"stage": DecisionRunStage.SPECIFICITY_FINAL_RANK.value}
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage.SPECIFICITY_FINAL_RANK,
                snapshot_id="specificity-final-rank",
                snapshot_sha256=self._state.foundation_sha256,
            )
            return DecisionRunStage.SPECIFICITY_FINAL_RANK

        return self._execute("run_specificity_revision", action, None)

    def finalize_specificity(self, run_id: str) -> DecisionRunStage:
        self._assert_run_id(run_id)
        self._require_stage(DecisionRunStage.SPECIFICITY_FINAL_RANK, None)
        if self._state.specificity_session_sha256 is not None:
            raise DecisionRunStateError("specificity session is already frozen")

        def action() -> DecisionRunStage:
            foundation = self._load_runtime() or self._load_foundation()
            final_candidates = self._current_specificity_candidates(selected_only=False)
            rank_candidates = self._current_specificity_candidates(selected_only=True)
            if not rank_candidates:
                raise DecisionRunStateError("specificity final rank requires selected candidates")
            rounds = self._specificity_rounds_before(self._state.specificity_round_index + 1)
            if not rounds:
                raise DecisionRunStateError("specificity final rank requires frozen audits")
            latest_versions = rounds[-1].audit.candidate_versions
            rank_versions = {
                candidate.candidate_id: latest_versions[candidate.candidate_id]
                for candidate in rank_candidates
            }
            rank_input = {
                "run_id": self._run_id,
                "candidate_ids": [candidate.candidate_id for candidate in rank_candidates],
                "candidate_versions": rank_versions,
                "candidates": [
                    candidate.model_dump(mode="json")
                    for candidate in rank_candidates
                ],
                "audit_rounds": [
                    round_record.audit.model_dump(mode="json")
                    for round_record in rounds
                ],
            }
            rank_input_sha256 = sha256_bytes(canonical_json_bytes(rank_input))
            final_rank_client = AuditedLLMClient(
                self._audited._client,
                self._store,
                input_artifact_sha256=rank_input_sha256,
                prompt_name="brand_specificity_final_rank",
                prompt_version="brand-specificity-final-rank-v1",
                prompt_sha256=SPECIFICITY_FINAL_RANK_PROMPT_SHA256,
            )
            final_rank = run_specificity_final_rank(
                final_rank_client,
                run_id=self._run_id,
                candidates=rank_candidates,
                candidate_versions=rank_versions,
                input_sha256=rank_input_sha256,
            )
            final_foundation = self._runner.apply_specificity_candidates(
                foundation, final_candidates
            )
            ranking_items = {
                item.candidate.candidate_id: item
                for item in final_foundation.ranking.ranked_candidates
            }
            expected_ids = set(ranking_items)
            if set(final_rank.candidate_ids) - expected_ids:
                raise DecisionRunStateError("final rank returned an unknown foundation candidate")
            ordered_ids = list(final_rank.ranked_candidate_ids) + [
                candidate_id
                for candidate_id in ranking_items
                if candidate_id not in set(final_rank.ranked_candidate_ids)
            ]
            if set(ordered_ids) != expected_ids or len(ordered_ids) != len(expected_ids):
                raise DecisionRunStateError("final rank must preserve the complete foundation ranking")
            reordered = [
                ranking_items[candidate_id].model_copy(
                    update={
                        "rank": rank,
                        "is_recommended": candidate_id == final_rank.recommended_candidate_id,
                    }
                )
                for rank, candidate_id in enumerate(ordered_ids, start=1)
            ]
            final_foundation = final_foundation.model_copy(
                update={
                    "ranking": final_foundation.ranking.model_copy(
                        update={
                            "ranked_candidates": reordered,
                            "recommended_candidate_id": final_rank.recommended_candidate_id,
                        }
                    )
                }
            )
            final_artifact = self._store.freeze_stage(
                "specificity_final_foundation", final_foundation
            )
            self._write_runtime(final_foundation)
            session = self._build_specificity_session()
            session_path = self._store.root / "stages" / "specificity_session.json"
            if not session_path.is_file() or session_path.is_symlink():
                raise DecisionRunStateError("specificity session was not atomically frozen")
            session_sha = sha256_bytes(session_path.read_bytes())
            self._state = self._state.model_copy(
                update={
                    "stage": DecisionRunStage.HOLDOUT.value,
                    "specificity_session_sha256": session_sha,
                    "foundation_sha256": _sha(final_artifact.manifest),
                }
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage.HOLDOUT,
                snapshot_id="specificity-session",
                snapshot_sha256=session_sha,
            )
            return DecisionRunStage.HOLDOUT

        return self._execute("finalize_specificity", action, None)

    def run_to_selection(
        self,
        *,
        specificity_audit: SpecificityAuditBatch,
        retry_call_id: str | None = None,
    ) -> PendingSelectionPackage:
        self._require_stage(DecisionRunStage.AWAITING_SELECTION, retry_call_id)
        if self._state.pending_selection_sha256 and self._stage_exists("pending_selection"):
            package = self._load_stage("pending_selection", PendingSelectionPackage, self._state.pending_selection_sha256)
            return package

        def action() -> PendingSelectionPackage:
            foundation = self._load_stage("foundation", DecisionFoundationState, self._state.foundation_sha256)
            snapshots = self._snapshots_for_foundation(
                foundation, limit=len(foundation.candidates)
            )
            package = PendingSelectionPackage(
                run_id=self._run_id,
                foundation_sha256=_sha(foundation),
                candidates=snapshots,
                recommended_candidate_id=foundation.ranking.recommended_candidate_id,
                specificity_audit=specificity_audit,
                created_at=_datetime.datetime.now(_datetime.timezone.utc),
            )
            artifact = self._store.freeze_stage("pending_selection", package)
            self._state = self._state.model_copy(
                update={"pending_selection_sha256": _sha(artifact.manifest)}
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage.AWAITING_SELECTION,
                snapshot_id="pending-selection",
                snapshot_sha256=self._state.pending_selection_sha256,
            )
            return package

        return self._execute("run_to_selection", action, retry_call_id)

    def run_holdout(self, *, retry_call_id: str | None = None) -> DecisionRunStage:
        self._require_stage(_HOLDOUT, retry_call_id)

        def action() -> DecisionRunStage:
            specificity_session_sha256, _ = self._require_specificity_session()
            selection = self._load_stage("selection", CandidateSelection, self._state.selection_sha256)
            foundation = self._load_stage("foundation", DecisionFoundationState, self._state.foundation_sha256)
            current = self._load_runtime()
            if current is None:
                current = self._runner.select_narratives(
                    foundation,
                    selection.selected_candidate_ids,
                    primary_candidate_id=selection.primary_candidate_id,
                )
            with self._audited.stage(DecisionRunStage.AWAITING_BLIND_EVIDENCE, retry_call_id):
                validated = self._runner.validate_holdout(current, self._input_bundle().holdout_atoms)
            self._write_runtime(validated)
            primary = next(
                item for item in validated.stress_results
                if item.candidate_id == validated.primary_candidate_id
            )
            blocked = [
                item
                for item in validated.stress_results
                if item.candidate_id in validated.selected_ids
                and derive_business_status(item).value == "BLOCKED"
            ]
            if blocked and len(blocked) == len(validated.selected_ids):
                raise ValueError("all selected candidates are BLOCKED after holdout")
            if derive_business_status(primary).value == "BLOCKED":
                self._state = self._state.model_copy(
                    update={"stage": DecisionRunStage.AWAITING_PRIMARY_RESELECTION.value}
                )
                self._save_state()
                return DecisionRunStage.AWAITING_PRIMARY_RESELECTION

            original = self._make_original_snapshot(
                validated,
                selection,
                specificity_session_sha256=specificity_session_sha256,
            )
            artifact = self._store.freeze_stage("original_snapshot", original)
            self._state = self._state.model_copy(
                update={
                    "stage": DecisionRunStage.AWAITING_BLIND_EVIDENCE.value,
                    "original_snapshot_sha256": _sha(artifact.manifest),
                }
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage.AWAITING_BLIND_EVIDENCE,
                snapshot_id=original.snapshot_id,
                snapshot_sha256=self._state.original_snapshot_sha256,
            )
            return DecisionRunStage.AWAITING_BLIND_EVIDENCE

        return self._execute("run_holdout", action, retry_call_id)

    def reselect_primary(
        self,
        selection: CandidateSelection,
        *,
        specificity_session_sha256: str,
        retry_call_id: str | None = None,
    ) -> DecisionRunStage:
        self._require_stage(DecisionRunStage.AWAITING_PRIMARY_RESELECTION, retry_call_id)

        def action() -> DecisionRunStage:
            frozen_session_sha256, _ = self._require_specificity_session()
            if specificity_session_sha256 != frozen_session_sha256:
                raise DecisionRunStateError("reselection specificity session hash does not match state")
            previous = self._load_stage("selection", CandidateSelection, self._state.selection_sha256)
            if not isinstance(selection, CandidateSelection):
                raise TypeError("selection must be a CandidateSelection")
            if selection.run_id != self._run_id:
                raise ValueError("selection run_id does not match run")
            if selection.pending_selection_sha256 != previous.pending_selection_sha256:
                raise ValueError("reselection must use the original pending selection")
            if selection.selected_candidate_ids != previous.selected_candidate_ids:
                raise ValueError("reselection must keep the original shortlist")
            current = self._load_runtime()
            if current is None:
                raise ValueError("holdout state is required before primary reselection")
            reselected = self._runner.reselect_primary(
                current, primary_candidate_id=selection.primary_candidate_id
            )
            self._write_runtime(reselected)
            original = self._make_original_snapshot(
                reselected,
                selection,
                specificity_session_sha256=frozen_session_sha256,
            )
            artifact = self._store.freeze_stage("original_snapshot", original)
            self._state = self._state.model_copy(
                update={
                    "stage": DecisionRunStage.AWAITING_BLIND_EVIDENCE.value,
                    "original_snapshot_sha256": _sha(artifact.manifest),
                }
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage.AWAITING_BLIND_EVIDENCE,
                snapshot_id=original.snapshot_id,
                snapshot_sha256=self._state.original_snapshot_sha256,
            )
            return DecisionRunStage.AWAITING_BLIND_EVIDENCE

        return self._execute("reselect_primary", action, retry_call_id)

    def freeze_blind(
        self,
        sources: Iterable[BlindSource],
        items: Iterable[BlindEvidenceItem],
        confirmed_by: str,
        *,
        retry_call_id: str | None = None,
    ) -> DecisionRunStage | str:
        self._require_stage(DecisionRunStage.AWAITING_BLIND_EVIDENCE, retry_call_id)

        def action() -> DecisionRunStage | str:
            original = self._load_stage(
                "original_snapshot", DecisionOriginalSnapshot, self._state.original_snapshot_sha256
            )
            package = freeze_blind_evidence(
                run_store=self._store,
                original_snapshot=original,
                sources=sources,
                items=items,
                confirmed_by=confirmed_by,
            )
            self._state = self._state.model_copy(
                update={
                    "stage": _READY_REASSESS,
                    "blind_evidence_sha256": _sha(package),
                }
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage.READY_REASSESS,
                snapshot_id="blind-evidence",
                snapshot_sha256=self._state.blind_evidence_sha256,
            )
            return DecisionRunStage.READY_REASSESS

        return self._execute("freeze_blind", action, retry_call_id)

    def run_blind_reassessment(
        self, *, retry_call_id: str | None = None
    ) -> DecisionEvolutionCheckpoint:
        self._require_stage(_READY_REASSESS, retry_call_id)

        def action() -> DecisionEvolutionCheckpoint:
            original = self._load_stage(
                "original_snapshot", DecisionOriginalSnapshot, self._state.original_snapshot_sha256
            )
            blind = self._load_stage(
                "blind_evidence", BlindEvidencePackage, self._state.blind_evidence_sha256
            )
            current = self._load_runtime()
            if current is None:
                raise ValueError("holdout state is required before blind reassessment")
            with self._audited.stage(DecisionRunStage.CHECKPOINT_00, retry_call_id):
                reassessment = reassess_from_blind_evidence(
                    client=self._audited,
                    original=original,
                    blind=blind,
                )
            checkpoint = self._runner.apply_blind_reassessment(
                original, reassessment, foundation=current
            )
            artifact = self._store.freeze_stage("checkpoint_00", checkpoint)
            release_state = getattr(self._runner, "_release_state", None)
            if not isinstance(release_state, DecisionFoundationState):
                raise TypeError("runner did not expose frozen release state")
            self._write_runtime(release_state)
            self._state = self._state.model_copy(
                update={
                    "stage": DecisionRunStage.CHECKPOINT_00.value,
                    "checkpoint_sha256": {
                        **self._state.checkpoint_sha256,
                        "checkpoint-00": _sha(artifact.manifest),
                    },
                }
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage.CHECKPOINT_00,
                snapshot_id=checkpoint.checkpoint_id,
                checkpoint_id=checkpoint.checkpoint_id,
                snapshot_sha256=self._state.checkpoint_sha256["checkpoint-00"],
            )
            return checkpoint

        return self._execute("run_blind_reassessment", action, retry_call_id)

    def run_next_release(
        self, *, retry_call_id: str | None = None
    ) -> DecisionEvolutionCheckpoint:
        stage = self._state.stage
        if stage == _FAILED:
            if retry_call_id is None:
                raise ValueError("run is FAILED; explicit retry_call_id is required")
            self._validate_retry_call(retry_call_id)
            stage = self._state.resume_stage or stage
        if stage not in {
            DecisionRunStage.CHECKPOINT_00.value,
            DecisionRunStage.CHECKPOINT_01.value,
            DecisionRunStage.CHECKPOINT_02.value,
            DecisionRunStage.CHECKPOINT_03.value,
        }:
            raise ValueError(f"invalid stage action: run_next_release is not legal at {stage}")
        next_index = int(stage[-2:]) + 1

        def action() -> DecisionEvolutionCheckpoint:
            checkpoint_name = f"checkpoint_{next_index:02d}"
            checkpoint_id = f"checkpoint-{next_index:02d}"
            current_checkpoint = self._load_stage(
                f"checkpoint_{next_index - 1:02d}",
                DecisionEvolutionCheckpoint,
                self._state.checkpoint_sha256.get(f"checkpoint-{next_index - 1:02d}"),
            )
            runtime = self._load_runtime()
            inputs = self._input_bundle()
            if runtime is None:
                raise ValueError("release state is required before next release")
            self._restore_runner(runtime, current_checkpoint)
            with self._audited.stage(DecisionRunStage(checkpoint_id.replace("checkpoint-", "CHECKPOINT_")), retry_call_id):
                checkpoint = self._runner.release_batch(
                    runtime,
                    batch_index=next_index,
                    atoms=inputs.release_batches[next_index - 1],
                )
            artifact = self._store.freeze_stage(checkpoint_name, checkpoint)
            release_state = getattr(self._runner, "_release_state", None)
            if not isinstance(release_state, DecisionFoundationState):
                raise TypeError("runner did not expose release state")
            self._write_runtime(release_state)
            self._state = self._state.model_copy(
                update={
                    "stage": DecisionRunStage(f"CHECKPOINT_{next_index:02d}").value,
                    "checkpoint_sha256": {
                        **self._state.checkpoint_sha256,
                        checkpoint_id: _sha(artifact.manifest),
                    },
                }
            )
            self._save_state()
            self._display(
                stage=DecisionRunStage(f"CHECKPOINT_{next_index:02d}"),
                snapshot_id=checkpoint.checkpoint_id,
                checkpoint_id=checkpoint.checkpoint_id,
                snapshot_sha256=self._state.checkpoint_sha256[checkpoint_id],
            )
            return checkpoint

        return self._execute("run_next_release", action, retry_call_id)

    def submit_final_selection(
        self,
        selection: FinalCandidateSelection,
        *,
        retry_call_id: str | None = None,
    ) -> DecisionRunStage:
        self._require_stage(DecisionRunStage.CHECKPOINT_04, retry_call_id)

        def action() -> DecisionRunStage:
            checkpoints = tuple(
                self._load_stage(
                    f"checkpoint_{index:02d}",
                    DecisionEvolutionCheckpoint,
                    self._state.checkpoint_sha256.get(f"checkpoint-{index:02d}"),
                )
                for index in range(5)
            )
            checkpoint = checkpoints[-1]
            if not isinstance(selection, FinalCandidateSelection):
                raise TypeError("selection must be a FinalCandidateSelection")
            if selection.run_id != self._run_id:
                raise ValueError("final selection run_id does not match run")
            if selection.checkpoint_04_sha256 != _sha(checkpoint):
                raise ValueError("final selection is not bound to checkpoint-04")
            final_shortlist_ids = set(selection.shortlisted_candidate_ids)
            for index, candidate_checkpoint in enumerate(checkpoints):
                if index >= 1:
                    checkpoint_ids = {
                        item.ranked_narrative.candidate.candidate_id
                        for item in candidate_checkpoint.candidates
                    }
                    if checkpoint_ids != final_shortlist_ids:
                        raise ValueError("checkpoint candidate IDs do not match final shortlist")
            checkpoint_04_recommended_ids = [
                item.ranked_narrative.candidate.candidate_id
                for item in checkpoint.candidates
                if item.ranked_narrative.is_recommended
            ]
            if len(checkpoint_04_recommended_ids) != 1:
                raise ValueError(
                    "checkpoint-04 must contain exactly one AI recommended candidate"
                )
            if selection.recommended_candidate_id != checkpoint_04_recommended_ids[0]:
                raise ValueError(
                    "final selection recommendation does not match checkpoint-04 AI recommendation"
                )
            if not {
                selection.selected_candidate_id,
                selection.recommended_candidate_id,
            }.issubset(final_shortlist_ids):
                raise ValueError("final selection candidates are not from final shortlist")
            self._store.freeze_stage("final_selection", selection)
            complete_state = self._validate_complete_workspace(None)
            self._state = self._state.model_copy(update=complete_state)
            self._save_state()
            self._display(
                stage=DecisionRunStage.COMPLETE,
                snapshot_id="final-selection",
                snapshot_sha256=self._state.final_selection_sha256,
            )
            return DecisionRunStage.COMPLETE

        return self._execute("submit_final_selection", action, retry_call_id)

    @property
    def _run_id(self) -> str:
        return self._store.manifest.decision_run_id

    def _require_stage(
        self,
        expected: DecisionRunStage | str,
        retry_call_id: str | None = None,
    ) -> None:
        expected_value = expected.value if isinstance(expected, DecisionRunStage) else expected
        current = self._state.stage
        if current == _FAILED:
            if retry_call_id is None:
                raise ValueError("run is FAILED; explicit retry_call_id is required")
            self._validate_retry_call(retry_call_id)
            current = self._state.resume_stage or current
        if current != expected_value:
            raise ValueError(f"invalid stage action: expected {expected_value}, current {self._state.stage}")

    def _validate_retry_call(self, retry_call_id: str) -> None:
        if not isinstance(retry_call_id, str) or not retry_call_id:
            raise ValueError("retry_call_id must identify a missing response request")
        call = self._store.load_call(retry_call_id)
        if call.response is not None:
            raise ValueError("retry_call_id must identify a request without response")
        if self._open_call_ids() != [retry_call_id]:
            raise ValueError("retry_call_id must identify the only missing response request")

    def _execute(self, action_name: str, action: Callable[[], Any], retry_call_id: str | None) -> Any:
        try:
            return action()
        except Exception as exc:
            failed_stage = self._state.stage
            resume_stage = self._state.resume_stage or self._state.stage
            self._state = self._state.model_copy(
                update={"stage": _FAILED, "resume_stage": resume_stage}
            )
            self._save_state()
            self._append_failure(action_name, failed_stage, exc, retry_call_id)
            raise

    def _load_or_infer_state(self) -> _CoordinatorState:
        if self._state_path.exists():
            value = _read_model(self._state_path, _CoordinatorState)
            if value.run_id != self._run_id:
                raise ValueError("coordinator state run_id does not match store")
            if value.stage == DecisionRunStage.COMPLETE.value or (
                self._store.root / "stages" / "final_selection.json"
            ).is_file():
                return value.model_copy(update=self._validate_complete_workspace(value))
            return value
        state = _CoordinatorState(
            run_id=self._run_id,
            stage=DecisionRunStage.PREPARED.value,
            specificity_round_index=1,
            specificity_candidate_ids=[],
            specificity_revision_count=0,
        )
        stages_path = self._store.root / "stages"
        names = {
            entry.stem for entry in stages_path.iterdir() if entry.is_file()
        }
        updates: dict[str, Any] = {}

        if "foundation" in names:
            foundation = self._store.load_stage("foundation", DecisionFoundationState)
            updates["foundation_sha256"] = _sha(foundation)
        if "pending_selection" in names:
            pending = self._store.load_stage("pending_selection", PendingSelectionPackage)
            updates["pending_selection_sha256"] = _sha(pending)
        if "selection" in names:
            selection = self._store.load_stage("selection", CandidateSelection)
            updates["selection_sha256"] = _sha(selection)
            updates["specificity_candidate_ids"] = list(selection.selected_candidate_ids)
        if "original_snapshot" in names:
            original = self._store.load_stage(
                "original_snapshot", DecisionOriginalSnapshot
            )
            updates["original_snapshot_sha256"] = _sha(original)
            updates["specificity_session_sha256"] = original.specificity_session_sha256
        task_names = sorted(
            name for name in names if name.startswith("specificity_task_round_")
        )
        if task_names:
            task = _read_model(stages_path / f"{task_names[-1]}.json", OfflineSpecificityTask)
            updates["specificity_round_index"] = task.round_index
            updates["pending_luna_task_sha256"] = sha256_bytes(
                (stages_path / f"{task_names[-1]}.json").read_bytes()
            )
            updates["specificity_candidate_ids"] = list(task.candidate_versions)
        if "specificity_session" in names:
            session = _read_model(
                stages_path / "specificity_session.json", SpecificitySession
            )
            updates["specificity_session_sha256"] = sha256_bytes(
                (stages_path / "specificity_session.json").read_bytes()
            )
        if (stages_path / "blind").is_dir():
            blind = self._store.load_blind_evidence()
            updates["blind_evidence_sha256"] = _sha(blind)
        checkpoint_hashes: dict[str, str] = {}
        for index in range(5):
            name = f"checkpoint_{index:02d}"
            if name not in names:
                continue
            checkpoint = self._store.load_stage(name, DecisionEvolutionCheckpoint)
            checkpoint_hashes[f"checkpoint-{index:02d}"] = _sha(checkpoint)
        updates["checkpoint_sha256"] = checkpoint_hashes
        if "final_selection" in names:
            final_selection = self._store.load_stage(
                "final_selection", FinalCandidateSelection
            )
            updates["final_selection_sha256"] = _sha(final_selection)

        if "final_selection" in names:
            return state.model_copy(update=self._validate_complete_workspace(None))
        elif "checkpoint_04" in names:
            updates["stage"] = DecisionRunStage.CHECKPOINT_04.value
        elif "checkpoint_03" in names:
            updates["stage"] = DecisionRunStage.CHECKPOINT_03.value
        elif "checkpoint_02" in names:
            updates["stage"] = DecisionRunStage.CHECKPOINT_02.value
        elif "checkpoint_01" in names:
            updates["stage"] = DecisionRunStage.CHECKPOINT_01.value
        elif "checkpoint_00" in names:
            updates["stage"] = DecisionRunStage.CHECKPOINT_00.value
        elif (stages_path / "blind").is_dir():
            updates["stage"] = _READY_REASSESS
        elif "original_snapshot" in names:
            updates["stage"] = DecisionRunStage.AWAITING_BLIND_EVIDENCE.value
        elif "specificity_session" in names:
            updates["stage"] = _HOLDOUT
        elif "pending_selection" in names:
            updates["stage"] = DecisionRunStage.AWAITING_SELECTION.value
        elif task_names:
            updates["stage"] = DecisionRunStage.AWAITING_SPECIFICITY_AUDIT.value
        elif "selection" in names:
            updates["stage"] = DecisionRunStage.SPECIFICITY_FINAL_RANK.value
        elif "pending_selection" in names or "foundation" in names:
            updates["stage"] = DecisionRunStage.AWAITING_SELECTION.value
        return state.model_copy(update=updates)

    def _validate_complete_workspace(
        self, persisted: _CoordinatorState | None
    ) -> dict[str, Any]:
        stages_path = self._store.root / "stages"
        required_files = {
            "foundation.json",
            "pending_selection.json",
            "selection.json",
            "original_snapshot.json",
            "checkpoint_00.json",
            "checkpoint_01.json",
            "checkpoint_02.json",
            "checkpoint_03.json",
            "checkpoint_04.json",
            "final_selection.json",
            "specificity_final_foundation.json",
            "specificity_session.json",
        }
        missing = sorted(name for name in required_files if not (stages_path / name).is_file())
        if not (stages_path / "blind").is_dir():
            missing.append("blind")
        if missing:
            raise ValueError(f"COMPLETE run is missing required artifacts: {missing[0]}")

        base_foundation = self._store.load_stage("foundation", DecisionFoundationState)
        foundation = (
            self._store.load_stage("specificity_final_foundation", DecisionFoundationState)
            if (stages_path / "specificity_final_foundation.json").is_file()
            else base_foundation
        )
        pending = self._store.load_stage("pending_selection", PendingSelectionPackage)
        selection = self._store.load_stage("selection", CandidateSelection)
        original = self._store.load_stage("original_snapshot", DecisionOriginalSnapshot)
        source_manifest = _read_model(
            stages_path / "blind" / "source_manifest.json", BlindSourceManifest
        )
        blind = self._store.load_blind_evidence()
        checkpoints = tuple(
            self._store.load_stage(
                f"checkpoint_{index:02d}", DecisionEvolutionCheckpoint
            )
            for index in range(5)
        )
        final_selection = self._store.load_stage("final_selection", FinalCandidateSelection)

        hashes = {
            "foundation": _sha(foundation),
            "pending_selection": _sha(pending),
            "selection": _sha(selection),
            "original_snapshot": _sha(original),
            "blind_evidence": _sha(blind),
            "final_selection": _sha(final_selection),
            "checkpoint_sha256": {
                f"checkpoint-{index:02d}": _sha(checkpoint)
                for index, checkpoint in enumerate(checkpoints)
            },
        }
        if persisted is not None:
            if persisted.stage != DecisionRunStage.COMPLETE.value:
                raise ValueError("final_selection exists but coordinator state is not COMPLETE")
            expected_state_hashes = {
                "foundation_sha256": hashes["foundation"],
                "pending_selection_sha256": hashes["pending_selection"],
                "selection_sha256": hashes["selection"],
                "original_snapshot_sha256": hashes["original_snapshot"],
                "blind_evidence_sha256": hashes["blind_evidence"],
                "final_selection_sha256": hashes["final_selection"],
            }
            for field, expected in expected_state_hashes.items():
                if getattr(persisted, field) != expected:
                    raise ValueError(f"coordinator state hash does not match {field}")
            if persisted.checkpoint_sha256 != hashes["checkpoint_sha256"]:
                raise ValueError("coordinator checkpoint hashes do not match frozen artifacts")

        values = [foundation, pending, selection, original, blind, final_selection, *checkpoints]
        if any(value.run_id != self._run_id for value in values) or source_manifest.run_id != self._run_id:
            raise ValueError("COMPLETE artifact run_id mismatch")
        foundation_candidate_ids = [
            candidate.candidate_id for candidate in foundation.candidates
        ]
        ranking_candidate_ids = [
            item.candidate.candidate_id for item in foundation.ranking.ranked_candidates
        ]
        pending_candidate_ids = [
            item.ranked_narrative.candidate.candidate_id
            for item in pending.candidates
        ]
        foundation_ids = set(foundation_candidate_ids)
        pending_ids = set(pending_candidate_ids)
        selection_ids = set(selection.selected_candidate_ids)
        original_ids = {
            item.ranked_narrative.candidate.candidate_id for item in original.candidates
        }
        if pending.foundation_sha256 != _sha(base_foundation):
            raise ValueError("pending selection is not bound to foundation")
        if ranking_candidate_ids != foundation_candidate_ids:
            raise ValueError("foundation candidate order does not match ranking")
        if pending_candidate_ids != foundation_candidate_ids:
            raise ValueError("pending candidate IDs do not match foundation")
        if pending.recommended_candidate_id not in pending_ids:
            raise ValueError("pending selection recommendation is not available")
        if selection.pending_selection_sha256 != hashes["pending_selection"]:
            raise ValueError("selection is not bound to pending selection")
        if not selection_ids.issubset(pending_ids) or selection.primary_candidate_id not in selection_ids:
            raise ValueError("selection candidate IDs are not a subset of pending candidates")
        if original.foundation_sha256 != hashes["foundation"] or original.selection_sha256 != hashes["selection"]:
            raise ValueError("original snapshot provenance does not match foundation/selection")
        if original_ids != selection_ids or original.primary_candidate_id != selection.primary_candidate_id:
            raise ValueError("original snapshot candidates do not match selection")
        if original.recommended_candidate_id not in foundation_ids:
            raise ValueError("original recommendation is not from foundation")
        original_hash = hashes["original_snapshot"]
        if source_manifest.original_snapshot_sha256 != original_hash:
            raise ValueError("blind source manifest is not bound to original snapshot")
        if blind.original_snapshot_sha256 != original_hash:
            raise ValueError("blind evidence is not bound to original snapshot")
        if blind.source_manifest_sha256 != _sha(source_manifest):
            raise ValueError("blind evidence is not bound to source manifest")
        final_shortlist_ids = set(final_selection.shortlisted_candidate_ids)
        checkpoint_04_recommended_ids: list[str] = []
        for index, checkpoint in enumerate(checkpoints):
            expected_id = f"checkpoint-{index:02d}"
            if checkpoint.checkpoint_id != expected_id or checkpoint.release_index != index:
                raise ValueError("checkpoint metadata does not match its artifact name")
            if index >= 1:
                checkpoint_ids = {
                    item.ranked_narrative.candidate.candidate_id
                    for item in checkpoint.candidates
                }
                if checkpoint_ids != final_shortlist_ids:
                    raise ValueError("checkpoint candidate IDs do not match final shortlist")
            if index == 4:
                checkpoint_04_recommended_ids = [
                    item.ranked_narrative.candidate.candidate_id
                    for item in checkpoint.candidates
                    if item.ranked_narrative.is_recommended
                ]
        if not final_shortlist_ids.issubset(pending_ids):
            raise ValueError("final shortlist candidates are not a subset of pending candidates")
        if len(checkpoint_04_recommended_ids) != 1:
            raise ValueError(
                "checkpoint-04 must contain exactly one AI recommended candidate"
            )
        if final_selection.recommended_candidate_id != checkpoint_04_recommended_ids[0]:
            raise ValueError(
                "final selection recommendation does not match checkpoint-04 AI recommendation"
            )
        if final_selection.checkpoint_04_sha256 != hashes["checkpoint_sha256"]["checkpoint-04"]:
            raise ValueError("final selection is not bound to checkpoint-04")
        if not {
            final_selection.selected_candidate_id,
            final_selection.recommended_candidate_id,
        }.issubset(final_shortlist_ids):
            raise ValueError("final selection candidates are not from final shortlist")

        return {
            "stage": DecisionRunStage.COMPLETE.value,
            "foundation_sha256": hashes["foundation"],
            "pending_selection_sha256": hashes["pending_selection"],
            "selection_sha256": hashes["selection"],
            "original_snapshot_sha256": hashes["original_snapshot"],
            "blind_evidence_sha256": hashes["blind_evidence"],
            "checkpoint_sha256": hashes["checkpoint_sha256"],
            "final_selection_sha256": hashes["final_selection"],
        }

    def _save_state(self) -> None:
        _write_atomic(self._state_path, canonical_json_bytes(self._state))

    def _assert_run_id(self, run_id: str) -> None:
        if not isinstance(run_id, str) or run_id != self._run_id:
            raise DecisionRunStateError("run_id does not match the loaded run")

    def _specificity_task_path(self, round_index: int) -> Path:
        if round_index not in {1, 2, 3}:
            raise DecisionRunStateError("specificity round must be 1..3")
        return self._store.root / "stages" / f"specificity_task_round_{round_index:02d}.json"

    def _assert_task_hash(self, task_path: Path) -> None:
        expected = self._state.pending_luna_task_sha256
        actual = sha256_bytes(task_path.read_bytes())
        if expected != actual:
            raise DecisionRunStateError("pending Luna task hash does not match coordinator state")

    def _load_foundation(self) -> DecisionFoundationState:
        value = self._load_stage(
            "foundation", DecisionFoundationState, self._state.foundation_sha256
        )
        if not isinstance(value, DecisionFoundationState):
            raise DecisionRunStateError("frozen foundation has an invalid type")
        return value

    def _require_specificity_session(self) -> tuple[str, SpecificitySession]:
        expected = self._state.specificity_session_sha256
        if expected is None:
            raise DecisionRunStateError("specificity_session_sha256 is required before holdout")
        path = self._store.root / "stages" / "specificity_session.json"
        try:
            session = self._load_stage("specificity_session", SpecificitySession, expected)
        except Exception as exc:
            raise DecisionRunStateError("specificity session is missing or hash-invalid") from exc
        if session.run_id != self._run_id:
            raise DecisionRunStateError("specificity session run_id does not match the run")
        if session.specificity_model != self._store.manifest.specificity_model:
            raise DecisionRunStateError("specificity session model profile does not match Luna profile")
        if not path.is_file() or path.is_symlink():
            raise DecisionRunStateError("specificity session must be a regular frozen stage")
        return expected, session

    def _export_specificity_task(
        self,
        *,
        round_index: int,
        candidate_ids: list[str],
        candidates: list[Any],
        prior_rounds: list[SpecificityRound],
    ) -> Path:
        if [candidate.candidate_id for candidate in candidates] != candidate_ids:
            raise DecisionRunStateError("specificity candidates must match the selected order")
        foundation = self._load_stage("foundation", DecisionFoundationState, self._state.foundation_sha256)
        task = export_specificity_task(
            self._store,
            round_index,
            candidates,
            list(foundation.visible_atoms),
            list(getattr(self._runner, "_brand_facts", [])),
            prior_rounds,
            [],
        )
        task_path = self._specificity_task_path(round_index)
        write_specificity_task(task_path, task)
        return task_path

    def _specificity_rounds_before(self, round_index: int) -> list[SpecificityRound]:
        if round_index == 1:
            return []
        if round_index not in {2, 3, 4}:
            raise DecisionRunStateError("specificity round boundary must be 1..3")
        audits: dict[int, tuple[Any, SpecificityAuditBatch]] = {}
        revisions: dict[int, tuple[Any, SpecificityRevisionBatch]] = {}
        for entry in sorted((self._store.root / "calls").iterdir(), key=lambda item: int(item.name)):
            if not entry.is_dir() or not (entry / "request_manifest.json").is_file():
                continue
            request_payload = json.loads((entry / "request_manifest.json").read_text(encoding="utf-8"))
            call = self._store.load_call(request_payload["call_id"])
            if call.response is None:
                continue
            if call.manifest.stage is DecisionRunStage.AWAITING_SPECIFICITY_AUDIT:
                parsed = _read_model_from_payload(call.response.manifest.response_payload, SpecificityAuditBatch)
                audits[parsed.round_index] = (call, parsed)
            elif call.manifest.stage is DecisionRunStage.SPECIFICITY_REVISION:
                parsed = _read_model_from_payload(call.response.manifest.response_payload, SpecificityRevisionBatch)
                revisions[parsed.round_index] = (call, parsed)
        expected = list(range(1, round_index))
        if sorted(audits) != expected:
            raise DecisionRunStateError("specificity prior rounds are not complete")
        rounds: list[SpecificityRound] = []
        for index in expected:
            audit_call, audit = audits[index]
            revision_item = revisions.get(index)
            if revision_item is None:
                rounds.append(
                    SpecificityRound(
                        round_index=index,
                        audit_call_sequence=audit_call.manifest.sequence,
                        audit_response_sha256=audit_call.response.manifest.response_sha256,
                        audit=audit,
                    )
                )
                continue
            revision_call, revision = revision_item
            rounds.append(
                SpecificityRound(
                    round_index=index,
                    audit_call_sequence=audit_call.manifest.sequence,
                    audit_response_sha256=audit_call.response.manifest.response_sha256,
                    audit=audit,
                    revision_call_sequence=revision_call.manifest.sequence,
                    revision_response_sha256=revision_call.response.manifest.response_sha256,
                    revision=revision,
                )
            )
        return rounds

    def _latest_specificity_audit(self) -> SpecificityAuditBatch:
        calls: list[SpecificityAuditBatch] = []
        for entry in (self._store.root / "calls").iterdir():
            if not entry.is_dir() or not (entry / "request_manifest.json").is_file():
                continue
            request_payload = json.loads((entry / "request_manifest.json").read_text(encoding="utf-8"))
            call = self._store.load_call(request_payload["call_id"])
            if call.manifest.stage is not DecisionRunStage.AWAITING_SPECIFICITY_AUDIT or call.response is None:
                continue
            calls.append(_read_model_from_payload(call.response.manifest.response_payload, SpecificityAuditBatch))
        if not calls:
            raise DecisionRunStateError("no frozen specificity audit is available")
        return max(calls, key=lambda item: item.round_index)

    def _current_specificity_candidates(self, *, selected_only: bool = True) -> list[Any]:
        foundation = self._load_stage("foundation", DecisionFoundationState, self._state.foundation_sha256)
        candidates = {candidate.candidate_id: candidate for candidate in foundation.candidates}
        revisions: list[SpecificityRevisionBatch] = []
        for entry in (self._store.root / "calls").iterdir():
            if not entry.is_dir() or not (entry / "request_manifest.json").is_file():
                continue
            request_payload = json.loads((entry / "request_manifest.json").read_text(encoding="utf-8"))
            call = self._store.load_call(request_payload["call_id"])
            if call.manifest.stage is DecisionRunStage.SPECIFICITY_REVISION and call.response is not None:
                revisions.append(_read_model_from_payload(call.response.manifest.response_payload, SpecificityRevisionBatch))
        for revision_batch in sorted(revisions, key=lambda item: item.round_index):
            for revision in revision_batch.revisions:
                if revision.candidate_id not in candidates:
                    raise DecisionRunStateError("specificity revision contains an unknown candidate")
                candidates[revision.candidate_id] = apply_specificity_revision(
                    candidates[revision.candidate_id], revision
                )
        if not selected_only:
            return list(candidates.values())
        selected = self._state.specificity_candidate_ids
        return [candidates[candidate_id] for candidate_id in selected if candidate_id in candidates]

    def _build_specificity_session(self) -> Any:
        task_path = self._specificity_task_path(1)
        first_task = _read_model(task_path, OfflineSpecificityTask)
        initial_candidates = {
            candidate.candidate_id: candidate
            for candidate in (
                _read_model_from_payload(item, NarrativeCandidate)
                for item in first_task.payload["candidates"]
            )
        }
        rounds = self._specificity_rounds_before(self._state.specificity_round_index + 1)
        if not rounds:
            raise DecisionRunStateError("specificity session requires a frozen audit round")
        histories: list[CandidateSpecificityHistory] = []
        for candidate_id, initial in sorted(initial_candidates.items()):
            current = initial
            version = 1
            unresolved: list[str] = []
            history_rounds: list[SpecificityRound] = []
            seen = False
            for specificity_round in rounds:
                audit_entry = next(
                    (item for item in specificity_round.audit.audits if item.candidate_id == candidate_id),
                    None,
                )
                if audit_entry is None:
                    if seen:
                        break
                    continue
                seen = True
                if specificity_round.audit.candidate_versions[candidate_id] != version:
                    raise DecisionRunStateError("specificity candidate version chain is invalid")
                revision = (
                    next(
                        (
                            item
                            for item in specificity_round.revision.revisions
                            if item.candidate_id == candidate_id
                        ),
                        None,
                    )
                    if specificity_round.revision is not None
                    else None
                )
                if revision is None:
                    history_rounds.append(
                        specificity_round.model_copy(
                            update={
                                "revision_call_sequence": None,
                                "revision_response_sha256": None,
                                "revision": None,
                            }
                        )
                    )
                    unresolved = [
                        finding.finding_id
                        for finding in audit_entry.findings
                        if finding.severity.value != "NOTE"
                    ]
                    continue
                if revision.from_version != version:
                    raise DecisionRunStateError("specificity revision version chain is invalid")
                history_rounds.append(specificity_round)
                current = apply_specificity_revision(current, revision)
                version = revision.to_version
                unresolved = list(revision.unresolved_finding_ids)
            if not history_rounds:
                continue
            histories.append(
                CandidateSpecificityHistory(
                    candidate_id=candidate_id,
                    initial_candidate=initial,
                    rounds=history_rounds,
                    current_candidate=current,
                    current_version=version,
                    outcome=(
                        SpecificityOutcome.HUMAN_REVIEW
                        if unresolved
                        else SpecificityOutcome.READY
                    ),
                    unresolved_finding_ids=unresolved,
                )
            )
        if not histories:
            raise DecisionRunStateError("specificity session has no candidate histories")
        prompt_hashes: dict[str, str] = {}
        schema_hashes: dict[str, str] = {}
        completed_at: list[_datetime.datetime] = []
        for entry in (self._store.root / "calls").iterdir():
            if not entry.is_dir() or not (entry / "request_manifest.json").is_file():
                continue
            request_payload = json.loads((entry / "request_manifest.json").read_text(encoding="utf-8"))
            call = self._store.load_call(request_payload["call_id"])
            if call.response is None or call.manifest.stage not in {
                DecisionRunStage.AWAITING_SPECIFICITY_AUDIT,
                DecisionRunStage.SPECIFICITY_REVISION,
                DecisionRunStage.SPECIFICITY_FINAL_RANK,
            }:
                continue
            prefix = {
                DecisionRunStage.AWAITING_SPECIFICITY_AUDIT: "audit",
                DecisionRunStage.SPECIFICITY_REVISION: "revision",
                DecisionRunStage.SPECIFICITY_FINAL_RANK: "final_rank",
            }[call.manifest.stage]
            prompt_key = f"{prefix}:{call.manifest.prompt_name}"
            previous_prompt = prompt_hashes.setdefault(prompt_key, call.manifest.prompt_sha256)
            if previous_prompt != call.manifest.prompt_sha256:
                raise DecisionRunStateError("specificity prompt hash changed across rounds")
            schema_name = call.response.manifest.response_model_name
            previous_schema = schema_hashes.setdefault(schema_name, call.manifest.response_schema_sha256)
            if previous_schema != call.manifest.response_schema_sha256:
                raise DecisionRunStateError("specificity response schema hash changed across rounds")
            completed_at.append(call.response.manifest.completed_at)
        session = SpecificitySession(
            run_id=self._run_id,
            decision_model=self._store.manifest.decision_model,
            specificity_model=self._store.manifest.specificity_model,
            prompt_hashes=prompt_hashes,
            schema_hashes=schema_hashes,
            confirmed_pattern_ids=[],
            candidate_histories=histories,
            final_candidate_ids=list(rounds[-1].audit.candidate_ids),
            finalized_at=max(completed_at),
        )
        session_path = self._store.root / "stages" / "specificity_session.json"
        if session_path.exists() or session_path.is_symlink():
            if session_path.is_symlink():
                raise DecisionRunStateError("specificity session must not be a symlink")
            existing = _read_model(session_path, SpecificitySession)
            if existing.model_dump(mode="json") != session.model_dump(mode="json"):
                raise DecisionRunStateError("specificity session already exists with different content")
            return existing
        _write_atomic(session_path, canonical_json_bytes(session))
        return session

    def _stage_exists(self, name: str) -> bool:
        return (self._store.root / "stages" / f"{name}.json").is_file()

    def _load_stage(self, name: str, model: type[BaseModel], expected_sha256: str | None) -> Any:
        if not expected_sha256:
            raise ValueError(f"missing dependency hash for {name}")
        if name == "foundation" and self._stage_exists("specificity_final_foundation"):
            name = "specificity_final_foundation"
        if name == "blind_evidence":
            value = self._store.load_blind_evidence()
        else:
            value = self._store.load_stage(name, model)
        actual = _sha(value)
        if actual != expected_sha256:
            raise ValueError(f"{name} hash does not match coordinator state")
        return value

    def _write_runtime(self, state: DecisionFoundationState) -> None:
        _write_atomic(self._runtime_path, canonical_json_bytes(state))

    def _load_runtime(self) -> DecisionFoundationState | None:
        if not self._runtime_path.exists():
            return None
        value = _read_model(self._runtime_path, DecisionFoundationState)
        if value.run_id != self._run_id:
            raise ValueError("runtime state run_id does not match store")
        return value

    def _write_inputs(self, inputs: DecisionInputBundle) -> None:
        value = {
            "baseline_records": [item.model_dump(mode="json") for item in inputs.baseline_records],
            "baseline_atoms": [item.model_dump(mode="json") for item in inputs.baseline_atoms],
            "challenge_atoms": [item.model_dump(mode="json") for item in inputs.challenge_atoms],
            "holdout_atoms": [item.model_dump(mode="json") for item in inputs.holdout_atoms],
            "release_batches": [
                [item.model_dump(mode="json") for item in batch]
                for batch in inputs.release_batches
            ],
            "analysis_package_sha256": inputs.analysis_package_sha256,
            "challenge_package_sha256": inputs.challenge_package_sha256,
            "holdout_package_sha256": inputs.holdout_package_sha256,
            "all_stable_ids_sha256": inputs.all_stable_ids_sha256,
        }
        _write_atomic(self._inputs_path, canonical_json_bytes(value))

    def _load_inputs(self) -> DecisionInputBundle | None:
        if not self._inputs_path.exists():
            return None
        data = self._inputs_path.read_bytes()
        value = json.loads(data.decode("utf-8"))
        if canonical_json_bytes(value) != data:
            raise ValueError("inputs.json is not canonical JSON")
        from src.schemas import CommentRecord, EvidenceAtom

        def parse_json_model(model: type[BaseModel], item: object) -> BaseModel:
            return model.model_validate_json(
                canonical_json_bytes(item), strict=True
            )

        return DecisionInputBundle(
            baseline_records=tuple(
                parse_json_model(CommentRecord, item)
                for item in value["baseline_records"]
            ),
            baseline_atoms=tuple(
                parse_json_model(EvidenceAtom, item)
                for item in value["baseline_atoms"]
            ),
            challenge_atoms=tuple(
                parse_json_model(EvidenceAtom, item)
                for item in value["challenge_atoms"]
            ),
            holdout_atoms=tuple(
                parse_json_model(EvidenceAtom, item)
                for item in value["holdout_atoms"]
            ),
            release_batches=tuple(
                tuple(parse_json_model(EvidenceAtom, item) for item in batch)
                for batch in value["release_batches"]
            ),
            analysis_package_sha256=value.get("analysis_package_sha256"),
            challenge_package_sha256=value.get("challenge_package_sha256"),
            holdout_package_sha256=value.get("holdout_package_sha256"),
            all_stable_ids_sha256=value.get("all_stable_ids_sha256"),
        )

    def _input_bundle(self) -> DecisionInputBundle:
        if self.inputs is None:
            self.inputs = self._load_inputs()
        if self.inputs is None:
            raise ValueError("DecisionInputBundle is required for this action")
        return self.inputs

    def _snapshots_for_foundation(
        self, foundation: DecisionFoundationState, *, limit: int
    ) -> list[CandidateDecisionSnapshot]:
        stress_by_id = {item.candidate_id: item for item in foundation.stress_results}
        return [
            CandidateDecisionSnapshot(
                ranked_narrative=item,
                stress_result=stress_by_id[item.candidate.candidate_id],
                business_status=derive_business_status(stress_by_id[item.candidate.candidate_id]),
                supporting_count=0,
                counter_count=0,
                risk_count=len(item.candidate.risks),
            )
            for item in foundation.ranking.ranked_candidates[:limit]
        ]

    def _snapshots_for_candidate_ids(
        self,
        foundation: DecisionFoundationState,
        candidates: list[Any],
    ) -> list[CandidateDecisionSnapshot]:
        wanted = {candidate.candidate_id for candidate in candidates}
        return [
            snapshot
            for snapshot in self._snapshots_for_foundation(
                foundation, limit=len(foundation.candidates)
            )
            if snapshot.ranked_narrative.candidate.candidate_id in wanted
        ]

    def _make_original_snapshot(
        self,
        foundation: DecisionFoundationState,
        selection: CandidateSelection,
        *,
        specificity_session_sha256: str,
    ) -> DecisionOriginalSnapshot:
        frozen_foundation = self._load_stage(
            "foundation", DecisionFoundationState, self._state.foundation_sha256
        )
        frozen_selection = self._store.load_stage("selection", CandidateSelection)
        selected = set(selection.selected_candidate_ids)
        all_snapshots = self._snapshots_for_foundation(foundation, limit=len(foundation.candidates))
        snapshots = [
            item for item in all_snapshots
            if item.ranked_narrative.candidate.candidate_id in selected
        ]
        return DecisionOriginalSnapshot(
            run_id=self._run_id,
            snapshot_id="original-ai-snapshot",
            foundation_sha256=_sha(frozen_foundation),
            selection_sha256=_sha(frozen_selection),
            candidates=snapshots,
            primary_candidate_id=selection.primary_candidate_id,
            recommended_candidate_id=foundation.ranking.recommended_candidate_id,
            holdout_evidence_ids=[item.evidence_id for item in foundation.holdout_atoms],
            specificity_session_sha256=specificity_session_sha256,
            created_at=_datetime.datetime.now(_datetime.timezone.utc),
        )

    def _restore_runner(
        self, runtime: DecisionFoundationState, current_checkpoint: DecisionEvolutionCheckpoint
    ) -> None:
        if getattr(self._runner, "_release_state", None) == runtime:
            return
        self._runner = DecisionRunner.restore(
            client=self._audited,
            brand_facts=list(getattr(self._runner, "_brand_facts", [])),
            state=runtime,
            services=getattr(self._runner, "_services"),
            checkpoints=self._load_checkpoints_through(current_checkpoint.release_index),
        )

    def _load_checkpoints_through(self, release_index: int) -> list[DecisionEvolutionCheckpoint]:
        result = []
        for index in range(release_index + 1):
            checkpoint_id = f"checkpoint-{index:02d}"
            expected = self._state.checkpoint_sha256.get(checkpoint_id)
            if expected is None:
                continue
            result.append(
                self._load_stage(
                    f"checkpoint_{index:02d}", DecisionEvolutionCheckpoint, expected
                )
            )
        return result

    def _display(
        self,
        *,
        stage: DecisionRunStage,
        snapshot_id: str,
        snapshot_sha256: str | None,
        checkpoint_id: str | None = None,
    ) -> None:
        if snapshot_sha256 is None:
            raise ValueError("display event requires snapshot hash")
        latest = self._store.latest_display_event()
        self._store.append_display_event(
            DecisionDisplayEvent(
                run_id=self._run_id,
                sequence=(latest.sequence + 1 if latest else 1),
                stage=stage,
                snapshot_id=snapshot_id,
                checkpoint_id=checkpoint_id,
                snapshot_sha256=snapshot_sha256,
                created_at=_datetime.datetime.now(_datetime.timezone.utc),
            )
        )

    def _open_call_ids(self) -> list[str]:
        result: list[str] = []
        for entry in (self._store.root / "calls").iterdir():
            if entry.is_dir() and (entry / "request_manifest.json").is_file() and not (entry / "response.json").exists():
                value = json.loads((entry / "request_manifest.json").read_text(encoding="utf-8"))
                result.append(value["call_id"])
        return result

    def _append_failure(
        self,
        action: str,
        stage: str,
        exc: Exception,
        retry_call_id: str | None,
    ) -> None:
        failure_dir = self._store.root / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(path.stem)
            for path in failure_dir.iterdir()
            if path.is_file() and path.stem.isdigit()
        ]
        sequence = max(existing, default=0) + 1
        record = _FailureRecord(
            run_id=self._run_id,
            sequence=sequence,
            action=action,
            stage=stage,
            error_type=type(exc).__name__,
            error_message=str(exc) or type(exc).__name__,
            retry_call_id=retry_call_id,
            created_at=_datetime.datetime.now(_datetime.timezone.utc),
        )
        path = failure_dir / f"{sequence}.json"
        with path.open("xb") as handle:
            data = canonical_json_bytes(record)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
