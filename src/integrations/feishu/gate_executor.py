"""Execute queued decision gates against the persisted coordinator run.

门执行镜像正式 CLI 的构建合同（build_gate_coordinator）；执行后的
产物投影与日志入队（S8）失败时以「门已执行」开头报错，避免被误读为门未执行。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from src.schemas import (
    CandidateSelection,
    DecisionEvolutionCheckpoint,
    DecisionFoundationState,
    FinalCandidateSelection,
)
from src.services.prepared_corpus import sha256_bytes

from .decision_gate_store import GateJob
from .decision_gates import _GATE_LABELS, available_gates, resolve_result_path
from .gate_cards import parameterized_gate_elements
from .notification_cards import _button, _button_elements, _text_element, validate_card
from .result_projection import (
    candidate_rows,
    checkpoint_rows,
    final_selection_row,
    run_log_fields,
)
from .writeback_store import WritebackStore


class GateExecutionError(RuntimeError):
    """门已执行但后续动作（写回入队/结果卡片）失败。"""


_ZERO_ARG_GATES = frozenset(
    {
        "EXPORT_SPECIFICITY_AUDIT",
        "RUN_HOLDOUT",
        "RUN_BLIND_REASSESSMENT",
        "RUN_NEXT_RELEASE",
    }
)


def next_gate_elements(
    run_id: str,
    stage: str,
    *,
    prefix: str,
    run_root: Path | None = None,
) -> list[dict]:
    """当前阶段可推进门的元素；提供 run_root 时带参数门从冻结产物生成真按钮。"""

    buttons: list[dict] = []
    parameter_labels: list[str] = []
    for index, action in enumerate(available_gates(stage)):
        label = _GATE_LABELS.get(action, action)
        if action in _ZERO_ARG_GATES:
            buttons.append(
                _button(
                    f"{prefix}_gate_{index}",
                    label,
                    {"type": "DECISION_GATE", "run_id": run_id, "gate_action": action},
                    button_type="primary",
                )
            )
        else:
            parameter_labels.append(label)
    elements: list[dict] = []
    if buttons:
        elements.extend(_button_elements(buttons))
    if parameter_labels:
        built = (
            parameterized_gate_elements(run_id, stage, run_root, prefix=prefix)
            if run_root is not None
            else []
        )
        if built:
            elements.extend(built)
        else:
            elements.append(
                _text_element(
                    f"{prefix}_gate_hint",
                    "、".join(parameter_labels) + "（需参数，暂由本地 CLI 提供）",
                )
            )
    return elements


class DecisionGateExecutor:
    """Consume one :class:`GateJob` by driving the persisted coordinator."""

    def __init__(
        self,
        *,
        run_root: Path,
        message_client: Any,
        chat_id: str,
        writeback_store: WritebackStore | None,
        coordinator_factory: Callable[[Path, str], tuple[Any, Any]],
    ) -> None:
        self._run_root = Path(run_root)
        self._message_client = message_client
        self._chat_id = chat_id
        self._writeback_store = writeback_store
        self._coordinator_factory = coordinator_factory

    def __call__(self, job: GateJob) -> None:
        coordinator, store = self._coordinator_factory(self._run_root, job.run_id)
        payload = json.loads(job.payload_json)
        if not isinstance(payload, dict):
            raise ValueError("gate payload must decode to an object")
        label = _GATE_LABELS.get(job.gate_action, job.gate_action)

        try:
            detail = self._dispatch(coordinator, store, job, payload)
        except Exception as exc:
            self._send_result_card(job, label, success=False, detail=str(exc), error=True)
            raise

        new_stage = self._stage_value(coordinator)
        summary = f"{label}完成，新阶段 {new_stage}"
        if detail:
            summary += f"；{detail}"
        try:
            written = self._enqueue_writebacks(job, store, new_stage, summary)
            self._send_result_card(
                job, label, success=True, detail=summary, written=written, new_stage=new_stage
            )
        except Exception as exc:
            raise GateExecutionError(f"门已执行（{summary}），但后续动作失败：{exc}") from exc

    def _dispatch(
        self,
        coordinator: Any,
        store: Any,
        job: GateJob,
        payload: dict[str, Any],
    ) -> str:
        action = job.gate_action
        run_id = job.run_id

        if action == "EXPORT_SPECIFICITY_AUDIT":
            task_path = coordinator.export_specificity_audit(run_id)
            return f"任务包 {Path(task_path).name}"

        if action == "IMPORT_SPECIFICITY_AUDIT":
            result_path = resolve_result_path(
                run_root=self._run_root, result_path=str(payload["result_path"])
            )
            stage = coordinator.import_specificity_audit(run_id, result_path)
            return f"导入后进入阶段 {self._stage_value(coordinator)}"

        if action == "RUN_SPECIFICITY_REVISION":
            coordinator.run_specificity_revision(run_id)
            return ""

        if action == "FINALIZE_SPECIFICITY":
            coordinator.finalize_specificity(run_id)
            return ""

        if action == "SUBMIT_SELECTION":
            selected = list(payload["selected_candidate_ids"])
            coordinator.submit_selection(run_id, selected, payload["primary_candidate_id"])
            return f"选线 {'、'.join(selected)}（主叙事 {payload['primary_candidate_id']}）"

        if action == "RUN_HOLDOUT":
            coordinator.run_holdout()
            return ""

        if action == "RESELECT_PRIMARY":
            selection = store.load_stage("selection", CandidateSelection)
            new_primary = str(payload["primary_candidate_id"])
            if new_primary not in selection.selected_candidate_ids:
                raise ValueError(
                    f"新主叙事 {new_primary} 不在已选线集合 "
                    f"{selection.selected_candidate_ids} 内"
                )
            new_selection = selection.model_copy(
                update={
                    "primary_candidate_id": new_primary,
                    "selected_by": job.requested_by,
                    "selected_at": datetime.now(timezone.utc),
                    "reason": payload.get("reason") or None,
                }
            )
            session_sha = sha256_bytes(
                (store.root / "stages" / "specificity_session.json").read_bytes()
            )
            coordinator.reselect_primary(new_selection, session_sha)
            return f"主叙事更换为 {new_primary}"

        if action == "RUN_BLIND_REASSESSMENT":
            coordinator.run_blind_reassessment()
            return ""

        if action == "RUN_NEXT_RELEASE":
            coordinator.run_next_release()
            return ""

        if action == "SUBMIT_FINAL_SELECTION":
            checkpoint_sha = sha256_bytes(
                (store.root / "stages" / "checkpoint_04.json").read_bytes()
            )
            selection = FinalCandidateSelection(
                run_id=run_id,
                checkpoint_04_sha256=checkpoint_sha,
                shortlisted_candidate_ids=list(payload["shortlisted_candidate_ids"]),
                selected_candidate_id=str(payload["selected_candidate_id"]),
                recommended_candidate_id=str(payload["recommended_candidate_id"]),
                selected_by=job.requested_by,
                selected_at=datetime.now(timezone.utc),
                reason=payload.get("reason") or None,
            )
            coordinator.submit_final_selection(selection)
            return f"最终选择 {payload['selected_candidate_id']}"

        raise ValueError(f"未接线的决策门：{action}")

    def _enqueue_writebacks(
        self,
        job: GateJob,
        store: Any,
        new_stage: str,
        summary: str,
    ) -> int:
        if self._writeback_store is None:
            return 0
        now = datetime.now(timezone.utc)
        count = 0

        log_fields = run_log_fields(
            job.run_id,
            action=job.gate_action,
            actor_open_id=job.requested_by,
            summary=summary,
        )
        self._writeback_store.enqueue(
            job.run_id, table_key="RUN_LOG", record_kind="RUN_LOG", fields=log_fields
        )
        count += 1

        if job.gate_action == "FINALIZE_SPECIFICITY":
            foundation = store.load_stage(
                "specificity_final_foundation", DecisionFoundationState
            )
            for row in candidate_rows(foundation, created_at=now):
                self._writeback_store.enqueue(
                    job.run_id, table_key="RESULTS", record_kind="CANDIDATE", fields=row
                )
                count += 1
        elif new_stage.startswith("CHECKPOINT_"):
            checkpoint = store.load_stage(
                f"checkpoint_{new_stage[-2:]}", DecisionEvolutionCheckpoint
            )
            for row in checkpoint_rows(checkpoint, created_at=now):
                self._writeback_store.enqueue(
                    job.run_id, table_key="RESULTS", record_kind="CHECKPOINT", fields=row
                )
                count += 1
        elif job.gate_action == "SUBMIT_FINAL_SELECTION":
            final = store.load_stage("final_selection", FinalCandidateSelection)
            self._writeback_store.enqueue(
                job.run_id,
                table_key="RESULTS",
                record_kind="FINAL_SELECTION",
                fields=final_selection_row(final, created_at=now),
            )
            count += 1
        return count

    def _send_result_card(
        self,
        job: GateJob,
        label: str,
        *,
        success: bool,
        detail: str,
        written: int = 0,
        error: bool = False,
        new_stage: str | None = None,
    ) -> None:
        title = f"决策门：{label} {'完成' if success else '失败'}"
        lines = [f"决策 run：{job.run_id}", f"操作者：{job.requested_by}", detail]
        if success and self._writeback_store is not None:
            lines.append(f"写回入队：{written} 条")
        elements = [
            {
                "tag": "div",
                "element_id": "gate_result",
                "text": {"tag": "lark_md", "content": "\n".join(lines)},
            }
        ]
        if success and new_stage is not None:
            elements.extend(
                next_gate_elements(
                    job.run_id, new_stage, prefix="gtx", run_root=self._run_root
                )
            )
        card = {
            "schema": "2.0",
            "config": {"update_multi": True, "summary": {"content": title}},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": "turquoise" if success else "red",
            },
            "body": {"elements": elements},
        }
        validate_card(card)
        self._message_client.send_card(self._chat_id, card, uuid=_gate_card_uuid(job))

    @staticmethod
    def _stage_value(coordinator: Any) -> str:
        stage = coordinator.current_stage
        return getattr(stage, "value", stage)


def _gate_card_uuid(job: GateJob) -> str:
    raw = f"{job.job_id}:{job.run_id}:{job.gate_action}".encode("utf-8")
    result = "mj-gate-" + hashlib.sha256(raw).hexdigest()[:32]
    if len(result) > 50:
        raise AssertionError("gate card UUID exceeds Feishu's UUID limit")
    return result


__all__ = ["DecisionGateExecutor", "GateExecutionError"]
