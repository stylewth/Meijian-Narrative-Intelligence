"""Consume queued gate jobs against the frozen-replay timeline.

回放执行器与正式 :class:`DecisionGateExecutor` 平行：接收同一个
``GateJob`` 队列合同，但执行动作改为读取演化 run 冻结产物并推进回放
时间线，零模型调用。结果卡与运行日志 summary 必须标明「冻结回放」。
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from src.services.replay_gate_timeline import (
    NODE_LABELS,
    REPLAY_SOURCE_MARK,
    ReplayTimeline,
    ReplayTimelineError,
    apply_action,
    initial_replay_state,
    load_replay_timeline,
    read_replay_state,
    run_log_summary,
    save_replay_state,
)

from .decision_gate_store import GateJob
from .decision_gates import (
    DECISION_GATE_ACTION_TYPE,
    _GATE_LABELS,
    available_gates,
)
from src.services.replay_gate_timeline import gate_milestone_binding
from .gate_executor import _ZERO_ARG_GATES, GateExecutionError
from .notification_cards import _button, _button_elements, _text_element, validate_card
from .result_projection import run_log_fields
from .writeback_store import WritebackStore


def replay_gate_elements(
    timeline: ReplayTimeline,
    run_id: str,
    stage: str,
    *,
    prefix: str,
    milestone_cursor: int | None = None,
) -> list[dict]:
    """当前回放阶段的门按钮；参数化门从冻结产物生成真按钮。

    传入演示游标时，未解锁到对应里程碑的门不显示，避免剧透后续内容。
    """

    if run_id != timeline.run_id:
        raise ReplayTimelineError("回放按钮请求与冻结产物不属于同一个 run")
    buttons: list[dict] = []
    locked: list[str] = []
    index = 0
    for action in available_gates(stage):
        label = _GATE_LABELS.get(action, action)
        if milestone_cursor is not None:
            required_node, _drives = gate_milestone_binding(action, stage)
            if milestone_cursor < required_node:
                locked.append(f"{label}（需节点 {required_node}）")
                index += 1
                continue
        if action in _ZERO_ARG_GATES:
            buttons.append(
                _button(
                    f"{prefix}_gate_{index}",
                    label,
                    {"type": DECISION_GATE_ACTION_TYPE, "run_id": run_id, "gate_action": action},
                    button_type="primary",
                )
            )
        elif action == "SUBMIT_SELECTION":
            for candidate_id in timeline.selected_candidate_ids:
                buttons.append(
                    _button(
                        f"{prefix}_sel_{index}",
                        f"选线主叙事：{candidate_id}",
                        {
                            "type": DECISION_GATE_ACTION_TYPE,
                            "run_id": run_id,
                            "gate_action": action,
                            "payload": {
                                "selected_candidate_ids": timeline.selected_candidate_ids,
                                "primary_candidate_id": candidate_id,
                            },
                        },
                        button_type="primary",
                    )
                )
                index += 1
        elif action == "SUBMIT_FINAL_SELECTION":
            ranked = timeline.candidates_sorted_by_rank(4)
            recommended = timeline.recommended_candidate_id()
            for candidate in ranked:
                candidate_id = str(candidate["candidate_id"])
                suffix = "（AI推荐）" if candidate_id == recommended else ""
                buttons.append(
                    _button(
                        f"{prefix}_fin_{index}",
                        f"终选：{candidate_id}{suffix}",
                        {
                            "type": DECISION_GATE_ACTION_TYPE,
                            "run_id": run_id,
                            "gate_action": action,
                            "payload": {
                                "shortlisted_candidate_ids": timeline.selected_candidate_ids,
                                "selected_candidate_id": candidate_id,
                                "recommended_candidate_id": recommended,
                            },
                        },
                        button_type="primary",
                    )
                )
                index += 1
        else:
            raise ReplayTimelineError(f"回放时间线未接线的参数化门：{action}")
        index += 1
    elements: list[dict] = []
    if buttons:
        elements.extend(_button_elements(buttons))
    if locked:
        elements.append(
            _text_element(
                f"{prefix}_locked_hint",
                "尚未解锁：" + "、".join(locked) + "；请先推进网页演示或用「下一步」远端推进。",
            )
        )
    return elements


class ReplayGateExecutor:
    """Consume one :class:`GateJob` by advancing the frozen-replay timeline."""

    def __init__(
        self,
        *,
        run_root: Path,
        state_path: Path,
        message_client: Any,
        chat_id: str,
        writeback_store: WritebackStore | None,
        notification_store: Any = None,
        timeline_factory: Callable[[], ReplayTimeline] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._run_root = Path(run_root)
        self._state_path = Path(state_path)
        self._message_client = message_client
        self._chat_id = chat_id
        self._writeback_store = writeback_store
        self._notification_store = notification_store
        self._timeline_factory = timeline_factory or (
            lambda: load_replay_timeline(self._run_root)
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def gate_elements(self, run_id: str, stage: str, *, prefix: str = "status") -> list[dict]:
        timeline = self._timeline_factory()
        return replay_gate_elements(timeline, run_id, stage, prefix=prefix)

    def __call__(self, job: GateJob) -> None:
        timeline = self._timeline_factory()
        if job.run_id != timeline.run_id:
            raise ValueError(
                f"门请求的 run {job.run_id!r} 与回放 run {timeline.run_id!r} 不一致"
            )
        payload = json.loads(job.payload_json)
        if not isinstance(payload, dict):
            raise ValueError("gate payload must decode to an object")
        label = _GATE_LABELS.get(job.gate_action, job.gate_action)
        state = read_replay_state(self._state_path, timeline.run_id) or initial_replay_state(
            timeline.run_id
        )

        try:
            outcome = apply_action(
                timeline,
                state,
                job.gate_action,
                payload,
                requested_by=job.requested_by,
                now=self._clock(),
                milestone_cursor=self._demo_cursor(),
            )
        except Exception as exc:
            self._send_result_card(job, label, success=False, detail=str(exc), error=True)
            raise

        advance_note = self._enqueue_advance_command(job, outcome.drives_node)
        summary = run_log_summary(label, outcome.new_stage, outcome.detail)
        try:
            written = self._enqueue_writebacks(job, summary, outcome)
            self._send_result_card(
                job,
                label,
                success=True,
                detail=outcome.detail + advance_note,
                summary=summary,
                written=written,
                timeline=timeline,
                new_stage=outcome.new_stage,
            )
        except Exception as exc:
            save_replay_state(self._state_path, outcome.state)
            raise GateExecutionError(
                f"回放门已执行（{summary}），但后续动作失败：{exc}"
            ) from exc
        save_replay_state(self._state_path, outcome.state)

    def _demo_cursor(self) -> int | None:
        """当前演示会话已同步的最大节点；没有会话返回 None。"""

        if self._notification_store is None:
            return None
        session = self._notification_store.active_session()
        if session is None:
            return None
        ordinals = [
            job.ordinal
            for job in self._notification_store.jobs_for_session(session.session_id)
        ]
        return max(ordinals) if ordinals else 0

    def _enqueue_advance_command(self, job: GateJob, drives_node: int | None) -> str:
        if drives_node is None or self._notification_store is None:
            return ""
        session = self._notification_store.active_session()
        if session is None:
            return ""
        self._notification_store.enqueue_demo_command(
            session.session_id,
            command="ADVANCE_TO_NODE",
            target_node=drives_node,
            requested_by=job.requested_by,
        )
        return f"；已发送网页推进指令（{NODE_LABELS.get(drives_node, drives_node)}）"

    def _enqueue_writebacks(self, job: GateJob, summary: str, outcome: Any) -> int:
        if self._writeback_store is None:
            return 0
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
        for row in outcome.result_rows:
            self._writeback_store.enqueue(
                job.run_id,
                table_key="RESULTS",
                record_kind=str(row["record_kind"]),
                fields=dict(row),
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
        summary: str = "",
        written: int = 0,
        timeline: ReplayTimeline | None = None,
        new_stage: str | None = None,
        error: bool = False,
    ) -> None:
        title = f"决策门回放：{label} {'完成' if success else '失败'}"
        lines = [
            f"模式：{REPLAY_SOURCE_MARK}（执行结果读取官方冻结产物，零模型调用）",
            f"决策 run：{job.run_id}",
            f"操作者：{job.requested_by}",
            detail,
        ]
        if summary:
            lines.append(summary)
        if success and self._writeback_store is not None:
            lines.append(f"写回入队：{written} 条")
        elements = [
            {
                "tag": "div",
                "element_id": "replay_gate_result",
                "text": {"tag": "lark_md", "content": "\n".join(lines)},
            }
        ]
        if success and timeline is not None and new_stage is not None:
            elements.extend(
                replay_gate_elements(timeline, job.run_id, new_stage, prefix="rgx")
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
        self._message_client.send_card(self._chat_id, card, uuid=_replay_card_uuid(job))


def _replay_card_uuid(job: GateJob) -> str:
    raw = f"{job.job_id}:{job.run_id}:{job.gate_action}".encode("utf-8")
    result = "mj-rgate-" + hashlib.sha256(raw).hexdigest()[:32]
    if len(result) > 50:
        raise AssertionError("gate card UUID exceeds Feishu's UUID limit")
    return result


__all__ = ["ReplayGateExecutor", "replay_gate_elements"]
