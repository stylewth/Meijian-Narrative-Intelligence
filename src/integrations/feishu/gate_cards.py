"""Build parameterized decision-gate buttons from frozen run artifacts.

按钮载荷全部来自冻结阶段文件（pending_selection / selection），
不做推断、不引入未冻结状态；文件缺失即抛错，暴露 run 完整性问题。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.schemas import CandidateSelection, PendingSelectionPackage

from .notification_cards import _button, _button_elements, _text_element


_MAX_TITLE_CHARS = 12


def _load_stage_model(run_root: Path, run_id: str, name: str, model: type) -> Any:
    path = Path(run_root) / run_id / "stages" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"阶段 {name} 未冻结：{path}")
    return model.model_validate_json(path.read_text(encoding="utf-8"), strict=True)


def _short_title(title: str) -> str:
    return title if len(title) <= _MAX_TITLE_CHARS else title[: _MAX_TITLE_CHARS - 1] + "…"


def submit_selection_elements(
    run_id: str, run_root: Path, *, prefix: str
) -> list[dict[str, Any]]:
    """AWAITING_SELECTION 阶段的选线按钮：每候选一键「唯一入围 + 主叙事」。

    多候选入围组合（2-3 个）仍走本地 CLI，卡片上如实说明。
    """

    package = _load_stage_model(run_root, run_id, "pending_selection", PendingSelectionPackage)
    buttons = []
    for index, snapshot in enumerate(package.candidates):
        candidate = snapshot.ranked_narrative.candidate
        payload = {
            "selected_candidate_ids": [candidate.candidate_id],
            "primary_candidate_id": candidate.candidate_id,
        }
        buttons.append(
            _button(
                f"{prefix}sel_{index}",
                f"选 {_short_title(candidate.title)}（主叙事）",
                {
                    "type": "DECISION_GATE",
                    "run_id": run_id,
                    "gate_action": "SUBMIT_SELECTION",
                    "payload": payload,
                },
                button_type="primary"
                if candidate.candidate_id == package.recommended_candidate_id
                else "default",
            )
        )
    elements = _button_elements(buttons)
    elements.append(
        _text_element(
            f"{prefix}sel_hint",
            "按钮为单候选选线；需要 2-3 个候选入围的组合仍由本地 CLI 提交。",
        )
    )
    return elements


def reselect_primary_elements(
    run_id: str, run_root: Path, *, prefix: str
) -> list[dict[str, Any]]:
    """AWAITING_PRIMARY_RESELECTION 阶段：每个已入围候选一个换主叙事按钮。"""

    selection = _load_stage_model(run_root, run_id, "selection", CandidateSelection)
    buttons = []
    for index, candidate_id in enumerate(selection.selected_candidate_ids):
        if candidate_id == selection.primary_candidate_id:
            continue
        buttons.append(
            _button(
                f"{prefix}rp_{index}",
                f"主叙事改为 {candidate_id}",
                {
                    "type": "DECISION_GATE",
                    "run_id": run_id,
                    "gate_action": "RESELECT_PRIMARY",
                    "payload": {"primary_candidate_id": candidate_id},
                },
                button_type="default",
            )
        )
    if not buttons:
        return []
    return _button_elements(buttons)


def final_selection_elements(
    run_id: str, run_root: Path, *, prefix: str
) -> list[dict[str, Any]]:
    """CHECKPOINT_04 阶段的终选按钮：入围集来自冻结 selection，推荐位为人工主叙事。

    非主叙事候选的偏离理由为事实性备注（人工经飞书卡片确认），不虚构分析。
    """

    selection = _load_stage_model(run_root, run_id, "selection", CandidateSelection)
    primary = selection.primary_candidate_id
    buttons = []
    for index, candidate_id in enumerate(selection.selected_candidate_ids):
        payload: dict[str, Any] = {
            "shortlisted_candidate_ids": list(selection.selected_candidate_ids),
            "selected_candidate_id": candidate_id,
            "recommended_candidate_id": primary,
        }
        if candidate_id != primary:
            payload["reason"] = "人工经飞书卡片确认偏离主叙事"
        buttons.append(
            _button(
                f"{prefix}fin_{index}",
                f"终选 {candidate_id}" + ("（主叙事）" if candidate_id == primary else ""),
                {
                    "type": "DECISION_GATE",
                    "run_id": run_id,
                    "gate_action": "SUBMIT_FINAL_SELECTION",
                    "payload": payload,
                },
                button_type="primary" if candidate_id == primary else "default",
            )
        )
    return _button_elements(buttons)


def import_audit_hint(*, prefix: str) -> dict[str, Any]:
    return _text_element(
        f"{prefix}imp_hint",
        "导入专属度审查结果需要 Luna 结果文件路径，仍由本地 CLI 执行 "
        "（import-specificity --result-path）。",
    )


_STAGE_PARAMETER_ELEMENTS = {
    "AWAITING_SELECTION": submit_selection_elements,
    "AWAITING_PRIMARY_RESELECTION": reselect_primary_elements,
    "CHECKPOINT_04": final_selection_elements,
}


def parameterized_gate_elements(
    run_id: str, stage: str, run_root: Path, *, prefix: str
) -> list[dict[str, Any]]:
    """当前阶段带参数门的卡片元素；阶段没有带参数门时返回空。"""

    builder = _STAGE_PARAMETER_ELEMENTS.get(stage)
    if builder is None:
        if stage == "AWAITING_SPECIFICITY_AUDIT":
            return [import_audit_hint(prefix=prefix)]
        return []
    return builder(run_id, Path(run_root), prefix=prefix)


__all__ = [
    "final_selection_elements",
    "import_audit_hint",
    "parameterized_gate_elements",
    "reselect_primary_elements",
    "submit_selection_elements",
]
