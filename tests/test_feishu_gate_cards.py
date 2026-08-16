"""带参数决策门卡片（S9）的合同测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.integrations.feishu.gate_cards import (
    final_selection_elements,
    parameterized_gate_elements,
    reselect_primary_elements,
    submit_selection_elements,
)
from src.integrations.feishu.gate_executor import next_gate_elements
from src.integrations.feishu.notification_cards import validate_card
from src.schemas import (
    CandidateSelection,
    CandidateSpecificityAudit,
    PendingSelectionPackage,
    SpecificityAuditBatch,
    SpecificityAuditType,
    SpecificityFinding,
    SpecificitySeverity,
)

from test_feishu_decision_gates import CREATED_AT, _snapshot


def _finding(finding_id: str, candidate_id: str, audit_type: SpecificityAuditType):
    return SpecificityFinding(
        finding_id=finding_id,
        candidate_id=candidate_id,
        audit_type=audit_type,
        severity=SpecificitySeverity.NOTE,
        claim="专属度可接受",
        evidence_ids=[],
        brand_fact_ids=[],
        competitor_replacement_result="竞品无法直接替换",
        delivery_condition="现有渠道可兑现",
        failure_mode="无明显失败模式",
        triggers_next_round=False,
    )


def _audit(candidate_id: str) -> CandidateSpecificityAudit:
    return CandidateSpecificityAudit(
        candidate_id=candidate_id,
        candidate_version=1,
        findings=[
            _finding(f"f-{candidate_id}-{kind.value}", candidate_id, kind)
            for kind in SpecificityAuditType
        ],
        consumer_evidence=[],
        meijian_assets=[],
        competitor_replacement_result="可防御",
        product_delivery_conditions=[],
        applicable_scenarios=[],
        failure_reasons=[],
        next_validation_question="下一步验证场景",
    )


def _write_stage(run_root: Path, run_id: str, name: str, model) -> None:
    stage_dir = run_root / run_id / "stages"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / f"{name}.json").write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )


def _write_selection(run_root: Path, run_id: str = "run-1") -> None:
    _write_stage(
        run_root,
        run_id,
        "selection",
        CandidateSelection(
            run_id=run_id,
            pending_selection_sha256="a" * 64,
            selected_candidate_ids=["C1", "C2"],
            primary_candidate_id="C1",
            selected_by="cli",
            selected_at=CREATED_AT,
        ),
    )


def _write_pending_selection(run_root: Path, run_id: str = "run-1") -> None:
    package = PendingSelectionPackage(
        run_id=run_id,
        foundation_sha256="b" * 64,
        candidates=[_snapshot("C1", 1), _snapshot("C2", 2)],
        recommended_candidate_id="C1",
        specificity_audit=SpecificityAuditBatch(
            run_id=run_id,
            round_index=1,
            candidate_ids=["C1", "C2"],
            candidate_versions={"C1": 1, "C2": 1},
            audits=[_audit("C1"), _audit("C2")],
        ),
        created_at=CREATED_AT,
    )
    _write_stage(run_root, run_id, "pending_selection", package)


def test_submit_selection_buttons_from_pending_selection(tmp_path: Path) -> None:
    _write_pending_selection(tmp_path)

    elements = submit_selection_elements("run-1", tmp_path, prefix="gtx")

    text = json.dumps(elements, ensure_ascii=False)
    assert text.count('"type": "callback"') + text.count('"type":"callback"') >= 2
    assert "SUBMIT_SELECTION" in text
    assert "selected_candidate_ids" in text
    assert "推荐" not in text  # 按钮标签只含标题与主叙事语义
    card = {"schema": "2.0", "config": {"update_multi": True}, "body": {"elements": elements}}
    validate_card(card)


def test_submit_selection_missing_stage_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pending_selection"):
        submit_selection_elements("run-1", tmp_path, prefix="gtx")


def test_reselect_buttons_exclude_current_primary(tmp_path: Path) -> None:
    _write_selection(tmp_path)

    elements = reselect_primary_elements("run-1", tmp_path, prefix="gtx")

    text = json.dumps(elements, ensure_ascii=False)
    assert "主叙事改为 C2" in text
    assert "RESELECT_PRIMARY" in text
    assert "C1" not in text.split("主叙事改为")[1].split('"')[0]


def test_final_selection_buttons_reason_only_for_non_primary(tmp_path: Path) -> None:
    _write_selection(tmp_path)

    elements = final_selection_elements("run-1", tmp_path, prefix="gtx")

    text = json.dumps(elements, ensure_ascii=False)
    assert "终选 C1（主叙事）" in text
    assert "终选 C2" in text
    c2_segment = text[text.index("终选 C2") : text.index("终选 C2") + 400]
    assert "人工经飞书卡片确认偏离主叙事" in c2_segment
    c1_segment = text[text.index("终选 C1") : text.index("终选 C2")]
    assert "reason" not in c1_segment


def test_parameterized_elements_by_stage(tmp_path: Path) -> None:
    assert parameterized_gate_elements("run-1", "HOLDOUT", tmp_path, prefix="gtx") == []
    hint = parameterized_gate_elements(
        "run-1", "AWAITING_SPECIFICITY_AUDIT", tmp_path, prefix="gtx"
    )
    assert any("CLI" in json.dumps(item, ensure_ascii=False) for item in hint)


def test_next_gate_elements_with_run_root_builds_real_buttons(tmp_path: Path) -> None:
    _write_pending_selection(tmp_path)

    elements = next_gate_elements(
        "run-1", "AWAITING_SELECTION", prefix="status", run_root=tmp_path
    )

    text = json.dumps(elements, ensure_ascii=False)
    assert "SUBMIT_SELECTION" in text
    assert "需参数" not in text

    without_root = next_gate_elements(
        "run-1", "AWAITING_SELECTION", prefix="status", run_root=None
    )
    assert "需参数" in json.dumps(without_root, ensure_ascii=False)
