"""干跑验证：脚本化审计者与修订响应驱动 pressure_audit 完整压力测试循环。

覆盖：离线代跑挂起（AWAITING_AUDITOR + 任务包落盘）→ 回填续跑 → DeepSeek
修订（版本递增、diff 落盘）→ 第二轮全 NOTE 收敛（READY 终态）。
生产路径不经过本文件；审计者实际通过 AUDITOR_* 配置的 LLMClient 调用真实
端点，或由外部审计者回填结果文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schemas import (
    CommentRecord,
    EvidenceAtom,
    NarrativeCandidate,
    SourceReference,
    SourceType,
    SpecificityAuditBatch,
    SpecificitySeverity,
)
from src.services.brand_specificity import (
    SpecificityCandidateRevisionV2,
    SpecificityEvidenceBindingV2,
    SpecificityFindingResponseV2,
    SpecificityNewBrandFactV2,
    SpecificityRevisionBatchV2,
)
from src.services.pressure_audit import run_pressure_loop

RUN_ID = "pressure-test-001"


def _comment(comment_id: str, text: str) -> CommentRecord:
    return CommentRecord(
        comment_id=comment_id,
        sample_type="目标品牌反馈",
        raw_content=text,
        source_platform="知乎",
        raw_id=comment_id,
        source=SourceReference(
            source_id=comment_id,
            source_type=SourceType.USER_COMMENT,
            source_ref=comment_id,
        ),
    )


def _candidate() -> NarrativeCandidate:
    return NarrativeCandidate(
        candidate_id="CAND-P1",
        primary_conflict_id="CONF-01",
        supporting_conflict_ids=["CONF-01"],
        title="测试候选：用可验证的兑法对照承接热度",
        target_audience="法定饮酒年龄成年人",
        user_conflict="想尝试很火的喝法，但担心照教程复刻后不符合预期",
        brand_opportunity="建议做口味×兑法对照表",
        why_brand="有用户观察到该口味出现频率高，营销热度真实存在",
        competitor_difference="当前语料未提供可核验的竞品事实",
        brand_role="配方校对者：帮用户分清配方偏差与口味偏好",
        draft_proposition="先看对照表，再决定这一杯怎么调",
        main_scenes=["法定饮酒年龄成年人居家调饮前查对照表"],
        content_theme="同一配方复刻对照",
        supporting_evidence=[{"comment_id": "C-001", "quote": "这种喝法很火，想试试"}],
        counter_evidence=[{"comment_id": "C-002", "quote": "照教程调出来太苦了"}],
        counter_evidence_note="存在负面反馈，主张限定为待验证提案",
        risks=["仅面向法定饮酒年龄成年人，倡导适量饮酒"],
    )


def _atoms_and_records() -> tuple[list[EvidenceAtom], dict[str, CommentRecord]]:
    records = {
        "C-001": _comment("C-001", "这种喝法很火，想试试"),
        "C-002": _comment("C-002", "照教程调出来太苦了"),
        "C-003": _comment("C-003", "换成别的基酒更好喝"),
    }
    atoms = [
        EvidenceAtom(
            evidence_id=comment_id,
            comment_id=comment_id,
            route="PRODUCT",
            experience_scope="ACTUAL_USE",
            evidence_grade="B",
            source=record.source,
            source_platform=record.source_platform,
        )
        for comment_id, record in records.items()
    ]
    return atoms, records


def _finding(finding_id: str, audit_type: str, severity: str, claim: str) -> dict:
    return {
        "finding_id": finding_id,
        "candidate_id": "CAND-P1",
        "audit_type": audit_type,
        "severity": severity,
        "claim": claim,
        "evidence_ids": ["C-003"],
        "brand_fact_ids": [],
        "competitor_replacement_result": "与竞争替换的关系说明",
        "delivery_condition": "兑现前提条件",
        "failure_mode": "失败模式说明",
        "suggested_patch": None,
        "triggers_next_round": severity != "NOTE",
    }


def _audit_batch(round_index: int, version: int, material: bool) -> SpecificityAuditBatch:
    severities = (
        {audit_type: "MATERIAL_RISK" for audit_type in (
            "COMPETITOR_REPLACEMENT",
            "BRAND_ASSET",
            "SCENE_DELIVERY",
            "FAILURE_PREMORTEM",
        )}
        if material
        else {audit_type: "NOTE" for audit_type in (
            "COMPETITOR_REPLACEMENT",
            "BRAND_ASSET",
            "SCENE_DELIVERY",
            "FAILURE_PREMORTEM",
        )}
    )
    findings = [
        _finding(
            f"CAND-P1-R{round_index}-{suffix}",
            audit_type,
            severities[audit_type],
            "第一轮实质风险：候选未回应基酒替换证据" if material else "边界记录，无新增风险",
        )
        for suffix, audit_type in (
            ("CR", "COMPETITOR_REPLACEMENT"),
            ("BA", "BRAND_ASSET"),
            ("SD", "SCENE_DELIVERY"),
            ("FP", "FAILURE_PREMORTEM"),
        )
    ]
    return SpecificityAuditBatch.model_validate_json(
        json.dumps(
            {
                "run_id": RUN_ID,
                "round_index": round_index,
                "candidate_ids": ["CAND-P1"],
                "candidate_versions": {"CAND-P1": version},
                "audits": [
                    {
                        "candidate_id": "CAND-P1",
                        "candidate_version": version,
                        "findings": findings,
                        "consumer_evidence": ["C-003"],
                        "brand_assets": [],
                        "competitor_replacement_result": "替换风险说明",
                        "product_delivery_conditions": ["前提条件"],
                        "applicable_scenarios": ["场景"],
                        "failure_reasons": ["失败原因"],
                        "next_validation_question": "待验证问题？",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        strict=True,
    )


class FakeApiAuditor:
    """API 模式替身：第一轮返回实质风险，第二轮收敛为 NOTE。"""

    provider = "api"
    model = "fake-auditor-model"

    def __init__(self) -> None:
        self.calls = 0

    def audit(self, task: dict) -> SpecificityAuditBatch:
        self.calls += 1
        material = task["round_index"] == 1
        return _audit_batch(task["round_index"], task["candidate_version"], material)


class FakeDeepseekClient:
    """修订替身：对每个非 NOTE finding 给 ACCEPT 并真实修改 why_brand。"""

    def __init__(self) -> None:
        self.calls = 0

    def generate_json(self, **kwargs):
        self.calls += 1
        audit = kwargs["response_model"] and None
        payload = json.loads(kwargs["user_prompt"])
        audit_batch = payload["audit"]
        candidate = payload["candidates"][0]
        non_note = [
            finding
            for audit_item in audit_batch["audits"]
            for finding in audit_item["findings"]
            if finding["severity"] != "NOTE"
        ]
        revised = dict(candidate)
        revised["why_brand"] = (
            candidate["why_brand"] + f"（v{payload['round_index'] + 1}：已纳入基酒替换验证前置）"
        )
        revised["risks"] = [*candidate["risks"], "替换实测通过前不作专属主张"]
        return SpecificityRevisionBatchV2(
            run_id=audit_batch["run_id"],
            round_index=audit_batch["round_index"],
            revisions=[
                SpecificityCandidateRevisionV2(
                    candidate_id=candidate["candidate_id"],
                    from_version=audit_batch["candidate_versions"][candidate["candidate_id"]],
                    to_version=audit_batch["candidate_versions"][candidate["candidate_id"]] + 1,
                    responses=[
                        SpecificityFindingResponseV2(
                            finding_id=finding["finding_id"],
                            disposition="ACCEPT",
                            reason="证据支持，已收窄主张",
                        )
                        for finding in non_note
                    ],
                    revised_candidate=revised,
                    field_diffs=[
                        SpecificityEvidenceBindingV2(
                            field="why_brand",
                            before=candidate["why_brand"],
                            after=revised["why_brand"],
                            source_finding_ids=[non_note[0]["finding_id"]],
                            comment_evidence_ids=["C-003"],
                            reason="按第一轮实质 finding 收窄主张并纳入验证前置",
                        )
                    ],
                    unresolved_finding_ids=[],
                    new_brand_facts=[],
                )
            ],
        )


def _write_round_result(attempt_dir: Path, candidate_id: str, round_index: int, batch: SpecificityAuditBatch) -> None:
    result_path = (
        attempt_dir
        / "candidates"
        / candidate_id
        / "rounds"
        / f"round-{round_index:02d}"
        / "auditor_result.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(json.loads(batch.model_dump_json()), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def test_pressure_loop_offline_then_api_full_run(tmp_path: Path) -> None:
    atoms, records = _atoms_and_records()
    candidate = _candidate()
    attempt_dir = tmp_path / "attempt"

    # 离线代跑模式：无审计者 → 挂起并落盘任务包
    pending = run_pressure_loop(
        run_id=RUN_ID,
        attempt_dir=attempt_dir,
        candidates=[candidate],
        evidence_atoms=atoms,
        records_by_id=records,
        deepseek_client=FakeDeepseekClient(),
        auditor=None,
    )
    assert not pending.all_terminal
    assert pending.states["CAND-P1"]["status"] == "AWAITING_AUDITOR"
    task_path = attempt_dir / "candidates" / "CAND-P1" / "rounds" / "round-01" / "auditor_task.json"
    assert task_path.is_file()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    assert task["round_index"] == 1
    assert {row["comment_id"] for row in task["evidence"]} == {"C-001", "C-002", "C-003"}

    # 回填第一轮结果 → DeepSeek 修订 → 第二轮任务包挂起
    _write_round_result(attempt_dir, "CAND-P1", 1, _audit_batch(1, 1, material=True))
    deepseek = FakeDeepseekClient()
    pending = run_pressure_loop(
        run_id=RUN_ID,
        attempt_dir=attempt_dir,
        candidates=[candidate],
        evidence_atoms=atoms,
        records_by_id=records,
        deepseek_client=deepseek,
        auditor=None,
    )
    assert pending.states["CAND-P1"]["status"] == "AWAITING_AUDITOR"
    assert deepseek.calls == 1
    state = json.loads((attempt_dir / "candidates" / "CAND-P1" / "state.json").read_text(encoding="utf-8"))
    assert state["current_version"] == 2
    assert state["rounds"][0]["revision_diffs"][0]["field"] == "why_brand"

    # 回填第二轮全 NOTE 结果 → READY 终态
    _write_round_result(attempt_dir, "CAND-P1", 2, _audit_batch(2, 2, material=False))
    result = run_pressure_loop(
        run_id=RUN_ID,
        attempt_dir=attempt_dir,
        candidates=[candidate],
        evidence_atoms=atoms,
        records_by_id=records,
        deepseek_client=deepseek,
        auditor=None,
    )
    assert result.all_terminal
    assert result.states["CAND-P1"]["status"] == "READY"
    final = json.loads(
        (attempt_dir / "candidates" / "CAND-P1" / "final_candidate.json").read_text(encoding="utf-8")
    )
    assert final["status"] == "READY"
    assert final["version"] == 2
    assert final["candidate"]["why_brand"].endswith("（v2：已纳入基酒替换验证前置）")


def test_pressure_loop_api_auditor_resolves_in_two_rounds(tmp_path: Path) -> None:
    atoms, records = _atoms_and_records()
    attempt_dir = tmp_path / "attempt-api"
    auditor = FakeApiAuditor()
    result = run_pressure_loop(
        run_id=RUN_ID,
        attempt_dir=attempt_dir,
        candidates=[_candidate()],
        evidence_atoms=atoms,
        records_by_id=records,
        deepseek_client=FakeDeepseekClient(),
        auditor=auditor,
    )
    assert result.all_terminal
    assert result.states["CAND-P1"]["status"] == "READY"
    assert auditor.calls == 2  # 第一轮实质风险 + 第二轮收敛确认
    assert (attempt_dir / "attempt_meta.json").is_file()
