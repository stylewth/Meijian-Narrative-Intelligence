"""干跑验证：脚本化 LLM 响应驱动 DecisionRunner 完整自由链路。

脚本响应按各服务的 user_prompt 动态绑定真实 ID 与原文引语，保证服务层
后校验（引语原文性、ID 绑定、数量合同）全部通过。生产路径不经过本文件；
生产仍通过 ``src.llm_client.LLMClient`` 调用真实模型。
"""

from __future__ import annotations

import json
from typing import Any

from src.schemas import (
    BrandFeedbackComparison,
    CandidateEvaluation,
    CandidateScores,
    CommentRecord,
    CorpusAnalysisResult,
    DiversityAssessment,
    EmotionalConflict,
    EvidenceAtom,
    ScoreItem,
    SourceReference,
    SourceType,
    StressCheckResult,
    StressCheckType,
    StressDecision,
    StressExecutionStatus,
)
from src.services.candidate_generation import _CandidateGenerationResult, _CandidateDraft
from src.services.candidate_ranking import _CandidateScoringResult
from src.services.decision_runner import DecisionRunner
from src.services.evidence_assessment import (
    EvidenceAssessmentProposal,
    _EvidenceAssessmentResult,
    _HoldoutAssessmentResult,
)
from src.services.evidence_routing import (
    _EvidenceAtomDraft,
    _EvidenceRoutingResponse,
    _SourceReferenceDraft,
)
from src.services.narrative_revision import (
    NarrativeRevisionProposal,
    _NarrativeRevisionResult,
)


def _comment(comment_id: str, text: str, platform: str) -> CommentRecord:
    return CommentRecord(
        comment_id=comment_id,
        sample_type="梅见反馈",
        raw_content=text,
        source_platform=platform,
        raw_id=f"raw-{comment_id}",
        source=SourceReference(
            source_id=f"src-{comment_id}",
            source_type=SourceType.USER_COMMENT,
            source_ref=f"raw-{comment_id}",
        ),
    )


def _atom(evidence_id: str, record: CommentRecord) -> EvidenceAtom:
    return EvidenceAtom(
        evidence_id=evidence_id,
        comment_id=record.comment_id,
        route="BRAND",
        experience_scope="ACTUAL_USE",
        evidence_grade="A",
        source=record.source,
        source_platform=record.source_platform,
        actual_use=True,
    )


def _comments(prefix: str, texts: list[str], platform: str) -> list[CommentRecord]:
    return [
        _comment(f"{prefix}-{index + 1:03d}", text, platform)
        for index, text in enumerate(texts)
    ]


def _atoms(records: list[CommentRecord], prefix: str) -> list[EvidenceAtom]:
    return [
        _atom(f"{prefix}-{index + 1:03d}", record)
        for index, record in enumerate(records)
    ]


BASELINE_TEXTS = [
    "梅见青梅酒度数低，闺蜜聚会喝不出压力，微醺刚好。",
    "朋友说梅见配冰气泡水很惊艳，青梅香气完全打开了。",
    "担心果酒太甜，但梅见入口清爽不齁，适合新手。",
    "竞品果子酒香精味重，梅见的青梅原果感明显更真实。",
    "约会小酌想要氛围感，梅见的东方梅酒包装很有记忆点。",
    "长辈聚会拿梅见招待，柔和口感老人也能接受。",
]
CHALLENGE_TEXTS = [
    "有人反馈梅见价格偏高，聚会批量购买有压力。",
    "夏天冰饮场景想要更便携的小瓶装。",
]
HOLDOUT_TEXTS = [
    "露营野餐带了一瓶梅见，轻负担又应景。",
    "同事团建选梅见，不喝酒的人也能接受低度果香。",
]
RELEASE_TEXTS = [
    "便利店随手就能买到梅见，补货很方便。",
    "调酒师用梅见做基底，出品稳定。",
]
RELEASE_2_TEXTS = [
    "梅见新包装开瓶更顺滑，不再洒漏。",
    "会员回购梅见成了固定习惯。",
]

BASELINE_RECORDS = _comments("c1", BASELINE_TEXTS, "小红书")
BASELINE_ATOMS = _atoms(BASELINE_RECORDS, "a1")
CHALLENGE_RECORDS = _comments("c2", CHALLENGE_TEXTS, "抖音")
CHALLENGE_ATOMS = _atoms(CHALLENGE_RECORDS, "a2")
HOLDOUT_RECORDS = _comments("c3", HOLDOUT_TEXTS, "哔哩哔哩")
HOLDOUT_ATOMS = _atoms(HOLDOUT_RECORDS, "a3")
RELEASE_RECORDS = _comments("c4", RELEASE_TEXTS, "小红书")
RELEASE_ATOMS = _atoms(RELEASE_RECORDS, "a4")
RELEASE_2_RECORDS = _comments("c5", RELEASE_2_TEXTS, "抖音")
RELEASE_2_ATOMS = _atoms(RELEASE_2_RECORDS, "a5")


def _quote(comment: dict[str, Any]) -> dict[str, str]:
    return {"comment_id": comment["comment_id"], "quote": comment["raw_content"][:6]}


def _conflict(conflict_id: str, comment: dict[str, Any]) -> EmotionalConflict:
    return EmotionalConflict(
        conflict_id=conflict_id,
        title=f"{conflict_id} 聚会微醺张力",
        user_need="聚会想微醺但怕失控",
        desired_state="轻松有氛围",
        rejected_state="喝醉失态",
        identity_need="懂生活的聚会组织者",
        main_concerns=["度数", "口感"],
        main_scenes=["朋友聚会"],
        supporting_evidence=[_quote(comment)],
        counter_evidence=[],
        counter_evidence_note="本批语料未发现反面证据",
        implication_for_meijian="以低度青梅果酒承接聚会微醺需求",
    )


class ScriptedLLMClient:
    """以合法样例响应各服务请求；仅用于开发验证，不进入生产路径。"""

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Any,
        evidence_catalog: dict[str, str] | None = None,
    ) -> Any:
        payload = json.loads(user_prompt)
        name = response_model.__name__
        builder = {
            "CorpusAnalysisResult": self._corpus_analysis,
            "DiversityAssessment": self._diversity,
            "_CandidateGenerationResult": self._generation,
            "_CandidateScoringResult": self._scoring,
            "StressCheckResult": self._stress,
            "_EvidenceAssessmentResult": self._incremental,
            "_HoldoutAssessmentResult": self._holdout,
            "_NarrativeRevisionResult": self._revision,
            "_EvidenceRoutingResponse": self._routing,
        }[name]
        return builder(payload)

    def _routing(self, payload: Any) -> _EvidenceRoutingResponse:
        comments = payload if isinstance(payload, list) else payload["comments"]
        drafts = [
            _EvidenceAtomDraft(
                comment_id=comment["comment_id"],
                route="BRAND",
                experience_scope="ACTUAL_USE",
                evidence_grade="A",
                ai_confidence=0.9,
                explanation="核心对象：青梅酒；体验：真实饮用；证据强度：直接表述。",
                source=_SourceReferenceDraft(
                    source_id=comment["source"]["source_id"],
                    source_type=comment["source"]["source_type"],
                    source_ref=comment["source"]["source_ref"],
                ),
            )
            for comment in comments
        ]
        return _EvidenceRoutingResponse(atoms=drafts)

    def _corpus_analysis(self, payload: Any) -> CorpusAnalysisResult:
        comments = payload if isinstance(payload, list) else payload["comments"]
        return CorpusAnalysisResult(
            emotional_conflicts=[
                _conflict("cf-weixun", comments[0]),
                _conflict("cf-xinnian", comments[3]),
            ],
            feedback_comparison=BrandFeedbackComparison(),
            main_scenes=["朋友聚会", "约会小酌"],
            data_limitations=["样本量有限"],
        )

    def _diversity(self, payload: Any) -> DiversityAssessment:
        conflict_count = len(payload["corpus_analysis"]["emotional_conflicts"])
        return DiversityAssessment(
            diversity_level="中等",
            distinct_conflict_count=conflict_count,
            evidence_density="充足",
            redundancy="低",
            contradictions=[],
            recommended_candidate_count=min(2, conflict_count),
            count_rationale="两个独立情绪冲突支撑两条候选叙事",
        )

    def _generation(self, payload: Any) -> _CandidateGenerationResult:
        conflicts = payload["emotional_conflicts"]
        comments = payload["original_evidence"]
        drafts = []
        for index, conflict in enumerate(conflicts[:2]):
            comment = comments[index]
            suffix = f"{index + 1:02d}"
            drafts.append(
                _CandidateDraft(
                    candidate_id=f"CAND-{suffix}",
                    supporting_conflict_ids=[conflict["conflict_id"]],
                    title=f"聚会微醺主线 {suffix}",
                    target_audience="聚会新手与组织者",
                    user_conflict=f"想微醺怕失控-{suffix}",
                    brand_opportunity=f"低度青梅场景-{suffix}",
                    why_meijian="青梅原果发酵",
                    competitor_difference="真实果感对香精感",
                    brand_role="氛围调节者",
                    draft_proposition=f"梅见让聚会微醺刚刚好（版本 {suffix}）",
                    main_scenes=["朋友聚会"],
                    content_theme="轻松真实",
                    supporting_evidence=[_quote(comment)],
                    counter_evidence=[],
                    counter_evidence_note="本批证据未构成反面",
                    risks=["证据规模有限"],
                )
            )
        return _CandidateGenerationResult(candidates=drafts)

    def _scoring(self, payload: Any) -> _CandidateScoringResult:
        evaluations = [
            CandidateEvaluation(
                candidate_id=candidate["candidate_id"],
                scores=CandidateScores(
                    evidence_strength=ScoreItem(score=82, rationale="多平台证据一致"),
                    emotional_tension=ScoreItem(score=80, rationale="张力明确"),
                    meijian_fit_and_exclusivity=ScoreItem(score=85, rationale="青梅资产强绑定"),
                    competitor_difference=ScoreItem(score=78, rationale="与香精果酒拉开差距"),
                    scene_conversion=ScoreItem(score=84, rationale="聚会场景可直接转化"),
                ),
                overall_assessment="证据与场景支撑充分",
            )
            for candidate in payload["candidates"]
        ]
        return _CandidateScoringResult(
            evaluations=evaluations,
            recommendation_reason="加权评分领先且场景可落地",
        )

    def _stress(self, payload: Any) -> StressCheckResult:
        attack_ids = [atom["evidence_id"] for atom in payload["attack_evidence"]]
        return StressCheckResult(
            check_type=StressCheckType(payload["check_type"]),
            execution_status=StressExecutionStatus.COMPLETED,
            decision=StressDecision.PASS,
            reference_ids=attack_ids[:1],
            rationale="攻击证据未推翻候选主张",
        )

    def _incremental(self, payload: Any) -> _EvidenceAssessmentResult:
        candidate_ids = sorted(payload.get("candidate_ids") or [])
        conflict_ids = sorted(payload.get("conflict_ids") or [])
        targets = [*candidate_ids, *conflict_ids]
        proposals = []
        for index, atom in enumerate(payload["evidence_atoms"]):
            if index == 0 and targets:
                proposals.append(
                    EvidenceAssessmentProposal(
                        evidence_id=atom["evidence_id"],
                        impact="SUPPORT",
                        target_id=targets[0],
                        reason="直接支撑首条主线的场景证据",
                    )
                )
            else:
                proposals.append(
                    EvidenceAssessmentProposal(
                        evidence_id=atom["evidence_id"],
                        impact="NEUTRAL",
                        target_id=None,
                        reason="与当前候选无直接冲突或支撑",
                    )
                )
        return _EvidenceAssessmentResult(proposals=proposals)

    def _holdout(self, payload: Any) -> _HoldoutAssessmentResult:
        candidate_id = sorted(payload["frozen_candidate_ids"])[0]
        impacts = [
            {"evidence_id": atom["evidence_id"], "candidate_id": candidate_id, "impact": "SUPPORT"}
            for atom in payload["evidence_atoms"]
        ]
        return _HoldoutAssessmentResult.model_validate({"impacts": impacts})

    def _revision(self, payload: Any) -> _NarrativeRevisionResult:
        candidate = payload["candidate"]
        atom_ids = [atom["evidence_id"] for atom in payload["incremental_evidence_atoms"]]
        return _NarrativeRevisionResult(
            revision=NarrativeRevisionProposal(
                candidate_id=candidate["candidate_id"],
                draft_proposition=candidate["draft_proposition"] + "（修订：补充便携场景）",
                main_scenes=[*candidate["main_scenes"], "便利店即饮"],
                competitor_difference=candidate["competitor_difference"],
                trigger_evidence_ids=atom_ids,
                reason="新增证据指向便携与价格敏感场景",
            )
        )


class _BaselineInputs:
    def __init__(self, records: list[CommentRecord], atoms: list[EvidenceAtom]) -> None:
        self.baseline_records = records
        self.baseline_atoms = atoms


def test_free_decision_chain_runs_end_to_end() -> None:
    runner = DecisionRunner(client=ScriptedLLMClient(), brand_facts=[])

    foundation = runner.build_baseline(
        _BaselineInputs(list(BASELINE_RECORDS), list(BASELINE_ATOMS))
    )
    assert len(foundation.candidates) == 2
    assert foundation.ranking.recommended_candidate_id is not None
    assert {item.candidate_id for item in foundation.stress_results} == {
        item.candidate.candidate_id for item in foundation.ranking.ranked_candidates
    }

    foundation = runner.apply_challenge(foundation, CHALLENGE_ATOMS)
    assert foundation.challenge_applied is True
    assert len(foundation.impacts) == len(CHALLENGE_ATOMS)

    selected = [
        item.candidate.candidate_id
        for item in foundation.ranking.ranked_candidates[:2]
    ]
    foundation = runner.select_narratives(
        foundation, selected, primary_candidate_id=selected[0]
    )
    assert foundation.primary_candidate_id == selected[0]

    foundation = runner.validate_holdout(foundation, HOLDOUT_ATOMS)
    assert foundation.holdout_validated is True
    assert len(foundation.holdout_impacts) == len(HOLDOUT_ATOMS)

    foundation = runner.freeze_original_snapshot(foundation)
    assert foundation.frozen is True

    checkpoint_one = runner.release_batch(
        foundation, batch_index=1, atoms=RELEASE_ATOMS
    )
    assert checkpoint_one.release_index == 1
    assert checkpoint_one.candidates

    runner.release_batch(foundation, batch_index=2, atoms=RELEASE_2_ATOMS)
    checkpoint_two = runner.checkpoints[-1]
    assert checkpoint_two.release_index == 2

    assert [item.checkpoint_id for item in runner.checkpoints] == [
        "checkpoint-01",
        "checkpoint-02",
    ]


def test_free_pipeline_from_import_to_decision() -> None:
    """自定义语料从导入到决策基线的自由全链：拆分 → 在线标注 → 发布 → 装载 → 基线。"""

    import shutil
    import tempfile
    from pathlib import Path

    from src.dataset_split import DATASET_VERSION
    from src.services.prepared_corpus import (
        load_prepared_corpus,
        publish_prepared_corpus,
    )
    from src.ui.custom_decision_workspace import CustomRunInputs
    from src.ui.preprocessing_workspace import (
        _build_custom_split_manifest,
        _build_prepared_package_online,
        _make_annotation_manifest,
        _normalize_imported_records,
        annotate_records_online,
    )

    workspace = Path(tempfile.mkdtemp(prefix="free-pipeline-"))
    try:
        published_dir = workspace / "prepared_corpora"
        published_dir.mkdir(parents=True, exist_ok=True)
        import_records = [
            *BASELINE_RECORDS,
            *CHALLENGE_RECORDS,
            *HOLDOUT_RECORDS,
            *RELEASE_RECORDS,
            *RELEASE_2_RECORDS,
        ]
        raw_records = [comment.model_copy(update={"source": None}) for comment in import_records]
        records = _normalize_imported_records(raw_records)
        assert all(record.source is not None for record in records)
        assert all(record.screening_status is not None for record in records)

        dataset_sha256 = "a" * 64
        split_manifest = _build_custom_split_manifest(
            records,
            dataset_sha256=dataset_sha256,
            dataset_version=DATASET_VERSION,
        )
        assert {item.split_role.value for item in split_manifest.assignments} == {"ANALYSIS"}

        atoms, batch_ids = annotate_records_online(
            client=ScriptedLLMClient(), analysis_records=records
        )
        assert {atom.comment_id for atom in atoms} == {record.comment_id for record in records}
        assert batch_ids == ["online-batch-01"]

        annotation_manifest = _make_annotation_manifest(
            dataset_sha256=dataset_sha256,
            dataset_version=DATASET_VERSION,
            batch_ids=batch_ids,
            result_sha256="b" * 64,
            model_id="scripted-model",
            reasoning_effort="none",
        )
        package = _build_prepared_package_online(
            records=records,
            split_manifest=split_manifest,
            annotation_manifest=annotation_manifest,
            evidence_atoms=atoms,
        )
        published = publish_prepared_corpus(published_dir, package)
        reloaded = load_prepared_corpus(published_dir, published.manifest.package_id)
        assert len(reloaded.package.evidence_atoms) == len(records)

        runner = DecisionRunner(client=ScriptedLLMClient(), brand_facts=[])
        foundation = runner.build_baseline(
            CustomRunInputs(
                list(reloaded.package.records), list(reloaded.package.evidence_atoms)
            )
        )
        assert len(foundation.candidates) == 2
        assert foundation.ranking.recommended_candidate_id is not None
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
