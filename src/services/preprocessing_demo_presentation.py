"""Read-only presentation models for the preprocessing competition demo."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from src.schemas import FiveCandidateUnifiedEvaluationV2
from src.services.candidate_ranking import calculate_weighted_score


class PreprocessingDemoPresentationError(ValueError):
    """A frozen preprocessing artifact cannot be projected safely."""


@dataclass(frozen=True, slots=True)
class PreprocessingStageDetailView:
    key: str
    label: str
    input_summary: str
    action_summary: str
    change_summary: str
    output_summary: str
    metrics: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class FoundationOpportunityView:
    candidate_id: str
    title: str
    argument: str
    evaluation: str
    original_score: float
    original_rank: int
    evidence_summary: str
    validation_risk: str


@dataclass(frozen=True, slots=True)
class PreprocessingDemoView:
    stages: tuple[PreprocessingStageDetailView, ...]
    opportunities: tuple[FoundationOpportunityView, ...]


def _object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreprocessingDemoPresentationError(f"正式展示产物缺失：{path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreprocessingDemoPresentationError(f"正式展示产物不可读取：{path}") from exc
    if not isinstance(value, dict):
        raise PreprocessingDemoPresentationError(f"正式展示产物必须是对象：{path}")
    return value


def _positive_int(value: object, *, label: str, path: Path) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise PreprocessingDemoPresentationError(f"{label} 不是有效数量：{path}")
    return value


def _candidate_set(validation_root: Path) -> list[dict[str, Any]]:
    payload = _object(validation_root / "canonical_input.json")
    candidates = payload.get("candidate_set")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise PreprocessingDemoPresentationError("基础机会必须恰好包含五条")
    if not all(isinstance(item, dict) for item in candidates):
        raise PreprocessingDemoPresentationError("基础机会格式无效")
    return candidates


def _opportunities(validation_root: Path) -> tuple[FoundationOpportunityView, ...]:
    evaluation_path = validation_root / "evaluation" / "result.json"
    try:
        evaluation = FiveCandidateUnifiedEvaluationV2.model_validate_json(
            evaluation_path.read_bytes(), strict=True
        )
    except (OSError, ValueError) as exc:
        raise PreprocessingDemoPresentationError(
            f"正式五候选评价不可读取：{evaluation_path}"
        ) from exc
    by_id = {item.candidate_id: item for item in evaluation.evaluations}
    rank_by_id = {
        candidate_id: index + 1
        for index, candidate_id in enumerate(evaluation.ranked_candidate_ids)
    }
    views: list[FoundationOpportunityView] = []
    for candidate in _candidate_set(validation_root):
        candidate_id = candidate.get("candidate_id")
        title = candidate.get("title")
        if not isinstance(candidate_id, str) or candidate_id not in by_id:
            raise PreprocessingDemoPresentationError("基础机会与正式评价候选不一致")
        if not isinstance(title, str) or not title:
            raise PreprocessingDemoPresentationError(f"基础机会标题无效：{candidate_id}")
        argument = candidate.get("brand_opportunity") or candidate.get("draft_proposition")
        if not isinstance(argument, str) or not argument:
            raise PreprocessingDemoPresentationError(f"基础机会论证缺失：{candidate_id}")
        supporting = candidate.get("supporting_evidence")
        counter = candidate.get("counter_evidence")
        risks = candidate.get("risks")
        if not isinstance(supporting, list) or not isinstance(counter, list):
            raise PreprocessingDemoPresentationError(f"基础机会证据格式无效：{candidate_id}")
        if not isinstance(risks, list) or not all(isinstance(item, str) for item in risks):
            raise PreprocessingDemoPresentationError(f"基础机会风险格式无效：{candidate_id}")
        item = by_id[candidate_id]
        views.append(
            FoundationOpportunityView(
                candidate_id=candidate_id,
                title=title,
                argument=argument,
                evaluation=item.overall_assessment,
                original_score=float(calculate_weighted_score(item.scores)),
                original_rank=rank_by_id[candidate_id],
                evidence_summary=f"{len(supporting)} 条直接证据 · {len(counter)} 条反证",
                validation_risk=risks[0] if risks else "未记录额外风险",
            )
        )
    return tuple(views)


def load_preprocessing_demo(
    screened_root: str | Path,
    validation_root: str | Path,
) -> PreprocessingDemoView:
    """Project the official manifests and pre-pressure candidate snapshot."""

    screened = Path(screened_root)
    validation = Path(validation_root)
    dataset = _object(screened / "dataset_manifest.json")
    split = _object(screened / "split_manifest.json")
    annotation = _object(
        screened
        / "annotations"
        / "formal_v2_luna_max"
        / "attempt-0001"
        / "annotation_run_manifest.json"
    )
    prepared_manifest = _object(
        screened.parents[1] / "prepared_corpora" / "formal-v1" / "manifest.json"
    )

    record_count = _positive_int(dataset.get("record_count"), label="原始语料", path=screened)
    effective_count = _positive_int(
        dataset.get("effective_record_count"), label="有效语料", path=screened
    )
    status_counts = dataset.get("status_counts")
    assignments = split.get("assignments")
    if not isinstance(status_counts, dict) or not isinstance(assignments, list):
        raise PreprocessingDemoPresentationError("正式筛选或拆分摘要格式无效")
    split_counts: dict[str, int] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict) or not isinstance(assignment.get("split_role"), str):
            raise PreprocessingDemoPresentationError("正式拆分 assignment 格式无效")
        role = assignment["split_role"]
        split_counts[role] = split_counts.get(role, 0) + 1

    batch_ids = annotation.get("batch_ids")
    if not isinstance(batch_ids, list):
        raise PreprocessingDemoPresentationError("正式标注摘要格式无效")
    model_id = annotation.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise PreprocessingDemoPresentationError("正式标注模型缺失")
    if (
        annotation.get("dataset_version") != split.get("dataset_version")
        or annotation.get("dataset_sha256") != split.get("dataset_sha256")
    ):
        raise PreprocessingDemoPresentationError("正式标注与拆分清单数据身份不一致")
    annotation_input_count = sum(
        split_counts.get(role, 0)
        for role in ("ANALYSIS", "CHALLENGE_POOL", "HOLDOUT")
    )
    package_ids = prepared_manifest.get("package_ids")
    if not isinstance(package_ids, dict) or not all(
        role in package_ids for role in ("ANALYSIS", "CHALLENGE", "HOLDOUT")
    ):
        raise PreprocessingDemoPresentationError("正式冻结包边界不完整")

    stages = (
        PreprocessingStageDetailView(
            "validate",
            "数据校验",
            "比赛原始评论文件与来源清单",
            "校验字段合同、来源记录和稳定 ID",
            f"{record_count} 条原始语料完成结构校验",
            "生成可进入筛选的统一原始数据集",
            (("原始语料", str(record_count)),),
        ),
        PreprocessingStageDetailView(
            "clean",
            "清洗筛选",
            f"{record_count} 条结构化原始语料",
            "执行有效性筛选、异常标记和重复边界核对",
            f"{effective_count} 条进入有效集，其余记录保留审计状态",
            "形成冻结的 screened_v2 有效语料集",
            (
                ("有效语料", str(effective_count)),
                ("排除", str(status_counts.get("EXCLUDE", 0))),
                ("待复核", str(status_counts.get("REVIEW", 0))),
            ),
        ),
        PreprocessingStageDetailView(
            "split",
            "分层拆分",
            f"{effective_count} 条有效语料",
            "按冻结 split manifest 隔离分析、校准、挑战和留出集合",
            "四类集合边界互斥，ID 总量守恒",
            "生成 ANALYSIS / GOLD / CHALLENGE_POOL / HOLDOUT",
            tuple(
                (role, str(split_counts.get(role, 0)))
                for role in ("ANALYSIS", "GOLD", "CHALLENGE_POOL", "HOLDOUT")
            ),
        ),
        PreprocessingStageDetailView(
            "annotate",
            "AI 标注",
            "ANALYSIS、CHALLENGE_POOL 与 HOLDOUT 的正式标注输入",
            "按冻结 Prompt 执行 Evidence Routing 并严格导入结果",
            f"{annotation_input_count} 条结果通过 Schema、ID 和输入哈希校验",
            "形成可供叙事决策读取的结构化证据",
            (
                ("正式标注", str(annotation_input_count)),
                ("批次", str(len(batch_ids))),
                ("模型", model_id),
            ),
        ),
        PreprocessingStageDetailView(
            "freeze",
            "证据冻结",
            "严格导入后的结构化证据与集合边界",
            "封装正式语料包并校验跨包边界",
            "三类正式包完成冻结，可交接基础机会层",
            "生成五条基础机会并进入叙事压力测试",
            (("正式包", "ANALYSIS · CHALLENGE · HOLDOUT"),),
        ),
    )
    return PreprocessingDemoView(stages=stages, opportunities=_opportunities(validation))


__all__ = [
    "FoundationOpportunityView",
    "PreprocessingDemoPresentationError",
    "PreprocessingDemoView",
    "PreprocessingStageDetailView",
    "load_preprocessing_demo",
]
