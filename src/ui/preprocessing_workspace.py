from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from html import escape
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Literal, Mapping, MutableMapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
import streamlit as st

from src.data_sources.competition_workbook_source import load_competition_workbook
from src.data_sources.file_source import load_tabular_file
from src.data_sources.feishu_bitable_source import load_feishu_bitable
from src.data_validator import validate_dataframe
from src.dataset_split import DATASET_VERSION, build_split_manifest, validate_split_manifest
from src.dataset_fingerprint import calculate_runtime_dataset_fingerprint
from src.prompt_loader import load_prompt_metadata
from src.schemas import (
    AnnotationRunManifest,
    CommentRecord,
    DatasetSplitManifest,
    DatasetSplitRole,
    EvidenceAtom,
    EvidenceGrade,
    EvidenceRoute,
    ExperienceScope,
    FormalPreparedCorpusGroupManifest,
    PackageValidationStatus,
    PreparedCorpusManifest,
    PreparedCorpusPackage,
    SampleType,
    SourceType,
)
from src.services.offline_annotation_import import (
    OfflineAnnotationImportResult,
    build_annotation_batches,
    import_offline_annotations,
)
from src.services.prepared_corpus import (
    canonical_json_bytes,
    list_prepared_corpora,
    package_digest,
    publish_prepared_corpus,
    sha256_bytes,
)
from src.services.formal_corpus_builder import (
    _annotation_manifest_identity,
    _group_sha256,
    _stable_ids_sha256,
)
from src.services.preprocessing_demo_presentation import (
    FoundationOpportunityView,
    PreprocessingStageDetailView,
    load_preprocessing_demo,
)
from src.ui.workspace_shell import Workspace, complete_workspace


OFFICIAL_REPLAY_MODE = "官方运行回放"
NEW_DATA_MODE = "新数据处理"
# Compatibility names keep existing imports and callers on the real path.
DEMO_MODE = OFFICIAL_REPLAY_MODE

_FOUNDATION_EVALUATION_TRANSLATIONS = {
    "EC-01-兑饮比例卡": (
        "Persuasive, evidence-rich drinking-pain point and useful safety boundaries, but the exact formulas are not yet validated and the differentiation from general mix guidance is unconfirmed, leaving real-world conversion deferred.",
        "直饮痛点有说服力、证据充足，安全边界也较完整；但具体配方尚未验证，和通用兑饮建议的差异也未确认，因此真实转化仍需后续验证。",
    ),
    "EC-02-火锅局自主饮酒": (
        "Strong emotional hook and a refreshing social-table concept, but the weakest evidence base, asset confirmation, and differentiation make it the least ready to drive measurable conversion.",
        "情绪钩子强，餐桌社交概念也新鲜；但证据基础、素材确认和差异化均是五个候选中最弱的，暂不适合直接推动可衡量转化。",
    ),
    "EC-03-双容量双剧本": (
        "Well-grounded in direct capacity-related user evidence and clear content scripts, with honest competitor boundary; it ranks below the two strongest candidates because availability and exclusivity are conditional.",
        "容量相关用户证据直接，内容脚本清晰，也诚实保留了竞品边界；但产品是否持续在售、是否具备专属度仍有条件限制，因此排在两个更强候选之后。",
    ),
    "EC-ADD-01-口味图鉴": (
        "Best combination of direct consumer evidence, clear scene actionability, and Meijian-specific flavor-choice insight; execution risk lies in unbuilt assets and unverified competitor comparison.",
        "直接消费者证据、场景可执行性和梅见专属的口味选择洞察结合最好；主要执行风险在于所需资产尚未搭建、竞品对比尚未核实。",
    ),
    "EC-ADD-02-梅见溯源记": (
        "Powerful emotional and brand-fit case for answering distrust, and a strong second-place candidate; it trails only because the foundational traceability facts and delivery chain are still unverified.",
        "回应消费者不信任的情绪力量和品牌契合度都很强，是稳固的第二候选；之所以未排第一，是因为追溯基础事实与交付链路仍未核实。",
    ),
}


def _localized_foundation_evaluation(item: FoundationOpportunityView) -> str:
    localized = _FOUNDATION_EVALUATION_TRANSLATIONS.get(item.candidate_id)
    if localized is None:
        raise ValueError(f"基础机会缺少中文展示映射：{item.candidate_id}")
    expected_source, translated = localized
    if item.evaluation != expected_source:
        raise ValueError(f"基础机会冻结评价已变化：{item.candidate_id}")
    return translated
REAL_MODE = NEW_DATA_MODE
PREPROCESSING_MODES = (OFFICIAL_REPLAY_MODE, NEW_DATA_MODE)
_ROUTING_PROMPT_METADATA = load_prompt_metadata("evidence_routing")
PROMPT_VERSION = _ROUTING_PROMPT_METADATA.version
PROMPT_SHA256 = _ROUTING_PROMPT_METADATA.sha256
ANNOTATION_VERSION = "annotation_sol_high_v1"
ANNOTATION_MODEL_ID = "gpt-5.6-sol"
ANNOTATION_REASONING_EFFORT = "high"
_PREPROCESSING_STATE_PREFIX = "preprocessing_"
_REAL_STATE_KEYS = (
    "preprocessing_records",
    "preprocessing_dataset_sha256",
    "preprocessing_dataset_version",
    "preprocessing_source_name",
    "preprocessing_split_manifest",
    "preprocessing_batches",
    "preprocessing_batch_index",
    "preprocessing_exported_batch_ids",
    "preprocessing_last_export_payload",
    "preprocessing_annotation_manifest",
    "preprocessing_import_result",
    "preprocessing_error",
    "preprocessing_published_path",
)


@dataclass(frozen=True)
class PreprocessingWorkspaceView:
    mode: str
    active_manifest: DatasetSplitManifest | None
    batch_rows: list[dict[str, object]]
    can_export_next_batch: bool
    can_import_annotations: bool
    can_publish: bool
    blocking_reason: str | None


@dataclass(frozen=True, slots=True)
class ProcessingStageView:
    key: str
    status: Literal["ready", "waiting"]
    ready: bool
    can_complete: bool
    label: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class OfficialReplayStageView:
    key: str
    label: str
    duration_ms: int
    ready: bool
    status: Literal["ready", "waiting"]
    reason: str | None = None
    summary: Mapping[str, object] = MappingProxyType({})


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _DatasetManifest(_StrictModel):
    manifest_type: Literal["ScreenedCompetitionDatasetManifest"]
    schema_version: Literal["screened_competition_dataset_v2"]
    dataset_version: Literal["screened_v2"]
    record_count: int = Field(gt=0)
    effective_record_count: int = Field(gt=0)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    screening_result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status_counts: dict[str, int]


class _SourceManifest(_StrictModel):
    dataset_version: Literal["raw_competition_v1"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    imported_at: datetime
    source_filename: str = Field(min_length=1)
    source_sheet: str = Field(min_length=1)
    record_count: int = Field(gt=0)


_OFFICIAL_STAGE_KEYS = ("validate", "clean", "split", "annotate", "freeze")
_OFFICIAL_STAGE_DURATIONS = {
    "validate": 5000,
    "clean": 5500,
    "split": 6000,
    "annotate": 6500,
    "freeze": 7000,
}
_OFFICIAL_ROOT_FILES = (
    "dataset_manifest.json",
    "screening_manifest.json",
    "source_manifest.json",
    "split_manifest.json",
    "type_mapping.json",
)
_ANNOTATION_RELATIVE_ROOT = Path(
    "annotations/formal_v2_luna_max/attempt-0001"
)


def _official_root(root: str | Path) -> Path:
    base = Path(root)
    candidates = (
        base,
        base / "screened_v2",
        base / "data" / "competition" / "screened_v2",
        base / "competition" / "screened_v2",
    )
    for candidate in candidates:
        if (candidate / "dataset_manifest.json").is_file():
            return candidate
    return candidates[2]


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"official artifact must be a JSON object: {path.name}")
    return payload


def _load_official_root_manifests(
    official_root: Path,
) -> tuple[_DatasetManifest, AnnotationRunManifest, _SourceManifest, DatasetSplitManifest, dict[str, str]]:
    paths = {name: official_root / name for name in _OFFICIAL_ROOT_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"missing official artifact: {missing[0]}")

    dataset = _DatasetManifest.model_validate_json(
        paths["dataset_manifest.json"].read_bytes(), strict=True
    )
    screening = AnnotationRunManifest.model_validate_json(
        paths["screening_manifest.json"].read_bytes(), strict=True
    )
    source = _SourceManifest.model_validate_json(
        paths["source_manifest.json"].read_bytes(), strict=True
    )
    split = DatasetSplitManifest.model_validate_json(
        paths["split_manifest.json"].read_bytes(), strict=True
    )
    type_mapping = TypeAdapter(dict[str, str]).validate_python(
        _json_object(paths["type_mapping.json"]), strict=True
    )
    for value in type_mapping.values():
        SampleType(value)

    if dataset.source_sha256 != source.source_sha256:
        raise ValueError("dataset and source manifest SHA-256 do not match")
    if dataset.screening_result_sha256 != screening.result_sha256:
        raise ValueError("dataset and screening manifest SHA-256 do not match")
    if dataset.dataset_sha256 != split.dataset_sha256:
        raise ValueError("dataset and split manifest SHA-256 do not match")
    if dataset.effective_record_count != len(split.assignments):
        raise ValueError("dataset effective record count does not match split assignments")
    if len({item.raw_id for item in split.assignments}) != len(split.assignments):
        raise ValueError("split manifest contains duplicate stable IDs")
    if sum(dataset.status_counts.values()) != dataset.record_count:
        raise ValueError("dataset status counts do not match record count")
    if dataset.status_counts.get("KEEP") != dataset.effective_record_count:
        raise ValueError("dataset KEEP count does not match effective record count")
    return dataset, screening, source, split, type_mapping


def _prepared_root_for(official_root: Path, supplied_root: str | Path) -> Path:
    base = Path(supplied_root)
    candidates = (
        base / "data" / "prepared_corpora" / "formal-v1",
        official_root.parent.parent.parent / "data" / "prepared_corpora" / "formal-v1",
        official_root.parent.parent.parent.parent / "prepared_corpora" / "formal-v1",
    )
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    return candidates[0]


def _official_annotation_manifest(
    official_root: Path,
    split: DatasetSplitManifest,
) -> AnnotationRunManifest:
    annotation_root = official_root / _ANNOTATION_RELATIVE_ROOT
    run_manifest = AnnotationRunManifest.model_validate_json(
        (annotation_root / "annotation_run_manifest.json").read_bytes(), strict=True
    )
    if run_manifest.dataset_sha256 != split.dataset_sha256:
        raise ValueError("annotation manifest dataset SHA-256 does not match split manifest")
    if run_manifest.dataset_version != split.dataset_version:
        raise ValueError("annotation manifest dataset version does not match split manifest")
    return run_manifest


def _validate_release_group_metadata(
    prepared_root: Path,
    split: DatasetSplitManifest,
    run_manifest: AnnotationRunManifest,
) -> FormalPreparedCorpusGroupManifest:
    group_manifest = FormalPreparedCorpusGroupManifest.model_validate_json(
        (prepared_root / "manifest.json").read_bytes(), strict=True
    )
    if group_manifest.group_sha256 != _group_sha256(group_manifest):
        raise ValueError("formal prepared corpus group SHA-256 does not match manifest")
    if group_manifest.annotation_identity != _annotation_manifest_identity(run_manifest):
        raise ValueError("formal group annotation identity does not match annotation manifest")

    expected_counts = {"ANALYSIS": 249, "CHALLENGE": 60, "HOLDOUT": 60}
    expected_roles = {
        "ANALYSIS": DatasetSplitRole.ANALYSIS,
        "CHALLENGE": DatasetSplitRole.CHALLENGE_POOL,
        "HOLDOUT": DatasetSplitRole.HOLDOUT,
    }
    assignments_by_role = {
        role: [
            assignment.raw_id
            for assignment in split.assignments
            if assignment.split_role is split_role
        ]
        for role, split_role in expected_roles.items()
    }
    all_ids = [
        stable_id
        for role in ("ANALYSIS", "CHALLENGE", "HOLDOUT")
        for stable_id in assignments_by_role[role]
    ]
    if len(all_ids) != 369 or len(set(all_ids)) != 369:
        raise ValueError("formal release metadata must contain exactly 369 unique stable IDs")
    if group_manifest.all_stable_ids_sha256 != _stable_ids_sha256(all_ids):
        raise ValueError("formal group stable ID hash does not match split manifest")
    if set(group_manifest.analysis_baseline_ids) | set(group_manifest.analysis_release_ids) != set(
        assignments_by_role["ANALYSIS"]
    ):
        raise ValueError("formal group baseline/release IDs do not cover ANALYSIS")

    for role, expected_count in expected_counts.items():
        if len(assignments_by_role[role]) != expected_count:
            raise ValueError(f"formal group {role} split count does not match role boundary")
        package_id = group_manifest.package_ids[role]
        package_manifest = PreparedCorpusManifest.model_validate_json(
            (prepared_root / "packages" / package_id / "manifest.json").read_bytes(),
            strict=True,
        )
        if package_manifest.package_id != package_id:
            raise ValueError(f"formal group {role} package ID does not match metadata")
        if package_manifest.package_sha256 != group_manifest.package_sha256[role]:
            raise ValueError(f"formal group {role} package SHA-256 does not match metadata")
        if package_manifest.record_count != expected_count:
            raise ValueError(f"formal group {role} package count does not match metadata")
        if (
            package_manifest.dataset_sha256 != run_manifest.dataset_sha256
            or package_manifest.dataset_version != run_manifest.dataset_version
            or package_manifest.annotation_model_id != run_manifest.model_id
            or package_manifest.annotation_reasoning_effort != run_manifest.reasoning_effort
            or package_manifest.annotation_version != run_manifest.annotation_version
            or package_manifest.annotation_prompt_version != run_manifest.prompt_version
            or package_manifest.annotation_prompt_sha256 != run_manifest.prompt_sha256
        ):
            raise ValueError(f"formal group {role} package annotation identity does not match")
    return group_manifest


def _official_stage(
    key: str,
    label: str,
    ready: bool,
    *,
    reason: str | None = None,
    summary: Mapping[str, object] | None = None,
) -> OfficialReplayStageView:
    return OfficialReplayStageView(
        key=key,
        label=label,
        duration_ms=_OFFICIAL_STAGE_DURATIONS[key],
        ready=ready,
        status="ready" if ready else "waiting",
        reason=reason,
        summary=MappingProxyType(dict(summary or {})),
    )


def build_official_replay_stages(root: str | Path) -> tuple[OfficialReplayStageView, ...]:
    """Return read-only stage views backed only by the official JSON artifacts.

    The function intentionally never looks inside annotation ``tasks/``. Every
    ready state is derived from strict manifest validation and published package
    validation, so replay controls cannot turn missing artifacts into success.
    """

    official_root = _official_root(root)
    stages: list[OfficialReplayStageView] = []
    try:
        dataset, screening, source, split, type_mapping = _load_official_root_manifests(
            official_root
        )
        stages.append(
            _official_stage(
                "validate",
                "校验",
                True,
                summary={"dataset_version": dataset.dataset_version, "record_count": dataset.record_count},
            )
        )
    except Exception as exc:
        reason = str(exc)
        return tuple(
            [
                _official_stage(key, {"validate": "校验", "clean": "清洗", "split": "拆分", "annotate": "标注", "freeze": "冻结"}[key], False, reason=reason)
                for key in _OFFICIAL_STAGE_KEYS
            ]
        )

    stages.append(
        _official_stage(
            "clean",
            "清洗",
            True,
            summary={"screening_result_sha256": screening.result_sha256, "source_filename": source.source_filename, "type_count": len(type_mapping)},
        )
    )
    stages.append(
        _official_stage(
            "split",
            "拆分",
            True,
            summary={"split_algorithm_version": split.split_algorithm_version, "assignment_count": len(split.assignments)},
        )
    )

    try:
        run_manifest = _official_annotation_manifest(official_root, split)
        annotated_roles = {
            DatasetSplitRole.ANALYSIS,
            DatasetSplitRole.CHALLENGE_POOL,
            DatasetSplitRole.HOLDOUT,
        }
        stages.append(
            _official_stage(
                "annotate",
                "标注",
                True,
                summary={
                    "model_id": run_manifest.model_id,
                    "input_count": sum(
                        assignment.split_role in annotated_roles
                        for assignment in split.assignments
                    ),
                    "batch_count": len(run_manifest.batch_ids),
                },
            )
        )
    except Exception as exc:
        reason = str(exc)
        stages.append(_official_stage("annotate", "标注", False, reason=reason))
        stages.append(_official_stage("freeze", "冻结", False, reason="等待正式标注导入产物"))
        return tuple(stages)

    try:
        prepared_root = _prepared_root_for(official_root, root)
        group_manifest = _validate_release_group_metadata(
            prepared_root,
            split,
            run_manifest,
        )
        stages.append(
            _official_stage(
                "freeze",
                "冻结",
                True,
                summary={
                    "group_id": group_manifest.group_id,
                    "package_ids": dict(group_manifest.package_ids),
                    "package_sha256": dict(group_manifest.package_sha256),
                },
            )
        )
    except Exception as exc:
        stages.append(_official_stage("freeze", "冻结", False, reason=str(exc)))
    return tuple(stages)


def build_official_stage_track_html(
    stages: tuple[OfficialReplayStageView, ...],
    *,
    current_index: int,
    completed_keys: set[str],
) -> str:
    """Build the compact five-card replay track from validated stage views."""

    if len(stages) != len(_OFFICIAL_STAGE_KEYS):
        raise ValueError("官方回放必须包含五个阶段")
    if not 0 <= current_index < len(stages):
        raise ValueError("current_index 不在官方回放阶段范围内")
    cards: list[str] = []
    for index, stage in enumerate(stages):
        state_classes = []
        if stage.key in completed_keys:
            state_classes.append("is-complete")
        if index == current_index:
            state_classes.append("is-current")
        if not stage.ready:
            state_classes.append("is-blocked")
        classes = " ".join(state_classes)
        state_label = "完成" if stage.key in completed_keys else ("就绪" if stage.ready else "等待")
        cards.append(
            f'<article class="mj-stage-card {classes}" data-stage-key="{escape(stage.key)}">'
            f'<span class="mj-stage-index">0{index + 1}</span>'
            f'<strong>{escape(stage.label)}</strong>'
            f'<small>{state_label} · {stage.duration_ms / 1000:.1f}s</small>'
            '<i aria-hidden="true"></i>'
            "</article>"
        )
    return '<section class="mj-stage-grid">' + "".join(cards) + "</section>"


def build_case_data_intake_html() -> str:
    """Present supported intake paths without pretending to import frozen case data."""

    return (
        "<style>"
        ".mj-case-intake{display:grid;grid-template-columns:minmax(15rem,.8fr) repeat(2,minmax(0,1fr));"
        "gap:.65rem;margin:.35rem 0 .7rem;padding:.7rem;border:1px solid #D8CCBE;background:#F8F3EC}"
        ".mj-case-intake>header{padding:.35rem .5rem;border-left:3px solid #8F2F4D}"
        ".mj-case-intake>header span{color:#8F2F4D;font-size:.62rem;font-weight:750;letter-spacing:.14em}"
        ".mj-case-intake>header h3{margin:.22rem 0!important;color:#352F2B!important;font-family:STZhongsong,'华文中宋',serif;font-size:1.05rem!important}"
        ".mj-case-intake>header p{margin:0;color:#756B64;font-size:.65rem;line-height:1.55}"
        ".mj-case-intake article{display:grid;grid-template-columns:auto 1fr;align-items:center;gap:.55rem;padding:.55rem .7rem;border:1px solid #D8CCBE;background:#FFFDF9}"
        ".mj-case-intake article i{display:grid;place-items:center;width:1.75rem;height:1.75rem;border-radius:50%;color:#FFF8EF;background:#365B4B;font-size:.58rem;font-style:normal;font-weight:750}"
        ".mj-case-intake article strong{display:block;color:#4D4540;font-size:.75rem}.mj-case-intake article small{color:#887D74;font-size:.6rem}"
        "@media(max-width:850px){.mj-case-intake{grid-template-columns:1fr}}"
        "</style>"
        '<section class="mj-case-intake"><header><span>数据接入 · 案例来源</span>'
        "<h3>先接入数据，再进入冻结流程</h3>"
        "<p>本次案例来源：XLSX · 484 条原始语料 · 数据已冻结</p></header>"
        "<article><i>XL</i><div><strong>XLSX</strong><small>本地结构化语料接入</small></div></article>"
        "<article><i>飞</i><div><strong>飞书多维表格</strong><small>系统支持的协同数据入口</small></div></article>"
        "</section>"
    )


def build_stage_detail_html(stage: PreprocessingStageDetailView) -> str:
    """Render one preprocessing change as a readable four-part workbench."""

    metrics = "".join(
        f"<li><strong>{escape(label)}</strong><span>{escape(value)}</span></li>"
        for label, value in stage.metrics
    )
    return (
        f'<section class="mj-stage-workbench" data-stage="{escape(stage.key)}">'
        '<header class="mj-stage-workbench__header">'
        '<span>当前步骤</span>'
        f'<h2>{escape(stage.label)}</h2>'
        "</header>"
        '<div class="mj-stage-workbench__grid">'
        '<article><small>输入</small>'
        f'<p>{escape(stage.input_summary)}</p></article>'
        '<article><small>处理动作</small>'
        f'<p>{escape(stage.action_summary)}</p></article>'
        '<article class="is-change"><small>发生变化</small>'
        f'<p>{escape(stage.change_summary)}</p><ul>{metrics}</ul></article>'
        '<article><small>阶段产出</small>'
        f'<p>{escape(stage.output_summary)}</p></article>'
        "</div></section>"
    )


def build_foundation_opportunities_html(
    opportunities: tuple[FoundationOpportunityView, ...],
) -> str:
    """Render the five pre-pressure opportunities without exposing raw artifacts."""

    if len(opportunities) != 5:
        raise ValueError("冻结结果必须恰好包含五条基础机会")
    cards: list[str] = []
    for item in opportunities:
        cards.append(
            '<article class="mj-opportunity-card">'
            '<header>'
            f'<span>基础机会 {item.original_rank:02d}</span>'
            f'<strong>{item.original_score:.1f}</strong>'
            "</header>"
            f'<h3>{escape(item.title)}</h3>'
            '<section><small>核心洞察</small>'
            f'<p>{escape(item.argument)}</p></section>'
            '<section><small>为什么值得验证</small>'
            f'<p>{escape(_localized_foundation_evaluation(item))}</p></section>'
            '<footer>'
            f'<span>{escape(item.evidence_summary)}</span>'
            f'<span>验证关注：{escape(item.validation_risk)}</span>'
            "</footer></article>"
        )
    return '<section class="mj-opportunity-grid">' + "".join(cards) + "</section>"


def derive_processing_stage(state: Mapping[str, Any]) -> ProcessingStageView:
    """Derive the new-data stage from real in-session processing state."""

    labels = {
        "validate": "校验",
        "clean": "清洗",
        "split": "拆分",
        "annotate": "标注",
        "freeze": "冻结",
    }
    if state.get("preprocessing_split_manifest") is None:
        if not state.get("preprocessing_records"):
            return ProcessingStageView("validate", "waiting", False, False, labels["validate"], "等待 Excel/CSV 或飞书数据")
        return ProcessingStageView("clean", "waiting", False, False, labels["clean"], "等待字段校验与清洗")
    batches = list(state.get("preprocessing_batches") or [])
    exported = set(state.get("preprocessing_exported_batch_ids") or [])
    batch_ids = {str(batch.get("batch_id")) for batch in batches if isinstance(batch, dict)}
    if not batches or exported != batch_ids:
        return ProcessingStageView("split", "waiting", False, False, labels["split"], "等待拆分并导出全部任务包")
    imported = state.get("preprocessing_import_result")
    annotation_manifest = state.get("preprocessing_annotation_manifest")
    if not isinstance(imported, OfflineAnnotationImportResult) or not isinstance(annotation_manifest, AnnotationRunManifest):
        return ProcessingStageView("annotate", "waiting", False, False, labels["annotate"], "等待严格校验通过的 annotation import")
    published_path = state.get("preprocessing_published_path")
    if not published_path or not Path(str(published_path)).is_dir():
        return ProcessingStageView("freeze", "waiting", False, False, labels["freeze"], "等待发布 PreparedCorpusPackage")
    return ProcessingStageView("freeze", "ready", True, True, labels["freeze"])


def apply_official_replay_action(state: MutableMapping[str, Any], action: str) -> None:
    """Apply a replay control without touching any official artifact."""

    if action not in {"start", "pause", "skip", "replay"}:
        raise ValueError(f"未知官方回放操作：{action}")
    completed = list(state.get("preprocessing_replay_completed_steps") or [])
    current = int(state.get("preprocessing_replay_step", 0) or 0)
    if action == "start":
        state["preprocessing_replay_running"] = True
    elif action == "pause":
        state["preprocessing_replay_running"] = False
    elif action == "replay":
        state["preprocessing_replay_running"] = True
        state["preprocessing_replay_step"] = 0
        state["preprocessing_replay_completed_steps"] = []
    elif action == "skip":
        if current < len(_OFFICIAL_STAGE_KEYS):
            if current not in completed:
                completed.append(current)
            state["preprocessing_replay_completed_steps"] = completed
            state["preprocessing_replay_step"] = min(current + 1, len(_OFFICIAL_STAGE_KEYS) - 1)
        state["preprocessing_replay_running"] = False


def advance_official_replay(
    state: MutableMapping[str, Any],
    stages: tuple[OfficialReplayStageView, ...],
) -> bool:
    """Advance exactly one validated stage while the read-only replay is running."""

    if not state.get("preprocessing_replay_running", False):
        return False
    if len(stages) != len(_OFFICIAL_STAGE_KEYS):
        raise ValueError("官方回放必须包含五个阶段")
    current = int(state.get("preprocessing_replay_step", 0) or 0)
    if not 0 <= current < len(stages):
        raise ValueError("preprocessing_replay_step 不在阶段范围内")
    if not stages[current].ready:
        state["preprocessing_replay_running"] = False
        return False

    completed = list(state.get("preprocessing_replay_completed_steps") or [])
    if current not in completed:
        completed.append(current)
    completed.sort()
    state["preprocessing_replay_completed_steps"] = completed
    if current == len(stages) - 1:
        state["preprocessing_replay_running"] = False
    else:
        state["preprocessing_replay_step"] = current + 1
    return True


def complete_official_replay(
    state: MutableMapping[str, Any], stages: tuple[OfficialReplayStageView, ...]
) -> bool:
    """Complete preprocessing only after the five official stages are replayed."""

    completed = set(state.get("preprocessing_replay_completed_steps") or [])
    if len(stages) != 5 or not all(stage.ready for stage in stages) or completed != set(range(5)):
        return False
    complete_workspace(state, Workspace.PREPROCESSING)
    return True


def clear_preprocessing_ui_state(state: MutableMapping[str, Any]) -> None:
    """Clear only preprocessing-owned ephemeral state when leaving the workspace."""

    for key in list(state):
        if key.startswith(_PREPROCESSING_STATE_PREFIX) and key != "preprocessing_mode":
            del state[key]


def _as_manifest(value: object) -> DatasetSplitManifest | None:
    if isinstance(value, DatasetSplitManifest):
        return value
    if value is None:
        return None
    return DatasetSplitManifest.model_validate_json(
        canonical_json_bytes(value), strict=True
    )


def build_workspace_view(
    state: MutableMapping[str, Any], *, mode: str | None = None
) -> PreprocessingWorkspaceView:
    active_manifest = _as_manifest(state.get("preprocessing_split_manifest"))
    batches = list(state.get("preprocessing_batches") or [])
    batch_index = int(state.get("preprocessing_batch_index", 0) or 0)
    exported_ids = set(state.get("preprocessing_exported_batch_ids") or [])
    batch_rows = list(batches[batch_index]["items"]) if batch_index < len(batches) else []
    can_export = active_manifest is not None and batch_index < len(batches)
    can_import = bool(batches) and len(exported_ids) == len(batches)
    imported = state.get("preprocessing_import_result")
    can_publish = can_import and isinstance(imported, OfflineAnnotationImportResult)

    blocking_reason = state.get("preprocessing_error")
    if blocking_reason is None:
        if active_manifest is None:
            blocking_reason = "请先导入有效的 484 条比赛语料并生成 split manifest。"
        elif not batches:
            blocking_reason = "当前 split manifest 没有可导出的 ANALYSIS 任务包。"
        elif not can_import:
            blocking_reason = "请先导出全部当前允许集合任务包，再导入严格 JSON 结果。"
        elif not can_publish:
            blocking_reason = "等待完整、严格校验通过的 AI 标注 JSON；当前不可发布。"

    return PreprocessingWorkspaceView(
        mode=mode or str(state.get("preprocessing_mode") or DEMO_MODE),
        active_manifest=active_manifest,
        batch_rows=batch_rows,
        can_export_next_batch=can_export,
        can_import_annotations=can_import,
        can_publish=can_publish,
        blocking_reason=blocking_reason if mode == REAL_MODE or mode is None else None,
    )


def _source_bytes(uploaded_file: Any) -> bytes:
    if uploaded_file is None or not hasattr(uploaded_file, "getvalue"):
        raise ValueError("请先上传 XLSX 或 CSV 文件")
    content = uploaded_file.getvalue()
    if not isinstance(content, bytes) or not content:
        raise ValueError("上传文件为空")
    return content


def _load_uploaded_dataset(uploaded_file: Any):
    filename = str(getattr(uploaded_file, "name", "uploaded.csv"))
    content = _source_bytes(uploaded_file)
    suffix = Path(filename).suffix.lower()
    if suffix == ".xlsx":
        with tempfile.NamedTemporaryFile(suffix=".xlsx") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            imported = load_competition_workbook(temporary_file.name)
        return imported.dataset, imported.source_manifest.source_sha256, imported.source_manifest.dataset_version
    if suffix == ".csv":
        frame = load_tabular_file(BytesIO(content), filename)
        dataset = validate_dataframe(frame)
        return dataset, hashlib.sha256(content).hexdigest(), DATASET_VERSION
    raise ValueError("只支持 .xlsx 或 .csv；.xlsx 必须符合比赛原始语料合同")


def _store_new_dataset(
    state: MutableMapping[str, Any],
    *,
    dataset: Any,
    dataset_sha256: str,
    dataset_version: str,
    source_name: str,
) -> tuple[int, int]:
    records = list(dataset.comments)
    split_manifest = build_split_manifest(
        records,
        dataset_sha256=dataset_sha256,
        dataset_version=dataset_version,
    )
    analysis_records = _analysis_records(records, split_manifest)
    batches = build_annotation_batches(
        analysis_records,
        dataset_sha256=dataset_sha256,
        batch_prefix="analysis",
    )
    state["preprocessing_records"] = records
    state["preprocessing_dataset_sha256"] = dataset_sha256
    state["preprocessing_dataset_version"] = dataset_version
    state["preprocessing_source_name"] = source_name
    state["preprocessing_split_manifest"] = split_manifest
    state["preprocessing_batches"] = batches
    state["preprocessing_batch_index"] = 0
    state["preprocessing_exported_batch_ids"] = []
    state["preprocessing_last_export_payload"] = None
    state["preprocessing_annotation_manifest"] = None
    state["preprocessing_import_result"] = None
    state["preprocessing_error"] = None
    return len(records), len(batches)


def _analysis_records(
    records: list[CommentRecord], manifest: DatasetSplitManifest
) -> list[CommentRecord]:
    analysis_ids = {
        assignment.raw_id
        for assignment in manifest.assignments
        if assignment.split_role is DatasetSplitRole.ANALYSIS
    }
    return [record for record in records if (record.raw_id or record.comment_id) in analysis_ids]


def _make_annotation_manifest(
    *,
    dataset_sha256: str,
    dataset_version: str,
    batch_ids: list[str],
    result_sha256: str = "0" * 64,
) -> AnnotationRunManifest:
    return AnnotationRunManifest(
        annotation_version=ANNOTATION_VERSION,
        model_id=ANNOTATION_MODEL_ID,
        reasoning_effort=ANNOTATION_REASONING_EFFORT,
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        prompt_version=PROMPT_VERSION,
        prompt_sha256=PROMPT_SHA256,
        batch_ids=batch_ids,
        result_sha256=result_sha256,
        generated_at=datetime.now(timezone.utc),
    )


def _task_package_bytes(
    batch: dict[str, object], manifest: AnnotationRunManifest
) -> bytes:
    payload = {
        "contract": "OFFLINE_ANNOTATION_TASK_PACKAGE",
        "manifest": manifest.model_dump(mode="json"),
        "batch": batch,
        "instructions": "仅填写 annotations 的 AI 字段；不得修改 input 中的原始字段。导入时必须提供完整结果包及真实 result_sha256。",
    }
    return canonical_json_bytes(payload)


def _annotation_payload(uploaded_file: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = _source_bytes(uploaded_file)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("AI 标注 JSON 必须是 UTF-8 且为合法 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("AI 标注 JSON 顶层必须是对象")
    manifest = payload.get("manifest")
    results = payload.get("results")
    if not isinstance(manifest, dict):
        raise ValueError("AI 标注 JSON 缺少 manifest 对象")
    if not isinstance(results, list) or not results:
        raise ValueError("AI 标注 JSON 缺少非空 results 数组")
    if not all(isinstance(item, dict) for item in results):
        raise ValueError("AI 标注 JSON results 必须全部是对象")
    return manifest, results


def _build_evidence_atoms(
    records: list[CommentRecord], imported: OfflineAnnotationImportResult
) -> list[EvidenceAtom]:
    records_by_id = {record.raw_id or record.comment_id: record for record in records}
    atoms: list[EvidenceAtom] = []
    for annotation in imported.annotations:
        record = records_by_id.get(annotation.raw_id)
        if record is None or record.source is None:
            raise ValueError(f"AI 标注引用了缺少来源的原始记录: {annotation.raw_id}")
        atoms.append(
            EvidenceAtom(
                evidence_id=f"EVIDENCE-{annotation.raw_id}",
                comment_id=record.comment_id,
                route=EvidenceRoute(annotation.route),
                experience_scope=ExperienceScope(annotation.experience_scope),
                evidence_grade=EvidenceGrade(annotation.evidence_grade),
                ai_confidence=annotation.ai_confidence,
                explanation=annotation.explanation,
                source=record.source,
                source_platform=record.source_platform,
                duplicate_group=record.duplicate_group,
                actual_use=record.actual_use,
            )
        )
    return atoms


def _build_prepared_package(
    *,
    records: list[CommentRecord],
    split_manifest: DatasetSplitManifest,
    annotation_manifest: AnnotationRunManifest,
    imported: OfflineAnnotationImportResult,
) -> PreparedCorpusPackage:
    validate_split_manifest(
        split_manifest,
        records,
        dataset_sha256=split_manifest.dataset_sha256,
    )
    evidence_atoms = _build_evidence_atoms(records, imported)
    package_id = (
        f"{split_manifest.dataset_version}-{split_manifest.dataset_sha256[:12]}-"
        f"{imported.result_sha256[:12]}"
    )
    artifact_values = {
        "records.json": canonical_json_bytes(records),
        "evidence_atoms.json": canonical_json_bytes(evidence_atoms),
        "split_manifest.json": canonical_json_bytes(split_manifest),
        "annotation_run_manifests.json": canonical_json_bytes([annotation_manifest]),
    }
    artifact_sha256 = {
        name: sha256_bytes(value) for name, value in artifact_values.items()
    }
    manifest = PreparedCorpusManifest(
        package_id=package_id,
        dataset_version=split_manifest.dataset_version,
        dataset_sha256=split_manifest.dataset_sha256,
        annotation_version=annotation_manifest.annotation_version,
        annotation_model_id=annotation_manifest.model_id,
        annotation_reasoning_effort=annotation_manifest.reasoning_effort,
        annotation_prompt_version=annotation_manifest.prompt_version,
        annotation_prompt_sha256=annotation_manifest.prompt_sha256,
        validation_status=PackageValidationStatus.PIPELINE_VALIDATED_ONLY,
        record_count=len(records),
        evidence_count=len(evidence_atoms),
        artifact_sha256=artifact_sha256,
        package_sha256=package_digest(artifact_sha256),
    )
    return PreparedCorpusPackage(
        manifest=manifest,
        records=records,
        evidence_atoms=evidence_atoms,
        split_manifest=split_manifest,
        annotation_run_manifests=[annotation_manifest],
    )


def _demo_audit_bytes(split_manifest: DatasetSplitManifest, prepared_root: Path) -> bytes:
    published = list_prepared_corpora(prepared_root)
    payload = {
        "contract": "PREGENERATED_PREPROCESSING_DEMO",
        "dataset_version": split_manifest.dataset_version,
        "split_manifest": split_manifest.model_dump(mode="json"),
        "gold_calibration": {
            "status": "NOT_PROVIDED",
            "note": "当前演示不包含官方 Gold 标注包，不得显示为已完成校准。",
        },
        "published_package_count": len(published),
        "model_called": False,
    }
    return canonical_json_bytes(payload)


def _load_demo_manifest(path: Path) -> DatasetSplitManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetSplitManifest.model_validate_json(
        canonical_json_bytes(payload), strict=True
    )


def _render_demo(split_manifest_path: Path, prepared_root: Path) -> None:
    try:
        manifest = _load_demo_manifest(split_manifest_path)
    except Exception as exc:
        st.error(f"PREGENERATED 演示合同加载失败：{exc}")
        return

    st.markdown("<span class='mode-seal'>PREGENERATED · 未调用模型</span>", unsafe_allow_html=True)
    st.info("演示结果只读展示：当前没有官方 Gold 标注包，也不会把不存在的标注结果显示为已完成。")
    st.caption(
        f"当前合同：dataset_version={manifest.dataset_version}；"
        f"split_manifest={manifest.split_algorithm_version}；"
        f"dataset_sha256={manifest.dataset_sha256}"
    )
    counts: dict[str, int] = {}
    for assignment in manifest.assignments:
        counts[assignment.split_role.value] = counts.get(assignment.split_role.value, 0) + 1
    st.write("当前已有 screened_v1 / split_manifest：" + "；".join(f"{key}={value}" for key, value in sorted(counts.items())))
    st.markdown("#### 查看 Gold 校准摘要")
    st.caption("官方 Gold 标注包：未提供；当前仅展示拆分合同，不代表 Gold 已校准或冻结。")
    st.markdown("#### 查看批次运行清单")
    st.caption("演示合同未包含可发布的 AI 标注批次运行结果。")
    st.markdown("#### 下载预处理审计包")
    st.button("导出下一批 Codex 任务包", disabled=True, key="demo_export_next_codex_batch")
    st.button("导入 AI 标注 JSON", disabled=True, key="demo_import_ai_annotation_json")
    st.button("发布 PreparedCorpusPackage", disabled=True, key="demo_publish_prepared_corpus")
    st.download_button(
        "下载预处理审计包",
        data=_demo_audit_bytes(manifest, prepared_root),
        file_name="preprocessing_audit_pregenerated.json",
        mime="application/json",
        key="download_preprocessing_audit",
    )


def _render_official_replay(split_manifest_path: Path) -> None:
    stages = build_official_replay_stages(split_manifest_path.parent)
    project_root = split_manifest_path.parent.parents[2]
    validation_root = (
        project_root
        / "data"
        / "competition"
        / "five_candidate_validation"
        / "official-20260813-five-candidate-validation-002"
    )
    demo_view = load_preprocessing_demo(split_manifest_path.parent, validation_root)
    state = st.session_state
    current = min(
        max(int(state.get("preprocessing_replay_step", 0) or 0), 0),
        len(stages) - 1,
    )
    completed = set(state.get("preprocessing_replay_completed_steps") or [])

    st.markdown(build_case_data_intake_html(), unsafe_allow_html=True)
    st.caption("只读官方运行回放：状态控制保存在当前会话，任何按钮都不会改写官方 JSON。")
    if not all(stage.ready for stage in stages):
        first_blocked = next(stage for stage in stages if not stage.ready)
        st.warning(f"官方运行回放未完成：{first_blocked.label}阶段等待真实产物。{first_blocked.reason or ''}")

    st.markdown(
        build_official_stage_track_html(
            stages,
            current_index=current,
            completed_keys={str(item) for item in completed},
        ),
        unsafe_allow_html=True,
    )
    current_stage = stages[current]
    st.markdown(
        build_stage_detail_html(demo_view.stages[current]),
        unsafe_allow_html=True,
    )

    controls = st.columns(4)
    with controls[0]:
        if st.button("自动播放", key="official_replay_start", use_container_width=True):
            apply_official_replay_action(state, "start")
            st.rerun()
    with controls[1]:
        if st.button("暂停", key="official_replay_pause", use_container_width=True):
            apply_official_replay_action(state, "pause")
            st.rerun()
    with controls[2]:
        if st.button("下一变化", key="official_replay_skip", use_container_width=True):
            apply_official_replay_action(state, "skip")
            st.rerun()
    with controls[3]:
        if st.button("重播", key="official_replay_replay", use_container_width=True):
            apply_official_replay_action(state, "replay")
            st.rerun()

    if complete_official_replay(state, stages):
        st.success("官方运行回放已完成；正式数据已冻结，可进入下一工作区。")
        with st.expander("查看冻结结果", expanded=False):
            st.caption("盲评前的五条基础机会：保留原始论证与评价，供后续压力测试对照。")
            st.markdown(
                build_foundation_opportunities_html(demo_view.opportunities),
                unsafe_allow_html=True,
            )
        if st.button("进入叙事压力测试", key="official_replay_enter_pressure", type="primary"):
            state["active_workspace"] = Workspace.PRESSURE_TEST.value
            st.rerun()

    with st.expander("高级操作", expanded=False):
        st.caption("高级操作仅影响当前会话回放位置，不会重跑、覆盖或修改任何官方产物。")
        if st.button("清除回放进度", key="official_replay_clear"):
            state.pop("preprocessing_replay_step", None)
            state.pop("preprocessing_replay_completed_steps", None)
            state.pop("preprocessing_replay_running", None)

    with st.expander("技术追溯", expanded=False):
        st.caption("案例展示默认收起。正式包已通过只读校验，技术标识仅用于审计追踪。")
        st.write(f"冻结组：{stages[-1].summary.get('group_id', '—')}")

    if state.get("preprocessing_replay_running", False):
        time.sleep(stages[current].duration_ms / 1000)
        advance_official_replay(state, stages)
        if complete_official_replay(state, stages):
            state["preprocessing_replay_running"] = False
        st.rerun()


def _render_real() -> None:
    state = st.session_state
    st.caption("真实链路：导入 → 拆分 → 导出 → 导入 → 发布")
    entry = st.radio(
        "数据入口",
        ["Excel/CSV", "飞书多维表格"],
        horizontal=True,
        key="preprocessing_new_data_entry",
    )
    if entry == "Excel/CSV":
        st.markdown("#### 1. 导入原始语料")
        uploaded_file = st.file_uploader(
            "上传比赛原始语料（XLSX / CSV）",
            type=["xlsx", "csv"],
            key="preprocessing_source_file",
        )
        if st.button(
            "导入并生成 split manifest",
            key="preprocessing_import_source",
            disabled=uploaded_file is None,
        ):
            clear_preprocessing_ui_state(state)
            try:
                dataset, dataset_sha256, dataset_version = _load_uploaded_dataset(uploaded_file)
                record_count, batch_count = _store_new_dataset(
                    state,
                    dataset=dataset,
                    dataset_sha256=dataset_sha256,
                    dataset_version=dataset_version,
                    source_name=str(getattr(uploaded_file, "name", "uploaded")),
                )
                st.success(f"已导入 {record_count} 条记录并生成真实 split manifest（{batch_count} 个批次）。")
            except Exception as exc:
                state["preprocessing_error"] = str(exc)
                st.error(f"原始语料导入或拆分失败：{exc}")
    else:
        st.markdown("#### 1. 导入飞书多维表格")
        feishu_url = st.text_input(
            "飞书多维表格链接",
            key="preprocessing_feishu_url",
        )
        if st.button(
            "读取飞书多维表格并生成 split manifest",
            key="preprocessing_import_feishu",
            disabled=not feishu_url,
        ):
            clear_preprocessing_ui_state(state)
            try:
                app_id = os.getenv("FEISHU_APP_ID", "")
                app_secret = os.getenv("FEISHU_APP_SECRET", "")
                if not app_id or not app_secret:
                    raise ValueError("缺少 FEISHU_APP_ID 或 FEISHU_APP_SECRET")
                dataset = load_feishu_bitable(
                    feishu_url,
                    app_id=app_id,
                    app_secret=app_secret,
                )
                dataset_sha256 = calculate_runtime_dataset_fingerprint(dataset.comments)
                record_count, batch_count = _store_new_dataset(
                    state,
                    dataset=dataset,
                    dataset_sha256=dataset_sha256,
                    dataset_version=DATASET_VERSION,
                    source_name=feishu_url,
                )
                st.success(f"已读取飞书真实记录 {record_count} 条并生成 split manifest（{batch_count} 个批次）。")
            except Exception as exc:
                state["preprocessing_error"] = str(exc)
                st.error(f"飞书多维表格导入失败：{exc}")

    split_manifest = _as_manifest(state.get("preprocessing_split_manifest"))
    if split_manifest is not None:
        batches = list(state.get("preprocessing_batches") or [])
        batch_index = int(state.get("preprocessing_batch_index", 0) or 0)
        exported_ids = set(state.get("preprocessing_exported_batch_ids") or [])
        st.markdown("#### 2. 拆分 split manifest")
        st.caption(
            f"dataset_version={split_manifest.dataset_version}；"
            f"dataset_sha256={split_manifest.dataset_sha256}；"
            f"当前允许集合=ANALYSIS，共 {sum(1 for item in split_manifest.assignments if item.split_role is DatasetSplitRole.ANALYSIS)} 条"
        )
        st.markdown("#### 3. 导出当前允许集合任务包")
        annotation_manifest = _make_annotation_manifest(
            dataset_sha256=split_manifest.dataset_sha256,
            dataset_version=split_manifest.dataset_version,
            batch_ids=[str(batch["batch_id"]) for batch in batches],
        )
        if batch_index < len(batches):
            current_batch = batches[batch_index]
            if st.button(
                "导出下一批 Codex 任务包",
                key="export_next_codex_batch",
                disabled=not build_workspace_view(state, mode=REAL_MODE).can_export_next_batch,
            ):
                state["preprocessing_exported_batch_ids"] = [
                    *state.get("preprocessing_exported_batch_ids", []),
                    current_batch["batch_id"],
                ]
                state["preprocessing_last_export_payload"] = _task_package_bytes(
                    current_batch, annotation_manifest
                )
                state["preprocessing_batch_index"] = batch_index + 1
                st.success(f"已生成任务包 {current_batch['batch_id']}，请下载后离线处理。")
        else:
            st.button("导出下一批 Codex 任务包", key="export_next_codex_batch", disabled=True)
        if state.get("preprocessing_last_export_payload"):
            st.download_button(
                "下载当前 Codex 任务包",
                data=state["preprocessing_last_export_payload"],
                file_name="codex_annotation_task.json",
                mime="application/json",
                key="download_codex_task",
            )
        st.caption(f"已导出批次：{len(exported_ids)} / {len(batches)}")

        st.markdown("#### 4. 导入 AI 标注 JSON")
        view = build_workspace_view(state, mode=REAL_MODE)
        result_file = st.file_uploader(
            "上传完整 AI 标注结果包（UTF-8 JSON）",
            type=["json"],
            key="preprocessing_annotation_file",
            disabled=not view.can_import_annotations,
        )
        if st.button(
            "导入 AI 标注 JSON",
            key="import_ai_annotation_json",
            disabled=not view.can_import_annotations or result_file is None,
        ):
            state["preprocessing_import_result"] = None
            state["preprocessing_error"] = None
            try:
                manifest_payload, results = _annotation_payload(result_file)
                imported = import_offline_annotations(
                    manifest_payload,
                    results,
                    dataset_sha256=split_manifest.dataset_sha256,
                    dataset_version=split_manifest.dataset_version,
                    prompt_metadata=_ROUTING_PROMPT_METADATA,
                    trusted_batch_inputs={
                        str(batch["batch_id"]): batch["items"]
                        for batch in batches
                    },
                )
                annotation_manifest = AnnotationRunManifest.model_validate_json(
                    canonical_json_bytes(manifest_payload), strict=True
                )
                expected_ids = {
                    assignment.raw_id
                    for assignment in split_manifest.assignments
                    if assignment.split_role is DatasetSplitRole.ANALYSIS
                }
                if set(imported.input_ids) != expected_ids:
                    raise ValueError("AI 标注结果必须恰好覆盖当前 ANALYSIS 允许集合")
                state["preprocessing_annotation_manifest"] = annotation_manifest
                state["preprocessing_import_result"] = imported
                st.success(f"严格 JSON 校验通过：已导入 {imported.input_count} 条 AI 标注。")
            except Exception as exc:
                state["preprocessing_error"] = str(exc)
                st.error(f"AI 标注 JSON 导入失败：{exc}")

        st.markdown("#### 5. 发布 PreparedCorpusPackage")
        view = build_workspace_view(state, mode=REAL_MODE)
        if view.blocking_reason:
            st.warning(f"阻断原因：{view.blocking_reason}")
        if st.button(
            "发布 PreparedCorpusPackage",
            key="publish_prepared_corpus",
            disabled=not view.can_publish,
        ):
            try:
                annotation_manifest = state.get("preprocessing_annotation_manifest")
                imported = state.get("preprocessing_import_result")
                records = list(state.get("preprocessing_records") or [])
                if not isinstance(annotation_manifest, AnnotationRunManifest):
                    raise ValueError("缺少已校验的 annotation run manifest")
                if not isinstance(imported, OfflineAnnotationImportResult):
                    raise ValueError("缺少已校验的 AI 标注结果")
                package = _build_prepared_package(
                    records=records,
                    split_manifest=split_manifest,
                    annotation_manifest=annotation_manifest,
                    imported=imported,
                )
                published = publish_prepared_corpus(
                    Path(__file__).resolve().parents[2] / "data" / "prepared_corpora",
                    package,
                )
                state["preprocessing_published_path"] = str(published.path)
                state["preprocessing_error"] = None
                st.success(
                    "已发布 PreparedCorpusPackage；状态：PIPELINE_VALIDATED_ONLY；"
                    "实验数据·未经 Gold 校准。"
                )
                st.caption(f"文件交接路径：{published.path}")
            except Exception as exc:
                state["preprocessing_error"] = str(exc)
                st.error(f"PreparedCorpusPackage 发布失败：{exc}")
    else:
        st.markdown("#### 2. 拆分 split manifest")
        st.markdown("#### 3. 导出当前允许集合任务包")
        st.button("导出下一批 Codex 任务包", disabled=True, key="empty_export_next_codex_batch")
        st.markdown("#### 4. 导入 AI 标注 JSON")
        st.button("导入 AI 标注 JSON", disabled=True, key="empty_import_ai_annotation_json")
        st.markdown("#### 5. 发布 PreparedCorpusPackage")
        view = build_workspace_view(state, mode=REAL_MODE)
        if view.blocking_reason:
            st.warning(f"阻断原因：{view.blocking_reason}")
        st.button("发布 PreparedCorpusPackage", disabled=True, key="empty_publish_prepared_corpus")


def render_preprocessing_workspace(
    *, split_manifest_path: Path, prepared_root: Path, mode: str | None = None
) -> None:
    st.markdown('<p class="hero-kicker">梅见 · 数据预处理工作台</p>', unsafe_allow_html=True)
    st.title("数据预处理")
    st.caption("完成五步冻结后解锁叙事压力测试；预处理临时状态不会写入后续决策业务状态。")
    selected_mode = mode
    if selected_mode is None:
        selected_mode = st.selectbox(
            "运行入口",
            list(PREPROCESSING_MODES),
            key="preprocessing_mode",
            on_change=_clear_preprocessing_mode_state,
        )
    elif selected_mode not in PREPROCESSING_MODES:
        raise ValueError(f"未知预处理入口：{selected_mode}")
    if selected_mode == OFFICIAL_REPLAY_MODE:
        _render_official_replay(split_manifest_path)
    else:
        _render_real()


def _clear_preprocessing_mode_state() -> None:
    state = st.session_state
    for key in _REAL_STATE_KEYS:
        state.pop(key, None)
