from __future__ import annotations

from datetime import datetime
import re
from typing import Literal, TypeAlias

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from src.string_enum import StringEnum


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_ROLES = ("ANALYSIS", "CHALLENGE", "HOLDOUT")
PackageRole: TypeAlias = Literal["ANALYSIS", "CHALLENGE", "HOLDOUT"]


def _validate_sha256_values(values: dict[str, str]) -> dict[str, str]:
    if any(not _SHA256_RE.fullmatch(value) for value in values.values()):
        raise ValueError("所有 SHA-256 必须是 64 位小写十六进制")
    return values


class SampleType(StringEnum):
    MEIJIAN_FEEDBACK = "梅见反馈"
    TARGET_BRAND_FEEDBACK = "目标品牌反馈"
    COMPETITOR_FEEDBACK = "竞品反馈"
    DRINKING_EMOTION = "饮酒情绪"
    SCENE_NEED = "场景需求"


class DatasetSplitRole(StringEnum):
    ANALYSIS = "ANALYSIS"
    GOLD = "GOLD"
    CHALLENGE_POOL = "CHALLENGE_POOL"
    HOLDOUT = "HOLDOUT"


class PackageValidationStatus(StringEnum):
    CALIBRATED_FROZEN = "CALIBRATED_FROZEN"
    PIPELINE_VALIDATED_ONLY = "PIPELINE_VALIDATED_ONLY"


class ModelRuntime(StringEnum):
    LOCAL = "local"
    CLOUD = "cloud"


class SourceType(StringEnum):
    USER_COMMENT = "USER_COMMENT"
    OFFICIAL_CONTENT = "OFFICIAL_CONTENT"
    MEDIA_CONTENT = "MEDIA_CONTENT"
    BRAND_DOC = "BRAND_DOC"
    PRODUCT_FACT = "PRODUCT_FACT"
    COMPETITOR_DOC = "COMPETITOR_DOC"
    VALIDATION = "VALIDATION"
    SIMULATED = "SIMULATED"


class ScreeningStatus(StringEnum):
    KEEP = "KEEP"
    EXCLUDE = "EXCLUDE"
    DUPLICATE = "DUPLICATE"
    REVIEW = "REVIEW"


class DataHealthStatus(StringEnum):
    READY = "READY"
    NEEDS_REBALANCE = "NEEDS_REBALANCE"


class EvidenceRoute(StringEnum):
    BRAND = "BRAND"
    PRODUCT = "PRODUCT"
    SERVICE = "SERVICE"
    SCENE = "SCENE"
    COMPETITOR = "COMPETITOR"
    OTHER = "OTHER"


class ExperienceScope(StringEnum):
    ACTUAL_USE = "ACTUAL_USE"
    PURCHASE_ONLY = "PURCHASE_ONLY"
    NON_USE = "NON_USE"
    UNKNOWN = "UNKNOWN"


class EvidenceGrade(StringEnum):
    A = "A"
    B = "B"
    C = "C"


class ResolutionStatus(StringEnum):
    AGREED = "AGREED"
    TEAM_CONSENSUS = "TEAM_CONSENSUS"
    UNRESOLVED = "UNRESOLVED"


class EvidenceImpact(StringEnum):
    SUPPORT = "SUPPORT"
    CHALLENGE = "CHALLENGE"
    NEW_SIGNAL = "NEW_SIGNAL"
    NEUTRAL = "NEUTRAL"


class StressExecutionStatus(StringEnum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class StressDecision(StringEnum):
    PASS = "PASS"
    REVISE = "REVISE"
    BLOCK = "BLOCK"


class BusinessDecisionStatus(StringEnum):
    """Executive-facing state; never expose internal check mechanics as a status."""

    ANALYSIS_IN_PROGRESS = "ANALYSIS_IN_PROGRESS"
    READY_FOR_VALIDATION = "READY_FOR_VALIDATION"
    NEEDS_REVISION = "NEEDS_REVISION"
    BLOCKED = "BLOCKED"


class DecisionEntryMode(StringEnum):
    CUSTOM = "CUSTOM"
    DEMO = "DEMO"


class EvolutionCheckpointRole(StringEnum):
    BLIND_REASSESSMENT = "BLIND_REASSESSMENT"
    DELTA = "DELTA"
    # Compatibility name for the existing runner; it serializes as the new role.
    BASELINE = "BLIND_REASSESSMENT"


class DecisionRunStage(StringEnum):
    PREPARED = "PREPARED"
    AWAITING_SPECIFICITY_AUDIT = "AWAITING_SPECIFICITY_AUDIT"
    SPECIFICITY_REVISION = "SPECIFICITY_REVISION"
    SPECIFICITY_FINAL_RANK = "SPECIFICITY_FINAL_RANK"
    AWAITING_SELECTION = "AWAITING_SELECTION"
    HOLDOUT = "HOLDOUT"
    AWAITING_PRIMARY_RESELECTION = "AWAITING_PRIMARY_RESELECTION"
    AWAITING_BLIND_EVIDENCE = "AWAITING_BLIND_EVIDENCE"
    READY_REASSESS = "READY_REASSESS"
    CHECKPOINT_00 = "CHECKPOINT_00"
    CHECKPOINT_01 = "CHECKPOINT_01"
    CHECKPOINT_02 = "CHECKPOINT_02"
    CHECKPOINT_03 = "CHECKPOINT_03"
    CHECKPOINT_04 = "CHECKPOINT_04"
    AWAITING_FINAL_SELECTION = "AWAITING_FINAL_SELECTION"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class DecisionTrigger(StringEnum):
    """The only real events permitted to create a decision checkpoint."""

    INITIAL_DECISION = "INITIAL_DECISION"
    RANK_CHANGED = "RANK_CHANGED"
    ADMISSION_CHANGED = "ADMISSION_CHANGED"
    COUNTER_EVIDENCE_ADDED = "COUNTER_EVIDENCE_ADDED"
    ROBUSTNESS_CHANGED = "ROBUSTNESS_CHANGED"
    HUMAN_VALIDATION_IMPORTED = "HUMAN_VALIDATION_IMPORTED"
    FINAL_DECISION_RECORDED = "FINAL_DECISION_RECORDED"


class BlindBrand(StringEnum):
    MEIJIAN = "梅见"
    CHOYA = "CHOYA"
    RIO = "RIO"
    SEVENTEEN_LIGHT_YEARS = "十七光年"
    UNKNOWN = "不知道"


class BrandDecision(StringEnum):
    GO = "GO"
    ITERATE = "ITERATE"
    NO_GO = "NO_GO"


class StressCheckType(StringEnum):
    EVIDENCE_COVERAGE = "EVIDENCE_COVERAGE"
    COUNTER_EVIDENCE = "COUNTER_EVIDENCE"
    COMPETITOR_SUBSTITUTION = "COMPETITOR_SUBSTITUTION"
    PRODUCT_GROUNDING = "PRODUCT_GROUNDING"
    ROBUSTNESS = "ROBUSTNESS"


class GroundingCategory(StringEnum):
    PRODUCT = "PRODUCT"
    PACKAGING = "PACKAGING"
    SERVING = "SERVING"
    SALES_SCENE = "SALES_SCENE"
    CONTENT_ACTION = "CONTENT_ACTION"


class GroundingFactCoverage(StringEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"


class RobustnessScenario(StringEnum):
    EXCLUDE_PLATFORM = "EXCLUDE_PLATFORM"
    ACTUAL_USE_ONLY = "ACTUAL_USE_ONLY"
    HOLDOUT = "HOLDOUT"


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataVersion(StrictBaseModel):
    dataset_version: str = Field(min_length=1)
    routing_prompt_version: str = Field(min_length=1)
    validation_round: str = Field(min_length=1)


class DatasetSplitAssignment(StrictBaseModel):
    raw_id: str = Field(min_length=1)
    split_role: DatasetSplitRole
    source_platform: str = Field(min_length=1)
    raw_sample_type: str = Field(min_length=1)
    platform_url_available: bool
    leakage_group: str = Field(pattern=r"^[0-9a-f]{64}$")


class DatasetSplitManifest(StrictBaseModel):
    dataset_version: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split_algorithm_version: str = Field(min_length=1)
    holdout_stratum_targets: dict[str, int] = Field(min_length=1)
    holdout_stratum_tolerance: int = Field(ge=0)
    assignments: list[DatasetSplitAssignment] = Field(min_length=1)


class AnnotationRunManifest(StrictBaseModel):
    annotation_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    dataset_version: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_ids: list[str] = Field(min_length=1)
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: datetime


class FormalAnnotationIdentity(StrictBaseModel):
    annotation_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"]
    dataset_version: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DecisionModelProfile(StrictBaseModel):
    profile_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    runtime: ModelRuntime
    response_format: Literal["json_schema", "json_object"]
    prompt_version: str = Field(min_length=1)
    endpoint_profile: str = Field(min_length=1)
    thinking_enabled: bool
    reasoning_effort: Literal["high", "max"]
    max_retries: int = Field(ge=0)


class SpecificityAuditType(StringEnum):
    COMPETITOR_REPLACEMENT = "COMPETITOR_REPLACEMENT"
    BRAND_ASSET = "BRAND_ASSET"
    SCENE_DELIVERY = "SCENE_DELIVERY"
    FAILURE_PREMORTEM = "FAILURE_PREMORTEM"


class SpecificitySeverity(StringEnum):
    HARD_CONFLICT = "HARD_CONFLICT"
    MATERIAL_RISK = "MATERIAL_RISK"
    NOTE = "NOTE"


class SpecificityDisposition(StringEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    DEFER = "DEFER"


class SpecificityOutcome(StringEnum):
    READY = "READY"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCKED = "BLOCKED"


_SPECIFICITY_PATCH_FIELDS = (
    "title",
    "target_audience",
    "user_conflict",
    "brand_opportunity",
    "why_brand",
    "why_meijian",
    "competitor_difference",
    "brand_role",
    "draft_proposition",
    "main_scenes",
    "content_theme",
    "risks",
)

LUNA_V2_PROFILE = {
    "profile_id": "codex-luna-specificity-max-v2",
    "model": "gpt-5.6-luna",
    "prompt_version": "brand-specificity-v2",
    "reasoning_effort": "max",
    "max_retries": 0,
}


def _assert_luna_specificity_profile(profile: DecisionModelProfile) -> DecisionModelProfile:
    v1_expected = {
        "profile_id": "codex-luna-specificity-max-v1",
        "provider": "codex",
        "model_id": "gpt-5.6-luna",
        "runtime": ModelRuntime.CLOUD,
        "response_format": "json_schema",
        "prompt_version": "brand-specificity-v1",
        "endpoint_profile": "codex-offline",
        "thinking_enabled": True,
        "reasoning_effort": "max",
        "max_retries": 0,
    }
    v2_expected = {
        **v1_expected,
        "profile_id": LUNA_V2_PROFILE["profile_id"],
        "prompt_version": LUNA_V2_PROFILE["prompt_version"],
    }
    if tuple(profile.model_dump().items()) not in {
        tuple(v1_expected.items()),
        tuple(v2_expected.items()),
    }:
        raise ValueError("specificity_model 必须精确使用 Codex gpt-5.6-luna Max 离线配置")
    return profile


class SpecificityFieldPatch(StrictBaseModel):
    field_name: Literal[
        "title",
        "target_audience",
        "user_conflict",
        "brand_opportunity",
        "why_brand",
        "why_meijian",
        "competitor_difference",
        "brand_role",
        "draft_proposition",
        "main_scenes",
        "content_theme",
        "risks",
    ]
    before: str | list[str]
    after: str | list[str]
    reason: str = Field(min_length=1)
    evidence_ids: list[str]
    brand_fact_ids: list[str]


class SpecificityFinding(StrictBaseModel):
    finding_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    audit_type: SpecificityAuditType
    severity: SpecificitySeverity
    claim: str = Field(min_length=1)
    evidence_ids: list[str]
    brand_fact_ids: list[str]
    competitor_replacement_result: str = Field(min_length=1)
    delivery_condition: str = Field(min_length=1)
    failure_mode: str = Field(min_length=1)
    suggested_patch: SpecificityFieldPatch | None = None
    triggers_next_round: bool


class CandidateSpecificityAudit(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    candidate_version: int = Field(ge=1)
    findings: list[SpecificityFinding] = Field(min_length=len(SpecificityAuditType))
    consumer_evidence: list[str]
    brand_assets: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("brand_assets", "meijian_assets"),
    )
    competitor_replacement_result: str = Field(min_length=1)
    product_delivery_conditions: list[str]
    applicable_scenarios: list[str]
    failure_reasons: list[str]
    next_validation_question: str = Field(min_length=1)

    @model_validator(mode="after")
    def contains_exactly_one_finding_per_audit_type(self) -> "CandidateSpecificityAudit":
        if any(finding.candidate_id != self.candidate_id for finding in self.findings):
            raise ValueError("finding 的 candidate_id 必须与候选审查一致")
        finding_ids = [finding.finding_id for finding in self.findings]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("finding_id 必须唯一")
        audit_types = [finding.audit_type for finding in self.findings]
        if len(audit_types) != len(SpecificityAuditType) or set(audit_types) != set(SpecificityAuditType):
            raise ValueError("每个候选必须恰好对应四类专属度审查")
        return self


class SpecificityAuditBatch(StrictBaseModel):
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=3)
    candidate_ids: list[str] = Field(min_length=1)
    candidate_versions: dict[str, int] = Field(min_length=1)
    audits: list[CandidateSpecificityAudit] = Field(min_length=1)

    @model_validator(mode="after")
    def candidates_and_audits_correspond_one_to_one(self) -> "SpecificityAuditBatch":
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids 必须唯一")
        expected_ids = set(self.candidate_ids)
        if set(self.candidate_versions) != expected_ids:
            raise ValueError("candidate_versions 必须与 candidate_ids 一一对应")
        audit_ids = [audit.candidate_id for audit in self.audits]
        if len(audit_ids) != len(set(audit_ids)) or set(audit_ids) != expected_ids:
            raise ValueError("audits 必须与 candidate_ids 一一对应")
        if any(self.candidate_versions[audit.candidate_id] != audit.candidate_version for audit in self.audits):
            raise ValueError("candidate_versions 必须与审查中的候选版本一致")
        return self


class SpecificityFindingResponse(StrictBaseModel):
    finding_id: str = Field(min_length=1)
    disposition: SpecificityDisposition
    reason: str = Field(min_length=1)
    applied_patch: SpecificityFieldPatch | None = None

    @model_validator(mode="after")
    def patch_matches_disposition(self) -> "SpecificityFindingResponse":
        if self.disposition is SpecificityDisposition.ACCEPT and self.applied_patch is None:
            raise ValueError("ACCEPT 必须携带 applied_patch")
        if self.disposition is not SpecificityDisposition.ACCEPT and self.applied_patch is not None:
            raise ValueError("只有 ACCEPT 可以携带 applied_patch")
        return self


class SpecificityFieldDiff(StrictBaseModel):
    field_name: Literal[
        "title",
        "target_audience",
        "user_conflict",
        "brand_opportunity",
        "why_brand",
        "why_meijian",
        "competitor_difference",
        "brand_role",
        "draft_proposition",
        "main_scenes",
        "content_theme",
        "risks",
    ]
    before: str | list[str]
    after: str | list[str]
    source_finding_ids: list[str] = Field(min_length=1)


class SpecificityCandidateRevision(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    from_version: int = Field(ge=1)
    to_version: int = Field(ge=1)
    responses: list[SpecificityFindingResponse]
    revised_candidate: "NarrativeCandidate"
    field_diffs: list[SpecificityFieldDiff]
    unresolved_finding_ids: list[str]

    @model_validator(mode="after")
    def version_and_candidate_are_consistent(self) -> "SpecificityCandidateRevision":
        if self.to_version != self.from_version + 1:
            raise ValueError("有效修订的 to_version 必须恰好比 from_version 大 1")
        if self.revised_candidate.candidate_id != self.candidate_id:
            raise ValueError("revised_candidate 的 candidate_id 必须一致")
        finding_ids = [response.finding_id for response in self.responses]
        if len(finding_ids) != len(set(finding_ids)):
            raise ValueError("response 的 finding_id 必须唯一")
        diff_fields = [diff.field_name for diff in self.field_diffs]
        if len(diff_fields) != len(set(diff_fields)):
            raise ValueError("field_diffs 的 field_name 必须唯一")
        unresolved = self.unresolved_finding_ids
        if len(unresolved) != len(set(unresolved)):
            raise ValueError("unresolved_finding_ids 必须唯一")
        return self


class SpecificityRevisionBatch(StrictBaseModel):
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=3)
    revisions: list[SpecificityCandidateRevision] = Field(min_length=1)

    @model_validator(mode="after")
    def revision_candidates_are_unique(self) -> "SpecificityRevisionBatch":
        candidate_ids = [revision.candidate_id for revision in self.revisions]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("revisions 的 candidate_id 必须唯一")
        return self


class SpecificityFinalRankResult(StrictBaseModel):
    run_id: str = Field(min_length=1)
    candidate_ids: list[str] = Field(min_length=1, max_length=3)
    ranked_candidate_ids: list[str] = Field(min_length=1, max_length=3)
    recommended_candidate_id: str = Field(min_length=1)
    candidate_versions: dict[str, int] = Field(min_length=1)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_identity: Literal["deepseek-specificity-final-rank-v1"]

    @model_validator(mode="after")
    def ranking_is_a_permutation_of_candidates(self) -> "SpecificityFinalRankResult":
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        if len(self.ranked_candidate_ids) != len(set(self.ranked_candidate_ids)):
            raise ValueError("ranked_candidate_ids must be unique")
        if set(self.candidate_ids) != set(self.ranked_candidate_ids):
            raise ValueError("ranked_candidate_ids must exactly match candidate_ids")
        if self.recommended_candidate_id not in self.ranked_candidate_ids:
            raise ValueError("recommended_candidate_id must come from ranked_candidate_ids")
        if set(self.candidate_versions) != set(self.candidate_ids):
            raise ValueError("candidate_versions must exactly match candidate_ids")
        if any(version < 1 for version in self.candidate_versions.values()):
            raise ValueError("candidate_versions must start at 1")
        return self


class SpecificityRound(StrictBaseModel):
    round_index: int = Field(ge=1, le=3)
    audit_call_sequence: int = Field(gt=0)
    audit_response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    audit: SpecificityAuditBatch
    revision_call_sequence: int | None = None
    revision_response_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    revision: SpecificityRevisionBatch | None = None

    @model_validator(mode="after")
    def revision_metadata_is_atomic(self) -> "SpecificityRound":
        revision_values = (
            self.revision_call_sequence,
            self.revision_response_sha256,
            self.revision,
        )
        if any(value is None for value in revision_values) and any(value is not None for value in revision_values):
            raise ValueError("revision_call_sequence、revision_response_sha256、revision 必须同时存在或同时为空")
        if self.audit.round_index != self.round_index:
            raise ValueError("round_index 必须与 audit 一致")
        if self.revision is not None and self.revision.round_index != self.round_index:
            raise ValueError("round_index 必须与 revision 一致")
        return self


class CandidateSpecificityHistory(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    initial_candidate: "NarrativeCandidate"
    rounds: list[SpecificityRound]
    current_candidate: "NarrativeCandidate"
    current_version: int = Field(ge=1)
    outcome: SpecificityOutcome
    blocking_gate_id: str | None = Field(default=None, min_length=1)
    unresolved_finding_ids: list[str]

    @model_validator(mode="after")
    def versions_and_outcome_are_consistent(self) -> "CandidateSpecificityHistory":
        if self.initial_candidate.candidate_id != self.candidate_id:
            raise ValueError("initial_candidate 的 candidate_id 必须一致")
        if self.current_candidate.candidate_id != self.candidate_id:
            raise ValueError("current_candidate 的 candidate_id 必须一致")
        round_indices = [item.round_index for item in self.rounds]
        if round_indices != list(range(1, len(round_indices) + 1)) or len(round_indices) > 3:
            raise ValueError("rounds 必须按 1..3 连续登记")
        revision_count = sum(item.revision is not None for item in self.rounds)
        if self.current_version != 1 + revision_count:
            raise ValueError("current_version 必须由有效修订次数递增得到")
        if self.outcome is SpecificityOutcome.BLOCKED and self.blocking_gate_id is None:
            raise ValueError("BLOCKED 必须由确定性校验或既有安全门提供 blocking_gate_id")
        if self.outcome is not SpecificityOutcome.BLOCKED and self.blocking_gate_id is not None:
            raise ValueError("非 BLOCKED 结果不能携带 blocking_gate_id")
        if len(self.unresolved_finding_ids) != len(set(self.unresolved_finding_ids)):
            raise ValueError("unresolved_finding_ids 必须唯一")
        return self


class SpecificitySession(StrictBaseModel):
    run_id: str = Field(min_length=1)
    decision_model: DecisionModelProfile
    specificity_model: DecisionModelProfile
    prompt_hashes: dict[str, str] = Field(min_length=1)
    schema_hashes: dict[str, str] = Field(min_length=1)
    confirmed_pattern_ids: list[str]
    candidate_histories: list[CandidateSpecificityHistory] = Field(min_length=1)
    final_candidate_ids: list[str] = Field(min_length=1)
    finalized_at: datetime

    @field_validator("prompt_hashes", "schema_hashes")
    @classmethod
    def hashes_are_sha256(cls, values: dict[str, str]) -> dict[str, str]:
        return _validate_sha256_values(values)

    @model_validator(mode="after")
    def session_identity_is_consistent(self) -> "SpecificitySession":
        _assert_luna_specificity_profile(self.specificity_model)
        history_ids = [history.candidate_id for history in self.candidate_histories]
        if len(history_ids) != len(set(history_ids)):
            raise ValueError("candidate_histories 的 candidate_id 必须唯一")
        if len(self.final_candidate_ids) != len(set(self.final_candidate_ids)):
            raise ValueError("final_candidate_ids 必须唯一")
        if not set(self.final_candidate_ids).issubset(history_ids):
            raise ValueError("final_candidate_ids 必须来自 candidate_histories")
        return self


class ConfirmedSpecificityPattern(StrictBaseModel):
    pattern_id: str = Field(min_length=1)
    audit_type: SpecificityAuditType
    scope_tags: list[str] = Field(min_length=1)
    description: str = Field(min_length=1)
    source_run_id: str = Field(min_length=1)
    source_candidate_id: str = Field(min_length=1)
    source_finding_id: str = Field(min_length=1)
    confirmation_type: str = Field(min_length=1)
    confirmed_by: str = Field(min_length=1)
    confirmed_at: datetime
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ConfirmedSpecificityPatterns(StrictBaseModel):
    schema_version: str = Field(min_length=1)
    patterns: list[ConfirmedSpecificityPattern]

    @model_validator(mode="after")
    def pattern_ids_are_unique(self) -> "ConfirmedSpecificityPatterns":
        pattern_ids = [pattern.pattern_id for pattern in self.patterns]
        if len(pattern_ids) != len(set(pattern_ids)):
            raise ValueError("pattern_id 必须唯一")
        return self


class OfflineSpecificityTask(StrictBaseModel):
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=3)
    model_profile: DecisionModelProfile
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_versions: dict[str, int] = Field(min_length=1)
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def task_uses_luna_profile(self) -> "OfflineSpecificityTask":
        _assert_luna_specificity_profile(self.model_profile)
        if any(version < 1 for version in self.candidate_versions.values()):
            raise ValueError("candidate_versions 必须从 1 开始")
        return self


class OfflineSpecificityResult(StrictBaseModel):
    task_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    round_index: int = Field(ge=1, le=3)
    model_profile: DecisionModelProfile
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_versions: dict[str, int] = Field(min_length=1)
    output: dict[str, JsonValue]

    @model_validator(mode="after")
    def result_uses_luna_profile(self) -> "OfflineSpecificityResult":
        _assert_luna_specificity_profile(self.model_profile)
        if any(version < 1 for version in self.candidate_versions.values()):
            raise ValueError("candidate_versions 必须从 1 开始")
        return self


class SpecificityImportReceipt(StrictBaseModel):
    task_id: str = Field(min_length=1)
    call_sequence: int = Field(gt=0)
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    imported_at: datetime


class PreparedCorpusManifest(StrictBaseModel):
    package_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_version: str = Field(min_length=1)
    annotation_model_id: str = Field(min_length=1)
    annotation_reasoning_effort: str = Field(min_length=1)
    annotation_prompt_version: str = Field(min_length=1)
    annotation_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_status: PackageValidationStatus
    gold_calibration_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    record_count: int = Field(gt=0)
    evidence_count: int = Field(gt=0)
    artifact_sha256: dict[str, str] = Field(min_length=1)
    package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("artifact_sha256")
    @classmethod
    def artifact_hashes_are_lowercase_sha256(
        cls, values: dict[str, str]
    ) -> dict[str, str]:
        if any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values.values()):
            raise ValueError(
                "artifact_sha256 values must be lowercase 64-character SHA-256"
            )
        return values

    @model_validator(mode="after")
    def gold_calibration_matches_validation_status(self) -> "PreparedCorpusManifest":
        has_gold_calibration = self.gold_calibration_sha256 is not None
        if self.validation_status is PackageValidationStatus.CALIBRATED_FROZEN:
            if not has_gold_calibration:
                raise ValueError(
                    "CALIBRATED_FROZEN requires gold_calibration_sha256"
                )
        elif has_gold_calibration:
            raise ValueError(
                "PIPELINE_VALIDATED_ONLY cannot include gold_calibration_sha256"
            )
        return self


class DecisionRunManifest(StrictBaseModel):
    decision_run_id: str = Field(min_length=1)
    package_ids: dict[PackageRole, str] = Field(min_length=3)
    package_sha256: dict[PackageRole, str] = Field(min_length=3)
    decision_model: DecisionModelProfile
    specificity_model: DecisionModelProfile
    created_at: datetime
    run_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("package_sha256")
    @classmethod
    def package_hashes_are_sha256(cls, values: dict[PackageRole, str]) -> dict[PackageRole, str]:
        _validate_sha256_values(values)
        return values

    @model_validator(mode="after")
    def registers_exact_package_roles(self) -> "DecisionRunManifest":
        if set(self.package_ids) != set(_PACKAGE_ROLES) or set(self.package_sha256) != set(_PACKAGE_ROLES):
            raise ValueError("package_ids 和 package_sha256 必须精确登记 ANALYSIS/CHALLENGE/HOLDOUT")
        _assert_luna_specificity_profile(self.specificity_model)
        return self


class SourceReference(StrictBaseModel):
    source_id: str = Field(min_length=1)
    source_type: SourceType
    source_ref: str = Field(min_length=1)


class BrandFact(StrictBaseModel):
    fact_id: str = Field(min_length=1)
    fact_type: GroundingCategory
    statement: str = Field(min_length=1)
    source: SourceReference
    verified_by: str = Field(min_length=1)
    grounding_coverage: GroundingFactCoverage = GroundingFactCoverage.COMPLETE

    @model_validator(mode="after")
    def source_must_be_a_human_verification_source(self) -> "BrandFact":
        if self.source.source_type not in {SourceType.BRAND_DOC, SourceType.PRODUCT_FACT}:
            raise ValueError("BrandFact 只能引用人工核验的 BRAND_DOC 或 PRODUCT_FACT")
        return self


class DataHealthResult(StrictBaseModel):
    status: DataHealthStatus
    effective_user_comment_count: int = Field(ge=0)
    platform_count: int = Field(ge=0)
    largest_platform: str | None = None
    largest_platform_share: float = Field(ge=0, le=1)
    traceable_rate: float = Field(ge=0, le=1)
    platform_url_coverage: float = Field(ge=0, le=1)
    actual_use_ratio: float | None = Field(default=None, ge=0, le=1)
    simulated_record_count: int = Field(ge=0)
    can_finalize_snapshot: bool


class CommentRecord(StrictBaseModel):
    comment_id: str = Field(min_length=1)
    sample_type: SampleType
    raw_sample_type: str | None = Field(default=None, min_length=1)
    raw_content: str = Field(min_length=1)
    context_content: str | None = Field(default=None, min_length=1)
    source_note: str | None = Field(default=None, min_length=1)
    source_platform: str | None = None
    drinking_scene: list[str] = Field(default_factory=list)
    emotion_keywords: list[str] = Field(default_factory=list)
    source_row: int | None = None
    possible_duplicate: bool = False
    raw_id: str | None = Field(default=None, min_length=1)
    original_url: str | None = Field(default=None, min_length=1)
    platform_url_available: bool | None = None
    collected_at: str | None = Field(default=None, min_length=1)
    screening_status: ScreeningStatus | None = None
    screening_reason: str | None = Field(default=None, min_length=1)
    duplicate_group: str | None = Field(default=None, min_length=1)
    source: SourceReference | None = None
    actual_use: bool | None = None
    versions: DataVersion | None = None


class EvidenceAtom(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    comment_id: str = Field(min_length=1)
    route: EvidenceRoute
    experience_scope: ExperienceScope
    evidence_grade: EvidenceGrade
    ai_confidence: float | None = Field(default=None, ge=0, le=1)
    explanation: str | None = Field(default=None, min_length=1)
    label_source: Literal["AI", "HUMAN_GOLD"] = "AI"
    source: SourceReference
    source_platform: str | None = Field(default=None, min_length=1)
    duplicate_group: str | None = Field(default=None, min_length=1)
    actual_use: bool | None = None


class PreparedCorpusPackage(StrictBaseModel):
    manifest: PreparedCorpusManifest
    records: list[CommentRecord] = Field(min_length=1)
    evidence_atoms: list[EvidenceAtom] = Field(min_length=1)
    split_manifest: DatasetSplitManifest
    annotation_run_manifests: list[AnnotationRunManifest] = Field(min_length=1)


class FormalPreparedCorpusGroupManifest(StrictBaseModel):
    group_id: str = Field(min_length=1)
    package_ids: dict[str, str] = Field(min_length=3)
    package_sha256: dict[str, str] = Field(min_length=3)
    analysis_baseline_ids: list[str] = Field(min_length=229, max_length=229)
    analysis_release_ids: list[str] = Field(min_length=20, max_length=20)
    all_stable_ids_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_identity: FormalAnnotationIdentity
    group_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("package_ids")
    @classmethod
    def package_ids_are_exactly_the_official_roles(
        cls, values: dict[str, str]
    ) -> dict[str, str]:
        if set(values) != {"ANALYSIS", "CHALLENGE", "HOLDOUT"}:
            raise ValueError("formal group package_ids must contain ANALYSIS/CHALLENGE/HOLDOUT")
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError("formal group package IDs must be non-empty strings")
        return values

    @field_validator("package_sha256")
    @classmethod
    def package_hashes_are_exactly_the_official_roles(
        cls, values: dict[str, str]
    ) -> dict[str, str]:
        if set(values) != {"ANALYSIS", "CHALLENGE", "HOLDOUT"}:
            raise ValueError("formal group package_sha256 must contain ANALYSIS/CHALLENGE/HOLDOUT")
        _validate_sha256_values(values)
        return values

    @model_validator(mode="after")
    def validate_analysis_boundaries(self) -> "FormalPreparedCorpusGroupManifest":
        if len(set(self.analysis_baseline_ids)) != len(self.analysis_baseline_ids):
            raise ValueError("analysis_baseline_ids must be unique")
        if len(set(self.analysis_release_ids)) != len(self.analysis_release_ids):
            raise ValueError("analysis_release_ids must be unique")
        if set(self.analysis_baseline_ids).intersection(self.analysis_release_ids):
            raise ValueError("analysis baseline and release IDs must be disjoint")
        return self


class EvidenceImpactRecord(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    impact: EvidenceImpact
    target_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def support_and_challenge_require_a_target(self) -> "EvidenceImpactRecord":
        if self.impact in {EvidenceImpact.SUPPORT, EvidenceImpact.CHALLENGE} and self.target_id is None:
            raise ValueError("SUPPORT / CHALLENGE 必须绑定目标 ID")
        if self.impact in {EvidenceImpact.NEW_SIGNAL, EvidenceImpact.NEUTRAL} and self.target_id is not None:
            raise ValueError("NEW_SIGNAL / NEUTRAL 不得绑定目标 ID")
        return self


class HoldoutEvidenceImpact(StrictBaseModel):
    """Explicit, auditable effect of one holdout atom on one candidate."""

    evidence_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    impact: EvidenceImpact

    @model_validator(mode="after")
    def holdout_impact_must_change_candidate_support(self) -> "HoldoutEvidenceImpact":
        if self.impact not in {EvidenceImpact.SUPPORT, EvidenceImpact.CHALLENGE}:
            raise ValueError("HOLDOUT 影响只能是 SUPPORT 或 CHALLENGE")
        return self


class CalibrationResult(StrictBaseModel):
    field_disagreements: dict[str, int]
    primary_confusion_pairs: dict[str, dict[str, int]]
    disagreements: list["AnnotationDisagreement"]


class CalibrationAnnotation(StrictBaseModel):
    annotator_id: str | None = Field(
        default=None, min_length=1, pattern=r"^\S(?:[\s\S]*\S)?$"
    )
    evidence_id: str = Field(min_length=1)
    route: EvidenceRoute
    experience_scope: ExperienceScope
    evidence_grade: EvidenceGrade
    annotator_note: str | None = Field(default=None, min_length=1)


class AnnotationDisagreement(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    fields: list[str] = Field(min_length=1)


def _validate_field_resolution_payloads(
    label: "AdjudicationDecision | FrozenCalibrationLabel",
    *,
    require_resolved_final: bool,
) -> None:
    for field in ("route", "experience_scope", "evidence_grade"):
        status = getattr(label, f"{field}_resolution_status")
        final_value = getattr(label, f"final_{field}")
        reviewer_ids = getattr(label, f"{field}_reviewer_ids")
        reason = getattr(label, f"{field}_resolution_reason")
        if any(
            not reviewer_id
            or reviewer_id != reviewer_id.strip()
            for reviewer_id in reviewer_ids
        ):
            raise ValueError(f"{field} reviewer_ids 不能包含首尾空白或空 ID")
        if status is ResolutionStatus.UNRESOLVED:
            if final_value is not None:
                raise ValueError(f"{field} UNRESOLVED 时 final 必须为空")
            if reviewer_ids or reason is not None:
                raise ValueError(f"{field} UNRESOLVED 不得伪造共识记录")
        elif status is ResolutionStatus.TEAM_CONSENSUS:
            if final_value is None:
                raise ValueError(f"{field} TEAM_CONSENSUS 必须提供 final")
        elif reviewer_ids or reason is not None:
            raise ValueError(f"{field} AGREED 不得附加共识记录")
        if require_resolved_final and status is not ResolutionStatus.UNRESOLVED and final_value is None:
            raise ValueError(f"{field} 已决字段必须提供 final")


class AdjudicationDecision(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    final_route: EvidenceRoute | None = None
    final_experience_scope: ExperienceScope | None = None
    final_evidence_grade: EvidenceGrade | None = None
    route_resolution_status: ResolutionStatus = ResolutionStatus.AGREED
    experience_scope_resolution_status: ResolutionStatus = ResolutionStatus.AGREED
    evidence_grade_resolution_status: ResolutionStatus = ResolutionStatus.AGREED
    route_reviewer_ids: list[str] = Field(default_factory=list)
    experience_scope_reviewer_ids: list[str] = Field(default_factory=list)
    evidence_grade_reviewer_ids: list[str] = Field(default_factory=list)
    route_resolution_reason: str | None = Field(default=None, min_length=1)
    experience_scope_resolution_reason: str | None = Field(default=None, min_length=1)
    evidence_grade_resolution_reason: str | None = Field(default=None, min_length=1)
    adjudicator_id: str | None = Field(
        default=None, min_length=1, pattern=r"^\S(?:[\s\S]*\S)?$"
    )
    adjudication_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validates_field_resolution_payloads(self) -> "AdjudicationDecision":
        _validate_field_resolution_payloads(self, require_resolved_final=False)
        return self


class FrozenCalibrationLabel(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    final_route: EvidenceRoute | None = None
    final_experience_scope: ExperienceScope | None = None
    final_evidence_grade: EvidenceGrade | None = None
    route_resolution_status: ResolutionStatus = ResolutionStatus.AGREED
    experience_scope_resolution_status: ResolutionStatus = ResolutionStatus.AGREED
    evidence_grade_resolution_status: ResolutionStatus = ResolutionStatus.AGREED
    route_reviewer_ids: list[str] = Field(default_factory=list)
    experience_scope_reviewer_ids: list[str] = Field(default_factory=list)
    evidence_grade_reviewer_ids: list[str] = Field(default_factory=list)
    route_resolution_reason: str | None = Field(default=None, min_length=1)
    experience_scope_resolution_reason: str | None = Field(default=None, min_length=1)
    evidence_grade_resolution_reason: str | None = Field(default=None, min_length=1)
    adjudicator_id: str | None = Field(
        default=None, min_length=1, pattern=r"^\S(?:[\s\S]*\S)?$"
    )
    adjudication_reason: str | None = Field(default=None, min_length=1)
    is_frozen: bool = True

    @model_validator(mode="after")
    def validates_frozen_field_resolution_payloads(self) -> "FrozenCalibrationLabel":
        _validate_field_resolution_payloads(self, require_resolved_final=True)
        return self


class AIHumanComparisonRecord(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    ai_route: EvidenceRoute
    final_route: EvidenceRoute | None = None
    route_match: bool | None = None
    ai_experience_scope: ExperienceScope
    final_experience_scope: ExperienceScope | None = None
    experience_scope_match: bool | None = None
    ai_evidence_grade: EvidenceGrade
    final_evidence_grade: EvidenceGrade | None = None
    evidence_grade_match: bool | None = None


class AIHumanComparisonResult(StrictBaseModel):
    records: list[AIHumanComparisonRecord]
    field_disagreements: dict[str, int]
    primary_confusion_pairs: dict[str, dict[str, int]]
    representative_disagreements: dict[str, list[str]]


class HumanAnnotation(StrictBaseModel):
    comment_id: str = Field(min_length=1)
    surface_need: str | None = None
    deep_emotion: str | None = None
    identity_need: str | None = None
    drinking_attitude: str | None = None
    brand_feedback: str | None = None
    narrative_opportunity: str | None = None


class EvidenceQuote(StrictBaseModel):
    comment_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class SingleCommentAnalysis(StrictBaseModel):
    comment_id: str = Field(min_length=1)
    user_need: str = Field(min_length=1)
    real_emotion: str = Field(min_length=1)
    core_conflict: str = Field(min_length=1)
    main_concerns: list[str] = Field(min_length=1)
    local_implication_for_brand: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "local_implication_for_brand", "local_implication_for_meijian"
        ),
    )
    evidence_quotes: list[EvidenceQuote] = Field(min_length=1)
    uncertainty: str | None = None

    @model_validator(mode="after")
    def evidence_must_match_comment(self) -> "SingleCommentAnalysis":
        if any(item.comment_id != self.comment_id for item in self.evidence_quotes):
            raise ValueError("单条分析的证据 comment_id 必须等于当前评论 ID")
        return self


class EmotionalConflict(StrictBaseModel):
    conflict_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    user_need: str = Field(min_length=1)
    desired_state: str = Field(min_length=1)
    rejected_state: str = Field(min_length=1)
    identity_need: str = Field(min_length=1)
    main_concerns: list[str] = Field(min_length=1)
    main_scenes: list[str] = Field(min_length=1)
    supporting_evidence: list[EvidenceQuote] = Field(min_length=1)
    counter_evidence: list[EvidenceQuote] = Field(default_factory=list)
    counter_evidence_note: str | None = None
    implication_for_brand: str = Field(
        min_length=1,
        validation_alias=AliasChoices(
            "implication_for_brand", "implication_for_meijian"
        ),
    )

    @model_validator(mode="after")
    def explain_missing_counter_evidence(self) -> "EmotionalConflict":
        if not self.counter_evidence and not (self.counter_evidence_note or "").strip():
            raise ValueError("counter_evidence 为空时必须说明未发现反面证据")
        return self


class FeedbackCluster(StrictBaseModel):
    summary: str = Field(min_length=1)
    evidence: list[EvidenceQuote] = Field(min_length=1)


class BrandFeedbackComparison(StrictBaseModel):
    brand_positive: list[FeedbackCluster] = Field(
        default_factory=list,
        validation_alias=AliasChoices("brand_positive", "meijian_positive"),
    )
    brand_negative: list[FeedbackCluster] = Field(
        default_factory=list,
        validation_alias=AliasChoices("brand_negative", "meijian_negative"),
    )
    competitor_positive: list[FeedbackCluster] = Field(default_factory=list)
    competitor_negative: list[FeedbackCluster] = Field(default_factory=list)


class CorpusAnalysisResult(StrictBaseModel):
    emotional_conflicts: list[EmotionalConflict]
    feedback_comparison: BrandFeedbackComparison
    main_scenes: list[str]
    data_limitations: list[str]


class DiversityAssessment(StrictBaseModel):
    diversity_level: str = Field(min_length=1)
    distinct_conflict_count: int = Field(ge=0)
    evidence_density: str = Field(min_length=1)
    redundancy: str = Field(min_length=1)
    contradictions: list[str]
    recommended_candidate_count: int = Field(ge=0, le=5)
    count_rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def candidate_count_cannot_exceed_conflicts(self) -> "DiversityAssessment":
        if self.recommended_candidate_count > self.distinct_conflict_count:
            raise ValueError("recommended_candidate_count 不得大于 distinct_conflict_count")
        return self


class ScoreItem(StrictBaseModel):
    score: float = Field(ge=0, le=100)
    rationale: str = Field(min_length=1)


class CandidateScores(StrictBaseModel):
    evidence_strength: ScoreItem
    emotional_tension: ScoreItem
    brand_fit_and_exclusivity: ScoreItem = Field(
        validation_alias=AliasChoices(
            "brand_fit_and_exclusivity", "meijian_fit_and_exclusivity"
        )
    )
    competitor_difference: ScoreItem
    scene_conversion: ScoreItem


class NarrativeCandidate(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    primary_conflict_id: str = Field(min_length=1)
    supporting_conflict_ids: list[str] = Field(min_length=1)
    title: str = Field(min_length=1)
    target_audience: str = Field(min_length=1)
    user_conflict: str = Field(min_length=1)
    brand_opportunity: str = Field(min_length=1)
    why_brand: str = Field(
        min_length=1,
        validation_alias=AliasChoices("why_brand", "why_meijian"),
    )
    competitor_difference: str = Field(min_length=1)
    brand_role: str = Field(min_length=1)
    draft_proposition: str = Field(min_length=1)
    main_scenes: list[str] = Field(min_length=1)
    content_theme: str = Field(min_length=1)
    supporting_evidence: list[EvidenceQuote] = Field(min_length=1)
    counter_evidence: list[EvidenceQuote] = Field(default_factory=list)
    counter_evidence_note: str | None = None
    risks: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_conflict_and_counter_evidence(self) -> "NarrativeCandidate":
        if self.primary_conflict_id not in self.supporting_conflict_ids:
            raise ValueError("primary_conflict_id 必须位于 supporting_conflict_ids")
        if not self.counter_evidence and not (self.counter_evidence_note or "").strip():
            raise ValueError("counter_evidence 为空时必须说明未发现反面证据")
        return self


class CandidateEvaluation(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    scores: CandidateScores
    overall_assessment: str = Field(min_length=1)


class RankedNarrative(StrictBaseModel):
    candidate: NarrativeCandidate
    evaluation: CandidateEvaluation
    weighted_score: float
    rank: int = Field(ge=1)
    is_recommended: bool = False

    @model_validator(mode="after")
    def evaluation_matches_candidate(self) -> "RankedNarrative":
        if self.evaluation.candidate_id != self.candidate.candidate_id:
            raise ValueError("评价结果与候选 ID 不一致")
        return self


class RankingResult(StrictBaseModel):
    ranked_candidates: list[RankedNarrative]
    recommended_candidate_id: str | None = None
    recommendation_reason: str | None = None

    @model_validator(mode="after")
    def validate_recommendation(self) -> "RankingResult":
        reason = (self.recommendation_reason or "").strip()
        if not self.ranked_candidates:
            if self.recommended_candidate_id is not None or not reason:
                raise ValueError("无候选时推荐 ID 必须为空且原因非空")
            return self
        candidate_ids = {item.candidate.candidate_id for item in self.ranked_candidates}
        if self.recommended_candidate_id not in candidate_ids or not reason:
            raise ValueError("有候选时推荐 ID 必须有效且推荐理由非空")
        return self


class NarrativeDecision(StrictBaseModel):
    recommended_candidate_id: str = Field(min_length=1)
    selected_candidate_id: str = Field(min_length=1)
    selection_changed_by_user: bool
    original_brand_role: str
    original_proposition: str
    original_main_scenes: list[str]
    edited_brand_role: str
    edited_proposition: str
    edited_main_scenes: list[str]


class TemplateCheckResult(StrictBaseModel):
    checked_text: str
    detected_keywords: list[str]
    template_risk_level: str
    user_problem_clear: bool
    competitor_difference_clear: bool
    meijian_fit_and_exclusivity_clear: bool
    scene_specific: bool
    evidence_supported: bool
    alcohol_safety_compliant: bool
    competitor_substitution_result: str
    problems: list[str]
    questions: list[str]
    revision_guidance: list[str]
    reference_ids: list[str] = Field(default_factory=list)


class StressCheckResult(StrictBaseModel):
    check_type: StressCheckType
    execution_status: StressExecutionStatus
    decision: StressDecision | None = None
    reference_ids: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)
    revision_boundary: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def execution_status_controls_business_decision(self) -> "StressCheckResult":
        if self.execution_status is StressExecutionStatus.COMPLETED:
            if self.decision is None:
                raise ValueError("COMPLETED 压力检查必须给出业务结论")
        elif self.decision is not None:
            raise ValueError("PENDING 或 ERROR 压力检查不能给出业务结论")
        return self


class CandidateStressResult(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    checks: list[StressCheckResult]

    @model_validator(mode="after")
    def contains_exactly_the_five_required_checks(self) -> "CandidateStressResult":
        check_types = [check.check_type for check in self.checks]
        if len(check_types) != len(StressCheckType) or set(check_types) != set(StressCheckType):
            raise ValueError("必须恰好包含五项压力检查，且每项只能出现一次")
        return self


class NarrativePatch(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    field_name: Literal["draft_proposition", "main_scenes", "competitor_difference"]
    before: str | list[str]
    after: str | list[str]
    added_phrases: list[str] = Field(default_factory=list)
    removed_phrases: list[str] = Field(default_factory=list)
    trigger_evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class CandidateDecisionSnapshot(StrictBaseModel):
    ranked_narrative: RankedNarrative
    stress_result: CandidateStressResult
    business_status: BusinessDecisionStatus
    supporting_count: int = Field(ge=0)
    counter_count: int = Field(ge=0)
    risk_count: int = Field(ge=0)

    @model_validator(mode="after")
    def candidate_id_matches_snapshots(self) -> "CandidateDecisionSnapshot":
        ranked_candidate_id = self.ranked_narrative.candidate.candidate_id
        if self.stress_result.candidate_id != ranked_candidate_id:
            raise ValueError("候选快照的排名与压力测试 candidate_id 必须一致")
        return self


class CandidateSelection(StrictBaseModel):
    run_id: str = Field(min_length=1)
    pending_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_candidate_ids: list[str] = Field(min_length=1, max_length=3)
    primary_candidate_id: str = Field(min_length=1)
    selected_by: str = Field(min_length=1)
    selected_at: datetime
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_selection(self) -> "CandidateSelection":
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ValueError("selected_candidate_ids 必须唯一")
        if self.primary_candidate_id not in self.selected_candidate_ids:
            raise ValueError("primary_candidate_id 必须存在于入围集合")
        return self


class BlindSourceManifest(StrictBaseModel):
    run_id: str = Field(min_length=1)
    original_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_sha256: dict[str, str] = Field(min_length=1)
    confirmed_by: str = Field(min_length=1)
    confirmed_at: datetime

    @field_validator("artifact_sha256")
    @classmethod
    def artifacts_are_sha256(cls, values: dict[str, str]) -> dict[str, str]:
        return _validate_sha256_values(values)


class BlindEvidenceItem(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_artifacts: list[str] = Field(min_length=1)


class BlindEvidencePackage(StrictBaseModel):
    run_id: str = Field(min_length=1)
    source_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    items: list[BlindEvidenceItem] = Field(min_length=1)


class DecisionDisplayEvent(StrictBaseModel):
    run_id: str = Field(min_length=1)
    sequence: int = Field(gt=0)
    stage: DecisionRunStage
    snapshot_id: str = Field(min_length=1)
    checkpoint_id: str | None = Field(default=None, min_length=1)
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


class DecisionFoundationState(StrictBaseModel):
    run_id: str = Field(min_length=1)
    baseline_records: list[CommentRecord] = Field(min_length=1)
    baseline_atoms: list[EvidenceAtom] = Field(min_length=1)
    visible_atoms: list[EvidenceAtom] = Field(min_length=1)
    corpus: CorpusAnalysisResult
    candidates: list[NarrativeCandidate] = Field(min_length=1)
    ranking: RankingResult
    stress_results: list[CandidateStressResult] = Field(min_length=1)
    impacts: list[EvidenceImpactRecord] = Field(default_factory=list)
    selected_ids: list[str] = Field(default_factory=list, max_length=3)
    primary_candidate_id: str | None = Field(default=None, min_length=1)
    holdout_impacts: list[HoldoutEvidenceImpact] = Field(default_factory=list)
    holdout_atoms: list[EvidenceAtom] = Field(default_factory=list)
    robustness_results: list[RobustnessScenarioResult] = Field(default_factory=list)
    challenge_applied: bool = False
    holdout_validated: bool = False
    frozen: bool = False

    @model_validator(mode="after")
    def validate_candidate_references(self) -> "DecisionFoundationState":
        candidate_ids = [candidate.candidate_id for candidate in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidates 的 candidate_id 必须唯一")
        candidate_id_set = set(candidate_ids)

        ranking_ids = [
            item.candidate.candidate_id for item in self.ranking.ranked_candidates
        ]
        if set(ranking_ids) != candidate_id_set or len(ranking_ids) != len(candidate_ids):
            raise ValueError("ranking 必须完整对应 candidates")

        stress_ids = [item.candidate_id for item in self.stress_results]
        if set(stress_ids) != candidate_id_set or len(stress_ids) != len(candidate_ids):
            raise ValueError("stress_results 必须完整对应 candidates")

        if len(self.selected_ids) != len(set(self.selected_ids)):
            raise ValueError("selected_ids 必须唯一")
        if not set(self.selected_ids).issubset(candidate_id_set):
            raise ValueError("selected_ids 必须来自 candidates")
        if self.primary_candidate_id is not None and self.primary_candidate_id not in self.selected_ids:
            raise ValueError("primary_candidate_id 必须存在于 selected_ids")
        return self


class PendingSelectionPackage(StrictBaseModel):
    run_id: str = Field(min_length=1)
    foundation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: list[CandidateDecisionSnapshot] = Field(min_length=1)
    recommended_candidate_id: str = Field(min_length=1)
    specificity_audit: SpecificityAuditBatch
    created_at: datetime

    @model_validator(mode="after")
    def recommended_candidate_is_available(self) -> "PendingSelectionPackage":
        candidate_ids = [
            item.ranked_narrative.candidate.candidate_id for item in self.candidates
        ]
        if self.recommended_candidate_id not in set(candidate_ids):
            raise ValueError("recommended_candidate_id must come from candidates")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidates 的 candidate_id 必须唯一")
        if self.specificity_audit.run_id != self.run_id:
            raise ValueError("specificity_audit 的 run_id 必须与 pending selection 一致")
        if set(self.specificity_audit.candidate_ids) != set(candidate_ids):
            raise ValueError("specificity_audit 必须覆盖 pending selection 的全部候选")
        return self


class DecisionOriginalSnapshot(StrictBaseModel):
    run_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    foundation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: list[CandidateDecisionSnapshot] = Field(min_length=1, max_length=3)
    primary_candidate_id: str = Field(min_length=1)
    recommended_candidate_id: str = Field(min_length=1)
    holdout_evidence_ids: list[str] = Field(min_length=1)
    specificity_session_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime

    @model_validator(mode="after")
    def selected_candidates_are_consistent(self) -> "DecisionOriginalSnapshot":
        candidate_ids = [
            item.ranked_narrative.candidate.candidate_id for item in self.candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidates 的 candidate_id 必须唯一")
        if self.primary_candidate_id not in candidate_ids:
            raise ValueError("primary_candidate_id 必须存在于 candidates")
        return self


class BlindCandidateReassessment(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    evaluation: CandidateEvaluation
    weighted_score: float = Field(ge=0, le=100)
    rank: int = Field(ge=1)
    business_status: BusinessDecisionStatus
    blind_impact_summary: str = Field(min_length=1)

    @model_validator(mode="after")
    def evaluation_matches_candidate(self) -> "BlindCandidateReassessment":
        if self.evaluation.candidate_id != self.candidate_id:
            raise ValueError("重评评价与候选 ID 不一致")
        return self


class BlindReassessmentResult(StrictBaseModel):
    run_id: str = Field(min_length=1)
    original_snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    blind_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidates: list[BlindCandidateReassessment] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def ranks_are_contiguous(self) -> "BlindReassessmentResult":
        candidate_ids = [item.candidate_id for item in self.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("重评候选 ID 必须唯一")
        if sorted(item.rank for item in self.candidates) != list(range(1, len(self.candidates) + 1)):
            raise ValueError("重评 rank 必须连续为 1..N")
        return self


class FinalCandidateSelection(StrictBaseModel):
    run_id: str = Field(min_length=1)
    checkpoint_04_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shortlisted_candidate_ids: list[str] = Field(min_length=1, max_length=3)
    selected_candidate_id: str = Field(min_length=1)
    recommended_candidate_id: str = Field(min_length=1)
    selected_by: str = Field(min_length=1)
    selected_at: datetime
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def selected_candidate_is_explainable(self) -> "FinalCandidateSelection":
        if len(self.shortlisted_candidate_ids) != len(set(self.shortlisted_candidate_ids)):
            raise ValueError("shortlisted_candidate_ids 必须唯一")
        if self.recommended_candidate_id not in self.shortlisted_candidate_ids:
            raise ValueError("recommended_candidate_id 必须存在于 shortlisted_candidate_ids")
        if self.selected_candidate_id not in self.shortlisted_candidate_ids:
            raise ValueError("selected_candidate_id 必须来自 shortlisted_candidate_ids")
        if self.selected_candidate_id != self.recommended_candidate_id and not (self.reason or "").strip():
            raise ValueError("偏离推荐候选时 reason 必填")
        return self


class DecisionCallRequestManifest(StrictBaseModel):
    call_id: str = Field(min_length=1)
    sequence: int = Field(gt=0)
    stage: DecisionRunStage
    input_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision_model: DecisionModelProfile
    prompt_name: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    system_prompt: str = Field(min_length=1)
    user_prompt: str = Field(min_length=1)
    response_schema_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


class DecisionCallResponseManifest(StrictBaseModel):
    call_id: str = Field(min_length=1)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_model_name: str = Field(min_length=1)
    response_payload: dict[str, JsonValue]
    response_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at: datetime


class CandidateSwitchSuggestion(StrictBaseModel):
    from_candidate_id: str = Field(min_length=1)
    to_candidate_id: str = Field(min_length=1)
    target_business_status: BusinessDecisionStatus
    trigger_evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def target_must_not_be_blocked(self) -> "CandidateSwitchSuggestion":
        if self.target_business_status is BusinessDecisionStatus.BLOCKED:
            raise ValueError("切换建议不得指向 BLOCKED 候选")
        if self.from_candidate_id == self.to_candidate_id:
            raise ValueError("切换建议的来源与目标候选必须不同")
        return self


class DecisionEvolutionCheckpoint(StrictBaseModel):
    run_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    entry_mode: DecisionEntryMode
    role: EvolutionCheckpointRole
    release_index: int = Field(ge=0, le=4)
    visible_evidence_ids: list[str] = Field(min_length=1)
    route_distribution: dict[str, int]
    experience_distribution: dict[str, int]
    grade_distribution: dict[str, int]
    candidates: list[CandidateDecisionSnapshot] = Field(min_length=1, max_length=3)
    selected_candidate_id: str = Field(min_length=1)
    patches: list[NarrativePatch] = Field(default_factory=list)
    switch_suggestion: CandidateSwitchSuggestion | None = None

    @model_validator(mode="after")
    def validates_release_and_candidate_references(
        self,
    ) -> "DecisionEvolutionCheckpoint":
        if self.role is EvolutionCheckpointRole.BLIND_REASSESSMENT and self.release_index != 0:
            raise ValueError("BLIND_REASSESSMENT 检查点的 release_index 必须为 0")
        if self.role is EvolutionCheckpointRole.DELTA and not 1 <= self.release_index <= 4:
            raise ValueError("DELTA 检查点的 release_index 必须在 1 到 4 之间")

        candidate_ids = [
            snapshot.ranked_narrative.candidate.candidate_id
            for snapshot in self.candidates
        ]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("检查点中的候选 ID 必须唯一")
        candidate_id_set = set(candidate_ids)
        if self.selected_candidate_id not in candidate_id_set:
            raise ValueError("selected_candidate_id 必须存在于 candidates")
        if any(patch.candidate_id not in candidate_id_set for patch in self.patches):
            raise ValueError("NarrativePatch 的 candidate_id 必须存在于 candidates")

        if self.switch_suggestion is not None:
            target_id = self.switch_suggestion.to_candidate_id
            if target_id not in candidate_id_set:
                raise ValueError("切换建议目标必须存在于 candidates")
            target_snapshot = next(
                snapshot
                for snapshot in self.candidates
                if snapshot.ranked_narrative.candidate.candidate_id == target_id
            )
            if target_snapshot.business_status is BusinessDecisionStatus.BLOCKED:
                raise ValueError("切换建议不得指向 BLOCKED 候选")
        return self


class DecisionFoundationAudit(StrictBaseModel):
    run_id: str = Field(min_length=1)
    foundation_id: str = Field(min_length=1)
    baseline_candidate_ids: list[str] = Field(min_length=1)
    selected_candidate_ids: list[str] = Field(min_length=1, max_length=3)
    challenge_evidence_ids: list[str] = Field(min_length=1)
    holdout_evidence_ids: list[str] = Field(min_length=1)
    holdout_validated: bool

    @model_validator(mode="after")
    def selected_candidates_are_from_baseline(self) -> "DecisionFoundationAudit":
        if len(self.baseline_candidate_ids) != len(set(self.baseline_candidate_ids)):
            raise ValueError("baseline_candidate_ids 必须唯一")
        if len(self.selected_candidate_ids) != len(set(self.selected_candidate_ids)):
            raise ValueError("selected_candidate_ids 必须唯一")
        if not set(self.selected_candidate_ids).issubset(self.baseline_candidate_ids):
            raise ValueError("selected_candidate_ids 必须来自 baseline_candidate_ids")
        return self


_DEMO_CHECKPOINT_IDS = tuple(f"checkpoint-{index:02d}" for index in range(5))


def _validate_demo_checkpoint_manifest(
    checkpoint_ids: list[str], checkpoint_sha256: dict[str, str] | None = None
) -> None:
    expected = list(_DEMO_CHECKPOINT_IDS)
    if checkpoint_ids != expected:
        raise ValueError("必须按顺序登记 checkpoint-00..04")
    if checkpoint_sha256 is not None:
        if list(checkpoint_sha256) != expected:
            raise ValueError("checkpoint_sha256 必须登记 checkpoint-00..04")
        _validate_sha256_values(checkpoint_sha256)


class DemoAttemptManifest(StrictBaseModel):
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    # The legacy fields remain readable by the legacy unit-test helper.  A
    # final Demo v2 attempt must use the three fields below; the final loader
    # rejects manifests that do not have them.
    foundation_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    checkpoint_ids: list[str] | None = Field(default=None, min_length=5, max_length=5)
    checkpoint_sha256: dict[str, str] | None = Field(default=None, min_length=5)
    run_manifest_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_profile: DecisionModelProfile | None = None
    specificity_model: DecisionModelProfile | None = None
    specificity_session_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    specificity_prompt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    specificity_schema_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    pending_candidate_ids: list[str] | None = Field(default=None, min_length=1)
    stage_sha256: dict[str, str] | None = Field(default=None, min_length=1)
    artifact_sha256: dict[str, str] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validates_attempt_checkpoints(self) -> "DemoAttemptManifest":
        if self.checkpoint_ids is not None:
            _validate_demo_checkpoint_manifest(self.checkpoint_ids, self.checkpoint_sha256)
        if self.artifact_sha256 is not None:
            _validate_sha256_values(self.artifact_sha256)
            for name in self.artifact_sha256:
                if (
                    not isinstance(name, str)
                    or not name
                    or name.startswith("/")
                    or "\\" in name
                    or any(part in {"", ".", ".."} for part in name.split("/"))
                ):
                    raise ValueError("artifact_sha256 keys must be relative forward-slash paths")
        if self.stage_sha256 is not None:
            _validate_sha256_values(self.stage_sha256)
        if self.pending_candidate_ids is not None and len(self.pending_candidate_ids) != len(set(self.pending_candidate_ids)):
            raise ValueError("pending_candidate_ids must be unique")
        if self.specificity_model is not None:
            _assert_luna_specificity_profile(self.specificity_model)
        return self


class DemoFeishuSyncArtifact(StrictBaseModel):
    """Offline, read-only sync projection derived from display events."""

    run_id: str = Field(min_length=1)
    event_count: int = Field(ge=0)
    last_sequence: int = Field(ge=0)
    last_snapshot_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PublishedDemoManifest(StrictBaseModel):
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    attempt_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_ids: list[str] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def validates_published_attempt(self) -> "PublishedDemoManifest":
        _validate_demo_checkpoint_manifest(self.checkpoint_ids)
        return self


class CandidateAttemptRecord(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    candidate_version: str = Field(min_length=1)
    ranking_snapshot: RankedNarrative
    revised_from_version: str | None = Field(default=None, min_length=1)
    human_edits: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def candidate_id_matches_payload(self) -> "CandidateAttemptRecord":
        if self.ranking_snapshot.candidate.candidate_id != self.candidate_id:
            raise ValueError("候选版本记录与候选 candidate_id 不一致")
        if self.revised_from_version == self.candidate_version:
            raise ValueError("修订后的候选必须使用新版本")
        return self


class StressAttemptRecord(StrictBaseModel):
    generation_id: int = Field(default=0, ge=0)
    attempt_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    candidate_version: str = Field(min_length=1)
    attempt_number: int = Field(ge=1)
    stress_result: CandidateStressResult
    next_candidate_version: str | None = Field(default=None, min_length=1)
    revision_reason: str | None = Field(default=None, min_length=1)
    human_edits: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def revision_requires_a_new_version(self) -> "StressAttemptRecord":
        if self.stress_result.candidate_id != self.candidate_id:
            raise ValueError("压力测试记录与候选 candidate_id 不一致")
        decisions = {check.decision for check in self.stress_result.checks}
        if StressDecision.REVISE in decisions:
            if (
                self.next_candidate_version is None
                or self.next_candidate_version == self.candidate_version
            ):
                raise ValueError("REVISE 必须产生新版本并重测")
            if not (self.revision_reason or "").strip():
                raise ValueError("REVISE 必须记录修订原因")
        return self


class AlternativeConsidered(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    candidate_version: str = Field(min_length=1)
    stress_attempt_id: str = Field(min_length=1)
    outcome: StressDecision
    reasons: list[str] = Field(min_length=1)
    human_edits: list[str] = Field(default_factory=list)
    reference_ids: list[str] = Field(default_factory=list)


class ValidationCandidateCard(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    target_audience: str = Field(min_length=1)
    user_conflict: str = Field(min_length=1)
    draft_proposition: str = Field(min_length=1)
    main_scenes: list[str] = Field(min_length=1)
    content_theme: str = Field(min_length=1)
    presentation_order: int = Field(ge=1, le=3)


class ValidationCardBatch(StrictBaseModel):
    participant_id: str = Field(min_length=1)
    validation_round: str = Field(min_length=1)
    cards: list[ValidationCandidateCard] = Field(min_length=2, max_length=3)


class HumanValidationRecord(StrictBaseModel):
    participant_id: str = Field(min_length=1)
    validation_round: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    presentation_order: int = Field(ge=1, le=3)
    submitted: bool
    blind_brand_guess: BlindBrand | None = None
    brand_revealed: bool = False
    meijian_likeness_score: int | None = Field(default=None, ge=1, le=5)
    distinctiveness_score: int | None = Field(default=None, ge=1, le=5)
    scene_clarity_score: int | None = Field(default=None, ge=1, le=5)
    trial_intent_score: int | None = Field(default=None, ge=1, le=5)
    resonance_reason: str | None = Field(default=None, min_length=1)
    improvement_suggestion: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def enforce_round_order_and_complete_submission(self) -> "HumanValidationRecord":
        round_two_values = (
            self.meijian_likeness_score,
            self.distinctiveness_score,
            self.scene_clarity_score,
            self.trial_intent_score,
            self.resonance_reason,
            self.improvement_suggestion,
        )
        if self.brand_revealed and self.blind_brand_guess is None:
            raise ValueError("必须先提交 Round 1 品牌盲猜，再进入 Round 2")
        if any(value is not None for value in round_two_values) and not self.brand_revealed:
            raise ValueError("Round 2 评分和开放文本只能在品牌揭示后填写")
        if self.submitted and (
            self.blind_brand_guess is None
            or not self.brand_revealed
            or any(value is None for value in round_two_values)
        ):
            raise ValueError("有效提交必须完成 Round 1 与 Round 2 全部字段")
        return self


class CandidateValidationSummary(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    submitted_response_count: int = Field(ge=1)
    blind_brand_ownership_numerator: int = Field(ge=0)
    blind_brand_ownership_denominator: int = Field(ge=1)
    blind_brand_ownership: float = Field(ge=0, le=1)
    blind_brand_confusion: dict[str, int]
    meijian_likeness_mean: float = Field(ge=1, le=5)
    distinctiveness_mean: float = Field(ge=1, le=5)
    scene_clarity_mean: float = Field(ge=1, le=5)
    trial_intent_mean: float = Field(ge=1, le=5)
    resonance_reasons: list[str]
    improvement_suggestions: list[str]
    significance_claimed: bool = False
    interpretation_boundary: str = Field(min_length=1)

    @model_validator(mode="after")
    def ownership_fraction_and_claim_boundary_are_consistent(
        self,
    ) -> "CandidateValidationSummary":
        expected = (
            self.blind_brand_ownership_numerator
            / self.blind_brand_ownership_denominator
        )
        if abs(self.blind_brand_ownership - expected) > 1e-12:
            raise ValueError("Blind Brand Ownership 必须等于分子除以分母")
        if self.significance_claimed:
            raise ValueError("小样本真人验证不得自动宣称统计显著性")
        return self


class BrandFinalDecision(StrictBaseModel):
    decision: BrandDecision
    selected_candidate_id: str | None = Field(default=None, min_length=1)
    reason: str = Field(min_length=1)
    reference_ids: list[str] = Field(min_length=1)
    decided_by: str = Field(min_length=1)


class EvidenceCoverage(StrictBaseModel):
    independent_evidence_count: int = Field(ge=0)
    platform_count: int = Field(ge=0)
    scene_count: int = Field(ge=0)
    grade_counts: dict[str, int]
    independent_brand_evidence_count: int = Field(ge=0)
    can_support_brand_candidate: bool
    reference_ids: list[str] = Field(default_factory=list)


class GroundingResult(StrictBaseModel):
    execution_status: StressExecutionStatus
    decision: StressDecision | None = None
    fact_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    missing_categories: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def grounding_execution_controls_business_decision(self) -> "GroundingResult":
        if self.execution_status is StressExecutionStatus.COMPLETED:
            if self.decision is None:
                raise ValueError("已完成 Grounding 必须给出业务结论")
        elif self.decision is not None:
            raise ValueError("未完成 Grounding 不能给出业务结论")
        return self


class RobustnessCandidateOutcome(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    baseline_rank: int = Field(ge=1)
    rank: int = Field(ge=1)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    new_risks: list[str] = Field(default_factory=list)


class RobustnessScenarioResult(StrictBaseModel):
    scenario: RobustnessScenario
    execution_status: StressExecutionStatus
    decision: StressDecision | None = None
    applied_filter: str = Field(min_length=1)
    reference_ids: list[str] = Field(default_factory=list)
    candidate_outcomes: list[RobustnessCandidateOutcome] = Field(default_factory=list)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def robustness_execution_controls_business_decision(self) -> "RobustnessScenarioResult":
        if self.execution_status is StressExecutionStatus.COMPLETED:
            if self.decision is None:
                raise ValueError("已完成鲁棒性场景必须给出业务结论")
        elif self.decision is not None:
            raise ValueError("未完成鲁棒性场景不能给出业务结论")
        return self


class NextBestEvidence(StrictBaseModel):
    target_audience: str = Field(min_length=1)
    scenario: str = Field(min_length=1)
    suggested_source: str = Field(min_length=1)
    core_question: str = Field(min_length=1)
    reference_ids: list[str] = Field(min_length=1)
    possible_decision_change: str = Field(min_length=1)


class FinalNarrativeReport(StrictBaseModel):
    target_audience: str
    core_emotional_conflict: str
    brand_worldview: str
    brand_role: str
    brand_proposition: str
    why_brand: str = Field(
        validation_alias=AliasChoices("why_brand", "why_meijian")
    )
    core_scenes: list[str]
    content_themes: list[str]
    campaign_idea: str
    supporting_conflict_ids: list[str]
    supporting_evidence: list[EvidenceQuote]
    counter_evidence: list[EvidenceQuote]
    risks: list[str]
    unsuitable_expressions: list[str]


class EvidenceBoundary(StrictBaseModel):
    """Limits on what the current evidence can support, not a market-size claim."""

    effective_user_comment_count: int = Field(ge=0)
    platform_count: int = Field(ge=0)
    human_validation_participant_count: int = Field(ge=0)
    actual_use_ratio: float | None = Field(default=None, ge=0, le=1)
    statement: str = Field(min_length=1)


class DecisionCheckpoint(StrictBaseModel):
    """A compact snapshot made only after a permitted, auditable trigger."""

    checkpoint_id: str = Field(min_length=1)
    occurred_at: str = Field(min_length=1)
    trigger: DecisionTrigger
    versions: DataVersion
    candidate_id: str | None = Field(default=None, min_length=1)
    candidate_rank: int | None = Field(default=None, ge=1)
    weighted_score: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    counter_evidence_ids: list[str] = Field(default_factory=list)
    business_status: BusinessDecisionStatus
    robustness_summary: str = Field(min_length=1)
    primary_risks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def candidate_fields_are_consistent(self) -> "DecisionCheckpoint":
        if (self.candidate_rank is None) != (self.weighted_score is None):
            raise ValueError("candidate_rank 与 weighted_score 必须同时存在或同时为空")
        if self.candidate_rank is not None and self.candidate_id is None:
            raise ValueError("候选排名必须关联 candidate_id")
        return self


class DecisionDelta(StrictBaseModel):
    """Difference between adjacent real checkpoints; it cannot invent a baseline."""

    previous_checkpoint_id: str = Field(min_length=1)
    checkpoint_id: str = Field(min_length=1)
    trigger: DecisionTrigger
    versions: DataVersion
    added_evidence_ids: list[str] = Field(default_factory=list)
    added_fact_ids: list[str] = Field(default_factory=list)
    added_supporting_evidence_ids: list[str] = Field(default_factory=list)
    added_counter_evidence_ids: list[str] = Field(default_factory=list)
    previous_rank: int | None = Field(default=None, ge=1)
    current_rank: int | None = Field(default=None, ge=1)
    rank_change: int | None = None
    previous_weighted_score: float | None = None
    current_weighted_score: float | None = None
    weighted_score_change: float | None = None
    previous_status: BusinessDecisionStatus
    current_status: BusinessDecisionStatus
    robustness_changed: bool
    added_risks: list[str] = Field(default_factory=list)
    removed_risks: list[str] = Field(default_factory=list)


class PublicNarrativeEvidence(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    source_title: str
    source_url: str
    source_type: str = Field(min_length=1)
    claim_text: str
    status: str = Field(min_length=1)
    confidence: str = Field(min_length=1)
    product_support: str = Field(min_length=1)
    raw_fields: dict[str, str]


class PublicEvidenceCorpusV2(StrictBaseModel):
    items: list[PublicNarrativeEvidence] = Field(min_length=54, max_length=54)
    file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def evidence_ids_are_unique(self) -> "PublicEvidenceCorpusV2":
        evidence_ids = [item.evidence_id for item in self.items]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("public evidence_id 必须唯一")
        return self


class FiveCandidateSharedInputV2(StrictBaseModel):
    candidate_ids: list[str] = Field(min_length=5, max_length=5)
    comment_evidence_ids: list[str] = Field(min_length=1)
    public_evidence: list[PublicNarrativeEvidence] = Field(min_length=54, max_length=54)
    confirmed_pattern_ids: list[str] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def validates_exact_shared_input_boundary(self) -> "FiveCandidateSharedInputV2":
        identifier_collections = (
            ("candidate_ids", self.candidate_ids),
            ("comment_evidence_ids", self.comment_evidence_ids),
            ("public evidence_id", [item.evidence_id for item in self.public_evidence]),
            ("confirmed_pattern_ids", self.confirmed_pattern_ids),
        )
        for label, identifiers in identifier_collections:
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{label} 必须唯一")
        return self


class SpecificityEvidenceBindingV2(StrictBaseModel):
    field: str = Field(min_length=1)
    before: str
    after: str
    source_finding_ids: list[str] = Field(min_length=1)
    comment_evidence_ids: list[str] = Field(default_factory=list)
    public_evidence_ids: list[str] = Field(default_factory=list)
    evidence_gap_ids: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def changed_text_requires_comment_evidence_or_a_gap_removal(self) -> "SpecificityEvidenceBindingV2":
        if any(
            len(identifiers) != len(set(identifiers))
            for identifiers in (
                self.source_finding_ids,
                self.comment_evidence_ids,
                self.public_evidence_ids,
                self.evidence_gap_ids,
            )
        ):
            raise ValueError("evidence binding IDs 必须唯一")
        if self.comment_evidence_ids:
            return self
        is_removal_or_narrowing = not self.after.strip() or self.after.strip() in self.before
        if not self.evidence_gap_ids or not is_removal_or_narrowing:
            raise ValueError("字段修改必须绑定评论证据；仅删除或收窄无支持主张可使用 evidence_gap_ids")
        return self


class HoldoutCandidateImpactV2(StrictBaseModel):
    evidence_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    impact: Literal["SUPPORT", "CHALLENGE", "NEUTRAL"]
    rationale: str = Field(min_length=1)
    is_blocking: bool


class FiveCandidateHoldoutResultV2(StrictBaseModel):
    holdout_evidence_ids: list[str] = Field(min_length=60, max_length=60)
    candidate_ids: list[str] = Field(min_length=5, max_length=5)
    impacts: list[HoldoutCandidateImpactV2] = Field(min_length=300, max_length=300)

    @model_validator(mode="after")
    def requires_the_complete_holdout_matrix(self) -> "FiveCandidateHoldoutResultV2":
        if len(self.holdout_evidence_ids) != len(set(self.holdout_evidence_ids)):
            raise ValueError("holdout_evidence_ids 必须唯一")
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids 必须唯一")
        expected_pairs = {
            (evidence_id, candidate_id)
            for evidence_id in self.holdout_evidence_ids
            for candidate_id in self.candidate_ids
        }
        actual_pairs = {(impact.evidence_id, impact.candidate_id) for impact in self.impacts}
        if actual_pairs != expected_pairs or len(actual_pairs) != len(self.impacts):
            raise ValueError("HOLDOUT 必须包含 60 × 5 个唯一 evidence/candidate 组合")
        return self


class FiveCandidateUnifiedEvaluationV2(StrictBaseModel):
    candidate_ids: list[str] = Field(min_length=5, max_length=5)
    evaluations: list[CandidateEvaluation] = Field(min_length=5, max_length=5)
    ranked_candidate_ids: list[str] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def evaluates_each_of_the_five_candidates_once(self) -> "FiveCandidateUnifiedEvaluationV2":
        expected = set(self.candidate_ids)
        evaluation_ids = [item.candidate_id for item in self.evaluations]
        if len(expected) != 5 or len(evaluation_ids) != len(set(evaluation_ids)):
            raise ValueError("统一重评的候选与评价必须唯一且恰好为五个")
        if set(evaluation_ids) != expected or set(self.ranked_candidate_ids) != expected:
            raise ValueError("统一重评必须恰好覆盖同一五个候选")
        if len(self.ranked_candidate_ids) != len(set(self.ranked_candidate_ids)):
            raise ValueError("ranked_candidate_ids 必须唯一")
        return self


class DecisionTimelineEvent(StrictBaseModel):
    """Append-only record of something that actually happened in this run."""

    event_id: str = Field(min_length=1)
    occurred_at: str = Field(min_length=1)
    trigger: DecisionTrigger
    description: str = Field(min_length=1)
    candidate_id: str | None = Field(default=None, min_length=1)
    reference_ids: list[str] = Field(default_factory=list)
    fact_ids: list[str] = Field(default_factory=list)
    validation_ids: list[str] = Field(default_factory=list)


for _specificity_model in (
    SpecificityCandidateRevision,
    CandidateSpecificityHistory,
):
    _specificity_model.model_rebuild()
