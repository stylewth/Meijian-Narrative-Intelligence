from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
from typing import Iterable, Protocol
import unicodedata

from src.schemas import DatasetSplitAssignment, DatasetSplitManifest, DatasetSplitRole


DATASET_VERSION = "screened_v2"
SPLIT_ALGORITHM_VERSION = "leakage_safe_stratified_v2"
HOLDOUT_STRATUM_TOLERANCE = 1
FIXED_TARGET_COUNTS = {
    DatasetSplitRole.GOLD: 50,
    DatasetSplitRole.CHALLENGE_POOL: 60,
    DatasetSplitRole.HOLDOUT: 60,
}
MINIMUM_RECORD_COUNT = sum(FIXED_TARGET_COUNTS.values())
_QUOTA_ERROR = "泄漏组约束下无法满足动态集合配额"


def expected_split_counts(record_count: int) -> dict[DatasetSplitRole, int]:
    if record_count < MINIMUM_RECORD_COUNT:
        raise ValueError(f"拆分至少需要 {MINIMUM_RECORD_COUNT} 条 KEEP 记录")
    return {
        DatasetSplitRole.ANALYSIS: record_count - MINIMUM_RECORD_COUNT,
        **FIXED_TARGET_COUNTS,
    }


# Compatibility for callers that display the original 484-record shape only.
TARGET_COUNTS = expected_split_counts(484)


class RawSplitRecord(Protocol):
    raw_id: str | None
    raw_content: str
    raw_sample_type: str | None
    source_platform: str | None
    context_content: str | None
    original_url: str | None
    platform_url_available: bool | None
    collected_at: str | None
    screening_status: object


def stable_rank(dataset_sha256: str, raw_id: str) -> str:
    return hashlib.sha256(f"{dataset_sha256}:{raw_id}".encode("utf-8")).hexdigest()


def leakage_group(record: RawSplitRecord) -> str:
    normalized = unicodedata.normalize("NFKC", record.raw_content).strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _record_fields(record: RawSplitRecord) -> tuple[str, str, str, bool]:
    raw_id = record.raw_id
    raw_sample_type = record.raw_sample_type
    source_platform = record.source_platform
    platform_url_available = record.platform_url_available
    if raw_id is None or raw_sample_type is None or source_platform is None or platform_url_available is None:
        raise ValueError("拆分要求原始 ID、类型、平台和链接可用性均存在")
    return raw_id, raw_sample_type, source_platform, platform_url_available


def _group_rank(dataset_sha256: str, group: list[RawSplitRecord]) -> str:
    return min(stable_rank(dataset_sha256, _record_fields(record)[0]) for record in group)


def _is_short(record: RawSplitRecord) -> bool:
    return len(record.raw_content) <= 50


def _is_compound(record: RawSplitRecord) -> bool:
    return any(marker in record.raw_content for marker in ("但", "却", "同时", "一边", "，", "、", ";", "；"))


def _gold_coverage_features(group: list[RawSplitRecord]) -> set[str]:
    features: set[str] = set()
    for record in group:
        _, raw_sample_type, source_platform, platform_url_available = _record_fields(record)
        features.add(f"platform:{source_platform}")
        features.add(f"raw_sample_type:{raw_sample_type}")
        features.add("text_length:short" if _is_short(record) else "text_length:long")
        features.add("expression:compound" if _is_compound(record) else "expression:single")
        features.add("experience:product" if "体验" in raw_sample_type else "experience:boundary")
        features.add("url:available" if platform_url_available else "url:unavailable")
    return features


def _challenge_coverage_features(group: list[RawSplitRecord]) -> set[str]:
    features: set[str] = set()
    for record in group:
        _, raw_sample_type, source_platform, platform_url_available = _record_fields(record)
        features.add(f"platform:{source_platform}")
        features.add(f"raw_sample_type:{raw_sample_type}")
        features.add("url:available" if platform_url_available else "url:unavailable")
        features.add("context:available" if record.context_content else "context:unavailable")
        features.add("collected_at:available" if record.collected_at else "collected_at:unavailable")
        features.add("text_length:short" if _is_short(record) else "text_length:long")
        features.add("expression:compound" if _is_compound(record) else "expression:single")
    return features


def _stratum_key(record: RawSplitRecord) -> str:
    _, raw_sample_type, source_platform, _ = _record_fields(record)
    return f"{source_platform}\x1f{raw_sample_type}"


def _holdout_stratum_targets(groups: list[list[RawSplitRecord]]) -> dict[str, int]:
    total_count = sum(map(len, groups))
    counts = Counter(_stratum_key(record) for group in groups for record in group)
    quotas = {
        stratum: count * FIXED_TARGET_COUNTS[DatasetSplitRole.HOLDOUT] / total_count
        for stratum, count in counts.items()
    }
    targets = {stratum: int(quota) for stratum, quota in quotas.items()}
    remaining = FIXED_TARGET_COUNTS[DatasetSplitRole.HOLDOUT] - sum(targets.values())
    for stratum in sorted(quotas, key=lambda item: (-(quotas[item] - targets[item]), item))[:remaining]:
        targets[stratum] += 1
    return dict(sorted(targets.items()))


def _can_select_exact(groups: list[list[RawSplitRecord]], target: int) -> bool:
    if target < 0:
        return False
    sizes = Counter(map(len, groups))
    if set(sizes) <= {1, 2}:
        singletons = sizes[1]
        pairs = sizes[2]
        minimum_pairs = max(0, (target - singletons + 1) // 2)
        maximum_pairs = min(pairs, target // 2)
        return minimum_pairs <= maximum_pairs
    reachable = 1
    mask = (1 << (target + 1)) - 1
    for group in groups:
        size = len(group)
        reachable |= (reachable << size) & mask
    return bool(reachable & (1 << target))


def _select_groups(
    groups: list[list[RawSplitRecord]],
    *,
    target: int,
    dataset_sha256: str,
    coverage_features,
    stratum_targets: dict[str, int] | None = None,
) -> tuple[list[list[RawSplitRecord]], list[list[RawSplitRecord]]]:
    rank_by_group = {id(group): _group_rank(dataset_sha256, group) for group in groups}
    features_by_group = {id(group): coverage_features(group) for group in groups}
    remaining = sorted(groups, key=lambda group: rank_by_group[id(group)])
    selected: list[list[RawSplitRecord]] = []
    covered: set[str] = set()
    selected_strata: Counter[str] = Counter()
    remaining_count = target

    while remaining_count:
        eligible: list[tuple[tuple[int, int, str], int, list[RawSplitRecord]]] = []
        for index, group in enumerate(remaining):
            if len(group) > remaining_count:
                continue
            candidates = remaining[:index] + remaining[index + 1 :]
            if not _can_select_exact(candidates, remaining_count - len(group)):
                continue
            feature_gain = len(features_by_group[id(group)] - covered)
            if stratum_targets is None:
                priority = (-feature_gain, 0, rank_by_group[id(group)])
            else:
                next_strata = selected_strata + Counter(_stratum_key(record) for record in group)
                deviation = sum(abs(next_strata[stratum] - expected) for stratum, expected in stratum_targets.items())
                priority = (deviation, -feature_gain, rank_by_group[id(group)])
            eligible.append((priority, index, group))
        if not eligible:
            raise ValueError(_QUOTA_ERROR)
        _, index, group = min(eligible, key=lambda item: item[0])
        selected.append(group)
        covered.update(features_by_group[id(group)])
        selected_strata.update(_stratum_key(record) for record in group)
        remaining.pop(index)
        remaining_count -= len(group)
    return selected, remaining


def _group_records(records: Iterable[RawSplitRecord]) -> list[list[RawSplitRecord]]:
    grouped: dict[str, list[RawSplitRecord]] = defaultdict(list)
    raw_ids: set[str] = set()
    for record in records:
        raw_id, _, _, _ = _record_fields(record)
        if raw_id in raw_ids:
            raise ValueError("拆分要求原始 ID 唯一")
        raw_ids.add(raw_id)
        grouped[leakage_group(record)].append(record)
    return list(grouped.values())


def build_split_manifest(
    records: Iterable[RawSplitRecord],
    *,
    dataset_sha256: str,
    dataset_version: str = DATASET_VERSION,
) -> DatasetSplitManifest:
    if re.fullmatch(r"[0-9a-f]{64}", dataset_sha256) is None:
        raise ValueError("dataset_sha256 必须是 64 位 SHA-256")
    record_list = list(records)
    targets = expected_split_counts(len(record_list))
    non_keep = [
        _record_fields(record)[0]
        for record in record_list
        if getattr(getattr(record, "screening_status", None), "value", getattr(record, "screening_status", None))
        != "KEEP"
    ]
    if non_keep:
        raise ValueError(f"拆分只允许 KEEP 记录: {non_keep[0]}")
    groups = _group_records(record_list)

    gold, remaining = _select_groups(
        groups,
        target=targets[DatasetSplitRole.GOLD],
        dataset_sha256=dataset_sha256,
        coverage_features=_gold_coverage_features,
    )
    holdout_targets = _holdout_stratum_targets(groups)
    holdout, remaining = _select_groups(
        remaining,
        target=targets[DatasetSplitRole.HOLDOUT],
        dataset_sha256=dataset_sha256,
        coverage_features=_gold_coverage_features,
        stratum_targets=holdout_targets,
    )
    challenge_pool, analysis = _select_groups(
        remaining,
        target=targets[DatasetSplitRole.CHALLENGE_POOL],
        dataset_sha256=dataset_sha256,
        coverage_features=_challenge_coverage_features,
    )
    if sum(map(len, analysis)) != targets[DatasetSplitRole.ANALYSIS]:
        raise ValueError(_QUOTA_ERROR)

    assignments: list[DatasetSplitAssignment] = []
    for role, selected_groups in (
        (DatasetSplitRole.ANALYSIS, analysis),
        (DatasetSplitRole.GOLD, gold),
        (DatasetSplitRole.CHALLENGE_POOL, challenge_pool),
        (DatasetSplitRole.HOLDOUT, holdout),
    ):
        for group in selected_groups:
            group_id = leakage_group(group[0])
            for record in group:
                raw_id, raw_sample_type, source_platform, platform_url_available = _record_fields(record)
                assignments.append(
                    DatasetSplitAssignment(
                        raw_id=raw_id,
                        split_role=role,
                        source_platform=source_platform,
                        raw_sample_type=raw_sample_type,
                        platform_url_available=platform_url_available,
                        leakage_group=group_id,
                    )
                )
    manifest = DatasetSplitManifest(
        dataset_version=dataset_version,
        dataset_sha256=dataset_sha256,
        split_algorithm_version=SPLIT_ALGORITHM_VERSION,
        holdout_stratum_targets=holdout_targets,
        holdout_stratum_tolerance=HOLDOUT_STRATUM_TOLERANCE,
        assignments=sorted(assignments, key=lambda assignment: assignment.raw_id),
    )
    validate_split_manifest(manifest, record_list, dataset_sha256=dataset_sha256)
    return manifest


def validate_split_manifest(
    manifest: DatasetSplitManifest,
    records: Iterable[RawSplitRecord],
    *,
    dataset_sha256: str | None = None,
) -> None:
    record_list = list(records)
    targets = expected_split_counts(len(record_list))
    non_keep = [
        _record_fields(record)[0]
        for record in record_list
        if getattr(getattr(record, "screening_status", None), "value", getattr(record, "screening_status", None))
        != "KEEP"
    ]
    if non_keep:
        raise ValueError(f"拆分只允许 KEEP 记录: {non_keep[0]}")
    if dataset_sha256 is not None and manifest.dataset_sha256 != dataset_sha256:
        raise ValueError("split manifest 数据集 SHA-256 不一致")
    if manifest.split_algorithm_version != SPLIT_ALGORITHM_VERSION:
        raise ValueError("split manifest 算法版本不一致")
    input_records = {_record_fields(record)[0]: record for record in record_list}
    assignment_ids = [assignment.raw_id for assignment in manifest.assignments]
    if assignment_ids != sorted(assignment_ids) or len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("split manifest 原始 ID 必须唯一且排序稳定")
    if set(assignment_ids) != set(input_records):
        raise ValueError("split manifest 原始 ID 与数据集不一致")
    counts = defaultdict(int)
    roles_by_group: dict[str, set[DatasetSplitRole]] = defaultdict(set)
    for assignment in manifest.assignments:
        record = input_records[assignment.raw_id]
        _, raw_sample_type, source_platform, platform_url_available = _record_fields(record)
        if (
            assignment.raw_sample_type != raw_sample_type
            or assignment.source_platform != source_platform
            or assignment.platform_url_available != platform_url_available
            or assignment.leakage_group != leakage_group(record)
        ):
            raise ValueError("split manifest 分层字段或记录内容不一致")
        counts[assignment.split_role] += 1
        roles_by_group[assignment.leakage_group].add(assignment.split_role)
    if dict(counts) != targets:
        raise ValueError(_QUOTA_ERROR)
    if any(len(roles) != 1 for roles in roles_by_group.values()):
        raise ValueError("泄漏组不得跨集合")
    expected_holdout_targets = _holdout_stratum_targets(_group_records(record_list))
    if manifest.holdout_stratum_targets != expected_holdout_targets:
        raise ValueError("split manifest HOLDOUT 分层目标不一致")
    if manifest.holdout_stratum_tolerance != HOLDOUT_STRATUM_TOLERANCE:
        raise ValueError("split manifest HOLDOUT 分层容差不一致")
    holdout_counts = Counter(
        f"{assignment.source_platform}\x1f{assignment.raw_sample_type}"
        for assignment in manifest.assignments
        if assignment.split_role is DatasetSplitRole.HOLDOUT
    )
    if any(
        abs(holdout_counts[stratum] - expected) > manifest.holdout_stratum_tolerance
        for stratum, expected in manifest.holdout_stratum_targets.items()
    ):
        raise ValueError("split manifest HOLDOUT 分层偏差超出容差")
