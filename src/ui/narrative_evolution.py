"""Task 4 的纯 UI/view-model 数据链路。

本模块只读取调用方显式提供的快照目录和表 9 最新记录，不调用模型，也不生成分析结果。
"""

from __future__ import annotations

from difflib import SequenceMatcher
from dataclasses import dataclass
from html import escape
import json
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, TypedDict

from src.schemas import DecisionEvolutionCheckpoint
from src.services.demo_package import load_published_demo


class SnapshotPair(TypedDict):
    baseline: dict[str, Any]
    delta: dict[str, Any]


_ACTION_TO_ROLE = {
    "SHOW_BASELINE": "BASELINE",
    "RESET_BASELINE": "BASELINE",
    "ACTIVATE_DELTA": "DELTA",
    "CONTINUE_TEST": "DELTA",
    "REVISE": "DELTA",
    "STOP": "DELTA",
}
_SNAPSHOT_METADATA = ("run_id", "snapshot_id", "generation_mode", "snapshot_role")
_DIMENSIONS = (
    "用户冲突支持度",
    "原始证据覆盖度",
    "品牌事实适配度",
    "竞品差异性",
    "场景兑现与市场可测试性",
)
_RISK_STATUSES = {"PASS", "REVISE", "BLOCK"}
_EVIDENCE_FIELDS = ("evidence_id", "raw_id", "impact", "reason")
_CANDIDATE_CHANGE_FIELDS = ("change_id", "candidate_id", "status", "reason")
_EVOLUTION_DIMENSIONS = (
    ("evidence_strength", "证据充分度"),
    ("emotional_tension", "情绪冲突张力"),
    ("meijian_fit_and_exclusivity", "梅见适配与独占性"),
    ("competitor_difference", "竞品差异度"),
    ("scene_conversion", "场景转化能力"),
)
_CHECKPOINT_IDS = tuple(f"checkpoint-{index:02d}" for index in range(5))


@dataclass(frozen=True)
class NarrativeChangeProjection:
    """Frozen, presentational projection for one candidate at one score node."""

    candidate_id: str
    checkpoint_index: int
    checkpoint_id: str
    changed: bool
    label: str
    before: str
    after: str
    patches: tuple[dict[str, Any], ...]
    diff_html: str | None

    @property
    def diff(self) -> str | None:
        """Compatibility alias for callers that call the rendered field ``diff``."""

        return self.diff_html


def snapshot_for_action(action: str) -> str:
    """返回动作对应的快照角色；不认识的动作直接失败。"""

    if not isinstance(action, str) or action not in _ACTION_TO_ROLE:
        raise ValueError(f"未知动作：{action!r}")
    return _ACTION_TO_ROLE[action]


def load_snapshot_pair(snapshot_dir: str | PathLike[str]) -> SnapshotPair:
    """从显式目录读取并校验 baseline_snapshot.json 与 delta_snapshot.json。"""

    if not isinstance(snapshot_dir, (str, PathLike)):
        raise ValueError("snapshot_dir 必须是显式目录路径")
    directory = Path(snapshot_dir)
    if not directory.exists():
        raise ValueError(f"快照目录不存在：{directory}")
    if not directory.is_dir():
        raise ValueError(f"snapshot_dir 不是目录：{directory}")

    loaded: dict[str, dict[str, Any]] = {}
    for role, filename in (("baseline", "baseline_snapshot.json"), ("delta", "delta_snapshot.json")):
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"缺少快照文件：{path.name}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"快照文件无法读取或 JSON 无法解析：{path}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"快照文件 JSON 根节点必须是对象：{path.name}")
        _validate_snapshot_metadata(payload, expected_role=role.upper(), filename=filename)
        loaded[role] = payload

    if loaded["baseline"]["run_id"] != loaded["delta"]["run_id"]:
        raise ValueError("Baseline 与 Delta 的 run_id 不一致")
    return SnapshotPair(baseline=loaded["baseline"], delta=loaded["delta"])


def load_evolution_checkpoints(
    published_path: str | PathLike[str],
) -> list[DecisionEvolutionCheckpoint]:
    """Project only the checkpoint order from the authoritative frozen-package loader."""

    package_root = Path(published_path).parent
    try:
        published = load_published_demo(package_root)
    except ValueError as exc:
        raise ValueError(f"冻结包已阻断：{exc}") from exc

    checkpoints = list(published.attempt.checkpoints)
    for release_index, checkpoint in enumerate(checkpoints):
        expected_id = _CHECKPOINT_IDS[release_index]
        if checkpoint.checkpoint_id != expected_id:
            raise ValueError(f"冻结包检查点顺序不连续：{expected_id}")
        if checkpoint.release_index != release_index:
            raise ValueError(f"冻结包 release_index 不连续：{expected_id}")
    return checkpoints


def build_evolution_view_model(
    checkpoint: DecisionEvolutionCheckpoint,
    *,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Project an already frozen checkpoint; it never recalculates a decision."""

    projected_candidate_id = candidate_id or checkpoint.selected_candidate_id
    selected = next(
        snapshot
        for snapshot in checkpoint.candidates
        if snapshot.ranked_narrative.candidate.candidate_id == projected_candidate_id
    )
    ranked = selected.ranked_narrative
    scores = ranked.evaluation.scores
    return {
        "checkpoint_id": checkpoint.checkpoint_id,
        "release_index": checkpoint.release_index,
        "selected_candidate_id": checkpoint.selected_candidate_id,
        "projected_candidate_id": projected_candidate_id,
        "candidate_ids": [snapshot.ranked_narrative.candidate.candidate_id for snapshot in checkpoint.candidates],
        "weighted_score": ranked.weighted_score,
        "rank": ranked.rank,
        "business_status": selected.business_status.value,
        "evidence_count": len(checkpoint.visible_evidence_ids),
        "supporting_count": selected.supporting_count,
        "counter_count": selected.counter_count,
        "risk_count": selected.risk_count,
        "route_distribution": dict(checkpoint.route_distribution),
        "experience_distribution": dict(checkpoint.experience_distribution),
        "grade_distribution": dict(checkpoint.grade_distribution),
        "dimensions": [
            {"key": key, "name": name, "score": getattr(scores, key).score,
             "reason": getattr(scores, key).rationale}
            for key, name in _EVOLUTION_DIMENSIONS
        ],
        "candidate": ranked.candidate,
        "patches": list(checkpoint.patches),
        "switch_suggestion": checkpoint.switch_suggestion,
    }


def narrative_change(
    run: Any,
    candidate_id: str,
    checkpoint_index: int,
) -> NarrativeChangeProjection:
    """Project only the patch recorded on one frozen numeric point.

    ``checkpoint_index`` is the six-node score-chart index: 0 is ``ai-original``;
    1--5 are ``checkpoint-00`` through ``checkpoint-04``.
    """

    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("candidate_id 必须是非空字符串")
    if not isinstance(checkpoint_index, int) or isinstance(checkpoint_index, bool):
        raise ValueError("checkpoint_index 必须是整数")

    points = getattr(run, "numeric_points", None)
    if not isinstance(points, (tuple, list)) or len(points) != 6:
        raise ValueError("正式演化 view 必须包含六个 numeric_points")
    if not 0 <= checkpoint_index < len(points):
        raise ValueError("checkpoint_index 必须在 0 到 5 之间")

    point = points[checkpoint_index]
    candidates = getattr(point, "candidates", None)
    if not isinstance(candidates, (tuple, list)):
        raise ValueError("numeric point 缺少 candidates")
    candidate = next(
        (
            item
            for item in candidates
            if getattr(item, "candidate_id", None) == candidate_id
        ),
        None,
    )
    if candidate is None:
        raise ValueError(f"正式演化 view 缺少候选 {candidate_id}: {getattr(point, 'checkpoint_id', '')}")

    raw_patches = getattr(candidate, "patches", None)
    if not isinstance(raw_patches, (tuple, list)):
        raise ValueError("candidate patches 必须是列表或元组")
    patches = tuple(raw_patches)
    if not all(isinstance(patch, Mapping) for patch in patches):
        raise ValueError("candidate patches 必须是对象列表")

    presentation_text = getattr(candidate, "presentation_text", None)
    if not isinstance(presentation_text, str) or not presentation_text:
        raise ValueError("candidate presentation_text 无效")

    if patches:
        before_values: list[str] = []
        after_values: list[str] = []
        for patch in patches:
            for field_name in ("before", "after"):
                value = patch.get(field_name)
                if not isinstance(value, str):
                    raise ValueError(f"patch 缺少有效 {field_name}")
            before_values.append(patch["before"])
            after_values.append(patch["after"])
        before = before_values[0]
        after = after_values[-1]
        if before == after:
            raise ValueError("patch before 与 after 不得相同")
        diff_html: str | None = render_word_diff(before, after)
        changed = True
        label = "本轮文案已调整"
    else:
        before = presentation_text
        after = presentation_text
        diff_html = None
        changed = False
        label = "本轮文案未调整"

    checkpoint_id = getattr(point, "checkpoint_id", None)
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("numeric point 缺少有效 checkpoint_id")
    return NarrativeChangeProjection(
        candidate_id=candidate_id,
        checkpoint_index=checkpoint_index,
        checkpoint_id=checkpoint_id,
        changed=changed,
        label=label,
        before=before,
        after=after,
        patches=patches,
        diff_html=diff_html,
    )


def render_word_diff(before: str | list[str], after: str | list[str]) -> str:
    """Return escaped, presentational-only text diff markup for a frozen patch."""

    old_tokens = list(_as_text(before))
    new_tokens = list(_as_text(after))
    parts: list[str] = []
    for operation, old_start, old_end, new_start, new_end in SequenceMatcher(
        a=old_tokens, b=new_tokens
    ).get_opcodes():
        if operation in {"delete", "replace"}:
            parts.append(_diff_span("word-diff-remove", old_tokens[old_start:old_end]))
        if operation == "equal":
            parts.append(escape("".join(old_tokens[old_start:old_end])))
        if operation in {"insert", "replace"}:
            parts.append(_diff_span("word-diff-add", new_tokens[new_start:new_end]))
    return "".join(parts)


def _as_text(value: str | list[str]) -> str:
    return value if isinstance(value, str) else "；".join(value)


def _diff_span(css_class: str, tokens: list[str]) -> str:
    if not tokens:
        return ""
    return f'<span class="{css_class}">{escape("".join(tokens))}</span>'


def build_view_model(
    latest_table9_record: Mapping[str, Any], snapshot_pair: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """把表 9 最新记录和已加载快照投影成普通 dict view model。"""

    if not isinstance(latest_table9_record, Mapping):
        raise ValueError("latest_table9_record 必须是对象")
    fields = latest_table9_record.get("fields", latest_table9_record)
    if not isinstance(fields, Mapping):
        raise ValueError("latest_table9_record.fields 必须是对象")

    action = fields.get("action")
    role = snapshot_for_action(action)
    snapshot = snapshot_pair.get(role.lower())
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"snapshot_pair 缺少 {role.lower()} 快照")

    record_run_id = fields.get("run_id")
    if record_run_id is not None and record_run_id != snapshot.get("run_id"):
        raise ValueError("表 9 记录与快照的 run_id 不一致")
    record_snapshot_id = fields.get("snapshot_id")
    if record_snapshot_id is not None and record_snapshot_id != snapshot.get("snapshot_id"):
        raise ValueError("表 9 记录与动作对应快照的 snapshot_id 不一致")

    analysis = snapshot.get("analysis")
    analysis_payload = analysis if isinstance(analysis, Mapping) else snapshot
    evidence_signals = _project_records(
        analysis_payload.get("evidence_impacts", snapshot.get("evidence_impacts")),
        required_fields=_EVIDENCE_FIELDS,
        label="evidence_impacts",
    )
    candidate_changes = _project_records(
        analysis_payload.get("candidate_changes", snapshot.get("candidate_changes")),
        required_fields=_CANDIDATE_CHANGE_FIELDS,
        label="candidate_changes",
    )
    risk_status = analysis_payload.get("risk_status", snapshot.get("risk_status", "未提供"))
    if risk_status is None:
        risk_status = "未提供"
    if risk_status != "未提供" and risk_status not in _RISK_STATUSES:
        raise ValueError("risk_status 只允许 PASS、REVISE、BLOCK")

    dimensions = _project_dimensions(analysis_payload.get("dimensions", snapshot.get("dimensions")))
    is_delta = role == "DELTA"
    first_text = (
        "20条策展式演示批次带来的新证据与候选变化：只展示快照已提供的分析字段。"
        if is_delta
        else "基线语料形成的核心洞察、候选叙事及其证据：只展示快照已提供的分析字段。"
    )
    first_screen = [
        {"title": "基线洞察", "text": first_text},
        {
            "title": "新证据信号",
            "text": "已提供快照中的证据信号。" if evidence_signals else "未提供",
        },
        {
            "title": "候选变化",
            "text": "已提供快照中的候选变化。" if candidate_changes else "未提供",
        },
        {"title": "下一步验证", "text": _next_step_text(analysis_payload, snapshot)},
    ]

    return {
        "action": action,
        "snapshot_role": role,
        "run_id": snapshot["run_id"],
        "snapshot_id": snapshot["snapshot_id"],
        "generation_mode": snapshot["generation_mode"],
        "first_screen": first_screen,
        "dimensions": dimensions,
        "risk_status": risk_status,
        "evidence_signals": evidence_signals,
        "candidate_changes": candidate_changes,
    }


def _validate_snapshot_metadata(
    payload: Mapping[str, Any], *, expected_role: str, filename: str
) -> None:
    missing = [
        field
        for field in _SNAPSHOT_METADATA
        if not isinstance(payload.get(field), str) or not payload[field].strip()
    ]
    if missing:
        raise ValueError(f"{filename} 缺少字段或字段为空：{', '.join(missing)}")
    if payload["snapshot_role"] != expected_role:
        raise ValueError(
            f"{filename} 的 snapshot_role 必须是 {expected_role}，实际为 {payload['snapshot_role']!r}"
        )


def _project_dimensions(raw_dimensions: Any) -> list[dict[str, Any]]:
    if raw_dimensions is not None and not isinstance(raw_dimensions, Mapping):
        raise ValueError("dimensions 必须是对象")
    dimensions = raw_dimensions if isinstance(raw_dimensions, Mapping) else {}
    result: list[dict[str, Any]] = []
    for name in _DIMENSIONS:
        raw_value = dimensions.get(name)
        if isinstance(raw_value, Mapping):
            result.append(
                {
                    "name": name,
                    "score": raw_value.get("score", "未提供"),
                    "status": raw_value.get("status", "未提供"),
                    "reason": raw_value.get("reason", raw_value.get("rationale", "未提供")),
                }
            )
        elif raw_value is None:
            result.append({"name": name, "score": "未提供", "status": "未提供", "reason": "未提供"})
        else:
            result.append({"name": name, "score": raw_value, "status": "未提供", "reason": "未提供"})
    return result


def _project_records(raw_records: Any, *, required_fields: tuple[str, ...], label: str) -> list[dict[str, Any]]:
    if raw_records is None:
        return []
    if not isinstance(raw_records, list):
        raise ValueError(f"{label} 必须是列表")
    projected: list[dict[str, Any]] = []
    for index, record in enumerate(raw_records, start=1):
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} 第 {index} 条必须是对象")
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise ValueError(f"{label} 第 {index} 条缺少字段：{', '.join(missing)}")
        projected.append({field: record[field] for field in required_fields})
    return projected


def _next_step_text(analysis_payload: Mapping[str, Any], snapshot: Mapping[str, Any]) -> str:
    next_steps = analysis_payload.get("next_steps", snapshot.get("next_steps"))
    if next_steps is None:
        return "未提供"
    if not isinstance(next_steps, list):
        raise ValueError("next_steps 必须是列表")
    return "；".join(str(item) for item in next_steps) if next_steps else "未提供"


__all__ = [
    "NarrativeChangeProjection",
    "SnapshotPair",
    "build_evolution_view_model",
    "build_view_model",
    "load_evolution_checkpoints",
    "load_snapshot_pair",
    "narrative_change",
    "render_word_diff",
    "snapshot_for_action",
]
