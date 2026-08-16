"""Minimal, auditable entry point for the official decision run."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings
from src.llm_client import LLMClient
from src.schemas import (
    DecisionFoundationState,
    DecisionModelProfile,
    DecisionRunManifest,
    DecisionRunStage,
    ModelRuntime,
    PendingSelectionPackage,
    CandidateSelection,
    OfflineSpecificityTask,
)
from src.services.decision_inputs import (
    DecisionInputBundle,
    assemble_official_demo_inputs,
)
from src.services.gold_decision_input import load_human_gold_decision_input
from src.services.decision_run_coordinator import DecisionRunCoordinator
from src.services.decision_run_store import (
    AuditedLLMClient,
    DecisionRunStore,
    _validate_run_id,
)
from src.services.decision_runner import DecisionRunner
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes
from src.services.brand_specificity_offline import write_specificity_task
from tools import preflight_decision_run as formal_preflight


EXPECTED_COUNTS = {"ANALYSIS": 249, "CHALLENGE": 60, "HOLDOUT": 60}
EXPECTED_PACKAGE_IDS = {
    "ANALYSIS": "formal-analysis",
    "CHALLENGE": "formal-challenge",
    "HOLDOUT": "formal-holdout",
}
EXPECTED_PROFILE = {
    "provider": "deepseek",
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-v4-flash",
    "thinking": True,
    "reasoning_effort": "max",
    "response_format": "json_object",
    "max_retries": 0,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class OfficialPreflight:
    report: dict[str, object]
    inputs: DecisionInputBundle
    settings: Settings


def _field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _report_of(preflight: object) -> Mapping[str, object]:
    report = preflight if isinstance(preflight, Mapping) else getattr(preflight, "report", None)
    if not isinstance(report, Mapping):
        raise TypeError("preflight must be a report mapping or OfficialPreflight")
    return report


def _inputs_of(preflight: object) -> object:
    inputs = getattr(preflight, "inputs", None)
    if inputs is None:
        raise TypeError("preflight inputs are required")
    return inputs


def _scalar(value: object) -> object:
    enum_value = getattr(value, "value", None)
    return enum_value if enum_value is not None else value


def _text(value: object, default: str = "") -> str:
    value = _scalar(value)
    return default if value is None else str(value)


def _sha(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _stable_record_id(record: object) -> str:
    stable_id = _field(record, "raw_id") or _field(record, "comment_id")
    if not isinstance(stable_id, str) or not stable_id:
        raise ValueError("decision input record is missing a stable ID")
    return stable_id


def _gold_metadata(inputs: object) -> dict[str, object] | None:
    """Return auditable metadata for the human-aligned Gold input."""

    gold_input = _field(inputs, "gold_input")
    gold_records = tuple(_field(inputs, "gold_records", ()) or ())
    gold_atoms = tuple(_field(inputs, "gold_atoms", ()) or ())
    if gold_input is None and not gold_records and not gold_atoms:
        return None
    if gold_input is None:
        raise ValueError("Human Gold records require a gold_input identity")
    if len(gold_records) != 50 or len(gold_atoms) != 50:
        raise ValueError("Human Gold decision input must contain exactly 50 records and atoms")
    input_records = tuple(_field(gold_input, "records", ()) or ())
    input_atoms = tuple(_field(gold_input, "atoms", ()) or ())
    if {
        _stable_record_id(record) for record in input_records
    } != {_stable_record_id(record) for record in gold_records}:
        raise ValueError("assembled Human Gold records do not match gold_input")
    if {
        _text(_field(atom, "evidence_id")) for atom in input_atoms
    } != {_text(_field(atom, "evidence_id")) for atom in gold_atoms}:
        raise ValueError("assembled Human Gold atoms do not match gold_input")
    if any(_text(_field(atom, "label_source")) != "HUMAN_GOLD" for atom in gold_atoms):
        raise ValueError("Human Gold atoms must have label_source=HUMAN_GOLD")
    brand_non_c_grade_count = sum(
        _text(_field(atom, "route")) == "BRAND"
        and _text(_field(atom, "evidence_grade")) != "C"
        for atom in gold_atoms
    )
    return {
        "label_source": "HUMAN_GOLD",
        "count": len(gold_records),
        "record_count": len(gold_records),
        "atom_count": len(gold_atoms),
        "calibration_workbook_sha256": _require_sha(
            _field(gold_input, "calibration_sha256"),
            "Human Gold calibration workbook SHA",
        ),
        "frozen_labels_sha256": _require_sha(
            _field(gold_input, "labels_json_sha256"),
            "frozen Human Gold labels SHA",
        ),
        "annotation_result_sha256": _require_sha(
            _field(gold_input, "result_sha256"),
            "Human Gold annotation result SHA",
        ),
        "stable_ids_sha256": _require_sha(
            _field(gold_input, "stable_ids_sha256"),
            "Human Gold stable IDs SHA",
        ),
        "brand_non_c_grade_count": brand_non_c_grade_count,
    }


def _decision_input_metadata(inputs: object) -> dict[str, object]:
    baseline_records = tuple(_field(inputs, "baseline_records", ()) or ())
    baseline_atoms = tuple(_field(inputs, "baseline_atoms", ()) or ())
    gold = _gold_metadata(inputs)
    human_gold_count = int(gold["record_count"]) if gold is not None else 0
    if human_gold_count > len(baseline_records):
        raise ValueError("Human Gold count cannot exceed baseline count")
    all_decision_sha = _field(inputs, "all_decision_stable_ids_sha256")
    return {
        "baseline_count": len(baseline_records),
        "baseline_atom_count": len(baseline_atoms),
        "analysis_baseline_count": len(baseline_records) - human_gold_count,
        "human_gold_count": human_gold_count,
        "all_decision_stable_ids_sha256": (
            _require_sha(all_decision_sha, "all decision stable IDs SHA")
            if all_decision_sha is not None
            else None
        ),
        "gold": gold,
    }


def _validate_official_assembled_inputs(inputs: object) -> dict[str, object]:
    baseline_records = tuple(_field(inputs, "baseline_records", ()) or ())
    metadata = _decision_input_metadata(inputs)
    if (
        metadata["baseline_count"] != 279
        or metadata["baseline_atom_count"] != 279
        or metadata["analysis_baseline_count"] != 229
        or metadata["human_gold_count"] != 50
        or len(tuple(_field(inputs, "challenge_atoms", ()) or ())) != 60
        or len(tuple(_field(inputs, "holdout_atoms", ()) or ())) != 60
    ):
        raise ValueError(
            "assembled official DecisionInputBundle must be ANALYSIS baseline 229 + Human Gold 50, "
            "CHALLENGE 60, HOLDOUT 60"
        )
    if len({_stable_record_id(record) for record in baseline_records}) != 279:
        raise ValueError("assembled official baseline records must have 279 unique stable IDs")
    gold = metadata["gold"]
    if not isinstance(gold, Mapping):
        raise ValueError("official decision baseline must include human-aligned Gold")
    if gold.get("brand_non_c_grade_count") != 4:
        raise ValueError("Human Gold must contain exactly four non-C BRAND evidence atoms")
    if metadata["all_decision_stable_ids_sha256"] is None:
        raise ValueError("all decision stable IDs SHA is required when Human Gold is included")
    return metadata


def _validate_formal_report(report: Mapping[str, object]) -> None:
    if report.get("status") != "PASS":
        raise ValueError("formal preflight did not PASS")
    if report.get("counts") != EXPECTED_COUNTS:
        raise ValueError("formal preflight counts must be ANALYSIS=249, CHALLENGE=60, HOLDOUT=60")
    if report.get("unique") != 369:
        raise ValueError("formal preflight must contain 369 unique stable IDs")
    if report.get("network_calls") != 0:
        raise ValueError("formal preflight must complete with network_calls=0")

    profile = report.get("profile")
    if not isinstance(profile, Mapping):
        raise ValueError("formal preflight profile is missing")
    for name, expected in EXPECTED_PROFILE.items():
        if profile.get(name) != expected:
            raise ValueError(f"formal DeepSeek profile field {name!r} is invalid")

    prompt = report.get("prompt")
    if not isinstance(prompt, Mapping) or prompt.get("name") != "evidence_routing":
        raise ValueError("formal evidence_routing prompt is required")
    _require_sha(prompt.get("sha256"), "formal prompt SHA")

    packages = report.get("packages")
    if not isinstance(packages, Mapping) or set(packages) != set(EXPECTED_PACKAGE_IDS):
        raise ValueError("formal package roles are incomplete")
    for role, package_id in EXPECTED_PACKAGE_IDS.items():
        package = packages.get(role)
        if not isinstance(package, Mapping) or package.get("package_id") != package_id:
            raise ValueError(f"formal {role} package identity is invalid")
        _require_sha(package.get("package_sha256"), f"formal {role} package SHA")

    _require_sha(report.get("all_stable_ids_sha256"), "all stable IDs SHA")
    if report.get("prepared_group_sha256") is not None:
        _require_sha(report.get("prepared_group_sha256"), "prepared group SHA")


def _package_report(report: Mapping[str, object], role: str) -> Mapping[str, object]:
    packages = report["packages"]
    if not isinstance(packages, Mapping):
        raise ValueError("formal package report is invalid")
    package = packages[role]
    if not isinstance(package, Mapping):
        raise ValueError(f"formal {role} package report is invalid")
    return package


def build_run_manifest(
    *,
    run_id: str,
    preflight: object,
    created_at: datetime | None = None,
) -> DecisionRunManifest:
    """Build the immutable run manifest from the strict preflight report."""

    _validate_run_id(run_id)
    report = _report_of(preflight)
    _validate_formal_report(report)
    profile = report["profile"]
    prompt = report["prompt"]
    if not isinstance(profile, Mapping) or not isinstance(prompt, Mapping):
        raise ValueError("formal profile and prompt are required")

    package_ids = {
        role: str(_package_report(report, role)["package_id"])
        for role in EXPECTED_PACKAGE_IDS
    }
    package_sha256 = {
        role: str(_package_report(report, role)["package_sha256"])
        for role in EXPECTED_PACKAGE_IDS
    }
    artifact_identity = {
        "run_id": run_id,
        "package_ids": package_ids,
        "package_sha256": package_sha256,
        "all_stable_ids_sha256": report["all_stable_ids_sha256"],
        "prepared_group_sha256": report.get("prepared_group_sha256"),
        "prompt": dict(prompt),
        "profile": dict(profile),
        "annotation": report.get("annotation"),
        "gold": report.get("gold"),
        "decision_inputs": report.get("decision_inputs"),
    }
    model_profile = DecisionModelProfile(
        profile_id="official-deepseek-v4-flash-thinking-max",
        provider=str(profile["provider"]),
        model_id=str(profile["model"]),
        runtime=ModelRuntime.CLOUD,
        response_format=str(profile["response_format"]),
        prompt_version=str(prompt["version"]),
        endpoint_profile="deepseek-official",
        thinking_enabled=bool(profile["thinking"]),
        reasoning_effort=str(profile["reasoning_effort"]),
        max_retries=int(profile["max_retries"]),
    )
    specificity_model = DecisionModelProfile(
        profile_id="codex-luna-specificity-max-v1",
        provider="codex",
        model_id="gpt-5.6-luna",
        runtime=ModelRuntime.CLOUD,
        response_format="json_schema",
        prompt_version="brand-specificity-v1",
        endpoint_profile="codex-offline",
        thinking_enabled=True,
        reasoning_effort="max",
        max_retries=0,
    )
    return DecisionRunManifest(
        decision_run_id=run_id,
        package_ids=package_ids,
        package_sha256=package_sha256,
        decision_model=model_profile,
        specificity_model=specificity_model,
        created_at=created_at or datetime.now(timezone.utc),
        run_artifact_sha256=_sha(artifact_identity),
    )


def _reject_existing_run(run_root: Path, run_id: str) -> None:
    _validate_run_id(run_id)
    target = Path(run_root) / run_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"run_id {run_id!r} already exists")


def load_official_preflight(
    *,
    prepared_root: Path,
    run_root: Path,
    run_id: str,
    gold_loader: Callable[[], object] | None = None,
) -> OfficialPreflight:
    """Run the strict zero-network gate and assemble formal plus Human Gold inputs."""

    _reject_existing_run(Path(run_root), run_id)
    output_root = Path(run_root) / run_id
    raw_report = formal_preflight.run_preflight(
        prepared_root=Path(prepared_root),
        run_root=Path(run_root),
        output_root=output_root,
    )
    if not isinstance(raw_report, Mapping):
        raise TypeError("formal preflight must return a report mapping")
    settings = formal_preflight.load_official_settings()
    settings.validate_official_decision_profile()

    group, packages = formal_preflight._load_group(Path(prepared_root))
    selection_manifest = formal_preflight._validate_boundaries(group, packages)
    gold_input = (gold_loader or load_human_gold_decision_input)()
    inputs = assemble_official_demo_inputs(
        analysis=packages["ANALYSIS"],
        challenge=packages["CHALLENGE"],
        holdout=packages["HOLDOUT"],
        selection_manifest=selection_manifest,
        gold_input=gold_input,
    )
    input_ids_sha256 = _require_sha(
        _field(inputs, "all_stable_ids_sha256"),
        "assembled all stable IDs SHA",
    )
    group_sha256 = _require_sha(
        _field(group, "group_sha256"),
        "assembled prepared group SHA",
    )
    report = dict(raw_report)
    for name, value in (
        ("all_stable_ids_sha256", input_ids_sha256),
        ("prepared_group_sha256", group_sha256),
    ):
        existing = report.get(name)
        if existing is not None and existing != value:
            raise ValueError(f"formal preflight {name} does not match assembled inputs")
        report[name] = value
    decision_inputs = _validate_official_assembled_inputs(inputs)
    report["gold"] = decision_inputs["gold"]
    report["decision_inputs"] = decision_inputs
    report["all_decision_stable_ids_sha256"] = decision_inputs[
        "all_decision_stable_ids_sha256"
    ]
    _validate_formal_report(report)
    return OfficialPreflight(report=report, inputs=inputs, settings=settings)


def _candidate_from_ranked(ranked: object) -> object:
    candidate = _field(ranked, "candidate")
    if candidate is None:
        raise ValueError("ranked candidate is missing candidate data")
    return candidate


def _evidence_reference(quote: object) -> str:
    comment_id = _field(quote, "comment_id")
    text = _field(quote, "quote")
    if text is None:
        return _text(comment_id)
    return f"{_text(comment_id)}: {_text(text)}"


def _stress_by_candidate(foundation: object) -> dict[str, object]:
    values = _field(foundation, "stress_results", ()) or ()
    return {_text(_field(item, "candidate_id")): item for item in values}


def _candidate_summary(ranked: object, stress: object | None = None) -> dict[str, object]:
    candidate = _candidate_from_ranked(ranked)
    return {
        "candidate_id": _text(_field(candidate, "candidate_id")),
        "title": _text(_field(candidate, "title")),
        "draft_proposition": _text(_field(candidate, "draft_proposition")),
        "main_scenes": list(_field(candidate, "main_scenes", ()) or ()),
        "rank": int(_field(ranked, "rank", 0)),
        "weighted_score": _field(ranked, "weighted_score"),
        "is_recommended": bool(_field(ranked, "is_recommended", False)),
        "supporting_evidence": [
            _evidence_reference(item)
            for item in (_field(candidate, "supporting_evidence", ()) or ())
        ],
        "counter_evidence": [
            _evidence_reference(item)
            for item in (_field(candidate, "counter_evidence", ()) or ())
        ],
        "counter_evidence_note": _field(candidate, "counter_evidence_note"),
        "stress_checks": [
            {
                "check_type": _text(_field(check, "check_type")),
                "execution_status": _text(_field(check, "execution_status")),
                "decision": _field(check, "decision"),
                "reference_ids": list(_field(check, "reference_ids", ()) or ()),
                "rationale": _text(_field(check, "rationale")),
            }
            for check in (_field(stress, "checks", ()) or ())
        ],
    }


def render_baseline_markdown(
    *,
    run_id: str,
    preflight: object,
    foundation: object,
    pending_selection: object,
) -> str:
    """Render the human-readable, evidence-linked baseline artifact."""

    report = _report_of(preflight)
    _validate_formal_report(report)
    profile = report["profile"]
    prompt = report["prompt"]
    packages = report["packages"]
    if not isinstance(profile, Mapping) or not isinstance(prompt, Mapping):
        raise ValueError("formal profile and prompt are required")
    if not isinstance(packages, Mapping):
        raise ValueError("formal packages are required")

    ranking = _field(foundation, "ranking")
    ranked_candidates = list(_field(ranking, "ranked_candidates", ()) or ())
    stress_map = _stress_by_candidate(foundation)
    recommendation_reason = _text(_field(ranking, "recommendation_reason"))
    lines = [
        f"# 正式决策基线与挑战结果",
        "",
        f"- run_id: `{run_id}`",
        f"- stage: `{_text(_field(foundation, 'challenge_applied')) and 'AWAITING_SELECTION'}`",
        f"- preflight: `ANALYSIS=249, CHALLENGE=60, HOLDOUT=60, unique=369, network_calls=0`",
        f"- model profile: `{json.dumps(dict(profile), ensure_ascii=False, sort_keys=True)}`",
        f"- prompt: `{_text(prompt.get('name'))}/{_text(prompt.get('version'))}` SHA `{_text(prompt.get('sha256'))}`",
        "",
        "## 正式数据包",
        "",
        f"- all stable IDs SHA: `{_text(report.get('all_stable_ids_sha256'))}`",
        f"- prepared group SHA: `{_text(report.get('prepared_group_sha256'))}`",
    ]
    decision_inputs = report.get("decision_inputs")
    if isinstance(decision_inputs, Mapping):
        lines.extend(
            [
                "",
                "## Decision baseline inputs",
                "",
                "- baseline: `ANALYSIS baseline 229 + human-aligned Gold 50 = 279`",
                f"- all decision stable IDs SHA: `{_text(decision_inputs.get('all_decision_stable_ids_sha256'))}`",
            ]
        )
        gold = decision_inputs.get("gold")
        if isinstance(gold, Mapping):
            lines.extend(
                [
                    "- Gold label source: `HUMAN_GOLD`",
                    f"- Gold workbook SHA: `{_text(gold.get('calibration_workbook_sha256'))}`",
                    f"- frozen Gold labels SHA: `{_text(gold.get('frozen_labels_sha256'))}`",
                    f"- Gold annotation result SHA: `{_text(gold.get('annotation_result_sha256'))}`",
                    f"- non-C BRAND Gold atoms: `{_text(gold.get('brand_non_c_grade_count'))}`",
                ]
            )
    for role in ("ANALYSIS", "CHALLENGE", "HOLDOUT"):
        package = packages[role]
        if not isinstance(package, Mapping):
            raise ValueError(f"formal {role} package is invalid")
        lines.append(
            f"- {role}: `{_text(package.get('package_id'))}` SHA `{_text(package.get('package_sha256'))}`"
        )

    lines.extend(["", "## 候选与推荐", ""])
    lines.append(f"推荐理由：{recommendation_reason}")
    for ranked in ranked_candidates:
        candidate = _candidate_from_ranked(ranked)
        candidate_id = _text(_field(candidate, "candidate_id"))
        lines.extend(
            [
                "",
                f"### {candidate_id}｜{_text(_field(candidate, 'title'))}",
                f"- 主张：{_text(_field(candidate, 'draft_proposition'))}",
                f"- 场景：{'; '.join(_text(item) for item in (_field(candidate, 'main_scenes', ()) or ())) }",
                f"- 排名：{_field(ranked, 'rank')}；总分：{_field(ranked, 'weighted_score')}；推荐：{bool(_field(ranked, 'is_recommended', False))}",
                f"- 支持证据：{'; '.join(_evidence_reference(item) for item in (_field(candidate, 'supporting_evidence', ()) or ())) }",
                f"- 反面证据：{'; '.join(_evidence_reference(item) for item in (_field(candidate, 'counter_evidence', ()) or ())) or _text(_field(candidate, 'counter_evidence_note'), '未发现')}",
            ]
        )
        stress = stress_map.get(candidate_id)
        lines.append("- 五项压力测试：")
        for check in (_field(stress, "checks", ()) or ()):
            lines.append(
                "  - "
                f"{_text(_field(check, 'check_type'))}: "
                f"{_text(_field(check, 'execution_status'))}/"
                f"{_text(_field(check, 'decision'))}; "
                f"引用={','.join(_text(item) for item in (_field(check, 'reference_ids', ()) or ())) }; "
                f"{_text(_field(check, 'rationale'))}"
            )

    lines.extend(
        [
            "",
            "## 运行边界",
            "",
            f"- foundation SHA: `{_sha(foundation)}`",
            f"- pending_selection SHA: `{_sha(pending_selection)}`",
            "- 本次入口只执行 `build_baseline + apply_challenge`，并冻结 foundation 与 pending_selection。",
            "- 未执行 selection、HOLDOUT 或盲测。",
        ]
    )
    return "\n".join(lines) + "\n"


def _stress_summary(foundation: object) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for stress in _field(foundation, "stress_results", ()) or ():
        checks = []
        for check in _field(stress, "checks", ()) or ():
            checks.append(
                {
                    "check_type": _text(_field(check, "check_type")),
                    "execution_status": _text(_field(check, "execution_status")),
                    "decision": _scalar(_field(check, "decision")),
                    "reference_ids": list(_field(check, "reference_ids", ()) or ()),
                    "rationale": _text(_field(check, "rationale")),
                }
            )
        result.append(
            {
                "candidate_id": _text(_field(stress, "candidate_id")),
                "checks": checks,
            }
        )
    return result


def build_challenge_report(
    *, run_id: str, inputs: object, foundation: object
) -> dict[str, object]:
    challenge_atoms = list(_field(inputs, "challenge_atoms", ()) or ())
    challenge_ids = [
        _text(_field(atom, "evidence_id")) for atom in challenge_atoms
    ]
    impacts = list(_field(foundation, "impacts", ()) or ())
    impact_counts = Counter(_text(_field(item, "impact")) for item in impacts)
    target_ids = sorted(
        {
            _text(target)
            for target in (_field(item, "target_id") for item in impacts)
            if target
        }
    )
    ranking = _field(foundation, "ranking")
    rerank = [
        {
            "candidate_id": _text(_field(_field(item, "candidate"), "candidate_id")),
            "rank": _field(item, "rank"),
            "weighted_score": _field(item, "weighted_score"),
            "is_recommended": bool(_field(item, "is_recommended", False)),
        }
        for item in (_field(ranking, "ranked_candidates", ()) or ())
    ]
    return {
        "run_id": run_id,
        "decision_inputs": _decision_input_metadata(inputs),
        "challenge_applied": bool(_field(foundation, "challenge_applied", False)),
        "challenge_evidence_count": len(challenge_atoms),
        "challenge_evidence_ids": challenge_ids,
        "impact_summary": {
            "total_impact_records": len(impacts),
            "by_impact": dict(sorted(impact_counts.items())),
            "target_candidate_ids": target_ids,
        },
        "revision_summary": {
            "revised_candidate_ids": target_ids,
            "revised_candidate_count": len(target_ids),
            "visible_evidence_count": len(_field(foundation, "visible_atoms", ()) or ()),
        },
        "rerank_summary": {
            "ranked_candidates": rerank,
            "recommended_candidate_id": _field(ranking, "recommended_candidate_id"),
            "recommendation_reason": _field(ranking, "recommendation_reason"),
        },
        "stress_test_summary": _stress_summary(foundation),
    }


def _snapshot_with_hash(payload: dict[str, object]) -> dict[str, object]:
    snapshot = dict(payload)
    snapshot["snapshot_sha256"] = _sha(payload)
    return snapshot


def build_snapshot_pair(
    *,
    run_id: str,
    foundation_sha256: str,
    pending_selection_sha256: str,
    stage: str = DecisionRunStage.AWAITING_SELECTION.value,
    checkpoint_id: str | None = None,
) -> dict[str, dict[str, object]]:
    """Build Feishu-compatible snapshots with immutable state metadata."""

    _require_sha(foundation_sha256, "foundation SHA")
    _require_sha(pending_selection_sha256, "pending_selection SHA")
    common = {
        "run_id": run_id,
        "stage": stage,
        "checkpoint_id": checkpoint_id,
        "foundation_sha256": foundation_sha256,
        "pending_selection_sha256": pending_selection_sha256,
    }
    baseline = _snapshot_with_hash(
        {
            **common,
            "checkpoint": "foundation",
            "snapshot_id": "foundation",
            "summary": "正式 baseline 已完成 challenge，并冻结 foundation；等待人工选择。",
        }
    )
    delta = _snapshot_with_hash(
        {
            **common,
            "checkpoint": "pending_selection",
            "snapshot_id": "pending-selection",
            "summary": "正式 challenge delta 已固化为 pending_selection；未执行 selection、HOLDOUT 或盲测。",
        }
    )
    return {"baseline": baseline, "delta": delta}


def _write_new_bytes(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path} already exists")
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_artifacts(
    run_root: Path,
    *,
    baseline_narrative: str,
    challenge_report: Mapping[str, object],
    snapshot_pair: Mapping[str, Mapping[str, object]],
    run_result: Mapping[str, object],
) -> dict[str, Path]:
    """Write all public artifacts once, refusing every existing target."""

    root = Path(run_root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("run artifact root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = root / "feishu_snapshots"
    if snapshot_dir.exists() and (snapshot_dir.is_symlink() or not snapshot_dir.is_dir()):
        raise ValueError("feishu_snapshots must be a directory")
    if not isinstance(snapshot_pair.get("baseline"), Mapping) or not isinstance(
        snapshot_pair.get("delta"), Mapping
    ):
        raise ValueError("snapshot_pair must contain baseline and delta mappings")

    paths = {
        "baseline_narrative": root / "baseline_narrative.md",
        "challenge_report": root / "challenge_report.json",
        "baseline_snapshot": snapshot_dir / "baseline_snapshot.json",
        "delta_snapshot": snapshot_dir / "delta_snapshot.json",
        "run_result": root / "run_result.json",
    }
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"artifact already exists: {path}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    _write_new_bytes(paths["baseline_narrative"], baseline_narrative.encode("utf-8"))
    _write_new_bytes(paths["challenge_report"], canonical_json_bytes(challenge_report))
    _write_new_bytes(
        paths["baseline_snapshot"],
        canonical_json_bytes(snapshot_pair["baseline"]),
    )
    _write_new_bytes(
        paths["delta_snapshot"],
        canonical_json_bytes(snapshot_pair["delta"]),
    )
    _write_new_bytes(paths["run_result"], canonical_json_bytes(run_result))
    return paths


def _call_count(store: object) -> int:
    calls_root = Path(_field(store, "root")) / "calls"
    if not calls_root.is_dir():
        return 0
    return sum(1 for entry in calls_root.iterdir() if entry.is_dir())


def _candidate_summaries(foundation: object) -> list[dict[str, object]]:
    stress_map = _stress_by_candidate(foundation)
    ranking = _field(foundation, "ranking")
    return [
        _candidate_summary(
            ranked,
            stress_map.get(_text(_field(_candidate_from_ranked(ranked), "candidate_id"))),
        )
        for ranked in (_field(ranking, "ranked_candidates", ()) or ())
    ]


def run_official_decision(
    *,
    prepared_root: Path,
    run_root: Path,
    run_id: str,
    execute_real_model: bool,
    preflight_loader: Callable[..., object] | None = None,
    store_factory: Callable[..., object] | None = None,
    llm_client_factory: Callable[[Settings], object] | None = None,
    audited_client_factory: Callable[..., object] | None = None,
    runner_factory: Callable[..., object] | None = None,
    coordinator_factory: Callable[..., object] | None = None,
    artifacts_writer: Callable[..., dict[str, Path]] = write_artifacts,
) -> dict[str, object]:
    """Execute exactly the official preparation path through the specificity gate."""

    if not execute_real_model:
        raise ValueError("--execute-real-model is required for the official run")
    _validate_run_id(run_id)
    loader = preflight_loader or load_official_preflight
    preflight = loader(
        prepared_root=Path(prepared_root),
        run_root=Path(run_root),
        run_id=run_id,
    )
    report = _report_of(preflight)
    _validate_formal_report(report)
    inputs = _inputs_of(preflight)
    settings = getattr(preflight, "settings", None)
    if not isinstance(settings, Settings):
        raise TypeError("official preflight settings are required")

    manifest = build_run_manifest(run_id=run_id, preflight=report)
    store_builder = store_factory or DecisionRunStore.create
    store = store_builder(Path(run_root), manifest)

    # This is the first point at which a real LLM client may be created.
    client_builder = llm_client_factory or LLMClient
    raw_client = client_builder(settings)
    audited_builder = audited_client_factory or AuditedLLMClient
    audited = audited_builder(
        raw_client,
        store,
        input_artifact_sha256=str(_field(inputs, "all_stable_ids_sha256")),
        prompt_name=str(_field(report["prompt"], "name")),
        prompt_version=str(_field(report["prompt"], "version")),
        prompt_sha256=str(_field(report["prompt"], "sha256")),
    )
    runner_builder = runner_factory or DecisionRunner
    runner = runner_builder(client=audited, brand_facts=[], run_id=run_id)
    coordinator_builder = coordinator_factory or DecisionRunCoordinator
    coordinator = coordinator_builder(
        run_store=store,
        runner=runner,
        audited_client=audited,
        inputs=inputs,
    )

    # Preparation deliberately stops at the Luna offline audit boundary.
    coordinator.prepare(inputs)

    foundation = store.load_stage("foundation", DecisionFoundationState)
    task_paths = sorted(Path(store.root).glob("stages/specificity_task_round_*.json"))
    if len(task_paths) != 1:
        raise ValueError("official prepare must freeze exactly one specificity task")
    task = OfflineSpecificityTask.model_validate_json(task_paths[0].read_bytes(), strict=True)
    foundation_sha256 = _sha(foundation)
    if _text(_field(foundation, "run_id")) != run_id:
        raise ValueError("foundation run_id does not match official run")
    if _field(foundation, "challenge_applied") is not True:
        raise ValueError("foundation must have challenge_applied=true")
    stage = _scalar(getattr(coordinator, "current_stage", None))
    if stage != DecisionRunStage.AWAITING_SPECIFICITY_AUDIT.value:
        raise ValueError("official run must stop at AWAITING_SPECIFICITY_AUDIT")

    challenge_report = build_challenge_report(
        run_id=run_id,
        inputs=inputs,
        foundation=foundation,
    )
    if challenge_report["challenge_evidence_count"] != EXPECTED_COUNTS["CHALLENGE"]:
        raise ValueError("official challenge evidence count must be exactly 60")
    decision_inputs = _decision_input_metadata(inputs)
    snapshot_pair = build_snapshot_pair(
        run_id=run_id,
        foundation_sha256=foundation_sha256,
        pending_selection_sha256=task.input_sha256,
        stage=DecisionRunStage.AWAITING_SPECIFICITY_AUDIT.value,
    )
    run_result: dict[str, object] = {
        "run_id": run_id,
        "stage": DecisionRunStage.AWAITING_SPECIFICITY_AUDIT.value,
        "foundation_sha256": foundation_sha256,
        "pending_luna_task_sha256": sha256_bytes(task_paths[0].read_bytes()),
        "task_path": str(task_paths[0]),
        "call_count": _call_count(store),
        "candidate_summaries": _candidate_summaries(foundation),
        "snapshot_dir": "feishu_snapshots",
        "decision_inputs": decision_inputs,
        "baseline_record_count": decision_inputs["baseline_count"],
        "analysis_baseline_count": decision_inputs["analysis_baseline_count"],
        "human_gold_count": decision_inputs["human_gold_count"],
        "all_decision_stable_ids_sha256": decision_inputs[
            "all_decision_stable_ids_sha256"
        ],
        "gold": decision_inputs["gold"],
        "preflight": {
            "counts": EXPECTED_COUNTS,
            "unique": 369,
            "network_calls": 0,
            "profile": report["profile"],
        },
        "selection_executed": False,
        "holdout_executed": False,
        "blind_test_executed": False,
        "artifact_files": [
            "baseline_narrative.md",
            "challenge_report.json",
            "feishu_snapshots/baseline_snapshot.json",
            "feishu_snapshots/delta_snapshot.json",
            "run_result.json",
        ],
    }
    artifacts_writer(
        Path(store.root),
        baseline_narrative=render_baseline_markdown(
            run_id=run_id,
            preflight=report,
            foundation=foundation,
            pending_selection=foundation,
        ),
        challenge_report=challenge_report,
        snapshot_pair=snapshot_pair,
        run_result=run_result,
    )
    return run_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the official DeepSeek decision preparation path."
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="prepare",
        choices=(
            "prepare",
            "export-specificity",
            "import-specificity",
            "revise-specificity",
            "finalize-specificity",
            "submit-selection",
            "continue",
        ),
    )
    parser.add_argument("--prepared-root", type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--result-path", type=Path)
    parser.add_argument("--selected-ids", type=str)
    parser.add_argument("--primary-id", type=str)
    parser.add_argument("--execute-real-model", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare" and not arguments.execute_real_model:
        print(
            "error: --execute-real-model is required; this command will invoke the formal real model",
            file=sys.stderr,
        )
        return 2
    try:
        if arguments.command == "prepare":
            if arguments.prepared_root is None:
                raise ValueError("--prepared-root is required for prepare")
            result = run_official_decision(
                prepared_root=arguments.prepared_root,
                run_root=arguments.run_root,
                run_id=arguments.run_id,
                execute_real_model=True,
            )
        else:
            result = _run_cli_gate_command(arguments)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def _run_cli_gate_command(arguments: argparse.Namespace) -> dict[str, object]:
    """Run one persisted coordinator gate and emit only immutable state metadata."""

    store = DecisionRunStore.load(arguments.run_root, arguments.run_id)
    input_sha256, prompt_name, prompt_version, prompt_sha256 = _cli_audit_binding(store)
    settings = Settings.from_env()
    raw_client = LLMClient(settings)
    audited = AuditedLLMClient(
        raw_client,
        store,
        input_artifact_sha256=input_sha256,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
    )
    runner = DecisionRunner(client=audited, brand_facts=[], run_id=arguments.run_id)
    coordinator = DecisionRunCoordinator(
        run_store=store,
        runner=runner,
        audited_client=audited,
    )
    command = arguments.command
    result: object
    if command == "export-specificity":
        result = coordinator.export_specificity_audit(arguments.run_id)
    elif command == "import-specificity":
        if arguments.result_path is None:
            raise ValueError("--result-path is required for import-specificity")
        result = coordinator.import_specificity_audit(arguments.run_id, arguments.result_path)
    elif command == "revise-specificity":
        result = coordinator.run_specificity_revision(arguments.run_id)
    elif command == "finalize-specificity":
        result = coordinator.finalize_specificity(arguments.run_id)
    elif command == "submit-selection":
        if arguments.selected_ids is None or arguments.primary_id is None:
            raise ValueError("--selected-ids and --primary-id are required for submit-selection")
        selected_ids = [item for item in arguments.selected_ids.split(",") if item]
        result = coordinator.submit_selection(arguments.run_id, selected_ids, arguments.primary_id)
    elif command == "continue":
        stage = coordinator.current_stage
        if stage is DecisionRunStage.AWAITING_SPECIFICITY_AUDIT:
            result = coordinator.export_specificity_audit(arguments.run_id)
        elif stage is DecisionRunStage.SPECIFICITY_REVISION:
            result = coordinator.run_specificity_revision(arguments.run_id)
        elif stage is DecisionRunStage.SPECIFICITY_FINAL_RANK:
            result = coordinator.finalize_specificity(arguments.run_id)
        else:
            raise ValueError(f"continue is not legal at stage {stage}")
    else:
        raise ValueError(f"unsupported command: {command}")
    path = result if isinstance(result, Path) else None
    payload = {
        "run_id": arguments.run_id,
        "stage": getattr(coordinator.current_stage, "value", coordinator.current_stage),
        "task_path": str(path) if path is not None else None,
        "result_path": str(arguments.result_path) if arguments.result_path is not None else None,
        "luna_model": "gpt-5.6-luna/max",
        "next_command": _next_cli_command(coordinator.current_stage, arguments),
    }
    return payload


def _cli_audit_binding(store: DecisionRunStore) -> tuple[str, str, str, str]:
    """Bind any future revision call to the frozen Luna task and revision prompt."""

    task_paths = sorted(store.root.glob("stages/specificity_task_round_*.json"))
    if task_paths:
        task_data = json.loads(task_paths[-1].read_text(encoding="utf-8"))
        task = OfflineSpecificityTask.model_validate(task_data, strict=True)
        input_sha256 = task.input_sha256
    else:
        session_path = store.root / "stages" / "specificity_session.json"
        if not session_path.is_file() or session_path.is_symlink():
            raise ValueError("no frozen specificity task or session is available for ledger binding")
        input_sha256 = sha256_bytes(session_path.read_bytes())
    prompt_path = PROJECT_ROOT / "prompts" / "brand_specificity_revision.md"
    if prompt_path.is_symlink() or not prompt_path.is_file():
        raise ValueError("specificity revision prompt is missing")
    prompt_bytes = prompt_path.read_bytes()
    prompt_bytes.decode("utf-8")
    return (
        input_sha256,
        "brand_specificity_revision",
        "brand-specificity-revision-v1",
        sha256_bytes(prompt_bytes),
    )


def _next_cli_command(stage: object, arguments: argparse.Namespace) -> str:
    value = getattr(stage, "value", stage)
    commands = {
        DecisionRunStage.AWAITING_SPECIFICITY_AUDIT.value: "export-specificity/import-specificity",
        DecisionRunStage.AWAITING_SELECTION.value: "submit-selection",
        DecisionRunStage.SPECIFICITY_REVISION.value: "revise-specificity",
        DecisionRunStage.SPECIFICITY_FINAL_RANK.value: "finalize-specificity",
        DecisionRunStage.HOLDOUT.value: "continue",
    }
    return commands.get(value, "none")


if __name__ == "__main__":
    raise SystemExit(main())
