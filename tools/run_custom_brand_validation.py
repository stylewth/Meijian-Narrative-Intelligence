"""第二品牌（果立方）自由链路验证驱动。

复用预处理工作区与 DecisionRunner 的生产函数，headless 依次执行：
  1. gold 双审（两遍独立在线标注 + 分歧仲裁 + 一致率报告 + 冻结）
  2. 语料包发布（baseline / challenge / holdout / 增量×2）
  3. 决策链（语料分析→候选→评分→压测→挑战修订→选线→holdout→冻结）
  4. 增量演化（按顺序释放 2 个增量批）
  5. 汇总总结报告

全部产物落盘到 果立方_第二品牌验证/。阶段幂等：已完成（存在哨兵文件）自动跳过；
决策链按 foundation 状态断点续跑（state.json + DecisionRunner.restore）。

用法：
    E:/Anaconda/python.exe -X utf8 tools/run_custom_brand_validation.py --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.brand_profile import BrandProfile, set_active_brand  # noqa: E402
from src.config import Settings  # noqa: E402
from src.data_validator import validate_dataframe  # noqa: E402
from src.dataset_split import DATASET_VERSION  # noqa: E402
from src.llm_client import LLMClient  # noqa: E402
from src.prompt_loader import load_prompt_metadata  # noqa: E402
from src.schemas import (  # noqa: E402
    BusinessDecisionStatus,
    DecisionEvolutionCheckpoint,
    DecisionFoundationState,
    EvidenceAtom,
    StrictBaseModel,
)
from src.services.pressure_audit import (  # noqa: E402
    ApiSpecificityAuditor,
    run_pressure_loop,
)
from src.services.decision_delta import derive_business_status  # noqa: E402
from src.services.decision_runner import DecisionRunner  # noqa: E402
from src.services.prepared_corpus import (  # noqa: E402
    list_prepared_corpora,
    publish_prepared_corpus,
)
from src.ui.custom_decision_workspace import CustomRunInputs  # noqa: E402
from src.ui.preprocessing_workspace import (  # noqa: E402
    _build_prepared_package_online,
    _normalize_imported_records,
    annotate_records_online,
    _store_new_dataset,
    run_online_annotation,
)

WORKSPACE_ROOT = REPO_ROOT.parent
EVIDENCE_ROOT = WORKSPACE_ROOT / "果立方_第二品牌验证"
PREP_CSV_DIR = EVIDENCE_ROOT / "00_准备" / "分包CSV"
BRAND_PROFILE_PATH = EVIDENCE_ROOT / "00_准备" / "brand_profile.json"
# 发布根放在仓库外的证据目录：仓库提交卫生约束禁止 data/ 下出现
# records.json / evidence_atoms.json 原始工件（tests/test_frozen_release_data.py）。
PREPARED_ROOT = EVIDENCE_ROOT / "prepared_corpora"
DOTENV_PATH = REPO_ROOT / ".env"

GOLD_DIR = EVIDENCE_ROOT / "01_gold双审"
PREPROCESS_DIR = EVIDENCE_ROOT / "02_预处理"
DECISION_DIR = EVIDENCE_ROOT / "03_决策链路"
PRESSURE_DIR = EVIDENCE_ROOT / "06_压力测试"
PRESSURE_ATTEMPT_DIR = PRESSURE_DIR / "attempt-gllf-pressure-001"
PRESSURE_RUN_ID = "gllf-pressure-001"
PREPROCESS_ROLES = ("baseline", "challenge", "holdout", "incremental_1", "incremental_2")
RELEASE_ROLES = ("incremental_1", "incremental_2")

_LOG_PATH = EVIDENCE_ROOT / "运行日志.txt"


def log(message: str) -> None:
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with _LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_role_csv(role: str) -> pd.DataFrame:
    return pd.read_csv(PREP_CSV_DIR / f"{role}.csv", dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- gold 双审

_ROUTE_VALUES = {"BRAND", "PRODUCT", "SERVICE", "SCENE", "COMPETITOR", "OTHER"}
_SCOPE_VALUES = {"ACTUAL_USE", "PURCHASE_ONLY", "NON_USE", "UNKNOWN"}
_GRADE_VALUES = {"A", "B", "C"}
_LABEL_KEYS = ("route", "experience_scope", "evidence_grade")


class _ArbitrationItem(StrictBaseModel):
    comment_id: str = Field(min_length=1)
    route: str = Field(pattern=r"^(BRAND|PRODUCT|SERVICE|SCENE|COMPETITOR|OTHER)$")
    experience_scope: str = Field(
        pattern=r"^(ACTUAL_USE|PURCHASE_ONLY|NON_USE|UNKNOWN)$"
    )
    evidence_grade: str = Field(pattern=r"^(A|B|C)$")
    ai_confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1)
    arbitration_reason: str = Field(min_length=1)


class _ArbitrationResponse(StrictBaseModel):
    judgments: list[_ArbitrationItem]


def _atoms_payload(atoms: list[EvidenceAtom]) -> list[dict]:
    return [atom.model_dump(mode="json") for atom in atoms]


def _compare_passes(
    first: list[EvidenceAtom], second: list[EvidenceAtom]
) -> tuple[dict, list[str]]:
    if {atom.comment_id for atom in first} != {atom.comment_id for atom in second}:
        raise ValueError("两次标注覆盖的评论集合不一致")
    total = len(first)
    matched = {key: 0 for key in _LABEL_KEYS}
    full_match = 0
    mismatch_ids: list[str] = []
    first_by_id = {atom.comment_id: atom for atom in first}
    second_by_id = {atom.comment_id: atom for atom in second}
    for comment_id in sorted(first_by_id):
        left, right = first_by_id[comment_id], second_by_id[comment_id]
        row_match = True
        for key in _LABEL_KEYS:
            if getattr(left, key).value == getattr(right, key).value:
                matched[key] += 1
            else:
                row_match = False
        if row_match:
            full_match += 1
        else:
            mismatch_ids.append(comment_id)
    agreement = {
        "total": total,
        "full_row_match": full_match,
        "full_row_match_rate": round(full_match / total, 4) if total else 0.0,
        "per_label_match": {
            key: {
                "count": matched[key],
                "rate": round(matched[key] / total, 4) if total else 0.0,
            }
            for key in _LABEL_KEYS
        },
    }
    return agreement, mismatch_ids


def stage_gold(client: LLMClient) -> None:
    routing_meta = load_prompt_metadata("evidence_routing")
    frozen_path = GOLD_DIR / "frozen_gold_labels.json"
    if frozen_path.exists():
        log("gold 双审已完成，跳过")
        return
    log("gold 双审开始：加载 gold 分包")
    dataset = validate_dataframe(load_role_csv("gold"))
    records = _normalize_imported_records(list(dataset.comments))
    log(f"gold 记录 {len(records)} 条，开始两遍独立标注")

    attempts: dict[str, list[dict]] = {}
    pass_atoms: list[list[EvidenceAtom]] = []
    for index in (1, 2):
        atoms, batch_ids = annotate_records_online(
            client=client, analysis_records=records
        )
        attempts[f"attempt-{index:04d}"] = _atoms_payload(atoms)
        pass_atoms.append(atoms)
        dump_json(GOLD_DIR / f"attempt-{index:04d}.json", attempts[f"attempt-{index:04d}"])
        log(f"第 {index} 遍标注完成：{len(atoms)} 条（批次 {batch_ids}）")

    agreement, mismatch_ids = _compare_passes(pass_atoms[0], pass_atoms[1])
    log(f"一致率：整行 {agreement['full_row_match']}/{agreement['total']}，分歧 {len(mismatch_ids)} 条")

    records_by_id = {record.comment_id: record for record in records}
    final_by_id: dict[str, EvidenceAtom] = {atom.comment_id: atom for atom in pass_atoms[0]}
    arbitration_log: list[dict] = []
    if mismatch_ids:
        pending = [
            {
                "comment_id": record.comment_id,
                "raw_content": record.raw_content,
                "sample_type": record.sample_type.value,
                "raw_sample_type": record.raw_sample_type,
                "source_platform": record.source_platform,
            }
            for record in records
            if record.comment_id in mismatch_ids
        ]
        first_by_id = {atom.comment_id: atom for atom in pass_atoms[0]}
        second_by_id = {atom.comment_id: atom for atom in pass_atoms[1]}
        opinions = [
            {
                "comment_id": comment_id,
                "attempt-0001": {
                    key: getattr(first_by_id[comment_id], key).value for key in _LABEL_KEYS
                }
                | {"explanation": first_by_id[comment_id].explanation},
                "attempt-0002": {
                    key: getattr(second_by_id[comment_id], key).value for key in _LABEL_KEYS
                }
                | {"explanation": second_by_id[comment_id].explanation},
            }
            for comment_id in mismatch_ids
        ]
        system_prompt = (
            "你是证据路由仲裁员。两条独立标注对同一批评论给出了不一致的"
            " route / experience_scope / evidence_grade 三标签。你必须逐条复核原文，"
            "依据路由规则独立裁决，不得简单采纳任何一方。原始评论是唯一事实证据；"
            "只分析达到法定饮酒年龄的成年人的饮酒相关评论；只返回符合 Schema 的 JSON。"
        )
        user_prompt = json.dumps(
            {"评论": pending, "两次标注": opinions}, ensure_ascii=False
        )
        response = client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=_ArbitrationResponse,
        )
        judged_ids = [item.comment_id for item in response.judgments]
        if set(judged_ids) != set(mismatch_ids):
            raise ValueError("仲裁未覆盖全部分歧评论")
        for item in response.judgments:
            if item.route not in _ROUTE_VALUES or item.experience_scope not in _SCOPE_VALUES or item.evidence_grade not in _GRADE_VALUES:
                raise ValueError(f"仲裁返回非法标签：{item.comment_id}")
            record = records_by_id[item.comment_id]
            if record.source is None:
                raise ValueError(f"评论 {item.comment_id} 缺少 SourceReference")
            final_by_id[item.comment_id] = EvidenceAtom(
                evidence_id=record.comment_id,
                comment_id=record.comment_id,
                route=item.route,
                experience_scope=item.experience_scope,
                evidence_grade=item.evidence_grade,
                ai_confidence=item.ai_confidence,
                explanation=item.explanation,
                source=record.source,
                source_platform=record.source_platform,
                duplicate_group=record.duplicate_group,
                actual_use=record.actual_use,
            )
            arbitration_log.append(item.model_dump(mode="json"))
        dump_json(GOLD_DIR / "arbitration.json", arbitration_log)
        log(f"仲裁完成：{len(arbitration_log)} 条分歧已裁决")

    final_labels = [_atoms_payload([final_by_id[cid]])[0] for cid in sorted(final_by_id)]
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "project": "PRJ-GLLF-001 第二品牌（果立方）gold 双审",
        "dataset_version": DATASET_VERSION,
        "record_count": len(records),
        "label_source": "AI_DOUBLE_REVIEW",
        "model": client._model,
        "prompt": {
            "name": routing_meta.name,
            "version": routing_meta.version,
            "sha256": routing_meta.sha256,
        },
        "label_rule": "两遍独立在线标注一致即定稿；分歧条目由仲裁遍（仅见分歧条目与双方理由）裁决。",
        "agreement": agreement,
        "final_labels": final_labels,
    }
    dump_json(frozen_path, frozen)
    frozen["sha256"] = hashlib.sha256(
        frozen_path.read_bytes()
    ).hexdigest()
    dump_json(frozen_path, frozen)

    per_label = agreement["per_label_match"]
    lines = [
        "# Gold 双审一致率报告",
        "",
        f"- 记录数：{agreement['total']}",
        f"- 整行一致：{agreement['full_row_match']} / {agreement['total']}"
        f"（{agreement['full_row_match_rate']:.1%}）",
        f"- route 一致：{per_label['route']['count']}（{per_label['route']['rate']:.1%}）",
        f"- experience_scope 一致：{per_label['experience_scope']['count']}"
        f"（{per_label['experience_scope']['rate']:.1%}）",
        f"- evidence_grade 一致：{per_label['evidence_grade']['count']}"
        f"（{per_label['evidence_grade']['rate']:.1%}）",
        f"- 分歧仲裁：{len(arbitration_log)} 条（见 arbitration.json）",
        "",
        "规则：两遍独立标注（同一模型、两次独立调用、互不可见）一致即定稿；"
        "分歧条目由第三遍仲裁，只提供原文与双方理由。",
    ]
    (GOLD_DIR / "一致率报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("gold 双审完成并已冻结")


# ---------------------------------------------------------------- 语料包发布

def _load_publish_record() -> dict:
    path = PREPROCESS_DIR / "发布记录.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def stage_preprocess(client: LLMClient) -> dict:
    record = _load_publish_record()
    for role in PREPROCESS_ROLES:
        if role in record:
            log(f"语料包 {role} 已发布（{record[role]['package_id']}），跳过")
            continue
        log(f"预处理开始：{role}")
        dataset = validate_dataframe(load_role_csv(role))
        content = (PREP_CSV_DIR / f"{role}.csv").read_bytes()
        state: dict = {}
        _store_new_dataset(
            state,
            dataset=dataset,
            dataset_sha256=hashlib.sha256(content).hexdigest(),
            dataset_version=DATASET_VERSION,
            source_name=f"{role}.csv",
        )
        atom_count = run_online_annotation(state, dotenv_path=DOTENV_PATH)
        package = _build_prepared_package_online(
            records=state["preprocessing_records"],
            split_manifest=state["preprocessing_split_manifest"],
            annotation_manifest=state["preprocessing_annotation_manifest"],
            evidence_atoms=state["preprocessing_online_atoms"],
        )
        published = publish_prepared_corpus(PREPARED_ROOT, package)
        record[role] = {
            "package_id": published.manifest.package_id,
            "records": len(published.package.records),
            "atoms": len(published.package.evidence_atoms),
        }
        dump_json(PREPROCESS_DIR / "发布记录.json", record)
        dump_json(
            PREPROCESS_DIR / f"{role}-atoms.json",
            _atoms_payload(list(published.package.evidence_atoms)),
        )
        log(f"语料包 {role} 已发布：{record[role]['package_id']}（{atom_count} 条标注）")
    return record


# ---------------------------------------------------------------- 决策链

def _find_published_package(package_id: str):
    for published in list_prepared_corpora(PREPARED_ROOT):
        if published.manifest.package_id == package_id:
            return published
    raise ValueError(f"未找到已发布语料包: {package_id}")


def _load_package_atoms(package_id: str) -> list[EvidenceAtom]:
    return list(_find_published_package(package_id).package.evidence_atoms)


def _load_package_inputs(package_id: str) -> tuple[list, list[EvidenceAtom]]:
    package = _find_published_package(package_id).package
    return list(package.records), list(package.evidence_atoms)


def _ranking_rows(foundation: DecisionFoundationState) -> list[dict]:
    stress_by_id = {item.candidate_id: item for item in foundation.stress_results}
    rows = []
    for ranked in foundation.ranking.ranked_candidates:
        stress = stress_by_id.get(ranked.candidate.candidate_id)
        scores = ranked.evaluation.scores
        rows.append(
            {
                "rank": ranked.rank,
                "candidate_id": ranked.candidate.candidate_id,
                "title": ranked.candidate.title,
                "weighted_score": ranked.weighted_score,
                "recommended": ranked.is_recommended,
                "business_status": (
                    derive_business_status(stress).value if stress else None
                ),
                "scores": {
                    "evidence_strength": scores.evidence_strength.score,
                    "emotional_tension": scores.emotional_tension.score,
                    "brand_fit_and_exclusivity": scores.brand_fit_and_exclusivity.score,
                    "competitor_difference": scores.competitor_difference.score,
                    "scene_conversion": scores.scene_conversion.score,
                },
            }
        )
    return rows


def _save_decision_state(
    foundation: DecisionFoundationState,
    checkpoints: list[DecisionEvolutionCheckpoint],
    step_name: str,
    step_counter: list[int],
) -> None:
    step_counter[0] += 1
    name = f"step-{step_counter[0]:02d}-{step_name}"
    dump_json(
        DECISION_DIR / "steps" / f"{name}.json",
        foundation.model_dump(mode="json"),
    )
    dump_json(
        DECISION_DIR / "state.json",
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "last_step": name,
            "foundation": foundation.model_dump(mode="json"),
            "checkpoints": [item.model_dump(mode="json") for item in checkpoints],
        },
    )
    dump_json(DECISION_DIR / "steps" / f"{name}-leaderboard.json", _ranking_rows(foundation))
    log(f"决策链进度：{name} 已落盘")


def stage_decision(client: LLMClient, publish_record: dict) -> DecisionFoundationState:
    DECISION_DIR.mkdir(parents=True, exist_ok=True)
    state_path = DECISION_DIR / "state.json"
    step_counter = [0]
    checkpoints: list[DecisionEvolutionCheckpoint] = []
    if state_path.exists():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        foundation = DecisionFoundationState.model_validate(payload["foundation"])
        checkpoints = [
            DecisionEvolutionCheckpoint.model_validate(item)
            for item in payload["checkpoints"]
        ]
        step_counter[0] = len(
            [
                path
                for path in (DECISION_DIR / "steps").glob("step-*.json")
                if not path.name.endswith("-leaderboard.json")
            ]
        )
        runner = DecisionRunner.restore(client, brand_facts=[], state=foundation, checkpoints=checkpoints)
        log(f"决策链断点恢复：{payload['last_step']}")
    else:
        baseline_id = publish_record["baseline"]["package_id"]
        records, atoms = _load_package_inputs(baseline_id)
        runner = DecisionRunner(client=client, brand_facts=[])
        foundation = runner.build_baseline(CustomRunInputs(records, atoms))
        log(f"基线完成：{len(records)} 条评论 → {len(foundation.candidates)} 个候选")
        _save_decision_state(foundation, checkpoints, "baseline", step_counter)

    if not foundation.challenge_applied:
        challenge_atoms = _load_package_atoms(publish_record["challenge"]["package_id"])
        foundation = runner.apply_challenge(foundation, challenge_atoms)
        log(f"挑战批完成：{len(challenge_atoms)} 条证据进入评估与修订")
        _save_decision_state(foundation, checkpoints, "challenge", step_counter)

    if not foundation.selected_ids:
        ranked = foundation.ranking.ranked_candidates
        selected = [item.candidate.candidate_id for item in ranked[:3]]
        foundation = runner.select_narratives(
            foundation, selected, primary_candidate_id=selected[0]
        )
        log(f"选线完成（按排名前三）：{selected}")
        _save_decision_state(foundation, checkpoints, "select", step_counter)

    if not foundation.holdout_validated:
        holdout_atoms = _load_package_atoms(publish_record["holdout"]["package_id"])
        foundation = runner.validate_holdout(foundation, holdout_atoms)
        log(f"holdout 验证完成：{len(holdout_atoms)} 条留出证据")
        _save_decision_state(foundation, checkpoints, "holdout", step_counter)

    if not foundation.frozen:
        stress_by_id = {item.candidate_id: item for item in foundation.stress_results}
        primary = foundation.primary_candidate_id
        if derive_business_status(stress_by_id[primary]) is BusinessDecisionStatus.BLOCKED:
            ranked_order = [
                item.candidate.candidate_id
                for item in foundation.ranking.ranked_candidates
                if item.candidate.candidate_id in foundation.selected_ids
            ]
            alternative = next(
                (
                    candidate_id
                    for candidate_id in ranked_order
                    if derive_business_status(stress_by_id[candidate_id])
                    is not BusinessDecisionStatus.BLOCKED
                ),
                None,
            )
            if alternative is None:
                raise ValueError("全部入围候选 BLOCKED，无法冻结（按纪律中止，不降级）")
            foundation = runner.reselect_primary(
                foundation, primary_candidate_id=alternative
            )
            log(f"主候选 BLOCKED，重选主候选：{alternative}")
            _save_decision_state(foundation, checkpoints, "reselect-primary", step_counter)
        foundation = runner.freeze_original_snapshot(foundation)
        log(f"原始快照已冻结：主候选 {foundation.primary_candidate_id}")
        _save_decision_state(foundation, checkpoints, "freeze", step_counter)

    for index, role in enumerate(RELEASE_ROLES, start=1):
        if any(checkpoint.release_index == index for checkpoint in checkpoints):
            continue
        batch_atoms = _load_package_atoms(publish_record[role]["package_id"])
        checkpoint = runner.release_batch(foundation, batch_index=index, atoms=batch_atoms)
        checkpoints.append(checkpoint)
        foundation = runner._release_state
        log(f"增量批 {index} 释放完成：新增 {len(batch_atoms)} 条证据")
        _save_decision_state(foundation, checkpoints, f"release-{index}", step_counter)

    dump_json(
        DECISION_DIR / "checkpoints.json",
        [item.model_dump(mode="json") for item in checkpoints],
    )
    return foundation


# ---------------------------------------------------------------- 压力测试

def _selected_candidates_in_rank_order(
    foundation: DecisionFoundationState,
) -> list:
    ranked = [
        item.candidate
        for item in sorted(
            foundation.ranking.ranked_candidates, key=lambda item: item.rank
        )
        if item.candidate.candidate_id in foundation.selected_ids
    ]
    return ranked


def _all_records_by_id() -> dict[str, Any]:
    records: dict[str, Any] = {}
    for published in list_prepared_corpora(PREPARED_ROOT):
        for record in published.package.records:
            records[record.comment_id] = record
    return records


def _finding_rows(round_item: dict) -> list[dict]:
    rows = []
    for audit_item in round_item["audit"]["audits"]:
        for finding in audit_item["findings"]:
            disposition = next(
                (
                    response
                    for revision_item in (round_item.get("revision") or {}).get(
                        "revisions", []
                    )
                    for response in revision_item["responses"]
                    if response["finding_id"] == finding["finding_id"]
                ),
                None,
            )
            rows.append(
                {
                    "finding_id": finding["finding_id"],
                    "audit_type": finding["audit_type"],
                    "severity": finding["severity"],
                    "claim": finding["claim"],
                    "evidence_ids": finding["evidence_ids"],
                    "triggers_next_round": finding["triggers_next_round"],
                    "disposition": (
                        f"{disposition['disposition']}（{disposition['reason']}）"
                        if disposition
                        else "—"
                    ),
                }
            )
    return rows


def _write_pressure_report(
    foundation: DecisionFoundationState, result
) -> Path:
    ranked_ids = [
        item.candidate.candidate_id
        for item in sorted(
            foundation.ranking.ranked_candidates, key=lambda item: item.rank
        )
        if item.candidate.candidate_id in foundation.selected_ids
    ]
    meta = json.loads(
        (PRESSURE_ATTEMPT_DIR / "attempt_meta.json").read_text(encoding="utf-8")
    )
    sections = [
        "# 果立方 · 品牌专属度压力测试报告",
        "",
        f"- 运行 ID：{PRESSURE_RUN_ID}",
        f"- 审计者（攻击方）：{json.dumps(meta['auditor'], ensure_ascii=False)}",
        f"- 轮次上限：{meta['limits']['max_audit_rounds']} 轮；修订上限："
        f"{meta['limits']['max_revisions']} 次（与官方五候选验证一致）",
        "- 合同：SpecificityAuditBatch（四类审计各恰好一条 finding，NOTE 不触发"
        "下一轮）+ SpecificityRevisionBatchV2（最小可审计修订，评论证据绑定）",
        "",
    ]
    final_candidates = {}
    for candidate_id in ranked_ids:
        state = result.states[candidate_id]
        sections.append(f"## {candidate_id}（终态：{state['status']}）")
        sections.append("")
        for round_item in state["rounds"]:
            sections.append(
                f"### 第 {round_item['round_index']} 轮（候选 v{round_item['candidate_version']}）"
            )
            sections.append("")
            sections.append(
                "| finding | 类别 | 严重度 | 主张 | 引用证据 | 触发下一轮 | 处置 |"
            )
            sections.append("|---|---|---|---|---|---|---|")
            for row in _finding_rows(round_item):
                sections.append(
                    f"| {row['finding_id']} | {row['audit_type']} | {row['severity']} "
                    f"| {row['claim']} | {', '.join(row['evidence_ids']) or '—'} "
                    f"| {row['triggers_next_round']} | {row['disposition']} |"
                )
            sections.append("")
            revision = round_item.get("revision")
            if revision:
                sections.append("**修订后文本变化（程序比对生成）：**")
                sections.append("")
                for diff in round_item.get("revision_diffs", []):
                    sections.append(f"- `{diff['field']}`：")
                    sections.append(f"  - 修前：{diff['before']}")
                    sections.append(f"  - 修后：{diff['after']}")
                sections.append("")
        final = state["current_candidate"]
        final_candidates[candidate_id] = final
        sections.append("**压测后终稿要点：**")
        sections.append("")
        sections.append(f"- 标题：{final['title']}")
        sections.append(f"- 主张：{final['draft_proposition']}")
        sections.append(f"- 品牌依据：{final['why_brand']}")
        sections.append(f"- 场景：{'；'.join(final['main_scenes'])}")
        sections.append("")
    report_path = PRESSURE_DIR / "压力测试报告.md"
    report_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    dump_json(
        PRESSURE_DIR / "final_candidates.json",
        {candidate_id: final_candidates[candidate_id] for candidate_id in ranked_ids},
    )
    return report_path


def stage_pressure(client: Any, settings: Settings) -> None:
    state_payload = json.loads(
        (DECISION_DIR / "state.json").read_text(encoding="utf-8")
    )
    foundation = DecisionFoundationState.model_validate(state_payload["foundation"])
    candidates = _selected_candidates_in_rank_order(foundation)
    if len(candidates) != 3:
        raise ValueError(f"入选候选应为 3 条，实际 {len(candidates)}")
    records_by_id = _all_records_by_id()
    missing = {
        atom.comment_id
        for atom in foundation.visible_atoms
        if atom.comment_id not in records_by_id
    }
    if missing:
        raise ValueError(f"可见证据缺少原文记录: {sorted(missing)[:5]}")
    auditor_settings = settings.auditor_client_settings()
    if auditor_settings is not None:
        auditor: Any = ApiSpecificityAuditor(
            LLMClient(auditor_settings), run_id=PRESSURE_RUN_ID
        )
        auditor_meta = {"provider": "api", "model": auditor_settings.llm_model}
        log(f"审计者 API 模式：{auditor_settings.llm_model}")
    else:
        auditor = None
        auditor_meta = {
            "provider": "zcode-subagent",
            "model": "glm-5.3-flash（ZCode 子代理离线代跑）",
        }
        log("审计者未配置 AUDITOR_* API，进入离线代跑模式（子代理回填）")
    result = run_pressure_loop(
        run_id=PRESSURE_RUN_ID,
        attempt_dir=PRESSURE_ATTEMPT_DIR,
        candidates=candidates,
        evidence_atoms=list(foundation.visible_atoms),
        records_by_id=records_by_id,
        deepseek_client=client,
        auditor=auditor,
        auditor_meta=auditor_meta,
    )
    if result.all_terminal:
        report_path = _write_pressure_report(foundation, result)
        for candidate_id, state in result.states.items():
            log(f"压测终态 {candidate_id}: {state['status']}（{len(state['rounds'])} 轮）")
        log(f"压力测试完成，报告已写入：{report_path}")
    else:
        awaiting = {
            "status": "AWAITING_AUDITOR",
            "pending_tasks": [str(path) for path in result.pending_tasks],
            "instruction": (
                "由离线代跑方读取每个 auditor_task.json（内含候选、历史、全部评论证据"
                "与输出合同），按 prompts/brand_specificity_audit_v3.md 产出严格 JSON "
                "写到同目录 auditor_result.json 后重跑 --stage pressure。"
            ),
        }
        dump_json(PRESSURE_DIR / "AWAITING_AUDITOR.json", awaiting)
        log(f"等待离线审计结果：{len(result.pending_tasks)} 个任务待回填")


# ---------------------------------------------------------------- 总结报告

def _fmt_score(value: float) -> str:
    return f"{value:.2f}"


def stage_report(foundation: DecisionFoundationState) -> None:
    profile = json.loads(BRAND_PROFILE_PATH.read_text(encoding="utf-8"))
    publish_record = _load_publish_record()
    gold = json.loads((GOLD_DIR / "frozen_gold_labels.json").read_text(encoding="utf-8"))
    arbitration_path = GOLD_DIR / "arbitration.json"
    arbitration_count = (
        len(json.loads(arbitration_path.read_text(encoding="utf-8")))
        if arbitration_path.exists()
        else 0
    )
    checkpoints = [
        DecisionEvolutionCheckpoint.model_validate(item)
        for item in json.loads(
            (DECISION_DIR / "checkpoints.json").read_text(encoding="utf-8")
        )
    ]
    baseline_board = json.loads(
        (DECISION_DIR / "steps" / "step-01-baseline-leaderboard.json").read_text(encoding="utf-8")
    )
    freeze_step = sorted((DECISION_DIR / "steps").glob("step-*-freeze-leaderboard.json"))[-1]
    freeze_board = json.loads(freeze_step.read_text(encoding="utf-8"))
    final_board = _ranking_rows(foundation)

    def board_table(board: list[dict]) -> str:
        header = (
            "| 排名 | 候选 | 标题 | 证据 | 情绪 | 品牌适配 | 竞品差异 | 场景 | 加权分 | 状态 |"
        )
        sep = "|---|---|---|---|---|---|---|---|---|---|"
        rows = [
            f"| {row['rank']} | {row['candidate_id']} | {row['title']} "
            f"| {_fmt_score(row['scores']['evidence_strength'])} "
            f"| {_fmt_score(row['scores']['emotional_tension'])} "
            f"| {_fmt_score(row['scores']['brand_fit_and_exclusivity'])} "
            f"| {_fmt_score(row['scores']['competitor_difference'])} "
            f"| {_fmt_score(row['scores']['scene_conversion'])} "
            f"| **{_fmt_score(row['weighted_score'])}** | {row['business_status'] or '—'} |"
            for row in board
        ]
        return "\n".join([header, sep, *rows])

    agreement = gold["agreement"]
    per_label = agreement["per_label_match"]
    selected_ids = foundation.selected_ids
    candidate_by_id = {candidate.candidate_id: candidate for candidate in foundation.candidates}

    # 压力测试终稿（若压测已完成，§8 展示压测后版本并附修订摘要）
    pressure_finals: dict[str, dict] = {}
    pressure_meta: dict[str, dict] = {}
    finals_path = PRESSURE_DIR / "final_candidates.json"
    if finals_path.exists():
        pressure_finals = json.loads(finals_path.read_text(encoding="utf-8"))
        for candidate_id in pressure_finals:
            state_path = PRESSURE_ATTEMPT_DIR / "candidates" / candidate_id / "state.json"
            if state_path.exists():
                pressure_meta[candidate_id] = json.loads(
                    state_path.read_text(encoding="utf-8")
                )

    def narrative_section(candidate: dict, candidate_id: str, rank: int, weighted: float) -> str:
        return "\n".join(
            [
                f"### {rank}. {candidate['title']}（{candidate_id}，加权分 {weighted:.2f}）",
                "",
                f"- **核心主张**：{candidate['draft_proposition']}",
                f"- **用户矛盾**：{candidate['user_conflict']}",
                f"- **品牌机会**：{candidate['brand_opportunity']}",
                f"- **品牌依据（why_brand）**：{candidate['why_brand']}",
                f"- **竞品差异**：{candidate['competitor_difference']}",
                f"- **品牌角色**：{candidate['brand_role']}",
                f"- **核心场景**：{'；'.join(candidate['main_scenes'])}",
                f"- **内容主题**：{candidate['content_theme']}",
                f"- **目标人群**：{candidate['target_audience']}",
                f"- **证据**：支持 {len(candidate['supporting_evidence'])} 条"
                f"（{', '.join(quote['comment_id'] for quote in candidate['supporting_evidence'])}）；"
                f"反面 {len(candidate['counter_evidence'])} 条。{candidate['counter_evidence_note']}",
                f"- **风险边界**：{'；'.join(candidate['risks'])}",
            ]
        )

    narrative_sections = []
    for candidate_id in selected_ids:
        ranked = next(
            item
            for item in foundation.ranking.ranked_candidates
            if item.candidate.candidate_id == candidate_id
        )
        if candidate_id in pressure_finals:
            pressure_state = pressure_meta.get(candidate_id, {})
            revision_rounds = [
                item
                for item in pressure_state.get("rounds", [])
                if item.get("revision")
            ]
            diff_summary = "；".join(
                f"第 {item['round_index']} 轮修订 {'、'.join(diff['field'] for diff in item['revision_diffs'])}"
                for item in revision_rounds
            ) or "无修订"
            sections_head = (
                f"（压测后终稿 v{pressure_state.get('current_version', 1)}，"
                f"压测终态 {pressure_state.get('status')}，{len(pressure_state.get('rounds', []))} 轮：{diff_summary}）"
            )
            narrative_sections.append(
                narrative_section(
                    pressure_finals[candidate_id],
                    candidate_id,
                    ranked.rank,
                    ranked.weighted_score,
                ).replace(
                    f"（{candidate_id}，加权分 {ranked.weighted_score:.2f}）",
                    f"（{candidate_id}，加权分 {ranked.weighted_score:.2f}）{sections_head}",
                )
            )
        else:
            narrative_sections.append(
                narrative_section(
                    candidate_by_id[candidate_id].model_dump(mode="json"),
                    candidate_id,
                    ranked.rank,
                    ranked.weighted_score,
                ).replace(
                    f"（{candidate_id}，加权分 {ranked.weighted_score:.2f}）",
                    f"（{candidate_id}，加权分 {ranked.weighted_score:.2f}，压测前版本）",
                )
            )

    checkpoint_lines = []
    for checkpoint in checkpoints:
        patches_text = (
            "；".join(
                f"{patch.candidate_id}·{patch.field_name}（{patch.reason}）"
                for patch in checkpoint.patches
            )
            or "无修订"
        )
        switch = (
            f"{checkpoint.switch_suggestion.from_candidate_id} → "
            f"{checkpoint.switch_suggestion.to_candidate_id}（{checkpoint.switch_suggestion.reason}）"
            if checkpoint.switch_suggestion is not None
            else "无换线建议"
        )
        top = min(checkpoint.candidates, key=lambda s: s.ranked_narrative.rank)
        checkpoint_lines.append(
            f"- **{checkpoint.checkpoint_id}**（release-{checkpoint.release_index}，"
            f"可见证据 {len(checkpoint.visible_evidence_ids)} 条）：榜首 "
            f"{top.ranked_narrative.candidate.title}"
            f"（{top.ranked_narrative.weighted_score:.2f}）；修订：{patches_text}；换线建议：{switch}"
        )

    package_rows = "\n".join(
        f"| {role} | {payload['package_id']} | {payload['records']} | {payload['atoms']} |"
        for role, payload in publish_record.items()
    )

    report = "\n".join(
        [
            "# 果立方第二品牌验证 · 总结报告",
            "",
            f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 目标品牌：{profile['brand_name']}（{profile['industry']}）",
            f"- 研究问题：{profile['research_question']}",
            f"- 主要竞品：{'、'.join(profile['competitors'])}",
            "- 链路：main 分支「自定义使用」自由链路（REAL_MODE，DeepSeek 实时推理）",
            "- **口径声明：本轮不含真人盲评；全部得分来自 AI 五维评分（0–100）"
            "＋与梅见完全一致的确定性加权（25/20/25/15/15）；gold 由 AI 两遍独立标注＋分歧仲裁产生。**",
            "",
            "## 1. 数据与拆分",
            "",
            "- 源表：果立方品牌认知迁移验证.xlsx（101 条，KEEP∩ANALYSIS_BASELINE 95 条，剔除 6 条）",
            "- 拆分种子 20260909：gold 12 / 基线 39 / 挑战 12 / holdout 12 / 增量 10+10"
            "（每个需 AI 标注的包 ≥10 条为系统硬约束）",
            "",
            "## 2. gold 双审（AI 两次审核）",
            "",
            f"- 整行一致率：{agreement['full_row_match']}/{agreement['total']}"
            f"（{agreement['full_row_match_rate']:.1%}）",
            f"- route 一致率 {per_label['route']['rate']:.1%}；experience_scope "
            f"{per_label['experience_scope']['rate']:.1%}；evidence_grade "
            f"{per_label['evidence_grade']['rate']:.1%}",
            f"- 分歧 {arbitration_count} 条，仲裁后冻结（frozen_gold_labels.json）",
            "",
            "## 3. 语料包发布",
            "",
            "| 角色 | package_id | 记录数 | 证据原子 |",
            "|---|---|---|---|",
            package_rows,
            "",
            "## 4. 基线排名（候选生成后）",
            "",
            board_table(baseline_board),
            "",
            "## 5. 冻结时排名（挑战批修订 + holdout 验证后）",
            "",
            board_table(freeze_board),
            "",
            "## 6. 增量演化",
            "",
            *checkpoint_lines,
            "",
            "## 7. 最终排名（增量批全部释放后，AI 得分直出）",
            "",
            board_table(final_board),
            "",
            "## 8. 入选叙事与场景（压测后终稿，按最终 AI 加权分）",
            "",
            *(
                [
                    "品牌专属度压力测试（攻击→修订→复审，与官方五候选验证同合同）已对全部"
                    "入选叙事完成：第一轮共提出 12 条实质风险（MATERIAL_RISK），DeepSeek "
                    "修订后第二轮复审全部收敛为 NOTE（READY 终态）。逐条 finding 与修订 "
                    "diff 见 `06_压力测试/压力测试报告.md`。",
                    "",
                ]
                if pressure_finals
                else ["（压力测试尚未完成，以下为压测前版本。）", ""]
            ),
            *narrative_sections,
            "",
            "## 9. 产物索引",
            "",
            "- `00_准备/`：品牌配置、清洗核对、拆分清单、分包 CSV",
            "- `01_gold双审/`：attempt-0001 / attempt-0002 / arbitration / frozen_gold_labels / 一致率报告",
            "- `02_预处理/`：发布记录、各包证据原子",
            "- `03_决策链路/`：逐步 foundation 快照、每步排行榜、state.json、checkpoints.json",
            "- `06_压力测试/`：attempt 目录（逐轮审计任务/结果/修订 diff/final_candidate）"
            "与压力测试报告.md",
            f"- 系统内语料包目录：`{PREPARED_ROOT}`",
            "",
        ]
    )
    report_path = EVIDENCE_ROOT / "总结报告.md"
    report_path.write_text(report + "\n", encoding="utf-8")
    log(f"总结报告已写入：{report_path}")


# ---------------------------------------------------------------- 主流程

class _RetryingClient:
    """操作员级重试：模型偶发输出不合 Schema（如多写一个 key）时重新采样一次。

    不改 prompt、不改 Schema、不改权重；LLM_MAX_RETRIES=0 的传输层纪律保持不变。
    每次重试都写运行日志留痕。
    """

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.model = inner._model

    def generate_json(self, **kwargs):
        try:
            return self._inner.generate_json(**kwargs)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            log(f"模型输出未通过校验（{type(exc).__name__}: {exc}），重新采样一次")
            return self._inner.generate_json(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="果立方第二品牌自由链路验证")
    parser.add_argument(
        "--stage",
        choices=["all", "gold", "preprocess", "decision", "pressure", "report"],
        default="all",
    )
    args = parser.parse_args()

    profile = json.loads(BRAND_PROFILE_PATH.read_text(encoding="utf-8"))
    set_active_brand(
        BrandProfile(
            brand_name=profile["brand_name"],
            industry=profile["industry"],
            target_audience=profile["target_audience"],
            research_question=profile["research_question"],
            competitors=tuple(profile["competitors"]),
            extended_alternatives=tuple(profile.get("extended_alternatives", ())),
            notes=profile.get("notes", ""),
        )
    )
    log(f"品牌档案已激活：{profile['brand_name']}；阶段：{args.stage}")

    settings = Settings.from_sources(dotenv_path=DOTENV_PATH)
    client = _RetryingClient(LLMClient(settings))
    started = datetime.now()
    try:
        if args.stage in {"all", "gold"}:
            stage_gold(client)
        publish_record = _load_publish_record()
        if args.stage in {"all", "preprocess"}:
            publish_record = stage_preprocess(client)
        if args.stage in {"all", "decision"}:
            if len(publish_record) < len(PREPROCESS_ROLES):
                raise ValueError("语料包未全部发布，不能运行决策链")
            foundation = stage_decision(client, publish_record)
        if args.stage == "pressure":
            stage_pressure(client, settings)
        if args.stage in {"all", "report"}:
            if args.stage == "report":
                state_payload = json.loads(
                    (DECISION_DIR / "state.json").read_text(encoding="utf-8")
                )
                foundation = DecisionFoundationState.model_validate(
                    state_payload["foundation"]
                )
            stage_report(foundation)
    except Exception:
        log("运行失败，堆栈如下：")
        log(traceback.format_exc())
        raise
    elapsed = (datetime.now() - started).total_seconds()
    log(f"阶段 {args.stage} 全部完成，耗时 {elapsed:.0f} 秒")


if __name__ == "__main__":
    main()
