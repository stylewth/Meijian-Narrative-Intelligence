"""品牌专属度压力测试循环（审计者 API 可配置）。

镜像官方五候选验证的「攻击 → 修订 → 复审」语义：Same Schema
（SpecificityAuditBatch / SpecificityRevisionBatchV2）、四类审计各恰好一条
finding、NOTE 不触发下一轮、上限 3 轮 / 2 次修订；差异是候选数与证据规模
可配置（不绑定官方 5/309/54/6 冻结合同）。

审计者（攻击方）两种供给方式：
- API 模式：AUDITOR_* 配置了 OpenAI 兼容端点时，用 LLMClient 直调；
- 离线代跑模式：auditor=None 时循环只落盘 auditor_task.json 并挂起；
  调用方（例如 ZCode 子代理）回填 auditor_result.json 后重跑即续。

状态逐候选落盘（candidates/<id>/state.json），天然断点续跑。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from src.brand_profile import active_brand_prompt_block
from src.llm_client import LLMClient
from src.schemas import (
    CommentRecord,
    EvidenceAtom,
    NarrativeCandidate,
    SpecificityAuditBatch,
    SpecificityAuditType,
    SpecificitySeverity,
)
from src.services.brand_specificity import SpecificityRevisionBatchV2
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes

MAX_AUDIT_ROUNDS = 3
MAX_REVISIONS = 2
AUDIT_TYPES = tuple(SpecificityAuditType)
NON_NOTE_SEVERITIES = {SpecificitySeverity.HARD_CONFLICT, SpecificitySeverity.MATERIAL_RISK}
ALLOWED_FIELDS = (
    "title",
    "target_audience",
    "user_conflict",
    "brand_opportunity",
    "why_brand",
    "competitor_difference",
    "brand_role",
    "draft_proposition",
    "main_scenes",
    "content_theme",
    "risks",
)
TERMINAL_STATUSES = {"READY", "HUMAN_REVIEW", "BLOCKED"}

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
_AUDIT_PROMPT_PATH = _PROMPT_DIR / "brand_specificity_audit_v3.md"
_REVISION_PROMPT_PATH = _PROMPT_DIR / "brand_specificity_revision_v3.md"


def _load_prompt_text(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} 必须是常规文件: {path}")
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} 不得包含 UTF-8 BOM")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} 必须是 UTF-8") from exc


def audit_system_prompt() -> str:
    return f"{_load_prompt_text(_AUDIT_PROMPT_PATH, '审计 prompt')}\n\n{active_brand_prompt_block()}"


def revision_system_prompt() -> str:
    return f"{_load_prompt_text(_REVISION_PROMPT_PATH, '修订 prompt')}\n\n{active_brand_prompt_block()}"


class Auditor(Protocol):
    def audit(self, task: dict[str, Any]) -> SpecificityAuditBatch: ...


class ApiSpecificityAuditor:
    """AUDITOR_* API 配置口：任意 OpenAI 兼容端点即可担任攻击方。"""

    def __init__(self, client: LLMClient, *, run_id: str) -> None:
        self._client = client
        self._run_id = run_id
        self.model = client._model
        self.provider = "api"

    def audit(self, task: dict[str, Any]) -> SpecificityAuditBatch:
        return self._client.generate_json(
            system_prompt=audit_system_prompt(),
            user_prompt=json.dumps(task, ensure_ascii=False),
            response_model=SpecificityAuditBatch,
        )


@dataclass
class PressureLoopResult:
    states: dict[str, dict[str, Any]]
    pending_tasks: list[Path] = field(default_factory=list)
    all_terminal: bool = False


def _write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    data = canonical_json_bytes(payload)
    if path.exists():
        if not replace and path.read_bytes() == data:
            return
        if not replace:
            raise ValueError(f"拒绝覆盖已有产物: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _read_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence_rows(
    evidence_atoms: list[EvidenceAtom], records_by_id: dict[str, CommentRecord]
) -> list[dict[str, str]]:
    rows = []
    for atom in sorted(evidence_atoms, key=lambda item: item.evidence_id):
        record = records_by_id[atom.comment_id]
        rows.append(
            {
                "comment_id": atom.comment_id,
                "raw_content": record.raw_content,
                "sample_type": record.sample_type.value,
                "route": atom.route.value,
                "evidence_grade": atom.evidence_grade.value,
                "explanation": atom.explanation,
            }
        )
    return rows


def _audit_task(
    *,
    run_id: str,
    candidate: NarrativeCandidate,
    round_index: int,
    candidate_version: int,
    history: list[dict[str, Any]],
    evidence_rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "attempt_id": run_id,
        "round_index": round_index,
        "candidate_id": candidate.candidate_id,
        "candidate_version": candidate_version,
        "candidate": candidate.model_dump(mode="json"),
        "history": history,
        "evidence": evidence_rows,
        "audit_types": [item.value for item in AUDIT_TYPES],
        "allowed_fields": list(ALLOWED_FIELDS),
        "brand_context": active_brand_prompt_block(),
    }


def _parse_audit(batch: Any, task: dict[str, Any]) -> SpecificityAuditBatch:
    if isinstance(batch, SpecificityAuditBatch):
        validated = batch
    else:
        validated = SpecificityAuditBatch.model_validate_json(
            canonical_json_bytes(batch), strict=True
        )
    if validated.run_id != task["attempt_id"]:
        raise ValueError("审计 run_id 与任务不一致")
    if validated.round_index != task["round_index"]:
        raise ValueError("审计 round_index 与任务不一致")
    if validated.candidate_ids != [task["candidate_id"]]:
        raise ValueError("审计候选与任务不一致")
    known_ids = {row["comment_id"] for row in task["evidence"]}
    for audit in validated.audits:
        for finding in audit.findings:
            unknown = set(finding.evidence_ids) - known_ids
            if unknown:
                raise ValueError(f"finding {finding.finding_id} 引用了未知评论: {sorted(unknown)}")
    return validated


def _outcome_for_audit(
    audit: SpecificityAuditBatch, round_index: int
) -> tuple[str | None, None]:
    if all(
        finding.severity is SpecificitySeverity.NOTE
        for item in audit.audits
        for finding in item.findings
    ):
        return "READY", None
    if round_index >= MAX_AUDIT_ROUNDS:
        return "HUMAN_REVIEW", None
    return None, None


def _history_from_rounds(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history = []
    for item in rounds:
        audit = item["audit"]
        findings = [
            {
                "finding_id": finding["finding_id"],
                "audit_type": finding["audit_type"],
                "severity": finding["severity"],
                "claim": finding["claim"],
            }
            for audit_item in audit["audits"]
            for finding in audit_item["findings"]
        ]
        revision = item.get("revision")
        dispositions = (
            [
                {"finding_id": response["finding_id"], "disposition": response["disposition"]}
                for revision_item in revision["revisions"]
                for response in revision_item["responses"]
            ]
            if revision
            else []
        )
        history.append(
            {
                "round_index": item["round_index"],
                "findings": findings,
                "dispositions": dispositions,
            }
        )
    return history


def _validate_revision(
    revision: SpecificityRevisionBatchV2,
    *,
    audit: SpecificityAuditBatch,
    candidate: NarrativeCandidate,
    version: int,
    round_index: int,
) -> None:
    if revision.run_id != audit.run_id or revision.round_index != audit.round_index:
        raise ValueError("修订 run_id/round_index 与审计不一致")
    if len(revision.revisions) != 1:
        raise ValueError("修订必须恰好覆盖一个候选")
    item = revision.revisions[0]
    if item.candidate_id != candidate.candidate_id:
        raise ValueError("修订候选 ID 不一致")
    if item.from_version != version or item.to_version != version + 1:
        raise ValueError("修订版本号必须恰好递增一")
    target_findings = [
        finding
        for audit_item in audit.audits
        for finding in audit_item.findings
        if finding.severity in NON_NOTE_SEVERITIES
    ]
    response_ids = [response.finding_id for response in item.responses]
    expected_ids = [finding.finding_id for finding in target_findings]
    if sorted(response_ids) != sorted(expected_ids):
        raise ValueError("修订 response 必须与全部非 NOTE finding 一一对应")
    accept_ids = {
        response.finding_id
        for response in item.responses
        if response.disposition == "ACCEPT"
    }
    unresolved_ids = [
        response.finding_id for response in item.responses if response.disposition != "ACCEPT"
    ]
    if sorted(item.unresolved_finding_ids) != sorted(unresolved_ids):
        raise ValueError("unresolved_finding_ids 必须恰为 REJECT/DEFER 集合")
    if item.new_brand_facts:
        raise ValueError("未提供公开证据时不得新增品牌事实")
    original = candidate.model_dump(mode="json")
    revised = item.revised_candidate.model_dump(mode="json")
    if item.revised_candidate.candidate_id != candidate.candidate_id:
        raise ValueError("修订不得改变候选身份")
    changed = {key for key in original if original[key] != revised[key]}
    if not changed.issubset(set(ALLOWED_FIELDS)):
        raise ValueError(f"修订改动了不允许的字段: {sorted(changed - set(ALLOWED_FIELDS))}")
    if accept_ids and not changed:
        raise ValueError("ACCEPT 必须实际修改候选文本")
    if not accept_ids and changed:
        raise ValueError("没有 ACCEPT 时不得修改候选文本")


def _apply_revision(
    candidate: NarrativeCandidate, revision: SpecificityRevisionBatchV2
) -> list[dict[str, Any]]:
    revised = revision.revisions[0].revised_candidate
    original = candidate.model_dump(mode="json")
    updated = revised.model_dump(mode="json")
    diffs = []
    for key in ALLOWED_FIELDS:
        if original[key] != updated[key]:
            diffs.append(
                {
                    "field": key,
                    "before": original[key],
                    "after": updated[key],
                }
            )
    return diffs


def run_pressure_loop(
    *,
    run_id: str,
    attempt_dir: Path,
    candidates: list[NarrativeCandidate],
    evidence_atoms: list[EvidenceAtom],
    records_by_id: dict[str, CommentRecord],
    deepseek_client: Any,
    auditor: Auditor | None = None,
    auditor_meta: dict[str, Any] | None = None,
) -> PressureLoopResult:
    if not candidates or len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("压测候选必须非空且 ID 唯一")
    attempt_dir = Path(attempt_dir)
    meta_path = attempt_dir / "attempt_meta.json"
    if not meta_path.exists():
        _write_json(
            meta_path,
            {
                "run_id": run_id,
                "auditor": auditor_meta or {"provider": "zcode-subagent"},
                "limits": {"max_audit_rounds": MAX_AUDIT_ROUNDS, "max_revisions": MAX_REVISIONS},
                "schema": "pressure-audit-custom-v1",
            },
        )
    evidence_rows = _evidence_rows(evidence_atoms, records_by_id)
    states: dict[str, dict[str, Any]] = {}
    pending_tasks: list[Path] = []
    for candidate in candidates:
        child_dir = attempt_dir / "candidates" / candidate.candidate_id
        state = _read_json(child_dir / "state.json") or {
            "candidate_id": candidate.candidate_id,
            "status": "PENDING",
            "next_step": "AUDITOR",
            "current_version": 1,
            "current_candidate": candidate.model_dump(mode="json"),
            "rounds": [],
            "failure": None,
        }
        rounds = list(state["rounds"])
        while state["status"] not in TERMINAL_STATUSES:
            if state["next_step"] == "AUDITOR":
                round_index = len(rounds) + 1
                if round_index > MAX_AUDIT_ROUNDS:
                    state.update({"status": "HUMAN_REVIEW", "next_step": "TERMINAL"})
                    break
                current = NarrativeCandidate.model_validate(state["current_candidate"])
                history = _history_from_rounds(rounds)
                version = 1 + sum(1 for item in rounds if item.get("revision") is not None)
                task = _audit_task(
                    run_id=run_id,
                    candidate=current,
                    round_index=round_index,
                    candidate_version=version,
                    history=history,
                    evidence_rows=evidence_rows,
                )
                round_dir = child_dir / "rounds" / f"round-{round_index:02d}"
                task_path = round_dir / "auditor_task.json"
                _write_json(task_path, task)
                result_path = round_dir / "auditor_result.json"
                raw_result = _read_json(result_path)
                try:
                    if raw_result is not None:
                        audit = _parse_audit(raw_result, task)
                    elif auditor is not None:
                        audit = _parse_audit(auditor.audit(task), task)
                        _write_json(result_path, json.loads(audit.model_dump_json()))
                    else:
                        state.update(
                            {
                                "status": "AWAITING_AUDITOR",
                                "next_step": "AUDITOR",
                                "failure": None,
                            }
                        )
                        pending_tasks.append(task_path)
                        break
                except (ValidationError, ValueError) as exc:
                    state.update(
                        {
                            "status": "FAILED",
                            "next_step": "AUDITOR",
                            "failure": f"审计结果无效: {exc}",
                        }
                    )
                    break
                outcome, _ = _outcome_for_audit(audit, round_index)
                rounds.append(
                    {
                        "round_index": round_index,
                        "candidate_version": version,
                        "auditor_result_sha256": sha256_bytes(result_path.read_bytes()),
                        "audit": json.loads(audit.model_dump_json()),
                        "revision": None,
                    }
                )
                if outcome is not None:
                    state.update(
                        {
                            "rounds": rounds,
                            "status": outcome,
                            "next_step": "TERMINAL",
                            "failure": None,
                        }
                    )
                    _write_json(
                        child_dir / "final_candidate.json",
                        {
                            "candidate": state["current_candidate"],
                            "status": outcome,
                            "version": version,
                        },
                    )
                    break
                state.update(
                    {
                        "rounds": rounds,
                        "status": "AWAITING_DEEPSEEK",
                        "next_step": "DEEPSEEK",
                        "failure": None,
                    }
                )
            if state["next_step"] == "DEEPSEEK":
                revision_count = sum(1 for item in rounds if item.get("revision") is not None)
                if revision_count >= MAX_REVISIONS:
                    state.update({"status": "HUMAN_REVIEW", "next_step": "TERMINAL"})
                    break
                round_index = len(rounds)
                current = NarrativeCandidate.model_validate(state["current_candidate"])
                version = 1 + revision_count
                audit = SpecificityAuditBatch.model_validate_json(
                    canonical_json_bytes(rounds[-1]["audit"]), strict=True
                )
                # 修订载荷只带候选与审计实际引用的评论证据：压缩推理负担，
                # 也约束修订只能绑定与主张直接相关的评论 ID。
                referenced = {
                    quote.comment_id
                    for quote in (*current.supporting_evidence, *current.counter_evidence)
                }
                for audit_item in audit.audits:
                    for finding in audit_item.findings:
                        referenced.update(finding.evidence_ids)
                focused_rows = [
                    row for row in evidence_rows if row["comment_id"] in referenced
                ]
                payload = {
                    "run_id": audit.run_id,
                    "round_index": round_index,
                    "candidate_versions": {candidate.candidate_id: version},
                    "audit": json.loads(audit.model_dump_json()),
                    "candidates": [json.loads(current.model_dump_json())],
                    "comment_evidence": focused_rows,
                    "public_evidence": [],
                }
                try:
                    revision = deepseek_client.generate_json(
                        system_prompt=revision_system_prompt(),
                        user_prompt=json.dumps(payload, ensure_ascii=False),
                        response_model=SpecificityRevisionBatchV2,
                    )
                    _validate_revision(
                        revision,
                        audit=audit,
                        candidate=current,
                        version=version,
                        round_index=round_index,
                    )
                except (ValidationError, ValueError) as exc:
                    state.update(
                        {
                            "status": "FAILED",
                            "next_step": "DEEPSEEK",
                            "failure": f"修订无效: {exc}",
                        }
                    )
                    break
                diffs = _apply_revision(current, revision)
                revised = revision.revisions[0].revised_candidate
                rounds[-1]["revision"] = json.loads(revision.model_dump_json())
                rounds[-1]["revision_diffs"] = diffs
                rounds[-1]["revision_sha256"] = sha256_bytes(
                    canonical_json_bytes(revision.model_dump(mode="json"))
                )
                state.update(
                    {
                        "rounds": rounds,
                        "current_candidate": json.loads(revised.model_dump_json()),
                        "current_version": version + 1,
                        "status": "AWAITING_AUDITOR",
                        "next_step": "AUDITOR",
                        "failure": None,
                    }
                )
        _write_json(
            child_dir / "state.json",
            {**state, "rounds": rounds},
            replace=True,
        )
        states[candidate.candidate_id] = {**state, "rounds": rounds}
    all_terminal = all(state["status"] in TERMINAL_STATUSES for state in states.values())
    return PressureLoopResult(
        states=states, pending_tasks=pending_tasks, all_terminal=all_terminal
    )
