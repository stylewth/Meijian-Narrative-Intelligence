"""Zero-network gate for the official decision run.

This module deliberately imports no LLM client, SDK, HTTP transport, or runner.
It only loads already-published prepared corpora and validates their contracts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Settings
from src.prompt_loader import PromptMetadata, load_prompt_metadata
from src.schemas import (
    DecisionModelProfile,
    DecisionRunManifest,
    DatasetSplitRole,
    EvidenceAtom,
    EvidenceGrade,
    EvidenceRoute,
    ExperienceScope,
    FormalPreparedCorpusGroupManifest,
    NarrativeCandidate,
    EvidenceQuote,
    PreparedCorpusPackage,
    SpecificityAuditBatch,
    SpecificityAuditType,
    SpecificityFinding,
    SpecificitySeverity,
    OfflineSpecificityResult,
    ModelRuntime,
    SourceReference,
    SourceType,
)
from src.services.decision_inputs import assemble_official_demo_inputs
from src.services.brand_specificity_offline import (
    LUNA_SPECIFICITY_PROFILE,
    export_specificity_task,
    import_specificity_result,
)
from src.services.demo_package import load_published_demo
from src.services.decision_run_store import DecisionRunStore
from src.services.prepared_corpus import (
    canonical_json_bytes,
    load_prepared_corpus,
    sha256_bytes,
)
from src.services.specificity_memory import load_confirmed_patterns


FORMAL_PROMPT_NAME = "evidence_routing"
EXPECTED_PACKAGE_IDS = {
    "ANALYSIS": "formal-analysis",
    "CHALLENGE": "formal-challenge",
    "HOLDOUT": "formal-holdout",
}
EXPECTED_COUNTS = {"ANALYSIS": 249, "CHALLENGE": 60, "HOLDOUT": 60}
SPECIFICITY_AUDIT_PROMPT = PROJECT_ROOT / "prompts" / "brand_specificity_audit.md"
SPECIFICITY_REVISION_PROMPT = PROJECT_ROOT / "prompts" / "brand_specificity_revision.md"
SPECIFICITY_MEMORY_DEFAULT = PROJECT_ROOT / "data" / "competition" / "narrative_memory" / "confirmed_patterns.json"
SPECIFICITY_DEMO_DEFAULT = PROJECT_ROOT / "data" / "competition" / "demo_v2"


def _contract_file_hash(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    data = path.read_bytes()
    if data.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"{label} must not contain a UTF-8 BOM")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} must be UTF-8") from exc
    return sha256_bytes(data)


def _specificity_contract_report() -> dict[str, object]:
    prompt_specs = {
        "audit": (SPECIFICITY_AUDIT_PROMPT, SpecificityAuditBatch),
        "revision": (
            SPECIFICITY_REVISION_PROMPT,
            __import__("src.schemas", fromlist=["SpecificityRevisionBatch"]).SpecificityRevisionBatch,
        ),
    }
    prompts: dict[str, object] = {}
    for name, (path, model) in prompt_specs.items():
        prompts[name] = {
            "path": str(path),
            "prompt_sha256": _contract_file_hash(path, f"specificity {name} prompt"),
            "schema_sha256": sha256_bytes(
                canonical_json_bytes(model.model_json_schema())
            ),
        }
    return {
        "profile": LUNA_SPECIFICITY_PROFILE.model_dump(mode="json"),
        "prompts": prompts,
    }


def _specificity_memory_report(path: Path) -> dict[str, object]:
    if not path.exists() and not path.is_symlink():
        return {
            "path": str(path),
            "status": "NOT_FOUND_EMPTY",
            "patterns": 0,
            "sha256": None,
        }
    memory = load_confirmed_patterns(path)
    return {
        "path": str(path),
        "status": "VALIDATED",
        "patterns": len(memory.patterns),
        "sha256": sha256_bytes(path.read_bytes()),
    }


def _specificity_demo_report(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "NOT_BUILT"
    if path.is_symlink() or not path.is_dir():
        raise ValueError("frozen demo package root must be a real directory")
    published = load_published_demo(path)
    attempt = published.attempt
    if attempt.manifest.specificity_session_sha256 is None:
        raise ValueError("frozen demo package lacks a specificity session")
    if not (attempt.path / "specificity_session.json").is_file():
        raise ValueError("frozen demo package specificity session is missing")
    return "VALIDATED"


def _specificity_has_imported_result(run_root: Path) -> bool:
    if not run_root.exists() or run_root.is_symlink() or not run_root.is_dir():
        return False
    for run_path in run_root.iterdir():
        calls_path = run_path / "calls"
        if run_path.is_symlink() or not calls_path.is_dir() or calls_path.is_symlink():
            continue
        for call_path in calls_path.iterdir():
            response_path = call_path / "response.json"
            request_path = call_path / "request_manifest.json"
            if (
                call_path.is_symlink()
                or not call_path.is_dir()
                or not request_path.is_file()
                or not response_path.is_file()
            ):
                continue
            try:
                request = json.loads(request_path.read_text(encoding="utf-8"))
                response = json.loads(response_path.read_text(encoding="utf-8"))
                if request.get("stage") != "AWAITING_SPECIFICITY_AUDIT":
                    continue
                payload = response["response_payload"]
                SpecificityAuditBatch.model_validate_json(
                    canonical_json_bytes(payload), strict=True
                )
                if response["response_sha256"] != sha256_bytes(
                    canonical_json_bytes(payload)
                ):
                    continue
                return True
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                continue
    return False


def _specificity_export_import_self_check() -> str:
    decision_profile = DecisionModelProfile(
        profile_id="preflight-decision-v1",
        provider="deepseek",
        model_id="deepseek-v4-flash",
        runtime=ModelRuntime.CLOUD,
        response_format="json_object",
        prompt_version="decision-v1",
        endpoint_profile="official",
        thinking_enabled=True,
        reasoning_effort="max",
        max_retries=0,
    )
    manifest = DecisionRunManifest(
        decision_run_id="preflight-specificity",
        package_ids={"ANALYSIS": "analysis", "CHALLENGE": "challenge", "HOLDOUT": "holdout"},
        package_sha256={"ANALYSIS": "a" * 64, "CHALLENGE": "b" * 64, "HOLDOUT": "c" * 64},
        decision_model=decision_profile,
        specificity_model=LUNA_SPECIFICITY_PROFILE,
        created_at=datetime.now(timezone.utc),
        run_artifact_sha256="d" * 64,
    )
    candidate = NarrativeCandidate(
        candidate_id="preflight-candidate",
        primary_conflict_id="preflight-conflict",
        supporting_conflict_ids=["preflight-conflict"],
        title="preflight candidate",
        target_audience="preflight audience",
        user_conflict="preflight conflict",
        brand_opportunity="preflight opportunity",
        why_meijian="preflight fit",
        competitor_difference="preflight difference",
        brand_role="preflight role",
        draft_proposition="preflight proposition",
        main_scenes=["preflight scene"],
        content_theme="preflight theme",
        supporting_evidence=[EvidenceQuote(comment_id="preflight-comment", quote="preflight quote")],
        counter_evidence_note="no counter evidence in contract self-check",
        risks=["preflight risk"],
    )
    atom = EvidenceAtom(
        evidence_id="preflight-evidence",
        comment_id="preflight-comment",
        route=EvidenceRoute.BRAND,
        experience_scope=ExperienceScope.ACTUAL_USE,
        evidence_grade=EvidenceGrade.A,
        explanation="preflight evidence",
        source=SourceReference(
            source_id="preflight-source",
            source_type=SourceType.USER_COMMENT,
            source_ref="preflight-self-check",
        ),
    )
    with tempfile.TemporaryDirectory(prefix="specificity-preflight-") as temporary:
        store = DecisionRunStore.create(Path(temporary) / "runs", manifest)
        task = export_specificity_task(
            store, 1, [candidate], [atom], [], [], []
        )
        findings = [
            {
                "finding_id": f"preflight-finding-{index}",
                "candidate_id": candidate.candidate_id,
                "audit_type": audit_type.value,
                "severity": SpecificitySeverity.NOTE.value,
                "claim": "preflight claim",
                "evidence_ids": [atom.evidence_id],
                "brand_fact_ids": [],
                "competitor_replacement_result": "preflight result",
                "delivery_condition": "preflight condition",
                "failure_mode": "preflight failure mode",
                "suggested_patch": None,
                "triggers_next_round": False,
            }
            for index, audit_type in enumerate(SpecificityAuditType, start=1)
        ]
        output = SpecificityAuditBatch(
            run_id=task.run_id,
            round_index=1,
            candidate_ids=[candidate.candidate_id],
            candidate_versions=task.candidate_versions,
            audits=[
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_version": 1,
                    "findings": findings,
                    "consumer_evidence": [],
                    "meijian_assets": [],
                    "competitor_replacement_result": "preflight result",
                    "product_delivery_conditions": [],
                    "applicable_scenarios": [],
                    "failure_reasons": [],
                    "next_validation_question": "preflight question",
                }
            ],
        )
        result_path = Path(temporary) / "specificity-result.json"
        result_path.write_bytes(
            canonical_json_bytes(
                OfflineSpecificityResult(
                    task_id=task.task_id,
                    run_id=task.run_id,
                    round_index=task.round_index,
                    model_profile=task.model_profile,
                    prompt_sha256=task.prompt_sha256,
                    schema_sha256=task.schema_sha256,
                    input_sha256=task.input_sha256,
                    candidate_versions=task.candidate_versions,
                    output=output.model_dump(mode="json"),
                )
            )
        )
        import_specificity_result(store, task, result_path)
    return "PASS"


def load_official_settings() -> Settings:
    return Settings.from_env()


def load_official_prompt() -> PromptMetadata:
    return load_prompt_metadata(FORMAL_PROMPT_NAME)


def _read_canonical_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required manifest is missing or not a regular file: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON manifest: {path}") from exc
    if canonical_json_bytes(payload) + b"\n" != raw:
        raise ValueError(f"manifest is not canonical JSON: {path}")
    return payload


def _locate_group_root(prepared_root: Path) -> Path:
    root = Path(prepared_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"prepared_root must be an existing directory: {root}")
    direct_manifest = root / "manifest.json"
    if direct_manifest.exists() or direct_manifest.is_symlink():
        return root
    candidates = [
        entry
        for entry in root.iterdir()
        if not entry.is_symlink() and entry.is_dir() and (entry / "manifest.json").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError(
            "prepared_root must contain exactly one formal prepared corpus group"
        )
    return candidates[0]


def _stable_id(record: Any) -> str:
    return record.raw_id or record.comment_id


def _annotation_identity(package: PreparedCorpusPackage) -> tuple[object, ...]:
    if len(package.annotation_run_manifests) != 1:
        raise ValueError("formal package must contain exactly one annotation manifest")
    annotation = package.annotation_run_manifests[0]
    return (
        annotation.annotation_version,
        annotation.model_id,
        annotation.reasoning_effort,
        annotation.dataset_version,
        annotation.dataset_sha256,
        annotation.prompt_version,
        annotation.prompt_sha256,
        annotation.result_sha256,
    )


def _load_group(
    prepared_root: Path,
) -> tuple[FormalPreparedCorpusGroupManifest, dict[str, PreparedCorpusPackage]]:
    group_root = _locate_group_root(prepared_root)
    if group_root != prepared_root:
        wrapper_entries = {entry.name for entry in Path(prepared_root).iterdir()}
        if wrapper_entries != {group_root.name}:
            raise ValueError("prepared root contains extra formal group entries")
    group_entries = {entry.name: entry for entry in group_root.iterdir()}
    if set(group_entries) != {"manifest.json", "packages"}:
        raise ValueError("formal group root contains extra entries")
    if group_entries["packages"].is_symlink() or not group_entries["packages"].is_dir():
        raise ValueError("formal group packages directory is missing")
    group_payload = _read_canonical_json(group_root / "manifest.json")
    try:
        group = FormalPreparedCorpusGroupManifest.model_validate(
            group_payload, strict=True
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("formal group manifest is invalid") from exc
    expected_group_sha = sha256_bytes(
        canonical_json_bytes(group.model_dump(mode="json", exclude={"group_sha256"}))
    )
    if group.group_sha256 != expected_group_sha:
        raise ValueError("formal group SHA-256 does not match manifest")
    if group.package_ids != EXPECTED_PACKAGE_IDS:
        raise ValueError("formal group package identity is not official")

    packages_root = group_root / "packages"
    if packages_root.is_symlink() or not packages_root.is_dir():
        raise ValueError("formal group packages directory is missing")
    package_entries = {entry.name: entry for entry in packages_root.iterdir()}
    if set(package_entries) != set(EXPECTED_PACKAGE_IDS.values()):
        raise ValueError("formal packages directory contains extra packages or files")
    if any(entry.is_symlink() or not entry.is_dir() for entry in package_entries.values()):
        raise ValueError("formal packages entries must be real directories")
    packages: dict[str, PreparedCorpusPackage] = {}
    for role, package_id in EXPECTED_PACKAGE_IDS.items():
        published = load_prepared_corpus(packages_root, package_id)
        package = published.package
        if package.manifest.package_id != package_id:
            raise ValueError(f"{role} package identity does not match manifest")
        if package.manifest.package_sha256 != group.package_sha256[role]:
            raise ValueError(f"{role} package SHA-256 does not match group manifest")
        if package.manifest.record_count != EXPECTED_COUNTS[role]:
            raise ValueError(
                f"{role} package must contain exactly {EXPECTED_COUNTS[role]} records"
            )
        expected_role = {
            "ANALYSIS": DatasetSplitRole.ANALYSIS,
            "CHALLENGE": DatasetSplitRole.CHALLENGE_POOL,
            "HOLDOUT": DatasetSplitRole.HOLDOUT,
        }[role]
        if {item.split_role for item in package.split_manifest.assignments} != {
            expected_role
        }:
            raise ValueError(f"{role} package split identity does not match role")
        annotation = package.annotation_run_manifests[0]
        if _annotation_identity(package) != (
            group.annotation_identity.annotation_version,
            group.annotation_identity.model_id,
            group.annotation_identity.reasoning_effort,
            group.annotation_identity.dataset_version,
            group.annotation_identity.dataset_sha256,
            group.annotation_identity.prompt_version,
            group.annotation_identity.prompt_sha256,
            group.annotation_identity.result_sha256,
        ):
            raise ValueError("formal package annotation identity mismatch")
        if (
            package.manifest.dataset_version != annotation.dataset_version
            or package.manifest.dataset_sha256 != annotation.dataset_sha256
            or package.manifest.annotation_version != annotation.annotation_version
            or package.manifest.annotation_model_id != annotation.model_id
            or package.manifest.annotation_reasoning_effort != annotation.reasoning_effort
            or package.manifest.annotation_prompt_version != annotation.prompt_version
            or package.manifest.annotation_prompt_sha256 != annotation.prompt_sha256
        ):
            raise ValueError("formal package dataset/annotation/prompt identity mismatch")
        packages[role] = package
    return group, packages


def _validate_boundaries(
    group: FormalPreparedCorpusGroupManifest,
    packages: dict[str, PreparedCorpusPackage],
) -> dict[str, object]:
    analysis_ids = [_stable_id(record) for record in packages["ANALYSIS"].records]
    baseline_ids = group.analysis_baseline_ids
    release_ids = group.analysis_release_ids
    if len(baseline_ids) != 229 or len(release_ids) != 20:
        raise ValueError("ANALYSIS baseline/release counts must be exactly 229/20")
    if len(set(baseline_ids)) != 229 or len(set(release_ids)) != 20:
        raise ValueError("ANALYSIS baseline/release IDs must be unique")
    if set(baseline_ids) & set(release_ids):
        raise ValueError("ANALYSIS baseline and release IDs overlap")
    if baseline_ids + release_ids != analysis_ids:
        raise ValueError("ANALYSIS baseline/release order does not match package order")
    batches = [release_ids[offset : offset + 5] for offset in range(0, 20, 5)]
    if len(batches) != 4 or any(len(batch) != 5 for batch in batches):
        raise ValueError("ANALYSIS release order is not four ordered batches of five")
    return {"selected_ids": release_ids}


def _check_run_root(run_root: Path) -> None:
    root = Path(run_root)
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("run_root must be a directory")
    root.mkdir(parents=True, exist_ok=True)
    probe_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".preflight-write-", dir=root, delete=False
        ) as handle:
            probe_path = Path(handle.name)
            handle.write(b"preflight")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if probe_path is not None:
            probe_path.unlink()


def _validate_existing_run_manifests(run_root: Path) -> None:
    if not run_root.exists():
        return
    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError("run_root must be a real directory")
    for entry in run_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            raise ValueError("run_root contains a non-directory run entry")
        manifest_path = entry / "run_manifest.json"
        payload = _read_canonical_json(manifest_path)
        try:
            manifest = DecisionRunManifest.model_validate(payload, strict=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid decision run manifest: {manifest_path}") from exc
        if manifest.decision_run_id != entry.name:
            raise ValueError("decision run manifest ID does not match its directory")
        if manifest.specificity_model != LUNA_SPECIFICITY_PROFILE:
            raise ValueError("decision run manifest lacks the exact Luna gpt-5.6-luna Max profile")


def _check_output_root(output_root: Path) -> None:
    root = Path(output_root)
    if not root.exists():
        return
    if root.is_symlink() or not root.is_dir():
        raise ValueError("output_root is an existing non-directory")
    if any(root.iterdir()):
        raise ValueError("output_root must be absent or empty; formal attempt conflicts")


def run_preflight(
    *,
    prepared_root: Path,
    run_root: Path,
    output_root: Path,
    specificity_memory_path: Path | None = None,
    demo_root: Path | None = None,
) -> dict[str, object]:
    settings = load_official_settings()
    settings.validate_official_decision_profile()
    prompt = load_official_prompt()
    group, packages = _load_group(Path(prepared_root))
    annotation = packages["ANALYSIS"].annotation_run_manifests[0]
    if prompt.name != FORMAL_PROMPT_NAME:
        raise ValueError("formal prompt name does not match")
    if prompt.version != annotation.prompt_version or prompt.sha256 != annotation.prompt_sha256:
        raise ValueError("formal prompt version/SHA does not match annotation manifest")
    selection_manifest = _validate_boundaries(group, packages)
    bundle = assemble_official_demo_inputs(
        analysis=packages["ANALYSIS"],
        challenge=packages["CHALLENGE"],
        holdout=packages["HOLDOUT"],
        selection_manifest=selection_manifest,
    )
    if [len(batch) for batch in bundle.release_batches] != [5, 5, 5, 5]:
        raise ValueError("ANALYSIS release batches must be exactly 4x5")
    all_ids = [
        _stable_id(record)
        for role in ("ANALYSIS", "CHALLENGE", "HOLDOUT")
        for record in packages[role].records
    ]
    unique_count = len(set(all_ids))
    if len(all_ids) != 369 or unique_count != 369:
        raise ValueError("formal packages must contain exactly 369 unique stable IDs")
    if bundle.all_stable_ids_sha256 != group.all_stable_ids_sha256:
        raise ValueError("formal group all-stable-ID SHA-256 does not match packages")
    _check_run_root(Path(run_root))
    _validate_existing_run_manifests(Path(run_root))
    _check_output_root(Path(output_root))
    specificity_contract = _specificity_contract_report()
    memory_report = _specificity_memory_report(
        Path(specificity_memory_path or SPECIFICITY_MEMORY_DEFAULT)
    )
    demo_status = _specificity_demo_report(
        Path(demo_root or SPECIFICITY_DEMO_DEFAULT)
    )
    offline_self_check = _specificity_export_import_self_check()
    execution_status = (
        "已导入并验证"
        if demo_status == "VALIDATED" or _specificity_has_imported_result(Path(run_root))
        else "真实 Luna 任务尚未执行"
    )
    return {
        "status": "PASS",
        "counts": dict(EXPECTED_COUNTS),
        "unique": unique_count,
        "profile": {
            "provider": "deepseek",
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "thinking": settings.llm_thinking_enabled,
            "reasoning_effort": settings.llm_reasoning_effort,
            "response_format": settings.llm_response_format.value,
            "max_retries": settings.llm_max_retries,
        },
        "prompt": {
            "name": prompt.name,
            "version": prompt.version,
            "sha256": prompt.sha256,
        },
        "annotation": {
            "version": annotation.annotation_version,
            "model": annotation.model_id,
            "result_sha256": annotation.result_sha256,
        },
        "packages": {
            role: {
                "package_id": packages[role].manifest.package_id,
                "package_sha256": packages[role].manifest.package_sha256,
            }
            for role in ("ANALYSIS", "CHALLENGE", "HOLDOUT")
        },
        "specificity": {
            **specificity_contract,
            "memory": memory_report,
            "offline_self_check": offline_self_check,
            "demo_package": demo_status,
        },
        "execution_status": execution_status,
        "network_calls": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the zero-network decision preflight.")
    parser.add_argument("--prepared-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--specificity-memory", type=Path, default=None)
    parser.add_argument("--demo-root", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = run_preflight(
        prepared_root=arguments.prepared_root,
        run_root=arguments.run_root,
        output_root=arguments.output_root,
        specificity_memory_path=arguments.specificity_memory,
        demo_root=arguments.demo_root,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
