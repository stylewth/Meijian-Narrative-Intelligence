from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.schemas import (
    BlindEvidencePackage,
    BlindSourceManifest,
    CandidateSelection,
    DecisionEntryMode,
    DecisionEvolutionCheckpoint,
    DecisionFoundationAudit,
    DecisionFoundationState,
    DecisionDisplayEvent,
    DecisionOriginalSnapshot,
    DemoAttemptManifest,
    DemoFeishuSyncArtifact,
    FinalCandidateSelection,
    PendingSelectionPackage,
    PublishedDemoManifest,
    SpecificitySession,
)
from src.services.brand_specificity_offline import LUNA_SPECIFICITY_PROFILE
from src.services.decision_run_store import DecisionRunStore
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes


_ALLOWED_ATTEMPT_IDS = frozenset({"attempt-0001", "attempt-0002"})
_CHECKPOINT_IDS = tuple(f"checkpoint-{index:02d}" for index in range(5))
_ATTEMPT_FILE_NAMES = frozenset({"manifest.json", "foundation.json", "checkpoints"})
_CHECKPOINT_FILE_NAMES = frozenset(
    f"{checkpoint_id}.json" for checkpoint_id in _CHECKPOINT_IDS
)
_PUBLISHED_FILE_NAME = "published.json"
_FINAL_ROOT_FILES = frozenset({
    "manifest.json", "foundation.json", "selection.json", "original_ai_snapshot.json",
    "specificity_session.json", "blind", "checkpoints", "final_selection.json",
    "display_events.json", "feishu_sync.json",
})
_FINAL_ARTIFACT_FILES = frozenset({
    "foundation.json", "selection.json", "original_ai_snapshot.json", "specificity_session.json",
    "blind/source_manifest.json", "blind/blind_evidence.json", "final_selection.json",
    "display_events.json", "feishu_sync.json",
    *(f"checkpoints/checkpoint-{index:02d}.json" for index in range(5)),
})


@dataclass(frozen=True)
class LoadedDemoAttempt:
    manifest: DemoAttemptManifest
    foundation: DecisionFoundationAudit | DecisionFoundationState
    checkpoints: tuple[DecisionEvolutionCheckpoint, ...]
    path: Path
    selection: CandidateSelection | None = None
    original_ai_snapshot: DecisionOriginalSnapshot | None = None
    final_selection: FinalCandidateSelection | None = None
    display_events: tuple[DecisionDisplayEvent, ...] = ()
    feishu_sync: DemoFeishuSyncArtifact | None = None


@dataclass(frozen=True)
class PublishedDemoAttempt:
    manifest: PublishedDemoManifest
    attempt: LoadedDemoAttempt
    path: Path


def _root_path(root: str | os.PathLike[str], *, create: bool) -> Path:
    root_path = Path(root)
    if root_path.exists() and (not root_path.is_dir() or root_path.is_symlink()):
        raise ValueError("demo package root must be a real directory")
    if not root_path.exists():
        if not create:
            raise ValueError("demo package root does not exist")
        root_path.mkdir(parents=True, exist_ok=True)
    return root_path.resolve(strict=True)


def _attempts_path(root: Path, *, create: bool) -> Path:
    attempts_path = root / "attempts"
    if attempts_path.exists():
        if not attempts_path.is_dir() or attempts_path.is_symlink():
            raise ValueError("attempts path must be a real directory")
    elif create:
        attempts_path.mkdir()
    else:
        raise ValueError("attempts directory does not exist")
    resolved = attempts_path.resolve(strict=True)
    if resolved.parent != root:
        raise ValueError("attempts path must remain inside demo package root")
    return resolved


def _validate_attempt_id(attempt_id: str) -> None:
    if attempt_id not in _ALLOWED_ATTEMPT_IDS:
        raise ValueError("demo attempt_id must be attempt-0001 or attempt-0002")


def _attempt_path(root: Path, attempt_id: str, *, create_parent: bool) -> Path:
    _validate_attempt_id(attempt_id)
    attempts_path = _attempts_path(root, create=create_parent)
    destination = (attempts_path / attempt_id).resolve(strict=False)
    if destination.parent != attempts_path:
        raise ValueError("attempt path must remain inside attempts directory")
    return destination


def _read_attempt_directory(directory: Path) -> dict[str, bytes]:
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("demo attempt path must be a real directory")
    entries = {entry.name: entry for entry in directory.iterdir()}
    missing = _ATTEMPT_FILE_NAMES - entries.keys()
    extra = entries.keys() - _ATTEMPT_FILE_NAMES
    if missing:
        raise ValueError(f"missing demo artifact: {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"unexpected demo artifact: {sorted(extra)[0]}")
    resolved_directory = directory.resolve(strict=True)
    for name, entry in entries.items():
        if name == "checkpoints":
            if entry.is_symlink() or not entry.is_dir():
                raise ValueError("checkpoints must be a real directory")
            if entry.resolve(strict=True).parent != resolved_directory:
                raise ValueError("checkpoints path escapes attempt")
            continue
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"demo artifact must be a regular file: {name}")
        if entry.resolve(strict=True).parent != resolved_directory:
            raise ValueError(f"demo artifact path escapes attempt: {name}")
    checkpoints_path = entries["checkpoints"]
    checkpoint_entries = {entry.name: entry for entry in checkpoints_path.iterdir()}
    missing_checkpoints = _CHECKPOINT_FILE_NAMES - checkpoint_entries.keys()
    extra_checkpoints = checkpoint_entries.keys() - _CHECKPOINT_FILE_NAMES
    if missing_checkpoints:
        raise ValueError(f"missing checkpoint artifact: {sorted(missing_checkpoints)[0]}")
    if extra_checkpoints:
        raise ValueError(f"unexpected checkpoint artifact: {sorted(extra_checkpoints)[0]}")
    resolved_checkpoints = checkpoints_path.resolve(strict=True)
    for name, entry in checkpoint_entries.items():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"checkpoint artifact must be a regular file: {name}")
        if entry.resolve(strict=True).parent != resolved_checkpoints:
            raise ValueError(f"checkpoint artifact path escapes checkpoints directory: {name}")
    return {
        "manifest.json": entries["manifest.json"].read_bytes(),
        "foundation.json": entries["foundation.json"].read_bytes(),
        **{
            f"checkpoints/{name}": checkpoint_entries[name].read_bytes()
            for name in _CHECKPOINT_FILE_NAMES
        },
    }


def _decode_canonical_json(data: bytes, name: str) -> Any:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {name}") from exc
    if canonical_json_bytes(payload) != data:
        raise ValueError(f"artifact is not canonical JSON: {name}")
    return payload


def _parse_attempt(
    artifacts: dict[str, bytes], attempt_id: str, path: Path
) -> LoadedDemoAttempt:
    try:
        _decode_canonical_json(artifacts["manifest.json"], "manifest.json")
        manifest = DemoAttemptManifest.model_validate_json(
            artifacts["manifest.json"], strict=True
        )
        _decode_canonical_json(artifacts["foundation.json"], "foundation.json")
        foundation = DecisionFoundationAudit.model_validate_json(
            artifacts["foundation.json"], strict=True
        )
        checkpoints_list: list[DecisionEvolutionCheckpoint] = []
        for checkpoint_id in _CHECKPOINT_IDS:
            artifact_name = f"checkpoints/{checkpoint_id}.json"
            _decode_canonical_json(artifacts[artifact_name], artifact_name)
            checkpoints_list.append(
                DecisionEvolutionCheckpoint.model_validate_json(
                    artifacts[artifact_name], strict=True
                )
            )
        checkpoints = tuple(checkpoints_list)
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("invalid JSON", "artifact is")):
            raise
        raise ValueError("invalid demo attempt artifact") from exc

    if manifest.attempt_id != attempt_id:
        raise ValueError("attempt manifest ID does not match requested attempt")
    if foundation.run_id != manifest.run_id:
        raise ValueError("foundation run_id does not match attempt manifest")
    if sha256_bytes(artifacts["foundation.json"]) != manifest.foundation_sha256:
        raise ValueError("foundation SHA-256 does not match attempt manifest")

    if tuple(manifest.checkpoint_ids) != _CHECKPOINT_IDS:
        raise ValueError("attempt manifest checkpoint IDs are invalid")
    if tuple(manifest.checkpoint_sha256) != _CHECKPOINT_IDS:
        raise ValueError("attempt manifest checkpoint hashes are invalid")
    for checkpoint_id, checkpoint in zip(_CHECKPOINT_IDS, checkpoints, strict=True):
        artifact_name = f"checkpoints/{checkpoint_id}.json"
        if checkpoint.checkpoint_id != checkpoint_id:
            raise ValueError("checkpoint ID does not match artifact name")
        if checkpoint.run_id != manifest.run_id:
            raise ValueError("checkpoint run_id does not match attempt manifest")
        if checkpoint.entry_mode is not DecisionEntryMode.DEMO:
            raise ValueError("demo attempts may only contain DEMO checkpoints")
        if sha256_bytes(artifacts[artifact_name]) != manifest.checkpoint_sha256[checkpoint_id]:
            raise ValueError("checkpoint SHA-256 does not match attempt manifest")

    return LoadedDemoAttempt(
        manifest=manifest,
        foundation=foundation,
        checkpoints=checkpoints,
        path=path,
    )


def _attempt_artifacts(
    manifest: DemoAttemptManifest,
    foundation: DecisionFoundationAudit,
    checkpoints: Sequence[DecisionEvolutionCheckpoint],
) -> dict[str, bytes]:
    if not isinstance(manifest, DemoAttemptManifest):
        raise TypeError("manifest must be a DemoAttemptManifest")
    if not isinstance(foundation, DecisionFoundationAudit):
        raise TypeError("foundation must be a DecisionFoundationAudit")
    if len(checkpoints) != len(_CHECKPOINT_IDS) or any(
        not isinstance(checkpoint, DecisionEvolutionCheckpoint) for checkpoint in checkpoints
    ):
        raise ValueError("checkpoints must contain exactly five DecisionEvolutionCheckpoint values")
    return {
        "manifest.json": canonical_json_bytes(manifest),
        "foundation.json": canonical_json_bytes(foundation),
        **{
            f"checkpoints/{checkpoint_id}.json": canonical_json_bytes(checkpoint)
            for checkpoint_id, checkpoint in zip(_CHECKPOINT_IDS, checkpoints, strict=True)
        },
    }


def write_demo_attempt(
    root: str | os.PathLike[str],
    manifest: DemoAttemptManifest,
    foundation: DecisionFoundationAudit,
    checkpoints: Sequence[DecisionEvolutionCheckpoint],
) -> LoadedDemoAttempt:
    """Freeze one validated demo attempt without overwriting prior attempts."""
    root_path = _root_path(root, create=True)
    if not isinstance(manifest, DemoAttemptManifest):
        raise TypeError("manifest must be a DemoAttemptManifest")
    destination = _attempt_path(root_path, manifest.attempt_id, create_parent=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"demo attempt already exists: {manifest.attempt_id}")

    artifacts = _attempt_artifacts(manifest, foundation, checkpoints)
    attempts_path = destination.parent
    temp_path = Path(tempfile.mkdtemp(prefix=f".{manifest.attempt_id}.", dir=str(attempts_path)))
    try:
        if temp_path.resolve(strict=True).parent != attempts_path:
            raise ValueError("temporary attempt path must remain inside attempts directory")
        for name, data in artifacts.items():
            artifact_path = temp_path / name
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_bytes(data)
        _parse_attempt(_read_attempt_directory(temp_path), manifest.attempt_id, temp_path)
        if destination.exists() or destination.is_symlink():
            raise ValueError(f"demo attempt already exists: {manifest.attempt_id}")
        os.rename(temp_path, destination)
        temp_path = Path()
    except Exception:
        if temp_path != Path() and temp_path.exists():
            shutil.rmtree(temp_path)
        raise
    return load_demo_attempt(root_path, manifest.attempt_id)


def load_demo_attempt(
    root: str | os.PathLike[str], attempt_id: str
) -> LoadedDemoAttempt:
    root_path = _root_path(root, create=False)
    attempt_path = _attempt_path(root_path, attempt_id, create_parent=False)
    if not attempt_path.exists():
        raise ValueError(f"demo attempt does not exist: {attempt_id}")
    if attempt_path.parent != _attempts_path(root_path, create=False):
        raise ValueError("attempt path must remain inside attempts directory")
    return _parse_attempt(_read_attempt_directory(attempt_path), attempt_id, attempt_path)


def _parse_published_demo(root: Path, data: bytes, path: Path) -> PublishedDemoAttempt:
    try:
        _decode_canonical_json(data, _PUBLISHED_FILE_NAME)
        manifest = PublishedDemoManifest.model_validate_json(data, strict=True)
    except (TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("invalid JSON", "artifact is")):
            raise
        raise ValueError("invalid published demo manifest") from exc

    _validate_attempt_id(manifest.attempt_id)
    attempt_path = _attempt_path(root, manifest.attempt_id, create_parent=False)
    if (attempt_path / "selection.json").is_file():
        attempt = load_final_demo_attempt(root, manifest.attempt_id)
    else:
        attempt = load_demo_attempt(root, manifest.attempt_id)
    if manifest.run_id != attempt.manifest.run_id:
        raise ValueError("published run_id does not match target attempt")
    if manifest.attempt_manifest_sha256 != sha256_bytes(
        canonical_json_bytes(attempt.manifest)
    ):
        raise ValueError("published attempt manifest SHA-256 does not match target attempt")
    attempt_checkpoint_ids = (
        attempt.manifest.checkpoint_ids
        if attempt.manifest.checkpoint_ids is not None
        else list(_CHECKPOINT_IDS)
    )
    if manifest.checkpoint_ids != attempt_checkpoint_ids:
        raise ValueError("published checkpoint IDs do not match target attempt")
    return PublishedDemoAttempt(manifest=manifest, attempt=attempt, path=path)


def publish_demo_attempt(
    root: str | os.PathLike[str], attempt_id: str
) -> PublishedDemoAttempt:
    """Atomically replace only the published pointer; attempts remain immutable."""
    root_path = _root_path(root, create=False)
    _validate_attempt_id(attempt_id)
    attempt = (
        load_final_demo_attempt(root_path, attempt_id)
        if (root_path / "attempts" / attempt_id / "selection.json").is_file()
        else load_demo_attempt(root_path, attempt_id)
    )
    checkpoint_ids = list(_CHECKPOINT_IDS)
    manifest = PublishedDemoManifest(
        run_id=attempt.manifest.run_id,
        attempt_id=attempt.manifest.attempt_id,
        attempt_manifest_sha256=sha256_bytes(canonical_json_bytes(attempt.manifest)),
        checkpoint_ids=checkpoint_ids,
    )
    data = canonical_json_bytes(manifest)
    _parse_published_demo(root_path, data, root_path / _PUBLISHED_FILE_NAME)

    destination = root_path / _PUBLISHED_FILE_NAME
    if destination.is_symlink():
        raise ValueError("published demo manifest must not be a symlink")
    temp_file = tempfile.NamedTemporaryFile(
        mode="xb", prefix=".published.", suffix=".json", dir=root_path, delete=False
    )
    temp_path = Path(temp_file.name)
    try:
        with temp_file:
            temp_file.write(data)
        os.replace(temp_path, destination)
        temp_path = Path()
    except Exception:
        if temp_path != Path() and temp_path.exists():
            temp_path.unlink()
        raise
    return load_published_demo(root_path)


def load_published_demo(root: str | os.PathLike[str]) -> PublishedDemoAttempt:
    """Load only published.json; this deliberately never falls back to demo_v1."""
    root_path = _root_path(root, create=False)
    published_path = root_path / _PUBLISHED_FILE_NAME
    if not published_path.exists():
        raise ValueError("published demo manifest does not exist")
    if published_path.is_symlink() or not published_path.is_file():
        raise ValueError("published demo manifest must be a regular file")
    if published_path.resolve(strict=True).parent != root_path:
        raise ValueError("published demo manifest must remain inside demo package root")
    return _parse_published_demo(root_path, published_path.read_bytes(), published_path)


# --- Demo v2 final package -------------------------------------------------

def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")


def _safe_file(path: Path, parent: Path, label: str) -> None:
    _regular_file(path, label)
    if path.resolve(strict=True).parent != parent.resolve(strict=True):
        raise ValueError(f"{label} escapes its directory")


def _decode_final(data: bytes, model: type[Any], name: str) -> Any:
    _decode_canonical_json(data, name)
    try:
        value = model.model_validate_json(data, strict=True)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"invalid final demo artifact: {name}") from exc
    if canonical_json_bytes(value) != data:
        raise ValueError(f"artifact is not canonical JSON: {name}")
    return value


def _final_paths(directory: Path) -> dict[str, Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("final demo attempt must be a real directory")
    entries = {entry.name: entry for entry in directory.iterdir()}
    if set(entries) != _FINAL_ROOT_FILES:
        raise ValueError("final demo artifact set is not exact")
    for name, entry in entries.items():
        if name in {"blind", "checkpoints"}:
            if entry.is_symlink() or not entry.is_dir() or entry.resolve(strict=True).parent != directory.resolve(strict=True):
                raise ValueError(f"{name} must be a contained real directory")
        else:
            _safe_file(entry, directory, name)

    checkpoints = entries["checkpoints"]
    expected = {f"checkpoint-{index:02d}.json" for index in range(5)}
    cp_entries = {entry.name: entry for entry in checkpoints.iterdir()}
    if set(cp_entries) != expected:
        raise ValueError("checkpoint artifact set is not exact")
    for name, path in cp_entries.items():
        _safe_file(path, checkpoints, f"checkpoint {name}")

    blind = entries["blind"]
    blind_entries = {entry.name: entry for entry in blind.iterdir()}
    if set(blind_entries) != {"source_manifest.json", "sources", "blind_evidence.json"}:
        raise ValueError("blind artifact set is not exact")
    for name in ("source_manifest.json", "blind_evidence.json"):
        _safe_file(blind_entries[name], blind, f"blind {name}")
    sources = blind_entries["sources"]
    if sources.is_symlink() or not sources.is_dir() or sources.resolve(strict=True).parent != blind.resolve(strict=True):
        raise ValueError("blind sources must be a contained real directory")

    return {
        "manifest.json": entries["manifest.json"],
        "foundation.json": entries["foundation.json"],
        "selection.json": entries["selection.json"],
        "original_ai_snapshot.json": entries["original_ai_snapshot.json"],
        "specificity_session.json": entries["specificity_session.json"],
        "blind/source_manifest.json": blind_entries["source_manifest.json"],
        "blind/blind_evidence.json": blind_entries["blind_evidence.json"],
        "final_selection.json": entries["final_selection.json"],
        "display_events.json": entries["display_events.json"],
        "feishu_sync.json": entries["feishu_sync.json"],
        **{f"checkpoints/{name}": path for name, path in cp_entries.items()},
    }


def _display_events_from_json(data: bytes, run_id: str) -> tuple[DecisionDisplayEvent, ...]:
    value = _decode_canonical_json(data, "display_events.json")
    if not isinstance(value, list):
        raise ValueError("display_events.json must be a JSON array")
    events = tuple(_decode_final(canonical_json_bytes(item), DecisionDisplayEvent, "display event") for item in value)
    if any(event.run_id != run_id or event.sequence != index for index, event in enumerate(events, start=1)):
        raise ValueError("display events run_id or sequence mismatch")
    return events


def _specificity_manifest_values(session: SpecificitySession) -> tuple[str, str]:
    audit_prompt_hashes = [
        value for key, value in session.prompt_hashes.items() if key.startswith("audit:")
    ]
    audit_schema_hash = session.schema_hashes.get("SpecificityAuditBatch")
    if len(audit_prompt_hashes) != 1 or audit_schema_hash is None:
        raise ValueError("specificity session lacks the frozen audit prompt/schema hashes")
    return audit_prompt_hashes[0], audit_schema_hash


def _parse_final_attempt(directory: Path, attempt_id: str) -> LoadedDemoAttempt:
    paths = _final_paths(directory)
    raw = {name: path.read_bytes() for name, path in paths.items()}
    manifest = _decode_final(raw["manifest.json"], DemoAttemptManifest, "manifest.json")
    if manifest.attempt_id != attempt_id or manifest.artifact_sha256 is None:
        raise ValueError("final manifest must contain artifact_sha256")
    if manifest.pending_candidate_ids is None:
        raise ValueError("final manifest must contain pending_candidate_ids")
    if manifest.run_manifest_sha256 is None or manifest.model_profile is None or manifest.stage_sha256 is None:
        raise ValueError("final manifest lacks run provenance")
    if (
        manifest.specificity_model is None
        or manifest.specificity_session_sha256 is None
        or manifest.specificity_prompt_sha256 is None
        or manifest.specificity_schema_sha256 is None
    ):
        raise ValueError("final manifest lacks complete specificity provenance")
    foundation = _decode_final(raw["foundation.json"], DecisionFoundationState, "foundation.json")
    selection = _decode_final(raw["selection.json"], CandidateSelection, "selection.json")
    original = _decode_final(raw["original_ai_snapshot.json"], DecisionOriginalSnapshot, "original_ai_snapshot.json")
    specificity_session = _decode_final(
        raw["specificity_session.json"], SpecificitySession, "specificity_session.json"
    )
    source_manifest = _decode_final(raw["blind/source_manifest.json"], BlindSourceManifest, "blind/source_manifest.json")
    blind = _decode_final(raw["blind/blind_evidence.json"], BlindEvidencePackage, "blind/blind_evidence.json")
    checkpoints = tuple(
        _decode_final(raw[f"checkpoints/checkpoint-{index:02d}.json"], DecisionEvolutionCheckpoint, f"checkpoint-{index:02d}.json")
        for index in range(5)
    )
    final_selection = _decode_final(raw["final_selection.json"], FinalCandidateSelection, "final_selection.json")
    display_events = _display_events_from_json(raw["display_events.json"], manifest.run_id)
    sync = _decode_final(raw["feishu_sync.json"], DemoFeishuSyncArtifact, "feishu_sync.json")

    if specificity_session.run_id != manifest.run_id:
        raise ValueError("specificity session run_id does not match final manifest")
    if specificity_session.specificity_model != LUNA_SPECIFICITY_PROFILE:
        raise ValueError("specificity session is not the exact Luna Max profile")
    if manifest.specificity_model != LUNA_SPECIFICITY_PROFILE:
        raise ValueError("final manifest specificity_model is not the exact Luna Max profile")
    session_hash = sha256_bytes(raw["specificity_session.json"])
    if manifest.specificity_session_sha256 != session_hash:
        raise ValueError("specificity session SHA-256 does not match final manifest")
    if original.specificity_session_sha256 != session_hash:
        raise ValueError("original snapshot specificity session SHA-256 does not match final session")
    prompt_hash, schema_hash = _specificity_manifest_values(specificity_session)
    if manifest.specificity_prompt_sha256 != prompt_hash:
        raise ValueError("specificity prompt SHA-256 does not match frozen session")
    if manifest.specificity_schema_sha256 != schema_hash:
        raise ValueError("specificity schema SHA-256 does not match frozen session")

    source_paths = {entry.name: entry for entry in (directory / "blind" / "sources").iterdir()}
    expected = set(_FINAL_ARTIFACT_FILES) | {f"blind/sources/{name}" for name in source_manifest.artifact_sha256}
    if set(manifest.artifact_sha256) != expected:
        raise ValueError("manifest artifact_sha256 keys do not match final file set")
    for name, digest in manifest.artifact_sha256.items():
        if name.startswith("blind/sources/"):
            source_name = name.removeprefix("blind/sources/")
            if source_name not in source_paths:
                raise ValueError(f"missing blind source: {source_name}")
            data = source_paths[source_name].read_bytes()
        else:
            data = raw[name]
        if sha256_bytes(data) != digest:
            raise ValueError(f"final artifact SHA-256 does not match: {name}")
    if set(source_paths) != set(source_manifest.artifact_sha256):
        raise ValueError("blind source files do not match source_manifest")
    for name, path in source_paths.items():
        _safe_file(path, directory / "blind" / "sources", f"blind source {name}")
    if any(item.run_id != manifest.run_id for item in (foundation, selection, original, specificity_session, blind, *checkpoints, final_selection, sync)):
        raise ValueError("final artifact run_id mismatch")
    if source_manifest.run_id != manifest.run_id or source_manifest.original_snapshot_sha256 != sha256_bytes(raw["original_ai_snapshot.json"]):
        raise ValueError("blind source manifest binding mismatch")
    if blind.run_id != manifest.run_id or blind.source_manifest_sha256 != sha256_bytes(raw["blind/source_manifest.json"]):
        raise ValueError("blind evidence binding mismatch")
    foundation_hash = sha256_bytes(raw["foundation.json"])
    selection_hash = sha256_bytes(raw["selection.json"])
    original_hash = sha256_bytes(raw["original_ai_snapshot.json"])
    foundation_candidate_ids = [
        candidate.candidate_id for candidate in foundation.candidates
    ]
    ranking_candidate_ids = [
        item.candidate.candidate_id for item in foundation.ranking.ranked_candidates
    ]
    if ranking_candidate_ids != foundation_candidate_ids:
        raise ValueError("foundation candidate order does not match ranking")
    if manifest.pending_candidate_ids != foundation_candidate_ids:
        raise ValueError("pending candidate IDs do not match foundation")
    if not set(selection.selected_candidate_ids).issubset(
        set(manifest.pending_candidate_ids)
    ):
        raise ValueError("selection candidate IDs are not a subset of pending candidates")
    if original.foundation_sha256 != foundation_hash or original.selection_sha256 != selection_hash:
        raise ValueError("original snapshot provenance does not match foundation/selection")
    foundation_ids = {candidate.candidate_id for candidate in foundation.candidates}
    selection_ids = set(selection.selected_candidate_ids)
    original_ids = {
        item.ranked_narrative.candidate.candidate_id for item in original.candidates
    }
    if not selection_ids.issubset(foundation_ids) or selection.primary_candidate_id not in selection_ids:
        raise ValueError("selection candidate IDs are not bound to foundation")
    if original_ids != selection_ids or original.primary_candidate_id != selection.primary_candidate_id:
        raise ValueError("original snapshot candidates do not match selection")
    if source_manifest.original_snapshot_sha256 != original_hash or blind.original_snapshot_sha256 != original_hash:
        raise ValueError("blind artifacts are not bound to original snapshot")
    if final_selection.checkpoint_04_sha256 != sha256_bytes(raw["checkpoints/checkpoint-04.json"]):
        raise ValueError("final selection is not bound to checkpoint-04")
    final_shortlist_ids = set(final_selection.shortlisted_candidate_ids)
    checkpoint_04_recommended_ids: list[str] = []
    for index, checkpoint in enumerate(checkpoints):
        expected_id = f"checkpoint-{index:02d}"
        if checkpoint.checkpoint_id != expected_id or checkpoint.release_index != index:
            raise ValueError("checkpoint metadata does not match its artifact name")
        if index >= 1:
            checkpoint_ids = {
                item.ranked_narrative.candidate.candidate_id
                for item in checkpoint.candidates
            }
            if checkpoint_ids != final_shortlist_ids:
                raise ValueError("checkpoint candidate IDs do not match final shortlist")
        if index == 4:
            checkpoint_04_recommended_ids = [
                item.ranked_narrative.candidate.candidate_id
                for item in checkpoint.candidates
                if item.ranked_narrative.is_recommended
            ]
    if len(checkpoint_04_recommended_ids) != 1:
        raise ValueError("checkpoint-04 must contain exactly one AI recommended candidate")
    if final_selection.recommended_candidate_id != checkpoint_04_recommended_ids[0]:
        raise ValueError("final selection recommendation does not match checkpoint-04 AI recommendation")
    if not {
        final_selection.selected_candidate_id,
        final_selection.recommended_candidate_id,
    }.issubset(final_shortlist_ids):
        raise ValueError("final selection candidates are not from final shortlist")
    expected_stage_hashes = {
        "foundation": sha256_bytes(raw["foundation.json"]),
        "pending_selection": manifest.stage_sha256.get("pending_selection"),
        "specificity_session": session_hash,
        "selection": sha256_bytes(raw["selection.json"]),
        "original_snapshot": sha256_bytes(raw["original_ai_snapshot.json"]),
        "blind_evidence": sha256_bytes(raw["blind/blind_evidence.json"]),
        **{f"checkpoint-{index:02d}": sha256_bytes(raw[f"checkpoints/checkpoint-{index:02d}.json"]) for index in range(5)},
        "final_selection": sha256_bytes(raw["final_selection.json"]),
    }
    if selection.pending_selection_sha256 != manifest.stage_sha256.get("pending_selection"):
        raise ValueError("selection is not bound to pending selection provenance")
    if manifest.stage_sha256.get("pending_selection") is None or manifest.stage_sha256 != expected_stage_hashes:
        raise ValueError("manifest stage_sha256 does not match final artifacts")
    if sync.event_count != len(display_events) or sync.last_sequence != (display_events[-1].sequence if display_events else 0):
        raise ValueError("feishu sync does not match display events")
    return LoadedDemoAttempt(manifest, foundation, checkpoints, directory, selection, original, final_selection, display_events, sync)


def load_final_demo_attempt(root: str | os.PathLike[str], attempt_id: str) -> LoadedDemoAttempt:
    root_path = _root_path(root, create=False)
    attempt_path = _attempt_path(root_path, attempt_id, create_parent=False)
    if not attempt_path.exists():
        raise ValueError(f"demo attempt does not exist: {attempt_id}")
    return _parse_final_attempt(attempt_path, attempt_id)


def _run_display_events(run_root: Path, run_id: str) -> tuple[DecisionDisplayEvent, ...]:
    path = run_root / "display" / "events.jsonl"
    _regular_file(path, "run display events")
    events: list[DecisionDisplayEvent] = []
    for line in path.read_bytes().splitlines():
        event = _decode_final(line, DecisionDisplayEvent, "run display event")
        if event.run_id != run_id or event.sequence != len(events) + 1:
            raise ValueError("run display events are not contiguous or cross-run")
        events.append(event)
    return tuple(events)


def _make_sync(run_id: str, events: tuple[DecisionDisplayEvent, ...]) -> DemoFeishuSyncArtifact:
    latest = events[-1] if events else None
    return DemoFeishuSyncArtifact(
        run_id=run_id,
        event_count=len(events),
        last_sequence=latest.sequence if latest else 0,
        last_snapshot_sha256=latest.snapshot_sha256 if latest else None,
    )


def write_final_demo_attempt(*, run_root: Path, run_id: str, attempt_id: str, output_root: Path, publish: bool = False) -> LoadedDemoAttempt:
    _validate_attempt_id(attempt_id)
    store = DecisionRunStore.load(Path(run_root), run_id)
    state = _decode_canonical_json((store.root / "coordinator" / "state.json").read_bytes(), "coordinator state")
    if not isinstance(state, dict) or state.get("run_id") != run_id or state.get("stage") != "COMPLETE":
        raise ValueError("only COMPLETE runs can be assembled")
    stages = store.root / "stages"
    foundation_path = (
        stages / "specificity_final_foundation.json"
        if (stages / "specificity_final_foundation.json").is_file()
        else stages / "foundation.json"
    )
    foundation = _decode_final(foundation_path.read_bytes(), DecisionFoundationState, "foundation")
    base_foundation = _decode_final((stages / "foundation.json").read_bytes(), DecisionFoundationState, "foundation")
    pending = _decode_final((stages / "pending_selection.json").read_bytes(), PendingSelectionPackage, "pending_selection")
    selection = _decode_final((stages / "selection.json").read_bytes(), CandidateSelection, "selection")
    original = _decode_final((stages / "original_snapshot.json").read_bytes(), DecisionOriginalSnapshot, "original_snapshot")
    specificity_session_path = stages / "specificity_session.json"
    _regular_file(specificity_session_path, "specificity_session")
    specificity_session = _decode_final(
        specificity_session_path.read_bytes(), SpecificitySession, "specificity_session"
    )
    if specificity_session.run_id != run_id or specificity_session.specificity_model != LUNA_SPECIFICITY_PROFILE:
        raise ValueError("frozen specificity session is not bound to this run or Luna profile")
    specificity_session_sha256 = sha256_bytes(
        canonical_json_bytes(specificity_session)
    )
    if state.get("specificity_session_sha256") != specificity_session_sha256:
        raise ValueError("coordinator specificity session hash does not match frozen session")
    if original.specificity_session_sha256 != specificity_session_sha256:
        raise ValueError("original snapshot specificity session hash does not match frozen session")
    specificity_prompt_sha256, specificity_schema_sha256 = _specificity_manifest_values(
        specificity_session
    )
    source_manifest = _decode_final((stages / "blind" / "source_manifest.json").read_bytes(), BlindSourceManifest, "source_manifest")
    blind = _decode_final((stages / "blind" / "blind_evidence.json").read_bytes(), BlindEvidencePackage, "blind_evidence")
    checkpoints = tuple(_decode_final((stages / f"checkpoint_{index:02d}.json").read_bytes(), DecisionEvolutionCheckpoint, "checkpoint") for index in range(5))
    final_selection = _decode_final((stages / "final_selection.json").read_bytes(), FinalCandidateSelection, "final_selection")
    events = _run_display_events(store.root, run_id)
    sync = _make_sync(run_id, events)
    files: dict[str, bytes] = {
        "foundation.json": canonical_json_bytes(foundation),
        "selection.json": canonical_json_bytes(selection),
        "original_ai_snapshot.json": canonical_json_bytes(original),
        "specificity_session.json": canonical_json_bytes(specificity_session),
        "blind/source_manifest.json": canonical_json_bytes(source_manifest),
        "blind/blind_evidence.json": canonical_json_bytes(blind),
        **{f"checkpoints/checkpoint-{index:02d}.json": canonical_json_bytes(value) for index, value in enumerate(checkpoints)},
        "final_selection.json": canonical_json_bytes(final_selection),
        "display_events.json": canonical_json_bytes(list(events)),
        "feishu_sync.json": canonical_json_bytes(sync),
    }
    coordinator_stage_hashes = {
        "foundation": sha256_bytes(files["foundation.json"]),
        "pending_selection": sha256_bytes(canonical_json_bytes(pending)),
        "specificity_session": sha256_bytes(files["specificity_session.json"]),
        "selection": sha256_bytes(files["selection.json"]),
        "original_snapshot": sha256_bytes(files["original_ai_snapshot.json"]),
        "blind_evidence": sha256_bytes(files["blind/blind_evidence.json"]),
        **{f"checkpoint-{index:02d}": sha256_bytes(files[f"checkpoints/checkpoint-{index:02d}.json"]) for index in range(5)},
        "final_selection": sha256_bytes(files["final_selection.json"]),
    }
    foundation_candidate_ids = [
        candidate.candidate_id for candidate in foundation.candidates
    ]
    ranking_candidate_ids = [
        item.candidate.candidate_id for item in foundation.ranking.ranked_candidates
    ]
    pending_candidate_ids = [
        item.ranked_narrative.candidate.candidate_id for item in pending.candidates
    ]
    if ranking_candidate_ids != foundation_candidate_ids:
        raise ValueError("foundation candidate order does not match ranking")
    if pending_candidate_ids != foundation_candidate_ids:
        raise ValueError("pending candidate IDs do not match foundation")
    if pending.foundation_sha256 != sha256_bytes(canonical_json_bytes(base_foundation)):
        raise ValueError("pending selection is not bound to the original foundation")
    if not set(selection.selected_candidate_ids).issubset(set(pending_candidate_ids)):
        raise ValueError("selection candidate IDs are not a subset of pending candidates")
    final_shortlist_ids = set(final_selection.shortlisted_candidate_ids)
    checkpoint_04_recommended_ids: list[str] = []
    for index, checkpoint in enumerate(checkpoints):
        expected_id = f"checkpoint-{index:02d}"
        if checkpoint.checkpoint_id != expected_id or checkpoint.release_index != index:
            raise ValueError("checkpoint metadata does not match its artifact name")
        checkpoint_ids = {
            item.ranked_narrative.candidate.candidate_id
            for item in checkpoint.candidates
        }
        if index >= 1 and checkpoint_ids != final_shortlist_ids:
            raise ValueError("checkpoint candidate IDs do not match final shortlist")
        if index == 4:
            checkpoint_04_recommended_ids = [
                item.ranked_narrative.candidate.candidate_id
                for item in checkpoint.candidates
                if item.ranked_narrative.is_recommended
            ]
    if len(checkpoint_04_recommended_ids) != 1:
        raise ValueError("checkpoint-04 must contain exactly one AI recommended candidate")
    if final_selection.recommended_candidate_id != checkpoint_04_recommended_ids[0]:
        raise ValueError("final selection recommendation does not match checkpoint-04 AI recommendation")
    if not {
        final_selection.selected_candidate_id,
        final_selection.recommended_candidate_id,
    }.issubset(final_shortlist_ids):
        raise ValueError("final selection candidates are not from final shortlist")
    if pending.foundation_sha256 != sha256_bytes(canonical_json_bytes(base_foundation)):
        raise ValueError("pending selection is not bound to foundation")
    if pending.recommended_candidate_id != base_foundation.ranking.recommended_candidate_id:
        raise ValueError("pending selection recommendation does not match the original foundation")
    if selection.pending_selection_sha256 != coordinator_stage_hashes["pending_selection"]:
        raise ValueError("selection is not bound to pending selection")
    coordinator_values = {
        "foundation": state.get("foundation_sha256"),
        "pending_selection": state.get("pending_selection_sha256"),
        "selection": state.get("selection_sha256"),
        "original_snapshot": state.get("original_snapshot_sha256"),
        "blind_evidence": state.get("blind_evidence_sha256"),
        "final_selection": state.get("final_selection_sha256"),
        "specificity_session": state.get("specificity_session_sha256"),
    }
    coordinator_values.update(state.get("checkpoint_sha256") or {})
    if coordinator_values != coordinator_stage_hashes:
        raise ValueError("coordinator stage hashes do not match frozen stages")
    source_dir = stages / "blind" / "sources"
    for name in source_manifest.artifact_sha256:
        source_path = source_dir / name
        _safe_file(source_path, source_dir, f"blind source {name}")
        files[f"blind/sources/{name}"] = source_path.read_bytes()
    stage_hashes = {
        "foundation": sha256_bytes(files["foundation.json"]),
        "pending_selection": sha256_bytes(canonical_json_bytes(pending)),
        "specificity_session": sha256_bytes(files["specificity_session.json"]),
        "selection": sha256_bytes(files["selection.json"]),
        "original_snapshot": sha256_bytes(files["original_ai_snapshot.json"]),
        "blind_evidence": sha256_bytes(files["blind/blind_evidence.json"]),
        **{f"checkpoint-{index:02d}": sha256_bytes(files[f"checkpoints/checkpoint-{index:02d}.json"]) for index in range(5)},
        "final_selection": sha256_bytes(files["final_selection.json"]),
    }
    manifest = DemoAttemptManifest(
        run_id=run_id,
        attempt_id=attempt_id,
        run_manifest_sha256=sha256_bytes((store.root / "run_manifest.json").read_bytes()),
        model_profile=store.manifest.decision_model,
        specificity_model=specificity_session.specificity_model,
        specificity_session_sha256=specificity_session_sha256,
        specificity_prompt_sha256=specificity_prompt_sha256,
        specificity_schema_sha256=specificity_schema_sha256,
        pending_candidate_ids=[
            item.ranked_narrative.candidate.candidate_id
            for item in pending.candidates
        ],
        stage_sha256=stage_hashes,
        artifact_sha256={name: sha256_bytes(data) for name, data in files.items()},
    )
    root = _root_path(output_root, create=True)
    destination = _attempt_path(root, attempt_id, create_parent=True)
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"demo attempt already exists: {attempt_id}")
    temp = Path(tempfile.mkdtemp(prefix=f".{attempt_id}.", dir=str(destination.parent)))
    try:
        for name, data in files.items():
            path = temp / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        (temp / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        _parse_final_attempt(temp, attempt_id)
        os.rename(temp, destination)
        temp = Path()
    except Exception:
        if temp != Path() and temp.exists():
            shutil.rmtree(temp)
        raise
    result = load_final_demo_attempt(root, attempt_id)
    if publish:
        publish_demo_attempt(root, attempt_id)
    return result
