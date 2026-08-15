from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel, ValidationError

from src.schemas import (
    BlindEvidencePackage,
    BlindSourceManifest,
    DecisionCallRequestManifest,
    DecisionCallResponseManifest,
    DecisionDisplayEvent,
    DecisionRunManifest,
    DecisionRunStage,
)
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes


T = TypeVar("T", bound=BaseModel)

_STAGE_NAMES = frozenset(
    {
        "pending_selection",
        "foundation",
        "selection",
        "original_snapshot",
        "checkpoint_00",
        "checkpoint_01",
        "checkpoint_02",
        "checkpoint_03",
        "checkpoint_04",
        "final_selection",
        "specificity_session",
        "specificity_final_foundation",
        "specificity_task_round_01",
        "specificity_task_round_02",
        "specificity_task_round_03",
    }
)
_CALL_DIR_RE = re.compile(r"^[1-9][0-9]*$")
_FAILURE_FILE_RE = re.compile(r"^[1-9][0-9]*\.json$")
_RUN_ID_RE = re.compile(r"^[^/\\]+$")


@dataclass(frozen=True, slots=True)
class FrozenArtifact:
    name: str
    manifest: BaseModel
    path: Path


@dataclass(frozen=True, slots=True)
class FrozenCall:
    call_id: str
    manifest: DecisionCallRequestManifest
    path: Path
    response: FrozenArtifact | None = None


def _validate_run_id(run_id: str) -> None:
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or not _RUN_ID_RE.fullmatch(run_id)
    ):
        raise ValueError("run_id must be a single path component")


def _validate_stage_name(name: str) -> None:
    if not isinstance(name, str) or name not in _STAGE_NAMES:
        raise ValueError(f"invalid stage name: {name!r}")


def _regular_file(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")


def _directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory")


def _decode_canonical(data: bytes, label: str) -> Any:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {label}") from exc
    if canonical_json_bytes(value) != data:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _validate_model_bytes(data: bytes, model: type[T], label: str) -> T:
    _decode_canonical(data, label)
    try:
        result = model.model_validate_json(data, strict=True)
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if canonical_json_bytes(result) != data:
        raise ValueError(f"{label} is not canonical JSON")
    return result


def _write_new_file(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"{path.name} already exists")
    temp_path: Path | None = None
    try:
        fd, temp_name = __import__("tempfile").mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _replace_file(path: Path, data: bytes) -> None:
    temp_path: Path | None = None
    try:
        fd, temp_name = __import__("tempfile").mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _append_line(path: Path, data: bytes) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError("display events file must be a regular file")
    with path.open("ab") as handle:
        handle.write(data + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_blind_source_name(name: str, sources_path: Path) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
    ):
        raise ValueError(f"blind source path must be relative and non-escaping: {name!r}")
    relative = Path(name)
    if relative.is_absolute() or len(relative.parts) != 1:
        raise ValueError(f"blind source path must be relative and non-escaping: {name!r}")
    source_path = sources_path / relative
    try:
        source_path.resolve().relative_to(sources_path.resolve())
    except ValueError as exc:
        raise ValueError(f"blind source path escapes sources: {name!r}") from exc
    return source_path


def _validate_blind_stage(
    blind_path: Path,
    run_id: str,
    original_snapshot_path: Path,
) -> None:
    _directory(blind_path, "blind stage")
    expected_entries = {"source_manifest.json", "blind_evidence.json", "sources"}
    entries = list(blind_path.iterdir())
    names = {entry.name for entry in entries}
    if names != expected_entries:
        missing = expected_entries - names
        extra = names - expected_entries
        if extra:
            raise ValueError(f"extra blind entries: {sorted(extra)}")
        raise ValueError(f"missing blind entries: {sorted(missing)}")
    for entry in entries:
        if entry.is_symlink():
            raise ValueError(f"blind entry {entry.name} must not be a symlink")

    source_manifest_path = blind_path / "source_manifest.json"
    package_path = blind_path / "blind_evidence.json"
    sources_path = blind_path / "sources"
    _regular_file(source_manifest_path, "blind source manifest")
    _regular_file(package_path, "blind evidence package")
    _directory(sources_path, "blind sources")
    source_manifest_bytes = source_manifest_path.read_bytes()
    package_bytes = package_path.read_bytes()
    source_manifest = _validate_model_bytes(
        source_manifest_bytes,
        BlindSourceManifest,
        "blind source manifest",
    )
    package = _validate_model_bytes(
        package_bytes,
        BlindEvidencePackage,
        "blind evidence package",
    )
    if source_manifest.run_id != run_id or package.run_id != run_id:
        raise ValueError("blind evidence run_id does not match run")

    _regular_file(original_snapshot_path, "original snapshot")
    original_snapshot_bytes = original_snapshot_path.read_bytes()
    _decode_canonical(original_snapshot_bytes, "original snapshot")
    original_snapshot_sha256 = sha256_bytes(original_snapshot_bytes)
    if source_manifest.original_snapshot_sha256 != original_snapshot_sha256:
        raise ValueError("blind source manifest original snapshot SHA does not match")
    if package.original_snapshot_sha256 != original_snapshot_sha256:
        raise ValueError("blind evidence original snapshot SHA does not match")
    if package.source_manifest_sha256 != sha256_bytes(source_manifest_bytes):
        raise ValueError("blind evidence source manifest SHA does not match")

    source_entries = list(sources_path.iterdir())
    source_names: set[str] = set()
    for source_entry in source_entries:
        if source_entry.is_symlink():
            raise ValueError(f"blind source {source_entry.name} must not be a symlink")
        if not source_entry.is_file():
            raise ValueError(f"blind source {source_entry.name} must be a regular file")
        _validate_blind_source_name(source_entry.name, sources_path)
        source_names.add(source_entry.name)

    manifest_names = set(source_manifest.artifact_sha256)
    for source_name in manifest_names:
        source_path = _validate_blind_source_name(source_name, sources_path)
        _regular_file(source_path, f"blind source {source_name}")
        actual_sha256 = sha256_bytes(source_path.read_bytes())
        if actual_sha256 != source_manifest.artifact_sha256[source_name]:
            raise ValueError(f"blind source SHA does not match: {source_name}")
    if source_names != manifest_names:
        missing = manifest_names - source_names
        extra = source_names - manifest_names
        if extra:
            raise ValueError(f"extra blind sources: {sorted(extra)}")
        raise ValueError(f"missing blind sources: {sorted(missing)}")

    for item in package.items:
        for source_name in item.source_artifacts:
            if source_name not in manifest_names:
                raise ValueError(
                    f"blind evidence source reference is not registered: {source_name}"
                )


def _validate_run_workspace(run_root: Path) -> None:
    coordinator_path = run_root / "coordinator"
    failures_path = run_root / "failures"
    _directory(coordinator_path, "coordinator")
    _directory(failures_path, "failures")

    allowed_coordinator_files = {"state.json", "runtime.json", "inputs.json"}
    coordinator_entries = list(coordinator_path.iterdir())
    for entry in coordinator_entries:
        if entry.is_symlink():
            raise ValueError(f"coordinator entry {entry.name} must not be a symlink")
        if entry.name not in allowed_coordinator_files or not entry.is_file():
            raise ValueError(f"extra coordinator entry: {entry.name}")
        _decode_canonical(entry.read_bytes(), f"coordinator {entry.name}")

    failure_sequences: list[int] = []
    for entry in failures_path.iterdir():
        if entry.is_symlink():
            raise ValueError(f"failure entry {entry.name} must not be a symlink")
        if not entry.is_file() or not _FAILURE_FILE_RE.fullmatch(entry.name):
            raise ValueError(f"extra failure entry: {entry.name}")
        failure_sequences.append(int(entry.stem))
        _decode_canonical(entry.read_bytes(), f"failure {entry.name}")
    if sorted(failure_sequences) != list(range(1, len(failure_sequences) + 1)):
        raise ValueError("failure sequences must be contiguous")


class DecisionRunStore:
    def __init__(self, root: Path, manifest: DecisionRunManifest) -> None:
        self.root = root
        self.manifest = manifest

    @classmethod
    def create(cls, root: Path, manifest: DecisionRunManifest) -> "DecisionRunStore":
        if not isinstance(manifest, DecisionRunManifest):
            raise TypeError("manifest must be a DecisionRunManifest")
        _validate_run_id(manifest.decision_run_id)
        parent = Path(root)
        if parent.exists() and parent.is_symlink():
            raise ValueError("run parent must not be a symlink")
        parent.mkdir(parents=True, exist_ok=True)
        run_root = parent / manifest.decision_run_id
        if run_root.exists() or run_root.is_symlink():
            raise FileExistsError(f"run {manifest.decision_run_id!r} already exists")
        run_root.mkdir()
        try:
            (run_root / "calls").mkdir()
            (run_root / "stages").mkdir()
            (run_root / "display").mkdir()
            (run_root / "coordinator").mkdir()
            (run_root / "failures").mkdir()
            (run_root / "display" / "events.jsonl").touch()
            _write_new_file(run_root / "run_manifest.json", canonical_json_bytes(manifest))
        except Exception:
            # Creation is intentionally fail-fast.  Do not turn a partial workspace
            # into a usable run, and do not remove a workspace owned by the caller.
            raise
        return cls(run_root, manifest)

    @classmethod
    def load(cls, root: Path, run_id: str) -> "DecisionRunStore":
        _validate_run_id(run_id)
        parent = Path(root)
        if parent.is_symlink():
            raise ValueError("run parent must not be a symlink")
        run_root = parent / run_id
        if not run_root.exists() and not run_root.is_symlink():
            raise FileNotFoundError(run_root)
        _directory(run_root, "run root")
        expected = {
            "run_manifest.json",
            "calls",
            "stages",
            "display",
            "coordinator",
            "failures",
        }
        entries = list(run_root.iterdir())
        names = {entry.name for entry in entries}
        if names != expected:
            missing = expected - names
            extra = names - expected
            if extra:
                raise ValueError(f"extra run entries: {sorted(extra)}")
            raise ValueError(f"missing run entries: {sorted(missing)}")
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"run entry {entry.name} must not be a symlink")

        manifest_path = run_root / "run_manifest.json"
        _regular_file(manifest_path, "run_manifest.json")
        manifest = _validate_model_bytes(
            manifest_path.read_bytes(), DecisionRunManifest, "run_manifest.json"
        )
        if manifest.decision_run_id != run_id:
            raise ValueError("run manifest id does not match run path")

        calls_path = run_root / "calls"
        stages_path = run_root / "stages"
        display_path = run_root / "display"
        _directory(calls_path, "calls")
        _directory(stages_path, "stages")
        _directory(display_path, "display")
        cls._validate_calls(calls_path, manifest.decision_run_id)
        cls._validate_stages(stages_path, manifest.decision_run_id)
        cls._validate_display(display_path, manifest.decision_run_id)
        _validate_run_workspace(run_root)
        return cls(run_root, manifest)

    @staticmethod
    def _validate_stages(stages_path: Path, run_id: str) -> None:
        for entry in stages_path.iterdir():
            if entry.is_symlink():
                raise ValueError(f"stage entry {entry.name} must not be a symlink")
            if entry.name == "blind":
                _validate_blind_stage(
                    entry,
                    run_id,
                    stages_path / "original_snapshot.json",
                )
                continue
            if not entry.is_file() or entry.suffix != ".json":
                raise ValueError(f"extra stage entry: {entry.name}")
            name = entry.stem
            _validate_stage_name(name)
            _decode_canonical(entry.read_bytes(), f"stage {name}")

    @staticmethod
    def _validate_calls(calls_path: Path, run_id: str) -> None:
        entries = list(calls_path.iterdir())
        sequences: list[int] = []
        call_ids: set[str] = set()
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f"call entry {entry.name} must not be a symlink")
            if not entry.is_dir() or not _CALL_DIR_RE.fullmatch(entry.name):
                raise ValueError(f"extra call entry: {entry.name}")
            sequence = int(entry.name)
            sequences.append(sequence)
            child_names = {child.name for child in entry.iterdir()}
            if child_names - {"request_manifest.json", "response.json"}:
                raise ValueError(f"extra call files: {sorted(child_names - {'request_manifest.json', 'response.json'})}")
            if "request_manifest.json" not in child_names:
                raise ValueError(f"missing request manifest for call {sequence}")
            request_path = entry / "request_manifest.json"
            _regular_file(request_path, "request_manifest.json")
            request = _validate_model_bytes(
                request_path.read_bytes(),
                DecisionCallRequestManifest,
                f"call {sequence} request manifest",
            )
            if request.sequence != sequence:
                raise ValueError("call sequence does not match request manifest")
            if request.call_id in call_ids:
                raise ValueError("call_id must be unique")
            call_ids.add(request.call_id)
            if "response.json" in child_names:
                response_path = entry / "response.json"
                _regular_file(response_path, "response.json")
                response = _validate_model_bytes(
                    response_path.read_bytes(),
                    DecisionCallResponseManifest,
                    f"call {sequence} response",
                )
                _validate_response_binding(request, response)
        if sorted(sequences) != list(range(1, len(sequences) + 1)):
            raise ValueError("call sequences must be contiguous")

    @staticmethod
    def _validate_display(display_path: Path, run_id: str) -> None:
        entries = list(display_path.iterdir())
        if {entry.name for entry in entries} != {"events.jsonl"}:
            names = {entry.name for entry in entries}
            if names - {"events.jsonl"}:
                raise ValueError(f"extra display entries: {sorted(names - {'events.jsonl'})}")
            raise ValueError("missing display entries: ['events.jsonl']")
        event_path = display_path / "events.jsonl"
        _regular_file(event_path, "display events")
        _read_events(event_path, run_id)

    def freeze_stage(self, name: str, value: BaseModel) -> FrozenArtifact:
        _validate_stage_name(name)
        if not isinstance(value, BaseModel):
            raise TypeError("stage value must be a Pydantic BaseModel")
        _directory(self.root / "stages", "stages")
        path = self.root / "stages" / f"{name}.json"
        _write_new_file(path, canonical_json_bytes(value))
        return FrozenArtifact(name=name, manifest=value, path=path)

    def load_stage(self, name: str, model: type[T]) -> T:
        _validate_stage_name(name)
        path = self.root / "stages" / f"{name}.json"
        _regular_file(path, f"stage {name}")
        return _validate_model_bytes(path.read_bytes(), model, f"stage {name}")

    def load_blind_evidence(self) -> BlindEvidencePackage:
        stages_path = self.root / "stages"
        _directory(stages_path, "stages")
        blind_path = stages_path / "blind"
        _validate_blind_stage(
            blind_path,
            self.manifest.decision_run_id,
            stages_path / "original_snapshot.json",
        )
        package_path = blind_path / "blind_evidence.json"
        return _validate_model_bytes(
            package_path.read_bytes(),
            BlindEvidencePackage,
            "blind evidence package",
        )

    def begin_call(self, request: DecisionCallRequestManifest) -> FrozenCall:
        if not isinstance(request, DecisionCallRequestManifest):
            raise TypeError("request must be a DecisionCallRequestManifest")
        calls_path = self.root / "calls"
        _directory(calls_path, "calls")
        self._validate_calls(calls_path, self.manifest.decision_run_id)
        existing = list(calls_path.iterdir())
        for entry in existing:
            if entry.is_symlink():
                raise ValueError("call entry must not be a symlink")
        existing_sequences = [
            int(entry.name)
            for entry in existing
            if entry.is_dir() and _CALL_DIR_RE.fullmatch(entry.name)
        ]
        call_path = calls_path / str(request.sequence)
        if call_path.exists() or call_path.is_symlink():
            raise FileExistsError(f"call sequence {request.sequence} already exists")
        expected_sequence = max(existing_sequences, default=0) + 1
        if request.sequence != expected_sequence:
            raise ValueError(
                f"call sequence must be {expected_sequence}, got {request.sequence}"
            )
        for sequence in existing_sequences:
            request_path = calls_path / str(sequence) / "request_manifest.json"
            if request_path.is_file() and not request_path.is_symlink():
                previous = _validate_model_bytes(
                    request_path.read_bytes(),
                    DecisionCallRequestManifest,
                    f"call {sequence} request manifest",
                )
                if previous.call_id == request.call_id:
                    raise FileExistsError(f"call {request.call_id!r} already exists")
        call_path.mkdir()
        request_path = call_path / "request_manifest.json"
        try:
            _write_new_file(request_path, canonical_json_bytes(request))
        except Exception:
            raise
        return FrozenCall(
            call_id=request.call_id,
            manifest=request,
            path=request_path,
        )

    def begin_offline_specificity_call(
        self,
        *,
        input_artifact_sha256: str,
        prompt_sha256: str,
        system_prompt: str,
        user_prompt: str,
        response_schema_sha256: str,
    ) -> FrozenCall:
        """Register a Luna offline exchange in the same immutable call ledger."""
        calls_path = self.root / "calls"
        _directory(calls_path, "calls")
        existing_sequences = [
            int(entry.name)
            for entry in calls_path.iterdir()
            if entry.is_dir() and _CALL_DIR_RE.fullmatch(entry.name)
        ]
        sequence = max(existing_sequences, default=0) + 1
        request = DecisionCallRequestManifest(
            call_id=f"call-{sequence}",
            sequence=sequence,
            stage=DecisionRunStage.AWAITING_SPECIFICITY_AUDIT,
            input_artifact_sha256=input_artifact_sha256,
            decision_model=self.manifest.specificity_model,
            prompt_name="brand_specificity_audit:codex_offline",
            prompt_version=self.manifest.specificity_model.prompt_version,
            prompt_sha256=prompt_sha256,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema_sha256=response_schema_sha256,
            created_at=datetime.now(timezone.utc),
        )
        return self.begin_call(request)

    def find_offline_specificity_call(self, task_id: str) -> FrozenCall:
        """Find the one ledger call whose offline request names ``task_id``."""
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be non-empty")
        calls_path = self.root / "calls"
        _directory(calls_path, "calls")
        self._validate_calls(calls_path, self.manifest.decision_run_id)
        matches: list[FrozenCall] = []
        for entry in calls_path.iterdir():
            if entry.is_symlink() or not entry.is_dir() or not _CALL_DIR_RE.fullmatch(entry.name):
                continue
            request_path = entry / "request_manifest.json"
            request = _validate_model_bytes(
                request_path.read_bytes(),
                DecisionCallRequestManifest,
                f"call {entry.name} request manifest",
            )
            if request.prompt_name != "brand_specificity_audit:codex_offline":
                continue
            try:
                request_payload = json.loads(request.user_prompt)
            except json.JSONDecodeError:
                continue
            if not isinstance(request_payload, dict) or request_payload.get("task_id") != task_id:
                continue
            matches.append(self.load_call(request.call_id))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one offline specificity call for task {task_id!r}, found {len(matches)}"
            )
        return matches[0]

    def load_call(self, call_id: str) -> FrozenCall:
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("call_id must be non-empty")
        calls_path = self.root / "calls"
        _directory(calls_path, "calls")
        self._validate_calls(calls_path, self.manifest.decision_run_id)
        found: FrozenCall | None = None
        for entry in calls_path.iterdir():
            if entry.is_symlink():
                raise ValueError("call entry must not be a symlink")
            if not entry.is_dir() or not _CALL_DIR_RE.fullmatch(entry.name):
                raise ValueError(f"extra call entry: {entry.name}")
            request_path = entry / "request_manifest.json"
            _regular_file(request_path, "request_manifest.json")
            request = _validate_model_bytes(
                request_path.read_bytes(),
                DecisionCallRequestManifest,
                f"call {entry.name} request manifest",
            )
            if request.call_id != call_id:
                continue
            response_artifact: FrozenArtifact | None = None
            response_path = entry / "response.json"
            if response_path.exists() or response_path.is_symlink():
                _regular_file(response_path, "response.json")
                response = _validate_model_bytes(
                    response_path.read_bytes(),
                    DecisionCallResponseManifest,
                    f"call {entry.name} response",
                )
                _validate_response_binding(request, response)
                response_artifact = FrozenArtifact(
                    name="response",
                    manifest=response,
                    path=response_path,
                )
            found = FrozenCall(
                call_id=request.call_id,
                manifest=request,
                path=request_path,
                response=response_artifact,
            )
        if found is None:
            raise FileNotFoundError(f"call {call_id!r} not found")
        return found

    def finish_call(
        self, call_id: str, response: DecisionCallResponseManifest
    ) -> FrozenCall:
        if not isinstance(response, DecisionCallResponseManifest):
            raise TypeError("response must be a DecisionCallResponseManifest")
        call = self.load_call(call_id)
        if call.response is not None:
            raise FileExistsError(f"response for call {call_id!r} already exists")
        if response.call_id != call_id:
            raise ValueError("response call_id does not match request")
        _validate_response_binding(call.manifest, response)
        response_path = call.path.parent / "response.json"
        _write_new_file(response_path, canonical_json_bytes(response))
        return FrozenCall(
            call_id=call.call_id,
            manifest=call.manifest,
            path=call.path,
            response=FrozenArtifact(
                name="response", manifest=response, path=response_path
            ),
        )

    def append_display_event(self, event: DecisionDisplayEvent) -> None:
        if not isinstance(event, DecisionDisplayEvent):
            raise TypeError("event must be a DecisionDisplayEvent")
        if event.run_id != self.manifest.decision_run_id:
            raise ValueError("display event run_id does not match run")
        event_path = self.root / "display" / "events.jsonl"
        _regular_file(event_path, "display events")
        events = _read_events(event_path, self.manifest.decision_run_id)
        expected = len(events) + 1
        if event.sequence != expected:
            raise ValueError(f"display event sequence must be {expected}")
        _append_line(event_path, canonical_json_bytes(event))

    def latest_display_event(self) -> DecisionDisplayEvent | None:
        event_path = self.root / "display" / "events.jsonl"
        _regular_file(event_path, "display events")
        events = _read_events(event_path, self.manifest.decision_run_id)
        return events[-1] if events else None

    def set_active(self) -> None:
        manifest_path = self.root / "run_manifest.json"
        _regular_file(manifest_path, "run_manifest.json")
        active = {
            "run_id": self.manifest.decision_run_id,
            "run_manifest_sha256": sha256_bytes(
                manifest_path.read_bytes()
            ),
        }
        _replace_file(self.root.parent / "active.json", canonical_json_bytes(active))


def _validate_response_binding(
    request: DecisionCallRequestManifest,
    response: DecisionCallResponseManifest,
) -> None:
    expected_request_sha = sha256_bytes(canonical_json_bytes(request))
    if response.request_sha256 != expected_request_sha:
        raise ValueError("response request SHA does not match request")
    expected_response_sha = sha256_bytes(canonical_json_bytes(response.response_payload))
    if response.response_sha256 != expected_response_sha:
        raise ValueError("response SHA does not match response payload")
    if response.call_id != request.call_id:
        raise ValueError("response call_id does not match request")


def _read_events(path: Path, run_id: str) -> list[DecisionDisplayEvent]:
    data = path.read_bytes()
    if not data:
        return []
    if not data.endswith(b"\n"):
        raise ValueError("display events must be newline-delimited JSON")
    events: list[DecisionDisplayEvent] = []
    for sequence, line in enumerate(data.splitlines(), start=1):
        event = _validate_model_bytes(line, DecisionDisplayEvent, "display event")
        if event.run_id != run_id:
            raise ValueError("display event run_id does not match run")
        if event.sequence != sequence:
            raise ValueError("display event sequences must be contiguous")
        events.append(event)
    return events


class AuditedLLMClient:
    def __init__(
        self,
        client: Any,
        store: DecisionRunStore,
        *,
        input_artifact_sha256: str,
        prompt_name: str,
        prompt_version: str,
        prompt_sha256: str,
    ) -> None:
        self._client = client
        self._store = store
        self._input_artifact_sha256 = input_artifact_sha256
        self._prompt_name = prompt_name
        self._prompt_version = prompt_version
        self._prompt_sha256 = prompt_sha256
        self._active_stage: DecisionRunStage | None = None
        self._retry_call_id: str | None = None

    @contextmanager
    def stage(
        self,
        stage: DecisionRunStage | str,
        retry_call_id: str | None = None,
    ) -> Iterator[None]:
        if self._active_stage is not None:
            raise RuntimeError("an audited stage is already active")
        try:
            selected_stage = DecisionRunStage(stage)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid decision stage: {stage!r}") from exc
        if retry_call_id is not None:
            call = self._store.load_call(retry_call_id)
            if call.response is not None:
                raise ValueError("retry call already has a response")
            if call.manifest.stage is not selected_stage:
                raise ValueError("retry call stage does not match active stage")
        self._active_stage = selected_stage
        self._retry_call_id = retry_call_id
        try:
            yield
        finally:
            self._active_stage = None
            self._retry_call_id = None

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        evidence_catalog: Mapping[str, str] | None = None,
    ) -> T:
        if self._active_stage is None:
            raise RuntimeError("generate_json requires an active audited stage")
        request = self._request_for_call(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
        )
        if self._retry_call_id is not None:
            retry_call = self._store.load_call(self._retry_call_id)
            if _requests_match(retry_call.manifest, request):
                call = retry_call
            else:
                existing = _find_matching_call(self._store, request)
                if existing is None or existing.response is None:
                    raise ValueError(
                        "retry_call_id must identify the current missing response request"
                    )
                return _load_response_model(existing, response_model)
        else:
            existing = _find_matching_call(self._store, request)
            if existing is not None:
                if existing.response is None:
                    raise ValueError(
                        "an existing request is missing a response; explicit retry_call_id is required"
                    )
                return _load_response_model(existing, response_model)
            call = self._store.begin_call(request)

        result = self._client.generate_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_model=response_model,
            evidence_catalog=evidence_catalog,
        )
        if not isinstance(result, BaseModel):
            raise TypeError("LLM client must return a Pydantic model")
        validated = response_model.model_validate_json(
            canonical_json_bytes(result.model_dump(mode="json")),
            strict=True,
        )
        payload = validated.model_dump(mode="json")
        response = DecisionCallResponseManifest(
            call_id=call.call_id,
            request_sha256=sha256_bytes(canonical_json_bytes(call.manifest)),
            response_model_name=response_model.__name__,
            response_payload=payload,
            response_sha256=sha256_bytes(canonical_json_bytes(payload)),
            completed_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
        )
        self._store.finish_call(call.call_id, response)
        if self._retry_call_id == call.call_id:
            self._retry_call_id = None
        return validated

    def _request_for_call(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
    ) -> DecisionCallRequestManifest:
        sequence = _next_call_sequence(self._store.root / "calls")
        call_id = f"call-{sequence}"
        return DecisionCallRequestManifest(
            call_id=call_id,
            sequence=sequence,
            stage=self._active_stage,
            input_artifact_sha256=self._input_artifact_sha256,
            decision_model=self._store.manifest.decision_model,
            prompt_name=self._prompt_name,
            prompt_version=self._prompt_version,
            prompt_sha256=self._prompt_sha256,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema_sha256=sha256_bytes(
                canonical_json_bytes(response_model.model_json_schema())
            ),
            created_at=__import__("datetime").datetime.now(
                __import__("datetime").timezone.utc
            ),
        )


def _next_call_sequence(calls_path: Path) -> int:
    _directory(calls_path, "calls")
    sequences: list[int] = []
    for entry in calls_path.iterdir():
        if entry.is_symlink():
            raise ValueError("call entry must not be a symlink")
        if not entry.is_dir() or not _CALL_DIR_RE.fullmatch(entry.name):
            raise ValueError(f"extra call entry: {entry.name}")
        sequences.append(int(entry.name))
    expected = max(sequences, default=0) + 1
    if sorted(sequences) != list(range(1, len(sequences) + 1)):
        raise ValueError("call sequences must be contiguous")
    return expected


def _assert_retry_request_matches(
    existing: DecisionCallRequestManifest,
    proposed: DecisionCallRequestManifest,
) -> None:
    if not _requests_match(existing, proposed):
        raise ValueError("retry request stage does not match frozen request")


def _requests_match(
    existing: DecisionCallRequestManifest,
    proposed: DecisionCallRequestManifest,
) -> bool:
    fields = (
        "stage",
        "input_artifact_sha256",
        "decision_model",
        "prompt_name",
        "prompt_version",
        "prompt_sha256",
        "system_prompt",
        "user_prompt",
        "response_schema_sha256",
    )
    return all(getattr(existing, field) == getattr(proposed, field) for field in fields)


def _find_matching_call(
    store: DecisionRunStore,
    proposed: DecisionCallRequestManifest,
) -> FrozenCall | None:
    calls_path = store.root / "calls"
    _directory(calls_path, "calls")
    matches: list[FrozenCall] = []
    for entry in sorted(calls_path.iterdir(), key=lambda item: int(item.name)):
        if entry.is_symlink() or not entry.is_dir() or not _CALL_DIR_RE.fullmatch(entry.name):
            continue
        request_path = entry / "request_manifest.json"
        if not request_path.is_file() or request_path.is_symlink():
            continue
        request = _validate_model_bytes(
            request_path.read_bytes(),
            DecisionCallRequestManifest,
            f"call {entry.name} request manifest",
        )
        if not _requests_match(request, proposed):
            continue
        response = None
        response_path = entry / "response.json"
        if response_path.exists() or response_path.is_symlink():
            response_manifest = _validate_model_bytes(
                response_path.read_bytes(),
                DecisionCallResponseManifest,
                f"call {entry.name} response",
            )
            _validate_response_binding(request, response_manifest)
            response = FrozenArtifact(
                name="response", manifest=response_manifest, path=response_path
            )
        matches.append(
            FrozenCall(
                call_id=request.call_id,
                manifest=request,
                path=request_path,
                response=response,
            )
        )
    if len(matches) > 1:
        raise ValueError("request ledger contains duplicate logical requests")
    return matches[0] if matches else None


def _load_response_model(call: FrozenCall, response_model: type[T]) -> T:
    if call.response is None:
        raise ValueError("response is missing")
    if call.response.manifest.response_model_name != response_model.__name__:
        raise ValueError("response model does not match frozen request")
    try:
        return response_model.model_validate(
            call.response.manifest.response_payload,
            strict=True,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise ValueError("frozen response does not match response model") from exc
