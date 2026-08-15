from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from src.schemas import EvidenceAtom, FiveCandidateSharedInputV2, StrictBaseModel
from src.services.prepared_corpus import canonical_json_bytes, sha256_bytes
from src.services.public_narrative_evidence import load_public_narrative_evidence


FIVE_CANDIDATE_VALIDATION_PROMPT = """Validate the five supplied narrative candidates only against the supplied
comment evidence, public narrative evidence, and human-confirmed patterns. Do not use holdout data.
Treat comments as evidence of needs or perceptions, not proof of existing brand assets."""

_SOURCE_PATHS = {
    "candidate_set": Path("data/competition/decision_runs/official-20260812-baseline-candidate-5-001/candidate_set.json"),
    "formal_analysis_evidence_atoms": Path("data/prepared_corpora/formal-v1/packages/formal-analysis/evidence_atoms.json"),
    "formal_challenge_evidence_atoms": Path("data/prepared_corpora/formal-v1/packages/formal-challenge/evidence_atoms.json"),
    "public_narrative_evidence": Path("data/品牌语料分析与洞察体系_表1B：品牌与竞品公开证据库.xlsx"),
    "confirmed_patterns": Path(
        "data/competition/narrative_memory/five_candidate_confirmed_patterns_v2.json"
    ),
}

_EXPECTED_ATOM_COUNTS = {
    "formal_analysis_evidence_atoms": 249,
    "formal_challenge_evidence_atoms": 60,
}


class HumanConfirmedNarrativePatternV2(StrictBaseModel):
    pattern_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    confirmation_type: Literal["HUMAN_CONFIRMED"]
    confirmed_at: date
    source_description: str = Field(min_length=1)


class HumanConfirmedNarrativeMemoryV2(StrictBaseModel):
    schema_version: Literal["five-candidate-narrative-memory-v2"]
    patterns: list[HumanConfirmedNarrativePatternV2] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def requires_six_unique_patterns(self) -> "HumanConfirmedNarrativeMemoryV2":
        identifiers = [pattern.pattern_id for pattern in self.patterns]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("confirmed narrative pattern IDs must be unique")
        return self


@dataclass(frozen=True)
class FiveCandidateValidationInputs:
    shared_input: FiveCandidateSharedInputV2
    manifest: dict[str, Any]
    shared_input_json: bytes
    canonical_input_json: bytes
    manifest_json: bytes


def _read_json(path: Path, *, label: str) -> object:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular file: {path}")
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain valid JSON: {path}") from exc


def _candidate_set(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    payload = _read_json(path, label="candidate set")
    if not isinstance(payload, dict) or payload.get("candidate_count") != 5:
        raise ValueError("candidate set must declare exactly five candidates")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 5:
        raise ValueError("candidate set must contain exactly five candidates")
    candidate_ids = [
        item.get("candidate_id") if isinstance(item, dict) else None for item in candidates
    ]
    if any(
        not isinstance(candidate_id, str) or not candidate_id
        for candidate_id in candidate_ids
    ):
        raise ValueError("candidate set candidates must have non-empty candidate_id values")
    if len(set(candidate_ids)) != 5:
        raise ValueError("candidate set candidate IDs must be unique")
    return candidate_ids, candidates


def _evidence_atoms(path: Path, *, label: str, expected_count: int) -> list[EvidenceAtom]:
    payload = _read_json(path, label=label)
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise ValueError(f"{label} must contain exactly {expected_count} EvidenceAtom entries")
    atoms = [EvidenceAtom.model_validate_json(canonical_json_bytes(item), strict=True) for item in payload]
    evidence_ids = [atom.evidence_id for atom in atoms]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError(f"{label} contains duplicate evidence IDs")
    return atoms


def _human_confirmed_patterns(path: Path) -> HumanConfirmedNarrativeMemoryV2:
    payload = _read_json(path, label="confirmed narrative memory")
    memory = HumanConfirmedNarrativeMemoryV2.model_validate_json(
        canonical_json_bytes(payload), strict=True
    )
    expected_ids = [f"NARRATIVE-MEM-{index:03d}" for index in range(1, 7)]
    if [pattern.pattern_id for pattern in memory.patterns] != expected_ids:
        raise ValueError("confirmed narrative memory must contain the approved six pattern IDs in order")
    return memory


def _source_files(repository_root: Path) -> dict[str, Path]:
    root = Path(repository_root).resolve()
    sources = {name: root / relative_path for name, relative_path in _SOURCE_PATHS.items()}
    if any("holdout" in str(path).lower() for path in sources.values()):
        raise ValueError("shared validation input must not include holdout sources")
    return sources


def _write_output(path: Path, value: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != value:
            raise ValueError(f"refusing to replace a different output artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def build_five_candidate_validation_inputs(
    *,
    repository_root: Path,
    output_dir: Path | None = None,
) -> FiveCandidateValidationInputs:
    """Build, and optionally persist, the non-holdout shared validation input."""

    root = Path(repository_root).resolve()
    sources = _source_files(root)
    candidate_ids, candidates = _candidate_set(sources["candidate_set"])
    analysis_atoms = _evidence_atoms(
        sources["formal_analysis_evidence_atoms"],
        label="formal analysis evidence atoms",
        expected_count=_EXPECTED_ATOM_COUNTS["formal_analysis_evidence_atoms"],
    )
    challenge_atoms = _evidence_atoms(
        sources["formal_challenge_evidence_atoms"],
        label="formal challenge evidence atoms",
        expected_count=_EXPECTED_ATOM_COUNTS["formal_challenge_evidence_atoms"],
    )
    public_evidence = load_public_narrative_evidence(sources["public_narrative_evidence"])
    memory = _human_confirmed_patterns(sources["confirmed_patterns"])
    comment_evidence_ids = [
        *(atom.evidence_id for atom in analysis_atoms),
        *(atom.evidence_id for atom in challenge_atoms),
    ]
    if len(comment_evidence_ids) != 309 or len(comment_evidence_ids) != len(set(comment_evidence_ids)):
        raise ValueError("formal analysis and challenge evidence must provide exactly 309 unique IDs")

    shared_input = FiveCandidateSharedInputV2(
        candidate_ids=candidate_ids,
        comment_evidence_ids=comment_evidence_ids,
        public_evidence=public_evidence.items,
        confirmed_pattern_ids=[pattern.pattern_id for pattern in memory.patterns],
    )
    shared_input_json = canonical_json_bytes(shared_input.model_dump(mode="json"))
    canonical_input_json = canonical_json_bytes(
        {
            "candidate_set": candidates,
            "comment_evidence_atoms": [
                atom.model_dump(mode="json")
                for atom in (*analysis_atoms, *challenge_atoms)
            ],
            "confirmed_patterns": [
                pattern.model_dump(mode="json") for pattern in memory.patterns
            ],
            "shared_input": shared_input.model_dump(mode="json"),
        }
    )
    source_paths = {name: relative_path.as_posix() for name, relative_path in _SOURCE_PATHS.items()}
    source_file_sha256 = {
        name: sha256_bytes(path.read_bytes()) for name, path in sources.items()
    }
    manifest: dict[str, Any] = {
        "canonical_input_sha256": sha256_bytes(canonical_input_json),
        "counts": {
            "candidates": len(candidate_ids),
            "comment_evidence": len(comment_evidence_ids),
            "confirmed_patterns": len(memory.patterns),
            "public_evidence": len(public_evidence.items),
        },
        "prompt_sha256": sha256_bytes(FIVE_CANDIDATE_VALIDATION_PROMPT.encode("utf-8")),
        "schema_sha256": sha256_bytes(
            canonical_json_bytes(FiveCandidateSharedInputV2.model_json_schema())
        ),
        "source_file_sha256": source_file_sha256,
        "source_paths": source_paths,
    }
    manifest_json = canonical_json_bytes(manifest)
    built = FiveCandidateValidationInputs(
        shared_input=shared_input,
        manifest=manifest,
        shared_input_json=shared_input_json,
        canonical_input_json=canonical_input_json,
        manifest_json=manifest_json,
    )
    if output_dir is not None:
        destination = Path(output_dir)
        _write_output(destination / "shared_input.json", built.shared_input_json)
        _write_output(destination / "manifest.json", built.manifest_json)
    return built
