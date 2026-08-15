from __future__ import annotations

import json
from pathlib import Path

from src.services.evolution_presentation import load_official_evolution_run
from src.services.preprocessing_demo_presentation import load_preprocessing_demo
from src.services.pressure_test_presentation import load_pressure_test_run
from src.ui.preprocessing_workspace import build_official_replay_stages


ROOT = Path(__file__).resolve().parents[1]
SCREENED_ROOT = ROOT / "data/competition/screened_v2"
VALIDATION_ROOT = (
    ROOT
    / "data/competition/five_candidate_validation/official-20260813-five-candidate-validation-002"
)
EVOLUTION_ROOT = (
    ROOT
    / "data/competition/evolution_runs/official-20260814-three-opportunity-evolution-001"
)
CATALOG_PATH = ROOT / "data/submission/evidence_catalog.json"


def test_release_uses_a_small_deidentified_evidence_catalog() -> None:
    assert CATALOG_PATH.is_file(), "minimal evidence catalog has not been created"
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    evidence = payload["evidence"]

    assert payload["schema_version"] == "submission_evidence_catalog_v1"
    assert len(evidence) == 40
    assert len({item["evidence_id"] for item in evidence}) == 40
    assert all(set(item) == {"evidence_id", "comment_id", "raw_content", "source_platform", "source_type"} for item in evidence)
    assert not list((ROOT / "data").rglob("records.json"))
    assert not list((ROOT / "data").rglob("evidence_atoms.json"))
    assert not list((ROOT / "data").rglob("annotation_import_summary.json"))


def test_all_three_official_workspaces_load_from_the_release_package() -> None:
    preprocessing = load_preprocessing_demo(SCREENED_ROOT, VALIDATION_ROOT)
    pressure = load_pressure_test_run(VALIDATION_ROOT, EVOLUTION_ROOT / "blind/selection_confirmation.json")
    evolution = load_official_evolution_run(EVOLUTION_ROOT)

    assert len(preprocessing.stages) == 5
    assert len(preprocessing.opportunities) == 5
    assert len(pressure.candidates) == 5
    assert len(pressure.evidence_catalog) == 40
    assert len(evolution.selected_candidate_ids) == 3
    assert len(evolution.numeric_points) == 6


def test_official_preprocessing_replay_is_fully_unlocked() -> None:
    stages = build_official_replay_stages(SCREENED_ROOT)

    assert len(stages) == 5
    assert all(stage.ready for stage in stages), [
        (stage.key, stage.reason) for stage in stages if not stage.ready
    ]
