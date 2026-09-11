from __future__ import annotations

import re
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP_LEVEL = {
    ".env.example",
    ".gitignore",
    ".streamlit",
    "README.md",
    "app.py",
    "assets",
    "data",
    "prompts",
    "requirements.txt",
    "src",
    "tests",
    "tools",
}
FORBIDDEN_NAMES = {
    ".env",
    "secrets.toml",
    "findings.md",
    "lessons.md",
    "progress.md",
    "task_plan.md",
}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"cli_[A-Za-z0-9]{12,}"),
    re.compile(r"(?i)(api[_-]?key|app[_-]?secret|password)\s*=\s*[\"'][^\"']{8,}"),
)


def tracked_candidate_files() -> list[Path]:
    """发布合同检查 git 跟踪文件；本地 .env 与 outputs/ 运行产物不属于提交范围。"""

    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [ROOT / line for line in result.stdout.splitlines() if line.strip()]


def test_release_tree_uses_the_approved_allowlist() -> None:
    assert (ROOT / "app.py").is_file(), "release app.py has not been created"
    tracked = tracked_candidate_files()
    assert tracked, "git ls-files returned no tracked files"
    actual = {path.relative_to(ROOT).parts[0] for path in tracked}
    assert actual <= ALLOWED_TOP_LEVEL
    assert {"app.py", "src", "data", "README.md", "requirements.txt"} <= actual


def test_release_tree_excludes_credentials_and_development_artifacts() -> None:
    files = tracked_candidate_files()
    assert not [path for path in files if path.name in FORBIDDEN_NAMES]
    assert not [path for path in files if path.suffix.lower() in {".log", ".sqlite", ".db", ".sqlite3", ".sqlite3-wal", ".sqlite3-shm"}]
    assert not [path for path in files if "superpowers" in path.parts]
    for path in files:
        if path.suffix.lower() not in {".py", ".md", ".toml", ".txt", ".json", ".example"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(pattern.search(text) for pattern in SECRET_PATTERNS), path


def test_release_contains_no_legacy_market_compatibility_route() -> None:
    app_text = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "_legacy_market_compatibility" not in app_text
    assert "src.replay" not in app_text
    assert "梅见评论分析.xlsx" not in app_text


def test_release_excludes_legacy_source_modules_and_prompts() -> None:
    forbidden_sources = (
        "src/replay",
        "src/ui/decision_workspace.py",
        "src/ui/brand_specificity.py",
        "src/services/competition_pipeline.py",
    )
    assert not [relative for relative in forbidden_sources if (ROOT / relative).exists()]
    assert {path.name for path in (ROOT / "prompts").glob("*.md")} == {
        "system.md",
        "evidence_routing.md",
        "brand_specificity_audit.md",
        "brand_specificity_audit_v2.md",
        "brand_specificity_audit_v3.md",
        "brand_specificity_revision.md",
        "brand_specificity_revision_v2.md",
        "brand_specificity_revision_v3.md",
        "corpus_analysis.md",
        "diversity_assessment.md",
        "candidate_generation.md",
        "candidate_scoring.md",
        "candidate_stress.md",
        "evidence_assessment.md",
        "narrative_revision.md",
        "blind_reassessment.md",
    }


def test_readme_uses_four_final_interface_screenshots() -> None:
    screenshots = (
        "assets/gateway.png",
        "assets/preprocessing.png",
        "assets/pressure-test.png",
        "assets/evolution.png",
    )
    assert not [relative for relative in screenshots if not (ROOT / relative).is_file()]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert all(f"]({relative})" in readme for relative in screenshots)
