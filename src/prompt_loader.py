from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


PROMPT_VERSION = "v1"
PROMPT_NAMES = frozenset(
    {
        "system",
        "corpus_analysis",
        "diversity_assessment",
        "candidate_generation",
        "candidate_scoring",
        "evidence_routing",
        "candidate_stress",
        "evidence_assessment",
        "narrative_revision",
        "blind_reassessment",
    }
)
PROMPT_VERSIONS = {
    name: (
        "v4"
        if name == "candidate_generation"
        else "v4"
        if name == "evidence_routing"
        else "v2"
        if name in {"system", "corpus_analysis", "candidate_scoring"}
        else PROMPT_VERSION
    )
    for name in PROMPT_NAMES
}

# Replay v2 was captured before the pre-data routing and stress prompts existed.
# Its sealed manifests must keep this exact historical set; the versions are
# written out literally because those prompt files no longer exist in runtime.
REPLAY_V2_PROMPT_NAMES = frozenset(
    {
        "system",
        "single_comment",
        "corpus_analysis",
        "diversity_assessment",
        "candidate_generation",
        "candidate_scoring",
        "template_check",
        "final_refinement",
    }
)
REPLAY_V2_PROMPT_VERSIONS = {
    "system": "v1",
    "single_comment": "v1",
    "corpus_analysis": "v1",
    "diversity_assessment": "v1",
    "candidate_generation": "v3",
    "candidate_scoring": "v1",
    "template_check": "v2",
    "final_refinement": "v1",
}


@dataclass(frozen=True)
class FileMetadata:
    path: Path
    version: str
    sha256: str
    raw_bytes: bytes
    text: str


@dataclass(frozen=True)
class PromptMetadata:
    name: str
    version: str
    sha256: str
    content: str
    raw_bytes: bytes


_PROMPT_HEADER = re.compile(r"\A<!-- prompt-version: ([^\r\n]+) -->")


def read_file_metadata(
    path: Path, *, expected_version: str | None = None
) -> FileMetadata:
    raw_bytes = path.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Prompt 文件必须是 UTF-8: {path}") from exc
    match = _PROMPT_HEADER.match(text)
    if match is None:
        raise ValueError(f"Prompt 文件缺少合法版本头: {path.name}")
    version = match.group(1)
    if expected_version is not None and version != expected_version:
        raise ValueError(f"Prompt 版本不受支持: {path.name}")
    return FileMetadata(
        path=path,
        version=version,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw_bytes=raw_bytes,
        text=text,
    )


def load_prompt_metadata(name: str, *, prompt_dir: Path | None = None) -> PromptMetadata:
    if name not in PROMPT_NAMES:
        raise ValueError(f"未知 Prompt: {name}")
    directory = prompt_dir or Path(__file__).parents[1] / "prompts"
    path = directory / f"{name}.md"
    version = PROMPT_VERSIONS[name]
    metadata = read_file_metadata(path, expected_version=version)
    return PromptMetadata(
        name=name,
        version=metadata.version,
        sha256=metadata.sha256,
        content=metadata.text,
        raw_bytes=metadata.raw_bytes,
    )


def load_prompt(name: str, *, prompt_dir: Path | None = None) -> str:
    return load_prompt_metadata(name, prompt_dir=prompt_dir).content
