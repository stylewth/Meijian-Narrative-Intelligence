from __future__ import annotations

from collections.abc import Iterable
from difflib import SequenceMatcher
import re

from pydantic import Field

from src.llm_client import LLMClient
from src.schemas import EvidenceAtom, NarrativeCandidate, NarrativePatch, StrictBaseModel
from src.services import _build_system_prompt, _serialize_prompt_input


class NarrativeRevisionProposal(StrictBaseModel):
    candidate_id: str = Field(min_length=1)
    draft_proposition: str = Field(min_length=1)
    main_scenes: list[str] = Field(min_length=1)
    competitor_difference: str = Field(min_length=1)
    trigger_evidence_ids: list[str] = Field(min_length=1)
    reason: str = Field(min_length=1)


class _NarrativeRevisionResult(StrictBaseModel):
    revision: NarrativeRevisionProposal


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]")


def _tokens(value: str | list[str]) -> list[str]:
    text = "\n".join(value) if isinstance(value, list) else value
    return _TOKEN_RE.findall(text)


def _phrases(before: str | list[str], after: str | list[str]) -> tuple[list[str], list[str]]:
    before_tokens = _tokens(before)
    after_tokens = _tokens(after)
    matcher = SequenceMatcher(a=before_tokens, b=after_tokens, autojunk=False)
    added: list[str] = []
    removed: list[str] = []
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            removed.append("".join(before_tokens[before_start:before_end]))
        if tag in {"insert", "replace"}:
            added.append("".join(after_tokens[after_start:after_end]))
    return added, removed


def revise_candidate_from_evidence(
    *,
    client: LLMClient,
    candidate: NarrativeCandidate,
    incremental_atoms: Iterable[EvidenceAtom],
) -> tuple[NarrativeCandidate, list[NarrativePatch]]:
    atoms = list(incremental_atoms)
    atom_ids = [atom.evidence_id for atom in atoms]
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError("当前新增批次的 evidence_id 必须唯一")
    response = client.generate_json(
        system_prompt=_build_system_prompt("narrative_revision"),
        user_prompt=_serialize_prompt_input(
            {
                "candidate": candidate.model_dump(mode="json"),
                "incremental_evidence_atoms": [atom.model_dump(mode="json") for atom in atoms],
            }
        ),
        response_model=_NarrativeRevisionResult,
    )
    proposal = response.revision
    if proposal.candidate_id != candidate.candidate_id:
        raise ValueError("局部叙事修订不得改变 candidate_id")
    if not set(proposal.trigger_evidence_ids).issubset(atom_ids):
        raise ValueError("局部叙事修订只能引用当前新增批次的 evidence_id")

    revised = candidate.model_copy(
        update={
            "draft_proposition": proposal.draft_proposition,
            "main_scenes": proposal.main_scenes,
            "competitor_difference": proposal.competitor_difference,
        }
    )
    patches: list[NarrativePatch] = []
    for field_name in ("draft_proposition", "main_scenes", "competitor_difference"):
        before = getattr(candidate, field_name)
        after = getattr(revised, field_name)
        if before == after:
            continue
        added_phrases, removed_phrases = _phrases(before, after)
        patches.append(
            NarrativePatch(
                candidate_id=candidate.candidate_id,
                field_name=field_name,
                before=before,
                after=after,
                added_phrases=added_phrases,
                removed_phrases=removed_phrases,
                trigger_evidence_ids=proposal.trigger_evidence_ids,
                reason=proposal.reason,
            )
        )
    return revised, patches
