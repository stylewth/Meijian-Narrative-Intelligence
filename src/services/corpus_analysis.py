from __future__ import annotations

from collections.abc import Iterable

from src.evidence import validate_evidence_quotes
from src.llm_client import LLMClient
from src.schemas import CommentRecord, CorpusAnalysisResult, EvidenceAtom, EvidenceQuote, EvidenceRoute
from src.services import _build_system_prompt, _serialize_prompt_input
from src.services.evidence_impact import eligible_brand_atoms


def _collect_evidence(result: CorpusAnalysisResult) -> list[EvidenceQuote]:
    evidence: list[EvidenceQuote] = []
    for conflict in result.emotional_conflicts:
        evidence.extend(conflict.supporting_evidence)
        evidence.extend(conflict.counter_evidence)
    comparison = result.feedback_comparison
    for clusters in (
        comparison.brand_positive,
        comparison.brand_negative,
        comparison.competitor_positive,
        comparison.competitor_negative,
    ):
        for cluster in clusters:
            evidence.extend(cluster.evidence)
    return evidence


def adapt_evidence_atoms_for_corpus(
    *,
    comments: Iterable[CommentRecord],
    evidence_atoms: Iterable[EvidenceAtom],
) -> list[CommentRecord]:
    """Pass eligible BRAND evidence and supplemental SCENE evidence to the existing analyzer."""

    comment_list = list(comments)
    comment_by_id = {comment.comment_id: comment for comment in comment_list}
    if len(comment_by_id) != len(comment_list):
        raise ValueError("comment_id 必须唯一")
    atom_list = list(evidence_atoms)
    if len({atom.evidence_id for atom in atom_list}) != len(atom_list):
        raise ValueError("evidence_id 必须唯一")
    for atom in atom_list:
        comment = comment_by_id.get(atom.comment_id)
        if comment is None:
            raise ValueError("Evidence Atom 的 comment_id 不存在")
        if comment.source is None or atom.source != comment.source:
            raise ValueError("Evidence Atom 的 SourceReference 必须与评论严格一致")

    brand_ids = {atom.comment_id for atom in eligible_brand_atoms(atom_list)}
    if not brand_ids:
        return []
    scene_ids = {
        atom.comment_id for atom in atom_list if atom.route is EvidenceRoute.SCENE
    }
    return [
        comment for comment in comment_list if comment.comment_id in brand_ids | scene_ids
    ]


def analyze_corpus(
    *,
    client: LLMClient,
    comments: Iterable[CommentRecord],
    evidence_atoms: Iterable[EvidenceAtom] | None = None,
) -> CorpusAnalysisResult:
    comment_list = list(comments)
    if evidence_atoms is not None:
        comment_list = adapt_evidence_atoms_for_corpus(
            comments=comment_list,
            evidence_atoms=evidence_atoms,
        )
        if not comment_list:
            raise ValueError("至少两条独立的非 C 级 BRAND Evidence Atom 才能进入候选分析")
    result = client.generate_json(
        system_prompt=_build_system_prompt("corpus_analysis"),
        user_prompt=_serialize_prompt_input(
            [comment.model_dump(mode="json") for comment in comment_list]
        ),
        response_model=CorpusAnalysisResult,
        evidence_catalog={
            comment.comment_id: comment.raw_content for comment in comment_list
        },
    )

    conflict_ids = [conflict.conflict_id for conflict in result.emotional_conflicts]
    if len(conflict_ids) != len(set(conflict_ids)):
        raise ValueError("conflict_id 重复")
    validate_evidence_quotes(_collect_evidence(result), comment_list)
    return result
