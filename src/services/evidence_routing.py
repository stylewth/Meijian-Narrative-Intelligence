from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import Field

from src.llm_client import LLMClient
from src.schemas import (
    CommentRecord,
    EvidenceAtom,
    EvidenceGrade,
    EvidenceRoute,
    ExperienceScope,
    ScreeningStatus,
    SourceType,
    StrictBaseModel,
)
from src.services import _build_system_prompt, _serialize_prompt_input


class _SourceReferenceDraft(StrictBaseModel):
    source_id: str = Field(min_length=1)
    source_type: str = Field(
        pattern=r"^(USER_COMMENT|OFFICIAL_CONTENT|MEDIA_CONTENT|BRAND_DOC|PRODUCT_FACT|COMPETITOR_DOC|VALIDATION|SIMULATED)$"
    )
    source_ref: str = Field(min_length=1)


class _EvidenceAtomDraft(StrictBaseModel):
    comment_id: str = Field(min_length=1)
    route: str = Field(pattern=r"^(BRAND|PRODUCT|SERVICE|SCENE|COMPETITOR|OTHER)$")
    experience_scope: str = Field(
        pattern=r"^(ACTUAL_USE|PURCHASE_ONLY|NON_USE|UNKNOWN)$"
    )
    evidence_grade: str = Field(pattern=r"^(A|B|C)$")
    ai_confidence: float = Field(ge=0, le=1)
    explanation: str = Field(min_length=1)
    source: _SourceReferenceDraft


class _EvidenceRoutingResponse(StrictBaseModel):
    atoms: list[_EvidenceAtomDraft]


def _validate_input_comments(comments: list[CommentRecord]) -> None:
    if not 10 <= len(comments) <= 20:
        raise ValueError("Evidence Routing 每批只接受 10–20 条评论")
    comment_ids = [comment.comment_id for comment in comments]
    if len(comment_ids) != len(set(comment_ids)):
        raise ValueError("comment_id 必须唯一")
    for comment in comments:
        if comment.source is None:
            raise ValueError(f"评论 {comment.comment_id} 缺少 SourceReference")


def select_routable_comments(comments: Iterable[CommentRecord]) -> list[CommentRecord]:
    """Return only contract-approved online records; never send excluded data to the model."""

    selected: list[CommentRecord] = []
    for comment in comments:
        if comment.screening_status is None:
            raise ValueError(f"评论 {comment.comment_id} 缺少 screening_status，不能进入在线 Routing")
        if comment.source is None:
            raise ValueError(f"评论 {comment.comment_id} 缺少 SourceReference，不能进入在线 Routing")
        if (
            comment.screening_status is ScreeningStatus.KEEP
            and comment.source.source_type is not SourceType.SIMULATED
        ):
            selected.append(comment)
    return selected


def partition_evidence_batches(items: Iterable[Any]) -> list[list[Any]]:
    """Split an online routing run into 10–20 item batches without a short tail."""

    values = list(items)
    if len(values) < 10:
        raise ValueError("Evidence Routing 总数不足 10 条，不能形成合规批次")
    batch_count = (len(values) + 19) // 20
    base, remainder = divmod(len(values), batch_count)
    if base < 10:
        raise ValueError("Evidence Routing 不能形成每批至少 10 条的批次")
    return [
        values[offset : offset + base + (1 if index < remainder else 0)]
        for index, offset in enumerate(
            [sum(base + (1 if prior < remainder else 0) for prior in range(index)) for index in range(batch_count)]
        )
    ]


def _as_routing_response(value: Any) -> _EvidenceRoutingResponse:
    if isinstance(value, _EvidenceRoutingResponse):
        return value
    return _EvidenceRoutingResponse.model_validate(value, strict=True)


def route_evidence_batch(
    *,
    client: LLMClient,
    comments: Iterable[CommentRecord],
) -> list[EvidenceAtom]:
    """Extract exactly one validated evidence atom for each input comment."""

    comment_list = list(comments)
    _validate_input_comments(comment_list)
    response = _as_routing_response(
        client.generate_json(
            system_prompt=_build_system_prompt("evidence_routing"),
            user_prompt=_serialize_prompt_input(
                [comment.model_dump(mode="json") for comment in comment_list]
            ),
            response_model=_EvidenceRoutingResponse,
        )
    )

    input_by_id = {comment.comment_id: comment for comment in comment_list}
    output_ids = [atom.comment_id for atom in response.atoms]
    if len(output_ids) != len(set(output_ids)):
        raise ValueError("输出 comment_id 不得重复")
    if set(output_ids) != set(input_by_id):
        raise ValueError("输出 comment_id 集合必须与输入严格一致")

    drafts_by_id = {atom.comment_id: atom for atom in response.atoms}
    atoms: list[EvidenceAtom] = []
    for comment in comment_list:
        draft = drafts_by_id[comment.comment_id]
        if draft.source.model_dump(mode="json") != comment.source.model_dump(mode="json"):
            raise ValueError(f"评论 {comment.comment_id} 的 SourceReference 必须与输入严格一致")
        atoms.append(
            EvidenceAtom(
                evidence_id=comment.comment_id,
                comment_id=comment.comment_id,
                route=EvidenceRoute(draft.route),
                experience_scope=ExperienceScope(draft.experience_scope),
                evidence_grade=EvidenceGrade(draft.evidence_grade),
                ai_confidence=draft.ai_confidence,
                explanation=draft.explanation,
                source=comment.source,
                source_platform=comment.source_platform,
                duplicate_group=comment.duplicate_group,
                actual_use=comment.actual_use,
            )
        )
    return atoms
