from __future__ import annotations

from collections.abc import Iterable

from src.schemas import CommentRecord, EvidenceQuote


def validate_evidence_quotes(
    quotes: Iterable[EvidenceQuote],
    comments: Iterable[CommentRecord],
) -> None:
    comment_index: dict[str, CommentRecord] = {}
    for comment in comments:
        if comment.comment_id in comment_index:
            raise ValueError(f"comment_id 重复: {comment.comment_id}")
        comment_index[comment.comment_id] = comment
    for evidence in quotes:
        comment = comment_index.get(evidence.comment_id)
        if comment is None:
            raise ValueError(f"证据 comment_id 不存在: {evidence.comment_id}")
        if evidence.quote not in comment.raw_content:
            raise ValueError(
                f"证据 quote 不是评论 {evidence.comment_id} 原文中的连续片段"
            )
