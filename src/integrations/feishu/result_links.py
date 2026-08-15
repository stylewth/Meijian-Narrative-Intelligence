"""Resolve notification result links without inventing table matches."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from src.services.demo_notifications import DemoNotificationNode


_RESULT_TABLE_IDS: Mapping[DemoNotificationNode, str] = {
    DemoNotificationNode.PREPROCESSING_COMPLETE: "tbl5D9q917nGJSsX",
    DemoNotificationNode.PRESSURE_TEST_COMPLETE: "tblTQejZoUta2yz0",
    DemoNotificationNode.BLIND_SELECTION_COMPLETE: "tbl6EHodQdhkReo2",
}


def resolve_result_url(
    node: DemoNotificationNode | str,
    *,
    wiki_url: str | None,
    web_url: str,
) -> str:
    """Return a matching table URL or the supplied web URL.

    Only the three nodes with a direct result-table contract are mapped. Delta
    checkpoints and the final synthesis intentionally remain web-only.
    """

    table_id = _table_id_for(node)
    if table_id is None:
        return web_url
    table_url = _build_table_url(wiki_url, table_id)
    return web_url if table_url is None else table_url


def _table_id_for(node: DemoNotificationNode | str) -> str | None:
    try:
        normalized = node if isinstance(node, DemoNotificationNode) else DemoNotificationNode(node)
    except (TypeError, ValueError):
        return None
    return _RESULT_TABLE_IDS.get(normalized)


def _build_table_url(wiki_url: str | None, table_id: str) -> str | None:
    if not isinstance(wiki_url, str) or not wiki_url.strip():
        return None
    parsed = urlsplit(wiki_url.strip())
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not _is_feishu_host(hostname):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0] != "wiki" or not parts[1].strip():
        return None

    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key not in {"table", "view"}
    ]
    query.append(("table", table_id))
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query),
            "",
        )
    )


def _is_feishu_host(hostname: str) -> bool:
    return hostname in {"feishu.cn", "larksuite.com"} or hostname.endswith(
        (".feishu.cn", ".larksuite.com")
    )


__all__ = ["resolve_result_url"]
