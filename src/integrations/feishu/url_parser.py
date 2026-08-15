"""解析飞书多维表格分享链接，不猜测或替换资源标识。"""

from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse


class FeishuURLParseError(ValueError):
    """飞书多维表格链接无法确定目标资源。"""


@dataclass(frozen=True, slots=True)
class BitableLocation:
    table_id: str
    app_token: str | None = None
    wiki_token: str | None = None


def parse_bitable_url(url: str) -> BitableLocation:
    """解析 ``/base`` 或 ``/wiki`` 链接。

    wiki token 不是 app token。这里原样保留，调用方必须通过飞书 API
    显式解析，不能把二者混用。
    """

    if not isinstance(url, str) or not url.strip():
        raise FeishuURLParseError("飞书多维表格链接不能为空")

    parsed = urlparse(url.strip())
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not _is_feishu_host(hostname):
        raise FeishuURLParseError("链接必须是有效的飞书域名 URL")

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) != 2 or parts[0] not in {"base", "wiki"}:
        raise FeishuURLParseError("链接路径必须为 /base/<app_token> 或 /wiki/<wiki_token>")

    token = parts[1].strip()
    if not token:
        token_name = "app_token" if parts[0] == "base" else "wiki_token"
        raise FeishuURLParseError(f"链接缺少 {token_name}")

    table_ids = parse_qs(parsed.query).get("table", [])
    table_id = table_ids[0].strip() if table_ids else ""
    if not table_id:
        raise FeishuURLParseError("链接无法确定 table_id")

    if parts[0] == "base":
        return BitableLocation(app_token=token, table_id=table_id)
    return BitableLocation(wiki_token=token, table_id=table_id)


def _is_feishu_host(hostname: str) -> bool:
    return hostname in {"feishu.cn", "larksuite.com"} or hostname.endswith(
        (".feishu.cn", ".larksuite.com")
    )
