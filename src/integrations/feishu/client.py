"""飞书多维表格只读 HTTP 客户端。"""

from typing import Any
from urllib.parse import quote

import requests

from .auth import get_tenant_access_token


OPEN_API_ROOT = "https://open.feishu.cn/open-apis"


class FeishuAPIError(RuntimeError):
    """飞书业务 API 返回明确失败或不完整数据。"""


class FeishuBitableClient:
    """提供飞书多维表格字段、记录读取和单条记录写入。"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        session: Any | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._session = session or requests.Session()
        self._timeout = timeout
        self._tenant_access_token: str | None = None

    def get_tables(
        self, app_token: str, *, page_size: int = 100
    ) -> list[dict[str, Any]]:
        path = f"/bitable/v1/apps/{quote(app_token, safe='')}/tables"
        return self._get_paginated(path, page_size=page_size, max_page_size=100)

    def get_fields(
        self, app_token: str, table_id: str, *, page_size: int = 100
    ) -> list[dict[str, Any]]:
        path = (
            f"/bitable/v1/apps/{quote(app_token, safe='')}/tables/"
            f"{quote(table_id, safe='')}/fields"
        )
        return self._get_paginated(path, page_size=page_size, max_page_size=100)

    def get_records(
        self, app_token: str, table_id: str, *, page_size: int = 500
    ) -> list[dict[str, Any]]:
        path = (
            f"/bitable/v1/apps/{quote(app_token, safe='')}/tables/"
            f"{quote(table_id, safe='')}/records"
        )
        return self._get_paginated(path, page_size=page_size, max_page_size=500)

    def create_record(
        self, app_token: str, table_id: str, fields: dict[str, Any]
    ) -> str:
        path = (
            f"/bitable/v1/apps/{quote(app_token, safe='')}/tables/"
            f"{quote(table_id, safe='')}/records"
        )
        data = self._request("POST", path, json={"fields": fields})
        record = data.get("record")
        if not isinstance(record, dict):
            raise FeishuAPIError("飞书创建记录响应缺少 record")
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise FeishuAPIError("飞书创建记录响应缺少 record_id")
        return record_id

    def resolve_wiki_app_token(self, wiki_token: str) -> str:
        data = self._request(
            "GET", "/wiki/v2/spaces/get_node", params={"token": wiki_token}
        )
        node = data.get("node")
        if not isinstance(node, dict):
            raise FeishuAPIError("飞书 Wiki 解析响应缺少 node")
        if node.get("obj_type") != "bitable":
            raise FeishuAPIError("该 Wiki 节点不是多维表格，无法解析 app_token")
        app_token = node.get("obj_token")
        if not isinstance(app_token, str) or not app_token:
            raise FeishuAPIError("飞书 Wiki 解析响应缺少 app_token")
        return app_token

    def _get_paginated(
        self, path: str, *, page_size: int, max_page_size: int
    ) -> list[dict[str, Any]]:
        if not 1 <= page_size <= max_page_size:
            raise ValueError(f"page_size 必须在 1 到 {max_page_size} 之间")

        items: list[dict[str, Any]] = []
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        while True:
            params: dict[str, Any] = {"page_size": page_size}
            if page_token is not None:
                params["page_token"] = page_token
            data = self._request("GET", path, params=params)
            page_items = data.get("items")
            if not isinstance(page_items, list):
                raise FeishuAPIError("飞书分页响应缺少 items")
            if not all(isinstance(item, dict) for item in page_items):
                raise FeishuAPIError("飞书分页响应中的 items 格式错误")
            items.extend(page_items)

            if not data.get("has_more"):
                return items
            next_token = data.get("page_token")
            if not isinstance(next_token, str) or not next_token:
                raise FeishuAPIError("飞书分页响应声明 has_more 但缺少 page_token")
            if next_token in seen_page_tokens:
                raise FeishuAPIError("飞书分页响应 page_token 重复，检测到分页循环")
            seen_page_tokens.add(next_token)
            page_token = next_token

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_kwargs: dict[str, Any] = {
            "headers": {
                "Authorization": f"Bearer {self._access_token()}",
                "Content-Type": "application/json; charset=utf-8",
            },
            "params": params,
            "timeout": self._timeout,
        }
        if json is not None:
            request_kwargs["json"] = json
        try:
            response = self._session.request(
                method,
                f"{OPEN_API_ROOT}{path}",
                **request_kwargs,
            )
            response.raise_for_status()
        except (requests.RequestException, RuntimeError) as exc:
            raise FeishuAPIError(f"飞书 HTTP 请求失败：{exc}") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise FeishuAPIError("飞书 API 未返回合法 JSON") from exc

        if not isinstance(payload, dict):
            raise FeishuAPIError("飞书 API 响应格式错误，顶层必须是对象")
        if payload.get("code") != 0:
            raise FeishuAPIError(
                f"飞书 API 请求失败（code={payload.get('code')}）：{payload.get('msg', '未知错误')}"
            )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise FeishuAPIError("飞书 API 响应缺少 data")
        return data

    def _access_token(self) -> str:
        if self._tenant_access_token is None:
            self._tenant_access_token = get_tenant_access_token(
                self._app_id,
                self._app_secret,
                session=self._session,
                timeout=self._timeout,
            )
        return self._tenant_access_token
