"""飞书自建应用 tenant access token 鉴权。"""

from typing import Any

import requests


AUTH_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"


class FeishuAuthError(RuntimeError):
    """飞书凭据或 token 响应无效。"""


def get_tenant_access_token(
    app_id: str,
    app_secret: str,
    *,
    session: Any | None = None,
    timeout: float = 10.0,
) -> str:
    if not app_id or not app_secret:
        raise ValueError("缺少 FEISHU_APP_ID 或 FEISHU_APP_SECRET")

    http = session or requests.Session()
    response = http.request(
        "POST",
        AUTH_URL,
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=timeout,
    )
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        raise FeishuAuthError("飞书 Token 接口未返回合法 JSON") from exc

    if payload.get("code") != 0:
        raise FeishuAuthError(
            f"飞书 Token 获取失败（code={payload.get('code')}）：{payload.get('msg', '未知错误')}"
        )
    token = payload.get("tenant_access_token")
    if not isinstance(token, str) or not token:
        raise FeishuAuthError("飞书 Token 响应缺少 tenant_access_token")
    return token
