"""飞书自建应用 tenant access token 鉴权。"""

import time
from typing import Any

import requests


AUTH_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
TOKEN_INVALID_CODE = 99991663
DEFAULT_REFRESH_MARGIN_SECONDS = 120.0


class FeishuAuthError(RuntimeError):
    """飞书凭据或 token 响应无效。"""


def get_tenant_access_token_details(
    app_id: str,
    app_secret: str,
    *,
    session: Any | None = None,
    timeout: float = 10.0,
) -> tuple[str, int]:
    """返回 (tenant_access_token, expire 秒数)；expire 缺失视为响应合同违约。"""

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
    expire = payload.get("expire")
    if not isinstance(expire, int) or isinstance(expire, bool) or expire <= 0:
        raise FeishuAuthError("飞书 Token 响应缺少有效的 expire")
    return token, expire


def get_tenant_access_token(
    app_id: str,
    app_secret: str,
    *,
    session: Any | None = None,
    timeout: float = 10.0,
) -> str:
    return get_tenant_access_token_details(
        app_id, app_secret, session=session, timeout=timeout
    )[0]


class TenantAccessTokenProvider:
    """缓存 tenant_access_token，并在过期前主动刷新。"""

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        session: Any | None = None,
        timeout: float = 10.0,
        clock: Any = time.monotonic,
        refresh_margin_seconds: float = DEFAULT_REFRESH_MARGIN_SECONDS,
    ) -> None:
        if not app_id or not app_secret:
            raise ValueError("缺少 FEISHU_APP_ID 或 FEISHU_APP_SECRET")
        if refresh_margin_seconds < 0:
            raise ValueError("refresh_margin_seconds 不能为负")
        self._app_id = app_id
        self._app_secret = app_secret
        self._session = session
        self._timeout = timeout
        self._clock = clock
        self._refresh_margin_seconds = refresh_margin_seconds
        self._token: str | None = None
        self._expires_at: float | None = None

    def get_token(self, *, force_refresh: bool = False) -> str:
        now = self._clock()
        if (
            not force_refresh
            and self._token is not None
            and self._expires_at is not None
            and now < self._expires_at
        ):
            return self._token
        token, expires_in = get_tenant_access_token_details(
            self._app_id,
            self._app_secret,
            session=self._session,
            timeout=self._timeout,
        )
        self._token = token
        self._expires_at = now + max(
            expires_in - self._refresh_margin_seconds, 1.0
        )
        return token
