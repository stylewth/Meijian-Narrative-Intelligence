"""Small requests-based Feishu IM and CardKit client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import requests

from .auth import FeishuAuthError, get_tenant_access_token
from .notification_cards import validate_card


OPEN_API_ROOT = "https://open.feishu.cn/open-apis"
REQUEST_TIMEOUT = (0.75, 1.5)


class FeishuMessageError(RuntimeError):
    """A Feishu message/CardKit request or response violated its contract."""

    def __init__(self, message: str, *, code: Any = None) -> None:
        super().__init__(message)
        self.code = code


class FeishuMessageClient:
    def __init__(
        self,
        app_id: str,
        app_secret: str,
        *,
        session: Any = requests,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._session = session
        self._token: str | None = None

    def send_card(self, chat_id: str, card: Mapping[str, Any], *, uuid: str) -> str:
        validate_card(card)
        payload = self._request(
            "POST",
            "/im/v1/messages?receive_id_type=chat_id",
            {
                "receive_id": chat_id,
                "msg_type": "interactive",
                "content": _serialize(card),
                "uuid": _required_uuid(uuid),
            },
        )
        return _required_id(payload, "message_id")

    def reply_card_entity(self, message_id: str, card_id: str, *, uuid: str) -> str:
        payload = self._request(
            "POST",
            f"/im/v1/messages/{message_id}/reply",
            {
                "msg_type": "interactive",
                "content": _serialize({"type": "card", "data": {"card_id": card_id}}),
                "uuid": _required_uuid(uuid),
            },
        )
        return _required_id(payload, "message_id")

    def create_card_entity(self, card: Mapping[str, Any]) -> str:
        validate_card(card)
        payload = self._request(
            "POST",
            "/cardkit/v1/cards",
            {"type": "card_json", "data": _serialize(card)},
        )
        return _required_id(payload, "card_id")

    def update_card_entity(
        self,
        card_id: str,
        *,
        sequence: int,
        uuid: str,
        card: Mapping[str, Any],
    ) -> None:
        validate_card(card)
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
            raise ValueError("CardKit sequence 必须是非负整数")
        self._request(
            "PUT",
            f"/cardkit/v1/cards/{card_id}",
            {
                "data": _serialize(card),
                "sequence": sequence,
                "uuid": _required_uuid(uuid),
            },
        )

    def _request(self, method: str, path: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            token = self._get_token()
            response = self._session.request(
                method,
                f"{OPEN_API_ROOT}{path}",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=dict(payload),
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except FeishuMessageError:
            raise
        except (FeishuAuthError, requests.RequestException, OSError) as exc:
            raise FeishuMessageError(str(exc), code="REQUEST_ERROR") from exc

        try:
            decoded = response.json()
        except (TypeError, ValueError) as exc:
            raise FeishuMessageError("飞书 API 未返回合法 JSON", code="INVALID_JSON") from exc
        if not isinstance(decoded, Mapping):
            raise FeishuMessageError("飞书 API 响应顶层必须是对象", code="INVALID_RESPONSE")
        code = decoded.get("code")
        if code != 0:
            raise FeishuMessageError(
                f"飞书 API 请求失败（code={code}）：{decoded.get('msg', '未知错误')}",
                code=code,
            )
        data = decoded.get("data")
        if not isinstance(data, Mapping):
            raise FeishuMessageError("飞书 API 响应缺少 data", code="MISSING_DATA")
        return data

    def _get_token(self) -> str:
        if self._token is None:
            try:
                self._token = get_tenant_access_token(
                    self._app_id,
                    self._app_secret,
                    session=self._session,
                    timeout=REQUEST_TIMEOUT,
                )
            except FeishuAuthError as exc:
                raise FeishuMessageError(str(exc)) from exc
        return self._token


def _serialize(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _required_uuid(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Feishu UUID 不能为空")
    return value


def _required_id(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise FeishuMessageError(f"飞书 API 响应缺少 {key}", code=f"MISSING_{key.upper()}")
    return value
