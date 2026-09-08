"""Deterministic Schema 2.0 cards for demo notifications and replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


MAX_CARD_BYTES = 30 * 1024
MAX_CARD_ELEMENTS = 200

_FORBIDDEN_FIELDS = ("FEISHU_APP_SECRET", "raw_id", "source_path", "model_raw_response")


def build_notification_card(
    snapshot: Any,
    *,
    session_short_id: str,
    web_url: str,
    result_url: str | None = None,
) -> dict[str, Any]:
    """Build a notification card without exposing source-only snapshot fields."""

    data = _snapshot_mapping(snapshot)
    title = _required_text(data, "title")
    current_result_url = web_url if result_url is None else result_url
    if not isinstance(current_result_url, str) or not current_result_url.strip():
        raise ValueError("result_url must be a non-empty URL")
    node_key = data.get("node", "notification")
    ordinal = data.get("ordinal", "unknown")
    prefix = _element_prefix("n", session_short_id, node_key, ordinal)
    elements = _summary_elements(data, prefix=prefix, session_short_id=session_short_id)
    elements.extend(
        _button_elements(
            [
                _button(
                    f"{prefix}_current",
                    "查看当前结果",
                    None,
                    button_type="primary",
                    behavior={"type": "open_url", "default_url": current_result_url},
                ),
                _button(
                    f"{prefix}_web",
                    "打开网页",
                    None,
                    button_type="default",
                    behavior={"type": "open_url", "default_url": web_url},
                ),
            ],
        )
    )
    card = _card(title, elements)
    validate_card(card)
    return card


def build_replay_card(
    snapshot: Any,
    *,
    replay_id: str,
    current_ordinal: int,
    web_url: str,
) -> dict[str, Any]:
    """Build a replay card with only the adjacent ordinal buttons enabled."""

    if not isinstance(current_ordinal, int) or isinstance(current_ordinal, bool):
        raise ValueError("回放序号必须是整数")
    if not 1 <= current_ordinal <= 8:
        raise ValueError("回放序号必须在 1 到 8 之间")
    if not replay_id:
        raise ValueError("回放 ID 不能为空")

    data = _snapshot_mapping(snapshot)
    title = _required_text(data, "title")
    node_key = data.get("node", "replay")
    prefix = _element_prefix("r", replay_id, node_key, current_ordinal)
    elements = _summary_elements(data, prefix=prefix, session_short_id=None)

    buttons: list[dict[str, Any]] = []
    if current_ordinal > 1:
        buttons.append(
            _button(
                f"{prefix}_previous",
                "上一步",
                {
                    "action": "REPLAY_PREVIOUS",
                    "replay_id": replay_id,
                    "target_ordinal": current_ordinal - 1,
                },
                button_type="default",
            )
        )
    if current_ordinal < 8:
        buttons.append(
            _button(
                f"{prefix}_next",
                "下一步",
                {
                    "action": "REPLAY_NEXT",
                    "replay_id": replay_id,
                    "target_ordinal": current_ordinal + 1,
                },
                button_type="primary",
            )
        )
    if buttons:
        elements.extend(_button_elements(buttons))

    elements.append(
        _text_element(f"{prefix}_source", f"当前节点 {current_ordinal}/8 · {web_url}")
    )
    card = _card(f"远程操控 · {title}", elements)
    validate_card(card)
    return card


def validate_card(card: Mapping[str, Any]) -> None:
    """Validate the public CardKit contract before a card reaches the network."""

    if not isinstance(card, Mapping) or card.get("schema") != "2.0":
        raise ValueError("飞书通知卡必须使用 schema 2.0")
    encoded = json.dumps(card, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) >= MAX_CARD_BYTES:
        raise ValueError("飞书卡片超过 30KB")
    if _count_elements(card) > MAX_CARD_ELEMENTS:
        raise ValueError("飞书卡片组件超过 200")
    _validate_element_ids(card)
    _validate_button_behaviors(card)
    text = encoded.decode("utf-8")
    if any(item in text for item in _FORBIDDEN_FIELDS):
        raise ValueError("飞书卡片包含禁止字段")


def _card(title: str, elements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "2.0",
        "config": {"update_multi": True, "summary": {"content": title}},
        "header": {
            "title": {"tag": "plain_text", "content": title},
            "template": "turquoise",
        },
        "body": {"elements": elements},
    }


def _summary_elements(
    data: Mapping[str, Any], *, prefix: str, session_short_id: str | None
) -> list[dict[str, Any]]:
    elements: list[dict[str, Any]] = []
    if session_short_id is not None:
        elements.append(_text_element(f"{prefix}_session", f"场次：{session_short_id}"))
    elements.append(_text_element(f"{prefix}_conclusion", _required_text(data, "conclusion")))

    source = data.get("source_checkpoint_id") or data.get("source_run_id") or data.get("node")
    source_sha256 = _required_text(data, "source_sha256")
    elements.append(
        _text_element(
            f"{prefix}_provenance",
            f"来源：{source}；节点：{data.get('node', 'unknown')}；SHA-256：{source_sha256[:12]}",
        )
    )

    metrics = data.get("metrics", ()) or ()
    if metrics:
        metric_text = "；".join(
            f"{_field(metric, 'label')}: {_field(metric, 'value')}" for metric in metrics
        )
        elements.append(_text_element(f"{prefix}_metrics", metric_text))

    candidates = data.get("candidates", ()) or ()
    if candidates:
        candidate_lines = []
        for candidate in candidates[:3]:
            parts = [_field(candidate, "title")]
            score = _field(candidate, "score", None)
            rank = _field(candidate, "rank", None)
            delta = _field(candidate, "score_delta", None)
            if score is not None:
                parts.append(f"分数 {score}")
            if delta is not None:
                parts.append(f"变化 {delta}")
            if rank is not None:
                parts.append(f"排名 {rank}")
            narrative_change = _field(candidate, "narrative_change", "")
            if narrative_change:
                parts.append(f"叙事变化 {narrative_change}")
            candidate_lines.append(" · ".join(parts))
        elements.append(_text_element(f"{prefix}_candidates", "\n".join(candidate_lines)))

    risk = _field(data, "primary_risk", "")
    if risk:
        elements.append(_text_element(f"{prefix}_risk", f"主要风险：{risk}"))
    return elements


def _text_element(element_id: str, content: str) -> dict[str, Any]:
    return {
        "tag": "div",
        "element_id": element_id,
        "text": {"tag": "lark_md", "content": content},
    }


def _button_elements(buttons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return buttons


def _button(
    element_id: str,
    label: str,
    value: dict[str, Any] | None,
    *,
    button_type: str,
    behavior: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "tag": "button",
        "element_id": element_id,
        "text": {"tag": "plain_text", "content": label},
        "type": button_type,
    }
    if behavior is None:
        if value is None:
            raise ValueError("callback button value must be provided")
        result["behaviors"] = [{"type": "callback", "value": value}]
    else:
        result["behaviors"] = [behavior]
    return result


def _snapshot_mapping(snapshot: Any) -> Mapping[str, Any]:
    if isinstance(snapshot, Mapping):
        return snapshot
    if hasattr(snapshot, "model_dump"):
        dumped = snapshot.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(snapshot, "__dict__"):
        return vars(snapshot)
    raise TypeError("snapshot 必须是映射或可导出的模型")


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"snapshot 缺少 {key}")
    return value


def _field(value: Any, key: str, default: Any = "") -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    if hasattr(value, key):
        return getattr(value, key)
    return default


def _element_prefix(kind: str, stable_value: Any, node_key: Any, ordinal: Any) -> str:
    raw = f"{kind}:{stable_value}:{node_key}:{ordinal}".encode("utf-8")
    return f"{kind}_{hashlib.sha256(raw).hexdigest()[:6]}"


def _validate_element_ids(card: Mapping[str, Any]) -> None:
    seen: set[str] = set()
    for mapping in _walk_mappings(card):
        if "element_id" not in mapping:
            continue
        element_id = mapping["element_id"]
        valid = (
            isinstance(element_id, str)
            and bool(element_id)
            and len(element_id) <= 20
            and element_id[0].isascii()
            and element_id[0].isalpha()
            and all(char.isascii() and (char.isalnum() or char == "_") for char in element_id)
        )
        if not valid:
            raise ValueError("飞书卡片 element_id 不符合 Schema 2.0 约束")
        if element_id in seen:
            raise ValueError("飞书卡片 element_id 必须唯一")
        seen.add(element_id)


def _validate_button_behaviors(card: Mapping[str, Any]) -> None:
    for mapping in _walk_mappings(card):
        if mapping.get("tag") != "button":
            continue
        if "value" in mapping:
            raise ValueError("飞书卡片按钮回传参数必须放在 behaviors")
        behaviors = mapping.get("behaviors")
        if not isinstance(behaviors, list) or len(behaviors) != 1:
            raise ValueError("飞书卡片按钮必须配置一个 behaviors")
        behavior = behaviors[0]
        if not isinstance(behavior, Mapping):
            raise ValueError("飞书卡片按钮 behaviors 格式无效")
        behavior_type = behavior.get("type")
        if behavior_type == "callback":
            if not isinstance(behavior.get("value"), Mapping):
                raise ValueError("飞书卡片 callback behaviors 必须包含对象 value")
        elif behavior_type == "open_url":
            if not isinstance(behavior.get("default_url"), str) or not behavior["default_url"].strip():
                raise ValueError("飞书卡片 open_url behaviors 必须包含 default_url")
        else:
            raise ValueError("飞书卡片按钮 behaviors 类型不受支持")


def _walk_mappings(value: Any):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_mappings(child)


def _count_elements(value: Any) -> int:
    if isinstance(value, Mapping):
        return (1 if "tag" in value else 0) + sum(_count_elements(item) for item in value.values())
    if isinstance(value, list):
        return sum(_count_elements(item) for item in value)
    return 0
