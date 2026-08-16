from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar, cast

from openai import OpenAI
from pydantic import BaseModel

from src.config import AppMode, LLMResponseFormat, Settings


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def _hydrate_evidence_references(
    payload: Any,
    schema: dict[str, Any],
    evidence_catalog: Mapping[str, str],
) -> Any:
    definitions = schema.get("$defs", {})

    def hydrate_evidence_quote(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        if set(value) != {"comment_id"}:
            raise ValueError("证据引用只能包含 comment_id")
        comment_id = value["comment_id"]
        if not isinstance(comment_id, str) or comment_id not in evidence_catalog:
            raise ValueError(f"证据 comment_id 不存在: {comment_id!r}")
        return {"comment_id": comment_id, "quote": evidence_catalog[comment_id]}

    def hydrate(value: Any, schema_fragment: Any) -> Any:
        if not isinstance(schema_fragment, dict):
            return value
        reference = schema_fragment.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/$defs/"):
            definition_name = reference.removeprefix("#/$defs/")
            if definition_name == "EvidenceQuote":
                return hydrate_evidence_quote(value)
            return hydrate(value, definitions.get(definition_name))
        if isinstance(value, list):
            return [
                hydrate(item, schema_fragment.get("items")) for item in value
            ]
        if isinstance(value, dict):
            properties = schema_fragment.get("properties", {})
            return {
                key: hydrate(item, properties.get(key))
                for key, item in value.items()
            }
        return value

    return hydrate(payload, schema)


class _Completions(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _Chat(Protocol):
    completions: _Completions


class _OpenAICompatibleClient(Protocol):
    chat: _Chat


class LLMClient:
    def __init__(
        self,
        settings: Settings,
        *,
        client: _OpenAICompatibleClient | None = None,
    ) -> None:
        settings.validate_mode(AppMode.ONLINE)
        self._model = cast(str, settings.llm_model)
        self._response_format = settings.llm_response_format
        self._thinking_enabled = settings.llm_thinking_enabled
        self._reasoning_effort = settings.llm_reasoning_effort or "max"
        if client is None:
            client_kwargs: dict[str, Any] = {
                "api_key": settings.llm_api_key,
                "timeout": settings.llm_timeout_seconds,
                "max_retries": settings.llm_max_retries,
            }
            if settings.llm_base_url:
                client_kwargs["base_url"] = settings.llm_base_url
            client = OpenAI(**client_kwargs)
        self._client = client

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseModel],
        evidence_catalog: Mapping[str, str] | None = None,
    ) -> ResponseModel:
        schema = response_model.model_json_schema()
        if evidence_catalog is not None:
            evidence_quote_schema = schema.get("$defs", {}).get("EvidenceQuote")
            if not isinstance(evidence_quote_schema, dict):
                raise TypeError("response_model 缺少 EvidenceQuote 定义")
            evidence_quote_schema["properties"] = {
                "comment_id": evidence_quote_schema["properties"]["comment_id"]
            }
            evidence_quote_schema["required"] = ["comment_id"]
        selected_system_prompt = system_prompt
        if self._response_format is LLMResponseFormat.JSON_SCHEMA:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }
        elif self._response_format is LLMResponseFormat.JSON_OBJECT:
            schema_json = json.dumps(
                schema,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            selected_system_prompt = (
                f"{system_prompt}\n\n"
                "只返回一个 JSON 对象，必须满足以下 JSON Schema：\n"
                f"{schema_json}"
            )
            response_format = {"type": "json_object"}
        else:
            raise ValueError(f"不支持的 LLM_RESPONSE_FORMAT：{self._response_format}")

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": selected_system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": response_format,
        }
        if self._thinking_enabled:
            request_kwargs["reasoning_effort"] = self._reasoning_effort
            request_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        response = self._client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("模型返回内容为空")
        payload = json.loads(content)
        if evidence_catalog is not None:
            payload = _hydrate_evidence_references(payload, schema, evidence_catalog)
        if response_model.model_config.get("extra") != "forbid":
            raise TypeError("response_model 必须配置 extra='forbid'")
        # Validate the JSON representation so enum values can be parsed from
        # their wire-format strings while strict numeric/type checks remain in
        # force. ``model_validate`` on the already-decoded dict would reject
        # every JSON enum value because strict mode expects enum instances.
        return response_model.model_validate_json(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            strict=True,
        )
