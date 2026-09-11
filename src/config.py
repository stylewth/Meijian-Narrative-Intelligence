from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from dotenv import dotenv_values

from src.string_enum import StringEnum


class AppMode(StringEnum):
    ONLINE = "online"
    REPLAY = "replay"


class LLMResponseFormat(StringEnum):
    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"


@dataclass(frozen=True, slots=True)
class Settings:
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_timeout_seconds: float = 60
    llm_max_retries: int = 2
    llm_max_tokens: int | None = None
    auditor_base_url: str | None = None
    auditor_api_key: str | None = None
    auditor_model: str | None = None
    auditor_timeout_seconds: float = 300
    auditor_max_tokens: int | None = None
    auditor_thinking_enabled: bool = False
    auditor_reasoning_effort: Literal["high", "max"] | None = None
    llm_response_format: LLMResponseFormat = LLMResponseFormat.JSON_SCHEMA
    llm_thinking_enabled: bool = False
    llm_reasoning_effort: Literal["high", "max"] | None = None
    max_records: int = 500

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.from_sources()

    @classmethod
    def from_sources(
        cls,
        *,
        secrets: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
        dotenv_path: str | Path | None = ".env",
    ) -> "Settings":
        dotenv_data = dict(dotenv_values(dotenv_path)) if dotenv_path else {}
        secret_data = dict(secrets or {})
        environment_data = dict(os.environ if environ is None else environ)

        def value(name: str, default: str | None = None) -> str | None:
            selected = environment_data.get(name)
            if selected in {None, ""}:
                selected = secret_data.get(name)
            if selected in {None, ""}:
                selected = dotenv_data.get(name)
            if selected in {None, ""}:
                return default
            return str(selected)

        response_format_value = value("LLM_RESPONSE_FORMAT", "json_schema")
        try:
            response_format = LLMResponseFormat(response_format_value)
        except ValueError as exc:
            raise ValueError(
                "LLM_RESPONSE_FORMAT 必须是 json_schema 或 json_object"
            ) from exc

        thinking_enabled_value = value("LLM_THINKING_ENABLED", "false")
        if thinking_enabled_value not in {"true", "false"}:
            raise ValueError("LLM_THINKING_ENABLED must be true or false")
        thinking_enabled = thinking_enabled_value == "true"

        reasoning_effort_value = value("LLM_REASONING_EFFORT")
        if reasoning_effort_value is not None and reasoning_effort_value not in {
            "high",
            "max",
        }:
            raise ValueError("LLM_REASONING_EFFORT must be high or max")

        max_tokens_value = value("LLM_MAX_TOKENS")
        llm_max_tokens = int(max_tokens_value) if max_tokens_value else None
        if llm_max_tokens is not None and llm_max_tokens <= 0:
            raise ValueError("LLM_MAX_TOKENS 必须为正整数")

        auditor_max_tokens_value = value("AUDITOR_MAX_TOKENS")
        auditor_max_tokens = int(auditor_max_tokens_value) if auditor_max_tokens_value else None
        if auditor_max_tokens is not None and auditor_max_tokens <= 0:
            raise ValueError("AUDITOR_MAX_TOKENS 必须为正整数")
        auditor_thinking_value = value("AUDITOR_THINKING_ENABLED", "false")
        if auditor_thinking_value not in {"true", "false"}:
            raise ValueError("AUDITOR_THINKING_ENABLED must be true or false")
        auditor_reasoning_value = value("AUDITOR_REASONING_EFFORT")
        if auditor_reasoning_value is not None and auditor_reasoning_value not in {"high", "max"}:
            raise ValueError("AUDITOR_REASONING_EFFORT must be high or max")

        return cls(
            llm_api_key=value("LLM_API_KEY"),
            llm_base_url=value("LLM_BASE_URL"),
            llm_model=value("LLM_MODEL"),
            llm_timeout_seconds=float(value("LLM_TIMEOUT_SECONDS", "60")),
            llm_max_retries=int(value("LLM_MAX_RETRIES", "2")),
            llm_max_tokens=llm_max_tokens,
            llm_response_format=response_format,
            llm_thinking_enabled=thinking_enabled,
            llm_reasoning_effort=reasoning_effort_value,
            auditor_base_url=value("AUDITOR_BASE_URL"),
            auditor_api_key=value("AUDITOR_API_KEY"),
            auditor_model=value("AUDITOR_MODEL"),
            auditor_timeout_seconds=float(value("AUDITOR_TIMEOUT_SECONDS", "300")),
            auditor_max_tokens=auditor_max_tokens,
            auditor_thinking_enabled=auditor_thinking_value == "true",
            auditor_reasoning_effort=auditor_reasoning_value,
            max_records=int(value("MAX_RECORDS", "500")),
        )

    def auditor_client_settings(self) -> "Settings | None":
        """压力测试攻击方（审计者）的 API 配置口。

        配置 AUDITOR_API_KEY + AUDITOR_MODEL 后返回一个可直接构造
        LLMClient 的 Settings（llm_* 字段被审计者配置替换）；未配置时
        返回 None，由调用方决定离线代跑方式（例如 ZCode 子代理）。
        """

        if not self.auditor_api_key or not self.auditor_model:
            return None
        return Settings(
            llm_api_key=self.auditor_api_key,
            llm_base_url=self.auditor_base_url,
            llm_model=self.auditor_model,
            llm_timeout_seconds=self.auditor_timeout_seconds,
            llm_max_retries=1,
            llm_response_format=LLMResponseFormat.JSON_OBJECT,
            llm_thinking_enabled=self.auditor_thinking_enabled,
            llm_reasoning_effort=self.auditor_reasoning_effort,
            llm_max_tokens=self.auditor_max_tokens,
        )

    def validate_official_decision_profile(self) -> None:
        expected_values = {
            "LLM_BASE_URL": self.llm_base_url == "https://api.deepseek.com",
            "LLM_MODEL": self.llm_model == "deepseek-v4-flash",
            "LLM_RESPONSE_FORMAT": self.llm_response_format
            is LLMResponseFormat.JSON_OBJECT,
            "LLM_THINKING_ENABLED": self.llm_thinking_enabled is True,
            "LLM_REASONING_EFFORT": self.llm_reasoning_effort == "max",
            "LLM_MAX_RETRIES": self.llm_max_retries == 0,
        }
        for field_name, is_valid in expected_values.items():
            if not is_valid:
                raise ValueError(f"正式决策配置要求 {field_name} 精确匹配")

    def validate_mode(self, mode: AppMode) -> None:
        if mode is AppMode.ONLINE and not self.llm_api_key:
            raise ValueError("在线分析模式缺少 LLM_API_KEY")
        if mode is AppMode.ONLINE and not self.llm_model:
            raise ValueError("在线分析模式缺少 LLM_MODEL")
