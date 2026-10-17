"""LLM client protocol, OpenAI-compatible client, and offline FakeLLM (M3)."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI
from pydantic import BaseModel, Field

from boundary.models import ToolCall, Usage, sanitize_error_message


class LLMError(RuntimeError):
    """Sanitized LLM or infrastructure error."""


class LLMResponse(BaseModel):
    """Normalized single-turn response from an LLM client."""

    assistant_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage


def parse_tool_arguments(raw_args: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse tool call arguments without raising on malformed JSON."""
    if raw_args is None or raw_args == "":
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return {
            "__raw_invalid_json__": raw_args,
            "__json_error__": exc.msg,
        }
    if not isinstance(parsed, dict):
        return {
            "__raw_invalid_json__": raw_args,
            "__json_error__": "expected a JSON object",
        }
    return parsed


def estimate_usage(
    messages: list[dict[str, Any]],
    assistant_content: str | None,
    tool_calls: list[ToolCall],
) -> Usage:
    """Estimate prompt and completion tokens via chars // 4 when API usage is missing."""
    prompt_chars = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            prompt_chars += len(content)
        if "tool_calls" in msg:
            prompt_chars += len(json.dumps(msg["tool_calls"], ensure_ascii=False))

    completion_chars = len(assistant_content or "")
    for tc in tool_calls:
        completion_chars += len(tc.name) + len(json.dumps(tc.arguments, ensure_ascii=False))

    return Usage(
        prompt_tokens=max(1, prompt_chars // 4) if prompt_chars > 0 else 0,
        completion_tokens=max(1, completion_chars // 4) if completion_chars > 0 else 0,
        source="estimated",
    )


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM chat completion clients."""

    @property
    def model(self) -> str:
        """Model identifier."""
        ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        """Execute a single chat completion turn."""
        ...


@dataclass
class FakeTurn:
    """Scripted turn for FakeLLM; if usage is None, usage is estimated via chars // 4."""

    assistant_content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage | None = None


class FakeLLM:
    """Deterministic, network-free LLM client for unit and integration tests."""

    def __init__(
        self,
        script: Sequence[LLMResponse | FakeTurn | Exception],
        *,
        model: str = "fake-llm",
    ) -> None:
        self._script: list[LLMResponse | FakeTurn | Exception] = list(script)
        self._index: int = 0
        self._model: str = model
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        """Return next scripted response or raise sanitized LLMError."""
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "seed": seed,
            }
        )
        if self._index >= len(self._script):
            raise LLMError("FakeLLM script exhausted.")

        item = self._script[self._index]
        self._index += 1

        if isinstance(item, Exception):
            raise LLMError(sanitize_error_message(str(item)))

        if isinstance(item, LLMResponse):
            return item

        usage = (
            item.usage
            if item.usage is not None
            else estimate_usage(messages, item.assistant_content, item.tool_calls)
        )
        return LLMResponse(
            assistant_content=item.assistant_content,
            tool_calls=list(item.tool_calls),
            usage=usage,
        )


class OpenAICompatClient:
    """OpenAI-compatible client supporting custom base_url and error sanitization (R7)."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("BOUNDARY_API_KEY", "")
        resolved_url = base_url if base_url is not None else os.environ.get("BOUNDARY_BASE_URL")
        resolved_model = (
            model if model is not None else os.environ.get("BOUNDARY_MODEL", "gpt-4o-mini")
        )

        self._api_key: str = resolved_key
        self._base_url: str | None = resolved_url or None
        self._model: str = resolved_model
        self.temperature: float = temperature
        self._client = OpenAI(
            api_key=self._api_key or "missing-api-key",
            base_url=self._base_url,
        )

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    def __repr__(self) -> str:
        return (
            f"OpenAICompatClient(model={self._model!r}, "
            f"base_url={self._base_url!r}, api_key='[REDACTED]')"
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        seed: int | None = None,
    ) -> LLMResponse:
        """Call chat.completions.create and normalize response; scrub secrets on error."""
        if not self._api_key:
            raise LLMError("BOUNDARY_API_KEY is not set.")

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if seed is not None:
            kwargs["seed"] = seed

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc), secrets=[self._api_key])
            raise LLMError(sanitized) from None

        if not response.choices:
            raise LLMError("LLM returned empty choices list.")

        choice_msg = response.choices[0].message
        assistant_content: str | None = choice_msg.content

        parsed_tool_calls: list[ToolCall] = []
        if choice_msg.tool_calls:
            for tc in choice_msg.tool_calls:
                func = getattr(tc, "function", None)
                if func is None:
                    continue
                parsed_tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=func.name,
                        arguments=parse_tool_arguments(func.arguments),
                    )
                )

        if (
            response.usage is not None
            and response.usage.prompt_tokens is not None
            and response.usage.completion_tokens is not None
        ):
            usage = Usage(
                prompt_tokens=int(response.usage.prompt_tokens),
                completion_tokens=int(response.usage.completion_tokens),
                source="api",
            )
        else:
            usage = estimate_usage(messages, assistant_content, parsed_tool_calls)

        return LLMResponse(
            assistant_content=assistant_content,
            tool_calls=parsed_tool_calls,
            usage=usage,
        )
