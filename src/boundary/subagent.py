"""Isolated sub-agent runner with consolidated budget accounting (M9 — R11, R12).

Rules:
- R11 (Isolated Context): The sub-agent never receives the parent agent's message history.
  It receives only the targeted query/file instruction and sandbox tools, returning a single
  concise `ToolResult` (<= 300 tokens) to the parent agent.
- R12 (Consolidated Budget): Every prompt and completion token spent by the sub-agent is
  reported back and deducted from the parent agent's global `token_budget` and recorded in
  `Trace.total_usage`.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from boundary.llm import LLMClient
from boundary.models import (
    StopReason,
    ToolCall,
    ToolResult,
    Usage,
    UsageSource,
    sanitize_error_message,
)
from boundary.tools import TOOL_SCHEMAS, TRUNCATED_SUFFIX, ToolSandbox, estimate_tokens

DEFAULT_SUBAGENT_MAX_STEPS: int = 3
DEFAULT_SUBAGENT_BUDGET: int = 8_000
DEFAULT_MAX_SUMMARY_TOKENS: int = 300

SUBAGENT_SYSTEM_PROMPT: str = (
    "You are an isolated inspection sub-agent. "
    "You do not have access to the parent conversation. "
    "Use the read-only sandbox tools (list_files, read_file, grep) to locate the exact "
    "information requested, then reply with a concise summary (at most 300 tokens) "
    "containing only the relevant facts and values."
)


class SubagentRunResult(BaseModel):
    """Result of an isolated SubagentRunner execution, including consolidated token usage."""

    tool_result: ToolResult
    usage: Usage
    steps_taken: int
    stop_reason: StopReason
    error: str | None = None


def _cap_summary_text(text: str, max_tokens: int) -> str:
    """Ensure sub-agent output returned to parent agent is at most `max_tokens` (chars // 4)."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - len(TRUNCATED_SUFFIX))
    return text[:keep] + TRUNCATED_SUFFIX


class SubagentRunner:
    """Executes an isolated mini agent loop and returns a bounded summary ToolResult (R11, R12)."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        sandbox: ToolSandbox,
        subagent_max_steps: int = DEFAULT_SUBAGENT_MAX_STEPS,
        subagent_budget: int = DEFAULT_SUBAGENT_BUDGET,
        max_summary_tokens: int = DEFAULT_MAX_SUMMARY_TOKENS,
    ) -> None:
        if subagent_max_steps < 1:
            raise ValueError("subagent_max_steps must be >= 1.")
        if subagent_budget < 1:
            raise ValueError("subagent_budget must be >= 1.")
        if max_summary_tokens < 1:
            raise ValueError("max_summary_tokens must be >= 1.")

        self.llm = llm
        self.sandbox = sandbox
        self.subagent_max_steps = subagent_max_steps
        self.subagent_budget = subagent_budget
        self.max_summary_tokens = max_summary_tokens

    def execute_tool_call(
        self,
        tc: ToolCall,
        *,
        fallback_query: str = "",
        remaining_global_budget: int | None = None,
        seed: int | None = None,
    ) -> SubagentRunResult:
        """Execute a delegated or intercepted ToolCall inside an isolated sub-agent context."""
        if "__json_error__" in tc.arguments:
            err_msg = (
                "Tool error (subagent): invalid JSON arguments "
                f"({tc.arguments['__json_error__']})."
            )
            return SubagentRunResult(
                tool_result=ToolResult(
                    call_id=tc.id,
                    content=err_msg,
                    size_chars=len(err_msg),
                    tokens_est=estimate_tokens(err_msg),
                ),
                usage=Usage(prompt_tokens=0, completion_tokens=0, source="api"),
                steps_taken=0,
                stop_reason="error",
                error=err_msg,
            )

        raw_path = tc.arguments.get("path")
        path = str(raw_path) if isinstance(raw_path, str) and raw_path.strip() else None

        raw_query = tc.arguments.get("query") or tc.arguments.get("pattern")
        if isinstance(raw_query, str) and raw_query.strip():
            query = raw_query.strip()
        elif fallback_query.strip():
            query = fallback_query.strip()
        else:
            query = "Inspect the target file and extract the key values concisely."

        return self.run(
            call_id=tc.id,
            query=query,
            path=path,
            remaining_global_budget=remaining_global_budget,
            seed=seed,
        )

    def run(
        self,
        *,
        call_id: str,
        query: str,
        path: str | None = None,
        remaining_global_budget: int | None = None,
        seed: int | None = None,
    ) -> SubagentRunResult:
        """Run an isolated sub-agent loop for a delegated inspection/search task.

        The parent agent's message history is never passed here (R11).
        All prompt + completion tokens consumed inside this loop are returned in
        `SubagentRunResult.usage` so the parent agent deducts them from its global budget (R12).
        """
        effective_budget = self.subagent_budget
        if remaining_global_budget is not None:
            effective_budget = min(self.subagent_budget, max(0, remaining_global_budget))

        user_prompt = f"Target path: {path}\nQuery: {query}" if path else f"Query: {query}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt_tokens = 0
        completion_tokens = 0
        any_estimated = False
        steps_taken = 0
        stop_reason: StopReason = "max_steps"
        error_text: str | None = None
        summary_content: str | None = None

        while True:
            if prompt_tokens + completion_tokens >= effective_budget:
                stop_reason = "budget"
                summary_content = (
                    "Tool error (subagent): subagent token budget exceeded before completion."
                )
                break

            if steps_taken >= self.subagent_max_steps:
                stop_reason = "max_steps"
                summary_content = (
                    f"Tool error (subagent): subagent reached max_steps "
                    f"({self.subagent_max_steps}) without a final summary."
                )
                break

            try:
                response = self.llm.complete(
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    seed=seed,
                )
            except Exception as exc:
                stop_reason = "error"
                error_text = sanitize_error_message(str(exc))
                summary_content = f"Tool error (subagent): {error_text}"
                break

            steps_taken += 1
            prompt_tokens += response.usage.prompt_tokens
            completion_tokens += response.usage.completion_tokens
            if response.usage.source == "estimated":
                any_estimated = True

            if not response.tool_calls:
                stop_reason = "final"
                summary_content = response.assistant_content or ""
                break

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.assistant_content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            messages.append(assistant_msg)

            for tc in response.tool_calls:
                result = self.sandbox.execute(tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.content,
                    }
                )

        bounded_text = _cap_summary_text(summary_content or "", self.max_summary_tokens)
        tool_result = ToolResult(
            call_id=call_id,
            content=bounded_text,
            size_chars=len(bounded_text),
            tokens_est=estimate_tokens(bounded_text),
        )
        usage_source: UsageSource = "estimated" if any_estimated else "api"
        return SubagentRunResult(
            tool_result=tool_result,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                source=usage_source,
            ),
            steps_taken=steps_taken,
            stop_reason=stop_reason,
            error=error_text,
        )
