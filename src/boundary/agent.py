"""Single-agent baseline loop with token budget and step limits (M3).

Token Budget Rule (R2):
The token budget is checked BEFORE each LLM call (`total_tokens >= token_budget`).
If the accumulated usage from prior steps is greater than or equal to `token_budget`,
the loop terminates before making another LLM request with `stop_reason="budget"`.
An in-flight LLM call is never interrupted mid-call, and its actual token consumption
is always recorded in `Trace.total_usage`.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import Any

from boundary.llm import LLMClient
from boundary.models import (
    Step,
    StopReason,
    Task,
    ToolResult,
    Trace,
    Usage,
    UsageSource,
    compute_config_hash,
    sanitize_error_message,
)
from boundary.tools import TOOL_NAMES, TOOL_SCHEMAS, ToolSandbox

SYSTEM_PROMPT: str = (
    "You are a precise repository question-answering agent. "
    "Use the provided read-only tools (list_files, read_file, grep) to inspect the codebase "
    "and answer the user's question concisely based strictly on repository contents."
)


def run_agent(
    *,
    task: Task,
    llm: LLMClient,
    sandbox: ToolSandbox,
    token_budget: int = 40_000,
    trial: int = 1,
    seed: int | None = None,
    parameters: dict[str, Any] | None = None,
    answer_checker: Callable[[str | None], bool] | None = None,
    trace_id: str | None = None,
) -> Trace:
    """Execute the single-agent baseline loop for one task trial.

    Token budget is checked BEFORE each LLM call
    (`total >= token_budget` -> `stop_reason="budget"`).
    Ongoing LLM calls are never interrupted; actual token usage is recorded in `Trace.total_usage`.
    Invalid tool calls or malformed JSON arguments are returned as readable `ToolResult` errors
    so the agent can recover; `stop_reason="error"` is reserved for LLM/infrastructure exceptions.
    """
    config_hash = compute_config_hash(
        model=llm.model,
        system_prompt=SYSTEM_PROMPT,
        tools=TOOL_NAMES,
        task_id=task.id,
        token_budget=token_budget,
        max_steps=task.max_steps,
        max_output_chars=sandbox.max_output_chars,
        grep_max_matches=sandbox.grep_max_matches,
        grep_max_line_chars=sandbox.grep_max_line_chars,
        grep_max_file_bytes=sandbox.grep_max_file_bytes,
        seed=seed,
        parameters=parameters,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]

    steps: list[Step] = []
    prompt_tokens = 0
    completion_tokens = 0
    any_estimated = False
    final_answer: str | None = None
    stop_reason: StopReason = "max_steps"
    error_text: str | None = None

    step_index = 0
    while True:
        if prompt_tokens + completion_tokens >= token_budget:
            stop_reason = "budget"
            break

        if step_index >= task.max_steps:
            stop_reason = "max_steps"
            break

        t0 = time.perf_counter()
        try:
            response = llm.complete(messages=messages, tools=TOOL_SCHEMAS, seed=seed)
        except Exception as exc:
            stop_reason = "error"
            error_text = sanitize_error_message(str(exc))
            break
        latency_ms = max(0, int((time.perf_counter() - t0) * 1000))

        prompt_tokens += response.usage.prompt_tokens
        completion_tokens += response.usage.completion_tokens
        if response.usage.source == "estimated":
            any_estimated = True

        if not response.tool_calls:
            steps.append(
                Step(
                    index=step_index,
                    assistant_content=response.assistant_content,
                    tool_calls=[],
                    tool_results=[],
                    usage=response.usage,
                    latency_ms=latency_ms,
                )
            )
            final_answer = response.assistant_content
            stop_reason = "final"
            break

        # Assistant requested one or more tool calls; execute all of them sequentially.
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

        tool_results: list[ToolResult] = []
        for tc in response.tool_calls:
            result = sandbox.execute(tc)
            tool_results.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.content,
                }
            )

        steps.append(
            Step(
                index=step_index,
                assistant_content=response.assistant_content,
                tool_calls=list(response.tool_calls),
                tool_results=tool_results,
                usage=response.usage,
                latency_ms=latency_ms,
            )
        )
        step_index += 1

    usage_source: UsageSource = "estimated" if any_estimated else "api"
    total_usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        source=usage_source,
    )

    passed = False
    if stop_reason == "final" and answer_checker is not None:
        passed = bool(answer_checker(final_answer))

    return Trace(
        trace_id=trace_id or f"tr_{uuid.uuid4().hex[:12]}",
        task_id=task.id,
        trial=trial,
        agent_variant="baseline",
        model=llm.model,
        config_hash=config_hash,
        steps=steps,
        final_answer=final_answer,
        passed=passed,
        total_usage=total_usage,
        stop_reason=stop_reason,
        error=error_text,
    )
