"""Single-agent and subagent-delegated loop with token budget and step limits (M3, M9, M11).

Token Budget Rule (R2, R12):
The token budget is checked BEFORE each LLM call (`total_tokens >= token_budget`).
If the accumulated usage from prior steps (including all tokens spent by any isolated
`SubagentRunner` per R12) is greater than or equal to `token_budget`, the loop terminates
before making another LLM request with `stop_reason="budget"`.
An in-flight LLM call is never interrupted mid-call, and its actual token consumption
is always recorded in `Trace.total_usage`.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from boundary.llm import LLMClient
from boundary.models import (
    Step,
    StopReason,
    Task,
    ToolCall,
    ToolResult,
    Trace,
    Usage,
    UsageSource,
    compute_config_hash,
    sanitize_error_message,
)
from boundary.tools import TOOL_NAMES, TOOL_SCHEMAS, ToolSandbox

if TYPE_CHECKING:
    from boundary.subagent import SubagentRunner

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
    subagent_runner: SubagentRunner | None = None,
    should_delegate: Callable[[ToolCall], bool] | None = None,
    agent_variant: str = "baseline",
    tool_schemas: list[dict[str, Any]] | None = None,
) -> Trace:
    """Execute the agent loop for one task trial (baseline or subagent variant).

    Token budget is checked BEFORE each LLM call
    (`total >= token_budget` -> `stop_reason="budget"`).
    When `subagent_runner` is active and `should_delegate(tc)` is True, the tool call is
    routed to an isolated `SubagentRunner` (R11) and all sub-agent prompt + completion tokens
    are added to the parent agent's global token counters and `Trace.total_usage` (R12).
    """
    active_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
    active_tool_names = [
        str(s.get("function", {}).get("name", ""))
        for s in active_schemas
        if isinstance(s.get("function"), dict)
    ] or list(TOOL_NAMES)

    extra_params = dict(parameters or {})
    if agent_variant != "baseline":
        extra_params["agent_variant"] = agent_variant

    config_hash = compute_config_hash(
        model=llm.model,
        system_prompt=SYSTEM_PROMPT,
        tools=active_tool_names,
        task_id=task.id,
        token_budget=token_budget,
        max_steps=task.max_steps,
        max_output_chars=sandbox.max_output_chars,
        grep_max_matches=sandbox.grep_max_matches,
        grep_max_line_chars=sandbox.grep_max_line_chars,
        grep_max_file_bytes=sandbox.grep_max_file_bytes,
        seed=seed,
        parameters=extra_params,
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
            response = llm.complete(messages=messages, tools=active_schemas, seed=seed)
        except Exception as exc:
            stop_reason = "error"
            error_text = sanitize_error_message(str(exc))
            break
        latency_ms = max(0, int((time.perf_counter() - t0) * 1000))

        step_prompt_tokens = response.usage.prompt_tokens
        step_completion_tokens = response.usage.completion_tokens
        step_estimated = response.usage.source == "estimated"

        prompt_tokens += step_prompt_tokens
        completion_tokens += step_completion_tokens
        if step_estimated:
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
            delegate = subagent_runner is not None and (
                should_delegate is None or should_delegate(tc)
            )
            if delegate and subagent_runner is not None:
                remaining_budget = max(0, token_budget - (prompt_tokens + completion_tokens))
                sub_res = subagent_runner.execute_tool_call(
                    tc,
                    fallback_query=task.prompt,
                    remaining_global_budget=remaining_budget,
                    seed=seed,
                )
                result = sub_res.tool_result
                prompt_tokens += sub_res.usage.prompt_tokens
                completion_tokens += sub_res.usage.completion_tokens
                step_prompt_tokens += sub_res.usage.prompt_tokens
                step_completion_tokens += sub_res.usage.completion_tokens
                if sub_res.usage.source == "estimated":
                    step_estimated = True
                    any_estimated = True
            else:
                result = sandbox.execute(tc)

            tool_results.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.content,
                }
            )

        step_usage_source: UsageSource = "estimated" if step_estimated else "api"
        steps.append(
            Step(
                index=step_index,
                assistant_content=response.assistant_content,
                tool_calls=list(response.tool_calls),
                tool_results=tool_results,
                usage=Usage(
                    prompt_tokens=step_prompt_tokens,
                    completion_tokens=step_completion_tokens,
                    source=step_usage_source,
                ),
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
        agent_variant=agent_variant,
        model=llm.model,
        config_hash=config_hash,
        steps=steps,
        final_answer=final_answer,
        passed=passed,
        total_usage=total_usage,
        stop_reason=stop_reason,
        error=error_text,
    )
