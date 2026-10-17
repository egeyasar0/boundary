"""Deterministic trace profiler for context carry cost, reuse ratio, and parallelism (M7).

Rules (R8, R9, R10):
- R8: Reads only trace data; makes zero network and zero LLM calls.
- R9: Outputs are upper-bound estimates/candidates with `confidence="low"`, not proof.
- R10: Every signal formula is documented in its docstring and unit-tested on synthetic traces.
"""

from __future__ import annotations

import json
import keyword
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from boundary.agent import SYSTEM_PROMPT
from boundary.models import Trace, UsageSource

TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[A-Za-z0-9_\-\.]{4,}")
PYTHON_KEYWORDS: frozenset[str] = frozenset(
    k.lower() for k in (*keyword.kwlist, *getattr(keyword, "softkwlist", ()))
)
DEFAULT_HIGH_FREQ_RATIO: float = 0.30
DEFAULT_MIN_EXTRACT_TOKENS: int = 2000
DEFAULT_REUSE_THRESHOLD: float = 0.1


class ToolResultProfile(BaseModel):
    """Profile metrics for a single ToolResult inside a Trace."""

    trace_id: str
    step_index: int
    call_id: str
    tool: str
    size_tokens: int
    remaining_calls: int
    carry_cost_tokens: int
    reuse_ratio: float


class TraceProfile(BaseModel):
    """Profile metrics for an entire Trace."""

    trace_id: str
    task_id: str
    num_calls: int
    carry_total_tokens: int
    carry_share: float
    usage_source: UsageSource
    results: list[ToolResultProfile]


class Opportunity(BaseModel):
    """Candidate boundary optimization opportunity (always confidence='low' per R9)."""

    id: str
    kind: Literal["extract_subagent", "parallelize"]
    task_id: str
    trace_ids: list[str]
    evidence: dict[str, Any]
    est_upper_bound_savings_tokens: int | None
    steps_saved: int | None
    confidence: Literal["low"] = "low"


class ProfileSummary(BaseModel):
    """Complete deterministic profiler output serialized to profile.json."""

    traces: list[TraceProfile] = Field(default_factory=list)
    opportunities: list[Opportunity] = Field(default_factory=list)


def extract_signal_tokens(
    text: str,
    *,
    excluded_tokens: frozenset[str] = frozenset(),
) -> set[str]:
    """Extract candidate signal tokens matching `[A-Za-z0-9_\\-\\.]{4,}`.

    Formula (S2 token extraction):
        tokens(text) = {
            m.lower() for m in re.findall(r"[A-Za-z0-9_\\-\\.]{4,}", text)
            if m.lower() not in PYTHON_KEYWORDS and m.lower() not in excluded_tokens
        }
    """
    if not text:
        return set()
    raw = TOKEN_PATTERN.findall(text)
    return {
        tok.lower()
        for tok in raw
        if tok.lower() not in PYTHON_KEYWORDS and tok.lower() not in excluded_tokens
    }


def compute_high_frequency_tokens(
    traces: Sequence[Trace],
    *,
    ratio_threshold: float = DEFAULT_HIGH_FREQ_RATIO,
) -> frozenset[str]:
    """Find tokens appearing in more than `ratio_threshold` (30%) of all tool results in a run.

    Formula (S2 corpus frequency filter):
        Given M total tool results across all traces in the run, let df(t) be the number
        of tool results containing token t. Token t is excluded if df(t) > 1 and
        df(t) / M > 0.30.
    """
    doc_counts: Counter[str] = Counter()
    total_results = 0

    for trace in traces:
        for step in trace.steps:
            for tr in step.tool_results:
                total_results += 1
                unique_in_result = extract_signal_tokens(tr.content)
                for tok in unique_in_result:
                    doc_counts[tok] += 1

    if total_results == 0:
        return frozenset()

    return frozenset(
        tok
        for tok, count in doc_counts.items()
        if count > 1 and (count / total_results) > ratio_threshold
    )


def compute_carry_cost(
    tokens_est: int,
    step_index: int,
    num_calls: int,
) -> tuple[int, int]:
    """Compute Signal S1 (carry cost) for a tool result produced at `step_index`.

    Formula (Signal S1 — Carry Cost):
        N = len(steps)  (total LLM calls in the trace)
        remaining = max(0, N - 1 - s)  (subsequent LLM calls where result is carried in prompt)
        carry_cost_tokens(result) = tokens_est(result) * remaining
        carry_total = sum(carry_cost_tokens(r) for r in trace_results)
        carry_share = carry_total / sum(step.usage.prompt_tokens for step in trace.steps)

    Returns:
        (remaining_calls, carry_cost_tokens)
    """
    remaining = max(0, num_calls - 1 - step_index)
    return remaining, tokens_est * remaining


def compute_reuse_ratio(
    result_content: str,
    downstream_texts: Sequence[str],
    *,
    excluded_tokens: frozenset[str] = frozenset(),
) -> float:
    """Compute Signal S2 (reuse ratio) for a tool result against downstream step texts.

    Formula (Signal S2 — Reuse Ratio):
        tokens(text) = matches of `[A-Za-z0-9_\\-\\.]{4,}` excluding Python keywords
                       and tokens appearing in >30% of tool results in the run.
        downstream = union of tokens(text) from subsequent steps' `assistant_content`,
                     `tool_calls.arguments`, and trace `final_answer`.
        non_empty_lines = [line for line in result_content.splitlines() if line.strip()]
        reused_lines = [
            line for line in non_empty_lines
            if len(tokens(line) & downstream) >= 1
        ]
        reuse_ratio = len(reused_lines) / len(non_empty_lines)  (0.0 if non_empty_lines is empty)
    """
    non_empty_lines = [line for line in result_content.splitlines() if line.strip()]
    if not non_empty_lines:
        return 0.0

    downstream_tokens: set[str] = set()
    for text in downstream_texts:
        downstream_tokens.update(extract_signal_tokens(text, excluded_tokens=excluded_tokens))

    if not downstream_tokens:
        return 0.0

    reused_count = 0
    for line in non_empty_lines:
        line_tokens = extract_signal_tokens(line, excluded_tokens=excluded_tokens)
        if line_tokens & downstream_tokens:
            reused_count += 1

    return reused_count / len(non_empty_lines)


def _collect_downstream_texts(trace: Trace, step_index: int) -> list[str]:
    """Collect assistant_content, tool_calls.arguments, and final_answer after step_index."""
    texts: list[str] = []
    for step in trace.steps:
        if step.index <= step_index:
            continue
        if step.assistant_content:
            texts.append(step.assistant_content)
        for tc in step.tool_calls:
            texts.append(json.dumps(tc.arguments, ensure_ascii=False))
    if trace.final_answer:
        texts.append(trace.final_answer)
    return texts


def find_independent_step_groups(
    trace: Trace,
    *,
    task_prompt: str = "",
    system_prompt: str = SYSTEM_PROMPT,
    excluded_tokens: frozenset[str] = frozenset(),
) -> list[list[int]]:
    """Compute Signal S3 (independence / parallelizability) across consecutive tool steps.

    Formula (Signal S3 — Step Independence):
        base_tokens = tokens(task_prompt) | tokens(system_prompt)
        produced(i) = union(tokens(r.content) for r in step[i].tool_results) - base_tokens
        args(j) = union(tokens(json.dumps(tc.arguments)) for tc in step[j].tool_calls)
        depends(j, i) (for i < j) <=> len(args(j) & produced(i)) > 0

        Consecutive tool-calling steps are grouped greedily: step j joins the current
        contiguous group G if and only if for every step i in G, depends(j, i) is False.
        Any group G with len(G) >= 2 is emitted as a parallelizable candidate with
        steps_saved = len(G) - 1.
    """
    base_tokens = extract_signal_tokens(
        f"{system_prompt}\n{task_prompt}",
        excluded_tokens=excluded_tokens,
    )

    produced_by_step: dict[int, set[str]] = {}
    args_by_step: dict[int, set[str]] = {}

    for step in trace.steps:
        if not step.tool_calls:
            continue
        prod: set[str] = set()
        for tr in step.tool_results:
            prod.update(extract_signal_tokens(tr.content, excluded_tokens=excluded_tokens))
        produced_by_step[step.index] = prod - base_tokens

        arg_toks: set[str] = set()
        for tc in step.tool_calls:
            arg_toks.update(
                extract_signal_tokens(
                    json.dumps(tc.arguments, ensure_ascii=False),
                    excluded_tokens=excluded_tokens,
                )
            )
        args_by_step[step.index] = arg_toks

    groups: list[list[int]] = []
    current_group: list[int] = []

    for step in trace.steps:
        if not step.tool_calls:
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = []
            continue

        j = step.index
        if not current_group:
            current_group = [j]
            continue

        # Check contiguous step index and independence from all steps in current_group
        is_contiguous = j == current_group[-1] + 1
        has_dependency = any(
            bool(args_by_step[j] & produced_by_step[i]) for i in current_group
        )
        if is_contiguous and not has_dependency:
            current_group.append(j)
        else:
            if len(current_group) >= 2:
                groups.append(current_group)
            current_group = [j]

    if len(current_group) >= 2:
        groups.append(current_group)

    return groups


def profile_traces(
    traces: Sequence[Trace],
    *,
    task_prompts: dict[str, str] | None = None,
    min_tokens: int = DEFAULT_MIN_EXTRACT_TOKENS,
    reuse_threshold: float = DEFAULT_REUSE_THRESHOLD,
) -> ProfileSummary:
    """Profile a sequence of Traces and generate deterministic low-confidence Opportunities.

    Opportunity Rules:
        - `extract_subagent`: emitted when a tool result has `size_tokens >= min_tokens`
          (default 2000) and `reuse_ratio < reuse_threshold` (default 0.1).
          `est_upper_bound_savings_tokens = carry_cost_tokens`.
        - `parallelize`: emitted for each independent contiguous step group (len >= 2)
          from Signal S3. `steps_saved = len(group) - 1`.
        - Every Opportunity carries `confidence="low"` (R9).
    """
    high_freq = compute_high_frequency_tokens(traces)
    prompts_map = task_prompts or {}

    trace_profiles: list[TraceProfile] = []
    opportunities: list[Opportunity] = []
    opp_counter = 0

    for trace in traces:
        num_calls = len(trace.steps)
        call_name_by_id: dict[str, str] = {}
        for step in trace.steps:
            for tc in step.tool_calls:
                call_name_by_id[tc.id] = tc.name

        result_profiles: list[ToolResultProfile] = []
        for step in trace.steps:
            downstream_texts = _collect_downstream_texts(trace, step.index)
            for tr in step.tool_results:
                tool_name = call_name_by_id.get(tr.call_id, "unknown")
                remaining, carry_cost = compute_carry_cost(
                    tokens_est=tr.tokens_est,
                    step_index=step.index,
                    num_calls=num_calls,
                )
                reuse = compute_reuse_ratio(
                    result_content=tr.content,
                    downstream_texts=downstream_texts,
                    excluded_tokens=high_freq,
                )
                rp = ToolResultProfile(
                    trace_id=trace.trace_id,
                    step_index=step.index,
                    call_id=tr.call_id,
                    tool=tool_name,
                    size_tokens=tr.tokens_est,
                    remaining_calls=remaining,
                    carry_cost_tokens=carry_cost,
                    reuse_ratio=reuse,
                )
                result_profiles.append(rp)

                if rp.size_tokens >= min_tokens and rp.reuse_ratio < reuse_threshold:
                    opp_counter += 1
                    opportunities.append(
                        Opportunity(
                            id=f"opp_{opp_counter:04d}",
                            kind="extract_subagent",
                            task_id=trace.task_id,
                            trace_ids=[trace.trace_id],
                            evidence={
                                "trace_id": trace.trace_id,
                                "step_index": rp.step_index,
                                "call_id": rp.call_id,
                                "tool": rp.tool,
                                "size_tokens": rp.size_tokens,
                                "remaining_calls": rp.remaining_calls,
                                "carry_cost_tokens": rp.carry_cost_tokens,
                                "reuse_ratio": round(rp.reuse_ratio, 6),
                            },
                            est_upper_bound_savings_tokens=rp.carry_cost_tokens,
                            steps_saved=None,
                            confidence="low",
                        )
                    )

        carry_total = sum(r.carry_cost_tokens for r in result_profiles)
        prompt_sum = sum(s.usage.prompt_tokens for s in trace.steps)
        if prompt_sum == 0:
            prompt_sum = trace.total_usage.prompt_tokens
        carry_share = (carry_total / prompt_sum) if prompt_sum > 0 else 0.0

        trace_profiles.append(
            TraceProfile(
                trace_id=trace.trace_id,
                task_id=trace.task_id,
                num_calls=num_calls,
                carry_total_tokens=carry_total,
                carry_share=carry_share,
                usage_source=trace.total_usage.source,
                results=result_profiles,
            )
        )

        indep_groups = find_independent_step_groups(
            trace,
            task_prompt=prompts_map.get(trace.task_id, ""),
        )
        steps_by_idx = {s.index: s for s in trace.steps}
        for group in indep_groups:
            opp_counter += 1
            group_tools = [
                tc.name
                for idx in group
                for tc in steps_by_idx[idx].tool_calls
            ]
            saved = len(group) - 1
            opportunities.append(
                Opportunity(
                    id=f"opp_{opp_counter:04d}",
                    kind="parallelize",
                    task_id=trace.task_id,
                    trace_ids=[trace.trace_id],
                    evidence={
                        "trace_id": trace.trace_id,
                        "step_indices": group,
                        "tools": group_tools,
                        "group_size": len(group),
                        "steps_saved": saved,
                    },
                    est_upper_bound_savings_tokens=None,
                    steps_saved=saved,
                    confidence="low",
                )
            )

    opportunities.sort(
        key=lambda o: (
            -(o.est_upper_bound_savings_tokens or 0),
            -(o.steps_saved or 0),
            o.id,
        )
    )

    return ProfileSummary(
        traces=trace_profiles,
        opportunities=opportunities,
    )
