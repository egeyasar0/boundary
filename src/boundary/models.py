"""Core Pydantic data models for Boundary."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

StopReason = Literal["final", "max_steps", "budget", "error"]
UsageSource = Literal["api", "estimated"]
CheckerType = Literal["contains_all", "regex", "exact_normalized"]


class Checker(BaseModel):
    """Deterministic task answer checker configuration."""

    type: CheckerType
    expected: list[str] | str


class Task(BaseModel):
    """Benchmark task specification loaded from YAML."""

    id: str
    prompt: str
    checker: Checker
    max_steps: int = 20
    tags: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    """Token usage for a single step or entire trace."""

    prompt_tokens: int
    completion_tokens: int
    source: UsageSource

    @property
    def total(self) -> int:
        """Total tokens consumed (prompt + completion)."""
        return self.prompt_tokens + self.completion_tokens


class ToolCall(BaseModel):
    """Tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """Result returned by a read-only sandbox tool."""

    call_id: str
    content: str
    size_chars: int
    tokens_est: int


class Step(BaseModel):
    """Single agent loop iteration."""

    index: int
    assistant_content: str | None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    usage: Usage
    latency_ms: int


class Trace(BaseModel):
    """Complete execution trace for one task trial."""

    trace_id: str
    task_id: str
    trial: int
    agent_variant: str = "baseline"
    model: str
    config_hash: str
    steps: list[Step]
    final_answer: str | None
    passed: bool
    total_usage: Usage
    stop_reason: StopReason


class TaskSummary(BaseModel):
    """Aggregated metrics for a single task across trials."""

    task_id: str
    trials: int
    passed_count: int
    pass_rate: float
    wilson_low: float
    wilson_high: float
    insufficient_evidence: bool
    mean_tokens: float
    tool_token_share: float
    stop_reasons: dict[StopReason, int]


class RunSummary(BaseModel):
    """Aggregated summary across all tasks in a run."""

    run_id: str
    agent_variant: str = "baseline"
    model: str
    total_trials: int
    total_passed: int
    overall_pass_rate: float
    wilson_low: float
    wilson_high: float
    insufficient_evidence: bool
    mean_tokens: float
    tool_token_share: float
    stop_reasons: dict[StopReason, int]
    tasks: list[TaskSummary]


def compute_config_hash(
    *,
    model: str,
    system_prompt: str,
    tools: list[str],
    task_id: str,
    seed: int | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    """Compute a deterministic SHA-256 prefix for run reproducibility (R6)."""
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "tools": sorted(tools),
        "task_id": task_id,
        "seed": seed,
        "parameters": parameters or {},
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
