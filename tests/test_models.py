"""Unit tests for Boundary Pydantic data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from boundary.models import (
    Checker,
    RunSummary,
    Step,
    Task,
    TaskSummary,
    ToolCall,
    ToolResult,
    Trace,
    Usage,
    compute_config_hash,
)


def test_usage_total_and_source() -> None:
    usage = Usage(prompt_tokens=120, completion_tokens=30, source="api")
    assert usage.total == 150
    assert usage.source == "api"

    est = Usage(prompt_tokens=40, completion_tokens=10, source="estimated")
    assert est.total == 50

    with pytest.raises(ValidationError):
        Usage(prompt_tokens=10, completion_tokens=5, source="invalid")  # type: ignore[arg-type]


def test_task_and_checker_defaults() -> None:
    task = Task(
        id="find-retry-limit",
        prompt="Which file defines retry count?",
        checker=Checker(type="contains_all", expected=["client.py", "5"]),
    )
    assert task.max_steps == 20
    assert task.tags == []


def test_trace_round_trip_and_stop_reason() -> None:
    step = Step(
        index=0,
        assistant_content=None,
        tool_calls=[ToolCall(id="call_1", name="list_files", arguments={"path": "."})],
        tool_results=[
            ToolResult(
                call_id="call_1",
                content="a.py\nb.py",
                size_chars=9,
                tokens_est=2,
            )
        ],
        usage=Usage(prompt_tokens=100, completion_tokens=20, source="api"),
        latency_ms=42,
    )
    trace = Trace(
        trace_id="tr_001",
        task_id="find-retry-limit",
        trial=1,
        model="gpt-4o-mini",
        config_hash="0123456789abcdef",
        steps=[step],
        final_answer="client.py defines 5",
        passed=True,
        total_usage=Usage(prompt_tokens=100, completion_tokens=20, source="api"),
        stop_reason="final",
    )

    raw_json = trace.model_dump_json()
    restored = Trace.model_validate_json(raw_json)
    assert restored == trace
    assert restored.agent_variant == "baseline"
    assert restored.total_usage.total == 120

    with pytest.raises(ValidationError):
        Trace(
            trace_id="tr_bad",
            task_id="t1",
            trial=1,
            model="m",
            config_hash="abc",
            steps=[],
            final_answer=None,
            passed=False,
            total_usage=Usage(prompt_tokens=0, completion_tokens=0, source="api"),
            stop_reason="timeout",  # type: ignore[arg-type]
        )


def test_compute_config_hash_determinism_and_system_prompt() -> None:
    h1 = compute_config_hash(
        model="test-model",
        system_prompt="You are a baseline agent.",
        tools=["read_file", "list_files", "grep"],
        task_id="task-1",
        seed=42,
        parameters={"temperature": 0.0},
    )
    h2 = compute_config_hash(
        model="test-model",
        system_prompt="You are a baseline agent.",
        tools=["grep", "list_files", "read_file"],
        task_id="task-1",
        seed=42,
        parameters={"temperature": 0.0},
    )
    assert h1 == h2
    assert len(h1) == 16

    h_diff_prompt = compute_config_hash(
        model="test-model",
        system_prompt="Different system prompt.",
        tools=["read_file", "list_files", "grep"],
        task_id="task-1",
        seed=42,
        parameters={"temperature": 0.0},
    )
    assert h1 != h_diff_prompt


def test_run_summary_model() -> None:
    task_summary = TaskSummary(
        task_id="task-1",
        trials=3,
        passed_count=2,
        pass_rate=2 / 3,
        wilson_low=0.20,
        wilson_high=0.90,
        insufficient_evidence=True,
        mean_tokens=500.0,
        tool_token_share=0.4,
        stop_reasons={"final": 2, "budget": 1},
    )
    summary = RunSummary(
        run_id="run-01",
        model="fake",
        total_trials=3,
        total_passed=2,
        overall_pass_rate=2 / 3,
        wilson_low=0.20,
        wilson_high=0.90,
        insufficient_evidence=True,
        mean_tokens=500.0,
        tool_token_share=0.4,
        stop_reasons={"final": 2, "budget": 1},
        tasks=[task_summary],
    )
    assert summary.tasks[0].task_id == "task-1"
