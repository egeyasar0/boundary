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

    for valid_reason in ("final", "max_steps", "budget", "error"):
        t = trace.model_copy(update={"stop_reason": valid_reason})
        assert Trace.model_validate_json(t.model_dump_json()).stop_reason == valid_reason

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


def test_compute_config_hash_all_fields_and_no_api_key() -> None:
    base_kwargs = {
        "model": "test-model",
        "system_prompt": "You are a baseline agent.",
        "tools": ["read_file", "list_files", "grep"],
        "task_id": "task-1",
        "token_budget": 40_000,
        "max_steps": 20,
        "max_output_chars": 20_000,
        "grep_max_matches": 200,
        "grep_max_line_chars": 500,
        "grep_max_file_bytes": 1_000_000,
        "seed": 42,
        "parameters": {"temperature": 0.0},
    }
    h1 = compute_config_hash(**base_kwargs)  # type: ignore[arg-type]
    h2 = compute_config_hash(
        model="test-model",
        system_prompt="You are a baseline agent.",
        tools=["grep", "list_files", "read_file"],
        task_id="task-1",
        token_budget=40_000,
        max_steps=20,
        max_output_chars=20_000,
        grep_max_matches=200,
        grep_max_line_chars=500,
        grep_max_file_bytes=1_000_000,
        seed=42,
        parameters={
            "temperature": 0.0,
            "api_key": "sk-secret-do-not-hash",
            "client_secret": "top-secret",
        },
    )
    assert h1 == h2
    assert len(h1) == 16

    assert compute_config_hash(**{**base_kwargs, "model": "other-model"}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "system_prompt": "Other"}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "tools": ["read_file"]}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "task_id": "task-2"}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "token_budget": 10_000}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "max_steps": 12}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "max_output_chars": 5_000}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "grep_max_matches": 50}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "grep_max_line_chars": 120}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "grep_max_file_bytes": 500_000}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "seed": 99}) != h1  # type: ignore[arg-type]
    assert compute_config_hash(**{**base_kwargs, "parameters": {"temperature": 0.7}}) != h1  # type: ignore[arg-type]
    # Prove safe_params keeps token_budget / max_tokens keys while stripping api_key / secret
    h_with_param_budget = compute_config_hash(
        **{**base_kwargs, "parameters": {"temperature": 0.0, "token_budget": 9999}}  # type: ignore[arg-type]
    )
    assert h_with_param_budget != h1


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


def test_trace_error_sanitization_and_length_cap() -> None:
    raw_err = (
        "HTTP 401: Authorization: Bearer sk-live1234567890abcdef "
        "api_key='nvapi-998877665544' headers={'Authorization': 'Bearer secret'} "
        + ("X" * 600)
    )
    trace = Trace(
        trace_id="tr_err",
        task_id="t1",
        trial=1,
        model="m",
        config_hash="0123456789abcdef",
        steps=[],
        final_answer=None,
        passed=False,
        total_usage=Usage(prompt_tokens=0, completion_tokens=0, source="api"),
        stop_reason="error",
        error=raw_err,
    )
    assert trace.error is not None
    assert len(trace.error) <= 500
    assert "sk-live1234567890abcdef" not in trace.error
    assert "nvapi-998877665544" not in trace.error
    assert "Bearer secret" not in trace.error
    assert "[REDACTED]" in trace.error

