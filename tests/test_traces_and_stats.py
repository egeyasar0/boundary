"""Unit tests for JSONL trace I/O and Wilson confidence intervals (M5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from boundary.models import Step, ToolCall, ToolResult, Trace, Usage
from boundary.stats import is_insufficient_evidence, wilson_interval
from boundary.traces import read_traces, write_trace, write_traces


def test_wilson_interval_values_and_bounds() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)

    low_0, high_0 = wilson_interval(0, 3)
    assert low_0 == pytest.approx(0.0, abs=1e-6)
    assert 0.50 < high_0 < 0.60

    low_3, high_3 = wilson_interval(3, 3)
    assert 0.40 < low_3 < 0.50
    assert high_3 == pytest.approx(1.0, abs=1e-6)

    low_half, high_half = wilson_interval(5, 10)
    assert 0.23 < low_half < 0.25
    assert 0.75 < high_half < 0.77

    with pytest.raises(ValueError):
        wilson_interval(4, 3)


def test_insufficient_evidence_flag() -> None:
    assert is_insufficient_evidence(3) is True
    assert is_insufficient_evidence(5) is False


def test_traces_jsonl_round_trip(tmp_path: Path) -> None:
    step = Step(
        index=0,
        assistant_content="Answer: 7919",
        tool_calls=[ToolCall(id="c1", name="list_files", arguments={"path": "."})],
        tool_results=[ToolResult(call_id="c1", content="a.py", size_chars=4, tokens_est=1)],
        usage=Usage(prompt_tokens=80, completion_tokens=20, source="api"),
        latency_ms=15,
    )
    t1 = Trace(
        trace_id="tr_1",
        task_id="retry-ceiling",
        trial=1,
        model="fake-llm",
        config_hash="abcdef0123456789",
        steps=[step],
        final_answer="7919 decorrelated-halcyon-v4",
        passed=True,
        total_usage=Usage(prompt_tokens=80, completion_tokens=20, source="api"),
        stop_reason="final",
    )
    t2 = t1.model_copy(update={"trace_id": "tr_2", "trial": 2, "stop_reason": "budget"})

    run_dir = tmp_path / "run_001"
    write_trace(run_dir, t1)
    write_trace(run_dir, t2)

    loaded = read_traces(run_dir)
    assert len(loaded) == 2
    assert loaded[0] == t1
    assert loaded[1] == t2
    assert all(len(t.config_hash) == 16 for t in loaded)

    # Also test write_traces directly to a .jsonl file path
    direct_file = tmp_path / "custom.jsonl"
    write_traces(direct_file, [t1])
    assert read_traces(direct_file) == [t1]
