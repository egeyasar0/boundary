"""Unit tests for the statistical comparison and R14/R15 decision engine (M12)."""

from __future__ import annotations

from pathlib import Path

import pytest

from boundary.compare import (
    compare_runs,
    compare_summaries,
    decide_boundary,
    intervals_overlap,
    render_comparison_report,
)
from boundary.models import Step, Trace, Usage
from boundary.report import summarize_run
from boundary.stats import wilson_interval
from boundary.traces import write_traces


def _make_trace(
    trace_id: str,
    task_id: str,
    trial: int,
    *,
    passed: bool,
    prompt_tokens: int,
    completion_tokens: int = 20,
    stop_reason: str = "final",
    agent_variant: str = "baseline",
) -> Trace:
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        source="api",
    )
    return Trace(
        trace_id=trace_id,
        task_id=task_id,
        trial=trial,
        agent_variant=agent_variant,
        model="fake-gpt-oss-20b",
        config_hash="cfg_hash_123",
        steps=[
            Step(
                index=0,
                assistant_content="done",
                tool_calls=[],
                tool_results=[],
                usage=usage,
                latency_ms=10,
            )
        ],
        final_answer="done",
        passed=passed,
        total_usage=usage,
        stop_reason=stop_reason,  # type: ignore[arg-type]
    )


def test_intervals_overlap_basic() -> None:
    """Verify interval overlap check."""
    assert intervals_overlap(0.1, 0.5, 0.4, 0.8) is True
    assert intervals_overlap(0.0, 0.3, 0.5, 0.9) is False


def test_r15_accept_when_pass_rate_increases_with_non_overlapping_wilson_ci() -> None:
    """Verify R15.1: ACCEPT when pass rate improves with non-overlapping Wilson 95% CIs."""
    b_wilson = wilson_interval(0, 10)  # [0.0, 0.278]
    v_wilson = wilson_interval(9, 10)  # [0.596, 0.982]
    decision, rationale = decide_boundary(
        baseline_trials=10,
        variant_trials=10,
        baseline_pass_rate=0.0,
        variant_pass_rate=0.9,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=False,
        token_delta_pct=10.0,
    )
    assert decision == "ACCEPT"
    assert "R15.1" in rationale


def test_r15_accept_when_pass_rate_increases_with_lower_bound_and_sufficient_trials() -> None:
    """Verify R15.1: ACCEPT when trials >= 5, pass rate up, v_low > b_low, tokens non-increasing."""
    b_wilson = wilson_interval(2, 6)
    v_wilson = wilson_interval(5, 6)
    decision, rationale = decide_boundary(
        baseline_trials=6,
        variant_trials=6,
        baseline_pass_rate=2 / 6,
        variant_pass_rate=5 / 6,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=False,
        token_delta_pct=-10.0,
    )
    assert decision == "ACCEPT"
    assert "R15.1" in rationale


def test_r15_accept_when_token_reduction_at_least_25_pct_at_equal_pass_rate() -> None:
    """Verify R15.2: ACCEPT when pass rate is maintained and tokens drop by >= 25%."""
    b_wilson = wilson_interval(3, 3)
    v_wilson = wilson_interval(3, 3)
    decision, rationale = decide_boundary(
        baseline_trials=3,
        variant_trials=3,
        baseline_pass_rate=1.0,
        variant_pass_rate=1.0,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=True,
        token_delta_pct=-65.0,
    )
    assert decision == "ACCEPT"
    assert "R15.2" in rationale


def test_r14_reject_when_tokens_increase_and_wilson_cis_overlap() -> None:
    """Verify R14: REJECT when token cost increases and pass rate gain has overlapping CIs."""
    b_wilson = wilson_interval(3, 6)
    v_wilson = wilson_interval(4, 6)
    decision, rationale = decide_boundary(
        baseline_trials=6,
        variant_trials=6,
        baseline_pass_rate=0.5,
        variant_pass_rate=4 / 6,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=False,
        token_delta_pct=18.5,
    )
    assert decision == "REJECT"
    assert "R14" in rationale


def test_r15_reject_on_tie_occams_razor() -> None:
    """Verify R15 Occam's Razor: equal pass rate and no token savings favors single agent."""
    b_wilson = wilson_interval(5, 5)
    v_wilson = wilson_interval(5, 5)
    decision, rationale = decide_boundary(
        baseline_trials=5,
        variant_trials=5,
        baseline_pass_rate=1.0,
        variant_pass_rate=1.0,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=False,
        token_delta_pct=0.0,
    )
    assert decision == "REJECT"
    assert "Occam's Razor" in rationale


def test_r15_reject_when_pass_rate_decreases_even_if_tokens_drop() -> None:
    """Verify R15: any decrease in pass rate results in REJECT."""
    b_wilson = wilson_interval(5, 5)
    v_wilson = wilson_interval(3, 5)
    decision, _ = decide_boundary(
        baseline_trials=5,
        variant_trials=5,
        baseline_pass_rate=1.0,
        variant_pass_rate=0.6,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=False,
        token_delta_pct=-50.0,
    )
    assert decision == "REJECT"


def test_r15_reject_when_token_reduction_below_25_pct_with_sufficient_trials() -> None:
    """Verify R15.2: < 25% token reduction at equal pass rate with sufficient trials is REJECT."""
    b_wilson = wilson_interval(5, 5)
    v_wilson = wilson_interval(5, 5)
    decision, rationale = decide_boundary(
        baseline_trials=5,
        variant_trials=5,
        baseline_pass_rate=1.0,
        variant_pass_rate=1.0,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=False,
        token_delta_pct=-12.0,
    )
    assert decision == "REJECT"
    assert "below the 25% threshold" in rationale


def test_inconclusive_when_insufficient_trials_and_overlapping_modest_gain() -> None:
    """Verify INCONCLUSIVE when trials < 5 and pass rate improves slightly with CI overlap."""
    b_wilson = wilson_interval(1, 3)
    v_wilson = wilson_interval(2, 3)
    decision, _ = decide_boundary(
        baseline_trials=3,
        variant_trials=3,
        baseline_pass_rate=1 / 3,
        variant_pass_rate=2 / 3,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=True,
        token_delta_pct=-10.0,
    )
    assert decision == "INCONCLUSIVE"


def test_compare_runs_and_render_comparison_report(tmp_path: Path) -> None:
    """Verify compare_runs and render_comparison_report across multiple tasks."""
    base_traces = [
        # Task 1 (needle): 0/3 pass (budget stop), 45000 tokens
        _make_trace(
            f"b_needle_{i}",
            "routing-shard-needle",
            i,
            passed=False,
            prompt_tokens=45_000,
            stop_reason="budget",
            agent_variant="baseline",
        )
        for i in range(1, 4)
    ] + [
        # Task 2 (easy): 3/3 pass, 1500 tokens
        _make_trace(
            f"b_easy_{i}",
            "retry-ceiling",
            i,
            passed=True,
            prompt_tokens=1500,
            stop_reason="final",
            agent_variant="baseline",
        )
        for i in range(1, 4)
    ]

    var_traces = [
        # Task 1 (needle): 3/3 pass, 4500 tokens (-90% tokens, +100% pass rate -> ACCEPT)
        _make_trace(
            f"v_needle_{i}",
            "routing-shard-needle",
            i,
            passed=True,
            prompt_tokens=4500,
            stop_reason="final",
            agent_variant="subagent_extracted",
        )
        for i in range(1, 4)
    ] + [
        # Task 2 (easy): 3/3 pass, 1500 tokens (0% token delta, 0% pass delta -> REJECT)
        _make_trace(
            f"v_easy_{i}",
            "retry-ceiling",
            i,
            passed=True,
            prompt_tokens=1500,
            stop_reason="final",
            agent_variant="subagent_extracted",
        )
        for i in range(1, 4)
    ]

    base_dir = tmp_path / "run_base"
    var_dir = tmp_path / "run_var"
    write_traces(base_dir, base_traces)
    write_traces(var_dir, var_traces)

    report = compare_runs(base_dir, var_dir, token_budget=40_000)
    assert report.overall.decision == "ACCEPT"
    assert report.overall.pass_rate_delta == pytest.approx(0.5)
    assert report.overall.token_delta_pct < -70.0

    by_task = {t.task_id: t for t in report.tasks}
    assert by_task["routing-shard-needle"].decision == "ACCEPT"
    assert by_task["routing-shard-needle"].budget_stop_delta == pytest.approx(-1.0)
    assert by_task["retry-ceiling"].decision == "REJECT"

    rendered = render_comparison_report(report)
    assert "ACCEPT (Boundary Justified)" in rendered
    assert "REJECT (Over-engineered)" in rendered
    assert "routing-shard-needle" in rendered
    assert "Comparison Verdict (R14 / R15)" in rendered

    # Also verify compare_summaries directly
    s_base = summarize_run(base_traces, run_id="b")
    s_var = summarize_run(var_traces, run_id="v")
    assert compare_summaries(s_base, s_var).overall.decision == "ACCEPT"
