"""Summary calculation and Rich report rendering for Boundary runs (M5)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from io import StringIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from boundary.models import RunSummary, StopReason, TaskSummary, Trace
from boundary.stats import is_insufficient_evidence, wilson_interval

ALL_STOP_REASONS: tuple[StopReason, ...] = ("final", "max_steps", "budget", "error")


def _compute_tool_tokens(trace: Trace) -> int:
    """Sum estimated tokens originating from tool outputs across all steps in a trace."""
    return sum(tr.tokens_est for step in trace.steps for tr in step.tool_results)


def _aggregate_traces(task_id: str, traces: Sequence[Trace]) -> TaskSummary:
    """Compute TaskSummary metrics for a group of traces belonging to one task."""
    trials = len(traces)
    passed_count = sum(1 for t in traces if t.passed)
    pass_rate = (passed_count / trials) if trials > 0 else 0.0
    w_low, w_high = wilson_interval(passed_count, trials)
    insufficient = is_insufficient_evidence(trials)

    total_tokens = sum(t.total_usage.total for t in traces)
    mean_tokens = (total_tokens / trials) if trials > 0 else 0.0

    tool_tokens = sum(_compute_tool_tokens(t) for t in traces)
    tool_share = min(1.0, (tool_tokens / total_tokens)) if total_tokens > 0 else 0.0

    stop_counts: dict[StopReason, int] = {reason: 0 for reason in ALL_STOP_REASONS}
    for t in traces:
        stop_counts[t.stop_reason] = stop_counts.get(t.stop_reason, 0) + 1

    return TaskSummary(
        task_id=task_id,
        trials=trials,
        passed_count=passed_count,
        pass_rate=pass_rate,
        wilson_low=w_low,
        wilson_high=w_high,
        insufficient_evidence=insufficient,
        mean_tokens=mean_tokens,
        tool_token_share=tool_share,
        stop_reasons=stop_counts,
    )


def summarize_run(traces: Sequence[Trace], *, run_id: str = "run") -> RunSummary:
    """Aggregate a list of Traces into a deterministic RunSummary."""
    if not traces:
        raise ValueError("Cannot summarize an empty trace list.")

    by_task: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        by_task[trace.task_id].append(trace)

    task_summaries: list[TaskSummary] = [
        _aggregate_traces(task_id, by_task[task_id]) for task_id in sorted(by_task.keys())
    ]

    overall = _aggregate_traces("__all__", traces)
    first = traces[0]

    return RunSummary(
        run_id=run_id,
        agent_variant=first.agent_variant,
        model=first.model,
        total_trials=overall.trials,
        total_passed=overall.passed_count,
        overall_pass_rate=overall.pass_rate,
        wilson_low=overall.wilson_low,
        wilson_high=overall.wilson_high,
        insufficient_evidence= all(ts.insufficient_evidence for ts in task_summaries)
        or overall.insufficient_evidence,
        mean_tokens=overall.mean_tokens,
        tool_token_share=overall.tool_token_share,
        stop_reasons=overall.stop_reasons,
        tasks=task_summaries,
    )


def _format_stop_reasons(stop_reasons: dict[StopReason, int], total: int) -> str:
    """Format stop_reason counts with explicit budget rate highlight."""
    parts = [
        f"{r}:{stop_reasons.get(r, 0)}"
        for r in ALL_STOP_REASONS
        if stop_reasons.get(r, 0) > 0
    ]
    joined = ", ".join(parts) if parts else "none"
    budget_cnt = stop_reasons.get("budget", 0)
    budget_pct = (budget_cnt / total * 100.0) if total > 0 else 0.0
    return f"{joined} (budget: {budget_pct:.0f}%)"


def render_report(summary: RunSummary, *, console: Console | None = None) -> str:
    """Render RunSummary via Rich table and summary panel, returning rendered plain/ANSI text."""
    buf = StringIO()
    out_console = Console(file=buf, width=140, force_terminal=False)

    table = Table(title=f"Boundary Baseline Report — {summary.run_id} (model: {summary.model})")
    table.add_column("Task ID", style="bold", no_wrap=True)
    table.add_column("Pass Rate", justify="right")
    table.add_column("Wilson 95% CI", justify="center")
    table.add_column("Evidence", justify="center")
    table.add_column("Mean Tokens", justify="right")
    table.add_column("Tool Token Share", justify="right")
    table.add_column("Stop Reasons (budget highlight)", justify="left")

    for ts in summary.tasks:
        evidence_str = "yetersiz kanıt" if ts.insufficient_evidence else "ok"
        table.add_row(
            ts.task_id,
            f"{ts.passed_count}/{ts.trials} ({ts.pass_rate * 100:.1f}%)",
            f"[{ts.wilson_low * 100:.1f}%, {ts.wilson_high * 100:.1f}%]",
            evidence_str,
            f"{ts.mean_tokens:.1f}",
            f"{ts.tool_token_share * 100:.1f}%",
            _format_stop_reasons(ts.stop_reasons, ts.trials),
        )

    out_console.print(table)

    budget_count = summary.stop_reasons.get("budget", 0)
    budget_rate = (
        (budget_count / summary.total_trials * 100.0) if summary.total_trials > 0 else 0.0
    )
    overall_ev = (
        "yetersiz kanıt (insufficient evidence)"
        if summary.insufficient_evidence
        else "yeterli"
    )
    stop_dist = ", ".join(
        f"{r}={summary.stop_reasons.get(r, 0)}" for r in ALL_STOP_REASONS
    )

    summary_lines = [
        f"Variant: {summary.agent_variant} | Model: {summary.model}",
        (
            f"Overall Pass Rate: {summary.total_passed}/{summary.total_trials} "
            f"({summary.overall_pass_rate * 100:.1f}%) "
            f"Wilson 95% CI: [{summary.wilson_low * 100:.1f}%, {summary.wilson_high * 100:.1f}%] "
            f"— Kanıt durumu: {overall_ev}"
        ),
        (
            f"Mean Used Tokens: {summary.mean_tokens:.1f} | "
            f"Tool Output Token Share: {summary.tool_token_share * 100:.1f}%"
        ),
        f"Stop Reason Distribution: {stop_dist}",
        (
            f"Budget Stop Rate (budget oranı): "
            f"{budget_count}/{summary.total_trials} ({budget_rate:.1f}%)"
        ),
    ]
    out_console.print(Panel("\n".join(summary_lines), title="Run Summary"))

    rendered = buf.getvalue()
    if console is not None:
        console.print(rendered, end="")
    return rendered
