"""Statistical comparison and Occam's-razor decision engine for Boundary runs (M12 — R14, R15).

Rules:
- R14 (Equal Budget Discipline): Baseline and Variant are compared under the same token budget.
  If the variant increases mean token cost (`token_delta_pct > 0`) and any pass-rate gain is not
  statistically significant (Wilson 95% confidence intervals overlap), the decision is `REJECT`.
- R15 (Occam's Razor for Agents): Ties always favor the single-agent baseline (`REJECT`).
  A sub-agent boundary receives `ACCEPT` if and only if at least one holds:
  1. Pass rate increases (`pass_rate_delta > 0`) with Wilson 95% lower-bound superiority
     (and non-overlapping CI, or sufficient evidence `trials >= 5` without token increase).
  2. At non-decreasing pass rate (`pass_rate_delta >= 0`), mean token consumption drops by
     at least 25% (`token_delta_pct <= -25.0`, i.e. Token Reduction >= 25%).
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from boundary.models import RunSummary, TaskSummary
from boundary.report import (
    DEFAULT_REPORT_TOKEN_BUDGET,
    _write_to_console,
    summarize_run,
)
from boundary.traces import read_traces

Decision = Literal["ACCEPT", "REJECT", "INCONCLUSIVE"]

SIGNIFICANT_TOKEN_REDUCTION_PCT: float = 25.0
EPSILON: float = 1e-9

DECISION_LABELS: dict[Decision, str] = {
    "ACCEPT": "ACCEPT (Boundary Justified)",
    "REJECT": "REJECT (Over-engineered)",
    "INCONCLUSIVE": "INCONCLUSIVE",
}


class TaskComparison(BaseModel):
    """Per-task comparison metrics and R14/R15 boundary decision."""

    task_id: str
    baseline_trials: int
    variant_trials: int
    baseline_pass_rate: float
    variant_pass_rate: float
    pass_rate_delta: float
    baseline_wilson: tuple[float, float]
    variant_wilson: tuple[float, float]
    wilson_overlap: bool
    insufficient_evidence: bool
    baseline_mean_tokens: float
    variant_mean_tokens: float
    token_delta_pct: float
    token_reduction_pct: float
    baseline_budget_stop_rate: float
    variant_budget_stop_rate: float
    budget_stop_delta: float
    decision: Decision
    decision_label: str
    rationale: str


class ComparisonReport(BaseModel):
    """Complete run-to-run statistical comparison report."""

    baseline_run_id: str
    variant_run_id: str
    baseline_variant_name: str
    candidate_variant_name: str
    model: str
    token_budget: int
    overall: TaskComparison
    tasks: list[TaskComparison] = Field(default_factory=list)


def intervals_overlap(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
) -> bool:
    """Return True if two closed intervals [low_a, high_a] and [low_b, high_b] overlap."""
    return max(low_a, low_b) <= min(high_a, high_b) + EPSILON


def decide_boundary(
    *,
    baseline_trials: int,
    variant_trials: int,
    baseline_pass_rate: float,
    variant_pass_rate: float,
    baseline_wilson: tuple[float, float],
    variant_wilson: tuple[float, float],
    insufficient_evidence: bool,
    token_delta_pct: float,
) -> tuple[Decision, str]:
    """Apply R14 and R15 decision rules to determine ACCEPT, REJECT, or INCONCLUSIVE.

    Formula / Decision Rules (R14, R15):
        pass_rate_delta = variant_pass_rate - baseline_pass_rate
        token_reduction_pct = -token_delta_pct
        wilson_overlap = max(b_low, v_low) <= min(b_high, v_high)

        1. If baseline_trials == 0 or variant_trials == 0 -> INCONCLUSIVE.
        2. If pass_rate_delta < 0 -> REJECT (quality regression).
        3. If token_delta_pct > 0 and (wilson_overlap or pass_rate_delta <= 0)
           -> REJECT (R14: higher token cost without statistically significant CI separation).
        4. If pass_rate_delta == 0 and token_delta_pct >= 0
           -> REJECT (R15 Occam's Razor: tie or higher cost favors single agent).
        5. If pass_rate_delta >= 0 and token_delta_pct <= -25.0
           -> ACCEPT (R15 Condition 2: >= 25% token reduction at non-decreasing pass rate).
        6. If pass_rate_delta > 0:
           - If not wilson_overlap and variant_wilson[0] > baseline_wilson[1]
             -> ACCEPT (R15 Condition 1: non-overlapping Wilson 95% CI superiority).
           - If not insufficient_evidence and variant_wilson[0] > baseline_wilson[0]
             and token_delta_pct <= 0
             -> ACCEPT (R15 Condition 1: Wilson lower-bound superiority with sufficient trials).
           - Otherwise -> INCONCLUSIVE (positive pass rate delta but insufficient trials/CI).
        7. If pass_rate_delta == 0 and -25.0 < token_delta_pct < 0:
           - If insufficient_evidence -> INCONCLUSIVE.
           - Otherwise -> REJECT (token reduction < 25% threshold with sufficient trials).
    """
    if baseline_trials == 0 or variant_trials == 0:
        return "INCONCLUSIVE", "Missing trials in baseline or variant."

    pass_rate_delta = variant_pass_rate - baseline_pass_rate
    b_low, b_high = baseline_wilson
    v_low, v_high = variant_wilson
    overlap = intervals_overlap(b_low, b_high, v_low, v_high)

    # Rule 2: Pass rate dropped
    if pass_rate_delta < -EPSILON:
        return (
            "REJECT",
            f"Pass rate decreased by {pass_rate_delta * 100:.1f}%.",
        )

    # Rule 3 (R14): Token cost increased without non-overlapping Wilson CI superiority
    if token_delta_pct > EPSILON and (overlap or pass_rate_delta <= EPSILON):
        return (
            "REJECT",
            f"Token usage increased (+{token_delta_pct:.1f}%) without significant CI gain (R14).",
        )

    # Rule 4 (R15 Occam's Razor): Tie in pass rate with no token savings
    if abs(pass_rate_delta) <= EPSILON and token_delta_pct >= -EPSILON:
        return (
            "REJECT",
            "Equal pass rate without token savings; Occam's Razor favors single agent (R15).",
        )

    # Rule 5 (R15 Condition 2): >= 25% token reduction at same or better pass rate
    if (
        pass_rate_delta >= -EPSILON
        and token_delta_pct <= -SIGNIFICANT_TOKEN_REDUCTION_PCT + EPSILON
    ):
        return (
            "ACCEPT",
            f"Token consumption reduced by {-token_delta_pct:.1f}% (>= 25% threshold, R15.2).",
        )

    # Rule 6 (R15 Condition 1): Pass rate increased
    if pass_rate_delta > EPSILON:
        if not overlap and v_low > b_high:
            return (
                "ACCEPT",
                (
                    f"Pass rate increased (+{pass_rate_delta * 100:.1f}%) with "
                    "non-overlapping Wilson CI (R15.1)."
                ),
            )
        if not insufficient_evidence and v_low > b_low + EPSILON and token_delta_pct <= EPSILON:
            return (
                "ACCEPT",
                (
                    f"Pass rate increased (+{pass_rate_delta * 100:.1f}%) with "
                    "Wilson lower-bound superiority (R15.1)."
                ),
            )
        return (
            "INCONCLUSIVE",
            "Pass rate improved but Wilson 95% CIs overlap under insufficient trial count.",
        )

    # Rule 7: Pass rate unchanged, modest token reduction (0% < reduction < 25%)
    if insufficient_evidence:
        return (
            "INCONCLUSIVE",
            (
                f"Modest token reduction ({-token_delta_pct:.1f}% < 25%) "
                "with insufficient trials (< 5)."
            ),
        )
    return (
        "REJECT",
        f"Token reduction ({-token_delta_pct:.1f}%) is below the 25% threshold required by R15.2.",
    )


def _compare_task_summaries(
    task_id: str,
    base: TaskSummary,
    var: TaskSummary,
) -> TaskComparison:
    pass_delta = var.pass_rate - base.pass_rate
    b_wilson = (base.wilson_low, base.wilson_high)
    v_wilson = (var.wilson_low, var.wilson_high)
    overlap = intervals_overlap(b_wilson[0], b_wilson[1], v_wilson[0], v_wilson[1])
    insufficient = base.insufficient_evidence or var.insufficient_evidence

    if base.mean_tokens > 0:
        token_delta_pct = ((var.mean_tokens - base.mean_tokens) / base.mean_tokens) * 100.0
    elif var.mean_tokens > 0:
        token_delta_pct = 100.0
    else:
        token_delta_pct = 0.0

    token_reduction_pct = -token_delta_pct

    base_budget_rate = (
        (base.stop_reasons.get("budget", 0) / base.trials) if base.trials > 0 else 0.0
    )
    var_budget_rate = (
        (var.stop_reasons.get("budget", 0) / var.trials) if var.trials > 0 else 0.0
    )
    budget_delta = var_budget_rate - base_budget_rate

    decision, rationale = decide_boundary(
        baseline_trials=base.trials,
        variant_trials=var.trials,
        baseline_pass_rate=base.pass_rate,
        variant_pass_rate=var.pass_rate,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        insufficient_evidence=insufficient,
        token_delta_pct=token_delta_pct,
    )

    return TaskComparison(
        task_id=task_id,
        baseline_trials=base.trials,
        variant_trials=var.trials,
        baseline_pass_rate=base.pass_rate,
        variant_pass_rate=var.pass_rate,
        pass_rate_delta=pass_delta,
        baseline_wilson=b_wilson,
        variant_wilson=v_wilson,
        wilson_overlap=overlap,
        insufficient_evidence=insufficient,
        baseline_mean_tokens=base.mean_tokens,
        variant_mean_tokens=var.mean_tokens,
        token_delta_pct=token_delta_pct,
        token_reduction_pct=token_reduction_pct,
        baseline_budget_stop_rate=base_budget_rate,
        variant_budget_stop_rate=var_budget_rate,
        budget_stop_delta=budget_delta,
        decision=decision,
        decision_label=DECISION_LABELS[decision],
        rationale=rationale,
    )


def compare_summaries(
    baseline: RunSummary,
    variant: RunSummary,
    *,
    token_budget: int = DEFAULT_REPORT_TOKEN_BUDGET,
) -> ComparisonReport:
    """Compare a baseline RunSummary and a candidate variant RunSummary (M12)."""
    base_by_task = {t.task_id: t for t in baseline.tasks}
    var_by_task = {t.task_id: t for t in variant.tasks}
    common_tasks = sorted(set(base_by_task.keys()) & set(var_by_task.keys()))

    task_comparisons: list[TaskComparison] = [
        _compare_task_summaries(tid, base_by_task[tid], var_by_task[tid])
        for tid in common_tasks
    ]

    overall_base = TaskSummary(
        task_id="__overall__",
        trials=baseline.total_trials,
        passed_count=baseline.total_passed,
        pass_rate=baseline.overall_pass_rate,
        wilson_low=baseline.wilson_low,
        wilson_high=baseline.wilson_high,
        insufficient_evidence=baseline.insufficient_evidence,
        mean_tokens=baseline.mean_tokens,
        tool_token_share=baseline.tool_token_share,
        max_budget_ratio=baseline.max_budget_ratio,
        stop_reasons=baseline.stop_reasons,
        last_error=baseline.last_error,
    )
    overall_var = TaskSummary(
        task_id="__overall__",
        trials=variant.total_trials,
        passed_count=variant.total_passed,
        pass_rate=variant.overall_pass_rate,
        wilson_low=variant.wilson_low,
        wilson_high=variant.wilson_high,
        insufficient_evidence=variant.insufficient_evidence,
        mean_tokens=variant.mean_tokens,
        tool_token_share=variant.tool_token_share,
        max_budget_ratio=variant.max_budget_ratio,
        stop_reasons=variant.stop_reasons,
        last_error=variant.last_error,
    )
    overall_cmp = _compare_task_summaries("__overall__", overall_base, overall_var)

    return ComparisonReport(
        baseline_run_id=baseline.run_id,
        variant_run_id=variant.run_id,
        baseline_variant_name=baseline.agent_variant,
        candidate_variant_name=variant.agent_variant,
        model=variant.model,
        token_budget=token_budget,
        overall=overall_cmp,
        tasks=task_comparisons,
    )


def compare_runs(
    baseline_run_dir: Path | str,
    variant_run_dir: Path | str,
    *,
    token_budget: int = DEFAULT_REPORT_TOKEN_BUDGET,
) -> ComparisonReport:
    """Load traces from `baseline_run_dir` and `variant_run_dir` and compute ComparisonReport."""
    b_path = Path(baseline_run_dir)
    v_path = Path(variant_run_dir)
    b_traces = read_traces(b_path)
    v_traces = read_traces(v_path)
    if not b_traces:
        raise ValueError(f"Baseline trace file at '{b_path}' is empty.")
    if not v_traces:
        raise ValueError(f"Variant trace file at '{v_path}' is empty.")

    b_id = b_path.stem if b_path.is_file() else b_path.name
    v_id = v_path.stem if v_path.is_file() else v_path.name
    b_summary = summarize_run(b_traces, run_id=b_id, token_budget=token_budget)
    v_summary = summarize_run(v_traces, run_id=v_id, token_budget=token_budget)
    return compare_summaries(b_summary, v_summary, token_budget=token_budget)


def render_comparison_report(
    report: ComparisonReport,
    *,
    console: Console | None = None,
) -> str:
    """Render ComparisonReport via Rich table and verdict summary panel (M12)."""
    buf = StringIO()
    out_console = Console(file=buf, width=260, force_terminal=False)

    title = (
        f"Boundary Comparison - {report.baseline_run_id} ({report.baseline_variant_name}) "
        f"vs {report.variant_run_id} ({report.candidate_variant_name})"
    )
    table = Table(title=escape(title), box=box.ASCII)
    table.add_column("Task ID", style="bold", no_wrap=True)
    table.add_column("Baseline Pass", justify="right")
    table.add_column("Variant Pass", justify="right")
    table.add_column("Pass Rate Delta", justify="right")
    table.add_column("Baseline Tokens", justify="right")
    table.add_column("Variant Tokens", justify="right")
    table.add_column("Token Delta %", justify="right")
    table.add_column("Budget Stop Delta", justify="right")
    table.add_column("Decision", justify="left", no_wrap=True)

    for tc in report.tasks:
        table.add_row(
            tc.task_id,
            f"{tc.baseline_pass_rate * 100:.1f}%",
            f"{tc.variant_pass_rate * 100:.1f}%",
            f"{tc.pass_rate_delta * 100:+.1f}%",
            f"{tc.baseline_mean_tokens:.1f}",
            f"{tc.variant_mean_tokens:.1f}",
            f"{tc.token_delta_pct:+.1f}%",
            f"{tc.budget_stop_delta * 100:+.1f}%",
            tc.decision_label,
        )

    out_console.print(table)

    ov = report.overall
    accepted_tasks = [t.task_id for t in report.tasks if t.decision == "ACCEPT"]
    rejected_tasks = [t.task_id for t in report.tasks if t.decision == "REJECT"]
    inconclusive_tasks = [t.task_id for t in report.tasks if t.decision == "INCONCLUSIVE"]

    panel_lines = [
        (
            f"Overall Verdict: {ov.decision_label} | "
            f"Model: {report.model} | Equal Budget: {report.token_budget} tokens"
        ),
        (
            f"Overall Pass Rate: {ov.baseline_pass_rate * 100:.1f}% -> "
            f"{ov.variant_pass_rate * 100:.1f}% ({ov.pass_rate_delta * 100:+.1f}%) | "
            f"Wilson Overlap: {'yes' if ov.wilson_overlap else 'no'}"
        ),
        (
            f"Overall Mean Tokens: {ov.baseline_mean_tokens:.1f} -> "
            f"{ov.variant_mean_tokens:.1f} ({ov.token_delta_pct:+.1f}%) | "
            f"Budget Stop Delta: {ov.budget_stop_delta * 100:+.1f}%"
        ),
        (
            f"Task Decisions: ACCEPT={len(accepted_tasks)}, "
            f"REJECT={len(rejected_tasks)}, INCONCLUSIVE={len(inconclusive_tasks)}"
        ),
        f"Rationale: {ov.rationale}",
    ]
    out_console.print(
        Panel("\n".join(panel_lines), title="Comparison Verdict (R14 / R15)", box=box.ASCII)
    )

    rendered = buf.getvalue()
    if console is not None:
        _write_to_console(console, rendered)
    return rendered
