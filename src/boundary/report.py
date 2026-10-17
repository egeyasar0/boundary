"""Summary calculation and Rich report rendering for Boundary runs and profiles (M5, M8)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from io import StringIO

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from boundary.models import RunSummary, StopReason, TaskSummary, Trace
from boundary.profile import Opportunity, ProfileSummary, ToolResultProfile, TraceProfile
from boundary.stats import is_insufficient_evidence, wilson_interval

ALL_STOP_REASONS: tuple[StopReason, ...] = ("final", "max_steps", "budget", "error")
DEFAULT_REPORT_TOKEN_BUDGET = 40_000


def _write_to_console(console: Console, rendered: str) -> None:
    """Write rendered text to console.file safely across UTF-8 and Windows cp1254 streams."""
    try:
        console.file.write(rendered)
    except UnicodeEncodeError:
        enc = getattr(console.file, "encoding", None) or "utf-8"
        safe_text = rendered.encode(enc, errors="replace").decode(enc, errors="replace")
        console.file.write(safe_text)
    console.file.flush()


def _compute_tool_tokens(trace: Trace) -> int:
    """Sum estimated tokens originating from tool outputs across all steps in a trace."""
    return sum(tr.tokens_est for step in trace.steps for tr in step.tool_results)


def _aggregate_traces(
    task_id: str,
    traces: Sequence[Trace],
    *,
    token_budget: int = DEFAULT_REPORT_TOKEN_BUDGET,
) -> TaskSummary:
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

    max_used = max((t.total_usage.total for t in traces), default=0)
    max_budget_ratio = (max_used / token_budget) if token_budget > 0 else 0.0

    stop_counts: dict[StopReason, int] = {reason: 0 for reason in ALL_STOP_REASONS}
    last_error: str | None = None
    for t in traces:
        stop_counts[t.stop_reason] = stop_counts.get(t.stop_reason, 0) + 1
        if t.stop_reason == "error" and t.error:
            last_error = t.error[:200]

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
        max_budget_ratio=max_budget_ratio,
        stop_reasons=stop_counts,
        last_error=last_error,
    )


def summarize_run(
    traces: Sequence[Trace],
    *,
    run_id: str = "run",
    token_budget: int = DEFAULT_REPORT_TOKEN_BUDGET,
) -> RunSummary:
    """Aggregate a list of Traces into a deterministic RunSummary."""
    if not traces:
        raise ValueError("Cannot summarize an empty trace list.")

    by_task: dict[str, list[Trace]] = defaultdict(list)
    for trace in traces:
        by_task[trace.task_id].append(trace)

    task_summaries: list[TaskSummary] = [
        _aggregate_traces(task_id, by_task[task_id], token_budget=token_budget)
        for task_id in sorted(by_task.keys())
    ]

    overall = _aggregate_traces("__all__", traces, token_budget=token_budget)
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
        insufficient_evidence=all(ts.insufficient_evidence for ts in task_summaries)
        or overall.insufficient_evidence,
        mean_tokens=overall.mean_tokens,
        tool_token_share=overall.tool_token_share,
        max_budget_ratio=overall.max_budget_ratio,
        stop_reasons=overall.stop_reasons,
        last_error=overall.last_error,
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
    out_console = Console(file=buf, width=260, force_terminal=False)

    table = Table(
        title=f"Boundary Baseline Report - {summary.run_id} (model: {summary.model})",
        box=box.ASCII,
    )
    table.add_column("Task ID", style="bold", no_wrap=True)
    table.add_column("Pass Rate", justify="right")
    table.add_column("Wilson 95% CI", justify="center")
    table.add_column("Evidence", justify="center")
    table.add_column("Mean Tokens", justify="right")
    table.add_column("Tool Token Share", justify="right")
    table.add_column("Max Used/Budget", justify="right")
    table.add_column("Stop Reasons (budget highlight)", justify="left")
    table.add_column("Son hata", justify="left", overflow="fold")

    for ts in summary.tasks:
        evidence_str = "yetersiz kanıt" if ts.insufficient_evidence else "ok"
        table.add_row(
            ts.task_id,
            f"{ts.passed_count}/{ts.trials} ({ts.pass_rate * 100:.1f}%)",
            f"[{ts.wilson_low * 100:.1f}%, {ts.wilson_high * 100:.1f}%]",
            evidence_str,
            f"{ts.mean_tokens:.1f}",
            f"{ts.tool_token_share * 100:.1f}%",
            f"{ts.max_budget_ratio * 100:.0f}%",
            _format_stop_reasons(ts.stop_reasons, ts.trials),
            ts.last_error or "-",
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
            f"- Kanıt durumu: {overall_ev}"
        ),
        (
            f"Mean Used Tokens: {summary.mean_tokens:.1f} | "
            f"Tool Output Token Share: {summary.tool_token_share * 100:.1f}% | "
            f"Max Used/Budget Ratio (aşım oranı): {summary.max_budget_ratio * 100:.0f}%"
        ),
        f"Stop Reason Distribution: {stop_dist}",
        (
            f"Budget Stop Rate (budget oranı): "
            f"{budget_count}/{summary.total_trials} ({budget_rate:.1f}%)"
        ),
        f"Son hata: {summary.last_error or '-'}",
    ]
    out_console.print(Panel("\n".join(summary_lines), title="Run Summary", box=box.ASCII))

    rendered = buf.getvalue()
    if console is not None:
        _write_to_console(console, rendered)
    return rendered


def _format_top_results(results: Sequence[ToolResultProfile], limit: int = 3) -> str:
    """Format top N tool results by carry_cost_tokens (then size_tokens)."""
    if not results:
        return "-"
    ranked = sorted(
        results,
        key=lambda r: (-r.carry_cost_tokens, -r.size_tokens, r.step_index, r.call_id),
    )[:limit]
    parts = [
        f"{r.tool}[s{r.step_index}](size={r.size_tokens}, carry={r.carry_cost_tokens}, "
        f"reuse={r.reuse_ratio:.2f})"
        for r in ranked
    ]
    return escape("; ".join(parts))


def _format_opportunity_metric(opp: Opportunity) -> str:
    if opp.kind == "extract_subagent":
        return f"<= {opp.est_upper_bound_savings_tokens or 0} tok (upper-bound)"
    return f"{opp.steps_saved or 0} steps saved"


def render_profile_report(
    profile: ProfileSummary,
    *,
    run_id: str = "run",
    top: int = 10,
    console: Console | None = None,
) -> str:
    """Render deterministic ProfileSummary via Rich tables and summary panel (M8)."""
    buf = StringIO()
    out_console = Console(file=buf, width=260, force_terminal=False)

    by_task: dict[str, list[TraceProfile]] = defaultdict(list)
    for tp in profile.traces:
        by_task[tp.task_id].append(tp)

    opps_by_task: dict[str, list[Opportunity]] = defaultdict(list)
    for opp in profile.opportunities:
        opps_by_task[opp.task_id].append(opp)

    task_table = Table(title=f"Boundary Trace Profile - {run_id}", box=box.ASCII)
    task_table.add_column("Task ID", style="bold", no_wrap=True)
    task_table.add_column("Traces", justify="right")
    task_table.add_column("Mean Carry Share", justify="right")
    task_table.add_column("Mean Carry Tokens", justify="right")
    task_table.add_column("Opportunities", justify="right")
    task_table.add_column("Top 3 Tool Results (by carry cost)", justify="left", overflow="fold")

    for task_id in sorted(by_task.keys()):
        t_profiles = by_task[task_id]
        n = len(t_profiles)
        mean_share = sum(tp.carry_share for tp in t_profiles) / n if n > 0 else 0.0
        mean_carry = sum(tp.carry_total_tokens for tp in t_profiles) / n if n > 0 else 0.0
        all_results = [r for tp in t_profiles for r in tp.results]
        opp_cnt = len(opps_by_task.get(task_id, []))
        task_table.add_row(
            task_id,
            str(n),
            f"{mean_share * 100:.1f}%",
            f"{mean_carry:.0f}",
            str(opp_cnt),
            _format_top_results(all_results, limit=3),
        )

    out_console.print(task_table)

    top_opps = profile.opportunities[: max(0, top)]
    opp_table = Table(
        title=f"Top {len(top_opps)} Candidate Boundary Opportunities (confidence=low)",
        box=box.ASCII,
    )
    opp_table.add_column("ID", style="bold", no_wrap=True)
    opp_table.add_column("Kind", no_wrap=True)
    opp_table.add_column("Task ID", no_wrap=True)
    opp_table.add_column("Confidence", justify="center")
    opp_table.add_column("Upper-Bound Estimate", justify="right")
    opp_table.add_column("Evidence", justify="left", overflow="fold")

    for opp in top_opps:
        if opp.kind == "extract_subagent":
            ev = (
                f"trace={opp.evidence.get('trace_id')} step={opp.evidence.get('step_index')} "
                f"tool={opp.evidence.get('tool')} size={opp.evidence.get('size_tokens')} "
                f"rem={opp.evidence.get('remaining_calls')} "
                f"reuse={opp.evidence.get('reuse_ratio')}"
            )
        else:
            ev = (
                f"trace={opp.evidence.get('trace_id')} "
                f"steps={opp.evidence.get('step_indices')} "
                f"tools={opp.evidence.get('tools')}"
            )
        opp_table.add_row(
            opp.id,
            opp.kind,
            opp.task_id,
            opp.confidence,
            _format_opportunity_metric(opp),
            escape(ev),
        )

    out_console.print(opp_table)

    total_traces = len(profile.traces)
    overall_mean_share = (
        sum(tp.carry_share for tp in profile.traces) / total_traces
        if total_traces > 0
        else 0.0
    )
    extract_cnt = sum(1 for o in profile.opportunities if o.kind == "extract_subagent")
    par_cnt = sum(1 for o in profile.opportunities if o.kind == "parallelize")
    total_est_savings = sum(
        o.est_upper_bound_savings_tokens or 0 for o in profile.opportunities
    )
    estimated_cnt = sum(1 for tp in profile.traces if tp.usage_source == "estimated")
    usage_note = (
        f"UYARI: {estimated_cnt}/{total_traces} izde usage_source='estimated' (tahmini token)."
        if estimated_cnt > 0
        else (
            f"Tüm izler ({total_traces}/{total_traces}) API tarafından raporlanan "
            "usage_source='api' kullanıyor."
        )
    )

    panel_lines = [
        (
            f"Traces Profiled: {total_traces} | "
            f"Overall Mean Carry Share: {overall_mean_share * 100:.1f}%"
        ),
        (
            f"Candidate Opportunities: {len(profile.opportunities)} "
            f"(extract_subagent={extract_cnt}, parallelize={par_cnt}) | "
            f"Est. Upper-Bound Savings: <= {total_est_savings} tokens"
        ),
        f"Usage Source: {usage_note}",
        (
            "Not (R9 / §8): Taşıma maliyeti sağlayıcı prompt önbelleğini (prompt caching) "
            "hesaba katmaz; tüm fırsatlar confidence='low' üst sınır adaylarıdır."
        ),
    ]
    out_console.print(Panel("\n".join(panel_lines), title="Profile Summary", box=box.ASCII))

    rendered = buf.getvalue()
    if console is not None:
        _write_to_console(console, rendered)
    return rendered
