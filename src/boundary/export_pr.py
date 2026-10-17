"""Deterministic refactor patch and Pull Request generator (M13 — R16).

Rule R16 (Deterministic Patch Generation):
`boundary export-pr` generates a unified diff patch (`boundary_refactor.patch` and
`subagent_patch.diff`) and an empirical Markdown justification (`PULL_REQUEST.md`)
deterministically without any LLM or network calls.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from boundary.compare import ComparisonReport, compare_runs
from boundary.report import DEFAULT_REPORT_TOKEN_BUDGET
from boundary.synthesize import VariantConfig, load_variant_config

PATCH_FILENAME = "boundary_refactor.patch"
DIFF_ALIAS_FILENAME = "subagent_patch.diff"
PR_MARKDOWN_FILENAME = "PULL_REQUEST.md"


class ExportPRResult(BaseModel):
    """Paths and content produced by `export_refactor_pr`."""

    out_dir: Path
    patch_path: Path
    diff_alias_path: Path
    pr_markdown_path: Path
    variant_name: str
    rule_count: int
    total_upper_bound_savings_tokens: int


def build_unified_patch(config: VariantConfig) -> str:
    """Build a deterministic unified diff patch wiring `VariantConfig` into agent tool routing."""
    sorted_task_ids = sorted(config.rules.keys())
    rule_lines: list[str] = []
    for tid in sorted_task_ids:
        rule = config.rules[tid]
        paths_repr = repr(rule.target_paths)
        tools_repr = repr(rule.intercepted_tools)
        rule_lines.append(
            f'+    "{tid}": {{"boundary_tool": "{rule.boundary_tool}", '
            f'"intercepted_tools": {tools_repr}, "target_paths": {paths_repr}}},'
        )
    if not rule_lines:
        rule_lines.append('+    # No extract_subagent rules triggered')

    patch_lines = [
        "--- a/agent_config.py",
        "+++ b/agent_config.py",
        "@@ -1,12 +1,34 @@",
        " # Agent tool routing configuration",
        '-AGENT_VARIANT = "baseline"',
        "-ENABLE_ISOLATED_SUBAGENT = False",
        "+from boundary.subagent import SubagentRunner",
        "+from boundary.synthesize import load_variant_config",
        "+",
        f'+AGENT_VARIANT = "{config.variant_name}"',
        "+ENABLE_ISOLATED_SUBAGENT = True",
        f"+SUBAGENT_MAX_STEPS = {config.subagent_max_steps}",
        f"+SUBAGENT_BUDGET = {config.subagent_budget}",
        f"+MAX_SUMMARY_TOKENS = {config.max_summary_tokens}",
        "+",
        "+DELEGATION_RULES = {",
        *rule_lines,
        "+}",
        "",
        " def execute_agent_tool(tc, *, task_id, llm, sandbox, remaining_budget):",
        "-    return sandbox.execute(tc)",
        "+    if ENABLE_ISOLATED_SUBAGENT and task_id in DELEGATION_RULES:",
        "+        runner = SubagentRunner(",
        "+            llm=llm,",
        "+            sandbox=sandbox,",
        "+            subagent_max_steps=SUBAGENT_MAX_STEPS,",
        "+            subagent_budget=SUBAGENT_BUDGET,",
        "+            max_summary_tokens=MAX_SUMMARY_TOKENS,",
        "+        )",
        "+        return runner.execute_tool_call(",
        "+            tc,",
        "+            remaining_global_budget=remaining_budget,",
        "+        ).tool_result",
        "+    return sandbox.execute(tc)",
    ]
    return "\n".join(patch_lines) + "\n"


def build_pr_markdown(
    config: VariantConfig,
    *,
    comparison: ComparisonReport | None = None,
) -> str:
    """Build deterministic `PULL_REQUEST.md` with empirical evidence and carry-cost savings."""
    sorted_task_ids = sorted(config.rules.keys())
    total_savings = sum(
        config.rules[tid].est_upper_bound_savings_tokens for tid in sorted_task_ids
    )
    all_target_paths: list[str] = sorted(
        {
            p
            for tid in sorted_task_ids
            for p in config.rules[tid].target_paths
        }
    )

    lines: list[str] = [
        f"# Refactor: Extract Isolated Sub-Agent Boundaries (`{config.variant_name}`)",
        "",
        "## 1. Summary",
        "",
        (
            "This automated refactor introduces isolated sub-agent boundaries "
            f"(`{config.variant_name}`) for `{len(sorted_task_ids)}` task(s) where high context "
            "carry cost and low downstream line reuse (`extract_subagent`) were detected."
        ),
        "",
        "- **Isolation Rule (R11):** Sub-agents never receive parent message history and return "
        f"a bounded summary (`<= {config.max_summary_tokens}` tokens).",
        "- **Consolidated Budget (R12):** All sub-agent prompt + completion tokens are deducted "
        f"from the parent agent's global budget (`subagent_budget={config.subagent_budget}`, "
        f"`subagent_max_steps={config.subagent_max_steps}`).",
        "",
        "## 2. Empirical Evidence Table (Baseline vs Variant)",
        "",
    ]

    if comparison is not None:
        ov = comparison.overall
        lines.extend(
            [
                f"- **Overall Decision:** `{ov.decision_label}`",
                f"- **Equal Token Budget:** `{comparison.token_budget}` tokens",
                f"- **Overall Pass Rate:** `{ov.baseline_pass_rate * 100:.1f}%` -> "
                f"`{ov.variant_pass_rate * 100:.1f}%` (`{ov.pass_rate_delta * 100:+.1f}%`)",
                f"- **Overall Mean Tokens:** `{ov.baseline_mean_tokens:.1f}` -> "
                f"`{ov.variant_mean_tokens:.1f}` (`{ov.token_delta_pct:+.1f}%`)",
                "",
                "| Task ID | Baseline Pass | Variant Pass | Pass Rate Delta | "
                "Baseline Tokens | Variant Tokens | Token Delta % | Decision |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for tc in comparison.tasks:
            lines.append(
                f"| `{tc.task_id}` | {tc.baseline_pass_rate * 100:.1f}% | "
                f"{tc.variant_pass_rate * 100:.1f}% | {tc.pass_rate_delta * 100:+.1f}% | "
                f"{tc.baseline_mean_tokens:.1f} | {tc.variant_mean_tokens:.1f} | "
                f"{tc.token_delta_pct:+.1f}% | `{tc.decision_label}` |"
            )
    else:
        lines.extend(
            [
                "| Task ID | Baseline Tool | Synthesized Boundary Tool | "
                "Est. Upper-Bound Carry Savings (Tokens) | Source Opportunities |",
                "| --- | --- | --- | ---: | --- |",
            ]
        )
        if sorted_task_ids:
            for tid in sorted_task_ids:
                rule = config.rules[tid]
                intercepted_str = ", ".join(f"`{t}`" for t in rule.intercepted_tools)
                opps_str = ", ".join(f"`{oid}`" for oid in rule.source_opportunity_ids) or "-"
                lines.append(
                    f"| `{tid}` | {intercepted_str} | `{rule.boundary_tool}` | "
                    f"<= {rule.est_upper_bound_savings_tokens} | {opps_str} |"
                )
        else:
            lines.append("| *(none)* | - | - | 0 | - |")

    lines.extend(
        [
            "",
            "## 3. Carry Cost (Taşıma Maliyeti) Savings Metrics",
            "",
            f"- **Total Estimated Upper-Bound Carry Savings:** `<= {total_savings}` tokens",
            f"- **Parent Context Summary Cap:** `<= {config.max_summary_tokens}` tokens per call",
            f"- **Sub-Agent Step Ceiling:** `{config.subagent_max_steps}` steps",
            f"- **Sub-Agent Token Sub-Budget:** `{config.subagent_budget}` tokens",
            "",
            "## 4. Context Leak Prevention Breakdown (Protected Files & Tools)",
            "",
        ]
    )

    if sorted_task_ids:
        for tid in sorted_task_ids:
            rule = config.rules[tid]
            paths_str = (
                ", ".join(f"`{p}`" for p in rule.target_paths)
                if rule.target_paths
                else "`*` (all task paths)"
            )
            tools_str = ", ".join(f"`{t}`" for t in rule.intercepted_tools)
            lines.append(
                f"- **`{tid}`**: Intercepts {tools_str} -> delegates to `{rule.boundary_tool}` "
                f"on {paths_str} (est. savings: `<= {rule.est_upper_bound_savings_tokens}` tokens)."
            )
    else:
        lines.append("- No tasks required sub-agent extraction.")

    if all_target_paths:
        lines.extend(
            [
                "",
                "### Protected Repository Files",
                "",
                *[f"- `{p}`" for p in all_target_paths],
            ]
        )

    return "\n".join(lines) + "\n"


def export_refactor_pr(
    variant_config_path: Path | str,
    out_dir: Path | str = Path("pr_output"),
    *,
    baseline_run_dir: Path | str | None = None,
    variant_run_dir: Path | str | None = None,
    token_budget: int = DEFAULT_REPORT_TOKEN_BUDGET,
) -> ExportPRResult:
    """Generate `boundary_refactor.patch`, `subagent_patch.diff`, and `PULL_REQUEST.md` (M13)."""
    config = load_variant_config(variant_config_path)

    comparison: ComparisonReport | None = None
    if baseline_run_dir is not None and variant_run_dir is not None:
        comparison = compare_runs(
            baseline_run_dir,
            variant_run_dir,
            token_budget=token_budget,
        )

    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    patch_content = build_unified_patch(config)
    pr_md_content = build_pr_markdown(config, comparison=comparison)

    patch_path = target_dir / PATCH_FILENAME
    diff_alias_path = target_dir / DIFF_ALIAS_FILENAME
    pr_md_path = target_dir / PR_MARKDOWN_FILENAME

    patch_path.write_text(patch_content, encoding="utf-8", newline="\n")
    diff_alias_path.write_text(patch_content, encoding="utf-8", newline="\n")
    pr_md_path.write_text(pr_md_content, encoding="utf-8", newline="\n")

    total_savings = sum(
        r.est_upper_bound_savings_tokens for r in config.rules.values()
    )
    return ExportPRResult(
        out_dir=target_dir,
        patch_path=patch_path,
        diff_alias_path=diff_alias_path,
        pr_markdown_path=pr_md_path,
        variant_name=config.variant_name,
        rule_count=len(config.rules),
        total_upper_bound_savings_tokens=total_savings,
    )
