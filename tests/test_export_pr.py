"""Unit tests for deterministic refactor patch and PULL_REQUEST.md generation (M13 — R16)."""

from __future__ import annotations

from pathlib import Path

from boundary.export_pr import (
    DIFF_ALIAS_FILENAME,
    PATCH_FILENAME,
    PR_MARKDOWN_FILENAME,
    export_refactor_pr,
)
from boundary.models import Step, Trace, Usage
from boundary.profile import Opportunity, ProfileSummary
from boundary.synthesize import save_variant_config, synthesize_variant
from boundary.traces import write_traces


def _make_sample_variant_config(tmp_path: Path) -> Path:
    profile = ProfileSummary(
        traces=[],
        opportunities=[
            Opportunity(
                id="opp_0011",
                kind="extract_subagent",
                task_id="routing-shard-needle",
                trace_ids=["tr_1"],
                evidence={
                    "trace_id": "tr_1",
                    "step_index": 1,
                    "call_id": "c1",
                    "tool": "read_file",
                    "size_tokens": 5003,
                    "remaining_calls": 5,
                    "carry_cost_tokens": 25015,
                    "reuse_ratio": 0.0,
                },
                est_upper_bound_savings_tokens=25015,
                steps_saved=None,
                confidence="low",
            ),
            Opportunity(
                id="opp_0012",
                kind="extract_subagent",
                task_id="ledger-anomaly-needle",
                trace_ids=["tr_2"],
                evidence={
                    "trace_id": "tr_2",
                    "step_index": 1,
                    "call_id": "c2",
                    "tool": "read_file",
                    "size_tokens": 5003,
                    "remaining_calls": 4,
                    "carry_cost_tokens": 20012,
                    "reuse_ratio": 0.01,
                },
                est_upper_bound_savings_tokens=20012,
                steps_saved=None,
                confidence="low",
            ),
        ],
    )
    cfg = synthesize_variant(
        profile,
        target_paths_by_call={
            ("tr_1", "c1"): "velox_grid/routing_matrix.py",
            ("tr_2", "c2"): "velox_grid/ledger.py",
        },
    )
    return save_variant_config(cfg, tmp_path / "variant_config.json")


def test_export_refactor_pr_generates_deterministic_patch_and_markdown(
    tmp_path: Path,
) -> None:
    """Verify R16: export_refactor_pr creates deterministic patch and PULL_REQUEST.md."""
    cfg_path = _make_sample_variant_config(tmp_path)
    out_dir_1 = tmp_path / "pr_out_1"
    out_dir_2 = tmp_path / "pr_out_2"

    res1 = export_refactor_pr(cfg_path, out_dir_1)
    res2 = export_refactor_pr(cfg_path, out_dir_2)

    assert (out_dir_1 / PATCH_FILENAME).exists()
    assert (out_dir_1 / DIFF_ALIAS_FILENAME).exists()
    assert (out_dir_1 / PR_MARKDOWN_FILENAME).exists()

    patch_1 = res1.patch_path.read_text(encoding="utf-8")
    patch_2 = res2.patch_path.read_text(encoding="utf-8")
    assert patch_1 == patch_2
    assert res1.diff_alias_path.read_text(encoding="utf-8") == patch_1

    # Verify unified diff headers and content
    assert "--- a/agent_config.py" in patch_1
    assert "+++ b/agent_config.py" in patch_1
    assert "@@ -1,12 +1,34 @@" in patch_1
    assert '+AGENT_VARIANT = "subagent_extracted"' in patch_1
    assert "routing-shard-needle" in patch_1
    assert "velox_grid/routing_matrix.py" in patch_1
    assert "SubagentRunner" in patch_1

    # Verify PULL_REQUEST.md sections and metrics
    md_1 = res1.pr_markdown_path.read_text(encoding="utf-8")
    md_2 = res2.pr_markdown_path.read_text(encoding="utf-8")
    assert md_1 == md_2
    assert "## 2. Empirical Evidence Table (Baseline vs Variant)" in md_1
    assert "## 3. Carry Cost (Taşıma Maliyeti) Savings Metrics" in md_1
    assert "## 4. Context Leak Prevention Breakdown (Protected Files & Tools)" in md_1
    assert "<= 45027" in md_1
    assert "`velox_grid/routing_matrix.py`" in md_1
    assert "`velox_grid/ledger.py`" in md_1
    assert "`delegate_inspection`" in md_1


def test_export_refactor_pr_with_comparison_runs(tmp_path: Path) -> None:
    """Verify PULL_REQUEST.md embeds Baseline vs Variant comparison table when runs provided."""
    cfg_path = _make_sample_variant_config(tmp_path)

    def _trace(tid: str, passed: bool, tokens: int, variant: str) -> Trace:
        u = Usage(prompt_tokens=tokens, completion_tokens=20, source="api")
        return Trace(
            trace_id=f"tr_{variant}_{tid}",
            task_id=tid,
            trial=1,
            agent_variant=variant,
            model="fake-gpt-oss-20b",
            config_hash="abc123456",
            steps=[
                Step(
                    index=0,
                    assistant_content="ans",
                    tool_calls=[],
                    tool_results=[],
                    usage=u,
                    latency_ms=10,
                )
            ],
            final_answer="ans",
            passed=passed,
            total_usage=u,
            stop_reason="final" if passed else "budget",
        )

    base_dir = tmp_path / "base_run"
    var_dir = tmp_path / "var_run"
    write_traces(
        base_dir,
        [
            _trace("routing-shard-needle", False, 58_000, "baseline"),
            _trace("ledger-anomaly-needle", True, 30_000, "baseline"),
        ],
    )
    write_traces(
        var_dir,
        [
            _trace("routing-shard-needle", True, 4_200, "subagent_extracted"),
            _trace("ledger-anomaly-needle", True, 3_800, "subagent_extracted"),
        ],
    )

    res = export_refactor_pr(
        cfg_path,
        tmp_path / "pr_enriched",
        baseline_run_dir=base_dir,
        variant_run_dir=var_dir,
        token_budget=40_000,
    )
    md = res.pr_markdown_path.read_text(encoding="utf-8")
    assert "ACCEPT (Boundary Justified)" in md
    assert "| `routing-shard-needle` |" in md
    assert "| `ledger-anomaly-needle` |" in md
