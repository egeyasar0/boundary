"""End-to-end variant runner, context load reduction, and CLI tests with FakeLLM (M11)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import boundary.cli as cli_mod
from boundary.cli import app
from boundary.llm import FakeLLM, FakeTurn
from boundary.models import ToolCall, Usage
from boundary.profile import profile_traces
from boundary.report import render_report, summarize_run
from boundary.runner import run_suite
from boundary.synthesize import save_variant_config, synthesize_variant
from boundary.tasks import load_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks" / "repo_qa"
FIXTURE_DIR = REPO_ROOT / "fixtures" / "sample_repo"


def test_variant_runner_reduces_parent_context_over_70_percent_within_budget(
    tmp_path: Path,
) -> None:
    """Verify M11: synthesized subagent variant reduces parent context load >70% within budget."""
    all_tasks = {t.id: t for t in load_tasks(TASKS_DIR)}
    needle_task = all_tasks["routing-shard-needle"]

    # 1. Baseline run: parent reads giant routing_matrix.py directly across 3 steps before answering
    baseline_script = [
        FakeTurn(
            tool_calls=[ToolCall(id="b0", name="list_files", arguments={"path": "."})],
            usage=Usage(prompt_tokens=250, completion_tokens=20, source="api"),
        ),
        FakeTurn(
            tool_calls=[
                ToolCall(
                    id="b1",
                    name="read_file",
                    arguments={
                        "path": "velox_grid/routing_matrix.py",
                        "start_line": 1,
                        "end_line": 400,
                    },
                )
            ],
            usage=Usage(prompt_tokens=450, completion_tokens=30, source="api"),
        ),
        FakeTurn(
            tool_calls=[
                ToolCall(
                    id="b2",
                    name="read_file",
                    arguments={
                        "path": "velox_grid/routing_matrix.py",
                        "start_line": 401,
                        "end_line": 820,
                    },
                )
            ],
            usage=Usage(prompt_tokens=5600, completion_tokens=30, source="api"),
        ),
        FakeTurn(
            assistant_content="SHARD-caldera-9043 and 59127",
            usage=Usage(prompt_tokens=10800, completion_tokens=25, source="api"),
        ),
    ]
    base_llm = FakeLLM(baseline_script, model="fake-gpt-oss-20b")
    base_dir, base_traces = run_suite(
        tasks=[needle_task],
        llm=base_llm,
        root_dir=FIXTURE_DIR,
        out_dir=tmp_path / "runs",
        trials=1,
        token_budget=25_000,
        run_id="run_baseline_needle",
    )
    assert len(base_traces) == 1
    base_trace = base_traces[0]
    assert base_trace.agent_variant == "baseline"

    # 2. Profile baseline trace and synthesize VariantConfig deterministically
    base_profile = profile_traces(base_traces)
    assert any(o.kind == "extract_subagent" for o in base_profile.opportunities)
    variant_cfg = synthesize_variant(
        base_profile,
        target_paths_by_call={
            (base_trace.trace_id, "b1"): "velox_grid/routing_matrix.py",
            (base_trace.trace_id, "b2"): "velox_grid/routing_matrix.py",
        },
    )
    assert "routing-shard-needle" in variant_cfg.rules

    # 3. Variant run: parent delegates inspection to isolated SubagentRunner
    variant_script = [
        # Parent step 0: list_files
        FakeTurn(
            tool_calls=[ToolCall(id="v0", name="list_files", arguments={"path": "."})],
            usage=Usage(prompt_tokens=250, completion_tokens=20, source="api"),
        ),
        # Parent step 1: calls delegate_inspection on velox_grid/routing_matrix.py
        FakeTurn(
            tool_calls=[
                ToolCall(
                    id="v1",
                    name="delegate_inspection",
                    arguments={
                        "path": "velox_grid/routing_matrix.py",
                        "query": "Find airgap_egress shard_code and witness_port",
                    },
                )
            ],
            usage=Usage(prompt_tokens=450, completion_tokens=35, source="api"),
        ),
        # Subagent step 0: greps inside routing_matrix.py
        FakeTurn(
            tool_calls=[
                ToolCall(
                    id="sub_g1",
                    name="grep",
                    arguments={
                        "pattern": "airgap_egress|SHARD-caldera-9043|59127",
                        "path": "velox_grid/routing_matrix.py",
                    },
                )
            ],
            usage=Usage(prompt_tokens=180, completion_tokens=30, source="api"),
        ),
        # Subagent step 1: returns concise <= 300 token summary to parent
        FakeTurn(
            assistant_content=(
                "airgap_egress entry: shard_code=SHARD-caldera-9043, witness_port=59127"
            ),
            usage=Usage(prompt_tokens=260, completion_tokens=20, source="api"),
        ),
        # Parent step 2: receives only concise summary and produces final answer
        FakeTurn(
            assistant_content="SHARD-caldera-9043 and 59127",
            usage=Usage(prompt_tokens=520, completion_tokens=25, source="api"),
        ),
    ]
    var_llm = FakeLLM(variant_script, model="fake-gpt-oss-20b")
    var_dir, var_traces = run_suite(
        tasks=[needle_task],
        llm=var_llm,
        root_dir=FIXTURE_DIR,
        out_dir=tmp_path / "runs",
        trials=1,
        token_budget=25_000,
        run_id="run_variant_needle",
        variant_config=variant_cfg,
    )
    assert len(var_traces) == 1
    var_trace = var_traces[0]
    assert var_trace.passed is True
    assert var_trace.stop_reason == "final"
    assert var_trace.agent_variant == "subagent_extracted"
    assert var_trace.total_usage.total < 25_000

    # Verify >70% reduction in parent context tool tokens and carry_total_tokens
    base_parent_tool_tokens = sum(
        tr.tokens_est for s in base_trace.steps for tr in s.tool_results
    )
    var_parent_tool_tokens = sum(
        tr.tokens_est for s in var_trace.steps for tr in s.tool_results
    )
    reduction_ratio = 1.0 - (var_parent_tool_tokens / base_parent_tool_tokens)
    assert reduction_ratio >= 0.70

    var_profile = profile_traces(var_traces)
    base_carry = base_profile.traces[0].carry_total_tokens
    var_carry = var_profile.traces[0].carry_total_tokens
    carry_reduction = 1.0 - (var_carry / base_carry)
    assert carry_reduction >= 0.70

    # Verify report labels Variant: subagent_extracted
    var_summary = summarize_run(var_traces, run_id=var_dir.name, token_budget=25_000)
    rendered = render_report(var_summary)
    assert "Variant: subagent_extracted" in rendered
    assert "Boundary Variant Report (subagent_extracted)" in rendered


def test_cli_run_with_variant_and_report_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `boundary run --variant <path>` and `boundary report` CLI flow."""
    runner = CliRunner()
    all_tasks = {t.id: t for t in load_tasks(TASKS_DIR)}
    needle_task = all_tasks["routing-shard-needle"]

    task_dir = tmp_path / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "08_routing_shard_needle.yaml").write_text(
        (TASKS_DIR / "08_routing_shard_needle.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    # Create a VariantConfig file via synthesize_variant
    from boundary.profile import Opportunity, ProfileSummary

    synth_cfg = synthesize_variant(
        ProfileSummary(
            traces=[],
            opportunities=[
                Opportunity(
                    id="opp_0001",
                    kind="extract_subagent",
                    task_id=needle_task.id,
                    trace_ids=["tr_base"],
                    evidence={
                        "trace_id": "tr_base",
                        "step_index": 1,
                        "call_id": "c1",
                        "tool": "read_file",
                        "size_tokens": 5003,
                        "remaining_calls": 3,
                        "carry_cost_tokens": 15009,
                        "reuse_ratio": 0.0,
                    },
                    est_upper_bound_savings_tokens=15009,
                    steps_saved=None,
                    confidence="low",
                )
            ],
        )
    )
    variant_path = save_variant_config(synth_cfg, tmp_path / "variant_config.json")

    script = [
        # Parent delegates inspection
        FakeTurn(
            tool_calls=[
                ToolCall(
                    id="call_del",
                    name="delegate_inspection",
                    arguments={
                        "path": "velox_grid/routing_matrix.py",
                        "query": "airgap_egress shard_code and witness_port",
                    },
                )
            ],
            usage=Usage(prompt_tokens=300, completion_tokens=30, source="api"),
        ),
        # Subagent returns concise summary
        FakeTurn(
            assistant_content="SHARD-caldera-9043 and witness_port 59127",
            usage=Usage(prompt_tokens=220, completion_tokens=20, source="api"),
        ),
        # Parent produces final answer
        FakeTurn(
            assistant_content="Airgap route: SHARD-caldera-9043 on port 59127.",
            usage=Usage(prompt_tokens=360, completion_tokens=20, source="api"),
        ),
    ]
    fake_llm = FakeLLM(script, model="fake-gpt-oss-20b")
    monkeypatch.setattr(cli_mod, "create_llm_client", lambda **kw: fake_llm)

    out_runs = tmp_path / "runs"
    res_run = runner.invoke(
        app,
        [
            "run",
            "--tasks",
            str(task_dir),
            "--root",
            str(FIXTURE_DIR),
            "--trials",
            "1",
            "--token-budget",
            "40000",
            "--out",
            str(out_runs),
            "--variant",
            str(variant_path),
        ],
    )
    assert res_run.exit_code == 0, res_run.output
    assert "Variant: subagent_extracted" in res_run.output
    assert "Boundary Variant Report (subagent_extracted)" in res_run.output

    run_dirs = list(out_runs.iterdir())
    assert len(run_dirs) == 1
    res_rep = runner.invoke(app, ["report", str(run_dirs[0])])
    assert res_rep.exit_code == 0, res_rep.output
    assert "Variant: subagent_extracted" in res_rep.output
