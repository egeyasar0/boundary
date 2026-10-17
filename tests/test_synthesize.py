"""Unit and CLI tests for deterministic candidate boundary synthesis (M10 — R13)."""

from __future__ import annotations

import shutil
from pathlib import Path

from typer.testing import CliRunner

from boundary.cli import app
from boundary.models import ToolCall
from boundary.profile import Opportunity, ProfileSummary, profile_traces
from boundary.synthesize import (
    load_variant_config,
    synthesize_variant,
)
from boundary.traces import read_traces

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_SAMPLE_PATH = REPO_ROOT / "tests" / "data" / "baseline_sample.jsonl"


def test_synthesize_variant_filters_extract_subagent_and_is_deterministic() -> None:
    """Verify R13: only extract_subagent opportunities generate rules deterministically."""
    profile = ProfileSummary(
        traces=[],
        opportunities=[
            Opportunity(
                id="opp_0001",
                kind="extract_subagent",
                task_id="heavy-needle-task",
                trace_ids=["tr_1"],
                evidence={
                    "trace_id": "tr_1",
                    "step_index": 1,
                    "call_id": "call_read_1",
                    "tool": "read_file",
                    "size_tokens": 5003,
                    "remaining_calls": 4,
                    "carry_cost_tokens": 20012,
                    "reuse_ratio": 0.0,
                },
                est_upper_bound_savings_tokens=20012,
                steps_saved=None,
                confidence="low",
            ),
            Opportunity(
                id="opp_0002",
                kind="parallelize",
                task_id="easy-parallel-task",
                trace_ids=["tr_2"],
                evidence={"step_indices": [1, 2], "tools": ["read_file", "read_file"]},
                est_upper_bound_savings_tokens=None,
                steps_saved=1,
                confidence="low",
            ),
            Opportunity(
                id="opp_0003",
                kind="extract_subagent",
                task_id="heavy-grep-task",
                trace_ids=["tr_3"],
                evidence={
                    "trace_id": "tr_3",
                    "step_index": 0,
                    "call_id": "call_grep_1",
                    "tool": "grep",
                    "size_tokens": 3200,
                    "remaining_calls": 3,
                    "carry_cost_tokens": 9600,
                    "reuse_ratio": 0.02,
                },
                est_upper_bound_savings_tokens=9600,
                steps_saved=None,
                confidence="low",
            ),
        ],
    )

    path_map = {
        ("tr_1", "call_read_1"): "velox_grid/routing_table.py",
        ("tr_3", "call_grep_1"): "velox_grid/shard_ledger_a.py",
    }

    cfg1 = synthesize_variant(profile, target_paths_by_call=path_map)
    cfg2 = synthesize_variant(profile, target_paths_by_call=path_map)

    assert cfg1.model_dump_json(indent=2) == cfg2.model_dump_json(indent=2)
    assert cfg1.variant_name == "subagent_extracted"
    assert cfg1.subagent_max_steps == 3
    assert cfg1.subagent_budget == 8000
    assert set(cfg1.rules.keys()) == {"heavy-grep-task", "heavy-needle-task"}
    assert "easy-parallel-task" not in cfg1.rules

    needle_rule = cfg1.rules["heavy-needle-task"]
    assert needle_rule.intercepted_tools == ["read_file"]
    assert needle_rule.boundary_tool == "delegate_inspection"
    assert needle_rule.target_paths == ["velox_grid/routing_table.py"]
    assert needle_rule.est_upper_bound_savings_tokens == 20012

    grep_rule = cfg1.rules["heavy-grep-task"]
    assert grep_rule.intercepted_tools == ["grep"]
    assert grep_rule.boundary_tool == "delegate_search"


def test_variant_config_schema_injection_and_delegation_filtering() -> None:
    """Verify VariantConfig replaces intercepted tool schema and filters tool calls accurately."""
    profile = ProfileSummary(
        traces=[],
        opportunities=[
            Opportunity(
                id="opp_0001",
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
            )
        ],
    )
    cfg = synthesize_variant(
        profile,
        target_paths_by_call={("tr_1", "c1"): "velox_grid/routing_table.py"},
    )

    # For routing-shard-needle, read_file is replaced by delegate_inspection
    needle_schemas = cfg.get_tool_schemas("routing-shard-needle")
    needle_tool_names = [s["function"]["name"] for s in needle_schemas]
    assert "read_file" not in needle_tool_names
    assert "delegate_inspection" in needle_tool_names
    assert "list_files" in needle_tool_names
    assert "grep" in needle_tool_names

    # For an unaffected task, baseline schemas remain intact
    baseline_schemas = cfg.get_tool_schemas("retry-ceiling")
    baseline_names = [s["function"]["name"] for s in baseline_schemas]
    assert baseline_names == ["list_files", "read_file", "grep"]

    # Delegation filtering checks
    assert cfg.should_delegate(
        "routing-shard-needle",
        ToolCall(
            id="t1",
            name="delegate_inspection",
            arguments={"path": "velox_grid/routing_table.py", "query": "airgap"},
        ),
    )
    assert cfg.should_delegate(
        "routing-shard-needle",
        ToolCall(
            id="t2",
            name="read_file",
            arguments={"path": "velox_grid/routing_table.py"},
        ),
    )
    assert not cfg.should_delegate(
        "routing-shard-needle",
        ToolCall(
            id="t3",
            name="read_file",
            arguments={"path": "velox_grid/client.py"},
        ),
    )
    assert not cfg.should_delegate(
        "retry-ceiling",
        ToolCall(
            id="t4",
            name="read_file",
            arguments={"path": "velox_grid/routing_table.py"},
        ),
    )


def test_cli_synthesize_command_end_to_end(tmp_path: Path) -> None:
    """Verify `boundary synthesize <run_dir> [--out ...]` reads profile.json and writes config."""
    runner = CliRunner()
    run_dir = tmp_path / "run_sample"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(BASELINE_SAMPLE_PATH, run_dir / "traces.jsonl")

    traces = read_traces(run_dir)
    profile = profile_traces(traces)
    (run_dir / "profile.json").write_text(
        profile.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )

    out_file = tmp_path / "custom_variant.json"
    res = runner.invoke(app, ["synthesize", str(run_dir), "--out", str(out_file)])
    assert res.exit_code == 0, res.output
    assert out_file.exists()

    loaded = load_variant_config(out_file)
    assert loaded.variant_name == "subagent_extracted"
    assert "routing-shard-needle" in loaded.rules
    assert loaded.rules["routing-shard-needle"].boundary_tool == "delegate_inspection"
    assert "velox_grid/routing_matrix.py" in loaded.rules["routing-shard-needle"].target_paths
