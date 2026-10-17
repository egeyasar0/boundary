"""Full lifecycle end-to-end tests for Faz 4 CLI commands with FakeLLM (M14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import boundary.cli as cli_mod
from boundary.cli import app
from boundary.compare import ComparisonReport
from boundary.llm import FakeLLM, FakeTurn
from boundary.models import ToolCall, Usage

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks" / "repo_qa"
FIXTURE_DIR = REPO_ROOT / "fixtures" / "sample_repo"


def test_full_lifecycle_run_profile_synthesize_variant_compare_export_pr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify M14: full 6-step CLI pipeline from run to export-pr."""
    runner = CliRunner()

    # Copy two representative tasks:
    # 1. 01_retry_ceiling.yaml (passes in baseline and variant at identical cost -> REJECT)
    # 2. 08_routing_shard_needle.yaml (heavy carry task -> blows budget in baseline -> ACCEPT)
    task_dir = tmp_path / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("01_retry_ceiling.yaml", "08_routing_shard_needle.yaml"):
        (task_dir / fname).write_text(
            (TASKS_DIR / fname).read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    # Step 1: Baseline run (trials=1, token_budget=25000)
    # Task 01 passes quickly; Task 08 reads giant routing_matrix.py across 3 steps and blows budget
    baseline_script = [
        # Task 01: retry-ceiling
        FakeTurn(
            assistant_content=(
                "MAX_BACKOFF_JITTER_MS is 7919 and BACKOFF_STRATEGY_TAG is decorrelated-halcyon-v4."
            ),
            usage=Usage(prompt_tokens=200, completion_tokens=20, source="api"),
        ),
        # Task 08: routing-shard-needle (reads giant file across steps, exceeds 25,000 budget)
        FakeTurn(
            tool_calls=[ToolCall(id="b0", name="list_files", arguments={"path": "."})],
            usage=Usage(prompt_tokens=300, completion_tokens=25, source="api"),
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
            usage=Usage(prompt_tokens=600, completion_tokens=30, source="api"),
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
            usage=Usage(prompt_tokens=9500, completion_tokens=30, source="api"),
        ),
        FakeTurn(
            tool_calls=[ToolCall(id="b3", name="list_files", arguments={"path": "velox_grid"})],
            usage=Usage(prompt_tokens=16000, completion_tokens=30, source="api"),
        ),
    ]
    base_llm = FakeLLM(baseline_script, model="fake-gpt-oss-20b")
    monkeypatch.setattr(cli_mod, "create_llm_client", lambda **kw: base_llm)

    base_out = tmp_path / "runs_base"
    res_base = runner.invoke(
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
            "25000",
            "--out",
            str(base_out),
        ],
    )
    assert res_base.exit_code == 0, res_base.output
    base_dirs = list(base_out.iterdir())
    assert len(base_dirs) == 1
    baseline_run_dir = base_dirs[0]

    # Step 2: Profile baseline run (`boundary profile <baseline_run_dir>`)
    res_prof = runner.invoke(app, ["profile", str(baseline_run_dir)])
    assert res_prof.exit_code == 0, res_prof.output
    profile_json_path = baseline_run_dir / "profile.json"
    assert profile_json_path.is_file()

    # Step 3: Synthesize VariantConfig (`boundary synthesize <baseline_run_dir>`)
    variant_cfg_path = tmp_path / "variant_config.json"
    res_synth = runner.invoke(
        app,
        ["synthesize", str(baseline_run_dir), "--out", str(variant_cfg_path)],
    )
    assert res_synth.exit_code == 0, res_synth.output
    assert variant_cfg_path.is_file()

    # Step 4: Variant run (`boundary run --variant <variant_cfg_path>`)
    variant_script = [
        # Task 01: retry-ceiling (identical cost -> Occam's Razor REJECT on task 01)
        FakeTurn(
            assistant_content=(
                "MAX_BACKOFF_JITTER_MS is 7919 and BACKOFF_STRATEGY_TAG is decorrelated-halcyon-v4."
            ),
            usage=Usage(prompt_tokens=200, completion_tokens=20, source="api"),
        ),
        # Task 08: routing-shard-needle (uses delegate_inspection via SubagentRunner)
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
            usage=Usage(prompt_tokens=350, completion_tokens=30, source="api"),
        ),
        # Subagent turn: returns concise summary
        FakeTurn(
            assistant_content="airgap_egress: shard_code=SHARD-caldera-9043, witness_port=59127",
            usage=Usage(prompt_tokens=220, completion_tokens=20, source="api"),
        ),
        # Parent final answer
        FakeTurn(
            assistant_content="SHARD-caldera-9043 and 59127",
            usage=Usage(prompt_tokens=420, completion_tokens=20, source="api"),
        ),
    ]
    var_llm = FakeLLM(variant_script, model="fake-gpt-oss-20b")
    monkeypatch.setattr(cli_mod, "create_llm_client", lambda **kw: var_llm)

    var_out = tmp_path / "runs_variant"
    res_var = runner.invoke(
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
            "25000",
            "--out",
            str(var_out),
            "--variant",
            str(variant_cfg_path),
        ],
    )
    assert res_var.exit_code == 0, res_var.output
    var_dirs = list(var_out.iterdir())
    assert len(var_dirs) == 1
    variant_run_dir = var_dirs[0]

    # Step 5: Compare baseline and variant (`boundary compare <baseline_dir> <variant_dir>`)
    res_cmp = runner.invoke(
        app,
        [
            "compare",
            str(baseline_run_dir),
            str(variant_run_dir),
            "--token-budget",
            "25000",
        ],
    )
    assert res_cmp.exit_code == 0, res_cmp.output
    assert "Boundary Comparison" in res_cmp.output
    assert "ACCEPT (Boundary Justified)" in res_cmp.output
    assert "REJECT (Over-engineered)" in res_cmp.output

    # Verify JSON comparison mode (`boundary compare --json`)
    res_cmp_json = runner.invoke(
        app,
        [
            "compare",
            str(baseline_run_dir),
            str(variant_run_dir),
            "--token-budget",
            "25000",
            "--json",
        ],
    )
    assert res_cmp_json.exit_code == 0, res_cmp_json.output
    parsed_cmp = ComparisonReport.model_validate_json(res_cmp_json.output)
    assert parsed_cmp.overall.decision == "ACCEPT"
    assert parsed_cmp.overall.decision_label == "ACCEPT (Boundary Justified)"
    by_task = {t.task_id: t for t in parsed_cmp.tasks}
    assert by_task["retry-ceiling"].decision == "REJECT"
    assert by_task["retry-ceiling"].decision_label == "REJECT (Over-engineered)"
    assert by_task["routing-shard-needle"].decision == "ACCEPT"
    assert by_task["routing-shard-needle"].decision_label == "ACCEPT (Boundary Justified)"

    # Step 6: Export PR artifacts (`boundary export-pr <variant_config.json>`)
    pr_out_dir = tmp_path / "pr_output"
    res_export = runner.invoke(
        app,
        [
            "export-pr",
            str(variant_cfg_path),
            "--out",
            str(pr_out_dir),
            "--baseline",
            str(baseline_run_dir),
            "--variant-run",
            str(variant_run_dir),
            "--token-budget",
            "25000",
        ],
    )
    assert res_export.exit_code == 0, res_export.output
    patch_file = pr_out_dir / "boundary_refactor.patch"
    diff_file = pr_out_dir / "subagent_patch.diff"
    pr_md_file = pr_out_dir / "PULL_REQUEST.md"
    assert patch_file.is_file()
    assert diff_file.is_file()
    assert pr_md_file.is_file()

    pr_md = pr_md_file.read_text(encoding="utf-8")
    assert "routing-shard-needle" in pr_md
    assert "ACCEPT (Boundary Justified)" in pr_md
    assert "Empirical Evidence Table (Baseline vs Variant)" in pr_md


def test_cli_compare_and_export_pr_error_handling(tmp_path: Path) -> None:
    """Verify graceful error exit codes when compare or export-pr receives missing inputs."""
    runner = CliRunner()

    res_cmp_err = runner.invoke(
        app,
        ["compare", str(tmp_path / "missing_base"), str(tmp_path / "missing_var")],
    )
    assert res_cmp_err.exit_code == 1
    assert "Comparison failed:" in res_cmp_err.output

    res_exp_err = runner.invoke(
        app,
        ["export-pr", str(tmp_path / "missing_variant.json"), "--out", str(tmp_path / "out")],
    )
    assert res_exp_err.exit_code == 1
    assert "Export failed:" in res_exp_err.output
    assert not json.dumps(res_exp_err.output).startswith("Traceback")
