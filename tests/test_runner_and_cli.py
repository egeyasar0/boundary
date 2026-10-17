"""End-to-end tests for runner, report, and Typer CLI using FakeLLM (M5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import boundary.cli as cli_mod
from boundary.cli import app
from boundary.llm import FakeLLM, FakeTurn, LLMClient
from boundary.models import ToolCall, Usage
from boundary.report import render_report, summarize_run
from boundary.runner import run_suite
from boundary.tasks import load_tasks
from boundary.traces import read_traces

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks" / "repo_qa"
FIXTURE_DIR = REPO_ROOT / "fixtures" / "sample_repo"

EXPECTED_ANSWERS_BY_ORDER: list[str] = [
    "7919 decorrelated-halcyon-v4",
    "VXG-KRYPTON-8831",
    "4391 and .vxg-cold-z9",
    "kr-obsidian-6173 and 3467",
    "1973 and 6217",
    "epoch-zircon-5521 and 9487",
    "AUD-zephyr-7309 and 8147",
    "SHARD-caldera-9043 and 59127",
]


def test_runner_and_report_highlight_budget_rate(tmp_path: Path) -> None:
    tasks = load_tasks(TASKS_DIR)[:2]
    # 2 tasks x 3 trials = 6 trials total.
    # For task 1: 2 pass ("final"), 1 hits "budget".
    # For task 2: 3 pass ("final").
    script: list[FakeTurn] = [
        # Task 1, Trial 1: tool call + final pass
        FakeTurn(
            tool_calls=[
                ToolCall(id="c1", name="read_file", arguments={"path": "velox_grid/client.py"})
            ],
            usage=Usage(prompt_tokens=100, completion_tokens=20, source="api"),
        ),
        FakeTurn(
            assistant_content="7919 decorrelated-halcyon-v4",
            usage=Usage(prompt_tokens=150, completion_tokens=20, source="api"),
        ),
        # Task 1, Trial 2: tool call exceeding token_budget=500 -> next step stops on budget
        FakeTurn(
            tool_calls=[ToolCall(id="c2", name="list_files", arguments={"path": "."})],
            usage=Usage(prompt_tokens=450, completion_tokens=100, source="api"),
        ),
        # Task 1, Trial 3: final pass directly
        FakeTurn(
            assistant_content="7919 decorrelated-halcyon-v4",
            usage=Usage(prompt_tokens=100, completion_tokens=20, source="api"),
        ),
        # Task 2, Trials 1..3: exact_normalized pass
        FakeTurn(
            assistant_content="VXG-KRYPTON-8831",
            usage=Usage(prompt_tokens=90, completion_tokens=10, source="api"),
        ),
        FakeTurn(
            assistant_content="VXG-KRYPTON-8831",
            usage=Usage(prompt_tokens=90, completion_tokens=10, source="api"),
        ),
        FakeTurn(
            assistant_content="VXG-KRYPTON-8831",
            usage=Usage(prompt_tokens=90, completion_tokens=10, source="api"),
        ),
    ]

    fake_llm = FakeLLM(script, model="fake-baseline")
    run_dir, traces = run_suite(
        tasks=tasks,
        llm=fake_llm,
        root_dir=FIXTURE_DIR,
        out_dir=tmp_path / "runs",
        trials=3,
        token_budget=500,
        seed=100,
        run_id="run_test_suite",
    )

    assert len(traces) == 6
    disk_traces = read_traces(run_dir)
    assert disk_traces == traces

    summary = summarize_run(disk_traces, run_id=run_dir.name)
    assert summary.total_trials == 6
    assert summary.total_passed == 5
    assert summary.stop_reasons["final"] == 5
    assert summary.stop_reasons["budget"] == 1
    assert summary.tool_token_share > 0.0

    report_text = render_report(summary)
    assert "retry-ceiling" in report_text
    assert "codec-magic-header" in report_text
    assert "yetersiz kanıt" in report_text
    assert "Budget Stop Rate (budget oranı): 1/6 (16.7%)" in report_text
    assert "(budget: 33%)" in report_text


def test_cli_run_and_report_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Build 8 tasks x 3 trials = 24 scripted turns (each task does 1 tool call + 1 final answer)
    turns: list[FakeTurn] = []
    for ans in EXPECTED_ANSWERS_BY_ORDER:
        for _ in range(3):
            turns.append(
                FakeTurn(
                    tool_calls=[ToolCall(id="tc_1", name="list_files", arguments={"path": "."})],
                    usage=Usage(prompt_tokens=120, completion_tokens=30, source="api"),
                )
            )
            turns.append(
                FakeTurn(
                    assistant_content=ans,
                    tool_calls=[],
                    usage=Usage(prompt_tokens=200, completion_tokens=40, source="api"),
                )
            )

    fake_llm = FakeLLM(turns, model="fake-cli-model")

    def _fake_factory(
        *, model: str | None = None, base_url: str | None = None
    ) -> LLMClient:
        return fake_llm

    monkeypatch.setattr(cli_mod, "create_llm_client", _fake_factory)

    runner = CliRunner()
    out_dir = tmp_path / "runs"
    res_run = runner.invoke(
        app,
        [
            "run",
            "--tasks",
            str(TASKS_DIR),
            "--root",
            str(FIXTURE_DIR),
            "--trials",
            "3",
            "--token-budget",
            "40000",
            "--out",
            str(out_dir),
            "--model",
            "fake-cli-model",
            "--seed",
            "7",
        ],
    )
    assert res_run.exit_code == 0, res_run.output
    assert "24 traces written" in res_run.output

    run_subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(run_subdirs) == 1
    created_run_dir = run_subdirs[0]

    res_report = runner.invoke(app, ["report", str(created_run_dir)])
    assert res_report.exit_code == 0, res_report.output
    assert "24/24 (100.0%)" in res_report.output
    assert "Budget Stop Rate (budget oranı): 0/24 (0.0%)" in res_report.output
