"""End-to-end tests for runner, report, and Typer CLI using FakeLLM (M5)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import boundary.cli as cli_mod
from boundary.cli import app
from boundary.llm import FakeLLM, FakeTurn, LLMClient, LLMResponse
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
    "REC-solstice-4409 and DIGEST-krypton-9183",
    "VOUCHER-titanium-6671 and PM-aurora-9317",
    "SEAL-vanguard-5823 and RECOVERY-onyx-4189",
    "CANARY-orion-6311 and OVERRIDE-cobalt-5149",
    "CORRIDOR-nebula-2743 and CLEARANCE-helix-7529",
    "VEC-polaris-3907 and ACK-zenith-8461",
    "Answer: 9",
    "Answer: 11",
    "Answer: 13",
    "Answer: 125",
    "hook-valkyrie-7741 and nonce-celeste-6029",
    "cipher-alabaster-8513 and salt-obsidia-4902",
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

    captured_timeout: list[float] = []

    def _fake_factory(
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> LLMClient:
        captured_timeout.append(timeout)
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
            "--timeout",
            "45",
        ],
    )
    assert res_run.exit_code == 0, res_run.output
    assert captured_timeout == [45.0]
    assert "60 traces written" in res_run.output

    run_subdirs = [p for p in out_dir.iterdir() if p.is_dir()]
    assert len(run_subdirs) == 1
    created_run_dir = run_subdirs[0]

    res_report = runner.invoke(app, ["report", str(created_run_dir)])
    assert res_report.exit_code == 0, res_report.output
    assert "60/60 (100.0%)" in res_report.output
    assert "Budget Stop Rate (budget oranı): 0/60 (0.0%)" in res_report.output


def test_runner_continues_after_timeout(tmp_path: Path) -> None:
    tasks = load_tasks(TASKS_DIR)[:1]
    fake_llm = FakeLLM(
        [
            TimeoutError("simulated timeout on trial 1"),
            FakeTurn(
                assistant_content="7919 decorrelated-halcyon-v4",
                usage=Usage(prompt_tokens=80, completion_tokens=20, source="api"),
            ),
        ]
    )
    _, traces = run_suite(
        tasks=tasks,
        llm=fake_llm,
        root_dir=FIXTURE_DIR,
        out_dir=tmp_path / "runs",
        trials=2,
        token_budget=5_000,
    )
    assert len(traces) == 2
    assert traces[0].stop_reason == "error"
    assert traces[0].error == "request timeout"
    assert traces[1].stop_reason == "final"
    assert traces[1].passed is True


def test_runner_creates_trace_file_at_start_and_preserves_completed_on_interrupt(
    tmp_path: Path,
) -> None:
    # 1. Even with empty task list, run_dir and traces.jsonl are created upfront
    empty_dir, empty_traces = run_suite(
        tasks=[],
        llm=FakeLLM([]),
        root_dir=FIXTURE_DIR,
        out_dir=tmp_path / "runs",
        run_id="run_empty_init",
    )
    assert empty_traces == []
    assert (empty_dir / "traces.jsonl").exists()

    # 2. If KeyboardInterrupt occurs during trial 2, trial 1 trace is already flushed to disk
    tasks = load_tasks(TASKS_DIR)[:1]

    class _InterruptOnSecondCallLLM:
        model = "interrupt-llm"

        def __init__(self) -> None:
            self.call_count = 0

        def complete(
            self,
            messages: list[dict[str, object]],
            tools: list[dict[str, object]] | None = None,
            seed: int | None = None,
            max_tokens: int | None = None,
        ) -> LLMResponse:
            self.call_count += 1
            if self.call_count == 1:
                return LLMResponse(
                    assistant_content="7919 decorrelated-halcyon-v4",
                    tool_calls=[],
                    usage=Usage(prompt_tokens=60, completion_tokens=15, source="api"),
                )
            raise KeyboardInterrupt("User pressed Ctrl+C during trial 2")

    interrupted_run_dir = tmp_path / "runs" / "run_interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_suite(
            tasks=tasks,
            llm=_InterruptOnSecondCallLLM(),
            root_dir=FIXTURE_DIR,
            out_dir=tmp_path / "runs",
            trials=2,
            run_id="run_interrupted",
        )

    saved = read_traces(interrupted_run_dir)
    assert len(saved) == 1
    assert saved[0].trial == 1
    assert saved[0].passed is True


def test_report_missing_or_empty_traces_and_last_error_display(tmp_path: Path) -> None:
    runner = CliRunner()

    # 1. Missing directory / traces.jsonl
    res_missing = runner.invoke(app, ["report", str(tmp_path / "does_not_exist")])
    assert res_missing.exit_code == 1
    assert "No traces found" in res_missing.output
    assert "Traceback" not in res_missing.output

    # 2. Empty traces.jsonl
    empty_run = tmp_path / "empty_run"
    empty_run.mkdir()
    (empty_run / "traces.jsonl").write_text("", encoding="utf-8")
    res_empty = runner.invoke(app, ["report", str(empty_run)])
    assert res_empty.exit_code == 1
    assert "No traces found" in res_empty.output
    assert "is empty" in res_empty.output
    assert "Traceback" not in res_empty.output

    # 3. Error trace with >200 chars error message displays first 200 chars under "Son hata"
    long_error = "ERR_PREFIX_" + ("Z" * 240)
    tasks = load_tasks(TASKS_DIR)[:1]
    err_run_dir, _ = run_suite(
        tasks=tasks,
        llm=FakeLLM([RuntimeError(long_error)]),
        root_dir=FIXTURE_DIR,
        out_dir=tmp_path / "runs",
        trials=1,
        run_id="run_with_err",
    )

    traces = read_traces(err_run_dir)
    summary = summarize_run(traces, run_id="run_with_err")
    assert summary.last_error is not None
    assert len(summary.last_error) == 200
    assert summary.tasks[0].last_error == long_error[:200]

    res_err_report = runner.invoke(app, ["report", str(err_run_dir)])
    assert res_err_report.exit_code == 0, res_err_report.output
    assert "Son hata" in res_err_report.output
    assert f"Son hata: {long_error[:200]}" in res_err_report.output


def test_cli_check_command_ok_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()

    # Case 1: Both chat and tool calling succeed
    fake_ok = FakeLLM(
        [
            FakeTurn(
                assistant_content="OK",
                usage=Usage(prompt_tokens=10, completion_tokens=2, source="api"),
            ),
            FakeTurn(
                tool_calls=[
                    ToolCall(id="call_ping", name="ping_tool", arguments={"echo": "ok"})
                ],
                usage=Usage(prompt_tokens=25, completion_tokens=12, source="api"),
            ),
        ]
    )
    monkeypatch.setattr(cli_mod, "create_llm_client", lambda **kw: fake_ok)

    res_ok = runner.invoke(
        app,
        ["check", "--model", "test-model", "--base-url", "http://127.0.0.1:9999/v1"],
    )
    assert res_ok.exit_code == 0, res_ok.output
    assert "OK (" in res_ok.output
    assert "ms)" in res_ok.output
    assert "tool calling: OK" in res_ok.output
    assert fake_ok.calls[0]["max_tokens"] == 16
    assert fake_ok.calls[1]["max_tokens"] == 64

    # Case 2: Chat request fails -> sanitized message, no leaked API key
    fake_chat_err = FakeLLM(
        [
            RuntimeError(
                "401 Unauthorized: Bearer sk-secret-check-key-999999"
            )
        ]
    )
    monkeypatch.setattr(cli_mod, "create_llm_client", lambda **kw: fake_chat_err)

    res_chat_fail = runner.invoke(app, ["check", "--model", "test-model"])
    assert res_chat_fail.exit_code == 1
    assert "FAIL:" in res_chat_fail.output
    assert "sk-secret-check-key-999999" not in res_chat_fail.output
    assert "[REDACTED]" in res_chat_fail.output

    # Case 3: Chat succeeds, but model does not emit a tool call
    fake_tool_fail = FakeLLM(
        [
            FakeTurn(
                assistant_content="OK",
                usage=Usage(prompt_tokens=10, completion_tokens=2, source="api"),
            ),
            FakeTurn(
                assistant_content="I cannot call tools.",
                tool_calls=[],
                usage=Usage(prompt_tokens=25, completion_tokens=8, source="api"),
            ),
        ]
    )
    monkeypatch.setattr(cli_mod, "create_llm_client", lambda **kw: fake_tool_fail)

    res_tool_fail = runner.invoke(app, ["check", "--model", "test-model"])
    assert res_tool_fail.exit_code == 1
    assert "OK (" in res_tool_fail.output
    assert "tool calling: FAIL" in res_tool_fail.output




