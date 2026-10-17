"""Unit tests for deterministic trace profiler (M7) and real trace verification (M8)."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

from boundary.models import Step, ToolCall, ToolResult, Trace, Usage
from boundary.profile import (
    compute_carry_cost,
    compute_high_frequency_tokens,
    compute_reuse_ratio,
    extract_signal_tokens,
    find_independent_step_groups,
    profile_traces,
)
from boundary.tasks import load_tasks
from boundary.traces import read_traces

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_SAMPLE_PATH = REPO_ROOT / "tests" / "data" / "baseline_sample.jsonl"
TASKS_DIR = REPO_ROOT / "tasks" / "repo_qa"


def _make_tool_result(call_id: str, content: str, tokens_est: int) -> ToolResult:
    return ToolResult(
        call_id=call_id,
        content=content,
        size_chars=len(content),
        tokens_est=tokens_est,
    )


def _make_step(
    index: int,
    *,
    tool_calls: list[ToolCall] | None = None,
    tool_results: list[ToolResult] | None = None,
    assistant_content: str = "",
    prompt_tokens: int = 1000,
    completion_tokens: int = 100,
) -> Step:
    return Step(
        index=index,
        assistant_content=assistant_content,
        tool_calls=tool_calls or [],
        tool_results=tool_results or [],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            source="api",
        ),
        latency_ms=50,
    )


def _make_trace(
    trace_id: str,
    task_id: str,
    steps: list[Step],
    *,
    final_answer: str = "done",
) -> Trace:
    total_prompt = sum(s.usage.prompt_tokens for s in steps)
    total_comp = sum(s.usage.completion_tokens for s in steps)
    return Trace(
        trace_id=trace_id,
        task_id=task_id,
        trial=0,
        model="fake-model",
        config_hash="abc123def456",
        steps=steps,
        final_answer=final_answer,
        passed=True,
        stop_reason="final",
        total_usage=Usage(
            prompt_tokens=total_prompt,
            completion_tokens=total_comp,
            source="api",
        ),
    )


def test_signal_s1_compute_carry_cost() -> None:
    """Verify S1 formula: remaining = max(0, N - 1 - s), carry_cost = tokens_est * remaining."""
    # Step 0 in a 4-step trace -> carried into steps 1, 2, 3 (remaining = 3)
    rem0, cost0 = compute_carry_cost(tokens_est=2500, step_index=0, num_calls=4)
    assert rem0 == 3
    assert cost0 == 7500

    # Step 2 in a 4-step trace -> carried into step 3 (remaining = 1)
    rem2, cost2 = compute_carry_cost(tokens_est=1200, step_index=2, num_calls=4)
    assert rem2 == 1
    assert cost2 == 1200

    # Last step (step 3 in a 4-step trace) -> remaining = 0
    rem3, cost3 = compute_carry_cost(tokens_est=5000, step_index=3, num_calls=4)
    assert rem3 == 0
    assert cost3 == 0


def test_signal_s1_trace_carry_total_and_share() -> None:
    """Verify TraceProfile carry_total_tokens and carry_share on a synthetic trace."""
    steps = [
        _make_step(
            0,
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
            tool_results=[_make_tool_result("c1", "x" * 8000, 2000)],
            prompt_tokens=1000,
        ),
        _make_step(
            1,
            tool_calls=[ToolCall(id="c2", name="read_file", arguments={"path": "b.py"})],
            tool_results=[_make_tool_result("c2", "y" * 4000, 1000)],
            prompt_tokens=3000,
        ),
        _make_step(
            2,
            assistant_content="Final answer ready.",
            prompt_tokens=6000,
        ),
    ]
    trace = _make_trace("tr-s1", "task-s1", steps)
    summary = profile_traces([trace])
    assert len(summary.traces) == 1
    tp = summary.traces[0]

    # Step 0: 2000 * (3 - 1 - 0) = 4000
    # Step 1: 1000 * (3 - 1 - 1) = 1000
    # Total carry = 5000; total prompt = 1000 + 3000 + 6000 = 10000 -> share = 0.5
    assert tp.num_calls == 3
    assert tp.carry_total_tokens == 5000
    assert tp.carry_share == pytest.approx(0.5)
    assert tp.usage_source == "api"
    assert [r.carry_cost_tokens for r in tp.results] == [4000, 1000]


def test_signal_s2_extract_tokens_and_filters() -> None:
    """Verify S2 regex token extraction, Python keyword exclusion, and >30% corpus filter."""
    text = "def return class import alpha_token_99 short xy z_999"
    tokens = extract_signal_tokens(text)
    # Python keywords ("return", "class", "import") and <4 char tokens ("def", "xy") excluded
    assert "return" not in tokens
    assert "class" not in tokens
    assert "import" not in tokens
    assert "alpha_token_99" in tokens
    assert "short" in tokens
    assert "z_999" in tokens

    # Corpus high-frequency (>30% across tool results)
    steps = [
        _make_step(
            i,
            tool_calls=[ToolCall(id=f"c{i}", name="read_file", arguments={"path": f"f{i}.py"})],
            tool_results=[
                _make_tool_result(
                    f"c{i}",
                    f"common_header_tok unique_tok_{i}",
                    20,
                )
            ],
        )
        for i in range(4)
    ]
    trace = _make_trace("tr-freq", "task-freq", steps)
    high_freq = compute_high_frequency_tokens([trace])
    assert "common_header_tok" in high_freq
    assert "unique_tok_0" not in high_freq


def test_signal_s2_reuse_ratio_and_extract_subagent_opportunity() -> None:
    """Verify S2 line reuse ratio and extract_subagent opportunity generation."""
    lines = [
        "line_alpha_unused_01 = 100",
        "line_beta_unused_02 = 200",
        "line_gamma_unused_03 = 300",
        "line_delta_reused_target = 9999",
    ]
    content = "\n".join(lines)
    # Only 1 of 4 non-empty lines shares a token with downstream_texts -> reuse_ratio = 0.25
    ratio = compute_reuse_ratio(
        result_content=content,
        downstream_texts=["The value comes from line_delta_reused_target in the module."],
    )
    assert ratio == pytest.approx(0.25)

    # Now build a trace with a large tool result (2500 tokens) where only 1 of 20 lines is reused
    twenty_lines = [f"noise_entry_unused_{i:03d} = {i}" for i in range(19)] + [
        "critical_needle_value_xyz = 42"
    ]
    steps = [
        _make_step(
            0,
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "big.py"})],
            tool_results=[_make_tool_result("c1", "\n".join(twenty_lines), 2500)],
            prompt_tokens=3000,
        ),
        _make_step(
            1,
            assistant_content="Found critical_needle_value_xyz.",
            prompt_tokens=4000,
        ),
    ]
    trace = _make_trace(
        "tr-s2",
        "task-s2",
        steps,
        final_answer="critical_needle_value_xyz",
    )
    summary = profile_traces([trace], min_tokens=2000, reuse_threshold=0.1)
    rp = summary.traces[0].results[0]
    assert rp.reuse_ratio == pytest.approx(1 / 20)

    extract_opps = [o for o in summary.opportunities if o.kind == "extract_subagent"]
    assert len(extract_opps) == 1
    opp = extract_opps[0]
    assert opp.task_id == "task-s2"
    assert opp.est_upper_bound_savings_tokens == 2500
    assert opp.confidence == "low"


def test_signal_s3_independent_vs_dependent_steps() -> None:
    """Verify S3 detects contiguous independent steps and splits on data dependency."""
    # Steps 0 and 1 read files mentioned in task_prompt (independent -> group [0, 1])
    # Step 2 reads a file discovered ONLY in step 1's output (dependent on step 1 -> breaks group)
    task_prompt = "Compare config_alpha.py and config_beta.py."
    steps = [
        _make_step(
            0,
            tool_calls=[
                ToolCall(id="c0", name="read_file", arguments={"path": "config_alpha.py"})
            ],
            tool_results=[_make_tool_result("c0", "ALPHA_PORT = 8080", 50)],
        ),
        _make_step(
            1,
            tool_calls=[
                ToolCall(id="c1", name="read_file", arguments={"path": "config_beta.py"})
            ],
            tool_results=[
                _make_tool_result(
                    "c1",
                    "see secret_downstream_module.py for override",
                    50,
                )
            ],
        ),
        _make_step(
            2,
            tool_calls=[
                ToolCall(
                    id="c2",
                    name="read_file",
                    arguments={"path": "secret_downstream_module.py"},
                )
            ],
            tool_results=[_make_tool_result("c2", "OVERRIDE = 9090", 50)],
        ),
        _make_step(3, assistant_content="Done."),
    ]
    trace = _make_trace("tr-s3", "task-s3", steps)
    groups = find_independent_step_groups(trace, task_prompt=task_prompt)
    assert groups == [[0, 1]]

    summary = profile_traces([trace], task_prompts={"task-s3": task_prompt})
    par_opps = [o for o in summary.opportunities if o.kind == "parallelize"]
    assert len(par_opps) == 1
    assert par_opps[0].steps_saved == 1
    assert par_opps[0].evidence["step_indices"] == [0, 1]
    assert par_opps[0].confidence == "low"


def test_baseline_sample_security_and_real_trace_profiling() -> None:
    """Verify M8 sample has no secrets, is deterministic, and separates needle carry_share."""
    assert BASELINE_SAMPLE_PATH.exists()
    raw_content = BASELINE_SAMPLE_PATH.read_text(encoding="utf-8")
    secret_regex = re.compile(r"(?i)(nvapi|bearer\s+[a-z0-9._\-]+|sk-[a-z0-9]{6,}|api_key)")
    assert secret_regex.search(raw_content) is None

    traces = read_traces(BASELINE_SAMPLE_PATH)
    assert len(traces) == 22

    task_prompts = {t.id: t.prompt for t in load_tasks(TASKS_DIR)}
    profile_run_1 = profile_traces(traces, task_prompts=task_prompts)
    profile_run_2 = profile_traces(traces, task_prompts=task_prompts)

    # 100% deterministic JSON output across runs
    assert profile_run_1.model_dump_json(indent=2) == profile_run_2.model_dump_json(indent=2)

    # Group carry_share by task_id
    shares_by_task: dict[str, list[float]] = defaultdict(list)
    for tp in profile_run_1.traces:
        shares_by_task[tp.task_id].append(tp.carry_share)

    mean_shares = {
        tid: sum(vals) / len(vals) for tid, vals in shares_by_task.items()
    }
    needle_tasks = {"ledger-anomaly-needle", "routing-shard-needle"}
    easy_tasks = set(mean_shares.keys()) - needle_tasks
    assert needle_tasks.issubset(mean_shares.keys())
    assert len(easy_tasks) == 6

    min_needle_share = min(mean_shares[tid] for tid in needle_tasks)
    max_easy_share = max(mean_shares[tid] for tid in easy_tasks)
    assert min_needle_share > 0.60
    assert max_easy_share < 0.35
    assert min_needle_share > max_easy_share

    # Every opportunity must have confidence="low"
    assert len(profile_run_1.opportunities) > 0
    for opp in profile_run_1.opportunities:
        assert opp.confidence == "low"
