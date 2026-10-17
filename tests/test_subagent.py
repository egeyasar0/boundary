"""Unit tests for isolated SubagentRunner and consolidated budget accounting (M9 — R11, R12)."""

from __future__ import annotations

from pathlib import Path

from boundary.agent import run_agent
from boundary.llm import FakeLLM, FakeTurn
from boundary.models import Checker, Task, ToolCall, Usage
from boundary.subagent import (
    DEFAULT_MAX_SUMMARY_TOKENS,
    SUBAGENT_SYSTEM_PROMPT,
    SubagentRunner,
)
from boundary.tools import ToolSandbox


def _make_sample_sandbox(tmp_path: Path) -> ToolSandbox:
    big_lines = [f"NOISE_RECORD_{i:04d} = 'nominal_padding_value_{i:04d}'" for i in range(600)]
    big_lines.append("TARGET_NEEDLE_KEY = 'NEEDLE-valkyrie-9901'")
    (tmp_path / "giant_module.py").write_text("\n".join(big_lines) + "\n", encoding="utf-8")
    return ToolSandbox(tmp_path)


def test_subagent_isolated_context_and_summary_bound(tmp_path: Path) -> None:
    """Verify R11: subagent never sees parent messages and caps returned summary <= 300 tokens."""
    sandbox = _make_sample_sandbox(tmp_path)
    overlong_summary = "NEEDLE-valkyrie-9901 " + ("extra_detail " * 200)
    fake_llm = FakeLLM(
        [
            FakeTurn(
                tool_calls=[
                    ToolCall(
                        id="sub_c1",
                        name="grep",
                        arguments={"pattern": "TARGET_NEEDLE_KEY", "path": "giant_module.py"},
                    )
                ],
                usage=Usage(prompt_tokens=120, completion_tokens=25, source="api"),
            ),
            FakeTurn(
                assistant_content=overlong_summary,
                usage=Usage(prompt_tokens=220, completion_tokens=80, source="api"),
            ),
        ]
    )

    runner = SubagentRunner(llm=fake_llm, sandbox=sandbox)
    res = runner.run(
        call_id="parent_call_1",
        query="Find TARGET_NEEDLE_KEY",
        path="giant_module.py",
    )

    # First call to subagent LLM has ONLY system + user prompt (no parent history)
    first_call_msgs = fake_llm.calls[0]["messages"]
    assert len(first_call_msgs) == 2
    assert first_call_msgs[0]["content"] == SUBAGENT_SYSTEM_PROMPT
    assert "giant_module.py" in first_call_msgs[1]["content"]

    assert res.stop_reason == "final"
    assert res.steps_taken == 2
    assert res.usage.prompt_tokens == 340
    assert res.usage.completion_tokens == 105
    assert res.tool_result.tokens_est <= DEFAULT_MAX_SUMMARY_TOKENS
    assert "NEEDLE-valkyrie-9901" in res.tool_result.content


def test_subagent_consolidated_budget_and_context_reduction_in_parent(tmp_path: Path) -> None:
    """Verify R11 & R12: subagent tokens deduct from parent budget while slashing parent context."""
    sandbox = _make_sample_sandbox(tmp_path)
    task = Task(
        id="needle-task",
        prompt="What is TARGET_NEEDLE_KEY in giant_module.py? SecretParentMarker_XYZ",
        checker=Checker(type="contains_all", expected=["NEEDLE-valkyrie-9901"]),
        max_steps=6,
    )

    # Turn 0 (parent step 0): parent calls read_file on giant_module.py (intercepted by subagent)
    # Turn 1 (subagent step 0): subagent calls grep on giant_module.py
    # Turn 2 (subagent step 1): subagent returns concise summary
    # Turn 3 (parent step 1): parent receives concise summary and answers
    fake_llm = FakeLLM(
        [
            FakeTurn(
                tool_calls=[
                    ToolCall(
                        id="p_call_1",
                        name="read_file",
                        arguments={"path": "giant_module.py"},
                    )
                ],
                usage=Usage(prompt_tokens=200, completion_tokens=30, source="api"),
            ),
            FakeTurn(
                tool_calls=[
                    ToolCall(
                        id="s_call_1",
                        name="grep",
                        arguments={"pattern": "TARGET_NEEDLE_KEY", "path": "giant_module.py"},
                    )
                ],
                usage=Usage(prompt_tokens=150, completion_tokens=25, source="api"),
            ),
            FakeTurn(
                assistant_content="TARGET_NEEDLE_KEY = NEEDLE-valkyrie-9901",
                usage=Usage(prompt_tokens=210, completion_tokens=15, source="api"),
            ),
            FakeTurn(
                assistant_content="The value is NEEDLE-valkyrie-9901.",
                usage=Usage(prompt_tokens=260, completion_tokens=20, source="api"),
            ),
        ]
    )

    sub_runner = SubagentRunner(llm=fake_llm, sandbox=sandbox)
    trace = run_agent(
        task=task,
        llm=fake_llm,
        sandbox=sandbox,
        token_budget=10_000,
        subagent_runner=sub_runner,
        should_delegate=lambda tc: tc.name == "read_file",
        agent_variant="subagent_extracted",
         answer_checker=lambda ans: ans is not None and "NEEDLE-valkyrie-9901" in ans,
    )

    assert trace.passed is True
    assert trace.stop_reason == "final"
    assert trace.agent_variant == "subagent_extracted"

    # Consolidated token usage = parent (200+30 + 260+20) + subagent (150+25 + 210+15) = 910
    assert trace.total_usage.prompt_tokens == 200 + 150 + 210 + 260
    assert trace.total_usage.completion_tokens == 30 + 25 + 15 + 20
    assert trace.total_usage.total == 910

    # Compare tool result size entering parent context vs direct read_file (~5000 tokens)
    direct_result = sandbox.execute(
        ToolCall(id="direct", name="read_file", arguments={"path": "giant_module.py"})
    )
    delegated_result = trace.steps[0].tool_results[0]
    assert delegated_result.tokens_est < direct_result.tokens_est * 0.10


def test_subagent_tokens_trigger_parent_global_budget_stop(tmp_path: Path) -> None:
    """Verify R12: subagent token spend pushing parent over budget stops parent on budget."""
    sandbox = _make_sample_sandbox(tmp_path)
    task = Task(
        id="tight-budget-task",
        prompt="Find TARGET_NEEDLE_KEY.",
        checker=Checker(type="contains_all", expected=["NEEDLE-valkyrie-9901"]),
        max_steps=5,
    )

    fake_llm = FakeLLM(
        [
            # Parent step 0 spends 300 tokens
            FakeTurn(
                tool_calls=[
                    ToolCall(id="p1", name="read_file", arguments={"path": "giant_module.py"})
                ],
                usage=Usage(prompt_tokens=250, completion_tokens=50, source="api"),
            ),
            # Subagent spends 800 tokens -> total becomes 1100 >= parent token_budget (1000)
            FakeTurn(
                assistant_content="NEEDLE-valkyrie-9901",
                usage=Usage(prompt_tokens=750, completion_tokens=50, source="api"),
            ),
        ]
    )
    sub_runner = SubagentRunner(llm=fake_llm, sandbox=sandbox)
    trace = run_agent(
        task=task,
        llm=fake_llm,
        sandbox=sandbox,
        token_budget=1000,
        subagent_runner=sub_runner,
        should_delegate=lambda tc: tc.name == "read_file",
    )

    assert trace.stop_reason == "budget"
    assert trace.total_usage.total == 1100
    assert len(trace.steps) == 1


def test_subagent_step_limit_and_error_handling_returns_safe_tool_result(tmp_path: Path) -> None:
    """Verify subagent max_steps=3 and LLM timeout/error return safe sanitized ToolResults."""
    sandbox = _make_sample_sandbox(tmp_path)

    # Case 1: Subagent hits LLM timeout/error with secret -> returns sanitized ToolResult
    err_llm = FakeLLM(
        [
            RuntimeError("request timeout with header Authorization: Bearer sk-secret-sub-999999")
        ]
    )
    err_runner = SubagentRunner(llm=err_llm, sandbox=sandbox)
    err_res = err_runner.run(call_id="c_err", query="Find needle", path="giant_module.py")
    assert err_res.stop_reason == "error"
    assert "Tool error (subagent):" in err_res.tool_result.content
    assert "sk-secret-sub-999999" not in err_res.tool_result.content
    assert "[REDACTED]" in err_res.tool_result.content

    # Case 2: Subagent exceeds subagent_max_steps=3 without final answer
    loop_llm = FakeLLM(
        [
            FakeTurn(
                tool_calls=[ToolCall(id=f"sc_{i}", name="list_files", arguments={"path": "."})],
                usage=Usage(prompt_tokens=100, completion_tokens=10, source="api"),
            )
            for i in range(4)
        ]
    )
    loop_runner = SubagentRunner(llm=loop_llm, sandbox=sandbox, subagent_max_steps=3)
    loop_res = loop_runner.run(call_id="c_loop", query="Find needle")
    assert loop_res.stop_reason == "max_steps"
    assert loop_res.steps_taken == 3
    assert loop_res.usage.total == 330
    assert "Tool error (subagent): subagent reached max_steps (3)" in loop_res.tool_result.content
