"""Unit tests for LLMClient, FakeLLM, and single-agent baseline loop (M3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from boundary.agent import SYSTEM_PROMPT, run_agent
from boundary.llm import (
    FakeLLM,
    FakeTurn,
    LLMClient,
    LLMError,
    OpenAICompatClient,
    parse_tool_arguments,
)
from boundary.models import Checker, Task, ToolCall, Usage
from boundary.tools import ToolSandbox


@pytest.fixture
def sandbox(tmp_path: Path) -> ToolSandbox:
    (tmp_path / "client.py").write_text("RETRY_LIMIT = 5\n", encoding="utf-8")
    (tmp_path / "config.py").write_text("TIMEOUT_SEC = 30\n", encoding="utf-8")
    return ToolSandbox(tmp_path)


@pytest.fixture
def sample_task() -> Task:
    return Task(
        id="find-retry",
        prompt="What is RETRY_LIMIT and TIMEOUT_SEC?",
        checker=Checker(type="contains_all", expected=["5", "30"]),
        max_steps=4,
    )


def test_multi_tool_calls_in_single_step_and_final_answer(
    sandbox: ToolSandbox, sample_task: Task
) -> None:
    fake_llm = FakeLLM(
        [
            FakeTurn(
                assistant_content="Checking both files.",
                tool_calls=[
                    ToolCall(id="c1", name="read_file", arguments={"path": "client.py"}),
                    ToolCall(id="c2", name="read_file", arguments={"path": "config.py"}),
                ],
                usage=Usage(prompt_tokens=100, completion_tokens=40, source="api"),
            ),
            FakeTurn(
                assistant_content="RETRY_LIMIT is 5 and TIMEOUT_SEC is 30.",
                tool_calls=[],
                usage=Usage(prompt_tokens=180, completion_tokens=20, source="api"),
            ),
        ]
    )
    assert isinstance(fake_llm, LLMClient)

    trace = run_agent(
        task=sample_task,
        llm=fake_llm,
        sandbox=sandbox,
        token_budget=5_000,
        answer_checker=lambda ans: ans is not None and "5" in ans and "30" in ans,
    )

    assert trace.stop_reason == "final"
    assert trace.passed is True
    assert trace.error is None
    assert len(trace.steps) == 2
    assert len(trace.steps[0].tool_calls) == 2
    assert len(trace.steps[0].tool_results) == 2
    assert "RETRY_LIMIT = 5" in trace.steps[0].tool_results[0].content
    assert "TIMEOUT_SEC = 30" in trace.steps[0].tool_results[1].content
    assert trace.total_usage.prompt_tokens == 280
    assert trace.total_usage.completion_tokens == 60
    assert trace.total_usage.total == 340
    assert trace.total_usage.source == "api"
    assert fake_llm.calls[0]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}


def test_broken_tool_and_invalid_json_recover_without_error_stop(
    sandbox: ToolSandbox, sample_task: Task
) -> None:
    malformed_args = parse_tool_arguments("{broken_json:")
    assert "__json_error__" in malformed_args

    fake_llm = FakeLLM(
        [
            FakeTurn(
                assistant_content=None,
                tool_calls=[
                    ToolCall(id="c_bad_json", name="read_file", arguments=malformed_args),
                    ToolCall(id="c_unknown", name="run_bash", arguments={"cmd": "ls"}),
                ],
                usage=Usage(prompt_tokens=80, completion_tokens=30, source="api"),
            ),
            FakeTurn(
                assistant_content="Recovered answer: 5 and 30",
                tool_calls=[],
                usage=Usage(prompt_tokens=150, completion_tokens=15, source="api"),
            ),
        ]
    )

    trace = run_agent(
        task=sample_task,
        llm=fake_llm,
        sandbox=sandbox,
        token_budget=5_000,
        answer_checker=lambda ans: ans is not None and "5" in ans,
    )

    assert trace.stop_reason == "final"
    assert trace.error is None
    assert len(trace.steps) == 2
    res_json_err = trace.steps[0].tool_results[0].content
    res_unknown_err = trace.steps[0].tool_results[1].content
    assert res_json_err.startswith("Error: invalid JSON arguments for tool 'read_file'")
    assert res_unknown_err == "Error: unknown tool 'run_bash'."


def test_token_budget_checked_before_llm_call_and_in_flight_preserved(
    sandbox: ToolSandbox, sample_task: Task
) -> None:
    """Verify Rule #3: budget checked BEFORE each LLM call; in-flight call is never cut."""
    fake_llm = FakeLLM(
        [
            # Step 0 starts with 0 < 200 budget; consumes 250 tokens (>= 200) and calls a tool.
            FakeTurn(
                assistant_content=None,
                tool_calls=[ToolCall(id="c1", name="list_files", arguments={"path": "."})],
                usage=Usage(prompt_tokens=200, completion_tokens=50, source="api"),
            ),
            # Step 1 should NEVER be called because 250 >= 200 before the 2nd LLM call.
            FakeTurn(
                assistant_content="Should not reach here",
                tool_calls=[],
                usage=Usage(prompt_tokens=100, completion_tokens=20, source="api"),
            ),
        ]
    )

    trace = run_agent(
        task=sample_task,
        llm=fake_llm,
        sandbox=sandbox,
        token_budget=200,
    )

    assert trace.stop_reason == "budget"
    assert len(fake_llm.calls) == 1
    assert len(trace.steps) == 1
    # Actual usage (250) from the completed step is recorded in Trace.total_usage
    assert trace.total_usage.total == 250
    assert trace.final_answer is None


def test_max_steps_stop_reason(sandbox: ToolSandbox, sample_task: Task) -> None:
    task_two_steps = sample_task.model_copy(update={"max_steps": 2})
    fake_llm = FakeLLM(
        [
            FakeTurn(
                tool_calls=[ToolCall(id="c1", name="list_files", arguments={"path": "."})],
                usage=Usage(prompt_tokens=50, completion_tokens=10, source="api"),
            ),
            FakeTurn(
                tool_calls=[ToolCall(id="c2", name="read_file", arguments={"path": "client.py"})],
                usage=Usage(prompt_tokens=70, completion_tokens=10, source="api"),
            ),
            FakeTurn(
                assistant_content="Too late",
                usage=Usage(prompt_tokens=90, completion_tokens=10, source="api"),
            ),
        ]
    )

    trace = run_agent(task=task_two_steps, llm=fake_llm, sandbox=sandbox, token_budget=10_000)
    assert trace.stop_reason == "max_steps"
    assert len(trace.steps) == 2
    assert len(fake_llm.calls) == 2


def test_estimated_usage_when_api_usage_missing(
    sandbox: ToolSandbox, sample_task: Task
) -> None:
    fake_llm = FakeLLM(
        [
            FakeTurn(
                assistant_content=" calling tool ",
                tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "client.py"})],
                usage=None,  # Triggers chars // 4 estimation
            ),
            FakeTurn(
                assistant_content="Final answer: 5 and 30",
                tool_calls=[],
                usage=None,
            ),
        ]
    )

    trace = run_agent(task=sample_task, llm=fake_llm, sandbox=sandbox)
    assert trace.stop_reason == "final"
    assert trace.total_usage.source == "estimated"
    assert trace.steps[0].usage.source == "estimated"
    assert trace.steps[0].usage.prompt_tokens > 0
    assert trace.steps[0].usage.completion_tokens > 0
    assert trace.total_usage.total == sum(s.usage.total for s in trace.steps)


def test_llm_exception_sets_error_stop_reason_and_sanitizes_secret(
    sandbox: ToolSandbox, sample_task: Task
) -> None:
    fake_llm = FakeLLM(
        [
            RuntimeError(
                "Connection failed with Authorization: Bearer sk-supersecretkey123456 "
                "and headers={'Authorization': 'Bearer sk-supersecretkey123456'} "
                + ("E" * 600)
            )
        ]
    )

    trace = run_agent(task=sample_task, llm=fake_llm, sandbox=sandbox)
    assert trace.stop_reason == "error"
    assert trace.passed is False
    assert trace.error is not None
    assert len(trace.error) <= 500
    assert "sk-supersecretkey123456" not in trace.error
    assert "[REDACTED]" in trace.error


def test_openai_compat_client_sanitizes_repr_and_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OpenAICompatClient(
        api_key="sk-test-secret-value-999999",
        base_url="http://127.0.0.1:9999/v1",
        model="local-model",
    )
    assert "sk-test-secret-value-999999" not in repr(client)
    assert "[REDACTED]" in repr(client)

    def _boom(**kwargs: Any) -> Any:
        raise RuntimeError(
            "Upstream error using key sk-test-secret-value-999999 and "
            "headers={'Authorization': 'Bearer sk-test-secret-value-999999'}"
        )

    monkeypatch.setattr(client._client.chat.completions, "create", _boom)

    with pytest.raises(LLMError) as exc_info:
        client.complete(messages=[{"role": "user", "content": "hi"}])

    err_str = str(exc_info.value)
    assert "sk-test-secret-value-999999" not in err_str
    assert "[REDACTED]" in err_str


def test_request_timeout_records_error_and_trace_message(
    sandbox: ToolSandbox, sample_task: Task, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = OpenAICompatClient(
        api_key="sk-test-secret",
        base_url="http://127.0.0.1:9999/v1",
        model="local-model",
    )
    assert client.timeout == 120.0

    custom_client = OpenAICompatClient(
        api_key="sk-test-secret",
        base_url="http://127.0.0.1:9999/v1",
        model="local-model",
        timeout=15.0,
    )
    assert custom_client.timeout == 15.0

    def _timeout_call(**kwargs: Any) -> Any:
        assert kwargs.get("timeout") == 15.0
        raise TimeoutError("Read timed out after 15.0s")

    monkeypatch.setattr(custom_client._client.chat.completions, "create", _timeout_call)

    trace = run_agent(task=sample_task, llm=custom_client, sandbox=sandbox)
    assert trace.stop_reason == "error"
    assert trace.error == "request timeout"
    assert trace.passed is False

