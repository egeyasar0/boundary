"""Task x trial benchmark runner and trace recorder (M5)."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from boundary.agent import run_agent
from boundary.llm import LLMClient
from boundary.models import Checker, Task, Trace
from boundary.tasks import evaluate_checker
from boundary.tools import ToolSandbox
from boundary.traces import init_trace_file, write_trace


def _make_answer_checker(checker: Checker) -> Callable[[str | None], bool]:
    """Create a bound answer checker callback for a single task."""

    def _check(ans: str | None) -> bool:
        return evaluate_checker(checker, ans)

    return _check


def generate_run_id() -> str:
    """Generate a timestamped unique run directory identifier."""
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"run_{stamp}_{suffix}"


def run_suite(
    *,
    tasks: Sequence[Task],
    llm: LLMClient,
    root_dir: Path | str,
    out_dir: Path | str,
    trials: int = 3,
    token_budget: int = 40_000,
    seed: int | None = None,
    parameters: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> tuple[Path, list[Trace]]:
    """Execute all tasks across trials, writing JSONL traces to out_dir/<run_id>/traces.jsonl."""
    if trials < 1:
        raise ValueError("trials must be >= 1.")
    if token_budget < 1:
        raise ValueError("token_budget must be >= 1.")

    resolved_run_id = run_id or generate_run_id()
    run_dir = Path(out_dir) / resolved_run_id
    init_trace_file(run_dir)

    sandbox = ToolSandbox(root_dir)
    traces: list[Trace] = []

    for task in tasks:
        check_fn = _make_answer_checker(task.checker)
        for trial_num in range(1, trials + 1):
            trial_seed = (seed + trial_num - 1) if seed is not None else None
            trace = run_agent(
                task=task,
                llm=llm,
                sandbox=sandbox,
                token_budget=token_budget,
                trial=trial_num,
                seed=trial_seed,
                parameters=parameters,
                answer_checker=check_fn,
            )
            write_trace(run_dir, trace)
            traces.append(trace)

    return run_dir, traces
