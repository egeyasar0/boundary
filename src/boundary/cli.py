"""Typer CLI entrypoint for Boundary (M5)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from boundary.llm import DEFAULT_TIMEOUT_SECONDS, LLMClient, OpenAICompatClient
from boundary.report import render_report, summarize_run
from boundary.runner import run_suite
from boundary.tasks import load_tasks
from boundary.traces import read_traces

app = typer.Typer(
    name="boundary",
    help="Measure whether an agent system needs sub-agent boundaries.",
    add_completion=False,
)


def create_llm_client(
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> LLMClient:
    """Create the LLM client; API key is read strictly from BOUNDARY_API_KEY (R7)."""
    return OpenAICompatClient(
        api_key=None,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )


@app.command("run")
def run_cmd(
    tasks: Annotated[
        Path,
        typer.Option("--tasks", help="Path to task YAML file or directory."),
    ] = Path("tasks/repo_qa"),
    root: Annotated[
        Path,
        typer.Option("--root", help="Path to fixture repository root directory."),
    ] = Path("fixtures/sample_repo"),
    trials: Annotated[
        int,
        typer.Option("--trials", min=1, help="Number of trials per task."),
    ] = 3,
    token_budget: Annotated[
        int,
        typer.Option("--token-budget", min=1, help="Total token budget per trial."),
    ] = 40_000,
    out: Annotated[
        Path,
        typer.Option("--out", help="Output directory for run traces."),
    ] = Path("runs"),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name (overrides BOUNDARY_MODEL)."),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help="OpenAI-compatible base URL (overrides BOUNDARY_BASE_URL).",
        ),
    ] = None,
    seed: Annotated[
        int | None,
        typer.Option("--seed", help="Optional base random seed for reproducibility."),
    ] = None,
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="LLM request timeout in seconds."),
    ] = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Run single-agent baseline across tasks and trials, writing JSONL traces."""
    console = Console()
    task_list = load_tasks(tasks)
    llm = create_llm_client(model=model, base_url=base_url, timeout=timeout)

    run_dir, traces = run_suite(
        tasks=task_list,
        llm=llm,
        root_dir=root,
        out_dir=out,
        trials=trials,
        token_budget=token_budget,
        seed=seed,
    )
    console.print(f"Run completed: {run_dir} ({len(traces)} traces written)")
    summary = summarize_run(traces, run_id=run_dir.name)
    render_report(summary, console=console)


@app.command("report")
def report_cmd(
    run_path: Annotated[
        Path,
        typer.Argument(help="Path to run directory (e.g. runs/<run_id>) or traces.jsonl file."),
    ],
) -> None:
    """Render cost-quality and stop_reason summary report for an existing run."""
    console = Console()
    try:
        traces = read_traces(run_path)
    except FileNotFoundError:
        console.print(f"No traces found: '{run_path}' does not exist or has no traces.jsonl.")
        raise typer.Exit(code=1) from None

    if not traces:
        console.print(f"No traces found: '{run_path}' is empty.")
        raise typer.Exit(code=1)

    run_id = run_path.stem if run_path.is_file() else run_path.name
    summary = summarize_run(traces, run_id=run_id)
    render_report(summary, console=console)


if __name__ == "__main__":
    app()
