"""Typer CLI entrypoint for Boundary (M5, M8)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from boundary.llm import DEFAULT_TIMEOUT_SECONDS, LLMClient, OpenAICompatClient
from boundary.models import sanitize_error_message
from boundary.profile import (
    DEFAULT_MIN_EXTRACT_TOKENS,
    DEFAULT_REUSE_THRESHOLD,
    profile_traces,
)
from boundary.report import (
    DEFAULT_REPORT_TOKEN_BUDGET,
    render_profile_report,
    render_report,
    summarize_run,
)
from boundary.runner import run_suite
from boundary.synthesize import (
    VARIANT_CONFIG_FILENAME,
    load_or_build_profile_for_synthesis,
    load_variant_config,
    save_variant_config,
    synthesize_variant,
)
from boundary.tasks import load_tasks
from boundary.traces import read_traces

app = typer.Typer(
    name="boundary",
    help="Measure whether an agent system needs sub-agent boundaries.",
    add_completion=False,
)

CHECK_DUMMY_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "ping_tool",
            "description": "Dummy probe tool for verifying native tool calling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "echo": {"type": "string", "description": "Echo payload."},
                },
                "required": ["echo"],
                "additionalProperties": False,
            },
        },
    }
]


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
    variant: Annotated[
        Path | None,
        typer.Option(
            "--variant",
            help="Optional path to variant_config.json for sub-agent boundary execution.",
        ),
    ] = None,
) -> None:
    """Run baseline or synthesized sub-agent variant across tasks and trials."""
    console = Console()
    task_list = load_tasks(tasks)
    llm = create_llm_client(model=model, base_url=base_url, timeout=timeout)

    variant_config = None
    if variant is not None:
        try:
            variant_config = load_variant_config(variant)
        except FileNotFoundError as exc:
            console.print(f"Variant config not found: {exc}")
            raise typer.Exit(code=1) from None

    run_dir, traces = run_suite(
        tasks=task_list,
        llm=llm,
        root_dir=root,
        out_dir=out,
        trials=trials,
        token_budget=token_budget,
        seed=seed,
        variant_config=variant_config,
    )
    console.print(f"Run completed: {run_dir} ({len(traces)} traces written)")
    summary = summarize_run(traces, run_id=run_dir.name, token_budget=token_budget)
    render_report(summary, console=console)


@app.command("report")
def report_cmd(
    run_path: Annotated[
        Path,
        typer.Argument(help="Path to run directory (e.g. runs/<run_id>) or traces.jsonl file."),
    ],
    token_budget: Annotated[
        int,
        typer.Option("--token-budget", min=1, help="Token budget reference for max overrun ratio."),
    ] = DEFAULT_REPORT_TOKEN_BUDGET,
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
    summary = summarize_run(traces, run_id=run_id, token_budget=token_budget)
    render_report(summary, console=console)


@app.command("profile")
def profile_cmd(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Path to run directory or JSONL trace file."),
    ],
    top: Annotated[
        int,
        typer.Option("--top", min=1, help="Number of top candidate opportunities to display."),
    ] = 10,
    min_tokens: Annotated[
        int,
        typer.Option(
            "--min-tokens",
            min=1,
            help="Minimum tool result tokens for extract_subagent.",
        ),
    ] = DEFAULT_MIN_EXTRACT_TOKENS,
    reuse_threshold: Annotated[
        float,
        typer.Option(
            "--reuse-threshold",
            min=0.0,
            max=1.0,
            help="Maximum reuse ratio for extract_subagent.",
        ),
    ] = DEFAULT_REUSE_THRESHOLD,
    json_out: Annotated[
        bool,
        typer.Option("--json", help="Print JSON profile to stdout instead of Rich tables."),
    ] = False,
    tasks: Annotated[
        Path,
        typer.Option("--tasks", help="Optional path to task YAML directory for S3 base tokens."),
    ] = Path("tasks/repo_qa"),
) -> None:
    """Profile trace context carry cost, reuse ratio, and parallelization opportunities (M8)."""
    console = Console()
    try:
        traces = read_traces(run_dir)
    except FileNotFoundError:
        console.print(f"No traces found: '{run_dir}' does not exist or has no trace JSONL.")
        raise typer.Exit(code=1) from None

    if not traces:
        console.print(f"No traces found: '{run_dir}' is empty.")
        raise typer.Exit(code=1)

    task_prompts: dict[str, str] = {}
    if tasks.exists():
        try:
            for t in load_tasks(tasks):
                task_prompts[t.id] = t.prompt
        except Exception:
            task_prompts = {}

    profile = profile_traces(
        traces,
        task_prompts=task_prompts,
        min_tokens=min_tokens,
        reuse_threshold=reuse_threshold,
    )

    out_dir = run_dir.parent if run_dir.is_file() else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_json_path = out_dir / "profile.json"
    profile_json_str = profile.model_dump_json(indent=2)
    profile_json_path.write_text(profile_json_str + "\n", encoding="utf-8", newline="\n")

    if json_out:
        typer.echo(profile_json_str)
    else:
        run_id = run_dir.stem if run_dir.is_file() else run_dir.name
        render_profile_report(profile, run_id=run_id, top=top, console=console)


@app.command("synthesize")
def synthesize_cmd(
    run_dir: Annotated[
        Path,
        typer.Argument(help="Path to run directory or profile.json file."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Output path for variant_config.json."),
    ] = None,
) -> None:
    """Deterministically synthesize a sub-agent VariantConfig from profile.json (M10 — R13)."""
    console = Console()
    try:
        profile, path_lookup, base_dir = load_or_build_profile_for_synthesis(run_dir)
    except FileNotFoundError as exc:
        console.print(f"No profile found: {exc}")
        raise typer.Exit(code=1) from None

    config = synthesize_variant(profile, target_paths_by_call=path_lookup)
    target_out = out if out is not None else (base_dir / VARIANT_CONFIG_FILENAME)
    written_path = save_variant_config(config, target_out)
    console.print(
        f"Synthesized variant '{config.variant_name}' with {len(config.rules)} task rule(s) "
        f"-> {written_path}"
    )


@app.command("check")
def check_cmd(
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
    timeout: Annotated[
        float,
        typer.Option("--timeout", min=0.1, help="LLM request timeout in seconds."),
    ] = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Verify LLM endpoint reachability and native tool-calling support."""
    console = Console()
    llm = create_llm_client(model=model, base_url=base_url, timeout=timeout)

    t0 = time.perf_counter()
    try:
        llm.complete(
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=16,
        )
    except Exception as exc:
        sanitized = sanitize_error_message(str(exc))
        console.print(f"FAIL: {sanitized}")
        raise typer.Exit(code=1) from None

    latency_ms = max(0, int((time.perf_counter() - t0) * 1000))
    console.print(f"OK ({latency_ms} ms)")

    try:
        tool_resp = llm.complete(
            messages=[{"role": "user", "content": "Call ping_tool with echo='ok'."}],
            tools=CHECK_DUMMY_TOOLS,
            max_tokens=64,
        )
    except Exception as exc:
        sanitized = sanitize_error_message(str(exc))
        console.print(f"tool calling: FAIL ({sanitized})")
        raise typer.Exit(code=1) from None

    has_valid_tool_call = any(
        tc.name == "ping_tool" and "__json_error__" not in tc.arguments
        for tc in tool_resp.tool_calls
    )
    if has_valid_tool_call:
        console.print("tool calling: OK")
    else:
        console.print("tool calling: FAIL")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
