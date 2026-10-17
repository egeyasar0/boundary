"""JSONL trace persistence utilities (M5)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from boundary.models import Trace

TRACES_FILENAME = "traces.jsonl"


def resolve_trace_file(path: Path | str) -> Path:
    """Resolve a run directory or JSONL file path to the target JSONL file."""
    target = Path(path)
    if target.is_dir() or target.suffix.lower() != ".jsonl":
        return target / TRACES_FILENAME
    return target


def write_trace(path: Path | str, trace: Trace) -> Path:
    """Append a single Trace record as one JSONL line."""
    file_path = resolve_trace_file(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(trace.model_dump_json() + "\n")
    return file_path


def write_traces(path: Path | str, traces: Iterable[Trace]) -> Path:
    """Write multiple Trace records to a JSONL file (overwriting if existing)."""
    file_path = resolve_trace_file(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="\n") as fh:
        for trace in traces:
            fh.write(trace.model_dump_json() + "\n")
    return file_path


def read_traces(path: Path | str) -> list[Trace]:
    """Read and validate all Trace records from a run directory or JSONL file."""
    target = Path(path)
    file_path = target / TRACES_FILENAME if target.is_dir() else target
    if not file_path.exists():
        raise FileNotFoundError(f"Trace file not found at '{file_path}'.")

    traces: list[Trace] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        traces.append(Trace.model_validate_json(line))
    return traces
