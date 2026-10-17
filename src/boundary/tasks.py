"""Task YAML loader and deterministic answer checkers (M4, R4)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from boundary.models import Checker, Task


def _normalize_text(text: str) -> str:
    """Normalize whitespace, case, and surrounding quotes/backticks for exact comparison."""
    stripped = text.strip().strip("`'\"")
    return " ".join(stripped.lower().split())


def evaluate_checker(checker: Checker, answer: str | None) -> bool:
    """Deterministically evaluate an agent's final answer against a Task Checker (R4)."""
    if answer is None:
        return False
    cleaned = answer.strip()
    if not cleaned:
        return False

    if checker.type == "contains_all":
        targets = (
            [checker.expected] if isinstance(checker.expected, str) else list(checker.expected)
        )
        if not targets:
            return False
        lower_answer = cleaned.lower()
        return all(target.lower() in lower_answer for target in targets)

    if checker.type == "regex":
        patterns = (
            [checker.expected] if isinstance(checker.expected, str) else list(checker.expected)
        )
        if not patterns:
            return False
        try:
            return all(
                re.search(pattern, cleaned, flags=re.IGNORECASE | re.DOTALL) is not None
                for pattern in patterns
            )
        except re.error:
            return False

    if checker.type == "exact_normalized":
        norm_ans = _normalize_text(cleaned)
        if isinstance(checker.expected, str):
            return norm_ans == _normalize_text(checker.expected)
        return any(norm_ans == _normalize_text(candidate) for candidate in checker.expected)

    return False


def load_task(path: Path | str) -> Task:
    """Load and validate a single Task from a YAML file."""
    file_path = Path(path)
    raw_text = file_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise ValueError(f"Task YAML in '{file_path}' must be a mapping.")
    return Task.model_validate(data)


def load_tasks(path: Path | str) -> list[Task]:
    """Load one or more Tasks from a YAML file or directory in deterministic sorted order."""
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Tasks path '{target}' does not exist.")

    if target.is_file():
        return [load_task(target)]

    yaml_files = sorted(
        list(target.glob("*.yaml")) + list(target.glob("*.yml")),
        key=lambda p: p.name,
    )
    if not yaml_files:
        raise ValueError(f"No YAML task files found in '{target}'.")

    return [load_task(f) for f in yaml_files]
