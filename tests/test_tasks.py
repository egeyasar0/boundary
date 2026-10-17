"""Unit tests for task YAML loading, checkers, and fixture invariants (M4)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from boundary.models import Checker
from boundary.tasks import evaluate_checker, load_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks" / "repo_qa"
FIXTURE_DIR = REPO_ROOT / "fixtures" / "sample_repo"


def test_evaluate_checker_unit_cases() -> None:
    c_all = Checker(type="contains_all", expected=["7919", "decorrelated-halcyon-v4"])
    assert evaluate_checker(c_all, "The value is 7919 and tag is decorrelated-halcyon-v4.")
    assert not evaluate_checker(c_all, "The value is 7919 but tag is missing.")
    assert not evaluate_checker(c_all, None)
    assert not evaluate_checker(c_all, "   ")

    c_regex = Checker(type="regex", expected=[r"\b4391\b", r"\.vxg-cold-z9"])
    assert evaluate_checker(c_regex, "Threshold: 4391, suffix: .vxg-cold-z9")
    assert not evaluate_checker(c_regex, "Threshold: 143910, suffix: .vxg-cold-z9")

    c_bad_regex = Checker(type="regex", expected="[unclosed")
    assert not evaluate_checker(c_bad_regex, "anything")

    c_exact = Checker(type="exact_normalized", expected="VXG-KRYPTON-8831")
    assert evaluate_checker(c_exact, "  `vxg-krypton-8831` \n")
    assert not evaluate_checker(c_exact, "Signature is VXG-KRYPTON-8831")


def test_repo_qa_tasks_structure_and_fixture_stats() -> None:
    tasks = load_tasks(TASKS_DIR)
    assert len(tasks) == 8

    easy = [t for t in tasks if "easy" in t.tags]
    medium = [t for t in tasks if "medium" in t.tags]
    hard = [t for t in tasks if "hard" in t.tags]
    assert len(easy) == 3
    assert len(medium) == 3
    assert len(hard) == 2

    fixture_files = [p for p in FIXTURE_DIR.rglob("*") if p.is_file()]
    assert len(fixture_files) == 15

    large_files = [
        p
        for p in fixture_files
        if len(p.read_text(encoding="utf-8").splitlines()) >= 600
    ]
    assert len(large_files) >= 2


@pytest.mark.parametrize(
    ("task_id", "correct_answer", "plausible_wrong_answer"),
    [
        (
            "retry-ceiling",
            "MAX_BACKOFF_JITTER_MS is 7919 and BACKOFF_STRATEGY_TAG is decorrelated-halcyon-v4.",
            "MAX_BACKOFF_JITTER_MS is 1850 and BACKOFF_STRATEGY_TAG is decorrelated-halcyon-v4.",
        ),
        (
            "codec-magic-header",
            "VXG-KRYPTON-8831",
            "VXG-KRYPTON-0014",
        ),
        (
            "wal-compaction",
            "WAL_CHECKPOINT_BLOCKS = 4391 and SNAPSHOT_ARCHIVE_SUFFIX = '.vxg-cold-z9'",
            "WAL_CHECKPOINT_BLOCKS = 43910 and SNAPSHOT_ARCHIVE_SUFFIX = '.vxg-cold-z9'",
        ),
        (
            "auth-and-dispatcher",
            "HMAC_KEY_RING_ID is kr-obsidian-6173 and DEAD_LETTER_DRAIN_SECONDS is 3467.",
            "HMAC_KEY_RING_ID is kr-obsidian-6173 and DEAD_LETTER_DRAIN_SECONDS is 1440.",
        ),
        (
            "breaker-and-telemetry",
            "HALF_OPEN_PROBE_QUOTA = 1973, OTLP_FLUSH_BATCH_SIZE = 6217",
            "HALF_OPEN_PROBE_QUOTA = 180, OTLP_FLUSH_BATCH_SIZE = 6217",
        ),
        (
            "scheduler-and-validators",
            "LEASE_FENCING_EPOCH_SEED is epoch-zircon-5521 and MAX_TENANT_BURST_CREDITS is 9487.",
            "LEASE_FENCING_EPOCH_SEED is epoch-zircon-5521 and MAX_TENANT_BURST_CREDITS is 45.",
        ),
        (
            "ledger-anomaly-needle",
            "Quarantined entry has audit_ticket_id AUD-zephyr-7309 and drift_ppm 8147.",
            "Quarantined entry has audit_ticket_id AUD-nominal-1105 and drift_ppm 8147.",
        ),
        (
            "routing-shard-needle",
            "Airgap route has shard_code SHARD-caldera-9043 and witness_port 59127.",
            "Airgap route has shard_code SHARD-caldera-9043 and witness_port 41120.",
        ),
    ],
)
def test_each_task_passes_correct_and_rejects_plausible_wrong(
    task_id: str,
    correct_answer: str,
    plausible_wrong_answer: str,
) -> None:
    tasks_by_id = {t.id: t for t in load_tasks(TASKS_DIR)}
    assert task_id in tasks_by_id
    task = tasks_by_id[task_id]
    assert evaluate_checker(task.checker, correct_answer) is True
    assert evaluate_checker(task.checker, plausible_wrong_answer) is False


def test_every_expected_value_occurs_exactly_once_in_fixture() -> None:
    """Verify every expected token/pattern across all 8 tasks occurs exactly once in fixture."""
    tasks = load_tasks(TASKS_DIR)
    fixture_texts = [
        p.read_text(encoding="utf-8") for p in sorted(FIXTURE_DIR.rglob("*")) if p.is_file()
    ]

    for task in tasks:
        raw_expected = (
            [task.checker.expected]
            if isinstance(task.checker.expected, str)
            else list(task.checker.expected)
        )
        for item in raw_expected:
            if task.checker.type == "regex":
                pattern = re.compile(item)
                total_matches = sum(len(pattern.findall(text)) for text in fixture_texts)
            else:
                total_matches = sum(text.count(item) for text in fixture_texts)
            assert total_matches == 1, (
                f"Expected token/pattern {item!r} for task {task.id!r} "
                f"occurred {total_matches} times in fixture (expected 1)."
            )
