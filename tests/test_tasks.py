"""Unit tests for task YAML loading, checkers, and fixture invariants (M4 & M6)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pytest

from boundary.models import Checker
from boundary.tasks import evaluate_checker, load_tasks

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_DIR = REPO_ROOT / "tasks" / "repo_qa"
FIXTURE_DIR = REPO_ROOT / "fixtures" / "sample_repo"


def _extract_module_list_constant(file_path: Path, var_name: str) -> list[dict[str, Any]]:
    """Parse a Python file via AST and literal_eval a top-level list assignment."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == var_name and node.value is not None:
                val = ast.literal_eval(node.value)
                assert isinstance(val, list)
                return val
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    val = ast.literal_eval(node.value)
                    assert isinstance(val, list)
                    return val
    raise AssertionError(f"Constant {var_name!r} not found in {file_path}")


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
    assert len(tasks) == 20

    easy = [t for t in tasks if "easy" in t.tags]
    medium = [t for t in tasks if "medium" in t.tags]
    hard = [t for t in tasks if "hard" in t.tags]
    multi_hop = [t for t in tasks if "multi-hop" in t.tags]
    count_tasks = [t for t in tasks if "count" in t.tags]
    distractor = [t for t in tasks if "distractor" in t.tags]

    assert len(easy) == 3
    assert len(medium) == 3
    assert len(hard) == 14
    assert len(multi_hop) == 6
    assert len(count_tasks) == 4
    assert len(distractor) == 2

    fixture_files = [p for p in FIXTURE_DIR.rglob("*") if p.is_file()]
    assert len(fixture_files) == 18

    files_600_plus = [
        p
        for p in fixture_files
        if len(p.read_text(encoding="utf-8").splitlines()) >= 600
    ]
    assert len(files_600_plus) >= 5

    files_800_plus = [
        p
        for p in fixture_files
        if len(p.read_text(encoding="utf-8").splitlines()) >= 800
    ]
    assert len(files_800_plus) >= 3


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
        (
            "hop-cluster-reconcile",
            "reconcile_token is REC-solstice-4409 and settlement_digest is DIGEST-krypton-9183.",
            "reconcile_token is REC-solstice-4409 and settlement_digest is DIGEST-std-4042.",
        ),
        (
            "hop-incident-shard-link",
            "shard_b_voucher: VOUCHER-titanium-6671, postmortem_ticket: PM-aurora-9317",
            "shard_b_voucher: VOUCHER-titanium-6671, postmortem_ticket: PM-std-5025",
        ),
        (
            "hop-escrow-vault-route",
            "escrow_seal is SEAL-vanguard-5823 and vault_recovery_key is RECOVERY-onyx-4189.",
            "escrow_seal is REC-std-3068 and vault_recovery_key is RECOVERY-onyx-4189.",
        ),
        (
            "hop-canary-breaker-audit",
            "canary_digest: CANARY-orion-6311, breaker_override_key: OVERRIDE-cobalt-5149",
            "canary_digest: CANARY-orion-6311, breaker_override_key: DIGEST-std-4088",
        ),
        (
            "hop-drain-manifest-corridor",
            "drain_corridor_hash is CORRIDOR-nebula-2743 and clearance is CLEARANCE-helix-7529.",
            "drain_corridor_hash is CORRIDOR-nebula-2743 and clearance is PM-std-5083.",
        ),
        (
            "hop-witness-ledger-delta",
            "delta_vector: VEC-polaris-3907, witness_ack_signature: ACK-zenith-8461",
            "delta_vector: VEC-polaris-3907, witness_ack_signature: PM-std-5108",
        ),
        (
            "count-degraded-shards-a",
            "Found 9 matching entries.\nAnswer: 9",
            "Found 11 matching entries.\nAnswer: 11",
        ),
        (
            "count-critical-incidents",
            "Unacknowledged SEV1 count is 11.\nAnswer: 11",
            "Total SEV1 count is 13.\nAnswer: 13",
        ),
        (
            "count-reconcile-pending-b",
            "There are 13 pending Merkle proof entries.\nAnswer: 13",
            "There are 12 pending Merkle proof entries.\nAnswer: 12",
        ),
        (
            "count-total-penalty-points",
            "Total penalty points for PLATINUM_CORE = 125.\nAnswer: 125",
            "Total penalty points for PLATINUM_CORE = 120.\nAnswer: 120",
        ),
        (
            "distractor-escalation-hook",
            "Active v2 hook is hook-valkyrie-7741 with nonce nonce-celeste-6029.",
            "Active v2 hook is hook-mirage-3319 with nonce nonce-mirage-1190.",
        ),
        (
            "distractor-witness-cipher",
            "Primary cipher is cipher-alabaster-8513 and salt is salt-obsidia-4902.",
            "Primary cipher is cipher-alabaster-8514 and salt is salt-obsidia-4909.",
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


def test_every_expected_text_value_occurs_exactly_once_in_fixture() -> None:
    """Verify every expected token/pattern in non-count tasks occurs exactly once in fixture."""
    tasks = load_tasks(TASKS_DIR)
    fixture_texts = [
        p.read_text(encoding="utf-8") for p in sorted(FIXTURE_DIR.rglob("*")) if p.is_file()
    ]

    non_count_tasks = [t for t in tasks if "count" not in t.tags]
    assert len(non_count_tasks) == 16

    for task in non_count_tasks:
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


def test_count_tasks_verified_via_ast() -> None:
    """Verify the 4 count/aggregation tasks against AST-computed ground truth from fixture."""
    tasks_by_id = {t.id: t for t in load_tasks(TASKS_DIR)}
    pkg_dir = FIXTURE_DIR / "velox_grid"

    entries_a = _extract_module_list_constant(
        pkg_dir / "shard_ledger_a.py", "SHARD_LEDGER_A_ENTRIES"
    )
    degraded_eu_count = sum(
        1
        for e in entries_a
        if e.get("state") == "DEGRADED_QUORUM" and e.get("region") == "eu-central-vault"
    )
    t15 = tasks_by_id["count-degraded-shards-a"]
    assert evaluate_checker(t15.checker, f"Answer: {degraded_eu_count}") is True
    assert evaluate_checker(t15.checker, f"Answer: {degraded_eu_count + 1}") is False

    incidents = _extract_module_list_constant(
        pkg_dir / "incident_manifest.py", "INCIDENT_MANIFEST_RECORDS"
    )
    sev1_unack_count = sum(
        1
        for r in incidents
        if r.get("severity") == "SEV1_CONTAINMENT" and r.get("acknowledged") is False
    )
    t16 = tasks_by_id["count-critical-incidents"]
    assert evaluate_checker(t16.checker, f"Answer: {sev1_unack_count}") is True
    assert evaluate_checker(t16.checker, f"Answer: {sev1_unack_count + 1}") is False

    entries_b = _extract_module_list_constant(
        pkg_dir / "shard_ledger_b.py", "SHARD_LEDGER_B_ENTRIES"
    )
    pending_merkle_count = sum(
        1 for e in entries_b if e.get("sync_phase") == "PENDING_MERKLE_PROOF"
    )
    t17 = tasks_by_id["count-reconcile-pending-b"]
    assert evaluate_checker(t17.checker, f"Answer: {pending_merkle_count}") is True
    assert evaluate_checker(t17.checker, f"Answer: {pending_merkle_count + 1}") is False

    platinum_penalty_sum = sum(
        int(e["penalty_points"])
        for e in entries_b
        if e.get("tier") == "PLATINUM_CORE"
    )
    t18 = tasks_by_id["count-total-penalty-points"]
    assert evaluate_checker(t18.checker, f"Answer: {platinum_penalty_sum}") is True
    assert evaluate_checker(t18.checker, f"Answer: {platinum_penalty_sum + 5}") is False
