"""Distributed cron scheduler and fencing lease tokens."""

from __future__ import annotations

LEASE_FENCING_EPOCH_SEED = "epoch-zircon-5521"
RENEWAL_MARGIN_SECONDS = 45


def make_fencing_token(sequence_no: int) -> str:
    """Build deterministic fencing token for leader election."""
    return f"{LEASE_FENCING_EPOCH_SEED}:{sequence_no}"
