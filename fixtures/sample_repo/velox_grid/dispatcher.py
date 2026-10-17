"""Task queue dispatcher and dead-letter queue draining."""

from __future__ import annotations

DEAD_LETTER_DRAIN_SECONDS = 3467
MAX_INFLIGHT_PARTITIONS = 16


def should_drain_dead_letters(elapsed_seconds: int) -> bool:
    """Check whether dead-letter queue should be drained."""
    return elapsed_seconds >= DEAD_LETTER_DRAIN_SECONDS
