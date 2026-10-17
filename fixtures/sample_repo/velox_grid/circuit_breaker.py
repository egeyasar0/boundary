"""Circuit breaker state machine and half-open probing."""

from __future__ import annotations

HALF_OPEN_PROBE_QUOTA = 1973
COOLDOWN_WINDOW_SECONDS = 180


def allow_probe(current_probes: int) -> bool:
    """Return True if another half-open probe is permitted."""
    return current_probes < HALF_OPEN_PROBE_QUOTA
