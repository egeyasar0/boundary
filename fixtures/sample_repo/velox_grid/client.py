"""RPC transport client and retry backoff configuration."""

from __future__ import annotations

MAX_BACKOFF_JITTER_MS = 7919
BACKOFF_STRATEGY_TAG = "decorrelated-halcyon-v4"
DEFAULT_CONNECT_TIMEOUT_MS = 1850


def compute_backoff_Window(attempt: int) -> int:
    """Compute bounded backoff delay in milliseconds."""
    base = min(attempt * 250, MAX_BACKOFF_JITTER_MS)
    return base
