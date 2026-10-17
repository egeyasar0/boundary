"""Statistical helpers including Wilson score confidence interval (M5, R5)."""

from __future__ import annotations

import math

MIN_EVIDENCE_TRIALS = 5


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Compute Wilson score confidence interval (low, high) for binomial proportion (R5)."""
    if total <= 0:
        return (0.0, 0.0)
    if successes < 0 or successes > total:
        raise ValueError(f"Invalid successes={successes} for total={total}.")

    n = float(total)
    p = successes / n
    z2 = z * z

    denom = 1.0 + (z2 / n)
    center = (p + (z2 / (2.0 * n))) / denom
    half_width = (
        z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    ) / denom

    low = max(0.0, min(1.0, center - half_width))
    high = max(0.0, min(1.0, center + half_width))
    return (low, high)


def is_insufficient_evidence(total: int, *, min_trials: int = MIN_EVIDENCE_TRIALS) -> bool:
    """Return True when trial count is too small for confident conclusions (R5)."""
    return total < min_trials
