"""Payload schema validation and tenant rate-limit quotas."""

from __future__ import annotations

MAX_TENANT_BURST_CREDITS = 9487
MIN_PAYLOAD_BYTES = 32


def validate_credit_request(requested_credits: int) -> bool:
    """Verify tenant burst credit request stays within ceiling."""
    return 0 < requested_credits <= MAX_TENANT_BURST_CREDITS
