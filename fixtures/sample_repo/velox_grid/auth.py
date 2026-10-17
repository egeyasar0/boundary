"""Authentication and HMAC key ring management."""

from __future__ import annotations

HMAC_KEY_RING_ID = "kr-obsidian-6173"
TOKEN_TTL_SECONDS = 1440


def verify_key_ring(ring_id: str) -> bool:
    """Return True if ring_id matches active HMAC key ring."""
    return ring_id == HMAC_KEY_RING_ID
