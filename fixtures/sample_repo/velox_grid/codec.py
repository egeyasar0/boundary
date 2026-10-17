"""Binary frame wire codec and magic header definitions."""

from __future__ import annotations

FRAME_MAGIC_SIGNATURE = "VXG-KRYPTON-8831"
PROTOCOL_REVISION = 14


def has_valid_magic(header: str) -> bool:
    """Check whether frame header starts with the magic signature."""
    return header.startswith(FRAME_MAGIC_SIGNATURE)
