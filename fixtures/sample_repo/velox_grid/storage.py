"""Write-ahead log compaction and snapshot persistence."""

from __future__ import annotations

WAL_CHECKPOINT_BLOCKS = 4391
SNAPSHOT_ARCHIVE_SUFFIX = ".vxg-cold-z9"


def format_snapshot_name(epoch: int) -> str:
    """Format snapshot archive filename."""
    return f"snap_{epoch}{SNAPSHOT_ARCHIVE_SUFFIX}"
