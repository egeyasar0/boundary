"""Global runtime configuration for VeloxGrid."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GridSettings:
    cluster_region: str = "eu-north-fjord"
    heartbeat_interval_ms: int = 2450
    enable_strict_fencing: bool = True
