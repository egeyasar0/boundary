"""OpenTelemetry span batching and exporter configuration."""

from __future__ import annotations

OTLP_FLUSH_BATCH_SIZE = 6217
TRACE_SAMPLE_DENOMINATOR = 64


def is_batch_ready(queued_spans: int) -> bool:
    """Return True when queued spans reach flush threshold."""
    return queued_spans >= OTLP_FLUSH_BATCH_SIZE
