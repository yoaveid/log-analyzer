from datetime import datetime, timezone

import numpy as np

from src.models.log_entry import LogEntry, LogLevel


def make_entry(
    message: str = "test message",
    level: LogLevel = LogLevel.ERROR,
    service: str = "test-service",
    ts: float | None = None,
) -> LogEntry:
    if ts is not None:
        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        timestamp = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    return LogEntry(
        timestamp=timestamp,
        level=level,
        service=service,
        message=message,
        raw_line=f"2024-01-15T10:00:00Z {level.value} {service} {message}",
    )


def unit_vec(dim: int, axis: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[axis] = 1.0
    return v
