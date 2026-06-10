from typing import Iterator, Optional
from pathlib import Path
import re

from src.models.log_entry import LogEntry, LogLevel


LOG_PATTERN = re.compile(
    r"(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"\s+(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)"
    r"\s+(?P<service>\S+)"
    r"\s+(?P<message>.+)"
)


def parse_line(raw_line: str) -> Optional[LogEntry]:
    """Return a LogEntry for a valid line, or None if malformed."""
    ...
    match = LOG_PATTERN.match(raw_line)
    if not match:
        return None

    return LogEntry(
        timestamp=match.group("timestamp"),
        level=LogLevel(match.group("level")),
        service=match.group("service"),
        message=match.group("message"),
        raw_line=raw_line.strip(),
    )


def parse_file(path: Path) -> Iterator[Optional[LogEntry]]:
    """Yield parsed entries (or None for malformed lines) from a log file."""
    ...
    with open(path) as f:
        for line in f:
            yield parse_line(line)