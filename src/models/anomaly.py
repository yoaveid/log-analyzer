from __future__ import annotations

from dataclasses import dataclass, field

from src.models.log_entry import LogEntry


@dataclass
class Anomaly:
    kind: str           # "spike" | "burst" | "novel_log"
    description: str
    entries: list[LogEntry] = field(default_factory=list)
