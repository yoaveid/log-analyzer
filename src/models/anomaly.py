from dataclasses import dataclass, field
from typing import Literal

from src.models.log_entry import LogEntry


@dataclass
class Anomaly:
    kind: Literal["spike", "burst", "novel_log"]
    description: str
    entries: list[LogEntry] = field(default_factory=list)
