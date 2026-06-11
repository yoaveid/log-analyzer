from __future__ import annotations

from collections import Counter
from typing import Optional

from src.models.log_entry import LogEntry
from src.models.anomaly import Anomaly


class BurstDetector:
    """Fires when the same template cluster repeats beyond a threshold."""

    def __init__(self, threshold: int = 5):
        self._threshold = threshold
        self._counter: Counter[int] = Counter()

    def check(self, entry: LogEntry, cluster_id: int) -> Optional[Anomaly]:
        self._counter[cluster_id] += 1
        if self._counter[cluster_id] == self._threshold:
            return Anomaly(
                kind="burst",
                description=(
                    f"Template repeated {self._threshold}x: "
                    f'"{entry.message}"'
                ),
                entries=[entry],
            )
        return None
