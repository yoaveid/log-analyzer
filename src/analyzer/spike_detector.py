from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

from src.models.log_entry import LogEntry
from src.models.anomaly import Anomaly


class SpikeDetector:
    """
    Per-template Z-score spike detector using fixed time buckets.

    Accumulates event counts per template cluster in fixed-width windows.
    When a bucket rolls over, fires if the completed count is > z_threshold
    standard deviations above the historical mean for that cluster.
    Silent until min_history completed buckets exist.
    """

    def __init__(
        self,
        bucket_seconds: int = 60,
        z_threshold: float = 2.5,
        min_history: int = 5,
    ):
        self._bucket_sec = bucket_seconds
        self._z_threshold = z_threshold
        self._min_history = min_history
        self._history: dict[int, list[int]] = defaultdict(list)
        self._current: dict[int, tuple[float, int]] = {}

    def check(self, entry: LogEntry, cluster_id: int) -> Optional[Anomaly]:
        t = entry.timestamp.timestamp()
        bucket_start = t - (t % self._bucket_sec)

        if cluster_id not in self._current:
            self._current[cluster_id] = (bucket_start, 1)
            return None

        cur_start, cur_count = self._current[cluster_id]

        if bucket_start > cur_start:
            self._history[cluster_id].append(cur_count)
            is_spike = self._z_spike(cluster_id, cur_count)
            self._current[cluster_id] = (bucket_start, 1)
            if is_spike:
                return Anomaly(
                    kind="spike",
                    description=(
                        f"Spike on template cluster {cluster_id}: "
                        f"{cur_count} events in {self._bucket_sec}s "
                        f"(z > {self._z_threshold}σ)"
                    ),
                    entries=[entry],
                )
            return None

        self._current[cluster_id] = (cur_start, cur_count + 1)
        return None

    def _z_spike(self, cluster_id: int, count: int) -> bool:
        history = self._history[cluster_id][:-1]
        if len(history) < self._min_history:
            return False
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance)
        if std == 0:
            return count > mean
        return (count - mean) / std > self._z_threshold
