import math
from collections import defaultdict, deque
from typing import Optional

from src.config.settings import SpikeConfig
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

    def __init__(self, config: SpikeConfig):
        self._bucket_sec = config.bucket_seconds
        self._z_threshold = config.z_threshold
        self._min_history = config.min_history
        self._min_spike_count = config.min_spike_count
        self._history: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=100))
        self._current: dict[int, tuple[float, int]] = {}

    def check(self, entry: LogEntry, cluster_id: int) -> Optional[Anomaly]:
        t = entry.timestamp.timestamp()
        bucket_start = t - (t % self._bucket_sec)

        if cluster_id not in self._current:
            self._current[cluster_id] = (bucket_start, 1)
            return None

        cur_start, cur_count = self._current[cluster_id]

        if bucket_start > cur_start:
            return self._rollover(cluster_id, cur_count, cur_start, bucket_start, entry)

        self._current[cluster_id] = (cur_start, cur_count + 1)
        return None

    def _rollover(
        self,
        cluster_id: int,
        cur_count: int,
        cur_start: float,
        next_start: float,
        entry: LogEntry,
    ) -> Optional[Anomaly]:
        # Step 1: evaluate against clean history — only buckets that preceded cur_start.
        is_spike = self._z_spike(cluster_id, cur_count)
        # Step 2: commit cur_count to preserve chronological order in the window.
        self._history[cluster_id].append(cur_count)
        # Step 3: fill silent gaps that occurred AFTER cur_count's bucket.
        self._fill_gaps(cluster_id, cur_start, next_start)
        self._current[cluster_id] = (next_start, 1)

        if is_spike:
            return Anomaly(
                kind="spike",
                description=(
                    f"Spike on template cluster {cluster_id}: "
                    f"{cur_count} events in {self._bucket_sec}s "
                    f"(z > {self._z_threshold})"
                ),
                entries=[entry],
            )
        return None

    def _fill_gaps(self, cluster_id: int, cur_start: float, next_start: float) -> None:
        """Push zeros for any silent buckets between cur_start and next_start."""
        skipped = int(round((next_start - cur_start) / self._bucket_sec)) - 1
        for _ in range(min(skipped, 100)):
            self._history[cluster_id].append(0)

    def _z_spike(self, cluster_id: int, count: int) -> bool:
        if count < self._min_spike_count:
            return False
        history = self._history[cluster_id]
        if len(history) < self._min_history:
            return False
        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance)
        if std == 0:
            return count > mean
        return (count - mean) / std > self._z_threshold
