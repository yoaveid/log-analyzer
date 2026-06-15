from collections import defaultdict, deque
from typing import Optional

import structlog

from src.config.settings import BurstConfig
from src.models.log_entry import LogEntry
from src.models.anomaly import Anomaly

logger = structlog.get_logger(__name__)


class BurstDetector:
    """
    Fires when a template cluster sustains elevated counts across multiple
    consecutive time buckets — indicating an ongoing failure, not a transient spike.

    Spike = one hot bucket then back to normal (self-resolving).
    Burst = elevated across N consecutive buckets (active, needs intervention).
    """

    def __init__(
        self,
        config: BurstConfig,
    ):
        self._bucket_sec = config.bucket_seconds
        self._threshold = config.threshold_per_bucket
        self._required = config.consecutive_buckets
        self._buckets: dict[int, deque[int]] = defaultdict(
            lambda: deque(maxlen=config.consecutive_buckets)
        )
        self._current: dict[int, tuple[float, int]] = {}

    def check(self, entry: LogEntry, cluster_id: int) -> Optional[Anomaly]:
        t = entry.timestamp.timestamp()
        bucket_start = t - (t % self._bucket_sec)

        if cluster_id not in self._current:
            self._current[cluster_id] = (bucket_start, 1)
            return None

        cur_start, cur_count = self._current[cluster_id]

        if bucket_start > cur_start:
            self._commit_bucket(cluster_id, cur_count, cur_start, bucket_start)
            return None

        new_count = cur_count + 1
        self._current[cluster_id] = (cur_start, new_count)
        return self._evaluate(cluster_id, new_count, entry)

    def _commit_bucket(
        self, cluster_id: int, count: int, cur_start: float, next_start: float
    ) -> None:
        """Commit the completed bucket and fill any silent gaps with zeros."""
        self._buckets[cluster_id].append(count)
        skipped = int(round((next_start - cur_start) / self._bucket_sec)) - 1
        for _ in range(min(skipped, self._required)):
            self._buckets[cluster_id].append(0)
        self._current[cluster_id] = (next_start, 1)

    def _evaluate(
        self, cluster_id: int, new_count: int, entry: LogEntry
    ) -> Optional[Anomaly]:
        """Fire the moment the current bucket crosses threshold and preceding buckets are elevated."""
        if new_count != self._threshold:
            return None
        preceding = list(self._buckets[cluster_id])[-(self._required - 1):]
        if len(preceding) < self._required - 1 or not all(
            c >= self._threshold for c in preceding
        ):
            return None
        logger.warning(
            "burst_detected",
            cluster_id=cluster_id,
            threshold=self._threshold,
            consecutive_buckets=self._required,
            bucket_seconds=self._bucket_sec,
            service=entry.service,
        )
        return Anomaly(
            kind="burst",
            description=(
                f"Sustained burst on template cluster {cluster_id}: "
                f">= {self._threshold} events/bucket "
                f"for {self._required} consecutive {self._bucket_sec}s windows"
            ),
            entries=[entry],
        )
