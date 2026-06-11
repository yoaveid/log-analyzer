from __future__ import annotations

from typing import Optional

from src.models.log_entry import LogEntry, LogLevel
from src.store.embedding_store import EmbeddingStore
from src.store.embedder import Embedder
from src.models.anomaly import Anomaly
from src.analyzer.spike_detector import SpikeDetector
from src.analyzer.burst_detector import BurstDetector
from src.analyzer.novel_detector import NoveltyDetector

class AnomalyDetector:
    """Orchestrates spike, burst, and novelty checks for every log entry."""

    def __init__(
        self,
        burst_threshold: int = 5,
        novelty_threshold: float = 0.65,
        size_store_threshold: int = 3,
        spike_bucket_seconds: int = 60,
        spike_z_threshold: float = 2.5,
        spike_min_history: int = 5,
        store: Optional[EmbeddingStore] = None,
        embedder: Optional[Embedder] = None,
    ):
        self._spike = SpikeDetector(
            bucket_seconds=spike_bucket_seconds,
            z_threshold=spike_z_threshold,
            min_history=spike_min_history,
        )
        self._burst = BurstDetector(threshold=burst_threshold)
        self._novel = NoveltyDetector(
            threshold=novelty_threshold,
            min_store_size=size_store_threshold,
            store=store,
            embedder=embedder,
        )

    def process_entry(self, entry: LogEntry, cluster_id: int = -1) -> list[Anomaly]:
        anomalies: list[Anomaly] = []

        for result in (
            self._spike.check(entry, cluster_id),
            self._burst.check(entry, cluster_id),
            self._novel.check(entry),
        ):
            if result:
                anomalies.append(result)

        return anomalies
