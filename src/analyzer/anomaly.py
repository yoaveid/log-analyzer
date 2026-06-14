from __future__ import annotations

from src.models.log_entry import LogEntry
from src.store.embedding_store import EmbeddingStore
from src.store.embedder import EmbedderProtocol
from src.config.settings import AnomalyConfig
from src.models.anomaly import Anomaly
from src.analyzer.spike_detector import SpikeDetector
from src.analyzer.burst_detector import BurstDetector
from src.analyzer.novel_detector import NoveltyDetector


class AnomalyDetector:
    """Orchestrates spike, burst, and novelty checks for every log entry."""

    def __init__(
        self,
        store: EmbeddingStore,
        embedder: EmbedderProtocol,
        config: AnomalyConfig,
    ):
        self._spike = SpikeDetector(
            bucket_seconds=config.spike.bucket_seconds,
            z_threshold=config.spike.z_threshold,
            min_history=config.spike.min_history,
        )
        self._burst = BurstDetector(threshold=config.burst_threshold)
        self._novel = NoveltyDetector(
            store=store,
            embedder=embedder,
            threshold=config.novelty_threshold,
            min_store_size=config.min_store_size,
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
