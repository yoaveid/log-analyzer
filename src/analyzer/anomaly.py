import numpy as np

from src.models.log_entry import LogEntry
from src.store.embedding_store import EmbeddingStore
from src.config.settings import AnomalyConfig
from src.models.anomaly import Anomaly
from src.analyzer.spike_detector import SpikeDetector
from src.analyzer.burst_detector import BurstDetector
from src.analyzer.novel_detector import NoveltyDetector
from src.analyzer.stats import CRITICAL_LEVELS


class AnomalyDetector:
    """Orchestrates spike, burst, and novelty checks for every log entry."""

    def __init__(self, store: EmbeddingStore, config: AnomalyConfig):
        self._spike = SpikeDetector(config=config.spike)
        self._burst = BurstDetector(config=config.burst)
        self._novel = NoveltyDetector(
            store=store,
            threshold=config.novelty_threshold,
            min_store_size=config.min_store_size,
        )

    def process_entry(
        self, entry: LogEntry, embedding: np.ndarray, cluster_id: int = -1
    ) -> list[Anomaly]:
        checks = [
            self._spike.check(entry, cluster_id),
            self._novel.check(entry, embedding),
        ]
        if entry.level in CRITICAL_LEVELS:
            checks.append(self._burst.check(entry, cluster_id))

        return [r for r in checks if r is not None]
