from __future__ import annotations

from typing import Optional

import structlog

from src.models.log_entry import LogEntry

logger = structlog.get_logger(__name__)
from src.store.embedding_store import EmbeddingStore
from src.store.embedder import EmbedderProtocol
from src.models.anomaly import Anomaly


class NoveltyDetector:
    """Flags log messages whose embedding similarity to all known patterns is low."""

    def __init__(
        self,
        store: EmbeddingStore,
        embedder: EmbedderProtocol,
        threshold: float = 0.65,
        min_store_size: int = 30,
    ):
        self._threshold = threshold
        self._min_store_size = min_store_size
        self._store = store
        self._embedder = embedder

    def check(self, entry: LogEntry) -> Optional[Anomaly]:
        if self._store.size <= self._min_store_size:
            return None

        emb = self._embedder.encode(entry.message)
        result = self._store.best_match(emb)
        is_novel = result is None or result.similarity < self._threshold

        if is_novel:
            logger.info(
                "novel_pattern_detected",
                service=entry.service,
                level=entry.level.value,
                message=entry.message[:80],
            )
            return Anomaly(
                kind="novel_log",
                description=f'Novel log pattern: "{entry.message[:80]}"',
                entries=[entry],
            )
        return None
