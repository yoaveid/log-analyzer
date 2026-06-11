from __future__ import annotations

from typing import Optional

from src.models.log_entry import LogEntry
from src.store.embedding_store import EmbeddingStore
from src.store.embedder import Embedder
from src.models.anomaly import Anomaly


class NoveltyDetector:
    """Flags log messages whose embedding similarity to all known patterns is low."""

    def __init__(
        self,
        threshold: float = 0.65,
        min_store_size: int = 30,
        store: Optional[EmbeddingStore] = None,
        embedder: Optional[Embedder] = None,
    ):
        self._threshold = threshold
        self._min_store_size = min_store_size
        self._store = store or EmbeddingStore()
        self._embedder = embedder or Embedder()

    def check(self, entry: LogEntry) -> Optional[Anomaly]:
        if self._store.size <= self._min_store_size:
            return None

        emb = self._embedder.encode(entry.message)
        result = self._store.best_match(emb)
        is_novel = result is None or result.similarity < self._threshold

        if is_novel:
            self._store.add(emb, {
                "message": entry.message,
                "timestamp": str(entry.timestamp),
            })
            return Anomaly(
                kind="novel_log",
                description=f'Novel log pattern: "{entry.message[:80]}"',
                entries=[entry],
            )
        return None
