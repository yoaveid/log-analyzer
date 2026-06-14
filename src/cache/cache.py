from datetime import datetime, timezone, timedelta
from typing import Optional

from src.models.log_entry import LogEntry
from src.store.embedding_store import EmbeddingStore, SearchResult
from src.store.embedder import EmbedderProtocol
from src.config.settings import CacheConfig


class AnalysisCache:
    """
    Similarity-based cache backed by the shared EmbeddingStore.

    Tier 1 — sim >= high_threshold AND age < staleness_days
        -> confident HIT

    Tier 2 — low_threshold <= sim < high_threshold AND same service AND same level AND age < recency_window_seconds
        -> conditional HIT

    Tier 3 — otherwise
        -> MISS, call LLM
    """

    def __init__(
        self,
        store: EmbeddingStore,
        embedder: EmbedderProtocol,
        config: CacheConfig,
    ):
        self._store = store
        self._embedder = embedder
        self._high = config.high_threshold
        self._low = config.low_threshold
        self._recency_window = timedelta(seconds=config.recency_window_seconds)
        self._staleness = timedelta(days=config.staleness_days)
        self._hits = 0
        self._misses = 0

    def get(self, entry: LogEntry) -> Optional[tuple[str, str]]:
        """Return (root_cause, mitigation) if a similar entry is already cached."""
        emb = self._embedder.encode(entry.message)
        result = self._store.best_match(emb)

        if result is None or result.metadata.get("root_cause") is None:
            self._misses += 1
            return None

        sim = result.similarity

        if sim >= self._high and not self._is_stale(result):
            return self._hit(result)

        if self._low <= sim < self._high and self._tier2_pass(entry, result):
            return self._hit(result)

        self._misses += 1
        return None

    def set(self, entry: LogEntry, root_cause: str, mitigation: str) -> None:
        """
        Persist LLM results for an entry.

        If AnomalyDetector already added a placeholder for this message,
        update it in place. Otherwise insert a new entry.
        """
        emb = self._embedder.encode(entry.message)
        result = self._store.best_match(emb)
        now = _now()

        if result is not None and result.similarity >= self._low:
            self._store.update_metadata(result.index, {
                "root_cause": root_cause,
                "mitigation": mitigation,
                "service": entry.service,
                "level": entry.level.value,
                "last_seen": now,
            })
        else:
            self._store.add(emb, {
                "message": entry.message,
                "service": entry.service,
                "level": entry.level.value,
                "root_cause": root_cause,
                "mitigation": mitigation,
                "first_seen": now,
                "last_seen": now,
                "hit_count": 0,
            })

    def top_recurring(self, k: int = 5) -> list[dict]:
        """Return the k most-hit cache entries, highest first."""
        entries = [
            m for m in self._store.iter_metadata()
            if m.get("hit_count", 0) > 0 and m.get("root_cause") is not None
        ]
        return sorted(entries, key=lambda m: m["hit_count"], reverse=True)[:k]

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / total, 4) if total else 0.0

    def _hit(self, result: SearchResult) -> tuple[str, str]:
        self._hits += 1
        self._store.update_metadata(result.index, {
            "last_seen": _now(),
            "hit_count": result.metadata.get("hit_count", 0) + 1,
        })
        return result.metadata["root_cause"], result.metadata["mitigation"]

    def _tier2_pass(self, entry: LogEntry, result: SearchResult) -> bool:
        """Tier 2: same service, same level, and seen within recency window."""
        meta = result.metadata
        if meta.get("service") != entry.service:
            return False
        if meta.get("level") != entry.level.value:
            return False
        last_seen_str = meta.get("last_seen")
        if not last_seen_str:
            return False
        last_seen = datetime.fromisoformat(last_seen_str)
        return (datetime.now(tz=timezone.utc) - last_seen) <= self._recency_window

    def _is_stale(self, result: SearchResult) -> bool:
        """True if the entry hasn't been seen for longer than staleness_days."""
        last_seen_str = result.metadata.get("last_seen")
        if not last_seen_str:
            return False
        last_seen = datetime.fromisoformat(last_seen_str)
        return (datetime.now(tz=timezone.utc) - last_seen) > self._staleness


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
