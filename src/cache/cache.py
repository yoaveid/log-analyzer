from typing import Optional

from src.store.embedding_store import EmbeddingStore
from src.store.embedder import Embedder


class AnalysisCache:
    """
    Similarity-based cache backed by the shared EmbeddingStore.

    get() returns a cached LLM result when a semantically similar message
    has already been analyzed. set() writes the LLM result back — either
    updating an existing store entry (added earlier by AnomalyDetector)
    or inserting a new one.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        embedder: Embedder,
        similarity_threshold: float = 0.85,
    ):
        self._store = store
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._hits = 0
        self._misses = 0

    def get(self, message: str) -> Optional[tuple[str, str]]:
        """Return (root_cause, mitigation) if a similar message is already cached."""
        emb = self._embedder.encode(message)
        result = self._store.best_match(emb)

        if (
            result is not None
            and result.similarity >= self._threshold
            and result.metadata.get("root_cause") is not None
        ):
            self._hits += 1
            return result.metadata["root_cause"], result.metadata["mitigation"]

        self._misses += 1
        return None

    def set(self, message: str, root_cause: str, mitigation: str) -> None:
        """
        Persist LLM results for a message.

        If AnomalyDetector already added this message to the store (as a novel
        error), we update that entry rather than creating a duplicate vector.
        """
        emb = self._embedder.encode(message)
        result = self._store.best_match(emb)

        if result is not None and result.similarity >= self._threshold:
            self._store.update_metadata(result.index, {
                "root_cause": root_cause,
                "mitigation": mitigation,
            })
        else:
            self._store.add(emb, {
                "message": message,
                "root_cause": root_cause,
                "mitigation": mitigation,
            })

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return round(self._hits / total, 4) if total > 0 else 0.0
