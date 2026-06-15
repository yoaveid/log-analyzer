from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np


_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension


@dataclass
class SearchResult:
    index: int
    similarity: float       # cosine similarity, 0-1 (higher = more similar)
    metadata: dict[str, Any]


class EmbeddingStore:
    """
    In-memory FAISS index with arbitrary per-vector metadata.
    """

    def __init__(self, dim: int = _EMBED_DIM):
        self._index = faiss.IndexFlatIP(dim)  # inner product = cosine sim on normalized vecs
        self._metadata: list[dict[str, Any]] = []

    def add(self, embedding: np.ndarray, metadata: dict[str, Any]) -> int:
        """Store a normalized embedding + metadata. Returns its index."""
        vec = embedding.astype(np.float32).reshape(1, -1)
        self._index.add(vec)
        self._metadata.append(metadata)
        return len(self._metadata) - 1

    def search(self, embedding: np.ndarray, k: int = 1) -> list[SearchResult]:
        """Return the k most similar stored vectors, closest first."""
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        vec = embedding.astype(np.float32).reshape(1, -1)
        similarities, indices = self._index.search(vec, k)
        return [
            SearchResult(
                index=int(idx),
                similarity=float(sim),
                metadata=self._metadata[int(idx)],
            )
            for sim, idx in zip(similarities[0], indices[0])
            if idx != -1
        ]

    def best_match(self, embedding: np.ndarray) -> Optional[SearchResult]:
        """Convenience wrapper — returns the single closest match or None."""
        results = self.search(embedding, k=1)
        return results[0] if results else None

    def add_if_novel(
        self, embedding: np.ndarray, metadata: dict[str, Any], dedup_threshold: float = 0.95
    ) -> bool:
        """Add only if no existing vector exceeds dedup_threshold similarity. Returns True if added."""
        match = self.best_match(embedding)
        if match is not None and match.similarity >= dedup_threshold:
            return False
        self.add(embedding, metadata)
        return True

    def update_metadata(self, index: int, updates: dict[str, Any]) -> None:
        """Merge updates into an existing entry's metadata (e.g. add LLM results)."""
        self._metadata[index].update(updates)

    def iter_metadata(self) -> list[dict[str, Any]]:
        """Return a shallow copy of all stored metadata dicts."""
        return list(self._metadata)

    @property
    def size(self) -> int:
        return self._index.ntotal