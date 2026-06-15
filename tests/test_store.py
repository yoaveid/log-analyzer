import numpy as np

from src.store.embedding_store import EmbeddingStore

DIM = 3


def _norm(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / float(np.linalg.norm(v))


E_X = _norm(1.0, 0.0, 0.0)
E_Y = _norm(0.0, 1.0, 0.0)
E_Z = _norm(0.0, 0.0, 1.0)
E_NEAR_X = _norm(0.95, 0.31, 0.0)  # cosine sim ≈ 0.95 with E_X


def test_add_and_best_match_returns_entry():
    store = EmbeddingStore(dim=DIM)
    store.add(E_X, {"label": "x"})
    result = store.best_match(E_X)
    assert result is not None
    assert result.similarity > 0.99
    assert result.metadata["label"] == "x"


def test_empty_store_returns_none():
    assert EmbeddingStore(dim=DIM).best_match(E_X) is None


def test_best_match_returns_closest_vector():
    store = EmbeddingStore(dim=DIM)
    store.add(E_X, {"label": "x"})
    store.add(E_Y, {"label": "y"})
    result = store.best_match(E_NEAR_X)
    assert result is not None
    assert result.metadata["label"] == "x"


def test_size_reflects_number_of_entries():
    store = EmbeddingStore(dim=DIM)
    assert store.size == 0
    store.add(E_X, {})
    assert store.size == 1
    store.add(E_Y, {})
    assert store.size == 2


def test_add_if_novel_skips_near_duplicate():
    store = EmbeddingStore(dim=DIM)
    store.add(E_X, {"label": "original"})
    added = store.add_if_novel(E_NEAR_X, {"label": "dup"}, dedup_threshold=0.9)
    assert added is False
    assert store.size == 1


def test_add_if_novel_adds_distinct_vector():
    store = EmbeddingStore(dim=DIM)
    store.add(E_X, {"label": "x"})
    added = store.add_if_novel(E_Y, {"label": "y"}, dedup_threshold=0.9)
    assert added is True
    assert store.size == 2


def test_add_if_novel_adds_when_store_empty():
    store = EmbeddingStore(dim=DIM)
    added = store.add_if_novel(E_X, {"label": "first"}, dedup_threshold=0.9)
    assert added is True
    assert store.size == 1


def test_update_metadata_merges_keys():
    store = EmbeddingStore(dim=DIM)
    idx = store.add(E_X, {"status": "new", "count": 0})
    store.update_metadata(idx, {"status": "updated", "count": 5})
    meta = store.iter_metadata()[idx]
    assert meta["status"] == "updated"
    assert meta["count"] == 5


def test_update_metadata_preserves_unmodified_keys():
    store = EmbeddingStore(dim=DIM)
    idx = store.add(E_X, {"a": 1, "b": 2})
    store.update_metadata(idx, {"b": 99})
    meta = store.iter_metadata()[idx]
    assert meta["a"] == 1
    assert meta["b"] == 99


