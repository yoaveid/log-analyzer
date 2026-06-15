from datetime import datetime, timedelta, timezone

import numpy as np

from src.cache.cache import AnalysisCache
from src.config.settings import CacheConfig
from src.models.log_entry import LogLevel
from src.store.embedding_store import EmbeddingStore
from tests.conftest import make_entry

DIM = 3


def _norm(*components: float) -> np.ndarray:
    v = np.array(components, dtype=np.float32)
    return v / float(np.linalg.norm(v))


# Vectors with controlled cosine similarity to E_BASE
E_BASE      = _norm(1.0,  0.0, 0.0)   # stored entry
E_HIGH_SIM  = _norm(0.95, 0.31, 0.0)  # sim ≈ 0.95 → tier 1 (≥ high=0.9)
E_MED_SIM   = _norm(0.85, 0.53, 0.0)  # sim ≈ 0.85 → tier 2 (0.8 ≤ sim < 0.9)
E_DIFF      = _norm(0.0,  0.0, 1.0)   # sim = 0.0  → miss  (< low=0.8)


class _MockEmbedder:
    def __init__(self, mapping: dict[str, np.ndarray]):
        self._map = mapping

    def encode(self, text: str) -> np.ndarray:
        return self._map.get(text, np.zeros(DIM, dtype=np.float32))


def _make_cache(mapping: dict[str, np.ndarray], **overrides) -> AnalysisCache:
    cfg = CacheConfig(
        **{"high_threshold": 0.9, "low_threshold": 0.8,
           "recency_window_seconds": 300, "staleness_days": 30, **overrides}
    )
    return AnalysisCache(
        store=EmbeddingStore(dim=DIM),
        embedder=_MockEmbedder(mapping),
        config=cfg,
    )


# ─── miss / hit ───────────────────────────────────────────────────────────────

def test_miss_on_empty_store():
    cache = _make_cache({"q": E_BASE})
    assert cache.get(make_entry(message="q")) is None


def test_tier1_hit_above_high_threshold():
    cache = _make_cache({"stored": E_BASE, "query": E_HIGH_SIM})
    cache.set(make_entry(message="stored"), "Root A", "Fix A")
    result = cache.get(make_entry(message="query"))
    assert result == ("Root A", "Fix A")


def test_tier1_miss_when_stale():
    cache = _make_cache({"stored": E_BASE, "query": E_HIGH_SIM})
    cache.set(make_entry(message="stored"), "Root A", "Fix A")
    stale = (datetime.now(tz=timezone.utc) - timedelta(days=35)).isoformat()
    cache._store.update_metadata(0, {"last_seen": stale})
    assert cache.get(make_entry(message="query")) is None


def test_tier2_hit_same_service_and_level_within_recency():
    cache = _make_cache({"stored": E_BASE, "query": E_MED_SIM})
    cache.set(make_entry(message="stored", service="svc", level=LogLevel.ERROR), "Root B", "Fix B")
    result = cache.get(make_entry(message="query", service="svc", level=LogLevel.ERROR))
    assert result == ("Root B", "Fix B")


def test_tier2_miss_different_service():
    cache = _make_cache({"stored": E_BASE, "query": E_MED_SIM})
    cache.set(make_entry(message="stored", service="svc-a"), "Root B", "Fix B")
    assert cache.get(make_entry(message="query", service="svc-b")) is None


def test_tier2_miss_different_level():
    cache = _make_cache({"stored": E_BASE, "query": E_MED_SIM})
    cache.set(make_entry(message="stored", service="svc", level=LogLevel.ERROR), "Root B", "Fix B")
    assert cache.get(make_entry(message="query", service="svc", level=LogLevel.CRITICAL)) is None


def test_tier2_miss_outside_recency_window():
    cache = _make_cache({"stored": E_BASE, "query": E_MED_SIM}, recency_window_seconds=1)
    cache.set(make_entry(message="stored", service="svc", level=LogLevel.ERROR), "Root B", "Fix B")
    old = (datetime.now(tz=timezone.utc) - timedelta(seconds=10)).isoformat()
    cache._store.update_metadata(0, {"last_seen": old})
    assert cache.get(make_entry(message="query", service="svc", level=LogLevel.ERROR)) is None


def test_miss_below_low_threshold():
    cache = _make_cache({"stored": E_BASE, "query": E_DIFF})
    cache.set(make_entry(message="stored"), "Root C", "Fix C")
    assert cache.get(make_entry(message="query")) is None


# ─── hit_rate ─────────────────────────────────────────────────────────────────

def test_hit_rate_zero_when_no_activity():
    assert _make_cache({}).hit_rate == 0.0


def test_hit_rate_calculated_correctly():
    cache = _make_cache({"stored": E_BASE, "q": E_HIGH_SIM, "miss": E_DIFF})
    cache.set(make_entry(message="stored"), "R", "M")
    cache.get(make_entry(message="q"))     # hit
    cache.get(make_entry(message="miss"))  # miss
    cache.get(make_entry(message="q"))     # hit
    assert cache.hit_rate == round(2 / 3, 4)


# ─── top_recurring ────────────────────────────────────────────────────────────

def test_top_recurring_sorted_by_hit_count():
    E_Y      = _norm(0.0, 1.0, 0.0)
    E_NEAR_Y = _norm(0.31, 0.95, 0.0)  # sim ≈ 0.95 with E_Y

    cache = _make_cache({
        "msg_a": E_BASE, "qa": E_HIGH_SIM,
        "msg_b": E_Y,    "qb": E_NEAR_Y,
    })
    cache.set(make_entry(message="msg_a"), "Root A", "Fix A")
    cache.set(make_entry(message="msg_b"), "Root B", "Fix B")

    for _ in range(3):
        cache.get(make_entry(message="qa"))
    cache.get(make_entry(message="qb"))

    top = cache.top_recurring(k=2)
    assert len(top) == 2
    assert top[0]["root_cause"] == "Root A"
    assert top[0]["hit_count"] == 3
    assert top[1]["root_cause"] == "Root B"
    assert top[1]["hit_count"] == 1


def test_top_recurring_excludes_zero_hit_entries():
    cache = _make_cache({"stored": E_BASE})
    cache.set(make_entry(message="stored"), "Root", "Fix")
    assert cache.top_recurring() == []
