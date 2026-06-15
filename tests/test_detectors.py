from collections import deque

import numpy as np

from src.analyzer.burst_detector import BurstDetector
from src.analyzer.novel_detector import NoveltyDetector
from src.analyzer.spike_detector import SpikeDetector
from src.config.settings import BurstConfig, SpikeConfig
from src.models.log_entry import LogLevel
from src.store.embedding_store import EmbeddingStore
from tests.conftest import make_entry, unit_vec

DIM = 3


def spike_cfg(**overrides) -> SpikeConfig:
    defaults = dict(bucket_seconds=60, z_threshold=2.0, min_history=3, min_spike_count=5)
    return SpikeConfig(**{**defaults, **overrides})


def burst_cfg(**overrides) -> BurstConfig:
    defaults = dict(bucket_seconds=60, threshold_per_bucket=5, consecutive_buckets=3)
    return BurstConfig(**{**defaults, **overrides})


# ═══════════════════════════════════════════
# SpikeDetector
# ═══════════════════════════════════════════

class TestSpikeDetector:
    def test_no_fire_during_warmup(self):
        detector = SpikeDetector(spike_cfg(min_history=3, min_spike_count=1))
        # 2 completed buckets with huge counts — still below min_history=3
        detector._history[0] = deque([100, 100], maxlen=100)
        detector._current[0] = (0.0, 100)
        result = detector.check(make_entry(ts=60), 0)
        assert result is None

    def test_fires_when_z_score_exceeded(self):
        detector = SpikeDetector(spike_cfg(z_threshold=2.0, min_spike_count=5))
        # mean=4, std≈0.89 → z(20)≈17.9
        detector._history[0] = deque([3, 5, 4, 3, 5], maxlen=100)
        detector._current[0] = (0.0, 20)
        result = detector.check(make_entry(ts=60), 0)
        assert result is not None
        assert result.kind == "spike"
        assert "cluster 0" in result.description

    def test_no_fire_when_z_below_threshold(self):
        detector = SpikeDetector(spike_cfg(z_threshold=3.0, min_spike_count=5))
        # mean=4, std≈0.89 → z(6)≈2.24 < threshold=3.0
        detector._history[0] = deque([3, 5, 4, 3, 5], maxlen=100)
        detector._current[0] = (0.0, 6)
        result = detector.check(make_entry(ts=60), 0)
        assert result is None

    def test_min_spike_count_blocks_low_volume(self):
        # Huge z-score but count=3 < min_spike_count=5 → no fire
        detector = SpikeDetector(spike_cfg(min_history=1, z_threshold=0.1, min_spike_count=5))
        detector._history[0] = deque([0, 0, 0, 0, 0], maxlen=100)
        detector._current[0] = (0.0, 3)
        result = detector.check(make_entry(ts=60), 0)
        assert result is None

    def test_std_zero_fires_when_count_exceeds_mean(self):
        # Flat baseline [10,10,10] → std=0, mean=10; count=15 > mean → spike
        detector = SpikeDetector(spike_cfg(min_history=3, min_spike_count=5))
        detector._history[0] = deque([10, 10, 10], maxlen=100)
        detector._current[0] = (0.0, 15)
        result = detector.check(make_entry(ts=60), 0)
        assert result is not None
        assert result.kind == "spike"

    def test_std_zero_no_fire_when_count_equals_mean(self):
        # count == mean → not strictly greater → no spike
        detector = SpikeDetector(spike_cfg(min_history=3, min_spike_count=5))
        detector._history[0] = deque([10, 10, 10], maxlen=100)
        detector._current[0] = (0.0, 10)
        result = detector.check(make_entry(ts=60), 0)
        assert result is None

    def test_gaps_filled_with_zeros_in_history(self):
        # Bucket at t=0 has count=5, then silence until t=240 (3 skipped buckets)
        detector = SpikeDetector(spike_cfg(min_history=1, min_spike_count=1))
        detector._history[0] = deque([5], maxlen=100)
        detector._current[0] = (0.0, 5)
        detector.check(make_entry(ts=240), 0)  # triggers rollover
        # After: commit cur_count=5, fill 3 gaps → [..., 5, 0, 0, 0]
        history = list(detector._history[0])
        assert history[-3:] == [0, 0, 0]

    def test_clusters_tracked_independently(self):
        detector = SpikeDetector(spike_cfg(min_spike_count=5))
        detector._history[0] = deque([3, 5, 4, 3, 5], maxlen=100)
        detector._history[1] = deque([3, 5, 4, 3, 5], maxlen=100)
        detector._current[0] = (0.0, 20)  # spike
        detector._current[1] = (0.0, 5)   # normal

        result_0 = detector.check(make_entry(ts=60), 0)
        result_1 = detector.check(make_entry(ts=60), 1)

        assert result_0 is not None
        assert result_1 is None

    def test_first_entry_for_cluster_never_fires(self):
        detector = SpikeDetector(spike_cfg())
        result = detector.check(make_entry(ts=0), cluster_id=99)
        assert result is None


# ═══════════════════════════════════════════
# BurstDetector
# ═══════════════════════════════════════════

class TestBurstDetector:
    def test_no_fire_on_first_event(self):
        detector = BurstDetector(burst_cfg())
        result = detector.check(make_entry(ts=0, level=LogLevel.CRITICAL), 0)
        assert result is None

    def test_no_fire_at_threshold_without_prior_history(self):
        # 5 events in bucket 0, no prior elevated buckets
        detector = BurstDetector(burst_cfg(threshold_per_bucket=5, consecutive_buckets=3))
        detector._current[0] = (0.0, 4)
        result = detector.check(make_entry(ts=5), 0)  # 5th event
        assert result is None  # preceding=[], len=0 < required-1=2

    def test_fires_when_sustained_across_required_buckets(self):
        detector = BurstDetector(burst_cfg(threshold_per_bucket=5, consecutive_buckets=3))
        detector._buckets[0] = deque([5, 6], maxlen=3)   # 2 elevated buckets
        detector._current[0] = (120.0, 4)
        result = detector.check(make_entry(ts=125), 0)   # 5th event
        assert result is not None
        assert result.kind == "burst"

    def test_no_fire_when_one_preceding_bucket_below_threshold(self):
        detector = BurstDetector(burst_cfg(threshold_per_bucket=5, consecutive_buckets=3))
        detector._buckets[0] = deque([5, 3], maxlen=3)   # 3 < threshold=5
        detector._current[0] = (120.0, 4)
        result = detector.check(make_entry(ts=125), 0)
        assert result is None

    def test_gap_zero_breaks_burst_streak(self):
        detector = BurstDetector(burst_cfg(threshold_per_bucket=5, consecutive_buckets=3))
        # Silent gap (0) between two elevated buckets
        detector._buckets[0] = deque([5, 0, 5], maxlen=3)
        detector._current[0] = (180.0, 4)
        result = detector.check(make_entry(ts=185), 0)
        assert result is None  # preceding[-2:]=[0,5], 0 < threshold → no burst

    def test_fires_exactly_at_threshold_not_before(self):
        detector = BurstDetector(burst_cfg(threshold_per_bucket=5, consecutive_buckets=3))
        detector._buckets[0] = deque([5, 5], maxlen=3)
        detector._current[0] = (120.0, 3)

        no_fire = detector.check(make_entry(ts=124), 0)   # 4th event
        assert no_fire is None

        fires = detector.check(make_entry(ts=125), 0)     # 5th event == threshold
        assert fires is not None
        assert fires.kind == "burst"

    def test_new_bucket_commits_without_firing(self):
        # Rolling into a new bucket should never directly fire
        detector = BurstDetector(burst_cfg(threshold_per_bucket=5, consecutive_buckets=3))
        detector._buckets[0] = deque([5, 5], maxlen=3)
        detector._current[0] = (0.0, 10)  # previous bucket count=10 ≥ threshold
        result = detector.check(make_entry(ts=60), 0)    # triggers rollover
        assert result is None


# ═══════════════════════════════════════════
# NoveltyDetector
# ═══════════════════════════════════════════

class TestNoveltyDetector:
    def test_silent_during_warmup(self):
        store = EmbeddingStore(dim=DIM)
        detector = NoveltyDetector(store=store, threshold=0.65, min_store_size=5)
        emb = unit_vec(DIM, 0)
        for _ in range(5):
            store.add(emb, {})
        # size==5 == min_store_size → still silent (condition is <=)
        result = detector.check(make_entry(), emb)
        assert result is None

    def test_fires_for_novel_pattern(self):
        store = EmbeddingStore(dim=DIM)
        detector = NoveltyDetector(store=store, threshold=0.65, min_store_size=0)
        store.add(unit_vec(DIM, 0), {})   # known pattern in X direction
        novel_emb = unit_vec(DIM, 1)      # Y direction — cosine sim=0 with X
        result = detector.check(make_entry(), novel_emb)
        assert result is not None
        assert result.kind == "novel_log"

    def test_no_fire_for_known_pattern(self):
        store = EmbeddingStore(dim=DIM)
        detector = NoveltyDetector(store=store, threshold=0.65, min_store_size=0)
        emb = unit_vec(DIM, 0)
        store.add(emb, {})
        result = detector.check(make_entry(), emb)  # exact match → sim=1.0
        assert result is None
