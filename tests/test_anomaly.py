from unittest.mock import MagicMock

from src.analyzer.anomaly import AnomalyDetector
from src.config.settings import AnomalyConfig, BurstConfig, SpikeConfig
from src.models.anomaly import Anomaly
from src.models.log_entry import LogLevel
from src.store.embedding_store import EmbeddingStore
from tests.conftest import make_entry, unit_vec

DIM = 3


def make_config() -> AnomalyConfig:
    return AnomalyConfig(
        novelty_threshold=0.65,
        min_store_size=1,
        spike=SpikeConfig(bucket_seconds=60, z_threshold=2.0, min_history=3, min_spike_count=5),
        burst=BurstConfig(bucket_seconds=60, threshold_per_bucket=5, consecutive_buckets=3),
    )


def make_detector() -> AnomalyDetector:
    return AnomalyDetector(store=EmbeddingStore(dim=DIM), config=make_config())


def test_spike_checked_for_every_log_level():
    detector = make_detector()
    detector._spike.check = MagicMock(return_value=None)
    detector._burst.check = MagicMock(return_value=None)
    detector._novel.check = MagicMock(return_value=None)

    for level in LogLevel:
        detector.process_entry(make_entry(level=level), unit_vec(DIM, 0), cluster_id=0)

    assert detector._spike.check.call_count == len(list(LogLevel))


def test_burst_not_called_for_debug_info_warning():
    detector = make_detector()
    detector._spike.check = MagicMock(return_value=None)
    detector._burst.check = MagicMock(return_value=None)
    detector._novel.check = MagicMock(return_value=None)

    for level in (LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING):
        detector.process_entry(make_entry(level=level), unit_vec(DIM, 0), cluster_id=0)

    detector._burst.check.assert_not_called()


def test_burst_called_for_error_and_critical():
    detector = make_detector()
    detector._spike.check = MagicMock(return_value=None)
    detector._burst.check = MagicMock(return_value=None)
    detector._novel.check = MagicMock(return_value=None)

    detector.process_entry(make_entry(level=LogLevel.ERROR), unit_vec(DIM, 0), 0)
    detector.process_entry(make_entry(level=LogLevel.CRITICAL), unit_vec(DIM, 0), 0)

    assert detector._burst.check.call_count == 2


def test_all_non_none_results_collected():
    detector = make_detector()
    spike_anomaly = Anomaly(kind="spike", description="s", entries=[])
    burst_anomaly = Anomaly(kind="burst", description="b", entries=[])

    detector._spike.check = MagicMock(return_value=spike_anomaly)
    detector._burst.check = MagicMock(return_value=burst_anomaly)
    detector._novel.check = MagicMock(return_value=None)

    results = detector.process_entry(make_entry(level=LogLevel.CRITICAL), unit_vec(DIM, 0), 0)
    assert len(results) == 2
    assert {r.kind for r in results} == {"spike", "burst"}


