import json

import pytest

from src.analyzer.anomaly import AnomalyDetector
from src.cache.cache import AnalysisCache
from src.config.settings import AnomalyConfig, BurstConfig, CacheConfig, SpikeConfig
from src.pipeline import PipelineServices, run
from src.store.embedding_store import EmbeddingStore
from src.store.normalizer import LogNormalizer
from tests.mocks.mock_embedder import MockEmbedder
from tests.mocks.mock_llm import MockLLMClient


def _make_services(llm: MockLLMClient) -> PipelineServices:
    embedder = MockEmbedder()
    normalizer = LogNormalizer()
    knowledge_store = EmbeddingStore()
    llm_store = EmbeddingStore()
    anomaly_cfg = AnomalyConfig(
        novelty_threshold=0.65,
        min_store_size=999,  # disable novelty in pipeline tests
        spike=SpikeConfig(bucket_seconds=60, z_threshold=2.0, min_history=3, min_spike_count=5),
        burst=BurstConfig(bucket_seconds=60, threshold_per_bucket=5, consecutive_buckets=3),
    )
    cache_cfg = CacheConfig(
        high_threshold=0.9, low_threshold=0.8, recency_window_seconds=300, staleness_days=30
    )
    return PipelineServices(
        llm=llm,
        embedder=embedder,
        knowledge_store=knowledge_store,
        llm_store=llm_store,
        normalizer=normalizer,
        detector=AnomalyDetector(store=knowledge_store, config=anomaly_cfg),
        cache=AnalysisCache(store=llm_store, embedder=embedder, config=cache_cfg),
    )


# ─── core flow ────────────────────────────────────────────────────────────────

def test_report_file_created(tmp_path):
    log = tmp_path / "test.log"
    log.write_text("2024-01-15T10:00:00Z ERROR svc JWT validation failed\n")
    output = tmp_path / "report.json"
    run(log, output, services=_make_services(MockLLMClient()))
    assert output.exists()


def test_report_has_expected_top_level_keys(tmp_path):
    log = tmp_path / "test.log"
    log.write_text("2024-01-15T10:00:00Z ERROR svc msg\n")
    output = tmp_path / "report.json"
    run(log, output, services=_make_services(MockLLMClient()))
    report = json.loads(output.read_text())
    assert set(report.keys()) >= {"summary", "enriched_errors", "anomalies", "llm_health"}


def test_valid_lines_counted_in_summary(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(
        "2024-01-15T10:00:00Z ERROR svc JWT validation failed\n"
        "2024-01-15T10:00:01Z INFO svc Request received\n"
        "2024-01-15T10:00:02Z CRITICAL svc Out of memory\n"
    )
    output = tmp_path / "report.json"
    run(log, output, services=_make_services(MockLLMClient()))
    report = json.loads(output.read_text())
    assert report["summary"]["total_parsed"] == 3
    assert report["summary"]["error_count"] == 2  # ERROR + CRITICAL


def test_malformed_lines_counted_not_crashing(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(
        "2024-01-15T10:00:00Z ERROR svc valid line\n"
        "this is not a log line\n"
        "also garbage\n"
        "2024-01-15T10:00:01Z INFO svc another valid\n"
    )
    output = tmp_path / "report.json"
    run(log, output, services=_make_services(MockLLMClient()))
    report = json.loads(output.read_text())
    assert report["summary"]["total_parsed"] == 2
    assert report["summary"]["malformed_lines"] == 2


# ─── enrichment routing ───────────────────────────────────────────────────────

def test_llm_called_only_for_error_and_critical(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(
        "2024-01-15T10:00:00Z DEBUG svc debug msg\n"
        "2024-01-15T10:00:01Z INFO svc info msg\n"
        "2024-01-15T10:00:02Z WARNING svc warn msg\n"
        "2024-01-15T10:00:03Z ERROR svc error msg\n"
        "2024-01-15T10:00:04Z CRITICAL svc critical msg\n"
    )
    output = tmp_path / "report.json"
    llm = MockLLMClient()
    run(log, output, services=_make_services(llm))
    assert llm.total_requests == 2


def test_enriched_errors_only_contain_error_and_critical(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(
        "2024-01-15T10:00:00Z INFO svc info msg\n"
        "2024-01-15T10:00:01Z ERROR svc error msg\n"
        "2024-01-15T10:00:02Z CRITICAL svc critical msg\n"
    )
    output = tmp_path / "report.json"
    run(log, output, services=_make_services(MockLLMClient()))
    report = json.loads(output.read_text())
    levels = {e["level"] for e in report["enriched_errors"]}
    assert levels <= {"ERROR", "CRITICAL"}


# ─── cache behavior ───────────────────────────────────────────────────────────

def test_second_identical_error_is_cache_hit(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(
        "2024-01-15T10:00:00Z ERROR svc Connection refused\n"
        "2024-01-15T10:00:01Z ERROR svc Connection refused\n"
    )
    output = tmp_path / "report.json"
    llm = MockLLMClient()
    run(log, output, services=_make_services(llm))
    report = json.loads(output.read_text())
    assert llm.total_requests == 1
    assert report["enriched_errors"][1]["cache_hit"] is True


def test_different_errors_each_call_llm(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(
        "2024-01-15T10:00:00Z ERROR svc Connection refused\n"
        "2024-01-15T10:00:01Z ERROR svc Out of memory: process killed\n"
    )
    output = tmp_path / "report.json"
    llm = MockLLMClient()
    run(log, output, services=_make_services(llm))
    assert llm.total_requests == 2
