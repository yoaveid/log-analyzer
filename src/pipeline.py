from dataclasses import dataclass, field
from pathlib import Path

import structlog

from src.ingestion.parser import parse_file
from src.models.log_entry import EnrichedLogEntry
from src.enrichment.llm_client import LLMClient, LLMClientProtocol
from src.cache.cache import AnalysisCache
from src.analyzer.anomaly import AnomalyDetector
from src.analyzer.stats import CRITICAL_LEVELS, LogStats
from src.output.reporter import write_report
from src.store.embedding_store import EmbeddingStore
from src.store.embedder import Embedder, EmbedderProtocol
from src.store.normalizer import LogNormalizer
from src.config.settings import AppConfig, load_config

logger = structlog.get_logger(__name__)


@dataclass
class PipelineServices:
    llm: LLMClientProtocol
    embedder: EmbedderProtocol
    knowledge_store: EmbeddingStore
    llm_store: EmbeddingStore
    normalizer: LogNormalizer
    detector: AnomalyDetector
    cache: AnalysisCache
    stats: LogStats = field(default_factory=LogStats)


def build_services(config: AppConfig) -> PipelineServices:
    normalizer = LogNormalizer()
    knowledge_store = EmbeddingStore()
    llm_store = EmbeddingStore()
    embedder = Embedder(config=config.embedder, normalizer=normalizer)
    return PipelineServices(
        llm=LLMClient(config=config.llm),
        embedder=embedder,
        knowledge_store=knowledge_store,
        llm_store=llm_store,
        normalizer=normalizer,
        detector=AnomalyDetector(store=knowledge_store, config=config.anomaly),
        cache=AnalysisCache(store=llm_store, embedder=embedder, config=config.cache),
    )


def run(
    log_path: Path,
    output_path: Path,
    services: PipelineServices | None = None,
    config: AppConfig | None = None,
) -> None:
    """Orchestrate ingestion -> anomaly detection -> enrichment -> reporting."""
    s = services or build_services(config or load_config())
    logger.info("pipeline_started", log_path=str(log_path))

    enriched_errors: list[EnrichedLogEntry] = []
    all_anomalies = []

    for entry in parse_file(log_path):
        if entry is None:
            s.stats.record_malformed()
            continue

        s.stats.update(entry)
        cluster_id = s.normalizer.parse(entry.message).cluster_id
        emb = s.embedder.encode(entry.message)
        all_anomalies += s.detector.process_entry(entry, emb, cluster_id)
        s.knowledge_store.add_if_novel(emb, {"message": entry.message, "level": entry.level.value})

        if entry.level in CRITICAL_LEVELS:
            cached = s.cache.get(entry)
            if cached:
                root_cause, mitigation = cached
            else:
                root_cause, mitigation = s.llm.analyze(entry)
                s.cache.set(entry, root_cause, mitigation)

            enriched_errors.append(EnrichedLogEntry(
                **entry.model_dump(),
                root_cause=root_cause,
                mitigation=mitigation,
                cache_hit=cached is not None,
            ))

    logger.info(
        "pipeline_completed",
        total_parsed=s.stats.total_parsed,
        error_count=s.stats.error_count,
        anomaly_count=len(all_anomalies),
        cache_hit_rate=s.cache.hit_rate,
    )
    write_report(
        entries=enriched_errors,
        stats=s.stats,
        anomalies=all_anomalies,
        cache_hit_rate=s.cache.hit_rate,
        llm_stats=s.llm.to_dict(),
        top_recurring=s.cache.top_recurring(k=5),
        output_path=output_path,
    )
