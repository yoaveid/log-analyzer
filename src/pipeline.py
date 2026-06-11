from pathlib import Path

from src.ingestion.parser import parse_file
from src.models.log_entry import EnrichedLogEntry
from src.enrichment.llm_client import LLMClient
from src.cache.cache import AnalysisCache
from src.analyzer.anomaly import AnomalyDetector
from src.analyzer.stats import CRITICAL_LEVELS, LogStats
from src.output.reporter import write_report
from src.store.embedding_store import EmbeddingStore
from src.store.embedder import Embedder
from src.store.normalizer import LogNormalizer


def run(log_path: Path, output_path: Path) -> None:
    """Orchestrate ingestion -> anomaly detection -> enrichment -> reporting."""
    normalizer = LogNormalizer()
    knowledge_store = EmbeddingStore()   # all log entries — feeds novelty detection
    llm_store = EmbeddingStore()         # CRITICAL/ERROR with LLM results — feeds cache
    embedder = Embedder(normalizer=normalizer)
    stats = LogStats()
    detector = AnomalyDetector(store=knowledge_store, embedder=embedder)
    cache = AnalysisCache(store=llm_store, embedder=embedder)
    llm = LLMClient()

    enriched_errors = []
    all_anomalies = []

    for entry in parse_file(log_path):
        if entry is None:
            stats.record_malformed()
            continue

        stats.update(entry)
        cluster_id = normalizer.parse(entry.message).cluster_id
        emb = embedder.encode(entry.message)
        knowledge_store.add(emb, {"message": entry.message, "level": entry.level.value})
        all_anomalies += detector.process_entry(entry, cluster_id)

        if entry.level in CRITICAL_LEVELS:
            cached = cache.get(entry)
            if cached:
                root_cause, mitigation = cached
            else:
                root_cause, mitigation = llm.analyze(entry)
                cache.set(entry, root_cause, mitigation)

            enriched_errors.append(EnrichedLogEntry(
                **entry.model_dump(),
                root_cause=root_cause,
                mitigation=mitigation,
                cache_hit=cached is not None,
            ))

    write_report(
        entries=enriched_errors,
        stats=stats,
        anomalies=all_anomalies,
        cache_hit_rate=cache.hit_rate,
        llm_stats=llm.to_dict(),
        top_recurring=cache.top_recurring(k=5),
        output_path=output_path,
    )