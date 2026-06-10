from pathlib import Path

from src.ingestion.parser import parse_file
from src.models.log_entry import EnrichedLogEntry
from src.enrichment.llm_client import LLMClient
from src.cache.cache import AnalysisCache
from src.analyzer.anomaly import AnomalyDetector, CRITICAL_LEVELS
from src.analyzer.stats import LogStats
from src.output.reporter import write_report
from src.store.embedding_store import EmbeddingStore
from src.store.embedder import Embedder


def run(log_path: Path, output_path: Path) -> None:
    """Orchestrate ingestion → enrichment → reporting."""
    store = EmbeddingStore()
    embedder = Embedder()
    stats = LogStats()
    detector = AnomalyDetector(store=store, embedder=embedder)
    cache = AnalysisCache(store=store, embedder=embedder)
    llm = LLMClient()

    enriched_errors = []
    all_anomalies = []

    for entry in parse_file(log_path):
        if entry is None:
            stats.record_malformed()
            continue

        stats.update(entry)
        all_anomalies += detector.process_entry(entry)

        if entry.level in CRITICAL_LEVELS:
            cached = cache.get(entry.message)
            if cached:
                root_cause, mitigation = cached
            else:
                root_cause, mitigation = llm.analyze(entry)
                cache.set(entry.message, root_cause, mitigation)

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
        output_path=output_path,
    )