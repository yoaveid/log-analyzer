import json
from pathlib import Path
from typing import Sequence

import structlog

from src.analyzer.anomaly import Anomaly
from src.analyzer.stats import LogStats
from src.models.log_entry import EnrichedLogEntry

logger = structlog.get_logger(__name__)


def write_report(
    entries: Sequence[EnrichedLogEntry],
    stats: LogStats,
    anomalies: Sequence[Anomaly],
    cache_hit_rate: float,
    llm_stats: dict,
    top_recurring: list[dict],
    output_path: Path,
) -> None:
    """Write a structured JSON summary to output_path."""
    report = {
        "summary": {
            **stats.to_dict(),
            "cache_hit_rate": cache_hit_rate,
        },
        "llm_health": llm_stats,
        "top_recurring_errors": [
            {
                "message": m.get("message"),
                "service": m.get("service"),
                "hit_count": m.get("hit_count"),
                "first_seen": m.get("first_seen"),
                "last_seen": m.get("last_seen"),
                "root_cause": m.get("root_cause"),
                "mitigation": m.get("mitigation"),
            }
            for m in top_recurring
        ],
        "anomalies": [
            {
                "kind": a.kind,
                "description": a.description,
                "affected_services": list({e.service for e in a.entries}),
            }
            for a in anomalies
        ],
        "enriched_errors": [e.model_dump(mode="json") for e in entries],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("report_written", path=str(output_path), enriched_errors=len(entries), anomalies=len(anomalies))
