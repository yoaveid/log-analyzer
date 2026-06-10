import json
from pathlib import Path
from typing import Sequence

from src.analyzer.anomaly import Anomaly
from src.analyzer.stats import LogStats
from src.models.log_entry import EnrichedLogEntry


def write_report(
    entries: Sequence[EnrichedLogEntry],
    stats: LogStats,
    anomalies: Sequence[Anomaly],
    cache_hit_rate: float,
    output_path: Path,
) -> None:
    """Write a structured JSON summary to output_path."""
    report = {
        "summary": {
            **stats.to_dict(),
            "cache_hit_rate": cache_hit_rate,
        },
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
