# Log Analyzer

A DevOps CLI tool that analyzes server log files, detects anomalies, and enriches errors with AI-generated root cause analysis using a local LLM.

**Features:**

- Detects spikes, bursts, and novel error patterns automatically
- Enriches every ERROR and CRITICAL entry with root cause and mitigation
- Semantic cache avoids redundant LLM calls for repeated errors
- Outputs a structured JSON report ready for dashboards or alerting pipelines

---

## How It Works

```
Parse → Normalize (DRAIN3) → Embed → Detect Anomalies → Enrich (LLM) → Report
```

**Anomaly Detection**

- **Spike** — Counts log entries per time bucket and computes a Z-score against historical bucket counts.
- **Burst** — Tracks ERROR and CRITICAL entries per bucket. Fires after a configurable number of consecutive over-threshold minutes.
- **Novelty** — Encodes each message as an embedding and computes cosine similarity against a FAISS vector index (a high-performance approximate nearest-neighbor library) of known patterns. Messages below the similarity threshold are flagged as novel and committed to the store.

**Semantic Cache**

LLM results are cached in a two-tier lookup:

- **Tier 1** — Similarity above the high threshold and not stale: return cached result directly.
- **Tier 2** — Medium similarity with matching service and log level within the recency window: return cached result.
- **Miss** — Call the LLM and cache the result.

---

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) installed and running

```bash
ollama pull gemma:2b   # pull the model once
ollama serve           # start the server (if not already running)
```

The sentence-transformers embedding model (`all-MiniLM-L6-v2`) downloads automatically on first run.

---

## Installation

```bash
git clone https://github.com/yoaveid/log-analyzer.git
cd log-analyzer
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py data/sample_logs/sample.log
```

With options:

```bash
python main.py path/to/app.log \
  --output data/output/report.json \
  --config-path config/settings.yaml
```

**Expected log format:**

```
2024-01-15T10:00:00Z ERROR    db-service    Connection pool exhausted
2024-01-15T10:00:01Z INFO     api-service   Request received
2024-01-15T10:00:02Z CRITICAL auth-service  JWT validation failed
```

---

## Configuration

Edit `config/settings.yaml`:

| Key                                  | Default                  | Description                                                  |
| ------------------------------------ | ------------------------ | ------------------------------------------------------------ |
| `llm.model`                          | `gemma:2b`               | Ollama model used for enrichment                             |
| `llm.base_url`                       | `http://localhost:11434` | Ollama server URL                                            |
| `llm.max_retries`                    | `3`                      | Retry attempts on LLM failure                                |
| `embedder.model`                     | `all-MiniLM-L6-v2`       | Sentence-transformers model for embeddings                   |
| `cache.high_threshold`               | `0.9`                    | Similarity above this returns cached result directly         |
| `cache.low_threshold`                | `0.8`                    | Similarity below this is always a miss                       |
| `cache.recency_window_seconds`       | `300`                    | Tier 2 match must be seen within this window                 |
| `cache.staleness_days`               | `30`                     | Tier 1 entries older than this are treated as stale          |
| `anomaly.novelty_threshold`          | `0.65`                   | Cosine similarity below this flags a message as novel        |
| `anomaly.min_store_size`             | `30`                     | Minimum known patterns before novelty detection activates    |
| `anomaly.spike.bucket_seconds`       | `60`                     | Time bucket width for spike counting                         |
| `anomaly.spike.z_threshold`          | `2.5`                    | Spike sensitivity — lower = more sensitive                   |
| `anomaly.spike.min_history`          | `5`                      | Minimum buckets of history required before spike can fire    |
| `anomaly.spike.min_spike_count`      | `5`                      | Minimum absolute count required before spike can fire        |
| `anomaly.burst.bucket_seconds`       | `60`                     | Time bucket width for burst counting                         |
| `anomaly.burst.threshold_per_bucket` | `5`                      | Errors per bucket before a burst fires                       |
| `anomaly.burst.consecutive_buckets`  | `3`                      | Consecutive over-threshold buckets required to confirm burst |
| `logging.level`                      | `INFO`                   | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR`           |
| `logging.format`                     | `text`                   | `text` for local dev, `json` for production ingestion        |

---

## Output

The report is written to `data/output/report.json` by default.

```json
{
  "summary": {
    "total_parsed": 1200,
    "error_count": 43,
    "malformed_lines": 2,
    "cache_hit_rate": 0.74
  },
  "enriched_errors": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "level": "ERROR",
      "service": "db-service",
      "message": "Connection pool exhausted",
      "root_cause": "Database connection pool saturated under peak load.",
      "mitigation": "Increase pool size or add connection retry backoff.",
      "cache_hit": false
    }
  ],
  "anomalies": [
    {
      "kind": "spike",
      "description": "Error rate spike detected in db-service",
      "affected_services": ["db-service"]
    }
  ],
  "top_recurring_errors": ["..."],
  "llm_health": {
    "total_requests": 43,
    "retry_count": 2,
    "failed_requests": 0
  }
}
```

---

## Design Decisions

- **Normalize before embedding** — Log messages are normalized using DRAIN3 to strip dynamic values (IDs, ports, timestamps) before encoding. This ensures that `"Connection refused on port 5432"` and `"Connection refused on port 3306"` produce the same embedding and hit the cache as one pattern.

- **Semantic cache over exact match** — Similar errors with different runtime values would never match an exact-string cache. Cosine similarity on embeddings catches them regardless of the specific value in the message.

- **Deduplication gate at write time** — Duplicate embeddings are rejected before entering the FAISS index rather than filtered at query time. This keeps the index size bounded to unique patterns, preventing memory growth during high-volume error bursts.

---

## Testing

```bash
pytest
```

- Parser behavior and timestamp format variants
- Malformed log line handling
- Spike, burst, and novelty detection logic
- Semantic cache tier hit/miss behavior
- LLM retry, fallback, and JSON extraction
- Log normalizer DRAIN3 state isolation
- End-to-end pipeline with mocked LLM and embedder
