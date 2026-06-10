from __future__ import annotations

from collections import deque, Counter
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from src.models.log_entry import LogEntry, LogLevel
from src.store.embedding_store import EmbeddingStore
from src.store.embedder import Embedder


CRITICAL_LEVELS = {LogLevel.CRITICAL, LogLevel.ERROR}


@dataclass
class Anomaly:
    kind: str           # "spike" | "burst" | "novel_error"
    description: str
    entries: list[LogEntry] = field(default_factory=list)


class AnomalyDetector:
    def __init__(
        self,
        spike_window_seconds: int = 60,
        spike_threshold: int = 3,
        burst_threshold: int = 2,
        novelty_threshold: float = 0.65,  # cosine similarity below this = novel
        size_store_threshold: int = 100,  # only check novelty if store has at least this many entries  
        store: Optional[EmbeddingStore] = None,
        embedder: Optional[Embedder] = None,
    ):
        self._spike_window = timedelta(seconds=spike_window_seconds)
        self._spike_threshold = spike_threshold
        self._burst_threshold = burst_threshold
        self._novelty_threshold = novelty_threshold
        self._size_store_threshold = size_store_threshold

        # Rule-based state
        self._recent_errors: deque[LogEntry] = deque()
        self._burst_counter: Counter[str] = Counter()
        self._in_spike = False  # prevents re-reporting the same ongoing spike

        # Shared with AnalysisCache — injected so both use one model + one index
        self._store = store or EmbeddingStore()
        self._embedder = embedder or Embedder()

    # ------------------------------------------------------------------
    # Stream entry point — call once per incoming log line
    # ------------------------------------------------------------------

    def process_entry(self, entry: LogEntry) -> list[Anomaly]:
        """Return any anomalies triggered by this single entry."""
        anomalies_check = [self._check_spike, self._check_novel]
        # check repeating messages only for critical levels
        if entry.level in CRITICAL_LEVELS:
            anomalies_check += [self._check_burst]
        
        anomalies: list[Anomaly] = []
        for check in anomalies_check:
            result = check(entry)
            if result:
                anomalies.append(result)
        return anomalies

    # ------------------------------------------------------------------
    # Spike — sliding time window over recent errors
    # ------------------------------------------------------------------

    def _check_spike(self, entry: LogEntry) -> Optional[Anomaly]:
        cutoff = entry.timestamp - self._spike_window
        while self._recent_errors and self._recent_errors[0].timestamp < cutoff:
            self._recent_errors.popleft()

        self._recent_errors.append(entry)

        if len(self._recent_errors) >= self._spike_threshold:
            if not self._in_spike:
                self._in_spike = True
                return Anomaly(
                    kind="spike",
                    description=(
                        f"{len(self._recent_errors)} errors within "
                        f"{self._spike_window.seconds}s ending at {entry.timestamp}"
                    ),
                    entries=list(self._recent_errors),
                )
        else:
            self._in_spike = False 
        return None

    # ------------------------------------------------------------------
    # Burst — same message repeating
    # ------------------------------------------------------------------

    def _check_burst(self, entry: LogEntry) -> Optional[Anomaly]:
        self._burst_counter[entry.message] += 1
        # Report only on the exact crossing point to avoid duplicate alerts
        if self._burst_counter[entry.message] == self._burst_threshold:
            return Anomaly(
                kind="burst",
                description=f'Message repeated {self._burst_threshold}x: "{entry.message[:80]}"',
                entries=[entry],
            )
        return None

    # ------------------------------------------------------------------
    # Novel error — embedding similarity vs store
    # ------------------------------------------------------------------

    def _check_novel(self, entry: LogEntry) -> Optional[Anomaly]:
        if self._store.size <= self._size_store_threshold:
            # Not enough data in the store to make a meaningful similarity comparison
            return None
        emb = self._embedder.encode(entry.message)
        result = self._store.best_match(emb)

        is_novel = result is None or result.similarity < self._novelty_threshold

        if is_novel:
            self._store.add(emb, {
                "message": entry.message,
                "timestamp": str(entry.timestamp),
            })
            return Anomaly(
                kind="novel_error",
                description=f'Novel error pattern: "{entry.message[:80]}"',
                entries=[entry],
            )
        return None
