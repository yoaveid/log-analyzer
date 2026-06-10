from collections import Counter

from src.models.log_entry import LogEntry, LogLevel


CRITICAL_LEVELS = {LogLevel.CRITICAL, LogLevel.ERROR}


class LogStats:
    """Incremental stats — call update() per entry, record_malformed() per bad line."""

    def __init__(self):
        self._total = 0
        self._malformed = 0
        self._by_level: Counter[str] = Counter()
        self._by_service: Counter[str] = Counter()
        self._error_count = 0

    def update(self, entry: LogEntry) -> None:
        self._total += 1
        self._by_level[entry.level.value] += 1
        self._by_service[entry.service] += 1
        if entry.level in CRITICAL_LEVELS:
            self._error_count += 1

    def record_malformed(self) -> None:
        self._malformed += 1

    @property
    def total_parsed(self) -> int:
        return self._total

    @property
    def malformed_count(self) -> int:
        return self._malformed

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def error_rate(self) -> float:
        return round(self._error_count / self._total, 4) if self._total else 0.0

    @property
    def by_level(self) -> dict[str, int]:
        return dict(self._by_level)

    @property
    def by_service(self) -> dict[str, int]:
        return dict(self._by_service)

    def to_dict(self) -> dict:
        return {
            "total_parsed": self.total_parsed,
            "malformed_lines": self.malformed_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "by_level": self.by_level,
            "by_service": self.by_service,
        }
