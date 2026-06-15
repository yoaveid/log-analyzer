from src.analyzer.stats import LogStats
from src.models.log_entry import LogLevel
from tests.conftest import make_entry


def test_error_rate_zero_when_no_entries():
    assert LogStats().error_rate == 0.0


def test_error_count_includes_error_and_critical_only():
    stats = LogStats()
    for level in (LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING):
        stats.update(make_entry(level=level))
    stats.update(make_entry(level=LogLevel.ERROR))
    stats.update(make_entry(level=LogLevel.CRITICAL))
    assert stats.error_count == 2


def test_error_rate_calculation():
    stats = LogStats()
    for _ in range(3):
        stats.update(make_entry(level=LogLevel.ERROR))
    for _ in range(7):
        stats.update(make_entry(level=LogLevel.INFO))
    assert stats.error_rate == 0.3
