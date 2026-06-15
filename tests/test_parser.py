from src.ingestion.parser import parse_file, parse_line
from src.models.log_entry import LogLevel


def test_valid_line():
    entry = parse_line("2024-01-15T10:00:10Z ERROR db-service Connection pool exhausted")
    assert entry is not None
    assert entry.level == LogLevel.ERROR
    assert entry.service == "db-service"
    assert entry.message == "Connection pool exhausted"


def test_malformed_line_returns_none():
    assert parse_line("this is garbage") is None


def test_empty_line_returns_none():
    assert parse_line("") is None


def test_timestamp_with_fractional_seconds():
    entry = parse_line("2024-01-15T10:00:10.123Z ERROR auth-service JWT failed")
    assert entry is not None
    assert entry.service == "auth-service"


def test_timestamp_with_positive_offset():
    entry = parse_line("2024-01-15T10:00:10+05:30 CRITICAL db-service OOM")
    assert entry is not None
    assert entry.level == LogLevel.CRITICAL


def test_all_log_levels_parsed():
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        entry = parse_line(f"2024-01-15T10:00:00Z {level} svc message")
        assert entry is not None, f"Failed to parse level {level}"
        assert entry.level.value == level

def test_parse_file_yields_none_for_malformed_lines(tmp_path):
    log = tmp_path / "test.log"
    log.write_text(
        "2024-01-15T10:00:00Z ERROR svc valid line\n"
        "this is garbage\n"
        "2024-01-15T10:00:01Z INFO svc another valid\n"
    )
    results = list(parse_file(log))
    assert len(results) == 3
    assert results[0] is not None
    assert results[1] is None
    assert results[2] is not None


