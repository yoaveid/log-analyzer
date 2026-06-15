from unittest.mock import patch

import pytest

from src.config.settings import LLMConfig
from src.enrichment.llm_client import LLMClient
from tests.conftest import make_entry


def _cfg(max_retries: int = 3) -> LLMConfig:
    return LLMConfig(model="test-model", base_url="http://localhost:11434", max_retries=max_retries)


@pytest.fixture
def mock_ollama():
    with patch("src.enrichment.llm_client.OllamaLLM") as cls:
        yield cls

def test_returns_parsed_root_cause_and_mitigation(mock_ollama):
    mock_ollama.return_value.invoke.return_value = (
        '{"root_cause": "DB timeout", "mitigation": "Increase pool size"}'
    )
    client = LLMClient(_cfg())
    root_cause, mitigation = client.analyze(make_entry(message="Connection refused"))
    assert root_cause == "DB timeout"
    assert mitigation == "Increase pool size"


def test_json_extracted_from_prose_response(mock_ollama):
    mock_ollama.return_value.invoke.return_value = (
        'Here is my analysis: {"root_cause": "Network timeout", "mitigation": "Retry with backoff"} '
        "I hope this helps."
    )
    client = LLMClient(_cfg())
    root_cause, mitigation = client.analyze(make_entry())
    assert root_cause == "Network timeout"
    assert mitigation == "Retry with backoff"


def test_total_requests_increments_per_call(mock_ollama):
    mock_ollama.return_value.invoke.return_value = '{"root_cause": "x", "mitigation": "y"}'
    client = LLMClient(_cfg())
    client.analyze(make_entry())
    client.analyze(make_entry())
    assert client.total_requests == 2


# ─── retry logic ──────────────────────────────────────────────────────────────

def test_retries_on_transient_exception(mock_ollama):
    mock_ollama.return_value.invoke.side_effect = [
        Exception("timeout"),
        '{"root_cause": "DB down", "mitigation": "Restart DB"}',
    ]
    client = LLMClient(_cfg(max_retries=3))
    root_cause, _ = client.analyze(make_entry())
    assert root_cause == "DB down"
    assert client.retry_count == 1


def test_fallback_when_all_retries_exhausted(mock_ollama):
    mock_ollama.return_value.invoke.side_effect = Exception("always fails")
    client = LLMClient(_cfg(max_retries=2))
    root_cause, mitigation = client.analyze(make_entry())
    assert root_cause == "Unable to determine root cause."
    assert "escalate" in mitigation
    assert client.failed_requests == 1


# ─── parse fallback ───────────────────────────────────────────────────────────

def test_fallback_on_malformed_json(mock_ollama):
    mock_ollama.return_value.invoke.return_value = "Sorry, I cannot help with that."
    client = LLMClient(_cfg())
    root_cause, _ = client.analyze(make_entry())
    assert root_cause == "Unable to determine root cause."


def test_fallback_on_missing_json_keys(mock_ollama):
    mock_ollama.return_value.invoke.return_value = '{"error": "unexpected format"}'
    client = LLMClient(_cfg())
    root_cause, _ = client.analyze(make_entry())
    assert root_cause == "Unable to determine root cause."


# ─── to_dict ──────────────────────────────────────────────────────────────────

def test_to_dict_reports_all_counters(mock_ollama):
    mock_ollama.return_value.invoke.side_effect = [
        Exception("fail"),
        Exception("fail"),  # exhausts max_retries=2 → failed_requests=1
    ]
    client = LLMClient(_cfg(max_retries=2))
    client.analyze(make_entry())
    stats = client.to_dict()
    assert stats["total_requests"] == 1
    assert stats["failed_requests"] == 1
    assert "retry_count" in stats
