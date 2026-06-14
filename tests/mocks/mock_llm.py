from src.models.log_entry import LogEntry


class MockLLMClient:
    """Offline stand-in for LLMClient. Returns fixed responses without hitting Ollama."""

    def __init__(
        self,
        root_cause: str = "Mock root cause.",
        mitigation: str = "Mock mitigation.",
    ):
        self.total_requests = 0
        self.retry_count = 0
        self.failed_requests = 0
        self._root_cause = root_cause
        self._mitigation = mitigation

    def analyze(self, entry: LogEntry) -> tuple[str, str]:
        self.total_requests += 1
        return self._root_cause, self._mitigation

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "retry_count": self.retry_count,
            "failed_requests": self.failed_requests,
        }
