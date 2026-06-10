import json
import re

from langchain_ollama import OllamaLLM

from src.models.log_entry import LogEntry

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)

_FALLBACK = ("Unable to determine root cause.", "Review logs and escalate if issue persists.")


class LLMClient:
    PROMPT_TEMPLATE = """You are a senior DevOps engineer analyzing server logs.

    Given a single log message, provide:
    1. A 1-sentence root cause analysis.
    2. A 1-sentence mitigation strategy.

    Rules:
    - Be concise and practical.
    - Focus only on operational/system causes (not generic explanations).
    - Assume this is a production system issue.
    - Do NOT include extra text, explanations, or formatting outside JSON.
    - Focus on system-level issues (network, DB, memory, latency, retries, timeouts).

    Output MUST be valid JSON in this exact format:
    {{"root_cause": "...", "mitigation": "..."}}

    Log message: {message}"""

    def __init__(self, max_retries: int = 3):
        self.llm = OllamaLLM(model="gemma:2b", base_url="http://localhost:11434")
        self._max_retries = max_retries
        self.total_requests = 0
        self.retry_count = 0
        self.failed_requests = 0

    def analyze(self, entry: LogEntry) -> tuple[str, str]:
        """Return (root_cause, mitigation) for a given log entry."""
        self.total_requests += 1
        prompt = self.PROMPT_TEMPLATE.format(message=entry.message)

        for attempt in range(self._max_retries):
            try:
                response = self.llm.invoke(prompt)
                return self._parse(response)
            except Exception:
                if attempt < self._max_retries - 1:
                    self.retry_count += 1  # only retries, not the first attempt

        self.failed_requests += 1
        return _FALLBACK

    def _parse(self, response: str) -> tuple[str, str]:
        match = _JSON_RE.search(response)
        if not match:
            return _FALLBACK
        try:
            data = json.loads(match.group())
            return data["root_cause"], data["mitigation"]
        except (json.JSONDecodeError, KeyError):
            return _FALLBACK

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "retry_count": self.retry_count,
            "failed_requests": self.failed_requests,
        }
