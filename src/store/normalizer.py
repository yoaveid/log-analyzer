import re

# Rules applied in order — most specific patterns must come first
# to avoid partial matches swallowing tokens meant for later rules.
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b', re.I), '<UUID>'),
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b'),       '<IP>'),
    (re.compile(r'https?://\S+'),                                              '<URL>'),
    (re.compile(r'(?<!\w)/(?:[\w./-]+)'),                                     '<PATH>'),
    (re.compile(r'\b0x[0-9a-fA-F]+\b'),                                       '<HEX>'),
    (re.compile(r'\b\d+\b'),                                                   '<ID>'),
]


class LogNormalizer:
    """
    Strips variable data from log messages before embedding.

    "auth failed for user_id=32"      → "auth failed for user_id=<ID>"
    "db retry in 30 seconds"          → "db retry in <ID> seconds"
    "db retry in 60 seconds"          → "db retry in <ID> seconds"  ← same result
    "connect to 10.0.0.1:5432 failed" → "connect to <IP> failed"
    """

    def normalize(self, message: str) -> str:
        text = message.lower().strip()
        for pattern, placeholder in _RULES:
            text = pattern.sub(placeholder, text)
        return re.sub(r'\s+', ' ', text).strip()
