from dataclasses import dataclass

from drain3 import TemplateMiner


@dataclass
class ParseResult:
    template: str
    cluster_id: int


class LogNormalizer:
    """
    Uses the DRAIN algorithm (via drain3) to extract log templates.
    """

    def __init__(self):
        self._miner = TemplateMiner()

    def parse(self, message: str) -> ParseResult:
        """
        Register a message with drain3 and return its template + cluster ID.
        Call this ONCE per message from the pipeline before any downstream
        processing.
        """
        try:
            result = self._miner.add_log_message(message)
            if result is None:
                return ParseResult(template=message, cluster_id=-1)
            return ParseResult(
                template=result["template_mined"],
                cluster_id=result["cluster_id"],
            )
        except Exception:
            return ParseResult(template=message, cluster_id=-1)

    def normalize(self, message: str) -> str:
        """
        Return the template for a message WITHOUT modifying drain3 state.
        Falls back to parse() only if the message was never registered.
        """
        try:
            cluster = self._miner.match(message)
            if cluster is not None:
                return cluster.get_template()
        except Exception:
            pass
        return self.parse(message).template
