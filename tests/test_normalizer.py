from unittest.mock import MagicMock

from src.store.normalizer import LogNormalizer



def test_parse_returns_template_and_cluster_id():
    n = LogNormalizer()
    n.parse("Connection refused on port 5432")
    result = n.parse("Connection refused on port 3306")
    assert result.cluster_id != -1
    assert "3306" not in result.template  # drain3 replaced the variable after seeing two examples


def test_similar_messages_share_cluster():
    n = LogNormalizer()
    r1 = n.parse("Connection refused on port 5432")
    r2 = n.parse("Connection refused on port 3306")
    assert r1.cluster_id == r2.cluster_id



def test_parse_fallback_on_exception():
    n = LogNormalizer()
    n._miner.add_log_message = MagicMock(side_effect=RuntimeError("drain3 broke"))
    result = n.parse("some message")
    assert result.template == "some message"
    assert result.cluster_id == -1


# ─── normalize ────────────────────────────────────────────────────────────────

def test_normalize_returns_template_for_known_message():
    n = LogNormalizer()
    n.parse("Connection refused on port 5432")
    n.parse("Connection refused on port 3306")
    template = n.normalize("Connection refused on port 3306")
    assert "3306" not in template  # drain3 learned the pattern; normalize returns the template


def test_normalize_fallback_for_unregistered_message():
    n = LogNormalizer()
    result = n.normalize("never seen before message")
    assert result == "never seen before message"
