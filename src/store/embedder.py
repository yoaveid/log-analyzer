import importlib
from typing import Optional, Protocol

import numpy as np
import structlog

from src.store.normalizer import LogNormalizer
from src.config.settings import EmbedderConfig

logger = structlog.get_logger(__name__)


class EmbedderProtocol(Protocol):
    def encode(self, text: str) -> np.ndarray: ...


class Embedder:
    """Lazy-loads the sentence transformer and keeps a single model instance."""

    def __init__(
        self,
        config: EmbedderConfig,
        normalizer: Optional[LogNormalizer] = None,
    ):
        self._model_name = config.model
        self._model = None
        self._normalizer = normalizer or LogNormalizer()

    def encode(self, text: str) -> np.ndarray:
        normalized = self._normalizer.normalize(text)
        return self._load().encode(normalized, normalize_embeddings=True, show_progress_bar=False)

    def _load(self):
        if self._model is None:
            logger.info("loading_embedding_model", model=self._model_name)
            sentence_transformers = importlib.import_module("sentence_transformers")
            self._model = sentence_transformers.SentenceTransformer(self._model_name)
            logger.info("embedding_model_ready", model=self._model_name)
        return self._model
