import importlib
from typing import Optional

import numpy as np

from src.store.normalizer import LogNormalizer


class Embedder:
    """Lazy-loads the sentence transformer and keeps a single model instance."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalizer: Optional[LogNormalizer] = None,
    ):
        self._model_name = model_name
        self._model = None
        self._normalizer = normalizer or LogNormalizer()

    def encode(self, text: str) -> np.ndarray:
        normalized = self._normalizer.normalize(text)
        return self._load().encode(normalized, normalize_embeddings=True)

    def _load(self):
        if self._model is None:
            sentence_transformers = importlib.import_module("sentence_transformers")
            self._model = sentence_transformers.SentenceTransformer(self._model_name)
        return self._model
