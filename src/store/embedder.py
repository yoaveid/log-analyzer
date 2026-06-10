import importlib
import numpy as np


class Embedder:
    """Lazy-loads the sentence transformer and keeps a single model instance."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def encode(self, text: str) -> np.ndarray:
        return self._load().encode(text, normalize_embeddings=True)

    def _load(self):
        if self._model is None:
            sentence_transformers = importlib.import_module("sentence_transformers")
            self._model = sentence_transformers.SentenceTransformer(self._model_name)
        return self._model
