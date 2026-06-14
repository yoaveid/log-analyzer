import hashlib

import numpy as np


class MockEmbedder:
    """Offline stand-in for Embedder. Returns deterministic normalized vectors
    without loading any model or hitting HuggingFace."""

    DIM = 384

    def encode(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)
