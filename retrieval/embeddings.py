"""Local, free embedding models (sentence-transformers). Downloaded once from Hugging Face."""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        name = model_name.lower()
        if "e5" in name:
            self.q_prefix, self.p_prefix = "query: ", "passage: "
        elif "bge" in name and "-en" in name:
            self.q_prefix, self.p_prefix = "Represent this sentence for searching relevant passages: ", ""
        else:
            self.q_prefix, self.p_prefix = "", ""
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            print(f"Loading embedding model {self.model_name} (first time downloads it)...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.model.encode([self.p_prefix + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True, show_progress_bar=len(texts) > 200)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode([self.q_prefix + text], normalize_embeddings=True)[0]


class HashEmbedder:
    """Tiny offline embedder used only by the unit tests (no download needed)."""

    model_name = "hash"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)


@lru_cache(maxsize=4)
def get_embedder(model_name: str):
    return HashEmbedder() if model_name == "hash" else Embedder(model_name)
