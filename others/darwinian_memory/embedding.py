"""
Darwinian Memory System — Embedding Module
===========================================
Provides the text-embedding function φ(·) used by Dual-Factor Retrieval
(§3.2.2).  The embedding backend is pluggable — default uses
sentence-transformers; fallback is a lightweight TF-IDF vectorizer
so the module works without GPU or heavy dependencies.

Key formula:
    Score(ˆp, p) = sim(φ(ˆp_pre), φ(p_pre)) · sim(φ(ˆp_goal), φ(p_goal))

where sim(·,·) is cosine similarity.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Sequence
from abc import ABC, abstractmethod


# ═══════════════════════════════════════════════════════════════════════
# Abstract embedding backend
# ═══════════════════════════════════════════════════════════════════════

class EmbeddingBackend(ABC):
    """Pluggable text-embedding backend."""

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        """Encode a single text string into a dense vector."""
        ...

    @abstractmethod
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts into a (N, dim) matrix."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


# ═══════════════════════════════════════════════════════════════════════
# TF-IDF fallback backend (zero-dependency)
# ═══════════════════════════════════════════════════════════════════════

class TFIDFBackend(EmbeddingBackend):
    """Lightweight TF-IDF vectorizer.  Works offline, no GPU required.

    Builds a vocabulary incrementally; unseen terms at inference time
    are simply ignored (their weight is 0).
    """

    def __init__(self, dim: int = 768):
        self._dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_count = 0

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        import re
        return re.findall(r'[a-zA-Z0-9_]+', text.lower())

    def fit(self, texts: list[str]):
        """Build vocabulary and compute IDF from a corpus."""
        df: dict[str, int] = {}
        for text in texts:
            tokens = set(self._tokenize(text))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)
        self._doc_count = len(texts)
        # IDF = log((N+1) / (df+1)) + 1  (smooth)
        for t, d in df.items():
            self._idf[t] = math.log((self._doc_count + 1) / (d + 1)) + 1.0

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1.0
        # Normalize TF
        norm = math.sqrt(sum(v * v for v in tf.values())) or 1.0
        vec = np.zeros(self._dim, dtype=np.float32)
        for t, f in tf.items():
            idx = self._vocab.get(t)
            if idx is not None and idx < self._dim:
                vec[idx] = (f / norm) * self._idf.get(t, 0.0)
        # L2-normalize the final vector
        vnorm = float(np.linalg.norm(vec)) or 1.0
        return vec / vnorm

    def encode(self, text: str) -> np.ndarray:
        return self._vectorize(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vectorize(t) for t in texts], axis=0)

    @property
    def dim(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════════
# Sentence-Transformers backend (preferred, requires pip install)
# ═══════════════════════════════════════════════════════════════════════

class SentenceTransformerBackend(EmbeddingBackend):
    """Wrapper around sentence-transformers for high-quality embeddings.

    Install:  pip install sentence-transformers
    Default model: all-MiniLM-L6-v2 (384-dim, fast, good quality).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerBackend. "
                "Install with: pip install sentence-transformers"
            )
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)

    @property
    def dim(self) -> int:
        return self._dim


# ═══════════════════════════════════════════════════════════════════════
# Cosine similarity
# ═══════════════════════════════════════════════════════════════════════

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized vectors.

    Assumes vectors are already normalized (‖a‖ = ‖b‖ = 1), so we can
    just compute the dot product.  If not normalized, this still works
    correctly via the full formula.
    """
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b
    if denom == 0.0:
        return 0.0
    return dot / denom


# ═══════════════════════════════════════════════════════════════════════
# Dual-Factor Similarity (§3.2.2, Eq. 1)
# ═══════════════════════════════════════════════════════════════════════

def dual_factor_similarity(
    emb_pre_query: np.ndarray,
    emb_goal_query: np.ndarray,
    emb_pre_candidate: np.ndarray,
    emb_goal_candidate: np.ndarray,
) -> float:
    """Compute the multiplicative dual-factor similarity score.

    Score(ˆp, p) = sim(φ(ˆp_pre), φ(p_pre)) · sim(φ(ˆp_goal), φ(p_goal))

    A high score requires BOTH the starting state context AND the
    intended objective to align, significantly reducing false positives
    from goal-only matching in dynamic GUI environments.
    """
    sim_pre = cosine_similarity(emb_pre_query, emb_pre_candidate)
    sim_goal = cosine_similarity(emb_goal_query, emb_goal_candidate)
    return sim_pre * sim_goal
