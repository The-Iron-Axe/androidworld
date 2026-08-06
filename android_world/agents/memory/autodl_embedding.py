"""AutoDL-backed embedding backend for the local U3 page graph.

Sends text to AutoDL's BGE-M3 `/embed` endpoint (via the SSH tunnel,
RAG_URL, default http://127.0.0.1:18180) and returns normalized dense
vectors.  No local embedding model is needed — the local page graph's
merge/retrieve vector math all happens against AutoDL vectors.
"""

from __future__ import annotations

import numpy as np

from android_world.agents.memory.dms_bridge import EmbeddingBackend


class AutoDLEmbeddingBackend(EmbeddingBackend):
    """EmbeddingBackend that calls AutoDL's /embed (BGE-M3) over the tunnel."""

    DIM = 1024  # BGE-M3 dense vector dimension

    def __init__(self, client=None, rag_url: str | None = None):
        """client: object with embed(texts)->{"dim","vectors"} (defaults to RagClient)."""
        if client is None:
            from client.rag_client import RagClient

            client = RagClient(base_url=rag_url)
        self._client = client
        self._dim = self.DIM

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        raw = self._client.embed(list(texts))
        vecs = np.asarray(raw["vectors"], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / (norms + 1e-12)

    @property
    def dim(self) -> int:
        return self._dim
