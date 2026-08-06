import unittest
from unittest import mock

import numpy as np

from android_world.agents.memory.autodl_embedding import AutoDLEmbeddingBackend
from android_world.agents.memory.environment import EnvKnowledge


class _FakeRagClient:
    """Stand-in for RagClient.embed returning a canned /embed response."""

    def __init__(self, dim: int = 1024):
        self._dim = dim
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> dict:
        self.calls.append(texts)
        rng = np.random.RandomState(0)
        vecs = rng.randn(len(texts), self._dim).astype(np.float32)
        return {"dim": self._dim, "vectors": vecs.tolist()}


class AutoDLEmbeddingBackendTest(unittest.TestCase):

    def _backend(self, dim: int = 1024):
        client = _FakeRagClient(dim=dim)
        backend = AutoDLEmbeddingBackend(client=client)
        return backend, client

    def test_encode_returns_1024_dim_normalized(self):
        backend, client = self._backend()
        v = backend.encode("Markor main screen")
        self.assertEqual(v.shape, (1024,))
        self.assertAlmostEqual(float(np.linalg.norm(v)), 1.0, places=4)
        self.assertEqual(client.calls, [["Markor main screen"]])

    def test_encode_batch_returns_n_dims(self):
        backend, client = self._backend()
        v = backend.encode_batch(["a", "b", "c"])
        self.assertEqual(v.shape, (3, 1024))
        norms = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), rtol=1e-4)
        self.assertEqual(client.calls, [["a", "b", "c"]])

    def test_encode_empty_batch_returns_empty(self):
        backend, _ = self._backend()
        v = backend.encode_batch([])
        self.assertEqual(v.shape, (0, 1024))

    def test_dim_property(self):
        backend, _ = self._backend()
        self.assertEqual(backend.dim, 1024)


class EnvKnowledgeDefaultEmbedderTest(unittest.TestCase):

    def test_defaults_to_autodl_backend(self):
        with mock.patch(
            "android_world.agents.memory.environment.AutoDLEmbeddingBackend"
        ) as mock_backend, mock.patch(
            "android_world.agents.memory.environment.PageGraph"
        ) as mock_graph:
            EnvKnowledge(rag_url="http://127.0.0.1:18180")
            mock_backend.assert_called_once_with(rag_url="http://127.0.0.1:18180")
            mock_graph.assert_called_once()
            args, kwargs = mock_graph.call_args
            self.assertIs(kwargs.get("embedder"), mock_backend.return_value)


if __name__ == "__main__":
    unittest.main()
