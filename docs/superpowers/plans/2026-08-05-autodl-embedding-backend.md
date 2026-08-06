# AutoDL Embedding Backend for Local U3 Page Graph — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the local U3 page graph do vector retrieval using AutoDL's BGE-M3 (`/embed`) instead of a local embedding model, so the local graph works without any local vector model.

**Architecture:** Add `AutoDLEmbeddingBackend` (implements `EmbeddingBackend` from `others/darwinian_memory/embedding.py`) that wraps the existing `RagClient.embed()` call. Wire it as `EnvKnowledge`'s default `PageGraph` embedder when none is supplied. Local graph learning/persistence/retrieval stays local; only the vector math goes to AutoDL over the SSH tunnel (RAG_URL, default `http://127.0.0.1:18180`). This matches the user's chosen architecture: **full dependence on AutoDL for embedding** (SiliconFlow API is used only for the MLLM decision calls, not for embeddings).

**Tech Stack:** Python 3, `android_world` (AndroidWorld), numpy, unittest, requests (via `RagClient`).

**Test command:** `python -m unittest discover -s android_world/agents/memory` from repo root `C:\Users\WRQ\Desktop\androidworld`.

**No git commits** — user prefers no-commit; leave changes in the working tree.

---

## Files

- Create: `android_world/agents/memory/autodl_embedding.py` — `AutoDLEmbeddingBackend`.
- Modify: `android_world/agents/memory/environment.py` — `EnvKnowledge.__init__` default embedder.
- Test: `android_world/agents/memory/test_autodl_embedding.py` — new unit tests.

No change to `PageGraph`, `RagClient`, `dms_bridge`, or `memory_agent.py`.

---

### Task 1: Create `AutoDLEmbeddingBackend`

**Files:**
- Create: `android_world/agents/memory/autodl_embedding.py`
- Test: `android_world/agents/memory/test_autodl_embedding.py`

- [ ] **Step 1: Write the failing test**

Create `android_world/agents/memory/test_autodl_embedding.py`:

```python
import unittest
from unittest import mock

import numpy as np

from android_world.agents.memory.autodl_embedding import AutoDLEmbeddingBackend


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

    @property
    def dim(self):
        return 1024

    def test_dim_property(self):
        backend, _ = self._backend()
        self.assertEqual(backend.dim, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest android_world.agents.memory.test_autodl_embedding -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'android_world.agents.memory.autodl_embedding'`.

- [ ] **Step 3: Write the implementation**

Create `android_world/agents/memory/autodl_embedding.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest android_world.agents.memory.test_autodl_embedding -v`
Expected: PASS (all 4 tests).

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory/autodl_embedding.py android_world/agents/memory/test_autodl_embedding.py
git commit -m "feat(memory): AutoDL embedding backend for local page graph"
```

(NO-COMMIT per user preference — skip this step, leave changes in working tree.)

---

### Task 2: Wire `AutoDLEmbeddingBackend` into `EnvKnowledge` default

**Files:**
- Modify: `android_world/agents/memory/environment.py` (`EnvKnowledge.__init__`, ~line 88)

- [ ] **Step 1: Update the default embedder logic**

In `android_world/agents/memory/environment.py`, `EnvKnowledge.__init__` currently ends with:

```python
    self._graph = PageGraph(persist_dir=persist_dir, embedder=embedder)
```

Change the constructor body so that when `embedder` is None, it defaults to `AutoDLEmbeddingBackend` pointed at `rag_url`. Add an import at the top of the file (after the existing `from android_world.agents.memory.page_graph import PageGraph`):

```python
from android_world.agents.memory.autodl_embedding import AutoDLEmbeddingBackend
```

And in `__init__`:

```python
    if embedder is None:
      embedder = AutoDLEmbeddingBackend(rag_url=self.rag_url)
    self._graph = PageGraph(persist_dir=persist_dir, embedder=embedder)
```

Note: `self.rag_url` must be set BEFORE this line (it is — `self.rag_url = rag_url or os.environ.get("RAG_URL", "")` at the top of `__init__`).

- [ ] **Step 2: Run the existing test suite to confirm nothing breaks**

Run: `python -m unittest discover -s android_world/agents/memory -v`
Expected: PASS — all tests. Existing `EnvKnowledgeLocalGraphTest` tests pass an explicit `embedder=FakeEmbedder()`, so they are unaffected. The new default only kicks in when no embedder is supplied (e.g. `AgentU3FeedTest` uses a `mock.Mock()` for `u3`, not a real `EnvKnowledge`, so it's unaffected too).

- [ ] **Step 3: Add a test that EnvKnowledge defaults to AutoDL backend**

In `test_autodl_embedding.py`, add a test class that patches `AutoDLEmbeddingBackend` and verifies `EnvKnowledge(rag_url=...)` constructs `PageGraph` with it. Use `unittest.mock.patch`:

```python
from unittest import mock
from android_world.agents.memory.environment import EnvKnowledge

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
```

Run: `python -m unittest android_world.agents.memory.test_autodl_embedding -v`
Expected: PASS.

- [ ] **Step 4: Run full suite**

Run: `python -m unittest discover -s android_world/agents/memory -v`
Expected: PASS.

- [ ] **Step 5: Commit**

(NO-COMMIT per user preference — skip.)

---

### Task 3: End-to-end verification (tunnel up optional)

**Files:**
- No source changes — verification only.

- [ ] **Step 1: Confirm AutoDL /embed works through the tunnel**

If the SSH tunnel (`pg_agent_rag\tunnel\start_tunnel.ps1`) is running and `RAG_URL=http://127.0.0.1:18180`, verify:

```bash
curl -s -X POST http://127.0.0.1:18180/embed -H "Content-Type: application/json" -d '{"texts":["hello world"]}'
```

Expected: `{"dim":1024,"vectors":[[...1024 floats...]]}`.

- [ ] **Step 2: Local smoke — EnvKnowledge records + retrieves via AutoDL**

Run a short Python check:

```bash
python -c "
from android_world.agents.memory.environment import EnvKnowledge
import tempfile
with tempfile.TemporaryDirectory() as d:
    ek = EnvKnowledge(rag_url='http://127.0.0.1:18180', persist_dir=d)
    ek.record_transition('Markor main screen', 'click new note', 'Create a note', 'Markor editor screen')
    print('transition recorded')
    hint = ek.retrieve_hint('', current_app='net.gsantner.markor', current_page='Markor main')
    print('hint:', hint[:200] if hint else '(empty)')
"
```

Expected: prints "transition recorded" and a non-empty hint (guidelines from the local graph, which now vectorized via AutoDL). If the tunnel is down, this fails with a connection error — that is expected and confirms AutoDL is now a hard dependency (per user's chosen mode).

- [ ] **Step 3: Full memory suite**

Run: `python -m unittest discover -s android_world/agents/memory -v`
Expected: PASS.

---

## Self-Review

**Spec coverage:**
- AutoDLEmbeddingBackend: Task 1 (encode/encode_batch/dim + tests).
- Default wiring: Task 2.
- Verification: Task 3.

**Placeholder scan:** No TBDs; all code provided inline.

**Type consistency:** `AutoDLEmbeddingBackend(client=...)` — `_FakeRagClient.embed(texts)->dict` and `RagClient.embed(texts)->dict` have the same signature; `encode_batch` expects `raw["vectors"]` list-of-lists. `dim` property returns 1024. `EnvKnowledge.__init__` passes `rag_url=self.rag_url` (already set). No name clashes.
