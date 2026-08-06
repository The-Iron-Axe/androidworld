# U3 Online Page Graph — Learn Environment Knowledge from Agent Experience

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give U3 (environment knowledge) the online graph-building capability from the PG-Agent paper (§3.1) — a local page graph that starts empty, grows as the agent executes, merges similar pages, and stores `before --action--> after` edges — so agent-discovered environment patterns (like the quick-settings trap) become retrievable knowledge instead of being discarded.

**Architecture:** A new pure-data `PageGraph` module (nodes = page summaries, edges = directed transitions with action + task) that merges nodes by embedding cosine similarity (deterministic — no local MLLM, keeping U3 pure data) and persists to JSON. `EnvKnowledge` composes it: `retrieve_hint` returns local learned guidelines first, then appends remote RAG guidelines if configured. The agent feeds each successful page transition into the graph via the existing `_on_step_complete` hook. This closes the gap identified in review: U3 was read-only against a pre-built remote graph; now it learns online like the paper's.

**Tech Stack:** Python 3, dataclasses, numpy, stdlib `unittest` (no pytest available). Reuses `dms_bridge` (already exports `EmbeddingBackend`, `TFIDFBackend`, `SentenceTransformerBackend`, `DMSConfig`).

---

## File Structure

- **Create** `android_world/agents/memory/page_graph.py` — `PageNode`, `PageEdge`, `PageGraph` (add_transition, merge-by-similarity, BFS guidelines retrieval, JSON persistence). Pure data, no LLM calls.
- **Modify** `android_world/agents/memory/environment.py` — `EnvKnowledge` gains a local `PageGraph`, `record_transition()`, and a merged local+remote `retrieve_hint` (remote failures now non-blocking).
- **Modify** `android_world/agents/memory/m3a.py` — add `self._current_goal = goal` at top of `step()`, and store `before_ui_elements_list` / `after_ui_elements` / `after_ui_elements_list` in `step_data` so hooks have page info.
- **Modify** `android_world/agents/memory/memory_agent.py` — `_on_step_complete` feeds the transition into U3 when `enable_u3`.
- **Modify** `android_world/agents/memory/__init__.py` — re-export `PageGraph`, `PageNode`, `PageEdge`.
- **Create** `android_world/agents/memory/test_page_graph.py` — unit tests for the graph + EnvKnowledge integration.

---

### Task 1: PageGraph data structures, add_transition, persistence

**Files:**
- Create: `android_world/agents/memory/page_graph.py`
- Create: `android_world/agents/memory/test_page_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# android_world/agents/memory/test_page_graph.py
import json
import os
import re
import tempfile
import unittest

import numpy as np

from android_world.agents.memory.page_graph import PageEdge, PageGraph, PageNode


class FakeEmbedder:
    """Bag-of-words embedder — overlapping tokens => high cosine similarity,
    disjoint tokens => low.  Deterministic, offline, discriminates properly
    (unlike the position-hash backend used elsewhere)."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def encode(self, text: str):
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in set(re.findall(r"[a-z0-9_]+", str(text).lower())):
            v[sum(ord(c) for c in tok) % self._dim] += 1.0
        norm = float(np.linalg.norm(v)) or 1.0
        return v / norm

    def encode_batch(self, texts: list[str]):
        return np.stack([self.encode(t) for t in texts], axis=0)

    @property
    def dim(self) -> int:
        return self._dim


class PageGraphAddTest(unittest.TestCase):

    def _graph(self, d: str) -> PageGraph:
        g = PageGraph(persist_dir=d, embedder=FakeEmbedder())
        return g

    def test_add_transition_creates_nodes_and_edge(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition(
                before_summary="Markor main screen",
                action_summary="click new note",
                task="Create a note",
                after_summary="Markor editor screen",
                before_app="net.gsantner.markor",
                after_app="net.gsantner.markor",
            )
            self.assertEqual(len(g.nodes), 2)
            self.assertEqual(len(g.edges), 1)
            e = g.edges[0]
            self.assertEqual(e.action_summary, "click new note")
            self.assertEqual(e.task, "Create a note")

    def test_identical_page_merges_into_existing_node(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main", "click A", "t1", "Markor editor")
            g.add_transition("Markor main", "click B", "t2", "Markor editor")
            # Both pages identical -> same nodes reused, one new edge for B
            self.assertEqual(len(g.nodes), 2)
            self.assertEqual(len(g.edges), 2)

    def test_repeated_edge_increments_count(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main", "click A", "t1", "Markor editor")
            g.add_transition("Markor main", "click A", "t2", "Markor editor")
            self.assertEqual(len(g.edges), 1)
            self.assertEqual(g.edges[0].count, 2)
            # Task list accumulates
            self.assertIn("t1", g.edges[0].task)
            self.assertIn("t2", g.edges[0].task)

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main", "click A", "t1", "Markor editor")
            g2 = PageGraph(persist_dir=d, embedder=FakeEmbedder())
            self.assertEqual(len(g2.nodes), 2)
            self.assertEqual(len(g2.edges), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'android_world.agents.memory.page_graph'`

- [ ] **Step 3: Write the PageGraph module**

```python
"""U3 Online Page Graph — learn app/page transitions from agent experience.

Faithful to PG-Agent (§3.1): the graph starts empty and grows as the agent
executes.  Each page transition  before --action--> after  becomes a directed
edge carrying the action summary and the task it served.  Pages are merged by
embedding cosine similarity (deterministic — no local MLLM, keeping U3 pure
data) instead of the paper's dual-level MLLM check.

U3 is pure data infrastructure: no LLM calls, no environment interaction.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from android_world.agents.memory import dms_bridge
from android_world.agents.memory.dms_bridge import EmbeddingBackend


@dataclass
class PageNode:
    """A page in the graph.  `page_summary` is its semantic identity."""
    page_id: str
    page_summary: str
    app: str = ""
    created_at: float = 0.0
    _embedding: Optional[list[float]] = field(default=None, repr=False)


@dataclass
class PageEdge:
    """A directed transition: source --action--> target, under task(s)."""
    source_id: str
    target_id: str
    action_summary: str
    task: str = ""  # semicolon-joined list of tasks this action served
    count: int = 1


class PageGraph:
    """Local, self-built page graph with merge-by-similarity + persistence."""

    def __init__(
        self,
        persist_dir: str = "",
        embedder: Optional[EmbeddingBackend] = None,
        merge_threshold: float = 0.85,
    ):
        self.persist_dir = persist_dir
        self.merge_threshold = merge_threshold
        self._nodes: dict[str, PageNode] = {}
        self._edges: list[PageEdge] = []
        self._embedder: Optional[EmbeddingBackend] = embedder
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            path = os.path.join(persist_dir, "page_graph.json")
            if os.path.exists(path):
                self._load(path)

    # ── Properties ────────────────────────────────────────────────

    @property
    def nodes(self) -> list[PageNode]:
        return list(self._nodes.values())

    @property
    def edges(self) -> list[PageEdge]:
        return list(self._edges)

    # ── Embedding ─────────────────────────────────────────────────

    def set_embedder(self, embedder: EmbeddingBackend) -> None:
        self._embedder = embedder

    def _ensure_embedder(self) -> EmbeddingBackend:
        if self._embedder is None:
            self._embedder = dms_bridge.SentenceTransformerBackend()
        return self._embedder

    def _embed(self, text: str) -> np.ndarray:
        return self._ensure_embedder().encode(text)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if denom == 0.0:
            return 0.0
        return float(np.dot(va, vb) / denom)

    # ── Core: add a page transition ───────────────────────────────

    def add_transition(
        self,
        before_summary: str,
        action_summary: str,
        task: str,
        after_summary: str,
        before_app: str = "",
        after_app: str = "",
    ) -> None:
        """Record a before --action--> after transition.

        Pages are merged by embedding similarity: an identical (or near-identical)
        page reuses the existing node instead of creating a duplicate.  Repeated
        identical transitions increment the edge count instead of duplicating.
        """
        before_id = self._merge_or_create(before_summary, before_app)
        after_id = self._merge_or_create(after_summary, after_app)
        self._upsert_edge(before_id, after_id, action_summary, task)

    def _merge_or_create(self, summary: str, app: str) -> str:
        summary = (summary or "").strip()
        if not summary:
            # Degenerate page — stable stub id so edges still link.
            return f"empty:{abs(hash((summary, app))) & 0xFFFFFFFF:08x}"
        emb = self._embed(summary).tolist()
        best_id, best_sim = None, -1.0
        for node in self._nodes.values():
            if node._embedding is None:
                continue
            sim = self._cosine(emb, node._embedding)
            if sim > best_sim:
                best_sim, best_id = sim, node.page_id
        if best_id is not None and best_sim >= self.merge_threshold:
            return best_id
        node = PageNode(
            page_id=uuid.uuid4().hex[:12],
            page_summary=summary,
            app=app,
            created_at=time.time(),
            _embedding=emb,
        )
        self._nodes[node.page_id] = node
        self.save()
        return node.page_id

    def _upsert_edge(self, src: str, tgt: str, action: str, task: str) -> None:
        for e in self._edges:
            if e.source_id == src and e.target_id == tgt and e.action_summary == action:
                e.count += 1
                if task and task not in e.task.split(";"):
                    e.task = f"{e.task};{task}" if e.task else task
                self.save()
                return
        self._edges.append(
            PageEdge(source_id=src, target_id=tgt, action_summary=action, task=task)
        )
        self.save()

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        p = path or (os.path.join(self.persist_dir, "page_graph.json") if self.persist_dir else "")
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        state = {
            "merge_threshold": self.merge_threshold,
            "nodes": [
                {"page_id": n.page_id, "page_summary": n.page_summary,
                 "app": n.app, "created_at": n.created_at, "embedding": n._embedding}
                for n in self._nodes.values()
            ],
            "edges": [
                {"source_id": e.source_id, "target_id": e.target_id,
                 "action_summary": e.action_summary, "task": e.task, "count": e.count}
                for e in self._edges
            ],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.merge_threshold = state.get("merge_threshold", 0.85)
        for nd in state.get("nodes", []):
            self._nodes[nd["page_id"]] = PageNode(
                page_id=nd["page_id"], page_summary=nd["page_summary"],
                app=nd.get("app", ""), created_at=nd.get("created_at", 0.0),
                _embedding=nd.get("embedding"),
            )
        for ed in state.get("edges", []):
            self._edges.append(PageEdge(
                source_id=ed["source_id"], target_id=ed["target_id"],
                action_summary=ed["action_summary"], task=ed.get("task", ""),
                count=ed.get("count", 1),
            ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory/page_graph.py android_world/agents/memory/test_page_graph.py
git commit -m "feat: add U3 online page graph (nodes, edges, merge, persistence)"
```

---

### Task 2: BFS guidelines retrieval from the learned graph

**Files:**
- Modify: `android_world/agents/memory/page_graph.py`
- Test: `android_world/agents/memory/test_page_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to test_page_graph.py
class PageGraphRetrievalTest(unittest.TestCase):

    def _graph(self, d: str) -> PageGraph:
        return PageGraph(persist_dir=d, embedder=FakeEmbedder())

    def test_empty_graph_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            self.assertEqual(g.retrieve_guidelines("any screen"), [])

    def test_retrieves_actions_and_tasks_around_similar_node(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main screen", "click new note",
                             "Create a note", "Markor editor screen")
            g.add_transition("Markor editor screen", "type content",
                             "Create a note", "Markor save screen")
            gl = g.retrieve_guidelines("Markor main screen")
            # BFS from the matched node should surface the outgoing action
            # and the achievable tasks.
            all_actions = [a for g in gl for a in g["actions"]]
            self.assertIn("click new note", all_actions)
            all_tasks = [t for g in gl for t in g["tasks"]]
            self.assertIn("Create a note", all_tasks)

    def test_disjoint_query_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main screen", "click new note",
                             "Create a note", "Markor editor screen")
            gl = g.retrieve_guidelines("OsmAnd satellite maps navigation")
            self.assertEqual(gl, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: FAIL with `AttributeError: 'PageGraph' object has no attribute 'retrieve_guidelines'`

- [ ] **Step 3: Add retrieve_guidelines to PageGraph**

Add a `hit_threshold` constructor param and the retrieval method:

```python
    def __init__(
        self,
        persist_dir: str = "",
        embedder: Optional[EmbeddingBackend] = None,
        merge_threshold: float = 0.85,
        hit_threshold: float = 0.5,
    ):
        # ... existing body ...
        self.hit_threshold = hit_threshold
```

Add the method (place after `_upsert_edge`):

```python
  def retrieve_guidelines(
      self,
      query_summary: str,
      top_k: int = 4,
      bfs_layers: int = 3,
  ) -> list[dict[str, Any]]:
    """Return [{actions: [...], tasks: [...]}] guidelines via BFS.

    Mirrors PG-Agent eq. (10)-(12): embed the query, retrieve the top-k most
    similar nodes, then BFS their outgoing edges to gather the achievable
    action queues and the tasks they serve.
    """
    if not self._nodes:
      return []
    emb = self._embed(query_summary).tolist()
    scored = []
    for node in self._nodes.values():
      if node._embedding is None:
        continue
      sim = self._cosine(emb, node._embedding)
      if sim >= self.hit_threshold:
        scored.append((sim, node.page_id))
    scored.sort(key=lambda t: t[0], reverse=True)
    seeds = [nid for _, nid in scored[:top_k]]
    if not seeds:
      return []

    seen_nodes = set(seeds)
    frontier = list(seeds)
    guidelines: list[dict[str, Any]] = []
    for _ in range(bfs_layers):
      next_frontier: list[str] = []
      for nid in frontier:
        for e in self._edges:
          if e.source_id != nid:
            continue
          actions = [e.action_summary] * min(e.count, 3)
          tasks = [t for t in e.task.split(";") if t]
          guidelines.append({"actions": actions, "tasks": tasks})
          if e.target_id not in seen_nodes:
            seen_nodes.add(e.target_id)
            next_frontier.append(e.target_id)
      frontier = next_frontier
      if not frontier:
        break

    # Dedupe by (actions, tasks)
    seen = set()
    uniq: list[dict[str, Any]] = []
    for g in guidelines:
      key = (tuple(g["actions"]), tuple(g["tasks"]))
      if key not in seen:
        seen.add(key)
        uniq.append(g)
    return uniq
```

Also update `save()` / `_load()` to persist `hit_threshold`:

In `save()`, add `"hit_threshold": self.hit_threshold` next to `merge_threshold`.
In `_load()`, add `self.hit_threshold = state.get("hit_threshold", 0.5)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory/page_graph.py android_world/agents/memory/test_page_graph.py
git commit -m "feat: add BFS guidelines retrieval to U3 page graph"
```

---

### Task 3: Integrate the local graph into EnvKnowledge

**Files:**
- Modify: `android_world/agents/memory/environment.py`
- Test: `android_world/agents/memory/test_page_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to test_page_graph.py
from android_world.agents.memory.environment import EnvKnowledge, build_screen_summary


class EnvKnowledgeLocalGraphTest(unittest.TestCase):

    def test_record_transition_then_retrieve(self):
        with tempfile.TemporaryDirectory() as d:
            ek = EnvKnowledge(rag_url="", persist_dir=d, embedder=FakeEmbedder())
            ek.record_transition(
                before_summary="Markor main screen",
                action_summary="click new note",
                task="Create a note",
                after_summary="Markor editor screen",
            )
            hint = ek.retrieve_hint(
                "Create a new note in Markor",
                "Some UI elements",
                current_app="net.gsantner.markor",
                current_page="Markor main",
            )
            self.assertIn("click new note", hint)

    def test_empty_graph_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            ek = EnvKnowledge(rag_url="", persist_dir=d, embedder=FakeEmbedder())
            hint = ek.retrieve_hint("anything", "ui", current_app="a")
            self.assertEqual(hint, "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: FAIL with `TypeError: EnvKnowledge.__init__() got an unexpected keyword argument 'persist_dir'`

- [ ] **Step 3: Modify EnvKnowledge to hold a local graph**

Rewrite `android_world/agents/memory/environment.py`:

```python
"""U3 Environment Knowledge — page-graph guidelines via local learned graph + AutoDL RAG.

U3 stores / retrieves environment knowledge (app/page transitions).  Two sources:

  1. Local online page graph (PG-Agent §3.1): starts empty, learns each
     successful page transition the agent makes, merges similar pages, and
     persists to disk.  This is where agent-discovered environment patterns
     live.
  2. Remote AutoDL RAG (optional): PG-Agent page-graph guidelines reached
     through the local SSH tunnel (RAG_URL, default http://127.0.0.1:18180).
     Used as a seed/pre-built knowledge source; failures are non-blocking.

U3 is pure data infrastructure from the agent's point of view: no LLM calls
locally; embedding + FAISS + BFS happen locally (graph) or on AutoDL (RAG).
"""

from __future__ import annotations

import os
import sys
from typing import Any

_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_client_dir = os.path.join(_repo_root, "client")
if _client_dir not in sys.path:
  sys.path.insert(0, _client_dir)

from rag_client import RagClient  # noqa: E402  pylint: disable=g-import-not-at-top

from android_world.agents.memory.page_graph import PageGraph
from android_world.agents.memory.dms_bridge import EmbeddingBackend


def build_screen_summary(
    goal: str,
    ui_elements_list: str,
    *,
    current_app: str = "",
    current_page: str = "",
    max_ui_chars: int = 1500,
) -> str:
  """Build a text screen summary S_It for RAG retrieve / graph nodes (no extra LLM call)."""
  parts: list[str] = []
  loc = []
  if current_app:
    loc.append(f"app={current_app}")
  if current_page:
    loc.append(f"page={current_page}")
  if loc:
    parts.append("Current screen: " + ", ".join(loc) + ".")
  parts.append(f"Task goal: {goal}")
  ui = (ui_elements_list or "").strip()
  if len(ui) > max_ui_chars:
    ui = ui[:max_ui_chars] + "\n..."
  if ui:
    parts.append("Visible UI elements:\n" + ui)
  return "\n".join(parts)


class EnvKnowledge:
  """U3: retrieve page-graph guidelines for the current screen.

  Composes a local learned PageGraph with an optional remote RAG client.
  """

  def __init__(
      self,
      rag_url: str | None = None,
      persist_dir: str = "",
      embedder: EmbeddingBackend | None = None,
      top_k: int = 4,
      bfs_layers: int = 3,
      max_guidelines: int = 12,
      timeout: float = 30.0,
  ):
    self.rag_url = rag_url or os.environ.get("RAG_URL", "")
    self.top_k = top_k
    self.bfs_layers = bfs_layers
    self.max_guidelines = max_guidelines
    self._client = RagClient(base_url=self.rag_url, timeout=timeout)
    self._last_raw: dict[str, Any] | None = None
    self._graph = PageGraph(persist_dir=persist_dir, embedder=embedder)

  def record_transition(
      self,
      before_summary: str,
      action_summary: str,
      task: str,
      after_summary: str,
      before_app: str = "",
      after_app: str = "",
  ) -> None:
    """Feed a page transition into the local learned graph (PG-Agent §3.1)."""
    self._graph.add_transition(
        before_summary, action_summary, task, after_summary,
        before_app=before_app, after_app=after_app,
    )

  def retrieve_hint(
      self,
      goal: str,
      ui_elements_list: str,
      *,
      current_app: str = "",
      current_page: str = "",
  ) -> str:
    """Return prompt-ready guidelines text.

    Local learned graph is always consulted first.  Remote RAG is appended if
    configured and reachable; its failure is non-blocking (logs a warning and
    returns local-only).  Empty guidelines from all sources return "".
    """
    summary = build_screen_summary(
        goal,
        ui_elements_list,
        current_app=current_app,
        current_page=current_page,
    )
    local = self._graph.retrieve_guidelines(summary, top_k=self.top_k, bfs_layers=self.bfs_layers)
    remote: list[dict[str, Any]] = []
    if self.rag_url:
      try:
        raw = self._client.retrieve(
            summary,
            top_k=self.top_k,
            bfs_layers=self.bfs_layers,
            max_guidelines=self.max_guidelines,
        )
        self._last_raw = raw
        remote = list(raw.get("guidelines") or [])
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[U3] remote RAG unavailable ({e}); using local graph only")
        remote = []
    guidelines = self._merge_guidelines(local, remote)
    if not guidelines:
      return ""
    return self._client.format_guidelines_for_prompt(guidelines)

  @staticmethod
  def _merge_guidelines(
      local: list[dict[str, Any]], remote: list[dict[str, Any]]
  ) -> list[dict[str, Any]]:
    """Dedupe local + remote guidelines by (actions, tasks)."""
    seen = set()
    out: list[dict[str, Any]] = []
    for g in list(local) + list(remote):
      key = (tuple(g.get("actions") or []), tuple(g.get("tasks") or []))
      if key not in seen:
        seen.add(key)
        out.append(g)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Verify the whole U3 module still imports (no network needed)**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -c "from android_world.agents.memory.environment import EnvKnowledge; print('env OK')"`
Expected: prints `env OK`

- [ ] **Step 6: Commit**

```bash
git add android_world/agents/memory/environment.py android_world/agents/memory/test_page_graph.py
git commit -m "feat: integrate local learned page graph into EnvKnowledge"
```

---

### Task 4: Agent hook — feed page transitions into U3

**Files:**
- Modify: `android_world/agents/memory/m3a.py`
- Modify: `android_world/agents/memory/memory_agent.py`
- Test: `android_world/agents/memory/test_page_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to test_page_graph.py
from unittest import mock


class AgentU3FeedTest(unittest.TestCase):
    """Verify MemoryAugmentedAgent._on_step_complete feeds U3 on success."""

    def test_on_step_complete_feeds_u3(self):
        from android_world.agents.memory.memory_agent import MemoryAugmentedAgent

        agent = MemoryAugmentedAgent.__new__(MemoryAugmentedAgent)
        agent.enable_u1 = False
        agent.enable_u2 = False
        agent.enable_u3 = True
        agent.u1 = None
        agent.u2 = None
        agent._current_goal = "Create a note"
        agent.u3 = mock.Mock()
        agent.u3.record_transition = mock.Mock()

        agent._on_step_complete({
            "before_ui_elements_list": "Markor main screen text",
            "before_ui_elements": [],
            "after_ui_elements_list": "Markor editor screen text",
            "after_ui_elements": [],
            "action_output_json": mock.Mock(action_type="click", index=3),
        })

        agent.u3.record_transition.assert_called_once()
        call = agent.u3.record_transition.call_args
        self.assertEqual(call.kwargs.get("action_summary"), "clicked 3")
        self.assertEqual(call.kwargs.get("task"), "Create a note")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: FAIL (either `_on_step_complete` doesn't feed U3, or `_current_goal` doesn't exist yet)

- [ ] **Step 3: Add page data to M3A.step()**

In `android_world/agents/memory/m3a.py` `step()`:

a) At the top of `step()` (right after `logging.info('----------step %s----------', ...)`), set the current goal:

```python
    self._current_goal = goal
```

b) After `before_ui_elements_list = _generate_ui_elements_description_list(...)`, store it:

```python
    step_data['before_ui_elements_list'] = before_ui_elements_list
```

c) Where the after state is fetched (`after_ui_elements = state.ui_elements` around the post-action section), store both:

```python
    after_ui_elements_list = _generate_ui_elements_description_list(
        after_ui_elements, logical_screen_size
    )
    step_data['after_ui_elements'] = after_ui_elements
    step_data['after_ui_elements_list'] = after_ui_elements_list
```

NOTE: `after_ui_elements_list` is already computed in the current code (line ~547). Just add the two `step_data[...]` assignments.

- [ ] **Step 4: Feed transitions in MemoryAugmentedAgent._on_step_complete**

In `android_world/agents/memory/memory_agent.py`, extend `_on_step_complete` to also feed U3 (keep the existing U1 update):

```python
  def _on_step_complete(self, step_data: dict[str, Any]) -> None:
    """Update U1 task state and feed U3 page graph from a completed step."""
    if self.enable_u1 and self.u1 is not None:
      before_ui_elements = step_data.get("before_ui_elements", [])
      app, page = extract_app_from_elements(before_ui_elements)
      action = step_data.get("action_output_json")
      effect = _action_effect_str(action)
      update_task_state(
          self.u1,
          current_app=app or None,
          current_page=page or None,
          last_action={"action_type": getattr(action, "action_type", "?")},
          last_effect=effect,
          failure=False,
      )

    if self.enable_u3 and self.u3 is not None:
      before_elements = step_data.get("before_ui_elements", [])
      after_elements = step_data.get("after_ui_elements", [])
      before_list = step_data.get("before_ui_elements_list", "")
      after_list = step_data.get("after_ui_elements_list", "")
      action = step_data.get("action_output_json")
      before_app, _ = extract_app_from_elements(before_elements)
      after_app, _ = extract_app_from_elements(after_elements)
      goal = getattr(self, "_current_goal", "")
      before_summary = build_screen_summary(
          goal, before_list, current_app=before_app or "")
      after_summary = build_screen_summary(
          goal, after_list, current_app=after_app or "")
      self.u3.record_transition(
          before_summary=before_summary,
          action_summary=_action_effect_str(action),
          task=goal,
          after_summary=after_summary,
          before_app=before_app,
          after_app=after_app,
      )
```

Add the import at the top of `memory_agent.py`:

```python
from android_world.agents.memory.environment import (
    EnvKnowledge, build_screen_summary,
)
```

(Replace the existing single-name import `from android_world.agents.memory.environment import EnvKnowledge`.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: PASS (10 tests)

- [ ] **Step 6: Verify existing tests still pass and agent imports cleanly**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_episodic -v`
Expected: PASS (10 tests)

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -c "from android_world.agents.memory_agent import MemoryAugmentedAgent; print('OK')"`
Expected: prints `OK` (harmless google.generativeai FutureWarning may appear)

- [ ] **Step 7: Commit**

```bash
git add android_world/agents/memory/m3a.py android_world/agents/memory/memory_agent.py android_world/agents/memory/test_page_graph.py
git commit -m "feat: feed page transitions into U3 from the agent step hook"
```

---

### Task 5: Package exports + full regression

**Files:**
- Modify: `android_world/agents/memory/__init__.py`

- [ ] **Step 1: Update the package exports**

Add the page graph types to `android_world/agents/memory/__init__.py`:

```python
from android_world.agents.memory.dms_bridge import (
    DMSConfig, MemoryBank, MemoryEntry, ObsAct, Plan, RetrievalResult,
)
from android_world.agents.memory.environment import EnvKnowledge, build_screen_summary
from android_world.agents.memory.episodic import EpisodicMemory, ObsAct
from android_world.agents.memory.page_graph import PageEdge, PageGraph, PageNode
from android_world.agents.memory.task_state import (
    TaskState,
    extract_app_from_elements,
    format_u1_context,
    init_task_state,
    update_task_state,
)
```

- [ ] **Step 2: Verify package imports**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -c "from android_world.agents.memory import PageGraph, EnvKnowledge; print('package OK')"`
Expected: prints `package OK`

- [ ] **Step 3: Run full unit test suite (both files)**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_page_graph android_world.agents.memory.test_episodic -v`
Expected: PASS (20 tests total — 10 + 10)

- [ ] **Step 4: Commit**

```bash
git add android_world/agents/memory/__init__.py
git commit -m "feat: export U3 page graph API from memory package"
```

---

### Task 6: Offline smoke — verify the learned graph survives a reload and retrieves

**Files:**
- None (uses a throwaway `python -c` snippet)

- [ ] **Step 1: Run the offline end-to-end smoke**

```bash
cd C:\Users\WRQ\Desktop\androidworld && python -c "
import tempfile
from android_world.agents.memory import PageGraph, EnvKnowledge, build_screen_summary

d = tempfile.mkdtemp()
ek = EnvKnowledge(rag_url='', persist_dir=d)
# Use the real SentenceTransformer embedder (downloads all-MiniLM-L6-v2 on first run).
ek._graph.set_embedder(ek._graph._ensure_embedder())

# Simulate two successful transitions the agent made in a prior session.
ek.record_transition(
    before_summary=build_screen_summary('Create a note', 'New note button', current_app='net.gsantner.markor', current_page='main'),
    action_summary='clicked 3',
    task='Create a note',
    after_summary=build_screen_summary('Create a note', 'Editor text field', current_app='net.gsantner.markor', current_page='editor'),
)
ek.record_transition(
    before_summary=build_screen_summary('Create a note', 'Editor text field', current_app='net.gsantner.markor', current_page='editor'),
    action_summary='typed content',
    task='Create a note',
    after_summary=build_screen_summary('Create a note', 'Save icon', current_app='net.gsantner.markor', current_page='editor'),
)

# New session: reload the graph from disk and retrieve on a similar screen.
ek2 = EnvKnowledge(rag_url='', persist_dir=d)
hint = ek2.retrieve_hint('Create a note in Markor', 'New note button', current_app='net.gsantner.markor', current_page='main')
print('HINT:', hint)
assert 'clicked 3' in hint, hint
print('OK: learned graph survived reload and retrieved')
"
```

Expected: `HINT:` line containing `clicked 3`, then `OK: learned graph survived reload and retrieved`. (First run downloads the embedding model; subsequent runs are offline.)

- [ ] **Step 2: Confirm the emulator run is untouched by this change**

The U2 two-round protocol (`python scripts/test_u1_u2.py --tasks=MarkorCreateNote --n=1 --seed=30`) is unaffected — this plan only adds a U3-only path. No regression expected. Optionally re-run it if you want full confidence; otherwise note that Task 6 is the offline smoke.

---

## Self-Review

**Spec coverage:** The PG-Agent §3.1 online graph-building is covered: Task 1 (nodes/edges/add_transition/merge/persistence), Task 2 (BFS guidelines retrieval matching eq. 10-12), Task 3 (EnvKnowledge composes local graph + optional remote RAG, non-blocking remote failures), Task 4 (agent feeds each successful page transition via the hook). The review-identified gap — "no write channel for agent-discovered environment patterns" — is closed. Deliberate deviation from the paper: node merge uses embedding cosine (deterministic, no local MLLM) instead of the dual-level MLLM check, to keep U3 pure data consistent with the existing U1/U2 modules.

**Placeholder scan:** Every step has concrete code and exact commands. No TBD/TODO. The smoke test in Task 6 is a complete, runnable `python -c` snippet.

**Type consistency:** `PageGraph(persist_dir, embedder, merge_threshold, hit_threshold)` consistent across Tasks 1-3. `EnvKnowledge(rag_url, persist_dir, embedder, top_k, bfs_layers, max_guidelines, timeout)` consistent with `memory_agent.py`'s existing `EnvKnowledge(rag_url=rag_url)` call (new params are optional defaults). `record_transition(before_summary, action_summary, task, after_summary, before_app, after_app)` consistent between EnvKnowledge (Task 3) and the agent hook (Task 4). `retrieve_guidelines(query_summary, top_k, bfs_layers)` consistent between PageGraph (Task 2) and EnvKnowledge (Task 3). `build_screen_summary(goal, ui_elements_list, *, current_app, current_page)` reused as-is. M3A's new `_current_goal` attribute is set in `step()` (Task 4) and read in the agent hook.
