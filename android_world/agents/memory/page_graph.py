"""U3 Online Page Graph — learn app/page transitions from agent experience.

Faithful to PG-Agent (§3.1): the graph starts empty and grows as the agent
executes.  Each page transition  before --action--> after  becomes a directed
edge carrying the action summary and the task it served.  Pages are merged by
embedding cosine similarity (deterministic — no local MLLM, keeping U3 pure
data) instead of the paper's dual-level MLLM check.

U3 is pure data infrastructure: no LLM calls, no environment interaction.
"""

from __future__ import annotations

import hashlib
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
        hit_threshold: float = 0.5,
    ):
        self.persist_dir = persist_dir
        self.merge_threshold = merge_threshold
        self.hit_threshold = hit_threshold
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
            # Degenerate page — stable stub id so edges still link across
            # processes (hash() is salted per-process; hashlib is stable).
            key = f"empty:{app}".encode("utf-8")
            return "empty:" + hashlib.md5(key).hexdigest()[:12]
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
                    # Repeat the action by edge count (frequency emphasis) —
                    # this also differentiates dedup keys: a count-1 and a
                    # count-3 edge for the same action yield different
                    # (actions, tasks) tuples, so they aren't collapsed.
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

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        p = path or (os.path.join(self.persist_dir, "page_graph.json") if self.persist_dir else "")
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        state = {
            "merge_threshold": self.merge_threshold,
            "hit_threshold": self.hit_threshold,
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
        self.hit_threshold = state.get("hit_threshold", 0.5)
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
