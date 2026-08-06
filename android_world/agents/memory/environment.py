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
from android_world.agents.memory.autodl_embedding import AutoDLEmbeddingBackend
from android_world.agents.memory.dms_bridge import EmbeddingBackend


def build_screen_summary(
    ui_elements_list: str,
    *,
    current_app: str = "",
    current_page: str = "",
    max_ui_chars: int = 1500,
) -> str:
  """Build a text screen summary S_It for RAG retrieve / graph nodes (no extra LLM call).

  Page identity is pure screen state (app/page/UI dump) — the task goal is
  intentionally NOT part of a node's summary, so the same physical page under
  different tasks merges into one node (PG-Agent §3.1 node semantics).  Task
  context lives on the graph edge (PageEdge.task), not the node.
  """
  parts: list[str] = []
  loc = []
  if current_app:
    loc.append(f"app={current_app}")
  if current_page:
    loc.append(f"page={current_page}")
  if loc:
    parts.append("Current screen: " + ", ".join(loc) + ".")
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
    if embedder is None:
      embedder = AutoDLEmbeddingBackend(rag_url=self.rag_url)
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
    """Feed a page transition into the page graph (PG-Agent §3.1).

    The primary store is the remote AutoDL graph: each transition is pushed
    incrementally (POST /add_transition), where new pages are merged by
    embedding similarity and new vectors are appended to FAISS without a
    rebuild.  The local graph is kept as a lightweight fallback cache so the
    agent still works offline; its updates mirror the same transition.
    """
    # Local fallback graph (cheap, offline-safe).
    self._graph.add_transition(
        before_summary, action_summary, task, after_summary,
        before_app=before_app, after_app=after_app,
    )
    # Remote incremental graph (real-time U3). Non-blocking on failure.
    if self.rag_url:
      try:
        self._client.add_transition(
            before_summary=before_summary,
            action_summary=action_summary,
            task=task,
            after_summary=after_summary,
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[U3] add_transition remote failed ({e}); local graph only")

  def retrieve_hint(
      self,
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
        ui_elements_list,
        current_app=current_app,
        current_page=current_page,
    )
    local = self._graph.retrieve_guidelines(summary, top_k=self.top_k, bfs_layers=self.bfs_layers)
    remote: list[dict[str, Any]] = []
    # AutoDL /retrieve rejects empty summary (422 string_too_short). Skip
    # remote until we have a non-empty screen summary (common on step 0).
    if self.rag_url and summary.strip():
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
