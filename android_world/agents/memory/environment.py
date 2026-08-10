"""U3 Environment Knowledge — AutoDL page-graph RAG only.

U3 stores / retrieves environment knowledge via the remote AutoDL service
(RAG_URL / --rag_url).  There is no local page-graph or local embedding
fallback: missing URL or remote failures raise immediately.
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

from android_world.agents.memory.autodl_embedding import AutoDLEmbeddingBackend
from android_world.agents.memory.dms_bridge import EmbeddingBackend


def build_screen_summary(
    ui_elements_list: str,
    *,
    current_app: str = "",
    current_page: str = "",
    max_ui_chars: int = 1500,
) -> str:
  """Build a text screen summary S_It for RAG retrieve (no extra LLM call).

  Page identity is pure screen state (app/page/UI dump) — the task goal is
  intentionally NOT part of a node's summary, so the same physical page under
  different tasks merges into one node (PG-Agent §3.1 node semantics).  Task
  context lives on the graph edge, not the node.
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
  """U3: retrieve / write page-graph guidelines via AutoDL only."""

  def __init__(
      self,
      rag_url: str | None = None,
      persist_dir: str = "",
      embedder: EmbeddingBackend | None = None,
      top_k: int = 4,
      bfs_layers: int = 3,
      max_guidelines: int = 12,
      timeout: float = 30.0,
      client: Any | None = None,
  ):
    del persist_dir  # No local page-graph store; AutoDL is the only backend.
    resolved = (rag_url if rag_url is not None else os.environ.get("RAG_URL", "")).strip()
    # Unit tests may inject an explicit client and/or embedder with a placeholder URL.
    if not resolved and client is None:
      raise ValueError(
          "U3 requires AutoDL rag_url (--rag_url or RAG_URL env). "
          "Local page-graph / embedding fallback is disabled."
      )
    self.rag_url = resolved or "http://test.invalid"
    self.top_k = top_k
    self.bfs_layers = bfs_layers
    self.max_guidelines = max_guidelines
    self._client = client if client is not None else RagClient(
        base_url=self.rag_url, timeout=timeout
    )
    self._last_raw: dict[str, Any] | None = None
    # Embedder kept for API compatibility / tests; production retrieve uses AutoDL /retrieve.
    if embedder is None:
      embedder = AutoDLEmbeddingBackend(rag_url=self.rag_url)
    self._embedder = embedder

  def record_transition(
      self,
      before_summary: str,
      action_summary: str,
      task: str,
      after_summary: str,
      before_app: str = "",
      after_app: str = "",
  ) -> None:
    """Push one transition to AutoDL. Failures raise (no local fallback)."""
    del before_app, after_app
    self._client.add_transition(
        before_summary=before_summary,
        action_summary=action_summary,
        task=task,
        after_summary=after_summary,
    )

  def retrieve_hint(
      self,
      ui_elements_list: str,
      *,
      current_app: str = "",
      current_page: str = "",
  ) -> str:
    """Return prompt-ready guidelines from AutoDL only.

    Empty screen summary skips the remote call (AutoDL rejects empty summary).
    Remote failures raise — they are not swallowed into a local graph.
    """
    summary = build_screen_summary(
        ui_elements_list,
        current_app=current_app,
        current_page=current_page,
    )
    if not summary.strip():
      return ""
    raw = self._client.retrieve(
        summary,
        top_k=self.top_k,
        bfs_layers=self.bfs_layers,
        max_guidelines=self.max_guidelines,
    )
    self._last_raw = raw
    remote = list(raw.get("guidelines") or [])
    if not remote:
      return ""
    return self._client.format_guidelines_for_prompt(remote)
