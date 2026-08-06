"""HTTP client for the AutoDL PG-Agent RAG service (run on local PC)."""

from __future__ import annotations

import os
from typing import Any

import requests


class RagClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("RAG_URL", "http://127.0.0.1:18180")).rstrip(
            "/"
        )
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def retrieve(
        self,
        summary: str,
        top_k: int = 4,
        bfs_layers: int = 3,
        max_guidelines: int = 20,
    ) -> dict[str, Any]:
        payload = {
            "summary": summary,
            "top_k": top_k,
            "bfs_layers": bfs_layers,
            "max_guidelines": max_guidelines,
        }
        r = requests.post(
            f"{self.base_url}/retrieve",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def embed(self, texts: list[str]) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/embed",
            json={"texts": texts},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def add_transition(
        self,
        before_summary: str,
        action_summary: str,
        task: str,
        after_summary: str,
        merge_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Incrementally add ONE page transition to the remote graph.

        The AutoDL server merges new pages by embedding similarity and
        appends any new vectors to its FAISS index (no rebuild).  Returns the
        server's delta report.
        """
        payload = {
            "before_summary": before_summary,
            "action_summary": action_summary,
            "task": task,
            "after_summary": after_summary,
            "merge_threshold": merge_threshold,
        }
        r = requests.post(
            f"{self.base_url}/add_transition",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def add_transitions(
        self,
        transitions: list[dict[str, Any]],
        merge_threshold: float = 0.85,
    ) -> dict[str, Any]:
        """Batch-add many transitions in one call (single persist at the end)."""
        r = requests.post(
            f"{self.base_url}/add_transitions",
            json={
                "transitions": [
                    {
                        "before_summary": t.get("before_summary", ""),
                        "action_summary": t.get("action_summary", ""),
                        "task": t.get("task", ""),
                        "after_summary": t.get("after_summary", ""),
                    }
                    for t in transitions
                ],
                "merge_threshold": merge_threshold,
            },
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def format_guidelines_for_prompt(self, guidelines: list[dict[str, Any]]) -> str:
        """Render guidelines as text for Sub-Task Planning / Decision prompts."""
        if not guidelines:
            return "No guidelines retrieved."
        lines: list[str] = ["Retrieved page-graph guidelines:"]
        for i, g in enumerate(guidelines, 1):
            actions = " -> ".join(g.get("actions") or [])
            tasks = ", ".join(g.get("tasks") or [])
            lines.append(f"{i}. Actions: [{actions}] | Achievable tasks: [{tasks}]")
        return "\n".join(lines)


def retrieve_guidelines(screen_summary: str, **kwargs: Any) -> list[dict[str, Any]]:
    """Convenience helper for agent loops."""
    client = RagClient()
    result = client.retrieve(screen_summary, **kwargs)
    return list(result.get("guidelines") or [])
