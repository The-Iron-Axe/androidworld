"""
Drop-in helper for local android_world agent loops.

Place `rag_client.py` next to your agent module (or add this folder to PYTHONPATH),
export RAG_URL=http://127.0.0.1:18180 after starting the SSH tunnel, then:

    from android_world_hook import inject_guidelines

    summary = mllm.summarize(screenshot)
    guidelines_text, raw = inject_guidelines(summary)
    action = mllm.decide(screenshot, summary, guidelines_text, history)
"""

from __future__ import annotations

from typing import Any

from rag_client import RagClient


def inject_guidelines(
    screen_summary: str,
    *,
    rag_url: str | None = None,
    top_k: int = 4,
    bfs_layers: int = 3,
    max_guidelines: int = 20,
) -> tuple[str, dict[str, Any]]:
    client = RagClient(base_url=rag_url)
    raw = client.retrieve(
        screen_summary,
        top_k=top_k,
        bfs_layers=bfs_layers,
        max_guidelines=max_guidelines,
    )
    text = client.format_guidelines_for_prompt(raw.get("guidelines") or [])
    return text, raw


def pg_agent_step_template(
    get_screenshot,
    summarize_fn,
    decide_fn,
    execute_fn,
    task: str,
    history: list | None = None,
    max_steps: int = 20,
) -> list:
    """
    Skeleton of one PG-Agent episode with remote RAG.

    Callables (all run on local PC except RAG HTTP):
      get_screenshot() -> image
      summarize_fn(image) -> str
      decide_fn(image, summary, guidelines_text, task, history) -> action
      execute_fn(action) -> None
    """
    history = list(history or [])
    for _ in range(max_steps):
        image = get_screenshot()
        summary = summarize_fn(image)
        guidelines_text, _ = inject_guidelines(summary)
        action = decide_fn(image, summary, guidelines_text, task, history)
        execute_fn(action)
        history.append({"summary": summary, "action": action})
        if getattr(action, "name", None) in ("status_complete", "complete", "done"):
            break
    return history
