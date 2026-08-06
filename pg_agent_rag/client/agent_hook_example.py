"""
Example hook for local android_world / PG-Agent loop.

Copy `client/rag_client.py` (or this whole `client/` folder) to your local
android_world project, start the SSH tunnel, then call `step_with_rag`
after you obtain a screen summary from the MLLM.
"""

from __future__ import annotations

from typing import Any, Callable

from rag_client import RagClient


def step_with_rag(
    screen_summary: str,
    *,
    plan_fn: Callable[[str, str], Any],
    rag: RagClient | None = None,
    top_k: int = 4,
    bfs_layers: int = 3,
    max_guidelines: int = 20,
) -> Any:
    """
    After MLLM produces `screen_summary` (S_It), retrieve guidelines and
    pass them into your planner/decision function.

    plan_fn(screen_summary, guidelines_text) -> action / plan result
    """
    client = rag or RagClient()
    result = client.retrieve(
        screen_summary,
        top_k=top_k,
        bfs_layers=bfs_layers,
        max_guidelines=max_guidelines,
    )
    guidelines_text = client.format_guidelines_for_prompt(result["guidelines"])
    return plan_fn(screen_summary, guidelines_text)


# --- Minimal local demo (no AVD required) ---------------------------------

def _demo_plan(summary: str, guidelines_text: str) -> dict[str, str]:
    print("--- screen summary ---")
    print(summary)
    print("--- guidelines ---")
    print(guidelines_text)
    # Fake decision: pick first guideline's first action if present
    first_line = guidelines_text.splitlines()[1] if "\n" in guidelines_text else ""
    return {"summary": summary, "decision_hint": first_line}


if __name__ == "__main__":
    import os

    # Local PC (after SSH -L): 18180. Direct on AutoDL for debug: set RAG_URL=http://127.0.0.1:6006
    os.environ.setdefault("RAG_URL", "http://127.0.0.1:18180")
    print(f"RAG_URL={os.environ['RAG_URL']}")
    out = step_with_rag(
        "I see the Android Settings list with Network & internet and Apps.",
        plan_fn=_demo_plan,
    )
    print("--- result ---")
    print(out)
