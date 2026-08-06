"""
PG-Agent local integration snippet (copy into your android_world agent).

This module does NOT call adb/AVD. Wire `inject_guidelines` after screen
summary and before sub-task planning. Start SSH tunnel only when you are
ready; set RAG_URL=http://127.0.0.1:18180.
"""

from __future__ import annotations

# --- paste near your agent imports ---
# from android_world_hook import inject_guidelines

INTEGRATION_SNIPPET = '''
# Inside your per-step loop (pseudocode aligned with PG-Agent):

state = env.get_state()                  # local AVD / android_world
screen_summary = mllm.summarize(state)   # S_It

# >>> RAG hook (AutoDL via SSH -L) <<<
from android_world_hook import inject_guidelines
guidelines_text, rag_raw = inject_guidelines(
    screen_summary,
    top_k=4,
    bfs_layers=3,
    max_guidelines=20,
)

global_plan = global_planner(state, task)           # P^G
observation = observation_agent(state, task, hist)  # O
subplan = subtask_planner(                          # P^S  <- inject guidelines
    state, observation, global_plan, guidelines_text, hist
)
action = decision_agent(                            # D    <- inject guidelines
    state, observation, subplan, guidelines_text, hist
)
env.execute_action(action)                          # local ADB — only when YOU run it
'''


def print_integration_help() -> None:
    print(INTEGRATION_SNIPPET)


if __name__ == "__main__":
    print_integration_help()
