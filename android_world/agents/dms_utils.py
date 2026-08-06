"""Darwinian Memory System (DMS) — Planner prompts and utility functions.

Integrates the DMS library (others/darwinian_memory) with AndroidWorld's
agent framework.  Provides the Planner prompt template, output parser,
and bridge helpers that convert between M3A-style step data and DMS
Plan / ObsAct types.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Optional

# Ensure repo root is importable (for direct invocation / test suites).
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
  sys.path.insert(0, _repo_root)

from others.darwinian_memory.memory_entry import Plan, ObsAct


# ── Planner Prompt (§3.1 of DMS paper) ────────────────────────────────

PLANNER_SYSTEM_PROMPT = """You are a hierarchical task planner for Android GUI automation.
Your job is to decompose a high-level user goal into a sequence of 2-4 sub-tasks.

For each sub-task, provide:
  - precondition: the expected UI state before starting this sub-task
  - goal: what this sub-task should achieve (a concrete, verifiable UI outcome)

RULES:
1. Each sub-task represents a logical phase of the workflow. Typical phases: (a) navigate to the relevant app, (b) execute the core operation, (c) verify or clean up.
2. Each goal must be DISTINCT — never duplicate or near-duplicate another goal.
3. Each goal must be independently verifiable.
4. Use app name and UI terms the agent can ground to actions.

Generic examples (from domains outside the benchmark):
- Send an email:
  [{"precondition": "Home screen", "goal": "Open email app and start composing a new message"}, {"precondition": "Compose screen open", "goal": "Enter recipient, subject and body, then send"}, {"precondition": "Message sent", "goal": "Return to home screen"}]
- Transfer money in a banking app:
  [{"precondition": "Home screen", "goal": "Open banking app and navigate to transfer page"}, {"precondition": "Transfer page open", "goal": "Enter amount and target account, confirm the transfer"}, {"precondition": "Transfer confirmed", "goal": "Return to home screen"}]

Output ONLY a JSON array of objects with "precondition" and "goal" keys."""


def parse_planner_output(raw_output: str) -> list[Plan]:
  """Parse the Planner LLM response into a list of Plan objects.

  Handles both pure JSON arrays and markdown-fenced JSON blocks.

  Args:
    raw_output: Raw text response from the Planner LLM.

  Returns:
    List of Plan objects.  Returns empty list if parsing fails.
  """
  # Try to extract JSON array from markdown code fences first
  fence_match = re.search(
      r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', raw_output, re.IGNORECASE
  )
  if fence_match:
    json_str = fence_match.group(1)
  else:
    # Fall back: find the first [...] in the text
    bracket_match = re.search(r'\[[\s\S]*\]', raw_output)
    if bracket_match:
      json_str = bracket_match.group(0)
    else:
      return []

  try:
    items = json.loads(json_str)
  except json.JSONDecodeError:
    return []

  if not isinstance(items, list):
    return []

  plans = []
  for item in items:
    if not isinstance(item, dict):
      continue
    pre = item.get('precondition', '').strip()
    goal = item.get('goal', '').strip()
    if goal:
      plans.append(Plan(precondition=pre, goal=goal))
  return plans


# ── Sub-plan action prompt builder ────────────────────────────────────

SUB_PLAN_ACTION_PROMPT_TEMPLATE = """You are an agent who can operate an Android phone on behalf of a user.
Based on user's goal/request, you may
- Answer back if the request/goal is a question (or a chat message), like user asks "What is my schedule for today?".
- Complete some tasks described in the requests/goals by performing actions (step by step) on the phone.

When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot (including the original screenshot and the same screenshot with bounding boxes and numeric indexes added to some UI elements) and a history of what you have done (in text). Based on these pieces of information and the goal, you must choose to perform one of the action in the following list (action description followed by the JSON format) by outputing the action in the correct JSON format.
- If you think the task has been completed, finish the task by using the status action with complete as goal_status: `{{"action_type": "status", "goal_status": "complete"}}`
- If you think the task is not feasible (including cases like you don't have enough information or can not perform some necessary actions), finish by using the `status` action with infeasible as goal_status: `{{"action_type": "status", "goal_status": "infeasible"}}`
- Answer user's question: `{{"action_type": "answer", "text": "<answer_text>"}}`
- Click/tap on an element on the screen. We have added marks (bounding boxes with numeric indexes on their TOP LEFT corner) to most of the UI elements in the screenshot, use the numeric index to indicate which element you want to click: `{{"action_type": "click", "index": <target_index>}}`.
- Long press on an element on the screen, similar with the click action above, use the numeric label on the bounding box to indicate which element you want to long press: `{{"action_type": "long_press", "index": <target_index>}}`.
- Type text into a text field (this action contains clicking the text field, typing in the text and pressing the enter, so no need to click on the target field to start), use the numeric label on the bounding box to indicate the target text field: `{{"action_type": "input_text", "text": <text_input>, "index": <target_index>}}`
- Press the Enter key: `{{"action_type": "keyboard_enter"}}`
- Navigate to the home screen: `{{"action_type": "navigate_home"}}`
- Navigate back: `{{"action_type": "navigate_back"}}`
- Scroll the screen or a scrollable UI element in one of the four directions, use the same numeric index as above if you want to scroll a specific UI element, leave it empty when scroll the whole screen: `{{"action_type": "scroll", "direction": <up, down, left, right>, "index": <optional_target_index>}}`
- Open an app (nothing will happen if the app is not installed): `{{"action_type": "open_app", "app_name": <name>}}`
- Wait for the screen to update: `{{"action_type": "wait"}}`

The overall user goal is: {overall_goal}
Current sub-task: {sub_goal}

Here is a history of what you have done so far for this sub-task:
{history}

{memory_hint}

The current screenshot and the same screenshot with bounding boxes and labels added are also given to you.
Here is a list of detailed information for some of the UI elements (notice that some elements in this list may not be visible in the current screen and so you can not interact with it, can try to scroll the screen to reveal it first), the numeric indexes are consistent with the ones in the labeled screenshot:
{ui_elements}

Here are some useful guidelines you need to follow:
General:
- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it doesn't (you can see that from the history), SWITCH to other solutions.
- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), you can just complete the task.
Action Related:
- Use the `open_app` action whenever you want to open an app (nothing will happen if the app is not installed), do not use the app drawer to open an app unless all other ways have failed.
- Use the `input_text` action whenever you want to type something (including password) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- For `click`, `long_press` and `input_text`, the index parameter you pick must be VISIBLE in the screenshot and also in the UI element list given to you.
- Consider exploring the screen by using the `scroll` action with different directions to reveal additional content.
- The direction parameter for the `scroll` action can be confusing sometimes as it's opposite to swipe, for example, to view content at the bottom, the `scroll` direction should be set to "down". If one does not work, try the opposite as well.

Now output an action from the above list in the correct JSON format, following the reason why you do that. Your answer should look like:
Reason: ...
Action: {{"action_type":...}}

Your Answer:
"""


def build_sub_plan_action_prompt(
    overall_goal: str,
    sub_goal: str,
    history: list[str],
    ui_elements: str,
    memory_hint: str = '',
) -> str:
  """Build the action-selection prompt for executing a single sub-plan step.

  Args:
    overall_goal: The user's top-level task description.
    sub_goal: The current sub-plan goal text.
    history: Summaries of previous steps within this sub-plan.
    ui_elements: Formatted UI element descriptions.
    memory_hint: Optional hint from DMS memory (e.g. cached action guidance).

  Returns:
    Complete prompt string for the Actor LLM.
  """
  if history:
    history_text = '\n'.join(history)
  else:
    history_text = 'You just started this sub-task, no action has been performed yet.'

  if memory_hint:
    hint_text = f'Memory hint (previously successful approach): {memory_hint}'
  else:
    hint_text = ''

  return SUB_PLAN_ACTION_PROMPT_TEMPLATE.format(
      overall_goal=overall_goal,
      sub_goal=sub_goal,
      history=history_text,
      memory_hint=hint_text,
      ui_elements=ui_elements if ui_elements else 'Not available',
  )


# ── Bridge helpers ─────────────────────────────────────────────────────

def step_data_to_obs_act(
    step_data: dict[str, Any], step_index: int = 0
) -> ObsAct:
  """Convert M3A-style step_data into a DMS ObsAct.

  The observation is the raw screenshot pixels; the action is the parsed
  JSONAction stored in step_data['action_output_json'].
  """
  return ObsAct(
      observation=step_data.get('raw_screenshot'),
      action=step_data.get('action_output_json'),
      step_index=step_index,
  )


def plan_to_key(plan: Plan) -> str:
  """Derive a stable string key for a plan — used for DMS plan-stats tracking."""
  return plan.goal.strip().lower()


def summarize_trajectory(trajectory: list[ObsAct]) -> str:
  """Create a short text summary of a trajectory for memory hints."""
  if not trajectory:
    return ''
  parts = []
  for oa in trajectory:
    action = oa.action
    if action is None:
      continue
    action_type = getattr(action, 'action_type', str(action))
    parts.append(str(action_type))
  return ' → '.join(parts) if parts else ''
