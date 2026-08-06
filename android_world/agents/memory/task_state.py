"""U1 Task State Memory — structured task-progress tracking.

U1 is the lightest-weight memory class.  It maintains structured, per-step
read/write state about the current task: what app/page we are on, what sub-goal
we are pursuing, what has been completed and what is still pending, the last
action and its observed effect, and a failure counter.

U1 is pure data infrastructure — no LLM calls, no agent inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskState:
  """Structured task state tracked across an episode.

  Attributes:
    goal: The original user goal for this task.
    current_app: Package name of the currently visible app ("" if unknown).
    current_page: Human-readable page/activity name ("" if unknown).
    current_subgoal: What the agent is currently trying to accomplish.
    completed: List of sub-goals that have been finished.
    pending: Sub-goals still to do (head = current).
    constraints: Named constraints discovered during execution (e.g. "date format must be YYYY-MM-DD").
    last_action: The most recent action dict (action_type + params).
    last_effect: One-line summary of what changed after last_action.
    failure_count: Consecutive failures on the current sub-goal.
    step_count: Total steps taken so far.
  """

  goal: str = ""
  current_app: str = ""
  current_page: str = ""
  current_subgoal: str = ""
  completed: list[str] = field(default_factory=list)
  pending: list[str] = field(default_factory=list)
  constraints: dict[str, str] = field(default_factory=dict)
  last_action: dict[str, Any] = field(default_factory=dict)
  last_effect: str = ""
  failure_count: int = 0
  step_count: int = 0


def init_task_state(goal: str) -> TaskState:
  """Create a fresh TaskState seeded with the user goal."""
  return TaskState(goal=goal)


def update_task_state(
    state: TaskState,
    *,
    current_app: str | None = None,
    current_page: str | None = None,
    current_subgoal: str | None = None,
    completed: str | None = None,
    pending: list[str] | None = None,
    constraint: tuple[str, str] | None = None,
    last_action: dict[str, Any] | None = None,
    last_effect: str | None = None,
    failure: bool = False,
) -> None:
  """Update U1 state fields in place.  Only supplied fields are modified.

  Args:
    state: The TaskState to mutate.
    current_app: New current app package name.
    current_page: New current page/activity description.
    current_subgoal: New current sub-goal text.
    completed: A sub-goal that was just finished (appended to completed list).
    pending: Replace the pending sub-goal queue wholesale.
    constraint: A (key, value) constraint to record.
    last_action: Dict representation of the action just taken.
    last_effect: One-line description of what happened.
    failure: If True, increment failure_count; otherwise reset it to 0.
  """
  if current_app is not None:
    state.current_app = current_app
  if current_page is not None:
    state.current_page = current_page
  if current_subgoal is not None:
    state.current_subgoal = current_subgoal
  if completed is not None:
    state.completed.append(completed)
    # Remove from pending if present
    if completed in state.pending:
      state.pending.remove(completed)
  if pending is not None:
    state.pending = pending
  if constraint is not None:
    state.constraints[constraint[0]] = constraint[1]
  if last_action is not None:
    state.last_action = last_action
  if last_effect is not None:
    state.last_effect = last_effect
  if failure:
    state.failure_count += 1
  else:
    state.failure_count = 0
  state.step_count += 1


def format_u1_context(state: TaskState) -> str:
  """Produce a compact inline string to inject into the action-selection prompt.

  Returns an empty string when there is nothing useful to report.
  """
  parts: list[str] = []

  if state.current_app:
    parts.append(f"Current app: {state.current_app}")
  if state.current_page:
    parts.append(f"Current page: {state.current_page}")
  if state.current_subgoal:
    parts.append(f"Current sub-goal: {state.current_subgoal}")

  if state.completed:
    parts.append(f"Completed: {', '.join(state.completed)}")
  if state.pending:
    parts.append(f"Pending: {', '.join(state.pending)}")

  if state.constraints:
    constraint_text = "; ".join(
        f"{k}: {v}" for k, v in state.constraints.items()
    )
    parts.append(f"Constraints: {constraint_text}")

  if state.last_action:
    action_type = state.last_action.get("action_type", "?")
    parts.append(f"Last action: {action_type}")
  if state.last_effect:
    parts.append(f"Last effect: {state.last_effect}")

  if state.failure_count > 0:
    parts.append(f"Consecutive failures: {state.failure_count}")

  if not parts:
    return ""
  return " | ".join(parts)


def extract_app_from_elements(
    ui_elements: list[Any],
) -> tuple[str, str]:
  """Heuristic: extract current app and page from UI elements.

  Uses the package_name of the first element that has one as the app,
  and the first meaningful text or content_description as the page hint.

  Args:
    ui_elements: List of UIElement objects from the accessibility tree.

  Returns:
    (app_package, page_hint) tuple.  Each may be "" if not detectable.
  """
  app = ""
  page = ""
  for el in ui_elements:
    pkg = getattr(el, "package_name", None) or ""
    if pkg and not app:
      app = pkg
    if not page:
      text = getattr(el, "text", None)
      cd = getattr(el, "content_description", None)
      if text and str(text).strip():
        page = str(text).strip()[:80]
      elif cd and str(cd).strip():
        page = str(cd).strip()[:80]
    if app and page:
      break
  return app, page
