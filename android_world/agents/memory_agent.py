"""Memory-Augmented M3A Agent.

Extends the M3A baseline with the five-class memory taxonomy (U1-U5).
Memory is pure data infrastructure — NOT a separate agent.

Usage (via run.py flags):
  --agent_name=m3a_qwen3_vl_32b_mem           # baseline (no memory)
  --agent_name=m3a_qwen3_vl_32b_mem --u1      # +U1 only
  --agent_name=m3a_qwen3_vl_32b_mem --u2      # +U2 only
  --agent_name=m3a_qwen3_vl_32b_mem --u3      # +U3 page-graph RAG
  --agent_name=m3a_qwen3_vl_32b_mem --u1 --u2 --u3
"""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a as m3a_lib
from android_world.agents import m3a_utils
from android_world.agents.memory.environment import (
    EnvKnowledge, build_screen_summary,
)
from android_world.agents.memory.episodic import EpisodicMemory, ObsAct, Plan
from android_world.agents.memory.procedural import ProceduralMemory
from android_world.agents.memory.task_state import (
    TaskState,
    extract_app_from_elements,
    format_u1_context,
    init_task_state,
    update_task_state,
)
from android_world.env import interface
from android_world.env import json_action


def _describe_element(ui_elements: Any, index: int | None) -> str:
  """Render an indexed UI element as semantic text (never its index).

  Priority: text > content_description > hint_text, capped at 40 chars.
  Falls back to the element's package when the element is missing or blank.

  `index` is the RAW UI-element list index — the same space the LLM's
  action.index uses.  m3a.py numbers elements with enumerate() over the raw
  list and only filters invisible ones out of the *text* (keeping the raw
  index), so do NOT filter ui_elements here or you'll misalign the index.
  """
  if ui_elements is None or index is None:
    return ""
  try:
    el = ui_elements[index]
  except (IndexError, TypeError):
    return ""
  for attr in ("text", "content_description", "hint_text"):
    val = getattr(el, attr, None) or ""
    val = str(val).strip()
    if val:
      return val[:40]
  pkg = getattr(el, "package_name", None) or ""
  pkg = str(pkg).strip()
  return f"element in {pkg}" if pkg else ""


def _action_effect_str(action: Any, ui_elements: Any = None) -> str:
  """Terse effect summary of an action for U1 tracking / U3 edge labels.

  Index-based actions render the target element semantically (text /
  content_description / hint_text) so summaries survive across episodes,
  screens, and tasks.  Falls back to the raw index only if the element
  cannot be described.
  """
  if action is None:
    return "no action"
  at = getattr(action, "action_type", str(action))
  index = getattr(action, "index", None)
  if at == "click":
    label = _describe_element(ui_elements, index)
    return f"clicked {label!r}" if label else f"clicked {index}"
  elif at == "input_text":
    text = getattr(action, "text", "")
    label = _describe_element(ui_elements, index)
    if label:
      return f"typed {text!r} into {label!r}"
    return f"typed {text!r}"
  elif at == "scroll":
    label = _describe_element(ui_elements, index)
    direction = getattr(action, "direction", "?")
    return f"scrolled {direction} on {label!r}" if label else f"scrolled {direction}"
  elif at == "open_app":
    return f"opened {getattr(action, 'app_name', '?')}"
  elif at in ("navigate_home", "navigate_back", "keyboard_enter", "wait"):
    return at.replace("_", " ")
  elif at == "long_press":
    label = _describe_element(ui_elements, index)
    return f"long-pressed {label!r}" if label else f"long-pressed {index}"
  elif at in ("status", "answer"):
    return f"{at} -> {getattr(action, 'goal_status', '') or getattr(action, 'text', '')}"
  return str(at)


class MemoryAugmentedAgent(m3a_lib.M3A):
  """M3A augmented with orthogonal memory modules.

  U1 (--u1): episode-level task state tracking — current app/page, failure
             counter, pending sub-goals.  Injected into each action prompt.
  U2 (--u2): cross-episode episodic memory — retrieves similar past
             trajectories via DMS dual-factor scoring.  Stores new trajectories
             on task completion.
  U3 (--u3): environment knowledge — PG-Agent page-graph guidelines retrieved
             from AutoDL RAG (RAG_URL) and injected into each action prompt.
  U4 (--u4): procedural skill memory — reusable parameterized skills mined
             from successful trajectories and injected as skill hints.

  U1/U2/U3/U4 are independent boolean flags for ablation.
  """

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      enable_u1: bool = False,
      enable_u2: bool = False,
      enable_u3: bool = False,
      enable_u4: bool = False,
      u2_persistence_dir: str = "",
      u3_persistence_dir: str = "",
      u4_persistence_dir: str = "",
      rag_url: str | None = None,
      name: str = "MemoryAugmentedAgent",
      wait_after_action_seconds: float = 2.0,
      screenshot_scale: float = 1.0,
  ):
    super().__init__(env, llm, name, wait_after_action_seconds)
    self.enable_u1 = enable_u1
    self.enable_u2 = enable_u2
    self.enable_u3 = enable_u3
    self.enable_u4 = enable_u4
    self.screenshot_scale = screenshot_scale

    # ── Memory state (pure data) ──
    self.u1: TaskState | None = None
    self.u2: EpisodicMemory | None = None
    self.u3: EnvKnowledge | None = None
    self.u4: ProceduralMemory | None = None
    if enable_u2:
      self.u2 = EpisodicMemory(persistence_dir=u2_persistence_dir)
    if enable_u3:
      self.u3 = EnvKnowledge(rag_url=rag_url, persist_dir=u3_persistence_dir)
    if enable_u4:
      self.u4 = ProceduralMemory(persistence_dir=u4_persistence_dir)
    # Goal buffered by _on_task_done/flush_memory; written by set_episode_success
    self._pending_trajectory_goal: str | None = None

    # ── U2 deterministic replay state (§3.2.2) ──
    self._replay_active = False
    self._replay_entry = None
    self._replay_index = 0
    self._replay_trajectory: list[ObsAct] = []

  # ── Lifecycle ───────────────────────────────────────────────────────

  def reset(self, go_home_on_reset: bool = False) -> None:
    super().reset(go_home_on_reset)
    self.u1 = None
    self._pending_trajectory_goal = None
    self._replay_active = False
    self._replay_entry = None
    self._replay_index = 0
    self._replay_trajectory = []

  # ── Hook: inject memory context into the action prompt ───────────────

  def _build_action_prompt(
      self,
      goal: str,
      history_lines: list[str],
      ui_elements_list: str,
  ) -> str:
    """Insert U1/U2/U3 memory blocks above the history."""
    memory_blocks: list[str] = []

    if self.enable_u1:
      if self.u1 is None:
        self.u1 = init_task_state(goal)
      u1_text = format_u1_context(self.u1)
      if u1_text:
        memory_blocks.append(f"## Task State (U1)\n{u1_text}")

    if self.enable_u2 and self.u2 is not None:
      hint = self.u2.retrieve_hint(goal)
      if hint:
        memory_blocks.append(f"## Memory Hint (U2)\nSimilar past trajectory: {hint}")

    if self.enable_u3 and self.u3 is not None:
      app = self.u1.current_app if self.u1 is not None else ""
      page = self.u1.current_page if self.u1 is not None else ""
      u3_text = self.u3.retrieve_hint(
          ui_elements_list,
          current_app=app,
          current_page=page,
      )
      if u3_text:
        memory_blocks.append(f"## Environment Knowledge (U3)\n{u3_text}")

    if self.enable_u4 and self.u4 is not None:
      app = self.u1.current_app if self.u1 is not None else ""
      page = self.u1.current_page if self.u1 is not None else ""
      u4_text = self.u4.retrieve_hint(goal, precondition=page)
      if u4_text:
        memory_blocks.append(f"## Procedural Skill (U4)\n{u4_text}")

    # Prepend memory blocks to the goal so they appear before the history
    # in the parent's prompt template.  This avoids any format-string issues
    # because the parent template only uses {goal} as a placeholder.
    if memory_blocks:
      augmented_goal = "\n\n".join(memory_blocks) + "\n\nGoal: " + goal
    else:
      augmented_goal = goal

    return m3a_lib._action_selection_prompt(
        augmented_goal,
        history_lines,
        ui_elements_list,
        self.additional_guidelines,
    )

  # ── Hook: update memory state after each successful step (U1, U3) ────

  def _on_step_complete(self, step_data: dict[str, Any]) -> None:
    """Update U1 task state and feed U3 page graph from a completed step."""
    if self.enable_u1 and self.u1 is not None:
      before_ui_elements = step_data.get("before_ui_elements", [])
      app, page = extract_app_from_elements(before_ui_elements)
      action = step_data.get("action_output_json")
      effect = _action_effect_str(action, before_ui_elements)
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
      before_app, before_page = extract_app_from_elements(before_elements)
      after_app, after_page = extract_app_from_elements(after_elements)
      goal = getattr(self, "_current_goal", "")
      before_summary = build_screen_summary(
          before_list, current_app=before_app or "", current_page=before_page or "")
      after_summary = build_screen_summary(
          after_list, current_app=after_app or "", current_page=after_page or "")
      self.u3.record_transition(
          before_summary=before_summary,
          action_summary=_action_effect_str(action, before_elements),
          task=goal,
          after_summary=after_summary,
          before_app=before_app,
          after_app=after_app,
      )

  # ── Hook: count failed steps in U1 ───────────────────────────────────

  def _on_step_failed(self, step_data: dict[str, Any]) -> None:
    """Increment U1 failure counter on a failed step."""
    del step_data
    if not self.enable_u1 or self.u1 is None:
      return
    update_task_state(self.u1, failure=True)

  # ── Hook: finalize memory when task is done ──────────────────────────

  def _on_task_done(self, goal: str, step_data: dict[str, Any]) -> None:
    """Buffer the trajectory when the agent declares the task done.

    The trajectory is NOT written to U2 yet — the real success signal comes
    from AndroidWorld's task evaluation (task.is_successful), which the runner
    provides via set_episode_success().  That call performs the final write so
    failed trajectories are not recorded as successful experiences.
    """
    del step_data
    if not self.enable_u2 or self.u2 is None:
      return
    self._pending_trajectory_goal = goal

  # ── Public: called by suite_utils after computing true task success ──

  def set_episode_success(self, success: bool) -> None:
    """Finalize the current episode's memory with the true outcome.

    Called by the evaluator after it computes task.is_successful().  Overrides
    the agent's own completion claim (which can be wrong) with ground truth.

    U2: feeds the global failure rate and flushes the trajectory.
    U4: successful episodes buffer their trajectory for skill mining; failed
        episodes drive score eviction.
    """
    if self.enable_u2 and self.u2 is not None:
      # Feed the global failure rate before flushing the trajectory.
      self.u2.record_episode_outcome(success)
      goal = self._pending_trajectory_goal
      self._pending_trajectory_goal = None
      self._flush_u2_trajectory(goal, {}, success=success)

    if self.enable_u4 and self.u4 is not None:
      self._flush_u4_trajectory(success)

  # ── Public: called by episode_runner when max_steps is reached ──────

  def flush_memory(self, goal: str) -> None:
    """Buffer the trajectory when max_steps is reached.

    Same as _on_task_done: the actual U2 write is deferred until
    set_episode_success() supplies the ground-truth outcome.
    """
    if not self.enable_u2 or self.u2 is None:
      return
    self._pending_trajectory_goal = goal

  # ── Sub-plan decomposition seam (Planner placeholder) ──────────────────
  #
  # This is the extension point for the future multi-agent design.  The
  # current implementation treats the whole task goal as a single sub-plan,
  # so all U2 memory operations below run at task granularity — exactly the
  # pre-existing behaviour.  When the multi-agent Planner is wired in, it
  # will return multiple Plan(precondition, goal) entries here and the
  # retrieval / replay / storage loops below will automatically operate at
  # sub-plan granularity with no further changes.

  def _decompose_into_subplans(self, goal: str) -> list[Plan]:
    """PLACEHOLDER: decompose `goal` into a sequence of sub-plans.

    Currently returns the whole task as one sub-plan (precondition empty),
    preserving today's task-level U2 behaviour.  The multi-agent Planner
    will override this to return a real {p1, ..., pk} sequence (§3.1).
    """
    return [Plan(precondition="", goal=goal)]

  # ── Internal ────────────────────────────────────────────────────────

  def _flush_u2_trajectory(
      self, goal: str, step_data: dict[str, Any], success: bool = False
  ) -> None:
    """Build ObsAct list from history and store in U2 memory bank.

    The observation field is stored as a lightweight text marker instead of
    the raw screenshot numpy array — the U2 retrieval hint only reads the
    action, never the observation, and full screenshots would bloat the
    trajectory pickles to many MB each.
    """
    del step_data
    if not self.enable_u2 or self.u2 is None:
      return

    # Replayed trajectories are re-consumed memories, not new experiences;
    # do not re-store them (§3.2.2).  Replayed history entries carry
    # u2_replayed=True, and _end_replay() clears the buffered goal.
    if goal is None or any(h.get("u2_replayed") for h in self.history):
      return

    trajectory: list[ObsAct] = []
    for i, hist in enumerate(self.history):
      action = hist.get("action_output_json")
      if action is None:
        continue
      trajectory.append(ObsAct(
          observation=f"step_{i}",  # lightweight marker, not raw pixels
          action=action,
          step_index=i,
      ))

    if len(trajectory) > 1:
      self.u2.add_trajectory(goal, trajectory)
      self.u2.finalize_task(goal, success=success)

  def _flush_u4_trajectory(self, success: bool) -> None:
    """Feed the finished episode's trajectory into U4 (ground-truth outcome).

    Successful episodes buffer their actions for later skill mining
    (write side).  Failed episodes update the score of any skill that was
    matched during the episode (feedback side).  The trajectory is read from
    the agent's own history — never from U2 — so U4 is an independent
    ablation flag.

    Each action is enriched with the semantic label of the UI element it
    acted on (resolved from that step's `before_ui_elements` via the action's
    index), so mined skills carry real targets instead of empty ones.  The
    precondition is the screen of the FIRST step (where the skill starts),
    not the final screen of the completed task.
    """
    goal = self._current_goal if hasattr(self, "_current_goal") else ""
    if not goal:
      return

    actions: list = []
    for h in self.history:
      action = h.get("action_output_json")
      if action is None or h.get("u2_replayed"):
        continue
      # Bind the semantic label of the element this action targets.
      ui_elements = h.get("before_ui_elements", [])
      index = getattr(action, "index", None)
      if index is not None and isinstance(ui_elements, list) and 0 <= index < len(ui_elements):
        try:
          action._semantic_target = _describe_element(ui_elements, index)
        except Exception:  # pylint: disable=broad-exception-caught
          pass
      actions.append(action)

    if not actions:
      return

    if success:
      # Precondition = the screen where the skill starts (first step's page).
      first_elements = self.history[0].get("before_ui_elements", []) if self.history else []
      first_app, first_page = extract_app_from_elements(first_elements)
      precondition = first_page or first_app
      self.u4.add_successful_trajectory(goal, actions, precondition=precondition)
      # Mine when a mine batch boundary is crossed (default: after every
      # successful episode, since mining is cheap and deterministic).
      self.u4.mine()
    else:
      self.u4.record_outcome(goal, success=False)

  # ── U2 deterministic replay (§3.2.2) ──────────────────────────────

  def _start_replay(self, trajectory: list[ObsAct], entry) -> None:
    """Begin deterministic replay of a cached trajectory for this task."""
    # §3.2.2 decoupled storage: ensure the trajectory is loaded from disk.
    loaded = self.u2.bank._load_trajectory(entry) if self.u2 is not None else []
    self._replay_active = True
    self._replay_entry = entry
    self._replay_index = 0
    self._replay_trajectory = loaded if loaded else list(trajectory)
    logging.info(
        "U2 — replaying cached trajectory (%d steps)", len(self._replay_trajectory)
    )

  def _resolve_action_target(self, action, ui_elements):
    """Rebind an action's index to the current screen's UI elements."""
    if action.action_type not in (
        json_action.CLICK,
        json_action.DOUBLE_TAP,
        json_action.LONG_PRESS,
    ):
      return action
    if action.index is None:
      return action
    if action.index < 0 or action.index >= len(ui_elements):
      return action
    logical_screen_size = self.env.logical_screen_size
    original = ui_elements[action.index]
    if m3a_utils.validate_ui_element(original, logical_screen_size):
      return action
    text = getattr(original, "text", None) or getattr(
        original, "content_description", None
    )
    if not text:
      return action
    matches = [
        (i, e)
        for i, e in enumerate(ui_elements)
        if m3a_utils.validate_ui_element(e, logical_screen_size)
        and (getattr(e, "text", None) or getattr(e, "content_description", None))
        == text
    ]
    if len(matches) == 1:
      rebound = copy.deepcopy(action)
      rebound.index = matches[0][0]
      return rebound
    return action

  def _end_replay(self) -> None:
    """Terminate replay and drop the buffered goal so nothing is re-stored."""
    self._replay_active = False
    self._replay_entry = None
    self._replay_index = 0
    self._replay_trajectory = []
    self._pending_trajectory_goal = None

  def _step_replay(self, goal: str):
    """Execute the next cached action. Returns (done, step_data)."""
    step_data = {
        "raw_screenshot": None,
        "before_screenshot_with_som": None,
        "before_ui_elements": [],
        "after_screenshot_with_som": None,
        "action_prompt": None,
        "action_output": None,
        "action_output_json": None,
        "action_reason": None,
        "action_raw_response": None,
        "summary_prompt": None,
        "summary": None,
        "summary_raw_response": None,
        "u2_replayed": True,
    }
    if self._replay_entry is not None:
      step_data["u2_memory_id"] = self._replay_entry.memory_id

    trajectory = self._replay_trajectory
    if self._replay_index >= len(trajectory):
      self._end_replay()
      step_data["summary"] = "Replayed full trajectory; task step complete."
      self.history.append(step_data)
      return True, step_data

    obs_act = trajectory[self._replay_index]
    self._replay_index += 1
    action = obs_act.action
    if action is None:
      step_data["summary"] = "Skipped replay step with no cached action."
      self.history.append(step_data)
      return False, step_data
    if not isinstance(action, json_action.JSONAction):
      action = json_action.JSONAction(
          **{k: v for k, v in vars(action).items() if k in json_action.ACTION_KEYS}
      )
    step_data["action_output_json"] = action

    if action.action_type == json_action.STATUS:
      done = action.goal_status != "infeasible"
      self._end_replay()
      step_data["summary"] = "Replayed memory declared sub-task %s." % (
          "completed" if done else "infeasible"
      )
      self.history.append(step_data)
      return done, step_data

    if action.action_type in (
        json_action.CLICK,
        json_action.DOUBLE_TAP,
        json_action.LONG_PRESS,
    ) and action.index is not None:
      try:
        state = self.get_post_transition_state()
        action = self._resolve_action_target(action, state.ui_elements)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("U2 replay — target resolution failed: %s", e)

    try:
      self.env.execute_action(action)
      time.sleep(self.wait_after_action_seconds)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("U2 [%s] — replay action failed: %s", goal, e)
      step_data["summary"] = f"Replay action failed: {e}"
      self.history.append(step_data)
      self._end_replay()
      return False, step_data

    remaining = len(trajectory) - self._replay_index
    step_data["summary"] = (
        f"Replayed step {self._replay_index}/{len(trajectory)}; {remaining} remaining."
    )
    self.history.append(step_data)
    return False, step_data

  def step(self, goal: str) -> base_agent.AgentInteractionResult:
    """Execute one interaction step; if U2 has a cached trajectory, replay it.

    The task goal is decomposed into sub-plans via _decompose_into_subplans
    (currently a single task-level sub-plan).  On the first step with U2
    enabled, each sub-plan is checked for a cached trajectory and replayed
    deterministically (§3.2.2) before falling back to the LLM.
    """
    # Active replay: execute the next cached action (§3.2.2).
    if self._replay_active:
      done, step_data = self._step_replay(goal)
      return base_agent.AgentInteractionResult(done, step_data)

    # First step of the task with U2 enabled: try deterministic replay for
    # each sub-plan.
    if self.enable_u2 and self.u2 is not None and len(self.history) == 0:
      for plan in self._decompose_into_subplans(goal):
        trajectory = self.u2.retrieve_sub_plan_replay(plan)
        if trajectory:
          self._start_replay(trajectory, self.u2._active_entry)
          done, step_data = self._step_replay(goal)
          return base_agent.AgentInteractionResult(done, step_data)

    return super().step(goal)
