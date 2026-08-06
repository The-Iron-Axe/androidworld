"""MultiAgentReflectorAgent — Planner / Executor / Reflector fused with memory.

The multi-agent layer is a *new* agent subclass of the existing memory agent.
It is orthogonal to the U1-U4 memory axis:

  --u1 --u2 --u3 --u4   memory axis  (existing flags, untouched)
  --multiagent          multi-agent axis (this module)

The 2x2 ablation:
  baseline                     m3a_qwen3_vl_32b
  +memory                      m3a_qwen3_vl_32b_mem --u1 --u2 --u3 --u4
  +multi-agent (no memory)     m3a_qwen3_vl_32b_mem --multiagent
  +memory +multi-agent         m3a_qwen3_vl_32b_mem --multiagent --u1 --u2 --u3 --u4

HARD CONSTRAINT: with enable_multiagent=False the execution path must be
byte-identical to MemoryAugmentedAgent.  Every overridden hook begins with
`if not self._multiagent:` and delegates to super(), so flag-off runs the
unmodified memory-agent path and none of the multi-agent code is reachable.

Architecture (docs/multi-agent-design.md §2):
  Planner   = methods on the agent (_planner_plan / _planner_replan) — needs
              cross-step state (current subgoal, progress ledger, replan cap).
  Executor  = the inherited M3A.step() action loop, untouched. The only
              addition is that the *system* extracts a structured ActionClaim
              from the Executor's action reasoning (never retro-edited).
  Reflector = three pure LLM-driven functions in multi_agent_verifier.py,
              wired into the memory-write gates here.

The ProgressLedger / AcceptanceItem / EvidenceBundle / ActionClaim are
agent-layer state — never TaskState/EpisodicMemory/etc. — so the
"--multiagent without memory" quadrant runs with all U flags off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from android_world.agents import infer
from android_world.agents import memory_agent as mem
from android_world.agents import multi_agent_verifier as mav
from android_world.agents.multi_agent_verifier import _log
from android_world.agents.memory.task_state import (
    extract_app_from_elements,
    init_task_state,
    update_task_state,
)
from android_world.env import interface

STALL_THRESHOLD = 3      # consecutive no-progress steps before replan
MAX_REPLANS = 2          # hard cap on planner replans per episode
UI_ACTION_TYPES = (
    "click", "input_text", "scroll", "open_app", "long_press",
    "navigate_home", "navigate_back",
)


# ── Agent-layer dataclasses (orthogonal to U1-U4) ────────────────────────

@dataclass(frozen=True)
class ActionClaim:
  """The Executor's pre-execution declaration, extracted by the system.

  frozen=True enforces the design rule (§4.1): produced before execution,
  never retro-edited by any verifier.
  """
  action_type: str
  target: str
  intent: str
  expected_effect: str
  subgoal: str
  screen_key_before: str


@dataclass
class ProgressLedger:
  """Planner-defined progress conditions; updated by the Progress Auditor.

  conditions: list of (id, text) — the id (e.g. "P1") is the stable key the
    LLM is told about AND that Progress Auditor / Evidence Certifier use, so
    the ids in `satisfied` always match what the prompt rendered.
  satisfied: set of condition ids currently met.
  """
  conditions: list[tuple[str, str]] = field(default_factory=list)
  satisfied: set[str] = field(default_factory=set)
  signature: str = ""
  last_advance_step: int = 0

  def recompute_signature(self) -> str:
    self.signature = ",".join(sorted(self.satisfied))
    return self.signature


@dataclass
class AcceptanceItem:
  """One acceptance criterion the final deliverable must satisfy."""
  item_id: str
  object: str
  expected: str
  mandatory: bool = True
  evidence_forms: list[str] = field(default_factory=list)


@dataclass
class EvidenceBundle:
  """Snapshotted evidence for one acceptance item (freshness-checked)."""
  item_id: str
  observed: str
  screenshot_idx: int
  region: str = ""
  valid: bool = True
  result: str | None = None


@dataclass
class PlannerState:
  """Cross-step state produced by the Planner, consumed by Executor/Reflector."""
  subgoals: list[str] = field(default_factory=list)
  current_idx: int = 0
  ledger: ProgressLedger = field(default_factory=ProgressLedger)
  checklist: list[AcceptanceItem] = field(default_factory=list)


# ── Prompt builders (Planner) ────────────────────────────────────────────

PLANNER_PROMPT_TEMPLATE = (
    "You are a planning module for an Android GUI agent.\n"
    "Break the user task into a small sequence of subgoals, define the\n"
    "progress conditions that indicate real advancement, and define an\n"
    "acceptance checklist the final state must satisfy.\n\n"
    "Task goal: {goal}\n\n"
    "Current screen (optional):\n{screen}\n\n"
    "Example (alarm task):\n"
    "  goal: create an alarm for 07:00 tomorrow\n"
    "  SUBGOALS:\n"
    "  1. open the alarm app\n"
    "  2. set the time to 07:00\n"
    "  3. set the date to tomorrow\n"
    "  4. enable and save the alarm\n"
    "  PROGRESS_CONDITIONS:\n"
    "  P1: alarm app is open\n"
    "  P2: time is set to 07:00\n"
    "  P3: date is set to tomorrow\n"
    "  P4: alarm is enabled and saved\n"
    "  ACCEPTANCE:\n"
    "  A1: alarm time: 07:00 [mandatory]\n"
    "  A2: alarm date: tomorrow [mandatory]\n"
    "  A3: alarm enabled: true [mandatory]\n\n"
    "Reply in EXACTLY this format:\n"
    "SUBGOALS:\n1. <subgoal>\n2. <subgoal>\n...\n"
    "PROGRESS_CONDITIONS:\nP1: <condition>\nP2: <condition>\n...\n"
    "ACCEPTANCE:\nA1: <object>: <expected value> [mandatory|optional]\n..."
)

REPLAN_PROMPT_TEMPLATE = (
    "You are a planning module for an Android GUI agent. The previous plan\n"
    "failed to make progress. Produce a revised plan.\n\n"
    "Task goal: {goal}\n"
    "Conditions already satisfied: {satisfied}\n"
    "Stalled subgoal: {stalled_subgoal}\n\n"
    "Reply in EXACTLY this format:\n"
    "SUBGOALS:\n1. <subgoal>\n2. <subgoal>\n...\n"
    "PROGRESS_CONDITIONS:\nP1: <condition>\nP2: <condition>\n...\n"
    "ACCEPTANCE:\nA1: <object>: <expected value> [mandatory|optional]\n..."
)


def _parse_plan_output(
    output: str,
) -> tuple[list[str], list[tuple[str, str]], list[AcceptanceItem]]:
  """Parse the planner's text output into (subgoals, conditions, checklist).

  conditions are returned as (id, text) pairs — the "P1"/"P2" ids from the
  planner's output are the stable keys the Progress Auditor reports back and
  the ledger's `satisfied` set stores, so prompt ids and code ids always agree.
  """
  subgoals: list[str] = []
  conditions: list[tuple[str, str]] = []
  checklist: list[AcceptanceItem] = []
  section = ""
  for line in output.splitlines():
    line = line.strip()
    if not line:
      continue
    upper = line.upper()
    if upper.startswith("SUBGOALS"):
      section = "subgoals"
      continue
    if upper.startswith("PROGRESS_CONDITIONS"):
      section = "conditions"
      continue
    if upper.startswith("ACCEPTANCE"):
      section = "acceptance"
      continue
    if section == "subgoals":
      text = re.sub(r"^\d+\.\s*", "", line).strip()
      if text:
        subgoals.append(text)
    elif section == "conditions":
      m = re.match(r"^(P\d+):\s*(.+)$", line, re.IGNORECASE)
      if m:
        conditions.append((m.group(1).upper(), m.group(2).strip()))
    elif section == "acceptance":
      text = re.sub(r"^A\d+:\s*", "", line).strip()
      if not text:
        continue
      mandatory = "optional" not in text.lower()
      obj, _, expected = text.partition(":")
      checklist.append(
          AcceptanceItem(
              item_id=f"A{len(checklist) + 1}",
              object=obj.strip(),
              expected=expected.split("[")[0].strip(),
              mandatory=mandatory,
          )
      )
  if not subgoals:
    subgoals = [output.strip()]  # degenerate fallback: whole goal as one subgoal
  if not conditions:
    conditions = [(f"P{i+1}", f"task {g}") for i, g in enumerate(subgoals)]
  if not checklist:
    checklist = [AcceptanceItem(item_id="A1", object="task", expected="complete")]
  return subgoals, conditions, checklist


# ── Action Claim extraction ──────────────────────────────────────────────

_CLAIM_VERB_RE = re.compile(
    r"\b(to|so that|in order to)\s+(.+?)(?:\.|$)", re.IGNORECASE
)


def _extract_intent_from_reason(reason: str) -> str:
  """Heuristically pull the intent from free-text action reasoning."""
  if not reason:
    return ""
  m = _CLAIM_VERB_RE.search(reason)
  if m:
    return m.group(2).strip()[:200]
  # No "to ..." phrase — fall back to the reason text itself.
  return reason[:200]


def _describe_target(action, ui_elements) -> str:
  """Semantic (index-free) description of the action's target element."""
  if action is None:
    return ""
  if getattr(action, "action_type", "") == "open_app":
    return getattr(action, "app_name", "") or ""
  index = getattr(action, "index", None)
  if index is None:
    return ""
  try:
    el = ui_elements[index]
  except (IndexError, TypeError):
    return ""
  for attr in ("text", "content_description", "hint_text"):
    val = getattr(el, attr, None) or ""
    if str(val).strip():
      return str(val).strip()[:40]
  return f"element[{index}]"


class MultiAgentReflectorAgent(mem.MemoryAugmentedAgent):
  """Memory agent fused with a Planner/Executor/Reflector orchestration layer.

  enable_multiagent=False (default) reproduces MemoryAugmentedAgent exactly.
  """

  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      enable_multiagent: bool = False,
      **kwargs,
  ):
    super().__init__(env, llm, **kwargs)
    self._multiagent = enable_multiagent
    self._planner_state: PlannerState | None = None
    self._certified: bool | None = None
    self._cert_report: Any | None = None
    self._stall_steps = 0
    self._replan_count = 0
    self._step_verdicts: list[Any] = []
    self._claims: list[ActionClaim] = []

  # ── Lifecycle ─────────────────────────────────────────────────────────

  def reset(self, go_home_on_reset: bool = False) -> None:
    if self._multiagent:
      self._planner_state = None
      self._certified = None
      self._cert_report = None
      self._stall_steps = 0
      self._replan_count = 0
      self._step_verdicts = []
      self._claims = []
    super().reset(go_home_on_reset)

  # ── Planner ───────────────────────────────────────────────────────────

  def _planner_plan(self, goal: str, ui_elements_list: str) -> None:
    """One-shot task decomposition at episode open. Writes U1."""
    prompt = PLANNER_PROMPT_TEMPLATE.format(
        goal=goal,
        screen=ui_elements_list[:1500] if ui_elements_list else "not available",
    )
    try:
      output, _is_safe, _raw = self.llm.predict_mm(prompt, [])
    except Exception as e:  # pylint: disable=broad-exception-caught
      output = f"SUBGOALS:\n1. {goal}"

    subgoals, conditions, checklist = _parse_plan_output(output)
    self._planner_state = PlannerState(
        subgoals=subgoals,
        ledger=ProgressLedger(conditions=conditions, satisfied=set()),
        checklist=checklist,
    )
    self._planner_state.ledger.recompute_signature()
    _log(f"[PLAN] subgoals={subgoals}")
    _log(f"[PLAN] conditions={conditions}")
    _log(f"[PLAN] checklist={[(i.item_id, i.object, i.expected) for i in checklist]}")

    # Write U1: seed pending subgoals + current subgoal.
    if self.enable_u1:
      if self.u1 is None:
        self.u1 = init_task_state(goal)
      update_task_state(
          self.u1,
          pending=list(subgoals),
          current_subgoal=subgoals[0] if subgoals else "",
      )

  def _planner_replan(self, goal: str) -> None:
    """Re-plan the remaining subgoals after a stall, with a hard cap."""
    if self._replan_count >= MAX_REPLANS:
      return
    self._replan_count += 1
    stalled = self._current_subgoal()
    state = self._planner_state
    satisfied = state.ledger.satisfied if state else set()
    prompt = REPLAN_PROMPT_TEMPLATE.format(
        goal=goal,
        satisfied=", ".join(sorted(satisfied)) or "none",
        stalled_subgoal=stalled,
    )
    try:
      output, _is_safe, _raw = self.llm.predict_mm(prompt, [])
    except Exception as e:  # pylint: disable=broad-exception-caught
      return  # give up on replan; keep executing current subgoal
    subgoals, conditions, checklist = _parse_plan_output(output)
    if state is None:
      state = PlannerState()
      self._planner_state = state
    state.subgoals = subgoals
    state.current_idx = 0
    state.ledger.conditions = conditions
    state.checklist = checklist
    state.ledger.recompute_signature()
    if self.enable_u1 and self.u1 is not None:
      update_task_state(
          self.u1,
          pending=list(subgoals),
          current_subgoal=subgoals[0] if subgoals else "",
      )

  def _current_subgoal(self) -> str:
    if self._planner_state and self._planner_state.subgoals:
      idx = min(self._planner_state.current_idx, len(self._planner_state.subgoals) - 1)
      return self._planner_state.subgoals[idx]
    return ""

  # ── Action prompt hook: inject the plan block ─────────────────────────

  def _build_action_prompt(
      self, goal: str, history_lines: list[str], ui_elements_list: str
  ) -> str:
    if not self._multiagent:
      return super()._build_action_prompt(goal, history_lines, ui_elements_list)

    # First step: plan once.  U2 replay steps return before this hook, so the
    # first call after a replay may already have history — rely on the planner
    # state being unset, not on an empty history.
    if self._planner_state is None:
      self._planner_plan(goal, ui_elements_list)

    plan_block = self._format_plan_block()
    if plan_block:
      goal = plan_block + "\n\n" + goal
    return super()._build_action_prompt(goal, history_lines, ui_elements_list)

  def _format_plan_block(self) -> str:
    state = self._planner_state
    if state is None:
      return ""
    lines = ["## Plan (multi-agent)"]
    lines.append(f"Current subgoal: {self._current_subgoal()}")
    if state.subgoals:
      lines.append(
          "Subgoals: " + " → ".join(
              f"[{'x' if i < state.current_idx else ' '}] {g}"
              for i, g in enumerate(state.subgoals)
          )
      )
    if state.ledger.satisfied:
      lines.append(
          "Progress satisfied: " + ", ".join(sorted(state.ledger.satisfied))
      )
    return "\n".join(lines)

  # ── Step-complete hook: Action Verifier → U3 gate, Progress Auditor → U1 ─

  def _on_step_complete(self, step_data: dict[str, Any]) -> None:
    if not self._multiagent:
      super()._on_step_complete(step_data)
      return

    # Replay steps have no action_reason / claim; skip all verifiers.
    if step_data.get("u2_replayed"):
      return

    # (2) Action Verifier → gate U3 edge drawing + record U4 step credit.
    verdict = self._verify_step_action(step_data)
    self._step_verdicts.append(verdict)
    self._apply_u3_gate(step_data, verdict)

    # (3) Progress Auditor → advance U1.completed only if the subgoal-level
    #     Evidence Certifier also passes (ordering: ADVANCING → certify).
    self._audit_and_advance(step_data)

    # (4) Replicate super()'s U1 bookkeeping (app/page/last_action).
    self._update_u1_bookkeeping(step_data)

  def _verify_step_action(self, step_data: dict[str, Any]) -> Any:
    action = step_data.get("action_output_json")
    action_type = getattr(action, "action_type", "")
    if action_type not in UI_ACTION_TYPES:
      # Non-UI actions (status/answer/wait) don't change screen state.
      return mav.ActionVerdict("CORRECT", "non-UI action, assumed correct")

    before = step_data.get("before_ui_elements", [])
    after = step_data.get("after_ui_elements", [])
    target = _describe_target(action, before)
    claim = self._build_claim(step_data, action_type, target)
    if claim is not None:
      self._claims.append(claim)

    before_img = step_data.get("before_screenshot_with_som")
    after_img = step_data.get("after_screenshot_with_som")
    return mav.verify_action(
        claim if claim is not None else _NullClaim(),
        action_type,
        target,
        before,
        after,
        self.llm,
        before_img,
        after_img,
    )

  def _build_claim(
      self, step_data: dict[str, Any], action_type: str, target: str
  ) -> ActionClaim | None:
    """System extracts a frozen ActionClaim from the Executor's reasoning."""
    reason = step_data.get("action_reason") or ""
    intent = _extract_intent_from_reason(reason)
    if not intent:
      return None
    before = step_data.get("before_ui_elements", [])
    app, page = extract_app_from_elements(before)
    return ActionClaim(
        action_type=action_type,
        target=target,
        intent=intent,
        expected_effect=intent,
        subgoal=self._current_subgoal(),
        screen_key_before=page or app,
    )

  def _apply_u3_gate(self, step_data: dict[str, Any], verdict: Any) -> None:
    """Only CORRECT transitions draw U3 page-graph edges."""
    if not self.enable_u3 or self.u3 is None:
      return
    if verdict.verdict != "CORRECT":
      _log(f"[U3] skip edge draw ({verdict.verdict} — not CORRECT)")
      return
    _log(f"[U3] draw edge (CORRECT)")
    before_elements = step_data.get("before_ui_elements", [])
    after_elements = step_data.get("after_ui_elements", [])
    before_list = step_data.get("before_ui_elements_list", "")
    after_list = step_data.get("after_ui_elements_list", "")
    action = step_data.get("action_output_json")
    before_app, before_page = extract_app_from_elements(before_elements)
    after_app, after_page = extract_app_from_elements(after_elements)
    goal = getattr(self, "_current_goal", "")
    from android_world.agents.memory.environment import build_screen_summary
    before_summary = build_screen_summary(
        before_list, current_app=before_app or "", current_page=before_page or "")
    after_summary = build_screen_summary(
        after_list, current_app=after_app or "", current_page=after_page or "")
    self.u3.record_transition(
        before_summary=before_summary,
        action_summary=mem._action_effect_str(action, before_elements),
        task=goal,
        after_summary=after_summary,
        before_app=before_app,
        after_app=after_app,
    )

  def _audit_and_advance(self, step_data: dict[str, Any]) -> None:
    state = self._planner_state
    if state is None:
      return
    goal = getattr(self, "_current_goal", "")
    pv = mav.audit_progress(
        goal=goal,
        conditions=state.ledger.conditions,
        satisfied_before=set(state.ledger.satisfied),
        after_elements=step_data.get("after_ui_elements", []),
        llm=self.llm,
        after_image=step_data.get("after_screenshot_with_som"),
    )
    if pv.verdict == "ADVANCING":
      self._stall_steps = 0
      new_satisfied = pv.new_satisfied
      if new_satisfied:
        state.ledger.satisfied.update(new_satisfied)
        state.ledger.recompute_signature()
      # Subgoal-level certification gates the ledger advance.
      if self._certify_current_subgoal(step_data):
        self._advance_subgoal()
      else:
        _log(f"[PA] ADVANCING but subgoal certifier FAILED — not advancing")
    else:
      self._stall_steps += 1
      _log(f"[PA] {pv.verdict} (stall_steps={self._stall_steps})")
      if self._stall_steps >= STALL_THRESHOLD:
        _log(f"[PA] reaching STALL_THRESHOLD — replanning")
        self._planner_replan(goal)
        self._stall_steps = 0

  def _certify_current_subgoal(self, step_data: dict[str, Any]) -> bool:
    """Subgoal-level Evidence Certifier: is the current subgoal truly done?"""
    state = self._planner_state
    if state is None:
      return True
    subgoal = self._current_subgoal()
    if not subgoal:
      return True
    item = AcceptanceItem(
        item_id="SG",
        object=subgoal,
        expected="achieved",
        mandatory=True,
    )
    result = mav.certify_evidence(
        goal=getattr(self, "_current_goal", ""),
        checklist=[item],
        final_elements=step_data.get("after_ui_elements", []),
        llm=self.llm,
        final_image=step_data.get("after_screenshot_with_som"),
    )
    return result.overall

  def _advance_subgoal(self) -> None:
    state = self._planner_state
    if state is None:
      return
    done = self._current_subgoal()
    # Note: the ledger's `satisfied` set is owned by the Progress Auditor and
    # holds *condition ids* (P1/P2).  We do NOT add the subgoal text here —
    # that would pollute the set with a different key format and break the
    # LOOPING signature comparison.
    if self.enable_u1 and self.u1 is not None:
      # Update completed/current_subgoal directly — NOT via update_task_state,
      # which would double-count step_count (bookkeeping already counted it).
      if done and done not in self.u1.completed:
        self.u1.completed.append(done)
      if done in self.u1.pending:
        self.u1.pending.remove(done)
      self.u1.current_subgoal = self._next_subgoal()
    state.current_idx += 1
    _log(f"[PLAN] advanced subgoal: {done!r} -> next={self._next_subgoal()!r}")

  def _next_subgoal(self) -> str:
    state = self._planner_state
    if state and state.subgoals and state.current_idx + 1 < len(state.subgoals):
      return state.subgoals[state.current_idx + 1]
    return ""

  def _update_u1_bookkeeping(self, step_data: dict[str, Any]) -> None:
    """Replicate MemoryAugmentedAgent._on_step_complete's U1 app/page tracking."""
    if not self.enable_u1 or self.u1 is None:
      return
    before_elements = step_data.get("before_ui_elements", [])
    app, page = extract_app_from_elements(before_elements)
    action = step_data.get("action_output_json")
    update_task_state(
        self.u1,
        current_app=app or None,
        current_page=page or None,
        last_action={"action_type": getattr(action, "action_type", "?")},
        last_effect=mem._action_effect_str(action, before_elements),
        failure=False,
    )

  # ── Task-done hook: global Evidence Certifier → _certified ────────────

  def _on_task_done(self, goal: str, step_data: dict[str, Any]) -> None:
    if not self._multiagent:
      super()._on_task_done(goal, step_data)
      return
    super()._on_task_done(goal, step_data)  # U2 buffer unchanged
    self._certified = self._certify_global(goal, step_data)

  def _certify_global(self, goal: str, step_data: dict[str, Any]) -> bool:
    state = self._planner_state
    if state is None or not state.checklist:
      return True  # no checklist -> nothing to certify (degenerate)
    result = mav.certify_evidence(
        goal=goal,
        checklist=state.checklist,
        final_elements=step_data.get("after_ui_elements", []),
        llm=self.llm,
        final_image=step_data.get("after_screenshot_with_som"),
    )
    self._cert_report = result
    return result.overall

  # ── Max-steps path: global certifier must also run ────────────────────

  def flush_memory(self, goal: str) -> None:
    if not self._multiagent:
      super().flush_memory(goal)
      return
    if self._certified is None:
      # Agent never declared done; certify against the last known state.
      self._certified = self._certify_global_from_history(goal)
    super().flush_memory(goal)

  def _certify_global_from_history(self, goal: str) -> bool:
    state = self._planner_state
    if state is None or not state.checklist or not self.history:
      return True
    last = self.history[-1]
    result = mav.certify_evidence(
        goal=goal,
        checklist=state.checklist,
        final_elements=last.get("after_ui_elements", []),
        llm=self.llm,
        final_image=last.get("after_screenshot_with_som"),
    )
    self._cert_report = result
    return result.overall

  # ── Episode-success fusion: internal certification vetoes external truth ─

  def set_episode_success(self, success: bool) -> None:
    if self._multiagent and self._certified is not None:
      success = success and self._certified
    super().set_episode_success(success)

  # ── U4 episode feedback (ground-truth outcome) ────────────────────────

  def _flush_u4_trajectory(self, success: bool) -> None:
    """Feed the finished episode into U4 using the ground-truth outcome.

    A successful episode is always buffered for positive-skill mining; a failed
    episode is buffered for negative-skill mining AND score-penalizes any
    matched skill (both handled by the parent).  The per-step Action Verifier
    verdicts (`_step_verdicts`) are still recorded and are available to
    callers, but they no longer gate skill mining — any ground-truth success
    feeds U4, so skills can accumulate even when individual steps weren't all
    CORRECT.
    """
    if not self._multiagent:
      super()._flush_u4_trajectory(success)
      return
    if not self.enable_u4 or self.u4 is None:
      return
    super()._flush_u4_trajectory(success)


class _NullClaim:
  """Fallback claim when the Executor's reasoning has no parseable intent."""

  intent = ""
  expected_effect = ""
