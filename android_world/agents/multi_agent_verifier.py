"""Reflector — three pure verification functions for the multi-agent layer.

The Reflector checks three *different objects* with three *different questions*
(see docs/multi-agent-design.md §1): the current action (Action Verifier), the
task progress (Progress Auditor), and the final deliverable (Evidence
Certifier).  Each is a pure function: it takes the step data it needs plus an
LLM wrapper, and returns a structured verdict.  No agent state is read or
written here — the agent (multi_agent.MultiAgentReflectorAgent) decides what
to do with each verdict (which memory layer to gate).

All three verifiers are LLM-driven per the design decision; they only reason
over the prompt inputs they are given.  Deterministic fast-paths are not
implemented (kept out of scope to match the ablation design).
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

from android_world.agents import infer

# Multi-agent execution log.  Appended with a timestamp; lives in the working
# directory (the repo root when run via run.py).
_LOG_FILE = "multi_agent.log"


def _log(msg: str) -> None:
  """Append one line to the multi-agent execution log (never stdout)."""
  try:
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
      f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
  except Exception:  # pylint: disable=broad-exception-caught
    pass

# ── Verdict containers ───────────────────────────────────────────────────

ACTION_VERDICTS = ("CORRECT", "MISGROUNDED", "NO_EFFECT")
PROGRESS_VERDICTS = ("ADVANCING", "STALLED", "LOOPING", "BUSY_WITHOUT_PROGRESS")
CERT_RESULTS = ("PASS", "FAIL", "NO_EVIDENCE")


class ActionVerdict:
  """Outcome of Action Verifier: did the action deliver what was claimed?"""

  def __init__(self, verdict: str, evidence: str = ""):
    self.verdict = verdict
    self.evidence = evidence

  def __repr__(self) -> str:
    return f"ActionVerdict({self.verdict}, {self.evidence!r})"


class ProgressVerdict:
  """Outcome of Progress Auditor: is the task really advancing?"""

  def __init__(self, verdict: str, new_satisfied: list[str] | None = None,
               evidence: str = ""):
    self.verdict = verdict
    self.new_satisfied = new_satisfied or []
    self.evidence = evidence

  def __repr__(self) -> str:
    return f"ProgressVerdict({self.verdict}, new={self.new_satisfied})"


class EvidenceResult:
  """Per-item certification result plus an overall outcome."""

  def __init__(self, results: dict[str, str], overall: bool, evidence: str = ""):
    self.results = results  # item_id -> PASS/FAIL/NO_EVIDENCE
    self.overall = overall  # True iff all mandatory items PASS
    self.evidence = evidence

  def __repr__(self) -> str:
    return f"EvidenceResult({self.results}, overall={self.overall})"


# ── Prompt builders ──────────────────────────────────────────────────────

def _format_elements(ui_elements: Any) -> str:
  """Render a UI element list to rich text for the prompt.

  Mirrors m3a's element description (m3a.py:_generate_ui_element_description):
  includes state flags (is_checked / is_selected / is_editable / is_clickable)
  that the verifiers need — e.g. Progress Auditor must see an is_checked flip
  to judge "the switch is now on".  Index is the raw UI-list index.
  """
  if not ui_elements:
    return "No UI elements"
  lines = []
  for i, el in enumerate(ui_elements):
    parts = [f"[{i}]"]
    text = getattr(el, "text", None) or ""
    cd = getattr(el, "content_description", None) or ""
    hint = getattr(el, "hint_text", None) or ""
    if text:
      parts.append(f"text={text!r}")
    if cd:
      parts.append(f"desc={cd!r}")
    if hint:
      parts.append(f"hint={hint!r}")
    flags = []
    for attr in ("is_clickable", "is_checked", "is_selected", "is_editable"):
      if getattr(el, attr, False):
        flags.append(attr.replace("is_", ""))
    if flags:
      parts.append("[" + ",".join(flags) + "]")
    lines.append(" ".join(parts))
  return "\n".join(lines[:40])


# ── Action Verifier ──────────────────────────────────────────────────────

GUI_ACTIONS_CONTEXT = (
    "Android GUI actions:\n"
    "  click <index>     — tap the UI element with that index\n"
    "  input_text <text> — type text into the focused text field\n"
    "  scroll <dir>      — scroll the screen/list in a direction\n"
    "  open_app <name>   — launch the named app\n"
    "  navigate_home / navigate_back — go to home / previous screen\n\n"
)


ACTION_PROMPT_TEMPLATE = (
    "You are an action verifier for an Android GUI agent.\n"
    "The agent performed one action. Decide whether the action actually\n"
    "delivered the effect it claimed.\n\n"
    + GUI_ACTIONS_CONTEXT +
    "Action claimed:\n{claim}\n\n"
    "Action executed:\n  {action_type} (target: {target})\n\n"
    "The intended effect was: {expected_effect}\n\n"
    "UI elements BEFORE the action:\n{before_elements}\n\n"
    "UI elements AFTER the action:\n{after_elements}\n\n"
    "Two screenshots (before and after) are also provided.\n\n"
    "Rules:\n"
    "- CORRECT: the claimed effect is present in the AFTER state.\n"
    "- MISGROUNDED: the action hit the wrong target or produced a different\n"
    "  screen transition than claimed (e.g. opened the date picker when the\n"
    "  claim said time picker).\n"
    "- NO_EFFECT: the screen state did not change in any way relevant to the\n"
    "  claim (or nothing happened at all).\n\n"
    "Reply with EXACTLY one line:\n"
    "VERDICT: CORRECT | MISGROUNDED | NO_EFFECT\n"
    "EVIDENCE: <one sentence>"
)


def verify_action(
    claim: Any,
    action_type: str,
    target: str,
    before_elements: Any,
    after_elements: Any,
    llm: infer.MultimodalLlmWrapper,
    before_image: Any = None,
    after_image: Any = None,
) -> ActionVerdict:
  """Verify that the executed action delivered its claimed effect.

  Returns CORRECT if the claimed effect actually occurred, MISGROUNDED if the
  action hit the wrong target / produced the wrong transition, NO_EFFECT if
  nothing the claim promised happened.  All-LLM per design decision.
  """
  expected = getattr(claim, "expected_effect", "") or ""
  prompt = ACTION_PROMPT_TEMPLATE.format(
      claim=str(getattr(claim, "intent", "")),
      action_type=action_type,
      target=target,
      expected_effect=expected,
      before_elements=_format_elements(before_elements),
      after_elements=_format_elements(after_elements),
  )
  images = []
  if before_image is not None:
    images.append(before_image)
  if after_image is not None:
    images.append(after_image)

  llm.begin_module('av')
  try:
    output, _is_safe, _raw = llm.predict_mm(prompt, images)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return ActionVerdict("NO_EFFECT", f"LLM call failed: {e}")
  finally:
    llm.end_module()

  # Fail-closed: unparseable / NO_RESPONSE must not draw U3 edges as CORRECT.
  verdict = _parse_verdict_line(output, ACTION_VERDICTS, default="NO_EFFECT")
  evidence = _parse_evidence_line(output)
  _log(f"[AV] {action_type}({target}) -> {verdict} | {evidence}")
  return ActionVerdict(verdict, evidence)


# ── Progress Auditor ─────────────────────────────────────────────────────

PROGRESS_PROMPT_TEMPLATE = (
    "You are a progress auditor for an Android GUI agent.\n"
    "You maintain a progress ledger of task conditions. Decide whether the\n"
    "task has really advanced toward its goal this step.\n\n"
    "Task goal: {goal}\n\n"
    "Progress conditions defined by the planner (each must be met for the\n"
    "task to be complete):\n{conditions}\n\n"
    "Conditions already satisfied BEFORE this step:\n{satisfied_before}\n\n"
    "Screen state AFTER this step:\n{after_elements}\n\n"
    "Example (alarm task):\n"
    "  conditions: P1: alarm app open / P2: time is 07:00\n"
    "  satisfied_before: P1\n"
    "  after state shows the time picker set to 07:00\n"
    "  -> VERDICT: ADVANCING, NEW_SATISFIED: P2\n\n"
    "Rules:\n"
    "- ADVANCING: at least one condition newly became satisfied this step.\n"
    "- STALLED: no condition newly satisfied (screen may or may not change).\n"
    "- LOOPING: the satisfied set is the same as in the recent past steps.\n"
    "- BUSY_WITHOUT_PROGRESS: screen keeps changing but the satisfied set\n"
    "  never grows.\n\n"
    "Reply with EXACTLY one line:\n"
    "VERDICT: ADVANCING | STALLED | LOOPING | BUSY_WITHOUT_PROGRESS\n"
    "NEW_SATISFIED: <comma-separated condition IDs newly met this step, or none>\n"
    "EVIDENCE: <one sentence>"
)


def audit_progress(
    goal: str,
    conditions: list[tuple[str, str]],
    satisfied_before: set[str],
    after_elements: Any,
    llm: infer.MultimodalLlmWrapper,
    after_image: Any = None,
    stall_count: int = 0,
    repeat_count: int = 0,
) -> ProgressVerdict:
  """Determine whether the task truly advanced (vs. stalled/looping/busy).

  ADVANCING: at least one new condition satisfied this step.
  STALLED: no new condition over several steps.
  LOOPING: the satisfied-set signature repeats (progress state, not screen).
  BUSY_WITHOUT_PROGRESS: screen changed but the condition set never grew.

  conditions is a list of (id, text) pairs — the id (e.g. "P1") is the key
  the LLM is told to report in NEW_SATISFIED and what `satisfied_before` /
  the returned new_satisfied contain, so prompt ids and ledger ids agree.

  The stall/repeat thresholds are the caller's decision (they accumulate
  across steps); this function only classifies the current step's delta.
  """
  prompt = PROGRESS_PROMPT_TEMPLATE.format(
      goal=goal,
      conditions="\n".join(f"  {cid}: {ctext}" for cid, ctext in conditions),
      satisfied_before=", ".join(sorted(satisfied_before)) or "none",
      after_elements=_format_elements(after_elements),
  )
  images = [after_image] if after_image is not None else []
  llm.begin_module('pa')
  try:
    output, _is_safe, _raw = llm.predict_mm(prompt, images)
  except Exception as e:  # pylint: disable=broad-exception-caught
    return ProgressVerdict("STALLED", evidence=f"LLM call failed: {e}")
  finally:
    llm.end_module()

  verdict = _parse_verdict_line(output, PROGRESS_VERDICTS, default="STALLED")
  new_satisfied = _parse_new_satisfied(output)
  evidence = _parse_evidence_line(output)
  _log(f"[PA] -> {verdict} new={new_satisfied} | {evidence}")
  return ProgressVerdict(verdict, new_satisfied, evidence)


# ── Evidence Certifier ───────────────────────────────────────────────────

EVIDENCE_PROMPT_TEMPLATE = (
    "You are an evidence certifier for an Android GUI agent.\n"
    "You verify whether each acceptance criterion of the task is actually\n"
    "satisfied by the final state, with valid evidence.\n\n"
    "Task goal: {goal}\n\n"
    "Acceptance checklist (each item must be met):\n{checklist}\n\n"
    "Final screen state:\n{final_elements}\n\n"
    "A final screenshot is provided.\n\n"
    "Example (alarm task):\n"
    "  checklist: [A1] alarm time = 07:00 (mandatory=True)\n"
    "  final screen shows an alarm listed as 07:00 with the switch on\n"
    "  ->\n"
    "  ITEM A1: PASS\n"
    "  EVIDENCE: The alarm list shows 07:00.\n"
    "  OVERALL: PASS\n\n"
    "Rules:\n"
    "- PASS: the final state satisfies the item AND you can point to evidence.\n"
    "- FAIL: the final state contradicts the item.\n"
    "- NO_EVIDENCE: only the action history suggests it, but no final-state\n"
    "  evidence supports it (do NOT infer completion from past actions).\n\n"
    "For each item reply with EXACTLY one line:\n"
    "ITEM <id>: PASS | FAIL | NO_EVIDENCE\n"
    "EVIDENCE: <one sentence per item>\n\n"
    "Then a final line:\n"
    "OVERALL: PASS | FAIL"
)


def certify_evidence(
    goal: str,
    checklist: list[Any],
    final_elements: Any,
    llm: infer.MultimodalLlmWrapper,
    final_image: Any = None,
) -> EvidenceResult:
  """Certify each acceptance item against the final state, per-item.

  Returns a dict item_id -> PASS/FAIL/NO_EVIDENCE plus an overall boolean
  (True iff all mandatory items PASS).  Evidence freshness is the caller's
  job (bundles invalidated when a later step modifies the value) — this
  function certifies against the final state snapshot.
  """
  checklist_lines = "\n".join(
      f"  [{i.item_id}] {i.object} = {i.expected} (mandatory={i.mandatory})"
      for i in checklist
  )
  prompt = EVIDENCE_PROMPT_TEMPLATE.format(
      goal=goal,
      checklist=checklist_lines,
      final_elements=_format_elements(final_elements),
  )
  images = [final_image] if final_image is not None else []
  llm.begin_module('ec')
  try:
    output, _is_safe, _raw = llm.predict_mm(prompt, images)
  except Exception as e:  # pylint: disable=broad-exception-caught
    results = {i.item_id: "NO_EVIDENCE" for i in checklist}
    _log(f"[EC] LLM call failed: {e} -> overall=False (fail closed)")
    return EvidenceResult(results, False, f"LLM call failed: {e}")
  finally:
    llm.end_module()

  results = _parse_cert_items(output, checklist)
  # Fail closed: the final verdict is driven by the per-item results (missing
  # items are NO_EVIDENCE and therefore FAIL), NOT by trusting the LLM's
  # OVERALL line, which can contradict its own item lines.
  mandatory = [i for i in checklist if i.mandatory]
  overall = bool(mandatory) and all(
      results.get(i.item_id) == "PASS" for i in mandatory
  )
  _log(f"[EC] per-item={results} overall={overall}")
  return EvidenceResult(results, overall, output)


# ── Output parsing helpers ───────────────────────────────────────────────

def _parse_verdict_line(output: str, valid: tuple[str, ...],
                        default: str | None = None) -> str:
  """Extract the first VERDICT: line and validate against `valid`."""
  line = _find_line(output, "VERDICT")
  if line:
    m = re.search(r"(?:VERDICT:)?\s*([A-Z_]+)", line)
    if m and m.group(1) in valid:
      return m.group(1)
  # LLM sometimes omits the VERDICT: prefix but still emits a bare label
  # (e.g. "CORRECT: everything fine"). Prefer an explicit token over default.
  for token in valid:
    if re.search(rf"\b{token}\b", output or ""):
      return token
  return default or valid[0]


def _parse_evidence_line(output: str) -> str:
  """Extract EVIDENCE: content, truncated."""
  line = _find_line(output, "EVIDENCE")
  if not line:
    return ""
  return line.split("EVIDENCE:", 1)[-1].strip()[:200]


def _parse_new_satisfied(output: str) -> list[str]:
  """Extract NEW_SATISFIED: content (condition IDs like P1/P2).

  Only matches IDs of the form <letter(s)><digits> (e.g. P1, A2) — the same
  ids the planner emitted and the prompt rendered, so the returned set always
  agrees with the ledger's condition keys.  Bare numbers / [0] / subgoal text
  are ignored (they can't be condition ids).
  """
  line = _find_line(output, "NEW_SATISFIED")
  if not line:
    return []
  content = line.split("NEW_SATISFIED:", 1)[-1]
  content = content.strip().lower()
  if not content or content == "none":
    return []
  tokens = re.findall(r"[a-z]+\d+", content)
  return [t.upper() for t in tokens if t]


def _parse_cert_items(output: str, checklist: list[Any]) -> dict[str, str]:
  """Extract per-item ITEM <id>: PASS/FAIL/NO_EVIDENCE lines."""
  results: dict[str, str] = {}
  for line in output.splitlines():
    m = re.match(r"ITEM\s+(\S+):\s*([A-Z_]+)", line.strip())
    if not m:
      continue
    item_id, result = m.group(1), m.group(2)
    if result in ("PASS", "FAIL", "NO_EVIDENCE"):
      results[item_id] = result
  # Fill in any checklist items the LLM omitted (fail closed).
  for item in checklist:
    results.setdefault(item.item_id, "NO_EVIDENCE")
  return results


def _find_line(output: str, prefix: str) -> str:
  """Return the first line starting with `prefix` (case-insensitive), or ''."""
  for line in output.splitlines():
    if line.strip().upper().startswith(prefix.upper()):
      return line.strip()
  return ""
