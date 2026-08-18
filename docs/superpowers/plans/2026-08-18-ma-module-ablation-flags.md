# Multi-Agent Module Ablation Flags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add four `--ma_no_*` flags that each disable exactly one multi-agent module (Planner, Action Verifier, Progress Auditor, Evidence Certifier), with the removed module's behavior reverting to the single-agent (m3a) baseline, plus no-leak unit tests.

**Architecture:** Four positive `enable_ma_*` kwargs on `MultiAgentReflectorAgent` (default `True` = full system), derived in `run.py` / `ablation_hierarchical.py` from the `--ma_no_*` flags (default `False`). Each affected hook in `multi_agent.py` is gated by its module flag. Where the module's only effect was the gate itself (AV → U3), the off path falls through to `super()` so U3 reverts to single-agent buffering. Module ownership follows the agreed semantics: `_certify_current_subgoal` belongs to EC (cut together with EC); the plan block's subgoal list belongs to Planner (kept under −PA), its "Current subgoal"/checkbox/"Progress satisfied" lines belong to PA (removed under −PA).

**Tech Stack:** Python 3, absl flags, absltest, AndroidWorld agent framework.

**Repo root:** `C:\Users\WRQ\Desktop\androidworld` (all paths below are relative to it).

---

## Task 0: Baseline check

**Files:**
- Run only, no edits.

- [ ] **Step 1: Confirm the existing test suite is green before touching anything**

Run: `python -m android_world.agents.multi_agent_test`
Expected: all existing tests PASS.

- [ ] **Step 2: Confirm working tree state**

Run: `git status`
Expected: no surprises before edits (untracked/new files are fine; note them so commits stay scoped).

---

## Task 1: Constructor module flags

**Files:**
- Modify: `android_world/agents/multi_agent.py` (`__init__`, lines 270-286)
- Test: `android_world/agents/multi_agent_test.py`

- [ ] **Step 1: Write the failing tests**

Append to `multi_agent_test.py`, right before the final `if __name__ == "__main__":` block:

```python
class _RecordingScriptedLlm(_ScriptedLlm):
    """_ScriptedLlm that also records every text prompt it sees."""

    def __init__(self, responses):
        super().__init__(responses)
        self.calls = []

    def predict_mm(self, text_prompt, images):
        self.calls.append(text_prompt)
        return super().predict_mm(text_prompt, images)


def _fake_action(action_type="click", index=0):
    """Minimal stand-in for the agent's action object."""
    return type("FakeAction", (), {"action_type": action_type, "index": index})()


class ModuleAblationTest(absltest.TestCase, _AdbMockMixin):
    """No-leak guards: each --ma_no_* flag removes exactly its module's effect."""

    def setUp(self):
        super().setUp()
        self._start_adb_mocks()

    def tearDown(self):
        super().tearDown()
        self._stop_adb_mocks()

    # ── Constructor ─────────────────────────────────────────────────

    def test_constructor_defaults_to_full_system(self):
        env = test_utils.FakeAsyncEnv()
        agent = ma.MultiAgentReflectorAgent(env, _ScriptedLlm([]), enable_multiagent=True)
        self.assertTrue(agent._ma_planner)
        self.assertTrue(agent._ma_av)
        self.assertTrue(agent._ma_pa)
        self.assertTrue(agent._ma_ec)

    def test_constructor_honors_module_flags(self):
        env = test_utils.FakeAsyncEnv()
        agent = ma.MultiAgentReflectorAgent(
            env, _ScriptedLlm([]), enable_multiagent=True,
            enable_ma_planner=False, enable_ma_av=False,
            enable_ma_pa=False, enable_ma_ec=False)
        self.assertFalse(agent._ma_planner)
        self.assertFalse(agent._ma_av)
        self.assertFalse(agent._ma_pa)
        self.assertFalse(agent._ma_ec)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m android_world.agents.multi_agent_test`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'enable_ma_planner'`.

- [ ] **Step 3: Implement the constructor flags**

In `android_world/agents/multi_agent.py`, change the `__init__` signature and body (current lines 270-286):

```python
  def __init__(
      self,
      env: interface.AsyncEnv,
      llm: infer.MultimodalLlmWrapper,
      enable_multiagent: bool = False,
      enable_ma_planner: bool = True,
      enable_ma_av: bool = True,
      enable_ma_pa: bool = True,
      enable_ma_ec: bool = True,
      **kwargs,
  ):
    super().__init__(env, llm, **kwargs)
    self._multiagent = enable_multiagent
    self._ma_planner = enable_ma_planner
    self._ma_av = enable_ma_av
    self._ma_pa = enable_ma_pa
    self._ma_ec = enable_ma_ec
```

(Keep every line after `self._ma_ec = enable_ma_ec` unchanged: `_planner_state`, `_certified`, etc.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m android_world.agents.multi_agent_test`
Expected: all tests PASS, including the two new constructor tests.

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/multi_agent.py android_world/agents/multi_agent_test.py
git commit -m "feat: add enable_ma_* module flags to MultiAgentReflectorAgent"
```

---

## Task 2: −Planner (no decomposition, no plan injection, no replan)

**Files:**
- Modify: `android_world/agents/multi_agent.py` (`_build_action_prompt`, `_planner_replan`)
- Test: `android_world/agents/multi_agent_test.py`

- [ ] **Step 1: Write the failing tests**

Append to the `ModuleAblationTest` class body (before the class's final blank line):

```python
    # ── Planner ─────────────────────────────────────────────────────

    def test_no_planner_skips_decomposition(self):
        env = test_utils.FakeAsyncEnv()
        llm = _RecordingScriptedLlm([])
        agent = ma.MultiAgentReflectorAgent(
            env, llm, enable_multiagent=True, enable_ma_planner=False)
        prompt = agent._build_action_prompt("do task", [], "")
        self.assertNotIn("## Plan", prompt)
        self.assertIsNone(agent._planner_state)
        self.assertEqual(len(llm.calls), 0)  # no planner LLM call

    def test_planner_injects_plan_block(self):
        env = test_utils.FakeAsyncEnv()
        llm = _RecordingScriptedLlm([
            "SUBGOALS:\n1. open app\n2. set time\n"
            "PROGRESS_CONDITIONS:\nP1: app open\nP2: time set\n"
            "ACCEPTANCE:\nA1: alarm: 07:00 [mandatory]",
        ])
        agent = ma.MultiAgentReflectorAgent(env, llm, enable_multiagent=True)
        prompt = agent._build_action_prompt("create alarm", [], "")
        self.assertIn("## Plan (multi-agent)", prompt)
        self.assertIn("Current subgoal: open app", prompt)

    def test_no_planner_replan_is_noop(self):
        env = test_utils.FakeAsyncEnv()
        agent = ma.MultiAgentReflectorAgent(
            env, _RecordingScriptedLlm([]), enable_multiagent=True,
            enable_ma_planner=False)
        agent._planner_state = None
        agent._replan_count = 0
        agent._planner_replan("goal")  # must not resurrect planner state
        self.assertIsNone(agent._planner_state)
        self.assertEqual(agent._replan_count, 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m android_world.agents.multi_agent_test.ModuleAblationTest`
Expected: FAIL — `test_no_planner_skips_decomposition` finds `## Plan` in the prompt (planner still runs); `test_no_planner_replan_is_noop` resurrects a planner state.

- [ ] **Step 3: Gate the planner call site and replan**

In `android_world/agents/multi_agent.py`, change `_build_action_prompt` (current lines 382-397):

```python
  def _build_action_prompt(
      self, goal: str, history_lines: list[str], ui_elements_list: str
  ) -> str:
    if not self._multiagent:
      return super()._build_action_prompt(goal, history_lines, ui_elements_list)

    # First step: plan once (only when the Planner module is enabled).
    if self._ma_planner and self._planner_state is None:
      self._planner_plan(goal, ui_elements_list)

    plan_block = self._format_plan_block()
    if plan_block:
      goal = plan_block + "\n\n" + goal
    return super()._build_action_prompt(goal, history_lines, ui_elements_list)
```

In the same file, change `_planner_replan` to no-op when the Planner is off (insert as the first statement of the method body, current line 337):

```python
    if not self._ma_planner:
      return
```

Keep the rest of `_planner_replan` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m android_world.agents.multi_agent_test`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/multi_agent.py android_world/agents/multi_agent_test.py
git commit -m "feat: gate Planner module with enable_ma_planner (decomposition + replan)"
```

---

## Task 3: −AV (no per-step action verification, U3 reverts to buffering)

**Files:**
- Modify: `android_world/agents/multi_agent.py` (`_on_step_complete`)
- Test: `android_world/agents/multi_agent_test.py`

- [ ] **Step 1: Write the failing tests**

Append to the `ModuleAblationTest` class body:

```python
    # ── Action Verifier (per-step) ──────────────────────────────────

    def test_no_av_removes_action_verify_and_buffers_u3(self):
        env = test_utils.FakeAsyncEnv()
        llm = _RecordingScriptedLlm([])
        agent = ma.MultiAgentReflectorAgent(
            env, llm, enable_multiagent=True, enable_ma_av=False,
            enable_ma_pa=False, enable_u1=True, enable_u3=True,
            rag_url="http://test.invalid")
        step_data = {
            "u2_replayed": False,
            "action_output_json": _fake_action("click", 0),
            "before_ui_elements": [],
            "after_ui_elements": [],
            "before_ui_elements_list": "",
            "after_ui_elements_list": "",
        }
        agent._on_step_complete(step_data)
        self.assertEqual(agent._step_verdicts, [])                # no AV verdict
        self.assertEqual(agent._last_reflector_feedback, "")      # no AV feedback
        self.assertEqual(len(agent._pending_u3_transitions), 1)   # U3 buffered
        self.assertEqual(len(llm.calls), 0)                       # no verifier calls

    def test_av_on_records_verdict(self):
        env = test_utils.FakeAsyncEnv()
        llm = _RecordingScriptedLlm([
            "VERDICT: NO_EFFECT\nEVIDENCE: screen unchanged",
        ])
        agent = ma.MultiAgentReflectorAgent(
            env, llm, enable_multiagent=True, enable_ma_pa=False)
        step_data = {
            "u2_replayed": False,
            "action_output_json": _fake_action("click", 0),
            "action_reason": "click to open",
            "before_ui_elements": [],
            "after_ui_elements": [],
        }
        agent._on_step_complete(step_data)
        self.assertEqual(len(agent._step_verdicts), 1)
        self.assertIn("Action Verifier", agent._last_reflector_feedback)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m android_world.agents.multi_agent_test.ModuleAblationTest`
Expected: FAIL — `test_no_av_removes_action_verify_and_buffers_u3` gets a non-empty `_step_verdicts` (AV still runs) and `_pending_u3_transitions` stays empty (no U3 buffer path).

- [ ] **Step 3: Gate the AV branch; off → super() fallback**

In `android_world/agents/multi_agent.py`, replace the whole body of `_on_step_complete` (current lines 448-468):

```python
  def _on_step_complete(self, step_data: dict[str, Any]) -> None:
    if not self._multiagent:
      super()._on_step_complete(step_data)
      return

    # Replay steps have no action_reason / claim; skip all verifiers.
    if step_data.get("u2_replayed"):
      return

    # (2) Action Verifier → gate U3 edge drawing + Executor feedback on miss.
    #     With AV off the U3 path reverts to the single-agent behavior:
    #     super()._on_step_complete buffers all transitions (flushed on
    #     episode GT success) instead of AV-gated immediate drawing.
    if self._ma_av:
      verdict = self._verify_step_action(step_data)
      self._step_verdicts.append(verdict)
      self._apply_u3_gate(step_data, verdict)
      self._apply_av_feedback(verdict)
    else:
      super()._on_step_complete(step_data)

    # (3) Progress Auditor → advance U1.completed only if the subgoal-level
    #     Evidence Certifier also passes (ordering: ADVANCING → certify).
    if self._ma_pa:
      self._audit_and_advance(step_data)

    # (4) Replicate super()'s U1 bookkeeping (app/page/last_action).  With AV
    #     off, super()._on_step_complete already did it — skip to avoid a
    #     double U1 write.
    if self._ma_av:
      self._update_u1_bookkeeping(step_data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m android_world.agents.multi_agent_test`
Expected: all PASS (new tests + all pre-existing, including `ProgressAuditorIntegrationTest`).

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/multi_agent.py android_world/agents/multi_agent_test.py
git commit -m "feat: gate Action Verifier module with enable_ma_av; U3 reverts to single-agent buffering"
```

---

## Task 4: −PA (no progress audit, no subgoal advancement, no stall-replan)

**Files:**
- Modify: `android_world/agents/multi_agent.py` (`_on_step_complete` PA gate, `_format_plan_block`)
- Test: `android_world/agents/multi_agent_test.py`

- [ ] **Step 1: Write the failing tests**

Append to the `ModuleAblationTest` class body:

```python
    # ── Progress Auditor (plan-block rendering) ─────────────────────

    def test_no_pa_hides_progress_lines_keeps_subgoals(self):
        env = test_utils.FakeAsyncEnv()
        agent = ma.MultiAgentReflectorAgent(
            env, _ScriptedLlm([]), enable_multiagent=True, enable_ma_pa=False)
        agent._planner_state = ma.PlannerState(
            subgoals=["open app", "set time"],
            current_idx=1,
            ledger=ma.ProgressLedger(
                conditions=[("P1", "app open"), ("P2", "time set")],
                satisfied={"P1"},
            ),
        )
        agent._last_reflector_feedback = "Action Verifier: NO_EFFECT — x"
        block = agent._format_plan_block()
        self.assertNotIn("Current subgoal", block)
        self.assertNotIn("Progress satisfied", block)
        self.assertNotIn("[", block)  # no progression checkboxes
        self.assertIn("Subgoals: open app → set time", block)
        self.assertIn("Action Verifier", block)  # AV feedback stays (AV on)

    def test_pa_shows_progress_lines(self):
        env = test_utils.FakeAsyncEnv()
        agent = ma.MultiAgentReflectorAgent(
            env, _ScriptedLlm([]), enable_multiagent=True)
        agent._planner_state = ma.PlannerState(
            subgoals=["open app", "set time"],
            current_idx=1,
            ledger=ma.ProgressLedger(
                conditions=[("P1", "app open"), ("P2", "time set")],
                satisfied={"P1"},
            ),
        )
        block = agent._format_plan_block()
        self.assertIn("Current subgoal: set time", block)
        self.assertIn("Progress satisfied: P1", block)
        self.assertIn("[x] open app", block)

    # ── Progress Auditor (per-step) ─────────────────────────────────

    def test_no_pa_removes_progress_audit(self):
        env = test_utils.FakeAsyncEnv()
        llm = _RecordingScriptedLlm([])
        agent = ma.MultiAgentReflectorAgent(
            env, llm, enable_multiagent=True, enable_ma_pa=False)
        agent._planner_state = ma.PlannerState(
            subgoals=["sg1", "sg2"],
            ledger=ma.ProgressLedger(conditions=[("P1", "c1")]),
        )
        step_data = {
            "u2_replayed": False,
            "action_output_json": _fake_action("click", 0),
            "before_ui_elements": [],
            "after_ui_elements": [],
        }
        agent._on_step_complete(step_data)
        self.assertEqual(agent._planner_state.current_idx, 0)
        self.assertEqual(agent._stall_steps, 0)
        self.assertEqual(agent._replan_count, 0)
        for p in llm.calls:
            self.assertNotIn("progress auditor", p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m android_world.agents.multi_agent_test.ModuleAblationTest`
Expected: FAIL — `test_no_pa_hides_progress_lines_keeps_subgoals` still sees "Current subgoal"; `test_no_pa_removes_progress_audit` still sees a progress-auditor LLM prompt.

- [ ] **Step 3: Gate PA in the plan block and the per-step path**

In `android_world/agents/multi_agent.py`, replace the whole body of `_format_plan_block` (current lines 399-418):

```python
  def _format_plan_block(self) -> str:
    state = self._planner_state
    if state is None:
      return ""
    lines = ["## Plan (multi-agent)"]
    # "Current subgoal", progression checkboxes and "Progress satisfied" are
    # Progress Auditor products: they disappear under −PA.  The subgoal list
    # itself is a Planner product and stays.  Verifier Feedback is AV's.
    if self._ma_pa:
      lines.append(f"Current subgoal: {self._current_subgoal()}")
    if state.subgoals:
      if self._ma_pa:
        rendered = " → ".join(
            f"[{'x' if i < state.current_idx else ' '}] {g}"
            for i, g in enumerate(state.subgoals))
      else:
        rendered = " → ".join(state.subgoals)
      lines.append("Subgoals: " + rendered)
    if self._ma_pa and state.ledger.satisfied:
      lines.append("Progress satisfied: " + ", ".join(sorted(state.ledger.satisfied)))
    if self._last_reflector_feedback:
      lines.append(f"## Verifier Feedback\n{self._last_reflector_feedback}")
    return "\n".join(lines)
```

In the same file, the `_on_step_complete` method already has the `if self._ma_pa:` gate around `_audit_and_advance` (added in Task 3). Confirm it is present; if the PA gate is missing, add it:

```python
    if self._ma_pa:
      self._audit_and_advance(step_data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m android_world.agents.multi_agent_test`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/multi_agent.py android_world/agents/multi_agent_test.py
git commit -m "feat: gate Progress Auditor module with enable_ma_pa; hide progress lines, keep subgoal list"
```

---

## Task 5: −EC (no acceptance certification, no completion veto, no fusion)

**Files:**
- Modify: `android_world/agents/multi_agent.py` (`_accept_task_done`, `_audit_and_advance`, `_on_task_done`, `flush_memory`, `set_episode_success`)
- Test: `android_world/agents/multi_agent_test.py`

- [ ] **Step 1: Write the failing tests**

Append to the `ModuleAblationTest` class body:

```python
    # ── Evidence Certifier ──────────────────────────────────────────

    def test_no_ec_accepts_done_without_certifier(self):
        env = test_utils.FakeAsyncEnv()
        llm = _RecordingScriptedLlm([])
        agent = ma.MultiAgentReflectorAgent(
            env, llm, enable_multiagent=True, enable_ma_ec=False)
        agent._planner_state = ma.PlannerState(
            checklist=[ma.AcceptanceItem("A1", "task", "done", True)],
        )
        self.assertTrue(
            agent._accept_task_done("goal",
                                    {"after_ui_elements": [],
                                     "after_screenshot_with_som": None}))
        self.assertIsNone(agent._certified)
        self.assertEqual(len(llm.calls), 0)  # no EC LLM call

    def test_no_ec_advances_subgoal_without_subgoal_cert(self):
        env = test_utils.FakeAsyncEnv()
        llm = _RecordingScriptedLlm([
            "VERDICT: ADVANCING\nNEW_SATISFIED: P2\nEVIDENCE: progressed",
        ])
        agent = ma.MultiAgentReflectorAgent(
            env, llm, enable_multiagent=True, enable_ma_ec=False, enable_u1=True)
        agent._planner_state = ma.PlannerState(
            subgoals=["sg1", "sg2"],
            ledger=ma.ProgressLedger(conditions=[("P1", "c1"), ("P2", "c2")]),
        )
        agent._audit_and_advance(
            {"after_ui_elements": [], "after_screenshot_with_som": None})
        self.assertEqual(agent._planner_state.current_idx, 1)  # advanced
        self.assertIn("P2", agent._planner_state.ledger.satisfied)
        self.assertEqual(len(llm.calls), 1)  # only PA, no subgoal-cert call

    def test_no_ec_flush_does_not_recertify(self):
        env = test_utils.FakeAsyncEnv()
        llm = _RecordingScriptedLlm([])
        agent = ma.MultiAgentReflectorAgent(
            env, llm, enable_multiagent=True, enable_ma_ec=False)
        agent._certified = False
        agent.flush_memory("goal")
        self.assertFalse(agent._certified)  # not flipped by re-cert
        self.assertEqual(len(llm.calls), 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m android_world.agents.multi_agent_test.ModuleAblationTest`
Expected: FAIL — `test_no_ec_accepts_done_without_certifier` calls the certifier LLM; `test_no_ec_flush_does_not_recertify` flips `_certified` to True.

- [ ] **Step 3: Gate every Evidence Certifier effect**

In `android_world/agents/multi_agent.py`, make the following five edits.

(1) `_accept_task_done` — add after the `if not self._multiagent:` guard (current line 423):

```python
    if not self._ma_ec:
      return True
```

(2) `_audit_and_advance` — change the subgoal-cert gate (current line 581):

```python
      if (not self._ma_ec) or self._certify_current_subgoal(step_data):
```

(3) `_on_task_done` — change the global-cert line (current line 674):

```python
    if self._ma_ec and self._certified is None:
      self._certified = self._certify_global(goal, step_data)
```

(4) `flush_memory` — change the re-cert line (current line 705):

```python
    if self._ma_ec and self._certified is not True:
      self._certified = self._certify_global_from_history(goal)
```

(5) `set_episode_success` — change the fusion guard (current line 727):

```python
    if self._multiagent and self._ma_ec and self._certified is not None:
      success = success and self._certified
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m android_world.agents.multi_agent_test`
Expected: all PASS (new + all pre-existing, including `CompletionVetoTest` and `FusionTest`).

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/multi_agent.py android_world/agents/multi_agent_test.py
git commit -m "feat: gate Evidence Certifier module with enable_ma_ec; no completion veto or success fusion"
```

---

## Task 6: `run.py` CLI flags

**Files:**
- Modify: `run.py`

- [ ] **Step 1: Add the four flags**

In `run.py`, after the `_MULTIAGENT` flag block (current lines 190-196), add:

```python
_MA_NO_PLANNER = flags.DEFINE_boolean(
    'ma_no_planner',
    False,
    'Disable the Planner module (decomposition, plan-block injection, replan).'
    ' Only meaningful with --multiagent.',
)
_MA_NO_AV = flags.DEFINE_boolean(
    'ma_no_av',
    False,
    'Disable the Action Verifier module (per-step action check, U3 gating,'
    ' U4 step credit). Only meaningful with --multiagent.',
)
_MA_NO_PA = flags.DEFINE_boolean(
    'ma_no_pa',
    False,
    'Disable the Progress Auditor module (subgoal advancement, stall-replan,'
    ' progress lines in the plan block). Only meaningful with --multiagent.',
)
_MA_NO_EC = flags.DEFINE_boolean(
    'ma_no_ec',
    False,
    'Disable the Evidence Certifier module (completion veto, subgoal cert,'
    ' success fusion). Only meaningful with --multiagent.',
)
```

- [ ] **Step 2: Wire them into `_get_agent`**

In `run.py`, change the `_MULTIAGENT.value` branch (current lines 335-338):

```python
    if _MULTIAGENT.value:
      agent = multi_agent.MultiAgentReflectorAgent(
          enable_multiagent=True,
          enable_ma_planner=not _MA_NO_PLANNER.value,
          enable_ma_av=not _MA_NO_AV.value,
          enable_ma_pa=not _MA_NO_PA.value,
          enable_ma_ec=not _MA_NO_EC.value,
          **agent_kwargs,
      )
    else:
      agent = memory_agent.MemoryAugmentedAgent(**agent_kwargs)
```

- [ ] **Step 3: Verify the flags register**

Run: `python run.py --help`
Expected: exit 0 and the help text lists `--ma_no_planner`, `--ma_no_av`, `--ma_no_pa`, `--ma_no_ec`. (No emulator is touched by `--help`.)

- [ ] **Step 4: Commit**

```bash
git add run.py
git commit -m "feat: add --ma_no_* CLI flags to run.py for multi-agent module ablation"
```

---

## Task 7: `ablation_hierarchical.py` flag pass-through

**Files:**
- Modify: `scripts/ablation_hierarchical.py`

- [ ] **Step 1: Add the four flags**

In `scripts/ablation_hierarchical.py`, after the `--multiagent` flag (current lines 131-137), add:

```python
flags.DEFINE_bool(
    'ma_no_planner', False,
    'Disable the Planner module (decomposition, plan-block injection, replan).')
flags.DEFINE_bool(
    'ma_no_av', False,
    'Disable the Action Verifier module (per-step action check, U3 gating,'
    ' U4 step credit).')
flags.DEFINE_bool(
    'ma_no_pa', False,
    'Disable the Progress Auditor module (subgoal advancement, stall-replan,'
    ' progress lines in the plan block).')
flags.DEFINE_bool(
    'ma_no_ec', False,
    'Disable the Evidence Certifier module (completion veto, subgoal cert,'
    ' success fusion).')
```

- [ ] **Step 2: Thread them into `_run_phase`**

In `scripts/ablation_hierarchical.py`, change the `if enable_multiagent:` block in `_run_phase` (current lines 238-239):

```python
  if enable_multiagent:
    agent_kwargs['enable_multiagent'] = True
    agent_kwargs['enable_ma_planner'] = not FLAGS.ma_no_planner
    agent_kwargs['enable_ma_av'] = not FLAGS.ma_no_av
    agent_kwargs['enable_ma_pa'] = not FLAGS.ma_no_pa
    agent_kwargs['enable_ma_ec'] = not FLAGS.ma_no_ec
```

- [ ] **Step 3: Verify the flags register**

Run: `python scripts/ablation_hierarchical.py --help`
Expected: exit 0 and the help text lists the four `--ma_no_*` flags. (No emulator is touched by `--help`.)

- [ ] **Step 4: Commit**

```bash
git add scripts/ablation_hierarchical.py
git commit -m "feat: pass --ma_no_* through ablation_hierarchical.py"
```

---

## Task 8: Full regression

**Files:**
- Run only, no edits.

- [ ] **Step 1: Run the full agent test suite**

Run: `python -m android_world.agents.multi_agent_test`
Expected: all PASS (existing verifier/planner/veto/fusion/replan tests + the new `ModuleAblationTest`).

- [ ] **Step 2: Run the memory-agent tests (touched base class path)**

Run: `python -m android_world.agents.memory_agent_test` if that file exists, else skip with a note.
Expected: PASS (the `_on_step_complete` super() fallback path is exercised by `test_no_av_removes_action_verify_and_buffers_u3` already).

- [ ] **Step 3: Sanity-check the four flags combine without error**

Run: `python run.py --help`
Expected: exit 0, flags listed. Also confirm `python run.py --multiagent --ma_no_av --ma_no_pa --ma_no_ec --ma_no_planner --help` exits 0 (all flags parse together).

- [ ] **Step 4: Summarize what each flag now does** (for the plan's reader)

Record in the commit/PR message the agreed ablation semantics:

| Flag | Removed module | Behavior reverts to m3a |
| --- | --- | --- |
| `--ma_no_planner` | Planner | no decomposition, no plan block, no replan |
| `--ma_no_av` | Action Verifier | no per-step verification, U3 buffers all transitions, U4 credit = 1.0 |
| `--ma_no_pa` | Progress Auditor | no subgoal advancement, no stall-replan, progress lines hidden (subgoal list kept) |
| `--ma_no_ec` | Evidence Certifier | no completion veto, no subgoal cert, no success fusion |

---

## Self-review notes

- **Spec coverage:** four flags → four gates (Tasks 2-5) + constructor (Task 1) + CLI wiring in both runners (Tasks 6-7) + regression (Task 8). The agreed "return to m3a" semantics are encoded per gate and pinned by tests.
- **No placeholders:** every step carries concrete code or a concrete command with expected output.
- **Type consistency:** `enable_ma_planner/av/pa/ec` are used identically across `multi_agent.py`, `run.py`, and `ablation_hierarchical.py`. `_certify_current_subgoal` follows the agreed EC ownership (`(not self._ma_ec) or ...` short-circuit in `_audit_and_advance`).
- **Out of scope (agreed earlier):** per-episode `episode_ma_stats()` instrumentation and the memory×module `CONFIGS` matrix expansion are follow-ups, to be planned after these flags are validated.
