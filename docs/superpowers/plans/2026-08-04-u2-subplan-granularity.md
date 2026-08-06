# U2 Episodic Memory — Add Sub-plan Granularity Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sub-plan granularity support to the U2 `EpisodicMemory` wrapper — the data-layer interface for storing and retrieving memories keyed by `Plan(precondition, goal)` instead of just a whole-task goal string.

**Architecture:** Two-phase upgrade (option A2). Phase 1 (this plan) adds the sub-plan-capable API surface to `EpisodicMemory` and its backend adapter, plus dynamic `T_global` threading — while keeping the existing whole-task API as the default path so nothing breaks. Phase 2 (future, separate plan) wires Planner-driven decomposition into `MemoryAugmentedAgent` to actually *use* the new interface. The memory module stays pure data — no LLM calls, no agent logic.

**Tech Stack:** Python 3, dataclasses, numpy, absl flags, `others/darwinian_memory/` (DMS library), `android_world/agents/memory/` (U1-U5 modules). No pytest available — use `python -m unittest` (stdlib).

---

## File Structure

- **Create** `android_world/agents/memory/dms_bridge.py` — thin adapter that isolates all `others/darwinian_memory` imports, exposing `Plan`, `ObsAct`, `MemoryEntry`, `RetrievalResult`, `MemoryBank`, `DMSConfig`. Keeps `episodic.py` free of direct DMS-library imports (single responsibility: U2 wrapper logic).
- **Modify** `android_world/agents/memory/episodic.py` — the U2 wrapper. Add sub-plan API methods and thread `T_global`. Keep existing methods working.
- **Create** `android_world/agents/memory/test_episodic.py` — stdlib unittest tests. Runs offline with a fake embedding backend (no sentence-transformers download).
- **Modify** `android_world/agents/memory/__init__.py` — re-export the sub-plan API.

---

### Task 1: Create the DMS bridge module

**Files:**
- Create: `android_world/agents/memory/dms_bridge.py`
- Test: `android_world/agents/memory/test_episodic.py`

- [ ] **Step 1: Write the failing test for the bridge imports**

```python
# android_world/agents/memory/test_episodic.py
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from android_world.agents.memory import dms_bridge
from android_world.agents.memory.dms_bridge import (
    DMSConfig, MemoryBank, MemoryEntry, ObsAct, Plan, RetrievalResult,
)


class DMSBridgeTest(unittest.TestCase):

    def test_plan_constructs(self):
        p = Plan(precondition="Home screen", goal="Open Markor")
        self.assertEqual(p.precondition, "Home screen")
        self.assertEqual(p.goal, "Open Markor")

    def test_obsact_constructs(self):
        oa = ObsAct(observation="step_0", action="click", step_index=0)
        self.assertEqual(oa.action, "click")

    def test_retrieval_result_defaults(self):
        r = RetrievalResult(hit=False)
        self.assertFalse(r)
        self.assertIsNone(r.entry)
        self.assertEqual(r.score, 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest android_world.agents.memory.test_episodic -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'android_world.agents.memory.dms_bridge'`

- [ ] **Step 3: Write the bridge module**

```python
"""DMS library bridge for the U2 episodic memory module.

Isolates all imports from others/darwinian_memory so the U2 wrapper
(episodic.py) depends only on this stable surface.  Keeping the heavy
DMS imports here means episodic.py can be reasoned about without
touching the DMS internals.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from others.darwinian_memory.config import DMSConfig
from others.darwinian_memory.embedding import EmbeddingBackend, TFIDFBackend
from others.darwinian_memory.memory_bank import MemoryBank, RetrievalResult
from others.darwinian_memory.memory_entry import MemoryEntry, ObsAct, Plan

__all__ = [
    "DMSConfig",
    "EmbeddingBackend",
    "TFIDFBackend",
    "MemoryBank",
    "RetrievalResult",
    "MemoryEntry",
    "ObsAct",
    "Plan",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest android_world.agents.memory.test_episodic -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory/dms_bridge.py android_world/agents/memory/test_episodic.py
git commit -m "feat: add DMS bridge module for U2 episodic memory"
```

---

### Task 2: Add sub-plan storage & retrieval API to EpisodicMemory

**Files:**
- Modify: `android_world/agents/memory/episodic.py`
- Test: `android_world/agents/memory/test_episodic.py`

- [ ] **Step 1: Write the failing test for sub-plan retrieval**

```python
# Append to test_episodic.py
from android_world.agents.memory import EpisodicMemory


class FakeBackend:
    """Deterministic fake embedding backend — no network, no ST download.

    A goal "exactly matches" itself and similar goals produce a high
    cosine score; unrelated goals score ~0.  This lets the sub-plan
    retrieval path be tested offline.
    """
    def __init__(self):
        self._dim = 8

    def encode(self, text: str):
        import numpy as np
        v = np.zeros(self._dim, dtype=np.float32)
        for i, ch in enumerate(text.lower()):
            v[i % self._dim] += ord(ch)
        norm = float(np.linalg.norm(v)) or 1.0
        return v / norm

    def encode_batch(self, texts: list[str]):
        import numpy as np
        return np.stack([self.encode(t) for t in texts], axis=0)

    @property
    def dim(self) -> int:
        return self._dim


class EpisodicMemorySubPlanTest(unittest.TestCase):

    def _make_memory(self, persistence_dir: str):
        config = DMSConfig()
        config.disk_storage_dir = persistence_dir
        mem = EpisodicMemory(config=config, persistence_dir=persistence_dir)
        mem._initialized = True
        mem.bank._embedder = FakeBackend()
        mem.bank._embedder_initialized = True
        return mem

    def test_add_and_retrieve_sub_plan(self):
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan = Plan(precondition="Markor main screen", goal="Create a new note")
            traj = [
                ObsAct(observation="step_0", action="click", step_index=0),
                ObsAct(observation="step_1", action="input_text", step_index=1),
            ]
            mem.add_sub_plan(plan, traj)
            self.assertEqual(mem.size, 1)

            hint = mem.retrieve_sub_plan_hint(plan)
            self.assertIn("click", hint)
            self.assertIn("input_text", hint)

    def test_retrieve_sub_plan_miss_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan = Plan(precondition="Home", goal="Send an SMS")
            self.assertEqual(mem.retrieve_sub_plan_hint(plan), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest android_world.agents.memory.test_episodic -v`
Expected: FAIL with `AttributeError: 'EpisodicMemory' object has no attribute 'add_sub_plan'`

- [ ] **Step 3: Refactor episodic.py to use the bridge and add sub-plan API**

Replace the DMS imports and add the sub-plan methods:

```python
# Top of episodic.py — replace the direct DMS imports:
from android_world.agents.memory import dms_bridge
from android_world.agents.memory.dms_bridge import (
    DMSConfig, MemoryBank, MemoryEntry, ObsAct, Plan, RetrievalResult,
)
```

Add these methods to the `EpisodicMemory` class (keep `retrieve_hint`, `add_trajectory`, `finalize_task` unchanged):

```python
  # ── Sub-plan granularity API (data-layer support for Planner
  #    decomposition).  The U2 wrapper does NOT generate sub-plans —
  #    that is the Planner agent's job.  It only stores and retrieves
  #    memories indexed by Plan(precondition, goal).
  #
  #    Default T_global = 0.5 keeps current behavior identical; callers
  #    that track global failure rate pass a real value for the dynamic
  #    thresholding in §3.2.4.

  def _build_retrieval_query(
      self, goal: str, precondition: str | None = None
  ) -> Plan:
    """Construct a Plan for retrieval, defaulting precondition to the goal."""
    return Plan(precondition=precondition or "", goal=goal)

  def retrieve_sub_plan_hint(
      self,
      plan: Plan,
      T_global: float = 0.5,
  ) -> str:
    """Retrieve a memory hint for a specific sub-plan.

    The hint is a compact action-type string suitable for injection into
    an action-selection prompt.

    Returns empty string on cache miss, risk block, or empty trajectory.
    """
    if not self._initialized:
      self.init_embedding()

    result: RetrievalResult = self.bank.retrieve(
        plan,
        current_logical_time=self.bank.logical_time,
        T_global=T_global,
    )

    if not result.hit:
      return ""
    if result.risk_blocked:
      print(f"[U2] sub-plan retrieve {plan.goal[:40]!r} -> RISK-BLOCKED")
      return ""
    if result.entry is None:
      return ""
    if result.should_mutate:
      print(f"[U2] sub-plan retrieve {plan.goal[:40]!r} -> ϵ-MUTATION")
      self._active_entry = result.entry
      return ""

    entry = result.entry
    trajectory = self.bank._load_trajectory(entry)
    if not trajectory:
      return ""

    hint = _trajectory_action_string(trajectory)
    print(
        f"[U2] sub-plan retrieve {plan.goal[:40]!r} -> HIT "
        f"score={result.score:.3f} hint={hint!r}"
    )
    self._active_entry = entry
    return hint

  def add_sub_plan(
      self,
      plan: Plan,
      trajectory: list[ObsAct],
  ) -> MemoryEntry | None:
    """Store a trajectory keyed by a sub-plan.

    Filters out atomic trajectories (|τ| ≤ 1) per DMS §3.2.1.
    Returns the new MemoryEntry, or None if filtered.
    """
    if not self._initialized:
      self.init_embedding(corpus_texts=[plan.goal])

    entry = self.bank.add(plan, trajectory)
    if entry is not None:
      print(
          f"[U2] add sub-plan {plan.goal[:40]!r} -> "
          f"stored {entry.memory_id[:8]} (|τ|={entry.trajectory_length})"
      )
      self._last_added_entry = entry
    else:
      print(f"[U2] add sub-plan {plan.goal[:40]!r} -> skipped (atomic |τ|<=1)")
    return entry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest android_world.agents.memory.test_episodic -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory/episodic.py android_world/agents/memory/test_episodic.py
git commit -m "feat: add sub-plan granularity API to U2 episodic memory"
```

---

### Task 3: Add `retrieve_hint` precondition parameter (whole-task path)

**Files:**
- Modify: `android_world/agents/memory/episodic.py`
- Test: `android_world/agents/memory/test_episodic.py`

- [ ] **Step 1: Write the failing test for precondition in whole-task retrieval**

```python
# Append to test_episodic.py
class EpisodicMemoryWholeTaskTest(unittest.TestCase):

    def _make_memory(self, persistence_dir: str):
        config = DMSConfig()
        config.disk_storage_dir = persistence_dir
        mem = EpisodicMemory(config=config, persistence_dir=persistence_dir)
        mem._initialized = True
        mem.bank._embedder = FakeBackend()
        mem.bank._embedder_initialized = True
        return mem

    def test_retrieve_hint_accepts_precondition(self):
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan = Plan(precondition="Markor main screen", goal="Create a new note")
            traj = [
                ObsAct(observation="step_0", action="open_app", step_index=0),
                ObsAct(observation="step_1", action="click", step_index=1),
            ]
            mem.add_sub_plan(plan, traj)
            hint = mem.retrieve_hint(
                "Create a new note", precondition="Markor main screen"
            )
            self.assertIn("open_app", hint)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest android_world.agents.memory.test_episodic -v`
Expected: FAIL with `TypeError: retrieve_hint() got an unexpected keyword argument 'precondition'`

- [ ] **Step 3: Modify retrieve_hint to accept precondition**

Change the signature and the query construction:

```python
  def retrieve_hint(
      self,
      goal: str,
      precondition: str | None = None,
      T_global: float = 0.5,
  ) -> str:
    """Return a compact memory hint string for injection into the prompt.

    Returns empty string on cache miss, risk block, or if no memory bank
    is initialised.

    Results are cached per (goal, precondition) so repeated calls within
    one episode only hit the bank once; the cache is cleared in
    finalize_task().

    Args:
      goal: The task or sub-task goal text.
      precondition: Optional UI-state description.  When provided, enables
        the dual-factor retrieval (§3.2.2) which matches both the starting
        state context AND the goal, reducing false positives.
      T_global: Global failure rate for dynamic thresholding (§3.2.4).
    """
    cache_key = (goal, precondition)
    if cache_key in self._retrieval_cache:
      return self._retrieval_cache[cache_key]

    if not self._initialized:
      self.init_embedding()

    plan = self._build_retrieval_query(goal, precondition)
    result: RetrievalResult = self.bank.retrieve(
        plan,
        current_logical_time=self.bank.logical_time,
        T_global=T_global,
    )

    if not result.hit:
      print(f"[U2] retrieve goal={goal[:50]!r} -> miss (no entry above threshold)")
      self._retrieval_cache[cache_key] = ""
      return ""
    if result.risk_blocked:
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but RISK-BLOCKED")
      self._retrieval_cache[cache_key] = ""
      return ""
    if result.entry is None:
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but entry is None")
      self._retrieval_cache[cache_key] = ""
      return ""
    if result.should_mutate:
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but ϵ-MUTATION (re-explore)")
      self._active_entry = result.entry
      self._retrieval_cache[cache_key] = ""
      return ""

    entry = result.entry
    trajectory = self.bank._load_trajectory(entry)
    if not trajectory:
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but empty trajectory")
      self._retrieval_cache[cache_key] = ""
      return ""

    hint = _trajectory_action_string(trajectory)
    print(
        f"[U2] retrieve goal={goal[:50]!r} -> HIT score={result.score:.3f} "
        f"reuse={entry.meta.reuse_count} hint={hint!r}"
    )
    self._active_entry = entry
    self._retrieval_cache[cache_key] = hint
    return hint
```

Also update `add_trajectory` to accept an optional `precondition` so whole-task storage can carry context:

```python
  def add_trajectory(
      self,
      goal: str,
      trajectory: list[ObsAct],
      precondition: str | None = None,
  ) -> MemoryEntry | None:
    """Store a new trajectory in the memory bank.

    Filters out atomic trajectories (|τ| ≤ 1) per DMS §3.2.1.

    Returns the new MemoryEntry, or None if filtered.
    """
    if not self._initialized:
      self.init_embedding(corpus_texts=[goal])

    plan = Plan(precondition=precondition or "", goal=goal)
    entry = self.bank.add(plan, trajectory)
    if entry is not None:
      print(
          f"[U2] add goal={goal[:50]!r} -> stored entry {entry.memory_id[:8]} "
          f"(|τ|={entry.trajectory_length}, bank_size={self.bank.size})"
      )
      self._last_added_entry = entry
    else:
      print(f"[U2] add goal={goal[:50]!r} -> skipped (atomic trajectory |τ|<=1)")
    return entry
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest android_world.agents.memory.test_episodic -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory/episodic.py android_world/agents/memory/test_episodic.py
git commit -m "feat: accept precondition and T_global in whole-task U2 retrieval"
```

---

### Task 4: Thread `T_global` through the agent path

**Files:**
- Modify: `android_world/agents/memory/episodic.py`
- Modify: `android_world/agents/memory_agent.py`

- [ ] **Step 1: Add the global-failure-rate tracker to EpisodicMemory**

Add a method and internal counter so `T_global` can be tracked from episode outcomes:

```python
  def record_episode_outcome(self, success: bool) -> None:
    """Track episode-level success/failure for the global failure rate.

    The global failure rate T_global feeds the dynamic risk threshold
    (§3.2.4).  This is a lightweight per-episode tracker — it is NOT a
    per-plan Bayesian model (that requires a Planner to name plans).
    """
    if success:
      self._episode_successes += 1
    else:
      self._episode_failures += 1

  @property
  def global_failure_rate(self) -> float:
    """Smoothed global failure rate T_global (Bayesian prior + observed)."""
    total = self._episode_failures + self._episode_successes
    if total == 0:
      return 0.5  # Uniform prior
    # Blend prior base rate with observed data.
    prior_fail = self.config.alpha_prior
    prior_total = self.config.alpha_prior + self.config.beta_prior
    prior_rate = prior_fail / prior_total
    return (self._episode_failures + prior_total * prior_rate) / (
        self._episode_failures + self._episode_successes + prior_total
    )
```

Initialize the counters in `__init__` (add to the existing attribute block):

```python
    # Episode-outcome counters for global failure rate T_global (§3.2.4)
    self._episode_successes: int = 0
    self._episode_failures: int = 0
```

- [ ] **Step 2: Wire record_episode_outcome into the agent's success callback**

In `memory_agent.py`, modify `set_episode_success`:

```python
  def set_episode_success(self, success: bool) -> None:
    """Finalize the current episode's U2 memory with the true outcome.

    Called by the evaluator after it computes task.is_successful().  Overrides
    the agent's own completion claim (which can be wrong) with ground truth.
    """
    if not self.enable_u2 or self.u2 is None:
      return
    # Feed the global failure rate before flushing the trajectory.
    self.u2.record_episode_outcome(success)
    goal = self._pending_trajectory_goal
    self._pending_trajectory_goal = None
    self._flush_u2_trajectory(goal, {}, success=success)
```

- [ ] **Step 3: Verify the full test suite passes**

Run: `python -m unittest android_world.agents.memory.test_episodic -v`
Expected: PASS (all tests)

Also verify the agent module imports cleanly (no pytest; this exercises the import graph):

Run: `python -c "from android_world.agents.memory_agent import MemoryAugmentedAgent; print('OK')"`
Expected: `OK` (may print a FutureWarning about google.generativeai — harmless)

- [ ] **Step 4: Commit**

```bash
git add android_world/agents/memory/episodic.py android_world/agents/memory_agent.py
git commit -m "feat: track and thread global failure rate into U2 dynamic threshold"
```

---

### Task 5: Re-export sub-plan API from the memory package

**Files:**
- Modify: `android_world/agents/memory/__init__.py`

- [ ] **Step 1: Update the package exports**

```python
"""Memory infrastructure for GUI agents.

This package contains the five-class memory taxonomy (U1-U5) as pure data
modules.  They are NOT agents — they have no LLM calls, no environment
interaction, and no agent inheritance.

U1: Task State     — structured per-step task progress
U2: Episodic       — per-task trajectory records (DMS-powered)
U3: Environment    — app/page/element knowledge (PG-Agent page-graph RAG)
U4: Procedural     — abstracted multi-trajectory workflows
U5: Control        — memory operations controller

U1-U4 are the *data layer* (what to save).
U5 is the *control layer* (when/how to write/retrieve/update).
"""

from android_world.agents.memory.dms_bridge import (
    DMSConfig, MemoryBank, MemoryEntry, ObsAct, Plan, RetrievalResult,
)
from android_world.agents.memory.environment import EnvKnowledge, build_screen_summary
from android_world.agents.memory.episodic import EpisodicMemory, ObsAct
from android_world.agents.memory.task_state import (
    TaskState,
    extract_app_from_elements,
    format_u1_context,
    init_task_state,
    update_task_state,
)
```

- [ ] **Step 2: Verify imports work from the package level**

Run: `python -c "from android_world.agents.memory import EpisodicMemory, Plan, ObsAct, DMSConfig; print('package imports OK')"`
Expected: `package imports OK`

- [ ] **Step 3: Run full unit test suite**

Run: `python -m unittest android_world.agents.memory.test_episodic -v`
Expected: PASS (all tests)

- [ ] **Step 4: Commit**

```bash
git add android_world/agents/memory/__init__.py
git commit -m "feat: re-export sub-plan API from memory package"
```

---

### Task 6: Manual smoke test with the two-round U2 protocol

**Files:**
- None (uses existing `scripts/test_u1_u2.py`)

- [ ] **Step 1: Confirm the code imports and the module graph is sound**

Run: `cd C:\Users\WRQ\Desktop\androidworld && python -m unittest android_world.agents.memory.test_episodic -v`
Expected: All tests PASS

- [ ] **Step 2: Run the two-round U2 protocol on a small task set**

This exercises the whole-task path (`retrieve_hint` with default `T_global=0.5`) end-to-end on the emulator.  Requires adb at `D:\Data\Android\platform-tools\adb.exe` and an emulator on port 5554.

Run:
```bash
python scripts/test_u1_u2.py --tasks=MarkorCreateNote,SimpleCalendarAddOneEvent --n=1 --seed=30
```

Expected:
- Round 1 (`u1u2_record`): U2 bank is empty; `[U2] retrieve ... -> miss` logs appear; trajectories get stored.
- Round 2 (`u1u2_verify`): with new seed (new task params), some `[U2] retrieve ... -> HIT` logs appear where goals embed closely.
- Both rounds finish without exceptions; `scripts/results/<run_id>_u1u2_record.json` and `scripts/results/<run_id>_u1u2_verify.json` written.

- [ ] **Step 3: Confirm no regression in the agent path**

The `--u2` single-round flag also exercises the path:
```bash
python scripts/test_u1_u2.py --u2 --tasks=MarkorCreateNote --n=1 --seed=30
```
Expected: One round, completes without error.

- [ ] **Step 4: (Optional) Verify sub-plan API directly in a REPL**

```bash
cd C:\Users\WRQ\Desktop\androidworld && python -c "
from android_world.agents.memory import EpisodicMemory, Plan, ObsAct, DMSConfig
import tempfile, os
d = tempfile.mkdtemp()
cfg = DMSConfig(); cfg.disk_storage_dir = d
m = EpisodicMemory(config=cfg, persistence_dir=d)
m.init_embedding()  # downloads all-MiniLM-L6-v2 on first run (network required)
p = Plan(precondition='Markor main', goal='Create a note')
traj = [ObsAct(observation='s0', action='click', step_index=0), ObsAct(observation='s1', action='input_text', step_index=1)]
m.add_sub_plan(p, traj)
print('sub-plan hint:', m.retrieve_sub_plan_hint(p))
"
```
Expected: prints a non-empty sub-plan hint like `click → input_text` after the model downloads.

---

## Self-Review

**Spec coverage:** The plan covers the full A2 phase-1 scope: sub-plan-capable storage/retrieval API (`add_sub_plan`, `retrieve_sub_plan_hint`), precondition threading into the existing whole-task path (`retrieve_hint(precondition=...)`, `add_trajectory(precondition=...)`), dynamic `T_global` tracking (`record_episode_outcome`, `global_failure_rate`, threading through `set_episode_success`), the DMS bridge to isolate library coupling, and package re-exports. Phase-2 Planner decomposition is explicitly deferred (future plan) — the data layer now supports it without requiring it.

**Placeholder scan:** Every step has concrete code and exact commands. No TBD/TODO.

**Type consistency:** `retrieve_sub_plan_hint(plan: Plan, T_global: float = 0.5)` and `add_sub_plan(plan: Plan, trajectory: list[ObsAct])` are consistent across Tasks 2-5. `retrieve_hint(goal, precondition=None, T_global=0.5)` and `add_trajectory(goal, trajectory, precondition=None)` are consistent in Task 3. `record_episode_outcome`/`global_failure_rate` match Task 4 usage in `memory_agent.py`. `DMSConfig`, `MemoryBank`, `MemoryEntry`, `ObsAct`, `Plan`, `RetrievalResult` are re-exported once in Task 1 and used consistently thereafter.
