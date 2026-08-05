"""U2 Episodic Memory — cross-task trajectory storage and retrieval.

Wraps the Darwinian Memory System (others/darwinian_memory/) as a pure data
module.  U2 is NOT an agent — it has no LLM calls, no environment interaction.

Usage (from an agent)::

    u2 = EpisodicMemory(persistence_dir="./mem_store")
    hint = u2.retrieve_hint("Create a note in Markor and share via SMS")
    # hint is a compact string like "open_app → click 3 → input_text ... →"
    if hint:
        # Inject into action-selection prompt

    # After task completes:
    u2.add_trajectory(goal, obs_act_list)
    u2.finalize_task(goal, success=True)

DMS mechanisms used:
  - Dual-Factor Retrieval:  sim(pre)×sim(goal) cos-similarity
  - ε-Mutation:  10% re-exploration probability
  - Bayesian Risk Gating:  LCB score → dynamic threshold
  - Survival Value:  S = Utility × AdaptiveDecay × Reliability
  - K-Verification:  K=3 strikes → prune
  - Elbow Method:  adaptive capacity regulation
"""

from __future__ import annotations

import os
from typing import Any

from android_world.agents.memory import dms_bridge
from android_world.agents.memory.dms_bridge import (
    DMSConfig, MemoryBank, MemoryEntry, ObsAct, Plan, RetrievalResult,
)


def _trajectory_action_string(trajectory: list[ObsAct]) -> str:
  """Produce a compact action-type summary of a trajectory (for prompt hints).

  Only action *types* are kept (open_app, click, input_text, scroll...).
  Element indices and typed text are deliberately dropped — they are stale on
  re-execution (the accessibility tree is re-indexed each step) and can
  mislead the agent into tapping the wrong element or repeating bad inputs.
  """
  parts = []
  for oa in trajectory:
    action = oa.action
    if action is None:
      continue
    at = getattr(action, "action_type", str(action))
    if at == "click":
      parts.append("click")
    elif at == "input_text":
      parts.append("input_text")
    elif at == "scroll":
      parts.append(f"scroll {getattr(action, 'direction', '?')}")
    elif at == "open_app":
      app = getattr(action, "app_name", None)
      parts.append(f"open_app({app})" if app else "open_app")
    elif at == "long_press":
      parts.append("long_press")
    else:
      parts.append(at)
  return " → ".join(parts) if parts else ""


class EpisodicMemory:
  """U2 episodic memory — wraps DMS MemoryBank with a simpler, goal-level API.

  The DMS internally uses Plan(precondition, goal) for fine-grained retrieval.
  This wrapper uses the full task goal as the plan goal, with an optional
  precondition for dual-factor retrieval.  Full Planner-based decomposition
  can be added later without changing this API.
  """

  def __init__(
      self,
      config: DMSConfig | None = None,
      persistence_dir: str = "",
  ):
    # Point the DMS trajectory store at the same directory as the bank index,
    # so traj_*.pkl and dms_bank.json always live together (and both survive
    # a persistence_dir move).  Otherwise trajectories silently fall back to
    # the default "./dms_memory_store" relative to cwd, splitting them from
    # the index and breaking retrieval after a reload.
    self.config = config or DMSConfig()
    if persistence_dir:
      self.config.disk_storage_dir = persistence_dir
    self.bank = MemoryBank(config=self.config)
    self._initialized = False
    self._persistence_dir = persistence_dir
    self._bank_path = (
        os.path.join(persistence_dir, "dms_bank.json") if persistence_dir else ""
    )

    # Track active memory entry for the current task (for reuse/penalize)
    self._active_entry: MemoryEntry | None = None
    # Freshly stored trajectory in the current task (credit its first outcome)
    self._last_added_entry: MemoryEntry | None = None
    # Per-key retrieval cache — cleared after each task so one episode only
    # queries the bank once.  Keys are (goal, precondition) for hints and
    # ("replay", goal, precondition) for replay trajectories; values are the
    # hint string, the replayed trajectory list, or None on a cache miss.
    self._retrieval_cache: dict[tuple[str, ...], str | list[ObsAct] | None] = {}

    # Episode-outcome counters for global failure rate T_global (§3.2.4)
    self._episode_successes: int = 0
    self._episode_failures: int = 0

    if persistence_dir:
      os.makedirs(persistence_dir, exist_ok=True)
      if os.path.exists(self._bank_path):
        try:
          self.bank = MemoryBank.load(self._bank_path, self.config)
        except Exception:
          pass  # Corrupted or empty — start fresh

  # ── Initialisation ──────────────────────────────────────────────────

  def init_embedding(self, corpus_texts: list[str] | None = None):
    """Initialise SentenceTransformer embedding. Call once before first retrieval.

    Raises on failure — does not fall back to TF-IDF.
    """
    if self._initialized:
      return

    self.bank.initialize_embedding(corpus_texts)
    backend = self.bank._embedder
    if backend is None or type(backend).__name__ != "SentenceTransformerBackend":
      raise RuntimeError(
          "[U2] SentenceTransformer embedding is required; "
          f"got {type(backend).__name__ if backend is not None else None}"
      )
    self._initialized = True

  # ── Retrieval ───────────────────────────────────────────────────────

  def _retrieve_entry(
      self, goal: str, precondition: str | None = None
  ) -> MemoryEntry | None:
    """Run the DMS retrieval decision and return the entry with a loaded
    trajectory, or None on miss / risk-block / mutation / empty trajectory.

    Shared by retrieve_hint (prompt injection) and retrieve_replay
    (deterministic replay).  Applies the same dual-factor scoring, Bayesian
    risk gate, epsilon mutation, and disk trajectory load as the paper's
    Algorithm 1.  The risk threshold uses the tracked global failure rate
    T_global (§3.2.4).
    """
    if not self._initialized:
      self.init_embedding()

    plan = self._build_retrieval_query(goal, precondition)
    result: RetrievalResult = self.bank.retrieve(
        plan,
        current_logical_time=self.bank.logical_time,
        T_global=self.global_failure_rate,
    )

    # Log every retrieval outcome so U2 effectiveness can be verified from
    # the terminal: record/verify rounds should show hits appearing in the
    # verify round only.
    if not result.hit:
      print(f"[U2] retrieve goal={goal[:50]!r} -> miss (no entry above threshold)")
      return None
    if result.risk_blocked:
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but RISK-BLOCKED")
      return None
    if result.entry is None:
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but entry is None")
      return None
    if result.should_mutate:
      # eps-MUTATION triggered — force re-exploration, but still hint
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but eps-MUTATION (re-explore)")
      self._active_entry = result.entry
      return None

    entry = result.entry
    # The DMS retrieve() already loaded the trajectory from disk on a
    # definitive hit, so entry.trajectory is guaranteed non-empty here
    # (miss / risk-block / mutation / None-entry all returned above).
    trajectory = entry.trajectory
    self._active_entry = entry
    print(
        f"[U2] retrieve goal={goal[:50]!r} -> HIT score={result.score:.3f} "
        f"reuse={entry.meta.reuse_count} hint={_trajectory_action_string(trajectory)!r}"
    )
    return entry

  def retrieve_hint(
      self,
      goal: str,
      precondition: str | None = None,
  ) -> str:
    """Return a compact memory hint string for injection into the prompt.

    Returns empty string on cache miss, risk block, or if no memory bank
    is initialised.

    Results are cached per (goal, precondition) so repeated calls within one
    episode only hit the bank once; the cache is cleared in finalize_task().

    Args:
      goal: The task or sub-task goal text.
      precondition: Optional UI-state description.  When provided, enables
        the dual-factor retrieval (§3.2.2) which matches both the starting
        state context AND the goal, reducing false positives.
    """
    cache_key = (goal, precondition or "")
    if cache_key in self._retrieval_cache:
      return self._retrieval_cache[cache_key]

    entry = self._retrieve_entry(goal, precondition)
    if entry is None:
      self._retrieval_cache[cache_key] = ""
      return ""

    hint = _trajectory_action_string(entry.trajectory)
    self._retrieval_cache[cache_key] = hint
    return hint

  def retrieve_replay(
      self, goal: str, precondition: str | None = None
  ) -> list[ObsAct] | None:
    """Return the full cached trajectory for deterministic replay, or None.

    Unlike retrieve_hint (which returns a compact prompt string), this returns
    the executable ObsAct list so the agent can replay the actions verbatim
    (§3.2.2: 'the Actor reuses the stored τ').  Results are cached per
    (goal, precondition) for the current episode, and a fresh list copy is
    returned so external mutation cannot pollute the bank.
    """
    cache_key = ("replay", goal, precondition or "")
    if cache_key in self._retrieval_cache:
      value = self._retrieval_cache[cache_key]
      return list(value) if isinstance(value, list) else None

    entry = self._retrieve_entry(goal, precondition)
    if entry is None:
      self._retrieval_cache[cache_key] = None
      return None

    self._retrieval_cache[cache_key] = list(entry.trajectory)
    return list(entry.trajectory)

  # ── Sub-plan granularity ────────────────────────────────────────────
  #
  # Sub-plan granularity API (data-layer support for Planner
  # decomposition).  The U2 wrapper does NOT generate sub-plans — that
  # is the Planner agent's job.  It only stores and retrieves memories
  # indexed by Plan(precondition, goal).
  #
  # Default T_global = 0.5 keeps current behavior identical; callers
  # that track global failure rate pass a real value for the dynamic
  # thresholding in §3.2.4.

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
      print(f"[U2] sub-plan retrieve {plan.goal[:40]!r} -> eps-MUTATION")
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

  # ── Writing ─────────────────────────────────────────────────────────

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
      # Freshly stored trajectory — track separately from a retrieved entry
      # so finalize_task can credit its first success/failure without counting
      # it as a "reuse" (reuse_count is reserved for retrieved memories).
      self._last_added_entry = entry
    else:
      print(f"[U2] add goal={goal[:50]!r} -> skipped (atomic trajectory |τ|<=1)")
    return entry

  # ── Finalisation ────────────────────────────────────────────────────

  def finalize_task(self, goal: str, success: bool):
    """Update metadata after task completion.

    On success with an active memory hit: increment reuse_count.
    On failure with an active memory hit: record failure, trigger K-Verification.

    Auto-persists the bank index to disk if persistence_dir was configured.
    """
    if self._active_entry is not None:
      if success:
        self._active_entry.meta.reuse_count += 1
        self._active_entry.meta.success_count += 1
      else:
        self._active_entry.meta.failure_count += 1
        # K-Verification: 3 strikes → prune
        self._active_entry.meta.verification_failures += 1
        if self._active_entry.meta.verification_failures >= self.config.K_verify:
          self.bank._remove_entry(self._active_entry)
          print(
              f"[U2] finalize goal={goal[:50]!r} success={success} -> "
              f"PRUNED after {self.config.K_verify} verification failures"
          )
      print(
          f"[U2] finalize goal={goal[:50]!r} success={success} -> "
          f"reused-entry: reuse={self._active_entry.meta.reuse_count} "
          f"fails={self._active_entry.meta.failure_count}"
      )
    elif self._last_added_entry is not None:
      # Newly stored trajectory — record its initial success/failure outcome.
      if success:
        self._last_added_entry.meta.success_count += 1
      else:
        self._last_added_entry.meta.failure_count += 1
      print(
          f"[U2] finalize goal={goal[:50]!r} success={success} -> "
          f"new-entry: success={self._last_added_entry.meta.success_count} "
          f"fails={self._last_added_entry.meta.failure_count}"
      )
    else:
      print(f"[U2] finalize goal={goal[:50]!r} success={success} -> no memory involved")

    # Advance logical clock
    self.bank.advance_time(1)
    self._active_entry = None
    self._last_added_entry = None
    self._retrieval_cache.clear()

    # Auto-persist after every task
    self.save()

  # ── Persistence ─────────────────────────────────────────────────────

  def save(self, path: str | None = None):
    """Persist memory bank to disk."""
    p = path or self._bank_path
    if p:
      os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
      self.bank.save(p)

  # ── Stats ───────────────────────────────────────────────────────────

  @property
  def size(self) -> int:
    return self.bank.size

  def stats(self) -> dict[str, Any]:
    return self.bank.stats()

  # ── Global failure rate T_global (§3.2.4) ──────────────────────────

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
