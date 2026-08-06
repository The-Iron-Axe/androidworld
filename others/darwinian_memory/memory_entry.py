"""
Darwinian Memory System — Memory Entry Data Structures
=======================================================
Defines the core data types used throughout DMS, as specified in §3.2.1.

A memory entry m is a tuple:
    m = (p, τ, s_meta)

where:
  p   = ⟨Precondition, Goal⟩  — natural-language plan, used as semantic index
  τ   = {(o₀,a₀), …, (o_T,a_T)} — dense execution trajectory
  s_meta = metadata (success count, failure count, timestamps, etc.)
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════
# Sub-task Plan  p = ⟨Precondition, Goal⟩
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Plan:
    """A Planner-generated sub-task decomposed into Precondition + Goal.

    Per §3.2.1, this structured format serves as the natural ground-truth
    summary for the atomic actions executed by the Actor, eliminating
    hallucination risks from post-hoc summarization.
    """
    precondition: str
    """Requisite UI state before the sub-task can begin.  Describes the
    expected screen / app / state context."""

    goal: str
    """Target state transformation.  What this sub-task should achieve."""

    def to_text(self) -> str:
        """Combined text used for embedding-based semantic retrieval."""
        return f"Precondition: {self.precondition}\nGoal: {self.goal}"

    def __repr__(self) -> str:
        pre = self.precondition[:60] + "…" if len(self.precondition) > 60 else self.precondition
        g = self.goal[:60] + "…" if len(self.goal) > 60 else self.goal
        return f"Plan(pre={pre!r}, goal={g!r})"


# ═══════════════════════════════════════════════════════════════════════
# Observation–Action pair (one step in a trajectory)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ObsAct:
    """A single (observation, action) pair in the execution trajectory τ."""
    observation: Any
    """Environment observation at this time step (screenshot, UI tree, etc.)."""

    action: Any
    """Atomic action executed by the Actor (tap, swipe, type, etc.)."""

    step_index: int = 0
    """Sequential index of this step within the trajectory."""


# ═══════════════════════════════════════════════════════════════════════
# Memory Metadata  s_meta
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryMeta:
    """Metadata tracked per memory entry for survival-value computation.

    Tracks the three pillars of §3.2.3:
      Utility      — via reuse_count (n_i)
      Temporal     — via created_at / last_used_at (Δt)
      Reliability  — via verification_failures (K_i)
    """
    reuse_count: int = 0
    """Number of times this memory has been successfully reused (n_i)."""

    success_count: int = 0
    """Cumulative execution successes (S_i for Bayesian reputation)."""

    failure_count: int = 0
    """Cumulative execution failures (F_i for Bayesian reputation)."""

    verification_failures: int = 0
    """Accumulated verification-failure strikes (K_i).  When this reaches
    K_limit (=3), the memory is permanently pruned (Appendix D)."""

    created_at: float = field(default_factory=time.time)
    """Wall-clock timestamp of memory creation (epoch seconds)."""

    last_used_at: float = field(default_factory=time.time)
    """Wall-clock timestamp of last successful reuse.  Used together with
    a logical-time counter to compute Δt for temporal decay."""

    last_logical_time: int = 0
    """Logical time step (planning-cycle index) of last reuse.
    Δt = current_logical_time - last_logical_time."""

    description: str = ""
    """Human-readable description of what this memory covers."""


# ═══════════════════════════════════════════════════════════════════════
# Memory Entry  m = (p, τ, s_meta)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """A single entry in the Darwinian Memory Bank.

    Per §3.2.1, each memory stores:
      - A sub-task plan p (the semantic index)
      - The execution trajectory τ (the reusable procedural knowledge)
      - Metadata s_meta (for survival-value tracking)
    """
    memory_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    """Unique identifier for this memory entry."""

    plan: Optional[Plan] = None
    """The Planner-generated sub-task p = ⟨Precondition, Goal⟩."""

    trajectory: list[ObsAct] = field(default_factory=list)
    """The dense execution trajectory τ = {(o₀,a₀), …, (o_T,a_T)}.
    Per §3.2.1, entries with |τ| = 1 are filtered out to prevent
    memory fragmentation from single atomic actions."""

    meta: MemoryMeta = field(default_factory=MemoryMeta)
    """Metadata for survival-value and risk computations."""

    # ── Cached embedding vectors (populated by MemoryBank) ────────
    _embedding_pre: Optional[list[float]] = field(default=None, repr=False)
    _embedding_goal: Optional[list[float]] = field(default=None, repr=False)

    @property
    def trajectory_length(self) -> int:
        """|τ| — number of atomic steps in the execution trajectory."""
        return len(self.trajectory)

    @property
    def is_atomic(self) -> bool:
        """True if this memory contains only a single atomic action.
        These are filtered out by MemoryBank.add() per §3.2.1."""
        return self.trajectory_length <= 1

    def __repr__(self) -> str:
        pid = self.memory_id[:8]
        g = self.plan.goal[:50] if self.plan else "(no plan)"
        return (f"MemoryEntry(id={pid}, goal={g!r}, "
                f"|τ|={self.trajectory_length}, "
                f"reuse={self.meta.reuse_count}, "
                f"fails={self.meta.verification_failures})")
