"""
Darwinian Memory System — Memory Bank (§3.2.1–§3.2.2)
========================================================
The MemoryBank implements the core storage, retrieval, and evolution
mechanisms of DMS:

  1. Memory Construction (§3.2.1):
     - Decomposes workflows into independent p = ⟨Precondition, Goal⟩ units
     - Filters out atomic actions (|τ| = 1)
     - Decoupled storage: trajectories on disk, semantic indices in memory

  2. Memory Utilization (§3.2.2):
     - Dual-Factor Retrieval: Score(ˆp, p) = sim(φ(ˆp_pre), φ(p_pre)) · sim(φ(ˆp_goal), φ(p_goal))
     - ϵ-Mutation: with probability ε, re-explore even on retrieval hit
     - Evolutionary Replacement: if mutated trajectory is shorter, overwrite

  3. Self-Regulation Integration (§3.2.3):
     - Survival value scoring and pruning
     - Adaptive capacity regulation via Elbow Method
"""

from __future__ import annotations

import os
import json
import pickle
import time
import random
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from .config import DMSConfig, default_config
from .memory_entry import MemoryEntry, Plan, ObsAct, MemoryMeta
from .embedding import (
    EmbeddingBackend,
    SentenceTransformerBackend,
    dual_factor_similarity,
)
from .survival import (
    compute_survival_value,
    adaptive_regulate,
    rank_by_survival,
)
from .risk import compute_memory_risk_score, compute_dynamic_threshold


# ═══════════════════════════════════════════════════════════════════════
# Retrieval result
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalResult:
    """Result of a memory retrieval operation."""
    hit: bool
    """True if a memory matched above the retrieval threshold."""

    entry: Optional[MemoryEntry] = None
    """The retrieved memory entry (if hit)."""

    score: float = 0.0
    """Dual-factor similarity score."""

    should_mutate: bool = False
    """Whether ϵ-mutation was triggered for this retrieval."""

    risk_blocked: bool = False
    """Whether the memory was blocked by risk assessment."""

    def __bool__(self) -> bool:
        return self.hit and not self.risk_blocked


# ═══════════════════════════════════════════════════════════════════════
# Memory Bank
# ═══════════════════════════════════════════════════════════════════════

class MemoryBank:
    """The Darwinian Memory Bank — a self-evolving memory ecosystem.

    Stores memory entries with decoupled architecture:
      - Semantic index:  embedding vectors + metadata in memory
      - Heavy trajectories:  persisted to disk as pickle files

    Provides dual-factor retrieval, evolutionary replacement, and
    self-regulation pruning.

    Usage
    -----
    >>> bank = MemoryBank()
    >>> bank.initialize_embedding(["sample text 1", "sample text 2"])
    >>> plan = Plan(precondition="Settings app open", goal="Turn off WiFi")
    >>> entry = bank.add(plan, trajectory_obs_acts)
    >>> result = bank.retrieve(plan, current_time=0)
    >>> if result.hit:
    ...     replayed = result.entry.trajectory
    """

    def __init__(
        self,
        config: DMSConfig = default_config,
        embedding_backend: Optional[EmbeddingBackend] = None,
    ):
        self.config = config
        self._entries: dict[str, MemoryEntry] = {}  # memory_id → entry
        self._logical_time: int = 0
        self._total_plans_executed: int = 0

        # Embedding
        self._embedder: Optional[EmbeddingBackend] = embedding_backend
        self._embedder_initialized = embedding_backend is not None

        # Disk storage directory
        self._disk_dir = config.disk_storage_dir
        os.makedirs(self._disk_dir, exist_ok=True)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    @property
    def logical_time(self) -> int:
        return self._logical_time

    # ── Embedding Initialization ────────────────────────────────────

    def initialize_embedding(
        self,
        corpus_texts: Optional[list[str]] = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the embedding backend with SentenceTransformer.

        Parameters
        ----------
        corpus_texts : list[str] or None
            Unused when SentenceTransformer loads successfully (kept for API compat).
        model_name : str
            Sentence-transformers model name.

        Raises
        ------
        Exception
            If SentenceTransformer fails to load (missing package, HF/network, etc.).
            Does not silently fall back to TF-IDF.
        """
        del corpus_texts  # ST path does not use a TF-IDF corpus
        if self._embedder_initialized:
            return

        self._embedder = SentenceTransformerBackend(model_name)
        self._embedder_initialized = True

    def _ensure_embedder(self):
        if self._embedder is None:
            raise RuntimeError(
                "Embedding backend not initialized; call initialize_embedding() first "
                "(SentenceTransformer is required)."
            )

    # ── Embedding helpers ───────────────────────────────────────────

    def _encode(self, text: str) -> np.ndarray:
        self._ensure_embedder()
        return self._embedder.encode(text)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_embedder()
        return self._embedder.encode_batch(texts)

    def _get_or_compute_embeddings(self, entry: MemoryEntry):
        """Compute and cache embeddings for a memory entry's plan."""
        if entry.plan is None:
            return
        if entry._embedding_pre is None:
            entry._embedding_pre = self._encode(entry.plan.precondition or "")
        if entry._embedding_goal is None:
            entry._embedding_goal = self._encode(entry.plan.goal or "")

    # ── Memory Construction (§3.2.1) ─────────────────────────────────

    def _build_memory_entry(
        self,
        plan: Plan,
        trajectory: list[ObsAct],
        description: str = "",
    ) -> Optional[MemoryEntry]:
        """Construct a memory entry from a sub-task plan and trajectory.

        Per §3.2.1, entries with |τ| ≤ 1 are filtered out to prevent
        memory fragmentation from single atomic actions.
        """
        if len(trajectory) <= 1:
            return None  # Filter atomic actions

        entry = MemoryEntry(
            plan=plan,
            trajectory=list(trajectory),
            meta=MemoryMeta(
                created_at=time.time(),
                last_used_at=time.time(),
                last_logical_time=self._logical_time,
                description=description,
            ),
        )
        # Pre-compute embeddings
        self._get_or_compute_embeddings(entry)

        # Persist trajectory to disk
        self._persist_trajectory(entry)

        return entry

    def add(
        self,
        plan: Plan,
        trajectory: list[ObsAct],
        description: str = "",
    ) -> Optional[MemoryEntry]:
        """Add a new memory to the bank.  Returns None if filtered out.

        Also triggers self-regulation check after addition.
        """
        entry = self._build_memory_entry(plan, trajectory, description)
        if entry is None:
            return None

        self._entries[entry.memory_id] = entry
        self._total_plans_executed += 1

        # Trigger regulation check
        self._maybe_regulate()

        return entry

    def add_from_planner_output(
        self,
        plan: Plan,
        obs_act_pairs: list[tuple],
        description: str = "",
    ) -> Optional[MemoryEntry]:
        """Convenience: add memory from raw (obs, act) tuples."""
        trajectory = [
            ObsAct(observation=obs, action=act, step_index=i)
            for i, (obs, act) in enumerate(obs_act_pairs)
        ]
        return self.add(plan, trajectory, description)

    # ── Disk Persistence ────────────────────────────────────────────

    def _trajectory_path(self, entry: MemoryEntry) -> str:
        return os.path.join(self._disk_dir, f"traj_{entry.memory_id}.pkl")

    def _persist_trajectory(self, entry: MemoryEntry):
        """Write trajectory to disk; keep only metadata + embeddings in memory."""
        path = self._trajectory_path(entry)
        with open(path, "wb") as f:
            pickle.dump(entry.trajectory, f)

    def _load_trajectory(self, entry: MemoryEntry) -> list[ObsAct]:
        """Load trajectory from disk on retrieval hit."""
        path = self._trajectory_path(entry)
        if not os.path.exists(path):
            return []
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Dual-Factor Retrieval (§3.2.2) ──────────────────────────────

    def retrieve(
        self,
        query_plan: Plan,
        current_logical_time: Optional[int] = None,
        T_global: float = 0.5,
    ) -> RetrievalResult:
        """Retrieve the best-matching memory for a query plan.

        Steps:
          1. Encode query plan's precondition and goal.
          2. Compute dual-factor similarity against all stored entries.
          3. Select the highest-scoring entry above retrieval_threshold.
          4. Check risk score — block if T_i > τ (dynamic threshold).
          5. Roll for ϵ-mutation — if triggered, force re-exploration.
          6. On definitive hit, load trajectory from disk and update metadata.

        Parameters
        ----------
        query_plan : Plan
            The Planner-generated sub-task to match against.
        current_logical_time : int or None
            Current logical time step.  Uses internal counter if None.
        T_global : float
            Global failure rate for dynamic thresholding.

        Returns
        -------
        RetrievalResult
        """
        if current_logical_time is not None:
            self._logical_time = current_logical_time

        if not self._entries:
            return RetrievalResult(hit=False)

        self._ensure_embedder()

        # Encode query
        emb_pre_q = self._encode(query_plan.precondition or "")
        emb_goal_q = self._encode(query_plan.goal or "")

        # Score all entries
        best_score = -1.0
        best_entry: Optional[MemoryEntry] = None

        for entry in self._entries.values():
            self._get_or_compute_embeddings(entry)
            if entry._embedding_pre is None or entry._embedding_goal is None:
                continue

            score = dual_factor_similarity(
                emb_pre_q, emb_goal_q,
                np.array(entry._embedding_pre),
                np.array(entry._embedding_goal),
            )
            if score > best_score:
                best_score = score
                best_entry = entry

        # Check retrieval threshold
        if best_entry is None or best_score < self.config.retrieval_threshold:
            return RetrievalResult(hit=False)

        # Risk gating (§3.2.4)
        T_i, _, _ = compute_memory_risk_score(best_entry, self.config)
        tau = compute_dynamic_threshold(T_global, self.config)
        risk_blocked = T_i > tau

        if risk_blocked:
            return RetrievalResult(
                hit=True, entry=best_entry, score=best_score,
                risk_blocked=True,
            )

        # ϵ-Mutation roll (§3.2.2)
        should_mutate = random.random() < self.config.epsilon

        if not should_mutate:
            # Definitive hit — load trajectory, update metadata
            best_entry.trajectory = self._load_trajectory(best_entry)
            best_entry.meta.last_used_at = time.time()
            best_entry.meta.last_logical_time = self._logical_time

        return RetrievalResult(
            hit=True,
            entry=best_entry,
            score=best_score,
            should_mutate=should_mutate,
            risk_blocked=False,
        )

    # ── Evolutionary Replacement (§3.2.2) ────────────────────────────

    def try_evolutionary_replace(
        self,
        old_entry: MemoryEntry,
        new_plan: Plan,
        new_trajectory: list[ObsAct],
    ) -> Optional[MemoryEntry]:
        """Attempt in-place evolutionary replacement.

        Per §3.2.2: if the new (mutated) trajectory is successful AND
        more efficient (|τ'| < |τ|), overwrite the existing entry.

        Returns the updated entry if replacement occurred, None otherwise.
        """
        if len(new_trajectory) <= 1:
            return None  # Don't replace with atomic actions

        if len(new_trajectory) >= len(old_entry.trajectory):
            return None  # New trajectory is not more efficient

        # In-place evolutionary update
        old_entry.plan = new_plan
        old_entry.trajectory = list(new_trajectory)
        old_entry.meta.reuse_count = 0  # Reset — it's effectively a new memory
        old_entry.meta.verification_failures = 0
        old_entry.meta.created_at = time.time()
        old_entry.meta.last_used_at = time.time()
        old_entry.meta.last_logical_time = self._logical_time
        old_entry.meta.success_count = 1

        # Re-compute embeddings for the new plan
        old_entry._embedding_pre = None
        old_entry._embedding_goal = None
        self._get_or_compute_embeddings(old_entry)

        # Persist updated trajectory
        self._persist_trajectory(old_entry)

        return old_entry

    # ── Self-Regulation (§3.2.3) ────────────────────────────────────

    def _maybe_regulate(self):
        """Check if regulation (pruning or expansion) is needed."""
        if self.size < self.config.C_min:
            return

        pruned, new_cap = adaptive_regulate(
            list(self._entries.values()),
            self._logical_time,
            self.config,
        )

        # Apply pruning
        for entry in pruned:
            self._remove_entry(entry)

        # Update capacity
        self.config.C_min = new_cap

    def force_regulate(self) -> tuple[int, int]:
        """Force a regulation cycle.  Returns (pruned_count, new_capacity)."""
        pruned, new_cap = adaptive_regulate(
            list(self._entries.values()),
            self._logical_time,
            self.config,
        )
        for entry in pruned:
            self._remove_entry(entry)
        self.config.C_min = new_cap
        return len(pruned), new_cap

    def _remove_entry(self, entry: MemoryEntry):
        """Remove an entry from the bank and its disk storage."""
        self._entries.pop(entry.memory_id, None)
        path = self._trajectory_path(entry)
        if os.path.exists(path):
            os.remove(path)

    # ── Utility ─────────────────────────────────────────────────────

    def advance_time(self, steps: int = 1):
        """Advance the logical clock."""
        self._logical_time += steps

    def get_survival_ranking(self) -> list[tuple[float, MemoryEntry]]:
        """Return entries sorted by current survival value (descending)."""
        return rank_by_survival(
            list(self._entries.values()),
            self._logical_time,
            self.config,
        )

    def stats(self) -> dict:
        """Diagnostic statistics about the memory bank."""
        if not self._entries:
            return {"size": 0}
        svals = [
            compute_survival_value(e, self._logical_time, self.config)
            for e in self._entries.values()
        ]
        return {
            "size": self.size,
            "total_plans_executed": self._total_plans_executed,
            "logical_time": self._logical_time,
            "survival_mean": float(np.mean(svals)),
            "survival_std": float(np.std(svals)),
            "survival_max": float(np.max(svals)),
            "survival_min": float(np.min(svals)),
            "total_reuses": sum(e.meta.reuse_count for e in self._entries.values()),
            "total_failures": sum(e.meta.verification_failures for e in self._entries.values()),
        }

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, path: str):
        """Serialize the full memory bank to disk.

        Note: trajectory files remain in self._disk_dir; this saves
        the index (metadata + embeddings + plan text).
        """

        def _conv(x):
            """Convert numpy arrays to lists so JSON can serialize them."""
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        state = {
            "logical_time": self._logical_time,
            "total_plans_executed": self._total_plans_executed,
            "entries": {},
        }
        for mid, entry in self._entries.items():
            state["entries"][mid] = {
                "memory_id": entry.memory_id,
                "plan_precondition": entry.plan.precondition if entry.plan else "",
                "plan_goal": entry.plan.goal if entry.plan else "",
                "trajectory_length": entry.trajectory_length,
                "meta": {
                    "reuse_count": entry.meta.reuse_count,
                    "success_count": entry.meta.success_count,
                    "failure_count": entry.meta.failure_count,
                    "verification_failures": entry.meta.verification_failures,
                    "created_at": entry.meta.created_at,
                    "last_used_at": entry.meta.last_used_at,
                    "last_logical_time": entry.meta.last_logical_time,
                    "description": entry.meta.description,
                },
                "embedding_pre": _conv(entry._embedding_pre),
                "embedding_goal": _conv(entry._embedding_goal),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        config: DMSConfig = default_config,
        embedding_backend: Optional[EmbeddingBackend] = None,
    ) -> "MemoryBank":
        """Load a memory bank from disk."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        bank = cls(config=config, embedding_backend=embedding_backend)
        bank._logical_time = state["logical_time"]
        bank._total_plans_executed = state["total_plans_executed"]

        for mid, edata in state["entries"].items():
            plan = Plan(
                precondition=edata["plan_precondition"],
                goal=edata["plan_goal"],
            )
            meta = MemoryMeta(**edata["meta"])
            emb_pre = edata.get("embedding_pre")
            emb_goal = edata.get("embedding_goal")
            entry = MemoryEntry(
                memory_id=edata["memory_id"],
                plan=plan,
                trajectory=[],  # Will be loaded from disk on retrieval
                meta=meta,
                _embedding_pre=np.array(emb_pre) if emb_pre is not None else None,
                _embedding_goal=np.array(emb_goal) if emb_goal is not None else None,
            )
            bank._entries[mid] = entry

        return bank

    def __len__(self) -> int:
        return self.size

    def __contains__(self, memory_id: str) -> bool:
        return memory_id in self._entries

    def __repr__(self) -> str:
        return (f"MemoryBank(entries={self.size}, "
                f"logical_time={self._logical_time}, "
                f"disk_dir={self._disk_dir!r})")
