"""
Darwinian Memory System (DMS) — Hyperparameters
================================================
All parameter values are taken directly from the paper:
  "Darwinian Memory: A Training-Free Self-Regulating Memory System
   for GUI Agent Evolution" (arXiv:2601.22528)

Reference: Appendix B (Experiment Setting & Baseline), Table 2, §3.2.
"""

from dataclasses import dataclass, field


@dataclass
class DMSConfig:
    # ── Survival Value (§3.2.3, Appendix B) ──────────────────────────
    V_new: float = 1.0
    """Novelty bonus for cold-start protection.  Shields nascent memories
    from premature pruning by boosting their initial utility score."""

    T_base: float = 30.0
    """Baseline retention span (logical time steps).  Every memory starts
    with this half-life before usage-based consolidation extends it."""

    mu: float = 15.0
    """Longevity coefficient (called α in Appendix B).  Controls the
    sensitivity of memory consolidation — how much each reuse extends
    the dynamic half-life.  Higher values → longer retention."""

    beta_decay: float = 0.5
    """Steepness of the sigmoid temporal-decay curve.  Higher values
    make the decay transition sharper around the half-life point."""

    gamma_penalty: float = 1.0
    """Penalty severity coefficient.  Each verification failure subtracts
    from the reliability factor as 1/(1 + γ·K_i)."""

    # ── Dynamic Thresholding (§3.2.4, Appendix B) ────────────────────
    tau_base: float = 0.5
    """Base risk-rejection threshold.  Plans with risk score T_i > τ
    are suppressed.  The threshold adapts dynamically based on the
    global error rate."""

    lam: float = 0.3
    """Penalty sensitivity for dynamic thresholding.  Determines how
    aggressively τ tightens when the global failure rate rises."""

    # ── Beta Prior for Bayesian Reputation (§3.2.4) ──────────────────
    alpha_prior: float = 1.0
    """Beta prior pseudo-count for failures.  Together with beta_prior
    determines the initial belief about plan failure rates.
    Uniform prior (1,1) → initial T_global = 0.5."""

    beta_prior: float = 1.0
    """Beta prior pseudo-count for successes.  Uniform prior → no
    initial bias toward success or failure."""

    # ── Verification (Appendix D) ────────────────────────────────────
    K_verify: int = 3
    """Verification depth.  K consecutive failure strikes are required
    before a memory is permanently pruned.  Higher K → stricter
    validation, higher memory purity, but more latency.
    Effective false-negative rate: P_FN^effective ≈ (P_FN)^K."""

    # ── ϵ-Mutation (§3.2.2) ─────────────────────────────────────────
    epsilon: float = 0.1
    """Probability of exploration (mutation) even when a high-confidence
    memory is retrieved.  Prevents the agent from being trapped in
    sub-optimal local optima.  Acts as a catalyst for evolutionary
    velocity (see Appendix C)."""

    # ── Memory Capacity (§3.2.3) ─────────────────────────────────────
    C_min: int = 100
    """Minimum capacity threshold.  Pruning is triggered when the
    memory bank reaches this size."""

    C_max: int = 500
    """Maximum capacity ceiling.  Expansion stops at this limit even
    if the elbow-method safeguard requests more space."""

    delta_step: int = 50
    """Capacity expansion increment.  When the elbow score exceeds
    the population mean, C_min is increased by this amount."""

    # ── Execution Limits (§3.1) ──────────────────────────────────────
    MaxA: int = 15
    """Local step limit for the Actor within a single sub-task."""

    MaxP: int = 50
    """Global step limit for the Planner across all sub-tasks."""

    # ── Retrieval ────────────────────────────────────────────────────
    retrieval_threshold: float = 0.6
    """Minimum dual-factor similarity score for a memory retrieval hit.
    Scores below this threshold are treated as cache misses."""

    # ── Persistence ──────────────────────────────────────────────────
    disk_storage_dir: str = "./dms_memory_store"
    """Directory for persisting high-dimensional trajectories to disk."""

    # ── Embedding dimension (model-dependent) ────────────────────────
    embedding_dim: int = 768
    """Dimensionality of the embedding vectors used for dual-factor
    similarity matching.  768 for sentence-transformers defaults."""


# Singleton-style default config — import this or override per-agent.
default_config = DMSConfig()
