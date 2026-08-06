"""
Darwinian Memory System — Survival Value Computation (§3.2.3)
==============================================================
The survival value S(m_i) is a composite metric that determines which
memories survive and which are pruned.  It has three multiplicative
components:

    S = Utility × AdaptiveDecay × Reliability

    1. Marginal Utility with Cold-Start Protection:
       U(n_i) = ln(1 + n_i) + V_new

    2. Adaptive Temporal Decay (Sigmoid-based):
       D(Δt, n_i) = 1 / (1 + e^{β·(Δt − T_half(n_i))})

       where T_half(n_i) = T_base + μ·ln(1 + n_i)

    3. Reliability Penalty:
       P(K_i) = 1 / (1 + γ·K_i)

Also implements the Adaptive Memory Regulation algorithm using the
Elbow Method for autonomous pruning decisions.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional
from dataclasses import dataclass

from .config import DMSConfig, default_config
from .memory_entry import MemoryEntry


# ═══════════════════════════════════════════════════════════════════════
# Component 1: Marginal Utility with Cold-Start Protection
# ═══════════════════════════════════════════════════════════════════════

def compute_utility(n_i: int, V_new: float = 1.0) -> float:
    """Compute marginal utility U(n_i) with diminishing returns.

    U(n_i) = ln(1 + n_i) + V_new

    The ln(1+n_i) term gives diminishing marginal returns — the first
    reuse raises utility from V_new to V_new+ln(2), the 10th reuse
    adds much less.  V_new provides cold-start protection so young
    memories survive initial selection pressure.
    """
    return math.log1p(n_i) + V_new   # log1p(x) = ln(1+x), numerically stable


# ═══════════════════════════════════════════════════════════════════════
# Component 2: Adaptive Temporal Decay
# ═══════════════════════════════════════════════════════════════════════

def compute_dynamic_half_life(
    n_i: int,
    T_base: float = 30.0,
    mu: float = 15.0,
) -> float:
    """Compute the dynamic half-life T_half(n_i).

    T_half(n_i) = T_base + μ · ln(1 + n_i)

    Frequently-used memories get logarithmically extended half-lives,
    making them decay much slower than dormant ones.
    """
    return T_base + mu * math.log1p(n_i)


def compute_temporal_decay(
    delta_t: int,
    n_i: int,
    T_base: float = 30.0,
    mu: float = 15.0,
    beta: float = 0.5,
) -> float:
    """Compute the sigmoid-based temporal decay D(Δt, n_i).

    D(Δt, n_i) = 1 / (1 + e^{β·(Δt − T_half(n_i))})

    When Δt ≪ T_half → D ≈ 1.0  (recently used → full weight)
    When Δt ≈ T_half → D = 0.5  (at half-life → half weight)
    When Δt ≫ T_half → D → 0    (long dormant → near-zero weight)

    Parameters
    ----------
    delta_t : int
        Logical time steps since last retrieval (current_time − last_used_logical_time).
    n_i : int
        Reuse count — more reuse → longer T_half → slower decay.
    """
    T_half = compute_dynamic_half_life(n_i, T_base, mu)
    exponent = beta * (delta_t - T_half)
    # Clamp exponent to avoid overflow in exp
    exponent = max(-50.0, min(50.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


# ═══════════════════════════════════════════════════════════════════════
# Component 3: Reliability Penalty
# ═══════════════════════════════════════════════════════════════════════

def compute_reliability(K_i: int, gamma: float = 1.0) -> float:
    """Compute the reliability penalty P(K_i).

    P(K_i) = 1 / (1 + γ·K_i)

    Each verification failure reduces the multiplier.  With γ=1.0:
      K_i=0 → 1.0   (perfect reliability)
      K_i=1 → 0.5   (one strike halves the score)
      K_i=2 → 0.33
      K_i=3 → 0.25  (pruning threshold reached)
    """
    return 1.0 / (1.0 + gamma * K_i)


# ═══════════════════════════════════════════════════════════════════════
# Full Survival Value
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SurvivalComponents:
    """Decomposed survival-value result for inspection / debugging."""
    utility: float
    temporal_decay: float
    reliability: float
    survival_value: float
    half_life: float
    delta_t: int
    reuse_count: int
    verification_failures: int


def compute_survival_value(
    entry: MemoryEntry,
    current_logical_time: int,
    config: DMSConfig = default_config,
) -> float:
    """Compute the full survival value S(m_i) for a memory entry.

    S = [ln(1+n_i) + V_new] · [1/(1+e^{β(Δt−T_half)})] · [1/(1+γ·K_i)]

    Parameters
    ----------
    entry : MemoryEntry
        The memory entry to score.
    current_logical_time : int
        Current logical time step (planning-cycle index) for Δt computation.
    config : DMSConfig
        Hyperparameter configuration.
    """
    n_i = entry.meta.reuse_count
    K_i = entry.meta.verification_failures
    delta_t = current_logical_time - entry.meta.last_logical_time

    U = compute_utility(n_i, config.V_new)
    D = compute_temporal_decay(delta_t, n_i, config.T_base, config.mu, config.beta_decay)
    R = compute_reliability(K_i, config.gamma_penalty)

    return U * D * R


def compute_survival_value_detailed(
    entry: MemoryEntry,
    current_logical_time: int,
    config: DMSConfig = default_config,
) -> SurvivalComponents:
    """Compute survival value with decomposed components for analysis."""
    n_i = entry.meta.reuse_count
    K_i = entry.meta.verification_failures
    delta_t = current_logical_time - entry.meta.last_logical_time

    U = compute_utility(n_i, config.V_new)
    D = compute_temporal_decay(delta_t, n_i, config.T_base, config.mu, config.beta_decay)
    R = compute_reliability(K_i, config.gamma_penalty)
    T_half = compute_dynamic_half_life(n_i, config.T_base, config.mu)

    return SurvivalComponents(
        utility=U,
        temporal_decay=D,
        reliability=R,
        survival_value=U * D * R,
        half_life=T_half,
        delta_t=delta_t,
        reuse_count=n_i,
        verification_failures=K_i,
    )


# ═══════════════════════════════════════════════════════════════════════
# Adaptive Memory Regulation — Elbow Method (§3.2.3)
# ═══════════════════════════════════════════════════════════════════════

def compute_elbow_cutoff(survival_values: np.ndarray) -> int:
    """Find the optimal pruning cutoff index k* using the Elbow Method.

    k* = argmax_k ∇²f(k)

    where f(k) = S(m_{(k)}) sorted in descending order of survival value.
    The discrete second-order gradient (second difference) captures the
    "elbow" — the point where the curve transitions from high-value
    memories to the long tail of low-value entries.

    Parameters
    ----------
    survival_values : np.ndarray
        Survival values sorted in DESCENDING order.  Must have length >= 3.

    Returns
    -------
    k_star : int
        Index of the elbow point (0-based).  Entries at indices >= k*
        are candidates for pruning.
    """
    n = len(survival_values)
    if n < 3:
        return n  # Too few to prune meaningfully

    # Second difference: ∇²f(k) = f(k+1) - 2f(k) + f(k-1)
    # We maximize this to find the sharpest change in slope.
    second_diff = np.zeros(n)
    for k in range(1, n - 1):
        second_diff[k] = (
            survival_values[k + 1]
            - 2.0 * survival_values[k]
            + survival_values[k - 1]
        )

    # Elbow is at the maximum positive second difference.
    # Positive second diff means the curve is convex (bending upward),
    # which in a descending sorted list means the drop is accelerating.
    k_star = int(np.argmax(second_diff))

    # If the maximum second diff is ≤ 0, the curve has no clear elbow;
    # return n (don't prune).
    if second_diff[k_star] <= 0:
        return n

    return k_star


def adaptive_regulate(
    entries: list[MemoryEntry],
    current_logical_time: int,
    config: DMSConfig = default_config,
) -> tuple[list[MemoryEntry], int]:
    """Perform adaptive memory regulation — prune or expand.

    Algorithm (§3.2.3, Adaptive Memory Regulation):
    1. If |M| < C_min, do nothing.
    2. Sort entries by survival value descending → f(k).
    3. Find elbow k* = argmax ∇²f(k).
    4. If f(k*) ≥ μ (population mean) → expand capacity.
       Else → prune entries at indices ≥ k*.

    Parameters
    ----------
    entries : list[MemoryEntry]
        Current memory bank entries.
    current_logical_time : int
        Current logical time step.
    config : DMSConfig
        Hyperparameter configuration.

    Returns
    -------
    pruned : list[MemoryEntry]
        Entries that should be removed from the bank.
    new_capacity : int
        Updated C_min after potential expansion.
    """
    if len(entries) < config.C_min:
        return [], config.C_min

    # Compute survival values and sort descending
    scored = [
        (compute_survival_value(e, current_logical_time, config), e)
        for e in entries
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    svals = np.array([s for s, _ in scored], dtype=np.float64)

    k_star = compute_elbow_cutoff(svals)
    elbow_score = svals[k_star] if k_star < len(svals) else 0.0
    population_mean = float(np.mean(svals))

    # Safeguard: if elbow score exceeds population mean,
    # the "overflow" is valuable experience → expand capacity.
    if k_star < len(svals) and elbow_score >= population_mean:
        new_cap = min(config.C_min + config.delta_step, config.C_max)
        return [], new_cap

    # Otherwise, prune the long tail (entries at and beyond the elbow).
    pruned = [e for _, e in scored[k_star:]] if k_star < len(scored) else []
    return pruned, config.C_min


def rank_by_survival(
    entries: list[MemoryEntry],
    current_logical_time: int,
    config: DMSConfig = default_config,
) -> list[tuple[float, MemoryEntry]]:
    """Return entries sorted by survival value (descending), with scores."""
    scored = [
        (compute_survival_value(e, current_logical_time, config), e)
        for e in entries
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored
