"""
Darwinian Memory System — Bayesian Risk Assessment (§3.2.4)
============================================================
Implements the negative feedback regulation mechanism that prevents the
agent from repeatedly exploring known failure modes.

Key components:
  1. Bayesian Reputation Modeling (Beta-Binomial conjugate prior)
  2. Uncertainty-Aware Risk Scoring (Lower Confidence Bound)
  3. Dynamic Thresholding (adapts to ecosystem health)

Mathematical details:
  - Prior:        p(θ_i) = Beta(θ_i | α, β)
  - Posterior:    p(θ_i | D_i) = Beta(θ_i | F_i+α, S_i+β)
  - Expected:     μ_i = (F_i + M·T_global) / (F_i + S_i + M)
  - Uncertainty:  σ_i = sqrt(μ_i·(1−μ_i) / (F_i + S_i + M + 1))
  - Risk Score:   T_i = μ_i − σ_i   (LCB approach)
  - Threshold:    τ = τ_base · (1 − λ · T_global)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import DMSConfig, default_config
from .memory_entry import MemoryEntry


# ═══════════════════════════════════════════════════════════════════════
# Global Bayesian statistics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BayesianStats:
    """Aggregated Bayesian reputation statistics for a plan.

    These can be per-plan (tracked by the Planner) or per-memory
    (tracked by memory metadata).  The paper uses per-plan tracking
    for Planner-generated sub-tasks (§3.2.4).
    """
    failures: int = 0      # F_i
    successes: int = 0     # S_i

    @property
    def total(self) -> int:
        return self.failures + self.successes


def compute_global_failure_rate(
    all_stats: list[BayesianStats],
    config: DMSConfig = default_config,
) -> float:
    """Compute the global failure rate T_global across all tracked plans.

    T_global = α_prior / (α_prior + β_prior)  [initial]
    Or, after observations: aggregate failure rate with prior smoothing.
    """
    total_F = sum(s.failures for s in all_stats)
    total_S = sum(s.successes for s in all_stats)
    M = config.alpha_prior + config.beta_prior
    T_global = config.alpha_prior / M  # prior base rate
    # Blend prior with observed data
    return (total_F + M * T_global) / (total_F + total_S + M)


# ═══════════════════════════════════════════════════════════════════════
# Bayesian Reputation Modeling
# ═══════════════════════════════════════════════════════════════════════

def bayesian_expected_failure(
    failures: int,
    successes: int,
    config: DMSConfig = default_config,
) -> float:
    """Bayesian-smoothed expected failure probability μ_i.

    μ_i = (F_i + M · T_global) / (F_i + S_i + M)

    where:
      M = α_prior + β_prior  (prior strength — pseudo-observation count)
      T_global = α_prior / (α_prior + β_prior)  (global base failure rate)

    The Bayesian smoothing prevents overfitting to small samples.
    With a uniform prior (1,1), a plan with 0F/0S gets μ_i = 0.5.
    """
    M = config.alpha_prior + config.beta_prior
    T_global = config.alpha_prior / M
    return (failures + M * T_global) / (failures + successes + M)


def bayesian_posterior_std(
    mu_i: float,
    failures: int,
    successes: int,
    config: DMSConfig = default_config,
) -> float:
    """Posterior standard deviation σ_i of the failure probability.

    σ_i = sqrt(μ_i · (1 − μ_i) / (F_i + S_i + M + 1))
    """
    M = config.alpha_prior + config.beta_prior
    variance = mu_i * (1.0 - mu_i) / (failures + successes + M + 1)
    return math.sqrt(variance)


# ═══════════════════════════════════════════════════════════════════════
# Uncertainty-Aware Risk Scoring
# ═══════════════════════════════════════════════════════════════════════

def compute_risk_score(
    failures: int,
    successes: int,
    config: DMSConfig = default_config,
) -> tuple[float, float, float]:
    """Compute the final risk score T_i using the Lower Confidence Bound.

    T_i = μ_i − σ_i

    The LCB approach is deliberately conservative: when uncertainty (σ)
    is high (few observations), the score is pulled LOWER, making it
    harder to trigger suppression.  This is the "Cold-Start" mitigation
    described in §3.2.4 — plans with little data get the benefit of the
    doubt rather than being prematurely suppressed.

    Returns
    -------
    T_i : float
        Risk score.  Plans with T_i > τ are suppressed.
    mu_i : float
        Expected failure probability.
    sigma_i : float
        Posterior standard deviation.
    """
    mu_i = bayesian_expected_failure(failures, successes, config)
    sigma_i = bayesian_posterior_std(mu_i, failures, successes, config)
    T_i = mu_i - sigma_i
    return T_i, mu_i, sigma_i


# ═══════════════════════════════════════════════════════════════════════
# Dynamic Thresholding
# ═══════════════════════════════════════════════════════════════════════

def compute_dynamic_threshold(
    T_global: float,
    config: DMSConfig = default_config,
) -> float:
    """Compute the dynamic rejection threshold τ.

    τ = τ_base · (1 − λ · T_global)

    When T_global is high (many failures system-wide), τ tightens
    (gets smaller), making it easier to suppress risky plans.
    When T_global is low (system healthy), τ relaxes, allowing more
    exploration.

    Parameters
    ----------
    T_global : float
        Global failure rate (from compute_global_failure_rate).
    config : DMSConfig
    """
    tau = config.tau_base * (1.0 - config.lam * T_global)
    return max(0.01, min(0.99, tau))  # Clamp to valid probability range


def should_suppress_plan(
    failures: int,
    successes: int,
    T_global: float,
    config: DMSConfig = default_config,
) -> tuple[bool, float, float, float]:
    """Determine whether a plan should be suppressed based on risk.

    Returns (suppress, T_i, mu_i, tau).
    """
    T_i, mu_i, sigma_i = compute_risk_score(failures, successes, config)
    tau = compute_dynamic_threshold(T_global, config)
    suppress = T_i > tau
    return suppress, T_i, mu_i, tau


# ═══════════════════════════════════════════════════════════════════════
# Per-Memory Risk (used for memory retrieval gating)
# ═══════════════════════════════════════════════════════════════════════

def compute_memory_risk_score(
    entry: MemoryEntry,
    config: DMSConfig = default_config,
) -> tuple[float, float, float]:
    """Compute risk score for a memory entry (using its metadata).

    This is used during retrieval to gate access to high-risk memories.
    Per Algorithm 1: "if m ≠ None ∧ ρ_m < τ_risk then" — memories
    whose risk score exceeds the threshold are skipped even if they
    match semantically.
    """
    return compute_risk_score(
        entry.meta.failure_count,
        entry.meta.success_count,
        config,
    )
