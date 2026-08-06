"""
Darwinian Memory System (DMS)
==============================
A self-evolving, training-free memory system for GUI agents,
governed by biological principles of natural selection.

Reference
---------
"Darwinian Memory: A Training-Free Self-Regulating Memory System
 for GUI Agent Evolution" (arXiv:2601.22528, 2026)

Quick Start
-----------
>>> from darwinian_memory import DMS, DMSConfig, Plan, ObsAct, MemoryBank
>>>
>>> config = DMSConfig()
>>> dms = DMS(config)
>>> dms.initialize_embedding()
>>>
>>> # Register your MLLM callbacks
>>> dms.set_planner(my_planner_fn)
>>> dms.set_actor(my_actor_fn)
>>>
>>> # Run a task
>>> result = dms.run_task("Turn off WiFi")
"""

from .config import DMSConfig, default_config
from .memory_entry import MemoryEntry, Plan, ObsAct, MemoryMeta
from .memory_bank import MemoryBank, RetrievalResult
from .survival import (
    compute_survival_value,
    compute_utility,
    compute_temporal_decay,
    compute_dynamic_half_life,
    compute_reliability,
    adaptive_regulate,
    rank_by_survival,
    SurvivalComponents,
)
from .risk import (
    BayesianStats,
    compute_risk_score,
    compute_dynamic_threshold,
    should_suppress_plan,
    compute_global_failure_rate,
    bayesian_expected_failure,
    bayesian_posterior_std,
)
from .verifier import (
    KVerifier,
    VerifierAgent,
    VerificationResult,
    heuristic_verify,
    VERIFIER_SYSTEM_PROMPT,
)
from .embedding import (
    EmbeddingBackend,
    TFIDFBackend,
    SentenceTransformerBackend,
    dual_factor_similarity,
    cosine_similarity,
)
from .dms import DMS, TaskResult, PlannerFn, ActorFn, VerifierFn

__all__ = [
    # Core
    "DMS",
    "DMSConfig",
    "default_config",
    "MemoryBank",
    # Data structures
    "MemoryEntry",
    "Plan",
    "ObsAct",
    "MemoryMeta",
    "RetrievalResult",
    "TaskResult",
    "SurvivalComponents",
    "BayesianStats",
    "VerificationResult",
    # Survival value
    "compute_survival_value",
    "compute_utility",
    "compute_temporal_decay",
    "compute_dynamic_half_life",
    "compute_reliability",
    "adaptive_regulate",
    "rank_by_survival",
    # Risk assessment
    "compute_risk_score",
    "compute_dynamic_threshold",
    "should_suppress_plan",
    "compute_global_failure_rate",
    "bayesian_expected_failure",
    "bayesian_posterior_std",
    # Verification
    "KVerifier",
    "VerifierAgent",
    "heuristic_verify",
    "VERIFIER_SYSTEM_PROMPT",
    # Embedding
    "EmbeddingBackend",
    "TFIDFBackend",
    "SentenceTransformerBackend",
    "dual_factor_similarity",
    "cosine_similarity",
    # Callback types
    "PlannerFn",
    "ActorFn",
    "VerifierFn",
]
