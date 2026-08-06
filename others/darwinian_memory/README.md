# Darwinian Memory System (DMS)

> **Paper**: "Darwinian Memory: A Training-Free Self-Regulating Memory System for GUI Agent Evolution"
> **arXiv**: [2601.22528](https://arxiv.org/abs/2601.22528) (2026-01-30)
> **Authors**: Hongze Mi, Yibo Feng, WenJie Lu, Song Cao et al.

A **training-free**, biologically inspired memory system for GUI agents. DMS treats agent memory as a **dynamic ecosystem** governed by survival of the fittest — memories that are frequently reused, recently active, and reliably successful survive; obsolete, error-prone, or dormant entries are autonomously pruned.

On the AndroidWorld benchmark, DMS boosts general-purpose MLLMs by **+12.4% to +25.4% success rate** and **+33.9% average execution stability** without any fine-tuning or architectural changes.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│   Planner   │────▶│  DMS Memory Bank │────▶│    Actor    │
│  (MLLM)     │     │                  │     │  (MLLM)     │
└─────────────┘     │ • Dual-Factor    │     └─────────────┘
                    │   Retrieval      │           │
                    │ • ϵ-Mutation     │           ▼
                    │ • Evolution      │     ┌─────────────┐
                    │ • Self-Regulation│     │  Verifier   │
                    │ • Risk Assessment│◀────│  (MLLM/H)   │
                    └──────────────────┘     └─────────────┘
```

### Key Mechanisms

| Component | Description | Section |
|-----------|-------------|---------|
| **Memory Construction** | Decomposes workflows into `⟨Precondition, Goal⟩` units; filters out single-step actions | §3.2.1 |
| **Dual-Factor Retrieval** | `Score = sim(φ(pre), φ(pre)) · sim(φ(goal), φ(goal))` — both context AND intent must match | §3.2.2 |
| **ϵ-Mutation** | With probability ε, re-explore even on retrieval hit; if shorter trajectory found, overwrite | §3.2.2 |
| **Survival Value** | `S = Utility × AdaptiveDecay × Reliability` — multi-factor score for pruning decisions | §3.2.3 |
| **Elbow Method Pruning** | Automatic cutoff detection via `k* = argmax ∇²f(k)` | §3.2.3 |
| **Bayesian Risk** | Beta-Binomial reputation model; LCB risk scoring; dynamic thresholding | §3.2.4 |
| **K-Verification** | K=3 consecutive failure strikes before memory deletion; `P_FN^effective ≈ (P_FN)^K` | App. D |

---

## Quick Start

```python
from darwinian_memory import DMS, DMSConfig, Plan, ObsAct

# 1. Create DMS with default hyperparameters from the paper
config = DMSConfig()
dms = DMS(config)

# 2. Initialize embedding backend
dms.initialize_embedding(model_name="all-MiniLM-L6-v2")

# 3. Register your Planner and Actor callbacks
def my_planner(observation, task, context):
    # Your MLLM call: decompose task → sub-plans
    return [Plan(precondition="Settings app is open", goal="Turn off WiFi")]

def my_actor(observation, sub_plan, context):
    # Your MLLM call: execute sub-plan → atomic actions
    return [ObsAct(observation, {"action": "tap", "x": 100, "y": 200})]

dms.set_planner(my_planner)
dms.set_actor(my_actor)

# 4. Optionally register a Verifier (falls back to heuristic if omitted)
dms.set_verifier(my_verifier_fn)

# 5. Run a task
result = dms.run_task("Turn off WiFi")
print(f"Success: {result.success}")
print(f"Memory reuse rate: {result.memory_reuse_rate:.1%}")
print(f"Memory hits: {result.memory_hits}, Mutations: {result.memory_mutations}")
```

### Multi-Round Execution (matching paper §4.3)

```python
tasks = ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]
results = dms.run_multi_round(tasks)

for i, r in enumerate(results):
    print(f"Round {i+1}: SR={r.success}, Reuse={r.memory_reuse_rate:.1%}")

# Memory reuse rate should climb from ~12% (R1) to >30% (R5)
# as the memory ecosystem matures.
```

---

## Module Structure

```
darwinian_memory/
├── __init__.py        # Public API
├── config.py          # Hyperparameters (from Appendix B)
├── memory_entry.py    # Data structures: Plan, MemoryEntry, ObsAct, MemoryMeta
├── embedding.py       # Embedding backends (TF-IDF, SentenceTransformer)
├── memory_bank.py     # Storage, dual-factor retrieval, evolutionary replacement
├── survival.py        # Survival value computation + Elbow Method pruning
├── risk.py            # Bayesian risk assessment + dynamic thresholding
├── verifier.py        # K-Verification policy + Verifier prompt (Figure 9)
└── dms.py             # Main orchestrator (Algorithm 1)
```

---

## Hyperparameters

All values from Appendix B of the paper:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `V_new` | 1.0 | Cold-start novelty bonus |
| `T_base` | 30.0 | Baseline retention span |
| `μ` | 15.0 | Longevity coefficient |
| `β_decay` | 0.5 | Sigmoid decay steepness |
| `γ_penalty` | 1.0 | Reliability penalty severity |
| `τ_base` | 0.5 | Base risk threshold |
| `λ` | 0.3 | Dynamic threshold sensitivity |
| `K_verify` | 3 | Verification depth |
| `ε` | 0.1 | Mutation probability |
| `C_min` | 100 | Pruning trigger capacity |
| `C_max` | 500 | Maximum capacity |
| `Δ_step` | 50 | Capacity expansion increment |

---

## Key Formulas

### Survival Value (§3.2.3)

```
S = [ln(1 + n_i) + V_new] · [1 / (1 + e^{β(Δt - T_half)})] · [1 / (1 + γ·K_i)]

where T_half(n_i) = T_base + μ · ln(1 + n_i)
```

### Risk Score (§3.2.4)

```
μ_i = (F_i + M·T_global) / (F_i + S_i + M)
σ_i = √(μ_i(1-μ_i) / (F_i + S_i + M + 1))
T_i = μ_i - σ_i          (Lower Confidence Bound)
τ   = τ_base · (1 - λ·T_global)
```

### Equilibrium Purity (Appendix C)

```
Q_ss = 1 / (1 + R_fail · R_ver)
where R_fail = p_fail/(1-p_fail), R_ver = P_FN/P_TP
```

---

## Dependencies

- **Required**: `numpy`
- **Recommended**: `sentence-transformers` (for high-quality embeddings)
- **Fallback**: Built-in TF-IDF vectorizer (zero extra dependencies)

```bash
pip install numpy sentence-transformers
```
