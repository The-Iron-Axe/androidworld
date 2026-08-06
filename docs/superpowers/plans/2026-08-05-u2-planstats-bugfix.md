# U2 记忆反馈（plan_stats）Bug 修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复把 DMSAgent 的 per-plan 贝叶斯反馈逻辑照搬到 U2（EpisodicMemory）时引入的两个 bug：`global_failure_rate` 双轨不一致、回放路径下 `_plan_stats` 永远不更新导致风险阈值失真。

**Architecture:** `_plan_stats`（per-plan 贝叶斯反馈）在 `finalize_task` 里更新，但回放路径的 `_flush_u2_trajectory` 守卫直接 return，跳过 `finalize_task`，导致 plan_stats 与 episode 计数不一致；同时 `global_failure_rate` 在 `_plan_stats` 非空时优先用 plan 统计、否则用 episode 统计，两条统计源切换导致动态风险阈值跳变。修复目标是让两条路径都走同一套反馈计数。

**Tech Stack:** Python 3.10, AndroidWorld, others/darwinian_memory (KVerifier/BayesianStats)

---

## 背景（已核实的代码事实）

`set_episode_success()`（memory_agent.py:281-293）是唯一入口：
```python
def set_episode_success(self, success: bool) -> None:
    if not self.enable_u2 or self.u2 is None:
      return
    self.u2.record_episode_outcome(success)      # 1) episode 计数 +1
    goal = self._pending_trajectory_goal
    self._pending_trajectory_goal = None
    self._flush_u2_trajectory(goal, {}, success=success)  # 2) 可能被守卫拦
```

`_flush_u2_trajectory`（memory_agent.py:309-342）：
```python
if goal is None or any(h.get("u2_replayed") for h in self.history):
    return   # 回放路径/无缓冲goal → 直接返回，不进 finalize_task
...
if len(trajectory) > 1:
    self.u2.add_trajectory(goal, trajectory)
    self.u2.finalize_task(goal, success=success)   # plan_stats 在这里更新
```

`finalize_task`（episodic.py:374-433）现在会 `_get_or_create_plan_stats(goal)` 并更新 successes/failures。

## 两个 bug 定义

### Bug 1：`global_failure_rate` 双轨不一致

`global_failure_rate`（episodic.py:479-...）现在：
- `_plan_stats` 非空 → 用 per-plan 统计
- 否则 → 用 episode 计数

但两条统计源的更新路径不同：
- **正常探索**：`add_trajectory` + `finalize_task` → plan_stats 更新 → 用 plan 分支
- **回放路径**：`_flush_u2_trajectory` 守卫 return → plan_stats 不更新 → 用 episode 分支
- **正常但无缓冲 goal**（`flush_memory` 后 goal 为空）：同上，plan_stats 不更新

同一个属性在轮次间会因统计源切换而跳变，且回放越多的轮次越失真。

### Bug 2：回放路径下 plan_stats 永远不更新

回放命中 → `done=True` → `set_episode_success(True)` → `record_episode_outcome(True)`（episode +1）→ `_flush_u2_trajectory` 守卫 return。**结果是这个 episode 的成功/失败只记入 episode 计数，不记入 plan_stats**。而 `global_failure_rate` 在 plan_stats 非空时忽略 episode 计数——回放多的轮次，成功回放对风险阈值**完全没有贡献**，违反论文 §3.2.4 反馈调节语义。

### Bug 3（防御性）：`_get_or_create_plan_stats` 的 `goal.strip()` 可能遇 None

`_get_or_create_plan_stats(goal)` 第一行 `key = plan_key.strip().lower()`。当前调用链保证进 `finalize_task` 时 goal 非 None（守卫在 `_flush_u2_trajectory`），但 `finalize_task` 是 public API，直接调用方（如 episodic.py:16 示例）可能传 None 导致 `AttributeError`。防御性修复。

---

## 修复方向（最终决定：撤销 _plan_stats，保留 KVerifier）

**核心认识（自查后修正）**：`_plan_stats` 在 U2 语境下是**死数据**——只有 `finalize_task` 写入，没有真正的消费者（`global_failure_rate` 在修复后不读它，U2 也没有 Planner 分解出有意义的 per-plan 语义）。引入它反而造成了双轨 bug。而 `KVerifier`（K=3 剪枝）是有价值的标准实现，保留。

**最终决定**：
1. **撤销 `_plan_stats` 引入**：移除 `_plan_stats` 字段、`_get_or_create_plan_stats` 方法、`finalize_task` 里的 plan_stats 更新、`global_failure_rate` 的 plan 优先分支。回到干净的 episode-only T_global。
2. **保留 KVerifier**：`_kverifier.record_failure` 替换手写 3-strike 是标准且正确的，保留。
3. `global_failure_rate` 只用 episode 计数——探索和回放都通过 `record_episode_outcome` 统一贡献，无双轨。

这比"保留 plan_stats 但让 global_failure_rate 不读它"更干净：不留死代码，消除双轨，改动更少。

---

### Task 1: 撤销 _plan_stats 引入（消除双轨 + 死代码）

**Files:**
- Modify: `android_world/agents/memory/episodic.py`
- Test: `android_world/agents/memory/test_episodic.py`

- [ ] **Step 1: 写失败测试**

在 `test_episodic.py` 的 `GlobalFailureRateTest` 类里追加（验证 plan_stats 不再影响 T_global）：

```python
def test_global_failure_rate_ignores_plan_stats(self):
    """T_global must come from episode outcomes only, so replay-heavy
    rounds don't skew the dynamic risk threshold via stale plan stats."""
    with tempfile.TemporaryDirectory() as d:
        config = DMSConfig()
        config.disk_storage_dir = d
        mem = EpisodicMemory(config=config, persistence_dir=d)
        # Even if a stale plan-stat entry exists, it must not affect T_global.
        mem._plan_stats["stale goal"] = mem._get_or_create_plan_stats("stale goal")
        mem._plan_stats["stale goal"].failures = 100
        # Without episode outcomes, rate stays at uniform prior 0.5.
        self.assertAlmostEqual(mem.global_failure_rate, 0.5)
        # With episode outcomes, rate reflects episodes only.
        mem.record_episode_outcome(True)
        mem.record_episode_outcome(True)
        mem.record_episode_outcome(False)
        self.assertAlmostEqual(mem.global_failure_rate, 0.4)
```

- [ ] **Step 2: 运行确认失败**

Run: `python -m pytest android_world/agents/memory/test_episodic.py::GlobalFailureRateTest::test_global_failure_rate_ignores_plan_stats -v`
Expected: FAIL（当前 `global_failure_rate` 在 `_plan_stats` 非空时优先用 plan 统计，`failures=100` 会让 rate 偏离 0.5）

- [ ] **Step 3: 移除 _plan_stats 与 _get_or_create_plan_stats**

在 `episodic.py` 中：

a) 删除 `__init__` 里的 `_plan_stats` 字段和 `_kverifier`（保留）：

```python
# 删除：
#     # Per-plan Bayesian feedback (§3.2.4) — mirrors DMSAgent._plan_stats.
#     self._plan_stats: dict[str, BayesianStats] = {}
# 保留：
#     self._kverifier = KVerifier(self.config)
```

b) 删除 `finalize_task` 里的 plan_stats 更新块（两处）：

```python
# 第一处（_active_entry 分支，原 396-401 行）删除：
#       # Per-plan Bayesian feedback (§3.2.4) — mirrors DMSAgent._finalize_sub_plan.
#       stats = self._get_or_create_plan_stats(goal)
#       if success:
#         stats.successes += 1
#       else:
#         stats.failures += 1
#
# 第二处（_last_added_entry 分支，原 413-417 行）删除：
#       stats = self._get_or_create_plan_stats(goal)
#       if success:
#         stats.successes += 1
#       else:
#         stats.failures += 1
```

c) 删除 `_get_or_create_plan_stats` 方法（原 467-476 行）：

```python
# 删除整个方法：
#   def _get_or_create_plan_stats(self, plan_key: str) -> BayesianStats:
#       ...
```

d) `global_failure_rate` 恢复为只用 episode 计数：

```python
@property
def global_failure_rate(self) -> float:
    """Smoothed global failure rate T_global (Bayesian prior + observed).

    T_global is a *global* failure rate (§3.2.4): it comes from episode
    outcomes only.  Both exploration and replay episodes feed the same
    counter via record_episode_outcome, so the dynamic risk threshold
    stays stable across rounds.  Per-goal feedback is intentionally NOT
    tracked here (U2 has no Planner to name plans).
    """
    total = self._episode_failures + self._episode_successes
    if total == 0:
      return 0.5  # Uniform prior
    prior_fail = self.config.alpha_prior
    prior_total = self.config.alpha_prior + self.config.beta_prior
    prior_rate = prior_fail / prior_total
    return (self._episode_failures + prior_total * prior_rate) / (
        self._episode_failures + self._episode_successes + prior_total
    )
```

e) 更新 `record_episode_outcome` 的 docstring（移除对 per-plan 的引用）：

```python
def record_episode_outcome(self, success: bool) -> None:
    """Track episode-level success/failure for the global failure rate.

    T_global feeds the dynamic risk threshold (§3.2.4).  Both exploration
    and replay episodes are counted here — replay is not re-stored, but
    its outcome still reflects system health.
    """
    if success:
      self._episode_successes += 1
    else:
      self._episode_failures += 1
```

f) 从 import 行移除 `BayesianStats`（若不再使用）：

```python
# episodic.py 顶部：
from android_world.agents.memory.dms_bridge import (
    DMSConfig, KVerifier, MemoryBank, MemoryEntry, ObsAct, Plan,
    RetrievalResult,
)
```

- [ ] **Step 4: 运行确认通过**

Run: `python -m pytest android_world/agents/memory/test_episodic.py -v`
Expected: 全部 PASS（含新增 test_global_failure_rate_ignores_plan_stats）

- [ ] **Step 5: 确认 dms_bridge 的 BayesianStats 导出可保留或移除**

`dms_bridge.py` 仍导出 `BayesianStats`。若不再被任何文件使用则移除；若保留也无碍（导出未用不报错）。检查：

```bash
grep -rn "BayesianStats" android_world/ --include="*.py"
```
若只剩 dms_bridge 自身的导出 → 从 dms_bridge import 和 __all__ 移除；若 episodic 或其他文件还用 → 保留。

- [ ] **Step 6: Commit**

```bash
git add android_world/agents/memory/episodic.py android_world/agents/memory/dms_bridge.py android_world/agents/memory/test_episodic.py
git commit -m "fix(u2): revert dead per-plan stats; global_failure_rate uses episode outcomes only"
```

---

### Task 2: 确认回放路径语义（不重存 + episode 计数）有测试锁定

**Files:**
- Verify: `android_world/agents/memory/test_replay_agent.py`

- [ ] **Step 1: 确认现有测试已锁定回放不重存**

Run: `python -m pytest android_world/agents/memory/test_replay_agent.py::TestReplay::test_replay_full_episode_does_not_restore -v`
Expected: PASS（回放后 bank.size 不增长——reuse not re-demonstrate）

- [ ] **Step 2: 确认回放 episode 仍贡献 T_global（通过 episode 计数）**

补一个测试到 `test_episodic.py`（若没有的话），验证 `record_episode_outcome` 在回放场景仍被调用：

```python
def test_replay_episode_still_feeds_global_failure_rate(self):
    """A replay episode calls record_episode_outcome, so T_global reflects
    overall system health regardless of replay vs exploration."""
    with tempfile.TemporaryDirectory() as d:
        config = DMSConfig()
        config.disk_storage_dir = d
        mem = EpisodicMemory(config=config, persistence_dir=d)
        mem.record_episode_outcome(True)   # a successful replay episode
        # 1 success, 0 failures, blended with uniform prior (1,1):
        # (0 + 2*0.5) / (1 + 2) = 1/3
        self.assertAlmostEqual(mem.global_failure_rate, 1 / 3)
```

- [ ] **Step 3: 运行确认通过**

Run: `python -m pytest android_world/agents/memory/test_episodic.py -v`
Expected: 全部 PASS

- [ ] **Step 4: Commit**

```bash
git add android_world/agents/memory/test_episodic.py
git commit -m "test(u2): lock replay-episode feeds global failure rate"
```

---

### Task 3: 全量回归 + 编译检查

**Files:**
- Verify: `android_world/agents/memory/`

- [ ] **Step 1: 全量回归**

Run: `python -m pytest android_world/agents/memory/ -q`
Expected: 全部 PASS

- [ ] **Step 2: 编译检查**

Run: `python -m py_compile android_world/agents/memory/episodic.py android_world/agents/memory/dms_bridge.py android_world/agents/memory_agent.py`
Expected: 无输出（成功）

- [ ] **Step 3: 确认 DMSAgent 未受影响**

Run: `python -m py_compile android_world/agents/dms_agent.py`
Expected: 无输出

- [ ] **Step 4: 确认最终状态无 plan_stats 死代码**

Run: `grep -rn "plan_stats\|_get_or_create_plan_stats" android_world/agents/memory/`
Expected: 无匹配（episodic.py 已清理）

- [ ] **Step 5: Commit（若 Task 2 后有未提交改动）**

```bash
git status --short android_world/agents/memory/
```
若有改动：`git add ... && git commit -m "chore(u2): final regression for plan_stats revert"`
若干净：跳过

---

## Self-Review

**1. Spec coverage:**
- Bug 1（双轨 + 死代码）：Task 1 撤销 plan_stats，global_failure_rate 统一 episode-only ✅
- Bug 3（_get_or_create_plan_stats None）：随 plan_stats 一起移除，不再有该方法 ✅
- Bug 2（回放不计入 plan_stats）：改为"回放计入 episode 计数（T_global 反映系统健康）"，Task 2 测试锁定 ✅

**2. Placeholder scan:** 无 TBD/TODO；每步含代码、命令、期望输出。

**3. Type consistency:**
- `global_failure_rate` 返回 float，测试 `assertAlmostEqual` ✅
- `_get_or_create_plan_stats` 移除后，finalize_task 里不再引用 ✅
- `KVerifier` 保留、`BayesianStats` 视使用情况移除——Task 1 Step 5 有 grep 检查 ✅
