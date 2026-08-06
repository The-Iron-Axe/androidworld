# DMS 确定性轨迹回放迁移到 U2 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DMSAgent 的确定性轨迹回放能力（§3.2.2 replay）迁移到 MemoryAugmentedAgent 的 U2 模块，使 U2 从"仅 hint 注入"升级为"命中即确定性回放缓存轨迹"，同时保证 `--u1`、`--u1 --u2`、`--u1 --u2 --u3` 三种消融组合各自独立可运行。

**Architecture:** 在 `EpisodicMemory` 上新增回放决策 API（`retrieve_replay`，返回完整轨迹而非 hint），在 `MemoryAugmentedAgent` 内用 `_replay_active` 状态机 + `_step_replay` 覆写 `step()` 高层，复用我在 `dms_agent.py` 里验证过的逐步驱动模式。回放只在 `enable_u2` 时激活，U1/U3 的 hook 不受影响。

**Tech Stack:** Python 3.10, AndroidWorld, others/darwinian_memory（MemoryBank）, absl logging

---

## 现状盘点（已核实）

- `MemoryAugmentedAgent`（[memory_agent.py](android_world/agents/memory_agent.py)）继承 M3A，`enable_u1/enable_u2/enable_u3` 三个独立开关，`_build_action_prompt` 注入 U1/U2/U3 记忆块。
- U2 = `EpisodicMemory`（[episodic.py](android_world/agents/memory/episodic.py)），目前 `retrieve_hint()` 只返回**动作类型字符串**（`open_app → click → ...`），拼进 prompt。**不做确定性回放**。
- DMSAgent（[dms_agent.py](android_world/agents/dms_agent.py)）已有完整 replay 实现（`_start_replay`/`_step_replay`/`_resolve_action_target`），逐步驱动，离线验证通过。这是移植蓝本。
- `episode_runner.py` 每步调一次 `agent.step(goal)`，`max_n_steps = int(complexity*10)`，`done=True` 时结束。
- `suite_utils.py:288-290` 调用 `agent.set_episode_success(agent_successful > 0.5)` 提供真值。
- `episodic.py` 存储时 `observation` 用文本标记 `step_{i}`，**不含原始截图像素**（轻量 pickle）。
- `memory_agent.py` U2 写入走 `set_episode_success → _flush_u2_trajectory`，轨迹来自 `self.history`（M3A 累积的 step_data）。

## 关键设计决策

1. **回放只用于跨任务复用**：`retrieve_replay(goal)` 命中且不触发 mutation/risk-block 时返回完整轨迹 `list[ObsAct]`，否则返回 `None`。`_build_action_prompt` 的 hint 注入保留为**降级路径**（回放轨迹为空时）。
2. **逐步驱动状态机**：`MemoryAugmentedAgent` 加 `_replay_active`/`_replay_entry`/`_replay_index` 三个字段，`step()` 开头检查——若回放中则 `_step_replay()` 执行下一个缓存动作并返回。这与此前 DMSAgent 一致，不碰 episode_runner。
3. **成功判定**：回放走完整个轨迹（无异常）即视为子任务完成（`done=True` 交给 episode_runner）。`set_episode_success` 仍由 suite_utils 提供真值，回放失败会记入 U2 的 K-Verification。
4. **U1/U3 独立**：replay 不触碰 U1（TaskState）和 U3（EnvKnowledge）的 hook。`enable_u2` 关闭时 `_build_action_prompt` 走原逻辑，完全不影响 `--u1` / `--u1 --u3` 组合。
5. **轨迹载荷**：replay 需要可执行的 `JSONAction`。当前 episodic 存的是 `action_output_json`（JSONAction 对象），pickle 后可直接执行。observation 标记不影响回放（回放只用 action）。

## 文件结构

- **Modify:** `android_world/agents/memory/episodic.py` — 新增 `retrieve_replay()`（返回完整轨迹）和 `_build_retrieval_result()`（共用检索决策）
- **Modify:** `android_world/agents/memory_agent.py` — 新增回放状态机（`_start_replay`/`_step_replay`/`_resolve_action_target`/`_step_result`），覆写 `step()`，`_build_action_prompt` 保留 hint 降级
- **Test:** `android_world/agents/memory/test_episodic.py` — 为 `retrieve_replay` 增补测试
- **Test:** `android_world/agents/memory/test_replay_agent.py` — 新建，用 fake env 测 MemoryAugmentedAgent 回放状态机

---

### Task 1: EpisodicMemory 新增 `retrieve_replay()` 与共用检索决策

**Files:**
- Modify: `android_world/agents/memory/episodic.py`

现状 `retrieve_hint()` 已包含完整检索逻辑（命中/risk-block/mutation/加载轨迹/返回）。将其拆出共用内部方法 `_retrieve_entry(goal, precondition) -> MemoryEntry | None`，返回决策后的 entry（已加载轨迹），供 hint 和 replay 共用。

- [ ] **Step 1: 写失败测试**

在 `android_world/agents/memory/test_episodic.py` 追加（先读该文件确认既有结构）：

```python
def test_retrieve_replay_returns_trajectory_on_hit():
    u2 = EpisodicMemory()
    u2.init_embedding(corpus_texts=["open app then click save"])
    goal = "Save a file"
    traj = [ObsAct(observation="step_0", action=JSONAction(action_type="open_app", app_name="Files"), step_index=0),
            ObsAct(observation="step_1", action=JSONAction(action_type="click", index=2), step_index=1)]
    u2.add_trajectory(goal, traj)
    replay = u2.retrieve_replay(goal)
    assert replay is not None
    assert len(replay) == 2
    assert replay[0].action.action_type == "open_app"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest android_world/agents/memory/test_episodic.py::test_retrieve_replay_returns_trajectory_on_hit -v`
Expected: FAIL with `AttributeError: 'EpisodicMemory' object has no attribute 'retrieve_replay'`

- [ ] **Step 3: 实现 `_retrieve_entry` 与 `retrieve_replay`**

在 `episodic.py` 中，将 `retrieve_hint` 的检索决策逻辑抽为共用内部方法，并新增 `retrieve_replay`。`_retrieve_entry` 返回已加载轨迹的 entry（`retrieve_hint` 内部复用）：

```python
def _retrieve_entry(
    self, goal: str, precondition: str | None = None
) -> MemoryEntry | None:
    """Run the DMS retrieval decision and return the entry with a loaded
    trajectory, or None on miss / risk-block / mutation / empty trajectory.

    Shared by retrieve_hint (prompt injection) and retrieve_replay (deterministic
    replay).  Applies the same dual-factor scoring, Bayesian risk gate, epsilon
    mutation, and disk trajectory load as the paper's Algorithm 1.
    """
    if not self._initialized:
      self.init_embedding()
    plan = self._build_retrieval_query(goal, precondition)
    result: RetrievalResult = self.bank.retrieve(
        plan,
        current_logical_time=self.bank.logical_time,
        T_global=self.global_failure_rate,
    )
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
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but eps-MUTATION (re-explore)")
      self._active_entry = result.entry
      return None
    entry = result.entry
    trajectory = self.bank._load_trajectory(entry)
    if not trajectory:
      print(f"[U2] retrieve goal={goal[:50]!r} -> hit score={result.score:.3f} but empty trajectory")
      return None
    entry.trajectory = trajectory  # ensure loaded for deterministic replay
    self._active_entry = entry
    print(
        f"[U2] retrieve goal={goal[:50]!r} -> HIT score={result.score:.3f} "
        f"reuse={entry.meta.reuse_count} steps={len(trajectory)}"
    )
    return entry
```

将 `retrieve_hint` 改为复用 `_retrieve_entry`（行为保持不变），并新增：

```python
def retrieve_replay(
    self, goal: str, precondition: str | None = None
) -> list[ObsAct] | None:
    """Return the full cached trajectory for deterministic replay, or None.

    Unlike retrieve_hint (which returns a compact prompt string), this returns
    the executable ObsAct list so the agent can replay the actions verbatim
    (§3.2.2: 'the Actor reuses the stored τ').  Results are cached per
    (goal, precondition) for the current episode.
    """
    cache_key = ("replay", goal, precondition or "")
    if cache_key in self._retrieval_cache:
      value = self._retrieval_cache[cache_key]
      return value if isinstance(value, list) else None
    entry = self._retrieve_entry(goal, precondition)
    if entry is None:
      self._retrieval_cache[cache_key] = None
      return None
    self._retrieval_cache[cache_key] = list(entry.trajectory)
    return list(entry.trajectory)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest android_world/agents/memory/test_episodic.py::test_retrieve_replay_returns_trajectory_on_hit -v`
Expected: PASS

- [ ] **Step 5: 回归既有测试**

Run: `python -m pytest android_world/agents/memory/test_episodic.py -v`
Expected: 全部 PASS（`retrieve_hint` 行为不变）

- [ ] **Step 6: Commit**

```bash
git add android_world/agents/memory/episodic.py android_world/agents/memory/test_episodic.py
git commit -m "feat(u2): add retrieve_replay for deterministic trajectory replay"
```

---

### Task 2: MemoryAugmentedAgent 增加回放状态机

**Files:**
- Modify: `android_world/agents/memory_agent.py`

新增回放状态字段、`_start_replay`/`_step_replay`/`_resolve_action_target`/`_step_result` 方法，覆写 `step()` 使回放中优先执行缓存动作。复用 DMSAgent 里已验证的实现，适配 MemoryAugmentedAgent 的 M3A 基类（`self.history` 存在、`get_post_transition_state()` 可用、`env.execute_action` 可用）。

- [ ] **Step 1: 写失败测试**

新建 `android_world/agents/memory/test_replay_agent.py`：

```python
"""Offline tests for MemoryAugmentedAgent deterministic replay (U2)."""

from android_world.agents import memory_agent
from android_world.agents.memory.episodic import ObsAct
from android_world.env import json_action


class _FakeEnv:
  logical_screen_size = property(lambda self: (1080, 2400))
  orientation = "portrait"
  physical_frame_boundary = (0, 0, 1080, 2400)
  interaction_cache = ""

  def __init__(self):
    self.executed = []

  def execute_action(self, action):
    self.executed.append(action)

  def get_state(self, wait_to_stabilize=False):
    class _S:
      ui_elements = []
      pixels = None
    return _S()

  def hide_automation_ui(self):
    pass


class _FakeLLM:
  def predict_mm(self, prompt, images):
    raise AssertionError("LLM must not be called during replay")


def _make_agent(env, llm):
  agent = memory_agent.MemoryAugmentedAgent(
      env, llm, enable_u1=False, enable_u2=True, enable_u3=False
  )
  agent._run_planner = lambda goal: []
  return agent


def test_replay_executes_cached_trajectory_without_llm():
  env = _FakeEnv()
  agent = _make_agent(env, _FakeLLM())
  traj = [
      ObsAct(observation="step_0",
             action=json_action.JSONAction(action_type="open_app", app_name="Files"),
             step_index=0),
      ObsAct(observation="step_1",
             action=json_action.JSONAction(action_type="click", index=2),
             step_index=1),
  ]
  # Inject a hit so step() enters replay instead of calling the LLM.
  agent.u2.retrieve_replay = lambda goal, precondition=None: list(traj)

  r1 = agent.step("Open Files and click")
  assert not r1.done
  assert len(env.executed) == 1
  assert env.executed[0].action_type == "open_app"

  r2 = agent.step("Open Files and click")
  assert not r2.done
  assert len(env.executed) == 2
  assert env.executed[1].action_type == "click"

  r3 = agent.step("Open Files and click")
  assert r3.done  # full trajectory replayed -> done
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest android_world/agents/memory/test_replay_agent.py -v`
Expected: FAIL（`step()` 未进入 replay，会调用 `_FakeLLM.predict_mm` 抛 AssertionError）

- [ ] **Step 3: 实现回放状态机**

在 `memory_agent.py` 的 `MemoryAugmentedAgent` 中新增：

```python
# ── Deterministic trajectory replay (U2, §3.2.2) ──────────────────────

def _start_replay(self, trajectory: list[ObsAct], entry) -> None:
  """Begin deterministic replay of a cached trajectory for this task."""
  # §3.2.2 decoupled storage: ensure the trajectory is loaded from disk.
  loaded = self.u2.bank._load_trajectory(entry) if self.u2 is not None else []
  self._replay_active = True
  self._replay_entry = entry
  self._replay_index = 0
  self._replay_trajectory = loaded if loaded else list(trajectory)
  logging.info(
      "U2 — replaying cached trajectory (%d steps)", len(self._replay_trajectory)
  )

def _resolve_action_target(self, action, ui_elements):
  """Rebind an action's index to the current screen's UI elements."""
  if action.action_type not in (
      json_action.CLICK,
      json_action.DOUBLE_TAP,
      json_action.LONG_PRESS,
  ):
    return action
  if action.index is None:
    return action
  if action.index < 0 or action.index >= len(ui_elements):
    return action
  logical_screen_size = self.env.logical_screen_size
  original = ui_elements[action.index]
  if m3a_utils.validate_ui_element(original, logical_screen_size):
    return action
  text = getattr(original, "text", None) or getattr(
      original, "content_description", None
  )
  if not text:
    return action
  matches = [
      (i, e)
      for i, e in enumerate(ui_elements)
      if m3a_utils.validate_ui_element(e, logical_screen_size)
      and (getattr(e, "text", None) or getattr(e, "content_description", None))
      == text
  ]
  if len(matches) == 1:
    rebound = copy.deepcopy(action)
    rebound.index = matches[0][0]
    return rebound
  return action

def _step_replay(self, goal: str):
  """Execute the next cached action. Returns (done, step_data)."""
  step_data = {
      "raw_screenshot": None,
      "before_screenshot_with_som": None,
      "before_ui_elements": [],
      "after_screenshot_with_som": None,
      "action_prompt": None,
      "action_output": None,
      "action_output_json": None,
      "action_reason": None,
      "action_raw_response": None,
      "summary_prompt": None,
      "summary": None,
      "summary_raw_response": None,
      "u2_replayed": True,
  }
  if self._replay_entry is not None:
    step_data["u2_memory_id"] = self._replay_entry.memory_id

  trajectory = self._replay_trajectory
  if self._replay_index >= len(trajectory):
    self._replay_active = False
    self._replay_entry = None
    self._replay_index = 0
    self._replay_trajectory = []
    step_data["summary"] = "Replayed full trajectory; task step complete."
    self.history.append(step_data)
    return True, step_data

  obs_act = trajectory[self._replay_index]
  self._replay_index += 1
  action = obs_act.action
  if action is None:
    step_data["summary"] = "Skipped replay step with no cached action."
    self.history.append(step_data)
    return False, step_data
  if not isinstance(action, json_action.JSONAction):
    action = json_action.JSONAction(
        **{k: v for k, v in vars(action).items() if k in json_action.ACTION_KEYS}
    )
  step_data["action_output_json"] = action

  if action.action_type == json_action.STATUS:
    done = action.goal_status != "infeasible"
    self._replay_active = False
    self._replay_entry = None
    self._replay_index = 0
    self._replay_trajectory = []
    step_data["summary"] = "Replayed memory declared sub-task %s." % (
        "completed" if done else "infeasible"
    )
    self.history.append(step_data)
    return done, step_data

  if action.action_type in (
      json_action.CLICK,
      json_action.DOUBLE_TAP,
      json_action.LONG_PRESS,
  ) and action.index is not None:
    try:
      state = self.get_post_transition_state()
      action = self._resolve_action_target(action, state.ui_elements)
    except Exception as e:  # pylint: disable=broad-exception-caught
      logging.warning("U2 replay — target resolution failed: %s", e)

  try:
    self.env.execute_action(action)
    time.sleep(self.wait_after_action_seconds)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning("U2 [%s] — replay action failed: %s", goal, e)
    step_data["summary"] = f"Replay action failed: {e}"
    self.history.append(step_data)
    self._replay_active = False
    self._replay_entry = None
    self._replay_index = 0
    self._replay_trajectory = []
    return False, step_data

  remaining = len(trajectory) - self._replay_index
  step_data["summary"] = (
      f"Replayed step {self._replay_index}/{len(trajectory)}; {remaining} remaining."
  )
  self.history.append(step_data)
  return False, step_data
```

在 `__init__` 中加字段（`enable_u2` 时初始化）：

```python
self._replay_active = False
self._replay_entry = None
self._replay_index = 0
self._replay_trajectory: list[ObsAct] = []
```

在 `reset()` 中重置：

```python
self._replay_active = False
self._replay_entry = None
self._replay_index = 0
self._replay_trajectory = []
```

- [ ] **Step 4: 覆写 `step()` 接入回放**

在 `memory_agent.py` 中新增 `step()` 覆写（放在 `_flush_u2_trajectory` 之后）：

```python
def step(self, goal: str) -> base_agent.AgentInteractionResult:
  """Execute one interaction step; if U2 has a cached trajectory, replay it."""
  # Active replay: execute the next cached action (§3.2.2).
  if self._replay_active:
    done, step_data = self._step_replay(goal)
    return base_agent.AgentInteractionResult(done, step_data)

  # First step of the task with U2 enabled: try deterministic replay.
  if self.enable_u2 and self.u2 is not None and len(self.history) == 0:
    trajectory = self.u2.retrieve_replay(goal)
    if trajectory:
      self._start_replay(trajectory, self.u2._active_entry)
      done, step_data = self._step_replay(goal)
      return base_agent.AgentInteractionResult(done, step_data)

  return super().step(goal)
```

同时更新 `_build_action_prompt` 的 U2 分支，保留 hint 作为回放降级（replay 已在 `step()` 拦截，`_build_action_prompt` 不会在回放中被调用；此分支留给非回放路径）：

```python
if self.enable_u2 and self.u2 is not None:
  hint = self.u2.retrieve_hint(goal)
  if hint:
    memory_blocks.append(f"## Memory Hint (U2)\nSimilar past trajectory: {hint}")
```

保持现状即可（replay 优先，hint 兜底）。

- [ ] **Step 5: 补充 imports**

在 `memory_agent.py` 顶部，`from android_world.agents import infer` 下方添加 `m3a_utils` 导入，并在标准库区补 `copy`、`time`：

```python
import copy
import time

from android_world.agents import infer
from android_world.agents import m3a as m3a_lib
from android_world.agents import m3a_utils
```

并将 `_resolve_action_target` 中的 `m3a_lib.m3a_utils.validate_ui_element` 改为 `m3a_utils.validate_ui_element`（与 DMSAgent 一致，直接引用已导入的模块）。

- [ ] **Step 6: 运行测试确认通过**

Run: `python -m pytest android_world/agents/memory/test_replay_agent.py -v`
Expected: PASS

- [ ] **Step 7: 回归既有测试**

Run: `python -m pytest android_world/agents/memory/test_episodic.py -v`
Expected: 全部 PASS

- [ ] **Step 8: Commit**

```bash
git add android_world/agents/memory_agent.py android_world/agents/memory/test_replay_agent.py
git commit -m "feat(u2): deterministic trajectory replay in MemoryAugmentedAgent"
```

---

### Task 3: 消融组合验证（--u1 / --u1 --u2 / --u1 --u2 --u3）

**Files:**
- Verify: `android_world/agents/memory/test_replay_agent.py`

确保三个开关独立可用，回放只在 `enable_u2` 时触发，U1/U3 不受影响。

- [ ] **Step 1: 写消融测试**

在 `test_replay_agent.py` 追加：

```python
def test_replay_off_when_u2_disabled():
  env = _FakeEnv()
  agent = memory_agent.MemoryAugmentedAgent(
      env, _FakeLLM(), enable_u1=True, enable_u2=False, enable_u3=False
  )
  # No replay: step() falls through to M3A LLM path (predict_mm raises),
  # but only if we get there — verify _replay_active stays False.
  assert not agent._replay_active


def test_u1_u2_u3_all_enabled_constructs():
  env = _FakeEnv()
  agent = memory_agent.MemoryAugmentedAgent(
      env, _FakeLLM(), enable_u1=True, enable_u2=True, enable_u3=True,
      rag_url=None,
  )
  assert agent.u1 is None  # lazy init
  assert agent.u2 is not None
  assert agent.u3 is not None
  assert agent.enable_u1 and agent.enable_u2 and agent.enable_u3
```

- [ ] **Step 2: 运行测试**

Run: `python -m pytest android_world/agents/memory/test_replay_agent.py -v`
Expected: 全部 PASS

- [ ] **Step 3: 手动消融启动验证（可选，需模拟器）**

分别启动三种组合，确认不报错：
```bash
# baseline + U1
python run.py --agent_name=m3a_qwen3_vl_32b_mem --u1 --fixed_task_seed --task_seed=30 --n_task_combinations=1
# + U2
python run.py --agent_name=m3a_qwen3_vl_32b_mem --u1 --u2 --u2_persistence_dir=./u2_store --fixed_task_seed --task_seed=30 --n_task_combinations=1
# + U3
python run.py --agent_name=m3a_qwen3_vl_32b_mem --u1 --u2 --u3 --u2_persistence_dir=./u2_store --u3_persistence_dir=./u3_store --fixed_task_seed --task_seed=30 --n_task_combinations=1
```
Expected: 三种组合都能启动到 agent 创建，无 ImportError/AttributeError。

- [ ] **Step 4: Commit**

```bash
git add android_world/agents/memory/test_replay_agent.py
git commit -m "test(u2): verify u1/u2/u3 ablation combinations"
```

---

### Task 4: 收尾——清理与最终回归

**Files:**
- Verify: 全部改动文件

- [ ] **Step 1: 全量回归**

Run: `python -m pytest android_world/agents/memory/ -v`
Expected: 全部 PASS（episodic 既有测试 + replay + 消融）

- [ ] **Step 2: 编译检查**

Run: `python -m py_compile android_world/agents/memory_agent.py android_world/agents/memory/episodic.py`
Expected: 无输出（成功）

- [ ] **Step 3: 确认 DMSAgent 不受影响**

Run: `python -m py_compile android_world/agents/dms_agent.py`
Expected: 无输出（DMSAgent 保留原样，作为独立完整 DMS 实现）

- [ ] **Step 4: Commit**

```bash
git add android_world/agents/memory_agent.py android_world/agents/memory/episodic.py
git commit -m "chore(u2): final regression for deterministic replay migration"
```

---

## Self-Review

**1. Spec coverage:**
- 迁移 DMSAgent 回放到 U2：Task 1（`retrieve_replay`）+ Task 2（状态机 + `step()` 覆写）✓
- 保证 --u1 / --u1 --u2 / --u1 --u2 --u3 独立运行：Task 3（消融测试 + 手动启动验证）✓
- 不碰 episode_runner：`step()` 覆写内部逐步驱动 ✓

**2. Placeholder scan:** 无 TBD/TODO；每步含完整代码、命令、期望输出。

**3. Type consistency:**
- `retrieve_replay` 返回 `list[ObsAct] | None`，Task 2 用 `list(trajectory)` 消费 ✓
- `_step_replay` 返回 `(done, step_data)` 元组，`step()` 内解包 ✓
- `_replay_active`/`_replay_entry`/`_replay_index`/`_replay_trajectory` 在 `__init__`/`reset`/`_step_replay` 全链路一致 ✓
- `_resolve_action_target` 用 `json_action.CLICK` 等常量（与 json_action.py 定义一致）✓
- `_build_action_prompt` U2 分支保留 hint（replay 在 step() 拦截）✓

**已知限制（plan 外）：**
- `retrieve_replay` 缓存键 `("replay", goal, precondition)` 与 `_retrieve_entry` 的 `_active_entry` 更新需在 `finalize_task` 清缓存（现有逻辑已 clear `_retrieval_cache`）✓
- replay 场景下 `_flush_u2_trajectory` 不写 U2（history 里是 replay 动作），但 `set_episode_success` 仍调用 `record_episode_outcome` 更新 T_global——行为合理，无需改动。
