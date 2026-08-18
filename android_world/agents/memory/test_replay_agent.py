"""Offline tests for MemoryAugmentedAgent deterministic replay (U2)."""

import types
import unittest

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


class _FakeEnvFail(_FakeEnv):
  """Env whose execute_action raises for a configured action type."""

  def __init__(self, fail_on_action_type):
    super().__init__()
    self.fail_on = fail_on_action_type

  def execute_action(self, action):
    if action.action_type == self.fail_on:
      raise RuntimeError("boom")
    super().execute_action(action)


class _FakeLLM:
  def predict_mm(self, prompt, images):
    raise AssertionError("LLM must not be called during replay")


def _make_agent(env, llm):
  agent = memory_agent.MemoryAugmentedAgent(
      env, llm, enable_u1=False, enable_u2=True, enable_u3=False
  )
  return agent


def _stub_hit(agent, traj, memory_id="stub"):
  """Stub U2 sub-plan retrieval to always hit with `traj` and set a stub
  active entry.

  _active_entry must be a non-None object for _start_replay's _load_trajectory;
  use a MemoryEntry-like stub.  Since retrieve_sub_plan_replay is stubbed,
  _start_replay calls u2.bank._load_trajectory(entry) on a real bank with no
  such file -> [] so it falls back to list(trajectory).
  """
  agent.u2.retrieve_sub_plan_replay = lambda plan, T_global=0.5: list(traj)

  # NOTE: the class body cannot read `memory_id` directly (assigning it in the
  # class namespace makes it a class-local before the RHS is evaluated), so
  # alias it to a name not assigned in the class body.
  _mid = memory_id

  class _StubEntry:
    memory_id = _mid
    meta = types.SimpleNamespace(
        reuse_count=0, success_count=0, failure_count=0
    )

  agent.u2._active_entry = _StubEntry()
  # Decoupled storage: bank load of stub id returns [] → replay uses traj list.
  agent.u2.bank._load_trajectory = lambda entry: []


def _traj_open_click():
  return [
      ObsAct(observation="step_0",
             action=json_action.JSONAction(action_type="open_app", app_name="Files"),
             step_index=0),
      ObsAct(observation="step_1",
             action=json_action.JSONAction(action_type="click", index=2),
             step_index=1),
  ]


class TestReplay(unittest.TestCase):
  def test_replay_executes_cached_trajectory_without_llm(self):
    env = _FakeEnv()
    agent = _make_agent(env, _FakeLLM())
    _stub_hit(agent, _traj_open_click())

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

  def test_replay_failure_ends_replay_not_done(self):
    env = _FakeEnvFail(fail_on_action_type="click")
    agent = _make_agent(env, _FakeLLM())
    _stub_hit(agent, _traj_open_click(), memory_id="stub_fail")

    r1 = agent.step("Open Files and click")
    assert not r1.done
    assert agent._replay_active  # still replaying after step 0

    r2 = agent.step("Open Files and click")
    assert not r2.done  # failing action -> episode not done
    assert "Replay action failed" in r2.data["summary"]
    assert not agent._replay_active
    assert agent._replay_entry is None
    assert agent._replay_index == 0

  def test_replay_full_episode_does_not_restore(self):
    env = _FakeEnv()
    agent = _make_agent(env, _FakeLLM())
    _stub_hit(agent, _traj_open_click(), memory_id="stub_restore")
    initial_size = agent.u2.bank.size

    r1 = agent.step("Open Files and click")
    assert not r1.done
    r2 = agent.step("Open Files and click")
    assert not r2.done
    r3 = agent.step("Open Files and click")
    assert r3.done

    # Evaluator finalizes the episode with ground truth; the replayed
    # trajectory must NOT be re-stored into the bank (§3.2.2).
    agent.set_episode_success(True)
    assert agent.u2.bank.size == initial_size

  def test_flush_skips_replayed_history_entries(self):
    # Even with a non-None buffered goal, a history containing any replayed
    # entry must not be flushed back to the bank (double-insurance guard).
    env = _FakeEnv()
    agent = _make_agent(env, _FakeLLM())
    agent.history = [
        {"action_output_json": json_action.JSONAction(
            action_type="open_app", app_name="Files"), "u2_replayed": True},
        {"action_output_json": json_action.JSONAction(
            action_type="click", index=0), "u2_replayed": True},
    ]
    agent._pending_trajectory_goal = "Some goal"

    agent._flush_u2_trajectory("Some goal", {})
    assert agent.u2.bank.size == 0

  def test_replay_status_complete_returns_done(self):
    env = _FakeEnv()
    agent = _make_agent(env, _FakeLLM())
    traj = [
        ObsAct(observation="step_0",
               action=json_action.JSONAction(action_type="open_app", app_name="Files"),
               step_index=0),
        ObsAct(observation="step_1",
               action=json_action.JSONAction(action_type="status", goal_status="complete"),
               step_index=1),
    ]
    _stub_hit(agent, traj, memory_id="stub_status_done")

    r1 = agent.step("Open Files and click")
    assert not r1.done
    r2 = agent.step("Open Files and click")
    assert r2.done
    assert not agent._replay_active
    assert agent.u2.bank.size == 0

  def test_replay_status_infeasible_returns_not_done(self):
    env = _FakeEnv()
    agent = _make_agent(env, _FakeLLM())
    traj = [
        ObsAct(observation="step_0",
               action=json_action.JSONAction(action_type="status", goal_status="infeasible"),
               step_index=0),
    ]
    _stub_hit(agent, traj, memory_id="stub_status_infeasible")

    r1 = agent.step("Open Files and click")
    assert not r1.done
    assert "infeasible" in r1.data["summary"]
    assert not agent._replay_active

  def test_replay_off_when_u2_disabled(self):
    env = _FakeEnv()
    agent = memory_agent.MemoryAugmentedAgent(
        env, _FakeLLM(), enable_u1=True, enable_u2=False, enable_u3=False
    )
    # U2 disabled -> replay never activates.
    self.assertFalse(agent._replay_active)
    self.assertIsNone(agent.u2)

  def test_u1_only_initializes_u1(self):
    env = _FakeEnv()
    agent = memory_agent.MemoryAugmentedAgent(
        env, _FakeLLM(), enable_u1=True, enable_u2=False, enable_u3=False
    )
    self.assertTrue(agent.enable_u1)
    self.assertFalse(agent.enable_u2)
    self.assertFalse(agent.enable_u3)
    self.assertIsNone(agent.u2)   # u2 未初始化
    self.assertIsNone(agent.u3)   # u3 未初始化

  def test_u1_u2_initializes_u2_only(self):
    env = _FakeEnv()
    agent = memory_agent.MemoryAugmentedAgent(
        env, _FakeLLM(), enable_u1=True, enable_u2=True, enable_u3=False
    )
    self.assertIsNotNone(agent.u2)
    self.assertIsNone(agent.u3)
    # u1 是 lazy 的：首次 _build_action_prompt 才 init，这里验证未主动初始化
    self.assertIsNone(agent.u1)

  def test_u1_u2_u3_all_enabled_constructs(self):
    env = _FakeEnv()
    from unittest import mock
    with mock.patch(
        "android_world.agents.memory_agent.EnvKnowledge"
    ) as mock_ek:
      mock_ek.return_value = mock.Mock()
      agent = memory_agent.MemoryAugmentedAgent(
          env, _FakeLLM(), enable_u1=True, enable_u2=True, enable_u3=True,
          rag_url="http://127.0.0.1:18180",
      )
    self.assertTrue(agent.enable_u1)
    self.assertTrue(agent.enable_u2)
    self.assertTrue(agent.enable_u3)
    self.assertIsNotNone(agent.u2)
    self.assertIsNotNone(agent.u3)
    mock_ek.assert_called_once()
    _, kwargs = mock_ek.call_args
    self.assertEqual(kwargs.get("rag_url"), "http://127.0.0.1:18180")

  def test_u3_requires_rag_url(self):
    env = _FakeEnv()
    with self.assertRaises(ValueError):
      memory_agent.MemoryAugmentedAgent(
          env, _FakeLLM(), enable_u3=True, rag_url=""
      )

  def test_no_replay_state_without_u2(self):
    env = _FakeEnv()
    agent = memory_agent.MemoryAugmentedAgent(
        env, _FakeLLM(), enable_u1=False, enable_u2=False, enable_u3=False
    )
    # step() 直接走 M3A LLM 路径；_replay_active 保持 False
    self.assertFalse(agent._replay_active)


class _FlushHistoryAgent(memory_agent.MemoryAugmentedAgent):
  """Agent whose history we can seed directly to exercise _flush_u2_trajectory
  without running the full M3A step loop."""

  def __init__(self, env, llm):
    super().__init__(
        env, llm, enable_u1=False, enable_u2=True, enable_u3=False
    )
    self.history = []


def _two_step_history():
  return [
      {"action_output_json": json_action.JSONAction(
          action_type="open_app", app_name="Markor")},
      {"action_output_json": json_action.JSONAction(
          action_type="click", index=2)},
  ]


class TestFlushOnlyStoresSuccess(unittest.TestCase):
  """失败执行不创建可执行宏(对齐论文 Algorithm 1 §3.2.1)。"""

  def test_failed_trajectory_not_stored(self):
    env = _FakeEnv()
    agent = _FlushHistoryAgent(env, _FakeLLM())
    agent.history = _two_step_history()
    before = agent.u2.bank.size
    agent._flush_u2_trajectory("Create a note", {}, success=False)
    # 失败:不 add_trajectory,bank 不新增条目。
    self.assertEqual(agent.u2.bank.size, before)

  def test_failed_trajectory_with_active_hit_penalized(self):
    """失败但复用命中过(_active_entry 非空)时,仍要累积 F_i。"""
    env = _FakeEnv()
    agent = _FlushHistoryAgent(env, _FakeLLM())
    agent.history = _two_step_history()
    # 构造一个已命中条目,模拟"复用后失败"。
    entry = types.SimpleNamespace(
        meta=types.SimpleNamespace(
            success_count=0, failure_count=0, reuse_count=0,
            verification_failures=0,
        )
    )
    agent.u2._active_entry = entry
    agent.u2._last_added_entry = None
    agent._flush_u2_trajectory("Create a note", {}, success=False)
    # 失败命中条目:F_i 累积(抑制机制保留)。
    self.assertEqual(entry.meta.failure_count, 1)


class _SlotLLM:
  """LLM that records slot-fill calls and returns the configured text."""

  def __init__(self, text="08:00"):
    self.text = text
    self.calls = []

  def predict_mm(self, prompt, images):
    self.calls.append((prompt, images))
    return self.text, True, "raw"


def _traj_with_input_text():
  return [
      ObsAct(observation="step_0",
             action=json_action.JSONAction(action_type="open_app", app_name="Clock"),
             step_index=0),
      ObsAct(observation="step_1",
             action=json_action.JSONAction(action_type="click", index=3),
             step_index=1),
      ObsAct(observation="step_2",
             action=json_action.JSONAction(
                 action_type="input_text", text="07:00", index=5),
             step_index=2),
      ObsAct(observation="step_3",
             action=json_action.JSONAction(action_type="click", index=7),
             step_index=3),
  ]


class TestSlotReplay(unittest.TestCase):
  def test_input_text_slot_filled_by_llm_and_replay_continues(self):
    env = _FakeEnv()
    llm = _SlotLLM(text="08:00")
    agent = _make_agent(env, llm)
    _stub_hit(agent, _traj_with_input_text())

    # 第一步:回放开始,执行 open_app。
    result = agent.step("create alarm 08:00")
    self.assertFalse(result.done)

    # 第二步:click。
    result = agent.step("create alarm 08:00")
    self.assertFalse(result.done)

    # 第三步:input_text —— 填槽:调 LLM,得到 08:00,替换旧值 07:00。
    result = agent.step("create alarm 08:00")
    self.assertFalse(result.done)
    self.assertEqual(len(llm.calls), 1)
    filled = result.data["action_output_json"]
    self.assertEqual(filled.action_type, "input_text")
    self.assertEqual(filled.text, "08:00")
    # 回放不终止:填槽后仍有步骤要执行。
    self.assertTrue(agent._replay_active)

    # 第四步:继续回放后续 click。
    result = agent.step("create alarm 08:00")
    self.assertFalse(result.done)

    # 轨迹耗尽:回放结束。
    result = agent.step("create alarm 08:00")
    self.assertTrue(result.done)
    self.assertFalse(agent._replay_active)

  def test_slot_fill_llm_failure_ends_replay(self):
    env = _FakeEnv()
    llm = _SlotLLM(text="")  # 空输出 → 填槽失败
    agent = _make_agent(env, llm)
    _stub_hit(agent, _traj_with_input_text())

    agent.step("create alarm 08:00")  # open_app
    agent.step("create alarm 08:00")  # click
    result = agent.step("create alarm 08:00")  # input_text
    self.assertFalse(result.done)
    self.assertEqual(result.data["action_output_json"], None)
    self.assertFalse(agent._replay_active)
    self.assertIn("slot-fill failed", result.data["summary"])

