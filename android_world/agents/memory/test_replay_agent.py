"""Offline tests for MemoryAugmentedAgent deterministic replay (U2)."""

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
  """Stub U2 retrieval to always hit with `traj` and set a stub active entry.

  _active_entry must be a non-None object for _start_replay's _load_trajectory;
  use a MemoryEntry-like stub.  Since retrieve_replay is stubbed, _start_replay
  calls u2.bank._load_trajectory(entry) on a real bank with no such file -> []
  so it falls back to list(trajectory).
  """
  agent.u2.retrieve_replay = lambda goal, precondition=None: list(traj)

  # NOTE: the class body cannot read `memory_id` directly (assigning it in the
  # class namespace makes it a class-local before the RHS is evaluated), so
  # alias it to a name not assigned in the class body.
  _mid = memory_id

  class _StubEntry:
    memory_id = _mid
  agent.u2._active_entry = _StubEntry()


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
    agent = memory_agent.MemoryAugmentedAgent(
        env, _FakeLLM(), enable_u1=True, enable_u2=True, enable_u3=True,
        rag_url=None,
    )
    self.assertTrue(agent.enable_u1)
    self.assertTrue(agent.enable_u2)
    self.assertTrue(agent.enable_u3)
    self.assertIsNotNone(agent.u2)
    self.assertIsNotNone(agent.u3)

  def test_no_replay_state_without_u2(self):
    env = _FakeEnv()
    agent = memory_agent.MemoryAugmentedAgent(
        env, _FakeLLM(), enable_u1=False, enable_u2=False, enable_u3=False
    )
    # step() 直接走 M3A LLM 路径；_replay_active 保持 False
    self.assertFalse(agent._replay_active)
