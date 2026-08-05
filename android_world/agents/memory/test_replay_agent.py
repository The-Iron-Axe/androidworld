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
  return agent


class TestReplay:
  def test_replay_executes_cached_trajectory_without_llm(self):
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
    # _active_entry must be a non-None object for _start_replay's _load_trajectory;
    # use a MemoryEntry-like stub.  Since retrieve_replay is stubbed, _start_replay
    # calls u2.bank._load_trajectory(entry) on a real bank with no such file -> []
    # so it falls back to list(trajectory).  Provide a minimal entry object.
    class _StubEntry:
      memory_id = "stub"
    agent.u2._active_entry = _StubEntry()

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
