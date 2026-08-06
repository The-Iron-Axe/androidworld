# android_world/agents/memory/test_procedural.py
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from android_world.agents.memory import ProceduralMemory
from android_world.agents.memory.skill import Skill, SkillAction
from android_world.env import json_action


def _act(action_type, **kwargs):
  return json_action.JSONAction(action_type=action_type, **kwargs)


class ProceduralMemoryTest(unittest.TestCase):

  def _make_memory(self, persistence_dir=""):
    return ProceduralMemory(persistence_dir=persistence_dir)

  def test_mine_from_successful_trajectories(self):
    mem = self._make_memory()
    traj = [
      _act("open_app", app_name="markor"),
      _act("click", index=1),
      _act("input_text", text="hello", index=2),
    ]
    mem.add_successful_trajectory(
        "Create a note", traj, precondition="Markor main screen")
    added = mem.mine()
    self.assertGreaterEqual(added, 1)
    self.assertEqual(mem.size, 1)

  def test_atomic_trajectory_skipped(self):
    mem = self._make_memory()
    mem.add_successful_trajectory("Open Markor", [_act("open_app", app_name="markor")])
    added = mem.mine()
    self.assertEqual(added, 0)
    self.assertEqual(mem.size, 0)

  def test_retrieve_hint_hit_and_miss(self):
    mem = self._make_memory()
    mem.library.add_skill(Skill(
        goal_hint="Create a new note",
        precondition="Markor main screen",
        actions=[
            SkillAction("click", target="Compose"),
            SkillAction("input_text", target="Subject"),
        ],
    ))
    hit = mem.retrieve_hint("Create a new note", precondition="Markor main screen")
    self.assertIn("click", hit)
    self.assertIn("input_text", hit)
    miss = mem.retrieve_hint("Buy groceries", precondition="Home")
    self.assertEqual(miss, "")

  def test_record_outcome_updates_and_evicts(self):
    mem = self._make_memory()
    mem.library.add_skill(Skill(
        goal_hint="Create a new note",
        actions=[SkillAction("click", target="Compose")],
        score=2.0,
    ))
    # One failure: score 2 -> 1, skill survives.
    mem.record_outcome("Create a new note", success=False)
    self.assertEqual(mem.size, 1)
    self.assertEqual(mem.library.get("Create a new note").score, 1.0)
    # Two more failures: 1 -> 0 -> evict at score <= 0.
    mem.record_outcome("Create a new note", success=False)
    mem.record_outcome("Create a new note", success=False)
    self.assertEqual(mem.size, 0)

  def test_persistence_roundtrip(self):
    with tempfile.TemporaryDirectory() as d:
      mem = self._make_memory(d)
      mem.library.add_skill(Skill(
          goal_hint="Create a new note",
          precondition="Markor main screen",
          actions=[SkillAction("click", target="Compose")],
      ))
      mem.save()

      mem2 = self._make_memory(d)
      self.assertEqual(mem2.size, 1)
      sk = mem2.library.get("Create a new note")
      self.assertIsNotNone(sk)
      self.assertEqual(sk.precondition, "Markor main screen")

  def test_stats(self):
    mem = self._make_memory()
    st = mem.stats()
    self.assertEqual(st["skills"], 0)
    self.assertEqual(st["buffered"], 0)


if __name__ == "__main__":
  unittest.main()
