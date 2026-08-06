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

  def test_failed_trajectory_mines_negative_skill(self):
    """Failed trajectories buffer and mine into negative 'avoid' skills."""
    mem = self._make_memory()
    traj = [
      _act("open_app", app_name="markor"),
      _act("long_press", index=1),
      _act("click", index=2),
    ]
    mem.add_failed_trajectory("Create a note", traj, precondition="Markor main screen")
    added = mem.mine()
    self.assertGreaterEqual(added, 1)
    self.assertEqual(mem.stats()["positive"], 0)
    self.assertGreaterEqual(mem.stats()["negative"], 1)

  def test_positive_and_negative_retrieved_together(self):
    """Both a positive and a negative skill for the same goal are retrieved —
    the prompt carries "how to do it" AND "what to avoid" simultaneously."""
    mem = self._make_memory()
    # Only a negative skill -> hint carries [Avoid].
    mem.add_failed_trajectory(
        "Create a note",
        [_act("open_app", app_name="markor"), _act("long_press", index=1)],
        precondition="Markor main screen",
    )
    mem.mine()
    neg_hint = mem.retrieve_hint("Create a note", precondition="Markor main screen")
    self.assertIn("[Avoid]", neg_hint)
    self.assertNotIn("[Skill]", neg_hint)

    # Add a positive skill for the same goal -> now BOTH appear.
    mem.add_successful_trajectory(
        "Create a note",
        [_act("open_app", app_name="markor"), _act("click", index=1)],
        precondition="Markor main screen",
    )
    mem.mine()
    both = mem.retrieve_hint("Create a note", precondition="Markor main screen")
    self.assertIn("[Skill]", both)
    self.assertIn("[Avoid]", both)

  def test_retrieve_blocks_separates_kinds(self):
    """retrieve_blocks returns positive and negative as independent keys."""
    mem = self._make_memory()
    mem.add_successful_trajectory(
        "Create a note",
        [_act("open_app", app_name="markor"), _act("click", index=1)],
        precondition="Markor main screen",
    )
    mem.add_failed_trajectory(
        "Create a note",
        [_act("open_app", app_name="markor"), _act("long_press", index=1)],
        precondition="Markor main screen",
    )
    mem.mine()
    blocks = mem.retrieve_blocks("Create a note", precondition="Markor main screen")
    self.assertIn("[Skill]", blocks["positive"])
    self.assertIn("[Avoid]", blocks["negative"])

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

  def test_record_outcome_evicts_negative_skill(self):
    """A negative skill that gets followed and still fails is penalized too."""
    mem = self._make_memory()
    mem.add_failed_trajectory(
        "Create a note",
        [_act("open_app", app_name="markor"), _act("long_press", index=1)],
        precondition="Markor main screen",
    )
    mem.mine()
    self.assertGreaterEqual(mem.stats()["negative"], 1)
    # score starts at 1.0; one failure -> 0 -> evict.
    mem.record_outcome("Create a note", success=False)
    self.assertEqual(mem.stats()["negative"], 0)

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
    self.assertEqual(st["positive"], 0)
    self.assertEqual(st["negative"], 0)
    self.assertEqual(st["buffered"], 0)
    self.assertEqual(st["failed_buffered"], 0)


if __name__ == "__main__":
  unittest.main()
