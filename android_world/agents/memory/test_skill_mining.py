# android_world/agents/memory/test_skill_mining.py
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from android_world.agents.memory.skill import Skill, SkillAction, SkillLibrary
from android_world.agents.memory.skill_mining import (
    bpe_merge,
    mine_skills,
    semantic_token,
    tokenize_trajectory,
)
from android_world.env import json_action


def _act(action_type, **kwargs):
  return json_action.JSONAction(action_type=action_type, **kwargs)


def _mk(t, text=""):
  """Fake UI element exposing only the semantic attrs semantic_token reads."""
  class _El:
    def __init__(self, text, cd, hint):
      self.text = text
      self.content_description = cd
      self.hint_text = hint
  return _El(t, "", "")


class TokenizeTest(unittest.TestCase):

  def test_semantic_token_click_uses_text(self):
    el = _mk("Compose")
    a = _act("click", index=3)
    a.element = el
    tok = semantic_token(a)
    self.assertEqual(tok, "click@Compose")

  def test_semantic_token_open_app(self):
    a = _act("open_app", app_name="markor")
    self.assertEqual(semantic_token(a), "open_app(markor)")

  def test_semantic_token_input_text_ignores_value(self):
    # input_text token must NOT carry the typed value (it would overfit a skill).
    a = _act("input_text", text="hello world", index=1)
    a.element = _mk("Subject")
    tok = semantic_token(a)
    self.assertEqual(tok, "input_text@Subject")

  def test_tokenize_skips_none(self):
    toks = tokenize_trajectory([_act("open_app", app_name="a"), None, _act("click", index=0)])
    self.assertEqual(toks, ["open_app(a)", "click"])


class BpeMergeTest(unittest.TestCase):

  def test_merges_frequent_pair(self):
    # (click, open_app) appears 3x across the two trajectories.
    t1 = ["click", "open_app(a)", "click"]
    t2 = ["click", "open_app(a)", "scroll:down"]
    merged = bpe_merge([t1, t2], min_freq=2, max_iters=2)
    # The pair (click, open_app(a)) should merge into one group.
    groups = [g for traj in merged for g in traj]
    self.assertTrue(any(g == ("click", "open_app(a)") for g in groups))

  def test_no_merge_below_min_freq(self):
    t1 = ["click", "scroll:up"]
    t2 = ["open_app(b)", "click", "scroll:up"]
    merged = bpe_merge([t1, t2], min_freq=3, max_iters=2)
    # (click, scroll:up) appears 2x < 3, so no group is longer than 1 token.
    groups = [g for traj in merged for g in traj]
    self.assertTrue(all(len(g) == 1 for g in groups))


class MineSkillsTest(unittest.TestCase):

  def test_mine_produces_parameterized_skill(self):
    # Two successful trajectories: click "Compose" -> input_text into "Subject"
    # -> click "Send".  Targets are identical -> no slots; goal/precondition seed.
    traj = [
      [_act("click", index=1), _act("input_text", text="hello", index=2),
       _act("click", index=5)],
      [_act("click", index=7), _act("input_text", text="world", index=9),
       _act("click", index=0)],
    ]
    for i, t in enumerate(traj):
      t[0].element = _mk("Compose")
      t[1].element = _mk("Subject")
      t[2].element = _mk("Send")
    skills = mine_skills(
        traj,
        goal_hints=["Create a new note"] * 2,
        preconditions=["Markor main screen"] * 2,
        min_freq=2,
        max_iters=4,
    )
    self.assertEqual(len(skills), 1)
    sk = skills[0]
    self.assertEqual(sk.goal_hint, "Create a new note")
    self.assertEqual(sk.precondition, "Markor main screen")
    self.assertGreaterEqual(len(sk.actions), 3)
    types = [a.action_type for a in sk.actions]
    self.assertEqual(types[:2], ["click", "input_text"])
    # Targets survive as semantic selectors, never indices.
    self.assertEqual(sk.actions[0].target, "Compose")
    self.assertEqual(sk.actions[1].target, "Subject")

  def test_mine_abstracts_varying_targets_to_slots(self):
    # Same structure, different target texts across trajectories -> slot.
    traj = [
      [_act("click", index=0), _act("input_text", text="a", index=1)],
      [_act("click", index=0), _act("input_text", text="b", index=1)],
    ]
    traj[0][1].element = _mk("Recipient A")
    traj[1][1].element = _mk("Recipient B")
    skills = mine_skills(
        traj,
        goal_hints=["Send SMS to X"] * 2,
        preconditions=["SMS app main"] * 2,
    )
    self.assertEqual(len(skills), 1)
    sk = skills[0]
    # The varying target should become a slot reference.
    self.assertIn("slot", sk.actions[1].target or "")
    self.assertTrue(sk.slots)

  def test_mine_negative_kind(self):
    """Failed trajectories mine into negative 'avoid' skills."""
    traj = [
      [_act("click", index=1), _act("input_text", text="hi", index=2)],
      [_act("click", index=1), _act("input_text", text="yo", index=2)],
    ]
    for t in traj:
      t[0].element = _mk("Trash")
      t[1].element = _mk("Subject")
    skills = mine_skills(
        traj,
        goal_hints=["Create a note"] * 2,
        preconditions=["Markor main"] * 2,
        kind="negative",
    )
    self.assertEqual(len(skills), 1)
    self.assertEqual(skills[0].kind, "negative")
    self.assertEqual(skills[0].goal_hint, "Create a note")


class GoalAbstractionTest(unittest.TestCase):

  def test_abstracts_concrete_filename(self):
    from android_world.agents.memory.skill_mining import _abstract_goal_hint
    g = _abstract_goal_hint("Delete the note in Markor named bold_king_edited.")
    self.assertNotIn("bold_king_edited", g)
    self.assertIn("Markor", g)

  def test_abstracts_timestamp_folder(self):
    from android_world.agents.memory.skill_mining import _abstract_goal_hint
    g = _abstract_goal_hint("Create a new folder in Markor named folder_20260806_143035.")
    self.assertNotIn("folder_20260806_143035", g)
    self.assertIn("Create a new folder", g)

  def test_keeps_task_verb(self):
    from android_world.agents.memory.skill_mining import _abstract_goal_hint
    g = _abstract_goal_hint("Open the Settings app and turn off wifi.")
    self.assertEqual(g, "Open the Settings app and turn off wifi.")

  def test_mine_uses_abstracted_goal(self):
    from android_world.agents.memory.skill_mining import mine_skills
    traj = [
      [_act("click", index=1), _act("input_text", text="hi", index=2)],
      [_act("click", index=1), _act("input_text", text="yo", index=2)],
    ]
    traj[0][1].element = _mk("Title")
    traj[1][1].element = _mk("Title")
    skills = mine_skills(
        traj,
        goal_hints=["Create a note named folder_20260806_143035."] * 2,
        preconditions=["Markor main"] * 2,
    )
    self.assertEqual(len(skills), 1)
    self.assertNotIn("folder_20260806_143035", skills[0].goal_hint)


class BareJSONActionTest(unittest.TestCase):
  """Real-agent path: actions are raw JSONAction WITHOUT .element attached."""

  def test_semantic_token_no_element_falls_back_to_bare_type(self):
    from android_world.agents.memory.skill_mining import semantic_token
    # 真实 agent 的 JSONAction: 没有 element 也没有 _semantic_target
    a = _act("click", index=3)
    self.assertIsNone(getattr(a, "element", None))
    tok = semantic_token(a)
    self.assertEqual(tok, "click")  # 退化为裸动作类型，不崩

  def test_semantic_token_uses_semantic_target(self):
    from android_world.agents.memory.skill_mining import semantic_token
    a = _act("click", index=3)
    a._semantic_target = "Compose"  # 真实路径由 _flush_u4_trajectory 绑定
    self.assertEqual(semantic_token(a), "click@Compose")


class SkillLibraryTest(unittest.TestCase):

  def test_add_and_get_dedup(self):
    lib = SkillLibrary()
    sk1 = Skill(goal_hint="Open Markor", actions=[SkillAction("open_app", app="markor")])
    lib.add_skill(sk1)
    sk2 = Skill(goal_hint="Open Markor", actions=[SkillAction("open_app", app="markor")])
    sid2 = lib.add_skill(sk2)
    self.assertEqual(len(lib.all()), 1)  # dedup by goal_hint
    self.assertEqual(lib.get("Open Markor").goal_hint, "Open Markor")
    self.assertEqual(sid2, lib._skill_id("Open Markor"))

  def test_positive_and_negative_same_goal_coexist(self):
    """A positive and a negative skill for the same goal must NOT overwrite
    each other — they are distinct memories keyed by (goal_hint, kind)."""
    lib = SkillLibrary()
    lib.add_skill(Skill(goal_hint="Create a note",
                        actions=[SkillAction("click", target="Compose")],
                        kind="positive"))
    lib.add_skill(Skill(goal_hint="Create a note",
                        actions=[SkillAction("long_press", target="Trash")],
                        kind="negative"))
    self.assertEqual(len(lib.all()), 2)
    self.assertIsNotNone(lib.get("Create a note", "positive"))
    self.assertIsNotNone(lib.get("Create a note", "negative"))

  def test_kind_roundtrip_through_persistence(self):
    with tempfile.TemporaryDirectory() as d:
      lib = SkillLibrary(persist_dir=d)
      lib.add_skill(Skill(goal_hint="Create a note",
                          actions=[SkillAction("click", target="Compose")],
                          kind="negative"))
      lib.save()
      lib2 = SkillLibrary(persist_dir=d)
      neg = lib2.get("Create a note", "negative")
      self.assertIsNotNone(neg)
      self.assertEqual(neg.kind, "negative")
      self.assertIsNone(lib2.get("Create a note", "positive"))

  def test_instantiate_binds_slots(self):
    sk = Skill(
        goal_hint="Send a note",
        slots=["recipient"],
        actions=[
            SkillAction("click", target="New"),
            SkillAction("input_text", target="Recipient", params={"text": "{recipient}"}),
        ],
    )
    bound = sk.instantiate({"recipient": "Mom"})
    self.assertEqual(bound[0].target, "New")
    self.assertEqual(bound[1].params["text"], "Mom")


if __name__ == "__main__":
  unittest.main()
