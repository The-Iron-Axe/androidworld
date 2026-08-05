# android_world/agents/memory/test_episodic.py
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
  sys.path.insert(0, _REPO_ROOT)

from android_world.agents.memory import EpisodicMemory
from android_world.agents.memory import dms_bridge
from android_world.agents.memory.dms_bridge import (
    DMSConfig, MemoryBank, MemoryEntry, ObsAct, Plan, RetrievalResult,
)
from android_world.env import json_action


class DMSBridgeTest(unittest.TestCase):

    def test_plan_constructs(self):
        p = Plan(precondition="Home screen", goal="Open Markor")
        self.assertEqual(p.precondition, "Home screen")
        self.assertEqual(p.goal, "Open Markor")

    def test_obsact_constructs(self):
        oa = ObsAct(observation="step_0", action="click", step_index=0)
        self.assertEqual(oa.action, "click")

    def test_retrieval_result_defaults(self):
        r = RetrievalResult(hit=False)
        self.assertFalse(r)
        self.assertIsNone(r.entry)
        self.assertEqual(r.score, 0.0)


class FakeBackend:
    """Deterministic fake embedding backend — no network, no ST download.

    A goal "exactly matches" itself and similar goals produce a high
    cosine score; unrelated goals score ~0.  This lets the sub-plan
    retrieval path be tested offline.
    """
    def __init__(self):
        self._dim = 8

    def encode(self, text: str):
        import numpy as np
        v = np.zeros(self._dim, dtype=np.float32)
        for i, ch in enumerate(text.lower()):
            v[i % self._dim] += ord(ch)
        norm = float(np.linalg.norm(v)) or 1.0
        return v / norm

    def encode_batch(self, texts: list[str]):
        import numpy as np
        return np.stack([self.encode(t) for t in texts], axis=0)

    @property
    def dim(self) -> int:
        return self._dim


class EpisodicMemorySubPlanTest(unittest.TestCase):

    def _make_memory(self, persistence_dir: str):
        config = DMSConfig()
        config.disk_storage_dir = persistence_dir
        # DMS rolls its ε-mutation on the global random; force it off so the
        # sub-plan retrieval test asserts the deterministic hit path.
        config.epsilon = 0.0
        mem = EpisodicMemory(config=config, persistence_dir=persistence_dir)
        mem._initialized = True
        mem.bank._embedder = FakeBackend()
        mem.bank._embedder_initialized = True
        return mem

    def test_add_and_retrieve_sub_plan(self):
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan = Plan(precondition="Markor main screen", goal="Create a new note")
            traj = [
                ObsAct(observation="step_0", action="click", step_index=0),
                ObsAct(observation="step_1", action="input_text", step_index=1),
            ]
            mem.add_sub_plan(plan, traj)
            self.assertEqual(mem.size, 1)

            hint = mem.retrieve_sub_plan_hint(plan)
            self.assertIn("click", hint)
            self.assertIn("input_text", hint)

    def test_retrieve_sub_plan_miss_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan = Plan(precondition="Home", goal="Send an SMS")
            self.assertEqual(mem.retrieve_sub_plan_hint(plan), "")


    def test_precondition_discriminates(self):
        # Same goal, different precondition must score strictly LOWER than an
        # exact match.  Dual-factor retrieval is sim(pre)·sim(goal), so the
        # precondition factor must measurably reduce the score even when the
        # goal term is 1.0.  (The FakeBackend's position-hash keeps even very
        # different precondition strings above the 0.6 hit threshold, so this
        # asserts the relative drop rather than a threshold miss.)
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan_a = Plan(precondition="Markor main screen", goal="Create a new note")
            traj = [
                ObsAct(observation="step_0", action="click", step_index=0),
                ObsAct(observation="step_1", action="input_text", step_index=1),
            ]
            mem.add_sub_plan(plan_a, traj)
            self.assertEqual(mem.size, 1)

            # Exact same precondition+goal -> full score.
            exact = mem.bank.retrieve(plan_a, T_global=0.5)
            self.assertTrue(exact.hit)
            self.assertAlmostEqual(exact.score, 1.0, places=3)

            # Same goal, very different precondition -> the precondition factor
            # must drag the dual-factor score strictly below the exact match.
            plan_b = Plan(
                precondition="Xylophone accounting board", goal="Create a new note"
            )
            differ = mem.bank.retrieve(plan_b, T_global=0.5)
            self.assertTrue(differ.hit)  # score still above the 0.6 threshold
            self.assertLess(differ.score, exact.score)

            # The sub-plan hint API must still retrieve the exact match.
            hit = mem.retrieve_sub_plan_hint(plan_a)
            self.assertNotEqual(hit, "")


class EpisodicMemoryWholeTaskTest(unittest.TestCase):

    def _make_memory(self, persistence_dir: str):
        config = DMSConfig()
        config.disk_storage_dir = persistence_dir
        # Force DMS ε-mutation off so the whole-task retrieval test asserts
        # the deterministic hit path (matches EpisodicMemorySubPlanTest).
        config.epsilon = 0.0
        mem = EpisodicMemory(config=config, persistence_dir=persistence_dir)
        mem._initialized = True
        mem.bank._embedder = FakeBackend()
        mem.bank._embedder_initialized = True
        return mem

    def test_retrieve_hint_accepts_precondition(self):
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan = Plan(precondition="Markor main screen", goal="Create a new note")
            traj = [
                ObsAct(observation="step_0", action="open_app", step_index=0),
                ObsAct(observation="step_1", action="click", step_index=1),
            ]
            mem.add_sub_plan(plan, traj)
            hint = mem.retrieve_hint(
                "Create a new note", precondition="Markor main screen"
            )
            self.assertIn("open_app", hint)


class GlobalFailureRateTest(unittest.TestCase):

    def test_global_failure_rate_starts_at_prior(self):
        with tempfile.TemporaryDirectory() as d:
            config = DMSConfig()
            config.disk_storage_dir = d
            mem = EpisodicMemory(config=config, persistence_dir=d)
            self.assertAlmostEqual(mem.global_failure_rate, 0.5)

    def test_global_failure_rate_after_observations(self):
        with tempfile.TemporaryDirectory() as d:
            config = DMSConfig()
            config.disk_storage_dir = d
            mem = EpisodicMemory(config=config, persistence_dir=d)
            mem.record_episode_outcome(True)
            mem.record_episode_outcome(True)
            mem.record_episode_outcome(False)
            rate = mem.global_failure_rate
            # 1 failure out of 3, blended with uniform prior (1,1):
            # (1 + 2*0.5) / (3 + 2) = 2/5 = 0.4
            self.assertAlmostEqual(rate, 0.4)

    def test_global_failure_rate_all_success(self):
        with tempfile.TemporaryDirectory() as d:
            config = DMSConfig()
            config.disk_storage_dir = d
            mem = EpisodicMemory(config=config, persistence_dir=d)
            for _ in range(10):
                mem.record_episode_outcome(True)
            rate = mem.global_failure_rate
            # (0 + 2*0.5) / (10 + 2) = 1/12 ≈ 0.0833
            self.assertAlmostEqual(rate, 1 / 12)


class EpisodicMemoryReplayTest(unittest.TestCase):

    def test_retrieve_replay_returns_trajectory_on_hit(self):
        config = DMSConfig()
        # Force DMS ε-mutation off so this test asserts the deterministic
        # replay hit path (matches the whole-task/sub-plan tests above).  With
        # the default epsilon=0.1 the mutation roll is non-deterministic and
        # this test would intermittently flake (retrieve_replay returns None).
        config.epsilon = 0.0
        with tempfile.TemporaryDirectory() as d:
            u2 = EpisodicMemory(config=config, persistence_dir=d)
            u2.init_embedding(corpus_texts=["open app then click save"])
            goal = "Save a file"
            traj = [ObsAct(observation="step_0", action=json_action.JSONAction(action_type="open_app", app_name="Files"), step_index=0),
                    ObsAct(observation="step_1", action=json_action.JSONAction(action_type="click", index=2), step_index=1)]
            u2.add_trajectory(goal, traj)
            replay = u2.retrieve_replay(goal)
            assert replay is not None
            assert len(replay) == 2
            assert replay[0].action.action_type == "open_app"


if __name__ == "__main__":
    unittest.main()
