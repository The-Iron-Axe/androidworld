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

    def test_retrieve_sub_plan_replay_returns_trajectory(self):
        """Sub-plan-level deterministic replay: retrieve_replay but keyed by
        Plan, so a future multi-agent Planner can fetch each sub-plan's cached
        trajectory independently."""
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan = Plan(precondition="Markor main screen", goal="Create a new note")
            traj = [
                ObsAct(observation="step_0", action="click", step_index=0),
                ObsAct(observation="step_1", action="input_text", step_index=1),
            ]
            mem.add_sub_plan(plan, traj)
            self.assertEqual(mem.size, 1)

            replay = mem.retrieve_sub_plan_replay(plan)
            self.assertIsNotNone(replay)
            self.assertEqual(len(replay), 2)
            self.assertEqual(replay[0].action, "click")

    def test_retrieve_sub_plan_replay_returns_none_on_miss(self):
        with tempfile.TemporaryDirectory() as d:
            mem = self._make_memory(d)
            plan = Plan(precondition="Home", goal="Send an SMS")
            self.assertIsNone(mem.retrieve_sub_plan_replay(plan))


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

    def test_global_failure_rate_ignores_plan_stats(self):
        """T_global must come from episode outcomes only; there is no
        per-plan feedback path in U2 (no Planner), so replay-heavy rounds
        don't skew the dynamic risk threshold."""
        with tempfile.TemporaryDirectory() as d:
            config = DMSConfig()
            config.disk_storage_dir = d
            mem = EpisodicMemory(config=config, persistence_dir=d)
            # No plan-stats mechanism exists on EpisodicMemory.
            self.assertFalse(hasattr(mem, "_plan_stats"))
            # Without episode outcomes, rate stays at uniform prior 0.5.
            self.assertAlmostEqual(mem.global_failure_rate, 0.5)
            # With episode outcomes, rate reflects episodes only.
            mem.record_episode_outcome(True)
            mem.record_episode_outcome(True)
            mem.record_episode_outcome(False)
            self.assertAlmostEqual(mem.global_failure_rate, 0.4)


class EpisodicMemoryReplayTest(unittest.TestCase):

    def _make_memory(self, persistence_dir: str):
        # Shared with the whole-task/sub-plan tests: FakeBackend (no network,
        # no SentenceTransformer) and ε-mutation off so the deterministic
        # replay hit path is asserted.  With the default epsilon=0.1 the
        # mutation roll would make retrieve_replay intermittently return None.
        config = DMSConfig()
        config.disk_storage_dir = persistence_dir
        config.epsilon = 0.0
        mem = EpisodicMemory(config=config, persistence_dir=persistence_dir)
        mem._initialized = True
        mem.bank._embedder = FakeBackend()
        mem.bank._embedder_initialized = True
        return mem

    def test_retrieve_replay_returns_trajectory_on_hit(self):
        with tempfile.TemporaryDirectory() as d:
            u2 = self._make_memory(d)
            goal = "Save a file"
            precondition = "Markor main screen"
            traj = [ObsAct(observation="step_0", action=json_action.JSONAction(action_type="open_app", app_name="Files"), step_index=0),
                    ObsAct(observation="step_1", action=json_action.JSONAction(action_type="click", index=2), step_index=1)]
            u2.add_trajectory(goal, traj, precondition=precondition)
            replay = u2.retrieve_replay(goal, precondition=precondition)
            self.assertIsNotNone(replay)
            self.assertEqual(len(replay), 2)
            self.assertEqual(replay[0].action.action_type, "open_app")

    def test_retrieve_replay_returns_none_on_miss(self):
        with tempfile.TemporaryDirectory() as d:
            u2 = self._make_memory(d)
            self.assertIsNone(u2.retrieve_replay("A goal never stored before"))


class ColdStartProtectionTest(unittest.TestCase):
    """失败任务创建的新条目必须保持 F=0,冷启动保护才生效(对齐论文
    Algorithm 1 第 20 行 S←1, F←0, K←0)。否则新条目第一次检索就被
    贝叶斯风险门控挡下,记忆永远无法被回放试用。"""

    def _make_memory(self, persistence_dir: str):
        config = DMSConfig()
        config.disk_storage_dir = persistence_dir
        config.epsilon = 0.0
        return EpisodicMemory(config=config, persistence_dir=persistence_dir)

    def _traj(self):
        return [
            ObsAct(observation="s0",
                   action=json_action.JSONAction(action_type="open_app", app_name="Markor"),
                   step_index=0),
            ObsAct(observation="s1",
                   action=json_action.JSONAction(action_type="click", index=2),
                   step_index=1),
            ObsAct(observation="s2",
                   action=json_action.JSONAction(action_type="input_text", text="hi", index=5),
                   step_index=2),
        ]

    def test_failed_task_new_entry_keeps_failure_count_zero(self):
        """失败任务留下的新轨迹 failure_count 必须为 0(冷启动豁免)。"""
        with tempfile.TemporaryDirectory() as d:
            u2 = self._make_memory(d)
            entry = u2.add_trajectory("Create a note", self._traj())
            u2.finalize_task("Create a note", success=False)
            self.assertEqual(entry.meta.failure_count, 0)
            self.assertEqual(entry.meta.success_count, 0)

    def test_cold_start_entry_not_risk_blocked(self):
        """F=0,S=0 的新条目检索时不应被风险门控挡掉。"""
        from others.darwinian_memory.risk import (
            compute_dynamic_threshold, compute_memory_risk_score,
        )
        with tempfile.TemporaryDirectory() as d:
            u2 = self._make_memory(d)
            entry = u2.add_trajectory("Create a note", self._traj())
            u2.finalize_task("Create a note", success=False)
            self.assertEqual(entry.meta.failure_count, 0)
            T_i, _, _ = compute_memory_risk_score(entry)
            tau = compute_dynamic_threshold(u2.global_failure_rate)
            self.assertLess(T_i, tau)

    def test_reused_entry_failure_accumulates_risk(self):
        """被复用后仍失败:failure_count 必须累积,风险分升高到会被挡。"""
        from others.darwinian_memory.risk import (
            compute_dynamic_threshold, compute_memory_risk_score,
        )
        with tempfile.TemporaryDirectory() as d:
            u2 = self._make_memory(d)
            entry = u2.add_trajectory("Create a note", self._traj())
            # 第一次:新条目创建(F=0,不挡)。
            u2.finalize_task("Create a note", success=False)
            # 第二次:该条目被复用(active)但任务仍失败 → F=1。
            u2._active_entry = entry
            u2.finalize_task("Create a note", success=False)
            self.assertEqual(entry.meta.failure_count, 1)
            T_i, _, _ = compute_memory_risk_score(entry)
            tau = compute_dynamic_threshold(u2.global_failure_rate)
            # F=1 时风险分应高于 F=0 时,并足以触发抑制(在默认阈值下)。
            self.assertGreater(T_i, 0.4)


if __name__ == "__main__":
    unittest.main()
