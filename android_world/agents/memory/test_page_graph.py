import re
import tempfile
import unittest
from unittest import mock

import numpy as np

from android_world.agents.memory.page_graph import PageGraph
from android_world.agents.memory.environment import EnvKnowledge, build_screen_summary
from android_world.agents.memory_agent import _action_effect_str, _describe_element


class FakeEmbedder:
    """Bag-of-words embedder — overlapping tokens => high cosine similarity,
    disjoint tokens => low.  Deterministic, offline, discriminates properly
    (unlike the position-hash backend used elsewhere)."""

    def __init__(self, dim: int = 64):
        self._dim = dim

    def encode(self, text: str):
        v = np.zeros(self._dim, dtype=np.float32)
        for tok in set(re.findall(r"[a-z0-9_]+", str(text).lower())):
            v[sum(ord(c) for c in tok) % self._dim] += 1.0
        norm = float(np.linalg.norm(v)) or 1.0
        return v / norm

    def encode_batch(self, texts: list[str]):
        return np.stack([self.encode(t) for t in texts], axis=0)

    @property
    def dim(self) -> int:
        return self._dim


class PageGraphAddTest(unittest.TestCase):

    def _graph(self, d: str) -> PageGraph:
        return PageGraph(persist_dir=d, embedder=FakeEmbedder())

    def test_add_transition_creates_nodes_and_edge(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition(
                before_summary="Markor main screen",
                action_summary="click new note",
                task="Create a note",
                after_summary="Markor editor screen",
                before_app="net.gsantner.markor",
                after_app="net.gsantner.markor",
            )
            self.assertEqual(len(g.nodes), 2)
            self.assertEqual(len(g.edges), 1)
            e = g.edges[0]
            self.assertEqual(e.action_summary, "click new note")
            self.assertEqual(e.task, "Create a note")

    def test_identical_page_merges_into_existing_node(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main", "click A", "t1", "Markor editor")
            g.add_transition("Markor main", "click B", "t2", "Markor editor")
            # Both pages identical -> same nodes reused, one new edge for B
            self.assertEqual(len(g.nodes), 2)
            self.assertEqual(len(g.edges), 2)

    def test_repeated_edge_increments_count(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main", "click A", "t1", "Markor editor")
            g.add_transition("Markor main", "click A", "t2", "Markor editor")
            self.assertEqual(len(g.edges), 1)
            self.assertEqual(g.edges[0].count, 2)
            # Task list accumulates
            self.assertIn("t1", g.edges[0].task)
            self.assertIn("t2", g.edges[0].task)

    def test_merge_threshold_boundary(self):
        # Pages with partial token overlap have cosine strictly between 0 and 1;
        # verify the 0.85 threshold decides merge vs new node (not just exact
        # string equality).  Computed with FakeEmbedder (bag-of-words, dim 64):
        #   "Markor main editor save"  vs "Markor main editor"      = 0.866 > 0.85 -> merge
        #   "Markor main editor save"  vs "Markor main editor back" = 0.750 < 0.85 -> new node
        #   "Markor main editor save"  vs "OsmAnd map navigation"   = 0.000 < 0.85 -> new node
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main editor save", "click A", "t1",
                             "Markor editor screen")
            # Just above threshold (0.866): merges into the existing node.
            g.add_transition("Markor main editor", "click B", "t2",
                             "Markor editor screen")
            # Just below threshold (0.750): partial overlap but new node.
            g.add_transition("Markor main editor back", "click C", "t3",
                             "Markor editor screen")
            # Disjoint (0.000): definitely a new node.
            g.add_transition("OsmAnd map navigation", "click D", "t4",
                             "Markor editor screen")

            summaries = {n.page_summary for n in g.nodes}
            # Original summary survives the merge (existing node is reused).
            self.assertIn("Markor main editor save", summaries)
            # "Markor main editor" merged (0.866 >= 0.85): no separate node.
            self.assertNotIn("Markor main editor", summaries)
            # 0.750 and 0.000 are both < 0.85: separate nodes.
            self.assertIn("Markor main editor back", summaries)
            self.assertIn("OsmAnd map navigation", summaries)
            # 4 distinct pages: save (merged), back, osmand, editor screen.
            self.assertEqual(len(g.nodes), 4)

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main", "click A", "t1", "Markor editor")
            g2 = PageGraph(persist_dir=d, embedder=FakeEmbedder())
            self.assertEqual(len(g2.nodes), 2)
            self.assertEqual(len(g2.edges), 1)


class PageGraphRetrievalTest(unittest.TestCase):

    def _graph(self, d: str) -> PageGraph:
        return PageGraph(persist_dir=d, embedder=FakeEmbedder())

    def test_empty_graph_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            self.assertEqual(g.retrieve_guidelines("any screen"), [])

    def test_retrieves_actions_and_tasks_around_similar_node(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main screen", "click new note",
                             "Create a note", "Markor editor screen")
            g.add_transition("Markor editor screen", "type content",
                             "Create a note", "Markor save screen")
            gl = g.retrieve_guidelines("Markor main screen")
            # BFS from the matched node should surface the outgoing action
            # and the achievable tasks.
            all_actions = [a for g_ in gl for a in g_["actions"]]
            self.assertIn("click new note", all_actions)
            all_tasks = [t for g_ in gl for t in g_["tasks"]]
            self.assertIn("Create a note", all_tasks)

    def test_disjoint_query_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            g.add_transition("Markor main screen", "click new note",
                             "Create a note", "Markor editor screen")
            gl = g.retrieve_guidelines("OsmAnd satellite maps navigation")
            self.assertEqual(gl, [])

    def test_bfs_layers_controls_multi_hop(self):
        # Build a genuine 2-hop chain A -> B -> C where only A is a seed.
        # Query = "Markor main screen"; FakeEmbedder cosines (computed):
        #   A "Markor main screen"  vs query = 1.0  -> seed
        #   B "OsmAnd map"          vs query = 0.0  -> NOT a seed (0.0 < 0.5)
        #   C "OsmAnd details"      vs query = 0.0  -> NOT a seed
        # So B and C are reachable ONLY by multi-hop BFS from A.
        # bfs_layers=1 finds A->B ("open editor") but not B->C ("tap marker");
        # bfs_layers=3 reaches C via B, surfacing "tap marker".
        with tempfile.TemporaryDirectory() as d:
            g = self._graph(d)
            # A -> B (Markor main screen -> OsmAnd map).
            g.add_transition("Markor main screen", "open editor", "t1",
                             "OsmAnd map")
            # B -> C (OsmAnd map -> OsmAnd details).
            g.add_transition("OsmAnd map", "tap marker", "t2",
                             "OsmAnd details")

            gl1 = g.retrieve_guidelines("Markor main screen", bfs_layers=1)
            actions1 = [a for g_ in gl1 for a in g_["actions"]]
            gl3 = g.retrieve_guidelines("Markor main screen", bfs_layers=3)
            actions3 = [a for g_ in gl3 for a in g_["actions"]]

            self.assertIn("open editor", actions1)
            self.assertNotIn("tap marker", actions1)
            self.assertIn("open editor", actions3)
            self.assertIn("tap marker", actions3)


class EnvKnowledgeLocalGraphTest(unittest.TestCase):

    def test_record_transition_then_retrieve(self):
        with tempfile.TemporaryDirectory() as d:
            ek = EnvKnowledge(rag_url="", persist_dir=d, embedder=FakeEmbedder())
            ek.record_transition(
                before_summary="Markor main screen",
                action_summary="click new note",
                task="Create a note",
                after_summary="Markor editor screen",
            )
            # Query must share enough tokens with the stored page for the
            # bag-of-words FakeEmbedder to clear the 0.5 hit threshold.  The
            # summary is "Current screen: app=net.gsantner.markor. page=Markor
            # main." (the goal is intentionally NOT embedded — task context
            # lives on the edge, not the node).  The app token is mostly noise
            # for this embedder, so the UI list is kept empty to avoid diluting
            # similarity against the stored "Markor main screen".
            hint = ek.retrieve_hint(
                "",
                current_app="net.gsantner.markor",
                current_page="Markor main",
            )
            self.assertIn("click new note", hint)

    def test_persistence_round_trip_via_envknowledge(self):
        with tempfile.TemporaryDirectory() as d:
            ek1 = EnvKnowledge(rag_url="", persist_dir=d, embedder=FakeEmbedder())
            ek1.record_transition(
                before_summary="Markor main screen",
                action_summary="click new note",
                task="Create a note",
                after_summary="Markor editor screen",
            )
            ek2 = EnvKnowledge(rag_url="", persist_dir=d, embedder=FakeEmbedder())
            hint = ek2.retrieve_hint(
                "",
                current_app="net.gsantner.markor",
                current_page="Markor main",
            )
            self.assertIn("click new note", hint)

    def test_empty_graph_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            ek = EnvKnowledge(rag_url="", persist_dir=d, embedder=FakeEmbedder())
            hint = ek.retrieve_hint("ui", current_app="a")
            self.assertEqual(hint, "")


class AgentU3FeedTest(unittest.TestCase):
    """Verify MemoryAugmentedAgent._on_step_complete feeds U3 on success."""

    def test_on_step_complete_feeds_u3(self):
        from android_world.agents.memory_agent import MemoryAugmentedAgent

        agent = MemoryAugmentedAgent.__new__(MemoryAugmentedAgent)
        agent.enable_u1 = False
        agent.enable_u2 = False
        agent.enable_u3 = True
        agent.u1 = None
        agent.u2 = None
        agent._current_goal = "Create a note"
        agent.u3 = mock.Mock()
        agent.u3.record_transition = mock.Mock()

        class _El:
            text = "New note"
            content_description = None
            hint_text = None
            package_name = "net.gsantner.markor"

        agent._on_step_complete({
            "before_ui_elements_list": "Markor main screen text",
            "before_ui_elements": [_El()],
            "after_ui_elements_list": "Markor editor screen text",
            "after_ui_elements": [],
            "action_output_json": mock.Mock(action_type="click", index=0),
        })

        agent.u3.record_transition.assert_called_once()
        call = agent.u3.record_transition.call_args
        self.assertEqual(call.kwargs.get("action_summary"), "clicked 'New note'")
        self.assertEqual(call.kwargs.get("task"), "Create a note")
        self.assertIn("New note", call.kwargs["before_summary"])
        # Node summaries must NOT contain the task goal (cross-task merging).
        self.assertNotIn("Create a note", call.kwargs["before_summary"])
        self.assertNotIn("Create a note", call.kwargs["after_summary"])

    def test_build_screen_summary_excludes_goal(self):
        s = build_screen_summary(
            "Markor main screen text",
            current_app="net.gsantner.markor",
            current_page="Markor main",
        )
        self.assertIn("net.gsantner.markor", s)
        self.assertIn("Markor main", s)
        self.assertNotIn("goal", s.lower())


class ActionEffectStrTest(unittest.TestCase):
    """Direct unit tests for _describe_element / _action_effect_str."""

    class _El:
        def __init__(self, text="", cd="", hint="", pkg="com.foo"):
            self.text = text or None
            self.content_description = cd or None
            self.hint_text = hint or None
            self.package_name = pkg or None

    def test_describe_prefers_text(self):
        el = self._El(text="New note", cd="create a note button", hint="tap to create")
        self.assertEqual(_describe_element([el], 0), "New note")

    def test_describe_uses_content_description_when_no_text(self):
        el = self._El(text="", cd="create a note button")
        self.assertEqual(_describe_element([el], 0), "create a note button")

    def test_describe_falls_back_to_package(self):
        el = self._El(text="", cd="", hint="", pkg="net.gsantner.markor")
        self.assertEqual(_describe_element([el], 0), "element in net.gsantner.markor")

    def test_describe_blank_element_returns_empty(self):
        el = self._El(text="", cd="", hint="", pkg="")
        self.assertEqual(_describe_element([el], 0), "")

    def test_describe_caps_at_40_chars(self):
        long_text = "x" * 80
        self.assertEqual(_describe_element([self._El(text=long_text)], 0), "x" * 40)

    def test_describe_index_zero_works(self):
        el = self._El(text="New note")
        self.assertEqual(_describe_element([el], 0), "New note")

    def test_describe_out_of_range_returns_empty(self):
        self.assertEqual(_describe_element([self._El(text="a")], 5), "")

    def test_describe_none_inputs_return_empty(self):
        self.assertEqual(_describe_element(None, 0), "")
        self.assertEqual(_describe_element([], None), "")

    def test_click_renders_semantic_label(self):
        action = mock.Mock(action_type="click", index=0)
        self.assertEqual(
            _action_effect_str(action, [self._El(text="Send email")]),
            "clicked 'Send email'",
        )

    def test_click_falls_back_to_index_when_undescribable(self):
        action = mock.Mock(action_type="click", index=3)
        self.assertEqual(_action_effect_str(action, []), "clicked 3")

    def test_input_text_renders_target_and_content(self):
        action = mock.Mock(action_type="input_text", index=0, text="hello")
        self.assertEqual(
            _action_effect_str(action, [self._El(text="Search field")]),
            "typed 'hello' into 'Search field'",
        )

    def test_input_text_without_target_renders_content_only(self):
        action = mock.Mock(action_type="input_text", index=1, text="hello")
        self.assertEqual(_action_effect_str(action, []), "typed 'hello'")

    def test_scroll_renders_direction_and_target(self):
        action = mock.Mock(action_type="scroll", index=0, direction="down")
        self.assertEqual(
            _action_effect_str(action, [self._El(text="Contacts")]),
            "scrolled down on 'Contacts'",
        )

    def test_scroll_without_target_renders_direction_only(self):
        action = mock.Mock(action_type="scroll", index=None, direction="up")
        self.assertEqual(_action_effect_str(action, []), "scrolled up")

    def test_long_press_renders_semantic_label(self):
        action = mock.Mock(action_type="long_press", index=0)
        self.assertEqual(
            _action_effect_str(action, [self._El(text="Edit")]),
            "long-pressed 'Edit'",
        )

    def test_non_index_actions_unchanged(self):
        self.assertEqual(_action_effect_str(None), "no action")
        self.assertEqual(
            _action_effect_str(mock.Mock(action_type="open_app", app_name="markor")),
            "opened markor",
        )
        self.assertEqual(
            _action_effect_str(mock.Mock(action_type="status", goal_status="complete")),
            "status -> complete",
        )
        self.assertEqual(
            _action_effect_str(mock.Mock(action_type="navigate_home")),
            "navigate home",
        )


if __name__ == "__main__":
    unittest.main()
