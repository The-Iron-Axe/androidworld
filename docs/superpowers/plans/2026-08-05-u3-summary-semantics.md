# U3 Node/Action Summary Semantics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the U3 page graph learn across tasks: page nodes no longer embed the task goal, and action summaries stop referencing per-screen UI indexes.

**Architecture:** Two changes. (1) `build_screen_summary` drops its `goal` param and the `Task goal:` line so a page's identity is pure screen state (app/page/UI dump) — same physical page under different tasks merges into one node. The query side (`retrieve_hint`) drops `goal` too so it shares the node's semantic space. (2) A new `_describe_element()` renders an index-based element into semantic text (text / content_description / hint_text, capped, never the index), and `_action_effect_str()` uses it so edge summaries like `clicked 5` become `clicked 'New note'`. Task info is preserved on `PageEdge.task`, and is surfaced by BFS just as before.

**Tech Stack:** Python 3, `android_world` (AndroidWorld), unittest, numpy.

**Test command:** `python -m unittest android_world.agents.memory.test_page_graph` from repo root `C:\Users\WRQ\Desktop\androidworld`.

---

## Files

- Modify: `android_world/agents/memory/environment.py` — `build_screen_summary` (drop goal), `EnvKnowledge.retrieve_hint` (drop goal param + call arg).
- Modify: `android_world/agents/memory_agent.py` — `_describe_element` (new), `_action_effect_str` (semantic output + optional ui_elements param), and three call sites (retrieve_hint, two build_screen_summary).
- Test: `android_world/agents/memory/test_page_graph.py` — update 4 tests, add 2 regression tests.

No change to `page_graph.py`'s graph structure or `EnvKnowledge.__init__` / `record_transition` signatures.

---

### Task 1: Drop `goal` from `build_screen_summary` and `retrieve_hint`

**Files:**
- Modify: `android_world/agents/memory/environment.py:36-59` (`build_screen_summary`)
- Modify: `android_world/agents/memory/environment.py:101-139` (`EnvKnowledge.retrieve_hint`)

- [ ] **Step 1: Update `build_screen_summary` signature + body**

Change:

```python
def build_screen_summary(
    goal: str,
    ui_elements_list: str,
    *,
    current_app: str = "",
    current_page: str = "",
    max_ui_chars: int = 1500,
) -> str:
  """Build a text screen summary S_It for RAG retrieve / graph nodes (no extra LLM call)."""
  parts: list[str] = []
  loc = []
  if current_app:
    loc.append(f"app={current_app}")
  if current_page:
    loc.append(f"page={current_page}")
  if loc:
    parts.append("Current screen: " + ", ".join(loc) + ".")
  parts.append(f"Task goal: {goal}")
  ui = (ui_elements_list or "").strip()
```

To:

```python
def build_screen_summary(
    ui_elements_list: str,
    *,
    current_app: str = "",
    current_page: str = "",
    max_ui_chars: int = 1500,
) -> str:
  """Build a text screen summary S_It for RAG retrieve / graph nodes (no extra LLM call).

  Page identity is pure screen state (app/page/UI dump) — the task goal is
  intentionally NOT part of a node's summary, so the same physical page under
  different tasks merges into one node (PG-Agent §3.1 node semantics).  Task
  context lives on the graph edge (PageEdge.task), not the node.
  """
  parts: list[str] = []
  loc = []
  if current_app:
    loc.append(f"app={current_app}")
  if current_page:
    loc.append(f"page={current_page}")
  if loc:
    parts.append("Current screen: " + ", ".join(loc) + ".")
  ui = (ui_elements_list or "").strip()
```

- [ ] **Step 2: Update `retrieve_hint` signature + body**

Change:

```python
  def retrieve_hint(
      self,
      goal: str,
      ui_elements_list: str,
      *,
      current_app: str = "",
      current_page: str = "",
  ) -> str:
```

to:

```python
  def retrieve_hint(
      self,
      ui_elements_list: str,
      *,
      current_app: str = "",
      current_page: str = "",
  ) -> str:
```

And inside, change the `summary = build_screen_summary(...)` call from:

```python
    summary = build_screen_summary(
        goal,
        ui_elements_list,
        current_app=current_app,
        current_page=current_page,
    )
```

to:

```python
    summary = build_screen_summary(
        ui_elements_list,
        current_app=current_app,
        current_page=current_page,
    )
```

- [ ] **Step 3: Update the local test helpers (the direct-arg tests now fail)**

Change `test_page_graph.py:184-204` — `EnvKnowledgeLocalGraphTest.test_record_transition_then_retrieve`:

```python
            hint = ek.retrieve_hint(
                "Markor main screen",
                "",
                current_app="net.gsantner.markor",
                current_page="Markor main",
            )
```

to:

```python
            hint = ek.retrieve_hint(
                "",
                current_app="net.gsantner.markor",
                current_page="Markor main",
            )
```

Change `test_page_graph.py:209` — `test_empty_graph_returns_empty`:

```python
            hint = ek.retrieve_hint("anything", "ui", current_app="a")
```

to:

```python
            hint = ek.retrieve_hint("ui", current_app="a")
```

- [ ] **Step 4: Run the tests to confirm the signature change is consistent**

Run: `python -m unittest android_world.agents.memory.test_page_graph -v`
Expected: PASS (all tests, including the two updated above). If anything still passes a positional `goal` to `build_screen_summary` / `retrieve_hint`, it fails here — fix before continuing.

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory/environment.py android_world/agents/memory/test_page_graph.py
git commit -m "feat(memory): drop goal from U3 node/query summaries"
```

---

### Task 2: Update the `memory_agent.py` call sites for the dropped `goal`

**Files:**
- Modify: `android_world/agents/memory_agent.py:21` (import line), `:133-140` (`_build_action_prompt`), `:184-187` (`_on_step_complete`)

- [ ] **Step 1: Remove the unused import**

`memory_agent.py:20-22` currently:

```python
from android_world.agents.memory.environment import (
    EnvKnowledge, build_screen_summary,
)
```

`build_screen_summary` is still used at the two `_on_step_complete` call sites (see Step 2), so this import stays as-is. No import change needed.

- [ ] **Step 2: Update the U3 retrieve call in `_build_action_prompt`**

`memory_agent.py:133-140` currently:

```python
      u3_text = self.u3.retrieve_hint(
          goal,
          ui_elements_list,
          current_app=app,
          current_page=page,
      )
```

Change to:

```python
      u3_text = self.u3.retrieve_hint(
          ui_elements_list,
          current_app=app,
          current_page=page,
      )
```

- [ ] **Step 3: Update the two `build_screen_summary` calls in `_on_step_complete`**

`memory_agent.py:184-187` currently:

```python
      before_summary = build_screen_summary(
          goal, before_list, current_app=before_app or "")
      after_summary = build_screen_summary(
          goal, after_list, current_app=after_app or "")
```

Change to:

```python
      before_summary = build_screen_summary(
          before_list, current_app=before_app or "")
      after_summary = build_screen_summary(
          after_list, current_app=after_app or "")
```

Note: the `goal = getattr(self, "_current_goal", "")` local at line 183 is still needed for the `record_transition(task=goal)` argument — do NOT delete it.

- [ ] **Step 4: Run the full memory test suite**

Run: `python -m unittest discover -s android_world/agents/memory -v`
Expected: PASS. Also `python -c "import android_world.agents.memory_agent"` imports cleanly (no leftover positional-goal calls).

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory_agent.py
git commit -m "feat(memory): drop goal from U3 call sites in agent"
```

---

### Task 3: Semantic action summaries via `_describe_element`

**Files:**
- Modify: `android_world/agents/memory_agent.py:34-53` (`_action_effect_str`)
- Modify: `android_world/agents/memory_agent.py:163-165` (U1 `_on_step_complete` call site)
- Modify: `android_world/agents/memory_agent.py:188-191` (U3 `_on_step_complete` call site)

- [ ] **Step 1: Add `_describe_element` and update `_action_effect_str`**

`memory_agent.py:34` currently starts with:

```python
def _action_effect_str(action: Any) -> str:
  """Terse effect summary of an action for U1 tracking."""
  if action is None:
    return "no action"
  at = getattr(action, "action_type", str(action))
  if at == "click":
    return f"clicked {getattr(action, 'index', '?')}"
  elif at == "input_text":
    return f"typed into {getattr(action, 'index', '?')}"
  elif at == "scroll":
    return f"scrolled {getattr(action, 'direction', '?')}"
  elif at == "open_app":
    return f"opened {getattr(action, 'app_name', '?')}"
  elif at in ("navigate_home", "navigate_back", "keyboard_enter", "wait"):
    return at.replace("_", " ")
  elif at == "long_press":
    return f"long-pressed {getattr(action, 'index', '?')}"
  elif at in ("status", "answer"):
    return f"{at} -> {getattr(action, 'goal_status', '') or getattr(action, 'text', '')}"
  return str(at)
```

Replace the whole function with:

```python
def _describe_element(ui_elements: Any, index: int | None) -> str:
  """Render an indexed UI element as semantic text (never its index).

  Priority: text > content_description > hint_text, capped at 40 chars.
  Falls back to the element's package when the element is missing or blank.
  """
  if ui_elements is None or index is None:
    return ""
  try:
    el = ui_elements[index]
  except (IndexError, TypeError):
    return ""
  for attr in ("text", "content_description", "hint_text"):
    val = getattr(el, attr, None) or ""
    val = str(val).strip()
    if val:
      return val[:40]
  pkg = getattr(el, "package_name", None) or ""
  pkg = str(pkg).strip()
  return f"element in {pkg}" if pkg else ""


def _action_effect_str(action: Any, ui_elements: Any = None) -> str:
  """Terse effect summary of an action for U1 tracking / U3 edge labels.

  Index-based actions render the target element semantically (text /
  content_description / hint_text) so summaries survive across episodes,
  screens, and tasks.  Falls back to the raw index only if the element
  cannot be described.
  """
  if action is None:
    return "no action"
  at = getattr(action, "action_type", str(action))
  index = getattr(action, "index", None)
  if at == "click":
    label = _describe_element(ui_elements, index)
    return f"clicked {label!r}" if label else f"clicked {index}"
  elif at == "input_text":
    text = getattr(action, "text", "")
    label = _describe_element(ui_elements, index)
    if label:
      return f"typed {text!r} into {label!r}"
    return f"typed {text!r}"
  elif at == "scroll":
    label = _describe_element(ui_elements, index)
    direction = getattr(action, "direction", "?")
    return f"scrolled {direction} on {label!r}" if label else f"scrolled {direction}"
  elif at == "open_app":
    return f"opened {getattr(action, 'app_name', '?')}"
  elif at in ("navigate_home", "navigate_back", "keyboard_enter", "wait"):
    return at.replace("_", " ")
  elif at == "long_press":
    label = _describe_element(ui_elements, index)
    return f"long-pressed {label!r}" if label else f"long-pressed {index}"
  elif at in ("status", "answer"):
    return f"{at} -> {getattr(action, 'goal_status', '') or getattr(action, 'text', '')}"
  return str(at)
```

- [ ] **Step 2: Pass `before_ui_elements` at the U1 call site**

`memory_agent.py:163-165` currently:

```python
      app, page = extract_app_from_elements(before_ui_elements)
      action = step_data.get("action_output_json")
      effect = _action_effect_str(action)
```

Change to:

```python
      app, page = extract_app_from_elements(before_ui_elements)
      action = step_data.get("action_output_json")
      effect = _action_effect_str(action, before_ui_elements)
```

- [ ] **Step 3: Pass `before_elements` at the U3 call site**

`memory_agent.py:188-191` currently:

```python
      self.u3.record_transition(
          before_summary=before_summary,
          action_summary=_action_effect_str(action),
          task=goal,
          after_summary=after_summary,
          before_app=before_app,
          after_app=after_app,
      )
```

Change the `action_summary=` line to:

```python
          action_summary=_action_effect_str(action, before_elements),
```

(`before_elements` is already bound at line 176.)

- [ ] **Step 4: Run the full memory test suite (expect the U3 feed assertion to fail)**

Run: `python -m unittest android_world.agents.memory.test_page_graph.AgentU3FeedTest -v`
Expected: FAIL — `test_on_step_complete_feeds_u3` asserts `action_summary == "clicked 3"` but the new code produces `"clicked ''"` (the mock element has no text).

- [ ] **Step 5: Commit**

```bash
git add android_world/agents/memory_agent.py
git commit -m "feat(memory): semantic action summaries in U3 edges"
```

---

### Task 4: Update and extend the U3 feed test

**Files:**
- Modify: `android_world/agents/memory/test_page_graph.py:213-240` (`AgentU3FeedTest`)
- Modify: `android_world/agents/memory/test_page_graph.py` (new `build_screen_summary` regression test)

- [ ] **Step 1: Update `test_on_step_complete_feeds_u3`**

Change `AgentU3FeedTest.test_on_step_complete_feeds_u3` (currently at `test_page_graph.py:216-240`) so the mock UI elements carry semantic text and the assertions check both the summary and the no-goal node summary. Replace the body:

```python
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

        agent._on_step_complete({
            "before_ui_elements_list": "Markor main screen text",
            "before_ui_elements": [],
            "after_ui_elements_list": "Markor editor screen text",
            "after_ui_elements": [],
            "action_output_json": mock.Mock(action_type="click", index=3),
        })

        agent.u3.record_transition.assert_called_once()
        call = agent.u3.record_transition.call_args
        self.assertEqual(call.kwargs.get("action_summary"), "clicked 3")
        self.assertEqual(call.kwargs.get("task"), "Create a note")
```

with:

```python
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
        # Node summaries must NOT contain the task goal (cross-task merging).
        self.assertNotIn("Create a note", call.kwargs["before_summary"])
        self.assertNotIn("Create a note", call.kwargs["after_summary"])
```

- [ ] **Step 2: Add a regression test for `build_screen_summary`**

Append a new method to `AgentU3FeedTest` (before the closing of the class):

```python
    def test_build_screen_summary_excludes_goal(self):
        s = build_screen_summary(
            "Markor main screen text",
            current_app="net.gsantner.markor",
            current_page="Markor main",
        )
        self.assertIn("net.gsantner.markor", s)
        self.assertIn("Markor main", s)
        self.assertNotIn("goal", s.lower())
```

- [ ] **Step 3: Run the full memory test suite**

Run: `python -m unittest discover -s android_world/agents/memory -v`
Expected: PASS — all tests, including the updated `test_on_step_complete_feeds_u3` (asserts `clicked 'New note'` and no goal in summaries) and the new `test_build_screen_summary_excludes_goal`.

- [ ] **Step 4: Commit**

```bash
git add android_world/agents/memory/test_page_graph.py
git commit -m "test(memory): U3 feed asserts semantic summary and goal-free nodes"
```

---

### Task 5: Final verification

- [ ] **Step 1: Run the entire memory test suite**

Run: `python -m unittest discover -s android_world/agents/memory -v`
Expected: PASS (all classes: PageGraphAddTest, PageGraphRetrievalTest, EnvKnowledgeLocalGraphTest, AgentU3FeedTest).

- [ ] **Step 2: Confirm no stale references to the old signatures**

Run: `python -c "import android_world.agents.memory_agent, android_world.agents.memory.environment; print('ok')"`
Expected: prints `ok` with no errors.

Run: `grep -rn "build_screen_summary(" android_world/agents/memory/ android_world/agents/memory_agent.py`
Expected: every call passes `ui_elements_list` as the first positional arg (no `goal` before it). Grep for `retrieve_hint(`:
- `memory_agent.py:126` — U2 `EpisodicMemory.retrieve_hint(goal)`. This is a **different** class (U2) with its own `goal`-first signature — leave it untouched.
- `memory_agent.py:133` — U3 `EnvKnowledge.retrieve_hint(ui_elements_list, ...)` — goal removed.
- `test_page_graph.py` — updated U3 calls.
- `episodic.py` — U2 only, untouched.

- [ ] **Step 3: Commit any stragglers (if Step 1/2 surfaced a missed call site, fix + commit)**

```bash
git add -A
git commit -m "fix(memory): resolve residual U3 signature call sites"
```

(If nothing to commit, `git status` will show a clean tree — skip this commit.)
