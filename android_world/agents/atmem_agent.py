"""ATMem Agent — Active Task-Driving Memory for AndroidWorld.

Based on "What Memory Do GUI Agents Really Need? From Passive Records to Active
Task-Driving States" (Liu et al., arXiv:2606.31612).

This agent implements a Planner-Actor-Memory architecture without any model
training — ATMem is maintained purely through prompt-driven structured output.

Architecture
------------
    ATMemAgent (EnvironmentInteractingAgent)
    ├── Planner → decomposes goal → subgoals + initial ATMem
    ├── M3A-based Actor → executes one action per step()
    └── ATMem state → cross-step structured memory, updated by Actor
"""

from __future__ import annotations

import time
from typing import Any, Optional

from absl import logging

from android_world.agents import agent_utils
from android_world.agents import atmem_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a as m3a_lib
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action


# ---------------------------------------------------------------------------
# Action-selection prompt augmented with ATMem slot
# ---------------------------------------------------------------------------

_ATMEM_ACTION_PROMPT_TEMPLATE = m3a_lib.ACTION_SELECTION_PROMPT_TEMPLATE + """
## ATMem — Active Task Memory

Current ATMem state:
{atmem_state}

After selecting your action, append an <atmem> block to update the memory:
- If you observed new task-relevant data: add items with status="remaining".
- If you completed an operation on an item: update its status to "finished".
- If an item does not satisfy constraints: set status="skipped" with a brief skipReason.
- If no memory update is needed: output {{}} inside the <atmem> block.

Output format:
Reason: ...\\nAction: {{"action_type":...}}\\n<atmem>{{... or {{}} }}</atmem>
"""


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------


class ATMemAgent(base_agent.EnvironmentInteractingAgent):
    """ATMem-augmented agent for AndroidWorld."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "ATMemAgent",
        wait_after_action_seconds: float = 2.0,
    ):
        super().__init__(env, name)
        self.llm = llm
        self.wait_after_action_seconds = wait_after_action_seconds

        # Per-task state (reset on each new task)
        self._goal: str = ""
        self._subgoals: list[str] = []
        self._subgoal_idx: int = 0
        self._atmem: atmem_utils.ATMem = atmem_utils.empty_atmem()
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.env.hide_automation_ui()
        self._goal = ""
        self._subgoals = []
        self._subgoal_idx = 0
        self._atmem = atmem_utils.empty_atmem()
        self._history = []

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------

    def _run_planner(self, goal: str) -> tuple[list[str], atmem_utils.ATMem]:
        """Decompose goal into subgoals and construct initial ATMem."""
        state = self.get_post_transition_state()
        screenshot = state.pixels.copy()

        prompt = atmem_utils.PLANNER_PROMPT_TEMPLATE.format(
            goal=goal,
            atmem_schema="""{
    "phase": "harvest" | "execute",
    "remainingFiles": [],
    "completedFiles": [],
    "constraints": {"logic": "AND|OR", "conditions": [{"field": "...", "op": "eq|neq|contains", "value": "..."}]},
    "schema": {"fields": ["attr1", "attr2", ...], "locked": false},
    "items": {"1": {"content": {...}, "status": "remaining|finished|skipped", "skipReason": ""}}
}"""
        )

        text, _, raw = self.llm.predict_mm(prompt, [screenshot])
        if not raw or text == infer.ERROR_CALLING_LLM:
            logging.warning("Planner LLM call failed, using single-goal fallback.")
            return [goal], atmem_utils.empty_atmem()

        subgoals = atmem_utils.parse_planner_subgoals(text)
        if not subgoals:
            subgoals = [goal]

        atmem = atmem_utils.parse_atmem_block(text) or atmem_utils.empty_atmem()

        logging.info("Planner: %d subgoals, ATMem phase=%s items=%d",
                     len(subgoals), atmem.get("phase", "?"), len(atmem.get("items", {})))
        return subgoals, atmem

    # ------------------------------------------------------------------
    # History formatting
    # ------------------------------------------------------------------

    def _history_for_prompt(self) -> str:
        if not self._history:
            return "You just started, no action has been performed yet."
        lines = []
        for i, step in enumerate(self._history):
            summary = step.get("summary", "(no summary)")
            full_atmem = (step.get("atmem") or {}).get("full", {})
            atmem_note = atmem_utils.atmem_summary_for_history(full_atmem)
            lines.append(f"Step {i + 1}- {summary} {atmem_note}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main step()
    # ------------------------------------------------------------------

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data: dict[str, Any] = {
            "raw_screenshot": None,
            "before_screenshot_with_som": None,
            "before_ui_elements": [],
            "after_screenshot_with_som": None,
            "action_prompt": None,
            "action_output": None,
            "action_output_json": None,
            "action_reason": None,
            "action_raw_response": None,
            "summary_prompt": None,
            "summary": None,
            "summary_raw_response": None,
            "atmem": None,
        }

        # ── Phase 0: Plan once at the start of the task ──────────────
        if not self._subgoals:
            self._goal = goal
            self._subgoals, self._atmem = self._run_planner(goal)

        current_subgoal = (
            self._subgoals[self._subgoal_idx]
            if self._subgoal_idx < len(self._subgoals)
            else "Complete the task and verify."
        )

        subgoal_note = f"\nCurrent subgoal ({self._subgoal_idx + 1}/{len(self._subgoals)}): {current_subgoal}"
        if self._subgoal_idx > 0:
            completed = "; ".join(self._subgoals[i] for i in range(self._subgoal_idx))
            subgoal_note += f"\nCompleted: {completed}"
        if self._subgoal_idx + 1 < len(self._subgoals):
            pending = "; ".join(self._subgoals[i] for i in range(self._subgoal_idx + 1, len(self._subgoals)))
            subgoal_note += f"\nRemaining: {pending}"

        # ── Phase 1: Get screen state ────────────────────────────────
        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        before_ui_elements = state.ui_elements
        step_data["before_ui_elements"] = before_ui_elements
        before_ui_elements_list = m3a_lib._generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )

        step_data["raw_screenshot"] = state.pixels.copy()
        before_screenshot = state.pixels.copy()
        for index, ui_element in enumerate(before_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    before_screenshot, ui_element, index,
                    logical_screen_size, physical_frame_boundary, orientation,
                )
        step_data["before_screenshot_with_som"] = before_screenshot.copy()

        # ── Phase 2: Build ATMem-augmented prompt ────────────────────
        atmem_str = atmem_utils.format_atmem_for_prompt(self._atmem)
        action_prompt = _ATMEM_ACTION_PROMPT_TEMPLATE.format(
            goal=goal,
            history=self._history_for_prompt(),
            ui_elements=before_ui_elements_list if before_ui_elements_list else "Not available",
            additional_guidelines=(
                f"Additional task-specific guidance:\n{subgoal_note}\n"
                if subgoal_note else ""
            ),
            atmem_state=atmem_str,
        )
        step_data["action_prompt"] = action_prompt

        # ── Phase 3: Call LLM ────────────────────────────────────────
        action_output, is_safe, raw_response = self.llm.predict_mm(
            action_prompt,
            [
                step_data["raw_screenshot"],
                before_screenshot,
            ],
        )
        if is_safe is False:
            action_output = (
                f'Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}\n'
                'Action: {"action_type": "status", "goal_status": "infeasible"}\n'
                '<atmem>{}</atmem>'
            )
        if not raw_response:
            raise RuntimeError("Error calling LLM in action selection phase.")

        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response

        # ── Phase 4: Parse ATMem update ──────────────────────────────
        new_atmem = atmem_utils.parse_atmem_block(action_output)
        # Merge new ATMem into existing state
        self._atmem = atmem_utils.merge_atmem(self._atmem, new_atmem)
        step_data["atmem"] = {
            "phase": self._atmem.get("phase"),
            "items_summary": atmem_utils.atmem_summary_for_history(self._atmem),
            "full": dict(self._atmem),
        }

        # ── Phase 5: Parse action ────────────────────────────────────
        reason, action = m3a_utils.parse_reason_action_output(action_output)
        if (not reason) or (not action):
            logging.info("Action prompt output is not in the correct format.")
            step_data["summary"] = (
                "Output for action selection is not in the correct format."
            )
            self._history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        logging.info("Action: %s", action)
        logging.info("Reason: %s", reason)
        logging.info("ATMem: %s", atmem_utils.atmem_summary_for_history(self._atmem))
        step_data["action_reason"] = reason

        try:
            converted_action = json_action.JSONAction(
                **agent_utils.extract_json(action),
            )
            step_data["action_output_json"] = converted_action
        except Exception as e:
            logging.info("Failed to convert output to valid action: %s", e)
            step_data["summary"] = (
                "Cannot parse the output to a valid action."
            )
            self._history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        # ── Phase 6: Validate index ──────────────────────────────────
        action_index = converted_action.index
        if (
            converted_action.action_type
            in ("click", "long_press", "input_text", "scroll")
            and action_index is not None
            and action_index >= len(before_ui_elements)
        ):
            logging.info(
                "Index out of range: %d >= %d", action_index, len(before_ui_elements)
            )
            step_data["summary"] = "Index out of range."
            self._history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        # ── Phase 7: Check for terminal action ────────────────────────
        if converted_action.action_type == "status":
            if converted_action.goal_status == "infeasible":
                logging.info("Agent declares task infeasible.")
            step_data["summary"] = "Agent thinks the task has been completed."
            self._history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        if converted_action.action_type == "answer":
            logging.info("Agent answered: %s", converted_action.text)

        # ── Phase 8: Execute action ──────────────────────────────────
        try:
            self.env.execute_action(converted_action)
        except Exception as e:
            logging.info("Failed to execute action: %s", e)
            step_data["summary"] = f"Execution error: {e}"
            self._history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        time.sleep(self.wait_after_action_seconds)

        # ── Phase 9: After-action capture + summary ──────────────────
        after_state = self.env.get_state(wait_to_stabilize=False)
        after_ui_elements_list = m3a_lib._generate_ui_elements_description_list(
            after_state.ui_elements, logical_screen_size
        )
        after_screenshot = after_state.pixels.copy()
        for index, ui_element in enumerate(after_state.ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    after_screenshot, ui_element, index,
                    logical_screen_size, physical_frame_boundary, orientation,
                )

        m3a_utils.add_screenshot_label(
            step_data["before_screenshot_with_som"], "before"
        )
        m3a_utils.add_screenshot_label(after_screenshot, "after")
        step_data["after_screenshot_with_som"] = after_screenshot.copy()

        summary_prompt = m3a_lib._summarize_prompt(
            action, reason, goal,
            before_ui_elements_list, after_ui_elements_list,
        )
        step_data["summary_prompt"] = summary_prompt

        summary, is_safe, summary_raw = self.llm.predict_mm(
            summary_prompt,
            [before_screenshot, after_screenshot],
        )
        if is_safe is False:
            summary = "Summary triggered LLM safety classifier."
        if not summary_raw:
            step_data["summary"] = f"LLM error during summarization: {summary}"
            self._history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        step_data["summary"] = f"Action selected: {action}. {summary}"
        step_data["summary_raw_response"] = summary_raw

        self._history.append(step_data)

        # ── Phase 10: Subgoal advancement heuristic ──────────────────
        # If schema items are all finished and phase is execute, advance
        items = self._atmem.get("items", {})
        if items and self._atmem.get("phase") == "execute":
            all_done = all(
                v.get("status") in ("finished", "skipped") for v in items.values()
            )
            if all_done and self._subgoal_idx + 1 < len(self._subgoals):
                self._subgoal_idx += 1
                logging.info(
                    "All items processed. Advancing to subgoal %d/%d.",
                    self._subgoal_idx + 1, len(self._subgoals),
                )

        return base_agent.AgentInteractionResult(False, step_data)

    # ------------------------------------------------------------------
    # Persistence (for research reproducibility)
    # ------------------------------------------------------------------

    def get_atmem_state(self) -> dict[str, Any]:
        """Return current ATMem + planner state for logging."""
        return {
            "goal": self._goal,
            "subgoals": self._subgoals,
            "subgoal_idx": self._subgoal_idx,
            "atmem": self._atmem,
            "history_len": len(self._history),
        }
