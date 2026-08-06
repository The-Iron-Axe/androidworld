"""U4 procedural-skill mining — deterministic abstraction from trajectories.

This module implements the U4 *abstraction* step: given a batch of successful
trajectories (as semantic action lists), extract recurring parameterized
sub-procedures and materialize them as Skill objects.  Everything here is
deterministic — no LLM calls, no environment interaction — so U4 stays a
pure data layer (and the ablation flag --u4 is a clean independent variable).

Pipeline (mirrors the cross-paper synthesis):
  1. Slot extraction      — values that differ at the same position across
                            trajectories become {slot} placeholders (AWM).
  2. Semantic tokenization— each action is rendered to a semantic token
                            (action_type + target text / content_description /
                            hint_text / app), never an element index, so skills
                            survive re-indexing (matches the repo's rule that
                            indices are stale on re-execution).
  3. BPE-style mining     — iteratively merge the most frequent adjacent
                            token pair into a single higher-level step when it
                            clears a frequency threshold (EAM Action Group
                            Mining).
  4. Skill materialization— an abstracted token sequence is converted into a
                            Skill with parameterized actions, goal_hint and
                            precondition drawn from the seed trajectories.

Retrieval/retention are NOT here — they live in ProceduralMemory
(procedural.py).
"""

from __future__ import annotations

from collections import Counter, defaultdict

from android_world.agents.memory.skill import Skill, SkillAction


def _element_semantics(el) -> str:
    """Semantic label of a UI element: text > content_description > hint_text."""
    for attr in ("text", "content_description", "hint_text"):
        val = getattr(el, attr, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def semantic_token(action) -> str:
    """Render an action to a stable semantic token (no element indices).

    The token is the *identity* used for cross-trajectory mining: two actions
    that act on the same semantic target collapse to the same token even when
    their raw element indices differ.

    The semantic target comes from `_semantic_target` (set by the agent when
    it binds an action to its UI element, see memory_agent._flush_u4_trajectory)
    or from `element` (used in tests).  If neither is present the token is a
    bare action type.
    """
    if action is None:
        return ""
    at = getattr(action, "action_type", str(action))
    if at in ("click", "double_tap", "long_press"):
        text = getattr(action, "_semantic_target", None) or _element_semantics(
            getattr(action, "element", None)
        )
        return f"{at}@{text}" if text else at
    if at == "input_text":
        text = getattr(action, "text", "")
        target = getattr(action, "_semantic_target", None) or _element_semantics(
            getattr(action, "element", None)
        )
        tag = f"@{target}" if target else ""
        return f"input_text{tag}"
    if at == "scroll":
        direction = getattr(action, "direction", "?")
        return f"scroll:{direction}"
    if at == "open_app":
        app = getattr(action, "app_name", None)
        return f"open_app({app})" if app else "open_app"
    if at in ("navigate_home", "navigate_back", "keyboard_enter", "wait"):
        return at
    if at in ("status", "answer"):
        text = getattr(action, "goal_status", "") or getattr(action, "text", "")
        return f"{at}:{text}" if text else at
    return at


def tokenize_trajectory(actions: list) -> list[str]:
    """Convert an action list into a list of semantic tokens (skips None)."""
    out = []
    for a in actions:
        t = semantic_token(a)
        if t:
            out.append(t)
    return out


def _extract_slots(trajectories: list[list[str]]) -> dict[str, str]:
    """Map positions where values vary across trajectories to {slot} names.

    Given aligned token lists (same action_type at the same position), a
    position whose concrete value differs is abstracted to a slot.  Slot
    names are generated deterministically (slot_0, slot_1, ...) so the same
    alignment always produces the same names.

    Returns {position -> slot_name} keyed by token-list index.
    """
    n = max((len(t) for t in trajectories), default=0)
    slots: dict[int, str] = {}
    for pos in range(n):
        values = {t[pos] for t in trajectories if len(t) > pos}
        if len(values) > 1:
            slots[pos] = f"slot_{len(slots)}"
    return slots


def _abstract_tokens(tokens: list[str], slots: dict[int, str]) -> list[str]:
    """Replace a varying concrete value at an aligned position with {slot}.

    Only the *value* is abstracted — the action part of the token is kept, so
    'input_text@Recipient A' becomes 'input_text@{slot_0}' (the varying
    target becomes a slot reference; the action type survives).
    """
    out: list[str] = []
    for pos, tok in enumerate(tokens):
        if pos in slots:
            # Token may carry a value after '@' (click@Text / input_text@Field).
            if "@" in tok:
                action, sep, _ = tok.rpartition("@")
                out.append(f"{action}{sep}{{{slots[pos]}}}")
            else:
                out.append(f"{{{slots[pos]}}}")
        else:
            out.append(tok)
    return out


def _abstract_token(tok: str, slot_name: str) -> str:
    """Abstract the value portion of one token to {slot_name}."""
    if "@" in tok:
        action, sep, _ = tok.rpartition("@")
        return f"{action}{sep}{{{slot_name}}}"
    return f"{{{slot_name}}}"


def bpe_merge(
    trajectories: list[list[str]],
    min_freq: int = 2,
    max_iters: int = 8,
) -> list[list[tuple[str, ...]]]:
    """BPE-style mining: merge frequent adjacent token pairs (EAM Action Group
    Mining).

    Groups are represented as tuples of leaf tokens throughout (never
    string-joined), so nested merges stay unambiguous even when leaf targets
    contain commas or parens.  At each iteration, count every adjacent group
    pair across all trajectories; merge the globally most frequent pair (ties
    broken lexicographically) whenever its total frequency >= min_freq.

    Returns one list of groups per input trajectory; a group is a tuple of
    leaf tokens, so a single unmerged token is a 1-tuple.
    """
    merged = [[(t,) for t in traj] for traj in trajectories]
    for _ in range(max_iters):
        pair_counts: Counter[tuple[tuple[str, ...], tuple[str, ...]]] = Counter()
        for traj in merged:
            for i in range(len(traj) - 1):
                pair_counts[(traj[i], traj[i + 1])] += 1
        if not pair_counts:
            break
        best_pair, best_count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
        if best_count < min_freq:
            break
        g1, g2 = best_pair
        merged = [_replace_group(traj, g1, g2, g1 + g2) for traj in merged]
    return merged


def _replace_group(
    traj: list[tuple[str, ...]],
    g1: tuple[str, ...],
    g2: tuple[str, ...],
    new_group: tuple[str, ...],
) -> list[tuple[str, ...]]:
    out: list[tuple[str, ...]] = []
    i = 0
    while i < len(traj):
        if i + 1 < len(traj) and traj[i] == g1 and traj[i + 1] == g2:
            out.append(new_group)
            i += 2
        else:
            out.append(traj[i])
            i += 1
    return out


def _flatten_groups(groups: list[list[tuple[str, ...]]]) -> list[list[str]]:
    """Collapse group lists back into flat leaf-token lists."""
    return [[leaf for g in traj for leaf in g] for traj in groups]


def _skill_from_token_list(
    tokens: list[str],
    goal_hint: str,
    precondition: str,
    slots: list[str],
    kind: str = "positive",
) -> Skill:
    """Materialize a flat leaf-token list into a parameterized Skill.

    Slots may already be substituted into a token's target (e.g.
    'input_text@{slot_0}'); `_leaf_to_action` carries them into the action.
    """
    actions = [_leaf_to_action(tok, slots) for tok in tokens]
    return Skill(
        goal_hint=goal_hint,
        precondition=precondition,
        actions=actions,
        slots=slots,
        score=1.0,
        kind=kind,
    )


def _leaf_to_action(leaf: str, slots: list[str]) -> SkillAction:
    """Map a leaf semantic token back to a SkillAction.

    Handles the formats emitted by semantic_token(): 'click@target',
    'input_text@target', 'scroll:up', 'open_app(app)', bare types, and any
    {slot} placeholders already substituted into the target.
    """
    if leaf.startswith("click@"):
        return SkillAction(action_type="click", target=leaf[len("click@"):])
    if leaf.startswith("input_text@"):
        return SkillAction(action_type="input_text", target=leaf[len("input_text@"):])
    if leaf.startswith("scroll:"):
        return SkillAction(action_type="scroll", target="", params={"direction": leaf[len("scroll:"):]})
    if leaf.startswith("open_app(") and leaf.endswith(")"):
        app = leaf[len("open_app("):-1]
        return SkillAction(action_type="open_app", target="", app=app)
    if leaf.startswith("status:") or leaf.startswith("answer:"):
        at, _, val = leaf.partition(":")
        return SkillAction(action_type=at, params={"text": val})
    # Bare action type or a {slot} in target position.
    if leaf.startswith("{") and leaf.endswith("}"):
        return SkillAction(action_type="click", target=leaf)
    return SkillAction(action_type=leaf)


def _abstract_goal_hint(goal: str) -> str:
    """Abstract a concrete task goal into a reusable task-category hint.

    Removes parameter-like tokens (file names, dates, timestamps, numbers,
    quoted strings) so a skill is retrievable for the whole task family, not
    one specific instance.  Examples:
      "Delete the note in Markor named bold_king_edited."
        -> "Delete the note in Markor named X."
      "Create a new folder in Markor named folder_20260806_143035."
        -> "Create a new folder in Markor named X."
    """
    import re
    text = (goal or "").strip()
    # Replace quoted strings with X.
    text = re.sub(r"['\"]([^'\"]+)['\"]", "'X'", text)
    # Replace timestamp-like / hash-like / numeric tokens with X.
    text = re.sub(r"\b[\w]+_\d{6,}\b", "X", text)  # folder_20260806_143035
    text = re.sub(r"\b\d{4}[-_]\d{1,2}[-_]\d{1,2}\b", "X", text)  # dates
    text = re.sub(r"\b\d+:\d+\b", "X", text)  # times
    # Replace words ending in _edited / _copy / trailing digit-suffixed names.
    text = re.sub(r"\b[\w]+_edited\b", "X", text)
    text = re.sub(r"\b[\w]+_copy\b", "X", text)
    # Anything that looks like a concrete value after "named/with title/for X"
    text = re.sub(r"named [\w.-]+", "named X", text, flags=re.IGNORECASE)
    text = re.sub(r"title [\w.-]+", "title X", text, flags=re.IGNORECASE)
    return text.strip()


def mine_skills(
    trajectories: list[list],
    goal_hints: list[str],
    preconditions: list[str],
    min_freq: int = 2,
    max_iters: int = 8,
    kind: str = "positive",
) -> list[Skill]:
    """Top-level abstraction entry: trajectories -> candidate skills.

    Args:
      trajectories: list of action lists (each a successful trajectory).
      goal_hints:   one goal hint per trajectory (task family label).  Each is
                    abstracted via `_abstract_goal_hint` so skills are keyed by
                    task category, not one concrete instance.
      preconditions: one starting-screen precondition per trajectory.
      min_freq:     minimum total occurrences of a token pair to merge (BPE).
      max_iters:    maximum BPE merge iterations.
      kind:         "positive" for skills mined from successful trajectories,
                    "negative" for avoidance skills mined from failed ones.
                    Only the skill's label changes — the abstraction pipeline
                    (slot extraction + BPE) is identical for both.

    Returns candidate Skill objects (not yet validated/committed).  The first
    trajectory's goal_hint/precondition seed the skill; cross-trajectory
    agreement is the caller's concern.
    """
    token_lists = [tokenize_trajectory(acts) for acts in trajectories]
    merged = bpe_merge(token_lists, min_freq=min_freq, max_iters=max_iters)
    slots = _extract_slots(token_lists)
    # Flatten the seed trajectory's merged groups back to leaf order (merging
    # only concatenates adjacent leaves, so order is preserved), then abstract
    # varying values to {slot} by original position.
    seed_leaves = [leaf for g in merged[0] for leaf in g] if merged else []
    abstracted = _abstract_tokens(seed_leaves, slots)
    return [
        _skill_from_token_list(
            abstracted,
            goal_hint=_abstract_goal_hint(goal_hints[0]),
            precondition=preconditions[0] if preconditions else "",
            slots=list(slots.values()),
            kind=kind,
        )
    ]
