"""U4 Procedural Skill — a reusable, parameterized action program.

A skill is the unit of U4 procedural memory: a parameterized sequence of
semantic actions (text/content-description tokens, never element indices)
that abstracts a repeated sub-procedure across multiple successful
trajectories.

Skill fields (assessed against the 6 reference papers):
  goal_hint    — the task family this skill serves (retrieval key #1, aligned
                 with U2's goal).  From AWM's workflow goal description.
  precondition — the screen state the skill applies to (retrieval key #2,
                 aligned with U2's precondition; doubles as Mobile-Agent-E's
                 invocation gate).
  actions      — ordered parameterized semantic actions (the procedure).
                 ProcMEM's π_w; EAM action-group mining output.
  slots        — parameter names the actions reference as {slot}; the
                 instantiation interface (AWM variable abstraction).
  score        — accumulated execution outcome; drives retention/eviction
                 (ProcMEM gain-based score).

This is pure data infrastructure: no LLM calls, no environment interaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SkillAction:
    """One parameterized semantic action inside a skill's procedure.

    `target` is a semantic selector (element text / content_description /
    hint_text, or a symbolic step like open_app/app_name) — never an element
    index.  `params` carries action arguments; values may reference the
    skill's slots via {slot_name}.
    """
    action_type: str
    target: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    app: str = ""


@dataclass
class Skill:
    """A reusable procedural skill."""
    goal_hint: str
    precondition: str = ""
    actions: list[SkillAction] = field(default_factory=list)
    slots: list[str] = field(default_factory=list)
    score: float = 1.0
    successes: int = 0
    failures: int = 0
    version: int = 1

    @property
    def capacity(self) -> int:
        """Maximum skill-library size.  Kept on the class for cheap access."""
        return SkillLibrary.capacity

    def action_types(self) -> str:
        """Compact action-type summary for prompt hints (mirrors U2's
        _trajectory_action_string — drops targets so hints stay stable)."""
        return " → ".join(a.action_type for a in self.actions) if self.actions else ""

    def instantiate(self, params: dict[str, str]) -> list[SkillAction]:
        """Bind {slot} placeholders in params/targets to concrete values.

        Missing slots are left unbound (caller decides whether that is fatal).
        """
        out: list[SkillAction] = []
        for a in self.actions:
            bound_target = a.target
            for k, v in params.items():
                bound_target = bound_target.replace(f"{{{k}}}", v)
            bound_params: dict[str, Any] = {}
            for k, v in a.params.items():
                sv = v
                if isinstance(sv, str):
                    for k2, v2 in params.items():
                        sv = sv.replace(f"{{{k2}}}", v2)
                bound_params[k] = sv
            out.append(SkillAction(a.action_type, bound_target, bound_params, a.app))
        return out


class SkillLibrary:
    """Persistent store of skills, mirroring PageGraph's JSON persistence.

    Skills are indexed by id (stable sha1 of goal_hint) for dedup; a skill
    with the same goal_hint reuses the id and updates in place.
    """

    capacity = 100

    def __init__(self, persist_dir: str = ""):
        import hashlib
        self._hash = hashlib.sha1
        self.persist_dir = persist_dir
        self._skills: dict[str, Skill] = {}
        if persist_dir:
            import os
            os.makedirs(persist_dir, exist_ok=True)
            path = self._path()
            if os.path.exists(path):
                self._load(path)

    def _path(self) -> str:
        import os
        return os.path.join(self.persist_dir, "u4_skills.json")

    def _skill_id(self, goal_hint: str) -> str:
        return self._hash(goal_hint.encode("utf-8")).hexdigest()[:12]

    # ── CRUD ────────────────────────────────────────────────────────

    def add_skill(self, skill: Skill) -> str:
        """Insert or update a skill, keyed by sha1(goal_hint).  Returns id."""
        import os
        sid = self._skill_id(skill.goal_hint)
        self._skills[sid] = skill
        if self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.save()
        return sid

    def get(self, goal_hint: str) -> Skill | None:
        return self._skills.get(self._skill_id(goal_hint))

    def remove(self, goal_hint: str) -> bool:
        sid = self._skill_id(goal_hint)
        if sid in self._skills:
            del self._skills[sid]
            self.save()
            return True
        return False

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    # ── Persistence ─────────────────────────────────────────────────

    def save(self, path: str | None = None) -> None:
        import os
        p = path or (self._path() if self.persist_dir else "")
        if not p:
            return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        state = {
            "skills": [
                {
                    "goal_hint": s.goal_hint,
                    "precondition": s.precondition,
                    "actions": [
                        {"action_type": a.action_type, "target": a.target,
                         "params": a.params, "app": a.app}
                        for a in s.actions
                    ],
                    "slots": s.slots,
                    "score": s.score,
                    "successes": s.successes,
                    "failures": s.failures,
                    "version": s.version,
                }
                for s in self._skills.values()
            ],
        }
        with open(p, "w", encoding="utf-8") as f:
            import json
            json.dump(state, f, ensure_ascii=False, indent=2)

    def _load(self, path: str) -> None:
        import json
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        for sd in state.get("skills", []):
            sid = self._skill_id(sd["goal_hint"])
            self._skills[sid] = Skill(
                goal_hint=sd["goal_hint"],
                precondition=sd.get("precondition", ""),
                actions=[
                    SkillAction(**a) for a in sd.get("actions", [])
                ],
                slots=sd.get("slots", []),
                score=sd.get("score", 1.0),
                successes=sd.get("successes", 0),
                failures=sd.get("failures", 0),
                version=sd.get("version", 1),
            )
