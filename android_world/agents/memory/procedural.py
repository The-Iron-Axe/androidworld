"""U4 Procedural Memory — facade over the skill library + mining pipeline.

U4 is pure data infrastructure: no LLM calls, no environment interaction.
It wraps SkillLibrary (storage) and skill_mining (abstraction) behind an
API aligned with U1/U2/U3 so the agent wires it identically.

Lifecycle:
  write:  add_successful_trajectory(goal, actions, precondition) buffers the
          trajectory; mine() abstracts the buffered successes into candidate
          skills (BPE + slot abstraction) and commits them.  Optionally
          validate_skills() replays a subset against a real verifier before
          commit (SkillWeaver-style testing) — see validation docs.
  read:   retrieve_hint(goal, precondition) returns a compact prompt hint.
  update: record_outcome(success) adjusts the matched skill's score; evicts
          when score <= 0 (ProcMEM gain-based pruning).
"""

from __future__ import annotations

from typing import Any

from android_world.agents.memory.skill import Skill, SkillLibrary
from android_world.agents.memory.skill_mining import mine_skills


class ProceduralMemory:
  """U4 procedural skill memory."""

  def __init__(
      self,
      persistence_dir: str = "",
      min_freq: int = 2,
      max_iters: int = 8,
  ):
    self.library = SkillLibrary(persist_dir=persistence_dir)
    self._persistence_dir = persistence_dir
    self.min_freq = min_freq
    self.max_iters = max_iters
    # Buffered successful trajectories awaiting the next mine().
    self._buffer: list[tuple[str, list, str]] = []  # (goal, actions, precondition)
    self._last_mined: int = 0

  # ── Writing: buffer + mine ─────────────────────────────────────────

  def add_successful_trajectory(
      self,
      goal: str,
      actions: list,
      precondition: str = "",
  ) -> None:
    """Buffer one successful trajectory (ground-truth confirmed) for mining.

    Only *successful* trajectories are buffered (validated experiences feed
    U4).  Trajectories with 1 action or fewer are skipped (atomic, mirroring
    U2's |τ|<=1 filter).
    """
    acts = [a for a in actions if a is not None]
    if len(acts) <= 1:
      return
    self._buffer.append((goal, acts, precondition))

  def mine(self) -> int:
    """Abstract buffered successful trajectories into skills; commit them.

    Returns the number of new skills added.  Candidate skills are committed
    only when they clear a structural sanity check (non-empty procedure).
    Retrieval scoring is delegated to SkillLibrary.
    """
    if not self._buffer:
      return 0
    trajectories = [acts for _, acts, _ in self._buffer]
    goal_hints = [g for g, _, _ in self._buffer]
    preconditions = [p for _, _, p in self._buffer]

    candidates = mine_skills(
        trajectories,
        goal_hints,
        preconditions,
        min_freq=self.min_freq,
        max_iters=self.max_iters,
    )
    added = 0
    for sk in candidates:
      if not sk.actions:
        continue
      self.library.add_skill(sk)
      added += 1
    self._last_mined = len(self._buffer)
    self._buffer.clear()
    return added

  # ── Retrieval ──────────────────────────────────────────────────────

  def _score_skill(self, skill: Skill, goal: str, precondition: str) -> float:
    """Dual-factor similarity of a skill to the query, aligned with U2.

    goal_hint and precondition are matched by exact string similarity here
    (fallback when no embedder is set); subclasses or callers can swap in an
    embedding backend.  Returns a value in [0, 1].
    """
    g = goal.lower().strip()
    gh = skill.goal_hint.lower().strip()
    goal_sim = 1.0 if g == gh else (0.0 if not g or not gh else _token_overlap(g, gh))
    p = precondition.lower().strip()
    ph = skill.precondition.lower().strip()
    pre_sim = 1.0 if p == ph else (0.5 if not p or not ph else _token_overlap(p, ph))
    return goal_sim * pre_sim

  def retrieve_hint(
      self,
      goal: str,
      precondition: str = "",
      k: int = 1,
  ) -> str:
    """Return a compact U4 skill hint for prompt injection, or "" on miss.

    Skills are scored by dual-factor similarity (goal_hint + precondition)
    mirroring U2's Plan(precondition, goal).  Returns at most k hints (default
    1, per ReasoningBank's k=1 finding).  Format is action-type summary with
    parameterized targets, stable across re-indexing.
    """
    skills = self.library.all()
    if not skills:
      return ""
    scored = [(self._score_skill(s, goal, precondition), s) for s in skills]
    scored.sort(key=lambda t: t[0], reverse=True)
    hits = [s for sc, s in scored if sc > 0.0][:k]
    if not hits:
      return ""
    parts = []
    for s in hits:
      parts.append(f"[Skill] {s.goal_hint}: " + s.action_types())
    return " | ".join(parts)

  # ── Update: outcome feedback ───────────────────────────────────────

  def record_outcome(
      self,
      goal: str,
      success: bool,
  ) -> None:
    """Update the best-matching skill's score from a real execution outcome.

    On success: score += 1; on failure: score -= 1.  Evict a skill whose
    score drops to <= 0 (ProcMEM gain-based pruning).
    """
    skills = self.library.all()
    if not skills:
      return
    scored = [(self._score_skill(s, goal, ""), s) for s in skills]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best = scored[0]
    if best_score <= 0.0:
      return
    if success:
      best.successes += 1
      best.score += 1.0
    else:
      best.failures += 1
      best.score -= 1.0
    if best.score <= 0.0:
      self.library.remove(best.goal_hint)
    else:
      self.library.add_skill(best)  # persist updated score

  # ── Persistence / stats ───────────────────────────────────────────

  def save(self, path: str | None = None) -> None:
    self.library.save(path)

  @property
  def size(self) -> int:
    return len(self.library.all())

  def stats(self) -> dict[str, Any]:
    skills = self.library.all()
    return {
        "skills": len(skills),
        "buffered": len(self._buffer),
        "last_mined": self._last_mined,
        "total_successes": sum(s.successes for s in skills),
        "total_failures": sum(s.failures for s in skills),
    }


def _token_overlap(a: str, b: str) -> float:
  """Tiny deterministic similarity: shared-word Jaccard in [0,1]."""
  if not a or not b:
    return 0.0
  sa, sb = set(a.split()), set(b.split())
  if not sa or not sb:
    return 0.0
  return len(sa & sb) / len(sa | sb)
