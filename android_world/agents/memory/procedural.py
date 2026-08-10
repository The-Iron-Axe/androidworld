"""U4 Procedural Memory — facade over the skill library + mining pipeline.

U4 is pure data infrastructure: no LLM calls, no environment interaction.
It wraps SkillLibrary (storage) and skill_mining (abstraction) behind an
API aligned with U1/U2/U3 so the agent wires it identically.

Lifecycle:
  write:  add_successful_trajectory(goal, actions, precondition) buffers the
          trajectory; add_failed_trajectory() buffers failed ones.  mine()
          abstracts the buffered successes into candidate positive skills AND
          the buffered failures into negative "avoid" skills (BPE + slot
          abstraction) and commits them.  Optionally validate_skills() replays
          a subset against a real verifier before commit (SkillWeaver-style
          testing) — see validation docs.
  read:   retrieve_hint(goal, precondition) returns a compact prompt hint.
          Positive skills ("do this") are returned first; a negative
          ("avoid this") hint is returned only when no positive skill matches,
          so the two never contradict each other in one prompt.
  update: record_outcome(success) adjusts the matched skill's score (of either
          kind); evicts when score <= 0 (ProcMEM gain-based pruning).
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
    # Buffered FAILED trajectories — mined into negative "avoid" skills.
    self._failed_buffer: list[tuple[str, list, str]] = []
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

  def add_failed_trajectory(
      self,
      goal: str,
      actions: list,
      precondition: str = "",
  ) -> None:
    """Buffer one FAILED trajectory for negative-skill mining.

    Failed trajectories are mined into negative "avoid" skills — reusable
    knowledge of what NOT to do — instead of being discarded.  This is the
    U4 channel through which failure experience reaches the decision loop.
    Trajectories with 1 action or fewer are skipped (atomic, mirroring the
    positive-skill filter).
    """
    acts = [a for a in actions if a is not None]
    if len(acts) <= 1:
      return
    self._failed_buffer.append((goal, acts, precondition))

  def mine(self) -> int:
    """Abstract buffered trajectories into skills; commit them.

    Successful trajectories become positive skills; failed ones become
    negative "avoid" skills (kind="negative").  Returns the number of new
    skills added (both kinds).  Candidate skills are committed only when they
    clear a structural sanity check (non-empty procedure).  Retrieval scoring
    is delegated to SkillLibrary.
    """
    added = 0
    if self._buffer:
      added += self._mine_batch(
          self._buffer, kind="positive", clear=True
      )
    if self._failed_buffer:
      added += self._mine_batch(
          self._failed_buffer, kind="negative", clear=True
      )
    return added

  def _mine_batch(
      self,
      buffer: list[tuple[str, list, str]],
      kind: str,
      clear: bool,
  ) -> int:
    """Mine one buffer batch (positive or negative) and commit the skills."""
    if not buffer:
      return 0
    trajectories = [acts for _, acts, _ in buffer]
    goal_hints = [g for g, _, _ in buffer]
    preconditions = [p for _, _, p in buffer]

    candidates = mine_skills(
        trajectories,
        goal_hints,
        preconditions,
        min_freq=self.min_freq,
        max_iters=self.max_iters,
        kind=kind,
    )
    added = 0
    for sk in candidates:
      if not sk.actions:
        continue
      self.library.add_skill(sk)
      added += 1
    self._last_mined = len(buffer)
    if clear:
      buffer.clear()
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

  def _top_skills(self, goal: str, precondition: str, kind: str) -> list[Skill]:
    """Best-matching skills of one kind, sorted by score desc."""
    skills = [s for s in self.library.all() if s.kind == kind]
    if not skills:
      return []
    scored = [(self._score_skill(s, goal, precondition), s) for s in skills]
    scored.sort(key=lambda t: t[0], reverse=True)
    return [s for sc, s in scored if sc > 0.0]

  def retrieve_hint(
      self,
      goal: str,
      precondition: str = "",
      k: int = 1,
  ) -> str:
    """Return a compact U4 skill hint for prompt injection, or "" on miss.

    Positive AND negative skills are retrieved together: a positive skill says
    "do it this way", a negative skill says "avoid this".  They are returned
    as separate sections so the prompt carries both what worked and what
    failed — failure experience is usable knowledge, not a fallback that is
    masked whenever a success exists.
    """
    blocks = self.retrieve_blocks(goal, precondition, k)
    return " | ".join(v for v in blocks.values() if v)

  def retrieve_blocks(
      self,
      goal: str,
      precondition: str = "",
      k: int = 1,
  ) -> dict[str, str]:
    """Retrieve positive and negative skill hints separately.

    Returns {"positive": ..., "negative": ...} — each either a formatted hint
    or "" when no skill of that kind matches.  The agent injects them as two
    independent prompt blocks so both "how to do it" and "what to avoid"
    reach the LLM at once.
    """
    positive = self._top_skills(goal, precondition, "positive")
    negative = self._top_skills(goal, precondition, "negative")
    return {
        "positive": (
            self._format_skills(positive[:k], negative=False) if positive else ""
        ),
        "negative": (
            self._format_skills(negative[:k], negative=True) if negative else ""
        ),
    }

  @staticmethod
  def _format_skills(skills: list[Skill], negative: bool) -> str:
    """Render matched skills as prompt text.

    Positive:  "[Skill] <goal_hint>: <action types>"
    Negative:  "[Avoid] <goal_hint>: <action types>"  (what NOT to do)
    """
    parts = []
    for s in skills:
      tag = "[Avoid]" if negative else "[Skill]"
      parts.append(f"{tag} {s.goal_hint}: " + s.action_types())
    return " | ".join(parts)

  # ── Update: outcome feedback ───────────────────────────────────────

  def record_outcome(
      self,
      goal: str,
      success: bool,
      strength: float = 1.0,
  ) -> None:
    """Update the best-matching skill's score from a real execution outcome.

    On success: score += strength; on failure: score -= strength.  `strength`
    in [0,1] lets the caller weight the credit by execution quality (e.g. the
    multi-agent Action Verifier pass rate), defaulting to full credit.  Evict
    a skill whose score drops to <= 0 (ProcMEM gain-based pruning).  The
    best-matching skill of EITHER kind is updated — a negative skill that gets
    followed and still fails should be penalized just like a positive one.
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
      best.score += strength
    else:
      best.failures += 1
      best.score -= strength
    if best.score <= 0.0:
      self.library.remove(best.goal_hint, best.kind)
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
        "positive": sum(1 for s in skills if s.kind == "positive"),
        "negative": sum(1 for s in skills if s.kind == "negative"),
        "buffered": len(self._buffer),
        "failed_buffered": len(self._failed_buffer),
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
