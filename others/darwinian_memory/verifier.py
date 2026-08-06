"""
Darwinian Memory System — Verification Mechanism (Appendix D)
==============================================================
Implements the K-Verification policy that guards memory reliability.

Key insight (Appendix C): system purity is bounded by
    Q_ss = 1 / (1 + R_fail · R_ver)

where R_ver = P_FN / P_TP.  By requiring K consecutive failure strikes
before deletion, the effective false-negative rate decays exponentially:
    P_FN^effective ≈ (P_FN)^K

The Verifier:
  1. Analyzes execution history (PRIMARY source of truth)
  2. Checks screenshot for contradictions (SECONDARY check)
  3. Defaults to success if history is sound and no contradiction found

The prompt in Figure 9 (Appendix D.2) is reproduced directly from the paper.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from .config import DMSConfig, default_config
from .memory_entry import MemoryEntry


# ═══════════════════════════════════════════════════════════════════════
# Verifier Prompt — directly from Figure 9 in the paper
# ═══════════════════════════════════════════════════════════════════════

VERIFIER_SYSTEM_PROMPT = """Role: You are an expert Android Task Verifier. Your job is to determine if the agent's execution history successfully achieved the user's goal.

Input Information:
1. Original Goal: The user's original objective.
2. Execution History: The (Thought, Code) steps the agent claims it just performed. This is your PRIMARY source of truth.
3. Final Screenshot: The ground truth screenshot. This is your SECONDARY check for contradictions.

YOUR VERIFICATION LOGIC (History-First):
1. Analyze History (Trust): Read the Execution History. Did the agent perform the logical actions required to complete the Original Goal? (e.g., for "Save recording," did the agent tap('Save')?)
2. Assume Success: If the history looks correct, your default verdict is {"verified success": true}.
3. Visual Veto (Contradiction Check): Now, look at the Final Screenshot. Does this screenshot explicitly contradict the agent's claim of success?
  • Contradiction (→Fail): The screenshot shows an error message (e.g., "Password incorrect").
  • Contradiction (→Fail): The screenshot shows the agent is in the wrong application.
  • Contradiction (→Fail): The goal was "Dismiss the 'OK' dialog," but the screenshot clearly shows the 'OK' dialog is still visible.
  • NO Contradiction (→Success): The goal was "Dismiss the 'OK' dialog," and the screenshot shows the dialog is gone. This confirms the history.
  • NO Contradiction (→Success): The goal was "Click the 'Save' button," and the screenshot shows the app has moved to a different screen. This confirms the history.

Key Rule: You must default to True (success) if the history is sound AND the screenshot does not provide strong, undeniable proof of failure.

Output Format: Respond ONLY with the JSON object: {"verified_success": <bool>, "reason": "<string>"}"""


# ═══════════════════════════════════════════════════════════════════════
# Verification result
# ═══════════════════════════════════════════════════════════════════════

class VerificationResult:
    """Result of a single verification check."""
    def __init__(self, success: bool, reason: str = ""):
        self.success = success
        self.reason = reason

    def __bool__(self) -> bool:
        return self.success

    def __repr__(self) -> str:
        status = "PASS" if self.success else "FAIL"
        return f"VerificationResult({status}, reason={self.reason!r})"


# ═══════════════════════════════════════════════════════════════════════
# K-Verification Policy (Appendix D)
# ═══════════════════════════════════════════════════════════════════════

class KVerifier:
    """Implements the K-Verification policy from Appendix D.

    A memory is only permanently pruned after K consecutive verification
    failures.  This exponentially reduces the false-negative rate:
        P_FN^effective ≈ (P_FN)^K

    The paper uses K = 3 (Appendix D.1).
    """

    def __init__(self, config: DMSConfig = default_config):
        self.K = config.K_verify
        # Per-memory strike counters are stored in MemoryEntry.meta.verification_failures

    def should_prune(self, entry: MemoryEntry) -> bool:
        """Check if a memory has accumulated enough strikes for pruning.

        Per Algorithm 1, line 26-28:
            if K_m >= K_limit then M := M without {m}  (prune obsolete memory)
        """
        return entry.meta.verification_failures >= self.K

    def record_failure(self, entry: MemoryEntry) -> bool:
        """Record a verification failure.  Returns True if memory should be pruned."""
        entry.meta.verification_failures += 1
        return self.should_prune(entry)

    def record_success(self, entry: MemoryEntry):
        """Successful reuse — strikes are NOT reset (reliability is
        tracked cumulatively for survival-value computation), but
        the success counter is incremented."""
        entry.meta.success_count += 1
        entry.meta.reuse_count += 1


# ═══════════════════════════════════════════════════════════════════════
# Verifier Agent (callable interface — plug in any MLLM)
# ═══════════════════════════════════════════════════════════════════════

class VerifierAgent:
    """Wraps an MLLM-based verifier that implements the logic in Figure 9.

    The verifier receives:
      - The original goal (sub-task plan)
      - The execution history (trajectory steps)
      - The final screenshot

    It returns a VerificationResult.

    The actual MLLM call is injected via the `verify_fn` callable so that
    the verifier works with any model backend (API, local, etc.).
    """

    def __init__(
        self,
        verify_fn: Callable[[str, list[Any], Any], VerificationResult],
        K: int = 3,
    ):
        """
        Parameters
        ----------
        verify_fn : callable
            Signature: (goal_text: str, execution_history: list, final_screenshot: Any)
            → VerificationResult
        K : int
            Verification depth (default 3).
        """
        self.verify_fn = verify_fn
        self.K = K
        self._kverifier = KVerifier()

    def verify(
        self,
        goal_text: str,
        execution_history: list[Any],
        final_screenshot: Any,
    ) -> VerificationResult:
        """Run a single verification check."""
        return self.verify_fn(goal_text, execution_history, final_screenshot)

    def verify_with_k_policy(
        self,
        entry: MemoryEntry,
        goal_text: str,
        execution_history: list[Any],
        final_screenshot: Any,
    ) -> tuple[VerificationResult, bool]:
        """Verify and update memory metadata per the K-Verification policy.

        Returns (result, should_prune).
        """
        result = self.verify(goal_text, execution_history, final_screenshot)

        if result.success:
            self._kverifier.record_success(entry)
            return result, False
        else:
            should_prune = self._kverifier.record_failure(entry)
            entry.meta.failure_count += 1
            return result, should_prune


# ═══════════════════════════════════════════════════════════════════════
# Heuristic verifier (no MLLM required — for offline testing)
# ═══════════════════════════════════════════════════════════════════════

def heuristic_verify(
    goal_text: str,
    execution_history: list[Any],
    final_screenshot: Any = None,
) -> VerificationResult:
    """Simple heuristic verifier for testing without an MLLM.

    Checks:
      1. Execution history is non-empty.
      2. History contains an action that looks like it addresses the goal.
      3. No explicit error keywords are found in the history.

    This is NOT a substitute for the full MLLM verifier — it's a
    lightweight stand-in for unit testing and offline development.
    """
    if not execution_history:
        return VerificationResult(False, "Empty execution history")

    # Check for error signals in the last few steps
    error_keywords = ["error", "failed", "exception", "denied", "incorrect"]
    for step in execution_history[-3:]:
        step_text = str(step).lower()
        for kw in error_keywords:
            if kw in step_text:
                return VerificationResult(False, f"Found error signal: '{kw}'")

    # Check that at least one action was taken
    if len(execution_history) >= 1:
        return VerificationResult(True, "Execution history is non-empty with no error signals")

    return VerificationResult(False, "Insufficient execution history")
