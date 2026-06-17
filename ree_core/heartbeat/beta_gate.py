"""
Beta Gate — MECH-090 Implementation

MECH-090: Beta oscillations gate E3→action_selection propagation.
Beta state controls whether E3's updated policy state propagates to
action selection, NOT whether E3 updates internally.

During committed sequence: beta_elevated=True
  → E3 continues updating internally
  → action_selector.update() is suppressed (policy output held)

At sequence completion / stop-change signal: beta_elevated=False
  → E3's current policy state propagates to action selection

This implements beta as a commitment gate, not a learning gate.
"""

import torch
from typing import Optional


class BetaGate:
    """
    Beta-gated policy propagation gate (MECH-090).

    Controls whether E3's current policy state propagates to action selection.
    E3 always updates internally. The gate only blocks propagation.
    """

    def __init__(
        self,
        initial_beta_elevated: bool = False,
        completion_release_threshold: float = 0.75,
        use_commit_readiness_gate: bool = False,
        commit_readiness_floor: float = 0.05,
        commit_readiness_strict_single_candidate: bool = False,
    ):
        self._beta_elevated: bool = initial_beta_elevated
        self._held_policy_state: Optional[torch.Tensor] = None
        self._propagation_count: int = 0
        self._hold_count: int = 0
        self._completion_release_threshold: float = completion_release_threshold
        self._last_completion_signal: float = 0.0
        # MECH-090 commit-entry readiness conjunction (R-c, 2026-05-28).
        # See HeartbeatConfig in ree_core/utils/config.py for full doc.
        self._use_commit_readiness_gate: bool = use_commit_readiness_gate
        self._commit_readiness_floor: float = commit_readiness_floor
        self._commit_readiness_strict_single_candidate: bool = (
            commit_readiness_strict_single_candidate
        )
        # MECH-090 diagnostics: counters for the gate path. Cleared on reset().
        self._n_elevation_admitted: int = 0
        self._n_elevation_blocked: int = 0
        self._n_elevation_single_candidate: int = 0
        self._last_readiness_score_margin: float = 0.0
        # SD-034 de-commitment hold / refractory (commitment-closure-control-plane,
        # 2026-06-12). A closure-driven release installs a refractory window via
        # apply_refractory(n); while it is active elevate() is a no-op so the latch
        # cannot immediately re-commit, producing a measurable latch-occupancy drop
        # (the V3-EXQ-468c committed_frac defect). Decremented once per propagate()
        # tick. Default 0 -> never installed -> bit-identical to pre-amend BetaGate.
        self._refractory_remaining: int = 0
        self._n_elevation_refractory_blocked: int = 0
        # SD-034 commitment-closure-control-plane BETA-ENGAGEMENT amend
        # (2026-06-17). Counts elevations driven by the closure-plane
        # commit->beta coupling (use_closure_commit_beta_coupling) rather than a
        # natural running_variance crossing (result.committed). Read by the
        # V3-EXQ-460f manifest as the non-vacuity readout that the coupling, not
        # the fragile natural commit-entry, is what engaged the latch.
        # Cleared per-episode in reset(). Stays 0 when the coupling flag is off.
        self._n_closure_coupled_elevations: int = 0

    @property
    def is_elevated(self) -> bool:
        """True when beta is elevated (commitment active, propagation suppressed)."""
        return self._beta_elevated

    @property
    def refractory_remaining(self) -> int:
        """Ticks remaining in the SD-034 post-closure de-commitment hold (0 = inactive)."""
        return self._refractory_remaining

    def apply_refractory(self, n_ticks: int) -> None:
        """
        Install a de-commitment refractory window (SD-034 closure hold).

        While the window is active (refractory_remaining > 0), elevate() is a
        no-op so a just-released latch cannot immediately re-commit. The window
        is decremented once per propagate() tick. n_ticks <= 0 is a no-op.
        Takes the max with any existing window so overlapping closures extend
        rather than truncate the hold.
        """
        if n_ticks > 0:
            self._refractory_remaining = max(self._refractory_remaining, int(n_ticks))

    def note_closure_coupled_elevation(self) -> None:
        """
        Record that the just-fired elevate() was driven by the SD-034
        closure-plane commit->beta coupling, not a natural running_variance
        crossing (commitment-closure-control-plane beta-engagement amend,
        2026-06-17). Pure diagnostic; does not change gate state. The caller
        (REEAgent.select_action) calls this immediately after elevate() when
        the coupling, not result.committed, was the trigger.
        """
        self._n_closure_coupled_elevations += 1

    def elevate(self) -> None:
        """
        Elevate beta state (sequence commitment begins).

        After this, propagate() will hold E3 state. NO-OP while a de-commitment
        refractory window is active (SD-034 hold) -- the just-released latch must
        survive the hold before it can re-commit. Bit-identical to pre-amend when
        no refractory has been installed (refractory_remaining stays 0).
        """
        if self._refractory_remaining > 0:
            self._n_elevation_refractory_blocked += 1
            return
        self._beta_elevated = True

    def release(self) -> None:
        """
        Release beta state (sequence complete or stop-change signal).

        After this, the next propagate() call will pass through E3 state.
        """
        self._beta_elevated = False

    def propagate(self, e3_policy_state: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Gate E3 policy state propagation to action selection (MECH-090).

        If beta is elevated: hold state, return None (action selector not updated).
        If beta is not elevated: pass state through, return it.

        Args:
            e3_policy_state: E3's current internal policy state [batch, state_dim]

        Returns:
            e3_policy_state if gate is open (beta not elevated), else None.
        """
        # SD-034 de-commitment hold: tick down the refractory window once per
        # propagate() call (propagate runs every select_action tick). No-op when
        # no window is active.
        if self._refractory_remaining > 0:
            self._refractory_remaining -= 1
        if self._beta_elevated:
            self._held_policy_state = e3_policy_state.detach()
            self._hold_count += 1
            return None
        else:
            self._held_policy_state = None
            self._propagation_count += 1
            return e3_policy_state

    def should_admit_elevation(
        self,
        score_margin: Optional[float],
        n_candidates: int,
    ) -> bool:
        """
        MECH-090 R-c commit-entry conjunction predicate (2026-05-28).

        Returns True when the gate is willing to admit a beta-elevation event
        (caller must still verify the rv-low precondition via the standard
        E3SelectionResult.committed signal -- this gate adds the readiness
        conjunction on top, it does NOT subsume the rv check).

        With use_commit_readiness_gate=False (default), the predicate is
        unconditionally True -- bit-identical to legacy rv-only elevation.

        With use_commit_readiness_gate=True, elevation is admitted iff:
            score_margin >= commit_readiness_floor
        where score_margin is the per-candidate first-action margin
        (REE lower-is-better convention: scores.sort().values[1] -
        scores.min(); positive when there is a clear winner; ~0 when
        the candidate pool collapsed to a near-tie -- the degenerate
        V3-EXQ-592 seed-42 signature).

        Single-candidate handling: when n_candidates < 2, score_margin
        is undefined. Default strict_single_candidate=False treats this
        as permissive (admit); True treats it as below-floor (block).
        See HeartbeatConfig docstring for the rationale.

        Diagnostics: increments _n_elevation_admitted / _n_elevation_blocked
        / _n_elevation_single_candidate counters. Caches the supplied margin
        in _last_readiness_score_margin for downstream consumers.

        Args:
            score_margin: per-candidate first-action margin scalar,
                or None when undefined (single-candidate / strict-mode
                fall-through).
            n_candidates: size of the E3 candidate pool this tick.

        Returns:
            True iff the gate admits beta-elevation this tick.
        """
        if not self._use_commit_readiness_gate:
            # Legacy rv-only path -- bit-identical.
            return True

        if n_candidates < 2:
            self._n_elevation_single_candidate += 1
            if self._commit_readiness_strict_single_candidate:
                self._n_elevation_blocked += 1
                return False
            # Permissive default: admit when the pool collapsed to a single
            # candidate (no margin to compute against).
            self._n_elevation_admitted += 1
            return True

        margin = float(score_margin) if score_margin is not None else 0.0
        self._last_readiness_score_margin = margin
        if margin >= self._commit_readiness_floor:
            self._n_elevation_admitted += 1
            return True
        self._n_elevation_blocked += 1
        return False

    def receive_hippocampal_completion(self, completion_signal: float) -> bool:
        """
        Receive hippocampal trajectory completion signal (ARC-028, MECH-105).

        When a good trajectory is found (high completion_signal), this acts as the
        subiculum->NAc->VP->VTA->dopamine loop analog (Lisman & Grace 2005):
        high hippocampal completion -> dopamine release -> beta drops -> gate opens.

        Args:
            completion_signal: float in [0, 1]. 0 = no completion; 1 = perfect trajectory.

        Returns:
            True if this signal triggered a beta release, False otherwise.
        """
        self._last_completion_signal = completion_signal
        if self._beta_elevated and completion_signal >= self._completion_release_threshold:
            self.release()
            return True
        return False

    def get_held_state(self) -> Optional[torch.Tensor]:
        """Return the held policy state (if any)."""
        return self._held_policy_state

    def get_state(self) -> dict:
        return {
            "beta_elevated": self._beta_elevated,
            "propagation_count": self._propagation_count,
            "hold_count": self._hold_count,
            "has_held_state": self._held_policy_state is not None,
            "completion_signal": self._last_completion_signal,
            "completion_release_threshold": self._completion_release_threshold,
            # MECH-090 R-c commit-entry gate diagnostics.
            "use_commit_readiness_gate": self._use_commit_readiness_gate,
            "commit_readiness_floor": self._commit_readiness_floor,
            "mech090_n_elevation_admitted": self._n_elevation_admitted,
            "mech090_n_elevation_blocked": self._n_elevation_blocked,
            "mech090_n_elevation_single_candidate": self._n_elevation_single_candidate,
            "mech090_last_readiness_score_margin": self._last_readiness_score_margin,
            # SD-034 de-commitment hold / refractory diagnostics.
            "sd034_refractory_remaining": self._refractory_remaining,
            "sd034_n_elevation_refractory_blocked": self._n_elevation_refractory_blocked,
            # SD-034 beta-engagement amend (2026-06-17).
            "sd034_n_closure_coupled_elevations": self._n_closure_coupled_elevations,
        }

    def reset(self) -> None:
        """Reset gate state (episode reset)."""
        self._beta_elevated = False
        self._held_policy_state = None
        self._last_completion_signal = 0.0
        # MECH-090 R-c gate diagnostics: cleared per-episode (cross-episode
        # counters would conflate distinct trials' degeneracy signatures).
        self._n_elevation_admitted = 0
        self._n_elevation_blocked = 0
        self._n_elevation_single_candidate = 0
        self._last_readiness_score_margin = 0.0
        # SD-034 de-commitment hold: clear the refractory window + diagnostic per
        # episode (a fresh trial starts with no carried-over hold).
        self._refractory_remaining = 0
        self._n_elevation_refractory_blocked = 0
        # SD-034 beta-engagement amend: per-episode reset of the coupling counter.
        self._n_closure_coupled_elevations = 0
