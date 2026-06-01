"""
infant_substrate:GAP-9 -- 4-phase infant curriculum scheduler.

Experiment-harness helper that manages phase-gated parameter switching
for the infant-stage developmental curriculum.  Not a ree_core substrate
scheduler; lives here alongside StepHarness / _metrics.py as a pure
helper that experiment scripts import.

Four phases (infant_substrate_expansion.md Section 6):
  Phase 0 (babbling):            ep 0..99  + H_pos exit condition
  Phase 1 (benefit discovery):   ep 100..499 + z_goal / benefit contacts exit
  Phase 2 (harm/benefit geography): ep 500..1999 + residue_coverage_pct exit
  Phase 3 (pre-gate readiness):  ep 2000+  (all features active)

Phase transitions:
- Episode-count gate is a HARD MINIMUM -- a phase cannot begin before the
  prescribed episode count regardless of telemetry.
- Telemetry gate (H_pos, z_goal_norm, benefit_contacts,
  residue_coverage_pct) is an ADDITIONAL requirement when the metric is
  provided.  If the metric is None, the hard episode count alone governs.
- Phases only advance, never retreat.  Even if metrics drop after a
  transition the scheduler stays in the later phase.

Usage:
    sched = InfantCurriculumScheduler(grid_size=12)
    for ep in range(max_episodes):
        env = CausalGridWorldV2(size=12, seed=ep,
                                **sched.env_kwargs())
        # ... run episode, collect telemetry ...
        sched.update(ep, h_pos=info["pos_entropy"],
                     z_goal_norm=z_norm,
                     benefit_contacts=n_contacts,
                     residue_coverage_pct=cov)
        if sched.phase_changed:
            print(f"Phase -> {sched.current_phase}")
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional


# Hard episode-count minimums:  phase i+1 cannot start before episode
# PHASE_EP_MIN[i+1].
PHASE_EP_MIN = [0, 100, 500, 2000]

# Design-spec exit thresholds (infant_substrate_expansion.md Section 6).
# H_POS_FRAC_OF_MAX recalibrated 2026-05-31 from 0.70 -> 0.20 per
# failure_autopsy_V3-EXQ-591_2026-05-27.md section 7 (GAP-C prereq 3): observed
# rolling-mean H_pos band 0.03-1.08 over 2000 episodes makes the legacy 0.70
# threshold (= 0.70 * ln(144) ~ 3.48) structurally unreachable. 0.20 * ln(144)
# ~ 0.99 sits inside the band with ~9% margin at the top.
H_POS_FRAC_OF_MAX = 0.20      # H_pos >= 0.20 * ln(grid_cells) to leave Phase 0
Z_GOAL_THRESHOLD = 0.30       # z_goal.norm() threshold for Phase 1 exit
BENEFIT_CONTACTS_REQUIRED = 5  # accidental benefit contacts in last 100 eps
RESIDUE_COVERAGE_THRESHOLD = 0.15  # residue_coverage_pct for Phase 2 exit

# Benefit-contacts rolling window length (episodes).
BENEFIT_CONTACTS_WINDOW = 100


class InfantCurriculumScheduler:
    """
    Phase-gated infant curriculum scheduler.

    Call ``update()`` once per episode end.  Query ``current_phase`` and
    ``phase_changed`` to detect transitions.  Use ``env_kwargs()`` and
    ``config_overrides()`` to get the parameter dicts for the current phase.
    """

    def __init__(
        self,
        grid_size: int = 12,
        on_phase3_entry: Optional[Callable[[], Any]] = None,
    ) -> None:
        """
        grid_size: side length of the CausalGridWorldV2 grid.  Used to
            compute the H_pos threshold for Phase 0 exit.
        on_phase3_entry: optional zero-arg callback fired EXACTLY ONCE,
            at the moment the scheduler advances into Phase 3 (INV-074 /
            MECH-333 / MECH-334 critical-period closure hook).  The
            experiment harness passes a closure that calls
            ``agent.gated_policy.crystallize()`` and
            ``agent.residue_field.snapshot_ewc_anchor()``.  Kept as a
            caller-supplied callback so this helper stays ree_core-free.
            None (default) = no-op, bit-identical legacy behaviour.
        """
        self._grid_size = grid_size
        self._current_phase: int = 0
        self._phase_changed: bool = False
        self._episode: int = -1
        # Rolling window of benefit-contact counts per episode.
        self._benefit_window: list[int] = []
        # INV-074 / MECH-334 Phase-3 crystallization hook.
        self._on_phase3_entry = on_phase3_entry
        self._phase3_hook_fired: bool = False

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current_phase(self) -> int:
        """Current curriculum phase (0-3)."""
        return self._current_phase

    @property
    def phase_changed(self) -> bool:
        """True if the most recent ``update()`` call advanced the phase."""
        return self._phase_changed

    @property
    def episode(self) -> int:
        """Episode index of the most recent ``update()`` call."""
        return self._episode

    # ------------------------------------------------------------------
    # Phase parameter dicts
    # ------------------------------------------------------------------

    def env_kwargs(self, phase: Optional[int] = None) -> Dict[str, Any]:
        """
        Return CausalGridWorldV2 constructor kwargs for the given phase
        (default: current phase).

        Only includes kwargs that differ from the env defaults; callers
        can spread this directly into the constructor alongside their own
        overrides.

        Phase 0: all infant features OFF.
        Phase 1: harm gradient (mild) + transient benefit patches.
        Phase 2: + microhabitat zones, harm gradient full.
        Phase 3: + environmental stochasticity (multi-source dynamics,
                 interoceptive noise, accelerated hazard/resource drift).
                 Destabilizing pressures for INV-074 crystallization-necessity
                 test (substrate_queue: test_bed_enrichment_crystallization_necessity).
        """
        p = self._current_phase if phase is None else phase
        if p == 0:
            return {
                "harm_gradient_enabled": False,
                "transient_benefit_enabled": False,
                "microhabitat_enabled": False,
            }
        if p == 1:
            return {
                "harm_gradient_enabled": True,
                "harm_gradient_scale": 0.15,
                "transient_benefit_enabled": True,
                "microhabitat_enabled": False,
            }
        if p == 2:
            return {
                "harm_gradient_enabled": True,
                "harm_gradient_scale": 0.30,
                "transient_benefit_enabled": True,
                "microhabitat_enabled": True,
            }
        # p == 3: Phase-3 post-crystallization destabilizing pressure.
        # Adds environmental dynamics (SD-047) + interoceptive noise (SD-048)
        # to force continued forward-model adaptation. Without crystallization,
        # the F-error-driven gradient overwriting should collapse diversity
        # established in Phase 2 (INV-074 prediction; test_bed_enrichment_
        # crystallization_necessity substrate item). With crystallization
        # (MECH-333/334), the locked diversity distribution should persist.
        return {
            "harm_gradient_enabled": True,
            "harm_gradient_scale": 0.30,
            "transient_benefit_enabled": True,
            "microhabitat_enabled": True,
            # SD-047: Multi-source environmental dynamics (weather + transients + drift).
            "multi_source_dynamics_enabled": True,
            "multi_source_intensity_scale": 0.8,
            "weather_field_enabled": True,
            "transient_events_enabled": True,
            "background_drift_enabled": True,
            "n_drift_sources": 2,
            # SD-048: Interoceptive noise (autonomic + sensitisation + fatigue).
            "interoceptive_noise_enabled": True,
            "interoceptive_noise_scale": 0.6,
            # Accelerated env drift to force forward-model reconsolidation.
            "env_drift_interval": 3,
            "env_drift_prob": 0.5,
        }

    def config_overrides(self, phase: Optional[int] = None) -> Dict[str, Any]:
        """
        Return agent config override dict for the given phase.

        Keys match REEConfig field names; callers apply them to the config
        object or pass them into REEConfig.from_dims kwargs.

        Phase 0: babbling -- high novelty bonus, no residue, fast sleep.
        Phase 1: benefit discovery -- higher novelty, gentle residue.
        Phase 2: geography -- structured curiosity dominant, fuller residue.
        Phase 3: pre-gate -- approach adult parameters.
        """
        p = self._current_phase if phase is None else phase
        _table = {
            0: {
                "novelty_bonus_weight": 0.5,
                "residue_scale_factor": 0.0,
                "offline_integration_frequency": 10,
            },
            1: {
                "novelty_bonus_weight": 0.7,
                "residue_scale_factor": 0.05,
                "offline_integration_frequency": 20,
            },
            2: {
                "novelty_bonus_weight": 0.5,
                "residue_scale_factor": 0.10,
                "offline_integration_frequency": 50,
            },
            3: {
                "novelty_bonus_weight": 0.5,
                "residue_scale_factor": 0.15,
                "offline_integration_frequency": 100,
            },
        }
        return dict(_table.get(p, _table[3]))

    # ------------------------------------------------------------------
    # Phase advancement
    # ------------------------------------------------------------------

    def update(
        self,
        episode: int,
        h_pos: Optional[float] = None,
        z_goal_norm: Optional[float] = None,
        benefit_contacts: Optional[int] = None,
        residue_coverage_pct: Optional[float] = None,
    ) -> int:
        """
        Advance scheduler state after one episode completes.

        Parameters
        ----------
        episode : 0-indexed episode number.
        h_pos : Shannon entropy of position histogram this episode (nats).
        z_goal_norm : z_goal.norm() at end of episode.
        benefit_contacts : transient-benefit + resource contact count this ep.
        residue_coverage_pct : fraction of residue centers with |w| > threshold.

        Returns
        -------
        current_phase : int (may be unchanged).
        """
        self._episode = episode
        self._phase_changed = False
        prev = self._current_phase

        # Update rolling benefit-contacts window.
        if benefit_contacts is not None:
            self._benefit_window.append(int(benefit_contacts))
            if len(self._benefit_window) > BENEFIT_CONTACTS_WINDOW:
                self._benefit_window = self._benefit_window[-BENEFIT_CONTACTS_WINDOW:]

        # Attempt each transition in order (phase only advances).
        self._try_phase_0_to_1(episode, h_pos)
        self._try_phase_1_to_2(episode, z_goal_norm)
        self._try_phase_2_to_3(episode, residue_coverage_pct)

        if self._current_phase != prev:
            self._phase_changed = True

        # INV-074 / MECH-334: fire the critical-period closure hook
        # exactly once, the first time the scheduler is in Phase 3.
        # Guarded so a callback exception cannot be double-fired.
        if (
            self._current_phase == 3
            and not self._phase3_hook_fired
            and self._on_phase3_entry is not None
        ):
            self._phase3_hook_fired = True
            self._on_phase3_entry()

        return self._current_phase

    # ------------------------------------------------------------------
    # Internal transition helpers
    # ------------------------------------------------------------------

    def _try_phase_0_to_1(self, episode: int, h_pos: Optional[float]) -> None:
        if self._current_phase != 0:
            return
        if episode < PHASE_EP_MIN[1]:
            return
        # Telemetry gate.
        if h_pos is not None:
            h_max = math.log(self._grid_size ** 2)
            if h_pos < H_POS_FRAC_OF_MAX * h_max:
                return  # threshold not met; stay in Phase 0
        # Either threshold met or no telemetry -> advance.
        self._current_phase = 1

    def _try_phase_1_to_2(self, episode: int, z_goal_norm: Optional[float]) -> None:
        if self._current_phase != 1:
            return
        if episode < PHASE_EP_MIN[2]:
            return
        # Telemetry gate: z_goal AND benefit contacts must both be satisfied
        # when the relevant metrics are provided.
        if z_goal_norm is not None and z_goal_norm < Z_GOAL_THRESHOLD:
            return
        recent_contacts = sum(self._benefit_window)
        # Only enforce the contacts gate when at least one contact count has
        # been logged (avoids blocking forever when the metric is never passed).
        if self._benefit_window and recent_contacts < BENEFIT_CONTACTS_REQUIRED:
            return
        self._current_phase = 2

    def _try_phase_2_to_3(
        self, episode: int, residue_coverage_pct: Optional[float]
    ) -> None:
        if self._current_phase != 2:
            return
        if episode < PHASE_EP_MIN[3]:
            return
        if residue_coverage_pct is not None:
            if residue_coverage_pct < RESIDUE_COVERAGE_THRESHOLD:
                return
        self._current_phase = 3

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def phase_summary(self) -> Dict[str, Any]:
        """Compact diagnostic dict for logging."""
        return {
            "current_phase": self._current_phase,
            "episode": self._episode,
            "phase_changed": self._phase_changed,
            "benefit_window_sum": sum(self._benefit_window),
            "benefit_window_len": len(self._benefit_window),
            "phase3_hook_fired": self._phase3_hook_fired,
        }
