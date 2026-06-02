"""MECH-342: maintenance-time readiness-driven commitment-release coupling.

Release-side complement to the MECH-090 commit-entry readiness conjunction
(R-c admission predicate). The MECH-090 audit (2026-06-02) + the
motor-program-cessation lit-pull (Resulaj 2009; Cavanagh/Frank 2011;
Falasconi/Arber 2025; Wessel 2022) established that the R-c readiness
conjunction is ADMISSION-ONLY by design: BetaGate.should_admit_elevation
(score_margin) and CommitReadiness.is_above_floor (nav_competence) gate
beta_gate.elevate() only on commit ENTRY (bistable: result.committed AND NOT
is_elevated). When execution readiness degrades WHILE the agent is already
beta-elevated, no existing release pathway covers it (B1 ruled out:
ARC-028/MECH-105 hippocampal-completion releases on POSITIVE completion;
MECH-091 is a z_harm threat gate; the V_s commit-release fires on
schema-staleness; SD-034 closure on rule-stability). V3-EXQ-592f measured
exactly this gap: with beta forced elevated and degraded readiness sustained,
state-occupancy suppression and decommit transitions were ZERO.

MECH-342 closes that gap: the SAME two R-c readiness signals that MECH-090
AND-composes to ADMIT a commitment here drive a graded, bounded-accumulation
RELEASE of an already-elevated beta latch when they degrade mid-commitment.

WHAT THIS IS NOT (falsifiable distinctions; the substrate must not collapse
into any of these):

  * NOT the MECH-090 admission predicate. That gate fires on commit ENTRY
    and is AND-composed (both axes must pass to admit). This is the release
    side, fires during MAINTENANCE, and is OR-composed (EITHER axis failing
    drives release -- the De Morgan dual of the AND admission).
  * NOT MECH-091 urgency-interrupt. MECH-091 fires on z_harm_a/z_harm_un
    norm > threshold (acute external threat / surprise). MECH-342 fires on
    internal decisiveness / motor-readiness decay with z_harm_a BELOW
    threshold. No harm-stream input.
  * NOT ARC-028 / MECH-105 hippocampal-completion. That releases when the
    completion_signal is HIGH (a good plan/state was found -- options GOOD).
    MECH-342 fires when decisiveness is LOW (poor candidate options) -- the
    opposite regime. No completion-signal input.
  * NOT MECH-269b / V_s commit-release. That fires on schema staleness
    (a committed-entry anchor key dropped from the active set -- the world
    changed under the agent). MECH-342 fires with a STABLE world schema.
    No anchor-set input.
  * NOT MECH-340 ghost-goal persistence/efficacy gate. That operates on the
    ghost-goal bank (whether to keep an UNATTAINABLE GOAL as a re-probe
    target) at goal-appraisal timescale. MECH-342 operates on the ACTIVE
    beta-gate commitment at motor-program timescale.

BINDING DESIGN CONSTRAINTS FROM THE BIOLOGY (lit-pull verdict B3b):

  1. GRADED / ONLINE, NOT a one-shot Schmitt flag. The trigger is a continued
     accumulation of low decisiveness / nav_competence -- a drift-to-a-
     release-bound, conflict-scaled by the deficit magnitude -- NOT "below
     floor for K consecutive ticks". Anchored in Resulaj 2009 bounded
     accumulation (an already-initiated action is reversed by continued
     internal evidence) + Cavanagh/Frank 2011 conflict-graded STN threshold.
     Converges with the goal-disengagement pull's contested-phase warning
     (Brandstaetter 2013 / Klinger 1975: disengagement is an extended
     contested PHASE, not a flag).

  2. TARGETED + HYSTERETIC, with a reengagement path. Release the specific
     committed program (beta latch + committed trajectory), not a global
     brake (Falasconi/Arber 2025 movement-specific BG-output suppression vs
     Wessel 2022 non-selective stop-signal -- the non-selective stop reflex
     is the WRONG model and is already covered by MECH-091). The
     premature-abort pole (reversing a correct-but-hard commitment -- the
     Resulaj failure signature) is guarded by a hysteresis band (separate
     floor vs higher reengage level) and a leak-toward-zero reengagement
     path when readiness recovers.

ACCUMULATOR DYNAMICS (per maintenance tick, only while beta is elevated):

    deficit_d = clip((score_margin_floor - score_margin) / score_margin_floor, 0, 1)
                # within-tick DECISIVENESS axis (Hanes & Schall accumulator
                # reading; 0 when margin healthy or no decisiveness signal).
    deficit_n = clip((nav_floor - nav_competence) / nav_floor, 0, 1)
                # across-tick MOTOR-READINESS axis (Cisek-Kalaska affordance
                # preparation + Roesch premature-commit; 0 when no nav signal).
    combined  = max(deficit_d, deficit_n)
                # OR-composition: either axis failing drives release pressure.

    recovered = (decisiveness recovered: margin >= score_margin_reengage OR
                 no margin signal) AND (nav recovered: nav >= nav_reengage OR
                 no nav signal)

    if combined > 0:            pressure += accumulation_rate * combined   # drift to bound
    elif recovered:            pressure  = max(0, pressure - leak_rate)    # reengagement
    else:                      pressure unchanged                          # dead-band hold
    pressure = clip(pressure, 0, pressure_cap)

    fire = pressure >= release_bound      # on fire, pressure resets to 0

The caller (REEAgent.select_action release block) acts on the returned bool:
beta_gate.release(); _committed_step_idx = 0; _committed_anchor_keys = None;
e3._committed_trajectory = None. Pressure is reset to 0 at commit ENTRY (each
committed program accumulates independently) and on fire.

MECH-094

  tick(simulation_mode=True) returns False without advancing pressure. A
  replay / DMN tick must not abort a committed motor program. Matches the
  SD-035 / MECH-279 / MECH-313 / MECH-320 / commit_readiness simulation_mode
  pattern.

INTEGRATION SITE

  REEAgent.__init__: instantiates self.maintenance_release when
  config.use_maintenance_release=True. None otherwise (bit-identical OFF).
  REEAgent.select_action release block (mirroring the MECH-091 template):
  when maintenance_release is not None AND beta_gate.is_elevated, reads
  decisiveness from self.e3.last_scores and nav_competence from
  self.commit_readiness.get_readiness() (None when CommitReadiness is not
  instantiated -- the nav axis is then inert and only the decisiveness axis
  drives release), calls tick(); on fire releases the committed program.
  REEAgent.reset: calls reset() per-episode. Commit-entry sites call
  reset_pressure() so each committed program starts at zero pressure.

See REE_assembly/docs/architecture/mech_342_commit_maintenance_release.md
and ree_core/policy/commit_readiness.py (the admission-side sibling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CommitMaintenanceReleaseConfig:
    """MECH-342 maintenance-time commitment-release configuration.

    Attributes:
        use_maintenance_release : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate
            CommitMaintenanceRelease when False.
        score_margin_floor : at/below this the within-tick decisiveness axis
            (per-candidate first-action score margin, REE lower-is-better)
            contributes release-pressure deficit. Mirrors the MECH-090
            commit_readiness_floor (0.05).
        score_margin_reengage : at/above this the decisiveness axis counts
            as recovered and contributes to the leak (reengagement) path.
            Must be >= score_margin_floor (hysteresis band).
        nav_floor : at/below this the across-tick nav_competence axis
            (CommitReadiness EMA) is failing. Mirrors mech090_readiness_floor
            (0.3).
        nav_reengage : at/above this the nav axis counts as recovered.
            Must be >= nav_floor (hysteresis band).
        accumulation_rate : per-tick pressure increment scale; pressure
            grows by accumulation_rate * combined_deficit each tick the
            combined deficit is positive (conflict-scaled drift-to-bound).
        leak_rate : per-tick pressure decay applied when both axes have
            recovered (the reengagement path).
        release_bound : pressure threshold at which release fires.
        pressure_cap : hard clamp on accumulated pressure.
    """

    use_maintenance_release: bool = False
    score_margin_floor: float = 0.05
    score_margin_reengage: float = 0.10
    nav_floor: float = 0.3
    nav_reengage: float = 0.5
    accumulation_rate: float = 0.2
    leak_rate: float = 0.1
    release_bound: float = 1.0
    pressure_cap: float = 1.5


class CommitMaintenanceRelease:
    """MECH-342 maintenance-time commitment-release regulator (waking-only).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance.
    Maintains a [0, pressure_cap] release-pressure accumulator driven by the
    OR-composition of the two R-c readiness deficits, with a hysteretic
    reengagement leak.

    Diagnostics tracked:
        _pressure                : float (the accumulator itself)
        _last_combined_deficit   : float
        _last_deficit_d          : float  (decisiveness axis)
        _last_deficit_n          : float  (nav_competence axis)
        _n_ticks                 : int    (maintenance ticks evaluated)
        _n_accumulate            : int    (ticks pressure rose)
        _n_leak                  : int    (ticks pressure leaked; reengagement)
        _n_hold                  : int    (dead-band hold ticks)
        _n_fires                 : int    (release events)
        _n_simulation_skips      : int    (MECH-094 skip count)
    """

    def __init__(
        self, config: "CommitMaintenanceReleaseConfig | None" = None
    ) -> None:
        self.config = (
            config if config is not None else CommitMaintenanceReleaseConfig()
        )
        c = self.config
        if c.score_margin_floor < 0.0:
            raise ValueError(
                f"score_margin_floor must be >= 0. Got {c.score_margin_floor}."
            )
        if c.score_margin_reengage < c.score_margin_floor:
            raise ValueError(
                "score_margin_reengage must be >= score_margin_floor "
                f"(hysteresis band). Got reengage={c.score_margin_reengage}, "
                f"floor={c.score_margin_floor}."
            )
        if not (0.0 <= c.nav_floor <= 1.0):
            raise ValueError(f"nav_floor must be in [0, 1]. Got {c.nav_floor}.")
        if not (c.nav_floor <= c.nav_reengage <= 1.0):
            raise ValueError(
                "nav_reengage must be in [nav_floor, 1] (hysteresis band). "
                f"Got reengage={c.nav_reengage}, floor={c.nav_floor}."
            )
        if c.accumulation_rate <= 0.0:
            raise ValueError(
                f"accumulation_rate must be > 0. Got {c.accumulation_rate}."
            )
        if c.leak_rate < 0.0:
            raise ValueError(f"leak_rate must be >= 0. Got {c.leak_rate}.")
        if c.release_bound <= 0.0:
            raise ValueError(
                f"release_bound must be > 0. Got {c.release_bound}."
            )
        if c.pressure_cap < c.release_bound:
            raise ValueError(
                "pressure_cap must be >= release_bound. Got "
                f"cap={c.pressure_cap}, bound={c.release_bound}."
            )
        self._pressure: float = 0.0
        self._last_combined_deficit: float = 0.0
        self._last_deficit_d: float = 0.0
        self._last_deficit_n: float = 0.0
        self._n_ticks: int = 0
        self._n_accumulate: int = 0
        self._n_leak: int = 0
        self._n_hold: int = 0
        self._n_fires: int = 0
        self._n_simulation_skips: int = 0

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def _deficit_decisiveness(
        self, score_margin: Optional[float], n_candidates: int
    ) -> Optional[float]:
        """Per-tick decisiveness deficit in [0, 1], or None when no signal.

        None when the score margin is undefined (fewer than 2 candidates, or
        the caller passed None). A None axis contributes neither deficit nor
        a recovery vote (it is inert -- the substrate does not abort on the
        absence of a decisiveness signal).
        """
        if score_margin is None or n_candidates < 2:
            return None
        floor = float(self.config.score_margin_floor)
        if floor <= 0.0:
            # Degenerate floor: any margin >= floor; deficit is 0.
            return 0.0
        deficit = (floor - float(score_margin)) / floor
        return max(0.0, min(1.0, deficit))

    def _deficit_nav(self, nav_competence: Optional[float]) -> Optional[float]:
        """Per-tick nav_competence deficit in [0, 1], or None when no signal."""
        if nav_competence is None:
            return None
        floor = float(self.config.nav_floor)
        if floor <= 0.0:
            return 0.0
        deficit = (floor - float(nav_competence)) / floor
        return max(0.0, min(1.0, deficit))

    def _axis_recovered(
        self,
        signal: Optional[float],
        reengage_level: float,
    ) -> bool:
        """True when an axis is recovered (>= reengage) OR has no signal.

        An axis with no signal does not block reengagement -- the leak path
        should fire when the present signals are all healthy.
        """
        if signal is None:
            return True
        return float(signal) >= float(reengage_level)

    def tick(
        self,
        score_margin: Optional[float],
        n_candidates: int,
        nav_competence: Optional[float],
        simulation_mode: bool = False,
    ) -> bool:
        """Advance the release-pressure accumulator one maintenance tick.

        Call only while beta is elevated (the caller guards on
        beta_gate.is_elevated). Returns True when accumulated pressure
        reaches release_bound this tick (the caller then releases the
        committed program); pressure is reset to 0 on fire.

        Args:
            score_margin : per-candidate first-action margin
                (sorted(scores)[1] - sorted(scores)[0], REE lower-is-better),
                or None when undefined. Drives the decisiveness axis.
            n_candidates : size of the candidate pool the margin came from.
            nav_competence : current CommitReadiness EMA in [0, 1], or None
                when CommitReadiness is not instantiated. Drives the
                nav-readiness axis.
            simulation_mode : MECH-094 gate. When True, no state advance and
                only the simulation-skip counter increments; returns False.

        Returns:
            True iff this tick fires a release.
        """
        if simulation_mode:
            self._n_simulation_skips += 1
            return False

        deficit_d = self._deficit_decisiveness(score_margin, n_candidates)
        deficit_n = self._deficit_nav(nav_competence)
        self._last_deficit_d = deficit_d if deficit_d is not None else 0.0
        self._last_deficit_n = deficit_n if deficit_n is not None else 0.0

        # OR-composition: either axis failing drives release pressure. A None
        # axis contributes 0 (inert). max() is the De Morgan dual of the
        # MECH-090 AND admission and is conflict-graded by the worse axis.
        present = [d for d in (deficit_d, deficit_n) if d is not None]
        combined = max(present) if present else 0.0
        self._last_combined_deficit = combined
        self._n_ticks += 1

        if combined > 0.0:
            self._pressure += float(self.config.accumulation_rate) * combined
            self._n_accumulate += 1
        else:
            recovered = self._axis_recovered(
                score_margin, self.config.score_margin_reengage
            ) and self._axis_recovered(
                nav_competence, self.config.nav_reengage
            )
            if recovered:
                self._pressure = max(
                    0.0, self._pressure - float(self.config.leak_rate)
                )
                self._n_leak += 1
            else:
                # Dead-band hold: deficit cleared the floor but neither axis
                # has reached the reengage level. The contested phase -- hold
                # pressure (do not accumulate, do not leak).
                self._n_hold += 1

        self._pressure = max(
            0.0, min(float(self.config.pressure_cap), self._pressure)
        )

        if self._pressure >= float(self.config.release_bound):
            self._n_fires += 1
            self._pressure = 0.0
            return True
        return False

    def get_pressure(self) -> float:
        """Return the current release-pressure accumulator."""
        return self._pressure

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset_pressure(self) -> None:
        """Reset the accumulator to 0 without touching diagnostic counters.

        Called by REEAgent at commit ENTRY so each committed program
        accumulates release pressure independently.
        """
        self._pressure = 0.0

    def reset(self) -> None:
        """Reset per-episode state and diagnostic counters."""
        self._pressure = 0.0
        self._last_combined_deficit = 0.0
        self._last_deficit_d = 0.0
        self._last_deficit_n = 0.0
        self._n_ticks = 0
        self._n_accumulate = 0
        self._n_leak = 0
        self._n_hold = 0
        self._n_fires = 0
        self._n_simulation_skips = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "pressure": self._pressure,
            "last_combined_deficit": self._last_combined_deficit,
            "last_deficit_decisiveness": self._last_deficit_d,
            "last_deficit_nav": self._last_deficit_n,
            "mech342_n_ticks": self._n_ticks,
            "mech342_n_accumulate": self._n_accumulate,
            "mech342_n_leak": self._n_leak,
            "mech342_n_hold": self._n_hold,
            "mech342_n_fires": self._n_fires,
            "mech342_n_simulation_skips": self._n_simulation_skips,
        }
