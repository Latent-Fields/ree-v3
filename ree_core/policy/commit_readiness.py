"""MECH-090: commit-entry readiness signal (R-c conjunction reading).

Child of MECH-090 (BG beta-gate commit-entry predicate). Pure-arithmetic
regulator that tracks a [0, 1] motor-program readiness scalar consumed by
the readiness-above-floor leg of the R-c conjunction at the BetaGate
commit-entry call site in REEAgent.select_action.

ARCHITECTURE

  Pure scalar regulator. Maintains a running EMA over per-tick outcome
  signals (default: env-emitted successful resource contacts) plus an
  explicit override seam (notify_outcome) for harnesses that wish to push
  a probe-derived readiness (e.g. committed_mode_curriculum.run_p0_warmup
  feeding the nav_competence scalar that precipitated this substrate
  amendment in V3-EXQ-592 seed 42). No learned parameters, no nn.Module.

  Two update paths:

    1. update(outcome_signal: float, simulation_mode: bool) -- per-tick
       call from REEAgent.sense (or the natural waking site). Outcome
       signal is the substrate-side default: env-emitted "did I just
       contact a resource / reach a goal cell this tick" boolean
       normalised to [0, 1]. Internal EMA:

           readiness <- (1 - alpha) * readiness + alpha * outcome_signal

       window is used by the harness override (commit_readiness_window)
       to determine the reset cadence; the EMA itself is windowless --
       the window is the rough "effective half-life" the alpha targets.

    2. notify_outcome(value: float, simulation_mode: bool) -- explicit
       hard-set or pushed-from-harness override. Replaces the EMA's
       current value. The substrate-side EMA continues to advance on
       subsequent update() calls; notify_outcome is a hard reset of the
       running value, NOT a per-call accumulator. Sole purpose: let the
       experiment harness substitute its own readiness measurement
       (e.g. probe-window nav_competence) for the substrate-side
       env-success-rate default.

  Readiness is initialised to 1.0 (fail-open semantics: an agent with no
  outcome history should not be blocked from elevating beta -- the
  conjunction returns to legacy rv-only behaviour until the EMA has
  collected real outcome data). Per-episode reset() returns readiness to
  this initial value so each episode starts with a clean readiness
  signal.

R-c CONJUNCTION SITE (consumer)

  REEAgent.select_action calls is_above_floor(mech090_readiness_floor)
  immediately before each beta_gate.elevate() entry call site. When the
  predicate is False AND use_mech090_readiness_conjunction is True, the
  elevation is blocked; the agent stays uncommitted on this tick. The
  agent-side commit-entry side effects (anchor snapshot, trajectory
  record, _committed_step_idx reset) are gated together with the elevate
  call -- a blocked entry produces NO commit-entry side effects (the
  cleanest semantic: the gate just didn't fire).

  The diagnostic counter _n_blocks_emitted increments only when the
  module's is_above_floor returns False on a real query (i.e. the caller
  WOULD have elevated under the conjunction if readiness had cleared
  the floor). The caller is responsible for advancing this counter via
  notify_block() at the actual block site; the module itself does not
  know whether is_above_floor was consulted under the conjunction or
  in a diagnostic context.

LITERATURE GROUNDING (R-c, post-pass disposition)

  See REE_assembly/evidence/literature/targeted_review_connectome_mech_090/
  synthesis.md (2026-05-28 author commit 9e68c5ca8a). R-c is the strongest
  reading of the 31-entry corpus:

    Hanes & Schall 1996 (Science, DOI 10.1126/science.274.5286.427).
      Accumulator-to-threshold: commitment is the threshold-crossing of
      a readiness/preparation signal in FEF saccadic countermanding. The
      gated quantity is readiness, not inverse precision. Conf 0.80
      weakens R-a (rv-only-is-fine).

    Cisek & Kalaska 2010 (Annu Rev Neurosci, DOI 10.1146/annurev.neuro
      .051508.135409). Affordance-competition: action selection is the
      competition resolution among prepared motor plans; "commitment
      to a regime without a prepared affordance" is not coherent within
      the model. Conf 0.78 weakens R-a.

    Roesch, Calu & Schoenbaum 2007 (Nat Neurosci, DOI 10.1038/nn2013).
      Premature-commit pathology: operationalises "committed before the
      motor program was specified" as a measurable impulsivity signature
      with a dopaminergic correlate. Conf 0.72 weakens R-a.

  These three Pass-2 entries converted V3-EXQ-592 seed 42 from "the rv-
  only substrate is satisfiable in a way we don't like" (R-a curriculum
  failure) to "the substrate is mis-architected against three load-
  bearing literatures" (R-c substrate failure).

MECH-094

  update(simulation_mode=True) and notify_outcome(..., simulation_mode=True)
  return without advancing the EMA. Replay / DMN consumers that wake the
  agent into a simulation tick cannot inherit a readiness signal that
  biologically belongs only to active behaviour. Match the SD-035 /
  MECH-279 / gated_policy / MECH-313 / MECH-320 simulation_mode pattern.

INTEGRATION SITE (Phase 1)

  REEAgent.__init__: instantiates self.commit_readiness when
  config.use_commit_readiness=True. None otherwise.
  REEAgent.select_action: computes _readiness_admits via
  commit_readiness.is_above_floor(config.mech090_readiness_floor) when
  the conjunction master switch is on; ANDs the result with the existing
  BetaGate.should_admit_elevation score-margin gate at both
  beta_gate.elevate() call sites (bistable + legacy). When the
  conjunction blocks, commit_readiness.notify_block() is called to
  advance the diagnostic counter.
  REEAgent.reset: calls commit_readiness.reset() per-episode.

  Per-tick outcome-signal source (Phase 1): the experiment harness pushes
  via commit_readiness.notify_outcome(value). The substrate itself does
  NOT compute outcome signals in Phase 1; the seam is wired and the
  harness is responsible for the per-tick update. This matches the user's
  "sketch the API for moving the computation into ree_core/" directive:
  Phase 1 lands the API (the module, the conjunction wiring, the reset
  hook, the notify_outcome push path); Phase 2 (separate
  /implement-substrate pass) wires an env-emitted "mech090_readiness_
  outcome" key reading in agent.sense() so the substrate can advance
  readiness automatically without harness involvement.

DEFAULTS

  ema_alpha=0.1 (~10-tick effective half-life), window=20 (~20-tick rough
  half-life target; informational, alpha is the load-bearing knob),
  initial_readiness=1.0 (fail-open). mech090_readiness_floor=0.3 (set at
  the REEConfig level, not here). All defaults intentionally permissive
  so master-OFF behaviour is bit-identical and master-ON-with-defaults
  produces a substrate-active-but-permissive readiness signal that the
  V3-EXQ-592b validation experiment can sweep.

SCOPE LIMITS (Phase 1)

  The Phase 1 instantiation is the simplest tractable readiness signal:
  EMA over env outcome signals plus a harness override seam. Three
  candidate alternative readiness signals named in the user's invocation
  are deferred unless the env-success-rate generalisation fails:

    CEM accumulator-to-threshold readiness (Hanes-Schall analog) --
      would require threading a per-candidate accumulator value through
      HippocampalModule.propose_trajectories; out-of-scope this pass.

    Affordance-preparation readiness (Cisek-Kalaska analog) -- would
      require an affordance-competition layer that REE does not yet
      have; out-of-scope this pass.

    Phasic dopaminergic burst on the leading candidate -- would require
      wiring through HippocampalModule.compute_completion_signal which
      already maps to a different dopamine analog (subiculum->NAc->VP->
      VTA loop per MECH-105); collapsing the two dopamine analogs into
      one substrate is a separate governance question.

  Phase 2 (if needed): the harness-override seam (notify_outcome) is
  ALREADY the API for plumbing any alternative readiness signal. The
  experiment harness can compute the alternative signal externally and
  push it into the substrate without touching this module's internal
  computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CommitReadinessConfig:
    """MECH-090 commit-entry readiness signal configuration.

    Attributes:
        use_commit_readiness : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate
            CommitReadiness when False.
        commit_readiness_window : nominal effective half-life (ticks)
            the EMA alpha targets. Informational; alpha is the load-
            bearing knob. Used by callers that wish to scale outcome
            signals by an external window (e.g. a probe harness
            averaging over its own window). Default 20 (~20-tick rough
            half-life).
        commit_readiness_ema_alpha : EMA update rate. Default 0.1
            (~10-tick half-life on the EMA itself). Q-deferred
            calibration may sweep this.
        commit_readiness_initial : initial readiness value. Default 1.0
            (fail-open: an agent with no outcome history defaults to
            "ready" so the conjunction reduces to rv-only behaviour
            until real outcome data has been collected).
    """

    use_commit_readiness: bool = False
    commit_readiness_window: int = 20
    commit_readiness_ema_alpha: float = 0.1
    commit_readiness_initial: float = 1.0


class CommitReadiness:
    """MECH-090 commit-entry readiness regulator (waking-only).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance.
    Maintains a [0, 1] readiness EMA over per-tick outcome signals plus
    a notify_outcome explicit override seam.

    Diagnostics tracked per call:
        _last_outcome_signal             : float
        _readiness                       : float (the EMA itself)
        _n_updates                       : int
        _n_simulation_skips              : int  (MECH-094 skip count)
        _n_blocks_emitted                : int  (caller-advanced via
                                                 notify_block; counts how
                                                 often is_above_floor was
                                                 consulted and returned
                                                 False under the
                                                 conjunction)
    """

    def __init__(self, config: "CommitReadinessConfig | None" = None) -> None:
        self.config = config if config is not None else CommitReadinessConfig()
        if not (0.0 <= self.config.commit_readiness_ema_alpha <= 1.0):
            raise ValueError(
                "commit_readiness_ema_alpha must be in [0, 1]. Got "
                f"{self.config.commit_readiness_ema_alpha}."
            )
        if not (0.0 <= self.config.commit_readiness_initial <= 1.0):
            raise ValueError(
                "commit_readiness_initial must be in [0, 1]. Got "
                f"{self.config.commit_readiness_initial}."
            )
        if self.config.commit_readiness_window < 1:
            raise ValueError(
                "commit_readiness_window must be >= 1. Got "
                f"{self.config.commit_readiness_window}."
            )
        self._readiness: float = float(self.config.commit_readiness_initial)
        self._last_outcome_signal: float = 0.0
        self._n_updates: int = 0
        self._n_simulation_skips: int = 0
        self._n_blocks_emitted: int = 0

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def update(
        self,
        outcome_signal: Optional[float] = None,
        simulation_mode: bool = False,
    ) -> float:
        """Advance the readiness EMA with a per-tick outcome signal.

        Args:
            outcome_signal : per-tick outcome scalar in [0, 1], OR None.
                Values outside [0, 1] are clipped (defensive; the EMA
                must stay in [0, 1] to remain comparable to the floor).
                None is the no-signal sentinel: no EMA advance, no
                counter increment. The default substrate-side source
                is env-emitted "did I just contact a resource / reach
                a goal cell this tick" normalised to [0, 1]; substrates
                that do not emit such a key should let the caller pass
                None (the agent reads obs_dict.get("mech090_readiness_
                outcome") which returns None when the key is absent),
                so readiness stays at its initial value (default 1.0,
                fail-open) until a real outcome signal arrives.
            simulation_mode : MECH-094 gate. When True, no state advance
                and only the simulation-skip counter increments. Match
                the SD-035 / MECH-279 / gated_policy / MECH-313 /
                MECH-320 pattern.

        Returns:
            the (possibly unchanged) readiness value after the update.
        """
        if simulation_mode:
            self._n_simulation_skips += 1
            return self._readiness
        if outcome_signal is None:
            # No-signal sentinel: substrates that do not emit an outcome
            # key let readiness sit at its initial value (fail-open).
            return self._readiness

        clipped = max(0.0, min(1.0, float(outcome_signal)))
        alpha = float(self.config.commit_readiness_ema_alpha)
        self._readiness = (1.0 - alpha) * self._readiness + alpha * clipped
        # Defensive clip after the EMA step in case of floating-point
        # drift over many ticks; both operands are already in [0, 1] so
        # this is a no-op in practice.
        self._readiness = max(0.0, min(1.0, self._readiness))
        self._last_outcome_signal = clipped
        self._n_updates += 1
        return self._readiness

    def notify_outcome(
        self,
        value: float,
        simulation_mode: bool = False,
    ) -> float:
        """Hard-set the readiness EMA to a pushed value (harness override).

        Sole purpose: let an experiment harness substitute a probe-
        derived readiness scalar (e.g. committed_mode_curriculum's
        nav_competence) for the substrate-side env-success-rate
        default. Replaces the EMA's running value; subsequent update()
        calls continue to advance the EMA normally from the new
        starting point.

        Args:
            value : new readiness scalar in [0, 1]. Values outside [0,
                1] are clipped.
            simulation_mode : MECH-094 gate (same semantics as update()).

        Returns:
            the (possibly unchanged) readiness value after the override.
        """
        if simulation_mode:
            self._n_simulation_skips += 1
            return self._readiness

        clipped = max(0.0, min(1.0, float(value)))
        self._readiness = clipped
        self._last_outcome_signal = clipped
        self._n_updates += 1
        return self._readiness

    def get_readiness(self) -> float:
        """Return the current readiness scalar in [0, 1]."""
        return self._readiness

    def is_above_floor(self, floor: float) -> bool:
        """True iff current readiness >= floor.

        Comparison is inclusive at the floor so the boundary case
        (readiness == floor) admits elevation. Callers (REEAgent
        commit-entry sites) supply the floor from
        config.mech090_readiness_floor.
        """
        return self._readiness >= float(floor)

    def notify_block(self) -> None:
        """Advance the diagnostic block counter.

        Called by REEAgent at the commit-entry call site when the
        readiness conjunction blocks an elevation (rv was low AND
        is_above_floor returned False). The module itself does not
        know whether is_above_floor was consulted under the conjunction
        or in a diagnostic context, so the caller is responsible for
        advancing this counter at the actual block site.
        """
        self._n_blocks_emitted += 1

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode state and diagnostic counters.

        Returns readiness to commit_readiness_initial (default 1.0,
        fail-open) so each episode starts with a clean readiness signal.
        Diagnostic counters are zeroed.
        """
        self._readiness = float(self.config.commit_readiness_initial)
        self._last_outcome_signal = 0.0
        self._n_updates = 0
        self._n_simulation_skips = 0
        self._n_blocks_emitted = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "readiness": self._readiness,
            "last_outcome_signal": self._last_outcome_signal,
            "n_updates": self._n_updates,
            "n_simulation_skips": self._n_simulation_skips,
            "n_blocks_emitted": self._n_blocks_emitted,
        }
