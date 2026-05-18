"""MECH-313: stochastic_noise_floor (LC-NE tonic / SAC max-entropy analog).

Child claim of ARC-065 (behavioral_diversity_generation_pathway). State-
independent stochastic floor on action selection that prevents complete
deterministic collapse of the policy. Distinct from MECH-260 dACC anti-
recency, which is state-dependent (penalises a candidate as a function of
recent action history).

ARCHITECTURE

  Pure scalar regulator. No internal buffer, no learned parameters.
  Each waking tick, computes an effective_temperature from the caller-
  supplied baseline_temperature using two knobs:

      effective_T = max(baseline_T + alpha, min_temperature)

  alpha           : SAC-entropy-bonus analog. Adds a constant lift to the
                    softmax temperature (raises action-selection entropy
                    floor; biologically: tonic LC-NE noise lifts baseline
                    decision noise across all policies).
  min_temperature : hard floor on the effective temperature. Guarantees
                    the policy NEVER collapses to argmax even if upstream
                    consumers pass a very small baseline (e.g. annealing
                    schedules).

  Both lift the softmax temperature only; the action-selection mechanism
  is unchanged. There is no candidate-level bias and no perturbation of
  scores.

LIT-PULL VERDICTS (resolved defaults; see SYNTHESIS for cited evidence)

  Pull 1 (ARC-065 behavioral_diversity_generation_pathway,
  evidence/literature/targeted_review_arc_065_behavioral_diversity_generation/
  SYNTHESIS.md, lit_conf 0.78-0.82, 9 entries):

    R1 -- Both noise-floor (this claim) AND structured-curiosity (MECH-314)
          channels needed for behavioural diversity (Wilson et al. 2014
          Horizon task; Faisal/Selen/Wolpert 2008 noise-substrate
          irreducibility; Friston 2015 active-inference complementary
          terms). Single-channel readings fail empirically (noise-only)
          or biologically (curiosity-only).

    R2 -- LC-NE tonic firing is the load-bearing biological substrate
          (Aston-Jones & Cohen 2005 adaptive-gain model). Tonic mode
          elevates baseline decision noise across all policies; phasic
          mode (already covered by MECH-104 volatility-surprise spike)
          gates committed exploitation.

    R4 -- Continuous, every tick, regardless of context (Aston-Jones &
          Cohen 2005). NOT triggered. Implementation: non-zero softmax
          temperature on E3 every waking tick.

  Magnitudes are NOT settled by the lit-pull (none of the cited papers
  pin numerical values). Q-043 captures the open question of relative
  weight calibration via parametric sweep on V3-EXQ-543b/c.

  Default values (alpha=0.1, min_temperature=1.0) are conservative
  starting points that preserve the existing E3 baseline temperature
  (1.0) when the master switch is on with defaults, then add a modest
  +0.1 lift. They are intentionally not zero so that turning the master
  switch on produces a non-trivial floor without pushing the policy
  too far from baseline before Q-043 calibrates them.

DISTINCTION FROM MECH-260

  MECH-260 (dACC anti-recency, ree_core/cingulate/dacc.py): adds a
  per-candidate score_bias proportional to the frequency of each
  candidate's action class in the recent action_history FIFO. STATE-
  DEPENDENT. After argmax-only-on-class-A behaviour, MECH-260 penalises
  class A specifically.

  MECH-313 (this module): lifts the softmax temperature uniformly.
  STATE-INDEPENDENT. After argmax-only-on-class-A behaviour, MECH-313
  raises the probability mass on every NON-A class equally.

  Q-045 falsifies whether they collapse into a single substrate: 4-arm
  ablation on V3-EXQ-543b/c (both-OFF / 313-only / 260-only / both-ON).
  This module's substrate guarantees they can coexist as independent
  flags.

INTEGRATION SITE

  REEAgent.select_action() reads noise_floor.compute_effective_temperature(
  baseline_temperature, simulation_mode) BEFORE calling e3.select(...,
  temperature=effective_T, ...). Phase 1 wires the regulator at the call
  site only -- no candidate-level changes, no E3 internal modification.
  See agent.py for the wiring.

MECH-094

  simulation_mode=True returns the baseline_temperature unchanged and
  does NOT advance diagnostic counters. Replay / DMN consumers calling
  this regulator (none today; reserved for forward-compat) get the
  un-floored baseline so simulation paths cannot inherit waking-tonic
  noise floor that biologically belongs only to active behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NoiseFloorConfig:
    """MECH-313 stochastic noise-floor configuration.

    Attributes:
        use_noise_floor : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate
            NoiseFloor when False.
        noise_floor_alpha : SAC-entropy-bonus analog; constant additive
            lift on the softmax temperature. Default 0.1 = modest +10%
            of the baseline E3 temperature (1.0). Q-043 calibrates.
        noise_floor_min_temperature : hard lower bound on the effective
            temperature. Default 1.0 matches the existing E3 default,
            so a baseline_temperature >= 1.0 already clears the floor
            (only annealed-low schedules engage it). Setting > 1.0
            engages the floor on every waking tick.
    """

    use_noise_floor: bool = False
    noise_floor_alpha: float = 0.1
    noise_floor_min_temperature: float = 1.0


class NoiseFloor:
    """MECH-313 stochastic noise-floor regulator (state-independent).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance.
    Each waking tick:
        effective_T = max(baseline_T + alpha, min_temperature)

    Diagnostics tracked per call:
        _last_baseline_temperature       : float
        _last_effective_temperature      : float
        _last_n_simulation_skips         : int  (count of MECH-094 skips)
        _n_waking_calls                  : int  (count of waking calls)
    """

    def __init__(self, config: "NoiseFloorConfig | None" = None) -> None:
        self.config = config if config is not None else NoiseFloorConfig()
        if self.config.noise_floor_alpha < 0.0:
            raise ValueError(
                "noise_floor_alpha must be >= 0 (it is an additive lift on "
                f"the softmax temperature). Got {self.config.noise_floor_alpha}."
            )
        if self.config.noise_floor_min_temperature <= 0.0:
            raise ValueError(
                "noise_floor_min_temperature must be > 0 (softmax "
                "temperature must be strictly positive). Got "
                f"{self.config.noise_floor_min_temperature}."
            )
        self._last_baseline_temperature: float = 0.0
        self._last_effective_temperature: float = 0.0
        self._last_n_simulation_skips: int = 0
        self._n_waking_calls: int = 0

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def compute_effective_temperature(
        self,
        baseline_temperature: float,
        simulation_mode: bool = False,
    ) -> float:
        """Return the noise-floored effective softmax temperature.

        Args:
            baseline_temperature : caller's baseline (e.g. the temperature
                kwarg threaded through REEAgent.select_action -> e3.select).
                Must be > 0 (validated; softmax temperature is strictly
                positive).
            simulation_mode : MECH-094 gate. When True, returns
                baseline_temperature unchanged and increments only the
                simulation-skip counter. Match the SD-035 / MECH-279 /
                gated_policy simulation_mode pattern.

        Returns:
            effective temperature = max(baseline + alpha, min_temperature),
            or baseline_temperature unchanged when simulation_mode=True.
        """
        if baseline_temperature <= 0.0:
            raise ValueError(
                "baseline_temperature must be > 0 (softmax temperature is "
                f"strictly positive). Got {baseline_temperature}."
            )
        if simulation_mode:
            self._last_n_simulation_skips += 1
            return float(baseline_temperature)

        self._n_waking_calls += 1
        lifted = float(baseline_temperature) + float(self.config.noise_floor_alpha)
        effective = max(lifted, float(self.config.noise_floor_min_temperature))
        self._last_baseline_temperature = float(baseline_temperature)
        self._last_effective_temperature = effective
        return effective

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode diagnostic counters.

        No persistent state to clear (regulator is stateless across ticks).
        """
        self._last_baseline_temperature = 0.0
        self._last_effective_temperature = 0.0
        self._last_n_simulation_skips = 0
        self._n_waking_calls = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "last_baseline_temperature": self._last_baseline_temperature,
            "last_effective_temperature": self._last_effective_temperature,
            "last_n_simulation_skips": self._last_n_simulation_skips,
            "n_waking_calls": self._n_waking_calls,
        }
