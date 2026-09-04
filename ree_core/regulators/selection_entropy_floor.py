"""SD-105: control_plane.selection_entropy_headroom_floor.

A TONIC BEHAVIOURAL-VARIABILITY SET-POINT on the E3 selection softmax
temperature. Leg (b) of the sd_phasic_burst_decay_and_warmup_headroom repair
(confirmed autopsy failure_autopsy_V3-EXQ-963a_2026-09-02, pillar 2).

THE DEFECT

  The SD-074 probe warmup exists to bring the agent to a non-degenerate E3
  action-value landscape before telemetry is collected. It succeeds -- and in
  succeeding it produces a CONFIDENT policy, which has almost no selection
  entropy left to move. V3-EXQ-963a's T0P0 baseline (tonic OFF, phasic OFF)
  measured 0.0195-0.153 normalized selection entropy against V3-EXQ-779a's
  0.152-0.610 on the same design: a 3-26x collapse on EVERY seed, not one
  unlucky one.

  That pins the R5 headroom precondition (baseline entropy inside the
  0.02-0.98 saturation band) at its floor reproducibly. Worse than the gate
  failing: the phasic lever SHARPENS (temp_delta is negative by default), so
  a baseline sitting at 0.0195 leaves ~2% of the readout's range for the
  manipulation to move into. A null measured there says nothing about the
  mechanism -- it says the readout had nowhere to go. That is the
  ANTI-CONSERVATIVE direction experiments/_lib/entropy_headroom.py already
  warns about: ceiling/floor compression biases a difference-of-arms readout
  toward zero, which is fatal exactly when the adjudication rests on a null.

WHY A SUBSTRATE FLOOR AND NOT A RE-DERIVED R5 BAND

  The autopsy left the choice open ("either the warmup must leave headroom or
  R5's band must be re-derived for warmed agents; state which and why"). This
  module takes the first branch, for two reasons.

  1. Re-deriving the band would let the gate pass while the condition it
     exists to detect is untouched. R5 is not an arbitrary threshold; it is
     the assertion that the DV has dynamic range. Lowering E_SAT_LOW to admit
     0.0195 converts an artifact into a citable result -- the warning
     experiments/_lib/precondition_gate.py already carries, and the same
     failure the dv-dynamic-range-precondition-class harness gate (ree-v3
     8e133d26ed) was built to catch from the other end. A criterion whose bar
     sits outside the DV's achievable range is not a weaker criterion, it is
     an unmeasurable one.

  2. The collapse is a property of ANY sufficiently trained policy, not of
     one warmup recipe. Repairing it inside probe_warmup would be a
     per-experiment band-aid that the next lineage rediscovers the first time
     it warms an agent by some other route. The headroom belongs where the
     temperature is set.

THE MECHANISM

  A ONE-SIDED INTEGRAL CONTROLLER IN LOG-TEMPERATURE.

    1. Read the realised normalized entropy H(p)/ln(K) of the PREVIOUS waking
       tick's E3 pre-commit selection distribution (agent.e3.last_precommit_probs).
       Previous, not current: the temperature is an INPUT to that softmax, so
       the current tick's distribution does not exist yet when the temperature
       must be chosen. The controller is therefore a feedback loop with one
       tick of lag, which is why it is deliberately slow.
    2. Smooth it: h_ema = (1 - a) * h_ema + a * h_t.
    3. Integrate the one-sided error into a log-domain multiplier:
         h_ema <  target                -> log_mult += gain * (target - h_ema)
         h_ema >  target + deadband     -> log_mult -= gain * (h_ema - target - deadband)
         otherwise                      -> unchanged (deadband, no chatter)
       log_mult is clamped to [0, ln(max_temperature_ratio)].
    4. Emit multiplier = exp(log_mult) >= 1.0, applied to the tonic effective
       softmax temperature at the agent's select_action() site.

  THE MULTIPLIER NEVER FALLS BELOW 1.0. The floor can only ADD exploration,
  never remove it. A controller allowed to go below 1.0 would be a general
  entropy REGULATOR -- it would clamp the readout from both sides and destroy
  exactly the dynamic range this exists to protect, and it would silently
  cancel a tonic manipulation that is trying to raise entropy. One-sidedness
  is the property that makes this a HEADROOM floor rather than a set-point
  servo.

  THE CAP IS LOAD-BEARING. When the scores are so peaked that the target is
  unreachable within max_temperature_ratio, the controller saturates and
  says so (`saturated` in get_state()). It does NOT keep integrating. A
  consumer that finds `headroom_met` False with `saturated` True has learned
  something real -- the policy is too confident for this readout at this
  budget -- and should declare the cell uninformative rather than reporting
  the compressed number as a measurement.

WHERE IT IS APPLIED, AND WHY THERE

  agent.select_action(), AFTER the MECH-313 noise_floor tonic lift and BEFORE
  the SD-069 phasic burst:

      tonic_T -> noise_floor -> [SD-105 multiplier] -> phasic delta -> e3.select

  It is a TONIC mechanism, so it belongs on the tonic side. Placing it before
  the phasic delta means the phasic contribution is still an ADDITIVE delta in
  absolute temperature units on top of a lifted baseline -- the event-locked
  transient keeps its magnitude instead of being rescaled, which is what the
  MECH-063 (ii) dissociation readout requires. And because the floor is
  enabled identically in every arm of a tonic contrast, both arms are lifted
  together: dS_tonic is preserved rather than compressed, which is the whole
  point.

  The `noise_floor_temp` control-vector field deliberately continues to report
  the PRE-multiplier tonic temperature, so the existing MECH-313 readout is
  uncontaminated; the multiplier is reported separately.

ARCHITECTURE

  Pure-arithmetic regulator (cf. ree_core/policy/noise_floor.py MECH-313,
  ree_core/regulators/phasic_surprise_burst.py SD-069). No nn.Module, no
  learned parameters, no gradient flow. Phased training does not apply.

MECH-094

  simulation_mode=True returns the cached multiplier unchanged and does NOT
  advance the entropy EMA, the integrator, or the counters. Replay / DMN
  content must not move the waking exploration set-point -- matching the
  noise_floor / phasic_surprise_burst simulation_mode contract.

EPISODE BOUNDARIES

  reset() clears the PER-EPISODE diagnostics only. The entropy EMA and the
  integrator SURVIVE, for the reason SD-075 gave for the phasic baseline: a
  set-point that re-converges from cold at every episode boundary measures
  episode LENGTH rather than the policy's confidence, and a seed with short
  episodes would then get a systematically different lift from a seed with
  long ones -- reintroducing the 779b defect on a new axis. Use a fresh
  instance for a fresh lifetime. `continuity_note` in get_state() records
  this, since it is the opposite of the phasic regulator's DEFAULT and a
  reader is entitled to be surprised.

BLAST RADIUS

  use_selection_entropy_floor defaults False, the agent does not instantiate
  the regulator when it is False, and the call site adds nothing -- so every
  existing experiment is bit-identical.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional


def normalized_entropy(probs) -> "Optional[float]":
    """Normalized Shannon entropy H(p)/ln(K) of a probability vector.

    Accepts a torch tensor or any sequence of floats. Returns None when there
    are fewer than 2 outcomes (entropy is not defined against a zero-width
    range) or the vector does not sum to anything usable.

    Kept here, next to its only consumer, so the agent call site does not need
    a torch import path of its own and every consumer normalizes identically.
    """
    try:
        vals = [float(x) for x in probs.detach().reshape(-1).tolist()]
    except AttributeError:
        vals = [float(x) for x in probs]
    k = len(vals)
    if k < 2:
        return None
    total = sum(v for v in vals if v > 0.0)
    if total <= 0.0:
        return None
    h = 0.0
    for v in vals:
        if v > 0.0:
            q = v / total
            h -= q * math.log(q)
    return h / math.log(k)


@dataclass
class SelectionEntropyFloorConfig:
    """SD-105 selection-entropy headroom floor configuration.

    Attributes:
        enabled : independent-testability mirror of the agent-level
            use_selection_entropy_floor flag. The agent gates INSTANTIATION on
            that flag; holding it here lets the regulator be unit-tested in
            isolation.
        target : target normalized selection entropy H/ln(K) in (0, 1). 0.15
            sits inside the R5 saturation band (0.02, 0.98) and inside
            V3-EXQ-779a's own healthy baseline range (0.152-0.610).
        gain : integral gain on log-temperature per tick, applied to the
            entropy error. Larger converges faster and overshoots more.
        max_temperature_ratio : hard cap on the emitted multiplier, so the
            controller cannot run away when the target is unreachable.
        ema_decay : EMA rate for the realised-entropy estimate.
        deadband : one-sided band ABOVE target within which the integrator is
            left alone, so it does not chatter around the set-point.
    """

    enabled: bool = True
    target: float = 0.15
    gain: float = 0.5
    max_temperature_ratio: float = 8.0
    ema_decay: float = 0.2
    deadband: float = 0.05


class SelectionEntropyFloor:
    """SD-105 selection-entropy headroom floor (tonic variability set-point).

    Public API:
      observe(normalized_entropy_value, simulation_mode=False) -> float
        Fold in one realised-entropy observation and return the multiplier.
      temperature_multiplier -> float
        Cached multiplier, always >= 1.0.
      apply_to_temperature(tonic_temperature) -> float
        Convenience: tonic_temperature * temperature_multiplier.
      reset()
        Clear PER-EPISODE diagnostics. The EMA and integrator survive -- see
        the EPISODE BOUNDARIES block in the module docstring.
      get_state() / diagnostics -> dict
        Read-only snapshot for experiment manifests and telemetry.
    """

    def __init__(
        self, config: "Optional[SelectionEntropyFloorConfig]" = None
    ) -> None:
        self.config = config if config is not None else SelectionEntropyFloorConfig()
        c = self.config
        if not (0.0 < float(c.target) < 1.0):
            raise ValueError(
                "target must be in (0, 1) (normalized entropy H/ln(K)). Got "
                f"{c.target}."
            )
        if float(c.gain) <= 0.0:
            raise ValueError(f"gain must be > 0. Got {c.gain}.")
        if float(c.max_temperature_ratio) < 1.0:
            raise ValueError(
                "max_temperature_ratio must be >= 1.0 (the floor can only ADD "
                f"exploration). Got {c.max_temperature_ratio}."
            )
        if not (0.0 < float(c.ema_decay) <= 1.0):
            raise ValueError(
                f"ema_decay must be in (0, 1] (EMA rate). Got {c.ema_decay}."
            )
        if float(c.deadband) < 0.0:
            raise ValueError(f"deadband must be >= 0. Got {c.deadband}.")
        self._log_mult_cap: float = math.log(float(c.max_temperature_ratio))
        # Lifetime state -- survives reset() (see module docstring).
        self._log_mult: float = 0.0
        self._entropy_ema: float = 0.0
        self._ema_initialized: bool = False
        self._lifetime_ticks: int = 0
        self._n_ticks_below_target: int = 0
        self._n_ticks_saturated: int = 0
        # Per-episode diagnostics.
        self._last_entropy: float = 0.0
        self._n_observations: int = 0
        self._n_simulation_skips: int = 0

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def observe(
        self, normalized_entropy_value: float, simulation_mode: bool = False
    ) -> float:
        """Fold in one realised-entropy observation; return the multiplier.

        Args:
            normalized_entropy_value : realised H/ln(K) in [0, 1] from the
                previous waking tick's selection distribution. Values outside
                [0, 1] are clamped (a caller passing an unnormalized entropy
                is a bug, but silently integrating it would be worse).
            simulation_mode : MECH-094 gate. True -> return the cached
                multiplier; do NOT advance the EMA, the integrator, or any
                counter.

        Returns:
            temperature multiplier >= 1.0.
        """
        if not self.config.enabled:
            return float(self.temperature_multiplier)
        if simulation_mode:
            self._n_simulation_skips += 1
            return float(self.temperature_multiplier)

        h = max(0.0, min(1.0, float(normalized_entropy_value)))
        self._last_entropy = h
        self._n_observations += 1
        self._lifetime_ticks += 1

        a = float(self.config.ema_decay)
        if not self._ema_initialized:
            self._entropy_ema = h
            self._ema_initialized = True
        else:
            self._entropy_ema = (1.0 - a) * float(self._entropy_ema) + a * h

        target = float(self.config.target)
        deadband = float(self.config.deadband)
        gain = float(self.config.gain)
        h_ema = float(self._entropy_ema)

        if h_ema < target:
            self._n_ticks_below_target += 1
            self._log_mult += gain * (target - h_ema)
        elif h_ema > target + deadband:
            # Relax -- but never below a multiplier of 1.0. See the
            # ONE-SIDEDNESS paragraph in the module docstring.
            self._log_mult -= gain * (h_ema - target - deadband)
        # else: inside the deadband, integrator untouched.

        if self._log_mult < 0.0:
            self._log_mult = 0.0
        if self._log_mult > self._log_mult_cap:
            self._log_mult = self._log_mult_cap
        if (
            self._log_mult >= self._log_mult_cap - 1e-12
            and self._log_mult_cap > 0.0
        ):
            self._n_ticks_saturated += 1

        return float(self.temperature_multiplier)

    def apply_to_temperature(self, tonic_temperature: float) -> float:
        """Return the lifted tonic temperature (multiplier is always >= 1)."""
        return float(tonic_temperature) * float(self.temperature_multiplier)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    @property
    def temperature_multiplier(self) -> float:
        return float(math.exp(self._log_mult))

    @property
    def entropy_ema(self) -> float:
        return float(self._entropy_ema)

    @property
    def headroom_met(self) -> bool:
        """True once the smoothed realised entropy has reached the target."""
        return bool(self._ema_initialized and self._entropy_ema >= self.config.target)

    @property
    def saturated(self) -> bool:
        """True when the multiplier is pinned at max_temperature_ratio."""
        return bool(
            self._log_mult_cap > 0.0
            and self._log_mult >= self._log_mult_cap - 1e-12
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear PER-EPISODE diagnostics only.

        The entropy EMA and the integrator deliberately SURVIVE -- see the
        EPISODE BOUNDARIES block in the module docstring for why re-converging
        from cold at every boundary would reintroduce the V3-EXQ-779b
        episode-length confound on a new axis.
        """
        self._last_entropy = 0.0
        self._n_observations = 0
        self._n_simulation_skips = 0

    def get_state(self) -> Dict[str, object]:
        """Diagnostic snapshot for experiment manifests / telemetry probe."""
        return {
            "temperature_multiplier": float(self.temperature_multiplier),
            "log_multiplier": float(self._log_mult),
            "entropy_ema": float(self._entropy_ema),
            "last_entropy": float(self._last_entropy),
            "target": float(self.config.target),
            "deadband": float(self.config.deadband),
            "max_temperature_ratio": float(self.config.max_temperature_ratio),
            "headroom_met": bool(self.headroom_met),
            "saturated": bool(self.saturated),
            "lifetime_ticks": int(self._lifetime_ticks),
            "n_ticks_below_target": int(self._n_ticks_below_target),
            "n_ticks_saturated": int(self._n_ticks_saturated),
            "n_observations": int(self._n_observations),
            "n_simulation_skips": int(self._n_simulation_skips),
            "continuity_note": "ema_and_integrator_survive_reset",
        }

    @property
    def diagnostics(self) -> Dict[str, object]:
        return self.get_state()
