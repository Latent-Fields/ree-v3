"""SD-069: phasic_surprise_burst (LC-NE phasic / adaptive-gain phasic mode).

PHASIC complement to MECH-313 stochastic_noise_floor (LC-NE tonic) on the
SAME E3 softmax-temperature channel. Together they instantiate MECH-063
sub-claim (ii): each control axis carries BOTH a slow tonic baseline AND a
fast phasic event-burst as independent, independently-toggleable degrees of
freedom on comparable readouts.

RELATIONSHIP TO MECH-104 (important -- do not conflate)

  This regulator reuses the MECH-104 volatility-surprise LIT BASIS (LC-NE
  phasic burst on unexpected/surprising events; Aston-Jones & Cohen 2005
  adaptive-gain model, phasic mode). It does NOT implement the MECH-104
  CLAIM. The active, evidenced MECH-104 claim
  (control_plane.volatility_interrupt, v3_exq_365) is the volatility spike
  routing to the ARC-016 commit / de-commit gate (e3_selector commit
  uncertainty). THIS module routes the same surprise event to the E3
  SELECTION softmax temperature instead -- the tonic/phasic axis of
  MECH-063, NOT the commit gate. Same biological substrate, same source
  signal, different consumer.

ARCHITECTURE

  Pure-arithmetic regulator (cf. ree_core/policy/noise_floor.py MECH-313 and
  ree_core/regulators/broadcast_override.py SD-037). No nn.Module, no learned
  parameters, no gradient flow.

  Per waking tick:

    1. surprise s_t = e3._running_variance (per-tick PE-MSE accumulator; the
       SAME signal MECH-314c learning-progress and MECH-320 recent_pe read,
       and the signal experiments already poke to fake MECH-104).
    2. Event test: an event fires when
           s_t >= trigger_ratio * max(ema_baseline, trigger_floor)
       i.e. a relative spike over the running EMA baseline of surprise, with
       an absolute floor so a quiescent (~0) stream cannot fire on numerical
       noise.
    3. On an event, an injection drive in [0, 1] is computed from the
       normalized surprise excess and the burst envelope is set to the MAX of
       its decayed previous value and the new drive (a fresh, larger event
       re-arms the transient; a smaller one does not cut a still-decaying
       burst short).
    4. The envelope decays geometrically every tick: level *= (1 - decay).
    5. EMA baseline is advanced with s_t AFTER the event test (so the tick's
       own spike does not pre-absorb into the baseline it is compared to).

  Output consumed at the E3 select() call site in
  REEAgent.select_action():

      temperature_delta = temp_delta * burst_level
      combined_T = max(tonic_effective_T + temperature_delta,
                       phasic_min_temperature)

  Default temp_delta is NEGATIVE: a phasic burst transiently SHARPENS the
  softmax (LC-NE phasic gain increase; "phasic mode gates committed
  exploitation" -- the reading noise_floor.py already commits to). The sign
  and magnitude are config-exposed; the load-bearing property for MECH-063
  (ii) is that the phasic contribution is EVENT-LOCKED and TRANSIENT, versus
  the tonic noise_floor's sustained every-tick offset.

DISTINCTION FROM MECH-313 (tonic)

  MECH-313 noise_floor: lifts the softmax temperature by a CONSTANT amount
  every waking tick (state-independent, sustained). Readout = a flat baseline
  offset present on every tick.

  SD-069 (this module): adds a TRANSIENT temperature delta only in the ticks
  following a surprise event, decaying to zero within a few ticks. Readout =
  a spiky, event-locked deviation that is ~0 on quiescent ticks.

  The two are independently toggleable (separate use_* flags) and act on the
  SAME effective-temperature readout, which is what makes the tonic-vs-phasic
  behavioural dissociation (MECH-063 ii) measurable.

MECH-094

  simulation_mode=True returns the cached burst_level unchanged and does NOT
  advance the EMA baseline, decay the envelope, or increment counters. Replay
  / DMN content must not trigger waking phasic arousal (matches the
  noise_floor / broadcast_override simulation_mode contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PhasicSurpriseBurstConfig:
    """SD-069 phasic surprise-burst configuration.

    Attributes:
        enabled : independent-testability mirror of the agent-level
            use_phasic_burst flag. The agent gates INSTANTIATION on
            use_phasic_burst; holding the flag here lets the regulator be
            unit-tested in isolation.
        surprise_ema_decay : EMA rate for the surprise baseline the event
            detector compares against. 0.1 ~= 20-tick EMA.
        trigger_ratio : event fires when surprise >= trigger_ratio *
            max(ema_baseline, trigger_floor). 1.5 = a 50% spike.
        trigger_floor : absolute floor on the baseline in the trigger test,
            so an ~0 baseline cannot fire on numerical noise.
        temp_delta : temperature delta at a full (level=1.0) burst. NEGATIVE
            = sharpening. Applied delta = temp_delta * burst_level.
        decay : geometric decay retained per tick on the envelope
            (level *= (1 - decay)). 0.5 halves every tick.
        min_temperature : hard lower bound on the COMBINED effective softmax
            temperature (enforced at the agent call site; carried here for
            the regulator's own clamp helper and ablation symmetry).
        excess_saturation : surprise-excess (s_t / baseline - trigger_ratio)
            at which the injection drive saturates to 1.0. Larger = a burst
            needs a bigger spike to reach full amplitude.
    """

    enabled: bool = True
    surprise_ema_decay: float = 0.1
    trigger_ratio: float = 1.5
    trigger_floor: float = 1e-6
    temp_delta: float = -0.5
    decay: float = 0.5
    min_temperature: float = 0.1
    excess_saturation: float = 1.0


class PhasicSurpriseBurst:
    """SD-069 phasic surprise-burst regulator (LC-NE phasic).

    Public API:
      tick(surprise, simulation_mode=False) -> float
        Advance one waking tick; return the current burst_level in [0, 1].
      temperature_delta -> float
        Cached transient temperature delta = temp_delta * burst_level.
      apply_to_temperature(tonic_temperature) -> float
        Convenience clamp: max(tonic_temperature + temperature_delta,
        min_temperature). The agent applies this at the e3.select() site.
      reset()
        Clear per-episode state (EMA baseline, envelope, diagnostics).
      get_state() / diagnostics -> dict
        Read-only snapshot for experiment manifests and the control-vector
        telemetry probe.
    """

    def __init__(self, config: "Optional[PhasicSurpriseBurstConfig]" = None) -> None:
        self.config = config if config is not None else PhasicSurpriseBurstConfig()
        c = self.config
        if not (0.0 < float(c.surprise_ema_decay) <= 1.0):
            raise ValueError(
                "surprise_ema_decay must be in (0, 1] (EMA rate). Got "
                f"{c.surprise_ema_decay}."
            )
        if float(c.trigger_ratio) < 1.0:
            raise ValueError(
                "trigger_ratio must be >= 1.0 (an event is a spike ABOVE the "
                f"running baseline). Got {c.trigger_ratio}."
            )
        if float(c.trigger_floor) <= 0.0:
            raise ValueError(
                f"trigger_floor must be > 0. Got {c.trigger_floor}."
            )
        if not (0.0 < float(c.decay) <= 1.0):
            raise ValueError(
                "decay must be in (0, 1] (per-tick geometric decay of the "
                f"envelope). Got {c.decay}."
            )
        if float(c.min_temperature) <= 0.0:
            raise ValueError(
                "min_temperature must be > 0 (softmax temperature is strictly "
                f"positive). Got {c.min_temperature}."
            )
        if float(c.excess_saturation) <= 0.0:
            raise ValueError(
                f"excess_saturation must be > 0. Got {c.excess_saturation}."
            )
        # Per-episode state.
        self._surprise_ema: float = 0.0
        self._ema_initialized: bool = False
        self._burst_level: float = 0.0
        self._temperature_delta: float = 0.0
        # Diagnostics.
        self._last_surprise: float = 0.0
        self._last_event_fired: bool = False
        self._n_events: int = 0
        self._n_waking_ticks: int = 0
        self._n_simulation_skips: int = 0

    # ------------------------------------------------------------------
    # Forward path
    # ------------------------------------------------------------------
    def tick(self, surprise: float, simulation_mode: bool = False) -> float:
        """Advance one waking tick and return the current burst level.

        Args:
            surprise : current per-tick surprise magnitude (e.g.
                e3._running_variance). Negative inputs are clamped to 0.
            simulation_mode : MECH-094 gate. True -> return the cached
                burst_level unchanged; do NOT advance the EMA baseline, decay
                the envelope, or increment counters.

        Returns:
            burst_level in [0, 1].
        """
        if not self.config.enabled:
            return float(self._burst_level)
        if simulation_mode:
            self._n_simulation_skips += 1
            return float(self._burst_level)

        s_t = max(0.0, float(surprise))
        self._n_waking_ticks += 1
        self._last_surprise = s_t

        # Decay the existing envelope first (a tick with no event still lets
        # a prior burst decay toward zero).
        decayed = float(self._burst_level) * (1.0 - float(self.config.decay))

        # Event test against the CURRENT baseline (before folding s_t in).
        baseline = self._surprise_ema if self._ema_initialized else 0.0
        eff_baseline = max(float(baseline), float(self.config.trigger_floor))
        threshold = float(self.config.trigger_ratio) * eff_baseline

        event_fired = self._ema_initialized and s_t >= threshold
        if event_fired:
            # Normalized surprise excess -> injection drive in [0, 1].
            # excess = s_t / eff_baseline - trigger_ratio, saturating at
            # excess_saturation.
            ratio = s_t / eff_baseline
            excess = ratio - float(self.config.trigger_ratio)
            drive = excess / float(self.config.excess_saturation)
            drive = max(0.0, min(1.0, drive))
            # A fresh, larger event re-arms; a smaller one does not truncate a
            # still-decaying burst.
            new_level = max(decayed, drive)
            self._n_events += 1
        else:
            new_level = decayed

        self._burst_level = max(0.0, min(1.0, new_level))
        self._temperature_delta = float(self.config.temp_delta) * self._burst_level
        self._last_event_fired = bool(event_fired)

        # Advance the EMA baseline with this tick's surprise AFTER the event
        # test (so the spike does not pre-absorb into its own threshold).
        a = float(self.config.surprise_ema_decay)
        if not self._ema_initialized:
            self._surprise_ema = s_t
            self._ema_initialized = True
        else:
            self._surprise_ema = (1.0 - a) * float(self._surprise_ema) + a * s_t

        return float(self._burst_level)

    def apply_to_temperature(self, tonic_temperature: float) -> float:
        """Return the combined effective temperature, floored strictly > 0.

        combined = max(tonic_temperature + temperature_delta, min_temperature)

        The agent applies this at the e3.select() call site AFTER the tonic
        MECH-313 noise_floor has produced tonic_temperature.
        """
        combined = float(tonic_temperature) + float(self._temperature_delta)
        return max(combined, float(self.config.min_temperature))

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    @property
    def burst_level(self) -> float:
        return float(self._burst_level)

    @property
    def temperature_delta(self) -> float:
        return float(self._temperature_delta)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear per-episode EMA baseline, envelope, and diagnostics."""
        self._surprise_ema = 0.0
        self._ema_initialized = False
        self._burst_level = 0.0
        self._temperature_delta = 0.0
        self._last_surprise = 0.0
        self._last_event_fired = False
        self._n_events = 0
        self._n_waking_ticks = 0
        self._n_simulation_skips = 0

    def get_state(self) -> Dict[str, object]:
        """Diagnostic snapshot for experiment manifests / telemetry probe."""
        return {
            "burst_level": float(self._burst_level),
            "temperature_delta": float(self._temperature_delta),
            "surprise_ema": float(self._surprise_ema),
            "last_surprise": float(self._last_surprise),
            "last_event_fired": bool(self._last_event_fired),
            "n_events": int(self._n_events),
            "n_waking_ticks": int(self._n_waking_ticks),
            "n_simulation_skips": int(self._n_simulation_skips),
        }

    # Alias for parity with broadcast_override's `.diagnostics` property.
    @property
    def diagnostics(self) -> Dict[str, object]:
        return self.get_state()
