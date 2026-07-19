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
  advance the EMA baseline, decay the envelope, or increment counters
  (INCLUDING the SD-075 lifetime counters). Replay / DMN content must not
  trigger waking phasic arousal (matches the noise_floor / broadcast_override
  simulation_mode contract).

SD-075: EPISODE-BOUNDARY BASELINE CONTINUITY + CONVERGENCE-GATED ACCOUNTING

  THE DEFECT. As shipped, reset() cleared the surprise-EMA cold at every
  episode boundary, and the first waking tick of an episode can never fire an
  event (it seeds the baseline; `event_fired` requires `_ema_initialized`).
  With surprise_ema_decay 0.1 the baseline has a ~10-tick time constant. A
  seed whose episodes are SHORTER than that never runs against a converged
  baseline, so n_event_ticks becomes a function of episode LENGTH rather than
  of surprise.

  This is not hypothetical. In V3-EXQ-779b
  (v3_exq_779b_mech063_tonic_phasic_dissociation_20260718T233554Z_v3) seed 23
  ran ~6.9-step episodes while seeds 29/37 ran 300-step episodes -- a 43x
  spread. Raising seed 23's budget from 835 to 2400 env steps delivered those
  steps as 345 episodes of ~7, NOT as longer episodes, and the
  phasic_fires_real_events precondition did not move (6 vs threshold 10). No
  step-budget increase can reach this: the binding axis is episode length.
  Autopsy: REE_assembly evidence/planning/failure_autopsy_V3-EXQ-779b_2026-07-19.json.

  LEG (a) -- baseline_continuity="carry". reset() preserves _surprise_ema and
  _ema_initialized while still clearing the envelope, the cached delta, and
  the PER-EPISODE diagnostics. The baseline then reflects the agent's surprise
  distribution across its lifetime instead of the current episode's first
  tick.

  Biologically this is the faithful setting, and the shipped default is the
  divergence: LC baseline adaptation is continuous across behavioural
  episodes, and a baseline that resets at every episode boundary has no
  biological counterpart. "reset" remains the DEFAULT purely for
  bit-identical backward compatibility with every run recorded against
  SD-069 -- not because it is the better model. New work should declare
  "carry" deliberately.

  A partial-decay "warm" mode (re-seed the new episode's baseline with a
  fraction of the carried value) is DELIBERATELY NOT BUILT. It is a second
  knob with no question that needs it; "carry" is the clean form and the
  autopsy's requirement is satisfied by it.

  LEG (b) -- warmup_ticks. Lifetime tick/episode counters survive reset() in
  BOTH continuity modes. While _lifetime_ticks < resolved_warmup_ticks the
  baseline is treated as unconverged and event ticks accrue to
  n_events_prewarmup instead of n_events_converged.

  ACCOUNTING ONLY -- IT DOES NOT SUPPRESS THE BURST. During warmup the
  regulator still fires, still sets the envelope, and still perturbs the
  softmax temperature exactly as before; only the REPORTING splits. This is
  deliberate: suppressing the burst would change agent BEHAVIOUR in the first
  ticks of a lifetime, layering a second mechanism change on top of the
  continuity fix, and the MECH-063 (ii) retest would then confound the two.
  The defect being repaired is a MEASUREMENT defect, so the gate is a
  measurement instrument.

  WHAT A CONSUMER SHOULD DO. Read n_events_converged (not n_events) as the
  event count, and n_converged_ticks as its denominator. When
  n_converged_ticks is too small to support the read, declare the cell
  UNINFORMATIVE rather than reporting a near-zero count -- a MIN-across-cells
  precondition otherwise treats a starved cell as a real measurement, which is
  precisely how 779b was withheld.

  BLAST RADIUS. Both fields are no-op by default, so every existing consumer
  of phasic_surprise_burst is bit-identical. Note separately that ANY edit to
  ree_core busts every experiments/_lib/probe_warmup.py cache, because
  _warmup_key() hashes compute_substrate_hash(scope=None) -- the whole
  substrate. That is by design there (a false HIT corrupts a conclusion, a
  false MISS only wastes compute) and needs no WarmupRecipe change.
"""

from __future__ import annotations

import math
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
        baseline_continuity : SD-075 leg (a). "reset" (default, no-op) clears
            the surprise-EMA baseline at every episode boundary -- the SD-069
            shipping behaviour. "carry" preserves the baseline across
            reset(), so it reflects the agent's surprise distribution rather
            than the current episode's first tick. See the SD-075 block in
            the module docstring for why "carry" is the biologically faithful
            setting and "reset" is nonetheless the default.
        warmup_ticks : SD-075 leg (b). Number of LIFETIME waking ticks the
            baseline is treated as unconverged. 0 (default, no-op) = no
            gating. -1 = DERIVE as ceil(3 / surprise_ema_decay), i.e. three
            EMA time constants (30 ticks at the default decay 0.1). A
            positive value is used verbatim. Gates ACCOUNTING ONLY: the
            regulator still fires and still perturbs temperature during
            warmup; the event counts are merely split into pre-warmup and
            converged so a consumer can report honestly or declare the cell
            uninformative. See the SD-075 block for why it does not suppress.
    """

    enabled: bool = True
    surprise_ema_decay: float = 0.1
    trigger_ratio: float = 1.5
    trigger_floor: float = 1e-6
    temp_delta: float = -0.5
    decay: float = 0.5
    min_temperature: float = 0.1
    excess_saturation: float = 1.0
    # SD-075. Both no-op by default -> SD-069 behaviour is bit-identical.
    baseline_continuity: str = "reset"
    warmup_ticks: int = 0


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
        # SD-075 leg (a).
        if str(c.baseline_continuity) not in ("reset", "carry"):
            raise ValueError(
                "baseline_continuity must be 'reset' (per-episode cold clear, "
                "SD-069 default) or 'carry' (preserve the surprise-EMA across "
                f"episode boundaries, SD-075). Got {c.baseline_continuity!r}."
            )
        # SD-075 leg (b). 0 = OFF, -1 = DERIVE, positive = verbatim. Any other
        # negative value is a typo for -1 and must not silently mean OFF.
        wt = int(c.warmup_ticks)
        if wt < -1:
            raise ValueError(
                "warmup_ticks must be 0 (no gating), -1 (derive as "
                "ceil(3 / surprise_ema_decay)), or a positive tick count. Got "
                f"{wt}."
            )
        if wt == -1:
            wt = int(math.ceil(3.0 / float(c.surprise_ema_decay)))
        self._resolved_warmup_ticks: int = wt
        # SD-075: LIFETIME counters -- these survive reset() in BOTH continuity
        # modes, because the convergence question is about the regulator's
        # whole history, not the current episode.
        self._lifetime_ticks: int = 0
        self._lifetime_episodes: int = 0
        self._n_events_converged: int = 0
        self._n_events_prewarmup: int = 0
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
        # SD-075 leg (b): the convergence verdict for THIS tick is taken
        # BEFORE incrementing, so the first lifetime tick (which only seeds the
        # baseline and cannot fire) is counted as pre-warmup, not converged.
        tick_is_converged = self._lifetime_ticks >= self._resolved_warmup_ticks
        self._lifetime_ticks += 1
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
            # SD-075 leg (b): ACCOUNTING ONLY. The burst above is unchanged --
            # the envelope is set and the temperature will be perturbed
            # whether or not the baseline has converged. Only the bookkeeping
            # splits, so a consumer can exclude events measured against an
            # unconverged baseline without altering behaviour.
            if tick_is_converged:
                self._n_events_converged += 1
            else:
                self._n_events_prewarmup += 1
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
        """Clear per-episode envelope and diagnostics.

        SD-075 leg (a): whether the surprise-EMA baseline is ALSO cleared is
        governed by config.baseline_continuity.

          "reset" (default) -- clear it. SD-069 shipping behaviour, retained
              bit-identically. Each episode re-seeds its baseline from its own
              first tick, which is the defect documented in the module
              docstring; kept as the default only for backward compatibility.
          "carry" -- preserve it. The envelope, the cached temperature delta,
              and the per-episode diagnostics are still cleared, so an
              in-flight burst never leaks across an episode boundary; only the
              slow baseline persists, which is the LC-faithful behaviour.

        The SD-075 LIFETIME counters are never cleared here in either mode --
        they measure the regulator's whole history, which is what the
        convergence gate is a question about. Use a fresh instance for a fresh
        lifetime.
        """
        self._lifetime_episodes += 1
        if str(self.config.baseline_continuity) == "reset":
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
            # ---- SD-075 ----
            # Config echo, so a manifest reader can tell which regime ran
            # without re-deriving it from the experiment's config block.
            "baseline_continuity": str(self.config.baseline_continuity),
            "warmup_ticks": int(self._resolved_warmup_ticks),
            "warmup_ticks_derived": bool(int(self.config.warmup_ticks) == -1),
            # Lifetime accounting (survives reset() in both continuity modes).
            "lifetime_ticks": int(self._lifetime_ticks),
            "lifetime_episodes": int(self._lifetime_episodes),
            "baseline_converged": bool(
                self._lifetime_ticks >= self._resolved_warmup_ticks
            ),
            "n_converged_ticks": int(
                max(0, self._lifetime_ticks - self._resolved_warmup_ticks)
            ),
            "n_prewarmup_ticks": int(
                min(self._lifetime_ticks, self._resolved_warmup_ticks)
            ),
            # THE event count a consumer should read. n_events above is the
            # unsplit total and is retained only for SD-069 continuity.
            "n_events_converged": int(self._n_events_converged),
            "n_events_prewarmup": int(self._n_events_prewarmup),
        }

    # Alias for parity with broadcast_override's `.diagnostics` property.
    @property
    def diagnostics(self) -> Dict[str, object]:
        return self.get_state()
