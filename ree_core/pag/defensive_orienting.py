"""
MECH-489 (SD-099): PAG defensive-orienting response.

Architectural commitment (see
REE_assembly/docs/architecture/sd_094_defensive_orienting_response.md):

  A phasic, unidentified-stimulus sibling of MECH-279 (pag/freeze_gate.py).
  Where MECH-279 freezes on ACCUMULATED harm/suffering (z_harm_a sustained
  over ticks), this gate freezes on a SUDDEN, UNEXPECTED, not-yet-identified
  onset -- the Sokolov orienting-reflex shape. It is deliberately a SEPARATE
  gate, not a modification of PAGFreezeGate: the two freeze causes are
  architecturally distinct and composed via OR at the action-constraint site
  (ree_core/agent.py select_action()), so PAGFreezeGate's existing behaviour
  is unchanged bit-for-bit.

Trigger (resolves observational_review_V3-EXQ-906b Section 12h's finding that
a naive `residue_surprise > p90` design under-fires on the ground-truth
injected events it exists to catch): a positive-derivative / onset detector
over TWO already-phasic substrate channels, not one absolute threshold --

  residue_surprise (unsigned VALENCE_SURPRISE, MECH-205 "surprise") -- catches
    non-nociceptive unexpected world-events (external_hazard_injected,
    delayed on world_rule_shift_occurred).
  z_harm_s norm (SD-010 sensory-discriminative harm, LatentState.z_harm) --
    catches nociceptive events (limb_damage_injected); Section 12g established
    this channel (NOT the chronic z_harm_a MECH-279 reads) has a clean
    event-locked phasic profile.

  trigger(t) = (residue_surprise(t) - surprise_baseline(t) > onset_delta_surprise)
            OR (z_harm_s_norm(t) - harm_s_baseline(t) > onset_delta_harm_s)
  baselines are slow EMAs, updated ONLY while NOT orienting -- frozen at
  their pre-trigger value for the duration of an active orienting episode,
  so the "has this subsided" read below measures genuine decay of the raw
  signal, not the baseline creeping up to chase a sustained elevation.

Logic:

  On trigger (only when not already orienting): orienting_active = True,
  identification_confidence = 0.0, peak excess recorded for the tick.

  While active: identification_confidence rises toward 1.0 at a rate scaled
  by how much the triggering channel's excess-over-baseline has decayed --
  NOT a fixed timer. A sustained, non-decaying elevation stalls confidence
  (freeze stays open-ended); a spike that resolves quickly resolves fast.

  Override (freeze release): identification_confidence >= sufficiency_threshold.
  Distinct from the SD-037 orexin `override` channel (broadcast-recruitment,
  a different mechanism) -- this is freeze-release-on-epistemic-sufficiency.

  Optional safety valve: max_orienting_duration ticks (0 = no cap, mirrors
  PAGFreezeGate's own max_freeze_duration=0 "no cap" convention -- 11b's
  design explicitly calls for an open-ended hold, not a fixed pulse).

Non-trainable: pure arithmetic over scalars and small counters. No gradient
flow. Reset per episode. The action-decision (approach/withdraw/resume) step
is deliberately NOT performed here -- it needs residue-field tensor reads
(benefit terrain vs z_harm) that belong at the agent level, mirroring
PAGFreezeGate's own scope of "pure arithmetic over scalars, nothing reaches
into the residue field."

Master switch: REEConfig.use_defensive_orienting (default False) gates
instantiation and wiring. With the flag off, agents behave bit-identically to
legacy.

MECH-094: simulation_mode=True ticks return a zeroed-output (orienting
inactive) without updating internal state -- baselines, confidence, and
trigger/override counters are all frozen. Replay / DMN content must not
commit the agent into an orienting-arrest state or silently advance
identification progress that only real waking perception should earn.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DefensiveOrientingConfig:
    """MECH-489 (SD-099) defensive-orienting configuration.

    All defaults are first-pass calibration seeded from
    observational_review_V3-EXQ-906b Section 12g/12h's own event-triggered
    numbers (not blind guesses), and are explicitly subject to revision by
    the validation experiment -- see the SD doc's "Trigger channel
    derivation" section for the reasoning.
    """

    # Master flag, mirrored on REEConfig. Held here too so the gate is
    # independently testable from the agent (same convention as
    # PAGFreezeGateConfig.enabled).
    enabled: bool = True

    # Slow EMA rates for the two rolling baselines. Low alpha so a single
    # spike does not itself immediately drag its own baseline up (which
    # would self-mask the onset detector).
    surprise_ema_alpha: float = 0.02
    harm_s_ema_alpha: float = 0.02

    # Onset thresholds: trigger fires when (channel - its baseline) exceeds
    # this delta. Seeded from 12h (residue_surprise external_hazard event
    # delta ~0.010) and 12g (z_harm_s phasic-bump range ~0.006-0.02).
    surprise_onset_delta: float = 0.010
    harm_s_onset_delta: float = 0.010

    # Identification-confidence dynamics. rise_rate scales the per-tick gain
    # by (1 - residual_excess), so confidence rises FASTER as the triggering
    # elevation decays back toward baseline. floor_rise is an optional
    # unconditional per-tick addition (0.0 = purely decay-driven, matching
    # 11b's "not a fixed timer" design).
    confidence_rise_rate: float = 0.15
    confidence_floor_rise: float = 0.0
    sufficiency_threshold: float = 0.8

    # Optional safety-valve cap on orienting duration (ticks). 0 = no cap,
    # mirroring PAGFreezeGateConfig.max_freeze_duration=0.
    max_orienting_duration: int = 0


@dataclass
class DefensiveOrientingOutput:
    """Per-tick DefensiveOrientingGate output."""

    orienting_active: bool = False
    # True on the tick where orienting commits this step (newly triggered).
    trigger_fired: bool = False
    # True on the tick where the freeze releases (epistemic sufficiency).
    override_fired: bool = False
    identification_confidence: float = 0.0
    ticks_in_orienting: int = 0
    # Which channel(s) crossed onset this tick: "surprise" | "harm_s" |
    # "both" | "" (empty when no trigger this tick).
    trigger_channel: str = ""
    surprise_delta: float = 0.0
    harm_s_delta: float = 0.0


class DefensiveOrientingGate:
    """MECH-489 (SD-099) PAG-analog phasic defensive-orienting gate.

    Public API:
      tick(residue_surprise_now, z_harm_s_norm_now, simulation_mode=False)
        -> DefensiveOrientingOutput
        Compute the orienting state for this step.

      reset()
        Clear per-episode state.

      is_active        Convenience property mirroring last orienting_active.

      diagnostics       Dict of running counters.

    State (per episode):
      _surprise_baseline, _harm_s_baseline   float, slow EMA baselines
      _orienting_active                      bool, current orienting status
      _identification_confidence             float, in [0, 1]
      _ticks_in_orienting                    int, how long orienting has been active
      _peak_excess                           float, largest onset excess recorded
                                              since the current trigger (used to
                                              normalise the decay-driven confidence rise)
      _last_output                           DefensiveOrientingOutput, last tick's output
      _n_ticks, _n_triggers, _n_overrides    diagnostic counters
    """

    def __init__(self, config: Optional[DefensiveOrientingConfig] = None):
        self.config = config or DefensiveOrientingConfig()

        self._surprise_baseline: float = 0.0
        self._harm_s_baseline: float = 0.0
        self._orienting_active: bool = False
        self._identification_confidence: float = 0.0
        self._ticks_in_orienting: int = 0
        self._peak_excess: float = 0.0
        self._last_output: DefensiveOrientingOutput = DefensiveOrientingOutput()

        self._n_ticks: int = 0
        self._n_triggers: int = 0
        self._n_overrides: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear per-episode state."""
        self._surprise_baseline = 0.0
        self._harm_s_baseline = 0.0
        self._orienting_active = False
        self._identification_confidence = 0.0
        self._ticks_in_orienting = 0
        self._peak_excess = 0.0
        self._last_output = DefensiveOrientingOutput()

    @property
    def is_active(self) -> bool:
        return bool(self._orienting_active)

    # -- Tick: main per-step computation --

    def tick(
        self,
        residue_surprise_now: float,
        z_harm_s_norm_now: float,
        simulation_mode: bool = False,
    ) -> DefensiveOrientingOutput:
        """Compute the defensive-orienting gate state for this step.

        Args:
            residue_surprise_now: Current unsigned VALENCE_SURPRISE magnitude
                (MECH-205 "surprise"; cached from the previous tick's
                update_residue() call -- see agent.py wiring notes).
            z_harm_s_norm_now: Current L2 norm of the SD-010
                sensory-discriminative harm latent (LatentState.z_harm).
            simulation_mode: MECH-094 hypothesis-tag equivalent. True ->
                return a zeroed output and do not update internal state.
                Replay / DMN content must not commit the agent into
                orienting-arrest or advance identification confidence.

        Returns:
            DefensiveOrientingOutput with orienting_active, trigger/override
            edges, identification_confidence, and diagnostic fields.
        """
        if not self.config.enabled:
            out = DefensiveOrientingOutput(
                orienting_active=False,
                trigger_fired=False,
                override_fired=False,
                identification_confidence=float(self._identification_confidence),
                ticks_in_orienting=0,
            )
            self._last_output = out
            return out

        if simulation_mode:
            # MECH-094 gate: replay / simulation -- return zeroed and do NOT
            # update baselines, confidence, or entry/exit counters.
            out = DefensiveOrientingOutput(
                orienting_active=False,
                trigger_fired=False,
                override_fired=False,
                identification_confidence=float(self._identification_confidence),
                ticks_in_orienting=int(self._ticks_in_orienting),
            )
            self._last_output = out
            return out

        self._n_ticks += 1

        surprise = float(residue_surprise_now)
        harm_s = float(z_harm_s_norm_now)

        # 1. Update slow rolling baselines ONLY while NOT orienting (mirrors
        #    PAGFreezeGate's own "the duration counter only accumulates while
        #    the gate is INACTIVE" pattern). This is load-bearing, not
        #    cosmetic: if the baseline kept chasing the raw signal upward
        #    while active, a genuinely SUSTAINED, non-resolving elevation
        #    would still show its excess-over-baseline shrink as the EMA
        #    caught up -- silently "resolving" a threat that never actually
        #    subsided, which is exactly the fixed-timer-in-disguise behaviour
        #    11b step 2/4 rules out ("not a fixed timer... open-ended while
        #    the unknown remains unidentified"). Freezing the baseline at
        #    its pre-trigger value means the confidence dynamics below are
        #    measuring genuine decay of the RAW signal, not baseline creep.
        if not self._orienting_active:
            a_s = float(self.config.surprise_ema_alpha)
            a_h = float(self.config.harm_s_ema_alpha)
            self._surprise_baseline = (
                (1 - a_s) * self._surprise_baseline + a_s * surprise
            )
            self._harm_s_baseline = (1 - a_h) * self._harm_s_baseline + a_h * harm_s

        surprise_delta = surprise - self._surprise_baseline
        harm_s_delta = harm_s - self._harm_s_baseline

        trigger_this_tick = False
        override_this_tick = False
        trigger_channel = ""

        # 2. Entry check (only when not already orienting -- a new trigger
        #    is refused while epistemic sufficiency has not yet been
        #    reached for the current one).
        if not self._orienting_active:
            surprise_fires = surprise_delta > float(self.config.surprise_onset_delta)
            harm_s_fires = harm_s_delta > float(self.config.harm_s_onset_delta)
            if surprise_fires or harm_s_fires:
                self._orienting_active = True
                self._ticks_in_orienting = 0  # will increment to 1 below
                self._identification_confidence = 0.0
                self._peak_excess = max(
                    surprise_delta if surprise_fires else 0.0,
                    harm_s_delta if harm_s_fires else 0.0,
                    1e-8,  # floor: avoid div-by-zero in the residual read below
                )
                trigger_this_tick = True
                self._n_triggers += 1
                if surprise_fires and harm_s_fires:
                    trigger_channel = "both"
                elif surprise_fires:
                    trigger_channel = "surprise"
                else:
                    trigger_channel = "harm_s"

        # 3. Identification-confidence dynamics + override check (only when
        #    active and not the tick that just triggered -- the freshly
        #    triggered tick starts at confidence 0.0 by construction).
        if self._orienting_active and not trigger_this_tick:
            current_excess = max(surprise_delta, harm_s_delta, 0.0)
            residual = min(1.0, current_excess / self._peak_excess)
            gain = float(self.config.confidence_rise_rate) * (1.0 - residual)
            gain += float(self.config.confidence_floor_rise)
            self._identification_confidence = min(
                1.0, max(0.0, self._identification_confidence + gain)
            )

            min_dur_met = self._identification_confidence >= float(
                self.config.sufficiency_threshold
            )
            max_dur_cap = (
                int(self.config.max_orienting_duration) > 0
                and self._ticks_in_orienting >= int(self.config.max_orienting_duration)
            )
            if min_dur_met or max_dur_cap:
                self._orienting_active = False
                override_this_tick = True
                self._n_overrides += 1
                self._ticks_in_orienting = 0
                self._identification_confidence = 0.0
                self._peak_excess = 0.0

        # 4. Tick the in-orienting counter when active.
        if self._orienting_active:
            self._ticks_in_orienting += 1

        out = DefensiveOrientingOutput(
            orienting_active=bool(self._orienting_active),
            trigger_fired=bool(trigger_this_tick),
            override_fired=bool(override_this_tick),
            identification_confidence=float(self._identification_confidence),
            ticks_in_orienting=int(self._ticks_in_orienting),
            trigger_channel=trigger_channel,
            surprise_delta=float(surprise_delta),
            harm_s_delta=float(harm_s_delta),
        )
        self._last_output = out
        return out

    # -- Read-only accessors --

    @property
    def last_output(self) -> DefensiveOrientingOutput:
        return self._last_output

    @property
    def diagnostics(self) -> dict:
        return {
            "n_ticks": int(self._n_ticks),
            "n_triggers": int(self._n_triggers),
            "n_overrides": int(self._n_overrides),
            "orienting_active": bool(self._orienting_active),
            "identification_confidence": float(self._identification_confidence),
            "ticks_in_orienting": int(self._ticks_in_orienting),
            "surprise_baseline": float(self._surprise_baseline),
            "harm_s_baseline": float(self._harm_s_baseline),
        }
