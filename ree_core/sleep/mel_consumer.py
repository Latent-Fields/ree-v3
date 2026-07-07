"""
SD-MEL-CONSUMER (sleep_substrate:GAP-5b): adaptive sleep-cadence MEL consumer.

The INV-050 THIRD / learning-demand sleep drive. Reads accumulated waking Model
Error Load (MEL = mean per-step e3 prediction error over the wake period, the same
signal V3-EXQ-701c demonstrated is measurable + monotone in graded novelty) and
modulates the offline (sleep) phase:

  * DURATION lever (primary): scales sws_consolidation_steps and/or
    rem_attribution_steps for a cycle by a relative, scale-free factor
        factor = clamp(1 + mel_gain * (mel/ref - 1), factor_min, factor_max)
    so high-MEL wake periods produce measurably more SWS schema writes + REM
    attribution rollouts -- the exact V3-EXQ-677 DV (cumulative_sws_writes /
    cumulative_rem_rollouts) that was scheduler-pinned (SWS=80/REM=60, zero
    cross-arm variance) on the K-episode-deterministic substrate.

  * ENTRY-timing lever (secondary): fire a cycle once accumulated MEL crosses
    mel_entry_threshold, with the K-episode counter as a safety-backstop ceiling.

DISTINCT from the SD-037 arousal entry gate (sleep_onset_gate.py / MECH-286 /
sleep_substrate:GAP-5, V4-deferred): that drive keys off override/staleness/threat;
this one keys off learning demand (MEL). They compose -- the MEL duration factor
applies to a cycle the MECH-286 gate has permitted.

Instrument-floor note (V3-EXQ-701c): the response is RELATIVE (mel/ref), never an
absolute spread gate. The 701c instrument's ABS_MEL_FLOOR=1e-4 was ~5x the entire
converged-base MEL magnitude (~2e-5); here relative_floor (~1e-6) only guards the
mel/ref division against ref ~ 0.

MECH-094: reads waking prediction error only (accumulation is gated on the caller's
hypothesis_tag=False waking step) and writes nothing to memory during non-waking
states. It only changes how many existing SWS/REM passes run.

See REE_assembly/docs/architecture/sd_mel_consumer.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


@dataclass
class MELConsumerConfig:
    """Knobs for the MEL consumer (mirrored on REEConfig)."""

    mel_gain: float = 1.0
    mel_reference: float = 0.0
    mel_reference_mode: str = "fixed"
    mel_ema_alpha: float = 0.1
    mel_duration_factor_min: float = 0.5
    mel_duration_factor_max: float = 3.0
    mel_relative_floor: float = 1e-6
    mel_scale_sws: bool = True
    mel_scale_rem: bool = True
    use_mel_entry: bool = False
    mel_entry_threshold: float = 0.0


class WakingMELAccumulator:
    """Accumulates per-step waking e3 prediction error over a wake window.

    MEL is the MEAN per-step prediction error (matches the V3-EXQ-701c
    instrument), so the window length does not bias the load estimate.
    """

    def __init__(self) -> None:
        self._sum: float = 0.0
        self._count: int = 0

    def accumulate(self, prediction_error: float) -> None:
        pe = float(prediction_error)
        if not math.isfinite(pe):
            return
        # PE magnitude: e3 prediction error is a non-negative MSE-like scalar,
        # but guard against any signed inputs so the load is a magnitude.
        self._sum += abs(pe)
        self._count += 1

    @property
    def count(self) -> int:
        return self._count

    def mean(self) -> float:
        if self._count == 0:
            return 0.0
        return self._sum / float(self._count)

    def reset(self) -> None:
        self._sum = 0.0
        self._count = 0


class MELConsumer:
    """Reads accumulated waking MEL and modulates offline-phase entry/duration."""

    def __init__(self, config: MELConsumerConfig) -> None:
        self.config = config
        self.accumulator = WakingMELAccumulator()
        # EMA / auto-calibrated reference set-point (per-step PE units). None
        # until the first cycle establishes it (fixed-auto or ema seed).
        self._reference_state: float | None = None
        # Diagnostics from the last computed factor.
        self._last_mel: float = 0.0
        self._last_reference: float = 0.0
        self._last_factor: float = 1.0

    # -- waking read path --

    def note_step_pe(self, prediction_error: float) -> None:
        """Accumulate one waking step's e3 prediction error.

        Caller (REEAgent.update_residue) invokes this ONLY on waking steps
        (hypothesis_tag=False), so replay / simulation PE never enters MEL.
        """
        self.accumulator.accumulate(prediction_error)

    def current_mel(self) -> float:
        """Mean per-step waking PE accumulated since the last cycle."""
        return self.accumulator.mean()

    # -- reference set-point --

    def _effective_reference(self, mel: float) -> float:
        """Resolve the homeostatic set-point for this cycle.

        fixed: constant mel_reference; if 0.0 (sentinel) auto-calibrate to the
               first cycle's MEL and hold it.
        ema:   slow EMA of per-cycle MEL, seeded on the first cycle.

        Floored at mel_relative_floor so mel/ref never divides by ~0.
        """
        cfg = self.config
        mode = str(cfg.mel_reference_mode)
        if mode == "ema":
            ref = self._reference_state if self._reference_state is not None else mel
        else:  # "fixed" (default) -- any unrecognised mode falls back to fixed
            if float(cfg.mel_reference) > 0.0:
                ref = float(cfg.mel_reference)
            elif self._reference_state is not None:
                ref = self._reference_state
            else:
                ref = mel  # auto-calibrate to first cycle
        return max(float(ref), float(cfg.mel_relative_floor))

    def duration_factor(self) -> float:
        """Relative, scale-free MEL -> offline-duration multiplier.

        factor = clamp(1 + gain*(mel/ref - 1), factor_min, factor_max).
        Returns 1.0 when no waking PE was accumulated (no signal to act on).
        """
        cfg = self.config
        mel = self.current_mel()
        ref = self._effective_reference(mel)
        self._last_mel = mel
        self._last_reference = ref
        if self.accumulator.count == 0 or ref <= 0.0:
            self._last_factor = 1.0
            return 1.0
        raw = 1.0 + float(cfg.mel_gain) * (mel / ref - 1.0)
        factor = max(
            float(cfg.mel_duration_factor_min),
            min(float(cfg.mel_duration_factor_max), raw),
        )
        if not math.isfinite(factor) or factor <= 0.0:
            factor = 1.0
        self._last_factor = factor
        return factor

    def scale_steps(self, base_steps: int) -> int:
        """Apply the current duration factor to a per-cycle step count.

        Floored at 1 so a scaled-down cycle still runs at least one pass.
        """
        scaled = int(round(int(base_steps) * self._last_factor))
        return max(1, scaled)

    # -- entry-timing lever --

    def entry_permitted(self, episodes_since_sleep: int, k_ceiling: int) -> bool:
        """Whether a cycle should fire this episode boundary.

        With the entry lever OFF: strict K-episode schedule (>= ceiling).
        With it ON: fire when accumulated MEL >= threshold OR the K ceiling is
        hit (safety backstop so sleep is never starved indefinitely).
        """
        at_ceiling = episodes_since_sleep >= int(k_ceiling)
        if not self.config.use_mel_entry:
            return at_ceiling
        mel = self.current_mel()
        crossed = (
            self.accumulator.count > 0
            and mel >= float(self.config.mel_entry_threshold)
        )
        return crossed or at_ceiling

    # -- lifecycle --

    def on_cycle_complete(self) -> None:
        """Update the EMA reference (if any) and reset the accumulator."""
        cfg = self.config
        mel = self.current_mel()
        if self.accumulator.count > 0:
            if str(cfg.mel_reference_mode) == "ema":
                if self._reference_state is None:
                    self._reference_state = mel
                else:
                    a = float(cfg.mel_ema_alpha)
                    self._reference_state = (1.0 - a) * self._reference_state + a * mel
            elif self._reference_state is None and float(cfg.mel_reference) <= 0.0:
                # fixed-auto: lock the set-point to the first cycle's MEL.
                self._reference_state = mel
        self.accumulator.reset()

    def reset(self) -> None:
        """Hard reset (e.g. between training stages)."""
        self.accumulator.reset()
        self._reference_state = None
        self._last_mel = 0.0
        self._last_reference = 0.0
        self._last_factor = 1.0

    # -- diagnostics --

    def get_metrics(self) -> Dict[str, float]:
        return {
            "mel_mean": float(self._last_mel),
            "mel_reference": float(self._last_reference),
            "mel_duration_factor": float(self._last_factor),
            "mel_n_steps_accumulated": float(self.accumulator.count),
        }
