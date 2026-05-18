"""
PACCAnalog -- SD-032e perigenual / subgenual cingulate-analog (pACC-autonomic
coupling; slow write-back from sustained z_harm_a into SD-012 drive_level).

SD-032e is the slowest member of the SD-032 cluster. It integrates z_harm_a
(affective pain, SD-011 C-fiber analog) over multi-episode timescales and
emits a signed bias that shifts the effective drive_level passed into
GoalState.update(), SalienceCoordinator, SD-032c AIC, and SD-032d PCC. This
is how sustained affective pain biases the interoceptive baseline -- the
architectural path for chronic-pain-like sensitisation (Baliki 2012).

Does NOT:
  - write into valence-signed mood state (REE has no valence-signed setpoint
    at V3; SD-032e compresses pACC/sgACC's valence-setpoint role onto the
    unsigned drive_level scalar -- see scoping lit-pull synthesis).
  - drive fast autonomic effectors (no fast-autonomic analogue in V3; the
    Gianaros/Critchley 2011 ACC->PAG->medulla route would be a future
    SD-032f).
  - write into ACC-internal synaptic gain (direct-write EMA compresses
    two biological steps -- Guo 2018 ACC LTP + ACC downstream influence --
    into one accumulator; see scoping synthesis entry).

Write target (per scoping lit-pull synthesis, 2026-04-19):
  drive_level as first-pass proxy. Biologically tighter target would be a
  valence-signed setpoint variable; none exists in V3. Documented
  simplification.

Timescale (per scoping lit-pull synthesis, 2026-04-19):
  Slow EMA over multi-episode accumulation. Default alpha=0.002
  (pacc_drive_ema = 0.998; half-life ~347 steps). Guo 2018 rodent ACC LTP
  saturates over days (thousands of REE steps); default is at the fast end
  of biological plausibility so the validation experiment can observe
  drift within a tractable runtime. Experiments studying long-horizon
  sensitisation should slow this further (alpha=0.0005 or less).

Offline decay (per scoping lit-pull synthesis, 2026-04-19):
  Default pacc_offline_decay=0.0 (no decay on offline entry). Non-zero
  values instantiate a DISTINCT sleep-recalibration claim (not yet
  registered) that would need its own literature pull and validation.
  SD-032e does NOT model sleep normalisation as a default; the hook
  exists so a future claim can be wired in without another implementation
  pass.

Operating-mode gating (SD-032a MECH-261):
  The coordinator's "autonomic" gate (active in external_task, attenuated
  in internal_planning / internal_replay / offline_consolidation) is
  applied to the EMA increment per tick -- this is what makes the
  write-back mode-conditioned rather than unconditional. When the
  coordinator is disabled (use_salience_coordinator=False), the gate
  defaults to 1.0 so SD-032e's slow drift remains observable under
  ablation baselines.

MECH-094 (hypothesis_tag):
  When hypothesis_tag=True (simulation / replay content), the tick is
  skipped -- no accumulation from hypothetical z_harm_a. This matches
  the MECH-094 write-gate convention used elsewhere in the agent.

Falsification signature (sd_032_cingulate_integration_substrate.md SD-032e):
  Sustained z_harm_a exposure produces drift in drive_level, which in turn
  modulates SD-032c's switch threshold and GoalState wanting gain. With
  SD-032e OFF, the same sustained z_harm_a leaves drive_level untouched
  (only obs_body[3] energy depletion moves it) -- no chronic-pain-like
  sensitisation signature is possible.

Non-trainable: pure arithmetic over scalars. EMA of a tanh-normalised
z_harm_a magnitude, scaled by coordinator gate. No gradient flow.

See CLAUDE.md: SD-032e. Spec:
REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
Scoping review:
REE_assembly/evidence/literature/targeted_review_pacc_autonomic_coupling_write_target/synthesis.md
"""

from dataclasses import dataclass
from typing import Dict, Optional

import math


@dataclass
class PACCConfig:
    """Configuration for SD-032e pACC-analog slow autonomic write-back.

    Defaults are conservative and no-op when use_pacc_analog=False (the
    module is not instantiated). When enabled, defaults produce a slow
    drift of the drive_bias toward tanh(z_harm_a_norm) over ~hundreds
    of steps, clipped to a bounded range so the substrate cannot by
    itself drive drive_level to saturation.
    """

    # EMA alpha for the drive_bias accumulator.
    # alpha=0.002 -> half-life ~347 steps, pacc_drive_ema=0.998.
    # Scoping synthesis called 0.005 "the fast end of biological plausibility"
    # and recommended 0.995 or slower; the default sits inside that envelope.
    # Experiments modelling chronic-pain-like sensitisation on long horizons
    # should use alpha=0.0005 or slower (pacc_drive_ema >= 0.9995).
    drive_alpha: float = 0.002

    # Scale on the tanh-normalised z_harm_a signal inside the EMA target.
    # tanh(z_harm_a_norm) is in [0, 1); multiplied by this scale produces
    # the per-step accumulation target. Keep <=1 to prevent the bias from
    # saturating at +cap on a single high-load tick.
    drive_scale: float = 1.0

    # Absolute cap on |drive_bias|. The substrate cannot drive
    # effective_drive_level outside [max(0, base - cap), min(1, base + cap)].
    # Bounded so SD-032e cannot masquerade as an unbounded amplifier of
    # SD-012 fatigue -- the structural effect we want is a shift in the
    # interoceptive baseline, not a replacement of it.
    drive_bias_cap: float = 0.5

    # Threshold below which z_harm_a_norm does not accumulate. Filters out
    # encoder noise at rest (z_harm_a rarely sits at exact zero). Keep
    # small so genuine low-grade sustained pain still accumulates.
    z_harm_a_min: float = 0.0

    # Offline decay on note_offline_entry(). DEFAULT 0.0 = no decay.
    # Non-zero values instantiate a DISTINCT sleep-recalibration claim
    # (see pACC-autonomic scoping synthesis). Hook exists so a future
    # claim can wire in without another implementation pass. Units:
    # multiplicative per offline entry: drive_bias *= (1 - offline_decay).
    offline_decay: float = 0.0


class PACCAnalog:
    """SD-032e perigenual cingulate-analog slow-EMA autonomic write-back.

    Stateful:
      _drive_bias              -- current bias in [-cap, +cap]; slow EMA of
                                  mode-gated, tanh-normalised z_harm_a_norm.
      _last_z_harm_a_norm      -- last input, for diagnostics.
      _last_write_gate         -- last gate, for diagnostics.
      _last_effective_drive    -- last effective_drive returned from
                                  effective_drive() call (cache).
      _n_ticks                 -- diagnostic counter.
      _n_offline_resets        -- diagnostic counter (note_offline_entry).
      _n_hypothesis_skipped    -- diagnostic counter (MECH-094 hypothesis
                                  ticks skipped).

    Per-episode reset() does NOT clear _drive_bias -- the whole point of
    SD-032e is cross-episode accumulation. Offline decay is the only way
    to clear bias, and is opt-in.
    """

    def __init__(self, config: Optional[PACCConfig] = None):
        self.config = config or PACCConfig()
        self._drive_bias: float = 0.0
        self._last_z_harm_a_norm: float = 0.0
        self._last_write_gate: float = 0.0
        self._last_effective_drive: float = 0.0
        self._n_ticks: int = 0
        self._n_offline_resets: int = 0
        self._n_hypothesis_skipped: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear per-episode state (diagnostics cache only).

        Does NOT reset _drive_bias -- cross-episode accumulation is the
        architectural purpose of SD-032e. Use note_offline_entry() with
        non-zero offline_decay for decay, or instantiate a fresh module
        for a hard reset.
        """
        self._last_z_harm_a_norm = 0.0
        self._last_write_gate = 0.0
        self._last_effective_drive = 0.0

    def note_offline_entry(self) -> None:
        """Optionally decay drive_bias on offline-mode entry.

        Called from agent.enter_offline_mode(). Default offline_decay=0.0
        makes this a no-op -- non-zero values are a distinct
        sleep-recalibration claim (not SD-032e default behaviour).
        """
        self._n_offline_resets += 1
        decay = float(self.config.offline_decay)
        if decay <= 0.0:
            return
        if decay >= 1.0:
            self._drive_bias = 0.0
            return
        self._drive_bias *= (1.0 - decay)

    # -- Tick: main per-step computation --

    def tick(
        self,
        z_harm_a_norm: float,
        write_gate: float = 1.0,
        hypothesis_tag: bool = False,
    ) -> Dict[str, float]:
        """Accumulate one EMA step into drive_bias.

        Args:
            z_harm_a_norm: ||z_harm_a||_2 from LatentState (SD-011 affective
                stream). Clipped at 0 from below.
            write_gate: gate scalar from SalienceCoordinator.write_gate(
                "autonomic") in [0, 1]. Defaults to 1.0 so the substrate
                behaves sensibly when the coordinator is disabled.
            hypothesis_tag: MECH-094 gate. When True, skip the EMA update
                -- simulation / replay content does not drive autonomic
                write-back.

        Returns a dict (all plain floats):
            drive_bias         current accumulator value in [-cap, +cap]
            z_harm_a_norm      echo of input
            write_gate         echo of input
            hypothesis_skipped 1.0 if this tick was skipped by MECH-094
        """
        self._n_ticks += 1

        if hypothesis_tag:
            self._n_hypothesis_skipped += 1
            return {
                "drive_bias": float(self._drive_bias),
                "z_harm_a_norm": float(z_harm_a_norm),
                "write_gate": float(write_gate),
                "hypothesis_skipped": 1.0,
            }

        z = float(z_harm_a_norm)
        if z < 0.0:
            z = 0.0
        self._last_z_harm_a_norm = z

        gate = float(write_gate)
        if gate < 0.0:
            gate = 0.0
        elif gate > 1.0:
            gate = 1.0
        self._last_write_gate = gate

        z_min = float(self.config.z_harm_a_min)
        if z <= z_min:
            # Below threshold -- decay toward zero at the EMA rate instead
            # of freezing. Lets drive_bias relax when affective pain goes
            # quiet for long stretches; matches the Guo 2018 reversibility
            # of LTP under nociceptive quiescence.
            target = 0.0
        else:
            target = math.tanh(z) * float(self.config.drive_scale)

        alpha = float(self.config.drive_alpha)
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0

        # Gated EMA: only gated fraction of the increment applies. With
        # gate=0 no accumulation; with gate=1 full EMA pull toward target.
        self._drive_bias = (1.0 - alpha * gate) * self._drive_bias + (
            alpha * gate * target
        )

        cap = float(self.config.drive_bias_cap)
        if cap < 0.0:
            cap = 0.0
        if self._drive_bias > cap:
            self._drive_bias = cap
        elif self._drive_bias < -cap:
            self._drive_bias = -cap

        return {
            "drive_bias": float(self._drive_bias),
            "z_harm_a_norm": float(z),
            "write_gate": float(gate),
            "hypothesis_skipped": 0.0,
        }

    # -- Drive-level read --

    def effective_drive(self, base_drive_level: float) -> float:
        """Compute base_drive_level + drive_bias, clipped to [0, 1].

        Caller is responsible for feeding the result into GoalState.update(),
        SalienceCoordinator.tick(), SD-032c AIC, SD-032d PCC -- anywhere a
        drive_level scalar is consumed -- so all SD-032 consumers see the
        SD-032e-adjusted value.
        """
        base = float(base_drive_level)
        if base < 0.0:
            base = 0.0
        elif base > 1.0:
            base = 1.0
        effective = base + float(self._drive_bias)
        if effective < 0.0:
            effective = 0.0
        elif effective > 1.0:
            effective = 1.0
        self._last_effective_drive = effective
        return effective

    # -- Read-only accessors --

    @property
    def drive_bias(self) -> float:
        return float(self._drive_bias)

    @property
    def last_effective_drive(self) -> float:
        return float(self._last_effective_drive)

    @property
    def diagnostics(self) -> Dict[str, int]:
        return {
            "n_ticks": self._n_ticks,
            "n_offline_resets": self._n_offline_resets,
            "n_hypothesis_skipped": self._n_hypothesis_skipped,
        }
