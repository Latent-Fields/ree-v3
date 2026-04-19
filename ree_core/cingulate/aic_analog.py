"""
AICAnalog -- SD-032c anterior-insula-analog (interoceptive salience / urgency).

SD-032c is NOT the affective-pain consumer (that is SD-032b DACCAdaptiveControl).
It is the module that decides WHEN the current operating mode is no longer
sustainable -- the urgency-interrupt source for the salience-network coordinator
(SD-032a). It also subsumes SD-021 descending pain modulation: in biology the
ACC/AIC -> PAG descending inhibitory pathway attenuates sensory nociceptive gain
as a function of the current operating mode, not as a raw function of whether
beta_gate is elevated.

Architectural scope (SD-032c v3 minimum-viable):

  Inputs (per tick, called once per select_action):
    z_harm_a_norm      scalar ||z_harm_a||_2 (SD-011 affective stream)
    drive_level        scalar in [0, 1]      (SD-012 homeostatic drive)
    beta_gate_elevated bool                  (MECH-090 committed-state signal)
    operating_mode     Dict[str, float]      (SD-032a coordinator readout from
                                              the previous tick; None if
                                              coordinator is disabled, in which
                                              case we treat p_external_task=1)
    extra_salient      Dict[str, float]      optional salient-event signals
                                              (unexpected z_goal drop, reward
                                              surprise, irreversibility flag)

  Computation:
    baseline <- (1 - alpha) * baseline + alpha * z_harm_a_norm
    urgency  = max(0, (z_harm_a_norm - baseline) / (baseline + eps))
    aic_salience = urgency * (1 + drive_coupling * drive_level)
                 + sum_i extra_salient[i] * extra_weight

    # Descending pain gain (subsumes SD-021)
    p_external   = operating_mode.get("external_task", 1.0)
    mode_weight  = p_external * float(beta_gate_elevated)
    drive_protect = max(0, 1.0 - drive_protect_weight * drive_level)
    harm_s_gain  = 1.0 - base_attenuation * mode_weight * drive_protect

  Outputs:
    aic_salience    float   fed to SalienceCoordinator via update_signal()
    harm_s_gain     float   multiplier on z_harm_s in agent.sense()
                            (replaces the raw beta_gate check of SD-021)
    urgency_signal  bool    aic_salience > urgency_threshold (diagnostic only;
                            coordinator's own threshold logic is authoritative)

Falsification signature (sd_032_cingulate_integration_substrate.md):
  Same z_harm_a produces different mode-switch behaviour in depleted vs
  well-resourced agents. Failure: mode-switch rate invariant to SD-012 state.
  Both `aic_salience` and `harm_s_gain` depend on drive_level -- this module
  is the only V3 substrate that makes that dependence structural.

Biological grounding:
  Craig 2009 (Nat Rev Neurosci) -- AIC as interoceptive-salience hub with
  autonomic and motor efferents; integrates interoceptive baseline with
  noxious / salient-event inputs.
  Menon & Uddin 2010 -- AIC is the rostral node of the salience network; its
  output switches DMN <-> CEN via dACC.
  SD-021 descending modulation (Basbaum 1984, Keltner 2006) is biologically an
  AIC/ACC function -- folded here per SD-032c spec.

Non-trainable: pure arithmetic over scalars. No gradient flow. Single EMA
buffer for interoceptive baseline.

MECH-094: not applicable (waking observation stream, not replay content).

See CLAUDE.md: SD-032c. Spec:
REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AICConfig:
    """Configuration for SD-032c AIC-analog.

    All defaults produce backward-compatible no-op behaviour: with default
    drive_coupling=1.0 and base_attenuation=0.5, the gain function reduces
    to (1 - 0.5 * p_external * beta_elevated * (1 - drive_level)) only when
    the module is actually instantiated (use_aic_analog=True). When not
    instantiated, agent.sense() never calls the module.
    """

    # EMA alpha for interoceptive baseline. Smaller = slower adaptation.
    # 0.02 matches MECH-205 pe_ema_alpha convention (~50-step window).
    baseline_alpha: float = 0.02

    # Initial baseline value. Zero means the first few ticks will show large
    # urgency spikes until the EMA converges. Small positive value is a
    # biologically plausible default "expected low-level interoception".
    baseline_init: float = 0.01

    # Numerical epsilon to avoid division-by-zero in the urgency ratio.
    baseline_eps: float = 1e-3

    # drive_coupling: how strongly drive_level (SD-012) scales urgency.
    # depleted agent (drive_level=1) -> aic_salience is (1 + drive_coupling)
    # times the urgency-ratio value. Falsification contract: must be nonzero
    # for drive-dependence to be present.
    drive_coupling: float = 1.0

    # urgency_threshold: reporting-only threshold on aic_salience for the
    # diagnostic urgency_signal flag. The actual MECH-259 trigger uses the
    # coordinator's switch_threshold applied to the salience aggregate.
    urgency_threshold: float = 1.0

    # base_attenuation: maximum descending attenuation of z_harm_s.
    # harm_s_gain = 1.0 - base_attenuation * mode_weight * drive_protect.
    # 0.5 matches the historical SD-021 descending_attenuation_factor meaning
    # "50% attenuation" at maximum mode_weight and drive_protect=1.
    base_attenuation: float = 0.5

    # drive_protect_weight: how strongly drive_level reduces the descending
    # attenuation. drive_protect = max(0, 1 - drive_protect_weight * drive_level).
    #   +1.0 (default): depleted agent -> drive_protect=0 -> no attenuation.
    #     Biological reading: preserve harm signal for an already-struggling
    #     agent so that SD-032c can still fire mode_switch_trigger.
    #    0.0: drive-independent attenuation (pure legacy SD-021 reading --
    #     use for ablation of the drive-dependence).
    #   -1.0: depleted agent -> drive_protect>1 -> MORE attenuation
    #     (opposite-sign reading; testable alternative hypothesis).
    # This is the knob flagged by SD-032c spec as configuration-testable.
    drive_protect_weight: float = 1.0

    # extra_weight: uniform weight applied to every entry in the extra_salient
    # dict on each tick. Zero by default (no-op). Non-zero wires the stub
    # inputs (unexpected z_goal drop, reward-surprise, irreversibility) into
    # the salience aggregate without requiring a rebuild of the coordinator's
    # affinity / salience tables.
    extra_weight: float = 0.0


class AICAnalog:
    """SD-032c anterior-insula-analog interoceptive-salience module.

    Stateful:
      _baseline       -- EMA over z_harm_a_norm; interoceptive "expected tone".
      _aic_salience   -- last computed urgency salience (scalar).
      _harm_s_gain    -- last computed descending multiplier in (0, 1].
      _urgency_signal -- last computed diagnostic threshold crossing (bool).
      _n_ticks        -- diagnostic counter.
      _n_urgency      -- number of ticks where urgency_signal fired.

    No gradient flow. Reset per episode via .reset().
    """

    def __init__(self, config: Optional[AICConfig] = None):
        self.config = config or AICConfig()
        self._baseline: float = float(self.config.baseline_init)
        self._aic_salience: float = 0.0
        self._harm_s_gain: float = 1.0
        self._urgency_signal: bool = False
        self._last_urgency_ratio: float = 0.0
        self._n_ticks: int = 0
        self._n_urgency: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear per-episode state. Call on env.reset()."""
        self._baseline = float(self.config.baseline_init)
        self._aic_salience = 0.0
        self._harm_s_gain = 1.0
        self._urgency_signal = False
        self._last_urgency_ratio = 0.0

    # -- Tick: main per-step computation --

    def tick(
        self,
        z_harm_a_norm: float,
        drive_level: float = 0.0,
        beta_gate_elevated: bool = False,
        operating_mode: Optional[Dict[str, float]] = None,
        extra_salient: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Compute aic_salience and harm_s_gain for this step.

        Args:
            z_harm_a_norm: scalar ||z_harm_a||_2 from SD-011 stream.
            drive_level: SD-012 GoalState drive_level in [0, 1].
            beta_gate_elevated: MECH-090 committed-state signal.
            operating_mode: SD-032a soft probability vector from the previous
                coordinator tick. None if coordinator is disabled; treated as
                p_external_task = 1.0 (waking baseline).
            extra_salient: optional salient-event signals (unexpected z_goal
                drop, reward-surprise, irreversibility). Uniform extra_weight
                applied. Defaults to no-op.

        Returns:
            Dict with aic_salience, harm_s_gain, urgency_signal (bool->float
            0/1), baseline, urgency_ratio. All plain python floats/bools.
        """
        self._n_ticks += 1

        z_norm = float(z_harm_a_norm)
        drive = float(drive_level)

        # EMA interoceptive baseline.
        alpha = float(self.config.baseline_alpha)
        self._baseline = (1.0 - alpha) * self._baseline + alpha * z_norm

        # Urgency ratio above baseline. Clamp at zero (below-baseline harm is
        # not salient; negative values would invert the signal meaning).
        ratio = (z_norm - self._baseline) / (self._baseline + float(self.config.baseline_eps))
        if ratio < 0.0:
            ratio = 0.0
        self._last_urgency_ratio = ratio

        # Drive-scaled urgency: depleted agent -> more salient at same z_harm_a.
        urgency_scaled = ratio * (1.0 + float(self.config.drive_coupling) * drive)

        # Extra salient-event aggregation (all defaults to zero).
        extra_sum = 0.0
        if extra_salient:
            extra_weight = float(self.config.extra_weight)
            if extra_weight != 0.0:
                for _, v in extra_salient.items():
                    extra_sum += extra_weight * float(v)

        self._aic_salience = urgency_scaled + extra_sum

        # Descending pain gain (SD-021 re-route).
        if operating_mode is not None:
            p_external = float(operating_mode.get("external_task", 1.0))
        else:
            # Coordinator disabled -- treat waking as external_task for the
            # attenuation path. Preserves a meaningful SD-021 function even
            # without SD-032a (still drive-gated; the thing EXQ-325a lacked).
            p_external = 1.0

        mode_weight = p_external * (1.0 if beta_gate_elevated else 0.0)
        drive_protect = 1.0 - float(self.config.drive_protect_weight) * drive
        if drive_protect < 0.0:
            drive_protect = 0.0
        # Note: drive_protect > 1.0 is permitted when drive_protect_weight is
        # negative; the base_attenuation clamp below keeps harm_s_gain >= 0.
        gain = 1.0 - float(self.config.base_attenuation) * mode_weight * drive_protect
        if gain < 0.0:
            gain = 0.0
        if gain > 1.0:
            gain = 1.0
        self._harm_s_gain = gain

        self._urgency_signal = self._aic_salience > float(self.config.urgency_threshold)
        if self._urgency_signal:
            self._n_urgency += 1

        return {
            "aic_salience": float(self._aic_salience),
            "harm_s_gain": float(self._harm_s_gain),
            "urgency_signal": 1.0 if self._urgency_signal else 0.0,
            "baseline": float(self._baseline),
            "urgency_ratio": float(self._last_urgency_ratio),
        }

    # -- Read-only accessors --

    @property
    def aic_salience(self) -> float:
        return float(self._aic_salience)

    @property
    def harm_s_gain(self) -> float:
        return float(self._harm_s_gain)

    @property
    def urgency_signal(self) -> bool:
        return bool(self._urgency_signal)

    @property
    def baseline(self) -> float:
        return float(self._baseline)

    @property
    def diagnostics(self) -> Dict[str, int]:
        return {"n_ticks": self._n_ticks, "n_urgency": self._n_urgency}
