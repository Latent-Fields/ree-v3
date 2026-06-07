"""
MECH-357 (SD-058): InstrumentalAvoidanceGate -- infralimbic-PFC-analog
freeze-suppression + instrumental-avoidance acquisition.

REE already has the Pavlovian / defensive REACTION side: SD-035 (amygdala
BLA / CeA salience) + MECH-279 (CeA->vlPAG freeze gate). What this module adds
is the instrumental-ACQUISITION side the biology lit-pull
(targeted_review_hazard_avoidance_learning) identifies as missing: active
avoidance learning is the RESOLUTION of a Pavlovian-instrumental conflict
(Moscarello & LeDoux 2013) -- the freezing reaction and the instrumental
avoidance action are mutually exclusive, and learning to avoid REQUIRES the
infralimbic prefrontal cortex (ilPFC) to suppress central-amygdala-driven
freezing (ilPFC lesion -> more freezing, less avoidance; CeA lesion -> the
opposite). A freeze-only substrate freezes instead of learning to avoid -- the
V3-EXQ-603g G_H 0/3 signature.

Three coordinated pieces (all behind use_instrumental_avoidance, default OFF):

  (a) Instrumental-avoidance ACTION pathway -- a per-candidate E3 score-bias
      that, under retained threat, PENALISES the no-op / freeze class (so the
      instrumental action is released) proportional to a learned
      avoidance-efficacy and the threat level. It does NOT compute the escape
      direction -- E3's existing harm-forward evaluation already favours
      low-harm trajectories; the gate's job (Moscarello & LeDoux) is to resolve
      the freeze-vs-act conflict and let the existing harm gradient pick the
      escape among the directed candidates (ARC-007-strict-compatible).

  (b) ilPFC freeze-SUPPRESSION gate -- a scalar freeze_suppression in [0, 1]
      that, when above suppression_threshold, overrides the MECH-279 freeze
      no-op so the agent takes its selected instrumental action instead of
      freezing. Consulted ONLY at the MECH-279 application site, so it is inert
      when use_pag_freeze_gate=False (the action-pathway half (a) still operates,
      since freezing is also a passive no-op choice the bias penalises).

  (c) Avoidance-efficacy LEARNING (the acquisition; eligibility trace) -- a
      scalar avoidance_efficacy in [0, 1] that starts low (initial_efficacy,
      default 0.0 = the freeze-default of an ilPFC-naive animal) and rises via
      an EMA credit when a directed action under threat DROPS z_harm_a, decays
      when the agent freezes or fails to avoid under threat. The gradual
      developmental acquisition Debiec & Sullivan 2017 / Thompson 2008 describe.
      PERSISTS across episodes within a curriculum stage (reset() clears only
      the within-episode threat trace, NOT the learned efficacy).

Protective-scaffold anneal (the curriculum; secondary): the effective efficacy
consumed by both consumers is max(avoidance_efficacy, scaffold_floor). The
curriculum (scaffolded_sd054_onboarding Stage-H) sets a high scaffold_floor
early in acquisition (an external instructor shows the agent that avoidance
works -- maternal-buffering / Turchetta 2020 reset-curriculum analogue) and
anneals it down as the measured avoidance_efficacy grows. Budget escalation is
explicitly the SECONDARY lever; this structural mechanism is primary.

Non-trainable: pure arithmetic over scalars + a small tensor bias. No gradient
flow. Mirrors the SD-035 CeA / BLA, MECH-279 PAG, MECH-313 NoiseFloor,
MECH-320 TonicVigor regulator pattern.

MECH-094: both compute methods AND the learning update accept simulation_mode
and are no-ops when True (replay / DMN content must not credit avoidance-efficacy
or suppress freeze on imagined outcomes).

See REE_assembly/docs/architecture/sd_058_instrumental_avoidance_acquisition.md
and evidence/literature/targeted_review_hazard_avoidance_learning/SYNTHESIS.md.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch


@dataclass
class InstrumentalAvoidanceGateConfig:
    """Configuration for MECH-357 instrumental-avoidance gate.

    All defaults produce backward-compatible no-op behaviour at rest:
    initial_efficacy=0.0 and scaffold_floor=0.0 -> effective_efficacy=0.0 ->
    zero action-bias and zero freeze-suppression until the agent learns (or the
    curriculum scaffolds) avoidance-efficacy above zero.
    """

    # -- Efficacy learning (eligibility trace) --

    # EMA credit rate when a directed action under threat drops z_harm_a.
    learn_rate: float = 0.05
    # Decay rate when the agent freezes / fails to avoid under threat.
    leak_rate: float = 0.02
    # Freeze-default for the ilPFC-naive agent. 0.0 = freezes until it learns.
    initial_efficacy: float = 0.0
    # Protective-scaffold floor (curriculum sets > 0 and anneals it down).
    # effective_efficacy = max(avoidance_efficacy, scaffold_floor).
    scaffold_floor: float = 0.0
    # Minimum harm-drop (z_harm_a_prev - z_harm_a_now) counted as a successful
    # avoidance for the credit branch.
    efficacy_reward_floor: float = 1e-4

    # -- Threat envelope --

    # z_harm_a norm below which there is no threat to avoid (no learning, no
    # bias, no suppression).
    threat_floor: float = 0.1
    # z_harm_a norm mapping to full threat_scale = 1.0.
    threat_ref: float = 0.5

    # -- Action pathway (a) --

    # Gain on the anti-passivity (penalise-no-op) bias.
    action_bias_gain: float = 0.1
    # Clamp on |bias| (mirrors lateral_pfc / curiosity / vigor bias_scale).
    bias_scale: float = 0.1
    # The passive / no-op action class index (matches MECH-279 / MECH-320).
    noop_class: int = 0

    # -- Freeze-suppression gate (b) --

    # effective_efficacy * threat_scale above which the MECH-279 freeze no-op
    # is suppressed (ilPFC suppresses CeA-driven freezing).
    suppression_threshold: float = 0.5


@dataclass
class InstrumentalAvoidanceGateOutput:
    """Per-tick diagnostic snapshot (not consumed by the agent loop directly;
    the agent reads the compute_* methods)."""

    avoidance_efficacy: float = 0.0
    effective_efficacy: float = 0.0
    threat_scale: float = 0.0
    freeze_suppression: float = 0.0
    freeze_suppressed: bool = False
    bias_max_abs: float = 0.0


class InstrumentalAvoidanceGate:
    """MECH-357 ilPFC-analog freeze-suppression + instrumental-avoidance gate.

    Stateful:
      _avoidance_efficacy   learned avoidance-efficacy in [0, 1]. CROSS-EPISODE
                            (reset() does NOT clear it -- developmental
                            acquisition persists across episodes within a stage).
      _z_harm_a_prev        previous tick's z_harm_a norm (threat the last action
                            responded to). Within-episode; cleared by reset().
      _last_action_directed whether the last emitted action was directed
                            (non-noop). Within-episode; cleared by reset().
      diagnostic counters   n_ticks / n_credit / n_decay / n_freeze_suppressed /
                            n_updates / n_sim_skipped.

    No gradient flow.
    """

    def __init__(self, config: Optional[InstrumentalAvoidanceGateConfig] = None):
        self.config = config or InstrumentalAvoidanceGateConfig()

        self._avoidance_efficacy: float = float(
            max(0.0, min(1.0, self.config.initial_efficacy))
        )
        self._z_harm_a_prev: Optional[float] = None
        self._last_action_directed: bool = False

        self._n_ticks: int = 0
        self._n_credit: int = 0
        self._n_decay: int = 0
        self._n_freeze_suppressed: int = 0
        self._n_updates: int = 0
        self._n_sim_skipped: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear the WITHIN-EPISODE threat trace. Does NOT clear the learned
        avoidance_efficacy -- developmental acquisition persists across episodes
        within a curriculum stage (the agent does not un-learn avoidance at every
        episode boundary). Call on agent.reset()."""
        self._z_harm_a_prev = None
        self._last_action_directed = False

    def set_scaffold_floor(self, value: float) -> None:
        """Curriculum hook (protective-scaffold anneal). Sets the floor under
        the effective efficacy consumed by both consumers."""
        self.config.scaffold_floor = float(max(0.0, min(1.0, value)))

    # -- Derived quantities --

    def effective_efficacy(self) -> float:
        """max(learned efficacy, protective-scaffold floor), clamped [0, 1]."""
        return float(
            max(
                0.0,
                min(
                    1.0,
                    max(self._avoidance_efficacy, float(self.config.scaffold_floor)),
                ),
            )
        )

    def threat_scale(self, z_harm_a_norm: float) -> float:
        """Linear ramp from 0 at threat_floor to 1 at threat_ref."""
        z = float(z_harm_a_norm)
        lo = float(self.config.threat_floor)
        hi = float(self.config.threat_ref)
        if z <= lo:
            return 0.0
        if hi <= lo:
            return 1.0
        return float(max(0.0, min(1.0, (z - lo) / (hi - lo))))

    # -- (a) Instrumental-avoidance action bias --

    def compute_action_bias(
        self,
        z_harm_a_norm: float,
        action_classes: Union[Sequence[int], torch.Tensor],
        noop_class: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """Per-candidate score-bias [K]. Penalises the no-op / freeze class
        (positive bias -- REE lower-is-better, so positive = unfavourable)
        proportional to effective_efficacy * threat_scale. Directed candidates
        get 0 (E3's existing harm gradient ranks them). Returns zeros when
        below threat, when efficacy is zero, or under simulation_mode.
        """
        if isinstance(action_classes, torch.Tensor):
            classes = action_classes.detach().flatten().tolist()
        else:
            classes = list(action_classes)
        k = len(classes)
        bias = torch.zeros(k, dtype=dtype, device=device)

        self._n_ticks += 1
        if simulation_mode:
            self._n_sim_skipped += 1
            return bias
        if k == 0:
            return bias

        eff = self.effective_efficacy()
        ts = self.threat_scale(z_harm_a_norm)
        if eff <= 0.0 or ts <= 0.0:
            return bias

        nc = int(self.config.noop_class if noop_class is None else noop_class)
        gain = float(self.config.action_bias_gain)
        scale = float(self.config.bias_scale)
        # Positive (unfavourable) bias on the no-op / freeze class -> release the
        # instrumental action. Clamped to bias_scale so the gate cannot dominate
        # the score-bias chain.
        penalty = float(max(-scale, min(scale, gain * eff * ts)))
        for i, cls in enumerate(classes):
            if int(cls) == nc:
                bias[i] = penalty
        return bias

    # -- (b) ilPFC freeze-suppression --

    def freeze_suppression(
        self, z_harm_a_norm: float, simulation_mode: bool = False
    ) -> float:
        """Scalar in [0, 1]: how strongly the ilPFC suppresses the MECH-279
        freeze this tick. effective_efficacy * threat_scale. 0 under
        simulation_mode."""
        if simulation_mode:
            return 0.0
        return float(self.effective_efficacy() * self.threat_scale(z_harm_a_norm))

    def should_suppress_freeze(
        self, z_harm_a_norm: float, simulation_mode: bool = False
    ) -> bool:
        """True when freeze_suppression >= suppression_threshold. The caller
        (MECH-279 site) skips the no-op override when this is True."""
        if simulation_mode:
            return False
        suppressed = self.freeze_suppression(z_harm_a_norm) >= float(
            self.config.suppression_threshold
        )
        if suppressed:
            self._n_freeze_suppressed += 1
        return suppressed

    # -- (c) Avoidance-efficacy learning (eligibility trace) --

    def update(
        self,
        z_harm_a_norm: float,
        action_was_directed: bool,
        simulation_mode: bool = False,
    ) -> None:
        """Advance the avoidance-efficacy eligibility trace by ONE tick.

        Compares the current z_harm_a norm to the threat the PREVIOUS action
        responded to (_z_harm_a_prev). When that previous tick was under threat:
          - directed action AND harm dropped  -> credit (efficacy rises)
          - directed action AND harm did not drop -> decay
          - froze (no-op) under threat        -> decay (freezing is not credited)
        A one-tick lag: the avoidance outcome is the just-experienced threat
        change. No-op under simulation_mode (MECH-094)."""
        z_now = float(z_harm_a_norm)
        if simulation_mode:
            self._n_sim_skipped += 1
            return
        self._n_updates += 1

        prev = self._z_harm_a_prev
        if prev is not None and prev > float(self.config.threat_floor):
            delta = prev - z_now  # > 0 = harm dropped after the action
            lr = float(self.config.learn_rate)
            leak = float(self.config.leak_rate)
            if action_was_directed and delta > float(self.config.efficacy_reward_floor):
                self._avoidance_efficacy += lr * (1.0 - self._avoidance_efficacy)
                self._n_credit += 1
            else:
                self._avoidance_efficacy -= leak * self._avoidance_efficacy
                self._n_decay += 1
            self._avoidance_efficacy = float(
                max(0.0, min(1.0, self._avoidance_efficacy))
            )

        self._z_harm_a_prev = z_now
        self._last_action_directed = bool(action_was_directed)

    # -- Read-only accessors --

    @property
    def avoidance_efficacy(self) -> float:
        return float(self._avoidance_efficacy)

    def last_output(self, z_harm_a_norm: float = 0.0) -> InstrumentalAvoidanceGateOutput:
        eff = self.effective_efficacy()
        ts = self.threat_scale(z_harm_a_norm)
        sup = eff * ts
        return InstrumentalAvoidanceGateOutput(
            avoidance_efficacy=float(self._avoidance_efficacy),
            effective_efficacy=float(eff),
            threat_scale=float(ts),
            freeze_suppression=float(sup),
            freeze_suppressed=bool(sup >= float(self.config.suppression_threshold)),
            bias_max_abs=float(
                min(
                    float(self.config.bias_scale),
                    float(self.config.action_bias_gain) * eff * ts,
                )
            ),
        )

    def get_state(self) -> dict:
        return {
            "mech357_avoidance_efficacy": float(self._avoidance_efficacy),
            "mech357_scaffold_floor": float(self.config.scaffold_floor),
            "mech357_effective_efficacy": float(self.effective_efficacy()),
            "mech357_n_credit": int(self._n_credit),
            "mech357_n_decay": int(self._n_decay),
            "mech357_n_freeze_suppressed": int(self._n_freeze_suppressed),
            "mech357_n_updates": int(self._n_updates),
            "mech357_n_sim_skipped": int(self._n_sim_skipped),
        }

    @property
    def diagnostics(self) -> dict:
        return self.get_state()
