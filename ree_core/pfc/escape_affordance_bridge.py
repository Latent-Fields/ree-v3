"""
SD-059 / MECH-358: EscapeAffordanceBridge -- relief/safety escape-affordance
bridge that gives the MECH-357 instrumental-avoidance gate a DIRECTED escape.

MECH-357 (SD-058) suppresses the MECH-279 PAG freeze and releases the
instrumental action, but its avoidance_efficacy is a GLOBAL SCALAR that only
penalises the no-op / freeze class -- compute_action_bias by design "does NOT
compute the escape direction". The V3-EXQ-603h Stage-H validation FAIL
(2026-06-08) showed the consequence: the agent un-freezes but acquires no
directed escape (seed-43: scalar efficacy 0.633 -> WORST survival 11.0). The
agent twitches away from freeze without learning that a particular
action / direction is a relief/safety affordance.

This module is the missing wire identified by the failure autopsy + the
Moscarello & LeDoux 2013 active-avoidance reference: REE built the ilPFC
freeze-SUPPRESSION half but not the LA/BA -> NAcc relief/safety action-credit
half. REE already owns relief (MECH-302 / SD-050) and safety
(MECH-303/304 / SD-052/SD-051) but they are UNWIRED to avoidance. The bridge
binds an action / direction to relief + safety so it becomes APPROACHABLE under
future threat.

Affordance-indexing (minimal V3 instantiation): the escape affordance is keyed
by FIRST-ACTION CLASS (the directed escape direction in the discrete action
space). This extends MECH-357's scalar efficacy into a per-class credit table.
Location / policy indexing is a deferred refinement; the validation experiment's
nav-competence positive control rules out the "directed escape needs richer
representation" branch.

Two independently-toggleable halves (so the 4-arm validation can dissociate):

  RELIEF half (MECH-302-consistent): a directed action under threat that DROPS
    z_harm_a (delta = prev - now > relief_reward_floor) emits phasic relief and
    credits relief_affordance[action_class] (EMA toward 1). This is the same
    d(z_harm_a)/dt < 0 signal MECH-302 fires on, attributed to the SPECIFIC
    action that produced it.

  SAFETY half (MECH-303/304-consistent): a directed action AFTER which threat is
    absent (threat_scale <= 0, i.e. z_harm_a below threat_floor -- the threat
    cue terminated) credits safety_affordance[action_class] (response-produced
    safety / conditioned inhibition). The directed action becomes a learned
    threat-absence predictor.

Approach bonus (the directed escape): under FUTURE threat (threat_scale > 0),
E3 receives a per-candidate NEGATIVE score-bias (REE lower-is-better, so
negative = favoured) toward each candidate whose first-action class carries
escape-affordance credit, proportional to approach_gain * threat_scale *
combined_affordance[class], CLAMPED to bias_scale. The no-op / freeze class
never gets an approach bonus.

Three guards keep it bounded and non-swamping (per the thought-intake design
discipline):
  (1) clamped to bias_scale (mirrors the MECH-357 / curiosity / vigor bias_scale
      so it cannot dominate the additive score-bias chain);
  (2) THREAT-CONTEXT-GATED -- exactly zero when threat_scale <= 0, so it never
      globally swamps food / goal approach when the agent is safe;
  (3) per-tick LEAK on both affordance tables (forgetting) so a stale credit
      cannot drive a pathological avoidance / habit loop.

DISTINCT from reflexive escape (SD-037 orexin / MECH-281 urgency-interrupt,
which are threat/arousal reflexes) and from the generic relief / safety rows
(MECH-302 / MECH-303/304, which fire on the CURRENT state): this is
learned-efficacy-gated DIRECTED APPROACH that binds an action to relief/safety
for use under future threat.

Non-trainable: pure arithmetic over scalars + per-action-class credit lists.
No gradient flow. Mirrors the SD-058/MECH-357, SD-035, MECH-279, MECH-313,
MECH-320 regulator pattern.

MECH-094: both compute methods AND the eligibility update accept simulation_mode
and are no-ops when True (replay / DMN content must not credit escape affordances
or bias action selection on imagined outcomes).

See REE_assembly/docs/architecture/sd_059_escape_affordance_bridge.md,
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603h_2026-06-08.md, and
REE_assembly/evidence/planning/thought_intake_2026-06-07_relief_safety_escape_affordance_bridge.md.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch


@dataclass
class EscapeAffordanceBridgeConfig:
    """Configuration for SD-059 / MECH-358 escape-affordance bridge.

    All defaults produce backward-compatible no-op behaviour at rest: the
    affordance tables start at zero, so the approach bonus is exactly zero
    until a relief / safety credit event raises an entry. Combined with the
    threat-context gate, the bridge is bit-identical to OFF whenever the agent
    is safe.
    """

    # Number of discrete first-action classes (set from action_dim by the agent).
    n_action_classes: int = 5

    # -- Half switches (consulted only when the master flag is on) --

    # RELIEF half: credit from negative d(z_harm_a)/dt to the last directed action.
    use_relief_credit: bool = True
    # SAFETY half: response-produced safety (threat absent after a directed action).
    use_safety_credit: bool = True

    # -- Credit learning (eligibility traces) --

    # EMA credit rate for the relief affordance on a successful harm-drop.
    relief_learn_rate: float = 0.1
    # EMA credit rate for the safety affordance on response-produced threat absence.
    safety_learn_rate: float = 0.1
    # Per-tick multiplicative leak on both tables (forgetting; pathological-loop guard).
    leak_rate: float = 0.01
    # Minimum harm-drop (z_harm_a_prev - z_harm_a_now) counted as relief.
    relief_reward_floor: float = 1e-4

    # -- Threat envelope (shared with the MECH-357 gate convention) --

    # z_harm_a norm below which there is no threat (no credit, no approach bonus).
    threat_floor: float = 0.1
    # z_harm_a norm mapping to full threat_scale = 1.0.
    threat_ref: float = 0.5

    # -- Trained threat-absence signal feeding the SAFETY half (603i fix) --
    #
    # The raw SAFETY credit (threat_scale(z_now) <= 0) almost never fires under
    # Stage-H: the threat does not go fully absent after a single directed action,
    # so on V3-EXQ-603i the safety half credited 0/3 in every arm (the relief half
    # fired 2/3). REE already owns TRAINED threat-absence predictors -- MECH-303
    # (contextual safety terrain) + MECH-304 (cue-specific conditioned safety) --
    # but they were unwired to the bridge. When use_trained_safety_signal is on,
    # a directed action under threat ALSO credits safety_affordance[class] when the
    # supplied trained safety_signal (response-produced safety prediction for the
    # post-action state) clears safety_signal_threshold. This is OR-composed with
    # the raw threat-absence check, so the original mechanism is retained as a
    # fallback. Default OFF -> bit-identical to the pre-603i bridge.
    use_trained_safety_signal: bool = False
    # Trained safety_prediction in [0, 1] above which the post-action state counts
    # as response-produced threat-absence (MECH-303/304 sigmoid output).
    safety_signal_threshold: float = 0.5

    # -- Approach bonus (directed escape) --

    # Gain on the (negative) approach bias toward credited escape affordances.
    approach_gain: float = 0.1
    # Clamp on |bias| (mirrors MECH-357 / curiosity / vigor bias_scale so the
    # bridge cannot dominate the additive score-bias chain).
    bias_scale: float = 0.1
    # The passive / no-op / freeze action class (matches MECH-279 / MECH-357).
    noop_class: int = 0


@dataclass
class EscapeAffordanceBridgeOutput:
    """Per-tick diagnostic snapshot (not consumed by the agent loop directly;
    the agent reads the compute_* methods)."""

    threat_scale: float = 0.0
    best_class: int = -1
    best_affordance: float = 0.0
    bias_max_abs: float = 0.0
    relief_affordance_max: float = 0.0
    safety_affordance_max: float = 0.0


class EscapeAffordanceBridge:
    """SD-059 / MECH-358 relief/safety escape-affordance bridge.

    Stateful (all CROSS-EPISODE within a curriculum stage -- developmental
    acquisition persists; reset() clears only the within-episode threat trace):
      _relief_affordance    per-action-class relief credit in [0, 1].
      _safety_affordance    per-action-class safety credit in [0, 1].
      _z_harm_a_prev        previous tick's z_harm_a norm (the threat the last
                            action responded to). Within-episode; cleared by reset().
      diagnostic counters   n_ticks / n_relief_credit / n_safety_credit /
                            n_decay / n_approach_fires / n_updates / n_sim_skipped.

    No gradient flow.
    """

    def __init__(self, config: Optional[EscapeAffordanceBridgeConfig] = None):
        self.config = config or EscapeAffordanceBridgeConfig()
        k = max(1, int(self.config.n_action_classes))
        self._relief_affordance: List[float] = [0.0] * k
        self._safety_affordance: List[float] = [0.0] * k
        self._z_harm_a_prev: Optional[float] = None

        self._n_ticks: int = 0
        self._n_relief_credit: int = 0
        self._n_safety_credit: int = 0
        # Subset of _n_safety_credit attributable to the trained MECH-303/304
        # threat-absence signal (vs the raw threat_scale<=0 path). Lets a
        # validation manifest confirm the safety half fires non-vacuously from
        # the trained predictor specifically (the 603i starvation gap).
        self._n_safety_credit_trained: int = 0
        self._n_decay: int = 0
        self._n_approach_fires: int = 0
        self._n_updates: int = 0
        self._n_sim_skipped: int = 0

    # -- State management --

    def reset(self) -> None:
        """Clear the WITHIN-EPISODE threat trace. Does NOT clear the learned
        affordance tables -- the escape affordance is retained across episodes
        within a curriculum stage (same persistence semantic as the MECH-357
        avoidance_efficacy). Call on agent.reset()."""
        self._z_harm_a_prev = None

    # -- Derived quantities --

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

    def _combined_affordance(self, cls: int) -> float:
        """relief + safety credit for an action class (each half gated by its
        flag), clamped to [0, 1]."""
        if cls < 0 or cls >= len(self._relief_affordance):
            return 0.0
        total = 0.0
        if self.config.use_relief_credit:
            total += self._relief_affordance[cls]
        if self.config.use_safety_credit:
            total += self._safety_affordance[cls]
        return float(max(0.0, min(1.0, total)))

    # -- Eligibility update (relief + safety credit; one-tick lag) --

    def update(
        self,
        z_harm_a_norm: float,
        last_action_class: Optional[int],
        last_action_directed: bool,
        simulation_mode: bool = False,
        safety_signal: Optional[float] = None,
    ) -> None:
        """Advance the escape-affordance eligibility traces by ONE tick.

        Compares the current z_harm_a norm to the threat the PREVIOUS action
        responded to (_z_harm_a_prev). When that previous tick was under threat
        AND the last action was directed (non-noop):
          - RELIEF: harm dropped (delta > relief_reward_floor) -> credit
            relief_affordance[last_action_class] (EMA toward 1).
          - SAFETY: threat now absent -> credit safety_affordance[last_action_class]
            (response-produced safety). Threat-absence is satisfied by the raw
            check (threat_scale(now) <= 0) OR, when use_trained_safety_signal is
            on, by the trained MECH-303/304 threat-absence predictor:
            safety_signal >= safety_signal_threshold. The trained path is the
            603i fix (the raw check almost never fires under Stage-H); it stays
            inside the under-threat + directed-action gate, so it credits genuine
            response-produced safety, not generic safe-context.
        Both tables leak per tick (forgetting / pathological-loop guard).
        One-tick lag: the avoidance outcome is the just-experienced threat
        change. No-op under simulation_mode (MECH-094).

        safety_signal is the post-action trained safety prediction in [0, 1]
        (None when the caller has no trained predictor enabled; ignored unless
        use_trained_safety_signal is on -> bit-identical OFF).
        """
        if simulation_mode:
            self._n_sim_skipped += 1
            return
        self._n_updates += 1

        prev = self._z_harm_a_prev
        z_now = float(z_harm_a_norm)
        k = len(self._relief_affordance)

        if prev is not None and prev > float(self.config.threat_floor):
            a = last_action_class
            nc = int(self.config.noop_class)
            directed = bool(last_action_directed) and a is not None and 0 <= int(a) < k and int(a) != nc
            if directed:
                a = int(a)
                # RELIEF: harm dropped after the directed action.
                if self.config.use_relief_credit:
                    delta = prev - z_now
                    if delta > float(self.config.relief_reward_floor):
                        lr = float(self.config.relief_learn_rate)
                        self._relief_affordance[a] += lr * (1.0 - self._relief_affordance[a])
                        self._n_relief_credit += 1
                # SAFETY: threat absent after the directed action (cue terminated).
                # Raw threat-absence OR a trained MECH-303/304 threat-absence
                # prediction clearing the threshold (603i fix -- the raw check
                # almost never fires under Stage-H, starving the safety half).
                if self.config.use_safety_credit:
                    raw_absent = self.threat_scale(z_now) <= 0.0
                    trained_absent = (
                        self.config.use_trained_safety_signal
                        and safety_signal is not None
                        and float(safety_signal) >= float(self.config.safety_signal_threshold)
                    )
                    if raw_absent or trained_absent:
                        sr = float(self.config.safety_learn_rate)
                        self._safety_affordance[a] += sr * (1.0 - self._safety_affordance[a])
                        self._n_safety_credit += 1
                        # Attribute to the trained predictor when the raw check
                        # did not already fire (the non-vacuity signal for 603j).
                        if trained_absent and not raw_absent:
                            self._n_safety_credit_trained += 1
            # Per-tick leak on both tables (applied once per under-threat update).
            leak = float(self.config.leak_rate)
            if leak > 0.0:
                self._relief_affordance = [v * (1.0 - leak) for v in self._relief_affordance]
                self._safety_affordance = [v * (1.0 - leak) for v in self._safety_affordance]
                self._n_decay += 1

        for i in range(k):
            self._relief_affordance[i] = float(max(0.0, min(1.0, self._relief_affordance[i])))
            self._safety_affordance[i] = float(max(0.0, min(1.0, self._safety_affordance[i])))

        self._z_harm_a_prev = z_now

    # -- Approach bonus (the directed escape) --

    def compute_approach_bias(
        self,
        z_harm_a_norm: float,
        action_classes: Union[Sequence[int], torch.Tensor],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """Per-candidate score-bias [K]. Under threat, REWARDS (negative bias --
        REE lower-is-better) each candidate whose first-action class carries
        escape-affordance credit, proportional to approach_gain * threat_scale *
        combined_affordance[class], clamped to bias_scale. The no-op / freeze
        class gets 0. Returns zeros when below threat (threat-context gate),
        when no class has credit, or under simulation_mode.
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

        ts = self.threat_scale(z_harm_a_norm)
        if ts <= 0.0:
            # Threat-context gate: zero when safe so the bridge never swamps
            # food / goal approach.
            return bias

        nc = int(self.config.noop_class)
        gain = float(self.config.approach_gain)
        scale = float(self.config.bias_scale)
        fired = False
        for i, cls in enumerate(classes):
            c = int(cls)
            if c == nc:
                continue
            aff = self._combined_affordance(c)
            if aff <= 0.0:
                continue
            mag = float(min(scale, gain * ts * aff))
            if mag > 0.0:
                bias[i] = -mag  # negative = favoured (REE lower-is-better)
                fired = True
        if fired:
            self._n_approach_fires += 1
        return bias

    # -- Read-only accessors --

    @property
    def relief_affordance(self) -> List[float]:
        return list(self._relief_affordance)

    @property
    def safety_affordance(self) -> List[float]:
        return list(self._safety_affordance)

    def best_escape_class(self) -> int:
        """Action class with the highest combined affordance (-1 if all zero)."""
        best_c, best_v = -1, 0.0
        for c in range(len(self._relief_affordance)):
            v = self._combined_affordance(c)
            if v > best_v:
                best_c, best_v = c, v
        return best_c

    def last_output(self, z_harm_a_norm: float = 0.0) -> EscapeAffordanceBridgeOutput:
        ts = self.threat_scale(z_harm_a_norm)
        best_c = self.best_escape_class()
        best_v = self._combined_affordance(best_c) if best_c >= 0 else 0.0
        return EscapeAffordanceBridgeOutput(
            threat_scale=float(ts),
            best_class=int(best_c),
            best_affordance=float(best_v),
            bias_max_abs=float(min(float(self.config.bias_scale),
                                   float(self.config.approach_gain) * ts * best_v)),
            relief_affordance_max=float(max(self._relief_affordance) if self._relief_affordance else 0.0),
            safety_affordance_max=float(max(self._safety_affordance) if self._safety_affordance else 0.0),
        )

    def get_state(self) -> dict:
        return {
            "mech358_relief_affordance_max": float(
                max(self._relief_affordance) if self._relief_affordance else 0.0
            ),
            "mech358_safety_affordance_max": float(
                max(self._safety_affordance) if self._safety_affordance else 0.0
            ),
            "mech358_best_escape_class": int(self.best_escape_class()),
            "mech358_n_relief_credit": int(self._n_relief_credit),
            "mech358_n_safety_credit": int(self._n_safety_credit),
            "mech358_n_safety_credit_trained": int(self._n_safety_credit_trained),
            "mech358_n_decay": int(self._n_decay),
            "mech358_n_approach_fires": int(self._n_approach_fires),
            "mech358_n_updates": int(self._n_updates),
            "mech358_n_sim_skipped": int(self._n_sim_skipped),
        }

    @property
    def diagnostics(self) -> dict:
        return self.get_state()
