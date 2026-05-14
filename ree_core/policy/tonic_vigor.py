"""MECH-320: tonic_vigor_coupling_score_bias (mesolimbic-DA-vigor / average-
reward-rate / opportunity-cost regulator).

First child mechanism for ARC-066 (action.tonic_vigor_coupling). Adds an
additive vigor bias to E3 trajectory scoring at the e3.select() call site,
biased toward action-trajectories (and away from no-op trajectories) by an
amount proportional to a slow EWMA over the realised E3-score-receipt
stream. Target-free: the bias applies regardless of whether any z_goal is
currently active. Sister mechanism to MECH-313 (stochastic_noise_floor /
LC-NE tonic noise) and MECH-314 (structured_curiosity_bonus) at the same
call site, on a different axis.

ARCHITECTURE

  Pure-arithmetic regulator. No internal nn.Module, no learned parameters.
  Two methods:

    update_score_receipt(score, simulation_mode):
        v_raw <- (1-alpha) * v_raw + alpha * (-score)
        # Note: REE convention is lower-score-is-better. We negate so that
        # a reward-rich history (low scores) drives the EWMA UP. v_t
        # therefore tracks "average reward rate" with the standard
        # high-is-good sign.

    compute_score_bias(per_candidate_scores, action_classes, energy, drive,
                       recent_pe, simulation_mode):
        v_t <- max(0, v_raw) * gate_energy(e) * gate_drive(d) * gate_pe(p)
        for i in 0..K-1:
          if additive form:
            bias[i] = -w_action * v_t   if action_classes[i] != noop_class
                       +w_passive * v_t   else
          if multiplicative form:
            gain[i] = (1 - w_action * v_t) if action_classes[i] != noop_class
                       (1 + w_passive * v_t) else
            bias[i] = (gain[i] - 1) * |per_candidate_scores[i]|
        bias = clamp(bias, [-bias_scale, +bias_scale])

  REE score convention: lower-is-better. A negative bias on action
  trajectories therefore favours action; a positive bias on no-op
  trajectories penalises passivity. The two terms compose at the candidate-
  score level and are mathematically equivalent to a single signed scalar
  applied to the action-vs-no-op contrast.

LIT-PULL VERDICTS (resolved defaults; see SYNTHESIS for cited evidence)

  ARC-066 lit-pull (evidence/literature/targeted_review_arc_066_tonic_vigor/
  synthesis.md, lit_conf 0.789, supports-direction, 7 entries):

    R1 -- mesolimbic DA-vigor is the load-bearing substrate (Niv 2007
          formalism + Salamone & Correa 2012 substrate identity +
          Beierholm 2013 human L-DOPA causal test). LC-NE-direction is
          REJECTED -- LC-NE tonic mode is one mechanism (noise = MECH-313),
          per Kane et al. 2017 DREADD test by the original Aston-Jones /
          Cohen authorship group.

    R3 -- ADDITIVE form is primary (Niv 2007 opportunity-cost derivation
          is naturally additive). MULTIPLICATIVE GAIN is the falsifiable
          secondary alternative; both implementable via tonic_vigor_form
          config. Discriminable by parametric sweep on a pre-existing
          action preference (additive scales linearly; multiplicative
          scales the score-gap differentially).

    R4 -- SLOW EWMA over realised E3-score-receipt is the primary
          scalar (Niv 2007 average-reward-rate formalism + Beierholm
          2013 empirical confirmation that the empirically tested
          scalar is across-trial reward history, NOT internal capacity
          composite). Internal-state proxies (energy, drive, recent PE)
          enter as SECONDARY MODULATORS.

  Magnitudes are NOT settled by the lit-pull; the discriminative-pair
  validation experiment will calibrate w_action / w_passive / bias_scale
  empirically.

DISTINCTION FROM SIBLING MECHANISMS

  MECH-313 (stochastic_noise_floor): orthogonal axis. MECH-313 lifts
    the softmax temperature uniformly (entropy on choice). MECH-320 adds
    a directional bias on the score axis (action over no-op). Both
    compose: MECH-320 says "favour action over no-op"; MECH-313 says
    "explore among action candidates". The R2 lit-pull verdict
    establishes that LC-NE tonic mode is one mechanism (noise =
    MECH-313), so MECH-320 must NOT be wired through an LC-NE-direction
    analog -- substrate is mesolimbic DA-vigor.

  MECH-314 (structured_curiosity_bonus): novelty / uncertainty / learning-
    progress driven bonus on E3 candidates. Per-candidate (or broadcast)
    scalar. MECH-320 is a target-free directional bias on action vs no-op
    only; not a function of candidate-level features.

  MECH-260 (dACC anti-recency): state-dependent recency penalty on
    recently-executed action classes. MECH-320 is state-INDEPENDENT
    (does not depend on action history) -- it depends only on the slow
    EWMA over realised reward.

  MECH-216 (e1_predictive_wanting): target-conditioned. MECH-320 is
    target-free (fires regardless of whether z_goal is active).

  SD-012 (homeostatic_drive_modulation): drive RISES as energy FALLS
    (deficit-keyed). MECH-320 vigor scales with capacity (rises when
    avg reward rate is high). Both can hold simultaneously: depletion
    recruits z_goal pursuit AND surplus recruits target-free action.
    The gate_drive secondary modulator handles the deficit-corner
    attribution.

  ARC-068 (opportunity_cost_no_op_penalty): mathematical complement.
    MECH-320's w_passive * v_t * (no-op trajectory) IS the opportunity-
    cost-on-passive computation Niv 2007 derives. Whether ARC-068
    eventually lands as a separate child mechanism with its own scalar,
    or is fully absorbed by MECH-320's w_passive term, is open until
    the ARC-068 lit-pull lands.

INTEGRATION SITE

  REEAgent.select_action() reads tonic_vigor.compute_score_bias(
  per_candidate_scores, action_classes, energy, drive, recent_pe,
  simulation_mode) AFTER the MECH-314 curiosity block and BEFORE the
  MECH-313 noise_floor temperature lift, composing additively into
  dacc_score_bias. After e3.select() returns, the realised E3 score of
  the selected candidate is fed back via update_score_receipt(score,
  simulation_mode) to advance the EWMA. See agent.py for the wiring.

MECH-094

  simulation_mode=True on either method returns zeros (compute_score_bias)
  / skips state advance (update_score_receipt) and increments only the
  simulation-skip counter. Match the SD-035 / MECH-279 / gated_policy /
  MECH-313 / MECH-314 simulation_mode pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch


# Form selector tokens.
FORM_ADDITIVE = "additive"
FORM_MULTIPLICATIVE = "multiplicative"
_VALID_FORMS = (FORM_ADDITIVE, FORM_MULTIPLICATIVE)


@dataclass
class TonicVigorConfig:
    """MECH-320 tonic-vigor regulator configuration.

    Attributes:
        use_tonic_vigor : master switch. False = disabled (default,
            backward-compatible). REEAgent does not instantiate
            TonicVigor when False.
        half_life : EWMA half-life in ticks. Default 100.0 = ~700-tick
            window for the EWMA to reach steady state. Niv 2007 / R4
            verdict: SLOW window (across-episode reward history).
        w_action : weight on the action-trajectory negative bias term.
            Default 0.1. Calibration via discriminative-pair validation.
        w_passive : weight on the no-op-trajectory positive bias (cost-
            of-passivity / ARC-068 complement). Default 0.1.
        bias_scale : clamp on the absolute magnitude of the per-candidate
            bias. Default 0.1 (mirrors lateral_pfc / curiosity bias_scale).
        gate_energy_min : energy reserve threshold below which the
            vigor scalar is gated DOWN linearly. Default 0.2 (energy
            below this is the depleted regime; SD-012 owns z_goal
            pursuit there).
        gate_drive_max : drive_level threshold above which the vigor
            scalar is gated DOWN linearly. Default 0.7. Above this the
            agent should be in z_goal-pursuit mode, not target-free
            action.
        gate_pe_max : recent prediction error threshold above which the
            vigor scalar is gated DOWN linearly. Default 1.0. Above this
            the agent should attend / consolidate / sleep, not act.
        form : implementation form selector. "additive" (default,
            primary) or "multiplicative" (falsifiable secondary).
        noop_class : action class index treated as no-op. Default 0
            (matches MECH-279 PAG freeze-gate convention).
    """

    use_tonic_vigor: bool = False
    half_life: float = 100.0
    w_action: float = 0.1
    w_passive: float = 0.1
    bias_scale: float = 0.1
    gate_energy_min: float = 0.2
    gate_drive_max: float = 0.7
    gate_pe_max: float = 1.0
    form: str = FORM_ADDITIVE
    noop_class: int = 0
    # V3-EXQ-563 diagnostic: hard floor applied to v_t before per-candidate
    # bias computation. Default 0.0 = standard behaviour (v_t gated to zero
    # when EWMA is negative, as normal). Set > 0 to force a minimum positive
    # tonic-vigor scalar regardless of the EWMA / gate state. Used in ARM_3
    # and ARM_5 of EXQ-563 to verify the score-bias -> action seam works
    # independently of the sign/scale failure on v_raw. Bit-identical to
    # prior behaviour when left at 0.0.
    v_t_floor: float = 0.0


@dataclass
class TonicVigorOutput:
    """Diagnostic snapshot for one compute_score_bias call.

    Attributes (all optional / informational):
        v_t : float, gated vigor scalar applied this tick.
        bias_max_abs : float, max(|bias|) over candidates.
        n_action_candidates : int, count of non-noop candidates.
        n_noop_candidates : int, count of noop candidates.
        gate_energy : float in [0, 1], energy gate value this tick.
        gate_drive : float in [0, 1], drive gate value this tick.
        gate_pe : float in [0, 1], PE gate value this tick.
    """

    v_t: float = 0.0
    bias_max_abs: float = 0.0
    n_action_candidates: int = 0
    n_noop_candidates: int = 0
    gate_energy: float = 1.0
    gate_drive: float = 1.0
    gate_pe: float = 1.0


class TonicVigor:
    """MECH-320 tonic-vigor regulator (target-free, capacity-keyed).

    Pure-arithmetic, no learned parameters, no nn.Module inheritance.
    Maintains a single scalar EWMA (_v_raw) over the realised E3-score-
    receipt stream. Each compute_score_bias call gates _v_raw down by
    secondary internal-state modulators (energy / drive / PE) to produce
    the effective vigor scalar v_t, then distributes a per-candidate
    bias over the K candidates by their first-step action class.

    Diagnostics tracked:
        _v_raw                       : float, ungated EWMA.
        _last_v_t                    : float, last gated vigor scalar.
        _last_output                 : TonicVigorOutput.
        _n_waking_score_updates      : int, EWMA advances on waking ticks.
        _n_simulation_score_skips    : int, EWMA skips on sim ticks.
        _n_waking_bias_calls         : int, score-bias calls on waking.
        _n_simulation_bias_skips     : int, score-bias skips on sim.
    """

    def __init__(self, config: "TonicVigorConfig | None" = None) -> None:
        self.config = config if config is not None else TonicVigorConfig()
        # Validate.
        if self.config.half_life <= 0.0:
            raise ValueError(
                "half_life must be > 0 (EWMA half-life is strictly positive). "
                f"Got {self.config.half_life}."
            )
        if self.config.w_action < 0.0:
            raise ValueError(
                "w_action must be >= 0 (action bias is a non-negative "
                f"weight). Got {self.config.w_action}."
            )
        if self.config.w_passive < 0.0:
            raise ValueError(
                "w_passive must be >= 0 (passive bias is a non-negative "
                f"weight). Got {self.config.w_passive}."
            )
        if self.config.bias_scale <= 0.0:
            raise ValueError(
                "bias_scale must be > 0 (clamp magnitude is strictly "
                f"positive). Got {self.config.bias_scale}."
            )
        if self.config.form not in _VALID_FORMS:
            raise ValueError(
                f"form must be one of {_VALID_FORMS}. Got "
                f"{self.config.form!r}."
            )
        # Derived: alpha = 1 - 0.5**(1/half_life). For half_life=100 ->
        # alpha ~ 0.00693.
        self._alpha: float = 1.0 - math.pow(0.5, 1.0 / float(self.config.half_life))
        # State.
        self._v_raw: float = 0.0
        self._last_v_t: float = 0.0
        self._last_output: TonicVigorOutput = TonicVigorOutput()
        self._n_waking_score_updates: int = 0
        self._n_simulation_score_skips: int = 0
        self._n_waking_bias_calls: int = 0
        self._n_simulation_bias_skips: int = 0

    # ------------------------------------------------------------------
    # Reward EWMA path
    # ------------------------------------------------------------------
    def update_score_receipt(
        self,
        score: float,
        simulation_mode: bool = False,
    ) -> None:
        """Advance the slow EWMA on the realised E3-score-receipt stream.

        Args:
            score : the realised E3 score of the SELECTED candidate this
                tick (REE convention: lower-is-better). The module
                internally negates this (avg reward rate = -avg score)
                so that v_raw climbs when the agent is in a reward-rich
                regime. Caller passes raw score; sign handling is internal.
            simulation_mode : MECH-094 gate. When True, EWMA is NOT
                advanced and only the simulation-skip counter is
                incremented. Match MECH-313 / MECH-314 / SD-035 / MECH-279
                pattern.
        """
        if simulation_mode:
            self._n_simulation_score_skips += 1
            return
        # Convert REE-low-is-better to high-is-good before EWMA so v_raw
        # has the standard sign.
        reward_signal = -float(score)
        self._v_raw = (1.0 - self._alpha) * self._v_raw + self._alpha * reward_signal
        self._n_waking_score_updates += 1

    # ------------------------------------------------------------------
    # Score-bias path
    # ------------------------------------------------------------------
    def compute_score_bias(
        self,
        per_candidate_scores: torch.Tensor,
        action_classes: torch.Tensor,
        energy: float,
        drive: float,
        recent_pe: float,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """Return the per-candidate additive bias vector.

        Args:
            per_candidate_scores : Tensor [K] of E3 trajectory scores.
                Read-only; multiplicative form uses |scores| as gain
                anchor. Caller may pass detached values.
            action_classes : Long tensor [K] of first-step action classes
                per candidate. The noop_class (config) is treated as the
                no-op trajectory; all other classes are action.
            energy : float in [0, 1] (or unconstrained; clipped). Energy
                reserve. Below gate_energy_min, vigor is gated DOWN.
            drive : float in [0, 1] (or unconstrained; clipped).
                Aggregate drive_level. Above gate_drive_max, vigor is
                gated DOWN.
            recent_pe : float >= 0 (clipped). Recent prediction error
                (e.g. e3._running_variance). Above gate_pe_max, vigor
                is gated DOWN.
            simulation_mode : MECH-094 gate. When True, returns zeros[K]
                + increments only the simulation-skip counter. State
                (v_raw, EWMA) is NOT advanced -- bias path is
                independent of update_score_receipt.

        Returns:
            Tensor [K], same dtype / device as per_candidate_scores,
            additive bias to be COMPOSED into dacc_score_bias upstream.
            REE convention lower-is-better: bias[i] is NEGATIVE on
            action-trajectories (favours action) and POSITIVE on
            no-op-trajectories (penalises passivity).
        """
        K = int(per_candidate_scores.shape[0])
        if simulation_mode:
            self._n_simulation_bias_skips += 1
            return torch.zeros(
                K,
                dtype=per_candidate_scores.dtype,
                device=per_candidate_scores.device,
            )

        # Apply secondary internal-state gates.
        gate_e = self._gate_energy(energy)
        gate_d = self._gate_drive(drive)
        gate_p = self._gate_pe(recent_pe)
        v_t = max(
            float(self.config.v_t_floor),
            max(0.0, self._v_raw) * gate_e * gate_d * gate_p,
        )

        # Cache.
        self._last_v_t = v_t
        self._n_waking_bias_calls += 1

        # If v_t is essentially zero, skip the per-candidate computation.
        if v_t <= 0.0:
            self._last_output = TonicVigorOutput(
                v_t=0.0,
                bias_max_abs=0.0,
                n_action_candidates=0,
                n_noop_candidates=0,
                gate_energy=gate_e,
                gate_drive=gate_d,
                gate_pe=gate_p,
            )
            return torch.zeros(
                K,
                dtype=per_candidate_scores.dtype,
                device=per_candidate_scores.device,
            )

        # Build per-candidate bias.
        device = per_candidate_scores.device
        dtype = per_candidate_scores.dtype
        action_classes_i = action_classes.long().to(device=device).view(-1)
        is_noop = action_classes_i == int(self.config.noop_class)
        n_noop = int(is_noop.sum().item())
        n_action = K - n_noop

        if self.config.form == FORM_ADDITIVE:
            # Additive primary form.
            action_term = -float(self.config.w_action) * v_t
            passive_term = float(self.config.w_passive) * v_t
            bias = torch.full((K,), action_term, dtype=dtype, device=device)
            if n_noop > 0:
                bias[is_noop] = passive_term
        else:  # FORM_MULTIPLICATIVE
            # Multiplicative gain form.
            # gain on action trajectories = (1 - w_action * v_t)
            # gain on no-op trajectories  = (1 + w_passive * v_t)
            # additive equivalent: bias[i] = (gain[i] - 1) * |score[i]|
            # = action: -w_action * v_t * |score[i]|
            # = noop:   +w_passive * v_t * |score[i]|
            score_mag = per_candidate_scores.detach().abs().to(dtype=dtype, device=device)
            action_term = -float(self.config.w_action) * v_t
            passive_term = float(self.config.w_passive) * v_t
            scale = torch.full((K,), action_term, dtype=dtype, device=device)
            if n_noop > 0:
                scale[is_noop] = passive_term
            bias = scale * score_mag

        # Clamp.
        scale_clamp = float(self.config.bias_scale)
        bias = torch.clamp(bias, min=-scale_clamp, max=scale_clamp)

        # Diagnostics.
        bias_max_abs = float(bias.detach().abs().max().item()) if K > 0 else 0.0
        self._last_output = TonicVigorOutput(
            v_t=v_t,
            bias_max_abs=bias_max_abs,
            n_action_candidates=n_action,
            n_noop_candidates=n_noop,
            gate_energy=gate_e,
            gate_drive=gate_d,
            gate_pe=gate_p,
        )
        return bias

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------
    def _gate_energy(self, energy: float) -> float:
        """gate_energy(e) = clip(e / e_min, 0, 1) when e < e_min; else 1.

        Energy below threshold linearly suppresses vigor.
        """
        e_min = float(self.config.gate_energy_min)
        if e_min <= 0.0:
            return 1.0
        e = float(energy)
        if e >= e_min:
            return 1.0
        if e <= 0.0:
            return 0.0
        return e / e_min

    def _gate_drive(self, drive: float) -> float:
        """gate_drive(d) suppresses vigor as drive rises above d_max.

        gate(d) = 1 if d <= d_max else max(0, 1 - (d - d_max) / (1 - d_max)).
        Saturated at 0 when d >= 1 (fully depleted -> SD-012 owns).
        """
        d_max = float(self.config.gate_drive_max)
        d = float(drive)
        if d <= d_max:
            return 1.0
        denom = max(1.0 - d_max, 1e-6)
        gated = 1.0 - (d - d_max) / denom
        if gated <= 0.0:
            return 0.0
        if gated >= 1.0:
            return 1.0
        return gated

    def _gate_pe(self, recent_pe: float) -> float:
        """gate_pe(p) suppresses vigor as recent PE rises above pe_max.

        gate(p) = 1 if p <= pe_max else max(0, 1 - (p - pe_max) / pe_max).
        Saturated at 0 when p >= 2 * pe_max (consolidate / sleep regime).
        """
        pe_max = float(self.config.gate_pe_max)
        if pe_max <= 0.0:
            return 1.0
        p = float(recent_pe)
        if p <= pe_max:
            return 1.0
        denom = max(pe_max, 1e-6)
        gated = 1.0 - (p - pe_max) / denom
        if gated <= 0.0:
            return 0.0
        if gated >= 1.0:
            return 1.0
        return gated

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset per-episode state.

        Clears the EWMA buffer (vigor scalar) and all diagnostic counters.
        Called from REEAgent.reset() at episode boundaries.
        """
        self._v_raw = 0.0
        self._last_v_t = 0.0
        self._last_output = TonicVigorOutput()
        self._n_waking_score_updates = 0
        self._n_simulation_score_skips = 0
        self._n_waking_bias_calls = 0
        self._n_simulation_bias_skips = 0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "v_raw": self._v_raw,
            "last_v_t": self._last_v_t,
            "last_bias_max_abs": self._last_output.bias_max_abs,
            "last_n_action_candidates": self._last_output.n_action_candidates,
            "last_n_noop_candidates": self._last_output.n_noop_candidates,
            "last_gate_energy": self._last_output.gate_energy,
            "last_gate_drive": self._last_output.gate_drive,
            "last_gate_pe": self._last_output.gate_pe,
            "n_waking_score_updates": self._n_waking_score_updates,
            "n_simulation_score_skips": self._n_simulation_score_skips,
            "n_waking_bias_calls": self._n_waking_bias_calls,
            "n_simulation_bias_skips": self._n_simulation_bias_skips,
            "alpha_derived": self._alpha,
            "form": self.config.form,
        }
