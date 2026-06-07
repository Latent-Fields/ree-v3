"""
DACCAdaptiveControl -- SD-032b dACC/aMCC-analog adaptive control.

Implements the dorsal-ACC / anterior-midcingulate-cortex analog hosting:
  - MECH-258: precision-weighted affective-pain prediction error (against the
    E2_harm_a forward model) as the core learning and control-demand signal.
  - MECH-260: bias suppression against recency and monostrategy -- dACC
    penalises repeated choice of the same option class when circumstances
    demand exploration (Scholl & Klein-Flugge 2018; Kolling 2015).

Architectural scope (SD-032b minimum-viable):
  Inputs  : z_harm_a (current), z_harm_a_pred (E2HarmAForward output from
            previous step), E3-derived precision scalar, candidate-level
            benefit/harm costs, drive_level, committed-action history.
  Outputs : Croxson-style integration bundle with four components:
              mode_ev          -- per-candidate expected value after effort cost
              choice_difficulty-- scalar spread across candidate EVs
              foraging_value   -- global exploration/exploitation signal
              harm_interaction -- reward x effort interaction term
            Consumed by the downstream striatal/E3-action-selection stage.
            (STOPGAP: consumed via DACCtoE3Adapter pending SD-032a.)

Biological grounding (see REE_assembly/evidence/literature/
targeted_review_cingulate_mcc/synthesis.md):
  Shenhav 2013     -- Expected Value of Control: mode_ev = payoff - control_required * cost
  Croxson 2009     -- dACC integrates reward and effort; supports exploration of alternatives
  Scholl 2017      -- neuromodulator-tunable learning rate in dACC; drive_level scales
                      the effective coupling into downstream selection.
  Scholl & Klein-Flugge 2018 -- MECH-260: suppression of repeated choice
  Kolling 2015     -- foraging account of dACC: signals the value of switching away
                      from the current option toward an alternative.

Explicitly NOT a selector: dACC integrates; striatum / E3 selects. The
adapter produces a bias on E3's score, it does not override the decision.

MECH-094: not applicable (waking control plane, no replay content).

Stopgap adapter:
  SD-032a (salience-network coordinator) is the proper consumer of the dACC
  bundle. Until SD-032a is implemented, DACCtoE3Adapter converts the bundle
  into a per-candidate score bias that E3.select() can consume directly.
  REMOVE DACCtoE3Adapter when SD-032a lands and the salience coordinator
  becomes the integration endpoint. The bundle shape is stable across this
  migration -- callers that consume the bundle will not need to change.

See CLAUDE.md: SD-032b, MECH-258, MECH-260, ARC-058.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from ree_core.utils.per_axis_drive import collapse_per_axis_drive


@dataclass
class DACCConfig:
    """Configuration for dACC/aMCC-analog adaptive control (SD-032b).

    All weights default to zero so the module is no-op when instantiated
    without explicit activation. Master switch lives on REEConfig.use_dacc.
    """
    # Bundle weight scales (STOPGAP adapter uses these to form score bias).
    dacc_weight: float = 0.0                 # overall gain on the bundle's action on E3 score
    dacc_interaction_weight: float = 0.0     # weight on reward x effort interaction
    dacc_foraging_weight: float = 0.0        # weight on foraging/exploration signal
    dacc_suppression_weight: float = 0.0     # MECH-260 recency/monostrategy suppression weight
    dacc_goal_readout_weight: float = 0.0    # SD-057 L7 (MECH-348): object-discriminative z_goal readout weight (0=no-op)

    # MECH-260 history window (number of recent action classes to track).
    dacc_suppression_memory: int = 8

    # Precision scaling matches ARC-016 / SD-019 convention (current_precision / 500).
    dacc_precision_scale: float = 500.0

    # Shenhav EVC effort cost: mode_ev = payoff - control_required * effort_cost
    dacc_effort_cost: float = 0.1

    # Scholl 2017 drive-gated learning-rate / coupling modulation (SD-012 hook).
    # 0.0 means drive_level has no effect on dACC coupling (backward compat).
    dacc_drive_coupling: float = 0.0

    # Absolute cap on the total score bias emitted by DACCtoE3Adapter.
    # Prevents score_bias from dominating E3 inter-candidate variation
    # (which is typically O(0.5-2.0)) when harm_eval_head raw scores are
    # large (O(10-40)). 0.0 = no cap (backward compat). Recommended: 2.0.
    dacc_bias_max_abs: float = 0.0

    # MECH-268 / SD-034 hook: absolute cap on precision-weighted PE after
    # closure events. None -> no cap (backward compat). When set (e.g. by
    # ClosureOperator on fire), subsequent _affective_pe() outputs are
    # clamped to this value, preventing the just-ended episode's residual
    # PE from continuing to drive control demand. This is the "cap / reset
    # MECH-268 pe buffer" component of SD-034's 5-part closure signal.
    dacc_pe_cap: Optional[float] = None

    # MECH-268: history-conditioned PE saturation (habituation under repeated
    # identical outcomes). Distinct from dacc_pe_cap (absolute clamp): this
    # attenuates the precision-weighted PE as a function of how often the
    # current outcome class recurs in a recent window. Closure events
    # (SD-034) reset the buffer so a new rule-state starts fresh.
    #
    # Saturation function:
    #     n_rec        = count(current_outcome_class in last saturation_window outcomes)
    #     excess       = max(0, n_rec - saturation_grace)
    #     sat_factor   = 1.0 / (1.0 + saturation_strength * excess)
    #     pe_saturated = pe_after_cap * sat_factor
    #
    # Defaults preserve backward compat: dacc_saturation_enabled=False ->
    # pe_saturated == pe_after_cap (no attenuation). When enabled with
    # default knobs (window=8, strength=0.3, grace=2), the bundle's pe
    # field smoothly decays toward zero as a single outcome class
    # dominates the recent window -- the rumination habituation route
    # identified in the 2026-04-20 GAP MEMO.
    dacc_saturation_enabled: bool = False
    dacc_saturation_window: int = 8
    dacc_saturation_strength: float = 0.3
    dacc_saturation_grace: int = 2


class DACCAdaptiveControl(nn.Module):
    """SD-032b dACC/aMCC-analog adaptive control.

    Stateful over steps:
      _pe_ema            -- slow EMA of precision-weighted affective PE,
                            used as the foraging signal baseline.
      _action_history    -- ring buffer of recent action-class indices for
                            MECH-260 recency/monostrategy suppression.

    Non-trainable (no gradient flow): this is a control-plane computation.
    State is reset by the caller via .reset() on episode boundaries.
    """

    def __init__(self, config: Optional[DACCConfig] = None):
        super().__init__()
        self.config = config or DACCConfig()
        self._pe_ema: Optional[float] = None
        self._pe_ema_alpha: float = 0.05
        self._action_history: List[int] = []
        # MECH-268: outcome-class FIFO for the saturation function. Distinct
        # from _action_history (MECH-260) -- outcomes are what happened
        # after the action (e.g. harm-vs-no-harm class), not the chosen
        # action class itself. Most recent at the tail.
        self._outcome_history: List[int] = []
        # MECH-268 last-cycle diagnostics (set by _affective_pe / forward).
        self._last_pe_unsaturated: Optional[float] = None
        self._last_saturation_factor: float = 1.0
        self._last_outcome_recurrence: int = 0
        # Stable diagnostic counters for experiment scripts.
        self._n_forward_calls: int = 0

    def reset(self) -> None:
        """Clear per-episode state. Call on env.reset()."""
        self._pe_ema = None
        self._action_history.clear()
        self._outcome_history.clear()
        self._last_pe_unsaturated = None
        self._last_saturation_factor = 1.0
        self._last_outcome_recurrence = 0

    def _affective_pe(
        self,
        z_harm_a: torch.Tensor,
        z_harm_a_pred: Optional[torch.Tensor],
        precision: float,
        current_outcome_class: Optional[int] = None,
    ) -> float:
        """MECH-258 + SD-034 cap + MECH-268 saturation -> control PE.

        Pipeline:
            pe_unweighted = ||z_harm_a - z_harm_a_pred||  (or norm if no pred)
            pe_weighted   = pe_unweighted * (1 + precision_norm)   # MECH-258
            pe_capped     = min(pe_weighted, dacc_pe_cap)          # SD-034 abs clamp
            pe_saturated  = pe_capped * sat_factor(history, class) # MECH-268

        sat_factor is computed against the outcome-history FIFO at the time
        of this call. When current_outcome_class is None (caller did not
        tag the outcome yet, e.g. first tick of an episode), saturation
        falls back to using the most-recent buffered class; if the buffer
        is empty saturation is 1.0 (no attenuation).

        Diagnostic side effects: self._last_pe_unsaturated stores the
        post-cap value; self._last_saturation_factor stores the factor
        applied; self._last_outcome_recurrence stores the recurrence
        count used. These let experiment scripts assert the saturation
        actually fired without re-deriving from inputs.
        """
        if z_harm_a_pred is None:
            pe = float(z_harm_a.norm().item())
        else:
            pe = float((z_harm_a - z_harm_a_pred).norm().item())
        prec_norm = min(precision / self.config.dacc_precision_scale, 3.0)
        pe_out = pe * (1.0 + prec_norm)
        # MECH-268 / SD-034: absolute post-closure precision-weighted PE cap.
        cap = self.config.dacc_pe_cap
        if cap is not None and pe_out > cap:
            pe_out = float(cap)
        self._last_pe_unsaturated = pe_out

        # MECH-268: history-conditioned saturation.
        sat_factor, n_rec = self._saturation_factor(current_outcome_class)
        self._last_saturation_factor = sat_factor
        self._last_outcome_recurrence = n_rec
        return pe_out * sat_factor

    def _saturation_factor(
        self, current_outcome_class: Optional[int]
    ) -> tuple:
        """MECH-268 f_sat: returns (sat_factor in (0, 1], n_recurrences).

        Returns 1.0 when saturation is disabled, when no class can be
        determined, or when n_recurrences is below the grace count.
        """
        if not self.config.dacc_saturation_enabled:
            return 1.0, 0
        cls = current_outcome_class
        if cls is None:
            if not self._outcome_history:
                return 1.0, 0
            cls = self._outcome_history[-1]
        cls = int(cls)
        window = max(1, int(self.config.dacc_saturation_window))
        recent = self._outcome_history[-window:]
        n_rec = sum(1 for o in recent if o == cls)
        excess = max(0, n_rec - int(self.config.dacc_saturation_grace))
        if excess <= 0:
            return 1.0, n_rec
        strength = float(self.config.dacc_saturation_strength)
        sat_factor = 1.0 / (1.0 + strength * excess)
        return sat_factor, n_rec

    def record_outcome(self, outcome_class: int) -> None:
        """MECH-268: push an outcome class onto the saturation FIFO.

        Caller decides what counts as an outcome class -- typical choices
        are harm-vs-no-harm (binary), reward-class id, or rule-state
        completion tag. The buffer is bounded by dacc_saturation_window;
        ClosureOperator clears it on rule-completion firing.
        """
        self._outcome_history.append(int(outcome_class))
        window = max(1, int(self.config.dacc_saturation_window))
        # Keep one extra slot of slack so window+1 oldest entries are
        # discarded; the saturation read is sliced to last `window` only.
        max_keep = window * 2
        if len(self._outcome_history) > max_keep:
            self._outcome_history = self._outcome_history[-max_keep:]

    def reset_outcome_history(self) -> None:
        """SD-034 closure hook: clear the MECH-268 outcome FIFO.

        Called from ClosureOperator._fire() so that a freshly-completed
        rule-state starts the next cycle with no habituation accrued
        from the previous one. Distinct from reset() which also clears
        action history and PE EMA -- closure is more targeted.
        """
        self._outcome_history.clear()
        self._last_saturation_factor = 1.0
        self._last_outcome_recurrence = 0

    def _update_pe_ema(self, pe: float) -> float:
        """Running EMA of the precision-weighted PE; baseline for foraging signal."""
        if self._pe_ema is None:
            self._pe_ema = pe
        else:
            self._pe_ema = (1.0 - self._pe_ema_alpha) * self._pe_ema + self._pe_ema_alpha * pe
        return self._pe_ema

    def _suppression_penalty(self, action_class: int) -> float:
        """MECH-260: recency/monostrategy penalty for a candidate action class.

        Penalty = frequency of this class in the recent history window, in [0, 1].
        Higher penalty means the class has been chosen more often recently.
        """
        if not self._action_history:
            return 0.0
        n = len(self._action_history)
        matches = sum(1 for a in self._action_history if a == action_class)
        return matches / n

    def record_action(self, action_class: int) -> None:
        """Push a chosen action class onto the history ring for MECH-260."""
        self._action_history.append(int(action_class))
        if len(self._action_history) > self.config.dacc_suppression_memory:
            self._action_history.pop(0)

    def inject_nogo(self, action_class: int, count: int) -> int:
        """SD-034 targeted No-Go: push `action_class` onto history `count` times.

        Used by the ClosureOperator when a rule-episode completes: biases
        MECH-260 recency suppression strongly against re-selecting the
        just-completed action class, implementing the "targeted No-Go on
        the just-completed rule_state" component of SD-034's closure
        signal. This is mechanistically identical to record_action() but
        semantically marked: the count reflects governance intent, not
        repeated execution.

        Args:
            action_class: int action-class tag to inject.
            count: number of times to push (clipped by history memory).

        Returns:
            int: number of entries actually pushed (may be less than
            count if history memory is smaller).
        """
        memory = self.config.dacc_suppression_memory
        n = int(max(0, min(count, memory)))
        for _ in range(n):
            self._action_history.append(int(action_class))
            if len(self._action_history) > memory:
                self._action_history.pop(0)
        return n

    def reset_episode_pe(self) -> None:
        """SD-034 MECH-268 hook: clear the PE EMA baseline after closure.

        Rebaselines the precision-weighted affective-pain PE EMA. Distinct
        from reset() which also clears _action_history (too destructive
        for a closure event -- targeted No-Go needs the history to persist).
        Call this from ClosureOperator._fire() when reset_pe_ema=True.
        """
        self._pe_ema = None

    def forward(
        self,
        z_harm_a: torch.Tensor,
        z_harm_a_pred: Optional[torch.Tensor],
        candidate_payoffs: torch.Tensor,
        candidate_effort: torch.Tensor,
        candidate_action_classes: List[int],
        precision: float,
        drive_level: float = 0.0,
        current_outcome_class: Optional[int] = None,
        per_axis_drive: Optional[Union[Sequence[float], np.ndarray, torch.Tensor]] = None,
        per_axis_combiner: str = "max",
        candidate_goal_proximity: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute the Croxson integration bundle for the current step.

        Args:
            z_harm_a:         [z_harm_a_dim] current affective-pain latent
            z_harm_a_pred:    [z_harm_a_dim] predicted from previous step's
                              E2HarmAForward rollout, or None on first step
            candidate_payoffs:[K] per-candidate benefit / value (E3 f-score
                              with sign flipped so higher = better, or any
                              caller-chosen payoff proxy)
            candidate_effort: [K] per-candidate effort / commitment cost
                              (e.g. harm-forward cost, trajectory length)
            candidate_action_classes: [K] int action-class tags used for
                              MECH-260 recency lookup
            precision:        scalar from E3.current_precision (ARC-016)
            drive_level:      scalar drive / homeostatic urgency in [0,1]
                              (SD-012 hook); scales dACC coupling via
                              dacc_drive_coupling.

        Returns a dict:
            mode_ev            [K]     Shenhav EVC per candidate
            choice_difficulty  scalar  std of mode_ev across candidates
            foraging_value     scalar  PE-above-baseline signal, Kolling 2015
            harm_interaction   [K]     reward x effort cross term (Croxson 2009)
            suppression        [K]     MECH-260 per-candidate recency penalty
            pe                 scalar  precision-weighted affective PE (MECH-258)
            drive_gain         scalar  effective dACC coupling after drive mod
        """
        self._n_forward_calls += 1

        pe = self._affective_pe(
            z_harm_a, z_harm_a_pred, precision,
            current_outcome_class=current_outcome_class,
        )
        pe_baseline = self._update_pe_ema(pe)

        # Shenhav 2013 EVC: payoff minus control-required * cost.
        # control_required scales with PE (more PE implies more dACC engagement).
        control_required = pe * self.config.dacc_effort_cost
        mode_ev = candidate_payoffs - control_required * candidate_effort

        # Croxson 2009 reward x effort interaction: dACC is sensitive to the
        # conjunction, not the linear sum. Signed so that high reward + low
        # effort candidates get a positive interaction boost.
        # Normalise each term to zero-mean before multiplying so the
        # interaction is about relative magnitudes, not absolute scale.
        payoff_c = candidate_payoffs - candidate_payoffs.mean()
        effort_c = candidate_effort - candidate_effort.mean()
        harm_interaction = payoff_c * (-effort_c)

        # Kolling 2015 foraging: how much the current PE exceeds baseline ->
        # value of switching away from the current option class.
        foraging_value = max(0.0, pe - pe_baseline)

        # Choice difficulty: spread of per-candidate EVs. Small spread -> hard
        # choice (ambiguity), large spread -> easy choice.
        if mode_ev.numel() > 1:
            choice_difficulty = float(mode_ev.std(unbiased=False).item())
        else:
            choice_difficulty = 0.0

        # MECH-260: per-candidate suppression penalty from action history.
        suppression = torch.tensor(
            [self._suppression_penalty(c) for c in candidate_action_classes],
            dtype=mode_ev.dtype,
            device=mode_ev.device,
        )

        # SD-049 Phase 3: when a per-axis drive vector is supplied, the dACC
        # control-demand coupling reads the worst-axis (max combiner default
        # -- biology: control demand follows the most-pressing deficit, not
        # the integrated load). Bit-identical to legacy when per_axis_drive
        # is None.
        eff_drive_for_dacc: float
        eff_drive_for_dacc_scalar: Optional[float] = collapse_per_axis_drive(
            per_axis_drive, mode=per_axis_combiner
        )
        if eff_drive_for_dacc_scalar is not None:
            eff_drive_for_dacc = eff_drive_for_dacc_scalar
        else:
            eff_drive_for_dacc = float(drive_level)
        # Scholl 2017: drive-gated coupling. Higher drive -> higher gain on
        # dACC influence into downstream selection. dacc_drive_coupling=0
        # disables this entirely (backward compat default).
        drive_gain = 1.0 + self.config.dacc_drive_coupling * eff_drive_for_dacc

        return {
            "mode_ev": mode_ev,
            "choice_difficulty": choice_difficulty,
            "foraging_value": float(foraging_value),
            "harm_interaction": harm_interaction,
            "suppression": suppression,
            "pe": pe,
            "drive_gain": drive_gain,
            # ControlVector logging (rec-B four-signal adjudication): expose
            # the Shenhav EVC effort term explicitly so the C_effort signal is
            # readable without re-deriving it from (payoff - mode_ev).
            # control_required = pe * dacc_effort_cost (PE-scaled control
            # demand); effort_term = control_required * candidate_effort, the
            # per-candidate quantity mode_ev subtracts. Additive keys -- no
            # existing consumer reads them; bit-identical for current callers.
            "control_required": float(control_required),
            "effort_term": (control_required * candidate_effort).detach(),
            # SD-057 L7 (MECH-348): per-candidate goal_proximity to the (now
            # object-bound, via SD-057 L4) z_goal. None when the caller does not
            # supply it -> the adapter's goal-readout term is skipped (no-op).
            "goal_readout": candidate_goal_proximity,
            # MECH-268 diagnostics: pe is the post-saturation value used by
            # downstream consumers; pe_unsaturated is post-cap pre-saturation;
            # saturation_factor and outcome_recurrence let scripts assert
            # the saturation actually fired.
            "pe_unsaturated": float(self._last_pe_unsaturated)
                if self._last_pe_unsaturated is not None
                else float(pe),
            "saturation_factor": float(self._last_saturation_factor),
            "outcome_recurrence": int(self._last_outcome_recurrence),
        }


class DACCtoE3Adapter(nn.Module):
    """STOPGAP adapter: dACC integration bundle -> per-candidate E3 score bias.

    This module exists solely because SD-032a (salience-network coordinator)
    has not been implemented yet. SD-032a is the proper consumer of the dACC
    bundle and will arbitrate between dACC, vACC, insular, and other inputs
    before biasing action selection. Until that lands, DACCtoE3Adapter takes
    the raw bundle and produces a [K] score-bias vector for E3.select().

    E3 convention: score = f + lambda_eff * m + rho_residue * phi ; lower
    score is better (softmax over -score). Therefore a favourable bias is
    NEGATIVE (reduces the score). We emit the bias on the same sign
    convention: score_bias is added directly to score.

    REMOVE this class when SD-032a lands. Callers of
    DACCAdaptiveControl.forward(...) that use the bundle directly do not
    need to be changed -- only the wiring at agent.select_action() does.

    Marked with explicit logging hooks (n_bias_calls) so scripts can assert
    the adapter is actually engaged during a run.
    """

    def __init__(self, config: Optional[DACCConfig] = None):
        super().__init__()
        self.config = config or DACCConfig()
        self._n_bias_calls: int = 0

    def forward(self, bundle: dict) -> torch.Tensor:
        """Convert a DACCAdaptiveControl bundle into an E3 score bias.

        bias = dacc_weight * drive_gain * (
            -mode_ev
          + dacc_interaction_weight * (-harm_interaction)
          + dacc_foraging_weight * foraging_value
          + dacc_suppression_weight * suppression
        )

        Signs:
          mode_ev          : higher = better candidate -> subtract from score
          harm_interaction : higher = better cross term -> subtract from score
          foraging_value   : scalar; uniformly raises all candidates' score,
                             making commitment harder overall (encourages
                             exploration without biasing within the set)
          suppression      : higher = more recent -> add to score (avoid)

        All multipliers default to 0, so with default config the bias is the
        zero vector regardless of bundle content. Flag activation is
        therefore a single entry point: setting dacc_weight > 0 (combined
        with at least one sub-weight > 0).
        """
        self._n_bias_calls += 1

        mode_ev = bundle["mode_ev"]
        device = mode_ev.device
        dtype = mode_ev.dtype

        weight = self.config.dacc_weight * float(bundle.get("drive_gain", 1.0))
        if weight == 0.0:
            scaled = torch.zeros_like(mode_ev)
        else:
            bias = -mode_ev.clone()
            if self.config.dacc_interaction_weight != 0.0:
                bias = bias + self.config.dacc_interaction_weight * (-bundle["harm_interaction"])
            if self.config.dacc_foraging_weight != 0.0:
                fv = torch.tensor(bundle["foraging_value"], dtype=dtype, device=device)
                bias = bias + self.config.dacc_foraging_weight * fv
            if self.config.dacc_suppression_weight != 0.0:
                bias = bias + self.config.dacc_suppression_weight * bundle["suppression"]

            scaled = weight * bias
            max_abs = self.config.dacc_bias_max_abs
            if max_abs > 0.0:
                scaled = scaled.clamp(-max_abs, max_abs)

        # SD-057 L7 (MECH-348): object-discriminative goal readout. Added
        # independently of dacc_weight (so a goal-conditioned consumer works
        # even if the legacy dACC bias is off), and AFTER the dacc_bias clamp
        # (it has its own weight controlling magnitude). proximity high ->
        # bias DOWN -> candidate favoured (REE lower-is-better). Skipped (and
        # bit-identical to the legacy adapter) when dacc_goal_readout_weight==0
        # or no goal_readout was supplied.
        grw = getattr(self.config, "dacc_goal_readout_weight", 0.0)
        gr = bundle.get("goal_readout", None)
        if grw != 0.0 and gr is not None:
            gr_t = gr if isinstance(gr, torch.Tensor) else torch.tensor(
                gr, dtype=dtype, device=device
            )
            scaled = scaled - grw * gr_t.to(device=device, dtype=dtype)
        return scaled
