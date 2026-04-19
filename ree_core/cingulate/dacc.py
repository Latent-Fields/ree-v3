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
from typing import List, Optional

import torch
import torch.nn as nn


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

    # MECH-260 history window (number of recent action classes to track).
    dacc_suppression_memory: int = 8

    # Precision scaling matches ARC-016 / SD-019 convention (current_precision / 500).
    dacc_precision_scale: float = 500.0

    # Shenhav EVC effort cost: mode_ev = payoff - control_required * effort_cost
    dacc_effort_cost: float = 0.1

    # Scholl 2017 drive-gated learning-rate / coupling modulation (SD-012 hook).
    # 0.0 means drive_level has no effect on dACC coupling (backward compat).
    dacc_drive_coupling: float = 0.0


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
        # Stable diagnostic counters for experiment scripts.
        self._n_forward_calls: int = 0

    def reset(self) -> None:
        """Clear per-episode state. Call on env.reset()."""
        self._pe_ema = None
        self._action_history.clear()

    def _affective_pe(
        self,
        z_harm_a: torch.Tensor,
        z_harm_a_pred: Optional[torch.Tensor],
        precision: float,
    ) -> float:
        """MECH-258: precision-weighted affective-pain PE magnitude.

        Returns a scalar float. When z_harm_a_pred is None (first step of an
        episode, before any E2HarmAForward rollout has been stored), returns
        the raw z_harm_a norm scaled by precision -- degrades gracefully to
        the urgency-style signal the system used pre-SD-032b.
        """
        if z_harm_a_pred is None:
            pe = float(z_harm_a.norm().item())
        else:
            pe = float((z_harm_a - z_harm_a_pred).norm().item())
        prec_norm = min(precision / self.config.dacc_precision_scale, 3.0)
        return pe * (1.0 + prec_norm)

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

    def forward(
        self,
        z_harm_a: torch.Tensor,
        z_harm_a_pred: Optional[torch.Tensor],
        candidate_payoffs: torch.Tensor,
        candidate_effort: torch.Tensor,
        candidate_action_classes: List[int],
        precision: float,
        drive_level: float = 0.0,
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

        pe = self._affective_pe(z_harm_a, z_harm_a_pred, precision)
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

        # Scholl 2017: drive-gated coupling. Higher drive -> higher gain on
        # dACC influence into downstream selection. dacc_drive_coupling=0
        # disables this entirely (backward compat default).
        drive_gain = 1.0 + self.config.dacc_drive_coupling * float(drive_level)

        return {
            "mode_ev": mode_ev,
            "choice_difficulty": choice_difficulty,
            "foraging_value": float(foraging_value),
            "harm_interaction": harm_interaction,
            "suppression": suppression,
            "pe": pe,
            "drive_gain": drive_gain,
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
            return torch.zeros_like(mode_ev)

        bias = -mode_ev.clone()
        if self.config.dacc_interaction_weight != 0.0:
            bias = bias + self.config.dacc_interaction_weight * (-bundle["harm_interaction"])
        if self.config.dacc_foraging_weight != 0.0:
            fv = torch.tensor(bundle["foraging_value"], dtype=dtype, device=device)
            bias = bias + self.config.dacc_foraging_weight * fv
        if self.config.dacc_suppression_weight != 0.0:
            bias = bias + self.config.dacc_suppression_weight * bundle["suppression"]

        return weight * bias
