"""SD-033b: OFC-analog (specific-outcome / task-structure substrate).

Second priority of the SD-033 PFC subdivision cluster. Holds a state_code
latent representing the agent's position in task structure -- the
biological substrate address E2's specific-outcome predictions read from.

ree-v3's E2 is already a specific-outcome predictor computationally
(per ARC-033 / SD-013). SD-033b does NOT reimplement E2. It registers a
named substrate that:
  (i)  carries an explicit state-space-structure latent (Stalnaker, Cooch
       & Schoenbaum 2015 cognitive-map function);
  (ii) consumes MECH-261's write_gate("sd_033b") so writes are gated by
       operating_mode (speculative under internal_planning, suppressed
       under internal_replay, consolidative under offline_consolidation,
       active read under external_task);
  (iii) projects a per-candidate top-down bias into E3 trajectory scoring,
       parallel to the SD-033a rule-bias path (Rudebeck & Murray 2014
       OFC role in option evaluation).

Two functional signatures (MECH-263, deferred to behavioural EXQs):
  (a) Devaluation sensitivity -- if an outcome's value changes while the
      state-action -> outcome mapping does not, behaviour changes
      appropriately. Tested via env-side outcome relabelling.
  (b) Same-sensory-input / different-task-role discrimination -- the
      state_code distinguishes states that are perceptually identical but
      occupy different positions in task structure.

This landing is the substrate address only; behavioural validation of
(a) and (b) requires environment work and is queued separately.

--------------------------------------------------------------------------------
DESIGN ALTERNATIVES (parallel to SD-033a)

  B1. State-code source: z_world only vs z_world + outcome signal
      CHOSEN: z_world + outcome_signal where outcome_signal is a pooled
              z_harm projection (the closest REE proxy for "what just
              happened" in outcome terms). Source =
                world_proj(z_world) + outcome_pool_weight * outcome_proj(z_harm)
              When z_harm is None, source = world_proj(z_world) alone.
      ALTERNATIVE: z_world only (state without outcome context). Cleaner
              but loses the OFC outcome-prediction grounding.
      Lit-pull question: does OFC state-coding require outcome context
      to identify task-structural role, or is structural role inferable
      from sensory state alone in primate single-unit data?

  B2. Per-candidate bias vs uniform bias
      CHOSEN: per-candidate, mirroring SD-033a A1. The bias head takes
              concat([state_code, per-candidate z_world summary]) ->
              scalar -> [K] bias.
      ALTERNATIVE: single state-only scalar bias.
      Same lit-pull question as SD-033a A1.

  B3. Frozen-random bias head with last layer zeroed vs trained head
      CHOSEN: frozen-random, last-Linear weights and bias zeroed at init
              so initial output is exactly zero -- bit-identical to
              baseline OFF until the head is deliberately trained.
      ALTERNATIVE: trained head via phased-training protocol (P0 encoder
              warmup, P1 frozen-encoder head training under devaluation
              labels, P2 eval).
      Same rationale as SD-033a A2.

  B4. EMA persistence vs richer dynamics
      CHOSEN: gate-modulated EMA (parallel to SD-033a A3). state_code <-
              (1 - eff_eta) * state_code + eff_eta * source where
              eff_eta = update_eta * gate.
      ALTERNATIVE: GRU / synaptic-hold (Mansouri 2020-style recurrent
              activity for task-set persistence).

--------------------------------------------------------------------------------
INPUTS

  z_world : [batch, world_dim]
      Exteroceptive latent. Primary source for state_code (OFC reads
      sensory state). Also used as per-candidate context for bias.
  z_harm : Optional[[batch, harm_dim]]
      Sensory-discriminative harm latent (SD-010/SD-011 z_harm_s). When
      provided, contributes to source as a coarse outcome-context signal.
      None -> outcome_proj contribution is zero.
  gate : float in [0, 1]
      SalienceCoordinator.write_gate("sd_033b"). Modulates the effective
      EMA rate. Registry default per mode: external_task=1.0,
      internal_planning=0.5, internal_replay=0.05, offline_consolidation=0.3.

OUTPUTS

  state_code : [1, state_dim] buffer
      Updated in place by update(). Reset on episode boundary via reset().
      Cross-tick persistence within episode.
  score_bias : [K] tensor (K = num_candidates)
      Additive bias on E3 per-candidate scores. Clamped to
      [-bias_scale, +bias_scale].

--------------------------------------------------------------------------------
MECH-094

state_code is gated by MECH-261, not by an explicit hypothesis_tag. The
sd_033b registry profile (suppressed under internal_replay, gate=0.05)
generalises the MECH-094 tag semantics: replayed content cannot
meaningfully overwrite the state-structure latent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class OFCConfig:
    """SD-033b configuration.

    Attributes:
        use_ofc_analog: master switch (default False, backward-compatible).
        state_dim: state_code dimensionality. Small by intent (task-structure
            summary, not a full state-space embedding).
        update_eta: base EMA rate; effective rate = update_eta * gate each tick.
            0.05 gives ~20-step rise time under gate=1.0.
        outcome_pool_weight: blend weight on mean(z_harm) in the source:
            source = world_proj(z_world) + outcome_pool_weight * outcome_proj(z_harm).
            Set 0.0 to ablate the outcome-context contribution (B1 alternative).
        bias_scale: clamp on |score_bias|.
        hidden_dim: bias head hidden-layer width.
        harm_dim: dimensionality of the optional z_harm input. When 0,
            the outcome projection is omitted entirely.
    """

    use_ofc_analog: bool = False
    state_dim: int = 16
    update_eta: float = 0.05
    outcome_pool_weight: float = 0.5
    bias_scale: float = 0.1
    hidden_dim: int = 32
    harm_dim: int = 0
    # When True, OFCAnalog.query_outcome() is active and delegates to the
    # E2HarmSForward on REEAgent for specific-outcome prediction (MECH-263
    # oracle path). Requires use_e2_harm_s_forward=True on the agent config.
    # Default False = oracle disabled; query_outcome() raises AssertionError
    # if called when False.
    use_outcome_oracle: bool = False
    # SD-033b GAP-8 (mirror of SD-033a GAP-D). When True, the state_bias_head's
    # last Linear is NOT zeroed at init (random init preserved). Enables the head
    # to be added to an experiment optimizer and trained via the E3 score-
    # aggregation gradient (the deferred trained-OFC-head behavioural arm).
    # Default False = last Linear zeroed (bit-identical to the original SD-033b
    # landing; bias output stays 0 until deliberately trained).
    train_state_bias_head: bool = False


class OFCAnalog(nn.Module):
    """SD-033b OFC-analog: state_code + per-candidate bias head.

    Stateful across ticks within an episode. reset() called by
    REEAgent.reset() on episode boundary.
    """

    def __init__(
        self,
        world_dim: int,
        config: Optional[OFCConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else OFCConfig()
        self.world_dim = world_dim
        state_dim = self.config.state_dim
        hidden_dim = self.config.hidden_dim

        self.world_proj = nn.Linear(world_dim, state_dim)

        if self.config.harm_dim > 0:
            self.outcome_proj: Optional[nn.Linear] = nn.Linear(
                self.config.harm_dim, state_dim
            )
        else:
            self.outcome_proj = None

        # When train_state_bias_head=False (default / GAP-8 off): last Linear
        # weights + bias ZEROED so initial bias output = 0. Bit-identical to
        # OFF until deliberately trained.
        # When train_state_bias_head=True (GAP-8 on): last Linear keeps random
        # init so gradient moves it from the first optimizer step.
        self.state_bias_head = nn.Sequential(
            nn.Linear(state_dim + world_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if not self.config.train_state_bias_head:
            with torch.no_grad():
                last_linear = self.state_bias_head[-1]
                last_linear.weight.zero_()
                last_linear.bias.zero_()

        self.register_buffer(
            "state_code",
            torch.zeros(1, state_dim),
            persistent=False,
        )

        self._last_gate: float = 0.0
        self._last_effective_eta: float = 0.0
        self._last_bias_abs_mean: float = 0.0
        self._last_outcome_prediction: Optional[torch.Tensor] = None

    def update(
        self,
        z_world: torch.Tensor,
        z_harm: Optional[torch.Tensor] = None,
        gate: float = 1.0,
    ) -> None:
        """Gate-modulated EMA update of state_code.

        Args:
            z_world: [batch, world_dim]. Detached.
            z_harm: Optional[[batch, harm_dim]]. Detached. When None or
                outcome_proj is unset, outcome contribution is zero.
            gate: float in [0, 1]. write_gate("sd_033b") from
                SalienceCoordinator.

        Effect:
            state_code <- (1 - eff_eta) * state_code + eff_eta * source
            where eff_eta = update_eta * clip(gate, 0, 1)
                  source  = world_proj(z_world).mean(0) +
                            outcome_pool_weight * outcome_proj(z_harm).mean(0)
            No gradient flow.
        """
        if not self.config.use_ofc_analog:
            return
        g = float(max(0.0, min(1.0, gate)))
        eff_eta = self.config.update_eta * g

        with torch.no_grad():
            zw = z_world.detach()
            if zw.dim() == 1:
                zw = zw.unsqueeze(0)
            world_contrib = self.world_proj(zw).mean(dim=0, keepdim=True)
            source = world_contrib

            if self.outcome_proj is not None and z_harm is not None:
                zh = z_harm.detach()
                if zh.dim() == 1:
                    zh = zh.unsqueeze(0)
                outcome_contrib = self.outcome_proj(zh).mean(dim=0, keepdim=True)
                source = source + self.config.outcome_pool_weight * outcome_contrib

            self.state_code.mul_(1.0 - eff_eta).add_(eff_eta * source)

        self._last_gate = g
        self._last_effective_eta = eff_eta

    def compute_bias(
        self,
        candidate_world_summaries: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-candidate score_bias from state_code + candidate summaries.

        Args:
            candidate_world_summaries: [K, world_dim].

        Returns:
            score_bias: [K] tensor, clamped to [-bias_scale, +bias_scale].
            Initial output is exactly zero (last Linear zeroed) so
            use_ofc_analog=True is bit-identical to OFF until the head is
            deliberately trained.
        """
        if not self.config.use_ofc_analog:
            return torch.zeros(
                candidate_world_summaries.shape[0],
                device=candidate_world_summaries.device,
                dtype=candidate_world_summaries.dtype,
            )

        k = candidate_world_summaries.shape[0]
        state_repeated = self.state_code.expand(k, -1)
        joined = torch.cat([state_repeated, candidate_world_summaries], dim=-1)
        bias_raw = self.state_bias_head(joined).squeeze(-1)
        bias = bias_raw.clamp(
            min=-self.config.bias_scale,
            max=self.config.bias_scale,
        )

        with torch.no_grad():
            self._last_bias_abs_mean = float(bias.abs().mean().item())

        return bias

    def bias_head_parameters(self):
        """Return state_bias_head parameters for experiment optimizer inclusion.

        SD-033b GAP-8 (mirror of SD-033a GAP-D). Experiment scripts that set
        train_state_bias_head=True should add these to their P1 optimizer:
            optim.Adam(list(agent.ofc.bias_head_parameters()), lr=LR)
        Gradient flows: E3 loss -> score_bias -> compute_bias() -> these weights.
        """
        return self.state_bias_head.parameters()

    @property
    def oracle_is_ready(self) -> bool:
        """True when the specific-outcome oracle path is enabled (MECH-263)."""
        return self.config.use_outcome_oracle

    def query_outcome(
        self,
        z_harm_s: torch.Tensor,
        action: torch.Tensor,
        e2_harm_s: "E2HarmSForward",  # noqa: F821 -- forward ref, avoid circular import
    ) -> torch.Tensor:
        """Predict next z_harm_s via the E2HarmSForward specific-outcome oracle.

        Named interface: OFC is the explicit query point for specific-outcome
        prediction. Delegates to e2_harm_s.forward() without new parameters.
        No gradient flow (no_grad wrapper + detach on output).

        Args:
            z_harm_s: [batch, harm_dim] sensory-discriminative harm latent.
            action:   [batch, action_dim] action to query.
            e2_harm_s: E2HarmSForward instance from REEAgent.

        Returns:
            z_harm_s_pred: [batch, harm_dim] predicted next harm latent.

        Raises:
            AssertionError when config.use_outcome_oracle is False.
        """
        assert self.config.use_outcome_oracle, (
            "query_outcome() called but OFCConfig.use_outcome_oracle is False"
        )
        with torch.no_grad():
            pred = e2_harm_s.forward(z_harm_s.detach(), action.detach())
        self._last_outcome_prediction = pred.detach()
        return pred.detach()

    def reset(self) -> None:
        """Zero state_code. Called on episode boundary via REEAgent.reset()."""
        with torch.no_grad():
            self.state_code.zero_()
        self._last_gate = 0.0
        self._last_effective_eta = 0.0
        self._last_bias_abs_mean = 0.0
        self._last_outcome_prediction = None

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        result = {
            "state_code_norm": float(self.state_code.norm().item()),
            "last_gate": self._last_gate,
            "last_effective_eta": self._last_effective_eta,
            "last_bias_abs_mean": self._last_bias_abs_mean,
            "oracle_enabled": self.config.use_outcome_oracle,
            "train_state_bias_head": self.config.train_state_bias_head,
        }
        if self._last_outcome_prediction is not None:
            result["last_oracle_pred_norm"] = float(
                self._last_outcome_prediction.norm().item()
            )
        return result
