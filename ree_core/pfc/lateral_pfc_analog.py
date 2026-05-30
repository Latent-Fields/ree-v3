"""SD-033a: Lateral-PFC-analog (mid-lateral rule/goal substrate).

Primary consumer of the MECH-261 write-gate registry (SalienceCoordinator). Holds a
small rule-state vector with gated persistence, projects that state (concatenated
with a per-candidate summary of z_world) to a per-trajectory scalar bias that enters
E3 as score_bias.

Four functional signatures (SD-033 architecture doc, section SD-033a):
  (i)   Stimulus-abstracted format: rule_state holds a latent rule vector; not tied
        to a specific perceptual template.
  (ii)  Distractor-resistant persistence: rule_state updates via gate-modulated EMA;
        when the MECH-261 gate for sd_033a is low (e.g. internal_replay mode),
        persistence is strong (low effective eta).
  (iii) Top-down bias into E3: rule_bias_head emits a per-trajectory scalar bias
        that is added to E3 scores (sign: lower-is-better, so positive bias
        penalises a trajectory).
  (iv)  Training-dependent emergence: the bias head is a trainable nn.Module. In
        this landing implementation the head is frozen-random with the LAST
        Linear zeroed so the initial bias output is exactly zero -- baseline
        behaviour when use_lateral_pfc_analog=True is identical to baseline OFF.
        A trained-head variant is a deliberate V3 choice deferred to a later
        ablation (see DESIGN ALTERNATIVES below).

--------------------------------------------------------------------------------
DESIGN ALTERNATIVES (documented per user guidance 2026-04-20)

These choices were made up-front; each has a documented alternative and a queued
lit-pull task (REE_assembly/evidence/planning/task_inbox.md) so a later pass can
retest if the landing choice turns out wrong.

  A1. Per-candidate bias vs uniform bias
      CHOSEN: per-candidate. The bias head takes concat([rule_state,
              per-candidate z_world summary]) -> scalar, producing a
              [K]-shaped bias where K = num_candidates.
      ALTERNATIVE: single scalar bias applied uniformly to all candidates
              (rule_state -> scalar via a state-only head).
      Lit-pull question: does biological lateral PFC produce
      trajectory-specific top-down bias (rule-conditional eval of each
      option), or a uniform state-dependent gain (overall bias toward
      rule-consistent vs rule-inconsistent behaviour)?

  A2. Frozen-random bias head with last layer zeroed vs trained head
      CHOSEN: frozen-random, last-Linear weights and bias zeroed at init
              so initial output is exactly zero. Head parameters are
              registered but not added to any optimizer by default.
              Phased-training for the head is deferred (P0 encoder warmup,
              P1 frozen-encoder head training, P2 eval).
      ALTERNATIVE: trained head via the standard phased-training protocol.
              Requires an identifiable training target for the bias -- the
              nearest candidates are (a) supervised rule-relevant labels
              or (b) reinforcement-style gradient from E3 action outcomes.
      Lit-pull question: the SD-033a spec signature (iv) is
      training-dependent emergence; we need the biology on how lateral PFC
      rule-bias projections are shaped by learning vs appear after
      generic cortical maturation.

  A3. Gate-modulated EMA persistence vs recurrent GRU / synaptic hold
      CHOSEN: gate-modulated EMA. rule_state ~ (1 - base_eta * gate) *
              rule_state + base_eta * gate * source. Simple scalar
              arithmetic, no recurrence, no learned persistence dynamics.
              gate is MECH-261 write_gate("sd_033a") in [0, 1]; when the
              gate is high (external_task, internal_planning) rule_state
              updates fast; when low (internal_replay) it persists.
      ALTERNATIVE 1: recurrent GRU / GRUCell with gate as an auxiliary
              input (Mansouri 2020 recurrent-activity persistence).
      ALTERNATIVE 2: synaptic-hold (short-term plasticity parameter slow
              relaxation, silent-synapse-style).
      Lit-pull question: is rule persistence in lateral PFC recurrent
      activity (spiking persistence) or synaptic-hold (STP/silent-synapse
      based)? Affects whether EMA is a biologically reasonable compression
      or a misleading simplification.

--------------------------------------------------------------------------------
INPUTS

  z_delta : [batch, delta_dim]
      SalienceCoordinator's regime/motivation latent. Biologically the rule
      signal arrives from dACC / salience network, not directly from sensory
      cortex. z_delta is the REE channel closest to this in the current
      latent stack.
  z_world : [batch, world_dim]
      Exteroceptive latent. Used both as source (pooled into the rule-update
      signal) and as per-candidate context for bias computation.
  gate : float in [0, 1]
      SalienceCoordinator.write_gate("sd_033a"). Modulates the effective
      EMA rate. Gate near 0 -> rule_state near-frozen (distractor
      resistance). Gate near 1 -> fast update.

OUTPUTS

  rule_state : [1, rule_dim] buffer
      Updated in place by update(). Reset on episode boundary via reset().
      Cross-tick persistence survives E3 ticks (consistent with MECH-262
      rule-selective persistence).
  score_bias : [K] tensor (K = num_candidates)
      Additive bias on E3 per-candidate scores. Clamped to
      [-bias_scale, +bias_scale] so it cannot dominate the E3 objective.

--------------------------------------------------------------------------------
MECH-094

rule_state is a gated latent variable, not replay content. MECH-261 generalises
the MECH-094 hypothesis_tag semantics: write_gate("sd_033a") is 0.05 under
internal_replay, so replay content cannot meaningfully update rule_state.
There is therefore no separate hypothesis_tag check here -- the gate is the
tag, via the registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class LateralPFCConfig:
    """SD-033a configuration.

    Attributes:
        use_lateral_pfc_analog: master switch (default False, backward-compatible).
        rule_dim: rule_state dimensionality. Small by intent (rule summary, not
            a full working-memory buffer).
        update_eta: base EMA rate; effective rate = update_eta * gate each tick.
            0.05 gives ~20-step rise time under gate=1.0.
        world_pool_weight: blend weight on mean(z_world) in the rule-update
            source: source = z_delta_proj + world_pool_weight * z_world_proj.
        bias_scale: clamp on |score_bias|. Keeps the bias from dominating E3.
        hidden_dim: bias head hidden layer width.
        use_discriminator_source: ARC-062 GAP-C. When True, the ARC-062
            GatedPolicy discriminator output (gating_weight scalar) is projected
            into rule_dim and added to the rule_state update source. Requires
            the gated_policy block to run before the lateral_pfc block in
            select_action() (handled by agent.py reorder). Default False =
            no-op, bit-identical backward compat.
        discriminator_pool_weight: weight of the discriminator contribution in
            the source formula. Source becomes:
              delta_proj(z_delta) + world_pool_weight * world_proj(z_world)
              + discriminator_pool_weight * discriminator_proj(disc_output)
        train_rule_bias_head: ARC-062 GAP-D. When True, the rule_bias_head's
            last Linear is NOT zeroed at init (random init preserved). Enables
            the head to be added to an experiment optimizer and trained via
            E3 score-aggregation gradient. Default False = last Linear zeroed
            (bit-identical to original landing, bias output stays 0 until
            deliberately trained).
    """

    use_lateral_pfc_analog: bool = False
    rule_dim: int = 16
    update_eta: float = 0.05
    world_pool_weight: float = 0.5
    bias_scale: float = 0.1
    hidden_dim: int = 32
    # ARC-062 GAP-C: discriminator output -> rule_state source
    use_discriminator_source: bool = False
    discriminator_pool_weight: float = 0.3
    # ARC-062 GAP-D: trainable rule_bias_head
    train_rule_bias_head: bool = False


class LateralPFCAnalog(nn.Module):
    """SD-033a lateral-PFC-analog: rule_state + per-candidate bias head.

    Stateful across ticks within an episode. reset() is called by
    REEAgent.reset() on episode boundary.
    """

    def __init__(
        self,
        delta_dim: int,
        world_dim: int,
        config: Optional[LateralPFCConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config if config is not None else LateralPFCConfig()
        self.delta_dim = delta_dim
        self.world_dim = world_dim
        rule_dim = self.config.rule_dim
        hidden_dim = self.config.hidden_dim

        # Rule-update source projections (frozen-random; not trained in landing).
        # Map raw latents into rule_dim so EMA accumulates in a uniform space.
        self.delta_proj = nn.Linear(delta_dim, rule_dim)
        self.world_proj = nn.Linear(world_dim, rule_dim)

        # ARC-062 GAP-C: discriminator_proj maps the 1-D GatedPolicy
        # gating_weight scalar into rule_dim to contribute to the rule_state
        # update source. Always constructed (cheap: Linear(1, rule_dim));
        # only used when use_discriminator_source=True.
        self.discriminator_proj = nn.Linear(1, rule_dim)

        # Per-candidate bias head:
        #   concat([rule_state, z_world_candidate_summary]) -> scalar.
        # When train_rule_bias_head=False (default / GAP-D off): last Linear
        # weights + bias ZEROED so initial bias output = 0. Bit-identical to
        # OFF until deliberately trained.
        # When train_rule_bias_head=True (GAP-D on): last Linear keeps random
        # init so gradient moves it from the first optimizer step.
        self.rule_bias_head = nn.Sequential(
            nn.Linear(rule_dim + world_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if not self.config.train_rule_bias_head:
            with torch.no_grad():
                last_linear = self.rule_bias_head[-1]
                last_linear.weight.zero_()
                last_linear.bias.zero_()

        # rule_state buffer: persistent across ticks within episode,
        # reset on episode boundary.
        self.register_buffer(
            "rule_state",
            torch.zeros(1, rule_dim),
            persistent=False,
        )

        # Diagnostics (not persisted, not part of state_dict contract)
        self._last_gate: float = 0.0
        self._last_effective_eta: float = 0.0
        self._last_bias_abs_mean: float = 0.0

    # ------------------------------------------------------------------
    # Update path
    # ------------------------------------------------------------------
    def update(
        self,
        z_delta: torch.Tensor,
        z_world: torch.Tensor,
        gate: float,
        disc_output: Optional[torch.Tensor] = None,
        override_signal: float = 0.0,
        override_eta_gain: float = 0.0,
    ) -> None:
        """Gate-modulated EMA update of rule_state.

        Args:
            z_delta: [batch, delta_dim]. Detached before use.
            z_world: [batch, world_dim]. Detached before use.
            gate: float in [0, 1]. write_gate("sd_033a") from SalienceCoordinator.
            disc_output: ARC-062 GAP-C. Optional [batch, 1] or [1, 1] tensor
                containing the GatedPolicy discriminator gating_weight. When
                use_discriminator_source=True and disc_output is not None,
                adds discriminator_pool_weight * discriminator_proj(disc_output)
                to the source vector. Default None = no-op (backward compat).
            override_signal: SD-037 BroadcastOverrideRegulator output in [0, 1].
                MECH-281 motor-coupling axis: orexin-recruited state accelerates
                rule_state learning. Default 0.0 = bit-identical OFF.
            override_eta_gain: scalar multiplier applied to eff_eta as
                (1 + override_eta_gain * override_signal). Default 0.0 = no-op.

        Effect:
            rule_state <- (1 - eff_eta) * rule_state + eff_eta * source
            where eff_eta = update_eta * clip(gate, 0, 1)
                            * (1 + override_eta_gain * override_signal)
                  source  = delta_proj(z_delta).mean(0)
                            + world_pool_weight * world_proj(z_world).mean(0)
                            [+ discriminator_pool_weight * discriminator_proj(disc_output)
                              if use_discriminator_source and disc_output is not None]
            No gradient flow (update is in-place on a buffer).
        """
        if not self.config.use_lateral_pfc_analog:
            return
        # Clip gate defensively
        g = float(max(0.0, min(1.0, gate)))
        eff_eta = self.config.update_eta * g
        # SD-037 MECH-281 motor-coupling axis: orexin-recruited state amplifies
        # the effective learning rate on rule_state. At override_eta_gain=0.0
        # (default), this is exactly 1.0 -> bit-identical to pre-MECH-281.
        if override_eta_gain != 0.0:
            ov = float(max(0.0, min(1.0, override_signal)))
            eff_eta = eff_eta * (1.0 + float(override_eta_gain) * ov)

        # Detach inputs; rule_state is a buffer, not a trainable param.
        with torch.no_grad():
            zd = z_delta.detach()
            zw = z_world.detach()
            if zd.dim() == 1:
                zd = zd.unsqueeze(0)
            if zw.dim() == 1:
                zw = zw.unsqueeze(0)
            delta_contrib = self.delta_proj(zd).mean(dim=0, keepdim=True)  # [1, rule_dim]
            world_contrib = self.world_proj(zw).mean(dim=0, keepdim=True)  # [1, rule_dim]
            source = delta_contrib + self.config.world_pool_weight * world_contrib

            # ARC-062 GAP-C: add discriminator contribution when enabled.
            if self.config.use_discriminator_source and disc_output is not None:
                do = disc_output.detach()
                if do.dim() == 1:
                    do = do.unsqueeze(0)  # [1, 1]
                disc_contrib = self.discriminator_proj(do).mean(dim=0, keepdim=True)  # [1, rule_dim]
                source = source + self.config.discriminator_pool_weight * disc_contrib

            self.rule_state.mul_(1.0 - eff_eta).add_(eff_eta * source)

        self._last_gate = g
        self._last_effective_eta = eff_eta

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------
    def compute_bias(
        self,
        candidate_world_summaries: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-candidate score_bias from rule_state + candidate summaries.

        Args:
            candidate_world_summaries: [K, world_dim] -- per-trajectory z_world
                summary. For the landing implementation the caller passes the
                first-step z_world of each candidate (trajectory world_states
                at t=0).

        Returns:
            score_bias: [K] tensor, clamped to [-bias_scale, +bias_scale].

        With the head's last layer zeroed at init, the initial output is a
        length-K vector of zeros -- no behavioural change from baseline OFF
        until the head is deliberately trained.
        """
        if not self.config.use_lateral_pfc_analog:
            return torch.zeros(
                candidate_world_summaries.shape[0],
                device=candidate_world_summaries.device,
                dtype=candidate_world_summaries.dtype,
            )

        k = candidate_world_summaries.shape[0]
        # Broadcast rule_state across K candidates
        rule_repeated = self.rule_state.expand(k, -1)  # [K, rule_dim]
        joined = torch.cat([rule_repeated, candidate_world_summaries], dim=-1)
        bias_raw = self.rule_bias_head(joined).squeeze(-1)  # [K]
        bias = bias_raw.clamp(
            min=-self.config.bias_scale,
            max=self.config.bias_scale,
        )

        with torch.no_grad():
            self._last_bias_abs_mean = float(bias.abs().mean().item())

        return bias

    # ------------------------------------------------------------------
    # GAP-D: trainable bias head parameter access
    # ------------------------------------------------------------------
    def bias_head_parameters(self):
        """Return rule_bias_head parameters for experiment optimizer inclusion.

        ARC-062 GAP-D. Experiment scripts that set train_rule_bias_head=True
        should add these to their P1 optimizer:
            optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=LR)
        Gradient flows: E3 loss -> score_bias -> compute_bias() -> these weights.
        """
        return self.rule_bias_head.parameters()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero rule_state. Called on episode boundary via REEAgent.reset()."""
        with torch.no_grad():
            self.rule_state.zero_()
        self._last_gate = 0.0
        self._last_effective_eta = 0.0
        self._last_bias_abs_mean = 0.0

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "rule_state_norm": float(self.rule_state.norm().item()),
            "last_gate": self._last_gate,
            "last_effective_eta": self._last_effective_eta,
            "last_bias_abs_mean": self._last_bias_abs_mean,
            "use_discriminator_source": self.config.use_discriminator_source,
            "train_rule_bias_head": self.config.train_rule_bias_head,
        }
