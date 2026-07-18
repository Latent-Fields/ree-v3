"""MECH-457: first-class RPE-driven actor-critic action-learning substrate.

A dorsal-striatal-analog ACTOR (a parameterized policy pi(a | z_world) over the
primitive action space) with a ventral-striatal-analog value CRITIC, taught by a
reward-prediction-error advantage. Architecturally distinct from -- and NOT reducible
to -- the thin lateral_pfc / ofc bias_head REINFORCE readout, which only emits a
clamped +/- bias_scale per-candidate nudge into E3's cost argmin and reconstructs a
policy post-hoc. This module emits its OWN categorical action distribution and its own
state-value.

Biological grounding:
  - O'Doherty et al. 2004 -- dorsal striatum = actor, dissociable from a
    ventral-striatal value-prediction critic.
  - Schultz, Dayan & Montague 1997 -- dopaminergic reward-prediction-error teacher.
  - cand-B successor-feature critic: Dayan 1993 (SR), Barreto et al. 2017 (SF+GPI),
    biological anchor Stachenfeld, Botvinick & Gershman 2017 (hippocampus as a
    predictive map -- place fields encode the SR).
ML statement: actor-critic policy-gradient (Sutton et al. 2000); decoupled
representation/policy (Stooke et al. 2021). Engineering counsel only -- the
architecture is fixed by the biology above, not by the ML framing.

Two critic forms, selected by config (use_sf_critic) so ONE module serves all four
validation arms (A0..A3 = cotrain_encoder x use_sf_critic):
  - plain value head V(z_world)                        (cand-A arms A0/A1)
  - successor-feature critic V_SF(z) = psi(z) . w      (cand-B arms A2/A3)
    where psi = expected discounted future state-features under the policy and w is a
    learned reward-weight vector (r ~= phi . w). w is grounded in the MECH-229
    VALENCE_WANTING reward channel (see agent.actor_critic_reward).

A third, ORTHOGONAL critic form is selected by use_distributional_critic (MECH-457
H-retention-critic, 2026-07-18): the plain value head becomes a categorical head over a
fixed symlog bin support, trained by cross-entropy and decoded to a scalar by
expectation (see distributional_value.ValueBins). It is a swap of the VALUE ESTIMATOR
only -- the actor, the log-prob and the advantage weighting are untouched, which is what
keeps it separable from the update-constraint locus (mech457_policy_kl_anchor).

CO-SHAPING (MECH-457 core, per V3-EXQ-737): whether the actor's gradient reaches the
z_world encoder is decided by the CALLER, not this module. The frozen arm feeds
z_world.detach() (= 737's refuted instantiation); the co-trained arm feeds live
z_world AND includes the encoder params in the optimizer. This module is
encoder-agnostic; the ablation lives in how z is supplied (see
REEAgent.actor_critic_step + actor_critic_encoder_parameters).

The training UPDATE (PPO / GAE / SF-TD losses) lives in the experiment, following the
established REE pattern where the bias_head REINFORCE update also lives in the
experiment. The MODULE, its parameters, its forward, the co-shaping gradient path, and
the RPE-teacher hookup live in the substrate.

MECH-094: not applicable -- this module performs no memory writes on simulated /
non-waking ticks; it reads z_world and emits an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .distributional_value import ValueBins


@dataclass
class ActorCriticStep:
    """One graph-connected actor-critic decision. The training hook (analogous to
    E3's SelectionResult.log_prob for the bias heads)."""

    action: torch.Tensor           # [batch] long -- chosen primitive action index
    log_prob: torch.Tensor         # [batch] -- log pi(action | z)
    value: torch.Tensor            # [batch] -- V(z) (plain) or psi(z).w (SF critic)
    entropy: torch.Tensor          # [batch] -- policy entropy (exploration bonus)
    logits: torch.Tensor           # [batch, action_dim]
    phi: Optional[torch.Tensor] = None    # [batch, sf_feature_dim] state features (SF only)
    psi: Optional[torch.Tensor] = None    # [batch, sf_feature_dim] successor features (SF only)
    # [batch, n_value_bins] critic logits (distributional critic only; None otherwise).
    # `value` above is their expectation-decode, so scalar consumers are unaffected.
    value_logits: Optional[torch.Tensor] = None


class ActorCriticPolicy(nn.Module):
    """Dorsal-striatal actor + value-baseline critic (plain or successor-feature).

    Args:
        world_dim: dimensionality of z_world (the input representation).
        action_dim: number of primitive actions the actor chooses among.
        hidden_dim: shared trunk / psi-head width. 128 matches the validated
            V3-EXQ-734/737 PPO trunk so the frozen arm reproduces 737.
        use_sf_critic: False -> plain learned value head V(z) (cand-A arms A0/A1).
            True -> successor-feature critic V_SF = psi(z).w (cand-B arms A2/A3).
        sf_feature_dim: dimensionality of the state-feature vector phi / successor
            feature vector psi (SF critic only).
    """

    def __init__(
        self,
        world_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        use_sf_critic: bool = False,
        sf_feature_dim: int = 32,
        use_distributional_critic: bool = False,
        n_value_bins: int = 41,
        value_bin_limit: float = 10.0,
        value_bin_sigma: float = 0.75,
    ) -> None:
        super().__init__()
        self.world_dim = int(world_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.use_sf_critic = bool(use_sf_critic)
        self.sf_feature_dim = int(sf_feature_dim)

        # Shared trunk feeding the actor (matches PPOPolicyNet shape: Tanh MLP).
        self.trunk = nn.Sequential(
            nn.Linear(self.world_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        # The ACTOR (dorsal striatum): its OWN categorical distribution over actions.
        self.policy_head = nn.Linear(self.hidden_dim, self.action_dim)

        # cand-A plain CRITIC (ventral striatum): a learned state-value V(z_world),
        # distinct from the scalar-EMA _lcg_value_baseline (ARC-108).
        #
        # MECH-457 H-retention-critic (2026-07-18): the critic may instead be
        # DISTRIBUTIONAL -- a [hidden -> n_value_bins] categorical head over a fixed
        # symlog bin support, trained by cross-entropy against a two-hot / HL-Gauss
        # projection of the return, and decoded to a scalar by expectation. The scalar
        # `value` contract is unchanged either way, so GAE / bootstrap / credit-replay /
        # eval consume the same quantity. Default False -> the exact scalar head above.
        # ANTI-ALIAS: this is a VALUE-ESTIMATOR swap only; nothing here constrains the
        # policy update (that locus is mech457_policy_kl_anchor).
        self.use_distributional_critic = bool(use_distributional_critic)
        if self.use_distributional_critic:
            self.value_bins = ValueBins(
                n_bins=int(n_value_bins),
                limit=float(value_bin_limit),
                sigma_ratio=float(value_bin_sigma),
            )
            self.value_head = nn.Linear(self.hidden_dim, int(n_value_bins))
        else:
            self.value_bins = None
            self.value_head = nn.Linear(self.hidden_dim, 1)

        # cand-B successor-feature CRITIC. phi(z) = state features; psi(z) = expected
        # discounted future features under the policy; V_SF = psi . w with w the
        # learned reward-weight vector (r ~= phi . w). Biological anchor: hippocampal
        # predictive map (Stachenfeld 2017). Only constructed when requested.
        if self.use_sf_critic:
            self.phi_head = nn.Linear(self.world_dim, self.sf_feature_dim)
            self.psi_head = nn.Sequential(
                nn.Linear(self.world_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.sf_feature_dim),
            )
            # Reward weights w (r ~= phi . w). Zero-init: no reward structure until the
            # reward-regression loss grounds it (interpretable, stable baseline start).
            self.reward_w = nn.Parameter(torch.zeros(self.sf_feature_dim))
        else:
            self.phi_head = None
            self.psi_head = None
            self.reward_w = None

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """z: [batch, world_dim] -> (logits, value, phi, psi).

        phi / psi are None unless use_sf_critic. value is V(z) (plain) or psi(z).w (SF).
        Kept 4-tuple for backward compatibility -- the distributional critic's bin logits
        are reached via forward_value_logits() or ActorCriticStep.value_logits.
        """
        logits, value, phi, psi, _vlogits = self._forward_full(z)
        return logits, value, phi, psi

    def _forward_full(self, z: torch.Tensor):
        """(logits, value, phi, psi, value_logits). value_logits is None unless the
        distributional critic is enabled."""
        if z.dim() == 1:
            z = z.unsqueeze(0)
        h = self.trunk(z)
        logits = self.policy_head(h)
        if self.use_sf_critic:
            phi = self.phi_head(z)
            psi = self.psi_head(z)
            value = (psi * self.reward_w).sum(dim=-1)
            return logits, value, phi, psi, None
        if self.use_distributional_critic:
            value_logits = self.value_head(h)
            # Expectation-decode: downstream still receives a scalar V(z).
            return logits, self.value_bins.decode(value_logits), None, None, value_logits
        value = self.value_head(h).squeeze(-1)
        return logits, value, None, None, None

    def forward_value_logits(self, z: torch.Tensor) -> Optional[torch.Tensor]:
        """[batch, n_value_bins] critic logits, or None on the scalar / SF critic."""
        return self._forward_full(z)[4]

    def critic_loss(
        self, value_logits: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy of the predicted value distribution against the projected
        return -- the distributional replacement for 0.5 * (V - G)^2. Raises if the
        distributional critic is not enabled, so a mis-wired ON arm fails loudly rather
        than silently training nothing (the degenerate-arm trap)."""
        if not self.use_distributional_critic:
            raise RuntimeError(
                "critic_loss requires use_distributional_critic=True "
                "(the scalar critic uses an MSE term at the call site)."
            )
        return self.value_bins.cross_entropy(value_logits, returns)

    def select(self, z: torch.Tensor, deterministic: bool = False) -> ActorCriticStep:
        """Sample (or, if deterministic, argmax) an action and return the full
        graph-connected step. Deterministic is used at eval (P2)."""
        logits, value, phi, psi, value_logits = self._forward_full(z)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        return ActorCriticStep(
            action=action,
            log_prob=dist.log_prob(action),
            value=value,
            entropy=dist.entropy(),
            logits=logits,
            phi=phi,
            psi=psi,
            value_logits=value_logits,
        )

    def sf_reward_prediction(self, phi: torch.Tensor) -> torch.Tensor:
        """r_hat = phi . w -- the SF reward regression prediction. The experiment
        regresses this toward the substrate reward R_t (benefit_eval - harm_eval) to
        ground the reward weights w. Raises if the SF critic is not enabled."""
        if not self.use_sf_critic:
            raise RuntimeError(
                "sf_reward_prediction requires use_sf_critic=True (cand-B critic)."
            )
        return (phi * self.reward_w).sum(dim=-1)

    def actor_parameters(self):
        """Trunk + actor head only (the policy object)."""
        params = list(self.trunk.parameters()) + list(self.policy_head.parameters())
        return iter(params)
