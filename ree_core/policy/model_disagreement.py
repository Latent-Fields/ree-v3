"""MECH-441: model_disagreement_directed_curiosity (RND / Plan2Explore analog).

The complementary directed-curiosity leg of ARC-065 substrate (b), sibling to
MECH-440's undirected tonic-noise floor (a). A per-candidate, self-annealing
intrinsic signal from E2 forward-model DISAGREEMENT: a SMALL K-head ensemble of
independent residual-delta predictors over (z_world, action). Where the world is
well-modelled the heads agree (low variance); where it is poorly sampled the
heads disagree (high variance = epistemic uncertainty). The per-candidate
cross-head variance is fed into E3 selection as a propagating per-candidate
curiosity BONUS (E3Config.use_model_disagreement_curiosity) -- unlike the
broadcast-novelty EMA channel found NON-PROPAGATING (V3-EXQ-590a / 141b), the
per-candidate disagreement can change the committed argmin.

SELF-ANNEALING is INTRINSIC (Plan2Explore; Sekar et al. 2020): as the ensemble
trains on visited transitions the cross-head variance collapses toward zero, so
the curiosity bonus vanishes exactly where experience accumulates -- no hand-set
decay schedule. disagreement_bootstrap_mask_prob > 0 gives each head a per-step
Bernoulli data mask so the heads diverge in poorly-sampled regions and converge
where data is dense (the RND/bootstrapped-ensemble epistemic-uncertainty proxy;
Osband et al. 2016).

Biology. Daw et al. 2006 (frontopolar cortex preferentially active during
exploratory decisions -- a dedicated directed-exploration substrate; substrate
existence only, the weakest of the cluster's three anchors). The
model-disagreement COMPUTATION is the engineering import (ARC-106: an ML
technique whose REE use is grounded in the directed-exploration substrate, not
imported as architecture).

PHASED TRAINING (mirrors SD-031 E2WorldForward P0/P1/P2):
  P0: z_world encoder warmup (SD-009 + SD-018) so z_world is discriminative.
  P1: train each head on the FROZEN z_world target (target = z_world_next.detach()
      -- stop-gradient on z_world inputs, do NOT propagate into the encoder).
  P2: read disagreement_per_candidate() on the waking select_action path (no_grad)
      and feed it into E3 selection.

MECH-094. disagreement_per_candidate() is a waking-only no_grad read with no
replay / memory-write surface, so it is not MECH-094-implicated; callers must
still gate it off on simulation ticks (the E3 consumer passes simulation_mode).

NO-OP DEFAULT. n_disagreement_heads <= 1 -> the ensemble is NOT built at the
agent level -> bit-identical OFF. Falsifier HELD (blocked_substrate) gated on
ARC-110 validation V3-EXQ-707: failure_autopsy_704b-706b-conversion-ceiling_
2026-06-27 found the single-arena collapse (not the curiosity channel) is the
binding constraint, so a MECH-441 run before ARC-110 is validated would
re-derive the arena ceiling (a vacuous FAIL).

See docs/architecture/state_conditioned_exploration_noise_floor.md (#mech-441)
and REE_assembly/docs/claims/claims.yaml (MECH-441).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from ree_core.latent.stack import ResidualHarmForward


@dataclass
class ModelDisagreementConfig:
    """MECH-441 disagreement-ensemble configuration.

    Attributes:
        world_dim : z_world dimensionality (REQUIRED; matches LatentStackConfig.world_dim).
        action_dim : action vector width.
        n_heads : ensemble size. <= 1 disables (the agent does not build it).
        hidden_dim : per-head transition-net hidden width.
        bootstrap_mask_prob : per-(head, sample) Bernoulli DROP probability during
            training (0.0 = every head sees every sample; > 0 diversifies heads).
        learning_rate : Adam LR for the internal optimizer.
    """

    world_dim: int = 32
    action_dim: int = 4
    n_heads: int = 0
    hidden_dim: int = 128
    bootstrap_mask_prob: float = 0.0
    learning_rate: float = 3e-4


class ModelDisagreementEnsemble(nn.Module):
    """MECH-441 K-head forward-model-disagreement ensemble over (z_world, action).

    Each head is a ResidualHarmForward (the generic residual-delta predictor,
    reused per ARC-106 reuse-the-mechanism) with a distinct random init. Provides
    a no_grad per-candidate disagreement read for E3 selection and a train_step
    for the phased-training driver.

    Diagnostics (get_state()):
        _last_disagreement_range : float (cross-candidate range of the read)
        _last_disagreement_mean  : float
        _n_train_steps           : int
    """

    def __init__(self, config: "ModelDisagreementConfig | None" = None) -> None:
        super().__init__()
        self.config = config if config is not None else ModelDisagreementConfig()
        n_heads = int(self.config.n_heads)
        if n_heads < 2:
            raise ValueError(
                "ModelDisagreementEnsemble requires n_heads >= 2 (disagreement is "
                f"cross-head variance); got {n_heads}. n_heads <= 1 -> do not build it."
            )
        world_dim = int(self.config.world_dim)
        action_dim = int(self.config.action_dim)
        hidden_dim = int(self.config.hidden_dim)
        # Distinct random inits give the heads initial disagreement; identical
        # architecture keeps the variance a clean epistemic signal.
        self.heads = nn.ModuleList([
            ResidualHarmForward(
                z_harm_dim=world_dim, action_dim=action_dim, hidden_dim=hidden_dim
            )
            for _ in range(n_heads)
        ])
        self.n_heads = n_heads
        self.world_dim = world_dim
        self.action_dim = action_dim
        self._optimizer = torch.optim.Adam(
            self.parameters(), lr=float(self.config.learning_rate)
        )
        self._last_disagreement_range: float = 0.0
        self._last_disagreement_mean: float = 0.0
        self._n_train_steps: int = 0

    # ------------------------------------------------------------------
    # Waking read (no_grad)
    # ------------------------------------------------------------------
    def disagreement_per_candidate(
        self,
        z0: torch.Tensor,
        actions: torch.Tensor,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """Per-candidate cross-head disagreement [K] for candidate first actions.

        Args:
            z0 : current observed z_world, [world_dim] or [1, world_dim].
            actions : per-candidate first-action vectors, [K, action_dim].
            simulation_mode : MECH-094 gate -- returns a zero vector.

        Returns:
            [K] tensor of per-candidate cross-head prediction variance, mean-
            pooled over world_dim. Returns zeros under simulation_mode.
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        k = int(actions.shape[0])
        device = actions.device
        if simulation_mode or k == 0:
            return torch.zeros(k, device=device)
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)
        z0 = z0[:1].to(device=device)
        z0_K = z0.expand(k, -1)                                  # [K, world_dim]
        actions_K = actions.to(device=device, dtype=z0.dtype)    # [K, action_dim]
        with torch.no_grad():
            preds = torch.stack(
                [head(z0_K, actions_K) for head in self.heads], dim=0
            )  # [n_heads, K, world_dim]
            # Cross-head variance per candidate, mean over world_dim.
            disagreement = preds.var(dim=0, unbiased=False).mean(dim=-1)  # [K]
        if k >= 2:
            self._last_disagreement_range = float(
                (disagreement.max() - disagreement.min()).item()
            )
        else:
            self._last_disagreement_range = 0.0
        self._last_disagreement_mean = float(disagreement.mean().item())
        return disagreement.detach()

    # ------------------------------------------------------------------
    # Phased training (P1)
    # ------------------------------------------------------------------
    def train_step(
        self,
        z_world_prev: torch.Tensor,
        action: torch.Tensor,
        z_world_next: torch.Tensor,
    ) -> float:
        """One P1 update of all heads on the FROZEN z_world target.

        Args:
            z_world_prev : [B, world_dim] previous observed z_world.
            action : [B, action_dim] action taken.
            z_world_next : [B, world_dim] observed next z_world (detached as the
                stop-gradient target).

        Returns:
            scalar mean training loss across heads (float).
        """
        z_prev = z_world_prev.detach()
        a = action.detach()
        target = z_world_next.detach()
        b = int(z_prev.shape[0])
        self._optimizer.zero_grad()
        total = z_prev.new_zeros(())
        mask_p = float(self.config.bootstrap_mask_prob)
        for head in self.heads:
            pred = head(z_prev, a)                          # [B, world_dim]
            se = (pred - target).pow(2).mean(dim=-1)        # [B]
            if mask_p > 0.0 and b >= 1:
                keep = (torch.rand(b, device=se.device) >= mask_p).to(se.dtype)
                denom = keep.sum().clamp(min=1.0)
                head_loss = (se * keep).sum() / denom
            else:
                head_loss = se.mean()
            total = total + head_loss
        loss = total / float(self.n_heads)
        loss.backward()
        self._optimizer.step()
        self._n_train_steps += 1
        return float(loss.item())

    def get_state(self) -> dict:
        """Diagnostic snapshot for experiment manifests."""
        return {
            "model_disagreement_last_range": self._last_disagreement_range,
            "model_disagreement_last_mean": self._last_disagreement_mean,
            "model_disagreement_n_heads": self.n_heads,
            "model_disagreement_n_train_steps": self._n_train_steps,
        }
