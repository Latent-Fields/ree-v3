"""
E2HarmSForward -- ARC-033: Sensory-Discriminative Harm Forward Model

Dedicated module for the E2_harm_s forward model (ARC-033 / SD-003 redesign).

Architecture:
    f(z_harm_s_t, a_t) -> z_harm_s_{t+1}

Implemented as a residual delta predictor (ResidualHarmForward in stack.py is
the underlying module). Predicting the delta avoids identity collapse on
autocorrelated z_harm_s signals (r~0.9). EXQ-166e PASS confirmed the
harm-delta architecture. EXQ-195 confirmed harm_forward_r2=0.914.

SD-003 redesigned pipeline (post-SD-011):
    z_harm_s_cf = E2HarmSForward(z_harm_s_t, a_cf)
    causal_sig  = E3.harm_eval_z_harm(z_harm_s_actual) - E3.harm_eval_z_harm(z_harm_s_cf)

Phased training required:
    P0: HarmEncoder warmup (z_harm_s encoder trains on proximity supervision)
    P1: E2HarmSForward trains on FROZEN z_harm_s (stop-gradient on encoder inputs)
    P2: Evaluation

Stop-gradient discipline:
    Use z_harm_s.detach() as target during P1 training. Do NOT propagate gradients
    from the forward model loss back into HarmEncoder -- this causes encoder drift
    toward representations that are trivially predictable.

Biological grounding:
    Keltner et al. (2006, J Neurosci): predictability suppresses sensory-discriminative
    pain (S1/S2). The brain maintains a forward model of expected nociceptive input
    for movement-induced reafference cancellation. This is the REE implementation.
    NOT applicable to z_harm_a (affective/C-fiber stream, which lacks predictive
    cancellation per Rainville et al. 1997).

MECH-094: not applicable (waking observation stream, not replay content).

See CLAUDE.md: SD-011, ARC-033. See ree_core/latent/stack.py: ResidualHarmForward.
See docs/architecture/arc_033_e2_harm_s_forward_model.md.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.latent.stack import ResidualHarmForward


@dataclass
class E2HarmSConfig:
    """Configuration for the E2_harm_s sensory-discriminative harm forward model (ARC-033).

    Disabled by default for full backward compatibility with all existing experiments.
    Set use_e2_harm_s_forward=True to enable.

    Phased training notes:
        P0: train HarmEncoder on proximity supervision (harm_obs -> z_harm_s MSE loss)
        P1: train E2HarmSForward on frozen z_harm_s targets (stop-gradient on z_harm_s)
        P2: evaluate forward_r2 and harm_s_cf_gap
    """
    # Master switch -- disabled by default (backward compat)
    use_e2_harm_s_forward: bool = False

    # Dimensions -- must match HarmEncoder outputs
    z_harm_dim: int = 32     # z_harm_s dimensionality (matches LatentStackConfig.z_harm_dim)
    action_dim: int = 4      # action space dimensionality

    # Architecture
    hidden_dim: int = 128    # hidden dimension for transition network (matches ResidualHarmForward default)
    action_enc_dim: int = 16  # action embedding dimensionality before concat

    # Training
    learning_rate: float = 5e-4   # small LR: harm stream is low-variance vs z_world
    # Note: apply stop-gradient to z_harm_s inputs during P1 training:
    #     target = z_harm_s_next.detach()  # <-- critical for encoder stability


class E2HarmSForward(nn.Module):
    """
    ARC-033: E2_harm_s sensory-discriminative harm forward model.

    f(z_harm_s_t, a_t) -> z_harm_s_{t+1}

    Wrapper around ResidualHarmForward (ree_core/latent/stack.py) that provides:
      - E2HarmSConfig-based instantiation
      - compute_loss() helper for phased training
      - counterfactual_forward() for SD-003 pipeline

    Residual delta architecture prevents identity collapse:
        delta = transition_net(cat(z_harm_s, action_enc))
        z_harm_s_next = z_harm_s + delta

    Usage in experiment scripts:
        from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig
        cfg = E2HarmSConfig(use_e2_harm_s_forward=True, z_harm_dim=32, action_dim=4)
        harm_fwd = E2HarmSForward(cfg)
        opt = torch.optim.Adam(harm_fwd.parameters(), lr=cfg.learning_rate)

        # P1 training step:
        z_pred = harm_fwd(z_harm_s_t, action_t)  # forward
        loss = harm_fwd.compute_loss(z_pred, z_harm_s_next.detach())  # stop-grad target
        opt.zero_grad(); loss.backward(); opt.step()

        # SD-003 counterfactual:
        z_harm_s_cf = harm_fwd.counterfactual_forward(z_harm_s_t, a_cf)
        causal_sig = e3.harm_eval_z_harm_head(z_harm_s_actual) - e3.harm_eval_z_harm_head(z_harm_s_cf)
    """

    def __init__(self, config: Optional[E2HarmSConfig] = None):
        super().__init__()
        self.config = config or E2HarmSConfig()
        # Delegate to ResidualHarmForward (canonical ARC-033 implementation)
        self._residual_fwd = ResidualHarmForward(
            z_harm_dim=self.config.z_harm_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        )

    def forward(
        self,
        z_harm_s: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict z_harm_s at next step via residual delta.

        IMPORTANT: do NOT propagate gradients from this output back into
        the HarmEncoder during P1 training. Use .detach() on z_harm_s_next
        as the training target.

        Args:
            z_harm_s: [batch, z_harm_dim] -- sensory-discriminative harm latent (current step)
            action:   [batch, action_dim] -- action taken (one-hot or continuous)

        Returns:
            z_harm_s_pred: [batch, z_harm_dim] -- predicted harm latent at next step
        """
        return self._residual_fwd(z_harm_s, action)

    def compute_loss(
        self,
        z_harm_s_pred: torch.Tensor,
        z_harm_s_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        MSE loss for P1 forward model training.

        The target z_harm_s_next MUST be detached from the encoder computation
        graph before calling this method. The caller is responsible for applying
        .detach() -- this method does NOT apply stop-gradient internally, to
        support any calling convention.

        Args:
            z_harm_s_pred: [batch, z_harm_dim] -- predicted next harm latent (from forward())
            z_harm_s_next: [batch, z_harm_dim] -- actual next harm latent (should be .detach()ed)

        Returns:
            loss: scalar MSE loss
        """
        return F.mse_loss(z_harm_s_pred, z_harm_s_next)

    def counterfactual_forward(
        self,
        z_harm_s: torch.Tensor,
        counterfactual_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        SD-003 counterfactual query: predicted z_harm_s under a_cf.

        Equivalent to forward(z_harm_s, counterfactual_action) but named explicitly
        for clarity in the SD-003 attribution pipeline:

            z_harm_s_cf  = harm_fwd.counterfactual_forward(z_harm_s_t, a_cf)
            z_harm_s_act = harm_fwd.forward(z_harm_s_t, a_actual)
            causal_sig   = e3.harm_eval_z_harm_head(z_harm_s_act) - e3.harm_eval_z_harm_head(z_harm_s_cf)

        Args:
            z_harm_s:              [batch, z_harm_dim] -- current sensory harm latent
            counterfactual_action: [batch, action_dim] -- the action not taken

        Returns:
            z_harm_s_cf: [batch, z_harm_dim] -- counterfactual predicted harm latent
        """
        return self._residual_fwd(z_harm_s, counterfactual_action)
