"""
E2WorldUncertaintyHead -- SD-063: conditional predictive-uncertainty head on the
E2 world-forward (distribution-free quantile / pinball form).

The current E2 world-forward is a POINT predictor trained on MSE (e2_fast.py
world_forward / e2_world.py); the only uncertainty signal reaching E3's commit
gate is the running-variance EMA (e3_selector.py current_precision), a
temporally-smoothed GLOBAL, state-BLIND estimate whose predicted uncertainty has
near-zero per-point correlation with realized error (precision_error_corr ~ 0.0
by construction). SD-063 gives the forward model a CONDITIONAL predictive
distribution so E3 can gate action commitment on where THIS prediction is
uncertain rather than on a running average -- the concrete realization of the
MECH-059 confidence channel (uncertainty-derived precision distinct from
residual error).

Preferred form (V3-EXQ-712 winner): distribution-free quantile regression with
the pinball loss. The 712 diagnostic held the encoder + transition set fixed and
swapped only the forward head + loss across four formulations on the same
held-out (z_world_t, a_t) -> z_world_{t+1} transitions (5 seeds, 4760 test
transitions):

    head              CRPS (lower=better)   precision_error_corr
    quantile_pinball  0.00486  (winner)     0.379  (winner)
    mse_point         0.00514               0.0    (the EMA null)
    mixture_gaussian  0.00682               0.038
    hetero_gaussian   0.00708               0.040

The load-bearing finding is narrow: ONLY the distribution-free quantile head
helped; imposing a Gaussian shape (hetero, mixture) on the next-state predictive
distribution did WORSE than the point baseline. The quantile head also delivered
a genuine per-point error signal (0.379) the EMA structurally cannot carry.

CRITICAL -- SD-031 agency-residual guard (design caveat):
    E2 is an agency detector via the residual
        residual_world = z_world_observed - E2WorldForward(z_world_prev, a_actual)
    (SD-031 / MECH-256). A predictive-variance head can absorb the agent-caused
    component of next-state variance into "expected spread," quietly killing the
    agency signal. THIS MODULE DEFENDS THAT STRUCTURALLY:
      - It is a SEPARATE nn.Module. It shares NO parameters with E2WorldForward
        or the z_world encoder. Training it cannot move the forward model that
        produces the agency residual.
      - Its P1 loss reads DETACHED z_world inputs and DETACHED z_world_next
        targets (the caller applies .detach()); its gradients never reach the
        encoder. This is the same stop-gradient discipline e2_world.py uses.
    The structural separation makes explaining-away impossible by construction;
    the SD-063 validation experiment must STILL confirm empirically that the
    E2WorldForward comparator residual is preserved when this head trains jointly.

Phased training (mirrors SD-031 / ARC-033):
    P0: z_world encoder warmup (SD-009 event-contrastive + SD-018 resource
        proximity) so z_world carries trained discriminative structure. An
        untrained encoder yields a near-invariant z_world -> the head fits a
        trivial spread and the comparison is vacuous (the MECH-353 / V3-EXQ-642
        lesson, and the 712 readiness gate).
    P1: the head trains on FROZEN z_world (stop-gradient: detach BOTH the
        (z_world_t, a_t) inputs AND the z_world_next target). Do NOT propagate
        head loss into the encoder or the point forward model.
    P2: evaluate held-out CRPS, precision_error_corr, and -- the SD-063
        falsifier -- that the E2WorldForward agency residual is unchanged.

MECH-094: this head is a WAKING online forward-model read used for commitment
gating. It does NOT write to memory and does NOT run under simulation / replay,
so the MECH-094 hypothesis_tag constraint DOES NOT APPLY. (Contrast the E2World
comparator's retrospective comparator-mode read, which IS MECH-094-gated.)

Biological grounding (shared with MECH-059 / SD-031): parietal / ACC-insula
outcome-prediction circuits carry a confidence estimate about predicted outcomes
distinct from the outcome prediction error itself; precision-weighting of
prediction (active inference, Friston) modulates action commitment on a
per-context basis, not via a single global gain.

See docs/architecture/sd_063_e2_conditional_uncertainty_head.md,
    ree_core/predictors/e2_world.py (the SD-031 agency comparator this must not
      disturb),
    ree_core/predictors/e3_selector.py (current_precision EMA -- the null this
      replaces at the commit gate under E3Config.use_conditional_precision_gate).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn


# Fixed quantile levels -- the V3-EXQ-712 winner config (9 levels, 0.1..0.9).
# Central interval [q0.1, q0.9] is the nominal-80% predictive interval.
QUANTILE_LEVELS: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Interquantile-range -> std conversion for the [q0.1, q0.9] span under a
# reference Gaussian: z_0.9 - z_0.1 = 2 * 1.2816 = 2.5631. Used only to turn the
# distribution-free IQR into a variance-scaled spread the E3 commit gate (which
# lives in variance space) can compare against its threshold.
IQR_TO_STD_10_90: float = 2.5631


@dataclass
class E2WorldUncertaintyConfig:
    """Configuration for the SD-063 E2 conditional predictive-uncertainty head.

    Disabled by default for full backward compatibility. Set
    use_e2_world_uncertainty=True to enable.

    z_world_dim has NO safe literal default (Optional[int]=None, REQUIRED when
    enabled) -- the same discipline as E2WorldConfig: a literal default would
    silently mis-size the head if a caller forgot to thread world_dim through.
    Unlike E2WorldConfig, there is NO world_dim>=128 hard-assert here: the SD-063
    head is a predictive-spread readout, not the SD-031 discriminative/attribution
    comparator, and the V3-EXQ-712 diagnostic that motivated it ran at world_dim=32.
    """

    # Master switch -- disabled by default (backward compat).
    use_e2_world_uncertainty: bool = False

    # Dimensions -- z_world_dim is REQUIRED (no safe literal default; see above).
    # Must match LatentStackConfig.world_dim of the agent.
    z_world_dim: Optional[int] = None
    action_dim: int = 4

    # Architecture (V3-EXQ-712 winner values).
    hidden_dim: int = 128

    # Quantile levels (fixed to the 712 winner set; overridable for ablation).
    quantile_levels: List[float] = field(default_factory=lambda: list(QUANTILE_LEVELS))

    # Training.
    learning_rate: float = 1e-3


class E2WorldUncertaintyHead(nn.Module):
    """SD-063: conditional predictive-uncertainty head over (z_world, action).

    f(z_world_t, a_t) -> per-dim quantiles of z_world_{t+1}, plus a per-input
    predictive-variance readout for the E3 commit gate.

    Input:  x = concat([z_world_t, a_onehot])  [batch, z_world_dim + action_dim]
    Output: quantiles [batch, z_world_dim, n_quantiles]

    This is a STANDALONE module (shares no parameters with E2WorldForward or the
    encoder). Callers MUST detach both the inputs and the target during training
    (P1 stop-gradient discipline) so the head cannot explain away the SD-031
    agency residual. See module docstring.

    Usage in experiment scripts:
        from ree_core.predictors.e2_world_uncertainty import (
            E2WorldUncertaintyHead, E2WorldUncertaintyConfig)
        cfg = E2WorldUncertaintyConfig(
            use_e2_world_uncertainty=True, z_world_dim=32, action_dim=4)
        head = E2WorldUncertaintyHead(cfg)
        opt = torch.optim.Adam(head.parameters(), lr=cfg.learning_rate)

        # P1 training step (detach BOTH inputs and target):
        q = head(z_world_t.detach(), action_t)
        loss = head.compute_loss(q, z_world_next.detach())
        opt.zero_grad(); loss.backward(); opt.step()

        # Commit-gate read (per-input predictive variance -> E3.select):
        pvar = head.predictive_variance(z_world_t, action_t)   # [batch]
        result = e3.select(candidates, conditional_predictive_variance=float(pvar.mean()))
    """

    def __init__(self, config: Optional[E2WorldUncertaintyConfig] = None):
        super().__init__()
        self.config = config or E2WorldUncertaintyConfig()

        z_world_dim = self.config.z_world_dim
        if z_world_dim is None:
            raise ValueError(
                "E2WorldUncertaintyConfig.z_world_dim is required (no safe "
                "default) -- pass config.latent.world_dim so the head is sized "
                "to the agent's z_world."
            )

        self.z_world_dim = int(z_world_dim)
        self.action_dim = int(self.config.action_dim)
        self.hidden_dim = int(self.config.hidden_dim)

        levels = list(self.config.quantile_levels)
        if len(levels) < 2:
            raise ValueError(
                "E2WorldUncertaintyConfig.quantile_levels needs >= 2 levels; "
                "got {}.".format(levels)
            )
        if any(not (0.0 < lv < 1.0) for lv in levels):
            raise ValueError(
                "quantile_levels must lie strictly in (0, 1); got {}.".format(levels)
            )
        if any(levels[i] >= levels[i + 1] for i in range(len(levels) - 1)):
            raise ValueError(
                "quantile_levels must be strictly increasing; got {}.".format(levels)
            )
        self.n_quantiles = len(levels)
        # Registered buffer so it moves with .to(device) and serializes with the
        # module, without being a trainable parameter.
        self.register_buffer("_levels", torch.tensor(levels, dtype=torch.float32))

        in_dim = self.z_world_dim + self.action_dim
        # Trunk matches V3-EXQ-712 _trunk (2-layer MLP, ReLU).
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.hidden_dim, self.z_world_dim * self.n_quantiles)

    # ------------------------------------------------------------------ #
    # Forward                                                            #
    # ------------------------------------------------------------------ #

    def forward(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Predict per-dim quantiles of z_world_{t+1}.

        IMPORTANT: during P1 training, pass z_world.detach() (and detach the
        target in compute_loss) so gradients never reach the z_world encoder or
        the E2WorldForward params -- this is what preserves the SD-031 agency
        residual.

        Args:
            z_world: [batch, z_world_dim] -- world-state latent (current step)
            action:  [batch, action_dim]  -- action taken (one-hot or continuous)

        Returns:
            quantiles: [batch, z_world_dim, n_quantiles]
        """
        if z_world.dim() == 1:
            z_world = z_world.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([z_world, action], dim=-1)
        q = self.out(self.trunk(x))
        return q.view(-1, self.z_world_dim, self.n_quantiles)

    # ------------------------------------------------------------------ #
    # Loss (pinball / quantile regression)                               #
    # ------------------------------------------------------------------ #

    def compute_loss(
        self,
        quantiles: torch.Tensor,
        z_world_next: torch.Tensor,
    ) -> torch.Tensor:
        """Pinball (quantile-regression) loss for P1 head training.

        The target z_world_next MUST be detached from the encoder graph before
        calling this (the caller applies .detach() -- this method does not, to
        support any calling convention). Detaching the target is load-bearing
        for the SD-031 agency-residual guard.

        pinball_tau(err) = max(tau * err, (tau - 1) * err),  err = y - q_tau

        Args:
            quantiles:    [batch, z_world_dim, n_quantiles] -- from forward()
            z_world_next: [batch, z_world_dim] -- actual next world latent
                          (should be .detach()ed by the caller)

        Returns:
            loss: scalar pinball loss (mean over batch, dims, quantiles)
        """
        if z_world_next.dim() == 1:
            z_world_next = z_world_next.unsqueeze(0)
        err = z_world_next.unsqueeze(-1) - quantiles          # [B, D, Q]
        lv = self._levels.view(1, 1, -1).to(err.device)       # [1, 1, Q]
        return torch.maximum(lv * err, (lv - 1.0) * err).mean()

    # ------------------------------------------------------------------ #
    # Predictive-spread readouts (for the E3 commit gate)                #
    # ------------------------------------------------------------------ #

    def predictive_std(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Per-input predictive standard deviation (mean over z_world dims).

        Distribution-free: derived from the [q0.1, q0.9] interquantile range,
        monotone-rearranged (torch.sort) to defend against quantile crossing,
        then scaled IQR / 2.5631 to a Gaussian-reference std so the E3 commit
        gate (variance space) has a comparable scale. Returned per batch item.

        Args:
            z_world: [batch, z_world_dim]
            action:  [batch, action_dim]

        Returns:
            std: [batch] -- per-input predictive std, meaned over z_world dims
        """
        q = self.forward(z_world, action)                     # [B, D, Q]
        qs, _ = torch.sort(q, dim=-1)                         # rearrange (no crossing)
        iqr = qs[..., -1] - qs[..., 0]                        # [B, D] (q_hi - q_lo)
        std = iqr / IQR_TO_STD_10_90
        return std.mean(dim=-1)                              # [B]

    def predictive_variance(self, z_world: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Per-input predictive variance (mean over z_world dims) -- [batch].

        This is the signal the SD-063 E3 commit gate consumes: pass its mean (a
        scalar) as E3.select(conditional_predictive_variance=...). HIGH exactly
        where THIS prediction is about to be wrong (the V3-EXQ-712 property the
        state-blind EMA structurally cannot carry). Computed under no_grad -- it
        is a read for gating, never a training path.
        """
        with torch.no_grad():
            q = self.forward(z_world, action)                # [B, D, Q]
            qs, _ = torch.sort(q, dim=-1)
            iqr = qs[..., -1] - qs[..., 0]                    # [B, D]
            std = iqr / IQR_TO_STD_10_90
            return (std ** 2).mean(dim=-1)                    # [B]
