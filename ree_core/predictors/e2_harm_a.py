"""
E2HarmAForward -- MECH-258: Affective-Pain Forward Model (prerequisite for SD-032b).

Dedicated module for the E2_harm_a forward model -- the affective-stream analog
of ARC-033's E2_harm_s sensory-discriminative forward model. Produces a
precision-weighted affective-pain prediction error that dACC (SD-032b) uses as
its primary learning and control-demand signal (MECH-258).

Architecture -- two paths behind `use_shared_harm_trunk`:

ARC-033 path (shared_trunk=None, independent module -- default):
    f(z_harm_a_t, a_t) -> z_harm_a_{t+1}  via own ResidualHarmForward.
    Mirrors E2HarmSForward structurally; the two streams have independent
    forward models. Biological read: dorsal-posterior insula (sensory PE) and
    anterior insula / pgACC (affective PE) as separate learned substrates.

ARC-058 path (shared_trunk=HarmForwardTrunk, competing claim):
    Shared HarmForwardTrunk (unsigned, modality-independent PE substrate)
    + stream-specific HarmForwardHead (signed, per-modality readout).
    Biological read: Horing & Buchel 2022 -- anterior insula encodes a
    modality-independent unsigned PE shared across aversive modalities;
    dorsal posterior insula encodes modality-specific signed PE. Trunk ~
    unsigned substrate, head ~ signed readout.

ARC-033 vs ARC-058 are registered as competing architectural claims. The same
code path switches between them via the constructor's `shared_trunk` argument;
validation experiments that ablate this flag arbitrate between the two.

Phased training required:
    P0: AffectiveHarmEncoder warmup (z_harm_a encoder trains on accumulated-harm
        / harm-surprise supervision per SD-020).
    P1: E2HarmAForward trains on FROZEN z_harm_a (stop-gradient on encoder
        inputs). Target must be `.detach()`ed by caller.
    P2: Evaluation.

Stop-gradient discipline:
    The caller passes `z_harm_a.detach()` as the training target. Do NOT
    propagate forward-model loss gradients back into AffectiveHarmEncoder --
    that causes encoder drift toward trivially predictable representations.

Biological grounding:
    Chen 2023, Hoskin 2023, Geuter 2017, Horing & Buchel 2022: anterior insula
    encodes an unsigned aversive PE that persists across pain, loud noise, and
    other modalities; per-modality signed PE lives in dorsal posterior insula.
    Rainville 1997 and Keltner 2006: affective pain (ACC) does not show the
    predictive cancellation seen in sensory pain (S1/S2) -- but this refers to
    SUBJECTIVE-REPORT cancellation, not to the absence of a forward model at
    the PE-substrate level. The forward model exists and is used for
    control-demand computation; what differs is the downstream consumer.

MECH-094: not applicable (waking observation stream, not replay content).

See CLAUDE.md: SD-011, SD-032b, ARC-033, ARC-058, MECH-258.
See ree_core/latent/stack.py: HarmForwardTrunk, HarmForwardHead, ResidualHarmForward.
See evidence/literature/targeted_review_pain_predictive_coding_substrate/synthesis.md.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.latent.stack import (
    HarmForwardHead,
    HarmForwardTrunk,
    ResidualHarmForward,
)


@dataclass
class E2HarmAConfig:
    """Configuration for the E2_harm_a affective-pain forward model (MECH-258).

    Disabled by default for full backward compatibility with all existing experiments.
    Set use_e2_harm_a=True at the REEConfig level to enable (this dataclass is
    instantiated by REEAgent when that flag is on).

    Dimensionality note: z_harm_a_dim defaults to 16 to match AffectiveHarmEncoder
    output in the ree-v3 default config, which differs from z_harm_s_dim=32. When
    running the shared-trunk path (ARC-058), the caller must ensure the trunk was
    constructed with a z_harm_dim matching whichever stream will feed it, or must
    project dims to a common size before the trunk call. See experiment scripts
    for the reconciliation pattern.

    Phased training notes:
        P0: train AffectiveHarmEncoder (SD-011 second source, SD-020 surprise target)
        P1: train E2HarmAForward on frozen z_harm_a targets (stop-gradient)
        P2: evaluate harm_a_forward_r2 and precision-weighted affective PE
    """
    # Dimensions -- must match AffectiveHarmEncoder outputs
    z_harm_a_dim: int = 16   # z_harm_a dimensionality (matches LatentStackConfig.z_harm_a_dim)
    action_dim: int = 4      # action space dimensionality

    # Architecture -- matches HarmForwardTrunk/Head defaults so shared-trunk is compatible
    hidden_dim: int = 128    # hidden dimension for transition network
    action_enc_dim: int = 16 # action embedding dimensionality before concat

    # Training
    learning_rate: float = 5e-4   # small LR, consistent with E2HarmSConfig


class E2HarmAForward(nn.Module):
    """MECH-258: E2_harm_a affective-pain forward model.

    f(z_harm_a_t, a_t) -> z_harm_a_{t+1}

    Two architectural paths (selected at construction time):

    (1) ARC-033 path -- `shared_trunk=None`:
        Owns an independent ResidualHarmForward. Parallel to E2HarmSForward:
        the two streams have independent forward-model substrates.

    (2) ARC-058 path -- `shared_trunk=<HarmForwardTrunk instance>`:
        Re-uses a caller-provided trunk; owns only a stream-specific head.
        The trunk is typically shared with E2HarmSForward (same nn.Module
        instance) so both streams update the same hidden substrate.

    compute_loss() helper is identical in both paths.

    Usage in experiment scripts (independent path, ARC-033):
        from ree_core.predictors.e2_harm_a import E2HarmAForward, E2HarmAConfig
        cfg = E2HarmAConfig(z_harm_a_dim=16, action_dim=4)
        harm_a_fwd = E2HarmAForward(cfg)
        opt = torch.optim.Adam(harm_a_fwd.parameters(), lr=cfg.learning_rate)

    Usage in experiment scripts (shared-trunk path, ARC-058):
        from ree_core.latent.stack import HarmForwardTrunk
        shared_trunk = HarmForwardTrunk(z_harm_dim=16, action_dim=4, hidden_dim=128)
        harm_s_fwd = E2HarmSForward(..., shared_trunk=shared_trunk)  # (future refactor)
        harm_a_fwd = E2HarmAForward(cfg, shared_trunk=shared_trunk)
        # Single optimizer over shared_trunk.parameters() + both heads' parameters.
    """

    def __init__(
        self,
        config: Optional[E2HarmAConfig] = None,
        shared_trunk: Optional[HarmForwardTrunk] = None,
    ):
        super().__init__()
        self.config = config or E2HarmAConfig()
        self._shared_mode = shared_trunk is not None

        if self._shared_mode:
            self._trunk = shared_trunk
            # Own the head only; trunk is external.
            self._head = HarmForwardHead(
                hidden_dim=self.config.hidden_dim,
                z_harm_dim=self.config.z_harm_a_dim,
            )
            self._residual_fwd = None
        else:
            # ARC-033 independent path: own ResidualHarmForward.
            self._trunk = None
            self._head = None
            self._residual_fwd = ResidualHarmForward(
                z_harm_dim=self.config.z_harm_a_dim,
                action_dim=self.config.action_dim,
                hidden_dim=self.config.hidden_dim,
            )

    def forward(
        self,
        z_harm_a: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict z_harm_a at next step.

        IMPORTANT: during P1 training, supply `.detach()`ed z_harm_a targets
        -- do NOT let forward-model loss gradients flow into the
        AffectiveHarmEncoder.

        Args:
            z_harm_a: [batch, z_harm_a_dim] -- affective-pain latent (current step)
            action:   [batch, action_dim]    -- action taken (one-hot or continuous)

        Returns:
            z_harm_a_pred: [batch, z_harm_a_dim] -- predicted affective-pain latent next step
        """
        if self._shared_mode:
            hidden = self._trunk(z_harm_a, action)
            return self._head(hidden, z_harm_a)
        return self._residual_fwd(z_harm_a, action)

    def compute_loss(
        self,
        z_harm_a_pred: torch.Tensor,
        z_harm_a_next: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss for P1 forward-model training.

        The target z_harm_a_next MUST be detached from the encoder computation
        graph before calling this method -- this method does NOT apply
        stop-gradient internally (symmetry with E2HarmSForward.compute_loss).

        Args:
            z_harm_a_pred: [batch, z_harm_a_dim] -- predicted next affective-pain latent
            z_harm_a_next: [batch, z_harm_a_dim] -- actual next affective-pain latent (detached)

        Returns:
            loss: scalar MSE loss
        """
        return F.mse_loss(z_harm_a_pred, z_harm_a_next)
