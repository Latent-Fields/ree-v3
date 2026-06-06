"""
E2WorldForward -- SD-031: z_world (causal-footprint) single-pass comparator.

Dedicated module for the E2_world forward model and the MECH-256 single-pass
agency comparator on the world-state reafferent stream (z_world). This is the
z_world instantiation of the general comparator mechanism (MECH-256); the
z_harm_s instantiation is ARC-033 / SD-029 (e2_harm_s.py), the z_self
instantiation is SD-030 (V4-deferred, not built).

Mechanism (single forward pass -- NO counterfactual second call):

    predicted_z_world = E2WorldForward(z_world_{t-1}, a_actual)
    residual_world    = z_world_observed - predicted_z_world      # agency signal

A self-caused world change is one the agent's own action predicted (small
residual); an externally-caused world change is one the action did not predict
(large residual). Residual magnitude IS the agency signal on this stream
(MECH-256), so there is no two-pass counterfactual -- the forward prediction
on the actually-taken action is the reference.

Architecture: residual-delta predictor (delegates to ResidualHarmForward, the
generic residual-delta module in stack.py -- its docstring notes it "mirrors
E2FastPredictor.world_forward()"; it is not harm-specific). Predicting the
delta avoids identity collapse on autocorrelated z_world signals, the same
lesson e2_harm_s learned on z_harm_s (r~0.9).

CRITICAL -- low-dimensionality carry-forward guard:
    The 2026-06-06 cluster failure-autopsy
    (failure_autopsy_zworld-integration-cluster_2026-06-06, V3-EXQ-177/145/170/
    215) established that z_world at world_dim=32 is a competent BULK dynamical
    predictor (world_forward_r2 0.72-0.94) but LACKS the discriminative /
    spatial granularity downstream attribution claims need (event-selectivity,
    counterfactual attribution, fine RBF residue all collapse to ~0). The
    E2_world comparator's discriminative/attribution arm therefore REQUIRES
    world_dim >= 128 (the REEConfig.large preset). E2WorldForward.__init__
    HARD-ASSERTS this by default so a dim=32 comparator cannot be silently
    constructed and produce a misleadingly-zero attribution gap. An explicit
    allow_subthreshold_dim escape hatch (default False) permits bulk-only /
    ablation construction below threshold but reports attribution_ready=False
    and returns a zeroed comparator residual rather than a vacuous gap.

Phased training (validation experiment, per SD-031 design doc):
    P0: z_world encoder warmup (SD-009 event-contrastive + SD-018 resource
        proximity) so z_world carries trained discriminative structure. A
        random/untrained encoder yields a near-position-invariant z_world ->
        world_forward fits identity trivially -> the comparator floors to a
        vacuous zero (the MECH-353 / V3-EXQ-642 lesson). Dimensionality is
        necessary, not sufficient: the encoder must be trained too.
    P1: E2WorldForward trains on FROZEN z_world (stop-gradient on z_world_t
        inputs: target = z_world_next.detach()). Do NOT propagate forward-model
        loss back into the encoder -- it drifts toward trivially-predictable
        representations.
    P2: attribution evaluation at world_dim >= 128 with ARC-065 behavioral
        diversity active (balanced agent-caused vs externally-caused events).

Biological grounding (SD-031 design doc): parietal outcome-prediction circuits
(Desmurget & Sirigu 2009), posterior-parietal / MD-thalamus / colliculus
corollary discharge (Sommer & Wurtz), ACC/insula outcome prediction-error
(Holroyd & Coles 2002), schizophrenia passivity-of-world-consequences (Frith
2000), sense-of-agency temporal binding (Haggard 2017).

MECH-094: forward() is a waking forward model and is unrestricted (it is the
evaluator/rollout-mode read, MECH-257). comparator_residual() is the
retrospective comparator-mode read and IS MECH-094-gated: it returns a zeroed
residual under simulation_mode so replay / DMN content cannot generate a
spurious agency signal.

See docs/architecture/sd_031_e2_world_forward_model.md,
docs/architecture/self_attribution_per_stream.md,
ree_core/predictors/e2_harm_s.py (the ARC-033 sibling template),
ree_core/latent/stack.py: ResidualHarmForward.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.latent.stack import ResidualHarmForward


# Minimum z_world dimensionality at which the discriminative / attribution arm
# of the comparator is considered viable. Set to the REEConfig.large preset
# (world_dim=128). The 2026-06-06 cluster autopsy established that world_dim=32
# lacks the discriminative granularity the attribution arm requires.
MIN_DISCRIMINATIVE_WORLD_DIM: int = 128


@dataclass
class E2WorldConfig:
    """Configuration for the E2_world causal-footprint forward model (SD-031).

    Disabled by default for full backward compatibility with all existing
    experiments. Set use_e2_world_forward=True to enable.

    z_world_dim has NO safe literal default (it is Optional[int]=None and
    REQUIRED when the model is enabled). This is deliberate: a literal default
    of 32 (the LatentStackConfig.world_dim default) would silently reintroduce
    the dim=32 granularity ceiling diagnosed by the 2026-06-06 cluster autopsy
    if a caller forgot to thread world_dim through. Construction reads
    z_world_dim from config.latent.world_dim at the agent level.

    Phased training notes (mirrors ARC-033 e2_harm_s):
        P0: train the z_world encoder (SD-009 event-contrastive + SD-018
            resource proximity) so z_world is discriminative BEFORE the
            forward model trains.
        P1: train E2WorldForward on frozen z_world targets (stop-gradient on
            z_world: target = z_world_next.detach()).
        P2: evaluate world_forward_r2 and the comparator attribution gap at
            world_dim >= 128 with ARC-065 diversity active.

    SD-013 interventional extension (use_interventional=True): contrastive
    margin loss pushing E2WorldForward predictions for a_actual vs a_cf apart.
    Higher priority for z_world than z_harm_s per the SD-031 design doc:
    world state carries strong ambient correlations (landmarks, resources,
    hazards persist across steps) that compress the action contribution
    without an interventional constraint (Scholkopf et al. 2021).
    """

    # Master switch -- disabled by default (backward compat).
    use_e2_world_forward: bool = False

    # Dimensions -- z_world_dim is REQUIRED (no safe literal default; see above).
    # Must match LatentStackConfig.world_dim of the agent.
    z_world_dim: Optional[int] = None
    action_dim: int = 4

    # Architecture.
    hidden_dim: int = 128          # transition-net hidden width (ResidualHarmForward default)
    action_enc_dim: int = 16       # action embedding width (ResidualHarmForward fixed at 16)

    # Training. E2.world_forward saturates world_forward_r2 ~0.95 at 3e-4
    # (EXQ-030b baseline); the dedicated module uses the same E2 LR.
    learning_rate: float = 3e-4

    # SD-013 interventional contrastive training (disabled by default).
    use_interventional: bool = False
    interventional_fraction: float = 0.3
    interventional_margin: float = 0.1

    # Carry-forward guard. By default the model HARD-ASSERTS
    # z_world_dim >= MIN_DISCRIMINATIVE_WORLD_DIM at construction. Set True
    # ONLY for deliberate bulk-only / ablation runs that knowingly operate the
    # forward model below the discriminative threshold; the comparator then
    # reports attribution_ready=False and returns a zeroed residual rather than
    # a misleadingly-zero attribution gap.
    allow_subthreshold_dim: bool = False


class E2WorldForward(nn.Module):
    """SD-031: E2_world single-pass forward-model comparator on z_world.

    f(z_world_t, a_t) -> z_world_{t+1}  (residual delta), plus the single-pass
    MECH-256 comparator residual = z_world_observed - f(z_world_prev, a_actual).

    Wrapper around ResidualHarmForward (the generic residual-delta module in
    stack.py) providing:
      - E2WorldConfig-based instantiation with the world_dim>=128 guard
      - forward() / compute_loss() for phased P1 training
      - comparator_residual() for the single-pass MECH-256 agency signal
      - attribution_ready for the dim-gated discriminative arm
      - compute_interventional_loss() for the SD-013 analogue (z_world)

    Usage in experiment scripts:
        from ree_core.predictors.e2_world import E2WorldForward, E2WorldConfig
        cfg = E2WorldConfig(use_e2_world_forward=True, z_world_dim=128, action_dim=4)
        world_fwd = E2WorldForward(cfg)
        opt = torch.optim.Adam(world_fwd.parameters(), lr=cfg.learning_rate)

        # P1 training step (stop-grad target):
        z_pred = world_fwd(z_world_t, action_t)
        loss = world_fwd.compute_loss(z_pred, z_world_next.detach())
        opt.zero_grad(); loss.backward(); opt.step()

        # P2 single-pass comparator (MECH-256 -- no counterfactual):
        residual = world_fwd.comparator_residual(z_world_obs, z_world_prev, a_actual)
        agency_signal = residual.norm(dim=-1)   # large -> externally-caused
    """

    def __init__(self, config: Optional[E2WorldConfig] = None):
        super().__init__()
        self.config = config or E2WorldConfig()

        z_world_dim = self.config.z_world_dim
        if z_world_dim is None:
            raise ValueError(
                "E2WorldConfig.z_world_dim is required (no safe default) -- "
                "pass config.latent.world_dim. A literal default of 32 would "
                "silently reintroduce the dim=32 granularity ceiling "
                "(failure_autopsy_zworld-integration-cluster_2026-06-06)."
            )

        # Carry-forward guard: hard-assert the discriminative dim threshold
        # unless an explicit bulk-only/ablation opt-in is set.
        if (not self.config.allow_subthreshold_dim) and z_world_dim < MIN_DISCRIMINATIVE_WORLD_DIM:
            raise ValueError(
                "E2_world comparator requires world_dim >= "
                "{min_dim} for its discriminative/attribution arm "
                "(failure_autopsy_zworld-integration-cluster_2026-06-06: "
                "z_world at dim=32 is a competent bulk predictor but lacks "
                "discriminative granularity). Got world_dim={got}. Use "
                "REEConfig.large() (world_dim=128), or set "
                "allow_subthreshold_dim=True ONLY for deliberate bulk-only / "
                "ablation runs (attribution_ready will be False).".format(
                    min_dim=MIN_DISCRIMINATIVE_WORLD_DIM, got=z_world_dim
                )
            )

        self.z_world_dim = z_world_dim
        self.action_dim = self.config.action_dim

        # Delegate to ResidualHarmForward -- the generic residual-delta forward
        # model (not harm-specific; its own docstring notes it mirrors
        # E2FastPredictor.world_forward()).
        self._residual_fwd = ResidualHarmForward(
            z_harm_dim=z_world_dim,
            action_dim=self.config.action_dim,
            hidden_dim=self.config.hidden_dim,
        )

    @property
    def attribution_ready(self) -> bool:
        """True when the discriminative/attribution arm is viable.

        False only on the explicit allow_subthreshold_dim bulk-only path with
        z_world_dim < MIN_DISCRIMINATIVE_WORLD_DIM. Callers MUST consult this
        before trusting comparator_residual as an agency signal -- a zeroed
        residual from a not-ready model is a "do not interpret" sentinel, not a
        genuine self-caused (zero-gap) attribution.
        """
        return self.z_world_dim >= MIN_DISCRIMINATIVE_WORLD_DIM

    def forward(
        self,
        z_world: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict z_world at next step via residual delta.

        This is the bulk / evaluator-mode read (MECH-257): unrestricted, used
        for rollout scoring as well as inside comparator_residual. It is
        dim-agnostic and not MECH-094-gated.

        IMPORTANT: during P1 training, do NOT propagate gradients from this
        output back into the z_world encoder. Use z_world_next.detach() as the
        training target (stop-gradient discipline).

        Args:
            z_world: [batch, z_world_dim] -- world-state latent (current step)
            action:  [batch, action_dim]  -- action taken (one-hot or continuous)

        Returns:
            z_world_pred: [batch, z_world_dim] -- predicted world latent at next step
        """
        return self._residual_fwd(z_world, action)

    def compute_loss(
        self,
        z_world_pred: torch.Tensor,
        z_world_next: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss for P1 forward-model training.

        The target z_world_next MUST be detached from the encoder computation
        graph before calling this (the caller applies .detach() -- this method
        does not, to support any calling convention).

        Args:
            z_world_pred: [batch, z_world_dim] -- predicted next world latent (from forward())
            z_world_next: [batch, z_world_dim] -- actual next world latent (should be .detach()ed)

        Returns:
            loss: scalar MSE loss
        """
        return F.mse_loss(z_world_pred, z_world_next)

    def comparator_residual(
        self,
        z_world_observed: torch.Tensor,
        z_world_prev: torch.Tensor,
        action: torch.Tensor,
        simulation_mode: bool = False,
    ) -> torch.Tensor:
        """Single-pass MECH-256 agency signal on z_world.

            residual_world = z_world_observed - E2WorldForward(z_world_prev, a_actual)

        One forward pass on the actually-taken action -- NO counterfactual
        second call. Self-caused world changes produce attenuated residuals
        (the prediction succeeds); externally-caused changes produce large
        residuals (the prediction fails).

        MECH-094: this is the retrospective comparator-mode read; under
        simulation_mode it returns a zeroed residual so replay / DMN content
        cannot generate a spurious agency signal.

        Carry-forward guard: when attribution_ready is False (bulk-only model
        below the discriminative dim threshold), returns a zeroed residual --
        a "do not interpret" sentinel, NOT a genuine zero-gap attribution.
        Callers must check attribution_ready.

        Args:
            z_world_observed: [batch, z_world_dim] -- observed world latent at t
            z_world_prev:     [batch, z_world_dim] -- world latent at t-1
            action:           [batch, action_dim]  -- the action actually taken at t-1
            simulation_mode:  MECH-094 gate (True -> zeroed residual)

        Returns:
            residual_world: [batch, z_world_dim] -- the agency signal (zeros when
                simulation_mode or not attribution_ready)
        """
        if simulation_mode or not self.attribution_ready:
            return torch.zeros_like(z_world_observed)
        predicted = self._residual_fwd(z_world_prev, action)
        return z_world_observed - predicted

    def compute_interventional_loss(
        self,
        z_world: torch.Tensor,
        a_actual: torch.Tensor,
        a_cf: torch.Tensor,
    ) -> torch.Tensor:
        """SD-013 analogue: contrastive interventional loss for action-sensitivity.

        Pushes E2WorldForward predictions for a_actual and a_cf apart by at
        least interventional_margin in L2. Forces the forward model to produce
        action-divergent world predictions so the comparator residual is
        action-discriminating even under the strong ambient correlations of
        world state (landmarks/resources/hazards persisting across steps).

        Margin loss: max(0, margin - ||z_pred_actual - z_pred_cf||_2)

        IMPORTANT: z_world MUST be detached from the encoder graph before
        calling this (same P1 stop-gradient discipline as compute_loss).

        Args:
            z_world:  [batch, z_world_dim] -- current world latent (detached)
            a_actual: [batch, action_dim]  -- action actually taken
            a_cf:     [batch, action_dim]  -- counterfactual action (must differ)

        Returns:
            loss: scalar contrastive margin loss (>= 0)
        """
        z_pred_actual = self._residual_fwd(z_world, a_actual)
        z_pred_cf = self._residual_fwd(z_world, a_cf)
        l2_dist = (z_pred_actual - z_pred_cf).norm(dim=-1)
        margin = self.config.interventional_margin
        return F.relu(margin - l2_dist).mean()
