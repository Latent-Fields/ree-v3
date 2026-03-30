"""
Latent Stack (L-space) — V3 Implementation

SD-005: z_gamma is split into two distinct streams:
  z_self  [self_dim]  — proprioceptive + interoceptive body-state  (E2 domain)
  z_world [world_dim] — exteroceptive world-state                   (E3/Hippocampal/ResidueField domain)

This split resolves three V2 failure modes:
1. SD-003 causal attribution: world_delta requires z_world to exist separately
2. MECH-069 incommensurability: signals are partially correlated in unified z_gamma
3. Residue accumulation: residue must track world-delta, not self-delta

Shared latent channels (unchanged from V2):
  z_beta  [beta_dim]  — affective latent (arousal/valence; integrates self + world signals)
  z_theta [theta_dim] — sequence context (temporal ordering)
  z_delta [delta_dim] — regime/motivation (long-horizon context)

Encoder is split at the bottom: body observation channels → z_self encoder,
world observation channels → z_world encoder. Both encoders receive top-down
conditioning from the shared z_beta / z_delta stack.

Module read/write contract:
  E1: reads z_self + z_world (associative prior, sensory prediction)
  E2: reads z_self, writes z_self_next (motor-sensory domain)
  E3 complex: reads z_world, writes z_world updates (harm/goal domain)
  HippocampalModule: reads z_world via action objects (planning domain)
  ResidueField: reads z_world, accumulates world_delta
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from ree_core.utils.config import LatentStackConfig


class ReafferencePredictor(nn.Module):
    """
    SD-007 / MECH-098 / MECH-101: Perspective-shift correction for z_world.

    Predicts the z_world change caused purely by self-motion (locomotion):
        Δz_world_loco = ReafferencePredictor(z_world_raw_prev, a_prev)

    Applied in LatentStack.encode():
        z_world_corrected = z_world_raw - Δz_world_loco

    Trained on empty-space steps (transition_type == "none") where the only
    source of z_world change is the perspective shift from locomotion.

    MECH-101: Input must be z_world_raw_prev (NOT z_self_prev). In local-view
    environments, Δz_world_raw from locomotion includes cell content entering
    view — inaccessible from body state alone but available in z_world_raw_prev.
    Biological basis: MSTd receives full visual optic flow (content-dependent)
    + vestibular + efference copy. The optic flow is scene-content-dependent;
    body state encodes position only. EXQ-027 confirmed R2=0.027 with z_self
    inputs (near-zero because cell content dominates the delta).

    Biological basis: MSTd congruent/incongruent neuron populations (Gu et al.
    2008) decompose optic flow into self-motion (reafference) vs genuine world
    change (exafference) using efference copy from premotor cortex (SLF pathway).

    See docs/architecture/sd_004_sd_005_encoder_codesign.md §3.
    """

    def __init__(self, world_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.world_dim = world_dim
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(world_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, world_dim),
        )

    def forward(
        self,
        z_world_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict z_world delta caused by self-motion (perspective shift).

        Args:
            z_world_prev: [batch, world_dim] — z_world_raw from previous step
            a_prev:       [batch, action_dim]

        Returns:
            Δz_world_loco: [batch, world_dim]
        """
        return self.net(torch.cat([z_world_prev, a_prev], dim=-1))

    def correct_z_world(
        self,
        z_world_raw: torch.Tensor,
        z_world_prev: torch.Tensor,
        a_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply reafference correction.

        z_world_corrected = z_world_raw - Δz_world_loco

        Args:
            z_world_raw:   [batch, world_dim] — current raw z_world from encoder
            z_world_prev:  [batch, world_dim] — z_world_raw from previous step
            a_prev:        [batch, action_dim]

        Returns:
            z_world_corrected: [batch, world_dim]
        """
        return z_world_raw - self.forward(z_world_prev, a_prev)


class HarmEncoder(nn.Module):
    """
    SD-010: Dedicated harm-stream encoder (nociceptive separation, ARC-027).

    Encodes the harm_obs vector into z_harm. This encoder is NEVER subject to
    reafference correction — nociception is not cancellable by self-motion
    prediction (contrast with z_world, which carries perspective-dependent
    exteroceptive content and requires SD-007 correction).

    Biological basis: the spinothalamic nociceptive pathway projects to anterior
    insula / ACC independently of the dorsal exteroceptive stream (V1→MT/MST→PPC)
    that carries the reafference signal. Hazard proximity is an interoceptive-
    adjacent signal, not a perspective-dependent world-state.

    harm_obs layout (CausalGridWorldV2 with use_proxy_fields=True):
      [0:25]   hazard_field_view   — normalised hazard proximity gradient (5×5 grid)
      [25:50]  resource_field_view — normalised resource proximity gradient (5×5 grid)
      [50]     harm_exposure       — nociceptive EMA from body_obs (env step accumulator)
      Total: harm_obs_dim = 51

    This class is instantiated directly in experiment scripts (not inside LatentStack)
    so that it can trivially bypass the reafference pipeline. Experiments import it as:
        from ree_core.latent.stack import HarmEncoder

    See CLAUDE.md: SD-010.
    """

    def __init__(self, harm_obs_dim: int = 51, z_harm_dim: int = 32):
        super().__init__()
        self.harm_obs_dim = harm_obs_dim
        self.z_harm_dim = z_harm_dim
        self.encoder = nn.Sequential(
            nn.Linear(harm_obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_harm_dim),
        )

    def forward(self, harm_obs: torch.Tensor) -> torch.Tensor:
        """
        Encode harm observations into z_harm latent.

        Args:
            harm_obs: [batch, harm_obs_dim] — hazard field + resource field + exposure

        Returns:
            z_harm: [batch, z_harm_dim]
        """
        return self.encoder(harm_obs)


class AffectiveHarmEncoder(nn.Module):
    """
    SD-011: Affective-motivational harm stream encoder (C-fiber/paleospinothalamic analog).

    Encodes accumulated harm exposure (harm_obs_a) into z_harm_a. The input is a
    temporally-integrated version of harm_obs -- an EMA maintained in the experiment
    loop over tau=10-30 steps -- representing accumulated homeostatic deviation rather
    than immediate proximity.

    Distinct from HarmEncoder (sensory-discriminative, Adelta-pathway analog) in:
    - Temporal scope: operates on integrated/accumulated signal, not immediate proximity
    - Functional role: feeds E3 commit gating directly as motivational urgency signal
    - NOT counterfactually predicted (no E2_harm_a forward model needed)
    - ARC-016 harm variance gating operates on z_harm_a (accumulated threat state
      should modulate commit threshold)

    Biological grounding: C-fiber / paleospinothalamic tract (medial system) projects
    via intralaminar thalamus (CM/PF) to ACC, anterior insula, amygdala. Encodes
    unpleasantness, motivational urgency, homeostatic deviation. Slower temporal
    profile than lateral (Adelta) system. Less suppressed by predictability --
    Rainville et al. (1997, Science): hypnotic modulation of unpleasantness modulates
    ACC (affective) not S1 (discriminative). Craig (2003): pain as homeostatic emotion.

    See CLAUDE.md: SD-011. See HarmEncoder for the sensory-discriminative partner.
    """

    def __init__(self, harm_obs_a_dim: int = 51, z_harm_a_dim: int = 16):
        super().__init__()
        self.harm_obs_a_dim = harm_obs_a_dim
        self.z_harm_a_dim = z_harm_a_dim
        self.encoder = nn.Sequential(
            nn.Linear(harm_obs_a_dim, 32),
            nn.ReLU(),
            nn.Linear(32, z_harm_a_dim),
        )

    def forward(self, harm_obs_a: torch.Tensor) -> torch.Tensor:
        """
        Encode accumulated harm observations into z_harm_a latent.

        Args:
            harm_obs_a: [batch, harm_obs_a_dim] -- EMA-accumulated harm proximity vector

        Returns:
            z_harm_a: [batch, z_harm_a_dim]
        """
        return self.encoder(harm_obs_a)


class HarmForwardModel(nn.Module):
    """
    ARC-033: Sensory-discriminative harm forward model (E2_harm_s).

    Predicts z_harm_s_next = E2_harm_s(z_harm_s, action), enabling SD-003
    counterfactual attribution to operate within the harm stream without a
    cross-stream bridge from z_world.

    This is the solution to the EXQ-093/094 bridge_r2=0 problem. HarmBridge
    (z_world -> z_harm) has R^2=0 by construction: SD-010 makes z_world perp
    z_harm. HarmForwardModel avoids this by predicting z_harm_s from z_harm_s
    (same stream) + action, exploiting the fact that sensory-discriminative
    proximity is PREDICTABLE from action: moving away from a hazard reduces
    proximity in a learnable, action-conditional way.

    SD-003 redesigned pipeline (post-SD-011):
        z_harm_s_cf = harm_fwd(z_harm_s, a_cf)
        causal_sig = e3.harm_eval_z_harm(z_harm_s_actual) - e3.harm_eval_z_harm(z_harm_s_cf)

    Biological grounding: Keltner et al. (2006, J Neurosci): predictability
    suppresses sensory-discriminative pain activity (S1/S2). The brain models
    expected nociceptive consequences of voluntary movement and cancels predicted
    input -- the same cancellation logic that is not available for the affective
    stream (ACC/insula, which integrates homeostatic context beyond single actions).

    See CLAUDE.md: SD-011, ARC-033. Compare: HarmBridge (deprecated counterfactual).
    """

    def __init__(self, z_harm_dim: int = 32, action_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.z_harm_dim = z_harm_dim
        self.action_dim = action_dim
        self.forward_net = nn.Sequential(
            nn.Linear(z_harm_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_harm_dim),
        )

    def forward(self, z_harm_s: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next harm latent given current harm latent and action.

        Args:
            z_harm_s: [batch, z_harm_dim] -- sensory-discriminative harm latent (current)
            action:   [batch, action_dim] -- action (one-hot or continuous)

        Returns:
            z_harm_s_next: [batch, z_harm_dim] -- predicted next harm latent
        """
        x = torch.cat([z_harm_s, action], dim=-1)
        return self.forward_net(x)


class HarmBridge(nn.Module):
    """
    SD-010 / SD-003: Counterfactual harm bridge.

    DEPRECATED for SD-003 counterfactual use (2026-03-24).
    EXQ-093/094 confirmed bridge_r2=0 -- z_world perp z_harm by SD-010 design;
    this bridge has nothing to learn. Use HarmForwardModel (ARC-033) instead:
        z_harm_s_cf = HarmForwardModel(z_harm_s, a_cf)
    This class is kept for backward compatibility with existing experiment scripts.

    Maps z_world latent -> z_harm latent space, allowing the SD-003 attribution
    pipeline to compute counterfactual harm without access to raw harm_obs.

    Used as:
        z_harm_cf = harm_bridge(E2.world_forward(z_world, a_cf))
        causal_sig = e3.harm_eval_z_harm(z_harm_actual) - e3.harm_eval_z_harm(z_harm_cf)

    This is a learned projector trained to minimise:
        ||HarmBridge(z_world) - HarmEncoder(harm_obs)||

    so that the harm latent derived from the world model approximates the latent
    derived from the direct nociceptive stream, enabling counterfactual queries
    where harm_obs is not available (the agent cannot observe harm_obs for actions
    it did not take).

    Biological analogy: the cortical harm evaluation pathway (slow, contextual)
    must be able to simulate "what would I have felt if I had acted differently"
    using the world model alone, without re-running the spinothalamic pathway.
    This is the bridge between the slow cortical and fast nociceptive routes.

    See CLAUDE.md: SD-010.
    """

    def __init__(self, world_dim: int = 32, z_harm_dim: int = 32):
        super().__init__()
        self.world_dim = world_dim
        self.z_harm_dim = z_harm_dim
        self.bridge = nn.Sequential(
            nn.Linear(world_dim, 64),
            nn.ReLU(),
            nn.Linear(64, z_harm_dim),
        )

    def forward(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Project world latent into harm latent space.

        Args:
            z_world: [batch, world_dim] — world-state latent (from E2.world_forward or encoder)

        Returns:
            z_harm_approx: [batch, z_harm_dim] — harm latent approximation for counterfactual
        """
        return self.bridge(z_world)


class _GradReverseFn(Function):
    """Gradient reversal function (Ganin et al. 2016)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:  # type: ignore[override]
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.alpha * grad_output, None


class AdversarialSplitHead(nn.Module):
    """
    SD-010 / EXQ-090: Adversarial z_self / harm boundary maintenance.

    Prevents "schizophrenic drift": the gradual blurring of the z_self encoder's
    boundary as harm training pressure inadvertently causes z_self to carry
    harm-predictive information (a z_world property).

    Mechanism: gradient reversal layer (Ganin et al. 2016 JMLR).
      - During forward pass: z_self -> harm_pred (normal MLP)
      - During backward pass: gradient into z_self encoder is NEGATED
      - Effect: z_self encoder learns to NOT be informative about harm outcomes

    The adversarial head trains normally to predict harm from z_self; the encoder
    receives the reversed gradient, pushing it toward harm-uninformative encodings.

    Usage in training loop:
        adv_loss = adv_head(z_self.detach(), z_self)  # train head on detached z_self
        # Then backward on adv_loss -- gradient reversal handles encoder update

        # Or use the helper:
        encoder_penalty, head_loss = adv_head.split_losses(z_self, harm_label)

    PASS criterion (EXQ-090):
        R2(z_self -> harm_outcome) < 0.05 with adversarial loss active
        vs R2 > 0.15 without adversarial loss (demonstrating drift prevention)

    References:
        Ganin et al. (2016). Domain-adversarial training of neural networks. JMLR.
    """

    def __init__(self, self_dim: int = 32, hidden_dim: int = 32, grl_alpha: float = 1.0):
        """
        Args:
            self_dim:   z_self dimensionality
            hidden_dim: hidden layer size for the adversarial harm predictor
            grl_alpha:  gradient reversal scale factor (1.0 = full reversal)
        """
        super().__init__()
        self.grl_alpha = grl_alpha
        self.harm_predictor = nn.Sequential(
            nn.Linear(self_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_self: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradient reversal.

        The gradient reversal layer sits between z_self and the harm predictor.
        Backward pass through this method will negate the gradient back to z_self.

        Args:
            z_self: [batch, self_dim] -- z_self encoder output (with gradients)

        Returns:
            harm_pred: [batch, 1] -- predicted harm (used for adversarial head loss)
        """
        z_reversed = _GradReverseFn.apply(z_self, self.grl_alpha)
        return self.harm_predictor(z_reversed)

    def split_losses(
        self,
        z_self: torch.Tensor,
        harm_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute split adversarial losses.

        Returns:
            encoder_penalty: loss that, when backpropagated into z_self encoder,
                             pushes it toward harm-uninformative encodings (via GRL)
            head_loss:       loss for updating the adversarial harm predictor head
                             (trained with normal gradients on detached z_self)
        """
        # Encoder update: backprop through GRL negates gradient into encoder
        harm_pred_grl = self.forward(z_self)
        encoder_penalty = F.mse_loss(harm_pred_grl, harm_label)

        # Head update: train harm predictor on detached z_self (normal gradients)
        harm_pred_direct = self.harm_predictor(z_self.detach())
        head_loss = F.mse_loss(harm_pred_direct, harm_label)

        return encoder_penalty, head_loss


@dataclass
class LatentState:
    """Complete latent state for V3 REE.

    SD-005: z_gamma replaced by z_self + z_world.

    Attributes:
        z_self:      Proprioceptive/interoceptive body-state  [batch, self_dim]
        z_world:     Exteroceptive world-state (perspective-corrected if SD-007 active)
                     [batch, world_dim]
        z_beta:      Affective latent (shared)                [batch, beta_dim]
        z_theta:     Sequence context (shared)                [batch, theta_dim]
        z_delta:     Regime/motivation (shared)               [batch, delta_dim]
        z_harm:      Harm-saliency latent from lateral encoder head (MECH-099)
                     [batch, harm_dim] or None if lateral head not enabled.
        z_world_raw: Raw z_world before reafference correction (SD-007 diagnostic).
                     None if SD-007 not enabled or prev_action not available.
        precision:   Per-channel precision values
        timestamp:   Optional timestep index
        hypothesis_tag: True = internally generated (replay/simulation).
            Must be True for all content produced by HippocampalModule.replay()
            and DMN-mode inference. ResidueField checks this before accumulating
            (MECH-094: tag loss = PTSD/psychosis mechanism).
    """
    z_self: torch.Tensor    # [batch, self_dim]
    z_world: torch.Tensor   # [batch, world_dim]
    z_beta: torch.Tensor    # [batch, beta_dim]
    z_theta: torch.Tensor   # [batch, theta_dim]
    z_delta: torch.Tensor   # [batch, delta_dim]
    precision: Dict[str, torch.Tensor]
    timestamp: Optional[int] = None
    hypothesis_tag: bool = False  # MECH-094
    z_harm: Optional[torch.Tensor] = None        # SD-010 sensory-discriminative harm [batch, harm_dim]
    z_harm_a: Optional[torch.Tensor] = None      # SD-011 affective-motivational harm [batch, z_harm_a_dim]
    z_world_raw: Optional[torch.Tensor] = None   # SD-007 diagnostic [batch, world_dim]
    event_logits: Optional[torch.Tensor] = None  # SD-009 [batch, 3] for CE loss; None if not enabled

    def to_tensor(self) -> torch.Tensor:
        """Concatenate all channels into a single tensor (excludes z_harm)."""
        return torch.cat([
            self.z_self, self.z_world,
            self.z_beta, self.z_theta, self.z_delta
        ], dim=-1)

    @property
    def device(self) -> torch.device:
        return self.z_self.device

    def detach(self) -> "LatentState":
        return LatentState(
            z_self=self.z_self.detach(),
            z_world=self.z_world.detach(),
            z_beta=self.z_beta.detach(),
            z_theta=self.z_theta.detach(),
            z_delta=self.z_delta.detach(),
            precision={k: v.detach() for k, v in self.precision.items()},
            timestamp=self.timestamp,
            hypothesis_tag=self.hypothesis_tag,
            z_harm=self.z_harm.detach() if self.z_harm is not None else None,
            z_harm_a=self.z_harm_a.detach() if self.z_harm_a is not None else None,
            z_world_raw=self.z_world_raw.detach() if self.z_world_raw is not None else None,
            event_logits=self.event_logits.detach() if self.event_logits is not None else None,
        )


class SplitEncoder(nn.Module):
    """SD-005 split encoder: body obs → z_self, world obs → z_world.

    Both streams receive top-down conditioning from the shared z_beta stack.

    MECH-099 (three-stream architecture): optionally includes a lateral_head
    that processes harm-salient channels (hazard entity + contamination view)
    directly to produce z_harm for E3, bypassing E2_world.

    Lateral head input (when enabled):
      - Hazard channel from local_view: 25 values (5×5 grid, entity type 3)
        Indices in world_obs [0:175]: [3, 10, 17, ..., 171] (every 7 from 3)
      - Contamination view: world_obs[175:200] (25 values)
      Total lateral input dim = 50

    The lateral head is optional (harm_dim=0 disables it).
    """

    # Indices of hazard channel within local_view (5×5×7 one-hot, entity type 3)
    # hazard is entity type index 3; for each of 25 cells, offset = cell*7 + 3
    HAZARD_INDICES = list(range(3, 175, 7))  # [3, 10, 17, ..., 171], length 25
    CONTAMINATION_SLICE = slice(175, 200)    # contamination_view within world_obs

    def __init__(
        self,
        body_obs_dim: int,
        world_obs_dim: int,
        self_dim: int,
        world_dim: int,
        topdown_dim: int,
        hidden_dim: int = 64,
        harm_dim: int = 0,
        use_event_classifier: bool = False,
    ):
        super().__init__()
        self.self_dim = self_dim
        self.world_dim = world_dim
        self.harm_dim = harm_dim
        self.use_event_classifier = use_event_classifier

        self.self_encoder = nn.Sequential(
            nn.Linear(body_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self_dim),
        )
        self.world_encoder = nn.Sequential(
            nn.Linear(world_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, world_dim),
        )

        # MECH-099: lateral head for harm-salient features (hazard + contamination)
        if harm_dim > 0:
            lateral_input_dim = 25 + 25  # hazard channel + contamination_view
            self.lateral_head = nn.Sequential(
                nn.Linear(lateral_input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, harm_dim),
            )
        else:
            self.lateral_head = None

        # SD-009 / MECH-100: event contrastive classifier head.
        # Maps z_world → logits over event types [none, env_caused, agent_caused].
        # Training loss: CE with transition_type labels from environment.
        if use_event_classifier:
            self.event_classifier = nn.Linear(world_dim, 3)
        else:
            self.event_classifier = None

        # Top-down conditioning projections
        if topdown_dim > 0:
            self.self_topdown = nn.Linear(topdown_dim, self_dim)
            self.world_topdown = nn.Linear(topdown_dim, world_dim)
        else:
            self.self_topdown = None
            self.world_topdown = None

        # Learnable precision logits
        self.self_precision_logit = nn.Parameter(torch.zeros(self_dim))
        self.world_precision_logit = nn.Parameter(torch.zeros(world_dim))

    def forward(
        self,
        body_obs: torch.Tensor,
        world_obs: torch.Tensor,
        topdown: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode body and world observations into latent streams.

        Returns:
            (z_self, z_world, prec_self, prec_world, z_harm, event_logits)

            z_harm is None if lateral_head is not enabled (harm_dim == 0).
            event_logits is None if use_event_classifier is False (SD-009 disabled).
            event_logits shape: [batch, 3] — [none, env_caused, agent_caused].
        """
        z_self = self.self_encoder(body_obs)
        z_world = self.world_encoder(world_obs)

        if topdown is not None:
            if self.self_topdown is not None:
                z_self = z_self + self.self_topdown(topdown)
            if self.world_topdown is not None:
                z_world = z_world + self.world_topdown(topdown)

        prec_self = torch.sigmoid(self.self_precision_logit).unsqueeze(0).expand(
            body_obs.shape[0], -1
        )
        prec_world = torch.sigmoid(self.world_precision_logit).unsqueeze(0).expand(
            world_obs.shape[0], -1
        )

        z_self = z_self * prec_self
        z_world = z_world * prec_world

        # MECH-099 lateral head (optional)
        z_harm = None
        if self.lateral_head is not None:
            hazard_ch = world_obs[:, self.HAZARD_INDICES]          # [batch, 25]
            cont_view = world_obs[:, self.CONTAMINATION_SLICE]     # [batch, 25]
            lateral_input = torch.cat([hazard_ch, cont_view], dim=-1)  # [batch, 50]
            z_harm = self.lateral_head(lateral_input)              # [batch, harm_dim]

        # SD-009 / MECH-100 event classifier (optional)
        event_logits = None
        if self.event_classifier is not None:
            event_logits = self.event_classifier(z_world)         # [batch, 3]

        return z_self, z_world, prec_self, prec_world, z_harm, event_logits


class SharedDepthEncoder(nn.Module):
    """Encoder for shared latent channels (z_beta, z_theta, z_delta)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        topdown_dim: int = 0,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        if topdown_dim > 0:
            self.topdown_proj = nn.Linear(topdown_dim, output_dim)
        else:
            self.topdown_proj = None

        self.precision_logit = nn.Parameter(torch.zeros(output_dim))

    def forward(
        self,
        x: torch.Tensor,
        topdown: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        if topdown is not None and self.topdown_proj is not None:
            encoded = encoded + self.topdown_proj(topdown)
        precision = torch.sigmoid(self.precision_logit).unsqueeze(0).expand(
            x.shape[0], -1
        )
        return encoded * precision, precision


class LatentStack(nn.Module):
    """
    V3 Multi-timescale Latent Stack (L-space).

    Implements SD-005 split encoder:
    - Bottom-up: body_obs → z_self, world_obs → z_world
    - Shared stack: (z_self + z_world) → z_beta → z_theta → z_delta
    - Top-down conditioning from z_delta/z_theta/z_beta back to z_self/z_world

    Preserves V2 architectural invariants:
    - Downstream processes cannot directly overwrite z_self or z_world
    - Influence restricted to prediction-error updates and precision modulation
    """

    def __init__(self, config: Optional[LatentStackConfig] = None):
        super().__init__()
        self.config = config or LatentStackConfig()

        hidden = 64  # internal hidden dim for encoders

        # SD-005: split bottom encoder (MECH-099: harm_dim enables lateral head;
        # SD-009: use_event_classifier enables event contrastive head)
        self.split_encoder = SplitEncoder(
            body_obs_dim=self.config.body_obs_dim,
            world_obs_dim=self.config.world_obs_dim,
            self_dim=self.config.self_dim,
            world_dim=self.config.world_dim,
            topdown_dim=self.config.topdown_dim,
            hidden_dim=hidden,
            harm_dim=getattr(self.config, "harm_dim", 0),
            use_event_classifier=getattr(self.config, "use_event_classifier", False),
        )

        # SD-007: optional ReafferencePredictor for perspective-shift correction.
        # Enabled when reafference_action_dim > 0.
        _reaf_action_dim = getattr(self.config, "reafference_action_dim", 0)
        if _reaf_action_dim > 0:
            self.reafference_predictor: Optional[ReafferencePredictor] = ReafferencePredictor(
                world_dim=self.config.world_dim,
                action_dim=_reaf_action_dim,
                hidden_dim=hidden,
            )
        else:
            self.reafference_predictor = None

        combined_dim = self.config.self_dim + self.config.world_dim
        # Q-007: volatility signal (NE/LC analog) appended to beta_encoder input when enabled.
        # 0 = disabled (default, backward compat). See LatentStackConfig.volatility_signal_dim.
        _vol_dim = getattr(self.config, "volatility_signal_dim", 0)
        self._volatility_signal_dim = _vol_dim

        # Shared depth stack
        self.beta_encoder = SharedDepthEncoder(
            input_dim=combined_dim + _vol_dim,
            output_dim=self.config.beta_dim,
            topdown_dim=self.config.topdown_dim,
            hidden_dim=hidden,
        )
        self.theta_encoder = SharedDepthEncoder(
            input_dim=self.config.beta_dim,
            output_dim=self.config.theta_dim,
            topdown_dim=self.config.topdown_dim,
            hidden_dim=hidden,
        )
        self.delta_encoder = SharedDepthEncoder(
            input_dim=self.config.theta_dim,
            output_dim=self.config.delta_dim,
            topdown_dim=0,
            hidden_dim=hidden,
        )

        # SD-010: dedicated HarmEncoder (nociceptive separation, ARC-027).
        # Instantiated only when use_harm_stream=True. Kept outside the reafference
        # pipeline by construction — z_harm is never perspective-corrected.
        if getattr(self.config, "use_harm_stream", False):
            self.harm_encoder: Optional[HarmEncoder] = HarmEncoder(
                harm_obs_dim=getattr(self.config, "harm_obs_dim", 51),
                z_harm_dim=getattr(self.config, "z_harm_dim", 32),
            )
        else:
            self.harm_encoder = None

        # SD-011: AffectiveHarmEncoder (C-fiber/paleospinothalamic analog, ARC-033).
        # Encodes harm_obs_a (EMA of proximity fields, tau~20 steps) into z_harm_a.
        # NOT counterfactually predicted -- feeds E3 commit gating directly.
        if getattr(self.config, "use_affective_harm_stream", False):
            self.affective_harm_encoder: Optional[AffectiveHarmEncoder] = AffectiveHarmEncoder(
                harm_obs_a_dim=getattr(self.config, "harm_obs_a_dim", 50),
                z_harm_a_dim=getattr(self.config, "z_harm_a_dim", 16),
            )
        else:
            self.affective_harm_encoder = None

        # Top-down projections
        self.delta_to_theta = nn.Linear(self.config.delta_dim, self.config.topdown_dim)
        self.theta_to_beta = nn.Linear(self.config.theta_dim, self.config.topdown_dim)
        self.beta_to_split = nn.Linear(self.config.beta_dim, self.config.topdown_dim)

        # Prediction networks (for prediction error computation)
        self.self_predictor = nn.Linear(self.config.self_dim, self.config.self_dim)
        self.world_predictor = nn.Linear(self.config.world_dim, self.config.world_dim)
        self.beta_predictor = nn.Linear(self.config.beta_dim, self.config.beta_dim)
        self.theta_predictor = nn.Linear(self.config.theta_dim, self.config.theta_dim)
        self.delta_predictor = nn.Linear(self.config.delta_dim, self.config.delta_dim)

        self.total_dim = (
            self.config.self_dim + self.config.world_dim +
            self.config.beta_dim + self.config.theta_dim + self.config.delta_dim
        )

    def init_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> LatentState:
        """Initialize latent state with zeros."""
        device = device or torch.device("cpu")
        harm_dim = getattr(self.config, "harm_dim", 0)
        return LatentState(
            z_self=torch.zeros(batch_size, self.config.self_dim, device=device),
            z_world=torch.zeros(batch_size, self.config.world_dim, device=device),
            z_beta=torch.zeros(batch_size, self.config.beta_dim, device=device),
            z_theta=torch.zeros(batch_size, self.config.theta_dim, device=device),
            z_delta=torch.zeros(batch_size, self.config.delta_dim, device=device),
            precision={
                "self":  torch.ones(batch_size, self.config.self_dim, device=device) * 0.5,
                "world": torch.ones(batch_size, self.config.world_dim, device=device) * 0.5,
                "beta":  torch.ones(batch_size, self.config.beta_dim, device=device) * 0.5,
                "theta": torch.ones(batch_size, self.config.theta_dim, device=device) * 0.5,
                "delta": torch.ones(batch_size, self.config.delta_dim, device=device) * 0.5,
            },
            timestamp=0,
            z_harm=torch.zeros(batch_size, harm_dim, device=device) if harm_dim > 0 else None,
        )

    def _split_observation(
        self,
        observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split flat observation into body and world channels (SD-005)."""
        body = observation[:, :self.config.body_obs_dim]
        world = observation[:, self.config.body_obs_dim:]
        return body, world

    def encode(
        self,
        observation: torch.Tensor,
        prev_state: Optional[LatentState] = None,
        prev_action: Optional[torch.Tensor] = None,
        harm_obs: Optional[torch.Tensor] = None,
        harm_obs_a: Optional[torch.Tensor] = None,
        volatility_signal: Optional[torch.Tensor] = None,
    ) -> LatentState:
        """
        Encode observation into latent state.

        Implements bottom-up pass with top-down conditioning:
        1. Split observation into body/world channels
        2. Encode z_self, z_world with top-down conditioning
        3. SD-007: Apply reafference correction to z_world_raw if enabled
        4. Encode shared stack (beta, theta, delta)
        5. Temporal smoothing with previous state
        6. SD-010: If harm_obs provided and harm_encoder present, produce z_harm
           independently (NOT subject to reafference correction).

        Args:
            observation:  Raw observation [batch, body_obs_dim + world_obs_dim]
            prev_state:   Previous latent state for temporal continuity
            prev_action:  Action taken at t-1; used by SD-007 reafference correction.
                          Shape [batch, action_dim] or [action_dim] (auto-unsqueezed).
                          None = no correction (first step or SD-007 disabled).
            harm_obs:     SD-010 nociceptive observation [batch, harm_obs_dim].
                          When provided and harm_encoder is active, overrides z_harm
                          from the MECH-099 lateral head with the dedicated stream.
            harm_obs_a:   SD-011 affective-motivational harm observation [batch, harm_obs_a_dim].
                          EMA of proximity fields maintained in environment (tau~20 steps).
                          When provided and affective_harm_encoder is active, produces z_harm_a.
                          None = disabled (default, backward compat).
            volatility_signal: Q-007/EXQ-051b — NE/LC analog [batch, volatility_signal_dim]
                          or scalar float/tensor auto-expanded. When provided and
                          LatentStackConfig.volatility_signal_dim > 0, appended to
                          combined_init before beta_encoder. Represents running_variance
                          from E3's harm prediction error (unexpected uncertainty signal).
                          None = disabled (default, backward compat).

        Returns:
            New LatentState (z_world is perspective-corrected if SD-007 enabled;
            z_world_raw holds the uncorrected value for diagnostic use).
        """
        batch_size = observation.shape[0]
        device = observation.device

        if prev_state is None:
            prev_state = self.init_state(batch_size, device)

        body_obs, world_obs = self._split_observation(observation)

        # First pass: no top-down (to get initial estimates for top-down computation)
        z_self_init, z_world_init, _, _, _, _ = self.split_encoder(body_obs, world_obs)
        combined_init = torch.cat([z_self_init, z_world_init], dim=-1)

        # Q-007: append volatility signal (NE/LC unexpected-uncertainty analog) to beta input.
        # Yu & Dayan 2005 (PMID 15944135): NE = unexpected uncertainty = running_variance spike.
        # Only active when volatility_signal_dim > 0 AND volatility_signal is provided.
        if volatility_signal is not None and self._volatility_signal_dim > 0:
            vs = volatility_signal
            if not isinstance(vs, torch.Tensor):
                vs = torch.tensor([[vs]], dtype=torch.float32, device=device)
            vs = vs.to(device).float()
            if vs.dim() == 0:
                vs = vs.unsqueeze(0).unsqueeze(0).expand(batch_size, self._volatility_signal_dim)
            elif vs.dim() == 1 and vs.shape[0] == self._volatility_signal_dim:
                vs = vs.unsqueeze(0).expand(batch_size, -1)
            elif vs.shape == (batch_size,):
                vs = vs.unsqueeze(-1).expand(-1, self._volatility_signal_dim)
            combined_init = torch.cat([combined_init, vs], dim=-1)

        z_beta_init, _ = self.beta_encoder(combined_init)
        z_theta_init, _ = self.theta_encoder(z_beta_init)
        z_delta, prec_delta = self.delta_encoder(z_theta_init)

        # Compute top-down signals
        topdown_theta = self.delta_to_theta(z_delta)
        z_theta, prec_theta = self.theta_encoder(z_beta_init, topdown_theta)
        topdown_beta = self.theta_to_beta(z_theta)
        z_beta, prec_beta = self.beta_encoder(combined_init, topdown_beta)
        topdown_split = self.beta_to_split(z_beta)

        # Second pass: split encoder with top-down from z_beta
        # Returns (z_self, z_world, prec_self, prec_world, z_harm, event_logits)
        z_self, z_world, prec_self, prec_world, z_harm, event_logits = self.split_encoder(
            body_obs, world_obs, topdown=topdown_split
        )

        # SD-007: Reafference correction — subtract expected perspective shift from z_world.
        # Applied before EMA so the corrected signal is what gets smoothed.
        # Condition: predictor enabled, prev_action provided, not the very first step.
        z_world_raw = z_world  # always keep uncorrected copy for diagnostics / training
        if (
            self.reafference_predictor is not None
            and prev_action is not None
            and (prev_state.timestamp or 0) > 0
        ):
            # Ensure prev_action has batch dim
            a = prev_action
            if a.dim() == 1:
                a = a.unsqueeze(0).expand(batch_size, -1)
            # MECH-101: use z_world_raw_prev (not z_self_prev) as predictor input.
            # z_world_raw_prev encodes what was visible at t-1, which is needed to
            # predict cell content entering view during locomotion.
            z_world_raw_prev = (
                prev_state.z_world_raw
                if prev_state.z_world_raw is not None
                else prev_state.z_world
            )
            z_world = self.reafference_predictor.correct_z_world(
                z_world_raw, z_world_raw_prev, a
            )

        # Temporal smoothing (EMA).
        # SD-008: alpha_world / alpha_self are configurable. Default 0.3 preserves
        # backward compat; set alpha_world >= 0.9 to fix event suppression.
        alpha_self   = getattr(self.config, "alpha_self",  0.3)
        alpha_world  = getattr(self.config, "alpha_world", 0.3)
        alpha_shared = 0.3  # z_beta/theta/delta use a shared alpha (body + world integrated)
        z_self  = alpha_self  * z_self  + (1 - alpha_self)  * prev_state.z_self
        z_world = alpha_world * z_world + (1 - alpha_world) * prev_state.z_world

        # Unified latent ablation (EXQ-044): fuse z_self and z_world into a single
        # shared representation, eliminating channel specialization. Both channels
        # receive the average, so E2 and E3 gradients flow through the same dimensions.
        if getattr(self.config, "unified_latent_mode", False):
            z_unified = (z_self + z_world) * 0.5
            z_self = z_unified
            z_world = z_unified

        z_beta  = alpha_shared * z_beta  + (1 - alpha_shared) * prev_state.z_beta
        z_theta = alpha_shared * z_theta + (1 - alpha_shared) * prev_state.z_theta
        z_delta = alpha_shared * z_delta + (1 - alpha_shared) * prev_state.z_delta

        # SD-010: dedicated nociceptive stream — overrides MECH-099 lateral head z_harm.
        # HarmEncoder output is NOT perspective-corrected (spinothalamic analogue).
        if harm_obs is not None and self.harm_encoder is not None:
            ho = harm_obs.to(device).float()
            if ho.dim() == 1:
                ho = ho.unsqueeze(0).expand(batch_size, -1)
            z_harm = self.harm_encoder(ho)

        # SD-011: affective-motivational harm stream — NOT perspective-corrected.
        # AffectiveHarmEncoder encodes the EMA-accumulated proximity signal into z_harm_a.
        z_harm_a = None
        if harm_obs_a is not None and self.affective_harm_encoder is not None:
            hoa = harm_obs_a.to(device).float()
            if hoa.dim() == 1:
                hoa = hoa.unsqueeze(0).expand(batch_size, -1)
            z_harm_a = self.affective_harm_encoder(hoa)

        return LatentState(
            z_self=z_self,
            z_world=z_world,
            z_beta=z_beta,
            z_theta=z_theta,
            z_delta=z_delta,
            precision={
                "self": prec_self, "world": prec_world,
                "beta": prec_beta, "theta": prec_theta, "delta": prec_delta,
            },
            timestamp=(prev_state.timestamp or 0) + 1,
            z_harm=z_harm,         # SD-010 or MECH-099: None if neither enabled
            z_harm_a=z_harm_a,     # SD-011: None if affective stream not enabled
            z_world_raw=z_world_raw,  # SD-007 diagnostic (uncorrected z_world)
            event_logits=event_logits,  # SD-009: None if event classifier not enabled
        )

    def predict(self, state: LatentState) -> LatentState:
        """Predict next latent state (for trajectory rollouts and prediction errors)."""
        return LatentState(
            z_self=self.self_predictor(state.z_self),
            z_world=self.world_predictor(state.z_world),
            z_beta=self.beta_predictor(state.z_beta),
            z_theta=self.theta_predictor(state.z_theta),
            z_delta=self.delta_predictor(state.z_delta),
            precision=state.precision,
            timestamp=(state.timestamp or 0) + 1,
        )

    def compute_prediction_error(
        self,
        predicted: LatentState,
        actual: LatentState,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute precision-weighted prediction errors at each channel.

        Error routing in V3:
          self error  → E2 motor-sensory loss
          world error → E3 harm/goal loss (and E1 world prior)
          beta error  → E1 affective prediction
          theta/delta → E1 deep model
        """
        errors: Dict[str, torch.Tensor] = {}

        errors["self"]  = (actual.z_self  - predicted.z_self)  * actual.precision["self"]
        errors["world"] = (actual.z_world - predicted.z_world) * actual.precision["world"]
        errors["beta"]  = (actual.z_beta  - predicted.z_beta)  * actual.precision["beta"]
        errors["theta"] = (actual.z_theta - predicted.z_theta) * actual.precision["theta"]
        errors["delta"] = (actual.z_delta - predicted.z_delta) * actual.precision["delta"]

        errors["total"] = sum(e.pow(2).mean() for e in errors.values())
        return errors

    def modulate_precision(
        self,
        state: LatentState,
        channel: str,
        gain: float,
    ) -> LatentState:
        """Modulate precision at a specific channel (attention as gain)."""
        new_precision = state.precision.copy()
        new_precision[channel] = torch.clamp(
            state.precision[channel] * gain, min=0.01, max=1.0
        )
        return LatentState(
            z_self=state.z_self, z_world=state.z_world,
            z_beta=state.z_beta, z_theta=state.z_theta, z_delta=state.z_delta,
            precision=new_precision,
            timestamp=state.timestamp,
            hypothesis_tag=state.hypothesis_tag,
        )

    def forward(
        self,
        observation: torch.Tensor,
        prev_state: Optional[LatentState] = None,
        prev_action: Optional[torch.Tensor] = None,
    ) -> LatentState:
        return self.encode(observation, prev_state, prev_action)
