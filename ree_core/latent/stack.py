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

from ree_core.utils.config import LatentStackConfig


@dataclass
class LatentState:
    """Complete latent state for V3 REE.

    SD-005: z_gamma replaced by z_self + z_world.

    Attributes:
        z_self:  Proprioceptive/interoceptive body-state  [batch, self_dim]
        z_world: Exteroceptive world-state                [batch, world_dim]
        z_beta:  Affective latent (shared)                [batch, beta_dim]
        z_theta: Sequence context (shared)                [batch, theta_dim]
        z_delta: Regime/motivation (shared)               [batch, delta_dim]
        precision: Per-channel precision values
        timestamp: Optional timestep index
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

    def to_tensor(self) -> torch.Tensor:
        """Concatenate all channels into a single tensor."""
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
        )


class SplitEncoder(nn.Module):
    """SD-005 split encoder: body obs → z_self, world obs → z_world.

    Both streams receive top-down conditioning from the shared z_beta stack.
    """

    def __init__(
        self,
        body_obs_dim: int,
        world_obs_dim: int,
        self_dim: int,
        world_dim: int,
        topdown_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.self_dim = self_dim
        self.world_dim = world_dim

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode body and world observations into z_self and z_world.

        Returns: (z_self, z_world, prec_self, prec_world)
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

        return z_self, z_world, prec_self, prec_world


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

        # SD-005: split bottom encoder
        self.split_encoder = SplitEncoder(
            body_obs_dim=self.config.body_obs_dim,
            world_obs_dim=self.config.world_obs_dim,
            self_dim=self.config.self_dim,
            world_dim=self.config.world_dim,
            topdown_dim=self.config.topdown_dim,
            hidden_dim=hidden,
        )

        combined_dim = self.config.self_dim + self.config.world_dim

        # Shared depth stack
        self.beta_encoder = SharedDepthEncoder(
            input_dim=combined_dim,
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
    ) -> LatentState:
        """
        Encode observation into latent state.

        Implements bottom-up pass with top-down conditioning:
        1. Split observation into body/world channels
        2. Encode z_self, z_world with top-down conditioning
        3. Encode shared stack (beta, theta, delta)
        4. Temporal smoothing with previous state

        Args:
            observation: Raw observation [batch, body_obs_dim + world_obs_dim]
            prev_state: Previous latent state for temporal continuity

        Returns:
            New LatentState
        """
        batch_size = observation.shape[0]
        device = observation.device

        if prev_state is None:
            prev_state = self.init_state(batch_size, device)

        body_obs, world_obs = self._split_observation(observation)

        # First pass: no top-down (to get initial estimates for top-down computation)
        z_self_init, z_world_init, _, _ = self.split_encoder(body_obs, world_obs)
        combined_init = torch.cat([z_self_init, z_world_init], dim=-1)
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
        z_self, z_world, prec_self, prec_world = self.split_encoder(
            body_obs, world_obs, topdown=topdown_split
        )

        # Temporal smoothing (EMA)
        alpha = 0.3
        z_self  = alpha * z_self  + (1 - alpha) * prev_state.z_self
        z_world = alpha * z_world + (1 - alpha) * prev_state.z_world
        z_beta  = alpha * z_beta  + (1 - alpha) * prev_state.z_beta
        z_theta = alpha * z_theta + (1 - alpha) * prev_state.z_theta
        z_delta = alpha * z_delta + (1 - alpha) * prev_state.z_delta

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
    ) -> LatentState:
        return self.encode(observation, prev_state)
