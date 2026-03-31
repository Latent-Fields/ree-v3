"""
E2 Fast Predictor — V3 Implementation

V3 extensions (SD-004, SD-005):

SD-005 — Self/World latent split:
  forward()       — motor-sensory: z_self_t + a → z_self_{t+1}   (E2 primary domain)
  world_forward() — world-state:   z_world_t + a → z_world_{t+1} (for SD-003 attribution)

SD-004 — Action objects:
  action_object() — compressed world-effect: (z_world_t, a_t) → o_t
  o_t is a low-dimensional representation of the world-effect of taking
  action a_t from z_world_t. HippocampalModule builds its map over
  action-object space O, not raw z_world.

SD-003 V3 attribution pipeline:
    z_world_actual = e2.world_forward(z_world, a_actual)
    z_world_cf     = e2.world_forward(z_world, a_cf)
    harm_actual    = e3.harm_eval(z_world_actual)
    harm_cf        = e3.harm_eval(z_world_cf)
    causal_sig     = harm_actual - harm_cf
    world_delta    = ||z_world_actual - z_world_cf||

E2 trains ONLY on motor-sensory prediction error (z_self). NOT harm/goal error.
harm_predict head from V2 is REMOVED — harm evaluation belongs to E3.

rollout_horizon must exceed E1.prediction_horizon (30 > 20).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ree_core.utils.config import E2Config
from ree_core.latent.stack import LatentState


@dataclass
class Trajectory:
    """Candidate trajectory through latent space.

    V3: states are z_self sequences (E2 primary domain).
    world_states (if present) are z_world projections for SD-003 use.
    action_objects (if present) are o_t sequences for HippocampalModule.
    harm_predictions removed — E3 handles harm evaluation.
    """
    states: List[torch.Tensor]           # z_self: List of [batch, self_dim]
    actions: torch.Tensor                # [batch, horizon, action_dim]
    world_states: Optional[List[torch.Tensor]] = None  # z_world: for SD-003 attribution
    action_objects: Optional[List[torch.Tensor]] = None  # o_t: for HippocampalModule

    @property
    def total_length(self) -> int:
        return len(self.states)

    def get_final_state(self) -> torch.Tensor:
        return self.states[-1]

    def get_state_sequence(self) -> torch.Tensor:
        """Stack z_self states [batch, horizon+1, self_dim]."""
        return torch.stack(self.states, dim=1)

    def get_world_state_sequence(self) -> Optional[torch.Tensor]:
        """Stack z_world states [batch, horizon+1, world_dim]. None if not tracked."""
        if self.world_states is None:
            return None
        return torch.stack(self.world_states, dim=1)

    def get_action_object_sequence(self) -> Optional[torch.Tensor]:
        """Stack action objects [batch, horizon, action_object_dim]. None if not computed."""
        if self.action_objects is None:
            return None
        return torch.stack(self.action_objects, dim=1)



class E2FastPredictor(nn.Module):
    """
    E2 Fast Predictor — V3.

    Primary role: fast motor-sensory transition model over z_self.

    Extended interface (SD-004, SD-005):
      forward(z_self, a)          → z_self_next  (motor-sensory prediction)
      world_forward(z_world, a)   → z_world_next (world prediction for attribution)
      action_object(z_world, a)   → o_t          (compressed world-effect, SD-004)
      forward_counterfactual(z_self, a_cf) → z_self_cf (SD-003 self-attr)

    NOT a harm predictor. harm_eval() belongs on E3Selector.
    """

    def __init__(self, config: Optional[E2Config] = None):
        super().__init__()
        self.config = config or E2Config()

        # --- Primary: motor-sensory transition model (z_self domain) ---
        # z_self_{t+1} = z_self_t + delta(z_self_t, a_t)
        self.self_transition = nn.Sequential(
            nn.Linear(self.config.self_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.self_dim),
        )
        self.self_action_encoder = nn.Linear(
            self.config.action_dim, self.config.action_dim
        )

        # --- SD-005: world_forward (z_world domain, for attribution only) ---
        # z_world_{t+1} = z_world_t + delta_w(z_world_t, a_t)
        # Lightweight head — may share hidden layers in future iterations.
        self.world_transition = nn.Sequential(
            nn.Linear(self.config.world_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.world_dim),
        )
        self.world_action_encoder = nn.Linear(
            self.config.action_dim, self.config.action_dim
        )

        # --- SD-004: action_object head ---
        # Compressed representation of (z_world_t, a_t) world-effect.
        # Bottleneck ensures action_object_dim << world_dim.
        self.action_object_head = nn.Sequential(
            nn.Linear(self.config.world_dim + self.config.action_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.action_object_dim),
        )

    # ------------------------------------------------------------------ #
    # Core interface                                                       #
    # ------------------------------------------------------------------ #

    def predict_next_self(
        self,
        z_self: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Motor-sensory prediction: z_self_t + a_t → z_self_{t+1}.

        This is E2's primary operation. Residual connection.

        Args:
            z_self:  [batch, self_dim]
            action:  [batch, action_dim]

        Returns:
            z_self_next [batch, self_dim]
        """
        a_enc = self.self_action_encoder(action)
        z_a = torch.cat([z_self, a_enc], dim=-1)
        delta = self.self_transition(z_a)
        return z_self + delta

    def world_forward(
        self,
        z_world: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        World-state prediction: z_world_t + a_t → z_world_{t+1}.

        Used in SD-003 V3 attribution pipeline only.
        NOT trained as E2's primary objective.

        Args:
            z_world: [batch, world_dim]
            action:  [batch, action_dim]

        Returns:
            z_world_next [batch, world_dim]
        """
        a_enc = self.world_action_encoder(action)
        z_a = torch.cat([z_world, a_enc], dim=-1)
        delta = self.world_transition(z_a)
        return z_world + delta

    def action_object(
        self,
        z_world: torch.Tensor,
        action: torch.Tensor,
        action_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Produce compressed world-effect action object o_t (SD-004).

        o_t = E2.action_object(z_world_t, a_t) encodes the world-effect
        of taking action a_t from world-state z_world_t.

        HippocampalModule builds its trajectory map in action-object space O.
        Planning horizon extends because action-object space is lower-dimensional
        and semantically grounded.

        SD-016 (MECH-151): optional action_bias [batch, action_object_dim] from
        E1.extract_cue_context(). When provided, added to o_t AFTER the base
        computation. This shifts the apparent affordance of each action-object
        by a cue-indexed contextual signal without replacing the base output.
        When None (all existing callers), behaviour is unchanged.

        Args:
            z_world:     [batch, world_dim]
            action:      [batch, action_dim]
            action_bias: [batch, action_object_dim] or None (SD-016)

        Returns:
            o_t [batch, action_object_dim]
        """
        z_a = torch.cat([z_world, action], dim=-1)
        o_t = self.action_object_head(z_a)
        if action_bias is not None:
            o_t = o_t + action_bias
        return o_t

    def forward_counterfactual(
        self,
        z_self: torch.Tensor,
        counterfactual_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Counterfactual query: what z_self under a different action?

        Used in the old z_gamma-level SD-003 (V2 style).
        In V3, counterfactual attribution uses world_forward() instead.

        Args:
            z_self: [batch, self_dim]
            counterfactual_action: [batch, action_dim]

        Returns:
            z_self_cf [batch, self_dim]
        """
        return self.predict_next_self(z_self, counterfactual_action)

    # ------------------------------------------------------------------ #
    # Rollout and candidate generation                                     #
    # ------------------------------------------------------------------ #

    def rollout_self(
        self,
        initial_z_self: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> "Trajectory":
        """
        Roll out z_self trajectory given an action sequence.

        Args:
            initial_z_self: [batch, self_dim]
            action_sequence: [batch, horizon, action_dim]

        Returns:
            Trajectory (z_self states only)
        """
        horizon = action_sequence.shape[1]
        states = [initial_z_self]
        current = initial_z_self

        for t in range(horizon):
            action = action_sequence[:, t, :]
            current = self.predict_next_self(current, action)
            states.append(current)

        return Trajectory(states=states, actions=action_sequence)

    def rollout_with_world(
        self,
        initial_z_self: torch.Tensor,
        initial_z_world: torch.Tensor,
        action_sequence: torch.Tensor,
        compute_action_objects: bool = True,
        action_bias: Optional[torch.Tensor] = None,
    ) -> "Trajectory":
        """
        Roll out trajectory tracking both z_self and z_world.

        Used by HippocampalModule for action-object proposals and
        by SD-003 attribution pipeline.

        SD-016 (MECH-151): optional action_bias [batch, action_object_dim] from
        E1.extract_cue_context(). Applied to each action_object() call within
        the rollout. When None (all existing callers), behaviour is unchanged.

        Args:
            initial_z_self:        [batch, self_dim]
            initial_z_world:       [batch, world_dim]
            action_sequence:       [batch, horizon, action_dim]
            compute_action_objects: whether to compute action objects (SD-004)
            action_bias:           [batch, action_object_dim] or None (SD-016)

        Returns:
            Trajectory with states (z_self), world_states (z_world),
            and optionally action_objects (o_t)
        """
        horizon = action_sequence.shape[1]
        states = [initial_z_self]
        world_states = [initial_z_world]
        action_objects = []

        z_self  = initial_z_self
        z_world = initial_z_world

        for t in range(horizon):
            action = action_sequence[:, t, :]

            if compute_action_objects:
                o_t = self.action_object(z_world, action, action_bias=action_bias)
                action_objects.append(o_t)

            z_self  = self.predict_next_self(z_self, action)
            z_world = self.world_forward(z_world, action)

            states.append(z_self)
            world_states.append(z_world)

        return Trajectory(
            states=states,
            actions=action_sequence,
            world_states=world_states,
            action_objects=action_objects if compute_action_objects else None,
        )

    def generate_random_actions(
        self,
        batch_size: int,
        horizon: int,
        device: torch.device,
    ) -> torch.Tensor:
        return torch.randn(batch_size, horizon, self.config.action_dim, device=device)

    def generate_candidates_random(
        self,
        initial_z_self: torch.Tensor,
        initial_z_world: Optional[torch.Tensor] = None,
        num_candidates: Optional[int] = None,
        horizon: Optional[int] = None,
        compute_action_objects: bool = True,
        action_bias: Optional[torch.Tensor] = None,
    ) -> List["Trajectory"]:
        """
        Generate candidate trajectories using random shooting.

        Used by HippocampalModule during CEM iterations.

        SD-016 (MECH-151): optional action_bias [batch, action_object_dim] passed
        through to each rollout's action_object() calls.

        Args:
            initial_z_self:        [batch, self_dim]
            initial_z_world:       [batch, world_dim] (optional; needed for SD-004 objects)
            num_candidates:        number of trajectories
            horizon:               rollout steps
            compute_action_objects: whether to compute action objects
            action_bias:           [batch, action_object_dim] or None (SD-016)

        Returns:
            List of Trajectory objects
        """
        n = num_candidates or self.config.num_candidates
        h = horizon or self.config.rollout_horizon
        device = initial_z_self.device
        batch_size = initial_z_self.shape[0]

        candidates = []
        for _ in range(n):
            actions = self.generate_random_actions(batch_size, h, device)
            if initial_z_world is not None:
                traj = self.rollout_with_world(
                    initial_z_self, initial_z_world, actions,
                    compute_action_objects, action_bias=action_bias,
                )
            else:
                traj = self.rollout_self(initial_z_self, actions)
            candidates.append(traj)

        return candidates

    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compatibility shim: maps V2 API to V3.

        In V2 this operated on z_gamma. In V3, treats input as z_self.
        For new code, use predict_next_self() directly.
        """
        return self.predict_next_self(current_state, action)

    def forward(
        self,
        initial_z_self: torch.Tensor,
        num_candidates: Optional[int] = None,
    ) -> List["Trajectory"]:
        """Forward pass: generate random candidate trajectories."""
        return self.generate_candidates_random(
            initial_z_self, num_candidates=num_candidates, compute_action_objects=False
        )
