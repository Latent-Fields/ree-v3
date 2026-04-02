"""
Goal state representation for REE V3.

Implements z_goal_latent: a slow-decay attractor in z_world space, updated
when benefit fires (MECH-112 wanting), maintained by E1 LSTM recurrently
(MECH-116 frontal working memory).

Claims: MECH-112, MECH-116, MECH-117, ARC-032
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class GoalConfig:
    """Configuration for z_goal persistent goal representation."""

    # Must match world_dim in LatentStackConfig
    goal_dim: int = 32

    # Slow attractor update rate when benefit fires
    # half-life ~ 1/alpha_goal reward-contact steps
    alpha_goal: float = 0.05

    # Very slow decay between benefit events
    # half-life ~ log(2)/decay_goal ~ 139 steps at 0.005
    decay_goal: float = 0.005

    # Minimum benefit_exposure to trigger z_goal update
    benefit_threshold: float = 0.1

    # Weight of goal proximity in trajectory scoring (subtracted from cost)
    goal_weight: float = 1.0

    # SD-012: drive modulation weight for z_goal update.
    # effective_benefit = benefit_exposure * (1.0 + drive_weight * drive_level)
    # drive_level = 1.0 - energy (obs_body[3]).
    # 0.0 disables drive modulation; 2.0 is the validated default for goal seeding.
    # Set to 0.0 explicitly for ablation baselines.
    drive_weight: float = 2.0

    # Whether E1 receives z_goal as conditioning input (MECH-116)
    e1_goal_conditioned: bool = True

    # Master switch -- disabled by default (ablation baseline)
    z_goal_enabled: bool = False


class GoalState:
    """
    Persistent goal representation in z_world latent space.

    z_goal_latent is a slow-decay attractor: pulled toward current z_world
    when benefit fires (alpha_goal update), decaying toward zero otherwise.
    E1's LSTM counteracts decay by maintaining goal context recurrently.

    goal_proximity = 1 / (1 + MSE(z_world, z_goal)) -- bounded [0,1].
    Higher = closer to goal (the wanting signal for trajectory scoring).
    """

    def __init__(self, config: GoalConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self._z_goal: torch.Tensor = torch.zeros(
            1, config.goal_dim, device=device
        )
        self._goal_norm_peak: float = 0.0

    @property
    def z_goal(self) -> torch.Tensor:
        """Current goal latent. Shape: [1, goal_dim]."""
        return self._z_goal

    def update(
        self,
        z_world_current: torch.Tensor,
        benefit_exposure: float,
        drive_level: float = 1.0,
    ) -> None:
        """
        Update z_goal from current world state and benefit signal.

        Always decays. Pulls toward z_world if drive-scaled benefit > threshold.

        Args:
            z_world_current: [batch, world_dim]
            benefit_exposure: scalar benefit this step (body_state[11])
            drive_level: homeostatic drive 0=sated, 1=depleted (SD-012).
                         effective_benefit = benefit_exposure * (1 + drive_weight * drive_level).
                         Default 1.0 for backward compat when drive_weight=0.
        """
        # Always decay toward zero
        self._z_goal = self._z_goal * (1.0 - self.config.decay_goal)

        # SD-012: scale benefit by drive level
        effective_benefit = benefit_exposure * (
            1.0 + self.config.drive_weight * drive_level
        )

        # Pull toward current z_world if effective benefit fires
        if effective_benefit > self.config.benefit_threshold:
            z_w = z_world_current.detach()
            if z_w.dim() == 2:
                z_w = z_w.mean(dim=0, keepdim=True)
            self._z_goal = (
                (1.0 - self.config.alpha_goal) * self._z_goal
                + self.config.alpha_goal * z_w
            )
            norm = self._z_goal.norm().item()
            if norm > self._goal_norm_peak:
                self._goal_norm_peak = norm

    def goal_proximity(self, z_world: torch.Tensor) -> torch.Tensor:
        """
        Bounded wanting signal. Higher = closer to goal.
        Returns 1 / (1 + MSE_sum(z_world, z_goal)). Shape: [batch].
        """
        z_goal_exp = self._z_goal.expand_as(z_world)
        dist = F.mse_loss(z_world, z_goal_exp, reduction="none").sum(dim=-1)
        return 1.0 / (1.0 + dist)

    def goal_distance(self, z_world: torch.Tensor) -> torch.Tensor:
        """Raw MSE distance from goal. Lower = closer. Shape: [batch]."""
        z_goal_exp = self._z_goal.expand_as(z_world)
        return F.mse_loss(z_world, z_goal_exp, reduction="none").sum(dim=-1)

    def is_active(self) -> bool:
        """True if z_goal has been updated at least once."""
        return self._z_goal.abs().sum().item() > 1e-6

    def goal_norm(self) -> float:
        """L2 norm of current z_goal."""
        return float(self._z_goal.norm().item())

    def reset(self) -> None:
        """Reset goal to zero."""
        self._z_goal = torch.zeros_like(self._z_goal)
        self._goal_norm_peak = 0.0

    def state_dict(self) -> dict:
        return {
            "z_goal": self._z_goal.cpu(),
            "goal_norm_peak": self._goal_norm_peak,
        }

    def load_state_dict(self, d: dict) -> None:
        self._z_goal = d["z_goal"].to(self.device)
        self._goal_norm_peak = float(d.get("goal_norm_peak", 0.0))
