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

    # MECH-187: gain multiplier on z_goal seeding signal.
    # Applied before drive modulation: effective_benefit = benefit_exposure * z_goal_seeding_gain * (...)
    # 1.0 = no change (default, fully backward compatible).
    # < 1.0 suppresses seeding (5-HT serotonergic inhibition of incentive salience).
    # > 1.0 elevates seeding (disinhibition or pharmacological augmentation).
    # Empirical range from Korte et al. 2016: suppression ~x0.6-0.8, elevation ~x1.5-2.5.
    z_goal_seeding_gain: float = 1.0

    # MECH-186: serotonergic benefit terrain maintenance (valence_wanting floor).
    # When > 0, the z_goal norm is prevented from decaying below this value.
    # Simulates tonic serotonergic support maintaining minimum wanting tone.
    # Default None/0.0 = disabled (no floor, backward-compatible).
    # Set to 0.05 for the MECH-186 floor-maintained condition.
    valence_wanting_floor: float = 0.0

    # MECH-188: PFC top-down z_goal injection (constant floor on effective z_goal norm
    # during action selection only -- does NOT modify the persistent z_goal attractor).
    # Simulates DRN-mPFC serotonergic top-down goal persistence (Miyazaki et al. 2020):
    # when terrain-based seeding has failed (LONG_HORIZON depression attractor), an
    # external PFC signal can maintain goal representation.
    # z_goal_inject=0.0 disables (default, fully backward compatible).
    # z_goal_inject=0.3 applies a constant norm floor of 0.3 to z_goal during
    # agent.select_action() -- does not affect update() or z_goal decay.
    # Used by EXQ-253 (condition B) to test whether top-down injection suffices to
    # maintain PLANNED/HABIT behavioral gap when bottom-up terrain seeding has collapsed.
    z_goal_inject: float = 0.0


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

        # MECH-186: valence_wanting floor -- prevent z_goal norm from dropping
        # below the floor value. Simulates tonic serotonergic benefit terrain
        # maintenance. Applied after decay, before any benefit-triggered update.
        # Disabled when valence_wanting_floor <= 0.0 (default).
        floor = getattr(self.config, "valence_wanting_floor", 0.0)
        if floor > 0.0:
            current_norm = self._z_goal.norm().item()
            if current_norm < floor and current_norm > 1e-9:
                # Scale up to floor norm while preserving direction
                self._z_goal = self._z_goal * (floor / current_norm)
            elif current_norm <= 1e-9 and floor > 0.0:
                # z_goal is zero vector: cannot preserve direction.
                # Floor clamp has no effect until first benefit contact seeds direction.
                pass

        # MECH-187: apply seeding gain before drive modulation
        # gain=1.0 (default) is identity -- fully backward compatible.
        # SD-012: scale benefit by drive level
        effective_benefit = benefit_exposure * self.config.z_goal_seeding_gain * (
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

    def with_injection(self, inject_norm: float) -> "GoalState":
        """
        MECH-188: Return a view of this GoalState with z_goal norm floored at inject_norm.

        Creates a lightweight wrapper that shares the same config but overrides
        _z_goal with a version that has a minimum L2 norm of inject_norm.
        Used by agent.select_action() when z_goal_inject > 0 -- applies to
        action selection only, does NOT modify the persistent attractor.

        If z_goal has no direction (norm=0), a constant unit vector is used so
        that goal_proximity still produces a non-trivial gradient for trajectory
        scoring. The first non-zero dimension (index 0) is set.

        Args:
            inject_norm: minimum L2 norm floor for the injected z_goal.

        Returns:
            A GoalState whose _z_goal has norm >= inject_norm.
        """
        injected = GoalState.__new__(GoalState)
        injected.config = self.config
        injected.device = self.device
        injected._goal_norm_peak = self._goal_norm_peak

        current_norm = self._z_goal.norm().item()
        if current_norm >= inject_norm:
            # Already above floor: no change
            injected._z_goal = self._z_goal
        elif current_norm > 1e-9:
            # Scale up to floor norm while preserving direction
            injected._z_goal = self._z_goal * (inject_norm / current_norm)
        else:
            # z_goal is zero: use first-dimension unit vector scaled to inject_norm
            z_seed = torch.zeros_like(self._z_goal)
            z_seed[0, 0] = inject_norm
            injected._z_goal = z_seed

        return injected

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
