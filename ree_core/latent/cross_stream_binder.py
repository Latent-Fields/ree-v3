"""Cross-stream binding factor (cross_stream_binding_substrate).

Installs a genuine shared latent cause between the z_self (`states`) and z_world
(`world_states`) rollout streams produced by E2FastPredictor.rollout_with_world.

WHY THIS EXISTS
---------------
failure_autopsy_V3-EXQ-641a_2026-06-06 adjudicated V3-EXQ-641a as a substrate
ceiling: with a fair contrast-matched control, cross-stream phase-alignment
C(tau) over the two rollout streams carried NO selection information beyond the
prediction-error cost E. The root cause is structural -- in rollout_with_world
the two streams advance through two INDEPENDENT forward models
(predict_next_self, world_forward) whose only shared input is the action. So any
co-variation the harness reads as "coherence" is exactly what E already scores.

This module makes the two streams genuinely bound: at each rollout step it
derives a shared factor from the JOINT (z_self, z_world) state and injects the
SAME additive perturbation into both post-transition states. The two streams'
step-deltas then share an explicit common component, so real cross-stream
coherence carries a per-candidate signature that a shuffle of the coherence
values destroys.

DESIGN NOTES
------------
- FIXED (untrained) projections. The 641a retest harness runs eval() with no P0
  curriculum for a new head, so a learned binder would be untrained-random and
  prove nothing. A fixed, joint-state-dependent shared field is the minimal
  substrate that installs a genuine common cause -- the ephaptic-field analog
  (MECH-270): a shared field co-located populations both feel, imposed
  structurally rather than learned. Phased training therefore does NOT apply to
  this substrate. A learned binder is a V4 extension, out of scope here.
- SAME perturbation into both streams. Two independent random projections of the
  shared factor would average to ~0 cross-stream alignment; adding the identical
  perturbation b_t into the first min(self_dim, world_dim) components of both
  streams guarantees a genuine shared delta-component.
- Theta-gated (MECH-089 theta-gamma nesting). The per-step shared code b_t is the
  gamma-rate content; the cosine theta window it is scaled by is the theta cycle
  it nests within.
- MECH-094 does NOT newly apply: no memory-write surface is added and
  hypothesis_tag semantics are unchanged.

See docs/architecture/sd_cross_stream_binding_substrate.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossStreamBinder(nn.Module):
    """Shared-latent-factor binding coupling for E2 rollouts.

    factor(z_self, z_world) -> g_t          shared factor from the joint state
    couple(z_self, z_world, g_t, t) -> (z_self', z_world')
        adds the SAME theta-gated perturbation b_t = W_out . g_t into the first
        min(self_dim, world_dim) components of both streams.

    Projections are fixed (no gradient is required for the substrate; the module
    is used under torch.no_grad-style eval in the retest). They are ordinary
    nn.Linear layers so state_dict save/load and .to(device) work normally and so
    initialisation is deterministic under the caller's seed.
    """

    def __init__(
        self,
        self_dim: int,
        world_dim: int,
        bind_dim: int = 16,
        strength: float = 0.15,
        theta_period: int = 4,
    ) -> None:
        super().__init__()
        self.self_dim = int(self_dim)
        self.world_dim = int(world_dim)
        self.bind_dim = int(bind_dim)
        self.strength = float(strength)
        # theta_period must be >= 1; a period of 1 degenerates the gate to a
        # constant 1.0 window (cos(2*pi*t) == 1), i.e. ungated coupling.
        self.theta_period = max(1, int(theta_period))
        # Shared output space = the overlap both streams can receive the SAME
        # perturbation in.
        self.bind_out_dim = min(self.self_dim, self.world_dim)

        # Joint state -> shared factor. Input is [z_self ; z_world].
        self.encode = nn.Linear(self.self_dim + self.world_dim, self.bind_dim)
        # Shared factor -> common perturbation in the overlap space.
        self.to_common = nn.Linear(self.bind_dim, self.bind_out_dim, bias=False)

    def factor(self, z_self: torch.Tensor, z_world: torch.Tensor) -> torch.Tensor:
        """Shared binding factor g_t from the joint pre-transition state.

        Args:
            z_self:  [batch, self_dim]
            z_world: [batch, world_dim]
        Returns:
            g_t: [batch, bind_dim]
        """
        joint = torch.cat([z_self, z_world], dim=-1)
        return torch.tanh(self.encode(joint))

    def _theta_gate(self, t: int) -> float:
        """MECH-089 theta window in [0, 1]: 0.5*(1 + cos(2*pi*t/theta_period))."""
        return 0.5 * (1.0 + math.cos(2.0 * math.pi * float(t) / float(self.theta_period)))

    def couple(
        self,
        z_self: torch.Tensor,
        z_world: torch.Tensor,
        g_t: torch.Tensor,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Inject the SAME theta-gated perturbation b_t into both streams.

        Args:
            z_self:  [batch, self_dim]  (post base-transition)
            z_world: [batch, world_dim] (post base-transition)
            g_t:     [batch, bind_dim]  from factor() on the pre-transition state
            t:       rollout step index (theta phase)
        Returns:
            (z_self', z_world') with b_t added into the first bind_out_dim dims.
        """
        k_t = self.strength * self._theta_gate(t)
        if k_t == 0.0:
            return z_self, z_world
        b_t = self.to_common(g_t)  # [batch, bind_out_dim]
        d = self.bind_out_dim
        # Out-of-place add on the overlap slice so autograd / downstream reads are
        # clean and the untouched tail dimensions pass through unchanged.
        z_self = torch.cat([z_self[..., :d] + k_t * b_t, z_self[..., d:]], dim=-1)
        z_world = torch.cat([z_world[..., :d] + k_t * b_t, z_world[..., d:]], dim=-1)
        return z_self, z_world
