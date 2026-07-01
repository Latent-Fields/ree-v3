"""SELF-1 / DR-13 (self_model_v4): dedicated z_self temporal-depth recurrence.

Turns z_self from an instantaneous body-state snapshot (the V3 single-MLP +
fixed-alpha EMA) into a STATEFUL temporal self-model. This is the substrate
floor of the self_model_v4 plan: DR-10 (z_self in E3 viability), DR-11
(z_self-domain goals), DR-12 (E2-PE -> E3 confidence), and the INV-064
maturational-stability gate all attach to a stable, inspectable, lesionable
self subject -- which a free-running shared latent cannot provide.

Mechanism (HYBRID, resolved on ARC-081 notes 2026-06-14):
  * a LIGHT DEDICATED self-recurrence -- a gated recurrent cell (GRUCell) whose
    hidden state is the previous stateful z_self and whose input is the current
    instantaneous encoded z_self. The gate lets the self-state selectively
    retain history a fixed-alpha EMA cannot, and the module is explicit /
    inspectable / lesionable / perturbation-isolated (its hidden input is
    z_self alone, so an experiment can perturb the self subject without
    touching z_world);
  * REGULARISED / ANCHORED by E1 generative feedback -- the recurrent output is
    blended toward E1's generative prediction of z_self (supplied by the agent
    from the cached E1 predicted-next z_self), so the self-latent stays
    consistent with the E-stream generative account of the body rather than
    drifting into a parallel self-model. The blend weight is the recorded
    residual tunable (light default preserves the stability-isolation benefit).

The E1-anchor BLEND lives at the encode call site (LatentStack.encode); this
module owns only the dedicated-recurrence half. See
REE_assembly/docs/architecture/dr13_self_recurrence_temporal_depth.md.

ML/AI parallel (engineering counsel only, not architectural authority): a gated
recurrent cell is the standard fix for a fixed-decay integrator that cannot
selectively retain (the EMA); GRU gates solve vanishing/over-smoothing at low
dimension. The E1-anchor blend is a hard analog of posterior<->prior
consistency (RSSM / world-models), but grounded neuroscientifically (the self
must track the E-stream body account), not as a KL/ELBO term.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SelfRecurrenceCell(nn.Module):
    """Gated temporal integrator for z_self (DR-13 dedicated self-recurrence).

    A thin wrapper over ``nn.GRUCell`` sized (self_dim -> self_dim): the hidden
    state carries the stateful z_self across ticks; the input is the current
    instantaneous encoded z_self. Kept deliberately minimal (one gated cell, no
    extra hidden dim) per the "light dedicated recurrence" design and the REE
    small-MLP convention.
    """

    def __init__(self, self_dim: int):
        super().__init__()
        self.self_dim = int(self_dim)
        # Gated cell: input = instantaneous z_self, hidden = previous stateful z_self.
        self.cell = nn.GRUCell(self.self_dim, self.self_dim)

    def forward(
        self,
        z_self_instant: torch.Tensor,
        z_self_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the stateful self-latent by one tick.

        Args:
            z_self_instant: current instantaneous encoded z_self [batch, self_dim]
                            (post top-down, post precision -- the value the EMA
                            would have smoothed).
            z_self_prev:    previous stateful z_self [batch, self_dim] (the
                            recurrent hidden state; init-zeros on the first step).

        Returns:
            z_self_next: the new stateful z_self [batch, self_dim].
        """
        # GRUCell expects (input, hidden); both [batch, self_dim]. z_self_prev is
        # the carried self-state -- perturbation-isolated from z_world by
        # construction (only z_self flows through this cell).
        return self.cell(z_self_instant, z_self_prev)
