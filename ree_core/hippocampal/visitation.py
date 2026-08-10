"""Unconditional per-region visitation-count tracker
(chip-20260810-fishtank-visitation-count-telemetry).

REE_assembly/evidence/planning/developmental_ecology_curiosity_foraging_correction_2026-08-10.md
Section 6 names three untelemetered signals needed to discriminate curiosity-driven
discovery from diffuse-gradient exploitation: SD-025's novelty term (landed --
chip-20260810-fishtank-sd025-novelty-telemetry), MECH-314's score-bias (left alone;
conditional on a flag no current fishtank driver enables), and a state-visitation
count (this module).

VisitationCounter is deliberately NOT the same substrate as either existing
visitation-shaped structure in this codebase:

* `REEAgent._zworld_visitation_buffer` (ree_core/agent.py) -- a raw FIFO of visited
  z_world points, only allocated when `use_structured_curiosity` /
  `curiosity_novelty_source="visitation"` (MECH-314) is on. Off in every
  fishtank-family driver today.
* `FamiliarityTracker` (ree_core/hippocampal/curiosity.py) -- a proximity-weighted
  EMA in [0, 1], only allocated when `curiosity_weight` (SD-025) > 0. It answers
  "how familiar is this region", not "how many times has it been visited", and is
  compute/memory-free when curiosity is disabled -- exactly the property this
  module must NOT rely on, since it has to be on always.

VisitationCounter mirrors FamiliarityTracker's kernel shape (nearest-anchor
association within a fixed `bandwidth`, FIFO overwrite of a fixed-size anchor set
when no anchor is close enough) because that shape is already tuned to the z_world
scale (see HippocampalConfig.familiarity_bandwidth's V3-EXQ-786a sweep note), but
records an exact per-anchor visit COUNT rather than an EMA, and is instantiated
UNCONDITIONALLY -- no flag gates it. It carries no scoring role: query()/update()
are read/write-only for telemetry and are never consulted by CEM trajectory scoring
or action selection.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisitationCounter(nn.Module):
    """Proximity-bucketed visit COUNT over a fixed-size FIFO anchor buffer.

    query(z) -> int visit count (0 = never visited); update(z) increments the
    count for z (allocating a fresh anchor on first visit to a region). All
    state lives in registered buffers so device/dtype follow the parent module.
    Always under torch.no_grad -- never touches autograd.
    """

    def __init__(
        self,
        world_dim: int,
        num_anchors: int = 128,
        bandwidth: float = 0.20,
    ):
        super().__init__()
        self.world_dim = int(world_dim)
        self.num_anchors = int(num_anchors)
        self.bandwidth = float(bandwidth)
        self._next_idx = 0
        self.register_buffer("anchors", torch.zeros(self.num_anchors, self.world_dim))
        self.register_buffer("counts", torch.zeros(self.num_anchors, dtype=torch.long))
        self.register_buffer(
            "active_mask", torch.zeros(self.num_anchors, dtype=torch.bool)
        )

    @torch.no_grad()
    def _nearest_active(self, z: torch.Tensor):
        """Returns (index, dist_sq) of the nearest ACTIVE anchor to z, or
        (-1, inf) when no anchor is active yet."""
        if not bool(self.active_mask.any()):
            return -1, float("inf")
        diffs = z.unsqueeze(0) - self.anchors  # [K, world_dim]
        dist_sq = (diffs ** 2).sum(dim=-1)  # [K]
        dist_sq = torch.where(
            self.active_mask, dist_sq, torch.full_like(dist_sq, float("inf"))
        )
        idx = int(torch.argmin(dist_sq).item())
        return idx, float(dist_sq[idx].item())

    @torch.no_grad()
    def query(self, z: torch.Tensor) -> int:
        """Visit count at the nearest active anchor within `bandwidth` of z.

        Args:
            z: [world_dim] or [1, world_dim] single query point.

        Returns:
            0 when no active anchor is within bandwidth (never-visited region),
            else the integer visit count of the nearest one. Read-only: never
            allocates or mutates an anchor.
        """
        if z.dim() > 1:
            z = z.reshape(-1)
        idx, dist_sq = self._nearest_active(z)
        thresh_sq = self.bandwidth * self.bandwidth
        if idx >= 0 and dist_sq <= thresh_sq:
            return int(self.counts[idx].item())
        return 0

    @torch.no_grad()
    def update(self, z: torch.Tensor) -> int:
        """Increment the visit count for z, returning the PRE-increment count.

        Associates to the nearest active anchor within `bandwidth` (raising its
        count by 1), or allocates a fresh anchor (FIFO overwrite) seeded at
        count=1 when no anchor is close enough.

        Args:
            z: [world_dim] or [1, world_dim] visited point.

        Returns:
            The visit count BEFORE this update (0 for a first-ever visit to a
            region) -- mirrors compute_novelty_score's pre-visit-read
            convention so a caller does not need a separate query() call.
        """
        if z.dim() > 1:
            z = z.reshape(-1)
        zi = z.detach()
        idx, dist_sq = self._nearest_active(zi)
        thresh_sq = self.bandwidth * self.bandwidth
        if idx >= 0 and dist_sq <= thresh_sq:
            pre = int(self.counts[idx].item())
            self.counts[idx] += 1
            return pre
        alloc_idx = self._next_idx
        self.anchors[alloc_idx] = zi
        self.counts[alloc_idx] = 1
        self.active_mask[alloc_idx] = True
        self._next_idx = (self._next_idx + 1) % self.num_anchors
        return 0

    @torch.no_grad()
    def reset(self) -> None:
        """Clear all anchors and counts (not wired to episode reset by
        default -- mirrors FamiliarityTracker.reset(), which is likewise
        unwired; a caller wanting per-episode counts can invoke this
        explicitly between episodes)."""
        self.anchors.zero_()
        self.counts.zero_()
        self.active_mask.zero_()
        self._next_idx = 0
