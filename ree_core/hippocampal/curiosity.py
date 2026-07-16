"""SD-025 (ARC-057 component 2): curiosity-drive familiarity tracking.

The curiosity drive biases hippocampal CEM trajectory scoring toward regions of
higher REPRESENTATIONAL DENSITY in the SD-024 benefit RBF map:

    novelty(z) = density(z) * (1 - familiarity(z))

`density(z)` is supplied by HippocampalModule.compute_representational_density
(the SD-024 weight-independent active-center count). `familiarity(z)` is supplied
by the FamiliarityTracker in this module: a visit-count EMA over a small FIFO set
of z_world anchors that rises toward 1 as a region is revisited, so novelty decays
there and the agent does not endlessly circle already-explored structure.

Design notes
------------
* WEIGHT-INDEPENDENT of the RBF benefit values -- familiarity is purely spatial
  (proximity to previously visited z_world), matching the density signal it
  multiplies. Together they measure "unexplored representational structure".
* Bounded: query() returns familiarity in [0, 1]; a fresh (empty) tracker returns
  0 everywhere -> all regions maximally novel.
* MECH-094: update() must be called on WAKING visits only. Replay / simulation /
  CEM-internal rollouts must NEVER advance familiarity (they write no real memory).
  The gating lives at the call site (HippocampalModule.update_familiarity / agent).

count-based novelty parallel (ML, engineering counsel only): visit-count
exploration bonuses (Bellemare 2016 pseudo-counts; Strehl & Littman MBIE-EB) share
the "novelty bonus that never decays -> perseveration" hazard; the (1 - familiarity)
EMA discount is the standard mitigation. Here novelty rides representational density
(RBF centers), not raw state-visit counts, and the discount is an EMA, not exact counts.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class FamiliarityTracker(nn.Module):
    """Proximity-weighted visit-count EMA over a FIFO buffer of z_world anchors.

    query(z) -> familiarity in [0, 1]; update(z) advances the EMA on a real visit.
    All state is held in registered buffers so device / dtype follow the parent
    module. Never touches autograd (all ops under no_grad).
    """

    def __init__(
        self,
        world_dim: int,
        num_anchors: int = 128,
        ema_alpha: float = 0.01,
        bandwidth: float = 1.0,
    ):
        super().__init__()
        self.world_dim = int(world_dim)
        self.num_anchors = int(num_anchors)
        self.ema_alpha = float(ema_alpha)
        self.bandwidth = float(bandwidth)
        self._next_idx = 0
        self.register_buffer("anchors", torch.zeros(self.num_anchors, self.world_dim))
        self.register_buffer("familiarity", torch.zeros(self.num_anchors))
        self.register_buffer(
            "active_mask", torch.zeros(self.num_anchors, dtype=torch.bool)
        )

    def _two_bw_sq(self, bandwidth: Optional[float]) -> float:
        bw = self.bandwidth if bandwidth is None else float(bandwidth)
        return 2.0 * (bw * bw)

    @torch.no_grad()
    def query(
        self, z: torch.Tensor, bandwidth: Optional[float] = None
    ) -> torch.Tensor:
        """Proximity-weighted familiarity in [0, 1] at each query point.

        Args:
            z:         [batch, world_dim] query points.
            bandwidth: optional scalar override for the proximity kernel.

        Returns:
            familiarity [batch] in [0, 1]. Zeros when no anchors are active.

        Uses a proximity-weighted SUM (a soft visitation kernel density), clamped
        to [0, 1] -- NOT a normalised average. Normalising by the weight sum would
        cancel the proximity term for a lone dominant anchor, making familiarity
        distance-independent; the clamped sum instead decays to 0 far from any
        visited anchor (everything distant stays novel) and saturates at 1 where
        visitation is dense.
        """
        batch = z.shape[0]
        if not bool(self.active_mask.any()):
            return torch.zeros(batch, device=z.device, dtype=z.dtype)
        diffs = z.unsqueeze(1) - self.anchors.unsqueeze(0)      # [batch, K, world_dim]
        dist_sq = (diffs ** 2).sum(dim=-1)                       # [batch, K]
        weights = torch.exp(-dist_sq / self._two_bw_sq(bandwidth))
        weights = weights * self.active_mask.to(weights.dtype).unsqueeze(0)
        fam = (weights * self.familiarity.unsqueeze(0)).sum(dim=-1)   # [batch]
        return fam.clamp_(0.0, 1.0).to(z.dtype)

    @torch.no_grad()
    def update(self, z: torch.Tensor, bandwidth: Optional[float] = None) -> None:
        """Advance the familiarity EMA at a visited z_world (WAKING only -- MECH-094).

        For each visited point: if the nearest ACTIVE anchor is within bandwidth,
        EMA-raise its familiarity toward 1 and nudge the anchor toward z; otherwise
        allocate a fresh anchor at z (FIFO overwrite) seeded at familiarity=alpha.

        Args:
            z:         [world_dim] or [batch, world_dim] visited point(s).
            bandwidth: optional scalar override for the association kernel.
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        alpha = self.ema_alpha
        bw = self.bandwidth if bandwidth is None else float(bandwidth)
        thresh_sq = bw * bw  # associate to nearest anchor within ~1 bandwidth
        for i in range(z.shape[0]):
            zi = z[i].detach()
            if bool(self.active_mask.any()):
                diffs = zi.unsqueeze(0) - self.anchors            # [K, world_dim]
                dist_sq = (diffs ** 2).sum(dim=-1)                # [K]
                dist_sq = torch.where(
                    self.active_mask, dist_sq, torch.full_like(dist_sq, float("inf"))
                )
                nearest = int(torch.argmin(dist_sq).item())
                nearest_d = float(dist_sq[nearest].item())
            else:
                nearest, nearest_d = -1, float("inf")
            if nearest >= 0 and nearest_d <= thresh_sq:
                # Revisit: raise familiarity toward 1, drift anchor toward z.
                self.familiarity[nearest] += alpha * (1.0 - self.familiarity[nearest])
                self.anchors[nearest] += alpha * (zi - self.anchors[nearest])
            else:
                # Novel region: allocate a fresh anchor (FIFO).
                idx = self._next_idx
                self.anchors[idx] = zi
                self.familiarity[idx] = alpha
                self.active_mask[idx] = True
                self._next_idx = (self._next_idx + 1) % self.num_anchors

    @torch.no_grad()
    def reset(self) -> None:
        """Clear all anchors (e.g. new lineage / fresh episode block if desired)."""
        self.anchors.zero_()
        self.familiarity.zero_()
        self.active_mask.zero_()
        self._next_idx = 0
