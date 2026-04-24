"""AnchorSet -- MECH-269 Phase 2 (ii) substrate.

Scale-tagged hippocampal anchor store with dual-trace preservation
(Bouton 2004) and per-anchor hysteresis on V_s_anchor (V_s minus a
local staleness proxy). Consumes MECH-288 BoundaryEvents to install
new active anchors per (scale, segment_id, stream_mixture) key and,
on remap, marks the outgoing anchor inactive rather than erasing it.

Design doc:  REE_assembly/docs/architecture/hippocampal_anchor_selection.md
Claim:       MECH-269 (docs/claims/claims.yaml)
Phase:       2 (ii) -- first-pass substrate. Key stand-ins documented
             in AnchorSetConfig docstring. Phase 3 replaces the local
             staleness proxy with MECH-284 and the tuple stream_mixture
             with a learned attribution head.

Key schema:
    AnchorKey = (scale: str, segment_id: str, stream_mixture: tuple[str, ...])

Stream mixture (Phase 2 stand-in):
    tuple(sorted(per_stream_vs.keys())) at anchor-creation tick.
    Learned attribution head deferred; this gives a deterministic,
    observable stream-membership signature sufficient for the first
    end-to-end validation.

Reset eligibility (hysteresis):
    V_s_anchor = V_s - staleness   (staleness monotonically increases
                                    per tick up to staleness_clip; reset
                                    to 0 on get_anchor / fresh write)
    If V_s_anchor < reset_threshold for hysteresis_k consecutive
    HippocampalModule ticks, mark_inactive fires on the active anchor
    for that (scale, stream_mixture) family and a new active anchor
    is installed for the incoming segment_id.

Dual-trace routing (Bouton 2004 / verdicts 2 of V_s foundation lit-pull):
    Old anchor -> mark_inactive, NOT erase. Retained in all_anchors()
    for retrieval / replay consumers; excluded from active_anchors().

MECH-094 gate:
    write_anchor is invoked only from HippocampalModule.tick_anchor_set,
    which is called from REEAgent.sense() (waking stream). Simulation
    / replay paths must not route through tick_anchor_set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ree_core.hippocampal.event_segmenter import BoundaryEvent
from ree_core.utils.config import AnchorSetConfig


AnchorKey = Tuple[str, str, Tuple[str, ...]]


@dataclass
class Anchor:
    """A single anchor record.

    Fields:
      key:            (scale, segment_id, stream_mixture) identifying tuple.
      z_world:        z_world snapshot at write time (detached clone).
      active:         True for the current live anchor on this
                      (scale, stream_mixture) family; False after
                      mark_inactive (dual-trace preserved).
      created_at:     HippocampalModule tick index at creation.
      last_accessed:  Tick index of last get_anchor / fresh write. Drives
                      the Phase 2 staleness proxy.
      below_threshold_streak: consecutive tick count of V_s_anchor
                      < reset_threshold (hysteresis counter). Reset on
                      any tick where V_s_anchor >= threshold.
    """
    key: AnchorKey
    z_world: torch.Tensor
    active: bool = True
    created_at: int = 0
    last_accessed: int = 0
    below_threshold_streak: int = 0


class AnchorSet:
    """Scale-tagged hippocampal anchor store (MECH-269 Phase 2 (ii))."""

    def __init__(self, config: AnchorSetConfig):
        self.config = config
        # Active index: (scale, stream_mixture) -> Anchor.
        # Only one active anchor per (scale, stream_mixture) family at a
        # time; incoming boundary installs a new one and the previous is
        # moved to inactive storage (dual-trace preserved).
        self._active: Dict[Tuple[str, Tuple[str, ...]], Anchor] = {}
        # Full index keyed by AnchorKey, containing both active and
        # inactive anchors. Preserved across remap (Bouton 2004).
        self._all: Dict[AnchorKey, Anchor] = {}
        # Count of active anchors per scale for the FIFO soft-cap.
        self._active_per_scale: Dict[str, int] = {}
        # Monotonic local tick counter; incremented by tick_hysteresis.
        self._tick: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def write_anchor(
        self,
        scale: str,
        segment_id: str,
        stream_mixture: Tuple[str, ...],
        z_world: torch.Tensor,
    ) -> Anchor:
        """Install a new active anchor for (scale, segment_id, stream_mixture).

        If an active anchor already exists on this (scale, stream_mixture)
        family, it is marked inactive first (Bouton 2004 dual-trace).
        Idempotent: if the same AnchorKey is written twice in a row,
        last_accessed is refreshed and the existing record is returned
        unchanged.
        """
        key: AnchorKey = (scale, segment_id, tuple(stream_mixture))
        existing = self._all.get(key)
        if existing is not None and existing.active:
            existing.last_accessed = self._tick
            return existing

        family = (scale, tuple(stream_mixture))
        prior_active = self._active.get(family)
        if prior_active is not None and prior_active.key != key:
            self._mark_inactive_internal(prior_active)

        z_snap = z_world.detach().clone()
        anchor = Anchor(
            key=key,
            z_world=z_snap,
            active=True,
            created_at=self._tick,
            last_accessed=self._tick,
            below_threshold_streak=0,
        )
        self._all[key] = anchor
        self._active[family] = anchor
        self._active_per_scale[scale] = self._active_per_scale.get(scale, 0) + 1
        self._enforce_scale_cap(scale)
        return anchor

    def get_anchor(
        self,
        scale: str,
        segment_id: str,
        stream_mixture: Tuple[str, ...],
    ) -> Optional[Anchor]:
        """Fetch an anchor by full key. Refreshes last_accessed when found."""
        key: AnchorKey = (scale, segment_id, tuple(stream_mixture))
        anchor = self._all.get(key)
        if anchor is not None:
            anchor.last_accessed = self._tick
        return anchor

    def mark_inactive(self, scale: str, stream_mixture: Tuple[str, ...]) -> Optional[Anchor]:
        """Mark the currently-active anchor on (scale, stream_mixture) inactive.

        Preserves the anchor in self._all (dual-trace). Returns the anchor
        that was deactivated, or None if no active anchor existed on that
        family. Used by reset_region and by the hysteresis reset path.
        """
        family = (scale, tuple(stream_mixture))
        anchor = self._active.get(family)
        if anchor is None:
            return None
        self._mark_inactive_internal(anchor)
        return anchor

    def reset_region(
        self,
        scale: str,
        stream_mixture: Tuple[str, ...],
        new_segment_id: str,
        z_world: torch.Tensor,
    ) -> Anchor:
        """Dual-trace remap: mark current active inactive; install new active.

        Consumed by MECH-287 broadcast-driven invalidation (Phase 3 wiring)
        and by the internal hysteresis reset path. The previous anchor is
        retained in all_anchors() as an inactive trace; the new anchor is
        written with the incoming segment_id.
        """
        self.mark_inactive(scale, stream_mixture)
        return self.write_anchor(scale, new_segment_id, stream_mixture, z_world)

    # ------------------------------------------------------------------ #
    # Per-tick hysteresis                                                #
    # ------------------------------------------------------------------ #
    def tick_hysteresis(
        self,
        per_stream_vs: Dict[str, float],
        staleness_lookup: Optional[Callable[[AnchorKey], float]] = None,
    ) -> List[Anchor]:
        """Advance staleness + hysteresis counters on all active anchors.

        For each active anchor:
          V_s_anchor = avg(V_s over anchor's stream_mixture) - staleness
          staleness  = min(staleness_clip, (tick - last_accessed) * staleness_rate)
        If V_s_anchor < reset_threshold increment below_threshold_streak;
        else reset it to 0. When the streak reaches hysteresis_k, the
        anchor is marked inactive and returned (caller installs the new
        active anchor via the boundary-event path).

        When staleness_lookup is provided (MECH-284 Phase 3 wiring), the
        per-anchor staleness is read from the supplied callable instead of
        the internal (tick - last_accessed) * staleness_rate proxy. The
        callable receives the anchor's full AnchorKey and returns a
        float in [0, staleness_clip]; values above staleness_clip are
        clamped for parity with the internal-proxy path.

        Returns: list of anchors that crossed the hysteresis threshold
        this tick (freshly marked inactive).
        """
        self._tick += 1
        fired: List[Anchor] = []
        cfg = self.config
        for anchor in list(self._active.values()):
            mixture = anchor.key[2]
            v_s_vals = [per_stream_vs.get(s) for s in mixture]
            v_s_vals = [v for v in v_s_vals if v is not None]
            if not v_s_vals:
                anchor.below_threshold_streak = 0
                continue
            avg_v_s = sum(v_s_vals) / len(v_s_vals)
            if staleness_lookup is not None:
                staleness = min(
                    cfg.staleness_clip,
                    float(staleness_lookup(anchor.key)),
                )
            else:
                staleness = min(
                    cfg.staleness_clip,
                    (self._tick - anchor.last_accessed) * cfg.staleness_rate,
                )
            v_s_anchor = avg_v_s - staleness
            if v_s_anchor < cfg.reset_threshold:
                anchor.below_threshold_streak += 1
            else:
                anchor.below_threshold_streak = 0
            if anchor.below_threshold_streak >= cfg.hysteresis_k:
                self._mark_inactive_internal(anchor)
                fired.append(anchor)
        return fired

    # ------------------------------------------------------------------ #
    # Boundary-event consumption                                         #
    # ------------------------------------------------------------------ #
    def consume_boundary_events(
        self,
        events: List[BoundaryEvent],
        z_world: Optional[torch.Tensor],
        stream_mixture: Tuple[str, ...],
    ) -> List[Anchor]:
        """Install / remap anchors for each BoundaryEvent whose scale is
        registered in config.scales.

        Phase 2 stand-in: stream_mixture is the tuple passed in by the
        caller (HippocampalModule builds it from per_stream_vs.keys()).
        Each BoundaryEvent triggers a write_anchor for (scale, segment_id_new,
        stream_mixture); any prior active on the (scale, stream_mixture)
        family is marked inactive by write_anchor's internal dual-trace.

        If z_world is None (e.g. no encoded state yet), the event is
        skipped. Returns the list of anchors that were installed.
        """
        if not self.config.subscribe_to_boundary_events:
            return []
        if z_world is None:
            return []
        installed: List[Anchor] = []
        scales = set(self.config.scales)
        for ev in events:
            if ev.scale not in scales:
                continue
            installed.append(
                self.write_anchor(
                    scale=ev.scale,
                    segment_id=ev.segment_id_new,
                    stream_mixture=stream_mixture,
                    z_world=z_world,
                )
            )
        return installed

    # ------------------------------------------------------------------ #
    # Query helpers                                                      #
    # ------------------------------------------------------------------ #
    def active_anchors(self, scale: Optional[str] = None) -> List[Anchor]:
        """Active anchors, optionally filtered by scale."""
        if scale is None:
            return list(self._active.values())
        return [a for a in self._active.values() if a.key[0] == scale]

    def all_anchors(self, scale: Optional[str] = None) -> List[Anchor]:
        """All anchors (active + inactive), optionally filtered by scale."""
        if scale is None:
            return list(self._all.values())
        return [a for a in self._all.values() if a.key[0] == scale]

    def reset(self) -> None:
        """Per-episode reset. Clears active + inactive anchors + tick."""
        self._active.clear()
        self._all.clear()
        self._active_per_scale.clear()
        self._tick = 0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _mark_inactive_internal(self, anchor: Anchor) -> None:
        if not anchor.active:
            return
        anchor.active = False
        family = (anchor.key[0], anchor.key[2])
        if self._active.get(family) is anchor:
            del self._active[family]
        self._active_per_scale[anchor.key[0]] = max(
            0, self._active_per_scale.get(anchor.key[0], 0) - 1
        )

    def _enforce_scale_cap(self, scale: str) -> None:
        cap = self.config.max_anchors_per_scale
        if cap <= 0:
            return
        if self._active_per_scale.get(scale, 0) <= cap:
            return
        # FIFO: oldest (smallest created_at) active anchor in this scale
        # gets marked inactive. Inactive anchors are preserved.
        scale_actives = sorted(
            [a for a in self._active.values() if a.key[0] == scale],
            key=lambda a: a.created_at,
        )
        while self._active_per_scale.get(scale, 0) > cap and scale_actives:
            oldest = scale_actives.pop(0)
            self._mark_inactive_internal(oldest)
