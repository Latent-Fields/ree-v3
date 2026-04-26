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
import torch.nn.functional as F

from ree_core.hippocampal.event_segmenter import BoundaryEvent
from ree_core.utils.config import AnchorSetConfig


AnchorKey = Tuple[str, str, Tuple[str, ...]]


@dataclass
class AnchorGoalPayload:
    """SD-039 dual-trace anchor goal-snapshot payload.

    Compact motivational payload attached to each anchor at write, remap, or
    invalidation time. Preserved across mark_inactive so inactive anchors
    remain queryable as blocked or deferred goal traces. The minimum
    substrate change that lets MECH-292 (ranked ghost-goal bank) and
    MECH-293 (waking ghost-goal probes) distinguish a still-wanted blocked
    path from generic unresolved terrain.

    Fields:
      z_goal_snapshot:    Detached clone of z_goal at payload-write time.
                          None when no goal was active. [1, goal_dim] when set.
      wanting_strength:   VALENCE_WANTING readout at the anchor's z_world
                          location (or last cached drive*benefit proxy).
                          Float in [0, +inf), reflecting motivational live-
                          ness at write time.
      arousal_tag:        BLA arousal-tag scalar at write time (LaBar &
                          Cabeza 2006 per-trace tag). 0.0 when amygdala
                          analog is disabled.
      last_vs:            Last V_s_anchor reading on the parent
                          (scale, stream_mixture) family at write/remap/
                          invalidate time, when MECH-269 Phase 1 / 2 are
                          active. None otherwise. Used by MECH-292's
                          recoverability term.
      staleness_at_write: MECH-284 region staleness scalar at write time
                          (or 0.0 when accumulator is disabled). Captured
                          so a later-frozen trace remembers how stale the
                          region already was when it was preserved.
      payload_written_step: HippocampalModule tick index at payload write.
                          Phase 2 stand-in for trace age; downstream
                          consumers compare against current tick.

    Notes:
      The payload is OPTIONAL: anchors carry goal_payload=None when SD-039
      is disabled. When SD-039 is enabled but no goal / valence / arousal
      is available at write time (e.g. very early training), the payload
      is still attached with whatever signals are present (z_goal_snapshot
      can be None; wanting_strength / arousal_tag default 0.0).

      Pure dataclass; does not own grad. z_goal_snapshot is .detach()'d on
      write (see HippocampalModule.build_goal_payload).
    """
    z_goal_snapshot: Optional[torch.Tensor] = None
    wanting_strength: float = 0.0
    arousal_tag: float = 0.0
    last_vs: Optional[float] = None
    staleness_at_write: Optional[float] = None
    payload_written_step: int = 0


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
      goal_payload:   SD-039 motivational payload. None when SD-039 is
                      disabled (or write was skipped, e.g. simulation).
                      Preserved across mark_inactive (dual-trace).
    """
    key: AnchorKey
    z_world: torch.Tensor
    active: bool = True
    created_at: int = 0
    last_accessed: int = 0
    below_threshold_streak: int = 0
    goal_payload: Optional[AnchorGoalPayload] = None

    def goal_match(self, current_z_goal: Optional[torch.Tensor]) -> float:
        """SD-039: cosine similarity between stored z_goal_snapshot and
        the supplied current z_goal latent. Returns 0.0 when this anchor
        has no payload, no z_goal_snapshot, or current_z_goal is None.

        Both vectors are flattened to 1-D for the cosine; shapes other
        than [goal_dim] or [1, goal_dim] are reduced via mean over leading
        dims. Negative cosines are clamped to 0.0 -- goal_match is a
        non-negative motivational-relevance signal, not a signed
        correlation.
        """
        if self.goal_payload is None:
            return 0.0
        z_snap = self.goal_payload.z_goal_snapshot
        if z_snap is None or current_z_goal is None:
            return 0.0
        a = z_snap.detach().reshape(-1).float()
        b = current_z_goal.detach().reshape(-1).float()
        if a.numel() == 0 or b.numel() == 0:
            return 0.0
        if a.numel() != b.numel():
            return 0.0
        if a.norm().item() < 1e-9 or b.norm().item() < 1e-9:
            return 0.0
        sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1)
        return max(0.0, float(sim.item()))


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
        goal_payload: Optional[AnchorGoalPayload] = None,
    ) -> Anchor:
        """Install a new active anchor for (scale, segment_id, stream_mixture).

        If an active anchor already exists on this (scale, stream_mixture)
        family, it is marked inactive first (Bouton 2004 dual-trace).
        Idempotent: if the same AnchorKey is written twice in a row,
        last_accessed is refreshed and the existing record is returned
        unchanged.

        SD-039: goal_payload is optional. When supplied (SD-039 master
        flag on AND not in simulation/replay), it is attached to the new
        Anchor and preserved across later mark_inactive remaps. When
        the call is idempotent (same key written twice in a row) and a
        non-None goal_payload is supplied, the payload is REFRESHED on
        the existing record so successive write/remap calls in the same
        tick capture the most recent motivational state.
        """
        key: AnchorKey = (scale, segment_id, tuple(stream_mixture))
        existing = self._all.get(key)
        if existing is not None and existing.active:
            existing.last_accessed = self._tick
            if goal_payload is not None:
                existing.goal_payload = goal_payload
            return existing

        family = (scale, tuple(stream_mixture))
        prior_active = self._active.get(family)
        if prior_active is not None and prior_active.key != key:
            # SD-039: dual-trace remap. Refresh prior anchor's payload
            # (so the trace it leaves behind reflects the most recent
            # motivational state of the path being abandoned), THEN mark
            # it inactive. This is the "cause-of-blockage payload" that
            # MECH-292 will rank against current goal-match.
            if goal_payload is not None:
                prior_active.goal_payload = goal_payload
            self._mark_inactive_internal(prior_active)

        z_snap = z_world.detach().clone()
        anchor = Anchor(
            key=key,
            z_world=z_snap,
            active=True,
            created_at=self._tick,
            last_accessed=self._tick,
            below_threshold_streak=0,
            goal_payload=goal_payload,
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

    def mark_inactive(
        self,
        scale: str,
        stream_mixture: Tuple[str, ...],
        goal_payload: Optional[AnchorGoalPayload] = None,
    ) -> Optional[Anchor]:
        """Mark the currently-active anchor on (scale, stream_mixture) inactive.

        Preserves the anchor in self._all (dual-trace). Returns the anchor
        that was deactivated, or None if no active anchor existed on that
        family. Used by reset_region and by the hysteresis reset path.

        SD-039: when a non-None goal_payload is supplied, it is written
        onto the anchor BEFORE mark_inactive fires. This captures the
        motivational state at the moment of invalidation (the third
        write site after WRITE and REMAP). The existing payload is NOT
        cleared on mark_inactive -- inactive anchors retain their
        motivational identity. Refresh-on-invalidate is the design
        choice in claims.yaml SD-039: payload reflects the most recent
        wanting/arousal at the point the trace is preserved, so MECH-292
        ranking sees the latest signal.
        """
        family = (scale, tuple(stream_mixture))
        anchor = self._active.get(family)
        if anchor is None:
            return None
        if goal_payload is not None:
            anchor.goal_payload = goal_payload
        self._mark_inactive_internal(anchor)
        return anchor

    def reset_region(
        self,
        scale: str,
        stream_mixture: Tuple[str, ...],
        new_segment_id: str,
        z_world: torch.Tensor,
        goal_payload: Optional[AnchorGoalPayload] = None,
    ) -> Anchor:
        """Dual-trace remap: mark current active inactive; install new active.

        Consumed by MECH-287 broadcast-driven invalidation (Phase 3 wiring)
        and by the internal hysteresis reset path. The previous anchor is
        retained in all_anchors() as an inactive trace; the new anchor is
        written with the incoming segment_id.

        SD-039: goal_payload, when supplied, is written onto BOTH the
        outgoing inactive anchor (preserving the cause-of-blockage state)
        AND the new active anchor (seeding the new trace's motivational
        identity).
        """
        self.mark_inactive(scale, stream_mixture, goal_payload=goal_payload)
        return self.write_anchor(
            scale, new_segment_id, stream_mixture, z_world,
            goal_payload=goal_payload,
        )

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
        goal_payload: Optional[AnchorGoalPayload] = None,
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

        SD-039: goal_payload, when supplied, is forwarded to each
        write_anchor call so anchors installed in response to this
        boundary tick carry the current motivational state. The same
        payload is shared across multiple events fired in the same tick
        (they reflect the same goal/wanting context).
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
                    goal_payload=goal_payload,
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

    def all_with_dual_trace(self, scale: Optional[str] = None) -> List[Anchor]:
        """Broad seed pool for MECH-285 sleep replay (Phase B).

        Returns active + inactive anchors with the Bouton 2004 dual-trace
        preserved -- the offline replay sampler treats inactive (mark_inactive
        but-not-erased) anchors as legitimate seeds. Alias of all_anchors()
        named for the design-doc API on sleep_aggregation_cluster.md.
        """
        return self.all_anchors(scale=scale)

    def query_by_goal_match(
        self,
        current_z_goal: Optional[torch.Tensor],
        threshold: float = 0.0,
        scale: Optional[str] = None,
        active_only: bool = False,
    ) -> List[Tuple[Anchor, float]]:
        """SD-039: motivational-relevance query over the dual-trace anchor pool.

        Scans active + inactive anchors (the Bouton 2004 dual-trace pool) and
        returns the subset whose stored goal_payload.z_goal_snapshot has
        non-negative cosine similarity to current_z_goal at or above
        threshold, paired with the goal_match score. Anchors without a
        payload (or without a z_goal_snapshot) score 0.0 and are filtered
        out at any positive threshold; with threshold=0.0 they are excluded
        unless explicitly desired (the wrapper returns only non-zero
        matches by default to keep MECH-292 ranking input clean).

        This is the substrate-side query MECH-292 (ranked ghost-goal bank)
        will consume. SD-039 itself does NOT rank or implement the bank --
        ranking by ghost_priority ~ wanting * goal_match * staleness *
        recoverability lives in MECH-292's ghost-goal bank module.

        Args:
            current_z_goal:  the agent's current z_goal latent. None -> all
                             scores degenerate to 0.0 -> empty result.
            threshold:       minimum goal_match for inclusion. Default 0.0
                             still excludes payload-less anchors via the
                             > 0.0 filter; pass threshold=-1.0 to include
                             every anchor with a payload regardless of
                             match.
            scale:           optional scale filter (e.g. "fast", "slow").
            active_only:     if True, restrict to the currently active
                             dual-trace half. Default False -- inactive
                             anchors are exactly the substrate's ghost-goal
                             use case (MECH-292 / MECH-293).

        Returns:
            List of (anchor, goal_match) pairs sorted by goal_match
            descending. Empty when current_z_goal is None or no anchor
            clears threshold.
        """
        if current_z_goal is None:
            return []
        pool = self.active_anchors(scale=scale) if active_only else self.all_anchors(scale=scale)
        scored: List[Tuple[Anchor, float]] = []
        for anchor in pool:
            score = anchor.goal_match(current_z_goal)
            # Default threshold=0.0 excludes payload-less / norm-zero traces.
            # threshold=-1.0 (or any negative) is the explicit "include
            # everything" path for diagnostic queries.
            if threshold < 0.0:
                if anchor.goal_payload is not None:
                    scored.append((anchor, score))
            elif score > threshold or (score == threshold and threshold > 0.0):
                scored.append((anchor, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

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
