"""GhostGoalBank -- MECH-292 ranked ghost-goal bank.

Derived ranking view over the SD-039 dual-trace anchor pool. Per the
MECH-292 spec (REE_assembly/docs/architecture/mech_292_ghost_goal_bank.md)
this module is intentionally a derived view, not a persistent store: SD-039
already preserves the per-anchor goal payload (z_goal_snapshot,
wanting_strength, arousal_tag, last_vs, staleness_at_write); MECH-292 just
arranges the existing data into a ranked bank.

Each rank() call walks the anchor pool, scores each anchor on a four-term
composite (wanting + goal_match + staleness + recoverability), and returns
a sorted list. Pure arithmetic; no trainable parameters; no gradient flow.

Architectural commitments (from MECH-292 spec):
  - The goal_match_floor is the rumination guard: anchors with no payload
    OR with cosine(z_goal_snapshot, current_z_goal) below the floor are
    invisible to the bank entirely. Pure low-V_s chasing is excluded by
    construction.
  - The bank does not own the per-anchor payload (SD-039 does), does not
    generate trajectories (MECH-293 does), does not gate write paths
    (MECH-094 / MECH-261 do), and does not modify the anchor pool. It is
    read-only over the existing dual-trace structure.
  - Staleness preference: when a StalenessAccumulator (MECH-284) is wired
    in, the per-anchor staleness is read from accumulator.snapshot() at
    the (scale, segment_id) region key. When the accumulator is absent,
    a local fallback ((current_tick - last_accessed) * staleness_proxy_rate,
    clipped to [0, 1]) is used so MECH-292 still works when MECH-284 is
    disabled.

MECH-094: substrate-side scope. The bank reads payloads whose provenance
was set at SD-039 population time; sense() always passes
simulation_mode=False, so the source anchors carry waking-stream
provenance. The bank itself has no write path -- nothing to gate.

Phased training: not applicable.

Falsifiable signature (from spec): in a reward-relocation task, anchors
from the now-obstructed but still-valued path should rank above equally
stale but goal-irrelevant anchors. Specifically: in a paired comparison
of a goal-relevant inactive anchor vs a goal-irrelevant inactive anchor
with matched staleness, the goal-relevant anchor's ghost_priority should
be strictly higher in >=4/7 seeds, with the gap dominated by the
goal_match component. Behavioural validation lands in V3-EXQ-495 when
MECH-293 consumes the bank.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ree_core.hippocampal.anchor_set import Anchor, AnchorSet
from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator
from ree_core.utils.config import GhostGoalBankConfig


@dataclass
class GhostGoalBankEntry:
    """One ranked entry in the ghost-goal bank.

    Fields:
      anchor:          reference to the source anchor in the dual-trace pool
                       (read-only; bank does not mutate).
      ghost_priority:  composite rank score (sum of weighted component
                       contributions; non-negative, magnitude not
                       semantically calibrated -- only ordering matters).
      components:      per-term breakdown for diagnostics, with keys
                       "wanting", "goal_match", "staleness",
                       "recoverability" each holding the WEIGHTED term
                       (i.e. weight * raw value). Sum of values matches
                       ghost_priority within float tolerance.
    """
    anchor: Anchor
    ghost_priority: float
    components: Dict[str, float] = field(default_factory=dict)


class GhostGoalBank:
    """MECH-292: derived ranking view over the dual-trace anchor pool.

    Stateless across calls beyond a small diagnostics cache. Each rank()
    call walks the anchor pool, scores each anchor, and returns a sorted
    list. The bank does NOT mutate the anchor set; it is a read-only
    consumer.

    Instantiation contract:
      anchor_set must have use_sd039_anchor_payload=True at the
      AnchorSetConfig level so that anchors carry a populated
      goal_payload (otherwise every anchor scores goal_match=0.0 and
      every entry is excluded by the floor; the bank degenerates to
      empty). HippocampalModule.__init__ enforces this precondition.

      staleness_accumulator is optional. When supplied, per-anchor
      staleness is read from accumulator.snapshot() at the
      (scale, segment_id) region key. When None, a local fallback
      based on the anchor's last_accessed tick is used.
    """

    def __init__(
        self,
        config: GhostGoalBankConfig,
        anchor_set: AnchorSet,
        staleness_accumulator: Optional[StalenessAccumulator] = None,
    ) -> None:
        self.config = config
        self.anchor_set = anchor_set
        self.staleness_accumulator = staleness_accumulator
        self._last_diagnostics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def rank(
        self,
        current_z_goal: Optional[torch.Tensor],
    ) -> List[GhostGoalBankEntry]:
        """Return the ranked ghost-goal bank for the supplied current z_goal.

        Anchors with no stored z_goal_snapshot (payload-less or written
        outside any goal regime) are excluded by goal_match_floor (their
        Anchor.goal_match() returns 0.0; the strict comparison
        score < floor with a positive floor excludes them).

        Returns an empty list when current_z_goal is None or when no
        anchor clears the floor. top_k caps the returned list size; the
        full pool is still scored for diagnostic purposes.
        """
        cfg = self.config
        if current_z_goal is None:
            self._last_diagnostics = self._empty_diagnostics(reason="no_z_goal")
            return []

        pool = self._pool_for_config()
        n_scanned = len(pool)

        scored: List[GhostGoalBankEntry] = []
        sums = {
            "wanting": 0.0,
            "goal_match": 0.0,
            "staleness": 0.0,
            "recoverability": 0.0,
        }
        n_below_floor = 0
        n_no_payload = 0

        for anchor in pool:
            payload = anchor.goal_payload
            if payload is None:
                n_no_payload += 1
                continue

            goal_match = anchor.goal_match(current_z_goal)
            if goal_match < cfg.goal_match_floor:
                n_below_floor += 1
                continue

            wanting = float(payload.wanting_strength)
            staleness = self._staleness_for_anchor(anchor)
            recoverability = self._recoverability_for_anchor(anchor)

            w_term = cfg.wanting_weight * wanting
            m_term = cfg.goal_match_weight * goal_match
            s_term = cfg.staleness_weight * staleness
            r_term = cfg.recoverability_weight * recoverability

            priority = w_term + m_term + s_term + r_term
            sums["wanting"] += w_term
            sums["goal_match"] += m_term
            sums["staleness"] += s_term
            sums["recoverability"] += r_term

            scored.append(GhostGoalBankEntry(
                anchor=anchor,
                ghost_priority=float(priority),
                components={
                    "wanting": float(w_term),
                    "goal_match": float(m_term),
                    "staleness": float(s_term),
                    "recoverability": float(r_term),
                },
            ))

        scored.sort(key=lambda e: e.ghost_priority, reverse=True)
        n_admitted = len(scored)
        if cfg.top_k is not None and cfg.top_k >= 0:
            scored = scored[: cfg.top_k]

        priorities = [e.ghost_priority for e in scored]
        max_priority = max(priorities) if priorities else 0.0
        mean_priority = (
            sum(priorities) / len(priorities) if priorities else 0.0
        )
        self._last_diagnostics = {
            "n_candidates_scanned": int(n_scanned),
            "n_no_payload": int(n_no_payload),
            "n_below_floor": int(n_below_floor),
            "n_admitted": int(n_admitted),
            "n_returned": int(len(scored)),
            "max_priority": float(max_priority),
            "mean_priority": float(mean_priority),
            "component_sums": {k: float(v) for k, v in sums.items()},
            "reason": "ok",
        }
        return scored

    def get_diagnostics(self) -> Dict[str, Any]:
        """Last-call summary. Cleared by reset()."""
        return dict(self._last_diagnostics)

    def reset(self) -> None:
        """Per-episode reset of the diagnostics cache.

        The bank is otherwise stateless across calls -- the source-of-
        truth anchor pool lives in AnchorSet, and AnchorSet.reset() is
        called separately by HippocampalModule.reset_anchor_set() on
        the episode boundary.
        """
        self._last_diagnostics = {}

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _pool_for_config(self) -> List[Anchor]:
        """Anchor pool to score, per include_active / include_inactive / scale."""
        cfg = self.config
        if cfg.include_active and cfg.include_inactive:
            return self.anchor_set.all_anchors(scale=cfg.scale)
        if cfg.include_active and not cfg.include_inactive:
            return self.anchor_set.active_anchors(scale=cfg.scale)
        if cfg.include_inactive and not cfg.include_active:
            # Inactive-only: derive from all_anchors filtered by active flag.
            return [
                a for a in self.anchor_set.all_anchors(scale=cfg.scale)
                if not a.active
            ]
        # Both include flags False -> empty pool (degenerate config).
        return []

    def _staleness_for_anchor(self, anchor: Anchor) -> float:
        """Per-anchor staleness in [0, 1].

        Preference order:
          1. payload.staleness_at_write -- used as a NON-PREFERRED last-
             resort fallback. Captures region staleness at write time;
             does not advance with current age. Skipped if both
             accumulator and tick-delta proxy are available.
          2. StalenessAccumulator.snapshot()[(scale, segment_id)] --
             MECH-284 region staleness, the spec-preferred source.
          3. (current_tick - last_accessed) * staleness_proxy_rate,
             clipped to [0, 1] -- local proxy, used when no accumulator
             is present.
        """
        cfg = self.config
        # (1) Preferred: MECH-284 region staleness when accumulator wired.
        if self.staleness_accumulator is not None:
            snap = self.staleness_accumulator.snapshot()
            region_key = (anchor.key[0], anchor.key[1])
            if region_key in snap:
                return max(0.0, min(1.0, float(snap[region_key])))
            # accumulator present but no entry for this region -> 0.0
            # (region was never broadcast-credited; not stale yet).
            return 0.0
        # (2) Local proxy: tick-delta since last access.
        try:
            current_tick = self.anchor_set._tick
        except AttributeError:
            current_tick = 0
        delta = max(0, int(current_tick) - int(anchor.last_accessed))
        proxy = delta * float(cfg.staleness_proxy_rate)
        return max(0.0, min(1.0, proxy))

    def _recoverability_for_anchor(self, anchor: Anchor) -> float:
        """Per-anchor recoverability in [0, 1].

        last_vs near 1.0 = the region was confidently grounded at
        preservation time; high recoverability. last_vs near 0.0 =
        already failing groundedness when preserved; low recoverability.
        last_vs is None (e.g. MECH-269 phase 1/2 disabled) -> use the
        configured default.
        """
        cfg = self.config
        payload = anchor.goal_payload
        if payload is None:
            return float(cfg.default_recoverability_when_unknown)
        last_vs = payload.last_vs
        if last_vs is None:
            return float(cfg.default_recoverability_when_unknown)
        return max(0.0, min(1.0, float(last_vs)))

    def _empty_diagnostics(self, reason: str) -> Dict[str, Any]:
        return {
            "n_candidates_scanned": 0,
            "n_no_payload": 0,
            "n_below_floor": 0,
            "n_admitted": 0,
            "n_returned": 0,
            "max_priority": 0.0,
            "mean_priority": 0.0,
            "component_sums": {
                "wanting": 0.0,
                "goal_match": 0.0,
                "staleness": 0.0,
                "recoverability": 0.0,
            },
            "reason": reason,
        }
