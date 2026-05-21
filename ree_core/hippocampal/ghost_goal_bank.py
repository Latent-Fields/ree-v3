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

MECH-339 Constraint 1 (composite cue + outshining gate; ghost_goal_search.md
Section 0.2): the retrieval cue must be composite -- z_goal cosine PLUS a
context channel built from the SD-039 payload fields the match ignores,
combined by an OUTSHINING gate so a strong direct goal_match suppresses the
context channel rather than it being summed in with fixed weight (Smith &
Vela 2001; encoding specificity is real but outshone by a strong direct
cue). When GhostGoalBankConfig.use_composite_cue_outshining is enabled, a
fifth gated term is added:

    context_salience = 1 - exp(-arousal_tag / arousal_scale)   in [0, 1)
    gate             = clip_[0,1]((outshine_pivot - goal_match)
                                   / outshine_pivot)
    context_term     = context_weight * gate * context_salience

so the context channel contributes nothing once goal_match reaches
outshine_pivot (the falsifiable (i): a strong direct match must not let
the context channel change the top entry) and dominates only when the
direct match is weak/absent (falsifiable (ii)). The overall ghost_priority
stays an additive sum of independently gateable channels -- Constraint 2
(dissociable additive channels) is unaffected; the outshining is a
within-channel multiplier on the context term, not a product across
wanting/goal_match. The smallest substrate step sources the context
channel from arousal_tag only -- the one SD-039 payload field that is both
already stored and entirely unused by the bank. last_vs is deferred (it
already drives the recoverability channel; reusing it would double-count)
and a `cause` tag is deferred (not present in the implemented
AnchorGoalPayload -- design-sketch only, would require an SD-039 payload
extension). Defaults are no-op: with the master switch off (or
context_weight 0.0) the bank is bit-identical to the pre-MECH-339 form.

MECH-340 (persistence / efficacy gate; ghost_goal_search.md Section 0.3;
ARC-079 / Q-053 front-runner): when
GhostGoalBankConfig.use_persistence_efficacy_gate is enabled, rank()
requires a global PersistenceAppraisal (control_efficacy and
goal_unattainability in [0, 1]) and excludes anchors whose persistence
license falls below persistence_floor. The license is:

    license = clip_[0,1](control_efficacy * (1 - goal_unattainability))

It deliberately does NOT read recoverability, staleness, wanting, or any
accumulated-failure proxy (Maier & Seligman / Bouton / Husain hard
negatives). REEAgent computes the appraisal via
persistence_appraisal_compute when the gate is on; when the gate is on
but appraisal is None, persistence_default_when_appraisal_missing
applies (default 1.0).

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

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from ree_core.hippocampal.anchor_set import Anchor, AnchorSet
from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator
from ree_core.utils.config import GhostGoalBankConfig


@dataclass
class PersistenceAppraisal:
    """MECH-340 global persistence gate inputs (goal-level, not per-anchor).

    Q-053 front-runner structural form: an internal control/efficacy
    unattainability appraisal. High control_efficacy and low
    goal_unattainability license persistence; the gate's absence is the
    biological default (disengagement).

    Fields:
      control_efficacy:       [0, 1] -- control / efficacy positive signal.
      goal_unattainability:   [0, 1] -- unattainability appraisal (0 =
                                still attainable). NOT an accumulated-failure
                                tally and NOT derived from recoverability.
    """
    control_efficacy: float = 1.0
    goal_unattainability: float = 0.0


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
                       (i.e. weight * raw value). When MECH-339's
                       composite cue is enabled, a fifth key "context"
                       holds the gated context term (context_weight *
                       outshine_gate * context_salience). Sum of values
                       matches ghost_priority within float tolerance.
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
        persistence_appraisal: Optional[PersistenceAppraisal] = None,
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

        # MECH-339 Constraint 1: composite cue + outshining gate. Off by
        # default -> the "context" channel is absent everywhere (sums,
        # per-entry components, diagnostics) so the bank is bit-identical
        # to the pre-MECH-339 four-term form.
        composite_on = bool(cfg.use_composite_cue_outshining)
        persistence_on = bool(cfg.use_persistence_efficacy_gate)
        persistence_license = self._persistence_license(persistence_appraisal)

        scored: List[GhostGoalBankEntry] = []
        sums = {
            "wanting": 0.0,
            "goal_match": 0.0,
            "staleness": 0.0,
            "recoverability": 0.0,
        }
        if composite_on:
            sums["context"] = 0.0
        n_below_floor = 0
        n_no_payload = 0
        n_below_persistence = 0

        for anchor in pool:
            payload = anchor.goal_payload
            if payload is None:
                n_no_payload += 1
                continue

            goal_match = anchor.goal_match(current_z_goal)
            if goal_match < cfg.goal_match_floor:
                n_below_floor += 1
                continue

            if persistence_on and persistence_license < cfg.persistence_floor:
                n_below_persistence += 1
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

            components = {
                "wanting": float(w_term),
                "goal_match": float(m_term),
                "staleness": float(s_term),
                "recoverability": float(r_term),
            }

            if composite_on:
                # MECH-339 C1: gated context channel. The gate is a
                # function of THIS anchor's direct goal_match -- a strong
                # direct match (>= outshine_pivot) drives the gate to 0 so
                # the context term cannot change the top-ranked entry
                # (falsifiable (i)); a weak/absent match opens the gate so
                # the context channel can decide the ordering (falsifiable
                # (ii)). Not a fixed-weight additive term -- the gate is
                # the outshining (Smith & Vela 2001).
                context_salience = self._context_salience_for_anchor(anchor)
                gate = self._outshine_gate(goal_match)
                c_term = cfg.context_weight * gate * context_salience
                priority += c_term
                sums["context"] += c_term
                components["context"] = float(c_term)

            if persistence_on:
                components["persistence_license"] = float(persistence_license)

            scored.append(GhostGoalBankEntry(
                anchor=anchor,
                ghost_priority=float(priority),
                components=components,
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
            "n_below_persistence": int(n_below_persistence),
            "persistence_license": (
                float(persistence_license) if persistence_on else None
            ),
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

    def _outshine_gate(self, goal_match: float) -> float:
        """MECH-339 C1 outshining gate, a function of the direct cue match.

        gate = clip_[0,1]((outshine_pivot - goal_match) / outshine_pivot)

        gate -> 1.0 as goal_match -> 0 (direct cue weak/absent: the
        context channel is allowed to decide ordering -- falsifiable (ii)).
        gate -> 0.0 once goal_match >= outshine_pivot (a strong direct
        match outshines context entirely: the context channel cannot
        change the top-ranked entry -- falsifiable (i); Smith & Vela 2001).
        A non-positive pivot is the degenerate "always outshone" case ->
        gate 0.0 (context channel effectively off).
        """
        pivot = float(self.config.outshine_pivot)
        if pivot <= 0.0:
            return 0.0
        gate = (pivot - float(goal_match)) / pivot
        return max(0.0, min(1.0, gate))

    def _context_salience_for_anchor(self, anchor: Anchor) -> float:
        """MECH-339 C1 context salience in [0, 1) from arousal_tag.

        The smallest substrate step sources the context channel from
        payload.arousal_tag (LaBar & Cabeza 2006 per-trace arousal tag;
        the one SD-039 field that is both already stored and entirely
        unused by the bank's match). A saturating transform maps the
        unbounded arousal scalar onto [0, 1):

            context_salience = 1 - exp(-arousal_tag / arousal_scale)

        0.0 when there is no payload, arousal_tag <= 0 (e.g. BLA analog
        disabled, the SD-039 default), or arousal_scale is non-positive
        (degenerate config). last_vs and a `cause` tag are deliberately
        NOT folded in here -- last_vs already drives the recoverability
        channel (reuse would double-count) and `cause` is not present in
        the implemented AnchorGoalPayload (design-sketch only; an SD-039
        payload extension, out of scope for this smallest step).
        """
        payload = anchor.goal_payload
        if payload is None:
            return 0.0
        arousal = max(0.0, float(payload.arousal_tag))
        if arousal <= 0.0:
            return 0.0
        scale = float(self.config.arousal_scale)
        if scale <= 0.0:
            return 0.0
        return 1.0 - math.exp(-arousal / scale)

    def _persistence_license(
        self,
        persistence_appraisal: Optional[PersistenceAppraisal],
    ) -> float:
        """MECH-340 global persistence license in [0, 1].

        Off when use_persistence_efficacy_gate is False (callers treat this
        as 1.0 and never exclude on persistence). On when enabled:

          license = clip(control_efficacy) * (1 - clip(goal_unattainability))

        Missing appraisal uses persistence_default_when_appraisal_missing
        (default 1.0 so agent wiring can land later without changing ranks
        until a consumer passes explicit appraisal).
        """
        cfg = self.config
        if not bool(cfg.use_persistence_efficacy_gate):
            return 1.0
        if persistence_appraisal is None:
            return max(
                0.0,
                min(
                    1.0,
                    float(cfg.persistence_default_when_appraisal_missing),
                ),
            )
        control = max(
            0.0,
            min(1.0, float(persistence_appraisal.control_efficacy)),
        )
        unattain = max(
            0.0,
            min(1.0, float(persistence_appraisal.goal_unattainability)),
        )
        return max(0.0, min(1.0, control * (1.0 - unattain)))

    def _empty_diagnostics(self, reason: str) -> Dict[str, Any]:
        component_sums = {
            "wanting": 0.0,
            "goal_match": 0.0,
            "staleness": 0.0,
            "recoverability": 0.0,
        }
        if bool(self.config.use_composite_cue_outshining):
            component_sums["context"] = 0.0
        return {
            "n_candidates_scanned": 0,
            "n_no_payload": 0,
            "n_below_floor": 0,
            "n_below_persistence": 0,
            "persistence_license": (
                float(self._persistence_license(None))
                if bool(self.config.use_persistence_efficacy_gate)
                else None
            ),
            "n_admitted": 0,
            "n_returned": 0,
            "max_priority": 0.0,
            "mean_priority": 0.0,
            "component_sums": component_sums,
            "reason": reason,
        }
