"""StalenessAccumulator -- MECH-284 Phase 3 substrate.

Region-indexed V_s residual schema-staleness accumulator. Integrates
MECH-287 BroadcastEvents (source_scale, source_segment_id_old) against
the currently active anchor set with an attribution_weight credit
assignment, leaks per tick, and exposes a getter consumed by MECH-269
online anchor-reset hysteresis (the online arm of the dual-readout).

Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
Claim:      MECH-284 (docs/claims/claims.yaml)
Phase:      3 -- online read-out only. MECH-285 offline sleep-priority
            read-out is deferred.

Operational definition (refined 2026-04-22 in claims.yaml):

    For each schema region r in active_anchor_set(t):
        if MECH-287 trigger(t):
            staleness[r] += attribution_weight(r, source_streams) * magnitude
        staleness[r] *= leak_factor

The accumulator is keyed on (scale, segment_id) -- the region key used
by per_region_vs (Phase 2 iii, T4). stream_mixture is NOT part of the
region key, matching the per-region readout: one (scale, segment_id)
region may be reached by multiple stream_mixture families; the
accumulator merges their staleness.

attribution_weight implementations (Phase 3 first pass):
  - "equal":          1 / N  across N active anchors (uniform credit).
  - "stream_overlap": |source_sources & stream_mixture| / max(|source_sources|, 1)
                       per anchor; cheap cosine-similarity surrogate over
                       the stream-name sets. An anchor with zero overlap
                       gets zero credit.

Under "equal", every active anchor is credited broadcast.strength / N
per broadcast. Under "stream_overlap", anchors with matching source
streams absorb more of the staleness; anchors with no overlap absorb
nothing.

MECH-094 gate:
    integrate() is invoked only from HippocampalModule.integrate_staleness,
    which is called from REEAgent.sense() (the waking observation stream).
    Simulation / replay paths must not route through integrate_staleness.

Numerical:
    staleness is clipped at config.staleness_clip to match the AnchorSet
    proxy range, so V_s_anchor = V_s(r) - staleness[r] stays in the same
    [-1, 1] band whether the Phase 2 proxy or the Phase 3 lookup drives
    hysteresis.

No trainable parameters. Pure float arithmetic.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

from ree_core.utils.config import StalenessAccumulatorConfig


RegionKey = Tuple[str, str]   # (scale, segment_id)


class StalenessAccumulator:
    """MECH-284 Phase 3 region-indexed staleness accumulator."""

    def __init__(self, config: StalenessAccumulatorConfig):
        self.config = config
        # region_key -> staleness in [0, staleness_clip].
        self._staleness: Dict[RegionKey, float] = {}
        # Diagnostic counters.
        self._n_integrations: int = 0
        self._n_leak_ticks: int = 0

    # ------------------------------------------------------------------ #
    # Integration                                                        #
    # ------------------------------------------------------------------ #
    def integrate(
        self,
        broadcasts: Iterable[Any],
        active_anchors: Iterable[Any],
    ) -> None:
        """Credit each broadcast's strength across the active anchor set.

        Args:
          broadcasts:     Iterable of BroadcastEvent-like objects carrying
                          .strength (float), .source_scale (str),
                          .source_segment_id_old (str), and
                          .source_sources (list[str]).
          active_anchors: Iterable of AnchorSet Anchor records; each has
                          .key = (scale, segment_id, stream_mixture).

        Staleness is accumulated per region_key = (scale, segment_id)
        drawn from each active anchor. Multiple anchors sharing a region
        key (different stream_mixtures under the same segment) add their
        credit to the same region bucket.
        """
        anchors = list(active_anchors)
        if not anchors:
            return
        mode = str(getattr(self.config, "attribution_mode", "equal")).lower()
        clip = float(getattr(self.config, "staleness_clip", 1.0))

        for bcast in broadcasts:
            self._n_integrations += 1
            strength = float(getattr(bcast, "strength", 0.0))
            if strength <= 0.0:
                continue
            if mode == "equal":
                weight = 1.0 / float(len(anchors))
                increments = [weight] * len(anchors)
            elif mode == "stream_overlap":
                src = set(getattr(bcast, "source_sources", []) or [])
                src_size = max(len(src), 1)
                increments = []
                for anchor in anchors:
                    mixture = set(anchor.key[2])
                    overlap = len(src & mixture) / float(src_size)
                    increments.append(overlap)
            else:
                # Unknown mode: treat as equal and keep integrating.
                weight = 1.0 / float(len(anchors))
                increments = [weight] * len(anchors)

            for anchor, incr in zip(anchors, increments):
                if incr <= 0.0:
                    continue
                region_key = (anchor.key[0], anchor.key[1])
                curr = self._staleness.get(region_key, 0.0)
                nxt = curr + incr * strength
                if nxt > clip:
                    nxt = clip
                self._staleness[region_key] = nxt

    # ------------------------------------------------------------------ #
    # Per-tick leak                                                      #
    # ------------------------------------------------------------------ #
    def tick_leak(self) -> None:
        """Apply leak_factor to all region entries; drop near-zero rows."""
        self._n_leak_ticks += 1
        leak = float(getattr(self.config, "leak_factor", 0.995))
        eps = float(getattr(self.config, "drop_epsilon", 1e-6))
        if leak >= 1.0 and leak != 1.0:
            # Treat out-of-range leak >= 1 defensively by clamping to 1.
            leak = 1.0
        if not self._staleness:
            return
        drop: List[RegionKey] = []
        for k, v in self._staleness.items():
            nv = v * leak
            if nv < eps:
                drop.append(k)
            else:
                self._staleness[k] = nv
        for k in drop:
            del self._staleness[k]

    # ------------------------------------------------------------------ #
    # Read-out                                                           #
    # ------------------------------------------------------------------ #
    def get(self, region_key: RegionKey) -> float:
        """Return staleness for (scale, segment_id), 0.0 if absent."""
        return float(self._staleness.get(region_key, 0.0))

    def lookup_by_anchor_key(self, anchor_key: Tuple[str, str, Tuple[str, ...]]) -> float:
        """Convenience: project an AnchorKey onto RegionKey and return staleness."""
        return self.get((anchor_key[0], anchor_key[1]))

    def snapshot(self) -> Dict[RegionKey, float]:
        """Shallow copy of the current staleness map (diagnostic)."""
        return dict(self._staleness)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "n_integrations": int(self._n_integrations),
            "n_leak_ticks": int(self._n_leak_ticks),
            "n_regions": int(len(self._staleness)),
            "max_staleness": float(max(self._staleness.values())) if self._staleness else 0.0,
            "mean_staleness": (
                float(sum(self._staleness.values()) / len(self._staleness))
                if self._staleness
                else 0.0
            ),
            "attribution_mode": str(getattr(self.config, "attribution_mode", "equal")),
            "leak_factor": float(getattr(self.config, "leak_factor", 0.995)),
            "staleness_clip": float(getattr(self.config, "staleness_clip", 1.0)),
        }

    # ------------------------------------------------------------------ #
    # Phase E targeted decay (MECH-273)                                  #
    # ------------------------------------------------------------------ #
    def partial_decay(
        self,
        replayed_regions: Iterable[RegionKey],
        decay_factor: float = 0.5,
    ) -> int:
        """Multiplicative decay applied only to the supplied regions.

        Distinct from tick_leak (which decays every region uniformly each
        waking tick): partial_decay implements the MECH-273 WRITEBACK-phase
        contract -- replayed regions had their schemas refreshed by the
        offline gradient pass, so their staleness is partially discharged.
        Regions absent from the supplied set are untouched.

        Args:
          replayed_regions: Iterable of (scale, segment_id) keys that the
                            sleep cycle replayed.
          decay_factor:     Multiplier in [0, 1] applied to staleness on
                            those regions. 0.5 = halve. Out-of-range values
                            are clamped to [0, 1].

        Returns:
          Number of region entries actually mutated (skipped if absent).
        """
        if decay_factor < 0.0:
            decay_factor = 0.0
        elif decay_factor > 1.0:
            decay_factor = 1.0
        if not self._staleness:
            return 0
        eps = float(getattr(self.config, "drop_epsilon", 1e-6))
        n_decayed = 0
        drop: List[RegionKey] = []
        seen: set = set()
        for region_key in replayed_regions:
            if region_key in seen:
                continue
            seen.add(region_key)
            curr = self._staleness.get(region_key)
            if curr is None:
                continue
            nxt = curr * decay_factor
            if nxt < eps:
                drop.append(region_key)
            else:
                self._staleness[region_key] = nxt
            n_decayed += 1
        for k in drop:
            del self._staleness[k]
        return n_decayed

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Per-episode reset: clear map + counters."""
        self._staleness.clear()
        self._n_integrations = 0
        self._n_leak_ticks = 0
