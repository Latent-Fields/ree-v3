"""
SleepReplaySampler -- MECH-285 Phase B substrate.

Offline sleep-priority arm of the V_s invalidation runtime's dual-readout
(MECH-284 is the online arm). At sleep entry, freezes a snapshot of the
MECH-284 StalenessAccumulator's region-keyed map. During the cycle, draws
seed anchors from the AnchorSet broad pool (active + inactive, dual-trace
preserved per Bouton 2004) with probability proportional to a softmax over
the frozen staleness signal.

Design contract (REE_assembly/docs/architecture/sleep_aggregation_cluster.md):

  * STATELESS within a cycle. The snapshot is captured ONCE at SLEEP_ENTRY
    and read-only thereafter; per-draw decisions do not mutate it.
  * BROAD POOL. Seeds are drawn from AnchorSet.all_with_dual_trace(),
    which returns active AND inactive anchors. Inactive anchors (Bouton
    2004 mark_inactive but-not-erased) are legitimate offline replay
    seeds because the dual-trace structure preserves their content.
  * UNIFORM FALLBACK. When no StalenessAccumulator is wired (pre-Phase-3
    configurations) or when every region's staleness is identically zero,
    fall back to a uniform draw over the broad pool. Controlled by
    allow_uniform_fallback (default True).
  * TEMPERATURE. softmax temperature controls the staleness skew. T=1.0
    is the canonical neutral value; lower T sharpens the distribution.

Phase B is a NO-OP CONSUMER. The draws produced here are recorded as
diagnostics on SleepCycleState.last_metrics; no downstream consumer
(routing gate, aggregator, writeback) sees them yet. Phases C / D / E
will pipe the draws into MECH-272 routing, MECH-275 aggregation, and
MECH-273 self-model writeback respectively.

Bit-identical OFF guarantee: the master flag use_mech285_sampler defaults
False; when False, REEAgent never instantiates this class and the sleep
cycle runs exactly as in Phase A.

No trainable parameters. Pure float arithmetic + numpy random draws.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ree_core.hippocampal.anchor_set import Anchor, AnchorSet
from ree_core.hippocampal.staleness_accumulator import StalenessAccumulator


RegionKey = Tuple[str, str]  # (scale, segment_id) -- matches MECH-284


class SleepReplaySampler:
    """MECH-285 Phase B offline replay-priority sampler.

    Constructor takes the AnchorSet broad-pool source and an optional
    StalenessAccumulator. The lifecycle is:

        sampler.freeze_snapshot()        # once at SLEEP_ENTRY
        for _ in range(N):
            anchor = sampler.draw(rng)   # stateless, repeatable

    The snapshot is the StalenessAccumulator.snapshot() at freeze time;
    if the accumulator is None or its snapshot is empty, the sampler
    runs in uniform-fallback mode (when allowed) or raises (when not).
    """

    def __init__(
        self,
        anchor_set: AnchorSet,
        staleness_accumulator: Optional[StalenessAccumulator],
        temperature: float = 1.0,
        allow_uniform_fallback: bool = True,
    ) -> None:
        if temperature <= 0.0:
            raise ValueError(
                f"SleepReplaySampler temperature must be > 0; got {temperature}"
            )
        self._anchor_set = anchor_set
        self._staleness = staleness_accumulator
        self._temperature = float(temperature)
        self._allow_uniform_fallback = bool(allow_uniform_fallback)

        # Frozen snapshot captured at SLEEP_ENTRY. None until freeze_snapshot().
        self._snapshot: Optional[Dict[RegionKey, float]] = None
        self._snapshot_is_uniform: bool = False

        # Diagnostics.
        self._n_draws: int = 0
        self._draw_region_counts: Dict[RegionKey, int] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def freeze_snapshot(self) -> None:
        """Capture the staleness snapshot. Call once at SLEEP_ENTRY."""
        if self._staleness is None:
            self._snapshot = {}
            self._snapshot_is_uniform = True
        else:
            snap = self._staleness.snapshot()
            self._snapshot = dict(snap)
            # Treat all-zero staleness the same as no accumulator: nothing
            # to skew the softmax against, so fall back to uniform.
            self._snapshot_is_uniform = not any(
                v > 0.0 for v in self._snapshot.values()
            )
        self._n_draws = 0
        self._draw_region_counts = {}

    def reset(self) -> None:
        """Drop the snapshot and zero diagnostics."""
        self._snapshot = None
        self._snapshot_is_uniform = False
        self._n_draws = 0
        self._draw_region_counts = {}

    # ------------------------------------------------------------------ #
    # Draw                                                               #
    # ------------------------------------------------------------------ #
    def draw(self, rng: Optional[np.random.Generator] = None) -> Optional[Anchor]:
        """Return one Anchor from the broad pool, or None if pool is empty.

        rng: optional numpy Generator for reproducibility. Defaults to the
        module-level numpy random state.
        """
        if self._snapshot is None:
            raise RuntimeError(
                "SleepReplaySampler.draw() called before freeze_snapshot()"
            )

        seeds: List[Anchor] = self._anchor_set.all_with_dual_trace()
        if not seeds:
            return None

        if self._snapshot_is_uniform:
            if not self._allow_uniform_fallback:
                raise RuntimeError(
                    "SleepReplaySampler has no staleness signal and "
                    "allow_uniform_fallback=False"
                )
            probs = None  # numpy uniform default
        else:
            probs = self._softmax_probs(seeds)

        idx = self._choose(len(seeds), probs, rng)
        chosen = seeds[idx]

        region_key: RegionKey = (chosen.key[0], chosen.key[1])
        self._n_draws += 1
        self._draw_region_counts[region_key] = (
            self._draw_region_counts.get(region_key, 0) + 1
        )
        return chosen

    def _softmax_probs(self, seeds: List[Anchor]) -> np.ndarray:
        snap = self._snapshot or {}
        T = self._temperature
        logits = np.array(
            [snap.get((a.key[0], a.key[1]), 0.0) / T for a in seeds],
            dtype=np.float64,
        )
        # Numerically-stable softmax.
        logits -= float(np.max(logits))
        exp = np.exp(logits)
        total = float(exp.sum())
        if total <= 0.0 or not math.isfinite(total):
            # Degenerate -- fall back to uniform within this draw.
            return np.full(len(seeds), 1.0 / len(seeds), dtype=np.float64)
        return exp / total

    @staticmethod
    def _choose(
        n: int,
        probs: Optional[np.ndarray],
        rng: Optional[np.random.Generator],
    ) -> int:
        if rng is None:
            if probs is None:
                return int(np.random.randint(0, n))
            return int(np.random.choice(n, p=probs))
        if probs is None:
            return int(rng.integers(0, n))
        return int(rng.choice(n, p=probs))

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #
    @property
    def n_draws(self) -> int:
        return int(self._n_draws)

    @property
    def draw_region_counts(self) -> Dict[RegionKey, int]:
        return dict(self._draw_region_counts)

    @property
    def snapshot_is_uniform(self) -> bool:
        return bool(self._snapshot_is_uniform)

    @property
    def snapshot_size(self) -> int:
        return 0 if self._snapshot is None else len(self._snapshot)

    def get_metrics(self) -> Dict[str, float]:
        """Return a flat metrics dict suitable for SleepCycleState.last_metrics."""
        return {
            "mech285_n_draws": float(self._n_draws),
            "mech285_n_distinct_regions_drawn": float(len(self._draw_region_counts)),
            "mech285_snapshot_size": float(self.snapshot_size),
            "mech285_snapshot_is_uniform": float(self._snapshot_is_uniform),
            "mech285_temperature": float(self._temperature),
        }
