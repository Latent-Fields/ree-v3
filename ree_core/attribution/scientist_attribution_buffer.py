"""
ScientistAttributionBuffer -- MECH-276 substrate.

The waking-phase mechanism that feeds the MECH-275 sleep-phase Bayesian
aggregator (ree_core/sleep/bayesian_aggregator.py). MECH-275 is only coherent
if the waking-phase feedstock is COUNTERFACTUAL-BACKED attribution -- a
deliberately produced contrast the agent had access to a counterfactual for --
not arbitrary correlation (aggregating correlation would produce noise-fit).
MECH-276 produces that feedstock.

Design contract (docs/architecture/scientist_agent_developmental_ordering.md
"MECH-276" section + claims.yaml MECH-276):

  * COUNTERFACTUAL-BACKED ATTRIBUTION. Each record is sourced from the
    single-pass comparators (SD-031 E2WorldForward on z_world; ARC-033
    E2HarmSForward on z_harm_s). The agency attribution is the comparator
    residual ||z_observed - E2(z_prev, a_actual)||; whether it is
    COUNTERFACTUAL-BACKED is decided by the counterfactual contrast
    ||E2(z_prev, a_actual) - E2(z_prev, a_cf)||: a discriminating action (the
    agent's choice mattered, contrast >= cf_margin) backs the attribution with
    a counterfactual; a near-zero-contrast action did not discriminate
    outcomes, so its attribution is mere correlation. This is the falsifiable
    distinction the MECH-275 claim turns on (claims.yaml MECH-276
    evidence_quality_note + the falsifiable secondary): correlational input
    must produce noise-fit, counterfactual-backed input schema revision.

  * PER-(DOMAIN, REGION) EMA. Records keyed on (domain, region) where
    region = (scale, segment_id) (matches the MECH-284 / MECH-269 anchor
    RegionKey the sleep loop routes on). V3 domains: "place" (E2WorldForward
    causal-footprint comparator) and "self" (E2HarmSForward / SD-003). The
    evidence_snapshot() merges across domains into a region -> attribution map
    (the v1 single-snapshot integration the phase_manager evidence loop
    consumes; per-domain snapshots are a documented follow-on).

  * ONLY-COUNTERFACTUAL-BACKED GATE. When only_counterfactual_backed (default
    True) a correlational (contrast < cf_margin) record is SKIPPED, so the
    feedstock the aggregator reads is structured. Setting it False is the
    correlational-control arm of the readiness diagnostic: feed everything ->
    the predicted noise-fit (no posterior movement / schema revision).

  * PER-CYCLE DECAY. decay_cycle() multiplies every region EMA by decay
    (default 1.0 = no decay), parallel to the BayesianAggregator snapshot
    decay; lets newer counterfactual-backed evidence overcome a stale prior.

  * GLOBAL SENTINEL. evidence_snapshot() also carries a GLOBAL_REGION sentinel
    key holding the global mean of all region attributions, so the sleep
    loop's region lookup can fall back to it when a routed region was not
    visited during the preceding waking phase.

MECH-094: record() is a NO-OP under simulation_mode (a replay / DMN tick must
not write attribution feedstock -- the attribution must come from real
consequential action). Doubly enforced: the comparators' comparator_residual /
forward reads the agent feeds are themselves MECH-094-gated upstream.

Bit-identical OFF guarantee: the master flag use_scientist_attribution defaults
False; when False, REEAgent never instantiates this class and the SleepLoop
evidence path sources the legacy MECH-284 staleness scalar exactly as before.

No trainable parameters. Pure float arithmetic + small dict state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


RegionKey = Tuple[str, str]            # (scale, segment_id) -- matches MECH-284
DomainRegionKey = Tuple[str, RegionKey]


@dataclass
class _RegionRecord:
    """Per-(domain, region) running attribution state."""

    ema: float = 0.0
    n: int = 0
    n_counterfactual_backed: int = 0
    last: float = 0.0


@dataclass
class ScientistAttributionConfig:
    """MECH-276 buffer knobs. Defaults are no-op-friendly."""

    # Counterfactual-contrast threshold above which an attribution is treated as
    # counterfactual-backed (a discriminating intervention). Records below it
    # are correlational.
    cf_margin: float = 0.05
    # When True, correlational (contrast < cf_margin) records are NOT buffered
    # (the structured feedstock). False = correlational-control arm.
    only_counterfactual_backed: bool = True
    # EMA rate per record applied to the per-region attribution.
    ema_alpha: float = 0.3
    # Per-cycle multiplicative decay applied to each region EMA at
    # decay_cycle(). 1.0 = no decay.
    decay: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1]; got {self.ema_alpha}"
            )
        if self.decay <= 0.0:
            raise ValueError(f"decay must be > 0; got {self.decay}")
        if self.cf_margin < 0.0:
            raise ValueError(f"cf_margin must be >= 0; got {self.cf_margin}")


class ScientistAttributionBuffer:
    """MECH-276 waking-phase counterfactual-backed attribution feedstock."""

    # Sentinel region carrying the global-mean attribution for sleep-loop
    # fallback lookups (a routed region not visited this waking phase).
    GLOBAL_REGION: RegionKey = ("__global__", "")

    def __init__(self, config: Optional[ScientistAttributionConfig] = None) -> None:
        self._config = config or ScientistAttributionConfig()
        self._records: Dict[DomainRegionKey, _RegionRecord] = {}

        # Diagnostics.
        self._n_records: int = 0
        self._n_counterfactual_backed: int = 0
        self._n_correlational_skipped: int = 0
        self._n_simulation_skipped: int = 0
        self._sum_attribution: float = 0.0
        self._sum_cf_contrast: float = 0.0
        self._n_cf_contrast_seen: int = 0
        self._n_decays: int = 0

    # ------------------------------------------------------------------ #
    # Configuration introspection                                        #
    # ------------------------------------------------------------------ #
    @property
    def config(self) -> ScientistAttributionConfig:
        return self._config

    # ------------------------------------------------------------------ #
    # Record                                                             #
    # ------------------------------------------------------------------ #
    def record(
        self,
        domain: str,
        region: RegionKey,
        attribution: float,
        counterfactual_contrast: float,
        simulation_mode: bool = False,
    ) -> bool:
        """Buffer one waking-phase attribution.

        Args:
            domain: "place" (E2WorldForward) or "self" (E2HarmSForward).
            region: (scale, segment_id) the attribution belongs to.
            attribution: the comparator agency residual magnitude
                (||z_observed - E2(z_prev, a_actual)||).
            counterfactual_contrast: ||E2(z_prev, a_actual) - E2(z_prev, a_cf)||;
                >= cf_margin -> counterfactual-backed (discriminating action).
            simulation_mode: MECH-094 gate (True -> no-op, returns False).

        Returns:
            True if the record was buffered; False if it was skipped
            (simulation tick, or correlational under only_counterfactual_backed).
        """
        if simulation_mode:
            self._n_simulation_skipped += 1
            return False

        attribution = float(attribution)
        counterfactual_contrast = float(counterfactual_contrast)
        self._sum_cf_contrast += counterfactual_contrast
        self._n_cf_contrast_seen += 1

        is_counterfactual_backed = (
            counterfactual_contrast >= float(self._config.cf_margin)
        )

        if self._config.only_counterfactual_backed and not is_counterfactual_backed:
            self._n_correlational_skipped += 1
            return False

        key: DomainRegionKey = (str(domain), (str(region[0]), str(region[1])))
        rec = self._records.get(key)
        if rec is None:
            rec = _RegionRecord(ema=attribution, n=0, n_counterfactual_backed=0, last=attribution)
            self._records[key] = rec
        else:
            alpha = float(self._config.ema_alpha)
            rec.ema = (1.0 - alpha) * rec.ema + alpha * attribution

        rec.n += 1
        rec.last = attribution
        if is_counterfactual_backed:
            rec.n_counterfactual_backed += 1
            self._n_counterfactual_backed += 1

        self._n_records += 1
        self._sum_attribution += attribution
        return True

    # ------------------------------------------------------------------ #
    # Evidence snapshot (the sleep-loop feedstock)                       #
    # ------------------------------------------------------------------ #
    def evidence_snapshot(self) -> Dict[RegionKey, float]:
        """Region -> attribution map for the MECH-275 aggregator evidence path.

        Merges across domains (the v1 single-snapshot integration): a region
        present in multiple domains contributes the mean of its domain EMAs.
        A GLOBAL_REGION sentinel carries the global mean of all region
        attributions for sleep-loop fallback lookups.

        Returns an empty dict (plus no sentinel) when nothing has been
        buffered, so the sleep loop falls back to a 0.0 prior pull.
        """
        if not self._records:
            return {}

        # Sum/count per region (scale, segment_id) across domains.
        per_region_sum: Dict[RegionKey, float] = {}
        per_region_n: Dict[RegionKey, int] = {}
        for (_domain, region), rec in self._records.items():
            per_region_sum[region] = per_region_sum.get(region, 0.0) + rec.ema
            per_region_n[region] = per_region_n.get(region, 0) + 1

        snapshot: Dict[RegionKey, float] = {
            region: per_region_sum[region] / max(1, per_region_n[region])
            for region in per_region_sum
        }

        if snapshot:
            snapshot[self.GLOBAL_REGION] = sum(snapshot.values()) / len(snapshot)
        return snapshot

    def lookup(self, region: RegionKey, snapshot: Optional[Dict[RegionKey, float]] = None) -> float:
        """Look up the attribution for a region, falling back to the global mean.

        Convenience for the sleep loop: returns the region's attribution if
        present, else the GLOBAL_REGION sentinel, else 0.0.
        """
        snap = snapshot if snapshot is not None else self.evidence_snapshot()
        if not snap:
            return 0.0
        rk = (str(region[0]), str(region[1]))
        if rk in snap:
            return float(snap[rk])
        return float(snap.get(self.GLOBAL_REGION, 0.0))

    # ------------------------------------------------------------------ #
    # Decay + reset                                                      #
    # ------------------------------------------------------------------ #
    def decay_cycle(self) -> None:
        """Apply per-cycle multiplicative decay to every region EMA."""
        decay = float(self._config.decay)
        self._n_decays += 1
        if decay == 1.0:
            return
        for rec in self._records.values():
            rec.ema = rec.ema * decay

    def reset(self) -> None:
        """Drop all buffered attributions and zero diagnostics."""
        self._records = {}
        self._n_records = 0
        self._n_counterfactual_backed = 0
        self._n_correlational_skipped = 0
        self._n_simulation_skipped = 0
        self._sum_attribution = 0.0
        self._sum_cf_contrast = 0.0
        self._n_cf_contrast_seen = 0
        self._n_decays = 0

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #
    @property
    def n_records(self) -> int:
        return int(self._n_records)

    @property
    def n_counterfactual_backed(self) -> int:
        return int(self._n_counterfactual_backed)

    @property
    def n_correlational_skipped(self) -> int:
        return int(self._n_correlational_skipped)

    @property
    def n_regions(self) -> int:
        return len({region for (_d, region) in self._records})

    def get_metrics(self) -> Dict[str, float]:
        """Flat metrics dict suitable for a manifest / SleepCycleState merge."""
        n_rec = max(1, self._n_records)
        n_contrast = max(1, self._n_cf_contrast_seen)
        return {
            "mech276_n_records": float(self._n_records),
            "mech276_n_counterfactual_backed": float(self._n_counterfactual_backed),
            "mech276_n_correlational_skipped": float(self._n_correlational_skipped),
            "mech276_n_simulation_skipped": float(self._n_simulation_skipped),
            "mech276_n_regions": float(self.n_regions),
            "mech276_mean_attribution": float(self._sum_attribution / n_rec),
            "mech276_mean_cf_contrast": float(self._sum_cf_contrast / n_contrast),
            "mech276_counterfactual_backed_fraction": float(
                self._n_counterfactual_backed / n_rec
            ),
            "mech276_n_decays": float(self._n_decays),
        }
