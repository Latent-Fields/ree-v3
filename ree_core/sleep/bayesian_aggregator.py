"""
BayesianAggregator -- MECH-275 Phase D substrate.

Per-domain per-region Gaussian posterior over residuals, updated from
probe-channel-routed replay events emitted by the MECH-272 RoutingGate.
Conjugate mean-and-variance update; standard arithmetic.

Design contract (REE_assembly/docs/architecture/sleep_aggregation_cluster.md
sections "Bayesian aggregator" and "Phase ordering within a sleep cycle"):

  * PER-DOMAIN PER-REGION. Posteriors keyed on (domain, region) where
    region = (scale, segment_id) (matches MECH-284 RegionKey). V3
    domains: "place" (this substrate; Bayesian upgrade of MECH-284's
    leaky integrator) and "self" (MECH-273 Phase E specialisation).
    "object" / "other" are V4-deferred.

  * CONJUGATE UPDATE. For posterior N(mu, sigma^2) at (domain, region)
    and an evidence value x with likelihood variance sigma_lik^2 and a
    weight w = probe_channel * probe_gain:

        tau          = 1 / sigma^2
        tau_lik      = w / sigma_lik^2         # weighted likelihood prec
        tau'         = tau + tau_lik
        mu'          = (tau * mu + tau_lik * x) / tau'
        sigma'^2     = 1 / tau'
        n_replays   += 1                        # only when w > 0

    When w <= 0 (probe-only ablation OFF, or routing-gate WAKING row),
    no update fires. The aggregator runs the whole pipeline but emits
    no posterior change -- diagnostic-only.

  * SNAPSHOT AT PHASE_SWITCH. snapshot() captures the current posterior
    map; a copy is preserved on _last_snapshot. Phase E writeback
    (MECH-273) reads this snapshot rather than the live posterior so
    REM-pass updates do not contaminate the SWS-only posterior the
    self-model writeback consumes.

  * PER-CYCLE DECAY. After snapshot() the posterior precision is
    multiplied by decay_factor (default 1.0 = no decay). Lower decay
    lets newer evidence overcome stale posteriors faster.

Phase D is a NO-OP CONSUMER. PosteriorUpdate messages emitted from
update() land on SleepCycleState.last_metrics as diagnostic counts +
domain-mean shifts. The Phase E self-model writeback (MECH-273) is the
first downstream consumer; "place" domain has no V3 consumer beyond
metrics.

Bit-identical OFF guarantee: the master flag use_mech275_aggregator
defaults False; when False, REEAgent never instantiates this class and
the SleepLoopManager runs exactly as in Phase C.

No trainable parameters. Pure float arithmetic + small dict state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Tuple

from ree_core.sleep.routing_gate import RoutedEvent


RegionKey = Tuple[str, str]   # (scale, segment_id) -- matches MECH-284
DomainRegionKey = Tuple[str, RegionKey]


@dataclass
class GaussianPosterior:
    """N(mean, variance) with sample count."""

    mean: float = 0.0
    variance: float = 1.0
    n: int = 0


@dataclass
class PosteriorUpdate:
    """Diagnostic record of one conjugate update.

    delta_mean = mean_after - mean_before
    delta_variance = variance_after - variance_before  (typically <= 0)
    n_replays = posterior.n after the update (cumulative for the region)
    """

    domain: str
    region: RegionKey
    delta_mean: float
    delta_variance: float
    n_replays: int


@dataclass
class BayesianAggregatorConfig:
    """Conjugate-update knobs. Defaults match the design-doc table."""

    domains: Tuple[str, ...] = ("place",)
    prior_mean: float = 0.0
    prior_variance: float = 1.0
    likelihood_variance: float = 1.0
    # Per-cycle multiplicative decay applied to posterior precision at
    # snapshot(). 1.0 = no decay.
    decay_factor: float = 1.0
    # Multiplier applied to probe_channel before forming the likelihood
    # weight w. probe_gain=0 -> aggregator never updates (diagnostic).
    probe_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.prior_variance <= 0.0:
            raise ValueError(
                f"prior_variance must be > 0; got {self.prior_variance}"
            )
        if self.likelihood_variance <= 0.0:
            raise ValueError(
                f"likelihood_variance must be > 0; got {self.likelihood_variance}"
            )
        if self.decay_factor <= 0.0:
            raise ValueError(
                f"decay_factor must be > 0; got {self.decay_factor}"
            )
        if self.probe_gain < 0.0:
            raise ValueError(
                f"probe_gain must be >= 0; got {self.probe_gain}"
            )
        if not self.domains:
            raise ValueError("domains tuple must not be empty")


class BayesianAggregator:
    """MECH-275 per-domain per-region Bayesian aggregator.

    Lifecycle within a sleep cycle:

        agg.update(routed, evidence, domain="place")  # SWS pass + REM
        ...                                           # re-route, gated
                                                      # on probe_channel
        agg.snapshot()                                # PHASE_SWITCH
        ...                                           # REM pass updates
                                                      # land on the live
                                                      # posterior; the
                                                      # snapshot is the
                                                      # SWS-only frozen
                                                      # copy.
        agg.reset()                                   # between cycles
                                                      # if desired.
    """

    def __init__(self, config: Optional[BayesianAggregatorConfig] = None) -> None:
        self._config = config or BayesianAggregatorConfig()
        self._posteriors: Dict[DomainRegionKey, GaussianPosterior] = {}
        self._last_snapshot: Optional[Dict[DomainRegionKey, GaussianPosterior]] = None

        # Diagnostics.
        self._n_updates: int = 0
        self._n_skipped_zero_probe: int = 0
        self._sum_weight: float = 0.0
        self._sum_anchor_channel_seen: float = 0.0
        self._sum_probe_channel_seen: float = 0.0
        self._n_routed_seen: int = 0
        self._n_snapshots: int = 0

    # ------------------------------------------------------------------ #
    # Configuration introspection                                        #
    # ------------------------------------------------------------------ #
    @property
    def config(self) -> BayesianAggregatorConfig:
        return self._config

    @property
    def domains(self) -> Tuple[str, ...]:
        return tuple(self._config.domains)

    # ------------------------------------------------------------------ #
    # Update                                                             #
    # ------------------------------------------------------------------ #
    def update(
        self,
        routed_event: RoutedEvent,
        evidence: float,
        domain: str = "place",
    ) -> Optional[PosteriorUpdate]:
        """Conjugate Gaussian update on (domain, region) keyed posterior.

        evidence is the scalar observation x. Region is extracted from
        routed_event.event (Anchor.key or a (scale, segment_id) tuple).
        Returns a PosteriorUpdate diagnostic or None when the event
        carries zero probe weight (or domain unknown / region missing).
        """
        self._n_routed_seen += 1
        self._sum_anchor_channel_seen += float(routed_event.anchor_channel)
        self._sum_probe_channel_seen += float(routed_event.probe_channel)

        if domain not in self._config.domains:
            return None

        region = self._region_from_event(routed_event.event)
        if region is None:
            return None

        weight = float(routed_event.probe_channel) * float(self._config.probe_gain)
        if weight <= 0.0:
            self._n_skipped_zero_probe += 1
            return None

        key = (domain, region)
        posterior = self._posteriors.get(key)
        if posterior is None:
            posterior = GaussianPosterior(
                mean=float(self._config.prior_mean),
                variance=float(self._config.prior_variance),
                n=0,
            )
            self._posteriors[key] = posterior

        mean_before = posterior.mean
        variance_before = posterior.variance

        tau = 1.0 / variance_before
        tau_lik = weight / float(self._config.likelihood_variance)
        tau_post = tau + tau_lik
        mean_after = (tau * mean_before + tau_lik * float(evidence)) / tau_post
        variance_after = 1.0 / tau_post

        posterior.mean = mean_after
        posterior.variance = variance_after
        posterior.n += 1

        self._n_updates += 1
        self._sum_weight += weight

        return PosteriorUpdate(
            domain=domain,
            region=region,
            delta_mean=mean_after - mean_before,
            delta_variance=variance_after - variance_before,
            n_replays=posterior.n,
        )

    def update_many(
        self,
        routed_events: Iterable[RoutedEvent],
        evidence_lookup,
        domain: str = "place",
    ) -> int:
        """Convenience: apply update() over an iterable. evidence_lookup is
        called as evidence_lookup(routed_event) -> float.
        Returns the number of posteriors that were actually updated.
        """
        n = 0
        for routed in routed_events:
            up = self.update(routed, evidence_lookup(routed), domain=domain)
            if up is not None:
                n += 1
        return n

    # ------------------------------------------------------------------ #
    # Snapshot + decay                                                   #
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[DomainRegionKey, GaussianPosterior]:
        """Capture the current posterior map and apply per-cycle decay.

        The captured snapshot is a deep copy preserved on _last_snapshot;
        decay is then applied to the live posterior (multiplicative on
        precision). Phase E writeback consumes the snapshot, NOT the
        live posterior, so REM-pass updates do not contaminate it.
        """
        snap = {
            key: GaussianPosterior(
                mean=p.mean, variance=p.variance, n=p.n
            )
            for key, p in self._posteriors.items()
        }
        self._last_snapshot = snap
        self._n_snapshots += 1

        decay = float(self._config.decay_factor)
        if decay != 1.0:
            for p in self._posteriors.values():
                # Decay multiplies precision; new variance = old / decay.
                p.variance = p.variance / decay
        return snap

    @property
    def last_snapshot(self) -> Optional[Dict[DomainRegionKey, GaussianPosterior]]:
        if self._last_snapshot is None:
            return None
        return {
            key: GaussianPosterior(
                mean=p.mean, variance=p.variance, n=p.n
            )
            for key, p in self._last_snapshot.items()
        }

    def reset(self) -> None:
        """Drop posteriors and zero diagnostics."""
        self._posteriors = {}
        self._last_snapshot = None
        self._n_updates = 0
        self._n_skipped_zero_probe = 0
        self._sum_weight = 0.0
        self._sum_anchor_channel_seen = 0.0
        self._sum_probe_channel_seen = 0.0
        self._n_routed_seen = 0
        self._n_snapshots = 0

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #
    @property
    def n_updates(self) -> int:
        return int(self._n_updates)

    @property
    def n_posteriors(self) -> int:
        return len(self._posteriors)

    def get_posterior(
        self, domain: str, region: RegionKey
    ) -> Optional[GaussianPosterior]:
        post = self._posteriors.get((domain, region))
        if post is None:
            return None
        # Defensive copy.
        return GaussianPosterior(
            mean=post.mean, variance=post.variance, n=post.n
        )

    def get_metrics(self) -> Dict[str, float]:
        """Flat metrics dict suitable for SleepCycleState.last_metrics."""
        n_seen = max(1, self._n_routed_seen)
        if self._posteriors:
            mean_post = sum(p.mean for p in self._posteriors.values()) / len(
                self._posteriors
            )
            mean_var = sum(p.variance for p in self._posteriors.values()) / len(
                self._posteriors
            )
        else:
            mean_post = 0.0
            mean_var = 0.0
        return {
            "mech275_n_updates": float(self._n_updates),
            "mech275_n_posteriors": float(len(self._posteriors)),
            "mech275_n_skipped_zero_probe": float(self._n_skipped_zero_probe),
            "mech275_n_snapshots": float(self._n_snapshots),
            "mech275_sum_weight": float(self._sum_weight),
            "mech275_mean_anchor_channel_seen": float(
                self._sum_anchor_channel_seen / n_seen
            ),
            "mech275_mean_probe_channel_seen": float(
                self._sum_probe_channel_seen / n_seen
            ),
            "mech275_mean_posterior_mean": float(mean_post),
            "mech275_mean_posterior_variance": float(mean_var),
        }

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _region_from_event(event: Any) -> Optional[RegionKey]:
        """Project the underlying replay event onto a (scale, segment_id) key.

        The replay event from MECH-285 is an Anchor with a 3-tuple key
        (scale, segment_id, stream_mixture). The aggregator drops the
        stream_mixture component to match MECH-284's RegionKey. Callers
        may also pass a bare 2-tuple (scale, segment_id) directly.
        """
        # Anchor with .key attribute.
        key_attr = getattr(event, "key", None)
        if key_attr is not None:
            if (
                isinstance(key_attr, tuple)
                and len(key_attr) >= 2
                and isinstance(key_attr[0], str)
                and isinstance(key_attr[1], str)
            ):
                return (key_attr[0], key_attr[1])
        # Plain tuple fallback.
        if (
            isinstance(event, tuple)
            and len(event) >= 2
            and isinstance(event[0], str)
            and isinstance(event[1], str)
        ):
            return (event[0], event[1])
        return None
