"""EventSegmenter -- MECH-288 substrate.

Per-tick hierarchical event-boundary detector emitting monotonic segment
IDs in nested "outer.inner" form. Feeds region-keyed substrates (MECH-269
anchor sets, MECH-284 staleness accumulator, MECH-287 broadcast trigger).

Design doc: REE_assembly/docs/architecture/event_segmenter.md
Claim:      MECH-288 (candidate v3_pending) in
            REE_assembly/docs/claims/claims.yaml

Two-level hierarchy (default):
  fast scale -- PE-threshold on sliding-window z-score over
                z_world + z_self. Boundary when pe_z > pe_threshold.
  slow scale -- BOCPD-Gaussian on z_goal (hazard=1/40, posterior
                pruned to top-k=20 run-lengths).

Cross-scale rule: when slow fires, inner resets to 0 (slow forces a
fast reset). Fast fires increment inner only.

MECH-094: simulation / replay content must not advance segment IDs.
The segmenter does NOT enforce this itself -- the caller (agent.sense)
invokes step() only on the waking observation stream. Replay paths
that need segment IDs can pass through force_boundary() with an
explicit reason, which is a logged API hook rather than a latent-driven
fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import math

import torch


# ------------------------------------------------------------------ #
# Dataclasses                                                        #
# ------------------------------------------------------------------ #

@dataclass
class BoundaryEvent:
    """Boundary event emitted at an event-segment transition.

    segment_id strings are "outer.inner" -- consumers parse them
    according to scale_id_format.
    """
    segment_id_old: str
    segment_id_new: str
    scale: str            # "fast" | "slow"
    posterior: float      # graded boundary strength in [0, 1]
    sources: List[str]    # contributing streams or "force"/"force:<reason>"
    t: int                # tick at which the boundary fired


@dataclass
class Scale:
    """Configuration for a single detection scale.

    Fields not applicable to a given algorithm are ignored:
      algorithm == "pe_threshold"    uses pe_threshold (+ window_length)
      algorithm == "bocpd_gaussian"  uses hazard + posterior_threshold
                                     (+ top_k, prior_var)
    """
    name: str
    streams: Tuple[str, ...]
    algorithm: str
    tau: int
    min_segment_length: int
    # pe_threshold algorithm
    pe_threshold: Optional[float] = None
    window_length: int = 200
    # bocpd_gaussian algorithm
    hazard: Optional[float] = None
    posterior_threshold: Optional[float] = None
    top_k: int = 20
    prior_var: float = 1.0


# ------------------------------------------------------------------ #
# Per-scale detector helpers                                         #
# ------------------------------------------------------------------ #

class _PEThresholdDetector:
    """Sliding-window z-score on a scalar PE signal.

    The input per step is an aggregated PE magnitude over the scale's
    registered streams:
      agg(t) = sum_s ||z_s(t) - z_s(t-1)||  (fallback when pe_dict
                                              does not supply the stream)
    or
      agg(t) = sum_s pe_dict[stream]        (when caller supplies
                                              external PEs, e.g. from
                                              forward-model residuals)
    Under either source, the window statistics normalise the signal
    so the threshold has a stable meaning.

    posterior = clip_[0,1]((pe_z - threshold) / (1 + |pe_z - threshold|)
                           + 0.5)  -> graded strength, 0.5 at threshold,
    monotonically rising above.
    """

    def __init__(self, window_length: int, pe_threshold: float):
        self.window_length = max(1, int(window_length))
        self.pe_threshold = float(pe_threshold)
        self._window: List[float] = []
        self._prev_streams: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self._window.clear()
        self._prev_streams.clear()

    def step(
        self,
        streams: Tuple[str, ...],
        latent_dict: Dict[str, Optional[torch.Tensor]],
        pe_dict: Dict[str, float],
    ) -> Tuple[bool, float, List[str]]:
        """Return (fired, posterior, sources) for the current tick."""
        agg = 0.0
        sources: List[str] = []
        for name in streams:
            if name in pe_dict:
                val = float(pe_dict[name])
                agg += val
                sources.append(name)
                continue
            z = latent_dict.get(name)
            if z is None:
                continue
            z_d = z.detach()
            prev = self._prev_streams.get(name)
            if prev is not None:
                diff = float((z_d - prev).norm().item())
                agg += diff
                sources.append(name)
            self._prev_streams[name] = z_d
        # Not enough window yet: seed with observation, do not fire.
        if len(self._window) < 2:
            self._window.append(agg)
            if len(self._window) > self.window_length:
                self._window.pop(0)
            return False, 0.0, sources
        mu = sum(self._window) / len(self._window)
        var = sum((x - mu) ** 2 for x in self._window) / len(self._window)
        sigma = math.sqrt(var) + 1e-6
        pe_z = (agg - mu) / sigma
        self._window.append(agg)
        if len(self._window) > self.window_length:
            self._window.pop(0)
        fired = pe_z > self.pe_threshold
        if fired:
            margin = pe_z - self.pe_threshold
            post = 0.5 + margin / (1.0 + abs(margin))
            posterior = max(0.0, min(1.0, post))
        else:
            posterior = 0.0
        return bool(fired), float(posterior), sources


class _BOCPDGaussianDetector:
    """Bayesian Online Change-Point Detection with Gaussian likelihood.

    Adams & MacKay 2007. Observations are scalar: per tick we take
    ||z_stream(t)|| for each registered stream and sum. Per-segment
    Gaussian with running mean / variance. Run-length posterior pruned
    to top-k to keep O(1) per step.

    Fires when P(r_t = 0) exceeds posterior_threshold. Posterior field
    on the emitted BoundaryEvent is P(r_t = 0) itself.
    """

    def __init__(
        self,
        hazard: float,
        posterior_threshold: float,
        top_k: int = 20,
        prior_var: float = 1.0,
    ):
        self.hazard = float(hazard)
        self.posterior_threshold = float(posterior_threshold)
        self.top_k = max(1, int(top_k))
        self.prior_var = float(prior_var)
        # run_lengths[i] = integer run-length (0 is "just changed")
        # run_probs[i]   = posterior prob
        # run_mean[i]    = running mean of observations in that segment
        # run_m2[i]      = sum of squared deviations (Welford) for variance
        self._run_lengths: List[int] = []
        self._run_probs: List[float] = []
        self._run_mean: List[float] = []
        self._run_m2: List[float] = []
        self._initialized = False

    def reset(self) -> None:
        self._run_lengths.clear()
        self._run_probs.clear()
        self._run_mean.clear()
        self._run_m2.clear()
        self._initialized = False

    def _predictive_log_prob(self, x: float, idx: int) -> float:
        """Gaussian predictive log-prob for a given run index.

        Uses running mean and Welford variance estimate for that segment.
        Falls back to prior_var for short segments (n < 2).
        """
        n = self._run_lengths[idx]
        mu = self._run_mean[idx]
        if n < 2:
            var = self.prior_var
        else:
            var = self._run_m2[idx] / n
            if var <= 0.0:
                var = self.prior_var
        var = max(var, 1e-6)
        diff = x - mu
        return -0.5 * (math.log(2.0 * math.pi * var) + diff * diff / var)

    def step(
        self,
        streams: Tuple[str, ...],
        latent_dict: Dict[str, Optional[torch.Tensor]],
        pe_dict: Dict[str, float],  # unused for BOCPD, kept for API symmetry
    ) -> Tuple[bool, float, List[str]]:
        agg = 0.0
        sources: List[str] = []
        for name in streams:
            z = latent_dict.get(name)
            if z is None:
                continue
            z_d = z.detach()
            agg += float(z_d.norm().item())
            sources.append(name)
        if not sources:
            return False, 0.0, sources

        if not self._initialized:
            # Seed with run-length 0 at probability 1.
            self._run_lengths = [0]
            self._run_probs = [1.0]
            self._run_mean = [agg]
            self._run_m2 = [0.0]
            self._initialized = True
            return False, 1.0, sources  # first tick is trivially a boundary

        # 1. Predictive log-prob for every existing run.
        n_runs = len(self._run_lengths)
        pred_log = [self._predictive_log_prob(agg, i) for i in range(n_runs)]

        # Implausible-under-all-runs fast path: if every existing run
        # assigns negligible log-probability to the observation, the
        # floating-point survivors (tiny but nonzero exp values) produce
        # a hazard-dominated posterior that fails to fire. Detect the
        # regime by the max(pred_log) cutoff and treat it as a decisive
        # change-point.
        if pred_log and max(pred_log) < -20.0:
            self._run_lengths = [0]
            self._run_probs = [1.0]
            self._run_mean = [agg]
            self._run_m2 = [0.0]
            return True, 1.0, sources

        # 2. Growth + change-point probabilities.
        # Growth: r increments, no change-point.
        growth_probs = [
            self._run_probs[i] * math.exp(pred_log[i]) * (1.0 - self.hazard)
            for i in range(n_runs)
        ]
        # Change-point prob at r=0 = sum_i P(r_{t-1}=i) * p(x|r_{t-1}=i) * hazard
        cp_prob = sum(
            self._run_probs[i] * math.exp(pred_log[i]) * self.hazard
            for i in range(n_runs)
        )

        # 3. Build new run distribution: [r=0, r=old+1 for each i].
        new_run_lengths = [0] + [r + 1 for r in self._run_lengths]
        new_run_probs = [cp_prob] + growth_probs
        # Updated Welford statistics for each run.
        new_run_mean: List[float] = [agg]
        new_run_m2: List[float] = [0.0]
        for i in range(n_runs):
            n_old = self._run_lengths[i]
            mu_old = self._run_mean[i]
            m2_old = self._run_m2[i]
            n_new = n_old + 1
            delta = agg - mu_old
            mu_new = mu_old + delta / n_new
            m2_new = m2_old + delta * (agg - mu_new)
            new_run_mean.append(mu_new)
            new_run_m2.append(m2_new)

        # 4. Normalise.
        total = sum(new_run_probs)
        if total <= 0.0:
            # Likelihoods underflowed to zero across every existing run --
            # the observation is effectively "impossible" under any current
            # segment hypothesis. That IS the signature of a decisive
            # change-point, so fire with posterior 1.0 and reseed the
            # posterior with a fresh run at r=0.
            self._run_lengths = [0]
            self._run_probs = [1.0]
            self._run_mean = [agg]
            self._run_m2 = [0.0]
            return True, 1.0, sources
        new_run_probs = [p / total for p in new_run_probs]

        # 5. Prune to top-k by posterior probability.
        if len(new_run_probs) > self.top_k:
            order = sorted(
                range(len(new_run_probs)),
                key=lambda i: new_run_probs[i],
                reverse=True,
            )[: self.top_k]
            order.sort()
            new_run_lengths = [new_run_lengths[i] for i in order]
            new_run_probs = [new_run_probs[i] for i in order]
            new_run_mean = [new_run_mean[i] for i in order]
            new_run_m2 = [new_run_m2[i] for i in order]
            # Re-normalise after pruning.
            total = sum(new_run_probs)
            if total > 0.0:
                new_run_probs = [p / total for p in new_run_probs]

        self._run_lengths = new_run_lengths
        self._run_probs = new_run_probs
        self._run_mean = new_run_mean
        self._run_m2 = new_run_m2

        # 6. Fire if P(r=0) exceeds threshold.
        try:
            idx_zero = self._run_lengths.index(0)
            p_zero = self._run_probs[idx_zero]
        except ValueError:
            p_zero = 0.0
        fired = p_zero > self.posterior_threshold
        return bool(fired), float(p_zero), sources


# ------------------------------------------------------------------ #
# EventSegmenter                                                     #
# ------------------------------------------------------------------ #

class EventSegmenter:
    """Hierarchical event-segment detector (MECH-288).

    Default two-scale config (fast PE-threshold on z_world+z_self; slow
    BOCPD-Gaussian on z_goal) is defined via EventSegmenterConfig in
    ree_core.utils.config. Scale order: the caller passes `scales` in
    any order; the segmenter locates the "slow" (outer) scale by name
    -- its fire increments outer and resets inner. All other scales
    increment inner only.

    segment_id_format "{outer}.{inner}" is the default. Consumers may
    pass a different format string, but the current implementation
    always produces two numeric fields separated by ".".
    """

    def __init__(
        self,
        scales: List[Scale],
        emit_to: Optional[List[str]] = None,
        scale_id_format: str = "{outer}.{inner}",
        slow_scale_name: str = "slow",
    ):
        if not scales:
            raise ValueError("EventSegmenter requires at least one scale.")
        self.scales: List[Scale] = list(scales)
        self.emit_to: List[str] = list(emit_to or [])
        self.scale_id_format: str = scale_id_format
        self.slow_scale_name: str = slow_scale_name
        # Build per-scale detector instances.
        self._detectors: Dict[str, Any] = {}
        for sc in self.scales:
            self._detectors[sc.name] = self._build_detector(sc)
        # Track last-fire tick per scale for min_segment_length guard.
        self._last_fire_t: Dict[str, int] = {sc.name: -10**9 for sc in self.scales}
        # Hierarchical segment counters.
        self._outer: int = 0
        self._inner: int = 0
        self._t_last_seen: int = -1

    # ------------------------------------------------------------------ #
    # Construction                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_detector(scale: Scale) -> Any:
        algo = scale.algorithm
        if algo == "pe_threshold":
            if scale.pe_threshold is None:
                raise ValueError(
                    f"Scale {scale.name}: pe_threshold required for "
                    "pe_threshold algorithm"
                )
            return _PEThresholdDetector(
                window_length=scale.window_length,
                pe_threshold=scale.pe_threshold,
            )
        if algo == "bocpd_gaussian":
            if scale.hazard is None or scale.posterior_threshold is None:
                raise ValueError(
                    f"Scale {scale.name}: hazard and posterior_threshold "
                    "required for bocpd_gaussian algorithm"
                )
            return _BOCPDGaussianDetector(
                hazard=scale.hazard,
                posterior_threshold=scale.posterior_threshold,
                top_k=scale.top_k,
                prior_var=scale.prior_var,
            )
        raise ValueError(f"Unknown algorithm: {algo}")

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def current_segment_id(self) -> str:
        return self.scale_id_format.format(outer=self._outer, inner=self._inner)

    def reset(self) -> None:
        for det in self._detectors.values():
            det.reset()
        self._last_fire_t = {sc.name: -10**9 for sc in self.scales}
        self._outer = 0
        self._inner = 0
        self._t_last_seen = -1

    def step(
        self,
        latent_dict: Dict[str, Optional[torch.Tensor]],
        pe_dict: Optional[Dict[str, float]],
        t: int,
    ) -> List[BoundaryEvent]:
        """Tick the segmenter once. Returns the list of BoundaryEvents fired.

        Evaluation order: slow first, then fast. Slow-fire resets inner
        and suppresses a fast event on the same tick (slow forces the
        inner reset; emitting both would double-count the boundary
        signal and mislead downstream consumers).
        """
        pe_dict = pe_dict or {}
        events: List[BoundaryEvent] = []
        # Slow scales first.
        slow_fired = False
        for sc in self.scales:
            if sc.name != self.slow_scale_name:
                continue
            det = self._detectors[sc.name]
            fired, posterior, sources = det.step(
                tuple(sc.streams), latent_dict, pe_dict
            )
            if fired and (t - self._last_fire_t[sc.name]) >= sc.min_segment_length:
                old_id = self.current_segment_id()
                self._outer += 1
                self._inner = 0
                new_id = self.current_segment_id()
                self._last_fire_t[sc.name] = t
                events.append(BoundaryEvent(
                    segment_id_old=old_id,
                    segment_id_new=new_id,
                    scale=sc.name,
                    posterior=float(posterior),
                    sources=list(sources),
                    t=int(t),
                ))
                slow_fired = True

        # Non-slow scales.
        for sc in self.scales:
            if sc.name == self.slow_scale_name:
                continue
            det = self._detectors[sc.name]
            fired, posterior, sources = det.step(
                tuple(sc.streams), latent_dict, pe_dict
            )
            if slow_fired:
                # Slow fire already reset inner; do not emit a second
                # event this tick. Still record last-fire so the fast
                # detector's min_segment_length suppression remains
                # correctly anchored.
                if fired:
                    self._last_fire_t[sc.name] = t
                continue
            if fired and (t - self._last_fire_t[sc.name]) >= sc.min_segment_length:
                old_id = self.current_segment_id()
                self._inner += 1
                new_id = self.current_segment_id()
                self._last_fire_t[sc.name] = t
                events.append(BoundaryEvent(
                    segment_id_old=old_id,
                    segment_id_new=new_id,
                    scale=sc.name,
                    posterior=float(posterior),
                    sources=list(sources),
                    t=int(t),
                ))

        self._t_last_seen = t
        return events

    def force_boundary(self, scale: str, reason: str, t: Optional[int] = None) -> BoundaryEvent:
        """Emit an explicit boundary event (supervised / scripted path).

        Bypasses the detector's latent-driven fire logic AND the
        min_segment_length suppression (callers use this as an API hook
        for task-marker injection; the contract is that the caller knows
        what they are doing).

        Increments counters per the scale-name rule: slow forces outer+1
        / inner=0; any other scale increments inner only.
        """
        if scale not in self._detectors:
            raise ValueError(f"Unknown scale: {scale}")
        if t is None:
            t = self._t_last_seen + 1 if self._t_last_seen >= 0 else 0
        old_id = self.current_segment_id()
        if scale == self.slow_scale_name:
            self._outer += 1
            self._inner = 0
        else:
            self._inner += 1
        self._last_fire_t[scale] = int(t)
        new_id = self.current_segment_id()
        return BoundaryEvent(
            segment_id_old=old_id,
            segment_id_new=new_id,
            scale=scale,
            posterior=1.0,
            sources=[f"force:{reason}"],
            t=int(t),
        )
