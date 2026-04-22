"""InvalidationTrigger -- MECH-287 substrate (Phase 2 iv).

Per-tick broadcast invalidation trigger. Subscribes to MECH-288 BoundaryEvents
emitted by the hierarchical event segmenter and re-emits them as graded
BroadcastEvent objects for downstream MECH-269 anchor-set reset and
MECH-284 staleness accumulator consumers.

Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
Claim:      MECH-287 (candidate v3_pending) in
            REE_assembly/docs/claims/claims.yaml

Verdict-3 architectural commitment (option c, V_s foundation lit-pull):
The trigger is a BoundaryEvent subscriber, NOT an independent comparator.
There is no latent-mismatch detector inside the trigger. The upstream
CA1/CA3 mismatch comparator substrate (Vinogradova 2001; O'Mara 2009;
Lisman & Grace 2005) is collapsed here to a subscription on the MECH-288
boundary queue -- whether to refactor MECH-287's claim text to make this
explicit is a downstream governance decision.

Consequences:
  - Dissociation test (C5): lesioning the event segmenter (disabling
    MECH-288) makes the trigger silent regardless of any internal state.
  - Trigger output is strictly a function of (boundary_events, config).
    No PE or latent input.

Guardrail -- phasic vs tonic broadcast (Aston-Jones & Cohen 2005;
Clewett 2025 failure signature 2): sustained high-frequency boundary
activity raises a tonic estimate. Once tonic exceeds threshold, the NEXT
phasic broadcast is suppressed; the mechanism prevents the trigger from
broadcasting continuously during chaotic regimes where every tick looks
like a boundary (clinical analogue: high-anxiety states with
"everything seems wrong"). The tonic signal decays passively via the
rolling-window estimate.

Graded output: broadcast_strength = posterior * gain. Posterior is
inherited from the BoundaryEvent (in [0, 1]); gain is a config scalar.
NO binary thresholding is applied to strength -- downstream consumers
see the graded posterior-weighted strength. Suppression gating is the
only binary operation; it is a whole-broadcast kill, not a threshold on
strength.

MECH-094: the trigger is called only from agent.sense() (the waking
observation stream), the same path that ticks the segmenter. Replay /
simulation paths that emit forced BoundaryEvents via
EventSegmenter.force_boundary() would still flow through here; that is
intentional -- forced boundaries are treated as real broadcast triggers
by design (the caller is responsible for the MECH-094 gate at the
force-boundary call site). No hypothesis_tag check is made inside this
module.

No trainable parameters. Pure arithmetic. No phased training.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence

# BoundaryEvent is imported lazily inside methods to avoid a circular
# import between ree_core.regulators and ree_core.hippocampal when this
# module is loaded before the hippocampal package finishes initialising.


# ------------------------------------------------------------------ #
# Output dataclass                                                   #
# ------------------------------------------------------------------ #

@dataclass
class BroadcastEvent:
    """Broadcast event emitted by the MECH-287 trigger.

    Payload mirrors the upstream BoundaryEvent enough that consumers can
    key on (source_scale, source_segment_id_new) without re-reading the
    boundary queue.
    """
    t: int
    strength: float             # posterior * gain, in [0, gain]
    posterior: float            # inherited from BoundaryEvent (in [0, 1])
    targets: List[str]          # configured broadcast targets
    source_scale: str           # "fast" | "slow" | (forced)
    source_segment_id_old: str
    source_segment_id_new: str
    source_sources: List[str]   # original BoundaryEvent.sources


# ------------------------------------------------------------------ #
# InvalidationTrigger                                                #
# ------------------------------------------------------------------ #

class InvalidationTrigger:
    """MECH-287 broadcast invalidation trigger.

    Public API:
      step(boundary_events, t) -> List[BroadcastEvent]
        Consume the BoundaryEvents emitted on tick t. Update the tonic
        estimate from this tick's activity. Emit one BroadcastEvent per
        boundary (with strength = posterior * gain) unless the tonic
        estimate (measured BEFORE this tick) exceeds tonic_threshold --
        in which case the whole tick's phasic broadcast is suppressed.

      reset()
        Clear tonic history and counters. Called per-episode.

      get_stats() -> dict
        Diagnostic snapshot: n_broadcast, n_suppressed, tonic_estimate,
        last_tick.

    State (per episode):
      _tonic_history: rolling window of per-tick aggregated posterior
                      (one scalar per tick seen, capped at tonic_window).
      _tonic_estimate: last computed rolling-mean estimate (populated on
                       each step()).
      _n_broadcast:    cumulative BroadcastEvents emitted.
      _n_suppressed:   cumulative BoundaryEvents that would have fired a
                       broadcast but were suppressed by the tonic gate.
      _last_tick:      last t seen (diagnostic; -1 before any tick).
    """

    def __init__(self, config: "InvalidationTriggerConfig"):  # noqa: F821
        self.config = config
        self._tonic_history: Deque[float] = deque(maxlen=int(config.tonic_window))
        self._tonic_estimate: float = 0.0
        self._n_broadcast: int = 0
        self._n_suppressed: int = 0
        self._last_tick: int = -1

    def reset(self) -> None:
        self._tonic_history.clear()
        self._tonic_estimate = 0.0
        self._n_broadcast = 0
        self._n_suppressed = 0
        self._last_tick = -1

    @property
    def tonic_estimate(self) -> float:
        return float(self._tonic_estimate)

    @property
    def n_broadcast(self) -> int:
        return int(self._n_broadcast)

    @property
    def n_suppressed(self) -> int:
        return int(self._n_suppressed)

    def get_stats(self) -> dict:
        return {
            "n_broadcast": self._n_broadcast,
            "n_suppressed": self._n_suppressed,
            "tonic_estimate": float(self._tonic_estimate),
            "last_tick": self._last_tick,
            "tonic_threshold": float(self.config.tonic_threshold),
            "gain": float(self.config.gain),
        }

    def step(
        self,
        boundary_events: Sequence,  # Sequence[BoundaryEvent]
        t: int,
    ) -> List[BroadcastEvent]:
        """Tick the trigger.

        Tonic estimate is computed from history BEFORE this tick, then
        this tick's activity is appended for the next tick's gate. This
        ordering maps to the "high-tonic period suppresses next phasic"
        failure signature -- a burst of boundary activity on tick t
        cannot simultaneously raise the tonic estimate AND gate itself.
        Suppression of tick t fires off the tonic estimate accumulated
        over ticks < t.
        """
        # 1. Tonic estimate from history up to (and NOT including) this tick.
        if self._tonic_history:
            self._tonic_estimate = (
                sum(self._tonic_history) / len(self._tonic_history)
            )
        else:
            self._tonic_estimate = 0.0

        # 2. This tick's aggregated activity (sum of posteriors across
        #    the tick's boundary events). Appended AFTER the gate decision.
        tick_activity = sum(float(ev.posterior) for ev in boundary_events)
        self._tonic_history.append(float(tick_activity))

        self._last_tick = int(t)

        # 3. No boundary events -> nothing to broadcast. Verdict-3
        #    dissociation: the trigger has no independent mismatch
        #    detector, so it is silent whenever the segmenter is silent.
        if not boundary_events:
            return []

        # 4. Tonic gate. If the rolling mean exceeds threshold, suppress
        #    the whole phasic broadcast for this tick. Each suppressed
        #    boundary increments the diagnostic counter so downstream
        #    monitoring can surface high-tonic regimes.
        if self._tonic_estimate > float(self.config.tonic_threshold):
            self._n_suppressed += len(boundary_events)
            return []

        # 5. Emit one BroadcastEvent per BoundaryEvent.
        gain = float(self.config.gain)
        targets = list(self.config.targets)
        events_out: List[BroadcastEvent] = []
        for ev in boundary_events:
            posterior = float(ev.posterior)
            strength = posterior * gain
            events_out.append(BroadcastEvent(
                t=int(t),
                strength=strength,
                posterior=posterior,
                targets=list(targets),
                source_scale=str(ev.scale),
                source_segment_id_old=str(ev.segment_id_old),
                source_segment_id_new=str(ev.segment_id_new),
                source_sources=list(ev.sources),
            ))
            self._n_broadcast += 1
        return events_out
