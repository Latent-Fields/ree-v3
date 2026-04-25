"""
SleepLoopManager: deterministic K-episode entry into the SD-017 sleep cycle.

Phase A of the sleep-aggregation cluster (see
REE_assembly/docs/architecture/sleep_aggregation_cluster.md). This module
provides ONLY the scaffolding:

  * SleepPhase enum (WAKING / SLEEP_ENTRY / SWS_ANALOG / PHASE_SWITCH /
    REM_ANALOG / WRITEBACK) -- the canonical phase ordering. Phase A only
    transitions through WAKING -> SWS_ANALOG -> REM_ANALOG -> WAKING via
    the existing REEAgent.run_sleep_cycle() convenience method; the
    intermediate PHASE_SWITCH / WRITEBACK states are reserved for Phase D
    (Bayesian aggregator) and Phase E (self-model writeback).

  * SleepCycleState dataclass: per-cycle counters and the current phase.

  * SleepLoopManager: tracks episodes_since_sleep, fires sleep cycles
    every cycle_every_k_episodes via the existing SD-017 surface on
    REEAgent. No new replay sampler, no routing gate, no aggregator, no
    self-model writeback. Those land in Phases B-E.

The manager is a no-op consumer of the existing surface: it does not
introduce any new mathematical content into the cycle. Bit-identical
waking semantics are preserved when the master flag is OFF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:  # pragma: no cover -- typing only
    from ree_core.agent import REEAgent
    from ree_core.sleep.bayesian_aggregator import BayesianAggregator
    from ree_core.sleep.replay_sampler import SleepReplaySampler
    from ree_core.sleep.routing_gate import RoutingGate


class SleepPhase(Enum):
    """Canonical phase ordering for the sleep-aggregation cluster.

    Phase A only visits WAKING / SWS_ANALOG / REM_ANALOG. The other
    phases are reserved tokens that later phases will populate:

      SLEEP_ENTRY    -- Phase B (replay sampler warm-up).
      PHASE_SWITCH   -- Phase D (Bayesian aggregator slot transition).
      WRITEBACK      -- Phase E (self-model offline gradient pass).
    """

    WAKING = "waking"
    SLEEP_ENTRY = "sleep_entry"
    SWS_ANALOG = "sws_analog"
    PHASE_SWITCH = "phase_switch"
    REM_ANALOG = "rem_analog"
    WRITEBACK = "writeback"


@dataclass
class SleepCycleState:
    """Per-cycle book-keeping. Reset at the start of each sleep cycle."""

    cycle_index: int = 0
    phase: SleepPhase = SleepPhase.WAKING
    episodes_since_sleep: int = 0
    last_metrics: Dict[str, float] = field(default_factory=dict)

    def reset_for_new_cycle(self, cycle_index: int) -> None:
        self.cycle_index = cycle_index
        self.phase = SleepPhase.WAKING
        self.last_metrics = {}


class SleepLoopManager:
    """
    Deterministic K-episode driver for the SD-017 sleep cycle.

    Phase A contract:
      * episodes_since_sleep is incremented by notify_episode_end().
      * When the counter reaches cycle_every_k_episodes AND the master
        flag is on AND the agent has at least one of sws_enabled /
        rem_enabled active, run_sleep_cycle is invoked through the
        agent's existing SD-017 entry point.
      * The phase field walks WAKING -> SWS_ANALOG -> REM_ANALOG ->
        WAKING during the cycle. Other phases are reserved.
      * Bit-identical OFF: the agent never instantiates this manager
        when use_sleep_loop is False, so notify_episode_end is never
        called and the existing waking pipeline is unchanged.
    """

    def __init__(
        self,
        cycle_every_k_episodes: int = 1,
        require_sleep_passes_enabled: bool = True,
        replay_sampler: Optional["SleepReplaySampler"] = None,
        draws_per_cycle: int = 0,
        routing_gate: Optional["RoutingGate"] = None,
        bayesian_aggregator: Optional["BayesianAggregator"] = None,
        aggregator_domain: str = "place",
    ) -> None:
        if cycle_every_k_episodes < 1:
            raise ValueError(
                "cycle_every_k_episodes must be >= 1; "
                f"got {cycle_every_k_episodes}"
            )
        if draws_per_cycle < 0:
            raise ValueError(
                "draws_per_cycle must be >= 0; "
                f"got {draws_per_cycle}"
            )
        self.cycle_every_k_episodes = int(cycle_every_k_episodes)
        self.require_sleep_passes_enabled = bool(require_sleep_passes_enabled)
        self.replay_sampler = replay_sampler
        self.draws_per_cycle = int(draws_per_cycle)
        self.routing_gate = routing_gate
        self.bayesian_aggregator = bayesian_aggregator
        self.aggregator_domain = str(aggregator_domain)
        self.state = SleepCycleState()
        self._cycle_history: List[Dict[str, float]] = []

    # -- public API --

    def notify_episode_end(self, agent: "REEAgent") -> Optional[Dict[str, float]]:
        """
        Called by REEAgent.reset() at the END of each episode (i.e. before
        the per-episode resets land for the next episode). Increments the
        counter and fires a sleep cycle when the counter reaches K.

        Returns the merged sleep-cycle metrics dict on a fired cycle, else
        None. Safe to call when the agent has no SD-017 passes enabled --
        the call is a no-op in that case.
        """
        self.state.episodes_since_sleep += 1
        if self.state.episodes_since_sleep < self.cycle_every_k_episodes:
            return None
        return self._run_cycle(agent)

    def force_cycle(self, agent: "REEAgent") -> Dict[str, float]:
        """
        Diagnostic / experiment hook: run a sleep cycle immediately,
        regardless of the K-episode counter. Resets the counter on
        completion. Honours require_sleep_passes_enabled.
        """
        return self._run_cycle(agent)

    def reset(self) -> None:
        """Hard reset (e.g. between training stages)."""
        self.state = SleepCycleState()
        self._cycle_history = []

    @property
    def cycle_history(self) -> List[Dict[str, float]]:
        return list(self._cycle_history)

    # -- internal --

    def _run_cycle(self, agent: "REEAgent") -> Optional[Dict[str, float]]:
        if self.require_sleep_passes_enabled and not (
            getattr(agent.config, "sws_enabled", False)
            or getattr(agent.config, "rem_enabled", False)
        ):
            # No SD-017 substrate to drive; reset counter and stay quiet.
            self.state.episodes_since_sleep = 0
            return None

        self.state.reset_for_new_cycle(self.state.cycle_index + 1)

        # Phase B: SLEEP_ENTRY -- freeze the staleness snapshot ONCE per
        # cycle and perform the configured N draws. Phase C: at SLEEP_ENTRY
        # the routing gate flips to the SWS row; each draw is routed
        # immediately. Phase D: each routed draw is consumed by the
        # Bayesian aggregator (probe-channel-gated). Place-domain evidence
        # = staleness scalar at the routed anchor's region, looked up
        # against a frozen-at-SLEEP_ENTRY copy of the staleness snapshot.
        sws_routed_draws: List = []
        evidence_snapshot: Dict = {}
        if self.replay_sampler is not None:
            self.state.phase = SleepPhase.SLEEP_ENTRY
            if self.routing_gate is not None:
                self.routing_gate.set_phase(SleepPhase.SWS_ANALOG)
            self.replay_sampler.freeze_snapshot()
            if self.bayesian_aggregator is not None:
                evidence_snapshot = self._build_evidence_snapshot(agent)
            for _ in range(self.draws_per_cycle):
                anchor = self.replay_sampler.draw()
                if anchor is None:
                    continue
                if self.routing_gate is not None:
                    routed = self.routing_gate.route(anchor)
                    sws_routed_draws.append(routed)
                    if self.bayesian_aggregator is not None:
                        self.bayesian_aggregator.update(
                            routed,
                            self._lookup_evidence(routed, evidence_snapshot),
                            domain=self.aggregator_domain,
                        )
            sampler_metrics = self.replay_sampler.get_metrics()
        else:
            sampler_metrics = {}

        if getattr(agent.config, "sws_enabled", False):
            self.state.phase = SleepPhase.SWS_ANALOG
        elif getattr(agent.config, "rem_enabled", False):
            self.state.phase = SleepPhase.REM_ANALOG

        # Phase C: PHASE_SWITCH between SWS and REM -- flip the gate to
        # the REM row and re-route the same draws as REM destinations.
        # The replay sampler does NOT redraw (the snapshot is frozen for
        # the cycle); the routing gate provides the SWS->REM channel
        # weight transition. Phase D: the aggregator snapshots the
        # SWS-only posterior at PHASE_SWITCH (Phase E writeback consumes
        # this snapshot); subsequent REM-pass updates land on the live
        # posterior.
        rem_routed_draws: List = []
        if (
            self.routing_gate is not None
            and getattr(agent.config, "rem_enabled", False)
            and sws_routed_draws
        ):
            self.state.phase = SleepPhase.PHASE_SWITCH
            if self.bayesian_aggregator is not None:
                self.bayesian_aggregator.snapshot()
            self.routing_gate.set_phase(SleepPhase.REM_ANALOG)
            for routed in sws_routed_draws:
                rem_routed = self.routing_gate.route(routed.event)
                rem_routed_draws.append(rem_routed)
                if self.bayesian_aggregator is not None:
                    self.bayesian_aggregator.update(
                        rem_routed,
                        self._lookup_evidence(rem_routed, evidence_snapshot),
                        domain=self.aggregator_domain,
                    )

        # Delegate to the existing SD-017 surface. run_sleep_cycle handles
        # the SWS -> REM ordering, mode entry/exit, and metric merging.
        metrics = agent.run_sleep_cycle()

        merged = dict(metrics)
        merged.update(sampler_metrics)
        if self.routing_gate is not None:
            merged.update(self.routing_gate.get_metrics())
            self.routing_gate.set_phase(SleepPhase.WAKING)
        if self.bayesian_aggregator is not None:
            merged.update(self.bayesian_aggregator.get_metrics())

        self.state.phase = SleepPhase.WAKING
        self.state.episodes_since_sleep = 0
        self.state.last_metrics = merged
        self._cycle_history.append(dict(merged))
        return merged

    # -- evidence lookup helpers (Phase D) --

    @staticmethod
    def _build_evidence_snapshot(agent: "REEAgent") -> Dict:
        """Snapshot the per-region staleness map at SLEEP_ENTRY.

        Phase D place-domain evidence is the staleness scalar at the
        replay anchor's region. Read from the agent's StalenessAccumulator
        when available; otherwise return an empty dict (lookups fall
        back to 0.0 evidence, so the posterior pulls toward the prior).
        """
        hippocampal = getattr(agent, "hippocampal", None)
        staleness = getattr(hippocampal, "staleness_accumulator", None)
        if staleness is None:
            return {}
        try:
            snap = staleness.snapshot()
        except Exception:  # pragma: no cover -- defensive
            return {}
        return dict(snap)

    @staticmethod
    def _lookup_evidence(routed, snapshot: Dict) -> float:
        """Look up the evidence scalar for a routed event's region key."""
        event = routed.event
        key_attr = getattr(event, "key", None)
        if (
            isinstance(key_attr, tuple)
            and len(key_attr) >= 2
            and isinstance(key_attr[0], str)
            and isinstance(key_attr[1], str)
        ):
            return float(snapshot.get((key_attr[0], key_attr[1]), 0.0))
        if (
            isinstance(event, tuple)
            and len(event) >= 2
            and isinstance(event[0], str)
            and isinstance(event[1], str)
        ):
            return float(snapshot.get((event[0], event[1]), 0.0))
        return 0.0
