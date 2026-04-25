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
    from ree_core.sleep.replay_sampler import SleepReplaySampler


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
        # cycle and perform the configured N draws. Diagnostic only:
        # no downstream consumer (routing / aggregator / writeback) sees
        # the draws yet -- they land in last_metrics for inspection.
        if self.replay_sampler is not None:
            self.state.phase = SleepPhase.SLEEP_ENTRY
            self.replay_sampler.freeze_snapshot()
            for _ in range(self.draws_per_cycle):
                self.replay_sampler.draw()
            sampler_metrics = self.replay_sampler.get_metrics()
        else:
            sampler_metrics = {}

        if getattr(agent.config, "sws_enabled", False):
            self.state.phase = SleepPhase.SWS_ANALOG
        elif getattr(agent.config, "rem_enabled", False):
            self.state.phase = SleepPhase.REM_ANALOG

        # Delegate to the existing SD-017 surface. run_sleep_cycle handles
        # the SWS -> REM ordering, mode entry/exit, and metric merging.
        metrics = agent.run_sleep_cycle()

        merged = dict(metrics)
        merged.update(sampler_metrics)

        self.state.phase = SleepPhase.WAKING
        self.state.episodes_since_sleep = 0
        self.state.last_metrics = merged
        self._cycle_history.append(dict(merged))
        return merged
