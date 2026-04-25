"""
RoutingGate -- MECH-272 Phase C substrate.

State-conditioned destination-weight routing for replay events. NOT a
proposer change: the gate consumes whatever event the upstream sampler /
hippocampal proposer produced and emits a RoutedEvent carrying two
non-negative channel weights (anchor_channel, probe_channel) that
downstream consumers multiply into their write strength.

Design contract (REE_assembly/docs/architecture/sleep_aggregation_cluster.md
sections "Routing-gate contract" and "Phase ordering within a sleep cycle"):

  * Per-phase weight table:

        Phase        | anchor_channel | probe_channel
        WAKING       |      1.0       |      0.0
        SWS_ANALOG   |      0.6       |      0.4
        REM_ANALOG   |      0.2       |      0.8

  * Phase ordering: SLEEP_ENTRY sets the SWS_ANALOG row before the SWS
    pass; PHASE_SWITCH sets the REM_ANALOG row before the REM pass.
  * route(event) -> RoutedEvent: pure read of the current weights; no
    state mutation. Calling route() outside a sleep cycle (current
    weights = WAKING row) emits anchor_channel=1.0, probe_channel=0.0,
    matching the bit-identical waking guarantee.

Phase C is a NO-OP CONSUMER. The downstream MECH-271 hippocampal router /
E1 ContextMemory consolidation consumer (anchor channel) and the Phase D
Bayesian aggregator (probe channel) do not exist yet, so RoutedEvents
land on SleepCycleState.last_metrics as diagnostics. The gate itself is
still exercised end-to-end so that Phase D / E can plug straight in.

Bit-identical OFF: the master flag use_mech272_routing defaults False;
when False, REEAgent never instantiates this class and the SleepLoopManager
runs exactly as in Phase B.

No trainable parameters. Pure float arithmetic over a per-phase weight
table and a small set of diagnostic counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ree_core.sleep.phase_manager import SleepPhase


@dataclass
class RoutingGateConfig:
    """Per-phase channel weights. Defaults match the design-doc table."""

    waking_anchor_weight: float = 1.0
    waking_probe_weight: float = 0.0
    sws_anchor_weight: float = 0.6
    sws_probe_weight: float = 0.4
    rem_anchor_weight: float = 0.2
    rem_probe_weight: float = 0.8

    def __post_init__(self) -> None:
        for name, value in (
            ("waking_anchor_weight", self.waking_anchor_weight),
            ("waking_probe_weight", self.waking_probe_weight),
            ("sws_anchor_weight", self.sws_anchor_weight),
            ("sws_probe_weight", self.sws_probe_weight),
            ("rem_anchor_weight", self.rem_anchor_weight),
            ("rem_probe_weight", self.rem_probe_weight),
        ):
            if value < 0.0:
                raise ValueError(
                    f"RoutingGateConfig.{name} must be >= 0; got {value}"
                )


@dataclass
class RoutedEvent:
    """A replay event tagged with channel destination weights.

    Downstream consumers multiply their write strength by the relevant
    channel weight. Weights are non-negative floats, NOT a probability
    simplex -- the SWS/REM rows sum to 1.0 by convention but a probe-only
    or anchor-only ablation may set the unused channel to 0 without
    rescaling the other.
    """

    event: Any
    anchor_channel: float
    probe_channel: float
    phase: SleepPhase = SleepPhase.WAKING


class RoutingGate:
    """MECH-272 state-conditioned routing gate.

    Lifecycle within a sleep cycle:

        gate.set_phase(SleepPhase.SLEEP_ENTRY)   # caller chooses the
        gate.set_phase(SleepPhase.SWS_ANALOG)    # destination row at
        ...                                       # SLEEP_ENTRY (SWS) and
        gate.set_phase(SleepPhase.PHASE_SWITCH)   # PHASE_SWITCH (REM).
        gate.set_phase(SleepPhase.REM_ANALOG)
        ...
        gate.set_phase(SleepPhase.WAKING)        # cycle complete

    SLEEP_ENTRY and PHASE_SWITCH are *transition* tokens in the design
    doc; the gate maps them to the destination row of the immediately
    following pass (SLEEP_ENTRY -> SWS row, PHASE_SWITCH -> REM row) so
    callers can either step through the canonical transitions or jump
    straight to the destination row.
    """

    def __init__(self, config: Optional[RoutingGateConfig] = None) -> None:
        self._config = config or RoutingGateConfig()
        self._phase: SleepPhase = SleepPhase.WAKING
        self._anchor_weight: float = self._config.waking_anchor_weight
        self._probe_weight: float = self._config.waking_probe_weight

        # Diagnostics.
        self._n_routed: int = 0
        self._n_routed_per_phase: Dict[SleepPhase, int] = {}
        self._sum_anchor_channel: float = 0.0
        self._sum_probe_channel: float = 0.0

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #
    def set_phase(self, phase: SleepPhase) -> None:
        """Flip the destination weights to the row implied by `phase`.

        SLEEP_ENTRY collapses to the SWS_ANALOG row (the next destination
        the SWS pass will consume); PHASE_SWITCH collapses to the
        REM_ANALOG row. Other phases (WAKING / WRITEBACK) park at the
        WAKING row.
        """
        cfg = self._config
        if phase in (SleepPhase.SWS_ANALOG, SleepPhase.SLEEP_ENTRY):
            self._anchor_weight = cfg.sws_anchor_weight
            self._probe_weight = cfg.sws_probe_weight
        elif phase in (SleepPhase.REM_ANALOG, SleepPhase.PHASE_SWITCH):
            self._anchor_weight = cfg.rem_anchor_weight
            self._probe_weight = cfg.rem_probe_weight
        else:
            self._anchor_weight = cfg.waking_anchor_weight
            self._probe_weight = cfg.waking_probe_weight
        self._phase = phase

    def reset(self) -> None:
        """Park at the WAKING row and zero diagnostics."""
        self._phase = SleepPhase.WAKING
        self._anchor_weight = self._config.waking_anchor_weight
        self._probe_weight = self._config.waking_probe_weight
        self._n_routed = 0
        self._n_routed_per_phase = {}
        self._sum_anchor_channel = 0.0
        self._sum_probe_channel = 0.0

    # ------------------------------------------------------------------ #
    # Route                                                              #
    # ------------------------------------------------------------------ #
    def route(self, event: Any) -> RoutedEvent:
        """Tag `event` with the current (anchor_channel, probe_channel)."""
        routed = RoutedEvent(
            event=event,
            anchor_channel=self._anchor_weight,
            probe_channel=self._probe_weight,
            phase=self._phase,
        )
        self._n_routed += 1
        self._n_routed_per_phase[self._phase] = (
            self._n_routed_per_phase.get(self._phase, 0) + 1
        )
        self._sum_anchor_channel += self._anchor_weight
        self._sum_probe_channel += self._probe_weight
        return routed

    # ------------------------------------------------------------------ #
    # Diagnostics                                                        #
    # ------------------------------------------------------------------ #
    @property
    def phase(self) -> SleepPhase:
        return self._phase

    @property
    def anchor_weight(self) -> float:
        return float(self._anchor_weight)

    @property
    def probe_weight(self) -> float:
        return float(self._probe_weight)

    @property
    def n_routed(self) -> int:
        return int(self._n_routed)

    def n_routed_in_phase(self, phase: SleepPhase) -> int:
        return int(self._n_routed_per_phase.get(phase, 0))

    def get_metrics(self) -> Dict[str, float]:
        """Flat metrics dict suitable for SleepCycleState.last_metrics."""
        n = max(1, self._n_routed)
        return {
            "mech272_n_routed": float(self._n_routed),
            "mech272_n_routed_sws": float(
                self._n_routed_per_phase.get(SleepPhase.SWS_ANALOG, 0)
            ),
            "mech272_n_routed_rem": float(
                self._n_routed_per_phase.get(SleepPhase.REM_ANALOG, 0)
            ),
            "mech272_mean_anchor_channel": float(self._sum_anchor_channel / n),
            "mech272_mean_probe_channel": float(self._sum_probe_channel / n),
            "mech272_current_anchor_weight": float(self._anchor_weight),
            "mech272_current_probe_weight": float(self._probe_weight),
        }
