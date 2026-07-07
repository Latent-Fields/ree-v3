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
    from ree_core.sleep.cross_module_consolidation import CrossModuleConsolidator
    from ree_core.sleep.mel_consumer import MELConsumer
    from ree_core.sleep.replay_sampler import SleepReplaySampler
    from ree_core.sleep.routing_gate import RoutingGate
    from ree_core.sleep.self_model_aggregator import SelfModelAggregator


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

    # welfare_relevant: descriptive ethics-perimeter marker, NOT a gate
    # (SENT-0; ethics_perimeter:P1-V3-WELFARE-TAG). Sleep/replay can re-sample
    # harm-bearing content offline -- a V3 welfare-relevant primitive,
    # pre-ethical, NOT claimed sentient.
    # See REE_assembly/docs/governance/sentience_welfare_risk_register.md.
    welfare_relevant = True

    def __init__(
        self,
        cycle_every_k_episodes: int = 1,
        require_sleep_passes_enabled: bool = True,
        replay_sampler: Optional["SleepReplaySampler"] = None,
        draws_per_cycle: int = 0,
        routing_gate: Optional["RoutingGate"] = None,
        bayesian_aggregator: Optional["BayesianAggregator"] = None,
        aggregator_domain: str = "place",
        self_model_aggregator: Optional["SelfModelAggregator"] = None,
        self_model_offline_n_steps: int = 100,
        self_model_partial_decay_factor: float = 0.5,
        self_model_domain: str = "self",
        use_rem_precision_recalibration: bool = False,
        rem_precision_recalibration_step: float = 0.1,
        use_mech272_routing_consumer: bool = False,
        cross_module_consolidator: Optional["CrossModuleConsolidator"] = None,
        cross_module_consolidation_steps: int = 0,
        cross_module_consolidation_schedule: str = "interleaved",
        cross_module_consolidation_lr: float = 1e-3,
        cross_module_consolidation_batch: int = 16,
        mel_consumer: Optional["MELConsumer"] = None,
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
        self.self_model_aggregator = self_model_aggregator
        self.self_model_offline_n_steps = int(self_model_offline_n_steps)
        if not 0.0 <= float(self_model_partial_decay_factor) <= 1.0:
            raise ValueError(
                "self_model_partial_decay_factor must be in [0, 1]; "
                f"got {self_model_partial_decay_factor}"
            )
        self.self_model_partial_decay_factor = float(self_model_partial_decay_factor)
        self.self_model_domain = str(self_model_domain)
        self.use_rem_precision_recalibration = bool(use_rem_precision_recalibration)
        if not 0.0 <= float(rem_precision_recalibration_step) <= 1.0:
            raise ValueError(
                "rem_precision_recalibration_step must be in [0, 1]; "
                f"got {rem_precision_recalibration_step}"
            )
        self.rem_precision_recalibration_step = float(rem_precision_recalibration_step)
        self.use_mech272_routing_consumer = bool(use_mech272_routing_consumer)
        # MECH-423 R3: module-tagged interleaved cross-module consolidation.
        self.cross_module_consolidator = cross_module_consolidator
        self.cross_module_consolidation_steps = int(cross_module_consolidation_steps)
        self.cross_module_consolidation_schedule = str(
            cross_module_consolidation_schedule
        )
        self.cross_module_consolidation_lr = float(cross_module_consolidation_lr)
        self.cross_module_consolidation_batch = int(cross_module_consolidation_batch)
        # SD-MEL-CONSUMER (GAP-5b): adaptive sleep-cadence MEL consumer. None ->
        # the K-episode-deterministic scheduler + fixed-duration cycle are
        # bit-identical to the pre-SD substrate.
        self.mel_consumer = mel_consumer
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
        # SD-MEL-CONSUMER (GAP-5b) entry-timing lever: when enabled, fire once
        # accumulated waking MEL crosses the threshold, with the K-episode
        # counter as a safety-backstop ceiling. Bit-identical strict-K when the
        # consumer is absent or its entry lever is off.
        if self.mel_consumer is not None:
            if not self.mel_consumer.entry_permitted(
                self.state.episodes_since_sleep, self.cycle_every_k_episodes
            ):
                return None
            return self._run_cycle(agent)
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
        if self.mel_consumer is not None:
            self.mel_consumer.reset()

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

        # MECH-286: override-gated sleep recruitment (orexin wake-stability axis).
        if getattr(agent.config, "use_mech286_sleep_onset_gate", False):
            from ree_core.sleep.sleep_onset_gate import evaluate_sleep_onset_permit

            permitted, gate_metrics = evaluate_sleep_onset_permit(agent)
            if not permitted:
                self.state.episodes_since_sleep = 0
                blocked = dict(gate_metrics)
                self.state.last_metrics = blocked
                return blocked

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
        replayed_regions: "set" = set()
        harm_replay_buffer_snapshot: list = []
        if self.replay_sampler is not None:
            self.state.phase = SleepPhase.SLEEP_ENTRY
            if self.routing_gate is not None:
                self.routing_gate.set_phase(SleepPhase.SWS_ANALOG)
            self.replay_sampler.freeze_snapshot()
            # GAP-4 / MECH-273: snapshot waking (z_harm_s, action) pairs before
            # any sleep-pass state changes the agent's waking-stream buffer.
            harm_replay_buffer_snapshot = list(
                getattr(agent, "_harm_replay_buffer", [])
            )
            if self.bayesian_aggregator is not None:
                evidence_snapshot = self._build_evidence_snapshot(agent)
            for _ in range(self.draws_per_cycle):
                anchor = self.replay_sampler.draw()
                if anchor is None:
                    continue
                if self.routing_gate is not None:
                    routed = self.routing_gate.route(anchor)
                    sws_routed_draws.append(routed)
                    region_key = self._extract_region_key(routed)
                    if region_key is not None:
                        replayed_regions.add(region_key)
                    if self.bayesian_aggregator is not None:
                        self.bayesian_aggregator.update(
                            routed,
                            self._lookup_evidence(routed, evidence_snapshot),
                            domain=self.aggregator_domain,
                        )
                    if self.self_model_aggregator is not None:
                        self.self_model_aggregator.update(
                            routed,
                            self._lookup_evidence(routed, evidence_snapshot),
                            domain=self.self_model_domain,
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
            if self.self_model_aggregator is not None:
                self.self_model_aggregator.snapshot()
            self.routing_gate.set_phase(SleepPhase.REM_ANALOG)
            for routed in sws_routed_draws:
                rem_routed = self.routing_gate.route(routed.event)
                rem_routed_draws.append(rem_routed)
                region_key = self._extract_region_key(rem_routed)
                if region_key is not None:
                    replayed_regions.add(region_key)
                if self.bayesian_aggregator is not None:
                    self.bayesian_aggregator.update(
                        rem_routed,
                        self._lookup_evidence(rem_routed, evidence_snapshot),
                        domain=self.aggregator_domain,
                    )
                if self.self_model_aggregator is not None:
                    self.self_model_aggregator.update(
                        rem_routed,
                        self._lookup_evidence(rem_routed, evidence_snapshot),
                        domain=self.self_model_domain,
                    )

        # GAP-8: forward mean anchor_channel to run_sleep_cycle ONLY when
        # use_mech272_routing_consumer is enabled. When OFF the weight stays
        # at 1.0 (bit-identical to pre-GAP-8: full-strength schema writes).
        if self.use_mech272_routing_consumer and sws_routed_draws:
            mean_anchor = float(
                sum(r.anchor_channel for r in sws_routed_draws) / len(sws_routed_draws)
            )
        else:
            mean_anchor = 1.0

        # SD-MEL-CONSUMER (GAP-5b) DURATION lever: scale this cycle's SWS/REM
        # step counts by the accumulated-waking-MEL duration factor. Temporarily
        # override agent.config.sws_consolidation_steps / rem_attribution_steps
        # (which run_sleep_cycle -> run_sws_schema_pass / run_rem_attribution_pass
        # read directly), then restore in finally so the override is scoped to
        # this cycle only. No-op (factor 1.0, no override) when the consumer is
        # absent -> bit-identical fixed-duration cycle.
        _mel_factor = 1.0
        _mel_orig_sws = None
        _mel_orig_rem = None
        if self.mel_consumer is not None:
            _mel_factor = self.mel_consumer.duration_factor()
            if getattr(agent.config, "mel_scale_sws", True):
                _mel_orig_sws = int(getattr(agent.config, "sws_consolidation_steps", 0))
                agent.config.sws_consolidation_steps = self.mel_consumer.scale_steps(
                    _mel_orig_sws
                )
            if getattr(agent.config, "mel_scale_rem", True):
                _mel_orig_rem = int(getattr(agent.config, "rem_attribution_steps", 0))
                agent.config.rem_attribution_steps = self.mel_consumer.scale_steps(
                    _mel_orig_rem
                )

        # Delegate to the existing SD-017 surface. run_sleep_cycle handles
        # the SWS -> REM ordering, mode entry/exit, and metric merging.
        # infant_substrate:GAP-8 -- capture z_goal norm before the cycle so
        # retention can be computed after. Non-invasive read; -1.0 sentinel
        # when goal_state is absent.
        _z_goal_before = self._safe_z_goal_norm(agent)
        try:
            metrics = agent.run_sleep_cycle(sws_anchor_weight=mean_anchor)
        finally:
            if _mel_orig_sws is not None:
                agent.config.sws_consolidation_steps = _mel_orig_sws
            if _mel_orig_rem is not None:
                agent.config.rem_attribution_steps = _mel_orig_rem
        _z_goal_after = self._safe_z_goal_norm(agent)

        # Phase E: WRITEBACK -- self-model offline gradient pass on
        # E2_harm_s using aggregator-corrected residuals as targets, plus
        # MECH-284 partial decay on replayed regions (the schemas they
        # encode have just been refreshed by the offline pass). MECH-094
        # simulation_mode tag is the EXPLICIT EXCEPTION here: parameter
        # update is gated to E2_harm_s ONLY (the optimiser is constructed
        # locally inside offline_gradient_pass over e2_harm_s.parameters()).
        writeback_metrics: Dict[str, float] = {}
        if (
            self.self_model_aggregator is not None
            and getattr(agent, "e2_harm_s", None) is not None
        ):
            self.state.phase = SleepPhase.WRITEBACK
            writeback_metrics = self.self_model_aggregator.offline_gradient_pass(
                e2_harm_s=agent.e2_harm_s,
                replayed_regions=replayed_regions,
                n_steps=self.self_model_offline_n_steps,
                domain=self.self_model_domain,
                use_snapshot=True,
                harm_replay_buffer=harm_replay_buffer_snapshot,
            )
            hippocampal = getattr(agent, "hippocampal", None)
            staleness = getattr(hippocampal, "staleness_accumulator", None)
            if staleness is not None and replayed_regions:
                n_decayed = staleness.partial_decay(
                    replayed_regions,
                    decay_factor=self.self_model_partial_decay_factor,
                )
                writeback_metrics["mech273_partial_decay_n_regions"] = float(
                    n_decayed
                )
                writeback_metrics["mech273_partial_decay_factor"] = float(
                    self.self_model_partial_decay_factor
                )

        # MECH-204 Option A: precision recalibration consumer (sibling step in
        # WRITEBACK). Reads SerotoninModule.compute_recalibration_target()
        # (the captured precision_at_rem_entry zero-point reference) and
        # nudges E3._running_variance toward 1.0/target by the configured
        # step. Runs independently of MECH-273 so the WRITEBACK phase fires
        # for either reason. Only fires when REM was actually entered this
        # cycle (rem_enabled gate) and serotonin is enabled (so the captured
        # target is meaningful) and the master flag is on.
        if (
            self.use_rem_precision_recalibration
            and getattr(agent.config, "rem_enabled", False)
            and getattr(agent, "serotonin", None) is not None
            and agent.serotonin.enabled
            and getattr(agent, "e3", None) is not None
        ):
            self.state.phase = SleepPhase.WRITEBACK
            target = float(agent.serotonin.compute_recalibration_target())
            if target > 0.0:
                rv_before, rv_after = agent.e3.recalibrate_precision_to(
                    target,
                    step=self.rem_precision_recalibration_step,
                )
                writeback_metrics["mech204_recalibration_target"] = target
                writeback_metrics["mech204_running_variance_before"] = float(
                    rv_before
                )
                writeback_metrics["mech204_running_variance_after"] = float(
                    rv_after
                )
                writeback_metrics["mech204_recalibration_step"] = float(
                    self.rem_precision_recalibration_step
                )
                writeback_metrics["mech204_recalibration_fired"] = 1.0
            else:
                writeback_metrics["mech204_recalibration_fired"] = 0.0

        merged = dict(metrics)
        if getattr(agent.config, "use_mech286_sleep_onset_gate", False):
            from ree_core.sleep.sleep_onset_gate import evaluate_sleep_onset_permit

            _, gate_metrics = evaluate_sleep_onset_permit(agent)
            merged.update(gate_metrics)
            merged["mech286_sleep_permitted"] = 1.0
        merged.update(sampler_metrics)
        if self.routing_gate is not None:
            merged.update(self.routing_gate.get_metrics())
            self.routing_gate.set_phase(SleepPhase.WAKING)
        if self.bayesian_aggregator is not None:
            merged.update(self.bayesian_aggregator.get_metrics())
        if self.self_model_aggregator is not None:
            merged.update(self.self_model_aggregator.get_metrics())
        if writeback_metrics:
            merged.update(writeback_metrics)

        # MECH-276: merge the scientist-attribution feedstock readout + apply the
        # per-cycle decay so newer counterfactual-backed evidence can overcome a
        # stale prior. No-op when the buffer is absent (bit-identical OFF).
        sci_buf = getattr(agent, "scientist_attribution_buffer", None)
        if sci_buf is not None:
            merged.update(sci_buf.get_metrics())
            sci_buf.decay_cycle()

        # MECH-423 R3: module-tagged interleaved cross-module consolidation.
        # Runs AFTER the existing writeback (additive). Builds default E1 + E2
        # loss closures from the agent's existing replay losses and runs the
        # configured schedule; merges the cross_module_replay_share / interleaved
        # / update-count readouts into the sleep-cycle metrics so they are a
        # readout of the LIVE MECH-121 consolidation pipeline. No-op (skipped
        # entirely) when the consolidator is None or steps <= 0 -> bit-identical.
        if (
            self.cross_module_consolidator is not None
            and self.cross_module_consolidation_steps > 0
            and getattr(agent, "e1", None) is not None
            and getattr(agent, "e2", None) is not None
        ):
            _batch = self.cross_module_consolidation_batch
            cmc_metrics = self.cross_module_consolidator.consolidate(
                module_losses={
                    # E1 world-model replay loss (over _world_experience_buffer).
                    "e1": lambda: agent.compute_prediction_loss(),
                    # E2 forward-model replay loss (over _e2_transition_buffer).
                    "e2": lambda: agent.compute_e2_loss(batch_size=_batch),
                },
                module_params={
                    "e1": list(agent.e1.parameters()),
                    "e2": list(agent.e2.parameters()),
                },
                n_steps=self.cross_module_consolidation_steps,
                schedule=self.cross_module_consolidation_schedule,
                lr=self.cross_module_consolidation_lr,
                simulation_mode=False,  # legitimate offline weight consolidation
            )
            merged.update(
                {f"cross_module_consolidation_{k}": v for k, v in cmc_metrics.items()}
            )

        # infant_substrate:GAP-8 -- post_sleep_z_goal_retention telemetry.
        # -1.0 sentinel when goal_state absent or z_goal_before <= 1e-8
        # (uninitialised). replay_diversity_index = distinct-regions / draws;
        # -1.0 sentinel when no draws occurred this cycle.
        merged["post_sleep_z_goal_before"] = _z_goal_before
        merged["post_sleep_z_goal_after"] = _z_goal_after
        merged["post_sleep_z_goal_retention"] = (
            _z_goal_after / _z_goal_before
            if _z_goal_before > 1e-8
            else -1.0
        )
        _n_draws = len(sws_routed_draws)
        merged["replay_diversity_index"] = (
            float(len(replayed_regions)) / float(_n_draws)
            if _n_draws > 0
            else -1.0
        )

        # SD-MEL-CONSUMER (GAP-5b): surface the MEL read + duration factor + the
        # effective scaled step counts this cycle, then update the reference
        # set-point and reset the accumulator for the next wake window.
        if self.mel_consumer is not None:
            merged.update(self.mel_consumer.get_metrics())
            merged["mel_sws_steps_effective"] = (
                float(self.mel_consumer.scale_steps(_mel_orig_sws))
                if _mel_orig_sws is not None
                else float(getattr(agent.config, "sws_consolidation_steps", 0.0))
            )
            merged["mel_rem_steps_effective"] = (
                float(self.mel_consumer.scale_steps(_mel_orig_rem))
                if _mel_orig_rem is not None
                else float(getattr(agent.config, "rem_attribution_steps", 0.0))
            )
            self.mel_consumer.on_cycle_complete()

        self.state.phase = SleepPhase.WAKING
        self.state.episodes_since_sleep = 0
        self.state.last_metrics = merged
        self._cycle_history.append(dict(merged))
        return merged

    # -- evidence lookup helpers (Phase D) --

    @staticmethod
    def _build_evidence_snapshot(agent: "REEAgent") -> Dict:
        """Snapshot the per-region evidence map at SLEEP_ENTRY.

        MECH-276 (when use_scientist_attribution is on): the aggregator
        evidence IS the waking-phase counterfactual-backed attribution
        feedstock (claims.yaml MECH-275: sleep aggregates counterfactual-backed
        attribution, not arbitrary correlation). The buffer's region -> attribution
        snapshot REPLACES the staleness scalar as the evidence source, and
        carries a GLOBAL_REGION sentinel for the _lookup_evidence fallback.

        Legacy (MECH-276 off): Phase D place-domain evidence is the MECH-284
        staleness scalar at the replay anchor's region. Read from the agent's
        StalenessAccumulator when available; otherwise return an empty dict
        (lookups fall back to 0.0 evidence, so the posterior pulls toward the
        prior). Bit-identical to the pre-MECH-276 path.
        """
        sci_buf = getattr(agent, "scientist_attribution_buffer", None)
        if sci_buf is not None:
            try:
                return dict(sci_buf.evidence_snapshot())
            except Exception:  # pragma: no cover -- defensive
                return {}

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
    def _safe_z_goal_norm(agent: "REEAgent") -> float:
        """Return z_goal L2 norm or -1.0 sentinel when unavailable.

        infant_substrate:GAP-8 helper -- non-invasive read, no state mutation.
        """
        goal_state = getattr(agent, "goal_state", None)
        if goal_state is None:
            return -1.0
        z_goal = getattr(goal_state, "_z_goal", None)
        if z_goal is None:
            return -1.0
        try:
            return float(z_goal.norm().item())
        except Exception:  # pragma: no cover -- defensive
            return -1.0

    @staticmethod
    def _extract_region_key(routed) -> Optional[tuple]:
        """Return (scale, segment_id) for a routed event, or None."""
        event = getattr(routed, "event", routed)
        key_attr = getattr(event, "key", None)
        if (
            isinstance(key_attr, tuple)
            and len(key_attr) >= 2
            and isinstance(key_attr[0], str)
            and isinstance(key_attr[1], str)
        ):
            return (key_attr[0], key_attr[1])
        if (
            isinstance(event, tuple)
            and len(event) >= 2
            and isinstance(event[0], str)
            and isinstance(event[1], str)
        ):
            return (event[0], event[1])
        return None

    # MECH-276 ScientistAttributionBuffer.GLOBAL_REGION sentinel: the snapshot
    # may carry a global-mean attribution under this key for fallback lookups of
    # routed regions not visited during the preceding waking phase. Staleness
    # snapshots never carry it (their .get() falls back to 0.0 as before), so
    # the legacy evidence path is bit-identical.
    _GLOBAL_REGION_SENTINEL = ("__global__", "")

    @classmethod
    def _lookup_evidence(cls, routed, snapshot: Dict) -> float:
        """Look up the evidence scalar for a routed event's region key.

        Falls back to the MECH-276 GLOBAL_REGION sentinel (global-mean
        attribution) when the routed region is absent from the snapshot, then to
        0.0. The fallback is a no-op for staleness snapshots (no sentinel key).
        """
        fallback = float(snapshot.get(cls._GLOBAL_REGION_SENTINEL, 0.0))
        event = routed.event
        key_attr = getattr(event, "key", None)
        if (
            isinstance(key_attr, tuple)
            and len(key_attr) >= 2
            and isinstance(key_attr[0], str)
            and isinstance(key_attr[1], str)
        ):
            return float(snapshot.get((key_attr[0], key_attr[1]), fallback))
        if (
            isinstance(event, tuple)
            and len(event) >= 2
            and isinstance(event[0], str)
            and isinstance(event[1], str)
        ):
            return float(snapshot.get((event[0], event[1]), fallback))
        return fallback
