"""
Sleep-aggregation cluster (MECH-272 / MECH-273 / MECH-275 / MECH-285).

Phase A: scaffolding only. The SleepLoopManager wraps the existing SD-017
surface on REEAgent (enter_sws_mode / run_sws_schema_pass / enter_rem_mode /
run_rem_attribution_pass / exit_sleep_mode / run_sleep_cycle) and adds a
deterministic K-episode trigger plus a SleepCycleState record. Subsequent
phases (B-E) layer on the replay sampler (MECH-285), routing gate (MECH-272),
Bayesian aggregator (MECH-275), and self-model writeback (MECH-273).

See REE_assembly/docs/architecture/sleep_aggregation_cluster.md for the
umbrella design and the master-flag matrix.
"""

from ree_core.sleep.bayesian_aggregator import (
    BayesianAggregator,
    BayesianAggregatorConfig,
    GaussianPosterior,
    PosteriorUpdate,
)
from ree_core.sleep.phase_manager import (
    SleepCycleState,
    SleepLoopManager,
    SleepPhase,
)
from ree_core.sleep.replay_sampler import SleepReplaySampler
from ree_core.sleep.routing_gate import (
    RoutedEvent,
    RoutingGate,
    RoutingGateConfig,
)
from ree_core.sleep.self_model_aggregator import (
    SelfModelAggregator,
    SelfModelAggregatorConfig,
)

__all__ = [
    "BayesianAggregator",
    "BayesianAggregatorConfig",
    "GaussianPosterior",
    "PosteriorUpdate",
    "SleepCycleState",
    "SleepLoopManager",
    "SleepPhase",
    "SleepReplaySampler",
    "RoutingGate",
    "RoutingGateConfig",
    "RoutedEvent",
    "SelfModelAggregator",
    "SelfModelAggregatorConfig",
]
