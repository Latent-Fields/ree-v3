"""
Regulator-layer substrates for REE-v3.

Currently hosts:
  - GABAergicDecayRegulator (SD-036): cross-stream tonic decay regulator.
  - InvalidationTrigger (MECH-287): broadcast invalidation trigger
    subscribing to MECH-288 BoundaryEvents.
  - BroadcastOverrideRegulator (SD-037): orexin-analog broadcast override
    driven by drive_level + sustained-threat magnitude.
"""

from ree_core.regulators.gabaergic_decay import (
    GABAergicDecayConfig,
    GABAergicDecayRegulator,
    StreamRegistration,
)
from ree_core.regulators.invalidation_trigger import (
    BroadcastEvent,
    InvalidationTrigger,
)
from ree_core.regulators.broadcast_override import (
    BroadcastOverrideConfig,
    BroadcastOverrideRegulator,
)
from ree_core.regulators.mech295_liking_bridge import (
    MECH295LikingBridge,
    MECH295LikingBridgeConfig,
    MECH295LikingBridgeOutput,
)
from ree_core.regulators.simulation_mode_rule_gate import (
    SimulationModeRuleGate,
    SimulationModeRuleGateConfig,
    SimulationModeRuleGateDiagnostics,
    SITE_DEFAULT,
    SITE_GATED_POLICY,
    SITE_LATERAL_PFC,
)

__all__ = [
    "GABAergicDecayConfig",
    "GABAergicDecayRegulator",
    "StreamRegistration",
    "BroadcastEvent",
    "InvalidationTrigger",
    "BroadcastOverrideConfig",
    "BroadcastOverrideRegulator",
    "MECH295LikingBridge",
    "MECH295LikingBridgeConfig",
    "MECH295LikingBridgeOutput",
    "SimulationModeRuleGate",
    "SimulationModeRuleGateConfig",
    "SimulationModeRuleGateDiagnostics",
    "SITE_DEFAULT",
    "SITE_GATED_POLICY",
    "SITE_LATERAL_PFC",
]
