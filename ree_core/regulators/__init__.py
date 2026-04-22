"""
Regulator-layer substrates for REE-v3.

Currently hosts:
  - GABAergicDecayRegulator (SD-036): cross-stream tonic decay regulator.
  - InvalidationTrigger (MECH-287): broadcast invalidation trigger
    subscribing to MECH-288 BoundaryEvents.
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

__all__ = [
    "GABAergicDecayConfig",
    "GABAergicDecayRegulator",
    "StreamRegistration",
    "BroadcastEvent",
    "InvalidationTrigger",
]
