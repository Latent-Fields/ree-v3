"""
Regulator-layer substrates for REE-v3.

Currently hosts:
  - GABAergicDecayRegulator (SD-036): cross-stream tonic decay regulator.
"""

from ree_core.regulators.gabaergic_decay import (
    GABAergicDecayConfig,
    GABAergicDecayRegulator,
    StreamRegistration,
)

__all__ = [
    "GABAergicDecayConfig",
    "GABAergicDecayRegulator",
    "StreamRegistration",
]
