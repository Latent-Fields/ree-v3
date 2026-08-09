"""
PAG (periaqueductal gray) substrate for REE-v3.

Currently hosts:
  - PAGFreezeGate (MECH-279): committed-freeze entry / exit gate (chronic,
    accumulated-suffering trigger).
  - DefensiveOrientingGate (MECH-489, SD-099): phasic defensive-orienting
    response (sudden/unidentified-onset trigger, distinct sibling gate).
"""

from ree_core.pag.freeze_gate import (
    PAGFreezeGate,
    PAGFreezeGateConfig,
    PAGFreezeGateOutput,
)
from ree_core.pag.defensive_orienting import (
    DefensiveOrientingGate,
    DefensiveOrientingConfig,
    DefensiveOrientingOutput,
)

__all__ = [
    "PAGFreezeGate",
    "PAGFreezeGateConfig",
    "PAGFreezeGateOutput",
    "DefensiveOrientingGate",
    "DefensiveOrientingConfig",
    "DefensiveOrientingOutput",
]
