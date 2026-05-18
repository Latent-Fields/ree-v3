"""
PAG (periaqueductal gray) substrate for REE-v3.

Currently hosts:
  - PAGFreezeGate (MECH-279): committed-freeze entry / exit gate.
"""

from ree_core.pag.freeze_gate import (
    PAGFreezeGate,
    PAGFreezeGateConfig,
    PAGFreezeGateOutput,
)

__all__ = [
    "PAGFreezeGate",
    "PAGFreezeGateConfig",
    "PAGFreezeGateOutput",
]
