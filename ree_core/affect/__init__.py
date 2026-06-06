"""ree_core.affect -- derived affect-stream readouts layered on the core latents.

Unlike the encoder-produced harm streams (z_harm_s / z_harm_a in latent/stack.py),
the modules here are pure-arithmetic regulators that integrate already-computed
signals over time. First member: MECH-353 blocked-agency / control-failure
(z_block).
"""

from ree_core.affect.blocked_agency import (
    BlockedAgency,
    BlockedAgencyConfig,
    BlockedAgencyOutput,
)

__all__ = [
    "BlockedAgency",
    "BlockedAgencyConfig",
    "BlockedAgencyOutput",
]
