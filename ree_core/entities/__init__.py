"""ree_core.entities -- token-instance entity-representation substrates.

Home of the ARC-006 / MECH-045 object-file buffer: the TOKEN projection of the
ARC-080 type/token/anchor triad (the missing third store; the TYPE store is the
SD-057 IncentiveTokenBank in goal.py and the ANCHOR store is the SD-039 /
MECH-292 ghost-goal bank in hippocampal/). See
REE_assembly/docs/architecture/mech_045_object_file_buffer.md.
"""

from ree_core.entities.object_file_buffer import (
    EntityObservation,
    ObjectFile,
    ObjectFileBuffer,
    ObjectFileBufferConfig,
)

__all__ = [
    "EntityObservation",
    "ObjectFile",
    "ObjectFileBuffer",
    "ObjectFileBufferConfig",
]
