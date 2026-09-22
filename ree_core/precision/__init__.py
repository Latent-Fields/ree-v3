"""ree_core.precision -- epistemic precision producers (SD-PP set, 2026-09-22).

Two producers live here, both default-OFF and pure-arithmetic (no RNG draws):

  * observation_reliability   -- SD-PP-1  evidence (sensory) precision
  * world_forward_epistemic_precision -- SD-PP-2  model precision of e2.world_forward

They feed the replay provenance packet (ree_core/hippocampal/replay_provenance.py,
SD-PP-3) and the provenance-conditioned consolidation gain
(ree_core/sleep/provenance_gain.py, SD-PP-4). Contract:
REE_assembly/docs/architecture/precision_provenance_substrate_spec.md
"""
