"""PFC subdivision modules (SD-033).

Subdivisions:
- SD-033a (lateral PFC analog): rule/goal persistence, top-down bias into E3.
  Instantiates MECH-262 (rule-selective persistence) and is the primary consumer
  of MECH-261's write-gate registry (SalienceCoordinator).
- SD-033b (vmPFC analog): pending.
- SD-033c (dlPFC analog, working-memory): pending.
"""

from ree_core.pfc.lateral_pfc_analog import LateralPFCAnalog

__all__ = ["LateralPFCAnalog"]
