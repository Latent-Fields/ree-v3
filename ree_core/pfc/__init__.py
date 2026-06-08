"""PFC subdivision modules (SD-033).

Subdivisions:
- SD-033a (lateral PFC analog): rule/goal persistence, top-down bias into E3.
  Instantiates MECH-262 (rule-selective persistence) and is the primary consumer
  of MECH-261's write-gate registry (SalienceCoordinator). IMPLEMENTED.
- SD-033b (OFC analog): specific-outcome / task-structure substrate, MECH-261
  second consumer; MECH-263 falsification target. IMPLEMENTED.
- SD-033c (dlPFC analog, working-memory): pending.
- SD-033e (frontopolar analog, BA 10): V4-scope STUB. Interface-only; enabling
  use_frontopolar_analog=True raises NotImplementedError until Prong D
  literature synthesis and SD-033e design doc land. See
  ree_core/pfc/frontopolar_analog.py module docstring and
  REE_assembly/evidence/planning/task_inbox.md line 21.
"""

from ree_core.pfc.lateral_pfc_analog import LateralPFCAnalog, LateralPFCConfig
from ree_core.pfc.ofc_analog import OFCAnalog, OFCConfig
from ree_core.pfc.frontopolar_analog import FrontopolarAnalog, FrontopolarConfig
from ree_core.pfc.escape_affordance_bridge import (
    EscapeAffordanceBridge,
    EscapeAffordanceBridgeConfig,
    EscapeAffordanceBridgeOutput,
)
from ree_core.pfc.trainable_escape_affordance_learner import (
    TrainableEscapeAffordanceLearner,
    TrainableEscapeAffordanceLearnerConfig,
    TrainableEscapeAffordanceLearnerOutput,
)

__all__ = [
    "LateralPFCAnalog",
    "LateralPFCConfig",
    "OFCAnalog",
    "OFCConfig",
    "FrontopolarAnalog",
    "FrontopolarConfig",
    "EscapeAffordanceBridge",
    "EscapeAffordanceBridgeConfig",
    "EscapeAffordanceBridgeOutput",
    "TrainableEscapeAffordanceLearner",
    "TrainableEscapeAffordanceLearnerConfig",
    "TrainableEscapeAffordanceLearnerOutput",
]
