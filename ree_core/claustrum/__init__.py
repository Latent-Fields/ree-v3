"""
Claustrum-analog substrate for REE-v3 (SD-091 / MECH-481 cluster).

Currently implements (MVP, steps 1-3 of sd_091_coalition_topology_control.md's
"Minimum-viable V3 implementation path" only -- see coalition_controller.py's
module docstring "Scope" note for what is deliberately NOT done in this
pass):
  - ControlDemandType (control_demand.py): the 10-class MECH-481 typed
    control-demand taxonomy. Only SENSORY_RESAMPLE and PROVENANCE_CHECK are
    templated; the other 8 are declared-but-untemplated.
  - COALITION_TEMPLATES (coalition_templates.py): static recruit/suppress/
    channel_gain data for the 2 MVP-templated types.
  - CoalitionController / CoalitionControllerConfig / CoalitionState
    (coalition_controller.py): the star-topology recruit/suppress/gain
    primitive (G_t), request_coalition() injection API,
    write_gate(target)/channel_gain(target) consumer-facing accessors, and
    minimal Gamma_t (timeout + completion-condition dissolution).

use_coalition_controller-style master switch: CoalitionControllerConfig.
enabled, default False -- bit-identical off. Nothing in ree_core imports
this package yet (no consumer wiring, no REEAgent.select_action
integration), so the package is inert by construction as of this landing,
independent of the config default.

See CLAUDE.md (ree-v3): SD-091, MECH-481. Spec:
REE_assembly/docs/architecture/sd_091_coalition_topology_control.md.
"""

from ree_core.claustrum.control_demand import (
    MVP_TEMPLATED_TYPES,
    ControlDemandType,
)
from ree_core.claustrum.coalition_controller import (
    CoalitionController,
    CoalitionControllerConfig,
    CoalitionState,
)
from ree_core.claustrum.coalition_templates import (
    COALITION_TEMPLATES,
    CoalitionTemplate,
)

__all__ = [
    "ControlDemandType",
    "MVP_TEMPLATED_TYPES",
    "CoalitionController",
    "CoalitionControllerConfig",
    "CoalitionState",
    "COALITION_TEMPLATES",
    "CoalitionTemplate",
]
