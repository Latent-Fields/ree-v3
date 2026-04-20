"""SD-034 governance substrate cluster.

Closure operator and (forthcoming) peer-governance modules that coordinate
signal emission across existing gate substrates (BetaGate MECH-090, dACC
MECH-260, ResidueField, SalienceCoordinator, MECH-268 pe-saturation) when
a committed episode reaches resolution.

See: REE_assembly/evidence/planning/sd033_governance_plan.md
"""

from ree_core.governance.closure_operator import (
    ClosureOperator,
    ClosureOperatorConfig,
    ClosureEvent,
)

__all__ = ["ClosureOperator", "ClosureOperatorConfig", "ClosureEvent"]
