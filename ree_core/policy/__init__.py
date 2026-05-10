"""Policy-layer modules.

ARC-062 (rule-apprehension layer, weak reading) lives here as the V3-tractable
instantiation of the rule-apprehension architectural slot identified by
MECH-309. Future ARC-063 strong-reading instantiations (distributed
CandidateRule field) would also land in this package.

See evidence/planning/arc_062_rule_apprehension_plan.md for the closure plan.
"""

from ree_core.policy.gated_policy import (
    GatedPolicy,
    GatedPolicyConfig,
    GatedPolicyOutput,
)
from ree_core.policy.noise_floor import (
    NoiseFloor,
    NoiseFloorConfig,
)
from ree_core.policy.structured_curiosity import (
    StructuredCuriosity,
    StructuredCuriosityConfig,
)

__all__ = [
    "GatedPolicy",
    "GatedPolicyConfig",
    "GatedPolicyOutput",
    "NoiseFloor",
    "NoiseFloorConfig",
    "StructuredCuriosity",
    "StructuredCuriosityConfig",
]
