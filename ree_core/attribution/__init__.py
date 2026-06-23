"""ree_core.attribution -- waking-phase attribution feedstock substrates.

MECH-276 scientist-agent counterfactual-backed attribution buffer: the
waking-phase feedstock the MECH-275 sleep-phase Bayesian aggregator consumes.
"""

from ree_core.attribution.scientist_attribution_buffer import (
    ScientistAttributionBuffer,
    ScientistAttributionConfig,
)

__all__ = [
    "ScientistAttributionBuffer",
    "ScientistAttributionConfig",
]
