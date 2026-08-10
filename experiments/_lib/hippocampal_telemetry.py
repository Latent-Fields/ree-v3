"""SD-025 curiosity-drive telemetry read (chip-20260810-fishtank-sd025-novelty-telemetry).

The fishtank-family drivers (v3_exq_906/906a/906b/906c/911/913) already surface
`residue_wanting` / `liking` / `surprise` per step via `_read_affect()`
(experiments/v3_exq_664_affective_fishtank_showcase.py). SD-025's own novelty
term -- `density(z) * (1 - familiarity(z))`, the information-seeking bonus
that biases CEM trajectory scoring toward under-visited representational
structure (see ree_core/hippocampal/curiosity.py) -- was NOT among those
fields; no per-step channel could discriminate curiosity-driven discovery
from diffuse-gradient exploitation (see
REE_assembly/evidence/planning/developmental_ecology_curiosity_foraging_correction_2026-08-10.md
Section 6). This module is the one-import read for that gap, mirroring the
`_read_affect` convention without editing the already-landed 664/906-lineage
scripts that other queued/landed runs already import.

`ree_core/hippocampal/module.py`'s `HippocampalModule.compute_novelty_score()`
caches its result to `agent.hippocampal.last_novelty_score` once per REAL
(waking) tick, from `ree_core/agent.py`'s familiarity-update call site -- see
that method's docstring for the pre-visit-vs-post-visit ordering rationale.
This module only reads that cache; it computes nothing itself.
"""

from __future__ import annotations

from typing import Optional

from ree_core.agent import REEAgent


def read_sd025_novelty(agent: REEAgent) -> Optional[float]:
    """Last-cached SD-025 novelty(z) term at the agent's current real z_world.

    Returns None when the curiosity drive is disabled
    (config.curiosity_weight <= 0.0 -- ree_core/hippocampal/module.py never
    builds a FamiliarityTracker in that regime) or before the first waking
    tick has run. A future fishtank-family driver logs it per step with:

        from experiments._lib.hippocampal_telemetry import read_sd025_novelty
        ...
        affect["sd025_novelty"] = read_sd025_novelty(agent)
    """
    hipp = getattr(agent, "hippocampal", None)
    if hipp is None:
        return None
    return getattr(hipp, "last_novelty_score", None)
