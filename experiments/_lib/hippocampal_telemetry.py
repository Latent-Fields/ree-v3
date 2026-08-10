"""Hippocampal telemetry reads for the fishtank-family drivers
(chip-20260810-fishtank-sd025-novelty-telemetry,
chip-20260810-fishtank-visitation-count-telemetry).

The fishtank-family drivers (v3_exq_906/906a/906b/906c/911/913) already surface
`residue_wanting` / `liking` / `surprise` per step via `_read_affect()`
(experiments/v3_exq_664_affective_fishtank_showcase.py).
REE_assembly/evidence/planning/developmental_ecology_curiosity_foraging_correction_2026-08-10.md
Section 6 names three further signals needed to discriminate curiosity-driven
discovery from diffuse-gradient exploitation that were NOT among those fields:
SD-025's novelty term, MECH-314's score-bias (left alone -- conditional on a flag
no current fishtank driver enables), and a state-visitation count. This module is
the one-import read for the first and third of those, mirroring the `_read_affect`
convention without editing the already-landed 664/906-lineage scripts that other
queued/landed runs already import.

Both reads are cache-only: the underlying values are computed and cached once per
REAL (waking) tick by `ree_core/agent.py`'s familiarity-update call site (see
`HippocampalModule.compute_novelty_score()` / `record_visitation()` docstrings for
the pre-visit-vs-post-visit ordering rationale). This module computes nothing
itself.
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


def read_state_visitation_count(agent: REEAgent) -> Optional[int]:
    """Last-cached unconditional visit count at the agent's current real
    z_world (0 = this tick is the first-ever visit to that region).

    Unlike read_sd025_novelty, this is NEVER None due to a disabled flag --
    ree_core/hippocampal/module.py's VisitationCounter (ree_core/hippocampal/
    visitation.py) is instantiated unconditionally, so this is available in
    every fishtank-family driver regardless of curiosity_weight or
    use_structured_curiosity. Returns None only before the first waking tick
    has run (agent freshly constructed, no sense() call yet). A future
    fishtank-family driver logs it per step with:

        from experiments._lib.hippocampal_telemetry import read_state_visitation_count
        ...
        affect["state_visitation_count"] = read_state_visitation_count(agent)

    From a per-step count stream, a driver can derive e.g. "fraction of steps
    spent in a never-before-visited cell" (count == 0) or "unique cells
    visited per episode" (count of steps where count == 0), directly from
    telemetry instead of re-deriving it from raw position traces.
    """
    hipp = getattr(agent, "hippocampal", None)
    if hipp is None:
        return None
    return getattr(hipp, "last_visitation_count", None)
