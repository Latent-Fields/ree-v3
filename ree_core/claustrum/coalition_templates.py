"""
COALITION_TEMPLATES -- the 2 MVP typed-demand -> star-topology templates.

Per sd_091_coalition_topology_control.md Section 2 ("Which typed
control-demand classes to build first"), only SENSORY_RESAMPLE and
PROVENANCE_CHECK are templated in this MVP pass. Both map onto ALREADY-BUILT
consumer substrate (no second implementation pass smuggled in as "just
wiring the coalition controller") -- the recruit/suppress lists below are
taken directly from Section 2's table and the source thought's worked
PROVENANCE_CHECK example, substituting the actual ree_core module each
target name corresponds to. The other 8 ControlDemandType members
(control_demand.py) stay declared-but-untemplated: consumer substrate for
them either doesn't exist yet in ree_core (social/language/reality-coherence
modules) or needs its own scoping pass (see the doc's Section 2 triage).

Target-name -> module map (for whoever wires consumer read sites in a
follow-up pass -- steps 4-5 of the doc's implementation path, NOT done by
this module):

  e1_sensory_encoder          ree_core/predictors/e1_deep.py (E1 precision-
                               routing target)
  e2_fast_forward_model       ree_core/predictors/e2_fast.py (E2 fast
                               forward model)
  e3_candidate_count          E3 selector's candidate-count ceiling -- a
                               PARAMETRIC side-effect (channel_gain axis),
                               deliberately not a participating/suppressed
                               axis entry (doc Section 2: "temporarily raise
                               candidate_count ceiling is a parametric
                               side-effect the coalition may request via its
                               own channel_gain, not a separate axis").
  hippocampal_anchor_set      ree_core/hippocampal/anchor_set.py (ARC-038
                               viability map)
  hippocampal_persistence_appraisal
                               ree_core/hippocampal/persistence_appraisal_
                               compute.py (MECH-269)
  e3_commitment_monitor       agent.py BetaGate / MECH-090 commit-entry
                               predicate, recruited (monitoring intensified,
                               never force-opened -- see the BetaGate
                               guardrail in coalition_controller.py)
  motor_commitment            BetaGate elevation eligibility -- attenuated
                               (suppressed), never raised
  hippocampal_write_consolidation
                               hippocampal write-consolidation gate --
                               attenuated (suppressed), never raised

Numeric weights (0.9 recruit / 0.6-0.5 suppress) are this module's own
placeholder magnitudes -- the doc leaves exact numbers to whoever builds
the MVP ("default TBD by whoever builds this"). They are chosen so that
SENSORY_RESAMPLE and PROVENANCE_CHECK are ALWAYS numerically distinguishable
via CoalitionController.write_gate() at every consumer site named above
(recruit weight < 1.0, not a no-op-identical 1.0), matching the doc's own
step-6 smoke-test requirement and the "primitive is present but miswired"
falsification signature ("log participating/suppressed per trial and
diff"). Retune when real consumer wiring (steps 4-5) lands -- these are not
load-bearing on any experiment yet, since nothing calls request_coalition()
outside this module's own tests.

Biological mapping stays provisional -- see control_demand.py's module
docstring and sd_091_coalition_topology_control.md Section 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ree_core.claustrum.control_demand import ControlDemandType


@dataclass(frozen=True)
class CoalitionTemplate:
    """Static recruit/suppress/gain data for one ControlDemandType.

    Consumed by CoalitionController.request_coalition() to build a live
    CoalitionState (which additionally carries opened_tick,
    max_duration_ticks, completion_condition -- per-request, not per-
    template). CoalitionState.__post_init__ re-clamps participating/
    suppressed to [0, 1] regardless of what is stored here, so this class
    does not need to duplicate that validation.
    """

    participating: Dict[str, float] = field(default_factory=dict)
    suppressed: Dict[str, float] = field(default_factory=dict)
    channel_gain: Dict[str, float] = field(default_factory=dict)


COALITION_TEMPLATES: Dict[ControlDemandType, CoalitionTemplate] = {
    ControlDemandType.SENSORY_RESAMPLE: CoalitionTemplate(
        participating={
            "e1_sensory_encoder": 0.9,
            "e2_fast_forward_model": 0.9,
        },
        suppressed={},
        channel_gain={"e3_candidate_count": 1.5},
    ),
    ControlDemandType.PROVENANCE_CHECK: CoalitionTemplate(
        participating={
            "hippocampal_anchor_set": 0.9,
            "hippocampal_persistence_appraisal": 0.9,
            "e3_commitment_monitor": 0.9,
        },
        suppressed={
            "motor_commitment": 0.6,
            "hippocampal_write_consolidation": 0.5,
        },
        channel_gain={},
    ),
}
