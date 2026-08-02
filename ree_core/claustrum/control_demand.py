"""
ControlDemandType -- MECH-481 typed control-demand taxonomy.

SD-091 asserts the control plane needs a second, graph-valued output G_t
(coalition/topology control) alongside the existing mode M_t and parameter
theta_t outputs (ARC-005/MECH-004). MECH-481 asserts the specific mechanism:
typed control demands map to typed coalition templates.

Biological mapping is explicitly PROVISIONAL: the claustrum correspondence
here is functional/hypothesis-generating (Krimmel et al Network Instantiation
in Cognitive Control account), not anatomical. This module does not claim the
claustrum is the seat of consciousness or metacognition -- see
REE_assembly/docs/architecture/sd_091_coalition_topology_control.md Section 4
("Guardrails carried forward as hard constraints") for the full list this
module and coalition_controller.py are structurally bound to.

All 10 taxonomy classes from MECH-481's functional_restatement are declared
here up front (cheap, no dependency, avoids an enum-schema break later), but
only 2 (SENSORY_RESAMPLE, PROVENANCE_CHECK) have real coalition templates as
of this MVP pass -- see coalition_templates.py. Requesting an unregistered
type is a no-op with a diagnostic counter increment (CoalitionController's
"safe no-op default" convention), never a crash.
"""

from __future__ import annotations

from enum import Enum


class ControlDemandType(Enum):
    """The 10 typed control-demand classes named in MECH-481's
    functional_restatement (claims.yaml). Only SENSORY_RESAMPLE and
    PROVENANCE_CHECK are templated in this MVP pass -- see
    coalition_templates.COALITION_TEMPLATES and
    sd_091_coalition_topology_control.md Section 2 for the build-order
    triage of the remaining 8.
    """

    # -- Templated in this MVP pass --
    SENSORY_RESAMPLE = "sensory_resample"
    PROVENANCE_CHECK = "provenance_check"

    # -- Register-only: declared for taxonomy stability, no template yet --
    COUNTERFACTUAL_EXPANSION = "counterfactual_expansion"
    CROSS_HORIZON_RECONCILIATION = "cross_horizon_reconciliation"
    SOCIAL_MODEL_CHECK = "social_model_check"
    INVARIANT_CONFLICT_REVIEW = "invariant_conflict_review"
    ACTION_OUTCOME_RECALIBRATION = "action_outcome_recalibration"
    LANGUAGE_EXPLICITATION = "language_explicitation"
    COMMITMENT_REOPEN = "commitment_reopen"
    SAFE_DEFER = "safe_defer"


# The subset with a real coalition template in this MVP pass. Kept as a
# frozenset here (rather than only in coalition_templates.py) so
# CoalitionController can classify "templated vs register-only" without
# importing the templates module, avoiding a circular import.
MVP_TEMPLATED_TYPES = frozenset(
    {ControlDemandType.SENSORY_RESAMPLE, ControlDemandType.PROVENANCE_CHECK}
)
