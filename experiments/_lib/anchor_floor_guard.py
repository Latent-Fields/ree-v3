"""Refuse a ceiling-class self-route on a learner scoring BELOW its own random anchor.

WHY THIS MODULE EXISTS (autopsy competence-objective cluster 734/737b/742a, 2026-07-22,
sec 5 Learning 2, user-adjudicated confirmed).

    "A learner below the random-walk anchor falsifies 'ceiling' readings by construction.
     Adopt this as a standing lint: any substrate_ceiling / learner_ceiling self-route on
     a cell scoring below its own declared random-policy floor should be refused at the
     manifest level. The information was present in 734/737b/742a and in every
     predecessor; nothing consumed it."

Under a capacity or representation CEILING a learner asymptotes toward the anchor from
below and plateaus; it does NOT end up systematically worse than acting at random on a
task a same-observation greedy solves comfortably. A trained learner scoring below its own
declared random_walk anchor has therefore NOT been shown to lack a mechanism -- it has been
shown to be optimising something other than the scored DV (the 734 survival-vs-forage
inversion: vanilla PPO survived 175.0 steps vs the greedy oracle's 20.4 while foraging 17x
less). Emitting a `substrate_ceiling` / `learner_or_observability_ceiling` verdict on that
run mislabels an objective-misspecification as a substrate limitation, which is the exact
mis-route the autopsy exists to prevent.

RELATIONSHIP TO zworld_encoder_guard. That guard catches a DIFFERENT failure: an untrained
world encoder (a frozen random projection) invalidating the run's premise. It fires on the
INPUT side (was z_world trained). This guard fires on the OUTPUT side (is the measured DV
below random), and catches the case the encoder guard cannot: 737b ran a genuinely
prediction-trained encoder (guard GREEN) and STILL scored ppo_ree_latent 0.233 vs
random_walk 0.933, and the ceiling reading was still wrong. Both are needed.

SCOPE: DETECTION + REFUSAL, no re-run. This module decides whether a computed self-route
label is supported by the measured scores and, if not, DOWNGRADES it to the safe non-verdict
`substrate_not_ready_requeue` (the same requeue route the readiness gates already use). It
never fabricates a positive verdict and never touches the substrate. A refused ceiling route
draws NO conclusion about the competence root -- it says only "these numbers cannot license
a ceiling reading; re-examine the objective / re-queue".

METHOD: strict `<` comparison against the run's OWN declared random_walk anchor at the SAME
rung. A cell with no measurement (None -- e.g. a zworld_encoder_guard-refused cell) is
skipped: a refused cell can neither support nor refute a verdict.

ASCII-only in all runtime strings (repo rule).

Reference:
  REE_assembly/evidence/planning/
  failure_autopsy_competence-objective-cluster-734-737b-742a_2026-07-22.md  (sec 5.2)
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

# The self-route labels a below-random learner falsifies. These are "ceiling"-class
# readings -- a run asserting that some LEARNER cannot clear a floor a control can, framed
# as a capacity / representation / observation-encoding limitation. Kept deliberately WIDE
# so a driver that spells the label its own way (`ree_substrate_ceiling`) is still covered;
# a driver adding a new ceiling spelling must add it here (and the source lint in
# validate_experiments.py keys on the same set).
CEILING_ROUTE_LABELS = frozenset({
    "substrate_ceiling",
    "ree_substrate_ceiling",
    "learner_ceiling",
    "learner_or_observability_ceiling",
})

# The conventional floor-anchor arm name across the capability_eval family
# (capability_eval.RandomPolicy; build_report(..., floor="random_walk")).
RANDOM_ANCHOR_ARM = "random_walk"

# The safe non-verdict a refused ceiling route downgrades to -- the same requeue route the
# readiness gates already emit, so a manifest reader and the indexer treat it identically.
REQUEUE_LABEL = "substrate_not_ready_requeue"

GUARD_NAME = "ceiling_route_not_below_random_anchor"

REFERENCE_DOC = (
    "REE_assembly/evidence/planning/"
    "failure_autopsy_competence-objective-cluster-734-737b-742a_2026-07-22.md"
)


class CeilingBelowRandomAnchorError(RuntimeError):
    """Raised (strict mode) when a ceiling-class self-route rests on a learner cell that
    scored below its own declared random_walk anchor."""


def is_ceiling_route(label: Optional[str]) -> bool:
    """True iff `label` is a ceiling-class self-route this guard governs."""
    return label in CEILING_ROUTE_LABELS


def sub_random_learner_arms(
    learner_scores: Mapping[str, Optional[float]],
    random_anchor: Optional[float],
) -> List[Tuple[str, float]]:
    """Sorted [(arm, score)] for learner arms scoring STRICTLY below the random anchor.

    `learner_scores` is the LEARNER arms only -- the caller excludes the random_walk anchor
    and the greedy_oracle ceiling, because a floor anchor is not below itself and an oracle
    is not a learner. A None score (an unmeasured / guard-refused cell) is skipped: it can
    neither support nor refute a verdict. Returns [] when the anchor is None (nothing to
    compare against) or no learner is below it.
    """
    if random_anchor is None:
        return []
    below: List[Tuple[str, float]] = []
    for arm, score in learner_scores.items():
        if score is None:
            continue
        if float(score) < float(random_anchor):
            below.append((str(arm), float(score)))
    return sorted(below, key=lambda t: (t[1], t[0]))


def ceiling_route_refusal(
    label: Optional[str],
    learner_scores: Mapping[str, Optional[float]],
    random_anchor: Optional[float],
    *,
    rung: str = "",
    context: str = "",
) -> Optional[Dict[str, Any]]:
    """Manifest-embeddable refusal record, or None when the route stands.

    Returns None unless `label` is a ceiling-class route AND at least one learner arm scored
    strictly below `random_anchor`. The record names the offending arms and their deficits,
    the anchor, the label refused, and the safe label it downgrades to, so a manifest reader
    (and the REE_assembly indexer) can re-derive the refusal from the numbers alone.
    """
    if not is_ceiling_route(label):
        return None
    below = sub_random_learner_arms(learner_scores, random_anchor)
    if not below:
        return None
    return {
        "name": GUARD_NAME,
        "kind": "verdict_guard",
        "refused": True,
        "original_label": label,
        "downgraded_to": REQUEUE_LABEL,
        "random_walk_anchor": float(random_anchor),
        "sub_random_arms": [
            {"arm": arm, "score": score,
             "deficit_below_random": round(float(random_anchor) - score, 6)}
            for arm, score in below
        ],
        "rung": str(rung),
        "context": str(context),
        "rationale": (
            "A learner scoring below its own declared random_walk anchor has NOT been shown "
            "to lack a mechanism; under a capacity/representation ceiling a learner "
            "asymptotes toward the anchor from below, it does not end up worse than random. "
            "A below-random score is the signature of optimising a different objective than "
            "the scored DV (the 734 survival-vs-forage inversion), so a ceiling verdict is "
            "unsupported. Downgraded to " + REQUEUE_LABEL + ": draw NO conclusion about the "
            "competence root; re-examine the objective / re-queue."
        ),
        "reference": REFERENCE_DOC,
    }


def refuse_ceiling_below_random(
    label: Optional[str],
    learner_scores: Mapping[str, Optional[float]],
    random_anchor: Optional[float],
    *,
    strict: bool = False,
    rung: str = "",
    context: str = "",
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Apply the standing lint to a computed self-route.

    Returns `(safe_label, record)`:
      * when the route stands (not a ceiling label, or every learner is at/above the
        anchor, or nothing measurable) -> `(label, None)`, unchanged;
      * when a ceiling route rests on a sub-random learner -> `(REQUEUE_LABEL, record)`,
        the label downgraded and the refusal recorded.

    With `strict=True` a refusal RAISES CeilingBelowRandomAnchorError instead of downgrading
    -- for a caller (contract / driver) that wants a hard stop rather than a silent requeue.
    """
    record = ceiling_route_refusal(
        label, learner_scores, random_anchor, rung=rung, context=context,
    )
    if record is None:
        return label, None
    if strict:
        arms = ", ".join(
            f"{d['arm']}={d['score']}" for d in record["sub_random_arms"]
        )
        raise CeilingBelowRandomAnchorError(
            f"CEILING ROUTE REFUSED: label={label} rests on learner(s) below the "
            f"random_walk anchor {record['random_walk_anchor']} "
            f"[{arms}] (rung={rung or '-'}, {context or '-'}). "
            "A learner below its own random floor cannot license a ceiling verdict; see "
            + REFERENCE_DOC + "."
        )
    return REQUEUE_LABEL, record


__all__ = [
    "CEILING_ROUTE_LABELS",
    "RANDOM_ANCHOR_ARM",
    "REQUEUE_LABEL",
    "GUARD_NAME",
    "REFERENCE_DOC",
    "CeilingBelowRandomAnchorError",
    "ceiling_route_refusal",
    "is_ceiling_route",
    "refuse_ceiling_below_random",
    "sub_random_learner_arms",
]
