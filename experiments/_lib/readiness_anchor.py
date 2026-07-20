"""Reachability guard for readiness anchors (Learning 1, SD-068 REM fanout autopsy).

THE FAILURE MODE THIS CLOSES
----------------------------
A readiness anchor asserts that a known-degenerate POSITIVE CONTROL reproduces the
signature it is supposed to reproduce. The anchor is scored by a predicate, and the
predicate is written by hand. If the predicate is NARROWER than the degeneracy it
anchors to, the control cannot score above the gate NO MATTER HOW FAITHFULLY IT
REPLICATES -- the precondition is unmeetable by construction, and every run reports
`met: false` forever.

That is not a substrate problem; it is an instrument-specification bug. But the
self-route reads it as one, and mislabels the cause (`substrate_not_ready_requeue`
when nothing about the substrate was unready). Confirmed instance:
`v3_exq_sd068_rem_unpaired_null_diagnostic.py` V3-EXQ-778d, whose
`null_zero_anchor_reproduces_778c_railed_signature` predicate detected only the
SATURATION rail of a two-rail degeneracy. The maximum score a bit-perfect replication
could achieve was 5/8 = 0.625 against a gate of 0.75. See
`REE_assembly/evidence/planning/failure_autopsy_SD-068-rem-fanout-cluster_2026-07-18.md`
section 2.

THE GUARD
---------
Before the expensive phase runs, replay the KNOWN-DEGENERATE REFERENCE through the
SHIPPED predicate and assert it clears the gate. A gate the reference itself cannot
pass is a guaranteed false negative, and this check catches it at design-audit time
rather than after a multi-hour run and an autopsy.

    from experiments._lib.readiness_anchor import assert_anchor_reachable

    guard = assert_anchor_reachable(
        anchor_name="null_zero_anchor_reproduces_778c_railed_signature",
        reference_cells=_REFERENCE_778C_NULL_ZERO,   # frozen recorded fixture
        score_fn=_railed,                            # THE SHIPPED PREDICATE, not a copy
        threshold=ANCHOR_MIN_RAILED_SEEDS_FRAC,
        reference_source="V3-EXQ-778c rem null, replicated bit-identically by 778d",
    )

Two rules make it worth anything:

1. `score_fn` MUST be the SAME callable the run scores its live cells with. A
   re-implementation for the guard defeats the entire purpose -- the bug being
   caught IS a mis-specified predicate.
2. `reference_cells` MUST be the recorded values of a control whose degeneracy is
   already established, frozen as a literal so the guard needs no compute and cannot
   drift with the substrate.

THE MIRROR FAILURE THIS GUARD DOES *NOT* CATCH
----------------------------------------------
This guard tests a FLOOR: can the reference clear the gate. It says nothing about a
gate that is too EASY. A VACUOUS anchor -- one almost nothing can fail -- is the exact
mirror of 778d and is equally a mislabel: 778d over-fails and blames the substrate,
a vacuous anchor under-fails and lets a run emit a confident verdict on an untrained
channel. `assert_anchor_reachable` will happily certify a vacuous anchor as reachable.

Note the trap: the AnchorUnreachable message advises "widen the predicate or lower the
gate". Applied to a vacuous anchor that is precisely BACKWARDS. Read the direction of
the defect before acting on that advice.

The usual cause is an EXISTENCE QUANTIFIER standing in for a POPULATION PROPERTY:

    max_h_pos = max(r["h_pos_max"] for r in seed_results)   # over seeds AND episodes
    movement_ok = max_h_pos >= H_POS_MOVEMENT_FLOOR

One seed's single lucky episode satisfies this while every other seed is stationary,
so it can essentially never report `met: false`.

    A global max is CORRECT when the anchor asks "does this exist at all / is the
    channel non-degenerate". It is WRONG when the anchor asks "is the population
    ready", because readiness is a per-unit property and max is an existence
    quantifier.

Correct uses of the max form, for contrast: `v3_exq_730_q080a_effort_harm_coupling.py`
(`max_perm_peak > 0.0` -- did exertion vary at all) and
`v3_exq_669b_mech329_wanting_first_goal_seeding.py` (`max_anchor >= 2` -- is the anchor
hierarchy non-degenerate). Both ask an existence question, so max is the right verb.

Two rules for a population-readiness anchor:

3. Score the SAME STATISTIC the load-bearing criterion routes on. If the criterion
   gates on a per-seed sustained mean, the anchor must too -- typically the FRACTION
   of seeds clearing the floor, not a global max. A floor orders of magnitude below
   the criterion's own gate is a smell (591: floor 0.20 vs criterion 0.994).
4. Use `margin_cells` whenever the reference passes by a thin margin. An anchor that
   scores exactly at its gate flips to unmeetable on any seed-level drift. Either set
   a margin, or record an explicit comment that the zero margin is known and intended.

ALREADY-RAN DEFECTS: RECORD THEM, DO NOT SILENCE THEM
-----------------------------------------------------
A script that HAS one of these defects, has ALREADY RUN, and whose repair belongs in a
successor EXQ letter is a THIRD status, distinct from both "guarded" and "exempt".
Adding a guard in place would force a threshold or predicate change that RETROACTIVELY
ALTERS WHAT THE RECORDED EVIDENCE MEANS: the manifest on disk was produced by the
shipped predicate, and a repaired predicate no longer describes it. So the old script
must keep its defect exactly as it ran.

Declare that with `ANCHOR_REACHABILITY_SUPERSEDED = "<successor EXQ + reason>"`.

    NOT `ANCHOR_REACHABILITY_EXEMPT`. EXEMPT means "there is no defect -- reachability
    holds by construction", and it SILENCES the lint. Using it here makes an unrepaired
    defect indistinguishable from a fixed one. SUPERSEDED deliberately does NOT silence
    the warning: it annotates it with the successor, so the backlog stays visible and
    stays triageable. Confirmed error, 2026-07-19: V3-EXQ-778d was given an EXEMPT on
    exactly these grounds. The reasoning was defensible and the reason string was
    specific; it was still wrong, and it silenced the anchor-reachability gate's own
    live regression specimen (breaking two contract tests). It was reverted -- 778d
    deliberately keeps its warning.

Known-defective instances, recorded but NOT fixed (lineage blocked, already ran -- a fix
needs a new EXQ letter, not an in-place edit):

  * the `591b/c/d/e/f` ISEF-005 family's `early_policy_produces_nontrivial_h_pos`
    (rule 3) and `591d/e/f`'s `false_advancer_present` (rule 4, zero margin on a
    0.06-nat threshold). See
    `REE_assembly/evidence/planning/infant_substrate_plan.md` `infant_substrate:GAP-14`
    governance_2026_07_18.
  * `v3_exq_sd068_rem_unpaired_null_diagnostic.py` V3-EXQ-778d -- the originating
    instance described at the top of this file, superseded by V3-EXQ-778h
    (`v3_exq_sd068_rem_unpaired_null_anchorfix_diagnostic.py`). It additionally serves
    as the live regression specimen for the lint, so its warning is load-bearing;
    `validate_experiments._LINT_SPECIMEN_FILES` records that and warns anyone about to
    mark it.
  * `v3_exq_sd068_consolidation_staged_damage_diagnostic.py` and
    `..._consolidation_staging_power_diagnostic.py` -- `intact_readouts_nondegenerate`
    (`INTACT_SIGNAL_FLOOR = 1e-9` against recorded readouts of 0.5 and ~5.4e3). A VACUOUS
    anchor: the mirror failure documented at the top of this file, in its strongest rule-3
    form. C1 routes on `H._normalise_degradation()`, a min-max rescale to [0,1] over each
    phase's own sigma series, so C1 is SCALE-INVARIANT to the raw levels this anchor gates;
    and the ratio denominators are separately guarded by inline `> 1e-12` tests at every
    sigma. So NO floor value would make it a readiness precondition -- the repair is a
    different anchor under a new EXQ letter, not a retune. Recorded 2026-07-19, not fixed
    (both drivers already ran). The other four SD-068 anchors audited in the same pass are
    legitimate denominator / existence gates whose wide margins are known and intended. See
    `REE_assembly/evidence/planning/vacuous_readiness_anchors_SD-068_2026-07-19.md`.

ASCII-only output (Windows runner terminals).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence


class AnchorUnreachable(AssertionError):
    """A readiness anchor's gate is unreachable by its own known-degenerate reference."""


def score_reference(
    *,
    reference_cells: Sequence[Any],
    score_fn: Callable[[Any], bool],
) -> Dict[str, Any]:
    """Apply the shipped predicate to the frozen reference; return the fraction scored True."""
    if not reference_cells:
        raise ValueError("reference_cells is empty; a guard with no reference proves nothing")
    flags: List[bool] = [bool(score_fn(cell)) for cell in reference_cells]
    n = len(flags)
    n_true = sum(1 for f in flags if f)
    return {
        "n_reference_cells": n,
        "n_reference_scored_true": n_true,
        "reference_score": float(n_true) / float(n),
        "per_cell_scored": flags,
    }


def assert_anchor_reachable(
    *,
    anchor_name: str,
    reference_cells: Sequence[Any],
    score_fn: Callable[[Any], bool],
    threshold: float,
    reference_source: str = "",
    margin_cells: int = 0,
) -> Dict[str, Any]:
    """Fail loudly at setup if the shipped predicate cannot score the reference above the gate.

    Args:
        anchor_name: the precondition name this guards, for the error message.
        reference_cells: recorded per-cell values of the known-degenerate positive
            control, frozen as a literal.
        score_fn: THE SHIPPED PREDICATE -- the same callable the live cells are
            scored with. Takes one reference cell, returns bool.
        threshold: the gate the anchor's measured fraction must clear (a floor).
        reference_source: provenance string, recorded in the returned payload.
        margin_cells: optional headroom, expressed in cells. Requires the reference
            to clear the gate by at least this many cells, so an anchor that only
            JUST passes on the reference (and would fail on any seed-level jitter)
            is also refused. Default 0 = clear the gate exactly.

    Returns:
        A payload suitable for embedding in the manifest `interpretation` block.

    Raises:
        AnchorUnreachable: the reference scores below the gate. The gate is a
            guaranteed false negative -- widen the predicate or lower the gate.
    """
    scored = score_reference(reference_cells=reference_cells, score_fn=score_fn)
    n = scored["n_reference_cells"]
    reference_score = scored["reference_score"]
    required = float(threshold) + (float(margin_cells) / float(n))
    reachable = reference_score >= required

    payload: Dict[str, Any] = {
        "anchor_name": anchor_name,
        "guard": "assert_anchor_reachable",
        "reference_source": reference_source,
        "threshold": float(threshold),
        "margin_cells": int(margin_cells),
        "required_score": required,
        "reachable": bool(reachable),
        **scored,
    }

    if not reachable:
        raise AnchorUnreachable(
            "READINESS ANCHOR UNREACHABLE BY ITS OWN REFERENCE -- refusing to run.\n"
            f"  anchor:           {anchor_name}\n"
            f"  reference:        {reference_source or '(unspecified)'}\n"
            f"  reference scores: {scored['n_reference_scored_true']}/{n} "
            f"= {reference_score:.4f} under the SHIPPED predicate\n"
            f"  gate demands:     >= {required:.4f} "
            f"(threshold {float(threshold):.4f}, margin {int(margin_cells)} cell(s))\n"
            "  A precondition that a bit-perfect replication of the known-degenerate\n"
            "  control CANNOT pass is a guaranteed false negative: it will report\n"
            "  met=false on every run and mislabel an instrument-specification gap as\n"
            "  a substrate or scientific verdict. Widen the predicate to cover the\n"
            "  whole degeneracy, or lower the gate. Do NOT run as-is.\n"
            "  See failure_autopsy_SD-068-rem-fanout-cluster_2026-07-18.md section 2."
        )
    return payload
