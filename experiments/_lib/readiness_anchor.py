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
