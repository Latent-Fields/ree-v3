"""The RESOLVED run-mode sampling budget, carried into arm contexts.

THE DEFECT THIS CLOSES
----------------------
`precondition_gate.assert_no_structurally_unsatisfiable_gate` can only reason
from a `structural_max` / `structural_min` a spec declares. For the commonest
shape in the corpus -- a RAW PER-TICK COUNTER against an integer sample floor --
the bound is trivially derivable: a cell cannot make more fresh selections (or
committed decisions, or collected transitions) than it takes env steps, so the
step budget IS the best attainable value.

It was not declarable, because THE ARM CONTEXT DID NOT CARRY THE BUDGET. Arm
contexts are built from the pre-registered arm table -- identity plus regime
flags -- and in several drivers that table is a module-level constant that never
learns whether this invocation is a `--dry-run`. So the one number the bound
needs was the one number absent at the point the bound would be written.

V3-EXQ-1062 is the confirmed instance: its `fresh_select_sample_floor` of 200
was structurally unreachable under its own --dry-run P2 budget of 60 steps, all
eight of its `PreconditionSpec`s omitted the bounds, and the guard ran clean.
See `failure_autopsy_V3-EXQ-1062_2026-09-22.md` section 6, findings 1 and 2, and
`precondition_gate_structural_bound_triage.md` (7 confirmed binding drivers).

THE TWO HALVES -- AND WHY DECLARING THE BOUND ALONE IS A REGRESSION
-------------------------------------------------------------------
Declaring `structural_max=budget_ceiling` and stopping there makes the guard
RAISE on every `--dry-run` of an affected driver: the reduced budget genuinely
cannot reach a floor sized for the full run. That is a true statement and a
useless one -- it removes the smoke test, and CLAUDE.md is explicit that a guard
which fires on correct code gets disabled. It is also not the defect: 1062's
harm was never that its smoke passed, it was that its smoke FAILED for a reason
indistinguishable from a real substrate failure (`outcome=FAIL
label=substrate_not_ready_requeue`, bit-identical to the real run, autopsy
finding 2).

So the remedy is TWO coupled halves, and this module supplies both:

  1. `RunBudget.scaled_floor(floor)` -- under `--dry-run` ONLY, scale the sample
     floor by the reduced budget, so the readiness gate is actually EVALUATED
     instead of being arithmetically foreclosed and routing
     `substrate_not_ready_requeue` vacuously.

     SCOPE, stated because it is easy to over-read: this makes the READINESS
     gate a real measurement and lets the analysis path run. It does NOT by
     itself make the smoke's VERDICT informative -- a driver whose dry run uses
     one seed can still have its load-bearing criterion predetermined by the
     seed count (V3-EXQ-1018: `signs_agree` requires >= 2 per-seed deltas, so
     `c1_passed` is False on every dry run regardless of substrate). Scaling
     moves the smoke off a vacuous readiness FAIL; it does not turn a one-seed
     smoke into a discriminating test, and nothing here should be read as
     claiming it does.
  2. `budget_ceiling` -- declare the counter's structural max against the
     RESOLVED budget the context now carries, so the guard proves something on
     the real run.

Half 1 without half 2 leaves the guard inert (the 1062 state). Half 2 without
half 1 wedges every smoke. V3-EXQ-1062a derived exactly this pair by hand
(`_active_sample_floors` + `structural_max=lambda ctx: float(ctx["p2_budget"])`);
this module is that pattern made reusable, and 1062a is its worked reference.

A REAL RUN NEVER HAS A THRESHOLD RELAXED. `scaled_floor` returns the
pre-registered constant unchanged when `dry_run` is False -- the scaling branch
is unreachable on a scored run. Pinned directly by
`test_real_run_branch_is_pinned_DIRECTLY_not_via_post_init`, which passes a
floor BELOW the clamp: that is the only input separating "the branch returned
early" from "__post_init__ happened to force scale == 1.0", and without it the
branch is untested even though the property holds.

USAGE -- a driver opts in with one line each side
-------------------------------------------------
    from experiments._lib.run_budget import RunBudget, budget_ceiling

    def run_experiment(dry_run: bool):
        n_ticks = 25 if dry_run else MEASURE_TICKS
        budget = RunBudget(ticks=n_ticks, nominal_ticks=MEASURE_TICKS,
                           dry_run=dry_run, cells=len(SEEDS) * len(ARMS))

        floor = budget.scaled_floor(MIN_FRESH_SELECTS)      # <- half 1
        arm_ctxs = budget.attach(_arms())                   # <- carries the budget
        assert_no_structurally_unsatisfiable_gate(SPECS, arm_ctxs)

    PreconditionSpec(
        name="fresh_selects_per_cell",
        threshold=floor,
        direction="lower",
        structural_max=budget_ceiling,                      # <- half 2
    )

`budget_ceiling` is a plain function, so it drops straight into the existing
`structural_max` slot: `precondition_gate`'s API is unchanged and it never
imports this module. Drivers that do not opt in behave exactly as before.

PER-CELL VERSUS RUN-AGGREGATE -- the distinction that decides the bound
-----------------------------------------------------------------------
Get this wrong and the bound is a fabrication rather than a proof, which is
worse than no bound at all (`precondition_gate`'s own docstring: manufacturing
support is the worse failure).

  * `budget_ceiling` -- the measured value is ONE CELL's counter, or the WORST
    cell's. Ceiling is `ticks`.
  * `budget_ceiling_total` -- the measured value is SUMMED OVER CELLS (e.g.
    `sum(r["n_committed"] for r in arm_results)`). Ceiling is `ticks * cells`.

Read the driver's measured-value expression before choosing. A `_worst_cell(...,
"min")` is per-cell; a `sum(... for r in arm_results)` is aggregate.

ASCII-only in printed output (Windows cp1252 terminals).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "RunBudget",
    "budget_ceiling",
    "budget_ceiling_total",
    "BUDGET_TICKS_KEY",
    "BUDGET_CELLS_KEY",
    "BUDGET_DRY_RUN_KEY",
]

# The context keys this module writes and `budget_ceiling` reads. Named
# constants rather than bare literals so a driver that builds its context dict
# by hand cannot drift from the reader.
BUDGET_TICKS_KEY = "budget_ticks"
BUDGET_CELLS_KEY = "budget_cells"
BUDGET_DRY_RUN_KEY = "budget_dry_run"

# The weakest floor a scaled SAMPLE-COUNT gate may take. A floor of 0 is
# vacuously green (every run has >= 0 samples) -- the inverse of 1062's failure
# and the easier one to miss -- so the scaled value is clamped here instead.
# 0.5 is deliberate and is the WEAKEST NON-VACUOUS value for a count: the gate
# tests `measured > threshold`, so it demands at least ONE observation.
#
# It is NOT an integer, and that is load-bearing. Truncating the scaled floor to
# an int rounds it DOWN (more lenient than proportional) and then clamps it back
# UP to a value the reduced budget may not be able to reach -- which reintroduces
# 1062's failure through the back door. Measured on V3-EXQ-1018: floor 30 scales
# to 0.833 at a 25/900 budget, and the worst arm yields exactly 1 fresh select,
# so the faithful float PASSES while `max(2, int(0.833))` would have routed
# substrate_not_ready_requeue on a smoke that behaved exactly as designed.
MIN_SCALED_FLOOR = 0.5


@dataclass(frozen=True)
class RunBudget:
    """The resolved per-cell sampling budget for ONE invocation.

    `ticks` is the number this invocation actually gets -- already reduced if
    this is a `--dry-run`. `nominal_ticks` is what a real run would use, and is
    the denominator of the scaling ratio. Keeping both is what lets
    `scaled_floor` report the pre-registered value alongside the scaled one, so
    a smoke's relaxed floor is stated rather than silently applied.

    `cells` is how many cells a RUN-AGGREGATE counter sums over -- normally
    `len(seeds) * len(arms)` for this invocation, NOT for a full run. It is used
    only by `budget_ceiling_total`; per-cell bounds ignore it.

    Frozen: the budget is resolved once, at the top of the run, and a bound that
    could be mutated between the guard call and the measurement would not be a
    design-time proof.
    """

    ticks: int
    nominal_ticks: int
    dry_run: bool
    cells: int = 1
    label: str = "ticks"

    def __post_init__(self) -> None:
        if int(self.ticks) < 0 or int(self.nominal_ticks) < 0:
            raise ValueError("RunBudget ticks/nominal_ticks must be non-negative")
        if int(self.cells) < 1:
            raise ValueError("RunBudget cells must be >= 1")
        if not self.dry_run and int(self.ticks) != int(self.nominal_ticks):
            # A real run whose resolved budget differs from its nominal one has
            # had a pre-registered quantity changed without saying so. Refuse:
            # every bound and every scaled floor downstream would be computed
            # against a budget the manifest does not record.
            raise ValueError(
                "RunBudget: a REAL run must have ticks == nominal_ticks "
                "(got %d vs %d). A scored run never runs on a reduced budget; "
                "if this invocation really is reduced, pass dry_run=True."
                % (int(self.ticks), int(self.nominal_ticks)))

    # -- scaling (half 1) ---------------------------------------------------- #

    @property
    def scale(self) -> float:
        """Reduced-budget ratio in (0, 1]; exactly 1.0 on a real run."""
        if not self.dry_run:
            return 1.0
        if int(self.nominal_ticks) <= 0:
            return 1.0
        return float(self.ticks) / float(self.nominal_ticks)

    def scaled_floor(self, floor: float, minimum: float = MIN_SCALED_FLOOR,
                     name: str = "", report: bool = True) -> float:
        """The SAMPLE-COUNT floor to USE for this invocation.

        Only for a floor whose units are SAMPLES (fresh selections, committed
        decisions, collected transitions) -- a quantity that scales with the
        budget. Do NOT scale a floor on a learned or bounded quantity (an R^2, a
        correlation, a fidelity error): those do not get easier to reach on a
        shorter run, so scaling one silently weakens a real gate.

        A REAL run returns `floor` unchanged -- no threshold is ever relaxed for
        a scored run, and that branch is pinned by contract. Under `--dry-run`
        the floor is scaled by `self.scale` and clamped at `minimum`, so the
        smoke tests the scoring path rather than routing
        `substrate_not_ready_requeue` bit-identically to a real failure
        (V3-EXQ-1062 autopsy section 6, finding 2).

        `report` prints the scaled and pre-registered values together, because a
        relaxed floor that is not stated is indistinguishable from the
        pre-registered one to anyone reading the smoke's output.
        """
        if not self.dry_run:
            return float(floor)
        scaled = max(float(minimum), float(floor) * self.scale)
        if report:
            print("  [smoke] sample floor%s scaled to the dry-run budget: "
                  "%.4g (pre-registered %g; budget %d/%d %s)"
                  % ((" '%s'" % name) if name else "", scaled, float(floor),
                     int(self.ticks), int(self.nominal_ticks), self.label))
        return float(scaled)

    # -- context injection --------------------------------------------------- #

    def as_ctx(self) -> Dict[str, Any]:
        """The budget keys to merge into an arm context."""
        return {
            BUDGET_TICKS_KEY: int(self.ticks),
            BUDGET_CELLS_KEY: int(self.cells),
            BUDGET_DRY_RUN_KEY: bool(self.dry_run),
        }

    def attach(self, arm_dicts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Copy each arm dict with the budget keys merged in.

        Copies rather than mutates: several drivers hold `ARM_CONTEXTS` as a
        module-level constant, and mutating it in place would leave a
        run-mode-specific budget on a shared object for the rest of the process
        -- which in a test session is a different invocation's budget.
        """
        return [dict(a, **self.as_ctx()) for a in arm_dicts]

    # -- manifest ------------------------------------------------------------ #

    def manifest_block(self) -> Dict[str, Any]:
        """Serialisable record of the resolved budget, for `full_config`."""
        return {
            "resolved_ticks": int(self.ticks),
            "nominal_ticks": int(self.nominal_ticks),
            "dry_run": bool(self.dry_run),
            "cells": int(self.cells),
            "scale": float(self.scale),
            "unit": str(self.label),
        }


# -- the bounds (half 2) ----------------------------------------------------- #

def _require_budget(ctx: Dict[str, Any], key: str) -> Optional[float]:
    """Budget value from `ctx`, or None when the context does not carry one.

    Returning None (rather than 0, or raising) is deliberate: `precondition_
    gate._spec_structural_verdict` reads a None bound as `not_evaluated` -- the
    explicit cannot-determine category -- so a driver that declares
    `budget_ceiling` but forgets `budget.attach(...)` is reported as UNCHECKED
    rather than silently bounded at zero. A zero would be read as a proof that
    the counter can never exceed 0, making every floor `unsatisfiable` and
    wedging the run on a bound that was never actually derived.
    """
    if key not in ctx:
        return None
    value = ctx[key]
    if value is None:
        return None
    return float(value)


def budget_ceiling(ctx: Dict[str, Any]) -> Optional[float]:
    """PER-CELL structural max: a cell's counter cannot exceed its tick budget.

    Drop straight into a floor spec's `structural_max`. Use for a measured value
    that is one cell's counter or the WORST cell's -- not a cross-cell sum.
    """
    return _require_budget(ctx, BUDGET_TICKS_KEY)


def budget_ceiling_total(ctx: Dict[str, Any]) -> Optional[float]:
    """RUN-AGGREGATE structural max: `ticks * cells`.

    Use only when the measured value is SUMMED over every cell of the run. On a
    per-cell measurement this over-states the ceiling by a factor of `cells`,
    which converts a binding bound into a non-binding one -- the bound would
    still be true, but it would stop proving the thing it was written for.
    """
    ticks = _require_budget(ctx, BUDGET_TICKS_KEY)
    cells = _require_budget(ctx, BUDGET_CELLS_KEY)
    if ticks is None or cells is None:
        return None
    return float(ticks) * float(cells)
