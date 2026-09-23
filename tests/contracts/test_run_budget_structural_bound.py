"""Contracts for `experiments._lib.run_budget` -- the budget-aware structural bound.

WHAT THIS LOCKS DOWN
--------------------
`precondition_gate.assert_no_structurally_unsatisfiable_gate` proves nothing for
a spec declaring no `structural_max` / `structural_min`. For a raw per-tick
counter against a sample floor the bound is the run's own step budget -- but the
arm context did not carry it, so the bound could not be written. `run_budget`
carries it, and supplies the floor-scaling half that keeps the smoke usable.

THE BLIND-SPOT MEASUREMENT (CLAUDE.md: "a guard that supplies the thing it
asserts is not a guard" -- run the new test against the OLD defect and confirm
it FAILS, and confirm the OLD guard still PASSES on it)
------------------------------------------------------------------------------
`TestV3EXQ1062Regression` reconstructs V3-EXQ-1062's exact shape -- a
`fresh_select_sample_floor` of 200 against a --dry-run P2 budget of 60 steps --
and pins all THREE states rather than only the fixed one:

  * `test_old_shape_passes_and_proves_nothing`  -- the pre-change defect. The
    guard returns cleanly with coverage 0.0. This test PASSES on the old code
    too, by construction: it is the control that shows the old guard was blind
    here, not that it was absent.
  * `test_bound_alone_wedges_the_smoke`         -- half 2 without half 1. The
    guard RAISES on --dry-run. This is why the enabler is not just a bound: a
    guard that fires on a correct driver's smoke gets disabled (CLAUDE.md).
  * `test_both_halves_smoke_passes_and_proves_something` -- the shipped shape.
    Coverage is non-zero AND the guard does not raise.

The middle case is the one that could not be written before this module existed,
and it is the finding that shaped the design. The third fails against the old
module (`budget_ceiling` does not exist; the floor cannot scale).

`TestBoundIsNotFabricated` is the other half: a bound derived from a context
that never received one must be reported as NOT EVALUATED, never as a numeric
proof. A zero there would make every floor `unsatisfiable` and wedge the run on
a bound nobody derived -- manufacturing a refusal the way `precondition_gate`'s
docstring warns about manufacturing support.
"""

import pytest

from experiments._lib.precondition_gate import (
    PreconditionSpec,
    StructurallyUnsatisfiableGate,
    assert_no_structurally_unsatisfiable_gate,
    structural_vacuity_verdict,
    summarize_structural_audit,
)
from experiments._lib.run_budget import (
    BUDGET_CELLS_KEY,
    BUDGET_DRY_RUN_KEY,
    BUDGET_TICKS_KEY,
    RunBudget,
    budget_ceiling,
    budget_ceiling_total,
)

# V3-EXQ-1062's real numbers (autopsy section 6, finding 2).
EXQ1062_FRESH_FLOOR = 200
EXQ1062_DRY_P2_BUDGET = 60
EXQ1062_P2_STEP_BUDGET = 1800


def _fresh_spec(threshold, structural_max=None):
    return PreconditionSpec(
        name="fresh_select_sample_floor",
        description="fresh E3 selections in the worst cell",
        control="V3-EXQ-1062a measured ~1780 per cell at P2_STEP_BUDGET 1800",
        threshold=float(threshold),
        direction="lower",
        structural_max=structural_max,
    )


# --- half 1: floor scaling --------------------------------------------------- #

class TestScaledFloor:

    def test_real_run_never_relaxes_the_floor(self):
        """The scaling branch must be unreachable on a scored run."""
        budget = RunBudget(ticks=EXQ1062_P2_STEP_BUDGET,
                           nominal_ticks=EXQ1062_P2_STEP_BUDGET, dry_run=False)
        assert budget.scale == 1.0
        assert budget.scaled_floor(EXQ1062_FRESH_FLOOR) == float(EXQ1062_FRESH_FLOOR)

    def test_real_run_branch_is_pinned_DIRECTLY_not_via_post_init(self):
        """Blind-spot repair (red-team note 1, 2026-09-23).

        `test_real_run_never_relaxes_the_floor` does NOT actually pin the
        `if not self.dry_run: return float(floor)` branch: delete that branch
        and it still passes, because __post_init__ forces scale == 1.0 and
        `max(0.5, floor) == floor` for every floor >= 0.5. The property held for
        a reason unrelated to the code under test -- a guard supplying the thing
        it asserts (CLAUDE.md General Rules, "The test half").

        A floor BELOW the clamp is the only input that separates them: with the
        branch present it is returned untouched; with it deleted the clamp
        raises it to 0.5, silently strengthening a pre-registered threshold on a
        SCORED run. Measured: this assertion FAILS with the branch deleted and
        PASSES with it present, while all 17 pre-existing tests pass either way.
        """
        budget = RunBudget(ticks=100, nominal_ticks=100, dry_run=False)
        assert budget.scaled_floor(0.25, report=False) == 0.25

    def test_dry_run_scales_by_the_budget_ratio(self):
        budget = RunBudget(ticks=EXQ1062_DRY_P2_BUDGET,
                           nominal_ticks=EXQ1062_P2_STEP_BUDGET, dry_run=True)
        # 200 * (60/1800) = 6.667, kept as a float -- NOT truncated to 6
        assert budget.scaled_floor(EXQ1062_FRESH_FLOOR, report=False) == pytest.approx(200 * 60 / 1800)

    def test_scaled_floor_is_clamped_above_zero(self):
        """A floor of 0 makes the smoke's gate vacuously GREEN -- the inverse of
        1062's failure, and the easier one to miss. The clamp is the WEAKEST
        non-vacuous value for a count: `measured > threshold` demands >= 1."""
        budget = RunBudget(ticks=1, nominal_ticks=10_000, dry_run=True)
        assert budget.scaled_floor(5, report=False) >= 0.5
        assert budget.scaled_floor(5, report=False) < 1.0   # >= 1 sample, not >= 2

    def test_scaled_floor_is_not_truncated_to_an_int(self):
        """V3-EXQ-1018's measured case: floor 30 at a 25/900 budget scales to
        0.833 and the worst arm yields exactly 1 fresh select. `int()` here
        would round to 0, clamp back up, and route substrate_not_ready_requeue
        on a smoke that behaved exactly as designed."""
        budget = RunBudget(ticks=25, nominal_ticks=900, dry_run=True)
        floor = budget.scaled_floor(30, report=False)
        assert floor == pytest.approx(30 * 25 / 900)
        assert 1.0 > floor > 0.5            # one fresh select CLEARS it
        assert float(1) > floor

    def test_real_run_with_reduced_ticks_is_refused(self):
        """A scored run on a reduced budget would compute every bound and every
        floor against a budget the manifest does not record."""
        with pytest.raises(ValueError, match="REAL run"):
            RunBudget(ticks=60, nominal_ticks=1800, dry_run=False)


# --- half 2: the bound ------------------------------------------------------- #

class TestBudgetCeiling:

    def test_per_cell_ceiling_is_the_tick_budget(self):
        budget = RunBudget(ticks=25, nominal_ticks=900, dry_run=True, cells=4)
        ctx = budget.attach([{"arm_id": "A"}])[0]
        assert budget_ceiling(ctx) == 25.0

    def test_aggregate_ceiling_multiplies_by_cells(self):
        budget = RunBudget(ticks=25, nominal_ticks=900, dry_run=True, cells=4)
        ctx = budget.attach([{"arm_id": "A"}])[0]
        assert budget_ceiling_total(ctx) == 100.0

    def test_attach_copies_and_does_not_mutate_shared_arm_tables(self):
        """Several drivers hold ARM_CONTEXTS as a module-level constant; an
        in-place merge would leave one invocation's budget on a shared object."""
        arms = [{"arm_id": "A"}, {"arm_id": "B"}]
        budget = RunBudget(ticks=25, nominal_ticks=900, dry_run=True, cells=2)
        out = budget.attach(arms)
        assert BUDGET_TICKS_KEY not in arms[0]
        assert BUDGET_TICKS_KEY not in arms[1]
        assert all(c[BUDGET_TICKS_KEY] == 25 for c in out)
        assert out[0][BUDGET_CELLS_KEY] == 2
        assert out[0][BUDGET_DRY_RUN_KEY] is True

    def test_manifest_block_is_serialisable_and_records_both_budgets(self):
        import json
        budget = RunBudget(ticks=25, nominal_ticks=900, dry_run=True, cells=4)
        block = json.loads(json.dumps(budget.manifest_block()))
        assert block["resolved_ticks"] == 25
        assert block["nominal_ticks"] == 900
        assert block["dry_run"] is True


class TestBoundIsNotFabricated:
    """A bound the context never supplied must read as NOT EVALUATED, not 0."""

    def test_missing_budget_key_returns_none(self):
        assert budget_ceiling({"arm_id": "A"}) is None
        assert budget_ceiling_total({"arm_id": "A"}) is None

    def test_aggregate_needs_both_keys(self):
        assert budget_ceiling_total({BUDGET_TICKS_KEY: 25}) is None

    def test_unattached_context_is_reported_not_evaluated_not_unsatisfiable(self):
        """The failure mode a 0 default would produce: every floor provably
        unmeetable, and the run refused on a bound nobody derived."""
        specs = [_fresh_spec(EXQ1062_FRESH_FLOOR, structural_max=budget_ceiling)]
        ctx = {"arm_id": "A"}          # driver forgot budget.attach(...)
        verdict = structural_vacuity_verdict(specs, ctx)
        assert verdict["verdict"] == "not_evaluated"
        assert verdict["n_evaluated"] == 0
        # and it must not refuse the run
        assert_no_structurally_unsatisfiable_gate(specs, [{"id": "A"}], report=False)


# --- the regression, all three states --------------------------------------- #

class TestV3EXQ1062Regression:
    """Floor 200 against a --dry-run P2 budget of 60. See module docstring."""

    def test_old_shape_passes_and_proves_nothing(self):
        """CONTROL. Passes on the pre-change module too -- that is the point:
        the old guard was blind here, so a new test that also passed here would
        be measuring nothing."""
        specs = [_fresh_spec(EXQ1062_FRESH_FLOOR)]          # no bound at all
        ctx = {"id": "ARM_0", "p2_budget": EXQ1062_DRY_P2_BUDGET}
        audited = assert_no_structurally_unsatisfiable_gate(specs, [ctx], report=False)
        summary = summarize_structural_audit(audited)
        assert summary["coverage"] == 0.0
        assert summary["proved_nothing"] is True

    def test_bound_alone_wedges_the_smoke(self):
        """Half 2 without half 1: the guard refuses a --dry-run it should allow.
        True, useless, and the reason the enabler ships both halves."""
        budget = RunBudget(ticks=EXQ1062_DRY_P2_BUDGET,
                           nominal_ticks=EXQ1062_P2_STEP_BUDGET, dry_run=True)
        specs = [_fresh_spec(EXQ1062_FRESH_FLOOR, structural_max=budget_ceiling)]
        ctxs = budget.attach([{"id": "ARM_0"}])
        with pytest.raises(StructurallyUnsatisfiableGate, match="fresh_select_sample_floor"):
            assert_no_structurally_unsatisfiable_gate(specs, ctxs, report=False)

    def test_both_halves_smoke_passes_and_proves_something(self):
        """The shipped shape. Fails against the old module: no `budget_ceiling`
        and no floor scaling."""
        budget = RunBudget(ticks=EXQ1062_DRY_P2_BUDGET,
                           nominal_ticks=EXQ1062_P2_STEP_BUDGET, dry_run=True)
        floor = budget.scaled_floor(EXQ1062_FRESH_FLOOR, report=False)
        specs = [_fresh_spec(floor, structural_max=budget_ceiling)]
        ctxs = budget.attach([{"id": "ARM_0"}])
        audited = assert_no_structurally_unsatisfiable_gate(specs, ctxs, report=False)
        summary = summarize_structural_audit(audited)
        assert summary["coverage"] == 1.0
        assert summary["proved_nothing"] is False
        assert summary["n_satisfiable"] == 1

    def test_real_run_bound_still_binds_against_the_preregistered_floor(self):
        """The real run must keep the pre-registered 200 and still be PROVED
        satisfiable -- the guard is load-bearing where it matters, not only in
        the smoke."""
        budget = RunBudget(ticks=EXQ1062_P2_STEP_BUDGET,
                           nominal_ticks=EXQ1062_P2_STEP_BUDGET, dry_run=False)
        floor = budget.scaled_floor(EXQ1062_FRESH_FLOOR, report=False)
        assert floor == 200.0
        specs = [_fresh_spec(floor, structural_max=budget_ceiling)]
        ctxs = budget.attach([{"id": "ARM_0"}])
        audited = assert_no_structurally_unsatisfiable_gate(specs, ctxs, report=False)
        assert summarize_structural_audit(audited)["n_satisfiable"] == 1

    def test_a_genuinely_unreachable_real_floor_still_raises(self):
        """The guard must not be defanged: a real run whose floor exceeds its
        own full budget is still refused before compute."""
        budget = RunBudget(ticks=100, nominal_ticks=100, dry_run=False)
        specs = [_fresh_spec(500, structural_max=budget_ceiling)]
        ctxs = budget.attach([{"id": "ARM_0"}])
        with pytest.raises(StructurallyUnsatisfiableGate):
            assert_no_structurally_unsatisfiable_gate(specs, ctxs, report=False)
