"""Contracts for experiments/_lib/duty_cycle_readiness.py.

The governing requirement, from failure_autopsy_V3-EXQ-851_2026-08-01.md
(Section 9 addendum) and failure_autopsy_V3-EXQ-858_2026-08-02.md (confirmed
directly, Sections 2/6):

    A robustly-engaged mechanism (excluded_count / suppression magnitude at
    parity with the healthy V3-EXQ-654j baseline) must NOT read as "dead"
    merely because its duty cycle (active_frac) varies across seeds/rungs and
    no single seed clears an AND-across-every-rung-simultaneous >= 0.8 floor.

The confirmed regression this locks down: V3-EXQ-858's own 4-rung x 3-seed
(12-cell) design had `active_frac` ranging 0.24-0.84 and never simultaneously
>= 0.8 on all four rungs for any seed, so the OLD compound gate
(`active_frac >= 0.8 AND magnitude > floor`) read C1e/C1f as 0/3 seeds meeting
on most rungs even though every single cell's magnitude was robustly positive.
"""

from experiments._lib.duty_cycle_readiness import magnitude_when_active_met

# --- the V3-EXQ-858 fixture: real per-cell numbers from the confirmed autopsy,
# Section 2's table (failure_autopsy_V3-EXQ-858_2026-08-02.md). Columns:
# seed, f_weight rung, demotion active_frac, excluded_count_mean (demotion
# magnitude-when-active), nogo active_frac, nogo_suppressed_per_tick_mean
# (nogo magnitude-when-active). -------------------------------------------- #

V3_EXQ_858_CELLS = [
    # seed, f_wt, demot_frac, excl_mean, nogo_frac, nogo_supp
    (42, 1.0, 0.643, 17.80, 0.643, 1.84),
    (43, 1.0, 0.240, 28.04, 0.240, 13.06),
    (44, 1.0, 0.778, 24.11, 0.778, 7.73),
    (42, 0.5, 0.632, 18.10, 0.632, 1.67),
    (43, 0.5, 0.242, 28.99, 0.242, 13.13),
    (44, 0.5, 0.619, 23.51, 0.619, 7.57),
    (42, 0.25, 0.609, 18.11, 0.609, 1.74),
    (43, 0.25, 0.241, 27.80, 0.241, 11.95),
    (44, 0.25, 0.808, 21.62, 0.808, 5.88),
    (42, 0.0, 0.843, 18.51, 0.843, 2.02),
    (43, 0.0, 0.243, 27.96, 0.243, 11.99),
    (44, 0.0, 0.808, 21.62, 0.808, 5.88),
]

DEMOTION_ACTIVE_FRAC_FLOOR = 0.8
EXCLUDED_COUNT_FLOOR = 0.0
NOGO_ACTIVE_FRAC_FLOOR = 0.8
NOGO_SUPPRESSED_FLOOR = 0.0

RUNGS = [1.0, 0.5, 0.25, 0.0]
SEEDS = [42, 43, 44]
MIN_SEEDS_FOR_PASS = 2  # matches the driver family's own majority-of-3 convention


def _old_buggy_demotion_non_vacuous(demot_frac: float, excl_mean: float) -> bool:
    """The compound gate this fix replaces -- reproduced here as a regression
    fixture, not imported from any driver (each driver had its own copy)."""
    return bool(
        demot_frac >= DEMOTION_ACTIVE_FRAC_FLOOR
        and excl_mean > EXCLUDED_COUNT_FLOOR
    )


def _old_buggy_nogo_non_vacuous(nogo_frac: float, nogo_supp: float) -> bool:
    return bool(
        nogo_frac >= NOGO_ACTIVE_FRAC_FLOOR
        and nogo_supp > NOGO_SUPPRESSED_FLOOR
    )


def test_every_858_cell_has_positive_magnitude_when_active():
    """Sanity check on the fixture itself: every cell's magnitude is positive
    (the autopsy's own finding -- the mechanisms are robustly engaged
    everywhere), so the corrected gate should read every cell as non-vacuous."""
    for _, _, _, excl_mean, _, nogo_supp in V3_EXQ_858_CELLS:
        assert excl_mean > 0.0
        assert nogo_supp > 0.0


def test_old_compound_gate_reproduces_the_confirmed_false_negative():
    """Regression: the OLD gate (active_frac>=0.8 AND magnitude>floor), ANDed
    across all 4 rungs simultaneously per seed, does NOT let C1e/C1f hold --
    reproducing 858's own confirmed per-rung pattern (Section 1's readiness
    table: "0/3 seeds at F100/F050, 1/3 at F025, 2/3 at F000"). F000 alone
    reaches the majority-of-3 floor (2/3), but F100/F050 (0/3) and F025 (1/3)
    do not, so the AND-across-all-4-rungs aggregate still reads False overall
    -- exactly the compound-gate artifact both autopsies diagnose."""
    demotion_met_per_rung = {rung: 0 for rung in RUNGS}
    nogo_met_per_rung = {rung: 0 for rung in RUNGS}
    for seed, rung, demot_frac, excl_mean, nogo_frac, nogo_supp in V3_EXQ_858_CELLS:
        if _old_buggy_demotion_non_vacuous(demot_frac, excl_mean):
            demotion_met_per_rung[rung] += 1
        if _old_buggy_nogo_non_vacuous(nogo_frac, nogo_supp):
            nogo_met_per_rung[rung] += 1

    # Matches 858's own reported per-rung counts exactly (rung order 1.0/0.5/0.25/0.0).
    assert demotion_met_per_rung == {1.0: 0, 0.5: 0, 0.25: 1, 0.0: 2}
    assert nogo_met_per_rung == {1.0: 0, 0.5: 0, 0.25: 1, 0.0: 2}
    c1e_holds_old = all(n >= MIN_SEEDS_FOR_PASS for n in demotion_met_per_rung.values())
    c1f_holds_old = all(n >= MIN_SEEDS_FOR_PASS for n in nogo_met_per_rung.values())
    assert c1e_holds_old is False
    assert c1f_holds_old is False


def test_corrected_gate_reads_every_858_cell_as_non_vacuous():
    """The fix: dropping the active_frac conjunct and scoring magnitude alone
    reads every one of the 12 cells as non-vacuous, so every rung clears the
    majority-of-3 floor and C1e/C1f both hold -- matching what both autopsies
    say the substrate is actually doing."""
    demotion_met_per_rung = {rung: 0 for rung in RUNGS}
    nogo_met_per_rung = {rung: 0 for rung in RUNGS}
    for seed, rung, demot_frac, excl_mean, nogo_frac, nogo_supp in V3_EXQ_858_CELLS:
        if magnitude_when_active_met(excl_mean, EXCLUDED_COUNT_FLOOR):
            demotion_met_per_rung[rung] += 1
        if magnitude_when_active_met(nogo_supp, NOGO_SUPPRESSED_FLOOR):
            nogo_met_per_rung[rung] += 1

    assert demotion_met_per_rung == {rung: len(SEEDS) for rung in RUNGS}
    assert nogo_met_per_rung == {rung: len(SEEDS) for rung in RUNGS}
    c1e_holds_new = all(n >= MIN_SEEDS_FOR_PASS for n in demotion_met_per_rung.values())
    c1f_holds_new = all(n >= MIN_SEEDS_FOR_PASS for n in nogo_met_per_rung.values())
    assert c1e_holds_new is True
    assert c1f_holds_new is True


def test_never_active_mechanism_still_correctly_fails():
    """A mechanism that never fires has an empty accumulator, whose caller-side
    fallback is 0.0 -- must not clear a 0.0 floor. Confirms the fix does not
    manufacture a pass for genuine inactivity (no `active_frac` blindness)."""
    assert magnitude_when_active_met(0.0, EXCLUDED_COUNT_FLOOR) is False
    assert magnitude_when_active_met(0.0, NOGO_SUPPRESSED_FLOOR) is False


def test_barely_above_floor_passes_and_at_floor_fails():
    """Strict-inequality floor semantics, matching every driver's own
    `magnitude_mean > FLOOR` convention (not >=)."""
    assert magnitude_when_active_met(1e-9, 0.0) is True
    assert magnitude_when_active_met(0.0, 0.0) is False


def test_does_not_accept_active_frac_as_a_parameter():
    """API-shape guard: the whole point of the fix is that this predicate is
    duty-cycle-blind by construction, not merely duty-cycle-lenient. A future
    edit that re-adds an active_frac parameter would silently reintroduce the
    confirmed false-negative class this module exists to close."""
    import inspect

    params = inspect.signature(magnitude_when_active_met).parameters
    assert "active_frac" not in params
    assert set(params) == {"magnitude_when_active", "magnitude_floor"}
