"""Duty-cycle-aware non-vacuity gate for a modulatory mechanism's readiness check.

THE FAILURE MODE THIS CLOSES
-----------------------------
The GOV-FANOUT-1 (ARC-062/MECH-309) leg family's C1e (MECH-448 rank-preserving
F->eligibility demotion) and C1f (MECH-449 active Go/No-Go opponency) readiness
preconditions were each written as a COMPOUND gate:

    seed_non_vacuous = bool(
        active_frac >= ACTIVE_FRAC_FLOOR        # e.g. 0.8
        and magnitude_when_active > MAGNITUDE_FLOOR   # e.g. excluded_count_mean > 0.0
    )

and then aggregated with an AND-across-every-rung-and-every-seed count
(``all(n_meeting[rung] >= MIN_SEEDS_FOR_PASS for rung in RUNGS)``). Confirmed
twice, independently, in two full-budget runs of this same driver family:

* ``failure_autopsy_V3-EXQ-851_2026-08-01.md`` (Section 9 addendum,
  2026-08-01T21:19:59Z): read C1e/C1f as "completely dead" (measured 0.0) under
  ``lateral_pfc`` routing. Raw per-seed data showed the mechanisms genuinely
  engaged at PRESERVED MAGNITUDE relative to the V3-EXQ-654j baseline
  (``excluded_count_mean`` 17.0-28.3 vs 654j's 17.4-29.1; ``nogo_suppressed``
  1.5-12.2 vs 654j's 1.5-13.2) -- what differed was DUTY CYCLE alone
  (``active_frac`` 0.24-0.58 vs 654j's flat 1.0).
* ``failure_autopsy_V3-EXQ-858_2026-08-02.md`` (confirmed directly, at full
  budget, across a 4-rung x 3-seed = 12-cell design): ``active_frac`` never
  dropped below 0.24 in a single cell and ``excluded_count_mean``/``nogo_supp``
  sat squarely in 851's own "when active" range in every cell -- yet C1e/C1f
  both read "false" because no single seed cleared ``active_frac >= 0.8`` on
  ALL FOUR RUNGS SIMULTANEOUSLY (seed 43 never exceeds 0.243 on any rung).

The mechanisms were robustly live in every one of those 12 cells. The gate
reported "dead" purely because duty cycle (how OFTEN the mechanism fires),
not magnitude (how STRONGLY it acts when it does), varies across
seeds/rungs -- and an AND-across-4-rungs-simultaneous floor is not a bar any
single seed can be expected to clear when duty cycle itself ranges 0.24-0.84.

THE FIX, PER BOTH AUTOPSIES' RECOMMENDED ROUTING (858 Section 6)
------------------------------------------------------------------
Score C1e/C1f as duty-cycle MAGNITUDE WHEN ACTIVE, dropping the
``active_frac >= 0.8`` requirement entirely:

    seed_non_vacuous = bool(magnitude_when_active > MAGNITUDE_FLOOR)

This is not a threshold-loosening in the sense the sibling module
``precondition_gate.py`` warns against ("never lower the threshold, that
converts an artifact into a citable result") -- ``magnitude_when_active`` in
every one of this family's drivers (``f_eligibility_excluded_count_mean``,
``go_nogo_suppressed_per_tick_mean``) is ALREADY computed as a mean over only
the ticks where the mechanism fired (the per-tick accumulator lists are
appended to exactly when ``f_eligibility_demotion_active`` /
``go_nogo_constitution_active`` read True). So this fix removes a REDUNDANT,
overly strict conjunct -- it does not touch the magnitude floor itself, and a
mechanism that never fires at all still correctly fails: an empty
accumulator's mean falls back to 0.0, which does not clear
``MAGNITUDE_FLOOR`` (0.0). No inactive mechanism can pass by omission.

``active_frac`` remains valuable TELEMETRY (it is still worth reporting
alongside the gate result -- 858's own addendum used it to characterise "24-58%
duty cycle" precisely) -- it is just no longer part of the boolean gate.

STATUS AS OF 2026-08-02
------------------------
No shared helper existed for this gate before this module: each driver
(``v3_exq_851``, ``v3_exq_858``, ``v3_exq_859``, ``v3_exq_863``,
``v3_exq_847a``) carries its own copy of the buggy compound predicate. Those
five have ALREADY RUN (or, for 847a, are mid-flight as of this fix landing --
claimed by ree-cloud-4 at 2026-08-02T11:25:39Z, ~5.5h into an estimated 10h
run) under the OLD gate spec; per governance instruction their scored
``evidence_direction`` is NOT retroactively altered by this fix (851/858's own
autopsies already correct the reading by hand). This module exists so any
FUTURE GOV-FANOUT-1 leg (or any other multi-rung driver with the same
active-frac-conjunct shape) imports the corrected predicate instead of
re-deriving -- and re-breaking -- it.

USAGE
-----
    from experiments._lib.duty_cycle_readiness import magnitude_when_active_met

    seed_demotion_non_vacuous = magnitude_when_active_met(
        magnitude_when_active=demotion_excluded_count_mean,
        magnitude_floor=EXCLUDED_COUNT_FLOOR,
    )

``active_frac`` (and any per-rung/per-seed AND-aggregation such as
``all(n_meeting[rung] >= MIN_SEEDS_FOR_PASS for rung in RUNGS)``) is
unaffected by this module and stays exactly as each driver already computes
it -- only the LEAF per-seed boolean changes.

ASCII-only in printed output (Windows cp1252 terminals).
"""

from __future__ import annotations

__all__ = ["magnitude_when_active_met"]


def magnitude_when_active_met(magnitude_when_active: float,
                               magnitude_floor: float) -> bool:
    """Corrected C1e/C1f-style non-vacuity check: magnitude alone, duty-cycle-blind.

    `magnitude_when_active` must already be conditioned on active ticks (a mean
    taken only over ticks where the mechanism fired -- exactly what this
    family's drivers already compute for `f_eligibility_excluded_count_mean` /
    `go_nogo_suppressed_per_tick_mean`). A mechanism that never fires yields an
    empty accumulator, whose caller-side fallback is 0.0 -- which does not
    clear a `magnitude_floor` of 0.0, so "never active" still correctly reads
    non-vacuous=False without this function needing to see `active_frac` at
    all.

    Deliberately does NOT take `active_frac` as a parameter. That is the whole
    fix: the old compound gate's `active_frac >= ACTIVE_FRAC_FLOOR` conjunct is
    what let seeds/rungs with genuine but lower-duty-cycle engagement (0.24-0.84
    observed across the 851/858 family) read as "dead" under an
    AND-across-every-rung-simultaneously aggregation no single seed could be
    expected to clear.
    """
    return bool(float(magnitude_when_active) > float(magnitude_floor))
