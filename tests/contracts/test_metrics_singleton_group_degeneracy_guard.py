"""Contracts for the singleton-group arity guard in
experiments/_metrics.py::metric_groups_are_degenerate.

THE BUG (chip-20260830-singleton-group-degeneracy-guard).

V3-EXQ-961's driver built its c3_geom_distance_spread non-degeneracy check as
a list of ONE-ELEMENT "groups" -- `[[r["distance_std"]] for r in results]` --
instead of a flat `values` list. `metric_groups_are_degenerate` reduces each
group with `metric_is_degenerate`, whose FIRST test is "spread <= eps ->
degenerate". A group of length 1 has a spread of exactly 0.0 by construction
(there is nothing else in the group to differ from it), so every singleton
group reads "pinned" regardless of what value it holds -- the check can never
pass no matter how varied the underlying data actually is.

Concretely: ARM_GEOM's three distance_std readings (1.20998, 1.09383,
1.04082) sit far above both the 0.5 criterion floor and this function's own
1e-6 floor -- a genuinely non-degenerate, well-spread signal -- yet wrapping
each seed's single reading in its own singleton group reported the metric
"every group pinned" and the run's non_degenerate flag came back False. The
REE_assembly indexer's non-degeneracy gate
(build_experiment_indexes.py ~3431-3436) then set scoring_excluded="degenerate"
on the manifest, silently dropping a sound run's evidence from claim
confidence/conflict scoring -- MECH-144's only experimental entry, until a
2026-08-30 manual rescue.

THE FIX has two independent parts, both pinned here:
  1. An ARITY GUARD in metric_groups_are_degenerate itself: a group of
     length < 2 carries no spread information and is now skipped rather than
     treated as pinned. If EVERY group is a singleton, the metric is reported
     NOT degenerate (there was nothing to measure, which is a different
     finding from "measured and found pinned").
  2. The driver (v3_exq_961) no longer wraps its per-seed distance_std
     readings in singleton groups at all -- it passes them as a flat
     "values" list, which is what metric_is_degenerate is for.

Both are asserted here because either one alone would have been enough to
close the V3-EXQ-961 signature, but the arity guard is the one that stops the
whole CLASS of bug recurring on some other driver that makes the same
groups-vs-values mistake in the future.
"""

from __future__ import annotations

import pytest

from experiments._metrics import (
    check_degeneracy,
    metric_groups_are_degenerate,
    metric_is_degenerate,
)


# -- 1. The arity guard on metric_groups_are_degenerate directly ------------


def test_all_singleton_groups_never_reported_degenerate():
    """The exact V3-EXQ-961 shape: one value per group, real spread across
    groups. Must NOT be reported degenerate -- there is nothing measurable
    in any single group, so this is an insufficient-arity finding, never a
    pinned finding."""
    values = [1.20998, 1.09383, 1.04082]
    groups = [[v] for v in values]

    is_deg, reason = metric_groups_are_degenerate(groups, floor=1e-6)

    assert is_deg is False
    # The reason must say WHY it wasn't measurable, not claim it was pinned.
    assert "insufficient arity" in reason
    # The verdict-summarising phrase used for a genuine pinned finding
    # ("every group pinned" / "every measurable group pinned") must not
    # appear -- this is an insufficient-arity finding, not a pinned one,
    # even though the per-group detail text says "not treated as pinned".
    assert "every group pinned" not in reason
    assert "every measurable group pinned" not in reason


def test_all_singleton_groups_never_reported_degenerate_even_when_constant():
    """Even if every singleton happens to carry the SAME value, that is
    still not evidence of pinning -- there was no within-group comparison
    possible, full stop. (A constant value across DIFFERENT groups is a
    cross-group finding metric_groups_are_degenerate does not make claims
    about; it only ever asks whether each group is internally pinned.)"""
    groups = [[0.5], [0.5], [0.5]]

    is_deg, reason = metric_groups_are_degenerate(groups)

    assert is_deg is False
    assert "insufficient arity" in reason


def test_mixed_singleton_and_multi_element_groups_skips_singletons():
    """A singleton group is skipped and does not itself contribute to a
    'pinned' verdict; the verdict is driven entirely by the groups that
    actually carry >= 2 elements."""
    # One singleton (unmeasurable) + one genuinely-pinned pair + one
    # genuinely-varying pair. The varying pair alone must make this
    # NOT degenerate (metric_groups_are_degenerate degenerates iff EVERY
    # measurable group is pinned).
    groups = [[1.0], [2.0, 2.0], [3.0, 4.0]]

    is_deg, reason = metric_groups_are_degenerate(groups)

    assert is_deg is False


def test_singleton_plus_only_pinned_multi_element_groups_still_degenerate():
    """A singleton alongside real multi-element groups that ARE all pinned
    must still report degenerate -- the guard skips singletons, it does not
    make the whole check permissive."""
    groups = [[1.0], [2.0, 2.0], [3.0, 3.0]]

    is_deg, reason = metric_groups_are_degenerate(groups)

    assert is_deg is True
    assert "insufficient arity" in reason  # the singleton is still named
    assert "pinned" in reason  # and the real groups are still named pinned


def test_empty_group_list_still_reported_degenerate():
    """Unchanged pre-existing behaviour: an empty groups list has no
    observations at all and must still read degenerate (this is a distinct
    finding from 'no group had sufficient arity' -- there were no groups to
    begin with)."""
    is_deg, reason = metric_groups_are_degenerate([])
    assert is_deg is True
    assert reason == "no groups"


# -- 2. Negative control: genuine multi-element pinned group ---------------
# (must NOT be affected by the guard -- pinning detection on real,
# sufficiently-sized groups is completely unchanged.)


def test_genuine_multi_element_pinned_group_still_reported_degenerate():
    """A real ON/OFF-style pair per seed, every seed bit-identical between
    its two arms -- the V3-EXQ-603 / 543e family this function exists for.
    Must still be reported degenerate; the arity guard must not weaken this
    detection."""
    # Three seeds, each an (arm_on, arm_off) pair that is bit-identical
    # within the seed (the pinned signature), even though the pinned VALUE
    # differs across seeds (so a flat pooled check would wrongly pass).
    groups = [[0.10, 0.10], [0.55, 0.55], [0.92, 0.92]]

    is_deg, reason = metric_groups_are_degenerate(groups)

    assert is_deg is True
    assert "every measurable group pinned" in reason


def test_genuine_multi_element_group_with_real_spread_not_degenerate():
    """Sanity check on the un-guarded path: a multi-element group with real
    spread is correctly NOT reported degenerate."""
    groups = [[0.10, 0.95], [0.20, 0.88]]

    is_deg, reason = metric_groups_are_degenerate(groups)

    assert is_deg is False
    assert reason == ""


def test_floor_and_ceiling_still_apply_to_multi_element_groups():
    """The floor/ceiling saturation rails documented on metric_is_degenerate
    are unchanged for groups that actually have >= 2 elements -- the arity
    guard only touches the special-case singleton branch."""
    # Every group has tiny jitter (real, nonzero spread) but every value
    # sits under the floor -- still degenerate via floor-pinning, same as
    # metric_is_degenerate's own floor test.
    groups = [[1e-8, 2e-8], [3e-8, 1e-8]]

    is_deg, reason = metric_groups_are_degenerate(groups, floor=1e-6)

    assert is_deg is True
    assert "floor-pinned" in reason


# -- 3. End-to-end through check_degeneracy (the manifest-facing surface) ---


def test_check_degeneracy_singleton_groups_do_not_exclude_a_sound_run():
    """Reproduces the exact V3-EXQ-961 c3_geom_distance_spread call shape and
    confirms check_degeneracy no longer marks a genuinely-varying,
    well-above-floor metric non_degenerate=False when it is (mis-)supplied
    as singleton groups."""
    distance_std_values = [1.20998, 1.09383, 1.04082]

    result = check_degeneracy({
        "c3_geom_distance_spread": {
            "groups": [[v] for v in distance_std_values],
            "floor": 1e-6,
        },
    })

    assert result["non_degenerate"] is True
    assert result["degeneracy_reason"] == ""
    assert result["degenerate_metrics"] == {}


def test_check_degeneracy_flat_values_is_the_recommended_shape():
    """The driver-side fix: the same data passed as a flat "values" list
    (what v3_exq_961 now does) gives the identical non-degenerate verdict --
    confirming "values" is the correct producer-side idiom for this shape of
    data (one reading per seed, no ON/OFF pairing)."""
    distance_std_values = [1.20998, 1.09383, 1.04082]

    result = check_degeneracy({
        "c3_geom_distance_spread": {
            "values": distance_std_values,
            "floor": 1e-6,
        },
    })

    assert result["non_degenerate"] is True
    assert result["degeneracy_reason"] == ""


def test_check_degeneracy_genuinely_pinned_multi_seed_groups_still_excludes():
    """Negative control at the check_degeneracy level: a real ON/OFF pinned
    signature (each seed's pair bit-identical) must still flip
    non_degenerate to False, exactly as before this fix."""
    result = check_degeneracy({
        "c2_on_off_delta": {"groups": [[0.3, 0.3], [0.7, 0.7], [0.9, 0.9]]},
    })

    assert result["non_degenerate"] is False
    assert "c2_on_off_delta" in result["degenerate_metrics"]
    assert "pinned" in result["degenerate_metrics"]["c2_on_off_delta"]


def test_metric_is_degenerate_unaffected_by_the_groups_guard():
    """metric_is_degenerate (the flat, ungrouped checker) is a completely
    separate function from metric_groups_are_degenerate and must be
    untouched by this change: a single flat observation is still, correctly,
    reported degenerate (there is genuinely no spread across the whole
    metric in that case -- this is NOT the singleton-group bug, it's the
    'only one data point was ever recorded' case, which is a real finding)."""
    is_deg, reason = metric_is_degenerate([0.5])
    assert is_deg is True
    assert "zero spread" in reason
