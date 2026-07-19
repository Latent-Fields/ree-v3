"""Contracts for subgroup-scoped aggregation in the SD-068 consolidation-lesion family.

THE DEFECT (V3-EXQ-778h autopsy, 2026-07-19a governance cycle). The C2 criterion
`C2_unpaired_ratio_content_contingent` is DEFINED on the seeds that passed the C1
de-rail predicate `C1_unpaired_null_derails`, but its summary statistics --
mean/sd/ci95 of `null_slope_ratio_unclamped`, and the derived `ceiling_inside_ci95`
flag -- were computed over ALL seeds, including seeds that FAILED C1.

Concrete instance, run v3_exq_sd068_rem_unpaired_null_anchorfix_diagnostic_
20260718T183746Z_v3 (queue V3-EXQ-778h): seed 7777 fails C1 (clamp_frac 0.6 > 0.2
ceiling; 2 unclamped sigmas < 3 minimum) yet contributes ratio 85.18 -- the sole
outlier against a 0.0011-3.30 range. Pooled: mean 11.55, sd 29.77, CI95
[-9.078, 32.179], `ceiling_inside_ci95: true`, which READS as "underpowered, cannot
conclude". Scoped to the 7 C1-passing seeds the reading is TIGHTER, not weaker:
6/7 above the 0.25 ceiling, median ~0.877, CI95 [0.252, 1.811], ceiling OUTSIDE.

That the defect erred conservative in this instance is LUCK, not a property -- a
C1-failing seed carrying a near-zero ratio pulls the interval the other way and
manufactures false confidence in content-contingency. Hence these contracts pin the
scoping itself, in both directions, not the sign of the 778h error.

V3-EXQ-778h is NOT re-run or re-queued by this fix: it is already adjudicated
(non_contributory / measurement_gap / failure_autopsy_V3-EXQ-778h_2026-07-19) and its
conclusion is unchanged. The fix makes the reported uncertainty honest for FUTURE runs.

Pinned here:
  C1  a C1-failing seed is excluded from C2's statistics.
  C2  the literal 778h 8-seed configuration reproduces the corrected subgroup reading.
  C3  the excluded-seed list (and subgroup n) is emitted, so the narrowing is auditable.
  C4  a C1-failing seed on the LOW side flips ceiling_inside_ci95 the other way --
      the defect is not conservative by construction.
  C5  eligible-but-unmeasurable is distinguishable from ineligible.
  C6  the shipped drivers actually route through the scoped aggregator (no regression
      back to an unfiltered comprehension over arm_results).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from _lib import consolidation_lesion_harness as H  # noqa: E402

CEILING = H.NULL_SLOPE_RATIO_CEILING  # 0.25


# --------------------------------------------------------------------------- #
# The literal V3-EXQ-778h configuration, transcribed from the shipped manifest
# arm_summary.NULL_UNPAIRED. Seed order is the run's own seed list.
# --------------------------------------------------------------------------- #
_778H_SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
_778H_CLAMP_FRAC = [0.0, 0.0, 0.0, 0.2, 0.0, 0.6, 0.2, 0.2]
_778H_N_UNCLAMPED = [5, 5, 5, 4, 5, 2, 4, 4]
_778H_RATIOS = [
    0.9946357009625552,
    0.9014339361414496,
    0.8770952900685055,
    0.0010989173069902271,
    0.576577543244651,
    85.18453599581865,  # seed 7777 -- the C1-FAILING outlier
    0.5739206576346799,
    3.295473796662413,
]

# The C1 per-seed de-rail predicate as documented in the 778h manifest:
# clamp_frac <= 0.2 AND >= 3 distinct values AND >= 3 unclamped sigma points.
# Distinct-value count is not varying across these seeds, so it is held at 5.
_DERAIL_CLAMP_CEILING = 0.2
_MIN_UNCLAMPED_SIGMAS = 3


def _row(seed, clamp_frac, n_unclamped, ratio, contingent=False, n_distinct=5.0):
    derailed = (
        clamp_frac <= _DERAIL_CLAMP_CEILING
        and n_distinct >= 3
        and n_unclamped >= _MIN_UNCLAMPED_SIGMAS
    )
    return {
        "seed": seed,
        "unpaired_derailed": bool(derailed),
        "unpaired_content_contingent": bool(contingent),
        "arms": {"NULL_UNPAIRED": {"null_slope_ratio_unclamped": ratio}},
    }


def _778h_rows():
    return [
        _row(s, c, u, r)
        for s, c, u, r in zip(
            _778H_SEEDS, _778H_CLAMP_FRAC, _778H_N_UNCLAMPED, _778H_RATIOS
        )
    ]


def _stats(rows):
    return H.subgroup_ratio_stats(
        rows,
        eligible=lambda r: bool(r["unpaired_derailed"]),
        value=lambda r: r["arms"]["NULL_UNPAIRED"]["null_slope_ratio_unclamped"],
        ceiling=CEILING,
    )


# --------------------------------------------------------------------------- #
# C1 -- a C1-failing seed is excluded from C2's statistics.
# --------------------------------------------------------------------------- #


def test_c1_failing_seed_is_excluded_from_c2_statistics():
    rows = _778h_rows()
    st = _stats(rows)

    assert st["subgroup_n"] == 7, "the 8th (C1-failing) seed must not contribute"
    assert 7777 in st["excluded_seeds"]
    assert st["n_excluded"] == 1

    # The excluded seed's value is genuinely absent from the point estimate: the
    # pooled mean was 11.55, the scoped mean is ~1.03.
    assert st["mean"] == pytest.approx(1.0314622631, rel=1e-9)
    assert st["mean"] < 2.0, "85.18 outlier must not be in the mean"


def test_excluded_seed_fails_c1_for_both_documented_reasons():
    """Guards the fixture itself: 7777 fails on clamp_frac AND on unclamped count."""
    i = _778H_SEEDS.index(7777)
    assert _778H_CLAMP_FRAC[i] > _DERAIL_CLAMP_CEILING
    assert _778H_N_UNCLAMPED[i] < _MIN_UNCLAMPED_SIGMAS
    assert _row(7777, _778H_CLAMP_FRAC[i], _778H_N_UNCLAMPED[i], _778H_RATIOS[i])[
        "unpaired_derailed"
    ] is False


# --------------------------------------------------------------------------- #
# C2 -- the literal 778h configuration reproduces the corrected subgroup reading.
# --------------------------------------------------------------------------- #


def test_778h_configuration_reproduces_corrected_subgroup_reading():
    st = _stats(_778h_rows())
    kept = [
        r for s, r in zip(_778H_SEEDS, _778H_RATIOS) if s != 7777
    ]

    # 6 of 7 C1-passing seeds sit above the 0.25 ceiling; median ~0.877.
    assert sum(1 for v in kept if v > CEILING) == 6
    assert len(kept) == 7
    assert sorted(kept)[3] == pytest.approx(0.8770952900685055)

    # The corrected interval is TIGHTER than the pooled one, and the ceiling falls
    # OUTSIDE it -- the pooled reading's "cannot conclude" was an artefact.
    assert st["sd"] == pytest.approx(1.0524379634, rel=1e-8)
    assert st["ci95_low"] == pytest.approx(0.2518053091, rel=1e-8)
    assert st["ci95_high"] == pytest.approx(1.8111192172, rel=1e-8)
    assert st["ceiling_inside_ci95"] is False

    # And the pooled computation the fix replaced DID put the ceiling inside --
    # pinned so a regression to unfiltered aggregation is caught by value, not by
    # code shape alone.
    pooled = H.subgroup_ratio_stats(
        _778h_rows(),
        eligible=lambda r: True,
        value=lambda r: r["arms"]["NULL_UNPAIRED"]["null_slope_ratio_unclamped"],
        ceiling=CEILING,
    )
    assert pooled["mean"] == pytest.approx(11.5505964797, rel=1e-9)
    assert pooled["sd"] == pytest.approx(29.7685550667, rel=1e-9)
    assert pooled["ceiling_inside_ci95"] is True


def test_778h_adjudicated_conclusion_is_unchanged_by_the_fix():
    """C2's PASS predicate stays False: 1 contingent seed is not a majority of 7.

    The 778h manifest reports n_seeds_content_contingent = 1 and
    C2 passed = false. Scoping the denominator to the de-railed subgroup cannot
    flip that (1 > 3.5 is false either way), which is why this fix does not
    re-open the already-adjudicated run.
    """
    rows = _778h_rows()
    rows[0]["unpaired_content_contingent"] = True  # the single contingent seed
    derailed = [r for r in rows if r["unpaired_derailed"]]
    n_contingent = sum(1 for r in derailed if r["unpaired_content_contingent"])
    assert n_contingent == 1
    assert (n_contingent > len(derailed) / 2) is False


# --------------------------------------------------------------------------- #
# C3 -- the exclusion is emitted, never silent.
# --------------------------------------------------------------------------- #


def test_subgroup_n_and_excluded_seeds_are_emitted():
    st = _stats(_778h_rows())
    for key in (
        "subgroup_n",
        "n_eligible",
        "n_non_finite",
        "excluded_seeds",
        "n_excluded",
    ):
        assert key in st, f"{key} must be emitted so the narrowing is auditable"

    assert st["excluded_seeds"] == [7777]
    assert st["subgroup_n"] + st["n_excluded"] + st["n_non_finite"] == len(_778H_SEEDS)


def test_no_exclusion_still_reports_an_empty_excluded_list():
    """An empty list, not a missing key -- absence of exclusions must be positive."""
    rows = [_row(s, 0.0, 5, 1.0) for s in (1, 2, 3)]
    st = _stats(rows)
    assert st["excluded_seeds"] == []
    assert st["n_excluded"] == 0
    assert st["subgroup_n"] == 3


# --------------------------------------------------------------------------- #
# C4 -- the defect is NOT conservative by construction.
# --------------------------------------------------------------------------- #


def test_low_side_ineligible_seed_would_manufacture_false_confidence():
    """A C1-failing seed near zero pulls the interval the OTHER way.

    Same shape as 778h but with the ineligible seed carrying ~0 instead of 85.18:
    pooling it drags the mean toward the ceiling and puts the ceiling INSIDE the
    interval where the scoped reading excludes it -- or, with a tight enough
    cluster, drags the whole interval BELOW the ceiling and reads as
    content-contingency that the eligible seeds do not show. Either way the pooled
    answer differs from the scoped one in the anti-conservative direction.
    """
    rows = [
        _row(1, 0.0, 5, 0.52),
        _row(2, 0.0, 5, 0.55),
        _row(3, 0.0, 5, 0.58),
        _row(7777, 0.6, 2, 0.0),  # ineligible, low side
    ]

    scoped = _stats(rows)
    pooled = H.subgroup_ratio_stats(
        rows,
        eligible=lambda r: True,
        value=lambda r: r["arms"]["NULL_UNPAIRED"]["null_slope_ratio_unclamped"],
        ceiling=CEILING,
    )

    assert scoped["subgroup_n"] == 3
    assert scoped["mean"] == pytest.approx(0.55)
    assert pooled["mean"] < scoped["mean"], "the ineligible seed drags the estimate down"
    assert pooled["sd"] > scoped["sd"], "and manufactures spread out of an ineligible seed"

    # THE POINT: the scoped reading excludes the ceiling (content-contingency is
    # ruled OUT for the eligible seeds); pooling the ineligible seed pulls the
    # interval down across the ceiling so the run reads as "cannot conclude" --
    # here in the ANTI-conservative direction, the opposite of 778h.
    assert scoped["ceiling_inside_ci95"] is False
    assert pooled["ceiling_inside_ci95"] is True


# --------------------------------------------------------------------------- #
# C5 -- ineligible vs unmeasurable are distinct.
# --------------------------------------------------------------------------- #


def test_eligible_but_unmeasurable_is_not_reported_as_excluded():
    rows = [
        _row(1, 0.0, 5, 1.0),
        _row(2, 0.0, 5, 1.2),
        _row(3, 0.0, 5, H.UNAVAILABLE),  # eligible, no measurement
        _row(4, 0.0, 5, float("nan")),  # eligible, non-finite
        _row(7777, 0.6, 2, 85.0),  # ineligible
    ]
    st = _stats(rows)
    assert st["n_eligible"] == 4
    assert st["subgroup_n"] == 2
    assert st["n_non_finite"] == 2
    assert st["excluded_seeds"] == [7777], "unmeasurable != ineligible"


def test_empty_subgroup_degrades_to_unavailable_not_a_crash():
    rows = [_row(7777, 0.6, 2, 85.0), _row(8888, 1.0, 0, 0.0)]
    st = _stats(rows)
    assert st["subgroup_n"] == 0
    assert st["mean"] == H.UNAVAILABLE
    assert st["sd"] == H.UNAVAILABLE
    assert st["ceiling_inside_ci95"] is False
    assert sorted(st["excluded_seeds"]) == [7777, 8888]


def test_single_seed_subgroup_reports_no_interval():
    st = _stats([_row(1, 0.0, 5, 0.9), _row(7777, 0.6, 2, 85.0)])
    assert st["subgroup_n"] == 1
    assert st["mean"] == pytest.approx(0.9)
    assert st["sd"] == H.UNAVAILABLE
    assert st["ceiling_inside_ci95"] is False


# --------------------------------------------------------------------------- #
# C6 -- the shipped drivers route through the scoped aggregator.
# --------------------------------------------------------------------------- #

_UNPAIRED_NULL_DRIVERS = [
    "v3_exq_sd068_rem_unpaired_null_anchorfix_diagnostic.py",
    "v3_exq_sd068_rem_unpaired_null_diagnostic.py",
]


@pytest.mark.parametrize("driver", _UNPAIRED_NULL_DRIVERS)
def test_driver_uses_subgroup_ratio_stats(driver):
    src = (REPO_ROOT / "experiments" / driver).read_text()
    assert "subgroup_ratio_stats(" in src, (
        f"{driver} must aggregate C2 through the subgroup-scoped helper"
    )
    for key in ("c2_subgroup_n", "c2_excluded_seeds"):
        assert key in src, f"{driver} must emit {key} in the manifest"


@pytest.mark.parametrize("driver", _UNPAIRED_NULL_DRIVERS)
def test_driver_has_no_unfiltered_ratio_comprehension(driver):
    """Regression guard: the pooled comprehension must not come back.

    Matches the exact defective shape -- a comprehension over `arm_results` pulling
    `null_slope_ratio_unclamped` with no de-rail predicate in the filter.
    """
    src = (REPO_ROOT / "experiments" / driver).read_text()
    tree = ast.parse(src)

    # `per_seed_*` manifest entries are per-seed REPORTING, not aggregation. They are
    # legitimately over all seeds and must stay that way -- the audit trail is how a
    # reader checks the exclusion. Exempt them by their dict key.
    reporting = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, val in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.startswith("per_seed")
            ):
                reporting.add(id(val))

    offenders = []
    for comp in ast.walk(tree):
        if not isinstance(comp, ast.ListComp) or id(comp) in reporting:
            continue
        body = ast.dump(comp)
        if "null_slope_ratio_unclamped" not in body or "arm_results" not in body:
            continue
        if "unpaired_derailed" in body:
            continue  # correctly scoped
        offenders.append(ast.unparse(comp)[:120])

    assert not offenders, (
        f"{driver} reintroduced an unfiltered C2 ratio aggregation over arm_results: "
        f"{offenders}"
    )


@pytest.mark.parametrize("driver", _UNPAIRED_NULL_DRIVERS)
def test_contingent_majority_denominator_is_the_derailed_subgroup(driver):
    """C2's PASS predicate is scoped too, not just its interval.

    A railed seed reports content_contingent_unclamped=False by construction (the
    unclamped-grid stub returns content_contingent=0.0 below 2 clean sigmas), so a
    full-seed denominator silently votes NO on seeds never eligible to vote.
    """
    src = (REPO_ROOT / "experiments" / driver).read_text()
    assert "n_contingent > n / 2" not in src, (
        f"{driver} still counts content-contingency against ALL seeds"
    )
    assert "n_derailed_group" in src


def test_staged_damage_order_statistics_scope_to_the_computable_subgroup():
    """Sibling instance found by the family audit: staging order is C3-scoped."""
    src = (
        REPO_ROOT
        / "experiments"
        / "v3_exq_sd068_consolidation_staged_damage_diagnostic.py"
    ).read_text()
    assert "staging_rows" in src
    assert "staging_excluded_seeds" in src
    assert "staging_subgroup_n" in src
    assert "n_match_pred >= need" not in src, (
        "staging match must be scored against the C3-computable denominator"
    )
