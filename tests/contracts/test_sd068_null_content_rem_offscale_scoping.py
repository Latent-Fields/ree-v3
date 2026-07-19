"""Contracts for off-scale scoping of the rem MAGNITUDE statistics in the SD-068
null-content control's confound register.

This is the second instance found by the family audit that produced the V3-EXQ-778h
C2 subgroup-aggregation fix (ree-v3 b42f69ffa3, see
tests/contracts/test_sd068_c2_subgroup_aggregation.py). It was deliberately left
unchanged by that fix because it is a scientific decision about what the confound
register asserts, not an unambiguous defect. THIS FILE PINS THE DECISION.

THE DECISION: SCOPE, not pool.

`v3_exq_sd068_null_content_control_diagnostic.py` aggregates `null_slope_ratio` per
phase into mean / sd / ci95 / ceiling_inside_ci95. For the rem phase it previously
filtered only on finiteness, ignoring `null_slope_ratio_rem_off_scale`. That flag is
the harness's own statement (consolidation_lesion_harness.py, the block setting
`null_slope_ratio_rem_off_scale`) that the ratio is NOT on a common scale with the
other seeds': when the null arm's rem precision reference collapses onto the 1e-3
positivity floor, the 1/1e-3 = 1000 term dominates the calibration error, so the value
reads "this leg is structurally content-free", never a calibrated N-fold noise
sensitivity. A mean over values with no common scale has no referent, and the 3-4
order-of-magnitude spread inflates the SEM enough to swallow the 0.25 ceiling, so
`ceiling_inside_ci95` reads "underpowered, cannot conclude" independent of the
evidence.

CONCRETE INSTANCE, run v3_exq_sd068_null_content_control_diagnostic_20260718T072318Z_v3
(queue V3-EXQ-778c): ALL 8 seeds are off-scale. Five (clamp_frac 1.0) report ratio
exactly 0.0; three (clamp_frac 0.2) report 1801.6 / 4348.5 / 9142.8. Pooled that is
mean 1911.6, sd 3306.1, CI95 [-379.4, 4202.6], ceiling_inside_ci95 true -- a published
point estimate and interval for a quantity never measured on scale, with a NEGATIVE
lower bound on a ratio of magnitudes. Scoped, subgroup_n is 0 and the statistics are
UNAVAILABLE: "no on-scale rem ratio exists at this n". That is a STRONGER and more
honest statement than a wide interval, and it is what the flag was written to convey.

WHAT IS DELIBERATELY NOT SCOPED, pinned here so a later audit does not "fix" it:
the per-seed confound VERDICT (`confounded_phases`, n_seeds_confounded,
confound_verdict_stable) stays over ALL seeds, per the register's standing rule that
confounded phases are reported and never dropped. Off-scale bears on the magnitude of
the ratio, not on whether the phase is confounded. The `per_seed_null_slope_ratio`
audit trail likewise stays complete.

V3-EXQ-778c and its siblings are NOT re-run or re-queued: they are adjudicated and
their conclusions are unchanged (this touches a descriptive statistic in the register,
not any PASS predicate). The fix makes the reported uncertainty honest for FUTURE runs.

Pinned here:
  D1  an off-scale seed is excluded from the rem magnitude statistics.
  D2  sws/nrem are NOT scoped -- they have no off-scale condition.
  D3  the literal 778c 8-seed configuration reproduces both readings, and the
      all-off-scale case degrades to UNAVAILABLE rather than a meaningless mean.
  D4  pooling is anti-conservative in BOTH directions, so the decision is not
      justified by the sign of the 778c error.
  D5  the narrowing is emitted (subgroup n + excluded seed ids), and the seed ids are
      real values rather than None.
  D6  the confound VERDICT stays unscoped.
  D7  the shipped driver routes through the scoped aggregator and has not regressed to
      an unfiltered comprehension.
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

DRIVER = "v3_exq_sd068_null_content_control_diagnostic.py"
CEILING = H.NULL_SLOPE_RATIO_CEILING  # 0.25


# --------------------------------------------------------------------------- #
# The literal V3-EXQ-778c configuration, transcribed from the shipped manifest
# (arm_results[*].seed / .null_control / .null_slope_ratio.rem).
# --------------------------------------------------------------------------- #
_778C_SEEDS = [42, 7, 123, 2024, 99, 7777, 314, 1000]
_778C_CLAMP_FRAC = [0.2, 1.0, 0.2, 1.0, 0.2, 1.0, 1.0, 1.0]
_778C_REM_RATIOS = [
    4348.4665081785715,
    0.0,
    9142.771353766831,
    0.0,
    1801.6453681003277,
    0.0,
    0.0,
    0.0,
]


def _score(seed, rem_ratio, clamp_frac, sws=1.0, nrem=0.144):
    """A seed_scores row in the shape `_score_seed` returns."""
    return {
        "seed": seed,
        # The harness flag is `any(clamped)`, i.e. ANY railed sigma point taints the
        # slope fit -- clamp_frac 0.2 is off-scale exactly as 1.0 is.
        "rem_ratio_off_scale": bool(clamp_frac > 0.0),
        "null_rem_target_clamped_frac": clamp_frac,
        "null_slope_ratio": {"sws": sws, "nrem": nrem, "rem": rem_ratio},
    }


def _778c_rows():
    return [
        _score(s, r, c)
        for s, r, c in zip(_778C_SEEDS, _778C_REM_RATIOS, _778C_CLAMP_FRAC)
    ]


def _on_scale(phase):
    """The driver's eligibility predicate, same shape as the shipped closure."""
    if phase != "rem":
        return lambda s: True
    return lambda s: not s["rem_ratio_off_scale"]


def _stats(rows, phase="rem"):
    return H.subgroup_ratio_stats(
        rows,
        eligible=_on_scale(phase),
        value=lambda s, _p=phase: s["null_slope_ratio"][_p],
        ceiling=CEILING,
    )


# --------------------------------------------------------------------------- #
# D1 -- an off-scale seed is excluded from the rem magnitude statistics.
# --------------------------------------------------------------------------- #


def test_off_scale_seed_is_excluded_from_rem_magnitude_statistics():
    rows = [
        _score(1, 0.9, 0.0),
        _score(2, 1.1, 0.0),
        _score(7777, 9142.77, 0.2),  # off-scale
    ]
    st = _stats(rows)

    assert st["subgroup_n"] == 2
    assert st["excluded_seeds"] == [7777]
    assert st["mean"] == pytest.approx(1.0)
    assert st["mean"] < 2.0, "the off-scale value must not enter the point estimate"


def test_any_clamped_point_makes_a_seed_off_scale_not_just_full_saturation():
    """The predicate follows the harness flag, which is `any(clamped)`.

    A 0.2-clamp seed still has the 1/1e-3 term in its slope fit, so its magnitude is
    contaminated even though four of five sigma points are clean. Pinned because the
    tempting "only exclude the fully-railed ones" variant readmits exactly the three
    seeds carrying 778c's 1801-9143 values.
    """
    partial = _score(42, 4348.47, 0.2)
    assert partial["rem_ratio_off_scale"] is True
    assert _stats([partial, _score(1, 0.9, 0.0)])["subgroup_n"] == 1


# --------------------------------------------------------------------------- #
# D2 -- sws/nrem are not scoped.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("phase", ["sws", "nrem"])
def test_non_rem_phases_are_not_narrowed_by_the_rem_off_scale_flag(phase):
    rows = _778c_rows()
    st = _stats(rows, phase=phase)

    assert st["subgroup_n"] == len(_778C_SEEDS), (
        f"{phase} has no off-scale condition; all seeds must contribute"
    )
    assert st["excluded_seeds"] == []
    assert st["n_excluded"] == 0


def test_778c_sws_and_nrem_readings_are_unchanged_by_the_fix():
    """The shipped 778c sws/nrem numbers must survive verbatim."""
    sws = _stats([_score(s, 0.0, 0.0, sws=v) for s, v in zip(
        _778C_SEEDS,
        [
            0.9999999985557226, 1.0, 0.9999999235226619, 0.9999999963187839,
            1.0000000013402017, 1.0, 1.0, 1.0000000004815457,
        ],
    )], phase="sws")
    assert sws["mean"] == pytest.approx(0.9999999900273644, rel=1e-12)
    assert sws["sd"] == pytest.approx(2.691473544676207e-08, rel=1e-9)
    assert sws["ceiling_inside_ci95"] is False


# --------------------------------------------------------------------------- #
# D3 -- the literal 778c configuration, both readings.
# --------------------------------------------------------------------------- #


def test_778c_pooled_reading_reproduces_the_shipped_manifest_numbers():
    """Pins what the fix replaced, by value -- not by code shape alone."""
    pooled = H.subgroup_ratio_stats(
        _778c_rows(),
        eligible=lambda s: True,
        value=lambda s: s["null_slope_ratio"]["rem"],
        ceiling=CEILING,
    )
    assert pooled["mean"] == pytest.approx(1911.6104037557163, rel=1e-12)
    assert pooled["sd"] == pytest.approx(3306.083577424908, rel=1e-12)
    assert pooled["ci95_low"] == pytest.approx(-379.3886306755837, rel=1e-9)
    assert pooled["ci95_high"] == pytest.approx(4202.609438187016, rel=1e-9)
    assert pooled["ceiling_inside_ci95"] is True

    # The tell that the pooled statistic is meaningless: a ratio of magnitudes cannot
    # be negative, yet its own 95% interval reaches well below zero.
    assert pooled["ci95_low"] < 0.0


def test_778c_scoped_reading_is_unavailable_because_every_seed_is_off_scale():
    st = _stats(_778c_rows())

    assert st["subgroup_n"] == 0
    assert st["n_eligible"] == 0
    assert sorted(st["excluded_seeds"]) == sorted(_778C_SEEDS)
    assert st["mean"] == H.UNAVAILABLE
    assert st["sd"] == H.UNAVAILABLE
    assert st["ci95_low"] == H.UNAVAILABLE

    # "No on-scale measurement" must NOT masquerade as an unresolved-at-this-n verdict:
    # ceiling_inside_ci95 goes False and ratio_subgroup_n == 0 carries the information.
    assert st["ceiling_inside_ci95"] is False


def test_all_778c_seeds_are_off_scale_including_the_zero_ratio_ones():
    """Guards the fixture: the five 0.0 ratios are railed-flat, not clean zeros.

    A fully-clamped null reference produces a flat series -> slope 0 -> ratio 0.0,
    which scores <= the 0.25 ceiling and reads as content-contingent. Those zeros are
    the ones most easily mistaken for a clean pass, so they must be inside the
    excluded set, not just the large values.
    """
    rows = _778c_rows()
    assert all(r["rem_ratio_off_scale"] for r in rows)
    zeros = [r["seed"] for r in rows if r["null_slope_ratio"]["rem"] == 0.0]
    assert zeros == [7, 2024, 7777, 314, 1000]
    assert set(zeros).issubset(set(_stats(rows)["excluded_seeds"]))


# --------------------------------------------------------------------------- #
# D4 -- pooling is anti-conservative in both directions.
# --------------------------------------------------------------------------- #


def test_high_side_off_scale_seed_manufactures_a_false_cannot_conclude():
    """778c's own direction: the off-scale seed widens the interval across the ceiling.

    The on-scale seeds sit clearly above 0.25 (rem is confounded); pooling one
    off-scale value inflates the SEM until the ceiling falls inside and the run reads
    "underpowered, cannot conclude" -- erasing a reading the eligible seeds do show.
    """
    rows = [
        _score(1, 0.80, 0.0),
        _score(2, 0.85, 0.0),
        _score(3, 0.90, 0.0),
        _score(7777, 9142.77, 0.2),  # off-scale, high side
    ]
    scoped = _stats(rows)
    pooled = H.subgroup_ratio_stats(
        rows, eligible=lambda s: True,
        value=lambda s: s["null_slope_ratio"]["rem"], ceiling=CEILING,
    )

    assert scoped["subgroup_n"] == 3
    assert scoped["mean"] == pytest.approx(0.85)
    assert scoped["ceiling_inside_ci95"] is False, "eligible seeds are clearly confounded"
    assert pooled["ceiling_inside_ci95"] is True, "pooling erases that reading"


def test_low_side_off_scale_seed_manufactures_false_content_contingency():
    """The other direction -- a railed-flat 0.0 ratio drags the estimate DOWN.

    This is 778c's actual majority shape (five railed seeds at exactly 0.0), and it is
    why the decision cannot be justified by "the error happened to be conservative".
    Pooling zeros pulls the mean toward and across the ceiling, reading as
    content-contingency the on-scale seeds do not show.
    """
    rows = [
        _score(1, 0.80, 0.0),
        _score(2, 0.85, 0.0),
        _score(3, 0.90, 0.0),
        _score(7, 0.0, 1.0),   # railed flat
        _score(2024, 0.0, 1.0),
        _score(314, 0.0, 1.0),
    ]
    scoped = _stats(rows)
    pooled = H.subgroup_ratio_stats(
        rows, eligible=lambda s: True,
        value=lambda s: s["null_slope_ratio"]["rem"], ceiling=CEILING,
    )

    assert scoped["mean"] == pytest.approx(0.85)
    assert pooled["mean"] < scoped["mean"], "railed zeros drag the estimate down"
    assert pooled["ci95_low"] < CEILING < pooled["ci95_high"], (
        "and pull the ceiling inside an interval the scoped reading excludes"
    )
    assert scoped["ceiling_inside_ci95"] is False


# --------------------------------------------------------------------------- #
# D5 -- the narrowing is emitted, with real seed ids.
# --------------------------------------------------------------------------- #


def test_excluded_seed_ids_are_real_values_not_none():
    """`H.subgroup_ratio_stats` default seed_of reads row['seed'].

    `_score_seed` therefore has to carry it -- an excluded_seeds list of
    [None, None, ...] is emitted but not auditable, which defeats the point.
    """
    st = _stats(_778c_rows())
    assert None not in st["excluded_seeds"]
    assert sorted(st["excluded_seeds"]) == sorted(_778C_SEEDS)


def test_score_seed_carries_the_seed_id():
    import importlib

    mod = importlib.import_module(DRIVER[:-3])
    score = mod._score_seed({"null_control_seed": 7777.0})
    assert score["seed"] == 7777
    assert isinstance(score["seed"], int)


def test_driver_emits_the_subgroup_keys_in_the_manifest():
    src = (REPO_ROOT / "experiments" / DRIVER).read_text()
    for key in (
        "ratio_subgroup_n",
        "ratio_excluded_seeds",
        "ratio_n_excluded",
        "ratio_subgroup_basis",
        "per_seed_rem_ratio_off_scale",
    ):
        assert key in src, f"{DRIVER} must emit {key} so the narrowing is auditable"


def test_driver_keeps_the_complete_per_seed_audit_trail():
    """The exclusion is checkable only if the unscoped values are still reported."""
    src = (REPO_ROOT / "experiments" / DRIVER).read_text()
    assert "per_seed_null_slope_ratio" in src
    assert "for s in seed_scores]" in src, (
        "per_seed_null_slope_ratio must stay an ALL-seeds comprehension"
    )


# --------------------------------------------------------------------------- #
# D6 -- the confound VERDICT stays unscoped.
# --------------------------------------------------------------------------- #


def test_confound_verdict_counts_all_seeds_not_the_on_scale_subgroup():
    """Off-scale bears on magnitude, not on whether the phase is confounded.

    Pinned as an intended NON-change: the register's standing rule is that confounded
    phases are reported and never dropped, so narrowing this counter would hide a
    confound rather than surface it -- the opposite of the register's purpose.
    """
    src = (REPO_ROOT / "experiments" / DRIVER).read_text()
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, val in zip(node.keys, node.values):
            if isinstance(key, ast.Constant) and key.value == "n_seeds_confounded":
                found = True
    assert found, "n_seeds_confounded must still be emitted"

    # The counter itself is over seed_scores with no eligibility predicate.
    assert 'sum(1 for s in seed_scores if p in s["confounded_phases"])' in src, (
        "n_seeds_confounded must remain an ALL-seeds count"
    )


# --------------------------------------------------------------------------- #
# D7 -- the shipped driver routes through the scoped aggregator.
# --------------------------------------------------------------------------- #


def test_driver_uses_subgroup_ratio_stats():
    src = (REPO_ROOT / "experiments" / DRIVER).read_text()
    assert "subgroup_ratio_stats(" in src, (
        f"{DRIVER} must aggregate the per-phase ratio through the scoped helper"
    )
    assert "rem_ratio_off_scale" in src


def test_driver_has_no_unfiltered_rem_ratio_comprehension():
    """Regression guard: the finiteness-only comprehension must not come back.

    Matches the defective shape -- a comprehension over `seed_scores` pulling
    `null_slope_ratio` with no off-scale predicate in the filter. `per_seed_*` entries
    are per-seed REPORTING and legitimately unfiltered (see D5), so they are exempted
    by their dict key, exactly as in test_sd068_c2_subgroup_aggregation.py.
    """
    src = (REPO_ROOT / "experiments" / DRIVER).read_text()
    tree = ast.parse(src)

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
        if "null_slope_ratio" not in body or "seed_scores" not in body:
            continue
        if "rem_ratio_off_scale" in body:
            continue  # correctly scoped
        offenders.append(ast.unparse(comp)[:120])

    assert not offenders, (
        f"{DRIVER} reintroduced an unfiltered per-phase ratio aggregation over "
        f"seed_scores: {offenders}"
    )
