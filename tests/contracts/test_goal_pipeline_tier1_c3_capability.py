"""Contracts for the goal_pipeline:GAP-4 Tier-1 C3-lift capability guard.

THE DEFECT THIS GUARDS (third defeat-by-construction in one cohort, 2026-07-30).

`evaluate_tier1_cohort`'s C3_lift_vs_baseline criterion was, for two months,
`goal_norm_peak_delta` -- `gap4.goal_norm_peak > base.goal_norm_peak + 0.01`
-- applied by default to a pairing on which it CANNOT be satisfied:

  * goal_norm_peak is a LIFETIME RUNNING MAXIMUM. GoalState.reset() zeroes it,
    REEAgent.reset() never calls GoalState.reset(), so it is the max over the
    agent's whole warmup+eval life (~12k steps), not an eval statistic.
  * It is a norm in each arm's own FREE-SCALE latent space (z_goal is an EMA
    toward an unnormalised z_world), and the two arms are built by DIFFERENT
    REEConfig constructors -- goal_stream(...) vs from_dims(...). An additive
    threshold across that boundary compares two unrelated units.
  * Measured on V3-EXQ-490i, the legacy arm sat ABOVE the gap4 arm on every
    seed, so the predicate returned False 3/3: inverted, not a near miss.

Each script then stamped its claim direction as
`"supports" if outcome == "PASS" else "weakens"`, so a criterion that could not
fire wrote `weakens` onto live claims. The 2026-05-29 cluster autopsy had to
correct exactly that class of stamp by hand across this cohort.

WHY A CAPABILITY GUARD AND NOT JUST A BETTER METRIC. This is the THIRD time a
GAP-4 Tier-1 criterion was defeated by construction, and the second was
introduced by the rebuild fixing the first:
  1. approach_commit_rate saturated at 1.0 in the 483c/524a baseline arm.
  2. mech295_bias_range_mean = 0.0 made an argmin-flip impossible (490k).
  3. goal_norm_peak_delta inverted on the between-path pairing.
Each was individually invisible in review and only found after a cohort had
run. Replacing metric #3 without a guard would just queue up #4.

WHAT IS DELIBERATELY *NOT* TESTED HERE: that the landed 483c / 490g-i manifests
get rewritten. Those are record-frozen; the replay cases below assert what the
guard WOULD have said about their rows, which is the regression pin.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
for _p in (str(REPO_ROOT), str(EXPERIMENTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _lib.goal_pipeline_tier1 import (  # noqa: E402
    C3_METRICS,
    DEFAULT_C3_LIFT_METRIC,
    TIER1_SEEDS_PASS_MIN,
    c3_lift_capability,
    evaluate_tier1_cohort,
    tier1_evidence_direction,
)

GAP4 = "ARM_1_gap4_operating"
BASE = "ARM_0_legacy_collapsed"

# Verbatim per-seed values from the landed V3-EXQ-490i manifest
# (v3_exq_490i_mech295_cascade_gap4_tier1_20260530T184434Z_v3.json). Inlined
# rather than read from REE_assembly: the contract suite runs on cloud workers
# that stage ree-v3 only.
ROWS_490I = [
    dict(arm=BASE, seed=42, gap4_operating=False,
         goal_norm_peak=0.7924675941467285, approach_commit_rate=0.0),
    dict(arm=GAP4, seed=42, gap4_operating=True,
         goal_norm_peak=0.2260640412569046, approach_commit_rate=1.0),
    dict(arm=BASE, seed=7, gap4_operating=False,
         goal_norm_peak=12.488753318786621, approach_commit_rate=0.0),
    dict(arm=GAP4, seed=7, gap4_operating=True,
         goal_norm_peak=0.09192166477441788, approach_commit_rate=1.0),
    dict(arm=BASE, seed=19, gap4_operating=False,
         goal_norm_peak=0.46786096692085266, approach_commit_rate=0.0),
    dict(arm=GAP4, seed=19, gap4_operating=True,
         goal_norm_peak=0.29584237933158875, approach_commit_rate=1.0),
]

REQUIRED_SPEC_KEYS = {
    "key", "direction", "delta", "ceiling", "floor",
    "cross_arm_valid", "scale_free", "note",
}


# --------------------------------------------------------------------------
# Registry shape
# --------------------------------------------------------------------------

def test_every_metric_declares_a_complete_spec():
    for name, spec in C3_METRICS.items():
        missing = REQUIRED_SPEC_KEYS - set(spec)
        assert not missing, f"C3_METRICS['{name}'] missing keys: {sorted(missing)}"
        assert spec["direction"] in ("higher", "lower"), (
            f"C3_METRICS['{name}'].direction must be 'higher' or 'lower'"
        )
        assert float(spec["delta"]) >= 0.0


def test_default_metric_is_registered_and_valid_across_the_gap4_boundary():
    """The harness's canonical contrast IS between-path, so the DEFAULT metric
    must be one that survives it. goal_norm_peak_delta was not, which is the
    whole defect."""
    assert DEFAULT_C3_LIFT_METRIC in C3_METRICS
    assert C3_METRICS[DEFAULT_C3_LIFT_METRIC]["cross_arm_valid"] is True, (
        "the default C3 lift metric must be comparable between a gap4_operating "
        "arm and a non-gap4 arm -- that is the pairing 471/475/490 all use"
    )


def test_goal_norm_peak_delta_is_pinned_not_cross_arm_valid():
    """Regression pin for the 2026-07-30 finding. If this flips back to True,
    the inverted criterion is live again."""
    assert C3_METRICS["goal_norm_peak_delta"]["cross_arm_valid"] is False


# --------------------------------------------------------------------------
# The 490i replay -- the actual incident
# --------------------------------------------------------------------------

def test_490i_rows_under_old_metric_are_refused_as_a_criterion_defect():
    acc = evaluate_tier1_cohort(
        ROWS_490I, gap4_arm_id=GAP4, baseline_arm_id=BASE,
        c3_lift_metric="goal_norm_peak_delta",
    )
    assert acc["C3_lift_vs_baseline"] is False
    assert acc["C3_lift_count"] == 0
    assert acc["C3_lift_status"] == "invalid_cross_arm"
    assert acc["criterion_valid"] is False


def test_490i_rows_under_old_metric_do_not_stamp_weakens():
    """The precise regression: a criterion that could not fire must not move
    claim confidence. This is the algorithm-generated-`weakens` pattern."""
    acc = evaluate_tier1_cohort(
        ROWS_490I, gap4_arm_id=GAP4, baseline_arm_id=BASE,
        c3_lift_metric="goal_norm_peak_delta",
    )
    assert acc["recommended_evidence_direction"] == "non_contributory"
    assert tier1_evidence_direction(acc) != "weakens"


def test_490i_rows_under_the_new_default_lift_cleanly():
    """approach_commit_rate on the between-path pairing: base 0.0 vs gap4 1.0
    on 3/3 seeds. The metric deprecated in 2026-05-29 as 'no headroom' is in
    fact maximally discriminative HERE -- the saturation was measured on a
    within-gap4 contrast (483c) and wrongly generalised."""
    acc = evaluate_tier1_cohort(ROWS_490I, gap4_arm_id=GAP4, baseline_arm_id=BASE)
    assert acc["C3_lift_metric"] == "approach_commit_rate"
    assert acc["C3_lift_status"] == "ok"
    assert acc["criterion_valid"] is True
    assert acc["C3_lift_count"] == 3
    assert acc["C3_lift_vs_baseline"] is True


# --------------------------------------------------------------------------
# Each capability failure mode
# --------------------------------------------------------------------------

def _pair_rows(metric_key, base_vals, gap4_vals, gap4_flag=True, base_flag=True):
    rows = []
    for i, (b, g) in enumerate(zip(base_vals, gap4_vals)):
        rows.append({"arm": BASE, "seed": i, "gap4_operating": base_flag, metric_key: b})
        rows.append({"arm": GAP4, "seed": i, "gap4_operating": gap4_flag, metric_key: g})
    return rows


def test_degenerate_when_the_toggle_does_not_move_the_metric():
    """483c's shape: all four arms gap4_operating=True and approach_commit_rate
    identically 1.0, so the sub-feature toggle cannot register."""
    rows = _pair_rows("approach_commit_rate", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    cap = c3_lift_capability(
        [r for r in rows if r["arm"] == GAP4],
        [r for r in rows if r["arm"] == BASE],
        "approach_commit_rate",
    )
    assert cap["status"] == "degenerate"
    assert cap["criterion_valid"] is False


def test_saturated_when_the_baseline_is_already_at_the_ceiling():
    rows = _pair_rows("approach_commit_rate", [1.0, 1.0, 1.0], [0.4, 0.9, 1.0])
    cap = c3_lift_capability(
        [r for r in rows if r["arm"] == GAP4],
        [r for r in rows if r["arm"] == BASE],
        "approach_commit_rate",
    )
    assert cap["status"] == "saturated"
    assert cap["criterion_valid"] is False


def test_inverted_when_the_baseline_beats_gap4_on_every_seed():
    rows = _pair_rows("approach_commit_rate", [0.8, 0.7, 0.9], [0.2, 0.1, 0.3])
    cap = c3_lift_capability(
        [r for r in rows if r["arm"] == GAP4],
        [r for r in rows if r["arm"] == BASE],
        "approach_commit_rate",
    )
    assert cap["status"] == "inverted"
    assert cap["criterion_valid"] is False


def test_ok_when_the_metric_can_move_in_the_claimed_direction():
    rows = _pair_rows("approach_commit_rate", [0.1, 0.2, 0.9], [0.8, 0.7, 0.3])
    cap = c3_lift_capability(
        [r for r in rows if r["arm"] == GAP4],
        [r for r in rows if r["arm"] == BASE],
        "approach_commit_rate",
    )
    assert cap["status"] == "ok"
    assert cap["criterion_valid"] is True


# --------------------------------------------------------------------------
# Direction-awareness -- the SD-036 mis-signing
# --------------------------------------------------------------------------

def test_a_lower_direction_metric_counts_a_DECREASE_as_lift():
    """SD-036 is a DECAY claim: a working regulator should REDUCE the sustain
    ratio. Judging it with a hardcoded `>` (as the pre-2026-07-30 comparator
    did for every metric) inverts the claim's sign."""
    rows = _pair_rows("harm_norm_sustain_ratio", [0.80, 0.75, 0.70], [0.30, 0.25, 0.20])
    acc = evaluate_tier1_cohort(
        rows, gap4_arm_id=GAP4, baseline_arm_id=BASE,
        c3_lift_metric="harm_norm_sustain_ratio",
    )
    assert acc["C3_lift_count"] == 3
    assert acc["C3_lift_vs_baseline"] is True
    assert acc["C3_lift_status"] == "ok"


def test_a_lower_direction_metric_flags_an_increase_as_inverted():
    rows = _pair_rows("harm_norm_sustain_ratio", [0.20, 0.25, 0.30], [0.80, 0.75, 0.70])
    acc = evaluate_tier1_cohort(
        rows, gap4_arm_id=GAP4, baseline_arm_id=BASE,
        c3_lift_metric="harm_norm_sustain_ratio",
    )
    assert acc["C3_lift_status"] == "inverted"
    assert acc["recommended_evidence_direction"] == "non_contributory"


def test_harm_sustain_ratio_is_the_sd036_readout_not_a_goal_metric():
    """The SD-036 regulator ticks z_harm / z_harm_a / z_beta and never touches
    z_goal, so no goal_* metric is causally reachable by the toggle."""
    spec = C3_METRICS["harm_norm_sustain_ratio"]
    assert spec["direction"] == "lower"
    assert spec["scale_free"] is True
    assert not spec["key"].startswith("goal_")


# --------------------------------------------------------------------------
# Back-compat + invariants
# --------------------------------------------------------------------------

def test_rows_without_gap4_provenance_skip_the_cross_arm_check():
    """Manifests predating the gap4_operating field must not be judged by a
    guessed boundary."""
    rows = [
        {"arm": BASE, "seed": 1, "goal_norm_peak": 0.10},
        {"arm": GAP4, "seed": 1, "goal_norm_peak": 0.50},
        {"arm": BASE, "seed": 2, "goal_norm_peak": 0.80},
        {"arm": GAP4, "seed": 2, "goal_norm_peak": 0.20},
    ]
    cap = c3_lift_capability(
        [r for r in rows if r["arm"] == GAP4],
        [r for r in rows if r["arm"] == BASE],
        "goal_norm_peak_delta",
    )
    assert cap["crosses_gap4_boundary"] is None
    assert cap["status"] != "invalid_cross_arm"


def test_an_invalid_criterion_never_yields_weakens():
    for status_rows, metric in [
        (_pair_rows("approach_commit_rate", [1.0, 1.0], [1.0, 1.0]), "approach_commit_rate"),
        (_pair_rows("approach_commit_rate", [0.9, 0.8], [0.2, 0.1]), "approach_commit_rate"),
        (ROWS_490I, "goal_norm_peak_delta"),
    ]:
        acc = evaluate_tier1_cohort(
            status_rows, gap4_arm_id=GAP4, baseline_arm_id=BASE, c3_lift_metric=metric,
        )
        assert acc["criterion_valid"] is False
        assert tier1_evidence_direction(acc) == "non_contributory"


def test_a_valid_criterion_that_fails_still_yields_weakens():
    """The guard must not launder every FAIL into non_contributory -- a real
    null result has to stay readable as one."""
    acc = {"pass": False, "criterion_valid": True}
    assert tier1_evidence_direction(acc) == "weakens"


def test_pass_always_yields_supports():
    assert tier1_evidence_direction({"pass": True, "criterion_valid": True}) == "supports"
    assert tier1_evidence_direction({"pass": True, "criterion_valid": False}) == "supports"


def test_unknown_metric_raises_rather_than_silently_defaulting():
    with pytest.raises(ValueError, match="Unknown c3_lift_metric"):
        c3_lift_capability([{"arm": GAP4, "seed": 1}], [{"arm": BASE, "seed": 1}], "not_a_metric")


def test_no_baseline_leaves_the_criterion_valid_and_inert():
    rows = [{"arm": GAP4, "seed": s, "gap4_operating": True} for s in (1, 2, 3)]
    acc = evaluate_tier1_cohort(rows, gap4_arm_id=GAP4, baseline_arm_id=None)
    assert acc["C3_lift_vs_baseline"] is True
    assert acc["criterion_valid"] is True
    assert acc["C3_lift_status"] == "no_pairs"


def test_seeds_pass_min_is_what_the_degeneracy_threshold_uses():
    """Two degenerate pairs out of three trips it; one does not."""
    assert TIER1_SEEDS_PASS_MIN == 2
    two = _pair_rows("approach_commit_rate", [0.5, 0.5, 0.1], [0.5, 0.5, 0.9])
    one = _pair_rows("approach_commit_rate", [0.5, 0.2, 0.1], [0.5, 0.6, 0.9])
    for rows, expected in ((two, "degenerate"), (one, "ok")):
        cap = c3_lift_capability(
            [r for r in rows if r["arm"] == GAP4],
            [r for r in rows if r["arm"] == BASE],
            "approach_commit_rate",
        )
        assert cap["status"] == expected
