"""Contracts for the NULL-ARM degeneracy guard on the SD-068 content-contingency
predicate.

This is the THIRD instance found by the family audit that produced the V3-EXQ-778h C2
subgroup-aggregation fix (ree-v3 b42f69ffa3,
tests/contracts/test_sd068_c2_subgroup_aggregation.py) and the rem off-scale magnitude
scoping (ree-v3 3718df6e73,
tests/contracts/test_sd068_null_content_rem_offscale_scoping.py). It was found during
the second fix's work and deliberately left unchanged there, because unlike those two
it moves a PASS PREDICATE rather than a descriptive statistic. THIS FILE PINS THE
DECISION.

THE DECISION: a CONSTANT null series is NOT content-contingent.

`content_contingent_<phase>` scored a phase as content-contingent whenever
`null_slope_ratio <= NULL_SLOPE_RATIO_CEILING` (0.25), with an `available` guard that
covers only the INJECTED (denominator) slope via NULL_MIN_INJECTED_SLOPE. Nothing
guarded the NULL arm. A fully-railed null arm produces a flat error series ->
null_slope == 0 -> ratio == 0.0 -> 0.0 <= 0.25 -> scored CONTENT-CONTINGENT. A zero
slope from a saturated constant is the ABSENCE of a measurement, not evidence that the
readout is inert on noise.

CONCRETE INSTANCE, run v3_exq_sd068_null_content_control_diagnostic_20260718T072318Z_v3
(queue V3-EXQ-778c): 5 of 8 seeds (7, 2024, 7777, 314, 1000) have
`null_rem_target_clamped_frac` 1.0 and `null_slope_ratio_rem` exactly 0.0, and each
reports `confounded_phases == ['sws']` only -- i.e. all five voted "rem is
content-contingent" off a railed-flat null reference. That is why
`confound_verdict_stable` is False for rem (n_seeds_confounded 3 of 8). The follow-on
leg's own docstring already named this
(v3_exq_sd068_rem_declamped_readout_diagnostic.py: "DEGENERATE AT BOTH RAILS ... the 5
apparently-clean seeds are clean only BY DEGENERACY").

WHY THIS WAS NOT ALREADY DELEGATED TO CONSUMERS. The harness emits
`null_series_sd_<phase>` / `null_series_n_distinct_<phase>` (V3-EXQ-778e) with a
comment saying the distinct-value count "lets a consumer gate on that difference
instead of reading a degenerate zero as a clean pass" -- so the guard may have been
intended as consumer-side. Audited: the ONLY shipped consumer of that telemetry is
`v3_exq_sd068_rem_declamped_readout_diagnostic.py::_null_series_non_degenerate`, and
it reads the UNPREFIXED keys emitted by the rem-scoped wrapper `rem_null_slope_ratio`,
applying the gate as a SEPARATE criterion while ignoring `content_contingent`
entirely. NO consumer of the per-phase `_<phase>`-suffixed fields ever gated on them --
`v3_exq_sd068_null_content_control_diagnostic.py::_score_seed` reads
`content_contingent_<p>` raw, guarded only by an injected-slope interpretability check
(C2). So the delegated-by-design reading is not supported: the telemetry shipped and no
per-phase consumer used it. The guard belongs in the predicate.

WHY THE THRESHOLD IS 2 AND NOT 3 -- the load-bearing part of this decision.
NULL_MIN_NULL_SERIES_DISTINCT is 2 (reject only a CONSTANT series), deliberately weaker
than the consumer-side NULL_SERIES_MIN_DISTINCT = 3 in the 778e driver. Importing the
consumer's 3 would FLIP AN ADJUDICATED PASS: run
v3_exq_sd068_sws_content_scored_readout_diagnostic_20260718T130139Z_v3 has
`null_series_n_distinct_sws` == 2.0 on 8/8 seeds while carrying a real non-zero null
slope (0.0376-0.0615, ratio 0.116-0.180), and its LOAD-BEARING C1 is
`C1_sws_content_contingent` on exactly that phase. A 2-distinct-value series with a
genuine graded slope is a real measurement that happens to be small -- not a rail. The
harness therefore guards only the absence-of-measurement rail; a consumer wanting
stricter gradedness imposes it on top, as 778e does.

VERDICT IMPACT ON COMPLETED RUNS: NONE. All 10 completed SD-068 manifests in
REE_assembly/evidence/experiments/ (V3-EXQ-778 through 778h) were checked at the
per-seed level.
  - 778g sws_content_scored (PASS): C1 is `C1_sws_content_contingent`, sws-only, and
    sws survives at n_distinct 2 (above). Its rem/nrem legs are explicitly
    `"gated": false` in context_summary -- reported context, not criteria.
  - 778b + 778c null_content_control (FAIL): C1 requires ALL THREE phases; the sws leg
    was confounded at ratio 0.99999999, `content_contingent.sws` False on EVERY
    recorded seed, so C1 was already False independent of rem. The railed rem seeds
    (5/8 on 778c) flip from contingent to degenerate, which corrects the register's
    per-seed count without touching the verdict.
  - 778d rem_unpaired_null (FAIL) + 778h rem_unpaired_null_anchorfix (PASS): the field
    DOES flip here -- on both runs, the NULL_ZERO *anchor* arm's full_grid has
    n_distinct 1.0 with content_contingent 1.0 on the same 5 seeds (7, 2024, 7777, 314,
    1000). No criterion reads it. C1 routes on `derailed` (clamp_frac / n_distinct /
    n_unclamped), C3 on `_railed` (the same fields plus ratio_full / min_ref) -- both
    read the degeneracy telemetry DIRECTLY and neither touches content_contingent. C2
    reads `content_contingent_unclamped`, but that comes from the NULL_UNPAIRED arm's
    unclamped_grid, where every seed has n_distinct >= 3. The flip also runs WITH the
    anchor's intent, not against it: those 5 seeds are the ones C3 wants scored railed.
  - 778e rem_declamped_readout (FAIL): already applies the stricter consumer-side gate
    and does not route its verdict through `content_contingent`.
  - 778 x2, 778a staging_power, 778f rem_gen_gain (PASS): carry no content-contingency
    fields at all.
No SD-068 experiment is re-run or re-queued: they are adjudicated, and this makes the
predicate honest for FUTURE runs.

Pinned here:
  D1  a CONSTANT null series is not content-contingent, even at ratio exactly 0.0.
  D2  a GENUINE near-zero null series IS content-contingent -- the guard rejects
      absence-of-measurement, not smallness. (D1+D2 pin the decision in BOTH
      directions, so it is not justified by the sign of the 778c instance.)
  D3  the threshold is 2, and the real 778g sws configuration (n_distinct 2, genuine
      slope) still passes -- the regression that a stricter threshold would cause.
  D4  the degeneracy is EMITTED, not silent, and a degenerate phase is named in
      confounded_phases rather than dropped.
  D5  the injected-arm guard is unchanged and still independent of the null-arm one.
  D6  the rem-scoped wrapper `rem_null_slope_ratio` carries the identical guard --
      the predicate exists in two places and both must move together.
  D7  the shipped 778e consumer's stricter gate still composes on top and has not
      regressed to the harness threshold.
"""

import math

import pytest

from experiments._lib import consolidation_lesion_harness as H


SIGMAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def _phase_rows(
    *,
    sws_margin,
    nrem_gap_after,
    rem_err,
    margin_clean=1.0,
    gap_before=1.0,
    clean_target_variance=1.0,
    rem_clamped=0.0,
):
    """One sigma's worth of the three phase readouts, as the harness consumes them."""
    return {
        "sws": {
            "sws_completion_margin": sws_margin,
            "sws_completion_margin_clean": margin_clean,
        },
        "nrem": {"gap_after": nrem_gap_after, "gap_before": gap_before},
        "rem": {
            "calibration_error": rem_err,
            "clean_target_variance": clean_target_variance,
            "target_clamped": rem_clamped,
        },
    }


def _arm(sws_of, nrem_of, rem_of, rem_clamped_of=lambda s: 0.0):
    return {
        s: _phase_rows(
            sws_margin=sws_of(s),
            nrem_gap_after=nrem_of(s),
            rem_err=rem_of(s),
            rem_clamped=rem_clamped_of(s),
        )
        for s in SIGMAS
    }


# The INJECTED arm is held identical across every case below, so the ONLY thing that
# differs between a degenerate and a genuine null is the null arm's own series. Each
# phase has a comfortably-above-floor injected slope.
#   sws series = 1 - margin/1.0 = 0.30*s   -> slope 0.30
#   nrem series = gap_after/1.0 = 0.20*s   -> slope 0.20
#   rem series = err/1.0 = 0.10 + 1.00*s   -> slope 1.00
INJECTED = _arm(
    sws_of=lambda s: 1.0 - 0.30 * s,
    nrem_of=lambda s: 0.20 * s,
    rem_of=lambda s: 0.10 + 1.00 * s,
)

# RAILED null: every phase saturated to a constant. Slope exactly 0 -> ratio exactly
# 0.0 -> clears the 0.25 ceiling. This is the 778c rem signature (the constant 998.5
# with target_clamped 1.0) generalised to all three phases.
NULL_RAILED = _arm(
    sws_of=lambda s: 0.50,
    nrem_of=lambda s: 0.40,
    rem_of=lambda s: 998.5009992509989,
    rem_clamped_of=lambda s: 1.0,
)

# GENUINE near-zero null: graded on every phase, but with a small slope well inside the
# ceiling. sws ratio 0.02/0.30, nrem 0.01/0.20, rem 0.05/1.00 -- all <= 0.25.
NULL_GENUINE = _arm(
    sws_of=lambda s: 1.0 - 0.02 * s,
    nrem_of=lambda s: 0.01 * s,
    rem_of=lambda s: 0.10 + 0.05 * s,
)

PHASES = ("sws", "nrem", "rem")


def _control(null_arm):
    return H.run_null_content_control(
        seed=0, sigmas=SIGMAS, injected_pr_by_sigma=INJECTED, null_pr_by_sigma=null_arm
    )


# --------------------------------------------------------------------------- #
# D1 -- a constant null series is NOT content-contingent                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("phase", PHASES)
def test_d1_railed_null_is_not_content_contingent(phase):
    out = _control(NULL_RAILED)

    # The premise: this is exactly the pathological configuration -- a flat null series
    # whose ratio sails under the ceiling.
    assert out[f"null_slope_{phase}"] == 0.0
    assert out[f"null_slope_ratio_{phase}"] == 0.0
    assert 0.0 <= H.NULL_SLOPE_RATIO_CEILING
    # ...and the injected arm is perfectly interpretable, so `available` cannot be
    # doing this work.
    assert out[f"null_control_available_{phase}"] == 1.0

    # The contract.
    assert out[f"null_series_n_distinct_{phase}"] == 1.0
    assert out[f"content_contingent_{phase}"] == 0.0, (
        f"{phase}: a railed-flat null series scored CONTENT-CONTINGENT off a zero "
        "slope -- that is the absence of a measurement, not content-contingency"
    )


def test_d1_railed_null_fails_the_all_phases_roll_up():
    out = _control(NULL_RAILED)
    assert out["all_phases_content_contingent"] == 0.0
    assert out["n_content_contingent_phases"] == 0.0
    assert out["n_confounded_phases"] == 3.0


# --------------------------------------------------------------------------- #
# D2 -- a genuine near-zero null IS content-contingent (the other direction)   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("phase", PHASES)
def test_d2_genuine_near_zero_null_is_content_contingent(phase):
    out = _control(NULL_GENUINE)

    ratio = out[f"null_slope_ratio_{phase}"]
    assert 0.0 < ratio <= H.NULL_SLOPE_RATIO_CEILING, (
        f"{phase}: fixture must be a genuinely small-but-nonzero null, got {ratio}"
    )
    assert out[f"null_series_n_distinct_{phase}"] >= H.NULL_MIN_NULL_SERIES_DISTINCT
    assert out[f"null_series_degenerate_{phase}"] == 0.0
    assert out[f"content_contingent_{phase}"] == 1.0, (
        f"{phase}: the guard rejects ABSENCE of measurement, not smallness -- a "
        "graded null series inside the ceiling must still score content-contingent"
    )


def test_d2_genuine_near_zero_null_passes_the_all_phases_roll_up():
    out = _control(NULL_GENUINE)
    assert out["all_phases_content_contingent"] == 1.0
    assert out["n_confounded_phases"] == 0.0
    assert H.confounded_phase_names(out) == []


def test_d2_guard_is_the_only_difference_between_the_two_readings():
    """The railed and genuine nulls differ ONLY in degeneracy, not in the ceiling test.

    Both clear `ratio <= ceiling` on all three phases. If the verdicts diverge, the
    divergence is attributable to the degeneracy guard and nothing else -- which is
    what makes D1 and D2 a matched pair rather than two unrelated fixtures.
    """
    railed = _control(NULL_RAILED)
    genuine = _control(NULL_GENUINE)
    for phase in PHASES:
        assert railed[f"null_slope_ratio_{phase}"] <= H.NULL_SLOPE_RATIO_CEILING
        assert genuine[f"null_slope_ratio_{phase}"] <= H.NULL_SLOPE_RATIO_CEILING
        assert railed[f"null_control_available_{phase}"] == 1.0
        assert genuine[f"null_control_available_{phase}"] == 1.0
        assert railed[f"content_contingent_{phase}"] == 0.0
        assert genuine[f"content_contingent_{phase}"] == 1.0


# --------------------------------------------------------------------------- #
# D3 -- the threshold is 2, and the real 778f sws configuration survives       #
# --------------------------------------------------------------------------- #


def test_d3_threshold_is_two_not_three():
    assert H.NULL_MIN_NULL_SERIES_DISTINCT == 2


def test_d3_two_distinct_null_with_a_real_slope_still_passes():
    """The regression a stricter threshold would cause.

    Reproduces the shape of run
    v3_exq_sd068_sws_content_scored_readout_diagnostic_20260718T130139Z_v3, whose sws
    leg carries n_distinct 2 on 8/8 seeds with a genuine non-zero slope and is that
    run's LOAD-BEARING C1. A step-shaped null series is a real graded response; only a
    CONSTANT one is the absence of a measurement.
    """
    # A 2-valued step across the grid: low for the first two sigmas, high after.
    step = _arm(
        sws_of=lambda s: 1.0 - (0.0 if s <= 0.25 else 0.06),
        nrem_of=lambda s: (0.0 if s <= 0.25 else 0.02),
        rem_of=lambda s: 0.10 + (0.0 if s <= 0.25 else 0.06),
    )
    out = _control(step)
    for phase in PHASES:
        assert out[f"null_series_n_distinct_{phase}"] == 2.0
        assert out[f"null_slope_{phase}"] != 0.0
        assert 0.0 < out[f"null_slope_ratio_{phase}"] <= H.NULL_SLOPE_RATIO_CEILING
        assert out[f"null_series_degenerate_{phase}"] == 0.0
        assert out[f"content_contingent_{phase}"] == 1.0, (
            f"{phase}: n_distinct 2 with a real slope is a measurement, not a rail -- "
            "raising the harness threshold to the consumer's 3 would flip the "
            "adjudicated V3-EXQ-778g PASS"
        )


def test_d3_harness_threshold_is_strictly_weaker_than_the_shipped_consumer_gate():
    """Pins the LAYERING, so the two thresholds cannot silently converge."""
    mod = pytest.importorskip(
        "experiments.v3_exq_sd068_rem_declamped_readout_diagnostic"
    )
    assert mod.NULL_SERIES_MIN_DISTINCT > H.NULL_MIN_NULL_SERIES_DISTINCT, (
        "the consumer gate must stay strictly stricter than the harness floor; if "
        "they converge, either the 778f PASS flips or the consumer's rail-(a) "
        "coverage weakens"
    )


# --------------------------------------------------------------------------- #
# D4 -- the degeneracy is emitted and named, not silent                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("phase", PHASES)
def test_d4_degeneracy_flag_is_emitted(phase):
    railed = _control(NULL_RAILED)
    genuine = _control(NULL_GENUINE)
    assert railed[f"null_series_degenerate_{phase}"] == 1.0
    assert genuine[f"null_series_degenerate_{phase}"] == 0.0


def test_d4_degenerate_phase_is_named_not_dropped():
    """Consistent with the register's standing rule: a phase that cannot pass is
    NAMED, never silently removed from the staging order."""
    out = _control(NULL_RAILED)
    assert set(H.confounded_phase_names(out)) == set(PHASES)


def test_d4_degeneracy_is_distinguishable_from_confounding():
    """A degenerate phase and a genuinely confounded phase both fail the predicate,
    but the flag lets a reader tell "no measurement" from "responds to noise"."""
    confounded = _arm(  # null tracks injected exactly -> ratio 1.0, graded
        sws_of=lambda s: 1.0 - 0.30 * s,
        nrem_of=lambda s: 0.20 * s,
        rem_of=lambda s: 0.10 + 1.00 * s,
    )
    out = _control(confounded)
    for phase in PHASES:
        assert out[f"content_contingent_{phase}"] == 0.0
        assert out[f"null_slope_ratio_{phase}"] == pytest.approx(1.0)
        # Failed for the OTHER reason -- not degenerate.
        assert out[f"null_series_degenerate_{phase}"] == 0.0


# --------------------------------------------------------------------------- #
# D5 -- the injected-arm guard is unchanged and independent                    #
# --------------------------------------------------------------------------- #


def test_d5_flat_injected_arm_is_still_unavailable_not_contingent():
    flat_injected = _arm(
        sws_of=lambda s: 1.0 - 0.30 * s,  # sws keeps a real injected slope
        nrem_of=lambda s: 0.0,  # nrem denominator collapses
        rem_of=lambda s: 0.10,  # rem denominator collapses
    )
    out = H.run_null_content_control(
        seed=0,
        sigmas=SIGMAS,
        injected_pr_by_sigma=flat_injected,
        null_pr_by_sigma=NULL_GENUINE,
    )
    for phase in ("nrem", "rem"):
        assert out[f"null_control_available_{phase}"] == 0.0
        assert out[f"null_slope_ratio_{phase}"] == H.UNAVAILABLE
        assert out[f"content_contingent_{phase}"] == 0.0
    # The unaffected phase is untouched by the injected-arm collapse elsewhere.
    assert out["null_control_available_sws"] == 1.0
    assert out["content_contingent_sws"] == 1.0


def test_d5_injected_slope_floor_constant_unchanged():
    assert H.NULL_MIN_INJECTED_SLOPE == 1e-9
    assert H.NULL_SLOPE_RATIO_CEILING == 0.25


# --------------------------------------------------------------------------- #
# D6 -- the rem-scoped wrapper carries the identical guard                     #
# --------------------------------------------------------------------------- #


def _rem_cell(null_arm, rem_error_key="calibration_error"):
    return H.rem_null_slope_ratio(
        sigmas=SIGMAS,
        injected_pr_by_sigma=INJECTED,
        null_pr_by_sigma=null_arm,
        rem_error_key=rem_error_key,
    )


def test_d6_wrapper_railed_null_is_not_content_contingent():
    cell = _rem_cell(NULL_RAILED)
    assert cell["null_slope_ratio"] == 0.0
    assert cell["available"] == 1.0
    assert cell["null_series_n_distinct"] == 1.0
    assert cell["null_series_degenerate"] == 1.0
    assert cell["content_contingent"] == 0.0


def test_d6_wrapper_genuine_null_is_content_contingent():
    cell = _rem_cell(NULL_GENUINE)
    assert 0.0 < cell["null_slope_ratio"] <= H.NULL_SLOPE_RATIO_CEILING
    assert cell["null_series_degenerate"] == 0.0
    assert cell["content_contingent"] == 1.0


def test_d6_wrapper_and_per_phase_predicate_agree_on_rem():
    """The predicate lives in two places; they must not drift apart."""
    for null_arm in (NULL_RAILED, NULL_GENUINE):
        per_phase = _control(null_arm)
        cell = _rem_cell(null_arm)
        assert cell["content_contingent"] == per_phase["content_contingent_rem"]
        assert cell["null_series_degenerate"] == per_phase["null_series_degenerate_rem"]
        assert cell["null_series_n_distinct"] == per_phase["null_series_n_distinct_rem"]


# --------------------------------------------------------------------------- #
# D7 -- the shipped consumer's stricter gate still composes on top             #
# --------------------------------------------------------------------------- #


def test_d7_consumer_gate_still_rejects_the_railed_cell():
    mod = pytest.importorskip(
        "experiments.v3_exq_sd068_rem_declamped_readout_diagnostic"
    )
    railed = _rem_cell(NULL_RAILED)
    genuine = _rem_cell(NULL_GENUINE)
    # The consumer gate is unchanged in behaviour by the harness guard.
    assert mod._null_series_non_degenerate(railed) is False
    assert mod._null_series_non_degenerate(genuine) is True
    # ...and the harness now agrees with it on the railed case, where it previously
    # contradicted it by scoring content_contingent 1.0.
    assert railed["content_contingent"] == 0.0


def test_d7_frozen_reference_series_all_clear_both_gates():
    """The 778e frozen known-positive references must clear the harness floor too.

    If the harness floor ever rose above these, the shipped run's own recorded
    positive control would stop validating.
    """
    mod = pytest.importorskip(
        "experiments.v3_exq_sd068_rem_declamped_readout_diagnostic"
    )
    for row in mod._REFERENCE_DECLAMPED_GRADED_SERIES:
        assert row["null_series_n_distinct"] >= H.NULL_MIN_NULL_SERIES_DISTINCT
        assert not math.isnan(row["null_series_sd"])
        assert mod._null_series_non_degenerate(row) is True
