"""
Contract: the Q-081 constrained-realisation surrogate null, VALIDATED BEFORE USE.

Q-081's non-degeneracy guard (claims.yaml, sharpened 2026-07-22 in REE_assembly
ca8e3d7fc8) states the requirement this file discharges:

    "VALIDATE THE NULL BEFORE USING IT: the surrogate ensemble must be shown to kill a
    deliberately-artefactual statistic (one computed purely from the update periods)
    while sparing a deliberately-injected real one, and that demonstration must be made
    BEFORE the surrogate adjudicates anything. Surrogate ensembles are cheap in
    simulation, so there is no cost argument for skipping this."

Shipped as a contract rather than a notebook precisely because "before it adjudicates
anything" has to keep being true as the substrate moves under it.

The four load-bearing tests, in the order they matter:

  1. KILL  -- test_null_kills_the_artefactual_statistic. A statistic computed purely from
     the configured update periods is bit-identical on every ensemble member (p = 1.0),
     and screen_statistic rules it out a priori.
  2. SPARE -- test_null_spares_an_injected_real_relation. A deliberately injected
     cross-stream coupling is detected at p <= 0.05 on every seed.
  3. The reason the null had to be sharpened -- test_block_surrogate_rejects_the_
     autocorrelation_false_positive and test_naive_shuffle_produces_the_false_positive.
     Two streams that are each strongly autocorrelated but PROVABLY UNCOUPLED (built from
     independent noise) are correctly non-significant under the block surrogate, and
     spuriously significant under the fresh-only shuffle that the block permutation
     replaces. This is the FALSE Outcome A the guard was rewritten to prevent, exhibited
     rather than asserted.
  4. The three preservation properties (a) tick times, (b) marginal, (c) within-stream
     autocorrelation -- each tested separately, because it is (c) alone that the
     superseded shuffle violated.

The synthetic streams are built here rather than driven from a live agent on purpose:
validating a null is a statement about the CONSTRUCTION, and it needs ground truth about
whether a real relation exists, which no agent trace can supply. Rates 1 / 3 / 10 mirror
SD-006 (E1 / E2 / E3).

ASCII-only output (repo rule).
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments._lib.q081_surrogate import (
    HOLD_CARRY,
    HOLD_FILLER,
    HOLD_NONE,
    HOLD_UNSTRUCTURED,
    SurrogateDesignError,
    artefactual_rate_statistic,
    autocorrelation_time,
    block_permute_stream,
    classify_hold_mode,
    cross_stream_xcorr,
    estimate_tick_period,
    lag_control_report,
    make_surrogate,
    plan_blocks,
    screen_statistic,
    surrogate_p_value,
    evaluate_against_null,
)

# Ensemble size: 99 gives a minimum attainable p of 0.01, enough resolution for a 0.05
# decision while keeping the whole file to a few seconds.
N_SURROGATES = 99
SEEDS = (0, 1, 2, 3)
ALPHA = 0.05


# --------------------------------------------------------------------------
# Synthetic multi-rate streams with KNOWN ground truth
# --------------------------------------------------------------------------


def _ar1(n, phi, rng):
    """AR(1) with unit stationary variance -- a stream that is smooth in time."""
    x = np.zeros(n)
    e = rng.standard_normal(n) * np.sqrt(1.0 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def _held_stream(tick_values, n_steps, period):
    """Expand per-tick values onto the step grid with carry-forward holds."""
    values = np.zeros(n_steps, dtype=np.float64)
    fresh = np.zeros(n_steps, dtype=bool)
    last = 0.0
    for i in range(n_steps):
        if i % period == 0:
            last = float(tick_values[i // period])
            fresh[i] = True
        values[i] = last
    return values, fresh


def make_streams(n_steps=4000, period_b=10, coupling=0.0, lag_steps=20,
                 phi_a=0.98, phi_b=0.9, seed=0):
    """Stream A at 1 step/tick, stream B at `period_b` steps/tick.

    `coupling` is the ground truth: 0.0 means B is built from noise INDEPENDENT of A, so
    any significant cross-stream statistic is a false positive by construction. Both
    streams are strongly autocorrelated either way -- that is the trap the block
    permutation exists to survive.
    """
    rng = np.random.default_rng(seed)
    a = _ar1(n_steps, phi_a, rng)
    fresh_a = np.ones(n_steps, dtype=bool)

    ticks = np.arange(0, n_steps, period_b)
    own = _ar1(ticks.size, phi_b, rng)
    driver = np.array([a[max(0, t - lag_steps)] for t in ticks])
    driver = (driver - driver.mean()) / (driver.std() + 1e-12)
    tick_values = coupling * driver + np.sqrt(max(0.0, 1.0 - coupling ** 2)) * own
    b, fresh_b = _held_stream(tick_values, n_steps, period_b)

    return {
        "A": a.reshape(-1, 1).astype(np.float32), "A__fresh": fresh_a,
        "B": b.reshape(-1, 1).astype(np.float32), "B__fresh": fresh_b,
    }


def _statistic(arrays):
    """The reference cross-stream statistic. Lag is discarded here -- it is a control."""
    return cross_stream_xcorr(arrays, "A", "B", max_lag_ticks=8)[0]


def _artefact(arrays):
    return artefactual_rate_statistic(arrays, "A", "B")


def _naive_fresh_shuffle_p(arrays, n=N_SURROGATES, seed=0):
    """The SUPERSEDED control: permute fresh samples individually, then carry.

    This is `stream_recorder.rate_matched_shuffle_index` in spirit -- it preserves tick
    times and the marginal but destroys within-stream autocorrelation. Reimplemented
    locally so this contract does not depend on that function surviving unchanged.
    """
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        member = dict(arrays)
        for name in ("A", "B"):
            v = np.asarray(arrays[name]).copy()
            f = np.asarray(arrays[f"{name}__fresh"], dtype=bool)
            pos = np.flatnonzero(f)
            v[pos] = v[pos][rng.permutation(pos.size)]
            src = np.where(f, np.arange(v.shape[0]), -1)
            np.maximum.accumulate(src, out=src)
            filled = src >= 0
            v[filled] = v[src[filled]]
            member[name] = v
        values.append(_statistic(member))
    return surrogate_p_value(_statistic(arrays), values)


# --------------------------------------------------------------------------
# 1. KILL the artefactual statistic
# --------------------------------------------------------------------------


def test_null_kills_the_artefactual_statistic():
    """A statistic that is a pure function of the update periods must not survive.

    The surrogate preserves every stream's freshness flags exactly, so such a statistic
    is bit-identical on every ensemble member. Asserted as exact equality rather than
    only via the p-value: the p-value being 1.0 is the consequence, the invariance is
    the mechanism, and the mechanism is what a future edit could break.
    """
    arrays = make_streams(seed=0)
    plan = plan_blocks(arrays, ["A", "B"])
    observed = _artefact(arrays)
    assert np.isfinite(observed)

    rng = np.random.default_rng(0)
    for _ in range(16):
        member = make_surrogate(arrays, ["A", "B"], plan, rng)
        assert _artefact(member) == observed, (
            "the artefactual rate statistic changed under the surrogate, which means the "
            "surrogate altered a stream's freshness flags -- tick times are a PRESERVED "
            "property and must pass through untouched")

    result = evaluate_against_null(arrays, ["A", "B"], _artefact,
                               n_surrogates=N_SURROGATES, seed=0, plan=plan)
    assert result["p_value"] == 1.0
    assert result["surrogate_n_distinct"] == 1


def test_screen_statistic_rules_out_the_rate_derived_statistic_a_priori():
    """Q-081's a-priori filter, applied mechanically rather than by inspection."""
    arrays = make_streams(seed=0)
    plan = plan_blocks(arrays, ["A", "B"])

    verdict = screen_statistic(arrays, ["A", "B"], _artefact, n_surrogates=16, plan=plan)
    assert verdict["verdict"] == "ruled_out"
    assert verdict["ensemble_n_distinct"] == 1

    admissible = screen_statistic(arrays, ["A", "B"], _statistic, n_surrogates=16, plan=plan)
    assert admissible["verdict"] == "admissible"
    assert admissible["ensemble_spread"] > 0.0


# --------------------------------------------------------------------------
# 2. SPARE an injected real relation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_null_spares_an_injected_real_relation(seed):
    """A deliberately injected cross-stream coupling must clear the null."""
    arrays = make_streams(coupling=0.5, lag_steps=20, seed=seed)
    result = evaluate_against_null(arrays, ["A", "B"], _statistic,
                               n_surrogates=N_SURROGATES, seed=100 + seed)
    assert result["p_value"] <= ALPHA, (
        "the surrogate killed a REAL injected cross-stream relation (p=%.3f). A null "
        "this conservative cannot adjudicate Q-081 -- it would report Outcome C for a "
        "system that genuinely coordinates." % result["p_value"])
    assert result["observed"] > result["surrogate_mean"]


# --------------------------------------------------------------------------
# 3. Why the null had to be sharpened: the autocorrelation false positive
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_block_surrogate_rejects_the_autocorrelation_false_positive(seed):
    """Two strongly autocorrelated but PROVABLY UNCOUPLED streams must not be significant.

    coupling=0.0 means B is built from noise independent of A, so there is no relation to
    find. A null that fires here is manufacturing Outcome A out of within-stream
    smoothness.
    """
    arrays = make_streams(coupling=0.0, seed=seed)
    result = evaluate_against_null(arrays, ["A", "B"], _statistic,
                               n_surrogates=N_SURROGATES, seed=100 + seed)
    assert result["p_value"] > ALPHA, (
        "FALSE Outcome A: the block surrogate reported p=%.3f on streams built from "
        "independent noise." % result["p_value"])


def test_naive_shuffle_produces_the_false_positive():
    """The superseded fresh-only shuffle fails the case above -- exhibited, not asserted.

    This is the evidence for the guard's sharpening, and the reason
    `stream_recorder.rate_matched_shuffle_index` must not be used to adjudicate Q-081.
    A majority of independent seeds is required to fire so the demonstration is about the
    construction rather than about one lucky draw.
    """
    naive_fp = 0
    block_fp = 0
    for seed in range(8):
        arrays = make_streams(coupling=0.0, seed=seed)
        if _naive_fresh_shuffle_p(arrays, seed=100 + seed) <= ALPHA:
            naive_fp += 1
        if evaluate_against_null(arrays, ["A", "B"], _statistic,
                             n_surrogates=N_SURROGATES, seed=100 + seed)["p_value"] <= ALPHA:
            block_fp += 1

    assert naive_fp >= 4, (
        "the fresh-only shuffle was expected to false-positive on a majority of seeds "
        "(it destroys within-stream autocorrelation); it fired on %d/8. If this stops "
        "reproducing, the demonstration underpinning Q-081's sharpened guard has changed "
        "and the guard needs re-deriving, not this threshold loosening." % naive_fp)
    assert block_fp == 0, (
        "the block surrogate false-positived on %d/8 uncoupled seeds" % block_fp)
    assert block_fp < naive_fp


# --------------------------------------------------------------------------
# 4. The three preserved properties
# --------------------------------------------------------------------------


def test_surrogate_preserves_tick_times_exactly():
    """(a) tick times / configured update period."""
    arrays = make_streams(seed=0)
    plan = plan_blocks(arrays, ["A", "B"])
    rng = np.random.default_rng(0)
    for _ in range(8):
        member = make_surrogate(arrays, ["A", "B"], plan, rng)
        for name in ("A", "B"):
            assert np.array_equal(member[f"{name}__fresh"], arrays[f"{name}__fresh"])
            assert estimate_tick_period(member[f"{name}__fresh"]) == \
                estimate_tick_period(arrays[f"{name}__fresh"])


def test_surrogate_preserves_the_marginal_exactly():
    """(b) marginal distribution of the stream's values.

    Checked on the FRESH subsequence, which the permutation reorders and therefore
    preserves exactly. The full-series marginal is exact only when hold run-lengths are
    constant; where they are not (MECH-091 phase reset), the fresh-sample marginal is the
    honest statement of what is preserved.
    """
    arrays = make_streams(seed=0)
    plan = plan_blocks(arrays, ["A", "B"])
    rng = np.random.default_rng(0)
    for _ in range(8):
        member = make_surrogate(arrays, ["A", "B"], plan, rng)
        for name in ("A", "B"):
            f = arrays[f"{name}__fresh"]
            before = np.sort(np.asarray(arrays[name])[f], axis=0)
            after = np.sort(np.asarray(member[name])[f], axis=0)
            assert np.array_equal(before, after)


def test_surrogate_preserves_within_stream_autocorrelation():
    """(c) the property the superseded shuffle destroyed, and the reason for this module.

    The block surrogate must retain most of each stream's correlation time; the fresh-only
    shuffle must collapse it. Tolerances are deliberately loose -- the claim is a large
    qualitative gap, not a calibrated number.
    """
    arrays = make_streams(seed=0)
    plan = plan_blocks(arrays, ["A", "B"])
    rng = np.random.default_rng(0)

    for name in ("A", "B"):
        tau_data = autocorrelation_time(arrays[name], arrays[f"{name}__fresh"])
        assert tau_data > 2.0, "the fixture must be autocorrelated for this test to mean anything"

        member = make_surrogate(arrays, ["A", "B"], plan, rng)
        tau_block = autocorrelation_time(member[name], member[f"{name}__fresh"])

        v = np.asarray(arrays[name]).copy()
        f = np.asarray(arrays[f"{name}__fresh"], dtype=bool)
        pos = np.flatnonzero(f)
        v[pos] = v[pos][np.random.default_rng(1).permutation(pos.size)]
        tau_shuffle = autocorrelation_time(v, f)

        assert tau_block >= 0.5 * tau_data, (
            "block surrogate lost the within-stream correlation time of '%s' "
            "(%.1f -> %.1f ticks)" % (name, tau_data, tau_block))
        assert tau_shuffle <= 0.25 * tau_data, (
            "the fresh-only shuffle was expected to destroy '%s' autocorrelation "
            "(%.1f -> %.1f ticks)" % (name, tau_data, tau_shuffle))
        assert tau_block > tau_shuffle


# --------------------------------------------------------------------------
# The unequal-rate design: blocks must be commensurate across 1 / 3 / 10
# --------------------------------------------------------------------------


def test_block_lengths_are_commensurate_across_e1_e2_e3_rates():
    """The unequal-rate rule: ONE block duration in STEPS, per-stream lengths in TICKS.

    With E1/E2/E3 at 1/3/10 steps, block lengths in ticks must differ by the rate ratios
    so that block DURATIONS match. A single block length in ticks applied to all three --
    the natural mistake, and the one the claim flags as hardest to notice -- would give
    the E3 stream blocks ten times longer in wall-clock than the E1 stream's.
    """
    n_steps = 4000
    rng = np.random.default_rng(0)
    arrays = {}
    for name, period, phi in (("E1", 1, 0.98), ("E2", 3, 0.95), ("E3", 10, 0.9)):
        ticks = np.arange(0, n_steps, period)
        values, fresh = _held_stream(_ar1(ticks.size, phi, rng), n_steps, period)
        arrays[name] = values.reshape(-1, 1).astype(np.float32)
        arrays[f"{name}__fresh"] = fresh

    plan = plan_blocks(arrays, ["E1", "E2", "E3"])
    W = plan.block_duration_steps

    for name, period in (("E1", 1), ("E2", 3), ("E3", 10)):
        info = plan.streams[name]
        assert info["period_steps"] == float(period), "period must be measured from the data"
        # Block duration in steps is within one tick of the common W.
        assert abs(info["block_len_ticks"] * period - W) <= period
        assert info["block_len_ticks"] >= 1

    # The payoff of measuring W in steps: block COUNT is equalised across streams, so the
    # ensemble is equally rich for the fine and the coarse stream.
    counts = [plan.streams[n]["n_blocks"] for n in ("E1", "E2", "E3")]
    assert max(counts) - min(counts) <= 2, "block counts diverged across rates: %s" % counts

    # And the lower bound actually binds: W covers the slowest stream's correlation time.
    assert W >= plan.safety_factor * max(
        plan.streams[n]["tau_steps"] for n in ("E1", "E2", "E3")) - 1.0


def test_plan_refuses_a_run_too_short_for_its_own_autocorrelation():
    """Refusal, not a silent downgrade.

    Shortening the blocks to fit a short run would break the within-stream autocorrelation
    the null exists to preserve, and nothing downstream could detect it.
    """
    arrays = make_streams(n_steps=300, phi_a=0.995, phi_b=0.98, seed=0)
    with pytest.raises(SurrogateDesignError, match="run too short"):
        plan_blocks(arrays, ["A", "B"])


def test_periods_are_measured_from_data_not_assumed_from_config():
    """A phase-shifted / irregular tick grid must still be read correctly.

    MECH-091 resets the E3 phase and MECH-093 modulates the rate, so a nominal
    `step % 10` is wrong on real traces.
    """
    n_steps = 1200
    fresh = np.zeros(n_steps, dtype=bool)
    pos = list(range(0, 400, 10)) + list(range(407, 1200, 10))  # phase reset at 400
    fresh[pos] = True
    assert estimate_tick_period(fresh) == 10.0


# --------------------------------------------------------------------------
# Hold semantics
# --------------------------------------------------------------------------


def test_hold_mode_classification_covers_the_recorder_semantics():
    n = 60
    fresh = np.zeros(n, dtype=bool)
    fresh[::10] = True

    carry, _ = _held_stream(np.arange(6, dtype=float) + 1.0, n, 10)
    assert classify_hold_mode(carry.reshape(-1, 1), fresh) == HOLD_CARRY

    # Event-stream semantics: a non-event step records a constant filler row.
    filler = np.tile(np.array([0.0, np.nan]), (n, 1))
    filler[fresh] = np.array([2.0, 0.7])
    assert classify_hold_mode(filler, fresh) == HOLD_FILLER

    assert classify_hold_mode(np.arange(n, dtype=float).reshape(-1, 1),
                              np.ones(n, dtype=bool)) == HOLD_NONE

    unstructured = np.random.default_rng(0).standard_normal((n, 1))
    assert classify_hold_mode(unstructured, fresh) == HOLD_UNSTRUCTURED


def test_permutation_refuses_an_unstructured_hold_pattern():
    n = 60
    fresh = np.zeros(n, dtype=bool)
    fresh[::10] = True
    unstructured = np.random.default_rng(0).standard_normal((n, 1))
    with pytest.raises(SurrogateDesignError, match="carry-forward|filler|Refusing"):
        block_permute_stream(unstructured, fresh, 2, np.random.default_rng(0),
                             hold_mode=HOLD_UNSTRUCTURED)


def test_filler_holds_are_not_carried_forward():
    """A filler hold must stay filler: carrying it would invent events that never fired."""
    n = 60
    fresh = np.zeros(n, dtype=bool)
    fresh[::10] = True
    values = np.tile(np.array([0.0, 0.0]), (n, 1))
    values[fresh] = np.array([[float(i + 1), 0.5] for i in range(int(fresh.sum()))])

    out = block_permute_stream(values, fresh, 2, np.random.default_rng(0),
                               hold_mode=HOLD_FILLER)
    assert np.array_equal(out[~fresh], values[~fresh])
    assert np.array_equal(np.sort(out[fresh], axis=0), np.sort(values[fresh], axis=0))


# --------------------------------------------------------------------------
# Lag is a CONTROL quantity
# --------------------------------------------------------------------------


def test_lag_is_reported_as_a_control_not_a_readout():
    """Q-081 records cross-stream lag as an Outcome-B (clock) detector.

    The contract is on the REPORTING SHAPE: the report must label itself a control and
    must expose the lag the scheduler alone accounts for, so a reader can see whether the
    measured lag is anything beyond the rate offset.
    """
    arrays = make_streams(coupling=0.5, lag_steps=20, seed=0)
    report = lag_control_report(arrays, "A", "B")

    assert report["role"] == "control"
    assert "Outcome-B" in report["interpretation"]
    assert "NOT evidence" in report["interpretation"]
    assert report["scheduler_expected_lag_steps"] == 9.0  # periods 1 and 10
    assert np.isfinite(report["statistic_at_argmax"])
    assert isinstance(report["lag_steps"], int)


# --------------------------------------------------------------------------
# The result carries its own null description
# --------------------------------------------------------------------------


def test_result_carries_the_block_plan_so_a_p_value_is_interpretable_later():
    arrays = make_streams(seed=0)
    result = evaluate_against_null(arrays, ["A", "B"], _statistic, n_surrogates=32, seed=0)
    plan = result["plan"]
    assert plan["construction"] == "block_permutation_within_own_tick_grid"
    assert set(plan["preserved"]) == {
        "tick_times", "marginal_of_fresh_samples", "within_stream_autocorrelation"}
    assert plan["destroyed"] == ["between_stream_relation"]
    for name in ("A", "B"):
        info = plan["streams"][name]
        for key in ("period_steps", "tau_ticks", "tau_steps", "block_len_ticks",
                    "n_blocks", "hold_mode"):
            assert key in info
