"""Contracts for SD-076b, the OU log-multiplier waking drift source (2026-09-24).

Surface under test: `E3TrajectorySelector._advance_wci_ou` / `_wci_ou_normal` and the
`waking_confidence_drift_source` branch of `update_running_variance`.

WHY THIS MECHANISM EXISTS -- and therefore what these contracts must actually pin.
V3-EXQ-794a (FAIL, 2026-07-24) tested the only drift source the substrate had, the
asymmetric EMA, and C1 (absolute overconfidence) failed at BOTH dose levels. The reason is
structural, not a tuning miss: the asymmetric EMA is a conditional-gain modification of the
same estimator on the same realised squared-error stream, so its fixed point is the
EXPECTILE of that stream's distribution at tau = (1 - asym) / 2, and an expectile's
displacement from the mean is proportional to the stream's DISPERSION. Measured in the
manifest: halving tau (asymmetry 0.6 -> 0.8) moved mean_rv/true_error_ref only 1.043 ->
1.023, i.e. 1.9%, against a C1 bar needing 10.5%.

So the property that matters -- the one the whole build is FOR -- is DISPERSION
INDEPENDENCE, and `test_displacement_is_independent_of_pe_dispersion` is the load-bearing
contract in this file. It is written as a differential test against the old mechanism on
the same two streams, so it cannot pass vacuously: the OLD source must show the dependence
and the NEW one must not. Measuring that blind spot is the point -- a contract that only
asserted "the OU source displaces rv" would have passed for the asymmetric EMA too, and
would therefore have certified the exact failure 794a already ran into.

Design record: REE_assembly/evidence/planning/
mech204_waking_drift_source_candidates_staged_20260924.md (candidates A/B/C, why B).
User ratification 2026-09-24: (B) OU log-multiplier with a sigma = 0 control arm.
"""
import math
import statistics
import sys
from collections import deque
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

from ree_core.predictors.e3_selector import E3TrajectorySelector  # noqa: E402
from ree_core.utils.config import E3Config  # noqa: E402

# V3-EXQ-794a's measured operating point.
TRUE_ERR = 0.0037
# The ratified dose ladder (arithmetic-mean multipliers), and the C1 bar it must clear.
G_LO, G_HI = 0.80, 0.65
C1_MARGIN = 0.10          # overconfidence_score = log(true_err / mean_rv) must exceed this
THETA = 0.02
SIGMA = 0.05
SEED = 20260924


def stationary_var(sigma=SIGMA, theta=THETA):
    """Closed form for Var[u] of u_t <- (1-theta) u_{t-1} + theta m + sigma xi_t.

    AR(1) with coefficient (1 - theta): Var = sigma^2 / (1 - (1-theta)^2)
                                            = sigma^2 / (theta * (2 - theta)).
    """
    return sigma * sigma / (theta * (2.0 - theta))


def mean_log_gain_for(g, sigma=SIGMA, theta=THETA):
    """m such that the ARITHMETIC-MEAN multiplier E[exp(u)] equals g.

    E[exp(u)] = exp(m + Var/2) for Gaussian u, so m = log(g) - Var/2. At sigma = 0 this
    reduces to log(g) exactly. Conflating the median multiplier exp(m) with the arithmetic
    mean is the easy error here, and the C1 DV is a function of the MEAN rv.
    """
    return math.log(g) - 0.5 * stationary_var(sigma, theta)


# CALIBRATED SYNTHETIC DISPERSION. The asymmetric EMA's displacement is a function of
# the error stream's dispersion (that is the whole finding), so a contract asserting it
# falls short of C1 is meaningless unless the stream is at the SUBSTRATE's dispersion. On
# a wide synthetic stream the old form clears C1 easily -- measured here: spread 1.6 gives
# score +0.53. Calibration target is V3-EXQ-794a's measured LO->HI dose separation, the
# one dispersion-sensitive quantity the manifest reports: 1.0428 -> 1.0234, i.e. 1.9%.
# Measured on this generator: spread 0.20 gives 0.9668 -> 0.9503, i.e. 1.7%. Nearest
# match, so SPREAD_794A = 0.20 is the substrate-calibrated stream and every C1 assertion
# runs on it.
SPREAD_794A = 0.20
SPREAD_WIDE = 1.8


def _err_seq(n, seed=0, spread=SPREAD_794A):
    """Deterministic error sequence with mean TRUE_ERR and a tunable dispersion.

    `spread` controls the width only; the mean is held at TRUE_ERR for every spread, so a
    dispersion comparison is not confounded by a level change.
    """
    import random

    r = random.Random(1234 + seed)
    lo = 1.0 - spread / 2.0
    return [TRUE_ERR * (lo + spread * r.random()) for _ in range(n)]


def _selector(**over):
    """A bare selector exercising only the running-variance path.

    `guard` picks the rv-floor configuration, and the choice is load-bearing:

    * "repaired" (DEFAULT) is the RATIFIED configuration -- scale-relative frac 0.2,
      softplus-saturating. Every science assertion runs here, because that is the
      configuration the validation run will use.
    * "off" removes the bound entirely (floor <= 0 short-circuits `_apply_wci_rv_floor`).
      Used only where the assertion is about the MULTIPLIER's exactness; the soft floor
      perturbs it by ~2e-5 at the doses, which is irrelevant to the science and would
      turn an exactness contract into a tolerance contract for no gain.

    The pre-2026-07-22 ABSOLUTE default (`waking_confidence_rv_floor` 0.01) is NOT
    available as a default here on purpose: at this substrate's error scale it clamps --
    see `test_the_legacy_absolute_floor_still_clamps_at_this_error_scale`.
    """
    cfg = E3Config()
    cfg.use_waking_confidence_inflation = over.pop("armed", True)
    guard = over.pop("guard", "repaired")
    if guard == "repaired":
        cfg.waking_confidence_rv_floor_relative_frac = 0.2
        cfg.waking_confidence_rv_floor_mode = "soft"
    elif guard == "off":
        cfg.waking_confidence_rv_floor = 0.0
        cfg.waking_confidence_rv_floor_relative_frac = 0.0
    elif guard != "legacy_absolute":
        raise AssertionError(f"unknown guard {guard!r}")
    for k, v in over.items():
        setattr(cfg, k, v)
    sel = E3TrajectorySelector.__new__(E3TrajectorySelector)
    sel.config = cfg
    sel._running_variance = cfg.precision_init
    sel._ema_alpha = cfg.precision_ema_alpha
    sel._wci_symmetric_rv_ref = cfg.precision_init
    sel._last_instantaneous_pe = 0.0
    sel._rv_history = deque(maxlen=50)
    sel._volatility_estimate = 0.0
    sel._wci_ou_log_gain = None
    sel._wci_ou_rng_obj = None
    sel._wci_ou_clamp_hits = 0
    return sel


def _drive(sel, errs):
    """Feed a squared-error sequence in; return the per-tick rv trace."""
    trace = []
    for e in errs:
        sel.update_running_variance(torch.tensor([math.sqrt(e)], dtype=torch.float64))
        trace.append(float(sel._running_variance))
    return trace


def _ou(g, sigma=SIGMA, theta=THETA, seed=SEED, **over):
    over.setdefault("waking_confidence_drift_source", "ou")
    over.setdefault("waking_confidence_ou_mean_log_gain",
                    mean_log_gain_for(g, sigma, theta))
    over.setdefault("waking_confidence_ou_sigma", sigma)
    over.setdefault("waking_confidence_ou_theta", theta)
    over.setdefault("waking_confidence_ou_seed", seed)
    return _selector(**over)


# ======================================================================================
# 1. Backward compatibility -- the new form must be invisible until selected
# ======================================================================================

def test_defaults_are_no_op_sentinels():
    cfg = E3Config()
    assert cfg.waking_confidence_drift_source == "asymmetric_ema"
    assert cfg.waking_confidence_ou_mean_log_gain == 0.0
    assert cfg.waking_confidence_ou_sigma == 0.0
    # theta / seed are UNSET sentinels on purpose (the ARC-029 precedent): both decide
    # what a run measures, so neither may carry a science-bearing default.
    assert cfg.waking_confidence_ou_theta == -1.0
    assert cfg.waking_confidence_ou_seed == -1


def test_off_path_is_bit_identical_to_the_plain_symmetric_ema():
    """Master flag OFF: rv must equal the original expression to the last bit."""
    errs = _err_seq(2000)
    got = _drive(_selector(armed=False), errs)
    cfg = E3Config()
    rv, a0 = cfg.precision_init, cfg.precision_ema_alpha
    want = []
    for e0 in errs:
        e = float(torch.tensor([math.sqrt(e0)], dtype=torch.float64).pow(2).mean().item())
        rv = (1 - a0) * rv + a0 * e
        want.append(rv)
    assert got == want


def test_asymmetric_ema_path_is_bit_identical_after_the_selector_was_added():
    """Adding the drift-source branch must not perturb the EXISTING inflation path.

    Reference arithmetic is written out here independently of the substrate, in the same
    order the pre-2026-09-24 code evaluated it.
    """
    errs = _err_seq(2000, seed=3)
    sel = _selector(waking_confidence_inflation_asymmetry=0.8,
                    waking_confidence_rv_floor_relative_frac=0.2,
                    waking_confidence_rv_floor_mode="soft")
    got = _drive(sel, errs)

    cfg = E3Config()
    rv, ref, a0 = cfg.precision_init, cfg.precision_init, cfg.precision_ema_alpha
    asy, frac, soft = 0.8, 0.2, 0.25
    want = []
    for e0 in errs:
        e = float(torch.tensor([math.sqrt(e0)], dtype=torch.float64).pow(2).mean().item())
        alpha = min(1.0, a0 * (1.0 + asy)) if e < rv else max(0.0, a0 * (1.0 - asy))
        rv_new = (1 - alpha) * rv + alpha * e
        ref = (1 - a0) * ref + a0 * e
        floor = frac * ref
        knee = soft * floor
        z = (rv_new - floor) / knee
        rv = floor + knee * (max(z, 0.0) + math.log1p(math.exp(-abs(z))))
        want.append(rv)
    assert got == want


def test_ou_source_at_zero_gain_and_zero_sigma_tracks_the_symmetric_reference():
    """m = 0, sigma = 0 is an exact no-op multiplier: rv must equal the reference."""
    sel = _ou(g=1.0, sigma=0.0, waking_confidence_ou_seed=-1, guard="off")
    _drive(sel, _err_seq(1500))
    assert sel._running_variance == pytest.approx(sel._wci_symmetric_rv_ref, rel=1e-12)


# ======================================================================================
# 2. The dose is exactly what config says -- the property the expectile form lacks
# ======================================================================================

@pytest.mark.parametrize("g", [G_LO, G_HI, 0.9, 0.5])
def test_sigma_zero_control_arm_is_an_exact_constant_discount(g):
    """sigma = 0 collapses to candidate A, the deterministic multiplicative discount.

    This IS the ratified control arm, so its exactness is a contract, not a detail.
    """
    sel = _ou(g=g, sigma=0.0, waking_confidence_ou_seed=-1, guard="off")
    _drive(sel, _err_seq(3000))
    ratio = sel._running_variance / sel._wci_symmetric_rv_ref
    assert ratio == pytest.approx(g, rel=1e-9)


def test_displacement_is_independent_of_pe_dispersion():
    """THE load-bearing contract: the new source escapes the expectile bound.

    Differential against the OLD source on the SAME two streams. A narrow and a wide
    error stream with the same mean:
      - asymmetric EMA: displacement must MOVE with dispersion (that is the 794a defect);
      - OU:             displacement must NOT (that is why this build exists).
    Written this way so it cannot pass vacuously -- see the module docstring.
    """
    narrow = _err_seq(6000, seed=11, spread=SPREAD_794A)
    wide = _err_seq(6000, seed=11, spread=SPREAD_WIDE)

    def displacement(make):
        out = []
        for errs in (narrow, wide):
            sel = make()
            _drive(sel, errs)
            out.append(1.0 - sel._running_variance / sel._wci_symmetric_rv_ref)
        return out

    old_narrow, old_wide = displacement(
        lambda: _selector(waking_confidence_inflation_asymmetry=0.8, guard="off")
    )
    new_narrow, new_wide = displacement(lambda: _ou(g=G_HI, sigma=0.0, guard="off",
                                                    waking_confidence_ou_seed=-1))

    # The old form's displacement is dispersion-bound: widening the stream 9x must move
    # it by more than a factor of two. (Measured here: 0.0587 -> 0.4769, ~8x.)
    assert old_wide > 2.0 * old_narrow, (old_narrow, old_wide)
    # The new form's displacement is set by config and must be flat across the same pair.
    assert new_narrow == pytest.approx(new_wide, rel=1e-6), (new_narrow, new_wide)
    assert new_narrow == pytest.approx(1.0 - G_HI, rel=1e-6)


def test_the_old_form_cannot_reach_the_c1_bar_and_the_new_one_can():
    """Pins the 794a diagnosis and the ratified dose ladder in one assertion pair.

    C1: overconfidence_score = log(true_error_ref / mean_rv) > 0.10.
    """
    errs = _err_seq(8000, seed=5)
    half = len(errs) // 2
    # true_err is measured over the SAME window as mean_rv (the 794a driver's own
    # convention); using the full-sequence mean against a tail mean_rv would fold the EMA
    # warm-in into the score and inflate it by ~0.04.
    true_err = statistics.fmean(errs[half:])

    def score(sel):
        trace = _drive(sel, errs)
        return math.log(true_err / statistics.fmean(trace[half:]))

    old = score(_selector(waking_confidence_inflation_asymmetry=0.8))
    assert old < C1_MARGIN, f"asymmetric EMA unexpectedly cleared C1 ({old:+.4f})"

    for g in (G_LO, G_HI):
        got = score(_ou(g=g))
        assert got > C1_MARGIN, f"g={g} failed C1 ({got:+.4f})"
        # Tolerance is set by the OU's own autocorrelation, not by slack: theta = 0.02
        # gives a ~50-tick correlation time, so a 4000-tick window holds ~80 effective
        # samples of exp(u) and the score's sampling sd is ~0.03. Measured across four
        # RNG seeds: +0.193 .. +0.266 at g = 0.80. That sd is a REAL constraint on the
        # validation run, carried into its readiness criteria as eval_ticks * theta >= 40.
        assert got == pytest.approx(-math.log(g), abs=0.08), (g, got)


# ======================================================================================
# 3. The OU process itself -- closed form, and a live stream
# ======================================================================================

def test_stationary_mean_and_variance_match_closed_form():
    """u ~ Normal(m, sigma^2 / (theta * (2 - theta))), measured against the formula."""
    m = mean_log_gain_for(G_HI)
    sel = _ou(g=G_HI)
    us = []
    for e in _err_seq(60000, seed=2):
        sel.update_running_variance(torch.tensor([math.sqrt(e)], dtype=torch.float64))
        us.append(sel.wci_ou_log_gain)
    tail = us[5000:]  # past any transient; u_0 is already at the mean, so this is cheap
    assert statistics.fmean(tail) == pytest.approx(m, abs=0.02)
    # rel is set by the estimator, not by slack: the samples are autocorrelated over
    # ~1/theta = 50 ticks, so 55000 draws are ~1100 effective ones and the variance
    # estimate's own sd is ~4%. 0.20 is ~5 of those.
    assert statistics.pvariance(tail) == pytest.approx(stationary_var(), rel=0.20)


def test_arithmetic_mean_multiplier_matches_exp_m_plus_half_var():
    """The mean/median distinction is load-bearing: the C1 DV reads the MEAN."""
    sel = _ou(g=G_HI)
    mults = []
    for e in _err_seq(60000, seed=4):
        sel.update_running_variance(torch.tensor([math.sqrt(e)], dtype=torch.float64))
        mults.append(math.exp(sel.wci_ou_log_gain))
    tail = mults[5000:]
    assert statistics.fmean(tail) == pytest.approx(G_HI, rel=0.05)
    assert statistics.median(tail) == pytest.approx(
        math.exp(mean_log_gain_for(G_HI)), rel=0.05
    )


def test_u_starts_at_the_stationary_mean_so_there_is_no_warmup_transient():
    sel = _ou(g=G_LO, sigma=0.0, waking_confidence_ou_seed=-1)
    assert sel.wci_ou_log_gain == pytest.approx(mean_log_gain_for(G_LO, 0.0), abs=1e-12)
    _drive(sel, _err_seq(5))
    assert sel.wci_ou_log_gain == pytest.approx(mean_log_gain_for(G_LO, 0.0), abs=1e-12)


def test_different_seeds_give_different_trajectories():
    """Negative-instrument check: a dead RNG would make every 'stochastic' arm identical."""
    a = _ou(g=G_HI, seed=1)
    b = _ou(g=G_HI, seed=2)
    ta, tb = _drive(a, _err_seq(500)), _drive(b, _err_seq(500))
    assert ta != tb
    # ... and the same seed must reproduce exactly.
    c = _ou(g=G_HI, seed=1)
    assert _drive(c, _err_seq(500)) == ta


def test_ou_stream_does_not_perturb_the_global_torch_rng():
    """A shared stream would destroy bit-identity of every unrelated arm."""
    torch.manual_seed(7)
    _drive(_ou(g=G_HI), _err_seq(2000))
    after = torch.rand(4, dtype=torch.float64).tolist()
    torch.manual_seed(7)
    baseline = torch.rand(4, dtype=torch.float64).tolist()
    assert after == baseline


# ======================================================================================
# 4. Guards -- unset sentinels raise, the clamp is counted, the floor still bounds
# ======================================================================================

def test_unset_theta_raises():
    sel = _selector(waking_confidence_drift_source="ou",
                    waking_confidence_ou_mean_log_gain=-0.2,
                    waking_confidence_ou_sigma=0.0)
    with pytest.raises(ValueError, match="waking_confidence_ou_theta"):
        _drive(sel, _err_seq(3))


def test_unset_seed_raises_only_when_sigma_is_positive():
    armed = _selector(waking_confidence_drift_source="ou",
                      waking_confidence_ou_theta=THETA,
                      waking_confidence_ou_sigma=SIGMA)
    with pytest.raises(ValueError, match="waking_confidence_ou_seed"):
        _drive(armed, _err_seq(3))
    # sigma = 0 draws nothing, so an unset seed is not an error there.
    quiet = _selector(waking_confidence_drift_source="ou",
                      waking_confidence_ou_theta=THETA,
                      waking_confidence_ou_sigma=0.0)
    _drive(quiet, _err_seq(3))


def test_unknown_drift_source_raises_rather_than_falling_back():
    """A silent fallback would look exactly like a mechanism that ran and did nothing."""
    sel = _selector(waking_confidence_drift_source="asymetric_ema")  # typo on purpose
    with pytest.raises(ValueError, match="waking_confidence_drift_source"):
        _drive(sel, _err_seq(3))


def test_clamp_is_not_reached_at_the_ratified_doses():
    """A silently-saturating lever is the V3-EXQ-794 rv-floor failure shape."""
    for g in (G_LO, G_HI):
        sel = _ou(g=g)
        _drive(sel, _err_seq(20000, seed=6))
        assert sel.wci_ou_clamp_hits == 0


def test_clamp_counts_when_it_does_bind():
    """The counter must be live, or its 0 above would be meaningless."""
    sel = _ou(g=G_HI, waking_confidence_ou_log_gain_clamp=0.001)
    _drive(sel, _err_seq(200))
    assert sel.wci_ou_clamp_hits > 0


def test_rv_floor_still_bounds_the_ou_path_but_does_not_bind_at_the_doses():
    deep = _ou(g=0.01, sigma=0.0, waking_confidence_ou_seed=-1,
               waking_confidence_rv_floor_relative_frac=0.2,
               waking_confidence_rv_floor_mode="hard")
    _drive(deep, _err_seq(3000))
    assert deep._running_variance == pytest.approx(
        0.2 * deep._wci_symmetric_rv_ref, rel=1e-9
    )
    for g in (G_LO, G_HI):
        sel = _ou(g=g, waking_confidence_rv_floor_relative_frac=0.2,
                  waking_confidence_rv_floor_mode="hard")
        _drive(sel, _err_seq(3000, seed=8))
        assert sel._running_variance > 0.2 * sel._wci_symmetric_rv_ref * 1.5


def test_the_legacy_absolute_floor_still_clamps_at_this_error_scale():
    """Why every science assertion above runs with the REPAIRED guard, pinned.

    `waking_confidence_rv_floor` still DEFAULTS to the pre-repair absolute 0.01, and the
    substrate's true error reference is ~0.0037, so the default bound sits ~2.7x ABOVE the
    operating point and pins rv regardless of the drift source. That is the V3-EXQ-794
    saturation defect, and it is reproduced here rather than assumed -- it caught this
    file's own first draft, which used the default and read as an OU source that failed C1.
    """
    sel = _ou(g=G_HI, sigma=0.0, waking_confidence_ou_seed=-1, guard="legacy_absolute")
    _drive(sel, _err_seq(3000))
    assert sel._running_variance == pytest.approx(0.01, rel=1e-9)
    # ... and the repaired guard at the same dose does NOT clamp.
    ok = _ou(g=G_HI, sigma=0.0, waking_confidence_ou_seed=-1)
    _drive(ok, _err_seq(3000))
    assert ok._running_variance < 0.01
    assert ok._running_variance / ok._wci_symmetric_rv_ref == pytest.approx(G_HI, rel=1e-3)
