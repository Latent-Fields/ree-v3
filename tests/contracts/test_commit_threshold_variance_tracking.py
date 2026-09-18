"""ARC-029 (D): the variance-TRACKING commitment bar.

WHAT THIS PINS. `committed` is `commit_variance < effective_threshold`, with
`commit_variance == E3TrajectorySelector._running_variance` (the EMA of the
world-forward prediction error) and an ABSOLUTE bar of 0.40
(`E3Config.commitment_threshold`). Training the forward model collapses rv to
~1e-6..1.6e-5 -- FIVE ORDERS below that bar -- measured under both phased and
joint training and at `alpha_world` 0.3 and 0.9, giving a measured
`committed_step_fraction` of 1.0000 even at MECH-108 `sweep_amplitude` 0.999.
Commitment is then an absorbing STATE, not one of two MODES, and ARC-029 (which
asserts two operating modes) has no bar the agent can be on both sides of.
Measurement:
`REE_assembly/evidence/planning/arc029_p1_lever_calibration_finding_20260918.md`
(REE_assembly 67520bc08d); GFLAG-0346. User decision 2026-09-18 (option D,
chip-20260918-arc029-p1-lever-operating-point) chose this build.

The lever makes the BASE bar the q-quantile of the run's own recent
commit-gate-variance distribution over a fixed-width sliding window, so
occupancy is ~q at ANY absolute rv scale.

METHOD / anti-vacuity. Every positive assertion is paired with its own negative
control. `test_off_path_reproduces_the_measured_saturation` REPRODUCES the
defect first: if that ever stops saturating, the rest of this file is vacuous.
`test_expanding_window_estimator_resaturates` is the control for the ESTIMATOR
choice specifically -- it shows the obvious alternative (a quantile over the
whole run so far) silently re-saturates under the measured ~5x within-run
drift, which is the one way this build could be inert while looking correct.

THE rv STREAM IS SYNTHETIC, THE GATE IS NOT. These tests drive the real
`select()` gate and the real `update_running_variance` EMA; only the
prediction-error stream is synthetic, calibrated to the measured post-training
band (rv ~1.2e-6 -> 1.1e-5, the measured ~5x drift). This is a SELECTOR-level
demonstration that the bar change removes saturation at the measured operating
point -- it is NOT an end-to-end trained-agent run, which belongs to the
re-queued ARC-029 experiment.

NO `multinomial` DRAW is involved in any assertion here (occupancy fractions and
run lengths are computed from python floats over recorded diagnostics), so this
carries no cross-machine class-contract hazard -- CLAUDE.md "Running the test
suite"; memory `reference-cross-machine-class-contract-divergence`.

BIT-IDENTITY. `test_arming_the_lever_while_warming_is_bit_identical` is the
strong form the build constraint asks for -- it proves identity where the new
code RUNS, not merely where it is skipped: the lever is ARMED (so the window is
being appended to on every tick and the helper is called on every tick) but the
window is wider than the run, so the bar never takes force. Scores, committed
flags, the parameter fingerprint and the torch RNG state must all match the
default-OFF run exactly. A cross-REVISION comparison against pristine
`origin/main` was NOT run: the dispatching box has no torch, and
`remote_pytest.sh` routes pytest rather than arbitrary scripts. The OFF path
adds exactly one `is None` identity check and no tensor op, RNG draw or
parameter, which this test pins behaviourally.
"""

import math

import pytest
import torch

from ree_core.utils.config import E3Config, REEConfig
from ree_core.predictors.e3_selector import E3TrajectorySelector
from ree_core.predictors.e2_fast import Trajectory

WORLD_DIM, HIDDEN_DIM, N_CAND = 8, 16, 4
TICKS = 1200
WINDOW = 200

# The measured post-training band and drift (finding doc, tables 1 and 2).
RV_LO, RV_HI = 1.2e-6, 1.1e-5


def _traj(seed: int) -> Trajectory:
    g = torch.Generator().manual_seed(seed)
    h = 4
    ws = torch.randn(1, h + 1, WORLD_DIM, generator=g)
    ss = torch.randn(1, h + 1, WORLD_DIM, generator=g)
    acts = torch.randn(1, h, 4, generator=g)
    return Trajectory(
        states=[ss[:, i, :] for i in range(h + 1)],
        actions=acts,
        world_states=[ws[:, i, :] for i in range(h + 1)],
    )


def _candidates():
    return [_traj(100 + i) for i in range(N_CAND)]


def _selector(**cfg_kw) -> E3TrajectorySelector:
    torch.manual_seed(0)
    # precision_init IS the initial _running_variance. Start it at the measured
    # post-training band rather than the 0.5 default: the scientific object here
    # is a TRAINED agent, and letting the alpha=0.05 EMA walk down from 0.5 would
    # spend most of the run in a warmup regime that is not what was measured.
    # Only this INITIAL CONDITION is set; rv then evolves through the real EMA.
    cfg = E3Config(world_dim=WORLD_DIM, hidden_dim=HIDDEN_DIM, precision_init=RV_LO)
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    sel = E3TrajectorySelector(cfg)
    sel.e3_score_decomp_enabled = True
    return sel


def _pe_stream(ticks: int, seed: int = 7):
    """Prediction errors whose MSE drifts RV_LO -> RV_HI with lognormal jitter.
    Fed through the REAL alpha=0.05 EMA, so rv is produced, not injected."""
    g = torch.Generator().manual_seed(seed)
    out = []
    span = math.log(RV_HI / RV_LO)
    for t in range(ticks):
        base = RV_LO * math.exp(span * (t / (ticks - 1)))
        jitter = math.exp(0.6 * float(torch.randn(1, generator=g).item()))
        out.append(torch.full((1, 1), math.sqrt(base * jitter)))
    return out


def _drive(sel, ticks=TICKS, sweep=0.0):
    """Returns (committed flags, gate variances, bars, final scores)."""
    cands = _candidates()
    pes = _pe_stream(ticks)
    flags, gvars, bars, scores = [], [], [], []
    for t in range(ticks):
        sel.update_running_variance(pes[t])
        sel.select(cands, sweep_threshold_reduction=sweep)
        d = sel.last_score_diagnostics
        flags.append(bool(d["committed"]))
        gvars.append(float(d["commit_variance"]))
        bars.append(d.get("variance_tracking_commit_bar"))
        scores.append(sel.last_scores.clone())
    return flags, gvars, bars, scores


def _occupancy(flags):
    """(fraction committed, mean committed-run length, n runs)."""
    frac = sum(flags) / len(flags)
    lens, cur = [], 0
    for f in flags:
        if f:
            cur += 1
        elif cur:
            lens.append(cur)
            cur = 0
    if cur:
        lens.append(cur)
    return frac, (sum(lens) / len(lens) if lens else 0.0), len(lens)


def _quantile(vals, q):
    v = sorted(vals)
    pos = (len(v) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return v[lo]
    frac = pos - lo
    return v[lo] * (1.0 - frac) + v[hi] * frac


def _detrended_bar(window_vals, q):
    """Independent re-implementation of the shipped estimator, for the
    causality test: least-squares line through log(v) vs index, q-quantile of
    the residuals, trend extrapolated one tick past the window's end."""
    n = len(window_vals)
    hi = max(window_vals)
    floor = hi * 1e-12
    logs = [math.log(v if v > floor else floor) for v in window_vals]
    mean_i = (n - 1) / 2.0
    mean_y = sum(logs) / n
    sxx = sum((i - mean_i) ** 2 for i in range(n))
    sxy = sum((i - mean_i) * (logs[i] - mean_y) for i in range(n))
    slope = sxy / sxx if sxx > 0 else 0.0
    resid = sorted(logs[i] - (mean_y + slope * (i - mean_i)) for i in range(n))
    trend_now = mean_y + slope * (n - mean_i)
    return math.exp(trend_now + _quantile(resid, q))


def _fingerprint(sel):
    return [float(p.detach().double().sum().item()) for p in sel.parameters()]


# --------------------------------------------------------------------------- #
# 1. The defect, reproduced (negative control for the whole file)              #
# --------------------------------------------------------------------------- #

def test_off_path_reproduces_the_measured_saturation():
    """With the fixed 0.40 bar and rv at the MEASURED post-training band, the
    agent is permanently committed -- the finding's 1.0000. If this ever stops
    saturating, every ON-path assertion below is vacuous."""
    sel = _selector()
    flags, gvars, bars, _ = _drive(sel)

    assert all(b is None for b in bars), "lever is OFF; no quantile bar may exist"
    assert min(gvars) > 0.0
    drift = max(gvars) / min(gvars)
    assert drift > 2.0, (
        "the stream must reproduce a real within-run rv drift (measured ~5x); "
        "got %.2fx over %.3e..%.3e" % (drift, min(gvars), max(gvars))
    )
    assert max(gvars) < 1e-3, (
        "rv must sit orders below the 0.40 bar, as measured; got max %.3e"
        % max(gvars)
    )
    frac, _, _ = _occupancy(flags)
    assert frac == 1.0, (
        "expected the measured absorbing state (committed_step_fraction "
        "1.0000) with the absolute bar; got %.4f" % frac
    )


# --------------------------------------------------------------------------- #
# 2. The ON path moves the thing the build exists to move                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("q", [0.25, 0.50, 0.75])
def test_on_path_leaves_saturation_and_tracks_the_requested_quantile(q):
    """Occupancy becomes ~q at an rv scale five orders below the absolute bar.
    This is the whole point of the build: NOT that the quantile is computed,
    but that committed_step_fraction leaves 1.0000."""
    sel = _selector(
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=q,
        commit_threshold_quantile_window=WINDOW,
    )
    flags, _, bars, _ = _drive(sel)

    # the first WINDOW ticks are still on the absolute bar, by design
    assert all(b is None for b in bars[:WINDOW]), "warmup must keep the absolute bar"
    assert all(b is not None for b in bars[WINDOW:]), "bar must take force once full"

    frac, mean_run, n_runs = _occupancy(flags[WINDOW:])
    assert frac < 0.95, (
        "ON path must leave saturation; got committed_step_fraction %.4f" % frac
    )
    assert abs(frac - q) < 0.10, (
        "occupancy should track the requested quantile q=%.2f; got %.4f" % (q, frac)
    )
    assert n_runs > 1, "a two-mode occupancy needs more than one committed run"


def test_on_path_keeps_committed_run_length_structure():
    """ARC-029's P1 asks for mean committed-run length >= 3 ticks, not merely
    an occupancy fraction. A bar that tracked rv instantaneously would hit the
    fraction and fail this."""
    sel = _selector(
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=0.50,
        commit_threshold_quantile_window=WINDOW,
    )
    flags, _, _, _ = _drive(sel)
    frac, mean_run, _ = _occupancy(flags[WINDOW:])
    assert mean_run >= 3.0, (
        "committed runs must have length structure; mean_run_len=%.2f at "
        "occupancy %.4f" % (mean_run, frac)
    )


def test_mech108_sweep_regains_dynamic_range_on_the_quantile_bar():
    """The finding measured the sweep INERT at the trained operating point:
    committed_step_fraction 1.0000 even at amplitude 0.999, because reaching a
    bar five orders above rv needs a > 0.99996. On the quantile bar the sweep
    must move occupancy GRADEDLY at MODEST amplitudes, because the bar now sits
    inside rv's own observed spread.

    Gradedness, not just "it moved", is the assertion that matters: a lever
    that jumped straight from 1.0 to 0.0 would be as unusable for an
    alternating-arm design as an inert one.
    """
    kw = dict(
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=0.50,
        commit_threshold_quantile_window=WINDOW,
    )
    fracs = [
        _occupancy(_drive(_selector(**kw), sweep=a)[0][WINDOW:])[0]
        for a in (0.0, 0.02, 0.05, 0.10)
    ]
    assert all(
        fracs[i] > fracs[i + 1] for i in range(len(fracs) - 1)
    ), "occupancy must fall MONOTONICALLY with sweep amplitude; got %r" % (fracs,)
    assert fracs[0] - fracs[-1] > 0.15, (
        "a modest amplitude (<= 0.10) must move occupancy substantially on the "
        "quantile bar; got %r" % (fracs,)
    )
    assert fracs[-1] > 0.0, (
        "amplitude 0.10 must not already have collapsed occupancy to zero -- "
        "the usable band would then be too narrow to alternate in; got %r"
        % (fracs,)
    )


# --------------------------------------------------------------------------- #
# 3. Negative control for the ESTIMATOR choice                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("q", [0.25, 0.50, 0.75])
def test_naive_quantile_estimators_miss_the_target_occupancy(q):
    """THE control that justifies DETRENDING, and the single most likely way
    this build could be inert while looking correct.

    Two obvious readings of "a quantile of the run's own rv distribution" are
    measured here against the shipped one, on the SAME gate-variance stream:

      - EXPANDING window (quantile over the whole run so far): lags the drift
        over the entire run.
      - plain TRAILING window: centred half a window BEHIND the tick it
        judges, so under a drift the current sample sits systematically on one
        side of it.

    Both miss the requested occupancy q by more than the detrended estimator
    does. If that ever stops being true, the detrending in
    `_variance_tracking_commit_bar` is dead weight and should be removed --
    this test is what would say so.
    """
    _, gvars, _, _ = _drive(_selector())

    expanding, trailing, detrended = [], [], []
    for t in range(WINDOW, len(gvars)):
        w = gvars[t - WINDOW:t]
        expanding.append(gvars[t] < _quantile(gvars[:t], q))
        trailing.append(gvars[t] < _quantile(w, q))
        detrended.append(gvars[t] < _detrended_bar(w, q))

    exp_err = abs(_occupancy(expanding)[0] - q)
    trl_err = abs(_occupancy(trailing)[0] - q)
    det_err = abs(_occupancy(detrended)[0] - q)

    assert det_err < 0.10, (
        "detrended estimator should hold occupancy near q=%.2f; error %.4f"
        % (q, det_err)
    )
    assert det_err < trl_err, (
        "detrending must beat the plain trailing window under drift: "
        "detrended err %.4f vs trailing err %.4f (q=%.2f)"
        % (det_err, trl_err, q)
    )
    assert det_err < exp_err, (
        "detrending must beat the expanding window under drift: "
        "detrended err %.4f vs expanding err %.4f (q=%.2f)"
        % (det_err, exp_err, q)
    )


# --------------------------------------------------------------------------- #
# 4. Bit-identity -- including where the new code RUNS                         #
# --------------------------------------------------------------------------- #

def test_arming_the_lever_while_warming_is_bit_identical():
    """The strong form: the lever is ARMED, so the window is appended on every
    tick and the bar helper is called on every tick -- but the window is wider
    than the run, so the bar never takes force. Trajectory, parameter
    fingerprint and torch RNG state must all be identical to default-OFF."""
    ticks = 300

    torch.manual_seed(1234)
    off = _selector()
    off_flags, off_gv, off_bars, off_scores = _drive(off, ticks=ticks)
    off_rng = torch.get_rng_state().clone()
    off_fp = _fingerprint(off)

    torch.manual_seed(1234)
    on = _selector(
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=0.50,
        commit_threshold_quantile_window=ticks + 1,   # never fills
    )
    on_flags, on_gv, on_bars, on_scores = _drive(on, ticks=ticks)
    on_rng = torch.get_rng_state().clone()
    on_fp = _fingerprint(on)

    # the new code really did run
    assert on._commit_gate_variance_window is not None
    assert len(on._commit_gate_variance_window) == ticks, "window must be filling"
    assert all(b is None for b in on_bars), "bar must not take force while warming"

    assert on_flags == off_flags
    assert on_gv == off_gv
    assert on_fp == off_fp, "parameter fingerprint must be untouched"
    assert torch.equal(on_rng, off_rng), "torch RNG state must be untouched"
    for a, b in zip(on_scores, off_scores):
        assert torch.equal(a, b), "scores must be bit-identical"


def test_off_path_creates_no_window_at_all():
    sel = _selector()
    assert sel._commit_gate_variance_window is None
    _drive(sel, ticks=20)
    assert sel._commit_gate_variance_window is None


# --------------------------------------------------------------------------- #
# 5. The bar is strictly causal                                                #
# --------------------------------------------------------------------------- #

def test_bar_is_a_quantile_of_strictly_earlier_ticks():
    """A sample must never contribute to the bar it is itself judged against."""
    q, ticks = 0.5, WINDOW + 40
    sel = _selector(
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=q,
        commit_threshold_quantile_window=WINDOW,
    )
    _, gvars, bars, _ = _drive(sel, ticks=ticks)
    for t in range(WINDOW, ticks):
        expected = _detrended_bar(gvars[t - WINDOW:t], q)
        assert bars[t] == pytest.approx(expected, rel=1e-9, abs=0.0), (
            "bar at tick %d must be the quantile of ticks [%d, %d)" % (t, t - WINDOW, t)
        )


# --------------------------------------------------------------------------- #
# 6. Wiring: the knobs must ARRIVE (from_dims silently swallows unknown kwargs) #
# --------------------------------------------------------------------------- #

def test_all_three_knobs_survive_from_dims():
    """`REEConfig.from_dims` swallows unknown kwargs silently -- an E3Config
    knob needs a field, a signature entry AND a `config.e3` mirror or the lever
    is structurally present and functionally inert. Assert, do not assume."""
    cfg = REEConfig.from_dims(
        body_obs_dim=4, world_obs_dim=4, action_dim=4,
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=0.37,
        commit_threshold_quantile_window=123,
    )
    assert cfg.e3.use_variance_tracking_commit_threshold is True
    assert cfg.e3.commit_threshold_quantile == 0.37
    assert cfg.e3.commit_threshold_quantile_window == 123


def test_from_dims_default_leaves_the_lever_off_with_unset_sentinels():
    cfg = REEConfig.from_dims(body_obs_dim=4, world_obs_dim=4, action_dim=4)
    assert cfg.e3.use_variance_tracking_commit_threshold is False
    assert cfg.e3.commit_threshold_quantile == -1.0
    assert cfg.e3.commit_threshold_quantile_window == -1


# --------------------------------------------------------------------------- #
# 7. No science-bearing default: arming without choosing must FAIL LOUDLY      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "kw",
    [
        dict(commit_threshold_quantile_window=WINDOW),                    # q unset
        dict(commit_threshold_quantile=0.5),                              # W unset
        dict(commit_threshold_quantile=0.0, commit_threshold_quantile_window=WINDOW),
        dict(commit_threshold_quantile=1.0, commit_threshold_quantile_window=WINDOW),
        dict(commit_threshold_quantile=0.5, commit_threshold_quantile_window=1),
    ],
)
def test_arming_without_an_explicit_operating_point_raises(kw):
    """The quantile IS the target occupancy and the window decides whether
    committed runs have length structure -- both belong to the experiment's
    pre-registration, so the substrate refuses to supply one silently. This
    also converts a from_dims silent-swallow into a loud construction error."""
    with pytest.raises(ValueError):
        _selector(use_variance_tracking_commit_threshold=True, **kw)
