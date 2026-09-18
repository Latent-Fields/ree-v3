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
    cfg = E3Config(world_dim=WORLD_DIM, hidden_dim=HIDDEN_DIM)
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
    """The finding measured the sweep INERT at the trained operating point
    (1.0000 committed at amplitude 0.999). On the quantile bar a MODEST
    amplitude must move occupancy, because the bar now sits inside rv's own
    observed spread."""
    kw = dict(
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=0.50,
        commit_threshold_quantile_window=WINDOW,
    )
    base, _, _ = _occupancy(_drive(_selector(**kw), sweep=0.0)[0][WINDOW:])
    swept, _, _ = _occupancy(_drive(_selector(**kw), sweep=0.3)[0][WINDOW:])
    assert base - swept > 0.05, (
        "a moderate sweep must lower occupancy on the quantile bar; "
        "%.4f -> %.4f at amplitude 0.3" % (base, swept)
    )


# --------------------------------------------------------------------------- #
# 3. Negative control for the ESTIMATOR choice                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("q", [0.25, 0.50, 0.75])
def test_expanding_window_estimator_resaturates(q):
    """THE control that justifies the fixed-width window. A quantile over the
    whole run so far LAGS the measured downward-in-time drift, so late samples
    fall on one side of a stale bar and occupancy walks away from q -- inert
    while looking correct. The sliding window on the same stream does not."""
    _, gvars, _, _ = _drive(_selector())

    expanding, sliding = [], []
    for t in range(WINDOW, len(gvars)):
        expanding.append(gvars[t] < _quantile(gvars[:t], q))
        sliding.append(gvars[t] < _quantile(gvars[t - WINDOW:t], q))

    exp_frac, _, _ = _occupancy(expanding)
    sld_frac, _, _ = _occupancy(sliding)

    assert abs(sld_frac - q) < 0.10, (
        "sliding window should hold occupancy at q=%.2f; got %.4f" % (q, sld_frac)
    )
    assert abs(exp_frac - q) > abs(sld_frac - q), (
        "expanding window must drift FURTHER from q than the sliding one "
        "(that is why the window is fixed-width): expanding %.4f vs sliding "
        "%.4f, target %.2f" % (exp_frac, sld_frac, q)
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
        expected = _quantile(gvars[t - WINDOW:t], q)
        assert bars[t] == pytest.approx(expected, rel=1e-12, abs=0.0), (
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
        use_variance_tracking_commit_threshold=True,
        commit_threshold_quantile=0.37,
        commit_threshold_quantile_window=123,
    )
    assert cfg.e3.use_variance_tracking_commit_threshold is True
    assert cfg.e3.commit_threshold_quantile == 0.37
    assert cfg.e3.commit_threshold_quantile_window == 123


def test_from_dims_default_leaves_the_lever_off_with_unset_sentinels():
    cfg = REEConfig.from_dims()
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
