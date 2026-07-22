"""Contracts for the SD-076 running-variance floor headroom repair (2026-07-22).

Surface under test: `E3TrajectorySelector._apply_wci_rv_floor` and the inflation branch
of `update_running_variance`.

WHY THIS GATE EXISTS. V3-EXQ-794 could not test SD-076 or MECH-204 because
`waking_confidence_rv_floor` defaults to 0.01 while the substrate's un-inflated operating
point is rv = 0.005420 and its true error reference is ~0.0037. The absolute floor
therefore sat 1.8x ABOVE the operating point, `max(floor, rv)` clamped from the first tick
inflation bit, and `rv_final` finished at EXACTLY 0.010000 on all four inflation arms with
`overconfidence_score` bit-identical to 15 significant figures (-1.004111904519277) at the
LO and HI asymmetry levels. Two doses producing one value is saturation, not a null.

WHY THE 2026-07-20 SMOKE PASSED 6/6 ANYWAY -- the property this file exists to stop
recurring. That smoke used a synthetic error sequence with true mean 0.05, ~13x the
substrate's real error scale, where 0.01 IS a floor with headroom. An absolute constant
validated at one scale silently became a clamp at another. So these contracts run at the
MEASURED 794 scale, and the repair is a scale-RELATIVE bound rather than a smaller
absolute one.

THE SIGN IS NOT THE BUG. The autopsy listed a wrong-sign inflation as candidate cause (b);
it is ruled out here by test_inflation_lowers_rv_below_the_uninflated_counterfactual.
Inflation drove rv DOWN correctly, into a floor that happens to sit ABOVE the OFF arm, so
the run's `inflation_lowers_rv` precondition saw rv rise.
"""
import math
import sys
from collections import deque
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # ree-v3/
sys.path.insert(0, str(REPO_ROOT))

from ree_core.predictors.e3_selector import E3TrajectorySelector  # noqa: E402
from ree_core.utils.config import E3Config  # noqa: E402

# V3-EXQ-794 measured `arm_true_error_ref` ~ 0.0037 and ARM_OFF_OFF `rv_final` 0.005420.
TRUE_ERR = 0.0037
N_STEPS = 4000
ASYM_LO, ASYM_HI = 0.6, 0.8  # the run's own two dose levels
# The repaired configuration: bound relative to the un-inflated counterfactual, approached
# softly. Both knobs are no-op at their defaults.
REPAIRED = dict(frac=0.2, mode="soft", softness=0.25)


def _err_seq(n, seed=0):
    """Deterministic error sequence, mean TRUE_ERR, with GENUINE variance.

    The variance is load-bearing: an asymmetric EMA has NO dose-response on a constant
    input (every alpha > 0 converges to the same fixed point), so a degenerate sequence
    would report a null that is an artefact of the harness rather than of the substrate.
    """
    import random

    r = random.Random(1234 + seed)
    return [TRUE_ERR * (0.2 + 1.6 * r.random()) for _ in range(n)]


def _selector(asym=None, floor=0.01, frac=0.0, mode="hard", softness=0.25):
    """A bare selector exercising only the running-variance path."""
    cfg = E3Config()
    cfg.use_waking_confidence_inflation = asym is not None
    cfg.waking_confidence_inflation_asymmetry = float(asym or 0.0)
    cfg.waking_confidence_rv_floor = floor
    cfg.waking_confidence_rv_floor_relative_frac = frac
    cfg.waking_confidence_rv_floor_mode = mode
    cfg.waking_confidence_rv_floor_softness = softness
    sel = E3TrajectorySelector.__new__(E3TrajectorySelector)
    sel.config = cfg
    sel._running_variance = cfg.precision_init
    sel._ema_alpha = cfg.precision_ema_alpha
    sel._wci_symmetric_rv_ref = cfg.precision_init
    sel._last_instantaneous_pe = 0.0
    sel._rv_history = deque(maxlen=50)
    sel._volatility_estimate = 0.0
    return sel


def _run(asym=None, seed=0, **kw):
    """Returns (rv_final, counterfactual_uninflated_ref)."""
    sel = _selector(asym=asym, **kw)
    for e in _err_seq(N_STEPS, seed):
        # float64 so bit-identity checks compare arithmetic, not float32 rounding of the
        # sqrt/square round-trip.
        sel.update_running_variance(torch.tensor([math.sqrt(e)], dtype=torch.float64))
    return float(sel._running_variance), float(sel._wci_symmetric_rv_ref)


def _legacy(asym, floor=0.01, seed=0):
    """The pre-repair arithmetic, written out independently of the substrate."""
    cfg = E3Config()
    rv = cfg.precision_init
    a0 = cfg.precision_ema_alpha
    for e0 in _err_seq(N_STEPS, seed):
        e = float(torch.tensor([math.sqrt(e0)], dtype=torch.float64).pow(2).mean().item())
        if asym is None:
            rv = (1 - a0) * rv + a0 * e
        else:
            asy = max(0.0, min(0.999, asym))
            alpha = min(1.0, a0 * (1.0 + asy)) if e < rv else max(0.0, a0 * (1.0 - asy))
            rv = max(floor, (1 - alpha) * rv + alpha * e)
    return rv


# --------------------------------------------------------------------------------------
# Backward compatibility -- the repair must be invisible until it is switched on
# --------------------------------------------------------------------------------------

def test_off_path_is_bit_identical():
    """Master flag False: the original symmetric expression, untouched."""
    assert _run(asym=None)[0] == _legacy(None)


@pytest.mark.parametrize("asym", [ASYM_LO, ASYM_HI])
def test_on_path_at_default_knobs_is_bit_identical_to_the_hard_clip(asym):
    """Both new knobs default to no-op, so ON reproduces the pre-repair arithmetic."""
    assert _run(asym=asym)[0] == _legacy(asym)


def test_default_config_still_clips_exactly():
    cfg = E3Config()
    assert cfg.waking_confidence_rv_floor_relative_frac == 0.0
    assert cfg.waking_confidence_rv_floor_mode == "hard"


# --------------------------------------------------------------------------------------
# The defect, and its repair
# --------------------------------------------------------------------------------------

def test_the_794_saturation_reproduces_under_the_old_config():
    """Pin the DEFECT, so a future 'simplification' back to an absolute hard floor fails
    here rather than in a run."""
    lo, _ = _run(asym=ASYM_LO)
    hi, _ = _run(asym=ASYM_HI)
    assert lo == 0.01 and hi == 0.01, (lo, hi)
    assert lo == hi  # the exact 794 signature


def test_repaired_config_separates_the_two_dose_levels():
    """ACCEPTANCE CRITERION: strict LO != HI separation in rv_final."""
    lo, _ = _run(asym=ASYM_LO, **REPAIRED)
    hi, _ = _run(asym=ASYM_HI, **REPAIRED)
    assert lo != hi
    assert hi < lo, "a HIGHER asymmetry must produce a MORE confident (lower) rv"
    # Not float noise, and not a tie the dose_saturation lint would have to refuse.
    assert abs(lo - hi) > 1e-9 * max(lo, hi)


@pytest.mark.parametrize("asym", [ASYM_LO, ASYM_HI])
def test_repaired_config_reaches_genuine_overconfidence(asym):
    """ACCEPTANCE CRITERION: rv must be able to sit BELOW true error, which is what
    `n_seeds_overconfident` counts. Unreachable under any bound anchored at 0.01."""
    rv, _ = _run(asym=asym, **REPAIRED)
    assert rv < TRUE_ERR, "rv %.6f is not below true error %.6f" % (rv, TRUE_ERR)


def test_inflation_lowers_rv_below_the_uninflated_counterfactual():
    """Rules out the autopsy's candidate cause (b), a wrong-sign inflation."""
    rv, ref = _run(asym=ASYM_HI, **REPAIRED)
    assert rv < ref, "inflation must push rv BELOW the symmetric reference"


def test_counterfactual_reference_tracks_the_off_path():
    """`_wci_symmetric_rv_ref` must equal what rv would be with inflation OFF -- that
    equality is what makes the relative floor a bound on the SUBSTRATE's own scale."""
    _, ref = _run(asym=ASYM_HI, **REPAIRED)
    off, _ = _run(asym=None)
    assert ref == pytest.approx(off, rel=1e-12)


# --------------------------------------------------------------------------------------
# Properties of the saturating nonlinearity
# --------------------------------------------------------------------------------------

def _soft(sel, xs):
    return [sel._apply_wci_rv_floor(x) for x in xs]


def test_soft_mode_is_strictly_monotonic():
    """The structural guarantee: two distinct doses can never map to one value, so a
    residual saturation SHRINKS a separation instead of collapsing it to a tie."""
    sel = _selector(floor=0.01, mode="soft")
    sel._wci_symmetric_rv_ref = 0.005
    ys = _soft(sel, [i * 1e-4 for i in range(400)])
    assert all(b > a for a, b in zip(ys, ys[1:]))
    assert len(set(ys)) == len(ys)


def test_soft_mode_never_crosses_the_floor():
    """The floor is load-bearing, not hygiene: rv feeds an ABSOLUTE commit threshold
    (ARC-016) and current_precision = 1/(rv + 1e-6)."""
    sel = _selector(floor=0.01, mode="soft")
    sel._wci_symmetric_rv_ref = 0.005
    assert all(y > 0.01 for y in _soft(sel, [i * 1e-4 for i in range(400)]))


def test_soft_mode_is_the_identity_well_above_the_floor():
    """Saturation must not distort the regime it is not bounding."""
    sel = _selector(floor=0.01, mode="soft", softness=0.25)
    sel._wci_symmetric_rv_ref = 0.005
    assert sel._apply_wci_rv_floor(0.5) == pytest.approx(0.5, rel=1e-9)


def test_relative_floor_scales_with_the_substrate():
    """The property the absolute floor lacked. Same frac, two error scales -> two floors."""
    sel = _selector(frac=0.2, mode="hard")
    sel._wci_symmetric_rv_ref = 0.004
    assert sel._apply_wci_rv_floor(1e-9) == pytest.approx(0.0008)
    sel._wci_symmetric_rv_ref = 0.04
    assert sel._apply_wci_rv_floor(1e-9) == pytest.approx(0.008)


def test_zero_softness_degenerates_to_the_hard_clip():
    """softplus -> max as the knee width -> 0; the limit must not divide by zero."""
    sel = _selector(floor=0.01, mode="soft", softness=0.0)
    assert sel._apply_wci_rv_floor(1e-9) == 0.01
    assert sel._apply_wci_rv_floor(0.5) == 0.5


def test_no_usable_bound_returns_rv_unchanged():
    sel = _selector(floor=0.0, mode="soft")
    assert sel._apply_wci_rv_floor(0.003) == 0.003


def test_floor_never_raises_on_extreme_inputs():
    """The softplus is evaluated in a stable form; large |z| must not overflow."""
    sel = _selector(floor=0.01, mode="soft")
    sel._wci_symmetric_rv_ref = 0.005
    for x in (0.0, 1e-30, 1e30, 1e300):
        y = sel._apply_wci_rv_floor(x)
        assert math.isfinite(y) and y > 0.0
