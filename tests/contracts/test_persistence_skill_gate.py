"""Contracts for experiments/_lib/persistence_skill_gate.py (SD-PP-B9 instrument).

THE BLIND-SPOT MEASUREMENT (CLAUDE.md "The test half") is
`test_blind_spot_new_gate_catches_what_the_old_floor_passed`: it runs the NEW
gate against the OLD defect (V3-EXQ-1062a's six landed cells) and requires it to
FAIL there, and in the same test confirms the OLD guard (an absolute
harm_a_forward_r2 >= 0.30 floor) still PASSES on all six. Without that second
half the first is not evidence the gate changed anything.
"""

import importlib.util
import math
import os
import sys

import pytest
import torch

_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "experiments", "_lib")
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

import persistence_skill_gate as psg  # noqa: E402

SUPERSEDED_FORWARD_R2_MIN = 0.30   # the floor this instrument replaces


# --------------------------------------------------------------------------- #
# 1. The blind-spot measurement.                                                #
# --------------------------------------------------------------------------- #

def test_blind_spot_new_gate_catches_what_the_old_floor_passed():
    """New gate FAILS on the old defect; old guard PASSES on it. Both halves."""
    cells = psg.CANARY_V3_EXQ_1062A["cells"]
    assert len(cells) == 6

    old_guard_passes = 0
    old_guard_control_passes = 0
    new_gate_fails = 0
    for _seed, model_r2, persistence_r2, _skill in cells:
        # OLD guard: an absolute floor on the model's own R2.
        if model_r2 >= SUPERSEDED_FORWARD_R2_MIN:
            old_guard_passes += 1
        # ... and the CONTROL clears that same floor, which is the defect.
        if persistence_r2 >= SUPERSEDED_FORWARD_R2_MIN:
            old_guard_control_passes += 1
        # NEW gate, through the shipped path.
        pred, tgt, prev, _ = psg._synthetic_pairs(model_r2, persistence_r2)
        v = psg.persistence_verdict(pred, tgt, prev, n_bootstrap=400)
        if v.status == "persistence_dominated":
            new_gate_fails += 1

    assert old_guard_passes == 6, "old guard must still pass -- else no blind spot"
    assert old_guard_control_passes == 6, (
        "the CONTROL clearing the old floor in 6/6 IS the defect; if this ever "
        "stops holding the canary's premise has changed")
    assert new_gate_fails == 6, (
        "the new gate must FAIL on every cell the old floor passed, or it has "
        "not closed the blind spot")


def test_model_sat_below_its_own_control_in_every_landed_cell():
    """The premise: model_r2 < persistence_r2 in 6/6, yet the old floor passed."""
    for _seed, model_r2, persistence_r2, _skill in psg.CANARY_V3_EXQ_1062A["cells"]:
        assert model_r2 < persistence_r2
        assert model_r2 >= SUPERSEDED_FORWARD_R2_MIN


# --------------------------------------------------------------------------- #
# 2. The canary (remedy 2).                                                     #
# --------------------------------------------------------------------------- #

def test_canary_reproduces_the_landed_manifest():
    c = psg.check_canary()
    assert c["ok"] is True, c
    assert c["n_cells"] == 6
    assert c["superseded_gate_control_pass_cells"] == 6
    for cell in c["cells"]:
        assert cell["reconstruction_faithful"]
        assert cell["battery_carries_pinned_r2"]
        assert cell["ci_non_degenerate"]
        assert cell["got"] == "persistence_dominated"


def test_canary_reconstruction_is_an_independent_route():
    """skill_from_r2 must reproduce the manifest's own SSE-derived skill."""
    for _seed, m_r2, p_r2, skill_recorded in psg.CANARY_V3_EXQ_1062A["cells"]:
        rec = psg.skill_from_r2(m_r2, p_r2)
        assert rec is not None
        assert abs(rec - skill_recorded) <= 1e-6 * max(1.0, abs(skill_recorded))


def test_canary_never_raises_and_reports_its_driver():
    c = psg.check_canary()
    assert "persistence_verdict" in c["drives"]
    assert "not a re-implementation" in c["drives"]


# --------------------------------------------------------------------------- #
# 3. The canary's own guard: a degenerate CI must be caught, not passed.        #
# --------------------------------------------------------------------------- #

def test_proportional_residual_battery_yields_a_degenerate_ci():
    """The regression the canary's ci_non_degenerate check exists to catch.

    Both predictors as scalings of the target make the per-row error ratio
    constant, so every bootstrap resample is identical and the CI has zero width.
    The classification is still correct -- which is exactly why a canary that
    only checked the classification would pass while testing nothing.
    """
    g = torch.Generator(device="cpu"); g.manual_seed(7)
    base = torch.randn(256, 4, generator=g)
    base = base - base.mean(dim=0, keepdim=True)
    pred = base * 0.2          # far from the target
    prev = base * 0.95         # close to it
    v = psg.persistence_verdict(pred, base, prev, n_bootstrap=400)
    assert v.status == "persistence_dominated"          # classification fine
    # ... but the CI is dead. NOTE it is NOT bit-exactly zero: float32
    # accumulation leaves ~1.8e-07 of width, which is why the canary's guard
    # cannot be a bare `width > 0`. This test pins that measurement.
    rel_width = (v.ci_high - v.ci_low) / abs(v.relative_skill)
    assert rel_width < psg.CI_LIVE_REL_WIDTH_MIN
    assert rel_width < 1e-5


def test_synthetic_pairs_produce_a_live_ci():
    for _seed, m_r2, p_r2, _skill in psg.CANARY_V3_EXQ_1062A["cells"]:
        pred, tgt, prev, _ = psg._synthetic_pairs(m_r2, p_r2)
        v = psg.persistence_verdict(pred, tgt, prev, n_bootstrap=400)
        rel_width = (v.ci_high - v.ci_low) / abs(v.relative_skill)
        assert rel_width > psg.CI_LIVE_REL_WIDTH_MIN


def test_the_canary_guard_would_reject_a_degenerate_battery():
    """Blind-spot check on the CANARY itself, not on the gate.

    Substituting the proportional-residual (degenerate-CI) battery for
    `_synthetic_pairs` must make the canary's own ci_live test fail. Otherwise
    the canary cannot notice a bootstrap that has stopped resampling.
    """
    g = torch.Generator(device="cpu"); g.manual_seed(7)
    base = torch.randn(256, 4, generator=g)
    base = base - base.mean(dim=0, keepdim=True)
    v = psg.persistence_verdict(base * 0.2, base, base * 0.95, n_bootstrap=400)
    ci_live = (v.ci_low is not None and v.ci_high is not None
               and abs(v.relative_skill) > 0.0
               and ((v.ci_high - v.ci_low) / abs(v.relative_skill))
               > psg.CI_LIVE_REL_WIDTH_MIN)
    assert ci_live is False


# --------------------------------------------------------------------------- #
# 4. No absolute bar: the control value is analytic, and it is 0.               #
# --------------------------------------------------------------------------- #

def test_persistence_scored_against_itself_reads_exactly_the_control_value():
    """The whole point: the baseline's own relative skill is 0, by construction."""
    assert psg.relative_skill(5.0, 5.0) == 0.0
    assert psg.relative_skill(1e-9, 1e-9) == 0.0
    assert psg.relative_skill(1e9, 1e9) == 0.0
    # and the same through the tensor path
    g = torch.Generator(device="cpu"); g.manual_seed(3)
    tgt = torch.randn(128, 3, generator=g)
    prev = tgt + torch.randn(128, 3, generator=g) * 0.3
    v = psg.persistence_verdict(prev.clone(), tgt, prev, n_bootstrap=200)
    assert v.relative_skill == pytest.approx(0.0, abs=1e-9)
    assert v.status == "cannot_determine"      # a tie is not readiness


def test_relative_skill_is_bounded_and_sign_agrees_with_skill():
    for sse_m, sse_p in [(1.0, 2.0), (2.0, 1.0), (1e-8, 1.0), (1.0, 1e-8),
                         (0.5, 0.5), (3.0, 7.0)]:
        d = psg.relative_skill(sse_m, sse_p)
        s = psg.skill_vs_persistence(sse_m, sse_p)
        assert d is not None and -1.0 <= d <= 1.0
        assert (d > 0) == (s > 0)
        assert (d < 0) == (s < 0)


def test_relative_skill_is_insensitive_to_the_denominator_scale():
    """The 16.4x denominator spread that makes `skill` unusable must not move `d`."""
    # same RATIO of errors, denominators three orders of magnitude apart
    a = psg.relative_skill(2.0e-5, 1.0e-5)
    b = psg.relative_skill(2.0e-2, 1.0e-2)
    assert a == pytest.approx(b, abs=1e-12)
    # while `skill` is identical in ratio terms but its ABSOLUTE error budget is not
    assert psg.skill_vs_persistence(2.0e-5, 1.0e-5) == pytest.approx(-1.0)


def test_r2_forms_agree_with_sse_forms():
    ss_tot = 100.0
    for m_r2, p_r2 in [(0.9, 0.99), (0.5, 0.2), (0.99, 0.9)]:
        sse_m = (1.0 - m_r2) * ss_tot
        sse_p = (1.0 - p_r2) * ss_tot
        assert psg.skill_from_r2(m_r2, p_r2) == pytest.approx(
            psg.skill_vs_persistence(sse_m, sse_p))
        assert psg.relative_skill_from_r2(m_r2, p_r2) == pytest.approx(
            psg.relative_skill(sse_m, sse_p))


# --------------------------------------------------------------------------- #
# 5. Three-valued status; cannot_determine is structural, never a bool.         #
# --------------------------------------------------------------------------- #

def _battery(model_r2, persistence_r2, n=256):
    return psg._synthetic_pairs(model_r2, persistence_r2, n=n)


def test_ready_when_the_head_genuinely_beats_persistence():
    pred, tgt, prev, _ = _battery(0.99, 0.60)
    v = psg.persistence_verdict(pred, tgt, prev, n_bootstrap=400)
    assert v.status == "ready"
    assert v.is_ready is True
    assert v.ci_low > 0.0


def test_thin_battery_is_cannot_determine_not_a_failure():
    pred, tgt, prev, _ = _battery(0.5, 0.9, n=8)
    v = psg.persistence_verdict(pred, tgt, prev)
    assert v.status == "cannot_determine"
    assert v.is_ready is False
    assert "too small" in v.reason


def test_motionless_target_is_cannot_determine():
    z = torch.ones(64, 3)
    v = psg.persistence_verdict(z.clone(), z, z.clone())
    assert v.status == "cannot_determine"
    assert "did not move" in v.reason


def test_row_count_mismatch_is_cannot_determine():
    g = torch.Generator(device="cpu"); g.manual_seed(11)
    tgt = torch.randn(64, 3, generator=g)
    v = psg.persistence_verdict(tgt.clone(), tgt, torch.randn(32, 3, generator=g))
    assert v.status == "cannot_determine"
    assert "mismatch" in v.reason


def test_near_tie_straddles_the_control_and_returns_cannot_determine():
    g = torch.Generator(device="cpu"); g.manual_seed(5)
    tgt = torch.randn(256, 3, generator=g)
    e1 = torch.randn(256, 3, generator=g) * 0.5
    e2 = torch.randn(256, 3, generator=g) * 0.5
    v = psg.persistence_verdict(tgt + e1, tgt, tgt + e2, n_bootstrap=600)
    assert v.status == "cannot_determine"
    assert v.ci_low < 0.0 < v.ci_high


def test_status_is_always_one_of_three_strings():
    for m, p in [(0.99, 0.5), (0.5, 0.99), (0.9, 0.9)]:
        pred, tgt, prev, _ = _battery(m, p)
        v = psg.persistence_verdict(pred, tgt, prev, n_bootstrap=300)
        assert v.status in ("ready", "persistence_dominated", "cannot_determine")
        assert not isinstance(v.status, bool)
        assert v.to_dict()["status"] == v.status


def test_is_ready_is_false_for_cannot_determine():
    pred, tgt, prev, _ = _battery(0.5, 0.9, n=8)
    assert psg.persistence_verdict(pred, tgt, prev).is_ready is False


# --------------------------------------------------------------------------- #
# 6. Printed denominator (remedy 3) and ASCII-only output.                      #
# --------------------------------------------------------------------------- #

def test_format_verdict_prints_the_denominators():
    pred, tgt, prev, _ = _battery(0.9, 0.99)
    out = psg.format_verdict(psg.persistence_verdict(pred, tgt, prev,
                                                     n_bootstrap=300), "cell0")
    for token in ("n_rows=", "sse_model=", "sse_persistence=", "sse_total=", "CI"):
        assert token in out


def test_all_printed_output_is_ascii():
    pred, tgt, prev, _ = _battery(0.9, 0.99)
    for v in (psg.persistence_verdict(pred, tgt, prev, n_bootstrap=200),
              psg.persistence_verdict(torch.ones(64, 2), torch.ones(64, 2),
                                      torch.ones(64, 2))):
        s = psg.format_verdict(v, "lbl")
        s.encode("ascii")            # raises if any non-ASCII reaches stdout
    src = open(psg.__file__, "r", encoding="utf-8").read()
    for i, line in enumerate(src.splitlines(), 1):
        if '"' in line or "'" in line:
            try:
                line.encode("ascii")
            except UnicodeEncodeError:            # pragma: no cover
                pytest.fail("non-ASCII on line %d of the gate module" % i)


# --------------------------------------------------------------------------- #
# 7. Determinism, and the optional headroom guard stays OFF by default.         #
# --------------------------------------------------------------------------- #

def test_verdict_is_deterministic_without_an_explicit_generator():
    pred, tgt, prev, _ = _battery(0.9, 0.99)
    a = psg.persistence_verdict(pred, tgt, prev, n_bootstrap=300)
    b = psg.persistence_verdict(pred, tgt, prev, n_bootstrap=300)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)


def test_headroom_guard_is_off_by_default_and_gates_when_asked():
    pred, tgt, prev, _ = _battery(0.90, 0.9999)
    off = psg.persistence_verdict(pred, tgt, prev, n_bootstrap=300)
    assert off.status == "persistence_dominated"        # default: no extra gate
    on = psg.persistence_verdict(pred, tgt, prev, n_bootstrap=300,
                                 min_persistence_headroom=0.01)
    assert on.status == "cannot_determine"
    assert "headroom" in on.reason


def test_undefined_statistics_return_none_not_zero():
    assert psg.skill_vs_persistence(1.0, 0.0) is None
    assert psg.relative_skill(0.0, 0.0) is None
    assert psg.skill_from_r2(0.5, 1.0) is None
    assert psg.skill_vs_persistence(float("nan"), 1.0) is None


def test_relative_form_is_defined_where_the_ratio_form_is_not():
    """The deliberate asymmetry, and the reason the gate routes on the relative form.

    At persistence_r2 == 1.0 the baseline is exact: `skill_from_r2` divides by
    zero, while the relative form correctly reports total domination.
    """
    assert psg.skill_from_r2(0.5, 1.0) is None
    assert psg.relative_skill_from_r2(0.5, 1.0) == pytest.approx(-1.0)
    assert psg.relative_skill(1.0, 0.0) == pytest.approx(-1.0)
    assert psg.relative_skill(0.0, 1.0) == pytest.approx(1.0)
