"""Contract tests for SD-091 scale-relative recruitment threshold (2026-09-17).

THE DEFECT THIS BUILD ADDRESSES
-------------------------------
The endogenous coalition trigger compares the previous tick's E3 candidate-score
margin (sorted[1] - sorted[0]) against an ABSOLUTE constant,
`endogenous_coalition_margin_threshold = 0.05`. E3 score magnitude is not
commensurable across seeds, so the gate is scale-sensitive. V3-EXQ-1038a measured
it: even with the commensurability operator ON -- which pulls margin MAGNITUDE
spread from 783.8x to 4.7x -- the fixed gate still leaves cross-seed recruitment
at CV 0.449 / spread 5.48x (one seed recruits on 14.7% of ticks, another on 2.7%).
Its pre-registered post-hoc re-thresholded the banked samples at
`0.05 * (cell_median / pooled_median)` and got CV 0.124 / spread 1.45x.

WHY A RUNNING ESTIMATOR SUFFICES (the thing that could have defeated this build)
-------------------------------------------------------------------------------
`pooled_median` is a cross-seed quantity, but it is ONE CONSTANT shared by every
cell, so the post-hoc threshold is algebraically

    0.05 * (cell_median / pooled_median) == (0.05 / pooled_median) * cell_median

i.e. a per-stream scale statistic times a fixed calibration constant. Replaying
the banked V3-EXQ-1038a ON-arm samples through a running single-stream window
median reproduced the collapse at every window from 25 to 500 (W=200/warmup=100:
CV 0.105, spread 1.347, against the target CV ~0.124 / spread ~1.45).

THE LEVER
---------
`REEConfig.endogenous_coalition_threshold_mode: str = "absolute"` (bit-identical
off; the whole trigger is additionally behind `use_endogenous_coalition_trigger`,
itself False by default).

CONTRACTS
  S1  defaults -- mode "absolute", and the three scale knobs carry the
      documented defaults.
  S2  OFF bit-identity -- in "absolute" mode the helper returns the config
      constant for every input and touches NO state (empty window, zero
      counters), so the call site is arithmetically the pre-build one.
  S3  warmup -- in "scale_relative" mode the absolute threshold applies until
      `scale_warmup` prior margins exist.
  S4  the arithmetic -- after warmup the threshold is exactly
      base * median(window) / reference, computed on strictly PRIOR margins.
  S5  THE PROPERTY THE BUILD EXISTS FOR -- two streams whose margin scales
      differ 5x produce ~5x different eligibility under the absolute gate and
      near-equal eligibility under the scale-relative gate.
  S6  reference calibration -- when the running scale equals the reference, the
      scale-relative threshold reduces EXACTLY to the absolute one.
  S7  cross-episode persistence -- reset() clears the per-trial request counter
      but NOT the margin-scale window.
  S8  validation -- a bogus mode and a non-positive window raise at
      construction; a non-positive reference falls back to absolute rather
      than dividing by zero.
  S9  end-to-end -- an agent with the trigger ON runs ticks in both modes
      without error, and only "scale_relative" fills the window.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

BASE = 0.05
REF = 0.44036865234375


def _build(seed: int = 7, **flags):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return agent, b, w


def _stream(median_scale: float, n: int = 600, seed: int = 0):
    """Deterministic positive-skewed margin stream with a known median."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, generator=g)
    # exponential-ish shape; median of -ln(1-u) is ln(2)
    vals = (-(1.0 - x).log()) * (median_scale / 0.6931471805599453)
    return [float(v) for v in vals]


def _eligibility(agent, samples, skip: int = 0):
    """Fraction of samples below the threshold the agent would apply.

    `skip` excludes the first N decisions from the FRACTION but still feeds
    them to the estimator -- used to measure the post-warmup regime, since
    warmup ticks legitimately use the absolute threshold (pinned by S3).
    """
    hits = 0
    counted = 0
    for i, m in enumerate(samples):
        thr = agent._endogenous_coalition_effective_threshold(m)
        if i >= skip:
            counted += 1
            if m < thr:
                hits += 1
    return hits / max(1, counted)


# ----------------------------------------------------------------------
# S1 defaults
# ----------------------------------------------------------------------
def test_s1_defaults():
    cfg = REEConfig()
    assert cfg.endogenous_coalition_threshold_mode == "absolute"
    assert cfg.endogenous_coalition_scale_window == 200
    assert cfg.endogenous_coalition_scale_warmup == 100
    assert cfg.endogenous_coalition_scale_reference == REF
    assert cfg.use_endogenous_coalition_trigger is False


# ----------------------------------------------------------------------
# S2 OFF bit-identity
# ----------------------------------------------------------------------
def test_s2_absolute_mode_returns_the_constant_and_touches_no_state():
    agent, _b, _w = _build()
    assert agent._endogenous_coalition_threshold_mode == "absolute"
    for m in _stream(2.0, n=300):
        assert agent._endogenous_coalition_effective_threshold(m) == BASE
    assert len(agent._endogenous_coalition_margin_window) == 0
    assert agent._endogenous_coalition_scale_relative_ticks == 0


def test_s2_absolute_mode_ignores_the_scale_knobs():
    agent, _b, _w = _build(
        endogenous_coalition_scale_window=5,
        endogenous_coalition_scale_warmup=1,
        endogenous_coalition_scale_reference=0.001,
    )
    for m in _stream(9.0, n=50):
        assert agent._endogenous_coalition_effective_threshold(m) == BASE


# ----------------------------------------------------------------------
# S3 warmup
# ----------------------------------------------------------------------
def test_s3_absolute_until_warmup_then_scale_relative():
    agent, _b, _w = _build(
        endogenous_coalition_threshold_mode="scale_relative",
        endogenous_coalition_scale_window=50,
        endogenous_coalition_scale_warmup=10,
    )
    samples = _stream(2.0, n=30)
    for i, m in enumerate(samples):
        thr = agent._endogenous_coalition_effective_threshold(m)
        if i < 10:
            assert thr == BASE, f"tick {i} must still be absolute"
    assert agent._endogenous_coalition_scale_relative_ticks == len(samples) - 10


# ----------------------------------------------------------------------
# S4 the arithmetic, on strictly prior margins
# ----------------------------------------------------------------------
def test_s4_threshold_is_base_times_prior_median_over_reference():
    agent, _b, _w = _build(
        endogenous_coalition_threshold_mode="scale_relative",
        endogenous_coalition_scale_window=1000,
        endogenous_coalition_scale_warmup=4,
    )
    samples = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 7.0]
    prior = []
    for m in samples:
        thr = agent._endogenous_coalition_effective_threshold(m)
        if len(prior) >= 4:
            expected = BASE * (statistics.median(prior) / REF)
            assert thr == pytest.approx(expected, rel=1e-12), (
                "threshold must use the median of STRICTLY PRIOR margins"
            )
        else:
            assert thr == BASE
        prior.append(m)


def test_s4_window_is_bounded_by_scale_window():
    agent, _b, _w = _build(
        endogenous_coalition_threshold_mode="scale_relative",
        endogenous_coalition_scale_window=17,
        endogenous_coalition_scale_warmup=2,
    )
    for m in _stream(1.0, n=200):
        agent._endogenous_coalition_effective_threshold(m)
    assert len(agent._endogenous_coalition_margin_window) == 17


# ----------------------------------------------------------------------
# S5 the property the build exists for
# ----------------------------------------------------------------------
_REL_KW = dict(
    endogenous_coalition_threshold_mode="scale_relative",
    endogenous_coalition_scale_window=200,
    endogenous_coalition_scale_warmup=100,
)


def _spread(a, b):
    return max(a, b) / max(min(a, b), 1e-12)


def test_s5_scale_equivariance_on_a_paired_5x_stream():
    """Same samples, 5x the scale: the gate must fire on the SAME ticks.

    lo and hi share a generator seed, so hi[i] == 5 * lo[i] exactly. The
    absolute gate therefore has to diverge and the scale-relative gate has to
    be (near-)exactly equivariant -- no sampling noise in the comparison.
    """
    lo = _stream(REF, n=600, seed=1)
    hi = _stream(REF * 5.0, n=600, seed=1)
    warm = _REL_KW["endogenous_coalition_scale_warmup"]

    abs_lo = _eligibility(
        _build(endogenous_coalition_threshold_mode="absolute")[0], lo, skip=warm
    )
    abs_hi = _eligibility(
        _build(endogenous_coalition_threshold_mode="absolute")[0], hi, skip=warm
    )
    rel_lo = _eligibility(_build(**_REL_KW)[0], lo, skip=warm)
    rel_hi = _eligibility(_build(**_REL_KW)[0], hi, skip=warm)

    assert _spread(abs_lo, abs_hi) > 3.0, (
        f"absolute gate must stay scale-sensitive (lo={abs_lo:.4f} "
        f"hi={abs_hi:.4f})"
    )
    assert rel_lo == rel_hi, (
        "post-warmup the scale-relative gate must fire on exactly the same "
        f"ticks at both scales (lo={rel_lo:.5f} hi={rel_hi:.5f})"
    )


def test_s5_collapses_spread_across_INDEPENDENT_streams():
    """The same property without the pairing, so sampling noise is in play.

    Five independently drawn streams whose scales span 8x. This is the shape
    of the V3-EXQ-1038a cross-seed measurement (CV 0.449 / spread 5.48x under
    the fixed gate, CV 0.124 / spread 1.45x under the post-hoc's
    scale-relative one).
    """
    scales = [REF * m for m in (0.5, 1.0, 2.0, 3.0, 4.0)]
    streams = [_stream(sc, n=800, seed=100 + i) for i, sc in enumerate(scales)]

    abs_e = [
        _eligibility(_build(endogenous_coalition_threshold_mode="absolute")[0], s)
        for s in streams
    ]
    rel_e = [_eligibility(_build(**_REL_KW)[0], s) for s in streams]

    abs_spread = max(abs_e) / max(min(abs_e), 1e-12)
    rel_spread = max(rel_e) / max(min(rel_e), 1e-12)

    def _cv(v):
        m = sum(v) / len(v)
        var = sum((x - m) ** 2 for x in v) / len(v)
        return (var ** 0.5) / m

    assert abs_spread > 4.0, f"fixed gate spread was {abs_spread:.3f}: {abs_e}"
    assert rel_spread < 1.5, (
        f"scale-relative gate spread was {rel_spread:.3f}: {rel_e}"
    )
    assert _cv(rel_e) < 0.13, f"scale-relative CV was {_cv(rel_e):.4f}: {rel_e}"
    assert rel_spread < abs_spread


# ----------------------------------------------------------------------
# S6 reference calibration
# ----------------------------------------------------------------------
def test_s6_reduces_to_absolute_when_scale_equals_reference():
    agent, _b, _w = _build(
        endogenous_coalition_threshold_mode="scale_relative",
        endogenous_coalition_scale_window=10,
        endogenous_coalition_scale_warmup=3,
    )
    for _ in range(5):
        agent._endogenous_coalition_effective_threshold(REF)
    assert agent._endogenous_coalition_effective_threshold(REF) == pytest.approx(
        BASE, rel=1e-12
    )


# ----------------------------------------------------------------------
# S7 cross-episode persistence
# ----------------------------------------------------------------------
def test_s7_reset_clears_the_counter_but_not_the_scale_window():
    agent, _b, _w = _build(
        endogenous_coalition_threshold_mode="scale_relative",
        endogenous_coalition_scale_window=50,
        endogenous_coalition_scale_warmup=5,
    )
    for m in _stream(1.0, n=40):
        agent._endogenous_coalition_effective_threshold(m)
    agent._endogenous_coalition_request_count = 3
    n_before = len(agent._endogenous_coalition_margin_window)
    assert n_before > 0

    agent.reset()

    assert agent._endogenous_coalition_request_count == 0
    assert len(agent._endogenous_coalition_margin_window) == n_before, (
        "the margin-scale estimator must survive an episode boundary -- the "
        "post-hoc it reproduces pooled all 20 episodes of a cell"
    )


# ----------------------------------------------------------------------
# S8 validation
# ----------------------------------------------------------------------
def test_s8_bogus_mode_raises():
    with pytest.raises(ValueError, match="endogenous_coalition_threshold_mode"):
        _build(endogenous_coalition_threshold_mode="nonsense")


@pytest.mark.parametrize("bad", [0, -1])
def test_s8_non_positive_window_raises(bad):
    with pytest.raises(ValueError, match="endogenous_coalition_scale_window"):
        _build(endogenous_coalition_scale_window=bad)


def test_s8_non_positive_reference_falls_back_to_absolute():
    agent, _b, _w = _build(
        endogenous_coalition_threshold_mode="scale_relative",
        endogenous_coalition_scale_window=20,
        endogenous_coalition_scale_warmup=2,
        endogenous_coalition_scale_reference=0.0,
    )
    for m in _stream(1.0, n=30):
        assert agent._endogenous_coalition_effective_threshold(m) == BASE


# ----------------------------------------------------------------------
# S9 end-to-end
# ----------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["absolute", "scale_relative"])
def test_s9_agent_runs_with_the_trigger_on(mode):
    agent, b, w = _build(
        use_coalition_controller=True,
        use_endogenous_coalition_trigger=True,
        endogenous_coalition_threshold_mode=mode,
        endogenous_coalition_scale_window=50,
        endogenous_coalition_scale_warmup=5,
    )
    for _ in range(25):
        with torch.no_grad():
            agent.act_with_split_obs(b, w)
    filled = len(agent._endogenous_coalition_margin_window)
    if mode == "scale_relative":
        assert filled > 0, "the live trigger path must feed the estimator"
    else:
        assert filled == 0
