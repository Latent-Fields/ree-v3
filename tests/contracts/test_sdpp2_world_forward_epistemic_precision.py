"""Contract tests for SD-PP-2 `precision.world_forward_epistemic_precision`.

Contract:
REE_assembly/docs/architecture/precision_provenance_substrate_spec.md section 3.

The eight contracts, in the spec's own order:

  T1  default OFF -- `use_world_forward_epistemic_precision` defaults False.
  T2  EMA maths exact -- v_tot / v_noise match a hand-rolled EMA on synthetic pe.
  T3  v_epi is floored at `v_floor` when the noise EMA overtakes v_tot.
  T4  noise split -- with evidence_variance_z = v_tot / (2 * noise_gain),
      v_epi is exactly half what it is with no evidence noise.
  T5  NO FUTURE INFO -- a `precision_at` read taken before
      `observe_outcome(pe=huge)` is identical to one taken with no observation
      at all, and the read AFTER differs. This is the ordering contract that
      makes "historical precision" mean anything.
  T6  sd063 fallback -- stub head with training_ready False -> source "ema";
      True -> source "sd063" and v_tot == head.predictive_variance(...).
  T7  NO RNG consumption -- torch.get_rng_state() byte-identical across 100
      precision_at + observe_outcome calls.
  T8  `prediction_at_test` equals `e2.world_forward` bitwise and carries no grad
      (built against a real E2FastPredictor, world_dim 16 / action_dim 4).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.precision.world_forward_epistemic_precision import (  # noqa: E402
    PrecisionRead,
    WorldForwardEpistemicPrecision,
    WorldForwardEpistemicPrecisionConfig,
)


D = 8
A = 4


def _mk(**kw) -> WorldForwardEpistemicPrecision:
    cfg = WorldForwardEpistemicPrecisionConfig(**kw)
    return WorldForwardEpistemicPrecision(cfg, world_dim=D)


def _pair(c: float):
    """A (pred, z_now) pair whose per-dim mean squared error is exactly c*c."""
    return torch.full((1, D), float(c)), torch.zeros(1, D)


class _StubHead:
    """Minimal stand-in for the SD-063 E2WorldUncertaintyHead read surface."""

    def __init__(self, ready: bool, value: float) -> None:
        self.training_ready = bool(ready)
        self._value = float(value)
        self.n_calls = 0

    def predictive_variance(self, z_world: torch.Tensor, action: torch.Tensor):
        self.n_calls += 1
        return torch.full((z_world.shape[0],), self._value)


# ---------------------------------------------------------------- T1
def test_t1_default_off():
    cfg = WorldForwardEpistemicPrecisionConfig()
    assert cfg.use_world_forward_epistemic_precision is False
    # The rest of the pre-registered defaults, pinned so a drift is a red test.
    assert cfg.source == "sd063_or_ema"
    assert cfg.pe_ema_alpha == 0.05
    assert cfg.v_floor == 1e-6
    assert cfg.noise_gain == 2.0
    assert cfg.v_init == 1e-2

    # Config validation (__post_init__).
    for bad in ({"source": "nope"}, {"pe_ema_alpha": 0.0}, {"pe_ema_alpha": 1.5},
                {"v_floor": 0.0}, {"noise_gain": -0.1}, {"v_init": 0.0}):
        with pytest.raises(ValueError):
            WorldForwardEpistemicPrecisionConfig(**bad)


# ---------------------------------------------------------------- T2
def test_t2_ema_maths_exact_against_hand_rolled():
    alpha = 0.05
    gain = 2.0
    est = _mk(pe_ema_alpha=alpha, noise_gain=gain, v_init=1e-2)

    v_tot = 1e-2          # v_init
    v_noise = 0.0
    evs = [0.0, 1e-4, 5e-4, 0.0, 2e-4, 1e-3, 0.0, 3e-4]
    for i, c in enumerate([0.1, 0.2, 0.05, 0.3, 0.15, 0.4, 0.02, 0.25]):
        pred, z_now = _pair(c)
        ev = evs[i]
        pe = est.observe_outcome(pred, z_now, ev)
        # pe is exactly mean_d (pred - z_now)^2
        assert pe == pytest.approx(float(c) ** 2, rel=1e-6, abs=1e-12)
        v_tot = (1.0 - alpha) * v_tot + alpha * pe
        v_noise = (1.0 - alpha) * v_noise + alpha * (gain * ev)
        assert est.v_tot == pytest.approx(v_tot, rel=1e-12, abs=0.0)
        assert est.v_noise == pytest.approx(v_noise, rel=1e-12, abs=0.0)

    assert est.n_obs == 8
    expected_v_epi = max(v_tot - v_noise, 1e-6)
    assert est.v_epi == pytest.approx(expected_v_epi, rel=1e-12)
    assert est.pi_cur == pytest.approx(1.0 / expected_v_epi, rel=1e-12)

    read = est.current_read()
    assert read.source == "ema"
    assert read.v_tot == pytest.approx(v_tot, rel=1e-12)
    assert read.v_ale == pytest.approx(v_noise, rel=1e-12)
    assert read.v_epi == pytest.approx(expected_v_epi, rel=1e-12)
    assert read.pi_epi == pytest.approx(1.0 / expected_v_epi, rel=1e-12)

    snap = est.snapshot()
    assert set(snap) == {"v_tot", "v_noise", "v_epi", "pi_cur", "n_obs"}
    assert snap["n_obs"] == 8.0
    mets = est.get_metrics()
    assert set(mets) == {"wf_precision_" + k for k in snap}
    assert mets["wf_precision_v_tot"] == pytest.approx(v_tot, rel=1e-12)


# ---------------------------------------------------------------- T3
def test_t3_v_epi_floored_at_v_floor():
    floor = 1e-6
    # alpha = 1.0 makes each EMA equal its latest input exactly.
    est = _mk(pe_ema_alpha=1.0, noise_gain=2.0, v_floor=floor)
    pred, z_now = _pair(0.01)                    # pe = 1e-4
    # evidence variance big enough that noise_gain * ev >> pe
    est.observe_outcome(pred, z_now, 1.0)
    assert est.v_tot == pytest.approx(1e-4, rel=1e-6)
    assert est.v_noise == pytest.approx(2.0, rel=1e-12)
    assert est.v_epi == floor                    # floored, not negative
    assert est.pi_cur == pytest.approx(1.0 / floor, rel=1e-12)

    read = est.current_read()
    assert read.v_epi == floor
    assert read.pi_epi == pytest.approx(1.0 / floor, rel=1e-12)

    # The floor also applies to the per-state (sd063) branch.
    head = _StubHead(ready=True, value=1e-5)
    r = est.precision_at(torch.zeros(1, D), torch.zeros(1, A), head=head)
    assert r.source == "sd063"
    assert r.v_epi == floor


# ---------------------------------------------------------------- T4
def test_t4_noise_split_halves_v_epi():
    gain = 2.0
    # alpha = 1.0 so both EMAs land exactly on their inputs in one step.
    clean = _mk(pe_ema_alpha=1.0, noise_gain=gain)
    noisy = _mk(pe_ema_alpha=1.0, noise_gain=gain)

    pred, z_now = _pair(0.2)                     # pe = 0.04
    pe = clean.observe_outcome(pred, z_now, 0.0)
    # evidence_variance_z = v_tot / (2 * noise_gain)  ->  v_noise = v_tot / 2
    noisy.observe_outcome(pred, z_now, pe / (2.0 * gain))

    assert clean.v_epi == pytest.approx(pe, rel=1e-9)
    assert noisy.v_noise == pytest.approx(pe / 2.0, rel=1e-9)
    assert noisy.v_epi == pytest.approx(clean.v_epi / 2.0, rel=1e-9)
    # ... and precision therefore doubles.
    assert noisy.pi_cur == pytest.approx(2.0 * clean.pi_cur, rel=1e-9)


# ---------------------------------------------------------------- T5
def test_t5_no_future_info_in_the_historical_read():
    z = torch.zeros(1, D)
    a = torch.zeros(1, A)

    est = _mk()
    untouched = _mk()

    before = est.precision_at(z, a)
    # a read with NO observation at all, from an independent estimator
    none_at_all = untouched.precision_at(z, a)
    assert before == none_at_all
    assert isinstance(before, PrecisionRead)
    assert before.source == "ema"

    # precision_at must not mutate state: a second read is identical
    assert est.precision_at(z, a) == before
    assert est.n_obs == 0

    pred, z_now = _pair(10.0)                    # pe = 100.0 -- huge
    est.observe_outcome(pred, z_now, 0.0)
    after = est.precision_at(z, a)
    assert after != before
    assert after.v_tot > before.v_tot
    assert after.pi_epi < before.pi_epi
    # and the pre-outcome read is STILL what the untouched estimator says
    assert before == untouched.precision_at(z, a)


# ---------------------------------------------------------------- T6
def test_t6_sd063_fallback_and_selection():
    z = torch.zeros(1, D)
    a = torch.zeros(1, A)
    pvar = 0.25

    # default source: sd063_or_ema
    est = _mk()
    not_ready = _StubHead(ready=False, value=pvar)
    r = est.precision_at(z, a, head=not_ready)
    assert r.source == "ema"
    assert r.v_tot == pytest.approx(est.v_tot, rel=1e-12)

    ready = _StubHead(ready=True, value=pvar)
    r2 = est.precision_at(z, a, head=ready)
    assert r2.source == "sd063"
    assert r2.v_tot == pytest.approx(pvar, rel=1e-12)
    assert r2.v_ale == pytest.approx(est.v_noise, rel=1e-12)
    assert r2.v_epi == pytest.approx(pvar - est.v_noise, rel=1e-12)
    assert r2.pi_epi == pytest.approx(1.0 / r2.v_epi, rel=1e-12)

    # No head at all -> ema.
    assert est.precision_at(z, a, head=None).source == "ema"

    # source="sd063" with a missing or not-ready head ALSO falls back to ema.
    strict = _mk(source="sd063")
    assert strict.precision_at(z, a, head=None).source == "ema"
    assert strict.precision_at(z, a, head=_StubHead(False, pvar)).source == "ema"
    assert strict.precision_at(z, a, head=_StubHead(True, pvar)).source == "sd063"

    # source="ema" never consults the head, even a ready one.
    ema_only = _mk(source="ema")
    spy = _StubHead(ready=True, value=pvar)
    assert ema_only.precision_at(z, a, head=spy).source == "ema"
    assert spy.n_calls == 0

    # current_read() is ALWAYS the ema read.
    assert _mk(source="sd063").current_read().source == "ema"


# ---------------------------------------------------------------- T7
def test_t7_no_rng_consumption():
    torch.manual_seed(1234)
    est = _mk()
    head = _StubHead(ready=True, value=0.05)
    z = torch.ones(1, D) * 0.3
    a = torch.zeros(1, A)
    a[0, 1] = 1.0
    pred, z_now = _pair(0.1)

    state_before = torch.get_rng_state()
    for _ in range(100):
        est.precision_at(z, a, head=head)
        est.precision_at(z, a, head=None)
        est.observe_outcome(pred, z_now, 1e-4)
        est.current_read()
        est.snapshot()
        est.get_metrics()
    state_after = torch.get_rng_state()

    assert torch.equal(state_before, state_after)
    assert est.n_obs == 100


# ---------------------------------------------------------------- T8
def test_t8_prediction_at_test_matches_world_forward_bitwise():
    from ree_core.predictors.e2_fast import E2FastPredictor
    from ree_core.utils.config import E2Config

    torch.manual_seed(7)
    e2 = E2FastPredictor(E2Config(world_dim=16, action_dim=4))
    est = _mk()

    torch.manual_seed(11)
    z_prev = torch.randn(3, 16)
    a_onehot = torch.zeros(3, 4)
    a_onehot[0, 2] = 1.0
    a_onehot[1, 0] = 1.0
    a_onehot[2, 3] = 1.0

    out = est.prediction_at_test(e2, z_prev, a_onehot)
    expected = e2.world_forward(z_prev[:1], a_onehot[:1])

    assert out.shape == (1, 16)
    assert torch.equal(out, expected.detach())     # bitwise
    assert not out.requires_grad
    assert out.grad_fn is None

    # It reads only the FIRST row (the transition under test).
    assert torch.equal(out, e2.world_forward(z_prev[0:1], a_onehot[0:1]).detach())

    # 1-D inputs are accepted and give the same answer.
    out_1d = est.prediction_at_test(e2, z_prev[0], a_onehot[0])
    assert torch.equal(out_1d, out)

    # ... and the prediction is usable directly by observe_outcome.
    z_now = torch.randn(1, 16)
    pe = est.observe_outcome(out, z_now, 0.0)
    assert pe == pytest.approx(float(((out - z_now) ** 2).mean()), rel=1e-6)


def test_t9_get_set_state_round_trip():
    """Pre-freeze repair (V3-EXQ-1073): calibrated EMA state is portable."""
    import torch
    from ree_core.precision.world_forward_epistemic_precision import (
        WorldForwardEpistemicPrecision, WorldForwardEpistemicPrecisionConfig)
    cfg = WorldForwardEpistemicPrecisionConfig(
        use_world_forward_epistemic_precision=True, source="ema")
    a = WorldForwardEpistemicPrecision(cfg, world_dim=16)
    for i in range(30):
        pred = torch.zeros(1, 16); z = torch.full((1, 16), 0.01 * (i % 3))
        a.observe_outcome(pred, z, 1e-6)
    b = WorldForwardEpistemicPrecision(cfg, world_dim=16)
    b.set_state(a.get_state())
    assert b.current_read() == a.current_read()
    assert b.n_obs == a.n_obs and b.v_tot == a.v_tot and b.v_noise == a.v_noise
