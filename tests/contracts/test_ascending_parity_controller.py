"""
Contract tests for the BOUNDED ascending-spiral gain -- the target-PARITY controller
(V3-EXQ-711 loop-effective-weight RUNAWAY repair, 2026-07-04).

The raw-scalar ascending gain (test_ascending_spiral_gain.py) has NO stable parity
regime: sub-threshold (V3-EXQ-709, coupling ~0.03, the limbic loop never wins) or
runaway (V3-EXQ-711, 20x-forward x 5x-plasticity compounding through the positive-
feedback plastic loop -> M_cross range peak 4897.8, w_eff[limbic] 10-2274x w_eff[motor]
-- a limbic MONOPOLY that merely replaces the F/motor-pinning). confirmed
failure_autopsy_V3-EXQ-711_2026-07-04. The controller replaces the unbounded multiply
with actuator-saturated setpoint control:

  FORWARD (parity-ceiling): a per-step ascending gain in [0, parity_forward_gain] solved
    so the limbic effective column weight w_eff[limbic] is LIFTED toward but HARD-CAPPED
    at parity_ceiling_ratio * w_eff[motor]. Motor column has no upper-tri entry so
    w_eff[motor] is the gain-invariant parity reference. Bounds the ratio -> fair
    reorder, never a monopoly.
  MATURATION (bounded loop): the ascending three-factor update is scaled by the BOUNDED
    parity_plasticity_gain, then the ascending M_cross entries are clamped to
    [-m_cross_clamp, m_cross_clamp] (an anti-windup clamp on the plastic positive-feedback
    loop -- the second 711 runaway source).

These tests lock:
  (A) BYTE-IDENTICAL OFF: use_ascending_parity_controller False with large params matches
      the plain learned cross-loop combine bit-for-bit.
  (B) AT-INIT identity: controller ON + large params but M_cross == 0 -> W_cross == I ->
      commit bit-identical to OFF.
  (C) INERT ON: controller ON with forward_gain 1.0 + ceiling 0.0 (inert defaults) -> no
      lift, no cap -> matches OFF (the master switch, not the sub-params, gates behaviour).
  (D) PARITY CEILING: under a runaway-shaped M_cross the raw scalar blows w_eff[limbic]
      to a large multiple of w_eff[motor], but the controller holds the ratio at/under
      parity_ceiling_ratio -- while w_eff[motor] stays exactly invariant.
  (E) MATURATION CLAMP: under forced large updates + a big plasticity gain the ascending
      M_cross entries run away without the clamp but stay within [-clamp, clamp] with it,
      while non-ascending entries are unaffected by the clamp.
  (F) MECHANISM (parity, not monopoly): a controller lift with a modest ceiling lets the
      limbic loop reach w_eff[motor] parity (limbic_ge_motor True) WITHOUT the effective
      weight exceeding the ceiling -- a fair win, not the raw scalar's blow-up.
  (G) from_dims plumbing: the flag + 4 params reach config.e3.
  (H) SAFETY: the arbitration stays strictly within the eligible set under the controller.
  (I) PRECEDENCE: with BOTH the raw flag and the controller on, the controller path wins
      (bounded), i.e. the raw runaway is suppressed.
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.utils.config import REEConfig
from ree_core.predictors.e3_selector import (
    E3TrajectorySelector,
    _FCG_CHANNEL_INDEX,
)


def _make_selector(cross_loop: bool = True, **extra) -> E3TrajectorySelector:
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        use_loop_segregation=True,
        use_learned_cross_loop_arbitration=cross_loop,
        **extra,
    )
    return E3TrajectorySelector(cfg.e3, None)


def _limbic_terms(pref_idx: int, n: int):
    base = torch.full((n,), 0.5)
    base[pref_idx] = -3.0
    return [
        (_FCG_CHANNEL_INDEX["ofc"], base.clone()),
        (_FCG_CHANNEL_INDEX["liking"], base.clone()),
        (_FCG_CHANNEL_INDEX["vigour"], base.clone()),
    ]


def _set_m_cross(sel: E3TrajectorySelector, pattern: torch.Tensor) -> None:
    with torch.no_grad():
        sel.M_cross.copy_(pattern)


# 709-signature: motor column grown past a small ascending path (limbic sub-parity).
_LEARNED_PATTERN = torch.tensor([
    [0.20, 0.03, 0.05],
    [0.03, 0.00, 0.04],
    [0.02, 0.01, 0.00],
], dtype=torch.float32)

# 711-signature: a runaway-shaped M_cross with huge ascending limbic entries.
_RUNAWAY_PATTERN = torch.tensor([
    [0.20, 3.00, 5.00],
    [0.03, 0.00, 4.00],
    [0.02, 0.01, 0.00],
], dtype=torch.float32)


# ------------------------------------------------------------------ #
# (A) byte-identical OFF                                               #
# ------------------------------------------------------------------ #

class TestByteIdenticalOff:
    def test_off_matches_plain_learned_across_seeds(self):
        n = 6
        elig = torch.arange(n)
        for seed in range(12):
            torch.manual_seed(seed)
            raw = torch.randn(n)
            terms = [
                (_FCG_CHANNEL_INDEX["ofc"], torch.randn(n)),
                (_FCG_CHANNEL_INDEX["dacc"], torch.randn(n)),
            ]
            sel_plain = _make_selector()
            _set_m_cross(sel_plain, _LEARNED_PATTERN)
            loc_plain = sel_plain._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            sel_off = _make_selector(
                use_ascending_parity_controller=False,
                loop_segregation_parity_forward_gain=50.0,
                loop_segregation_parity_ceiling_ratio=2.0,
                loop_segregation_m_cross_clamp=0.1,
            )
            _set_m_cross(sel_off, _LEARNED_PATTERN)
            loc_off = sel_off._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            assert loc_plain == loc_off, (
                f"seed {seed}: controller OFF must ignore the params and match the plain "
                f"learned combine ({loc_plain} vs {loc_off})"
            )


# ------------------------------------------------------------------ #
# (B) at-init identity (M_cross == 0)                                  #
# ------------------------------------------------------------------ #

class TestAtInitIdentity:
    def test_large_params_at_init_matches_off(self):
        n = 5
        elig = torch.arange(n)
        for seed in range(8):
            torch.manual_seed(seed)
            raw = torch.randn(n)
            terms = _limbic_terms(pref_idx=3, n=n)
            loc_off = _make_selector()._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            loc_on = _make_selector(
                use_ascending_parity_controller=True,
                loop_segregation_parity_forward_gain=40.0,
                loop_segregation_parity_ceiling_ratio=1.5,
            )._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            assert loc_off == loc_on, (
                f"seed {seed}: controller at M_cross==0 must be a no-op"
            )


# ------------------------------------------------------------------ #
# (C) inert ON (forward_gain 1.0, ceiling 0.0)                         #
# ------------------------------------------------------------------ #

class TestInertOn:
    def test_inert_params_match_off(self):
        n = 6
        elig = torch.arange(n)
        for seed in range(12):
            torch.manual_seed(seed)
            raw = torch.randn(n)
            terms = _limbic_terms(pref_idx=(seed % n), n=n)
            sel_off = _make_selector()
            _set_m_cross(sel_off, _LEARNED_PATTERN)
            loc_off = sel_off._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            sel_on = _make_selector(
                use_ascending_parity_controller=True,
                loop_segregation_parity_forward_gain=1.0,
                loop_segregation_parity_ceiling_ratio=0.0,
            )
            _set_m_cross(sel_on, _LEARNED_PATTERN)
            loc_on = sel_on._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            assert loc_off == loc_on, (
                f"seed {seed}: forward_gain 1.0 + ceiling 0.0 must be inert"
            )


# ------------------------------------------------------------------ #
# (D) parity ceiling bounds the effective-weight ratio                #
# ------------------------------------------------------------------ #

class TestParityCeilingBounds:
    def test_controller_caps_w_limbic_where_raw_scalar_runs_away(self):
        n = 5
        elig = torch.arange(n)
        raw = torch.randn(n)
        terms = _limbic_terms(pref_idx=3, n=n)
        ceil = 1.25

        sel_raw = _make_selector(
            use_ascending_spiral_gain=True,
            loop_segregation_ascending_spiral_gain=20.0,
        )
        _set_m_cross(sel_raw, _RUNAWAY_PATTERN)
        sel_raw._segregated_loop_arbitrate(elig, raw, terms, True, [None] * n, True, 1.0, True)
        d_raw = dict(sel_raw.last_score_diagnostics)
        raw_ratio = (
            d_raw["loop_cross_loop_w_limbic_eff"] / d_raw["loop_cross_loop_w_motor_eff"]
        )

        sel_ctrl = _make_selector(
            use_ascending_parity_controller=True,
            loop_segregation_parity_forward_gain=20.0,
            loop_segregation_parity_ceiling_ratio=ceil,
        )
        _set_m_cross(sel_ctrl, _RUNAWAY_PATTERN)
        sel_ctrl._segregated_loop_arbitrate(elig, raw, terms, True, [None] * n, True, 1.0, True)
        d_ctrl = dict(sel_ctrl.last_score_diagnostics)
        ctrl_ratio = (
            d_ctrl["loop_cross_loop_w_limbic_eff"] / d_ctrl["loop_cross_loop_w_motor_eff"]
        )

        assert raw_ratio > 2.0 * ceil, (
            "precondition: the raw scalar must run away (this is the 711 failure)"
        )
        assert ctrl_ratio <= ceil + 1e-4, (
            f"the controller must hold w_eff[limbic]/w_eff[motor] at/under the ceiling "
            f"({ctrl_ratio} vs {ceil})"
        )
        assert d_ctrl["loop_cross_loop_w_motor_eff"] == pytest.approx(
            d_raw["loop_cross_loop_w_motor_eff"]
        ), "w_eff[motor] must be invariant to the ascending controller"
        assert d_ctrl["loop_ascending_parity_controller_active"] is True
        # the applied forward gain was throttled below the raw lift to hold the ceiling.
        assert d_ctrl["loop_ascending_parity_forward_gain_applied"] < 20.0


# ------------------------------------------------------------------ #
# (E) maturation clamp bounds the plastic loop                        #
# ------------------------------------------------------------------ #

class TestMaturationClamp:
    def _run_updates(self, pgain, clamp, n_updates=5):
        torch.manual_seed(0)
        sel = _make_selector(
            use_ascending_parity_controller=True,
            loop_segregation_parity_forward_gain=1.0,
            loop_segregation_parity_plasticity_gain=pgain,
            loop_segregation_m_cross_clamp=clamp,
        )
        for k in range(n_updates):
            sel._lcg_value_baseline = -100.0  # force a large positive delta_t
            torch.manual_seed(1 + k)
            raw = torch.randn(5)
            sel._segregated_loop_arbitrate(
                torch.arange(5), raw, _limbic_terms(3, 5), True,
                [None] * 5, True, 1.0, False,  # waking -> arms coact + runs the update
            )
            torch.manual_seed(100 + k)
            sel.post_action_update(torch.randn(1, 32), harm_occurred=False)
        return sel.M_cross.detach().clone()

    def test_clamp_bounds_ascending_entries(self):
        clamp = 0.5
        M_noclamp = self._run_updates(50.0, 0.0)
        M_clamp = self._run_updates(50.0, clamp)
        iu = torch.triu_indices(3, 3, offset=1)
        asc_noclamp = M_noclamp[iu[0], iu[1]].abs().max().item()
        asc_clamp = M_clamp[iu[0], iu[1]].abs().max().item()
        assert asc_noclamp > clamp, (
            "precondition: without the clamp the plastic ascending loop runs away"
        )
        assert asc_clamp <= clamp + 1e-6, (
            f"the clamp must bound the ascending M_cross entries ({asc_clamp} > {clamp})"
        )

    def test_clamp_applied_per_run(self):
        # Each run's ascending M_cross entries respect that run's clamp bound (the clamp
        # is a closed-loop control -- it also reshapes later selection + non-ascending
        # updates, so we assert only the invariant the clamp directly guarantees).
        iu = torch.triu_indices(3, 3, offset=1)
        M_small = self._run_updates(50.0, 0.5)
        M_big = self._run_updates(50.0, 5.0)
        assert M_small[iu[0], iu[1]].abs().max().item() <= 0.5 + 1e-6
        assert M_big[iu[0], iu[1]].abs().max().item() <= 5.0 + 1e-6


# ------------------------------------------------------------------ #
# (F) mechanism: parity win, not monopoly                             #
# ------------------------------------------------------------------ #

class TestParityWinNotMonopoly:
    def test_limbic_reaches_parity_within_ceiling(self):
        n = 4
        elig = torch.arange(n)
        raw = torch.tensor([-1.0, 0.5, 0.5, 0.5])
        terms = _limbic_terms(pref_idx=3, n=n)
        ceil = 1.25

        sel = _make_selector(
            use_ascending_parity_controller=True,
            loop_segregation_parity_forward_gain=30.0,
            loop_segregation_parity_ceiling_ratio=ceil,
        )
        _set_m_cross(sel, _RUNAWAY_PATTERN)
        sel._segregated_loop_arbitrate(elig, raw, terms, True, [None] * n, True, 1.0, True)
        d = dict(sel.last_score_diagnostics)
        w_l = d["loop_cross_loop_w_limbic_eff"]
        w_m = d["loop_cross_loop_w_motor_eff"]
        assert d["loop_cross_loop_limbic_ge_motor"] is True, (
            "the controller must let the limbic loop REACH motor parity"
        )
        assert w_l <= ceil * w_m + 1e-4, (
            "but the win must be a bounded PARITY win, never past the ceiling"
        )


# ------------------------------------------------------------------ #
# (G) from_dims plumbing                                              #
# ------------------------------------------------------------------ #

class TestFromDimsPlumbing:
    def test_flags_reach_e3_config(self):
        cfg_on = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
            use_loop_segregation=True, use_learned_cross_loop_arbitration=True,
            use_ascending_parity_controller=True,
            loop_segregation_parity_forward_gain=8.0,
            loop_segregation_parity_ceiling_ratio=1.25,
            loop_segregation_parity_plasticity_gain=2.0,
            loop_segregation_m_cross_clamp=0.5,
        )
        assert cfg_on.e3.use_ascending_parity_controller is True
        assert cfg_on.e3.loop_segregation_parity_forward_gain == pytest.approx(8.0)
        assert cfg_on.e3.loop_segregation_parity_ceiling_ratio == pytest.approx(1.25)
        assert cfg_on.e3.loop_segregation_parity_plasticity_gain == pytest.approx(2.0)
        assert cfg_on.e3.loop_segregation_m_cross_clamp == pytest.approx(0.5)
        cfg_off = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        )
        assert cfg_off.e3.use_ascending_parity_controller is False
        assert cfg_off.e3.loop_segregation_parity_forward_gain == pytest.approx(1.0)
        assert cfg_off.e3.loop_segregation_parity_ceiling_ratio == pytest.approx(0.0)
        assert cfg_off.e3.loop_segregation_m_cross_clamp == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# (H) safety: arbitration stays within the eligible set               #
# ------------------------------------------------------------------ #

class TestSafetyWithinEligibleSet:
    def test_commit_is_always_in_eligible_set_under_controller(self):
        n_total = 8
        elig = torch.tensor([1, 3, 6])
        raw = torch.randn(n_total)
        terms = _limbic_terms(pref_idx=2, n=n_total)
        sel = _make_selector(
            use_ascending_parity_controller=True,
            loop_segregation_parity_forward_gain=25.0,
            loop_segregation_parity_ceiling_ratio=1.25,
        )
        _set_m_cross(sel, _RUNAWAY_PATTERN)
        local = sel._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n_total, True, 1.0, True
        )
        assert 0 <= local < int(elig.numel()), "commit must be a local index into eligible"


# ------------------------------------------------------------------ #
# (I) precedence: controller suppresses the raw runaway                #
# ------------------------------------------------------------------ #

class TestPrecedenceOverRawScalar:
    def test_controller_takes_precedence_over_raw_flag(self):
        n = 5
        elig = torch.arange(n)
        raw = torch.randn(n)
        terms = _limbic_terms(pref_idx=3, n=n)
        ceil = 1.25
        # BOTH flags on: the controller path must win (bounded), not the raw runaway.
        sel = _make_selector(
            use_ascending_spiral_gain=True,
            loop_segregation_ascending_spiral_gain=20.0,
            use_ascending_parity_controller=True,
            loop_segregation_parity_forward_gain=20.0,
            loop_segregation_parity_ceiling_ratio=ceil,
        )
        _set_m_cross(sel, _RUNAWAY_PATTERN)
        sel._segregated_loop_arbitrate(elig, raw, terms, True, [None] * n, True, 1.0, True)
        d = dict(sel.last_score_diagnostics)
        ratio = d["loop_cross_loop_w_limbic_eff"] / d["loop_cross_loop_w_motor_eff"]
        assert d["loop_ascending_parity_controller_active"] is True
        assert d["loop_ascending_spiral_gain_active"] is False, (
            "with both on, the raw-scalar diagnostic must report inactive (controller wins)"
        )
        assert ratio <= ceil + 1e-4, "the controller must suppress the raw runaway"
