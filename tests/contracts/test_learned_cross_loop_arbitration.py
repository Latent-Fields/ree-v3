"""
Contract tests for ARC-108 x ARC-110 LEARNED (dopamine-gated) CROSS-LOOP
arbitration -- the named next attack on the F-dominance conversion ceiling
(MECH-439) after V3-EXQ-707b.

707b built ARC-110 loop segregation fully-live (all 6 non-degeneracy gates passed;
the limbic loop carried per-candidate range 1.414) yet the limbic loop NEVER won,
because the cross-loop combine was STATIC ARITHMETIC (fixed spiral gains) that still
inherited F's dominance. This substrate replaces the fixed combine with a LEARNED
[3,3] cross-loop matrix W_cross = I + M_cross, updated by the SAME ARC-108 signed-RPE
three-factor rule (shared delta_t) via an outer-product Hebbian co-activation trace,
so the limbic value loop can LEARN to drive the motor commit.

These tests lock:
  (A) BYTE-IDENTICAL OFF / at-init: with the flag off, arbitration is the legacy
      static combine; with the flag on but M_cross == 0 (init), the committed index
      is bit-identical to the static combine (W_cross == I).
  (B) NON-VACUITY: a waking commit arms the outer-product coact trace, and the
      shared-delta_t post_action_update moves M_cross off its zero init (the learned
      cross-loop weights actually change).
  (C) MECH-094 waking gate: a simulation tick arms NO cross-loop trace.
  (D) MECHANISM (limbic can win): a learned M_cross that boosts the limbic column
      makes the committed index follow the limbic loop's preference against a motor
      (F) loop that prefers a different candidate -- the conversion the static combine
      structurally could not produce.
  (E) from_dims plumbing: the flag reaches config.e3.
  (F) SAFETY: cross-loop arbitration stays strictly within the eligible set (never
      re-admits a candidate outside it).
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
    _LOOP_NAMES,
)


def _make_selector(cross_loop: bool, **extra) -> E3TrajectorySelector:
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        use_loop_segregation=True,
        use_learned_cross_loop_arbitration=cross_loop,
        **extra,
    )
    return E3TrajectorySelector(cfg.e3, None)


def _limbic_terms(pref_idx: int, n: int):
    """Named limbic channels (ofc/liking/vigour -> limbic loop by the default map)
    that strongly prefer `pref_idx` (COST convention: lower == better)."""
    base = torch.full((n,), 0.5)
    base[pref_idx] = -3.0
    return [
        (_FCG_CHANNEL_INDEX["ofc"], base.clone()),
        (_FCG_CHANNEL_INDEX["liking"], base.clone()),
        (_FCG_CHANNEL_INDEX["vigour"], base.clone()),
    ]


# ------------------------------------------------------------------ #
# (A) byte-identical OFF / at-init                                     #
# ------------------------------------------------------------------ #

class TestByteIdenticalAtInit:
    def test_on_at_init_matches_off(self):
        """W_cross = I + M_cross with M_cross == 0 -> W_cross == I -> the committed
        index is bit-identical to the static-arithmetic combine, across seeds."""
        n = 6
        elig = torch.arange(n)
        for seed in range(12):
            torch.manual_seed(seed)
            raw = torch.randn(n)
            terms = [
                (_FCG_CHANNEL_INDEX["ofc"], torch.randn(n)),
                (_FCG_CHANNEL_INDEX["dacc"], torch.randn(n)),
            ]
            loc_off = _make_selector(cross_loop=False)._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            loc_on = _make_selector(cross_loop=True)._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            assert loc_off == loc_on, (
                f"seed {seed}: cross-loop ON-at-init ({loc_on}) must match the static "
                f"combine ({loc_off}) -- W_cross==I at M_cross==0"
            )

    def test_on_at_init_diagnostics_identity(self):
        """At init the effective column weights equal the static gains (all 1.0) and
        the M_cross range is exactly 0 (non-vacuity baseline)."""
        n = 5
        sel = _make_selector(cross_loop=True)
        sel._segregated_loop_arbitrate(
            torch.arange(n), torch.randn(n), _limbic_terms(3, n), True,
            [None] * n, True, 1.0, True,
        )
        d = sel.last_score_diagnostics
        assert d["loop_learned_cross_loop_active"] is True
        assert d["loop_cross_loop_m_range"] == 0.0
        assert d["loop_cross_loop_w_motor_eff"] == pytest.approx(1.0)
        assert d["loop_cross_loop_w_limbic_eff"] == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# (B) non-vacuity: the learned weights actually move                   #
# ------------------------------------------------------------------ #

class TestLearningMovesMCross:
    def test_waking_commit_arms_coact(self):
        """A waking commit (simulation_mode False) with range-carrying loops arms the
        outer-product coact trace + pending flag (both zero before)."""
        n = 5
        sel = _make_selector(cross_loop=True)
        assert sel._clg_pending is False
        assert float(sel._clg_coact_trace.abs().sum()) == 0.0
        sel._segregated_loop_arbitrate(
            torch.arange(n), torch.randn(n), _limbic_terms(3, n), True,
            [None] * n, True, 1.0, False,  # simulation_mode=False -> waking
        )
        assert sel._clg_pending is True, "waking commit must arm the cross-loop update"
        assert float(sel._clg_coact_trace.abs().sum()) > 0.0, "coact trace must be non-zero"

    def test_post_action_update_moves_m_cross(self):
        """The shared-delta_t three-factor update moves M_cross off its zero init and
        clears the pending flag / increments the update count."""
        n = 5
        sel = _make_selector(cross_loop=True)
        # Force a large positive dopaminergic delta_t regardless of the random heads:
        # delta_t = R_t - V-hat_t, so a very negative baseline guarantees a big signal.
        sel._lcg_value_baseline = -100.0
        sel._segregated_loop_arbitrate(
            torch.arange(n), torch.randn(n), _limbic_terms(3, n), True,
            [None] * n, True, 1.0, False,
        )
        assert sel._clg_pending is True
        metrics = sel.post_action_update(torch.randn(1, 32), harm_occurred=False)
        assert sel._clg_pending is False, "update must clear the pending flag"
        assert sel._clg_n_updates == 1
        assert "clg_delta_t" in metrics
        m_range = float((sel.M_cross.max() - sel.M_cross.min()).item())
        assert m_range > 0.0, "M_cross must move off its zero init (non-vacuity)"


# ------------------------------------------------------------------ #
# (C) MECH-094 waking gate                                             #
# ------------------------------------------------------------------ #

class TestWakingGate:
    def test_simulation_tick_arms_nothing(self):
        """A simulation/replay tick must NOT arm the cross-loop trace (no delta_t, no
        M_cross write -- MECH-094)."""
        n = 5
        sel = _make_selector(cross_loop=True)
        sel._segregated_loop_arbitrate(
            torch.arange(n), torch.randn(n), _limbic_terms(3, n), True,
            [None] * n, True, 1.0, True,  # simulation_mode=True
        )
        assert sel._clg_pending is False
        assert float(sel._clg_coact_trace.abs().sum()) == 0.0


# ------------------------------------------------------------------ #
# (D) mechanism: the limbic loop can learn to win                      #
# ------------------------------------------------------------------ #

class TestLimbicCanWin:
    def test_boosted_limbic_column_wins(self):
        """A learned M_cross that boosts the limbic column makes the committed index
        follow the limbic loop's preference against a motor (F) loop that prefers a
        different candidate -- the conversion the static combine could not produce."""
        n = 4
        elig = torch.arange(n)
        # Motor (F) strictly prefers candidate 0; limbic strictly prefers candidate 3.
        raw = torch.tensor([-3.0, 0.5, 0.5, 0.5])
        terms = _limbic_terms(pref_idx=3, n=n)

        # At init (W_cross == I) the strong motor F-gap dominates -> commit == 0.
        sel_init = _make_selector(cross_loop=True)
        loc_init = sel_init._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n, True, 1.0, True
        )
        assert loc_init == 0, "at init the motor F-gap should win (commit == 0)"

        # Crank the limbic column: every row receives a large limbic contribution, so
        # the effective limbic column weight dominates -> commit follows limbic (3).
        sel_win = _make_selector(cross_loop=True)
        with torch.no_grad():
            sel_win.M_cross[:, 2] = 8.0   # limbic column -> all three rows
        loc_win = sel_win._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n, True, 1.0, True
        )
        assert loc_win == 3, (
            f"a learned limbic-dominant M_cross must let the limbic loop override the "
            f"motor F-winner (expected commit 3, got {loc_win})"
        )
        assert loc_win != loc_init, "the learned matrix must be able to flip the commit"
        d = sel_win.last_score_diagnostics
        assert d["loop_cross_loop_w_limbic_eff"] > d["loop_cross_loop_w_motor_eff"]


# ------------------------------------------------------------------ #
# (E) from_dims plumbing                                              #
# ------------------------------------------------------------------ #

class TestFromDimsPlumbing:
    def test_flag_reaches_e3_config(self):
        cfg_on = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
            use_loop_segregation=True, use_learned_cross_loop_arbitration=True,
            learned_cross_loop_eta=0.02,
        )
        assert cfg_on.e3.use_learned_cross_loop_arbitration is True
        assert cfg_on.e3.learned_cross_loop_eta == pytest.approx(0.02)
        cfg_off = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        )
        assert cfg_off.e3.use_learned_cross_loop_arbitration is False


# ------------------------------------------------------------------ #
# (F) safety: arbitration stays within the eligible set                #
# ------------------------------------------------------------------ #

class TestSafetyWithinEligibleSet:
    def test_commit_is_always_in_eligible_set(self):
        """The returned local index is always a valid position in eligible_idx -- the
        learned weights reorder WITHIN the F+Go/No-Go eligible set and can never point
        outside it (a suppressed candidate is never a candidate here)."""
        n_total = 8
        # A restricted eligible set (only 3 of 8 candidates admitted upstream).
        elig = torch.tensor([1, 3, 6])
        raw = torch.randn(n_total)
        terms = _limbic_terms(pref_idx=2, n=n_total)  # terms are indexed onto elig internally
        sel = _make_selector(cross_loop=True)
        with torch.no_grad():
            sel.M_cross[:, 2] = 6.0  # limbic-dominant
        local = sel._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n_total, True, 1.0, True
        )
        assert 0 <= local < int(elig.numel()), "commit must be a local index into the eligible set"
