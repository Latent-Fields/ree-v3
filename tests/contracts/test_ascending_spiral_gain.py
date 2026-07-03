"""
Contract tests for the ARC-110 x ARC-108 ASCENDING-SPIRAL GAIN -- the V3-EXQ-709/710
loop-effective-weight ceiling repair.

The 709/710 autopsies found the learned cross-loop matrix ENGAGES (M_cross moves off
its zero init) yet the ascending path M_cross[motor,limbic] peaks ~0.03 -- functionally
too weak to lift a non-motor (limbic) loop to the motor loop's effective column weight
``w_eff[j] = sum_i gain_i * W_cross[i,j]``, so the readiness gate ``limbic_loop_can_win``
was met on only 1/4 divergent seeds. Biology (Haber 2000): the striato-nigro-striatal
spiral is anatomically ASYMMETRIC -- ascending (limbic -> associative -> motor) influence
is the developmentally-strengthened, load-bearing direction. This substrate scales ONLY
the ascending (upper-triangular row<col, in the motor(0)/assoc(1)/limbic(2) ordering)
entries of M_cross -- in the forward W_cross (anatomical strength) AND in the three-factor
update (maturation rate) -- so w_eff[limbic] rises WITHOUT raising w_eff[motor] (its
column is diagonal + descending, un-amplified).

These tests lock:
  (A) BYTE-IDENTICAL OFF: with use_ascending_spiral_gain False the arbitration matches
      the plain learned cross-loop combine bit-for-bit, across seeds and a non-trivial
      hand-set M_cross.
  (B) BYTE-IDENTICAL at gain==1.0: with the flag ON but both gains 1.0 the G matrices
      are all-ones -> W_cross == I + M_cross and the update is unchanged.
  (C) AT-INIT identity: flag ON + large gain but M_cross == 0 -> gain*0 == 0 ->
      W_cross == I -> commit bit-identical to OFF (bit-identical-at-init preserved).
  (D) ASYMMETRY: the forward gain leaves w_eff[motor] EXACTLY unchanged (motor column is
      diagonal + descending, never scaled) while raising w_eff[limbic]/w_eff[assoc] when
      their ascending entries are positive -- the implicit motor(F) de-pinning.
  (E) MECHANISM (limbic CAN now win): a SMALL learned ascending M_cross that does NOT
      flip the commit at gain 1.0 DOES flip it (limbic overrides the motor F-winner)
      under a large forward gain -- the repair the 709 substrate structurally lacked.
  (F) PLASTICITY MATURATION: with plasticity_gain > 1 the ascending (upper-tri) entries
      of M_cross accrue MORE per update than at gain 1.0, while descending/diagonal
      entries are bit-identical (the coact trace is the same because forward gain 1.0
      keeps selection byte-identical).
  (G) from_dims plumbing: the flag + both gains reach config.e3.
  (H) SAFETY: the arbitration stays strictly within the eligible set under any gain.
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
    """Named limbic channels (ofc/liking/vigour -> limbic loop by the default map)
    that strongly prefer `pref_idx` (COST convention: lower == better)."""
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


# A non-trivial (learned-looking) M_cross with motor-column growth beating the small
# ascending path -- exactly the 709 signature (w_eff[motor] > w_eff[limbic] at gain 1).
_LEARNED_PATTERN = torch.tensor([
    [0.20, 0.03, 0.05],   # motor row: diag grew (0.20); ascending assoc->motor (0.03), limbic->motor (0.05)
    [0.03, 0.00, 0.04],   # assoc row: descending motor->assoc (0.03); ascending limbic->assoc (0.04)
    [0.02, 0.01, 0.00],   # limbic row: descending only
], dtype=torch.float32)


# ------------------------------------------------------------------ #
# (A) byte-identical OFF                                               #
# ------------------------------------------------------------------ #

class TestByteIdenticalOff:
    def test_off_matches_plain_learned_across_seeds(self):
        """use_ascending_spiral_gain False -> the committed index equals the plain
        learned-cross-loop combine bit-for-bit, with the same hand-set M_cross."""
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
            sel_off = _make_selector(use_ascending_spiral_gain=False,
                                     loop_segregation_ascending_spiral_gain=50.0)
            _set_m_cross(sel_off, _LEARNED_PATTERN)
            loc_off = sel_off._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            assert loc_plain == loc_off, (
                f"seed {seed}: flag OFF must ignore the gain and match the plain "
                f"learned combine ({loc_plain} vs {loc_off})"
            )


# ------------------------------------------------------------------ #
# (B) byte-identical at gain == 1.0                                    #
# ------------------------------------------------------------------ #

class TestByteIdenticalAtUnitGain:
    def test_on_gain_one_matches_off(self):
        """Flag ON but both gains 1.0 -> all-ones G -> selection unchanged, across seeds."""
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
            sel_on = _make_selector(use_ascending_spiral_gain=True,
                                    loop_segregation_ascending_spiral_gain=1.0,
                                    loop_segregation_ascending_plasticity_gain=1.0)
            _set_m_cross(sel_on, _LEARNED_PATTERN)
            loc_on = sel_on._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            assert loc_off == loc_on, f"seed {seed}: gain==1.0 must be a no-op"


# ------------------------------------------------------------------ #
# (C) at-init identity (M_cross == 0)                                  #
# ------------------------------------------------------------------ #

class TestAtInitIdentity:
    def test_large_gain_at_init_matches_off(self):
        """Flag ON + large gain but M_cross == 0 (init) -> gain*0 == 0 -> W_cross == I
        -> committed index bit-identical to OFF."""
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
                use_ascending_spiral_gain=True,
                loop_segregation_ascending_spiral_gain=40.0,
            )._segregated_loop_arbitrate(
                elig, raw, terms, True, [None] * n, True, 1.0, True
            )
            assert loc_off == loc_on, f"seed {seed}: large gain at M_cross==0 must be a no-op"


# ------------------------------------------------------------------ #
# (D) asymmetry: motor column never amplified                         #
# ------------------------------------------------------------------ #

class TestAsymmetryMotorUnamplified:
    def test_forward_gain_leaves_w_motor_eff_unchanged(self):
        """The forward gain scales only ascending (upper-tri) entries, so w_eff[motor]
        (diagonal + descending column) is EXACTLY unchanged while w_eff[limbic] and
        w_eff[assoc] rise (their ascending entries are positive)."""
        n = 5
        elig = torch.arange(n)
        raw = torch.randn(n)
        terms = _limbic_terms(pref_idx=3, n=n)

        sel1 = _make_selector(use_ascending_spiral_gain=True,
                              loop_segregation_ascending_spiral_gain=1.0)
        _set_m_cross(sel1, _LEARNED_PATTERN)
        sel1._segregated_loop_arbitrate(elig, raw, terms, True, [None] * n, True, 1.0, True)
        d1 = dict(sel1.last_score_diagnostics)

        selG = _make_selector(use_ascending_spiral_gain=True,
                              loop_segregation_ascending_spiral_gain=8.0)
        _set_m_cross(selG, _LEARNED_PATTERN)
        selG._segregated_loop_arbitrate(elig, raw, terms, True, [None] * n, True, 1.0, True)
        dG = dict(selG.last_score_diagnostics)

        assert dG["loop_cross_loop_w_motor_eff"] == pytest.approx(
            d1["loop_cross_loop_w_motor_eff"]
        ), "motor effective weight must be invariant to the ascending gain"
        # limbic column ascending entries (0.05 + 0.04) are positive -> w_eff rises.
        assert dG["loop_cross_loop_w_limbic_eff"] > d1["loop_cross_loop_w_limbic_eff"]
        assert dG["loop_cross_loop_w_assoc_eff"] > d1["loop_cross_loop_w_assoc_eff"]
        assert dG["loop_ascending_spiral_gain_active"] is True
        # raw learned coupling diagnostic stays RAW (measures learning, not effective wt).
        assert dG["loop_cross_loop_limbic_to_motor"] == pytest.approx(0.05)


# ------------------------------------------------------------------ #
# (E) mechanism: the limbic loop can now win                          #
# ------------------------------------------------------------------ #

class TestLimbicCanNowWin:
    def test_small_ascending_coupling_flips_under_gain(self):
        """With the 709-signature M_cross (motor column grew past the tiny ascending
        path) the limbic loop does NOT win at gain 1.0 but DOES under a large forward
        gain -- the committed index flips to the limbic loop's pick, and w_eff[limbic]
        crosses above w_eff[motor]."""
        n = 4
        elig = torch.arange(n)
        # Motor (F) prefers candidate 0 by a modest margin; limbic prefers candidate 3.
        raw = torch.tensor([-1.0, 0.5, 0.5, 0.5])
        terms = _limbic_terms(pref_idx=3, n=n)

        sel1 = _make_selector(use_ascending_spiral_gain=True,
                              loop_segregation_ascending_spiral_gain=1.0)
        _set_m_cross(sel1, _LEARNED_PATTERN)
        loc1 = sel1._segregated_loop_arbitrate(elig, raw, terms, True, [None] * n, True, 1.0, True)
        d1 = dict(sel1.last_score_diagnostics)
        assert d1["loop_cross_loop_w_motor_eff"] > d1["loop_cross_loop_w_limbic_eff"], (
            "precondition: at gain 1.0 the motor column must still out-weigh limbic "
            "(the 709 ceiling this test repairs)"
        )
        assert d1["loop_cross_loop_limbic_ge_motor"] is False

        selG = _make_selector(use_ascending_spiral_gain=True,
                              loop_segregation_ascending_spiral_gain=30.0)
        _set_m_cross(selG, _LEARNED_PATTERN)
        locG = selG._segregated_loop_arbitrate(elig, raw, terms, True, [None] * n, True, 1.0, True)
        dG = dict(selG.last_score_diagnostics)
        assert dG["loop_cross_loop_w_limbic_eff"] > dG["loop_cross_loop_w_motor_eff"], (
            "the ascending gain must lift w_eff[limbic] above w_eff[motor]"
        )
        assert dG["loop_cross_loop_limbic_ge_motor"] is True
        assert locG == 3, (
            f"under the ascending gain the limbic loop must override the motor F-winner "
            f"(expected commit 3, got {locG})"
        )
        assert locG != loc1, "the gain must be able to flip the commit off the motor pick"


# ------------------------------------------------------------------ #
# (F) plasticity maturation                                           #
# ------------------------------------------------------------------ #

class TestPlasticityMaturation:
    def test_ascending_entries_accrue_faster(self):
        """With plasticity_gain > 1 (forward gain kept 1.0 so selection + coact are
        byte-identical) the ascending (upper-tri) M_cross entries move MORE per update
        than at gain 1.0, while descending/diagonal entries are bit-identical."""
        n = 5

        def _run(pgain):
            torch.manual_seed(0)
            sel = _make_selector(
                use_ascending_spiral_gain=True,
                loop_segregation_ascending_spiral_gain=1.0,       # keep selection identical
                loop_segregation_ascending_plasticity_gain=pgain,
            )
            sel._lcg_value_baseline = -100.0   # force a large positive delta_t
            torch.manual_seed(1)
            raw = torch.randn(n)
            sel._segregated_loop_arbitrate(
                torch.arange(n), raw, _limbic_terms(3, n), True,
                [None] * n, True, 1.0, False,  # waking -> arms the coact trace
            )
            torch.manual_seed(2)
            sel.post_action_update(torch.randn(1, 32), harm_occurred=False)
            return sel.M_cross.detach().clone()

        M1 = _run(1.0)
        M5 = _run(5.0)
        iu = torch.triu_indices(3, 3, offset=1)   # ascending entries
        il = torch.tril_indices(3, 3, offset=0)   # diagonal + descending
        # Descending + diagonal entries identical (same coact, gain 1 there).
        assert torch.allclose(M5[il[0], il[1]], M1[il[0], il[1]], atol=1e-9), (
            "non-ascending entries must be unaffected by the plasticity gain"
        )
        # Ascending entries scaled ~5x (the update delta there is x5; from a zero init
        # a single update makes the accrued value 5x).
        asc1 = M1[iu[0], iu[1]]
        asc5 = M5[iu[0], iu[1]]
        moved = asc1.abs() > 1e-9
        assert moved.any(), "at least one ascending entry must have moved (non-vacuity)"
        assert torch.allclose(asc5[moved], 5.0 * asc1[moved], rtol=1e-4), (
            "ascending entries must accrue plasticity_gain x faster"
        )


# ------------------------------------------------------------------ #
# (G) from_dims plumbing                                              #
# ------------------------------------------------------------------ #

class TestFromDimsPlumbing:
    def test_flags_reach_e3_config(self):
        cfg_on = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
            use_loop_segregation=True, use_learned_cross_loop_arbitration=True,
            use_ascending_spiral_gain=True,
            loop_segregation_ascending_spiral_gain=12.0,
            loop_segregation_ascending_plasticity_gain=3.0,
        )
        assert cfg_on.e3.use_ascending_spiral_gain is True
        assert cfg_on.e3.loop_segregation_ascending_spiral_gain == pytest.approx(12.0)
        assert cfg_on.e3.loop_segregation_ascending_plasticity_gain == pytest.approx(3.0)
        cfg_off = REEConfig.from_dims(
            body_obs_dim=8, world_obs_dim=8, action_dim=5, self_dim=32, world_dim=32,
        )
        assert cfg_off.e3.use_ascending_spiral_gain is False
        assert cfg_off.e3.loop_segregation_ascending_spiral_gain == pytest.approx(1.0)
        assert cfg_off.e3.loop_segregation_ascending_plasticity_gain == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# (H) safety: arbitration stays within the eligible set               #
# ------------------------------------------------------------------ #

class TestSafetyWithinEligibleSet:
    def test_commit_is_always_in_eligible_set_under_gain(self):
        """Under a large ascending gain the returned local index is still a valid
        position in the (restricted) eligible set -- the gain reorders WITHIN eligible
        candidates and can never point outside it."""
        n_total = 8
        elig = torch.tensor([1, 3, 6])
        raw = torch.randn(n_total)
        terms = _limbic_terms(pref_idx=2, n=n_total)
        sel = _make_selector(use_ascending_spiral_gain=True,
                             loop_segregation_ascending_spiral_gain=25.0)
        _set_m_cross(sel, _LEARNED_PATTERN)
        local = sel._segregated_loop_arbitrate(
            elig, raw, terms, True, [None] * n_total, True, 1.0, True
        )
        assert 0 <= local < int(elig.numel()), "commit must be a local index into eligible"
