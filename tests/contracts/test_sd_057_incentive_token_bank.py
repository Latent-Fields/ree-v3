"""SD-057 contract: object-bound incentive-salience token bank (GAP-7 L2-L3-L4).

C1 default OFF: GoalConfig.use_incentive_token_bank defaults False; GoalState
   has no incentive_bank; legacy single-attractor seeding is unchanged.
C2 L2 bind: update(k, benefit, z) creates a per-type token; tag 0 (no resource)
   is skipped.
C3 L3 per-axis wanting: wanting[k] = base_value[k]*(1 + kappa*drive_axis[k]);
   most_wanted is drive-specific (food when hungry, water when thirsty).
C4 L3 revaluation + decay: base_value EMAs toward benefit; decay() shrinks it.
C5 [1,n] per-axis tensor shape robustness.
C6 reset clears the bank (per-episode state).
"""
import torch
from ree_core.goal import GoalConfig, GoalState, IncentiveTokenBank


def _bank(**kw):
    cfg = GoalConfig(goal_dim=4, use_incentive_token_bank=True, **kw)
    return IncentiveTokenBank(cfg, torch.device("cpu"))


def test_c1_default_off_no_bank_and_legacy_seeding():
    cfg = GoalConfig(goal_dim=4)
    assert cfg.use_incentive_token_bank is False
    gs = GoalState(cfg, torch.device("cpu"))
    assert gs.incentive_bank is None
    gs.update(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), benefit_exposure=1.0, drive_level=1.0)
    assert gs.is_active()


def test_c2_l2_bind_and_skip_no_resource():
    bank = _bank(incentive_decay=0.0, incentive_value_alpha=1.0)
    assert bank.is_empty()
    bank.update(1, 0.5, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    bank.update(0, 0.9, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))  # tag 0 -> skipped
    assert set(bank._base_value.keys()) == {1}


def test_c3_l3_per_axis_wanting_is_drive_specific():
    bank = _bank(incentive_decay=0.0, incentive_value_alpha=1.0,
                 incentive_drive_kappa_weight=2.0)
    food = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    water = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    bank.update(1, 0.5, food)
    bank.update(2, 0.5, water)
    # hungry (axis0 high) -> food; thirsty (axis1 high) -> water
    k_h, z_h, _ = bank.most_wanted(per_axis_drive=torch.tensor([0.9, 0.1, 0.1]))
    assert k_h == 1 and torch.allclose(z_h, food)
    k_t, z_t, _ = bank.most_wanted(per_axis_drive=torch.tensor([0.1, 0.9, 0.1]))
    assert k_t == 2 and torch.allclose(z_t, water)
    w = bank.wanting(per_axis_drive=torch.tensor([0.9, 0.1, 0.1]))
    assert abs(w[1] - 0.5 * (1 + 2 * 0.9)) < 1e-6


def test_c4_revaluation_and_decay():
    bank = _bank(incentive_decay=0.1, incentive_value_alpha=0.5)
    z = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    bank.update(1, 1.0, z)
    v1 = bank._base_value[1]            # 0.5*0 + 0.5*1.0 = 0.5
    assert abs(v1 - 0.5) < 1e-6
    bank.update(1, 1.0, z)
    assert bank._base_value[1] > v1     # revalues upward toward benefit
    before = bank._base_value[1]
    bank.decay()
    assert bank._base_value[1] < before  # slow decay shrinks the token


def test_c5_per_axis_2d_shape_robust():
    bank = _bank(incentive_decay=0.0, incentive_value_alpha=1.0)
    bank.update(1, 0.5, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    bank.update(2, 0.5, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    k, _, _ = bank.most_wanted(per_axis_drive=torch.tensor([[0.1, 0.9, 0.1]]))
    assert k == 2


def test_c6_reset_clears_bank():
    cfg = GoalConfig(goal_dim=4, use_incentive_token_bank=True)
    gs = GoalState(cfg, torch.device("cpu"))
    gs.incentive_bank.update(1, 0.5, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert not gs.incentive_bank.is_empty()
    gs.reset()
    assert gs.incentive_bank.is_empty()
