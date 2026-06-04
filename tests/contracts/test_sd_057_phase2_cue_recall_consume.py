"""SD-057 phase-2 contracts: L6 cue-recall (MECH-347) + L7 dACC readout (MECH-348).

C1 L6 cue_pull: GoalState.cue_pull nudges z_goal toward an object embedding
   without the benefit gate; 0-strength is a no-op.
C2 L6 config defaults are no-op (use_cue_recall False; bank-less GoalState has no
   cue effect).
C3 L7 dACC goal-readout: OFF (weight 0) is bit-identical; ON favours the
   high-proximity candidate; None goal_readout is safe.
C4 preconditions: use_mech_consume requires use_dacc; use_cue_recall requires
   use_incentive_token_bank (loud-not-silent ValueError).
"""
import torch
import pytest

from ree_core.goal import GoalConfig, GoalState
from ree_core.cingulate.dacc import DACCConfig, DACCtoE3Adapter
from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent


def test_c1_cue_pull_nudges_zgoal_no_benefit_gate():
    gs = GoalState(GoalConfig(goal_dim=4, use_incentive_token_bank=True,
                              use_cue_recall=True), torch.device("cpu"))
    z = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert gs.goal_norm() == 0.0
    gs.cue_pull(z, 0.5)
    assert gs.goal_norm() > 0.0
    assert torch.nn.functional.cosine_similarity(gs.z_goal, z).item() > 0.99
    before = gs.z_goal.clone()
    gs.cue_pull(z, 0.0)  # no-op
    assert torch.allclose(gs.z_goal, before)


def test_c2_defaults_noop():
    cfg = GoalConfig(goal_dim=4)
    assert cfg.use_cue_recall is False
    assert cfg.cue_recall_gain == 0.05


def test_c3_dacc_goal_readout_off_bit_identical_on_favours_proximity():
    K = 3
    prox = torch.tensor([0.9, 0.1, 0.5])
    bundle = {"mode_ev": torch.zeros(K), "harm_interaction": torch.zeros(K),
              "foraging_value": 0.0, "suppression": torch.zeros(K),
              "drive_gain": 1.0, "goal_readout": prox}
    off = DACCtoE3Adapter(DACCConfig(dacc_weight=1.0, dacc_goal_readout_weight=0.0))
    on = DACCtoE3Adapter(DACCConfig(dacc_weight=1.0, dacc_goal_readout_weight=1.0))
    assert torch.allclose(off(bundle), torch.zeros(K))           # OFF ignores readout
    b_on = on(bundle)
    assert b_on[0] < b_on[1]                                     # closest favoured (lower=better)
    assert float(b_on[0]) == pytest.approx(-0.9)
    nogr = dict(bundle); nogr["goal_readout"] = None
    assert torch.allclose(on(nogr), torch.zeros(K))             # None-safe


def _mini_cfg(**kw):
    return REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4,
                               world_dim=32, z_goal_enabled=True, **kw)


def test_c4_precondition_mech_consume_requires_dacc():
    with pytest.raises(ValueError):
        REEAgent(_mini_cfg(use_mech_consume=True, use_dacc=False))


def test_c4_precondition_cue_recall_requires_bank():
    with pytest.raises(ValueError):
        REEAgent(_mini_cfg(use_cue_recall=True, use_incentive_token_bank=False))
