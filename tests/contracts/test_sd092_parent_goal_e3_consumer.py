"""Contract: SD-092 residual -- a default-OFF E3 scoring consumer for
GoalState._z_goal_parent (E3Config.parent_goal_weight).

Background (chip-20260902-zgoal-parent-e3-consumer; pre-flight 2026-09-24). The SD-092
parent attractor (MECH-427 cross-level subgoal credit) had NO reader in ree_core: E3's only
goal channel is child-only (compute_goal_score -> goal_state.goal_proximity reads _z_goal),
so any arm that restores or ablates cross-level credit was DV-invariant by construction.

The consumer is an ADDITIVE term, not a blend inside goal_proximity (goal_proximity has many
other callers that must not move):
    score -= parent_goal_weight * sum_t 1 / (1 + MSE_sum(z_t, _z_goal_parent))
gated on goal_state.parent_is_active() and its own weight -- NOT inside the child
`goal_state.is_active() and goal_weight > 0` block -- and deliberately outside the
_COMMENSURABILITY_CHANNELS partition.

Guards (real E3TrajectorySelector + real GoalState; parent != child fixture):
  C1 weight 0.0 (default) with an ACTIVE parent -> scores bit-identical to no parent.
  C2 ON -> the parent term has non-zero cross-candidate RANGE and CHANGES THE ARGMIN
     (a candidate heading to the parent beats one heading to the child) -- FAILS on the
     pre-change tree, where the knob does not exist and nothing reads the parent.
  C3 ON with the child INACTIVE and goal_weight 0 -> the parent term still applies.
  C4 ON with the parent INACTIVE -> bit-identical.
  C5 from_dims reaches parent_goal_weight and the SD-092 GoalConfig knobs.
"""

from __future__ import annotations

import torch

from ree_core.goal import GoalConfig, GoalState
from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector
from ree_core.utils.config import REEConfig

WD = 6
H = 3


def _traj_to(target: torch.Tensor) -> Trajectory:
    """Candidate whose z_world rollout moves linearly from 0 to `target`."""
    ws = [(t / H) * target.reshape(1, WD) for t in range(H + 1)]
    actions = torch.zeros(1, H, 5)
    actions[:, 0, 0] = 1.0
    return Trajectory(states=[torch.zeros(1, WD) for _ in range(H + 1)],
                      actions=actions, world_states=ws)


CHILD = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
PARENT = torch.tensor([0.0, 0.0, 0.0, -1.0, 0.0, 0.0])


def _goal_state(child_active: bool = True, parent_active: bool = True) -> GoalState:
    gs = GoalState(
        GoalConfig(goal_dim=WD, z_goal_enabled=True, use_hierarchical_goal_credit=True,
                   parent_goal_alpha=1.0),
        torch.device("cpu"),
    )
    if child_active:
        gs._z_goal = CHILD.reshape(1, WD).clone()
    if parent_active:
        out = gs.credit_subgoal_attainment(PARENT.reshape(1, WD), credit=1.0)
        assert out and gs.parent_is_active()
    return gs


def _selector(**kw) -> E3TrajectorySelector:
    torch.manual_seed(0)
    sel = E3TrajectorySelector(E3Config(world_dim=WD, hidden_dim=8, **kw))
    sel._running_variance = 0.0
    return sel


def _scores(sel, gs, cands):
    return torch.cat([sel.score_trajectory(c, goal_state=gs).reshape(-1) for c in cands])


def test_c1_weight_zero_is_bit_identical_with_active_parent():
    sel = _selector(goal_weight=1.0)
    cands = [_traj_to(CHILD), _traj_to(PARENT), _traj_to(torch.zeros(WD))]
    with_parent = _scores(sel, _goal_state(parent_active=True), cands)
    no_parent = _scores(sel, _goal_state(parent_active=False), cands)
    assert torch.equal(with_parent, no_parent)


def test_c2_parent_term_has_range_and_changes_argmin():
    cands = [_traj_to(CHILD), _traj_to(PARENT)]
    gs = _goal_state()
    sel = _selector(goal_weight=1.0)
    off = _scores(sel, gs, cands)
    sel.config.parent_goal_weight = 5.0
    on = _scores(sel, gs, cands)
    delta = on - off
    assert float(delta.max() - delta.min()) > 1e-3, "parent term is a uniform shift"
    # lower is better: OFF prefers the child-bound candidate (index 0) ...
    assert int(off.argmin()) == 0
    # ... and a strong parent channel reverses it toward the parent-bound one.
    assert int(on.argmin()) == 1


def test_c3_parent_term_live_when_child_inactive_and_goal_weight_zero():
    cands = [_traj_to(CHILD), _traj_to(PARENT)]
    gs = _goal_state(child_active=False, parent_active=True)
    assert not gs.is_active()
    sel = _selector(goal_weight=0.0)
    off = _scores(sel, gs, cands)
    sel.config.parent_goal_weight = 1.0
    on = _scores(sel, gs, cands)
    delta = on - off
    assert float(delta.max() - delta.min()) > 1e-3


def test_c4_inactive_parent_is_bit_identical():
    cands = [_traj_to(CHILD), _traj_to(PARENT)]
    gs = _goal_state(parent_active=False)
    sel = _selector(goal_weight=1.0)
    off = _scores(sel, gs, cands)
    sel.config.parent_goal_weight = 5.0
    on = _scores(sel, gs, cands)
    assert torch.equal(on, off)


def test_c5_defaults_and_from_dims_passthrough():
    assert E3Config().parent_goal_weight == 0.0
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=20, action_dim=5,
        parent_goal_weight=0.7, use_hierarchical_goal_credit=True,
        parent_goal_alpha=0.2, parent_goal_decay=0.01, subgoal_credit_min=0.3,
    )
    assert cfg.e3.parent_goal_weight == 0.7
    assert cfg.goal.use_hierarchical_goal_credit is True
    assert cfg.goal.parent_goal_alpha == 0.2
    assert cfg.goal.parent_goal_decay == 0.01
    assert cfg.goal.subgoal_credit_min == 0.3
    d = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=20, action_dim=5)
    assert d.e3.parent_goal_weight == 0.0
    assert d.goal.use_hierarchical_goal_credit is False
