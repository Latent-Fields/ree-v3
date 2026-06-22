"""Contract tests: closure-exclusive de-commit eval mode (rung-6 BUILD, 2026-06-22,
the named dissociable substrate from failure_autopsy_V3-EXQ-460j).

460j ARM_LEVER_OFF showed the natural-commit latch-hold NEVER armed
(ncl_hold_reassert_total=0): it arms only on a decisive natural commit
(result.committed), which does not form on the full closure-coupling substrate, so
natural-commit and the SD-034 closure de-commit were NON-DISSOCIABLE (no sustained
occupancy for the de-commit to act on). closure_exclusive_decommit_eval fixes the
ARM SOURCE: beta elevation becomes closure-exclusive (result.committed suppressed
from _commit_for_beta) AND the latch-hold arms on the closure-coupled commit
(_closure_commit_active), so a sustained occupancy forms via the closure plane
independently of the fragile natural commit and the SD-034 closure de-commit can
act on it (the existing yield-on-refractory).

Contracts (interface-level guarantees, NOT magnitude thresholds):

  C1  config defaults: closure_exclusive_decommit_eval False; from_dims surfaces it;
      a default agent never enters the eval branch (_ncl_hold_closure_armed_count 0).
  C2  preconditions raise: eval True without use_closure_commit_beta_coupling -> ValueError;
      eval True without use_natural_commit_latch_hold -> ValueError.
  C3  arm-source fix (LOAD-BEARING): eval ON, a closure-coupled commit
      (result.committed False + committed trajectory present) ARMS the hold and
      increments _ncl_hold_closure_armed_count; the same closure-coupled commit with
      eval OFF (legacy hold) does NOT arm.
  C4  closure-exclusive elevation: eval ON, a NATURAL commit with NO closure
      trajectory does NOT elevate beta (natural path suppressed); the same natural
      commit with eval OFF DOES elevate.
  C5  dissociation preserved: eval ON, arm via closure-coupled commit, then an active
      SD-034 refractory disarms the hold (the MECH-446 occupancy-drop DV is intact).
  C6  bit-identical OFF: a default (eval-off) agent's closure-coupled commit does not
      arm the hold and its latch trace matches a reference OFF run.
"""

from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import SelectionResult
from ree_core.utils.config import REEConfig

ACTION_DIM = 4
SELF_DIM = 8
WORLD_DIM = 8
BODY_OBS_DIM = 4
WORLD_OBS_DIM = 8
H = 3


def _one_hot(i: int) -> torch.Tensor:
    a = torch.zeros(1, H, ACTION_DIM)
    a[:, :, i] = 1.0
    return a


def _traj(i: int, off: float) -> Trajectory:
    states = [torch.full((1, SELF_DIM), off + 0.01 * k) for k in range(H + 1)]
    world = [torch.full((1, WORLD_DIM), off + 0.02 * k) for k in range(H + 1)]
    return Trajectory(states=states, actions=_one_hot(i), world_states=world)


def _candidates():
    return [_traj(1, 0.1), _traj(2, 0.2)]


def _scores() -> torch.Tensor:
    return torch.tensor([0.0, 1.0, 1.0])


class _Stub:
    def __init__(self, committed: bool):
        self.committed = committed
        self.scores = _scores()

    def select(self, candidates, temperature: float = 1.0, **kw):
        return SelectionResult(
            selected_trajectory=candidates[0],
            selected_index=0,
            selected_action=candidates[0].actions[:, 0, :],
            scores=self.scores.clone(),
            precision=1.0,
            committed=self.committed,
            log_prob=torch.tensor(0.0),
            urgency=0.0,
        )


def _build_agent(**kwargs) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_sleep_loop=False,
        sws_enabled=False,
        rem_enabled=False,
        use_sleep_aggregation_cluster=False,
        **kwargs,
    )
    cfg.heartbeat.beta_gate_bistable = True
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _tick(agent, stub, cands, *, committed_trajectory):
    """One E3 select_action tick with a chosen stub committed state + an explicit
    committed-trajectory presence (committed_trajectory drives _closure_commit_active
    when the closure->beta coupling is on)."""
    agent.e3.select = stub.select
    agent.e3._running_variance = 0.0
    agent.e3.last_scores = stub.scores.clone()
    agent.e3._committed_trajectory = cands[0] if committed_trajectory else None
    agent.select_action(cands, {"e3_tick": True})


_EVAL = dict(
    use_closure_commit_beta_coupling=True,
    use_natural_commit_latch_hold=True,
    closure_exclusive_decommit_eval=True,
)


# ----------------------------------------------------------------------
# C1 config defaults + default agent never enters the eval branch
# ----------------------------------------------------------------------
def test_c1_config_defaults():
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
    )
    assert cfg.closure_exclusive_decommit_eval is False
    cfg2 = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        closure_exclusive_decommit_eval=False,
    )
    assert cfg2.closure_exclusive_decommit_eval is False
    off = _build_agent()
    assert off._ncl_hold_closure_armed_count == 0


# ----------------------------------------------------------------------
# C2 preconditions raise
# ----------------------------------------------------------------------
def test_c2_precondition_requires_coupling():
    with pytest.raises(ValueError, match="use_closure_commit_beta_coupling"):
        _build_agent(
            closure_exclusive_decommit_eval=True,
            use_natural_commit_latch_hold=True,
        )


def test_c2_precondition_requires_latch_hold():
    with pytest.raises(ValueError, match="use_natural_commit_latch_hold"):
        _build_agent(
            closure_exclusive_decommit_eval=True,
            use_closure_commit_beta_coupling=True,
        )


# ----------------------------------------------------------------------
# C3 arm-source fix (LOAD-BEARING): closure-coupled commit arms the hold under eval
# ----------------------------------------------------------------------
def test_c3_closure_coupled_commit_arms_hold_under_eval():
    cands = _candidates()
    on = _build_agent(**_EVAL)
    # closure-coupled commit: NOT a natural commit, but a committed trajectory exists
    _tick(on, _Stub(committed=False), cands, committed_trajectory=True)
    assert on.beta_gate.is_elevated, "closure-coupled commit must elevate beta"
    assert on._ncl_hold_active is True, "hold must arm on the closure-coupled commit"
    assert on._ncl_hold_closure_armed_count >= 1

    # Legacy hold (eval OFF): the SAME closure-coupled commit does NOT arm the hold.
    legacy = _build_agent(
        use_closure_commit_beta_coupling=True,
        use_natural_commit_latch_hold=True,
    )
    _tick(legacy, _Stub(committed=False), cands, committed_trajectory=True)
    assert legacy._ncl_hold_active is False
    assert legacy._ncl_hold_closure_armed_count == 0


# ----------------------------------------------------------------------
# C4 closure-exclusive elevation: natural commit alone is suppressed
# ----------------------------------------------------------------------
def test_c4_natural_commit_suppressed_under_eval():
    cands = _candidates()
    # eval ON: a NATURAL commit with NO closure trajectory -> _commit_for_beta is
    # _closure_commit_active only (False here) -> beta does NOT elevate.
    on = _build_agent(**_EVAL)
    _tick(on, _Stub(committed=True), cands, committed_trajectory=False)
    assert not on.beta_gate.is_elevated, "eval must suppress the natural-commit path"

    # eval OFF (legacy coupling): the SAME natural commit DOES elevate.
    legacy = _build_agent(
        use_closure_commit_beta_coupling=True,
        use_natural_commit_latch_hold=True,
    )
    _tick(legacy, _Stub(committed=True), cands, committed_trajectory=False)
    assert legacy.beta_gate.is_elevated, "legacy path elevates on a natural commit"


# ----------------------------------------------------------------------
# C5 dissociation preserved: hold yields to the SD-034 closure de-commit
# ----------------------------------------------------------------------
def test_c5_yield_to_closure_refractory_under_eval():
    cands = _candidates()
    on = _build_agent(**_EVAL)
    _tick(on, _Stub(committed=False), cands, committed_trajectory=True)  # arm
    assert on._ncl_hold_active is True
    # SD-034 closure de-commit: drop the latch + install a refractory window.
    stub = _Stub(committed=False)
    on.e3.select = stub.select
    on.beta_gate.release()
    on.beta_gate.apply_refractory(5)
    on.e3.last_scores = stub.scores.clone()
    on.e3._committed_trajectory = cands[0]
    on.select_action(cands, {"e3_tick": True})
    assert on._ncl_hold_active is False, "hold must yield to the closure de-commit"


# ----------------------------------------------------------------------
# C6 bit-identical OFF
# ----------------------------------------------------------------------
def test_c6_bit_identical_off():
    cands = _candidates()
    # A default agent with the closure coupling on but eval OFF: a closure-coupled
    # commit takes the legacy path (no closure-arm) -> action stream matches a
    # reference run built the same way.
    a = _build_agent(use_closure_commit_beta_coupling=True)
    b = _build_agent(use_closure_commit_beta_coupling=True)
    acts_a, acts_b = [], []
    for agent, acc in ((a, acts_a), (b, acts_b)):
        for _ in range(5):
            stub = _Stub(committed=False)
            agent.e3.select = stub.select
            agent.e3._running_variance = 0.0
            agent.e3.last_scores = stub.scores.clone()
            agent.e3._committed_trajectory = cands[0]
            act = agent.select_action(cands, {"e3_tick": True})
            acc.append(act.detach().clone())
    assert a._ncl_hold_closure_armed_count == 0
    for x, y in zip(acts_a, acts_b):
        assert torch.equal(x, y)
