"""Contract tests: F-independent closure-plane commit-ENTRY TRAJECTORY primitive
(rung-6 amend extension, 2026-06-23, commitment_closure:GAP-4;
failure_autopsy_V3-EXQ-460k/460l).

The bool latch e3._closure_committed_active (use_closure_commit_entry) arms + SUSTAINS
the closure-formed beta occupancy (C-KEY), but a bare bool cannot be STEPPED: the
between-E3-tick path (agent.py) reads e3._committed_trajectory to advance a committed
PROGRAM, so a closure-armed hold with only a bool falls through to repeating
_last_action (no closure-formed program executes -- the C-STEP gap). This extension adds
a PARALLEL sticky trajectory e3._closure_committed_trajectory (SET to the goal/rule-
directed result.selected_trajectory when use_closure_commit_entry_trajectory is on,
CLEARED at the same de-commit / closure-fire / reset sites as the bool); the
_closure_commit_active arm gate, the latch-hold persistence check, and the between-tick
stepping all read the UNION (_committed_trajectory OR _closure_committed_trajectory).

Contracts (interface-level guarantees, NOT magnitude thresholds):

  C-KEY   trajectory ON + eval ON, running_variance pinned above commit (committed never
          True -> _committed_trajectory stays None), a goal-active rule-directed commitment
          -> the F-independent TRAJECTORY latch is installed AND the hold sustains beta.
  C-STEP  under the same regime the between-tick path ADVANCES the closure-formed
          trajectory (the 4900 stepping union), not a repeated _last_action -- proven with
          a trajectory whose per-step actions differ.
  C-YIELD the hold still YIELDS to the SD-034 de-commit refractory (the safety yields are
          preserved -- the union-aware persistence did not break them).
  C-OFF   default-OFF (trajectory flag off) reproduces A's bool-latch path bit-for-bit
          (latch stays None); episode-reset clears the trajectory latch; the precondition
          (trajectory requires the bool flag) raises.
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


def _one_hot_uniform(i: int) -> torch.Tensor:
    """All H timesteps the same one-hot (action class i)."""
    a = torch.zeros(1, H, ACTION_DIM)
    a[:, :, i] = 1.0
    return a


def _one_hot_per_step(classes) -> torch.Tensor:
    """Distinct one-hot per timestep -> stepping is observable vs repeating _last_action."""
    a = torch.zeros(1, H, ACTION_DIM)
    for t, c in enumerate(classes):
        a[:, t, c] = 1.0
    return a


def _traj(actions: torch.Tensor, off: float) -> Trajectory:
    states = [torch.full((1, SELF_DIM), off + 0.01 * k) for k in range(H + 1)]
    world = [torch.full((1, WORLD_DIM), off + 0.02 * k) for k in range(H + 1)]
    return Trajectory(states=states, actions=actions, world_states=world)


def _candidates_uniform():
    return [_traj(_one_hot_uniform(1), 0.1), _traj(_one_hot_uniform(2), 0.2)]


def _candidates_stepped():
    # selected (index 0): action class 0 -> 1 -> 2 across the horizon. selected_action is
    # class 0 (t=0); a genuine STEP to t=1 yields class 1 != _last_action (class 0).
    return [_traj(_one_hot_per_step([0, 1, 2]), 0.1), _traj(_one_hot_uniform(3), 0.2)]


def _scores() -> torch.Tensor:
    return torch.tensor([0.0, 1.0, 1.0])


class _Stub:
    """An E3 selection that is NEVER a natural commit (committed=False) -- running_variance
    pinned above commit_threshold, so the F path never sets _committed_trajectory."""

    def __init__(self, candidates):
        self.candidates = candidates
        self.scores = _scores()

    def select(self, candidates, temperature: float = 1.0, **kw):
        return SelectionResult(
            selected_trajectory=candidates[0],
            selected_index=0,
            selected_action=candidates[0].actions[:, 0, :],
            scores=self.scores.clone(),
            precision=1.0,
            committed=False,
            log_prob=torch.tensor(0.0),
            urgency=0.0,
        )


def _build_agent(*, with_goal_rule: bool = False, **kwargs) -> REEAgent:
    if with_goal_rule:
        kwargs.setdefault("z_goal_enabled", True)
        kwargs.setdefault("use_lateral_pfc_analog", True)
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
    if with_goal_rule:
        assert agent.goal_state is not None
        agent.goal_state.is_active = lambda: True  # type: ignore[assignment]
        assert agent.lateral_pfc is not None
        agent.lateral_pfc.rule_state.fill_(0.5)  # norm >> default floor 0.01
    return agent


def _e3_tick(agent, stub, cands):
    agent.e3.select = stub.select
    agent.e3._running_variance = 0.0
    agent.e3.last_scores = stub.scores.clone()
    agent.e3._committed_trajectory = None
    return agent.select_action(cands, {"e3_tick": True})


def _between_tick(agent, cands):
    # Between E3 ticks: select_action returns early via the stepping/hold branch; e3.select
    # is not invoked. _committed_trajectory stays None (F never commits).
    agent.e3._committed_trajectory = None
    return agent.select_action(cands, {"e3_tick": False})


_EVAL = dict(
    use_closure_commit_beta_coupling=True,
    use_natural_commit_latch_hold=True,
    closure_exclusive_decommit_eval=True,
)
_ENTRY = dict(_EVAL, use_closure_commit_entry=True)  # A's bool latch
_TRAJ = dict(_ENTRY, use_closure_commit_entry_trajectory=True)  # + C-STEP extension


# ----------------------------------------------------------------------
# C-KEY: F-independent TRAJECTORY latch installs + the hold sustains beta
# ----------------------------------------------------------------------
def test_ckey_trajectory_arm_and_sustain():
    cands = _candidates_uniform()
    stub = _Stub(cands)
    on = _build_agent(with_goal_rule=True, **_TRAJ)
    for _ in range(4):
        _e3_tick(on, stub, cands)
    assert on.e3._committed_trajectory is None, "F path must never have committed"
    assert on.e3._closure_committed_trajectory is not None, (
        "the F-independent TRAJECTORY latch must be installed"
    )
    assert on.e3._closure_committed_trajectory is cands[0], (
        "the latch holds the goal/rule-directed selected_trajectory"
    )
    assert on._ncl_hold_closure_armed_count > 0, "the closure entry must arm the hold"
    assert on.beta_gate.is_elevated, "the closure-formed occupancy must elevate beta"
    assert on.beta_gate.committed_run_length >= 2, "the hold must SUSTAIN beta across ticks"
    # is_committed telemetry reflects the closure-formed commit (honest readout).
    assert on.e3.get_commitment_state()["is_committed"] is True


# ----------------------------------------------------------------------
# C-STEP (LOAD-BEARING): the between-tick path STEPS the closure trajectory,
# not a repeated _last_action
# ----------------------------------------------------------------------
def test_cstep_between_tick_advances_closure_trajectory():
    cands = _candidates_stepped()
    stub = _Stub(cands)
    on = _build_agent(with_goal_rule=True, **_TRAJ)

    # One E3 tick installs the closure trajectory + elevates beta; _last_action = the
    # selected action = class-0 one-hot (t=0). _committed_step_idx reset to 0 (fresh arm).
    _e3_tick(on, stub, cands)
    assert on.e3._closure_committed_trajectory is cands[0]
    assert on.beta_gate.is_elevated
    last0 = on._last_action.detach().clone()
    assert torch.equal(last0, cands[0].actions[:, 0, :]), "last_action = selected (t=0)"

    # Between-tick #1: step_idx 0 -> action = trajectory[t=0] (class 0); idx -> 1.
    a1 = _between_tick(on, cands)
    assert torch.equal(a1, cands[0].actions[:, 0, :])

    # Between-tick #2: step_idx 1 -> action = trajectory[t=1] (class 1) -- the STEP. A
    # bool-only / un-widened path would repeat _last_action (class 0) here; the closure
    # program executing produces class 1.
    a2 = _between_tick(on, cands)
    assert torch.equal(a2, cands[0].actions[:, 1, :]), (
        "between-tick must STEP the closure-formed trajectory (t=1, class 1), "
        "not repeat _last_action (t=0, class 0)"
    )
    assert not torch.equal(a2, cands[0].actions[:, 0, :])


# ----------------------------------------------------------------------
# C-YIELD: the hold still yields to the SD-034 de-commit refractory
# ----------------------------------------------------------------------
def test_cyield_refractory_disarms_hold():
    cands = _candidates_uniform()
    stub = _Stub(cands)
    on = _build_agent(with_goal_rule=True, **_TRAJ)
    for _ in range(3):
        _e3_tick(on, stub, cands)
    assert on._ncl_hold_active is True, "hold armed before the refractory"

    # An SD-034 closure de-commit installs a beta refractory. The union-aware persistence
    # must NOT paper over it: refractory_remaining > 0 is a principled yield.
    on.beta_gate.apply_refractory(5)
    _e3_tick(on, stub, cands)
    assert on.beta_gate.refractory_remaining > 0
    assert on._ncl_hold_active is False, (
        "the latch-hold must yield to the SD-034 de-commit refractory (safety yield intact)"
    )


# ----------------------------------------------------------------------
# C-OFF: default-OFF bit-identical to A's bool latch + reset clears + precondition
# ----------------------------------------------------------------------
def test_coff_trajectory_flag_off_bit_identical_to_bool_latch():
    cands = _candidates_stepped()
    # A (bool latch only) vs B (bool latch + trajectory flag explicitly False): the
    # trajectory latch never installs and the action stream is identical.
    a = _build_agent(with_goal_rule=True, **_ENTRY)
    b = _build_agent(
        with_goal_rule=True, **_ENTRY, use_closure_commit_entry_trajectory=False
    )
    acts_a, acts_b = [], []
    for agent, acc in ((a, acts_a), (b, acts_b)):
        stub = _Stub(cands)
        for _ in range(3):
            acc.append(_e3_tick(agent, stub, cands).detach().clone())
            acc.append(_between_tick(agent, cands).detach().clone())
    assert a.e3._closure_committed_trajectory is None
    assert b.e3._closure_committed_trajectory is None
    for x, y in zip(acts_a, acts_b):
        assert torch.equal(x, y)


def test_coff_reset_clears_trajectory_latch():
    cands = _candidates_uniform()
    on = _build_agent(with_goal_rule=True, **_TRAJ)
    _e3_tick(on, _Stub(cands), cands)
    assert on.e3._closure_committed_trajectory is not None
    on.reset()
    assert on.e3._closure_committed_trajectory is None, "episode reset must clear the latch"


def test_coff_precondition_trajectory_requires_bool_flag():
    with pytest.raises(
        ValueError, match="use_closure_commit_entry_trajectory=True requires"
    ):
        _build_agent(
            use_closure_commit_entry_trajectory=True,
            use_closure_commit_beta_coupling=True,
            use_natural_commit_latch_hold=True,
        )
