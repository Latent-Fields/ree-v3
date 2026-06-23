"""Contract tests: F-independent closure-plane commit-ENTRY primitive (rung-6 amend,
2026-06-23, commitment_closure:GAP-4; failure_autopsy_V3-EXQ-460k/460l).

The 2026-06-22 closure-exclusive de-commit eval arms the latch-hold ONLY via
_closure_commit_active (agent.py), which gates on e3._committed_trajectory is not None
-- whose ONLY non-None writer is e3_selector.py under `if committed:` (pure
running_variance/F). On the 460j substrate the F-commit never sustains, so the
trajectory is rarely non-None and the eval rarely arms (ncl_hold_closure_armed_total=0,
the 460k/460l signature). This amend adds an F-INDEPENDENT latch
e3._closure_committed_active, SET (Option A) on a goal-active, rule-directed commitment
and CLEARED on the SD-034 closure fire / de-commit refractory install / episode reset;
_closure_commit_active becomes the UNION (legacy F-commit OR the F-independent entry),
and the latch-hold persistence check is union-aware, so a sustained closure-formed
occupancy forms with ZERO F-commits.

Contracts (interface-level guarantees, NOT magnitude thresholds):

  C1  config defaults: use_closure_commit_entry False; from_dims surfaces it + the rule
      floor; a default agent has the latch attr init False and never arms.
  C2  preconditions raise: entry True without use_closure_commit_beta_coupling -> ValueError;
      entry True without use_natural_commit_latch_hold -> ValueError.
  C3  C-KEY (LOAD-BEARING): entry ON + eval ON, running_variance pinned above commit
      (committed never True -> _committed_trajectory stays None), a goal-active rule-directed
      commitment -> the latch arms AND the hold SUSTAINS beta (ncl_hold_closure_armed_count
      > 0 AND a multi-tick committed_run_length). The SAME scenario with entry OFF (the
      legacy eval, == pre-fix code) does NOT arm (count exactly 0) and beta never sustains.
  C4  default-OFF bit-identical action stream (entry off -> latch never set -> legacy path).
  C5  episode-reset clears the latch.
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
    """An E3 selection that is NEVER a natural commit (committed=False) -- models
    running_variance pinned above commit_threshold, so the F path never sets
    _committed_trajectory."""

    def __init__(self, committed: bool = False):
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


def _build_agent(*, with_goal_rule: bool = False, **kwargs) -> REEAgent:
    if with_goal_rule:
        # The Option-A SET predicate reads goal_state + lateral_pfc, so they must exist.
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
        # Force the Option-A SET predicate: an active goal + a followed rule (rule_state
        # norm above the floor). Monkeypatching is_active isolates the latch WIRING from
        # goal-seeding mechanics (not under test here).
        assert agent.goal_state is not None
        agent.goal_state.is_active = lambda: True  # type: ignore[assignment]
        assert agent.lateral_pfc is not None
        agent.lateral_pfc.rule_state.fill_(0.5)  # norm >> default floor 0.01
    return agent


def _tick(agent, stub, cands):
    """One E3 select_action tick. _committed_trajectory is left None (the F path never
    commits -- the whole point: the occupancy must form F-independently)."""
    agent.e3.select = stub.select
    agent.e3._running_variance = 0.0
    agent.e3.last_scores = stub.scores.clone()
    agent.e3._committed_trajectory = None
    return agent.select_action(cands, {"e3_tick": True})


_EVAL = dict(
    use_closure_commit_beta_coupling=True,
    use_natural_commit_latch_hold=True,
    closure_exclusive_decommit_eval=True,
)
_ENTRY = dict(_EVAL, use_closure_commit_entry=True)


# ----------------------------------------------------------------------
# C1 config defaults + latch attr present/False + default agent never arms
# ----------------------------------------------------------------------
def test_c1_config_defaults():
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM, world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM, self_dim=SELF_DIM, world_dim=WORLD_DIM,
    )
    assert cfg.use_closure_commit_entry is False
    assert cfg.closure_commit_entry_rule_norm_floor == pytest.approx(0.01)
    cfg2 = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM, world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM, self_dim=SELF_DIM, world_dim=WORLD_DIM,
        use_closure_commit_entry=True, closure_commit_entry_rule_norm_floor=0.05,
        use_closure_commit_beta_coupling=True, use_natural_commit_latch_hold=True,
    )
    assert cfg2.use_closure_commit_entry is True
    assert cfg2.closure_commit_entry_rule_norm_floor == pytest.approx(0.05)
    off = _build_agent()
    assert hasattr(off.e3, "_closure_committed_active")
    assert off.e3._closure_committed_active is False
    assert off._ncl_hold_closure_armed_count == 0


# ----------------------------------------------------------------------
# C2 preconditions raise
# ----------------------------------------------------------------------
def test_c2_precondition_requires_coupling():
    with pytest.raises(ValueError, match="use_closure_commit_entry=True requires"):
        _build_agent(
            use_closure_commit_entry=True,
            use_natural_commit_latch_hold=True,
        )


def test_c2_precondition_requires_latch_hold():
    with pytest.raises(ValueError, match="use_closure_commit_entry=True requires"):
        _build_agent(
            use_closure_commit_entry=True,
            use_closure_commit_beta_coupling=True,
        )


# ----------------------------------------------------------------------
# C3 C-KEY (LOAD-BEARING): F-independent arm + sustain; entry-OFF == pre-fix 0
# ----------------------------------------------------------------------
def test_c3_ckey_f_independent_arm_and_sustain():
    cands = _candidates()
    stub = _Stub(committed=False)  # F never commits; _committed_trajectory stays None

    # ENTRY ON: the goal-active rule-directed predicate SETs the F-independent latch ->
    # the eval arms + the hold sustains beta WITHOUT any F-commit.
    on = _build_agent(with_goal_rule=True, **_ENTRY)
    for _ in range(4):
        _tick(on, stub, cands)
    assert on.e3._committed_trajectory is None, "F path must never have committed"
    assert on.e3._closure_committed_active is True, "F-independent latch must be SET"
    assert on._ncl_hold_closure_armed_count > 0, (
        "the closure-plane commit entry must arm the latch-hold with zero F-commits"
    )
    assert on.beta_gate.is_elevated, "the closure-formed occupancy must elevate beta"
    assert on.beta_gate.committed_run_length >= 2, (
        "the hold must SUSTAIN beta across ticks (multi-tick committed_run_length)"
    )
    assert on._ncl_hold_active is True, (
        "the hold must stay armed across ticks (union-aware persistence sustains it)"
    )

    # ENTRY OFF (the legacy eval == pre-fix code): the SAME scenario does NOT arm.
    # _closure_commit_active reduces to (_committed_trajectory is not None) = False.
    off = _build_agent(with_goal_rule=True, **_EVAL)  # use_closure_commit_entry absent
    for _ in range(4):
        _tick(off, stub, cands)
    assert off.e3._closure_committed_active is False
    assert off._ncl_hold_closure_armed_count == 0, "pre-fix: arms exactly 0"
    assert not off.beta_gate.is_elevated, "pre-fix: the occupancy never forms"


# ----------------------------------------------------------------------
# C4 default-OFF bit-identical action stream
# ----------------------------------------------------------------------
def test_c4_default_off_bit_identical():
    cands = _candidates()
    # A reference agent with the closure coupling on but entry OFF, vs the same built
    # with use_closure_commit_entry explicitly False -> identical action stream + no arm.
    a = _build_agent(use_closure_commit_beta_coupling=True)
    b = _build_agent(use_closure_commit_beta_coupling=True, use_closure_commit_entry=False)
    acts_a, acts_b = [], []
    for agent, acc in ((a, acts_a), (b, acts_b)):
        for _ in range(5):
            act = _tick(agent, _Stub(committed=False), cands)
            acc.append(act.detach().clone())
    assert a.e3._closure_committed_active is False
    assert b.e3._closure_committed_active is False
    assert a._ncl_hold_closure_armed_count == 0
    for x, y in zip(acts_a, acts_b):
        assert torch.equal(x, y)


# ----------------------------------------------------------------------
# C5 episode reset clears the latch
# ----------------------------------------------------------------------
def test_c5_reset_clears_latch():
    cands = _candidates()
    on = _build_agent(with_goal_rule=True, **_ENTRY)
    _tick(on, _Stub(committed=False), cands)
    assert on.e3._closure_committed_active is True
    on.reset()
    assert on.e3._closure_committed_active is False, "episode reset must clear the latch"
