"""Contract tests: natural-commit LATCH-HOLD lever (rung-6 amend, 2026-06-21,
failure_autopsy_V3-EXQ-460i).

The HOLD is SEPARATE from the rung-6 RELEASE (NaturalCommitUrgencyRelease): it
ESTABLISHES the sustained natural-commit beta-latch occupancy the release acts on.
V3-EXQ-460i found the release fired ZERO because the 460h sustained ~2400-step
monolithic natural-commit hold did not reproduce -- the active SD-034 de-commit
control-plane fragments the latch to ~1-tick blips even with the release OFF, so
there was no sustained occupancy to shorten. The hold re-asserts a natural-commit-
armed beta latch each tick (against whatever churns it) so the OFF baseline
sustains BY CONSTRUCTION, while YIELDING to the three principled releases so it
never papers over them.

Contracts (interface-level guarantees, NOT magnitude thresholds):

  C1  config defaults: use_natural_commit_latch_hold False; max_ticks 0; from_dims
      surfaces both with no-op defaults; master OFF -> the hold never arms
      (_ncl_hold_active stays False after a committed select_action tick).
  C2  arm site: a fresh NATURAL commit (result.committed) with the flag ON arms
      the hold (_ncl_hold_active True). A purely closure-coupled elevation does NOT.
  C3  re-assert against churn (LOAD-BEARING): once armed, with the committed
      trajectory persisting but the agent no longer re-committing (no natural
      commit, no coupling) and the latch dropped each tick, the hold ON re-elevates
      the latch (sustained) where the hold OFF leaves it dropped.
  C4  yield to closure de-commit: an active SD-034 refractory (apply_refractory>0)
      disarms the hold and the latch is NOT re-elevated (the MECH-446 occupancy-
      drop DV is preserved).
  C5  yield when the committed trajectory ends: e3._committed_trajectory None
      disarms the hold (the natural commit genuinely ended).
  C6  max-ticks safety cap: the hold disarms after natural_commit_latch_hold_max_ticks
      re-assert ticks (guards a degenerate config from latching forever).
  C7  bit-identical OFF: the flag-OFF agent never touches the hold state
      (_ncl_hold_reassert_count == 0) and its latch trace matches a reference OFF run.
"""

from __future__ import annotations

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


def _scores(gap_norm: float) -> torch.Tensor:
    return torch.tensor([0.0, float(gap_norm), 1.0])


class _Stub:
    """E3.select stub with a toggleable committed flag (simulate the agent
    re-committing or NOT re-committing while a committed trajectory persists)."""

    def __init__(self, committed: bool = True, gap_norm: float = 1.0):
        self.committed = committed
        self.scores = _scores(gap_norm)

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


def _build_agent(**hold_kwargs) -> REEAgent:
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
        **hold_kwargs,
    )
    cfg.heartbeat.beta_gate_bistable = True
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _commit_tick(agent, stub, cands):
    """One E3 select_action tick with the given stub committed state."""
    agent.e3.select = stub.select
    agent.e3._running_variance = 0.0
    agent.e3.last_scores = stub.scores.clone()
    agent.e3._committed_trajectory = cands[0]
    agent.select_action(cands, {"e3_tick": True})


# ----------------------------------------------------------------------
# C1 config defaults + master OFF
# ----------------------------------------------------------------------
def test_c1_config_defaults_and_master_off():
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
    )
    assert cfg.use_natural_commit_latch_hold is False
    assert cfg.natural_commit_latch_hold_max_ticks == 0
    cfg2 = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        use_natural_commit_latch_hold=True,
        natural_commit_latch_hold_max_ticks=25,
    )
    assert cfg2.use_natural_commit_latch_hold is True
    assert cfg2.natural_commit_latch_hold_max_ticks == 25

    off = _build_agent()  # master off (default)
    stub = _Stub(committed=True)
    _commit_tick(off, stub, _candidates())
    assert off._ncl_hold_active is False
    assert off._ncl_hold_reassert_count == 0


# ----------------------------------------------------------------------
# C2 arm site
# ----------------------------------------------------------------------
def test_c2_arm_on_natural_commit():
    on = _build_agent(use_natural_commit_latch_hold=True)
    _commit_tick(on, _Stub(committed=True), _candidates())
    assert on._ncl_hold_active is True


# ----------------------------------------------------------------------
# C3 re-assert against churn (LOAD-BEARING)
# ----------------------------------------------------------------------
def test_c3_reassert_against_churn_load_bearing():
    cands = _candidates()
    # Hold ON: arm on a natural commit, then the agent stops re-committing
    # (stub.committed=False -> bistable would NOT re-elevate) while the committed
    # trajectory persists and the latch is dropped each tick -> the hold sustains.
    on = _build_agent(use_natural_commit_latch_hold=True)
    _commit_tick(on, _Stub(committed=True), cands)  # arm + elevate
    assert on.beta_gate.is_elevated
    stub_off = _Stub(committed=False)
    on.e3.select = stub_off.select
    elevated_on = []
    for _ in range(6):
        on.beta_gate.release()  # simulate the de-commit churn dropping the latch
        on.e3.last_scores = stub_off.scores.clone()
        on.e3._committed_trajectory = cands[0]  # commitment persists
        on.select_action(cands, {"e3_tick": True})
        elevated_on.append(bool(on.beta_gate.is_elevated))
    assert all(elevated_on), "hold ON must re-assert the latch against churn"
    assert on._ncl_hold_reassert_count >= 1

    # Hold OFF: same churn, no re-commit -> the latch stays dropped (the bistable
    # path cannot re-elevate without a commit; nothing else holds it).
    off = _build_agent()
    _commit_tick(off, _Stub(committed=True), cands)
    stub_off2 = _Stub(committed=False)
    off.e3.select = stub_off2.select
    elevated_off = []
    for _ in range(6):
        off.beta_gate.release()
        off.e3.last_scores = stub_off2.scores.clone()
        off.e3._committed_trajectory = cands[0]
        off.select_action(cands, {"e3_tick": True})
        elevated_off.append(bool(off.beta_gate.is_elevated))
    assert not any(elevated_off), "hold OFF must NOT re-assert the dropped latch"


# ----------------------------------------------------------------------
# C4 yield to closure de-commit (refractory)
# ----------------------------------------------------------------------
def test_c4_yield_to_closure_refractory():
    cands = _candidates()
    on = _build_agent(use_natural_commit_latch_hold=True)
    _commit_tick(on, _Stub(committed=True), cands)  # arm
    assert on._ncl_hold_active is True
    # A closure de-commit drops the latch and installs a refractory window.
    stub_off = _Stub(committed=False)
    on.e3.select = stub_off.select
    on.beta_gate.release()
    on.beta_gate.apply_refractory(5)
    on.e3.last_scores = stub_off.scores.clone()
    on.e3._committed_trajectory = cands[0]
    on.select_action(cands, {"e3_tick": True})
    assert on._ncl_hold_active is False, "hold must yield to the closure de-commit"
    assert not on.beta_gate.is_elevated, "hold must NOT re-elevate during refractory"


# ----------------------------------------------------------------------
# C5 yield when the committed trajectory ends
# ----------------------------------------------------------------------
def test_c5_yield_when_commit_ends():
    cands = _candidates()
    on = _build_agent(use_natural_commit_latch_hold=True)
    _commit_tick(on, _Stub(committed=True), cands)  # arm
    assert on._ncl_hold_active is True
    stub_off = _Stub(committed=False)
    on.e3.select = stub_off.select
    on.beta_gate.release()
    on.e3.last_scores = stub_off.scores.clone()
    on.e3._committed_trajectory = None  # the natural commit genuinely ended
    on.select_action(cands, {"e3_tick": True})
    assert on._ncl_hold_active is False
    assert not on.beta_gate.is_elevated


# ----------------------------------------------------------------------
# C6 max-ticks safety cap
# ----------------------------------------------------------------------
def test_c6_max_ticks_cap():
    cands = _candidates()
    on = _build_agent(
        use_natural_commit_latch_hold=True,
        natural_commit_latch_hold_max_ticks=3,
    )
    _commit_tick(on, _Stub(committed=True), cands)  # arm
    stub_off = _Stub(committed=False)
    on.e3.select = stub_off.select
    for _ in range(5):
        on.beta_gate.release()
        on.e3.last_scores = stub_off.scores.clone()
        on.e3._committed_trajectory = cands[0]
        on.select_action(cands, {"e3_tick": True})
    assert on._ncl_hold_active is False, "hold must disarm at the max-ticks cap"
    assert on._ncl_hold_ticks <= 3


# ----------------------------------------------------------------------
# C7 bit-identical OFF
# ----------------------------------------------------------------------
def test_c7_bit_identical_off():
    cands = _candidates()
    # Reference OFF run (deterministic via the stub).
    ref = _build_agent()
    _commit_tick(ref, _Stub(committed=True), cands)
    ref_stub = _Stub(committed=True)
    ref.e3.select = ref_stub.select
    ref_trace = []
    for _ in range(8):
        ref.e3.last_scores = ref_stub.scores.clone()
        ref.e3._committed_trajectory = cands[0]
        ref.select_action(cands, {"e3_tick": True})
        ref_trace.append(bool(ref.beta_gate.is_elevated))

    # A second OFF agent must produce the same latch trace and never touch the
    # hold state (the flag-OFF path is inert).
    off = _build_agent(use_natural_commit_latch_hold=False)
    _commit_tick(off, _Stub(committed=True), cands)
    off_stub = _Stub(committed=True)
    off.e3.select = off_stub.select
    off_trace = []
    for _ in range(8):
        off.e3.last_scores = off_stub.scores.clone()
        off.e3._committed_trajectory = cands[0]
        off.select_action(cands, {"e3_tick": True})
        off_trace.append(bool(off.beta_gate.is_elevated))

    assert off_trace == ref_trace
    assert off._ncl_hold_active is False
    assert off._ncl_hold_reassert_count == 0
