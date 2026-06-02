"""MECH-342 contract tests: maintenance-time readiness-driven commitment
release (B3b).

Contracts (interface-level guarantees, NOT magnitude thresholds):

  C1  config defaults: use_maintenance_release False; from_dims surfaces the
      knobs with no-op defaults.
  C2  master OFF: agent.maintenance_release is None; bit-identical (the
      release branch is skipped).
  C3  GRADED, not a one-shot Schmitt flag: a single below-floor tick does not
      fire; sustained deficit fires after ~release_bound/accumulation_rate
      ticks, scaled by deficit magnitude.
  C4  OR-composition (De Morgan dual of the MECH-090 AND admission): either
      axis failing drives release pressure; a None axis is inert.
  C5  hysteretic reengagement: recovery above the reengage level leaks
      pressure; the dead-band (between floor and reengage) holds.
  C6  targeted release in the agent loop: degraded readiness while
      beta-elevated releases the latch, resets _committed_step_idx, clears
      _committed_anchor_keys AND e3._committed_trajectory.
  C7  no false abort: healthy readiness never releases (premature-abort guard).
  C8  MECH-094: simulation_mode=True is a no-op (no accumulate, returns False).
  C9  lifecycle: reset_pressure() zeroes the accumulator; reset() zeroes
      accumulator + diagnostics.
  C10 config validation: reengage<floor and cap<bound raise ValueError.
  C11 592f gap reproduction: with use_maintenance_release=False, degraded
      readiness while beta-elevated produces NO release (the exact 592f
      FAIL_NO_RELEASE_AUTHORITY signature).
"""

from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.policy import (
    CommitMaintenanceRelease,
    CommitMaintenanceReleaseConfig,
)
from ree_core.predictors.e2_fast import Trajectory
from ree_core.predictors.e3_selector import SelectionResult
from ree_core.utils.config import REEConfig

ACTION_DIM = 4
SELF_DIM = 8
WORLD_DIM = 8
BODY_OBS_DIM = 4
WORLD_OBS_DIM = 8
H = 3


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
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


def _scores(margin: float) -> torch.Tensor:
    # REE lower-is-better: winner 0.0, runner-up = margin.
    return torch.tensor([0.0, float(margin)])


class _Stub:
    def __init__(self):
        self.committed = True
        self.scores = _scores(0.5)

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


def _reg(**overrides) -> CommitMaintenanceRelease:
    cfg = CommitMaintenanceReleaseConfig(use_maintenance_release=True, **overrides)
    return CommitMaintenanceRelease(cfg)


def _build_agent(mr_on: bool) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
        commit_readiness_initial=1.0,
        use_maintenance_release=mr_on,
        maintenance_release_nav_floor=0.3,
        maintenance_release_accumulation_rate=0.2,
        use_sleep_loop=False,
        sws_enabled=False,
        rem_enabled=False,
        use_sleep_aggregation_cluster=False,
    )
    cfg.heartbeat.beta_gate_bistable = True
    cfg.heartbeat.use_commit_readiness_gate = True
    cfg.heartbeat.commit_readiness_floor = 0.05
    agent = REEAgent(cfg)
    agent.reset()
    return agent


def _run_forced(mr_on: bool, nav: float, margin: float, ticks: int = 8):
    agent = _build_agent(mr_on)
    stub = _Stub()
    agent.e3.select = stub.select
    agent.beta_gate.elevate()
    agent.e3._committed_trajectory = _candidates()[0]
    agent.e3._running_variance = 0.0
    agent._committed_step_idx = 0
    stub.scores = _scores(margin)
    release_tick = None
    cands = _candidates()
    for t in range(ticks):
        agent.commit_readiness.notify_outcome(nav)
        agent.e3.last_scores = _scores(margin)
        agent.select_action(cands, {"e3_tick": True})
        if release_tick is None and not agent.beta_gate.is_elevated:
            release_tick = t
    return agent, release_tick


# ----------------------------------------------------------------------
# C1 config defaults
# ----------------------------------------------------------------------
def test_c1_config_defaults():
    c = CommitMaintenanceReleaseConfig()
    assert c.use_maintenance_release is False
    assert c.score_margin_floor == 0.05
    assert c.score_margin_reengage == 0.10
    assert c.nav_floor == 0.3
    assert c.nav_reengage == 0.5
    assert c.accumulation_rate == 0.2
    assert c.leak_rate == 0.1
    assert c.release_bound == 1.0
    assert c.pressure_cap == 1.5
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
    )
    assert cfg.use_maintenance_release is False
    assert cfg.maintenance_release_nav_floor == 0.3


# ----------------------------------------------------------------------
# C2 master OFF
# ----------------------------------------------------------------------
def test_c2_master_off_no_module():
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
    )
    agent = REEAgent(cfg)
    assert agent.maintenance_release is None


# ----------------------------------------------------------------------
# C3 graded, not one-shot
# ----------------------------------------------------------------------
def test_c3_graded_not_oneshot():
    # A single below-floor tick must NOT fire.
    mr = _reg()
    assert mr.tick(score_margin=1.0, n_candidates=4, nav_competence=0.0) is False
    assert mr.get_pressure() == pytest.approx(0.2)
    # Sustained max nav deficit fires after release_bound/accumulation_rate
    # ticks (5 at default 1.0/0.2), proportional to deficit magnitude.
    mr2 = _reg()
    fires = [
        mr2.tick(score_margin=1.0, n_candidates=4, nav_competence=0.0)
        for _ in range(6)
    ]
    assert sum(fires) == 1
    fire_idx = fires.index(True)
    assert fire_idx in (4, 5)  # FP boundary -> 5th or 6th tick


def test_c3_half_deficit_fires_later():
    # nav=0.15 -> deficit 0.5 -> ~0.1/tick -> fires ~tick 10-11 (FP boundary).
    mr = _reg()
    fires = [
        mr.tick(score_margin=1.0, n_candidates=4, nav_competence=0.15)
        for _ in range(12)
    ]
    assert sum(fires) == 1
    assert fires.index(True) in (9, 10)


# ----------------------------------------------------------------------
# C4 OR-composition
# ----------------------------------------------------------------------
def test_c4_either_axis_drives_release():
    # Decisiveness axis alone (nav healthy, margin << floor).
    mr_d = _reg()
    f_d = [
        mr_d.tick(score_margin=0.0, n_candidates=4, nav_competence=1.0)
        for _ in range(6)
    ]
    assert any(f_d)
    # nav axis alone (margin healthy, nav << floor).
    mr_n = _reg()
    f_n = [
        mr_n.tick(score_margin=1.0, n_candidates=4, nav_competence=0.0)
        for _ in range(6)
    ]
    assert any(f_n)


def test_c4_none_axes_inert():
    # Both axes absent -> combined deficit 0, recovered True -> never fires.
    mr = _reg()
    assert not any(
        mr.tick(score_margin=None, n_candidates=0, nav_competence=None)
        for _ in range(30)
    )
    assert mr.get_pressure() == 0.0


def test_c4_max_combine_takes_worse_axis():
    # deficit_d (margin 0.025 -> 0.5) vs deficit_n (nav 0.0 -> 1.0) -> max=1.0.
    mr = _reg()
    mr.tick(score_margin=0.025, n_candidates=4, nav_competence=0.0)
    st = mr.get_state()
    assert st["last_deficit_nav"] == pytest.approx(1.0)
    assert st["last_deficit_decisiveness"] == pytest.approx(0.5)
    assert st["last_combined_deficit"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# C5 hysteresis / reengagement
# ----------------------------------------------------------------------
def test_c5_reengage_leak():
    mr = _reg()
    mr.tick(1.0, 4, 0.0)
    mr.tick(1.0, 4, 0.0)  # pressure 0.4
    assert mr.get_pressure() == pytest.approx(0.4)
    mr.tick(1.0, 4, 0.9)  # nav 0.9 >= reengage 0.5 -> leak 0.1
    assert mr.get_pressure() == pytest.approx(0.3)


def test_c5_deadband_holds():
    mr = _reg()
    mr.tick(1.0, 4, 0.0)
    mr.tick(1.0, 4, 0.0)  # pressure 0.4
    # nav 0.4: above floor 0.3 (deficit 0) but below reengage 0.5 -> hold.
    mr.tick(1.0, 4, 0.4)
    assert mr.get_pressure() == pytest.approx(0.4)
    assert mr.get_state()["mech342_n_hold"] == 1


# ----------------------------------------------------------------------
# C6 targeted release in agent loop
# ----------------------------------------------------------------------
def test_c6_agent_release_on_degraded_nav():
    agent, rel = _run_forced(mr_on=True, nav=0.0, margin=0.0)
    assert rel is not None
    assert agent.beta_gate.is_elevated is False
    assert agent.e3._committed_trajectory is None
    assert agent._committed_step_idx == 0
    assert agent._committed_anchor_keys is None
    assert agent.maintenance_release.get_state()["mech342_n_fires"] >= 1


# ----------------------------------------------------------------------
# C7 no false abort
# ----------------------------------------------------------------------
def test_c7_no_false_abort_healthy():
    agent, rel = _run_forced(mr_on=True, nav=1.0, margin=1.0)
    assert rel is None
    assert agent.beta_gate.is_elevated is True
    assert agent.e3._committed_trajectory is not None


# ----------------------------------------------------------------------
# C8 MECH-094 simulation gate
# ----------------------------------------------------------------------
def test_c8_simulation_mode_noop():
    mr = _reg()
    out = mr.tick(score_margin=0.0, n_candidates=4, nav_competence=0.0,
                  simulation_mode=True)
    assert out is False
    assert mr.get_pressure() == 0.0
    assert mr.get_state()["mech342_n_simulation_skips"] == 1
    assert mr.get_state()["mech342_n_ticks"] == 0


# ----------------------------------------------------------------------
# C9 lifecycle
# ----------------------------------------------------------------------
def test_c9_reset_pressure_and_reset():
    mr = _reg()
    mr.tick(1.0, 4, 0.0)
    mr.tick(1.0, 4, 0.0)
    assert mr.get_pressure() > 0.0
    mr.reset_pressure()
    assert mr.get_pressure() == 0.0
    # diagnostics survive reset_pressure
    assert mr.get_state()["mech342_n_accumulate"] == 2
    mr.reset()
    assert mr.get_pressure() == 0.0
    assert mr.get_state()["mech342_n_accumulate"] == 0
    assert mr.get_state()["mech342_n_ticks"] == 0


# ----------------------------------------------------------------------
# C10 config validation
# ----------------------------------------------------------------------
def test_c10_config_validation():
    with pytest.raises(ValueError):
        CommitMaintenanceRelease(
            CommitMaintenanceReleaseConfig(nav_floor=0.3, nav_reengage=0.1)
        )
    with pytest.raises(ValueError):
        CommitMaintenanceRelease(
            CommitMaintenanceReleaseConfig(release_bound=1.0, pressure_cap=0.5)
        )
    with pytest.raises(ValueError):
        CommitMaintenanceRelease(
            CommitMaintenanceReleaseConfig(accumulation_rate=0.0)
        )


# ----------------------------------------------------------------------
# C11 592f gap reproduction (master OFF)
# ----------------------------------------------------------------------
def test_c11_592f_gap_reproduced_when_off():
    # With use_maintenance_release=False, degraded readiness while
    # beta-elevated produces NO release: the FAIL_NO_RELEASE_AUTHORITY
    # signature 592f measured. (commit_readiness_gate is still on, mirroring
    # the 592f config -- the admission gate fires predicates but has no
    # maintenance authority.)
    agent, rel = _run_forced(mr_on=False, nav=0.0, margin=0.0)
    assert rel is None
    assert agent.beta_gate.is_elevated is True
    assert agent.e3._committed_trajectory is not None
    assert agent.maintenance_release is None
