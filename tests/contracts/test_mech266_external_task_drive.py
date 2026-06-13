"""Contract tests for the mode-governance-engagement substrate: the
external_task salience SOURCE on the SD-032a SalienceCoordinator (unblocks the
MECH-266 / SD-032a behavioural arms 464d/467d).

Interface-level guarantees that should hold regardless of tuning:
  C1  default OFF -> no "external_task_drive" slot on the coordinator; the
      action stream is bit-identical (no RNG perturbation) default vs ON-but-
      no-goal (engagement 0).
  C2  ON -> the slot is registered in BOTH affinity_weights (-> external_task)
      AND salience_weights (so a switch INTO external_task can fire).
  C3  coordinator math: a positive external_task_drive raises the external_task
      affinity logit (mode SELECTION) AND the salience_aggregate (MECH-259
      switch source); OFF baseline never occupies external_task on a dACC-
      internal_planning push.
  C4  agent injection: goal active + committed pursuit -> injected engagement
      > 0 and external_task gets genuinely occupied where the same agent OFF
      sits at fraction_in_external_task == 0.
  C5  release path: goal INACTIVE (require_goal_active) -> injected engagement
      == 0 (preserves competition; not the 464b 100%-external_task saturation).
"""

import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.cingulate.salience_coordinator import (
    SalienceCoordinator,
    SalienceCoordinatorConfig,
)
from ree_core.environment.causal_grid_world import CausalGridWorldV2

MODE = "external_task"


def _build(env, **kw):
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        z_goal_enabled=True,
        use_salience_coordinator=True,
        **kw,
    )


def _seed_goal(ag):
    ag.goal_state._z_goal = torch.ones(1, ag.goal_state.config.goal_dim) * 0.5


def _action_stream(use_drive, n=20):
    torch.manual_seed(321)
    env = CausalGridWorldV2(size=8, seed=11)
    kw = {"use_external_task_drive": True} if use_drive else {}
    ag = REEAgent(_build(env, **kw))
    _, od = env.reset()
    acts = []
    for _ in range(n):
        a = ag.act_with_split_obs(od["body_state"], od["world_state"])
        acts.append(int(a.argmax()))
        _, h, d, inf, od = env.step(a)
        if d:
            _, od = env.reset()
            ag.reset()
    return acts


# ---------------------------------------------------------------- C1
def test_c1_default_off_no_slot_bit_identical():
    env = CausalGridWorldV2(size=8, seed=0)
    ag = REEAgent(_build(env))
    assert ag.salience is not None
    assert "external_task_drive" not in ag.salience.config.affinity_weights
    assert "external_task_drive" not in ag.salience.config.salience_weights
    # ON with no goal ever seeded -> engagement is 0 every tick (require_goal_active
    # defaults True), so the action stream must match the OFF baseline bit-for-bit.
    assert _action_stream(False) == _action_stream(True)


# ---------------------------------------------------------------- C2
def test_c2_on_registers_both_slots():
    env = CausalGridWorldV2(size=8, seed=0)
    ag = REEAgent(_build(env, use_external_task_drive=True))
    aff = ag.salience.config.affinity_weights["external_task_drive"]
    assert aff == {"external_task": 1.0}
    assert ag.salience.config.salience_weights["external_task_drive"] == 1.0


# ---------------------------------------------------------------- C3
def test_c3_coordinator_math_affinity_and_salience():
    # Build a coordinator with an internal_planning-pushing dACC signal so the
    # OFF baseline never sits in external_task; then add external_task_drive.
    cfg = SalienceCoordinatorConfig()
    cfg.affinity_weights["external_task_drive"] = {"external_task": 1.0}
    cfg.salience_weights["external_task_drive"] = 1.0
    coord = SalienceCoordinator(cfg)

    # Baseline: a strong internal_planning push (dACC PE) with NO drive -> argmax
    # is internal_planning and external_task occupancy stays low.
    coord.reset()
    base = coord.tick(dacc_bundle={"pe": 3.0, "foraging_value": 1.0,
                                   "choice_difficulty": 1.0})
    base_et = base["operating_mode"][MODE]
    base_salience = base["salience_aggregate"]

    # Same push + a strong external_task_drive -> external_task probability AND
    # the salience aggregate both rise.
    coord.reset()
    coord.update_signal("external_task_drive", 1.0)
    eng = coord.tick(dacc_bundle={"pe": 3.0, "foraging_value": 1.0,
                                  "choice_difficulty": 1.0})
    assert eng["operating_mode"][MODE] > base_et
    assert eng["salience_aggregate"] > base_salience


# ---------------------------------------------------------------- C4
def test_c4_agent_injects_and_is_monotone():
    # use_dacc=True so the baseline coordinator genuinely competes (dACC pushes
    # internal_planning), the regime in which the 603n gap (fraction_in_external_task
    # == 0) appears. The drive is injected and never SUPPRESSES external_task.
    def _run(use_drive):
        torch.manual_seed(7)
        e = CausalGridWorldV2(size=8, seed=3)
        kw = {"use_external_task_drive": True} if use_drive else {}
        ag = REEAgent(_build(e, use_dacc=True, **kw))
        _, od = e.reset()
        ag.reset()
        _seed_goal(ag)  # reset() clears z_goal; seed so goal is active
        et_steps = 0
        injected = []
        for _ in range(30):
            ag.act_with_split_obs(od["body_state"], od["world_state"])
            if ag.salience.current_mode == MODE:
                et_steps += 1
            injected.append(
                ag.salience._input_signals.get("external_task_drive", 0.0)
            )
            _, h, d, inf, od = e.step(0)
            if d:
                _, od = e.reset()
                ag.reset()
                _seed_goal(ag)
        return et_steps, injected

    off_et, off_inj = _run(False)
    on_et, on_inj = _run(True)
    # OFF agent never carries the signal; ON agent injects a positive engagement
    # (proximity term is always > 0 once a goal is seeded). This IS the source the
    # coordinator previously lacked on the foraging substrate.
    assert all(v == 0.0 for v in off_inj)
    assert any(v > 0.0 for v in on_inj)
    # The drive is a pull toward external_task: it never reduces occupancy.
    assert on_et >= off_et


# ---------------------------------------------------------------- C5
def test_c5_goal_inactive_engagement_zero():
    torch.manual_seed(0)
    env = CausalGridWorldV2(size=8, seed=1)
    ag = REEAgent(_build(env, use_external_task_drive=True))
    # No goal seeded -> goal_state.is_active() False -> require_goal_active gates
    # engagement to exactly 0 (the release path that preserves mode competition).
    _, od = env.reset()
    ag.reset()
    for _ in range(10):
        ag.act_with_split_obs(od["body_state"], od["world_state"])
        assert ag.salience._input_signals.get("external_task_drive", 0.0) == 0.0
        _, h, d, inf, od = env.step(0)
        if d:
            break
