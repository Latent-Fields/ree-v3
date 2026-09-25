"""MECH-279 PAG freeze / SD-099 defensive-orienting "hold still" action is STAY.

Until 2026-09-25 REEConfig.pag_freeze_noop_action_class defaulted to the
literal 0, which in CausalGridWorld(V2) is (-1, 0) -- a MOVE, not a no-op. So a
frozen agent walked one cell per frozen tick into the top wall (the V3-EXQ-475
"catatonic lock"; REE_assembly
evidence/planning/mech287_lock_dv_inert_pag_path_20260925.md, GFLAG-0506).
SD-099 defensive orienting reuses the same setting and had the same defect.

The fix derives the default from CausalGridWorld.ACTIONS (the unique (0, 0)
entry). These contracts pin:

  C1. The default is DERIVED from the env's own action table: it is the unique
      zero-displacement class, and both the dataclass default and the
      from_dims() default resolve to it. It is not 0.
  C2. The ONLY behaviour change is on freeze/orienting-active ticks: with the
      freeze gate OFF, and with it ON but never firing, a fixed-seed rollout is
      byte-identical whether the no-op class is the new default or the old 0.
  C3. When the freeze fires, the executed action leaves the agent's position
      unchanged (and the same for SD-099 orienting). A positive control with
      the old class 0 must MOVE the agent, so this test is not vacuous against
      the defect it guards.

The gate TRIGGER is forced in C3 (the real tick() runs, its output is then
marked active), because the subject under test is the action override plus the
real env step, not the gate's entry dynamics (test_mech_279_pag_freeze_gate.py
owns those).
"""

from __future__ import annotations

import dataclasses
import hashlib

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils import config as config_mod
from ree_core.utils.config import REEConfig

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_loop import step_once

OLD_DEFECTIVE_NOOP_CLASS = 0  # the pre-2026-09-25 default, (-1, 0) = UP
STEPS = 12


# ---------------------------------------------------------------- helpers ---

def _cfg(env, **overrides) -> REEConfig:
    # action_dim=env.action_dim (5) so the stay class is inside the action
    # space; the tiny fixture's ACTION_DIM=4 has no stay action at all.
    return make_tiny_config(env, action_dim=env.action_dim, **overrides)


def _pos(env):
    return (int(env.agent_x), int(env.agent_y))


def _rollout(seed: int, force=None, **overrides):
    """Fixed-seed rollout. Returns (action_idxs, positions, state_digest, agent).

    force: None | "freeze" | "orienting" -- wrap the real gate tick so its
    output is marked active every tick.
    """
    set_all_seeds(seed)
    env = make_tiny_env(seed=seed)
    agent = REEAgent(_cfg(env, **overrides))
    if force == "freeze":
        real = agent.pag_freeze_gate.tick
        agent.pag_freeze_gate.tick = lambda *a, **k: dataclasses.replace(
            real(*a, **k), freeze_active=True
        )
    elif force == "orienting":
        real = agent.defensive_orienting.tick
        agent.defensive_orienting.tick = lambda *a, **k: dataclasses.replace(
            real(*a, **k), orienting_active=True
        )
    agent.reset()
    _flat, obs = env.reset()
    actions, positions = [], [_pos(env)]
    for _ in range(STEPS):
        _a, idx, _t, obs = step_once(agent, env, obs)
        actions.append(idx)
        positions.append(_pos(env))
    h = hashlib.sha256()
    lat = agent._current_latent
    for t in (lat.z_world, lat.z_self):
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    h.update(torch.get_rng_state().numpy().tobytes())
    return actions, positions, h.hexdigest(), agent


# -------------------------------------------------------------------- C1 ---

def test_c1_default_is_the_env_stay_class():
    zero = [k for k, d in CausalGridWorld.ACTIONS.items() if tuple(d) == (0, 0)]
    assert len(zero) == 1, f"env must have exactly one stay action, got {zero}"
    stay = zero[0]
    assert config_mod.CAUSAL_GRID_WORLD_STAY_ACTION_CLASS == stay
    assert REEConfig().pag_freeze_noop_action_class == stay
    fd = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=5)
    assert fd.pag_freeze_noop_action_class == stay
    # Pin away from the old defect explicitly: class 0 is a MOVE.
    assert tuple(CausalGridWorld.ACTIONS[OLD_DEFECTIVE_NOOP_CLASS]) != (0, 0)
    assert stay != OLD_DEFECTIVE_NOOP_CLASS


# -------------------------------------------------------------------- C2 ---

def test_c2_freeze_gate_off_rollout_unchanged_by_noop_class():
    new = _rollout(7)
    old = _rollout(7, pag_freeze_noop_action_class=OLD_DEFECTIVE_NOOP_CLASS)
    assert new[3].pag_freeze_gate is None
    assert new[0] == old[0]
    assert new[1] == old[1]
    assert new[2] == old[2]


def test_c2_freeze_gate_on_but_silent_rollout_unchanged_by_noop_class():
    # Gate constructed and ticking, but theta so high it can never commit.
    # Instrumental avoidance on too, so the MECH-357 directed-action credit
    # (the one other consumer touched by the fix) is exercised.
    kw = dict(
        use_pag_freeze_gate=True,
        pag_theta_freeze=1e9,
        use_instrumental_avoidance=True,
    )
    new = _rollout(11, **kw)
    old = _rollout(11, pag_freeze_noop_action_class=OLD_DEFECTIVE_NOOP_CLASS, **kw)
    gate = new[3].pag_freeze_gate
    assert gate is not None and not gate.is_active
    assert new[0] == old[0]
    assert new[1] == old[1]
    assert new[2] == old[2]
    assert (
        new[3]._ia_last_action_directed == old[3]._ia_last_action_directed
    )


# -------------------------------------------------------------------- C3 ---

def _assert_held_still(actions, positions, stay):
    assert all(a == stay for a in actions), actions
    assert all(p == positions[0] for p in positions), positions


def test_c3_freeze_fires_position_unchanged():
    stay = config_mod.CAUSAL_GRID_WORLD_STAY_ACTION_CLASS
    actions, positions, _d, agent = _rollout(
        3, force="freeze", use_pag_freeze_gate=True
    )
    _assert_held_still(actions, positions, stay)


def test_c3_freeze_does_not_count_as_directed_avoidance():
    _a, _p, _d, agent = _rollout(
        3, force="freeze", use_pag_freeze_gate=True, use_instrumental_avoidance=True
    )
    assert agent._ia_last_action_directed is False


def test_c3_orienting_fires_position_unchanged():
    stay = config_mod.CAUSAL_GRID_WORLD_STAY_ACTION_CLASS
    actions, positions, _d, _agent = _rollout(
        3, force="orienting", use_defensive_orienting=True
    )
    _assert_held_still(actions, positions, stay)


def test_c3_positive_control_old_class_moves_the_agent():
    # The old default must fail C3's predicate, else C3 is vacuous.
    actions, positions, _d, _agent = _rollout(
        3,
        force="freeze",
        use_pag_freeze_gate=True,
        pag_freeze_noop_action_class=OLD_DEFECTIVE_NOOP_CLASS,
    )
    assert all(a == OLD_DEFECTIVE_NOOP_CLASS for a in actions)
    assert any(p != positions[0] for p in positions), (
        "positive control: class 0 (UP) should move the agent from "
        f"{positions[0]}; pick a seed whose start is not on the top edge"
    )
