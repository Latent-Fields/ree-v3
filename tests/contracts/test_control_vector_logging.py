"""Contract tests for ControlVector logging (recommendation B).

Read-only, default-OFF telemetry that surfaces four separately-inspectable
control signals each E3 tick (V_outcome / C_effort / C_time / G_vigor) and
EXPOSES the ARC-068-vs-MECH-320 collapse (C_time and G_vigor are both w*v_t
for the same MECH-320 v_t scalar). See the four-signal control adjudication
2026-06-07 and ree-v3/CLAUDE.md.

C1  default OFF -> _last_control_vector stays empty (no-op).
C2  ON -> _last_control_vector populated with the four signals + shared + authority.
C3  collapse: C_time.potential == w_passive*v_t and G_vigor.potential == w_action*v_t
    for the SAME shared v_t.
C4  bit-identical OFF: ON-vs-OFF emit identical action streams under matched seeds
    (pure telemetry, no scoring/selection effect).
"""

import random

import numpy as np
import torch

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments._harness import StepHarness


def _mk_env(seed=None):
    # A seed MUST be threaded through for any test that asserts bit-identity:
    # CausalGridWorldV2 builds its internal RNG as np.random.default_rng(seed),
    # and seed=None pulls fresh OS entropy, so the per-episode microhabitat /
    # Voronoi zone map is non-deterministic. Two envs built without a seed draw
    # independently-random maps, which desynchronises the action stream and made
    # the c4 OFF-vs-ON bit-identity assertion flaky (passes ~3/4 of the time).
    return CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, seed=seed)


def _dims(env):
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )


def _run(cfg, steps=6, seed=0):
    # Reseed every global RNG this run can touch so the run is fully
    # deterministic regardless of which sibling tests (or other test modules
    # during full-suite collection) ran first. torch drives the agent's
    # stochastic ops; the env RNG is seeded via _mk_env(seed) below; numpy /
    # python-random are reset defensively for any incidental global draw.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = _mk_env(seed=seed)
    agent = REEAgent(cfg)
    results = StepHarness(agent, env, train_mode=True, seed=seed).run_episode(
        max_steps=steps
    )
    actions = [int(r.action.argmax().item()) for r in results]
    return agent, actions


def test_c1_default_off_no_population():
    env = _mk_env()
    cfg = REEConfig.from_dims(**_dims(env))
    assert cfg.use_control_vector_logging is False
    agent, _ = _run(cfg)
    assert agent._last_control_vector == {}


def test_c2_on_populates_four_signals():
    env = _mk_env()
    cfg = REEConfig.from_dims(
        use_control_vector_logging=True,
        use_tonic_vigor=True,
        use_dacc=True,
        **_dims(env),
    )
    cfg.tonic_vigor_v_t_floor = 0.5
    agent, _ = _run(cfg)
    cv = agent._last_control_vector
    assert cv, "ON must populate the control vector"
    for k in ("V_outcome", "C_effort", "C_time", "G_vigor", "shared", "authority"):
        assert k in cv, f"missing control signal: {k}"
    # V_outcome is the primary value axis (present once E3 has scored).
    assert cv["V_outcome"]["present"] is True
    assert "value_mean" in cv["V_outcome"]


def test_c3_ctime_gvigor_collapse_to_one_vt():
    env = _mk_env()
    cfg = REEConfig.from_dims(
        use_control_vector_logging=True,
        use_tonic_vigor=True,
        **_dims(env),
    )
    cfg.tonic_vigor_v_t_floor = 0.5  # force a non-zero shared scalar
    agent, _ = _run(cfg)
    cv = agent._last_control_vector
    s = cv["shared"]
    vt = s["tonic_vigor_v_t"]
    assert vt > 0.0, "v_t should be > 0 under the forced floor"
    # The collapse: both halves are w * v_t for the SAME v_t.
    assert abs(cv["C_time"]["potential"] - s["w_passive"] * vt) < 1e-9
    assert abs(cv["G_vigor"]["potential"] - s["w_action"] * vt) < 1e-9


def test_c4_bit_identical_off_vs_on_action_stream():
    env = _mk_env()
    base = _dims(env)
    cfg_off = REEConfig.from_dims(use_tonic_vigor=True, use_dacc=True, **base)
    cfg_on = REEConfig.from_dims(
        use_control_vector_logging=True, use_tonic_vigor=True, use_dacc=True, **base
    )
    _, actions_off = _run(cfg_off, seed=7)
    _, actions_on = _run(cfg_on, seed=7)
    assert actions_off == actions_on, (
        "ControlVector logging must not change the action stream "
        f"(off={actions_off} on={actions_on})"
    )
