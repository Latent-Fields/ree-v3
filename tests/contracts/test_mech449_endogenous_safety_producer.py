"""Contracts for the MECH-449 ENDOGENOUS safety producer (2026-09-24).

REEAgent._endogenous_gng_safety populates the Go/No-Go gate's ``safety`` axis
from the harm valuation pathway (E3.harm_eval_head over each candidate's
PREDICTED z_world states), calibrated against a per-agent running harm scale
(bias-corrected EMA mean/variance; z >= gng_safety_z_threshold fires; sd floored
at gng_safety_sd_floor). Default OFF -> never runs -> bit-identical.

The values under test are read from the module (REEAgent / REEConfig), never
restated: the harm HEAD is the producer's input and is set deterministically
here so the calibration arithmetic downstream of it can be checked; the
producer, its running scale, its mapping and its wiring into e3.select are the
code under test.
"""

from __future__ import annotations

import math

import torch

from ree_core.agent import REEAgent
from ree_core.predictors.e2_fast import Trajectory
from ree_core.utils.config import E3Config, REEConfig

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_env import make_tiny_env
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_loop import run_episode


WORLD_DIM = 16


def _agent(**overrides) -> REEAgent:
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env, **overrides)
    return REEAgent(cfg)


def _linear_harm_head(agent: REEAgent, slope: float, bias: float = 0.0) -> None:
    """harm(z) = sigmoid(bias + slope * z[0]) -- a deterministic, monotone head."""
    head = agent.e3.harm_eval_head
    with torch.no_grad():
        first = head[0]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1.0  # hidden unit 0 = relu(z[0])
        last = head[2]
        last.weight.zero_()
        last.bias.fill_(bias)
        last.weight[0, 0] = slope


def _cand(world_val: float, horizon: int = 3, action_dim: int = 5) -> Trajectory:
    states = [torch.zeros(1, WORLD_DIM) for _ in range(horizon + 1)]
    ws = [torch.zeros(1, WORLD_DIM)] + [
        torch.full((1, WORLD_DIM), float(world_val)) for _ in range(horizon)
    ]
    actions = torch.zeros(1, horizon, action_dim)
    actions[:, 0, 0] = 1.0
    return Trajectory(states=states, actions=actions, world_states=ws)


def test_default_off_and_from_dims_plumbs_every_knob():
    assert E3Config().use_gng_endogenous_safety is False
    cfg = REEConfig.from_dims(
        body_obs_dim=4, world_obs_dim=8, action_dim=5,
        use_gng_endogenous_safety=True,
        gng_safety_z_threshold=3.0,
        gng_safety_ema_decay=0.9,
        gng_safety_sd_floor=0.02,
        gng_safety_warmup_samples=7,
    )
    e3 = cfg.e3
    assert e3.use_gng_endogenous_safety is True
    assert e3.gng_safety_z_threshold == 3.0
    assert e3.gng_safety_ema_decay == 0.9
    assert e3.gng_safety_sd_floor == 0.02
    assert e3.gng_safety_warmup_samples == 7


def test_warmup_emits_no_veto():
    agent = _agent(use_gng_endogenous_safety=True, gng_safety_warmup_samples=50)
    _linear_harm_head(agent, slope=5.0)
    out = agent._endogenous_gng_safety([_cand(0.0), _cand(3.0)])
    assert out is not None and out.shape == (2,)
    assert float(out.abs().sum()) == 0.0
    d = agent.gng_safety_diagnostics()
    assert d["n_samples"] == 2 and d["n_signal_fired"] == 0


def test_flat_landscape_never_fires_sd_floor():
    """An undiscriminative head (spread << sd floor) must not veto, even though
    z-scores of pure spread against a collapsed running sd would cross +2."""
    agent = _agent(use_gng_endogenous_safety=True, gng_safety_warmup_samples=10)
    _linear_harm_head(agent, slope=0.004)  # harm spread ~0.001 over z in [0, 1]
    vals = [0.0] * 9 + [1.0]  # one candidate sits ~3 SD above the rest
    for t in range(60):
        agent._endogenous_gng_safety([_cand(v) for v in vals])
    d = agent.gng_safety_diagnostics()
    assert d["n_candidates_scored"] == 600
    assert d["n_signal_fired"] == 0
    # The floor is what bites: the true running sd is far below it.
    assert math.sqrt(d["ema_var"]) < agent.config.e3.gng_safety_sd_floor
    # Measure the blind spot: WITHOUT the floor the same landscape does fire.
    agent2 = _agent(use_gng_endogenous_safety=True, gng_safety_warmup_samples=10,
                    gng_safety_sd_floor=1e-9)
    _linear_harm_head(agent2, slope=0.004)
    for t in range(60):
        agent2._endogenous_gng_safety([_cand(v) for v in vals])
    assert agent2.gng_safety_diagnostics()["n_signal_fired"] > 0


def test_outlier_candidate_fires_and_maps_onto_gate_floor():
    agent = _agent(use_gng_endogenous_safety=True, gng_safety_warmup_samples=20)
    _linear_harm_head(agent, slope=2.0, bias=-2.0)
    # Baseline: a spread of ordinary candidates builds the running scale.
    for t in range(40):
        agent._endogenous_gng_safety([_cand(v) for v in (0.0, 0.1, 0.2, 0.3)])
    out = agent._endogenous_gng_safety([_cand(0.1), _cand(0.2), _cand(3.0)])
    last = agent.gng_safety_diagnostics()["last"]
    floor = agent.config.e3.gng_safety_floor
    z_thr = agent.config.e3.gng_safety_z_threshold
    assert last["armed"] is True
    assert last["fired"] == [False, False, True]
    assert last["z"][2] >= z_thr and max(last["z"][:2]) < z_thr
    # safety >= floor exactly where z >= threshold (the gate's own test).
    for s, z in zip(out.tolist(), last["z"]):
        assert (s >= floor) == (z >= z_thr)
        assert 0.0 <= s <= 1.0


def test_running_scale_persists_across_episode_reset():
    agent = _agent(use_gng_endogenous_safety=True)
    _linear_harm_head(agent, slope=1.0)
    agent._endogenous_gng_safety([_cand(0.5), _cand(1.0)])
    n_before = agent.gng_safety_diagnostics()["n_samples"]
    agent.reset()
    assert agent.gng_safety_diagnostics()["n_samples"] == n_before == 2
    agent.reset_gng_safety_state()
    assert agent.gng_safety_diagnostics()["n_samples"] == 0


def _capture_select(agent: REEAgent):
    calls = []
    orig = agent.e3.select

    def _wrapped(*a, **kw):
        res = orig(*a, **kw)
        calls.append({
            "signals": kw.get("go_nogo_signals"),
            "active": bool(agent.e3.last_score_diagnostics.get(
                "go_nogo_constitution_active", False)),
            "n_safety_nogo": int(agent.e3.last_score_diagnostics.get(
                "go_nogo_n_safety_nogo", 0)),
            "k": len(a[0]) if a else None,
        })
        return res

    agent.e3.select = _wrapped
    return calls


# The gate's arming chain: constitution + an F-built eligible set (f_demotion)
# + a live modulatory accumulator (MECH-341 score diversity supplies one on the
# tiny config; dACC needs the affective harm stream the tiny env lacks) + K >= 2.
_ARMED = dict(use_go_nogo_constitution=True, use_f_eligibility_demotion=True,
              use_e3_score_diversity=True)


def test_flag_off_sends_no_safety_axis_on_a_real_rollout():
    agent = _agent(**_ARMED)
    calls = _capture_select(agent)
    set_all_seeds(1)
    run_episode(agent, make_tiny_env(seed=1), steps=30)
    assert calls, "e3.select never ran"
    for c in calls:
        assert c["signals"] is None or "safety" not in c["signals"]
    assert agent.gng_safety_diagnostics()["n_ticks_scored"] == 0


def test_flag_on_wires_safety_into_gate_and_accumulates_count():
    agent = _agent(use_gng_endogenous_safety=True, gng_safety_warmup_samples=5,
                   gng_safety_sd_floor=1e-6, **_ARMED)
    # Default-init head; the sd floor is lowered so the small spread it gives on
    # REAL candidates (this is a wiring test, not a calibration test) can fire.
    calls = _capture_select(agent)
    set_all_seeds(1)
    run_episode(agent, make_tiny_env(seed=1), steps=60)
    d = agent.gng_safety_diagnostics()
    assert calls and d["n_ticks_scored"] == len(calls)
    for c in calls:
        assert c["signals"] is not None and "safety" in c["signals"]
        assert c["signals"]["safety"].numel() == c["k"]
    assert d["n_gate_active_ticks"] == sum(1 for c in calls if c["active"])
    assert d["n_gate_active_ticks"] > 0, "arming chain did not arm the gate"
    assert d["n_safety_nogo_applied"] == sum(c["n_safety_nogo"] for c in calls if c["active"])
    assert d["n_signal_fired"] > 0
