"""Contract tests for SD-069 phasic_surprise_burst (MECH-063 sub-claim ii).

LC-NE PHASIC complement to MECH-313 stochastic_noise_floor (tonic) on the
SAME E3 softmax-temperature channel. Reuses the MECH-104 volatility-surprise
lit basis (Aston-Jones & Cohen 2005 phasic mode) but routes to the E3
selection softmax, NOT the ARC-016 commit gate (the existing MECH-104 claim).

Contracts (C1-C8):
  C1: default-off no-op. use_phasic_burst=False -> agent.phasic_burst is None.
  C2: flag-on wires the regulator. use_phasic_burst=True -> agent.phasic_burst
      is a PhasicSurpriseBurst with the config threaded from REEConfig.
  C3: event dynamics. A surprise spike over the EMA baseline fires a burst
      (level 1.0 at a large excess) that decays geometrically over ticks; a
      quiescent stream fires nothing.
  C4: temperature transient + strict-positive floor. temperature_delta =
      temp_delta * burst_level; apply_to_temperature floors the combined
      temperature at min_temperature > 0.
  C5: MECH-094 simulation gate. simulation_mode=True returns the cached
      burst_level and does NOT advance the EMA baseline, envelope, or event
      counter; only the simulation-skip counter increments.
  C6: input validation. Out-of-range config raises at construction.
  C7: reset clears per-episode state.
  C8: agent-level bit-identity when NO event fires. With defaults on a
      settling (non-spiking) surprise stream, the ON action stream is
      identical to OFF (the phasic path adds nothing until an event fires),
      and n_events == 0 confirms the identity is earned, not incidental.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.regulators import PhasicSurpriseBurst, PhasicSurpriseBurstConfig
from ree_core.utils.config import REEConfig
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments._harness import StepHarness


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _mk_env(seed):
    return CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, seed=seed)


def _dims(env):
    return dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )


def _run(cfg, steps=12, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    agent = REEAgent(cfg)
    res = StepHarness(agent, _mk_env(seed), train_mode=True, seed=seed).run_episode(
        max_steps=steps
    )
    actions = [int(r.action.argmax().item()) for r in res]
    return agent, actions


# ----------------------------------------------------------------------
# C1 default-off no-op
# ----------------------------------------------------------------------
def test_c1_default_off_no_op():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(**_dims(env))
    assert cfg.use_phasic_burst is False
    agent, _ = _run(cfg, seed=7)
    assert agent.phasic_burst is None


# ----------------------------------------------------------------------
# C2 flag-on wiring
# ----------------------------------------------------------------------
def test_c2_agent_wires_phasic_burst_when_flag_on():
    env = _mk_env(0)
    cfg = REEConfig.from_dims(use_phasic_burst=True, **_dims(env))
    agent, _ = _run(cfg, seed=7)
    assert isinstance(agent.phasic_burst, PhasicSurpriseBurst)
    c = agent.phasic_burst.config
    assert c.trigger_ratio == 1.5
    assert c.temp_delta == -0.5
    assert c.decay == 0.5
    assert c.min_temperature == 0.1


# ----------------------------------------------------------------------
# C3 event dynamics
# ----------------------------------------------------------------------
def test_c3_spike_fires_burst_then_decays():
    pb = PhasicSurpriseBurst(
        PhasicSurpriseBurstConfig(
            surprise_ema_decay=0.1, trigger_ratio=1.5, temp_delta=-0.5, decay=0.5
        )
    )
    # Quiescent warmup -> no event.
    for _ in range(4):
        pb.tick(1.0)
    assert pb.get_state()["n_events"] == 0
    assert pb.burst_level == 0.0
    # Spike (5.0 vs baseline ~1.0, ratio 5 >= 1.5) -> full burst.
    lvl = pb.tick(5.0)
    assert lvl == pytest.approx(1.0, abs=1e-9)
    assert pb.get_state()["n_events"] == 1
    # Geometric decay (decay=0.5 -> halved each tick) with no further events.
    l1 = pb.tick(1.0)
    l2 = pb.tick(1.0)
    assert l1 == pytest.approx(0.5, abs=1e-9)
    assert l2 == pytest.approx(0.25, abs=1e-9)


def test_c3_quiescent_stream_never_fires():
    pb = PhasicSurpriseBurst(PhasicSurpriseBurstConfig(trigger_ratio=1.5))
    for _ in range(10):
        pb.tick(1.0)
    assert pb.get_state()["n_events"] == 0
    assert pb.temperature_delta == 0.0


# ----------------------------------------------------------------------
# C4 temperature transient + strict-positive floor
# ----------------------------------------------------------------------
def test_c4_temperature_delta_and_floor():
    pb = PhasicSurpriseBurst(
        PhasicSurpriseBurstConfig(
            trigger_ratio=1.5, temp_delta=-0.8, decay=0.5, min_temperature=0.1
        )
    )
    pb.tick(1.0)
    pb.tick(5.0)  # full burst
    assert pb.temperature_delta == pytest.approx(-0.8, abs=1e-9)
    # tonic 1.0 + (-0.8) = 0.2 (floor 0.1 does not bind).
    assert pb.apply_to_temperature(1.0) == pytest.approx(0.2, abs=1e-9)

    # Large negative delta must be floored strictly > 0.
    pb2 = PhasicSurpriseBurst(
        PhasicSurpriseBurstConfig(trigger_ratio=1.5, temp_delta=-5.0, min_temperature=0.1)
    )
    pb2.tick(1.0)
    pb2.tick(100.0)
    assert pb2.apply_to_temperature(1.0) == pytest.approx(0.1, abs=1e-9)
    assert pb2.apply_to_temperature(1.0) > 0.0


# ----------------------------------------------------------------------
# C5 MECH-094 simulation gate
# ----------------------------------------------------------------------
def test_c5_mech094_simulation_gate():
    pb = PhasicSurpriseBurst(PhasicSurpriseBurstConfig(trigger_ratio=1.5))
    pb.tick(1.0)  # initialize baseline
    state_before = pb.get_state()
    ema_before = state_before["surprise_ema"]
    n_events_before = state_before["n_events"]

    # A huge surprise under simulation must NOT fire or advance state.
    lvl = pb.tick(100.0, simulation_mode=True)
    state_after = pb.get_state()
    assert lvl == pb.burst_level  # cached value returned
    assert state_after["n_events"] == n_events_before
    assert state_after["surprise_ema"] == pytest.approx(ema_before, abs=1e-12)
    assert state_after["n_simulation_skips"] == 1
    assert state_after["n_waking_ticks"] == state_before["n_waking_ticks"]


# ----------------------------------------------------------------------
# C6 input validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"surprise_ema_decay": 0.0},
        {"surprise_ema_decay": 1.5},
        {"trigger_ratio": 0.9},
        {"trigger_floor": 0.0},
        {"decay": 0.0},
        {"decay": 1.5},
        {"min_temperature": 0.0},
        {"min_temperature": -1.0},
        {"excess_saturation": 0.0},
    ],
)
def test_c6_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        PhasicSurpriseBurst(PhasicSurpriseBurstConfig(**kwargs))


# ----------------------------------------------------------------------
# C7 reset
# ----------------------------------------------------------------------
def test_c7_reset_clears_state():
    pb = PhasicSurpriseBurst(PhasicSurpriseBurstConfig(trigger_ratio=1.5))
    pb.tick(1.0)
    pb.tick(5.0)
    assert pb.get_state()["n_events"] == 1
    assert pb.burst_level > 0.0
    pb.reset()
    st = pb.get_state()
    assert st["n_events"] == 0
    assert st["n_waking_ticks"] == 0
    assert st["burst_level"] == 0.0
    assert st["surprise_ema"] == 0.0


# ----------------------------------------------------------------------
# C8 agent-level bit-identity when no event fires
# ----------------------------------------------------------------------
def test_c8_bit_identical_off_vs_on_when_no_event():
    env = _mk_env(0)
    base = _dims(env)
    cfg_off = REEConfig.from_dims(**base)
    cfg_on = REEConfig.from_dims(use_phasic_burst=True, **base)
    ag_off, act_off = _run(cfg_off, seed=13)
    ag_on, act_on = _run(cfg_on, seed=13)
    # Defaults on a settling surprise stream fire no event, so the phasic
    # path adds nothing and the action stream is bit-identical to OFF.
    assert ag_on.phasic_burst.get_state()["n_events"] == 0
    assert act_off == act_on, (
        f"phasic-ON with no event must be bit-identical to OFF "
        f"(off={act_off} on={act_on})"
    )
