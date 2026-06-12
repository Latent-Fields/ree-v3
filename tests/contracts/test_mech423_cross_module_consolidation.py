"""
Contract tests for MECH-423 R3: module-tagged interleaved cross-module
consolidation.

The legacy MECH-121 offline pass trains e2_harm_s alone over region-keyed traces
with no module identity. The EXP-0380 (MECH-423) R3 readiness check needs a
cross_module_replay_share readout and an interleaved E1<->E2 consolidation
schedule. This suite pins:

  C1 default-OFF: agent.cross_module_consolidator is None.
  C2 config validation (schedule / n_steps / lr).
  C3 interleaved schedule -> share == 1.0 when both modules have replay content.
  C4 blocked schedule -> share == 0.0 (the catastrophic-interference control).
  C5 only-one-module-has-data -> interleaved share == 0.0 (honest: no integration).
  C6 simulation_mode -> all-zero no-op.
  C7 SleepLoopManager hook merges the readout into the live MECH-121 sleep cycle.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.agent import REEAgent
from ree_core.sleep.cross_module_consolidation import (
    CrossModuleConsolidator,
    CrossModuleConsolidatorConfig,
)
from ree_core.utils.config import REEConfig


def _build(seed: int = 7, **flags):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return agent, b, w


def _fill_buffers(agent, b, w, n: int = 14):
    sd = agent.config.latent.self_dim
    for _ in range(n):
        with torch.no_grad():
            a = agent.act_with_split_obs(b, w)
        act = a if a.dim() == 2 else a.unsqueeze(0)
        agent.record_transition(
            torch.randn(1, sd), act.float(), torch.randn(1, sd)
        )


def _module_maps(agent):
    losses = {
        "e1": lambda: agent.compute_prediction_loss(),
        "e2": lambda: agent.compute_e2_loss(batch_size=8),
    }
    params = {
        "e1": list(agent.e1.parameters()),
        "e2": list(agent.e2.parameters()),
    }
    return losses, params


# ----------------------------------------------------------------------
# C1 default-OFF
# ----------------------------------------------------------------------
def test_c1_default_off_no_consolidator():
    agent, _b, _w = _build()
    assert agent.cross_module_consolidator is None


# ----------------------------------------------------------------------
# C2 config validation
# ----------------------------------------------------------------------
def test_c2_config_validation():
    with pytest.raises(ValueError):
        CrossModuleConsolidatorConfig(schedule="nonsense")
    with pytest.raises(ValueError):
        CrossModuleConsolidatorConfig(n_steps=-1)
    with pytest.raises(ValueError):
        CrossModuleConsolidatorConfig(lr=0.0)
    # consolidate() also rejects a bad schedule override
    c = CrossModuleConsolidator()
    with pytest.raises(ValueError):
        c.consolidate({}, {}, n_steps=1, schedule="bogus")


# ----------------------------------------------------------------------
# C3 interleaved -> share 1.0 (both modules have data)
# ----------------------------------------------------------------------
def test_c3_interleaved_share_one():
    agent, b, w = _build(
        use_cross_module_consolidation=True, cross_module_consolidation_steps=4
    )
    _fill_buffers(agent, b, w, 14)
    losses, params = _module_maps(agent)
    m = agent.cross_module_consolidator.consolidate(
        module_losses=losses, module_params=params, n_steps=4, schedule="interleaved"
    )
    assert m["interleaved"] == 1.0
    assert m["cross_module_replay_share"] == 1.0
    assert m["n_traces"] == 4.0
    assert m["n_cross_module_traces"] == 4.0
    assert m["n_updates"] == 8.0
    assert m["updates_e1"] == 4.0 and m["updates_e2"] == 4.0


# ----------------------------------------------------------------------
# C4 blocked -> share 0.0
# ----------------------------------------------------------------------
def test_c4_blocked_share_zero():
    agent, b, w = _build(
        use_cross_module_consolidation=True, cross_module_consolidation_steps=4
    )
    _fill_buffers(agent, b, w, 14)
    losses, params = _module_maps(agent)
    m = agent.cross_module_consolidator.consolidate(
        module_losses=losses, module_params=params, n_steps=4, schedule="blocked"
    )
    assert m["interleaved"] == 0.0
    assert m["cross_module_replay_share"] == 0.0
    assert m["n_cross_module_traces"] == 0.0
    # both modules still trained (sequentially)
    assert m["updates_e1"] == 4.0 and m["updates_e2"] == 4.0


# ----------------------------------------------------------------------
# C5 only-one-module-has-data -> interleaved share 0.0 (honest readout)
# ----------------------------------------------------------------------
def test_c5_single_module_data_share_zero():
    agent, b, w = _build(
        use_cross_module_consolidation=True, cross_module_consolidation_steps=3
    )
    _fill_buffers(agent, b, w, 14)
    # e2 reports an exactly-zero sentinel -> not a genuine update -> not touched
    e2_zero = next(agent.e2.parameters()).sum() * 0.0
    losses = {
        "e1": lambda: agent.compute_prediction_loss(),
        "e2": lambda: next(agent.e2.parameters()).sum() * 0.0,
    }
    params = {
        "e1": list(agent.e1.parameters()),
        "e2": list(agent.e2.parameters()),
    }
    m = agent.cross_module_consolidator.consolidate(
        module_losses=losses, module_params=params, n_steps=3, schedule="interleaved"
    )
    # interleaved schedule, but only e1 ever touches a trace -> no cross-module
    assert m["cross_module_replay_share"] == 0.0
    assert m["updates_e1"] == 3.0
    assert m["updates_e2"] == 0.0
    assert float(e2_zero.item()) == 0.0  # sentinel sanity


# ----------------------------------------------------------------------
# C6 simulation_mode no-op (MECH-094)
# ----------------------------------------------------------------------
def test_c6_simulation_mode_no_op():
    agent, b, w = _build(
        use_cross_module_consolidation=True, cross_module_consolidation_steps=4
    )
    _fill_buffers(agent, b, w, 14)
    losses, params = _module_maps(agent)
    m = agent.cross_module_consolidator.consolidate(
        module_losses=losses,
        module_params=params,
        n_steps=4,
        schedule="interleaved",
        simulation_mode=True,
    )
    assert m["n_updates"] == 0.0
    assert m["n_traces"] == 0.0
    assert m["cross_module_replay_share"] == 0.0


# ----------------------------------------------------------------------
# C7 SleepLoopManager hook merges the readout into the live sleep cycle
# ----------------------------------------------------------------------
def test_c7_sleep_loop_hook_merges_readout():
    agent, b, w = _build(
        use_sleep_loop=True,
        sws_enabled=True,
        rem_enabled=True,
        use_cross_module_consolidation=True,
        cross_module_consolidation_steps=3,
        cross_module_consolidation_schedule="interleaved",
    )
    assert agent.sleep_loop is not None
    assert agent.sleep_loop.cross_module_consolidator is agent.cross_module_consolidator
    _fill_buffers(agent, b, w, 14)
    m = agent.sleep_loop.force_cycle(agent)
    assert "cross_module_consolidation_cross_module_replay_share" in m
    assert m["cross_module_consolidation_interleaved"] == 1.0
    assert m["cross_module_consolidation_cross_module_replay_share"] > 0.0
    assert m["cross_module_consolidation_n_updates"] > 0.0
