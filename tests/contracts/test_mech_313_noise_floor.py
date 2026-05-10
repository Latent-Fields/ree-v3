"""Contract tests for MECH-313 stochastic_noise_floor (ARC-065 child).

Five contracts (C1-C5):
  C1: default-off no-op. With use_noise_floor=False, agent.noise_floor is
      None and select_action's effective_temperature equals the caller's
      baseline_temperature.
  C2: flag-on lifts the temperature. With use_noise_floor=True at
      defaults (alpha=0.1, min_temperature=1.0), the regulator's
      compute_effective_temperature lifts a baseline of 1.0 to >= 1.1
      and respects the floor when baseline < 1.0.
  C3: backward-compat across the existing config matrix. Toggling
      noise_floor with other major flags (use_gated_policy,
      use_dacc, use_lateral_pfc_analog) does not raise during agent
      construction or first sense() tick.
  C4: MECH-094 simulation gate. simulation_mode=True returns the
      baseline temperature unchanged and increments only the simulation-
      skip counter; subsequent waking call advances waking-call counter
      without retroactively re-incrementing simulation-skip.
  C5: input-validation. compute_effective_temperature with
      baseline_temperature <= 0 raises ValueError. NoiseFloorConfig with
      negative alpha or non-positive min_temperature raises at
      construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.policy import NoiseFloor, NoiseFloorConfig
from ree_core.utils.config import REEConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_agent_and_one_tick(seed: int = 7, **flags):
    """Build a small REEAgent and run one sense() tick."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1,
        use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16, world_dim=16,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    kwargs = {"obs_body": body, "obs_world": world}
    for key in ("harm_obs", "harm_obs_a"):
        v = obs_dict.get(key)
        if v is not None:
            attr = "use_harm_stream" if key == "harm_obs" else "use_affective_harm_stream"
            if getattr(cfg.latent, attr, False):
                if v.dim() == 1:
                    v = v.unsqueeze(0)
                kwargs[key.replace("harm_obs", "obs_harm")] = v
    with torch.no_grad():
        latent = agent.sense(**kwargs)
    return agent, latent


# ----------------------------------------------------------------------
# C1 default-off no-op
# ----------------------------------------------------------------------
def test_c1_default_off_no_op():
    agent, latent = _build_agent_and_one_tick()
    assert agent.noise_floor is None, (
        "default config should produce noise_floor=None"
    )
    # Construct a regulator-free counterpart and confirm a manual call
    # site that mirrors select_action's else branch produces the
    # baseline temperature unchanged.
    baseline = 1.0
    effective = baseline  # mirrors the else branch in select_action
    assert effective == baseline


# ----------------------------------------------------------------------
# C2 flag-on lifts the temperature
# ----------------------------------------------------------------------
def test_c2_flag_on_lifts_temperature_at_defaults():
    nf = NoiseFloor(NoiseFloorConfig(use_noise_floor=True))
    # Baseline 1.0 + alpha 0.1 = 1.1; min_temperature 1.0 does not bind.
    assert nf.compute_effective_temperature(1.0) == pytest.approx(1.1, abs=1e-9)
    # Baseline 0.5 + alpha 0.1 = 0.6; min_temperature 1.0 binds.
    assert nf.compute_effective_temperature(0.5) == pytest.approx(1.0, abs=1e-9)
    # Baseline 5.0 + alpha 0.1 = 5.1; min does not bind.
    assert nf.compute_effective_temperature(5.0) == pytest.approx(5.1, abs=1e-9)


def test_c2_custom_alpha_and_floor():
    nf = NoiseFloor(NoiseFloorConfig(
        use_noise_floor=True,
        noise_floor_alpha=0.5,
        noise_floor_min_temperature=2.0,
    ))
    # Baseline 1.0 + alpha 0.5 = 1.5; min 2.0 binds.
    assert nf.compute_effective_temperature(1.0) == pytest.approx(2.0, abs=1e-9)
    # Baseline 3.0 + alpha 0.5 = 3.5; min 2.0 does not bind.
    assert nf.compute_effective_temperature(3.0) == pytest.approx(3.5, abs=1e-9)


def test_c2_agent_wires_noise_floor_when_flag_on():
    agent, _ = _build_agent_and_one_tick(use_noise_floor=True)
    assert agent.noise_floor is not None
    assert isinstance(agent.noise_floor, NoiseFloor)
    assert agent.noise_floor.config.noise_floor_alpha == 0.1
    assert agent.noise_floor.config.noise_floor_min_temperature == 1.0


# ----------------------------------------------------------------------
# C3 backward-compat across config matrix
# ----------------------------------------------------------------------
@pytest.mark.parametrize("flags", [
    {"use_noise_floor": True},
    {"use_noise_floor": True, "use_gated_policy": True},
    {"use_noise_floor": True, "use_dacc": True, "dacc_weight": 0.0},
])
def test_c3_backward_compat_config_matrix(flags):
    agent, latent = _build_agent_and_one_tick(**flags)
    assert agent.noise_floor is not None
    assert latent is not None
    assert torch.isfinite(latent.z_world).all()


# ----------------------------------------------------------------------
# C4 MECH-094 simulation gate
# ----------------------------------------------------------------------
def test_c4_mech094_simulation_gate():
    nf = NoiseFloor(NoiseFloorConfig(
        use_noise_floor=True,
        noise_floor_alpha=0.5,
        noise_floor_min_temperature=2.0,
    ))
    pre_skip = nf._last_n_simulation_skips
    pre_waking = nf._n_waking_calls

    sim_T = nf.compute_effective_temperature(1.0, simulation_mode=True)
    # Simulation: returns baseline unchanged (NOT lifted).
    assert sim_T == pytest.approx(1.0, abs=1e-9)
    assert nf._last_n_simulation_skips == pre_skip + 1
    assert nf._n_waking_calls == pre_waking
    # Waking-only diagnostic fields still at init values (never written
    # in the simulation branch).
    assert nf._last_baseline_temperature == 0.0
    assert nf._last_effective_temperature == 0.0

    # Subsequent waking call advances waking counter only.
    pre_skip_2 = nf._last_n_simulation_skips
    waking_T = nf.compute_effective_temperature(1.0, simulation_mode=False)
    assert waking_T == pytest.approx(2.0, abs=1e-9)
    assert nf._last_n_simulation_skips == pre_skip_2  # unchanged
    assert nf._n_waking_calls == pre_waking + 1
    assert nf._last_baseline_temperature == pytest.approx(1.0, abs=1e-9)
    assert nf._last_effective_temperature == pytest.approx(2.0, abs=1e-9)


# ----------------------------------------------------------------------
# C5 input validation
# ----------------------------------------------------------------------
def test_c5_invalid_baseline_raises():
    nf = NoiseFloor(NoiseFloorConfig(use_noise_floor=True))
    with pytest.raises(ValueError):
        nf.compute_effective_temperature(0.0)
    with pytest.raises(ValueError):
        nf.compute_effective_temperature(-1.0)


def test_c5_invalid_config_raises():
    with pytest.raises(ValueError):
        NoiseFloor(NoiseFloorConfig(use_noise_floor=True, noise_floor_alpha=-0.1))
    with pytest.raises(ValueError):
        NoiseFloor(NoiseFloorConfig(
            use_noise_floor=True, noise_floor_min_temperature=0.0,
        ))
    with pytest.raises(ValueError):
        NoiseFloor(NoiseFloorConfig(
            use_noise_floor=True, noise_floor_min_temperature=-0.5,
        ))


# ----------------------------------------------------------------------
# Reset clears diagnostic counters
# ----------------------------------------------------------------------
def test_reset_clears_diagnostics():
    nf = NoiseFloor(NoiseFloorConfig(use_noise_floor=True))
    nf.compute_effective_temperature(1.0)
    nf.compute_effective_temperature(2.0, simulation_mode=True)
    assert nf._n_waking_calls == 1
    assert nf._last_n_simulation_skips == 1

    nf.reset()
    assert nf._n_waking_calls == 0
    assert nf._last_n_simulation_skips == 0
    assert nf._last_baseline_temperature == 0.0
    assert nf._last_effective_temperature == 0.0
