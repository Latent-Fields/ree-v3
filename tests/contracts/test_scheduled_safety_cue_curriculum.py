"""
Contract tests for SD-065 conditioned-safety cue channel (MECH-304
promote-to-active behavioural falsifier substrate, 2026-07-14).

The channel is the observable Pavlovian CS the SD-051 ConditionedSafetyStore
needs: an AMBIENT (uniform-when-active) 25-dim field view emitted into
world_state (feeds z_world) that can be paired with the MECH-302 relief window
during teaching and presented concurrently with a threat at test.

Seven contracts:
  C1  Off by default -- bit-identical to pre-extension behaviour; info-dict
      sentinel keys always present but inert when master is OFF; no obs channel;
      world_obs_dim unchanged.
  C2  Preconditions: safety_cue_enabled requires use_proxy_fields;
      safety_cue_on_relief requires limb_damage_enabled. Both raise ValueError.
  C3  Channel geometry: enabling grows world_obs_dim by exactly 25; world_state's
      trailing 25 dims carry the cue; obs_dict["safety_cue_field_view"] present
      when enabled, absent when disabled.
  C4  Manual override (test API): set_safety_cue(True) forces a uniform-scale
      view; (False) forces zeros even inside a relief window; (None) follows the
      schedule. Scale is clipped to [0, 1].
  C5  Relief pairing: with limb_damage + scheduled injection + on_relief, the cue
      is active EXACTLY on ticks where sum(limb_damage) > safety_cue_heal_floor.
  C6  Per-episode reset: reset() clears _safety_cue_active + event_count; the
      manual override PERSISTS across reset (deliberate experimenter control).
  C7  RNG isolation: enabling the master switch (never activating the cue) draws
      no env RNG -- bit-identical RNG-driven trajectory vs a control that differs
      only by the cue flag.
"""

import numpy as np
import pytest
import torch

from ree_core.environment.causal_grid_world import CausalGridWorld, CausalGridWorldV2


def _act(a: int) -> torch.Tensor:
    return torch.tensor(a, dtype=torch.long)


def _step(env, action_int: int):
    flat, harm, done, info, obs = env.step(_act(action_int))
    return flat, harm, done, info, obs


# ---------------------------------------------------------------------------
# C1  Off by default; sentinel keys present but inert; no channel; dim unchanged
# ---------------------------------------------------------------------------

def test_c1_off_by_default_inert():
    env = CausalGridWorldV2(size=12, num_hazards=3, num_resources=2, seed=42)
    assert env.safety_cue_enabled is False
    assert env.safety_cue_on_relief is False
    assert env._safety_cue_active is False
    assert env._safety_cue_event_count == 0
    assert env._safety_cue_manual_override is None

    baseline_dim = CausalGridWorldV2(
        size=12, num_hazards=3, num_resources=2, seed=42
    ).world_obs_dim
    assert env.world_obs_dim == baseline_dim  # no dim change when OFF

    _flat, _h, _d, info, obs = _step(env, 0)
    # Info sentinels always present but inert.
    assert info["safety_cue_enabled"] is False
    assert info["safety_cue_active"] is False
    assert info["safety_cue_event_count"] == 0
    # No obs channel leaked.
    assert "safety_cue_field_view" not in obs


# ---------------------------------------------------------------------------
# C2  Preconditions raise ValueError
# ---------------------------------------------------------------------------

def test_c2_precondition_cue_requires_proxy():
    with pytest.raises(ValueError):
        CausalGridWorld(
            size=12, seed=1, use_proxy_fields=False, safety_cue_enabled=True
        )


def test_c2_precondition_on_relief_requires_limb_damage():
    with pytest.raises(ValueError):
        # V2 forces proxy fields; limb_damage stays off -> on_relief must raise.
        CausalGridWorldV2(size=12, seed=1, safety_cue_on_relief=True)


def test_c2_enabled_alone_ok_without_limb_damage():
    # Manual-only use (no relief pairing) does not require limb_damage.
    env = CausalGridWorldV2(size=12, seed=1, safety_cue_enabled=True)
    assert env.safety_cue_enabled is True
    assert env.safety_cue_on_relief is False


# ---------------------------------------------------------------------------
# C3  Channel geometry: +25 dims, trailing block, obs key presence
# ---------------------------------------------------------------------------

def test_c3_channel_geometry():
    off = CausalGridWorldV2(size=12, num_hazards=3, num_resources=2, seed=5)
    on = CausalGridWorldV2(
        size=12, num_hazards=3, num_resources=2, seed=5, safety_cue_enabled=True
    )
    assert on.world_obs_dim == off.world_obs_dim + 25

    on.set_safety_cue(True)
    _f, _h, _d, _info, obs = _step(on, 0)
    assert "safety_cue_field_view" in obs
    scv = obs["safety_cue_field_view"]
    assert scv.shape[0] == 25
    # world_state trailing 25 dims == the cue view (appended last).
    assert torch.allclose(obs["world_state"][-25:], scv)
    assert torch.allclose(scv, torch.ones(25))


# ---------------------------------------------------------------------------
# C4  Manual override semantics + scale clip
# ---------------------------------------------------------------------------

def test_c4_manual_override_semantics():
    env = CausalGridWorldV2(
        size=12, seed=9, safety_cue_enabled=True, safety_cue_scale=0.7
    )
    env.set_safety_cue(True)
    _f, _h, _d, info, obs = _step(env, 0)
    assert info["safety_cue_active"] is True
    assert torch.allclose(obs["safety_cue_field_view"], torch.full((25,), 0.7))

    env.set_safety_cue(False)
    _f, _h, _d, info, obs = _step(env, 0)
    assert info["safety_cue_active"] is False
    assert torch.allclose(obs["safety_cue_field_view"], torch.zeros(25))

    env.set_safety_cue(None)  # follow schedule; no relief pairing -> inactive
    _f, _h, _d, info, _obs = _step(env, 0)
    assert info["safety_cue_active"] is False


def test_c4_scale_clip():
    hi = CausalGridWorldV2(size=12, seed=9, safety_cue_enabled=True, safety_cue_scale=5.0)
    assert hi.safety_cue_scale == 1.0
    lo = CausalGridWorldV2(size=12, seed=9, safety_cue_enabled=True, safety_cue_scale=-2.0)
    assert lo.safety_cue_scale == 0.0


# ---------------------------------------------------------------------------
# C5  Relief pairing tracks the damage window exactly
# ---------------------------------------------------------------------------

def test_c5_relief_pairing_tracks_damage_window():
    env = CausalGridWorldV2(
        size=14, num_hazards=1, num_resources=1, seed=3,
        limb_damage_enabled=True, heal_rate=0.05,
        scheduled_limb_damage_enabled=True, scheduled_limb_damage_interval=5,
        scheduled_limb_damage_prob=1.0, scheduled_limb_damage_magnitude=0.6,
        safety_cue_enabled=True, safety_cue_on_relief=True,
        safety_cue_heal_floor=0.05,
    )
    env.reset()
    n_active = 0
    for _t in range(30):
        _f, _h, done, info, _obs = _step(env, 4)
        dmg = float(env.limb_damage.sum())
        # Cue active <=> body-damage above the relief-window floor.
        assert bool(info["safety_cue_active"]) == (dmg > env.safety_cue_heal_floor)
        n_active += int(info["safety_cue_active"])
        if done:
            break
    assert n_active > 0  # the pairing fired at least once


# ---------------------------------------------------------------------------
# C6  Per-episode reset clears dynamics; manual override persists
# ---------------------------------------------------------------------------

def test_c6_reset_clears_dynamics_override_persists():
    env = CausalGridWorldV2(size=12, seed=11, safety_cue_enabled=True)
    env.set_safety_cue(True)
    _step(env, 0)
    assert env._safety_cue_active is True
    assert env._safety_cue_event_count >= 1

    env.reset()
    # Dynamics cleared...
    assert env._safety_cue_active is False
    assert env._safety_cue_event_count == 0
    # ...but the deliberate experimenter override persists across reset.
    assert env._safety_cue_manual_override is True
    _f, _h, _d, info, _obs = _step(env, 0)
    assert info["safety_cue_active"] is True


# ---------------------------------------------------------------------------
# C7  RNG isolation: master ON (never activated) draws no env RNG
# ---------------------------------------------------------------------------

def test_c7_rng_isolation():
    # on_relief=False and no manual override => the cue never activates and its
    # code path draws no RNG. RNG-driven state (hazard positions after drift,
    # agent health) must match a control identical but for the cue flag.
    ctrl = CausalGridWorldV2(size=12, num_hazards=3, num_resources=2, seed=77)
    cue = CausalGridWorldV2(
        size=12, num_hazards=3, num_resources=2, seed=77, safety_cue_enabled=True
    )
    ctrl.reset()
    cue.reset()
    for _t in range(25):
        _f0, h0, d0, info0, _o0 = _step(ctrl, 2)
        _f1, h1, d1, info1, _o1 = _step(cue, 2)
        assert float(h0) == float(h1)
        assert bool(d0) == bool(d1)
        assert info0["health"] == info1["health"]
        assert info1["safety_cue_active"] is False  # never activated
        if d0:
            break
