"""Contract tests for the MECH-090 R-c continuation (nav_competence axis)
env-emitted readiness-outcome wiring -- the Phase-2 follow-on (2026-06-02).

The 2026-05-29 landing wired the CONSUMER (CommitReadiness.is_above_floor
AND-composed at both beta_gate.elevate() sites) plus the notify_outcome seam,
but left no automatic SOURCE: in any ecological run nothing advanced the
readiness EMA, so the across-tick nav_competence axis sat fail-open (readiness
pinned at the initial 1.0). This pass closes the gap with:
  (a) CausalGridWorldV2.mech090_readiness_outcome_enabled (env-only kwarg,
      default False) emitting info["mech090_readiness_outcome"] = clip(1 -
      mean(limb_damage), 0, 1) -- a [0,1] motor-program-readiness scalar.
  (b) REEAgent.sense(mech090_readiness_outcome=...) advancing the
      CommitReadiness EMA from that per-tick signal.

Six contracts (C1-C6):
  C1: env default OFF -> info has NO "mech090_readiness_outcome" key
      (absent-when-disabled; agent reads None -> EMA un-advanced -> bit-identical).
  C2: env ON -> info["mech090_readiness_outcome"] present, in [0,1], EXACTLY
      1 - mean(limb_damage); monotone in accumulated damage (degrade) and
      recovers toward 1.0 as damage heals.
  C3: agent.sense(mech090_readiness_outcome=None) with the conjunction ON
      leaves the readiness EMA unchanged (the None-sentinel no-op path).
  C4: agent.sense() with sustained LOW outcomes drives the readiness EMA below
      the floor (is_above_floor flips False); sustained HIGH outcomes keep it
      above. This is the load-bearing across-tick degrade/recover dynamic.
  C5: master OFF (commit_readiness is None) -> sense(mech090_readiness_outcome=
      0.0) is a no-op (no crash, no state); bit-identical to omitting the kwarg.
  C6: MECH-094 -- the agent forwards simulation_mode=hypothesis_tag into
      update(); a simulation/replay latent does not advance the EMA, a waking
      latent does. Verified at the wiring layer (waking sense advances) plus the
      module-level simulation_mode no-op the wiring relies on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_agent(**flags) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        harm_obs_dim=50,
        harm_obs_a_dim=50,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        **flags,
    )
    return REEAgent(cfg)


def _obs(agent: REEAgent):
    """Zero body/world obs at the configured input dims."""
    body = torch.zeros(1, 12, device=agent.device)
    world = torch.zeros(1, 250, device=agent.device)
    return body, world


# ----------------------------------------------------------------------
# C1: env OFF -> key absent (bit-identical)
# ----------------------------------------------------------------------
def test_c1_env_off_no_outcome_key():
    env = CausalGridWorldV2(seed=0, limb_damage_enabled=True)
    assert env.mech090_readiness_outcome_enabled is False
    env.reset()
    _, _, _, info, _ = env.step(4)
    assert "mech090_readiness_outcome" not in info, (
        "with mech090_readiness_outcome_enabled=False the env must NOT emit "
        "the key (absent -> agent reads None -> EMA un-advanced -> bit-identical)"
    )


# ----------------------------------------------------------------------
# C2: env ON -> key present, == 1 - mean(limb_damage), monotone
# ----------------------------------------------------------------------
def test_c2_env_on_emits_exact_outcome():
    env = CausalGridWorldV2(
        seed=0, limb_damage_enabled=True, mech090_readiness_outcome_enabled=True
    )
    env.reset()
    # zero damage -> outcome 1.0
    env.limb_damage[:] = 0.0
    _, _, _, info, _ = env.step(4)
    assert "mech090_readiness_outcome" in info
    assert info["mech090_readiness_outcome"] == pytest.approx(1.0, abs=1e-6)

    # high damage -> low outcome, exactly 1 - mean(limb_damage) post-step
    env.limb_damage[:] = 0.8
    _, _, _, info, _ = env.step(4)
    expected = 1.0 - float(np.mean(env.limb_damage))
    assert info["mech090_readiness_outcome"] == pytest.approx(expected, abs=1e-6)
    assert 0.0 <= info["mech090_readiness_outcome"] <= 1.0


def test_c2_outcome_monotone_in_damage():
    env = CausalGridWorldV2(
        seed=0, limb_damage_enabled=True, mech090_readiness_outcome_enabled=True
    )
    env.reset()
    outcomes = []
    for dmg in (0.0, 0.25, 0.5, 0.9):
        env.limb_damage[:] = dmg
        _, _, _, info, _ = env.step(4)
        outcomes.append(info["mech090_readiness_outcome"])
    # readiness outcome strictly decreases as damage increases
    assert outcomes[0] > outcomes[1] > outcomes[2] > outcomes[3]


def test_c2_outcome_recovers_on_heal():
    env = CausalGridWorldV2(
        seed=0,
        limb_damage_enabled=True,
        mech090_readiness_outcome_enabled=True,
        heal_rate=0.05,
    )
    env.reset()
    env.limb_damage[:] = 0.8
    # stay action heals each step -> outcome climbs toward 1.0
    first = None
    last = None
    for _ in range(20):
        _, _, _, info, _ = env.step(4)
        if first is None:
            first = info["mech090_readiness_outcome"]
        last = info["mech090_readiness_outcome"]
    assert last > first, "healing should raise the readiness outcome over ticks"


def test_c2_limb_damage_off_outcome_constant_one():
    # fail-open: no damage substrate -> outcome is a constant 1.0
    env = CausalGridWorldV2(
        seed=0, limb_damage_enabled=False, mech090_readiness_outcome_enabled=True
    )
    env.reset()
    _, _, _, info, _ = env.step(4)
    assert info["mech090_readiness_outcome"] == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------
# C3: sense(None) leaves EMA unchanged
# ----------------------------------------------------------------------
def test_c3_sense_none_is_noop():
    agent = _build_agent(use_mech090_readiness_conjunction=True)
    assert agent.commit_readiness is not None
    before = agent.commit_readiness.get_readiness()
    body, world = _obs(agent)
    agent.sense(body, world, mech090_readiness_outcome=None)
    after = agent.commit_readiness.get_readiness()
    assert after == pytest.approx(before), (
        "None outcome (key absent) must not advance the readiness EMA"
    )
    # n_updates should not have advanced for a None signal
    assert agent.commit_readiness.get_state()["n_updates"] == 0


# ----------------------------------------------------------------------
# C4: sustained low outcomes drive readiness below the floor; high keep above
# ----------------------------------------------------------------------
def test_c4_low_outcomes_drive_below_floor():
    agent = _build_agent(
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
        commit_readiness_ema_alpha=0.3,  # faster convergence for the test
    )
    body, world = _obs(agent)
    floor = float(agent.config.mech090_readiness_floor)
    assert agent.commit_readiness.is_above_floor(floor)  # fail-open initial 1.0
    for _ in range(40):
        agent.sense(body, world, mech090_readiness_outcome=0.0)
    assert not agent.commit_readiness.is_above_floor(floor), (
        "sustained zero readiness outcomes must drive the EMA below the floor"
    )


def test_c4_high_outcomes_stay_above_floor():
    agent = _build_agent(
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
        commit_readiness_ema_alpha=0.3,
    )
    body, world = _obs(agent)
    floor = float(agent.config.mech090_readiness_floor)
    for _ in range(40):
        agent.sense(body, world, mech090_readiness_outcome=1.0)
    assert agent.commit_readiness.is_above_floor(floor)


def test_c4_recover_after_degrade():
    agent = _build_agent(
        use_mech090_readiness_conjunction=True,
        mech090_readiness_floor=0.3,
        commit_readiness_ema_alpha=0.3,
    )
    body, world = _obs(agent)
    floor = float(agent.config.mech090_readiness_floor)
    for _ in range(40):
        agent.sense(body, world, mech090_readiness_outcome=0.0)
    assert not agent.commit_readiness.is_above_floor(floor)
    for _ in range(40):
        agent.sense(body, world, mech090_readiness_outcome=1.0)
    assert agent.commit_readiness.is_above_floor(floor), (
        "readiness must recover above the floor once outcomes recover"
    )


# ----------------------------------------------------------------------
# C5: master OFF -> sense(outcome) is a no-op (bit-identical)
# ----------------------------------------------------------------------
def test_c5_master_off_outcome_is_noop():
    agent = _build_agent()  # conjunction off -> commit_readiness is None
    assert agent.commit_readiness is None
    body, world = _obs(agent)
    # passing an outcome must not crash and must not create state
    latent_with = agent.sense(body, world, mech090_readiness_outcome=0.0)
    assert latent_with is not None
    assert agent.commit_readiness is None


def test_c5_kwarg_omitted_vs_none_identical():
    # bit-identical: omitting the kwarg == passing None (master OFF).
    # Seed identically so the only variable is the kwarg (separate agents
    # otherwise carry independent random encoder init).
    torch.manual_seed(123)
    a1 = _build_agent()
    torch.manual_seed(123)
    a2 = _build_agent()
    b1, w1 = _obs(a1)
    b2, w2 = _obs(a2)
    l1 = a1.sense(b1, w1)
    l2 = a2.sense(b2, w2, mech090_readiness_outcome=None)
    assert torch.allclose(l1.z_world, l2.z_world)


# ----------------------------------------------------------------------
# C6: MECH-094 -- simulation latents do not advance the EMA
# ----------------------------------------------------------------------
def test_c6_waking_sense_advances_ema():
    agent = _build_agent(use_mech090_readiness_conjunction=True)
    body, world = _obs(agent)
    agent.sense(body, world, mech090_readiness_outcome=0.0)
    # a waking sense() (hypothesis_tag False) advances the EMA -> n_updates 1
    assert agent.commit_readiness.get_state()["n_updates"] == 1


def test_c6_module_simulation_mode_noop_underpins_wiring():
    # The wiring passes simulation_mode=hypothesis_tag into update(); the
    # module's simulation_mode path is the MECH-094 guard the wiring relies on.
    agent = _build_agent(use_mech090_readiness_conjunction=True)
    cr = agent.commit_readiness
    before = cr.get_readiness()
    cr.update(outcome_signal=0.0, simulation_mode=True)
    assert cr.get_readiness() == pytest.approx(before)
    assert cr.get_state()["n_simulation_skips"] == 1
    assert cr.get_state()["n_updates"] == 0
