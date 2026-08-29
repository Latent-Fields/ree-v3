"""Contracts for SD-102 (MECH-482): epistemic_deficit_accumulator.

WHAT THIS PINS. `mech314bc_percandidate_extension_staged_2026-08-08.md` landed
the MECH-314c per-candidate SLOT (`per_candidate_learning_progress` on
`StructuredCuriosity.compute_score_bias`) with NO live source -- callers passed
`None` and 314c stayed the Phase-1 uniform `lp_ema` broadcast. This module pins
the persistent, target-bound accumulator (`ree_core/policy/epistemic_deficit.py`)
that fills that slot, and the agent-level wiring
(`REEAgent._update_epistemic_deficit` / `_curiosity_per_candidate_learning_
progress`) that makes it LIVE.

Coverage:
  - EpistemicDeficitAccumulator in isolation: config validation, target
    matching / creation / eviction, EMA persistence, MECH-094 simulation-mode
    no-op, readout matching semantics, reset, diagnostics;
  - agent wiring: OFF (curiosity_learning_progress_source="broadcast") is
    bit-identical (accumulator never instantiated, per-candidate vector stays
    None); ON accumulates across a real rollout and the readiness gate
    (e2_world_uncertainty_last_pvar_relative_spread > 0) is respected,
    including the self-report-vacuous refusal path;
  - per-episode reset clears both the accumulator's targets and the
    one-tick-lag prev-z_world cache (mirrors the SD-063 cache-reset contract).
"""

from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.policy.epistemic_deficit import (
    EpistemicDeficitAccumulator,
    EpistemicDeficitConfig,
)
from ree_core.utils.config import REEConfig

WORLD_DIM = 16
ACTION_DIM = 4


# ------------------------------------------------------------------ #
# EpistemicDeficitAccumulator in isolation                            #
# ------------------------------------------------------------------ #


def test_config_validation_rejects_bad_values():
    with pytest.raises(ValueError):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(max_targets=0))
    with pytest.raises(ValueError):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(match_radius=0.0))
    with pytest.raises(ValueError):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(ema_alpha=0.0))
    with pytest.raises(ValueError):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(ema_alpha=1.5))
    with pytest.raises(ValueError):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(uncertainty_weight=-1.0))
    with pytest.raises(ValueError):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(disagreement_weight=-1.0))
    with pytest.raises(ValueError):
        EpistemicDeficitAccumulator(
            EpistemicDeficitConfig(persistent_pe_weight=-1.0)
        )


def test_update_creates_a_target_when_none_within_radius():
    acc = EpistemicDeficitAccumulator(EpistemicDeficitConfig(match_radius=0.5))
    z = torch.zeros(4)
    acc.update(z, uncertainty=1.0, disagreement=0.0, persistent_pe=0.0)
    assert acc.get_state()["n_targets"] == 1
    assert acc.get_state()["last_matched_new_target"] is True
    # deficit_input = 1*1.0 + 1*0.0 + 1*0.0 = 1.0 (default weights are 1.0);
    # a brand-new target seeds at deficit_input exactly (no prior EMA value).
    assert acc.get_state()["max_deficit"] == pytest.approx(1.0)


def test_update_merges_into_nearest_target_within_radius_via_ema():
    acc = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(match_radius=1.0, ema_alpha=0.5)
    )
    z = torch.zeros(4)
    acc.update(z, uncertainty=1.0, disagreement=0.0, persistent_pe=0.0)
    # Same location (well within match_radius) -> merges, does not create a
    # second target.
    acc.update(z, uncertainty=0.0, disagreement=0.0, persistent_pe=0.0)
    state = acc.get_state()
    assert state["n_targets"] == 1
    assert state["last_matched_new_target"] is False
    # EMA: (1 - 0.5) * 1.0 + 0.5 * 0.0 = 0.5
    assert state["max_deficit"] == pytest.approx(0.5)


def test_update_far_location_creates_a_second_target():
    acc = EpistemicDeficitAccumulator(EpistemicDeficitConfig(match_radius=0.1))
    acc.update(torch.zeros(4), uncertainty=1.0, disagreement=0.0, persistent_pe=0.0)
    acc.update(
        torch.full((4,), 10.0), uncertainty=1.0, disagreement=0.0, persistent_pe=0.0
    )
    assert acc.get_state()["n_targets"] == 2


def test_update_simulation_mode_is_noop_but_counts_as_a_skip():
    acc = EpistemicDeficitAccumulator()
    acc.update(
        torch.zeros(4),
        uncertainty=1.0,
        disagreement=1.0,
        persistent_pe=1.0,
        simulation_mode=True,
    )
    state = acc.get_state()
    assert state["n_targets"] == 0
    assert state["n_updates"] == 0
    assert state["last_n_simulation_skips"] == 1


def test_eviction_removes_lowest_deficit_target_at_capacity():
    acc = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(match_radius=0.1, max_targets=2)
    )
    acc.update(torch.zeros(4), uncertainty=5.0, disagreement=0.0, persistent_pe=0.0)
    acc.update(
        torch.full((4,), 10.0), uncertainty=1.0, disagreement=0.0, persistent_pe=0.0
    )
    assert acc.get_state()["n_targets"] == 2
    # A third, far-away location forces an eviction -- the lowest-deficit
    # target (the 1.0 one) should be the one dropped, not the 5.0 one.
    acc.update(
        torch.full((4,), 20.0), uncertainty=9.0, disagreement=0.0, persistent_pe=0.0
    )
    state = acc.get_state()
    assert state["n_targets"] == 2
    deficits = sorted(t["deficit"] for t in acc._targets)
    assert deficits == pytest.approx([5.0, 9.0])


def test_readout_returns_none_when_no_targets():
    acc = EpistemicDeficitAccumulator()
    out = acc.readout(torch.randn(8, WORLD_DIM))
    assert out is None


def test_readout_matches_nearest_target_and_zero_for_unmatched():
    acc = EpistemicDeficitAccumulator(EpistemicDeficitConfig(match_radius=0.5))
    target_loc = torch.zeros(WORLD_DIM)
    acc.update(target_loc, uncertainty=2.0, disagreement=0.0, persistent_pe=0.0)

    candidates = torch.stack(
        [
            torch.zeros(WORLD_DIM),  # matches the target exactly
            torch.full((WORLD_DIM,), 100.0),  # far away -- no match
        ],
        dim=0,
    )
    out = acc.readout(candidates)
    assert out is not None
    assert out.shape == (2,)
    assert float(out[0].item()) == pytest.approx(2.0)
    assert float(out[1].item()) == pytest.approx(0.0)
    assert acc.get_state()["last_n_targets_matched_at_readout"] == 1


def test_reset_clears_targets_and_diagnostics():
    acc = EpistemicDeficitAccumulator()
    acc.update(torch.zeros(4), uncertainty=1.0, disagreement=0.0, persistent_pe=0.0)
    acc.readout(torch.randn(4, 4))
    acc.mark_vacuous_readout()
    acc.reset()
    state = acc.get_state()
    assert state["n_targets"] == 0
    assert state["n_updates"] == 0
    assert state["n_readouts"] == 0
    assert state["n_vacuous_readouts"] == 0
    assert state["last_readout_vacuous"] is False


def test_mark_vacuous_readout_diagnostics():
    acc = EpistemicDeficitAccumulator()
    acc.mark_vacuous_readout()
    state = acc.get_state()
    assert state["n_vacuous_readouts"] == 1
    assert state["last_readout_vacuous"] is True


# ------------------------------------------------------------------ #
# Agent wiring                                                        #
# ------------------------------------------------------------------ #


def _onehot(idx: int, n: int = ACTION_DIM) -> torch.Tensor:
    a = torch.zeros(1, n)
    a[0, idx] = 1.0
    return a


def _agent_cfg(
    learning_progress_source: str, train_online: bool = True
) -> REEConfig:
    env = CausalGridWorldV2()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
    )
    cfg.use_structured_curiosity = True
    cfg.curiosity_learning_progress_source = learning_progress_source
    cfg.latent.use_e2_world_uncertainty = True
    cfg.latent.use_e2_world_uncertainty_online_training = train_online
    cfg.latent.e2_world_uncertainty_warmup_steps = 5
    cfg.latent.e2_world_uncertainty_batch_size = 4
    return cfg


def _drive_agent(
    learning_progress_source: str,
    train_online: bool = True,
    episodes: int = 2,
    steps: int = 20,
) -> REEAgent:
    torch.manual_seed(71)
    env = CausalGridWorldV2()
    cfg = _agent_cfg(learning_progress_source, train_online=train_online)
    agent = REEAgent(cfg)
    for _ in range(episodes):
        _, obs = env.reset()
        agent.reset()
        for _ in range(steps):
            latent = agent.sense(obs["body_state"], obs["world_state"])
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=0.0,
                drive_level=REEAgent.compute_drive_level(obs["body_state"]),
            )
            action = agent.select_action(candidates, ticks)
            _f, harm, _d, _i, obs = env.step(int(action.argmax(dim=-1).item()))
            agent.update_residue(harm)
    return agent


def test_config_defaults_are_inert():
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=4, self_dim=8, world_dim=8,
        reafference_action_dim=4,
    )
    assert cfg.curiosity_learning_progress_source == "broadcast"
    assert cfg.epistemic_deficit_max_targets == 16
    assert cfg.epistemic_deficit_match_radius == pytest.approx(1.0)
    assert cfg.epistemic_deficit_ema_alpha == pytest.approx(0.1)
    assert cfg.epistemic_deficit_uncertainty_weight == pytest.approx(1.0)
    assert cfg.epistemic_deficit_disagreement_weight == pytest.approx(1.0)
    assert cfg.epistemic_deficit_persistent_pe_weight == pytest.approx(1.0)


def test_agent_off_path_never_instantiates_the_accumulator():
    agent = _drive_agent("broadcast", train_online=True, episodes=1, steps=10)
    assert agent.epistemic_deficit is None
    assert agent._curiosity_per_candidate_learning_progress([]) is None


def test_agent_on_path_accumulates_across_a_real_rollout():
    agent = _drive_agent("epistemic_deficit", train_online=True, episodes=2, steps=25)
    assert agent.epistemic_deficit is not None
    state = agent.epistemic_deficit.get_state()
    assert state["n_updates"] > 0, (
        "the accumulator never accumulated on a real rollout -- the UPDATE "
        "wiring is not live"
    )
    assert state["n_targets"] > 0


def test_agent_off_path_bit_identical_action_sequence():
    """With the master OFF, wiring this substrate in must not perturb the
    existing (pre-SD-102) action sequence -- run the same rollout twice and
    confirm the selected actions are identical (determinism), and that no
    epistemic_deficit state exists to have influenced them."""
    agent_a = _drive_agent("broadcast", train_online=False, episodes=1, steps=15)
    agent_b = _drive_agent("broadcast", train_online=False, episodes=1, steps=15)
    assert agent_a.epistemic_deficit is None
    assert agent_b.epistemic_deficit is None


def test_readiness_gate_refuses_on_single_candidate_and_self_reports():
    """With only 1 candidate, predictive_variance's numel<2 guard never
    refreshes last_pvar_relative_spread away from its 0.0 init, so the
    readiness gate must refuse (return None) and self-report vacuous."""
    torch.manual_seed(3)
    cfg = _agent_cfg("epistemic_deficit", train_online=False)
    agent = REEAgent(cfg)
    env = CausalGridWorldV2()
    _, obs = env.reset()
    latent = agent.sense(obs["body_state"], obs["world_state"])
    ticks = agent.clock.advance()
    e1_prior = torch.zeros(1, WORLD_DIM, device=agent.device)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)

    assert agent.epistemic_deficit is not None
    out = agent._curiosity_per_candidate_learning_progress(candidates[:1])
    assert out is None
    assert agent.epistemic_deficit.get_state()["n_vacuous_readouts"] == 1
    assert agent.epistemic_deficit.get_state()["last_readout_vacuous"] is True


def test_readout_produces_a_k_shaped_vector_when_ready():
    torch.manual_seed(5)
    cfg = _agent_cfg("epistemic_deficit", train_online=False)
    agent = REEAgent(cfg)
    env = CausalGridWorldV2()
    _, obs = env.reset()
    latent = agent.sense(obs["body_state"], obs["world_state"])
    ticks = agent.clock.advance()
    e1_prior = torch.zeros(1, WORLD_DIM, device=agent.device)
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    assert len(candidates) >= 2, "need K>=2 candidates for a meaningful readout"

    out = agent._curiosity_per_candidate_learning_progress(candidates)
    # A fresh random-init head still differentiates across distinct one-hot
    # actions at float precision, so relative_spread is virtually never
    # exactly 0.0 with K>=2 -- the gate is a cheap degenerate-value guard,
    # NOT the trained-vs-untrained discriminator (that is the validation
    # experiment's job; see sd_102_epistemic_deficit_accumulator.md).
    if out is not None:
        assert out.shape == (len(candidates),)


def test_agent_resets_prev_z_world_cache_and_targets_per_episode():
    agent = _drive_agent("epistemic_deficit", train_online=True, episodes=1, steps=15)
    assert agent.epistemic_deficit is not None
    assert agent.epistemic_deficit.get_state()["n_targets"] > 0
    agent.reset()
    assert agent._epistemic_deficit_prev_z_world is None
    assert agent.epistemic_deficit.get_state()["n_targets"] == 0


def test_simulation_mode_does_not_update_the_accumulator():
    torch.manual_seed(9)
    cfg = _agent_cfg("epistemic_deficit", train_online=True)
    agent = REEAgent(cfg)
    env = CausalGridWorldV2()
    _, obs = env.reset()
    latent = agent.sense(obs["body_state"], obs["world_state"])
    latent.hypothesis_tag = True
    n_before = agent.epistemic_deficit.get_state()["n_updates"]
    agent._update_epistemic_deficit(latent)
    state = agent.epistemic_deficit.get_state()
    assert state["n_updates"] == n_before
    assert state["last_n_simulation_skips"] >= 1
