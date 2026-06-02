"""
Contract tests for scaffolded_sd054_onboarding substrate (2026-05-31).

Substrate: NEW env kwarg reef_bipartite_agent_spawn_in_reef_half on
CausalGridWorldV2 + NEW experiment-harness scheduler module
experiments/scaffolded_sd054_onboarding.py.

Plan-of-record: REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
Substrate_queue entry: scaffolded_sd054_onboarding (status pending_implementation
-> implemented). IGW-20260531-029.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.scaffolded_sd054_onboarding import (
    P0OnboardingResult,
    P1OnboardingResult,
    P2OnboardingMetrics,
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _lerp,
    _set_goal_pipeline_frozen,
    _set_p1_anneal_state,
)


# ---------------------------------------------------------------------------
# C1: bit-identical OFF -- new env kwarg default does not change anything.
# ---------------------------------------------------------------------------


def test_c1_env_kwarg_default_is_false():
    """
    Default constructor must produce reef_bipartite_agent_spawn_in_reef_half=False.
    Bit-identical to pre-2026-05-31 SD-054-bipartite spawn behaviour.
    """
    env = CausalGridWorldV2(
        size=12,
        num_hazards=2,
        num_resources=3,
        reef_enabled=True,
        reef_bipartite_layout=True,
        seed=42,
    )
    assert env.reef_bipartite_agent_spawn_in_reef_half is False


def test_c1_default_vs_explicit_false_bit_identical():
    """
    Constructing with the new kwarg explicitly False must produce the same
    spawn position as the default constructor (no implicit RNG draw added).
    """
    env_default = CausalGridWorldV2(
        size=12,
        num_hazards=2,
        num_resources=3,
        reef_enabled=True,
        reef_bipartite_layout=True,
        seed=99,
    )
    env_explicit_false = CausalGridWorldV2(
        size=12,
        num_hazards=2,
        num_resources=3,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_agent_spawn_in_reef_half=False,
        seed=99,
    )
    env_default.reset()
    env_explicit_false.reset()
    assert (env_default.agent_x, env_default.agent_y) == (
        env_explicit_false.agent_x,
        env_explicit_false.agent_y,
    )


def test_c1_default_constructor_keeps_midline_band():
    """
    Across 30 seeds, the legacy bipartite spawn restricts agent_x to the
    midline band [5, 6, 7] at size=12, radius=1.
    """
    spawn_rows = []
    for s in range(30):
        env = CausalGridWorldV2(
            size=12,
            num_hazards=2,
            num_resources=3,
            reef_enabled=True,
            reef_bipartite_layout=True,
            seed=s,
        )
        env.reset()
        spawn_rows.append(env.agent_x)
    assert set(spawn_rows).issubset({5, 6, 7}), (
        f"legacy spawn must stay in midline band, got {sorted(set(spawn_rows))}"
    )


# ---------------------------------------------------------------------------
# C2: spawn-admissibility extension fires when the new kwarg is on.
# ---------------------------------------------------------------------------


def test_c2_spawn_in_reef_half_produces_reef_spawns():
    """
    When reef_bipartite_agent_spawn_in_reef_half=True, spawn admissibility
    widens to include reef-half rows (8, 9, 10 at size=12). Across 30 seeds,
    at least one spawn must land in the reef half (rows >= 8).
    """
    spawn_rows = []
    for s in range(30):
        env = CausalGridWorldV2(
            size=12,
            num_hazards=2,
            num_resources=3,
            reef_enabled=True,
            reef_bipartite_layout=True,
            reef_bipartite_agent_spawn_in_reef_half=True,
            seed=s,
        )
        env.reset()
        spawn_rows.append(env.agent_x)
    in_reef_half = sum(1 for r in spawn_rows if r >= 8)
    assert in_reef_half > 0, (
        "scaffolded onboarding spawn admissibility must produce some reef-half "
        f"spawns over 30 seeds, got rows {sorted(set(spawn_rows))}"
    )
    # Should still produce at most reef + agent-band rows.
    assert all(r in {5, 6, 7, 8, 9, 10} for r in spawn_rows), (
        f"spawn rows must be in agent-band union reef-half, got {sorted(set(spawn_rows))}"
    )


def test_c2_forage_pool_unchanged_under_spawn_extension():
    """
    Hazards and resources must still draw from the forage half (rows < 5)
    regardless of the new spawn-admissibility flag. The scaffold widens
    AGENT spawn only, not hazard / resource placement.
    """
    env = CausalGridWorldV2(
        size=12,
        num_hazards=2,
        num_resources=3,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_agent_spawn_in_reef_half=True,
        seed=42,
    )
    env.reset()
    for h in env.hazards:
        assert h[0] < 5, f"hazard must stay in forage half, got row {h[0]}"
    for r in env.resources:
        assert r[0] < 5, f"resource must stay in forage half, got row {r[0]}"


def test_c2_runtime_mutation_back_to_off_restricts_spawn():
    """
    The flag is runtime-mutable: setting reef_bipartite_agent_spawn_in_reef_half
    back to False on an existing env must restore the midline-band restriction
    on the next reset(). This is what the scheduler relies on between phases.
    """
    env = CausalGridWorldV2(
        size=12,
        num_hazards=2,
        num_resources=3,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_agent_spawn_in_reef_half=True,
        seed=0,
    )
    env.reset()
    env.reef_bipartite_agent_spawn_in_reef_half = False
    for s in range(20):
        env._rng = np.random.default_rng(s)
        env.reset()
        assert env.agent_x in {5, 6, 7}, (
            f"runtime flip back to OFF must restore midline band, got {env.agent_x} at seed {s}"
        )


# ---------------------------------------------------------------------------
# C3: scheduler module surface + master-switch default.
# ---------------------------------------------------------------------------


def test_c3_config_master_default_off():
    cfg = ScaffoldedSD054OnboardingConfig()
    assert cfg.use_scaffolded_sd054_onboarding_scheduler is False


def test_c3_config_memo_default_values():
    """
    Default values must match the substrate-design memo's Config Surface table
    so a downstream caller that only flips the master switch gets the
    plan-of-record curriculum.
    """
    cfg = ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_p0_episode_budget == 30
    assert cfg.scaffold_p1_episode_budget == 30
    assert cfg.scaffold_p2_episode_budget == 30
    assert cfg.scaffold_p0_proximity_harm_scale == 0.05
    assert cfg.scaffold_p1_anneal_hazard_food_attraction_min == 0.0
    assert cfg.scaffold_p1_anneal_hazard_food_attraction_max == 0.7
    assert cfg.scaffold_p1_anneal_proximity_harm_scale_min == 0.05
    assert cfg.scaffold_p1_anneal_proximity_harm_scale_max == 0.1
    assert cfg.scaffold_p1_anneal_mech295_min_drive_to_fire_max == 1.0
    assert cfg.scaffold_p1_anneal_mech295_min_drive_to_fire_min == 0.01
    assert cfg.scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_max == 0.6
    assert cfg.scaffold_p1_anneal_mech307_conjunction_z_beta_threshold_min == 0.3
    assert cfg.scaffold_p1_survival_gate_steps == 75


def test_c3_scheduler_disabled_short_circuits_on_master_off():
    """
    With master switch OFF, all three phase methods short-circuit and return
    aborted=True (or empty metrics). The scheduler must be a strict no-op
    on the OFF path -- no env construction, no agent mutation.
    """
    import torch

    cfg = ScaffoldedSD054OnboardingConfig()  # master False default
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    assert sched.enabled is False

    # Pass an obviously-fake agent; the scheduler must not touch it.
    class FakeAgent:
        class _E3:
            _running_variance = 0.42
        e3 = _E3()

    fake = FakeAgent()
    device = torch.device("cpu")
    p0 = sched.run_p0(fake, device)
    assert isinstance(p0, P0OnboardingResult)
    assert p0.aborted is True
    assert p0.abort_reason == "master_switch_off"
    p1 = sched.run_p1(fake, device)
    assert isinstance(p1, P1OnboardingResult)
    assert p1.aborted is True
    assert p1.survival_gate_passed is False
    p2 = sched.run_p2(fake, device)
    assert isinstance(p2, P2OnboardingMetrics)
    assert p2.n_episodes == 0
    assert p2.z_goal_norm_peak_max == 0.0


# ---------------------------------------------------------------------------
# C4: P1 anneal arithmetic + endpoint values.
# ---------------------------------------------------------------------------


def test_c4_lerp_endpoints():
    assert math.isclose(_lerp(1.0, 0.01, 0.0), 1.0)
    assert math.isclose(_lerp(1.0, 0.01, 1.0), 0.01, abs_tol=1e-9)
    assert math.isclose(_lerp(0.0, 0.7, 0.0), 0.0)
    assert math.isclose(_lerp(0.0, 0.7, 1.0), 0.7, abs_tol=1e-9)


def test_c4_lerp_clamps_out_of_range_t():
    # t < 0 clamps to 0; t > 1 clamps to 1.
    assert math.isclose(_lerp(0.0, 1.0, -0.5), 0.0)
    assert math.isclose(_lerp(0.0, 1.0, 1.7), 1.0)


def test_c4_p1_anneal_mutates_bridge_config():
    """
    _set_p1_anneal_state at anneal_t=0.0 sets mech295_min_drive_to_fire=1.0
    (bridge silent at default drive) and z_beta_threshold=0.6 (legacy
    pre-recalibration value). At anneal_t=1.0 it sets them to 0.01 / 0.3
    (the 2026-05-12 defaults).
    """
    cfg = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=True)

    class BC:
        min_drive_to_fire = 0.01
        mech307_conjunction_z_beta_threshold = 0.3

    class Bridge:
        def __init__(self):
            self.config = BC()

    class AC:
        use_mech295_liking_bridge = True
        use_mech307_conjunction = True
        mech295_min_drive_to_fire = 0.01
        mech307_conjunction_z_beta_threshold = 0.3

    class Agent:
        def __init__(self):
            self.config = AC()
            self.mech295_bridge = Bridge()

    agent = Agent()

    _set_p1_anneal_state(agent, cfg, anneal_t=0.0)
    assert math.isclose(agent.mech295_bridge.config.min_drive_to_fire, 1.0)
    assert math.isclose(agent.mech295_bridge.config.mech307_conjunction_z_beta_threshold, 0.6)
    assert math.isclose(agent.config.mech295_min_drive_to_fire, 1.0)
    assert math.isclose(agent.config.mech307_conjunction_z_beta_threshold, 0.6)

    _set_p1_anneal_state(agent, cfg, anneal_t=1.0)
    assert math.isclose(agent.mech295_bridge.config.min_drive_to_fire, 0.01, abs_tol=1e-9)
    assert math.isclose(
        agent.mech295_bridge.config.mech307_conjunction_z_beta_threshold, 0.3, abs_tol=1e-9
    )

    # Midpoint: drive 0.505, z_beta 0.45 (linear interp).
    _set_p1_anneal_state(agent, cfg, anneal_t=0.5)
    assert math.isclose(agent.mech295_bridge.config.min_drive_to_fire, 0.505)
    assert math.isclose(
        agent.mech295_bridge.config.mech307_conjunction_z_beta_threshold, 0.45
    )


def test_c4_goal_pipeline_freeze_toggle():
    class AC:
        use_mech295_liking_bridge = True
        use_mech307_conjunction = True

    class A:
        def __init__(self):
            self.config = AC()

    agent = A()
    _set_goal_pipeline_frozen(agent, frozen=True)
    assert agent.config.use_mech295_liking_bridge is False
    assert agent.config.use_mech307_conjunction is False
    _set_goal_pipeline_frozen(agent, frozen=False)
    assert agent.config.use_mech295_liking_bridge is True
    assert agent.config.use_mech307_conjunction is True


# ---------------------------------------------------------------------------
# C5: phase-specific env construction matches memo.
# ---------------------------------------------------------------------------


def test_c5_p0_env_scaffold_engaged():
    """
    P0 env constructor: scaffold spawn admissibility ON,
    hazard_food_attraction = 0.0, proximity_harm_scale = sub-target.
    """
    cfg = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=True)
    env = _build_env(cfg, "p0")
    assert env.reef_bipartite_agent_spawn_in_reef_half is True
    assert env.hazard_food_attraction == 0.0
    assert env.proximity_harm_scale == cfg.scaffold_p0_proximity_harm_scale


def test_c5_p1_env_anneal_endpoints():
    """
    P1 env at anneal_t=0.0 matches P0 hazard floor; at anneal_t=1.0 matches
    P2 target. Spawn admissibility is OFF throughout P1 (narrowed back to
    midline band).
    """
    cfg = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=True)
    env_start = _build_env(cfg, "p1", anneal_t=0.0)
    assert env_start.reef_bipartite_agent_spawn_in_reef_half is False
    assert math.isclose(env_start.hazard_food_attraction, 0.0)
    assert math.isclose(env_start.proximity_harm_scale, 0.05)

    env_end = _build_env(cfg, "p1", anneal_t=1.0)
    assert env_end.reef_bipartite_agent_spawn_in_reef_half is False
    assert math.isclose(env_end.hazard_food_attraction, 0.7, abs_tol=1e-9)
    assert math.isclose(env_end.proximity_harm_scale, 0.1, abs_tol=1e-9)


def test_c5_p2_env_pinned_to_target():
    """
    P2 env: full target config matching V3-EXQ-603b GAP-4 Tier-1 measurement env.
    """
    cfg = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=True)
    env = _build_env(cfg, "p2")
    assert env.reef_bipartite_agent_spawn_in_reef_half is False
    assert math.isclose(env.hazard_food_attraction, 0.7)
    assert math.isclose(env.proximity_harm_scale, 0.1)


def test_c5_build_env_rejects_unknown_phase():
    cfg = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=True)
    with pytest.raises(ValueError):
        _build_env(cfg, "p3")


# ---------------------------------------------------------------------------
# C6: Stage-0 positive control -- the scheduler MUST drive z_goal.
#
# V3-EXQ-603d / V3-EXQ-625b harness-fix regression guard. Before the
# 2026-06-02 amend, _train_episode (P0/P1) and _eval_episode (P2) never called
# agent.update_z_goal, so GoalState.update was never reached and z_goal stayed
# zero-init across every step of every arm (C4 z_goal_norm_peak=0.0 FAIL).
# These contracts make a z_goal=0 scheduler structurally unshippable.
#
# NOTE: a non-zero z_goal also requires the agent config to set
# z_goal_enabled=True (otherwise agent.goal_state is None and update_z_goal
# early-returns). 603d's config omitted it; the V3-EXQ-603e validation MUST
# set z_goal_enabled=True + drive_weight=2.0. These tests build a correctly
# configured agent to isolate the scheduler-wiring contract.
# ---------------------------------------------------------------------------


def _build_goal_enabled_agent(cfg: ScaffoldedSD054OnboardingConfig):
    """Build a minimal REEAgent with the goal pipeline enabled (goal_state
    present) whose obs dims match the scheduler's P2 env."""
    import torch  # noqa: F401
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent

    env = _build_env(cfg, "p2")
    rcfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        z_goal_enabled=True,
        drive_weight=2.0,
    )
    return REEAgent(rcfg)


def test_c6_stage0_positive_control_p2_seeds_zgoal(monkeypatch):
    """
    Stage-0 positive control: with forced supra-threshold benefit+drive, the
    scheduler's P2 measurement must produce a NON-ZERO z_goal_norm_peak. A
    scheduler that does not call update_z_goal yields exactly 0.0 here (the
    603d C4 FAIL signature), so this assertion is the unshippable gate.
    """
    import experiments.scaffolded_sd054_onboarding as sched_mod
    import torch

    # Force every seeding call to supra-threshold inputs so z_goal formation is
    # deterministic and independent of stochastic resource-contact in a short
    # eval episode.
    monkeypatch.setattr(
        sched_mod, "_benefit_and_drive", lambda obs_body: (0.5, 1.0)
    )

    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_p2_episode_budget=1,
        scaffold_steps_per_episode=15,
    )
    agent = _build_goal_enabled_agent(cfg)
    assert agent.goal_state is not None, (
        "precondition: z_goal_enabled config must create a goal_state"
    )

    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    p2 = sched.run_p2(agent, torch.device("cpu"))

    assert p2.z_goal_norm_peak_max > 0.0, (
        "Stage-0 FAIL: scheduler P2 produced z_goal_norm_peak_max=0.0 -- "
        "update_z_goal is not wired into _eval_episode (603d/625b regression)"
    )


def test_c6_update_z_goal_called_in_p1_not_p0(monkeypatch):
    """
    Phase-gating contract: update_z_goal must fire during P1 (goal pipeline
    unfrozen) and must NOT fire during P0 (warm-up stays goal-frozen by
    design). Spy on agent.update_z_goal and count calls per phase.
    """
    import experiments.scaffolded_sd054_onboarding as sched_mod
    import torch

    monkeypatch.setattr(
        sched_mod, "_benefit_and_drive", lambda obs_body: (0.5, 1.0)
    )

    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_p0_episode_budget=1,
        scaffold_p1_episode_budget=1,
        scaffold_steps_per_episode=12,
    )
    agent = _build_goal_enabled_agent(cfg)

    calls = {"n": 0}
    real_update = agent.update_z_goal

    def _spy(*a, **k):
        calls["n"] += 1
        return real_update(*a, **k)

    monkeypatch.setattr(agent, "update_z_goal", _spy)

    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    device = torch.device("cpu")

    sched.run_p0(agent, device)
    assert calls["n"] == 0, (
        f"P0 must stay goal-frozen: update_z_goal called {calls['n']} times in P0"
    )

    sched.run_p1(agent, device)
    assert calls["n"] > 0, (
        "P1 must seed z_goal: update_z_goal was never called during run_p1"
    )
