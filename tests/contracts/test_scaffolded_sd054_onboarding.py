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


# ---------------------------------------------------------------------------
# C6: nursery / forced-benefit feeding amend (2026-06-03).
# Routed by failure_autopsy_V3-EXQ-603e-626a-622_2026-06-03.
# ---------------------------------------------------------------------------

from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    Stage0NurseryResult,
    classify_interpretation_branch,
    evaluate_substrate_gate,
    stage_plan,
)


def test_c6_amend_config_defaults_are_noop():
    """New amend knobs default to no-op (Stage-0 disabled, P2 guard off, no hold)."""
    cfg = ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_stage0_enabled is False
    assert cfg.scaffold_stage0_forced_benefit == 1.0
    assert cfg.scaffold_stage0_forced_drive == 0.9
    assert cfg.scaffold_stage0_num_hazards == 0
    assert cfg.scaffold_stage0_z_goal_peak_gate == 0.4
    assert cfg.scaffold_p2_hazard_food_attraction_guard == -1.0  # <0 = no guard
    assert cfg.scaffold_p1_anneal_hold_fraction == 0.0  # pure-linear anneal
    # Existing memo defaults untouched (anti-regression).
    assert cfg.scaffold_p2_hazard_food_attraction == 0.7
    assert cfg.scaffold_p0_episode_budget == 30


def test_c6_stage0_aborts_on_master_off():
    cfg = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=False)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    r = sched.run_stage0_nursery(object(), device=None)
    assert isinstance(r, Stage0NurseryResult)
    assert r.aborted is True
    assert r.abort_reason == "master_switch_off"
    assert r.z_goal_formed is False


def test_c6_stage0_aborts_when_disabled():
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=False,
    )
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    r = sched.run_stage0_nursery(object(), device=None)
    assert r.aborted is True
    assert r.abort_reason == "stage0_disabled"


def test_c6_stage0_aborts_without_goal_state():
    """
    Forced-feed nursery needs a live goal_state (z_goal_enabled=True). A
    z_goal-disabled agent aborts loudly rather than silently producing z_goal=0.
    """
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
    )
    sched = ScaffoldedSD054OnboardingScheduler(cfg)

    class NoGoalAgent:
        goal_state = None

    r = sched.run_stage0_nursery(NoGoalAgent(), device=None)
    assert r.aborted is True
    assert r.abort_reason == "goal_state_none_set_z_goal_enabled_true"


def test_c6_stage0_env_is_safe_dense_nursery():
    cfg = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=True)
    env = _build_env(cfg, "stage0")
    assert env.reef_bipartite_agent_spawn_in_reef_half is True
    assert env.hazard_food_attraction == 0.0
    assert env.proximity_harm_scale == cfg.scaffold_stage0_proximity_harm_scale
    assert env.num_hazards == cfg.scaffold_stage0_num_hazards == 0


def test_c6_p2_hazard_guard_overrides_when_nonneg():
    """guard >= 0 overrides P2 hfa; guard < 0 keeps the (hard) 0.7 default."""
    base = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=True)
    env_default = _build_env(base, "p2")
    assert env_default.hazard_food_attraction == 0.7  # no guard

    guarded = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_p2_hazard_food_attraction_guard=0.3,
    )
    env_guarded = _build_env(guarded, "p2")
    assert math.isclose(env_guarded.hazard_food_attraction, 0.3)


def test_c6_p2_metrics_carry_contact_readout():
    """P2OnboardingMetrics exposes the foraging-contact-rate guard fields."""
    cfg = ScaffoldedSD054OnboardingConfig(use_scaffolded_sd054_onboarding_scheduler=False)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    m = sched.run_p2(object(), device=None)  # master-off short-circuit
    assert hasattr(m, "contact_steps")
    assert hasattr(m, "contact_rate")
    assert hasattr(m, "hazard_food_attraction_used")
    assert m.contact_rate == 0.0


def test_c6_substrate_gate_full_pass():
    gate = evaluate_substrate_gate(
        stage0_z_goal_peaks_per_seed=[0.6, 0.7, 0.55],
        p1_survival_pass_per_seed=[True, True, False],
        p2_z_goal_peaks_per_seed=[0.5, 0.45, 0.1],
        p2_contact_rates_per_seed=[0.2, 0.1, 0.0],
    )
    assert gate["stage0_positive_control"] is True
    assert gate["g1_survival"] is True   # 2/3
    assert gate["g2_contact"] is True    # 2/3
    assert gate["g3_zgoal"] is True      # 2/3
    assert gate["substrate_gate_passed"] is True


def test_c6_substrate_gate_blocks_without_stage0():
    """
    z_goal=0 is NOT interpretable without the feeding positive control. A
    same-substrate retest with the nursery disabled supplies no Stage-0 z_goal
    peaks -> stage0_positive_control False -> gate cannot pass -> the old path
    cannot masquerade as 603f.
    """
    gate = evaluate_substrate_gate(
        stage0_z_goal_peaks_per_seed=[],          # nursery never run
        p1_survival_pass_per_seed=[True, True, True],
        p2_z_goal_peaks_per_seed=[0.5, 0.5, 0.5],
        p2_contact_rates_per_seed=[0.2, 0.2, 0.2],
    )
    assert gate["stage0_positive_control"] is False
    assert gate["substrate_gate_passed"] is False


def test_c6_substrate_gate_blocks_on_starvation():
    """Fed-but-no-contact / starvation: g2_contact fails -> gate fails."""
    gate = evaluate_substrate_gate(
        stage0_z_goal_peaks_per_seed=[0.6, 0.6, 0.6],
        p1_survival_pass_per_seed=[False, False, True],   # 1/3 survive
        p2_z_goal_peaks_per_seed=[0.0, 0.0, 0.0],
        p2_contact_rates_per_seed=[0.0, 0.0, 0.0],        # never fed in P2
    )
    assert gate["g1_survival"] is False
    assert gate["g2_contact"] is False
    assert gate["g3_zgoal"] is False
    assert gate["substrate_gate_passed"] is False


def test_c6_interpretation_branches():
    not_engaged = evaluate_substrate_gate([], [False], [0.0], [0.0])
    assert classify_interpretation_branch(not_engaged) == "substrate_not_engaged"

    # Fed (stage0 + survival + contact) but z_goal does not form ecologically.
    fed_no_goal = evaluate_substrate_gate(
        [0.6, 0.6, 0.6], [True, True, True], [0.0, 0.0, 0.0], [0.2, 0.2, 0.2]
    )
    assert fed_no_goal["g2_contact"] is True and fed_no_goal["g3_zgoal"] is False
    assert classify_interpretation_branch(fed_no_goal) == "fed_but_no_goal"

    full = evaluate_substrate_gate(
        [0.6, 0.6, 0.6], [True, True, True], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]
    )
    assert full["substrate_gate_passed"] is True
    assert classify_interpretation_branch(full, diversity_resolved=False) == "goal_formed_diversity_inert"
    assert classify_interpretation_branch(full, diversity_resolved=True) == "goal_formed_mechanisms_load_bearing"
    assert classify_interpretation_branch(full, behaviour_harmful=True) == "goal_formed_behaviour_random_harmful"
    assert classify_interpretation_branch(full) == "goal_formed_diversity_undetermined"


def test_c6_stage_plan_has_five_stages_nursery_first():
    plan = stage_plan()
    assert len(plan) == 5
    assert plan[0]["stage"] == "0"
    assert plan[0]["method"] == "run_stage0_nursery"
    assert plan[-1]["method"] == "run_p2"


# ---------------------------------------------------------------------------
# C7: developmental-window / consolidation amend (2026-06-03b).
# Routed by the V3-EXQ-634 design-error review: GoalState.update() always decays
# the persistent z_goal attractor, and P1/P2 call update_z_goal every step incl.
# unfed steps, washing out the Stage-0 trace before ecological contact. These
# contracts prove: (1) Stage-0 trace persists across Stage-0b consolidation;
# (2) decay-only updates are blocked during the protected/contact-gated window;
# (3) ecological contact still refreshes z_goal; (4) flags-off is bit-identical
# to the pre-amend (legacy 634) path.
# ---------------------------------------------------------------------------

from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    Stage0bConsolidationResult,
    GOAL_WRITE_MODES,
)


def _dw_cfg(**overrides):
    """Small developmental-window config for fast contract runs."""
    base = dict(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=2,
        scaffold_stage0b_episode_budget=3,
        scaffold_p1_episode_budget=1,
        scaffold_p2_episode_budget=1,
        scaffold_steps_per_episode=10,
        scaffold_p1_survival_gate_steps=1,
    )
    base.update(overrides)
    return ScaffoldedSD054OnboardingConfig(**base)


def test_c7_config_defaults_are_noop():
    """New developmental-window knobs default to no-op (window disabled)."""
    cfg = ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_developmental_window_enabled is False
    assert cfg.scaffold_stage0b_enabled is False
    assert cfg.scaffold_stage0b_episode_budget == 10
    assert cfg.scaffold_stage0b_retention_gate == 0.75
    assert cfg.scaffold_contact_gated_goal_updates is False
    assert len(GOAL_WRITE_MODES) == 5


def test_c7_stage0_trace_persists_across_stage0b(monkeypatch):
    """Stage-0 trace must survive the protected Stage-0b consolidation window:
    update_z_goal is never called, so retention_ratio == 1.0 (>= the 0.75 gate)
    and the end norm equals the start norm. This is the developmental-window
    acceptance the V3-EXQ-634 review said was missing."""
    import experiments.scaffolded_sd054_onboarding as sched_mod
    import torch
    # Deterministic supra-threshold feed so Stage-0 reliably forms z_goal.
    monkeypatch.setattr(sched_mod, "_benefit_and_drive", lambda ob: (0.5, 1.0))

    cfg = _dw_cfg(
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
    )
    agent = _build_goal_enabled_agent(cfg)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    dev = torch.device("cpu")

    s0 = sched.run_stage0_nursery(agent, dev)
    assert s0.z_goal_norm_peak > 0.0, "precondition: Stage-0 must form z_goal"
    norm_before = agent.goal_state.goal_norm()

    s0b = sched.run_stage0b_consolidation(agent, dev, stage0_baseline_norm=s0.z_goal_norm_peak)
    assert isinstance(s0b, Stage0bConsolidationResult)
    assert not s0b.aborted
    assert s0b.n_episodes == 3
    # Protected: z_goal not touched -> norm unchanged across the window.
    assert abs(s0b.z_goal_norm_end - norm_before) < 1e-6
    assert s0b.retention_ratio >= 0.75
    assert s0b.retention_gate_passed is True


def test_c7_decay_only_blocked_in_protected_window(monkeypatch):
    """KEY contract: under contact-gating, an UNFED P1 must not wash out z_goal.
    With the window OFF the same unfed P1 applies decay-only updates and the
    norm drops -- the contrast that defines the bug and its fix."""
    import experiments.scaffolded_sd054_onboarding as sched_mod
    import torch
    dev = torch.device("cpu")

    def _run(window_on):
        # Stage-0 feeds (forced), then P1 runs fully UNFED (benefit=0).
        feeds = {"n": 0}

        def _bd(ob):
            # Stage-0 uses forced_benefit so this is only consulted for drive
            # there; in P1 it returns benefit=0 (unfed) to force decay-only.
            return (0.0, 0.5)

        monkeypatch.setattr(sched_mod, "_benefit_and_drive", _bd)
        cfg = _dw_cfg(
            scaffold_developmental_window_enabled=window_on,
            scaffold_contact_gated_goal_updates=window_on,
            scaffold_p1_episode_budget=2,
            scaffold_steps_per_episode=15,
        )
        agent = _build_goal_enabled_agent(cfg)
        sched = ScaffoldedSD054OnboardingScheduler(cfg)
        sched.run_stage0_nursery(agent, dev)
        norm_after_stage0 = agent.goal_state.goal_norm()
        p1 = sched.run_p1(agent, dev)
        norm_after_p1 = agent.goal_state.goal_norm()
        return norm_after_stage0, norm_after_p1, p1

    # Window ON: unfed P1 is contact-gated -> no decay-only -> norm preserved.
    on_s0, on_p1, p1_on = _run(window_on=True)
    assert on_s0 > 0.0
    assert p1_on.contact_gated is True
    assert p1_on.n_decay_only_updates == 0
    assert p1_on.n_skipped_protected_updates > 0
    assert abs(on_p1 - on_s0) < 1e-6, "protected window must not wash out z_goal"

    # Window OFF: legacy path applies decay-only every unfed step -> norm drops.
    off_s0, off_p1, p1_off = _run(window_on=False)
    assert p1_off.contact_gated is False
    assert p1_off.n_decay_only_updates > 0
    assert p1_off.n_skipped_protected_updates == 0
    assert off_p1 < off_s0, "legacy path must wash out z_goal under unfed decay-only"


def test_c7_ecological_contact_still_refreshes(monkeypatch):
    """Contact-gating must not block legitimate refresh: a FED step still calls
    update_z_goal (n_contact_refresh > 0) and moves z_goal."""
    import experiments.scaffolded_sd054_onboarding as sched_mod
    import torch
    dev = torch.device("cpu")
    # Every P1 step is a validated contact (benefit supra-threshold).
    monkeypatch.setattr(sched_mod, "_benefit_and_drive", lambda ob: (0.5, 1.0))

    cfg = _dw_cfg(
        scaffold_developmental_window_enabled=True,
        scaffold_contact_gated_goal_updates=True,
        scaffold_stage0_episode_budget=1,
        scaffold_p1_episode_budget=1,
        scaffold_steps_per_episode=12,
    )
    agent = _build_goal_enabled_agent(cfg)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    sched.run_stage0_nursery(agent, dev)
    p1 = sched.run_p1(agent, dev)
    assert p1.contact_gated is True
    assert p1.n_contact_refresh_updates > 0, "fed steps must still refresh z_goal"
    assert p1.n_skipped_protected_updates == 0, "no step should be skipped when all are contacts"


def test_c7_flags_off_bit_identical_legacy_path(monkeypatch):
    """With the developmental-window flags OFF, P1/P2 take the legacy every-step
    decay-only path and Stage-0b aborts (disabled) -- bit-identical to 634."""
    import experiments.scaffolded_sd054_onboarding as sched_mod
    import torch
    dev = torch.device("cpu")
    monkeypatch.setattr(sched_mod, "_benefit_and_drive", lambda ob: (0.0, 0.5))

    cfg = _dw_cfg()  # window flags default off
    agent = _build_goal_enabled_agent(cfg)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)

    s0b = sched.run_stage0b_consolidation(agent, dev)
    assert s0b.aborted is True
    assert s0b.abort_reason == "stage0b_disabled"

    sched.run_stage0_nursery(agent, dev)
    p1 = sched.run_p1(agent, dev)
    assert p1.contact_gated is False
    assert p1.n_skipped_protected_updates == 0
    assert p1.n_decay_only_updates > 0  # legacy: update called every unfed step
    p2 = sched.run_p2(agent, dev)
    assert p2.contact_gated is False
    assert p2.n_skipped_protected_updates == 0


# ---------------------------------------------------------------------------
# C8: seeding-calibration amend (V3-EXQ-634b autopsy, 2026-06-03c).
# Decouples the contact-GATING threshold from the contact-RATE readout,
# propagates GoalConfig seeding magnitudes from the scaffold, and adds the
# consumption-event-gated z_goal readout (the fair G3 input).
# ---------------------------------------------------------------------------


def test_c8_seeding_calibration_config_defaults_are_noop():
    """New seeding-calibration knobs default to no-op (sentinel / None)."""
    cfg = ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_contact_gating_benefit_threshold == -1.0
    assert cfg.scaffold_z_goal_seeding_gain is None
    assert cfg.scaffold_benefit_threshold is None
    assert cfg.scaffold_drive_floor is None


def test_c8_gating_threshold_falls_back_to_readout_then_decouples():
    """Sentinel < 0 -> gating uses the contact-rate readout threshold
    (bit-identical). >= 0 -> gating is decoupled from the readout."""
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_p2_contact_benefit_threshold=1e-6)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    assert sched._gating_threshold() == 1e-6  # fallback
    cfg.scaffold_contact_gating_benefit_threshold = 0.1
    assert sched._gating_threshold() == 0.1  # decoupled


def test_c8_apply_seeding_calibration_noop_and_applies():
    """Default config leaves the agent's GoalConfig untouched; set knobs are
    propagated onto GoalState.config so genuine wild contact can seed."""
    import torch  # noqa: F401

    # (1) defaults -> no-op
    cfg = ScaffoldedSD054OnboardingConfig()
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    agent = _build_goal_enabled_agent(cfg)
    gc = agent.goal_state.config
    before = (gc.z_goal_seeding_gain, gc.benefit_threshold, gc.drive_floor)
    sched._apply_goal_seeding_calibration(agent)
    assert (gc.z_goal_seeding_gain, gc.benefit_threshold, gc.drive_floor) == before

    # (2) set knobs -> applied
    cfg.scaffold_z_goal_seeding_gain = 1.5
    cfg.scaffold_benefit_threshold = 0.02
    cfg.scaffold_drive_floor = 0.9
    sched._apply_goal_seeding_calibration(agent)
    assert gc.z_goal_seeding_gain == 1.5
    assert gc.benefit_threshold == 0.02
    assert gc.drive_floor == 0.9

    # (3) arithmetic: wild benefit 0.03 now clears the firing threshold
    eff = 0.03 * gc.z_goal_seeding_gain * (1.0 + gc.drive_weight * gc.drive_floor)
    assert eff > gc.benefit_threshold


def test_c8_calibration_applied_by_run_p1():
    """run_p1 applies the scaffold seeding calibration to the live GoalConfig."""
    import torch
    cfg = _dw_cfg(
        scaffold_benefit_threshold=0.02,
        scaffold_z_goal_seeding_gain=1.5,
        scaffold_drive_floor=0.9,
    )
    agent = _build_goal_enabled_agent(cfg)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    sched.run_p1(agent, torch.device("cpu"))
    gc = agent.goal_state.config
    assert gc.benefit_threshold == 0.02
    assert gc.z_goal_seeding_gain == 1.5
    assert gc.drive_floor == 0.9


def test_c8_contact_gating_decoupled_protects_subseeding_whiff(monkeypatch):
    """KEY decoupling contract: a benefit in the band (readout_floor, seeding_floor)
    is counted as ecological contact (readout) yet PROTECTED (skipped) by the
    gating threshold -- so it cannot decay-only erode the trace. The SAME benefit
    with the gating sentinel (fallback to readout) instead SEEDS (refresh)."""
    import experiments.scaffolded_sd054_onboarding as sched_mod
    import torch
    dev = torch.device("cpu")
    # P1 benefit sits in the band: above readout (1e-6) but below gating (0.1).
    monkeypatch.setattr(sched_mod, "_benefit_and_drive", lambda ob: (0.05, 1.0))

    def _run(gating):
        cfg = _dw_cfg(
            scaffold_developmental_window_enabled=True,
            scaffold_contact_gated_goal_updates=True,
            scaffold_p2_contact_benefit_threshold=1e-6,
            scaffold_contact_gating_benefit_threshold=gating,
            scaffold_stage0_episode_budget=1,
            scaffold_p1_episode_budget=1,
            scaffold_steps_per_episode=12,
        )
        agent = _build_goal_enabled_agent(cfg)
        sched = ScaffoldedSD054OnboardingScheduler(cfg)
        sched.run_stage0_nursery(agent, dev)
        return sched.run_p1(agent, dev)

    # Decoupled (gating 0.1): the 0.05 whiff is protected, never decays.
    decoupled = _run(gating=0.1)
    assert decoupled.n_skipped_protected_updates > 0
    assert decoupled.n_contact_refresh_updates == 0
    assert decoupled.n_decay_only_updates == 0

    # Sentinel (-1 -> fallback to readout 1e-6): the 0.05 step now SEEDS.
    legacy = _run(gating=-1.0)
    assert legacy.n_contact_refresh_updates > 0
    assert legacy.n_skipped_protected_updates == 0


def test_c8_p2_consumption_gated_peak_distinct_from_frozen_peak(monkeypatch):
    """G3 redesign contract: with z_goal carried from Stage-0 but NO genuine
    seeding contact in P2 (every P2 benefit is sub-seeding -> protected), the
    frozen peak (z_goal_norm_peak_max) stays > 0 (carried nursery trace) while
    the consumption-event-gated readout is exactly 0 with num_contact_events==0
    -- the seed-42 artifact and its fix."""
    import experiments.scaffolded_sd054_onboarding as sched_mod
    import torch
    dev = torch.device("cpu")

    # P2 benefit 0.05 is above readout (so it counts as contact_rate) but below
    # the gating/seeding floor (0.1) -> protected, never seeds z_goal in P2.
    monkeypatch.setattr(sched_mod, "_benefit_and_drive", lambda ob: (0.05, 1.0))

    cfg = _dw_cfg(
        scaffold_developmental_window_enabled=True,
        scaffold_contact_gated_goal_updates=True,
        scaffold_p2_contact_benefit_threshold=1e-6,
        scaffold_contact_gating_benefit_threshold=0.1,
        scaffold_stage0_episode_budget=1,
        scaffold_p2_episode_budget=1,
        scaffold_steps_per_episode=12,
    )
    agent = _build_goal_enabled_agent(cfg)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    # Stage-0 forced-feed seeds a non-zero carried trace.
    s0 = sched.run_stage0_nursery(agent, dev)
    assert s0.z_goal_norm_peak > 0.0

    p2 = sched.run_p2(agent, dev)
    # Frozen peak reflects the carried trace (the misleading G3 signal).
    assert p2.z_goal_norm_peak_max > 0.0
    # Consumption-gated readout is honest: no genuine seeding event occurred.
    assert p2.num_contact_events == 0
    assert p2.z_goal_norm_at_contact_peak == 0.0
    # ... yet the contact-RATE readout still registers ecological engagement.
    assert p2.contact_steps > 0
