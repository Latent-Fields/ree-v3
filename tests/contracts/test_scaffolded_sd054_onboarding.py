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


# ---------------------------------------------------------------------------
# C9: SD-057 cue-recall bridge (2026-06-04 amend). GAP-2 foraging-contact lever.
# ---------------------------------------------------------------------------

import torch  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    _sd049_kwargs,
    _contacted_resource_type,
    _maybe_cue_recall,
    _strongest_perceived_type,
    _new_cue_diag,
)


def test_c9_config_defaults_are_noop():
    cfg = ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_cue_recall_bridge_enabled is False
    assert cfg.scaffold_cue_n_resource_types == 3
    assert cfg.scaffold_cue_recall_min_proximity == 0.0


def test_c9_sd049_kwargs_off_empty_on_populated():
    off = ScaffoldedSD054OnboardingConfig()
    assert _sd049_kwargs(off) == {}
    on = ScaffoldedSD054OnboardingConfig(scaffold_cue_recall_bridge_enabled=True,
                                         scaffold_cue_n_resource_types=3)
    k = _sd049_kwargs(on)
    assert k.get("multi_resource_heterogeneity_enabled") is True
    assert k.get("per_axis_drive_enabled") is True
    assert k.get("n_resource_types") == 3


@pytest.mark.parametrize("phase", ["stage0", "p0", "p1", "p2"])
def test_c9_build_env_off_has_no_sd049(phase):
    env = _build_env(ScaffoldedSD054OnboardingConfig(), phase)
    assert getattr(env, "multi_resource_heterogeneity_enabled", False) is False


@pytest.mark.parametrize("phase", ["stage0", "p0", "p1", "p2"])
def test_c9_build_env_on_enables_sd049_with_views(phase):
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_cue_recall_bridge_enabled=True)
    env = _build_env(cfg, phase)
    assert getattr(env, "multi_resource_heterogeneity_enabled", False) is True
    _, obs = env.reset()
    assert any(kk.startswith("resource_field_view_") for kk in obs)
    assert "per_axis_drive" in obs


def test_c9_contacted_resource_type_helper():
    assert _contacted_resource_type({}) is None
    assert _contacted_resource_type({"resource_type_at_agent": torch.tensor([0])}) is None
    assert _contacted_resource_type({"resource_type_at_agent": torch.tensor([2])}) == 2
    # consumed-this-tick takes precedence
    assert _contacted_resource_type(
        {"sd049_consumed_type_tag_this_tick": 1, "resource_type_at_agent": torch.tensor([3])}
    ) == 1


def _bridge_agent():
    rc = REEConfig.from_dims(body_obs_dim=17, world_obs_dim=325, action_dim=4,
        world_dim=32, z_goal_enabled=True, drive_weight=2.0,
        use_incentive_token_bank=True, use_cue_recall=True, cue_recall_gain=0.2)
    rc.latent.use_resource_encoder = True
    return REEAgent(rc)


def test_c9_maybe_cue_recall_off_is_noop():
    agent = _bridge_agent()
    obs = {"resource_field_view_food": torch.ones(25)}

    class _Env:
        resource_type_names = ("food",)
    # bridge OFF -> 0 regardless
    assert _maybe_cue_recall(agent, _Env(), obs, 0.9,
                             ScaffoldedSD054OnboardingConfig()) == 0


def test_c9_maybe_cue_recall_fires_for_token_match_only():
    agent = _bridge_agent()
    # seed a token for food (type 1) only
    z = torch.zeros(1, 32); z[0, 0] = 1.0
    agent.goal_state.incentive_bank.update(1, 0.5, z)
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_cue_recall_bridge_enabled=True)

    class _Env:
        resource_type_names = ("food", "water", "novelty")
    # food strongest-perceived + token present -> fires
    obs_food = {"resource_field_view_food": torch.ones(25),
                "resource_field_view_water": torch.zeros(25),
                "resource_field_view_novelty": torch.zeros(25)}
    assert _maybe_cue_recall(agent, _Env(), obs_food, 0.9, cfg) == 1
    # water strongest-perceived but NO token -> no fire (identity-matched)
    obs_water = {"resource_field_view_food": torch.zeros(25),
                 "resource_field_view_water": torch.ones(25),
                 "resource_field_view_novelty": torch.zeros(25)}
    assert _maybe_cue_recall(agent, _Env(), obs_water, 0.9, cfg) == 0


# --- C9 cont.: SD-057 cue-recall FORMATION fix + instrumentation (V3-EXQ-638) ---


def test_c9_stage0_bind_flag_default_is_noop():
    # The formation-fix flag must default OFF (bit-identical Stage-0).
    assert ScaffoldedSD054OnboardingConfig().scaffold_stage0_bind_incentive_token is False


def test_c9_strongest_perceived_type_helper():
    class _Env:
        resource_type_names = ("food", "water", "novelty")
    # no field views at all -> (0, -1.0)
    assert _strongest_perceived_type(_Env(), {}) == (0, -1.0)
    # food field strongest -> tag 1
    obs = {"resource_field_view_food": torch.full((25,), 0.7),
           "resource_field_view_water": torch.zeros(25),
           "resource_field_view_novelty": torch.zeros(25)}
    bt, bp = _strongest_perceived_type(_Env(), obs)
    assert bt == 1 and abs(bp - 0.7) < 1e-6


def test_c9_cue_diag_attributes_no_token_reason():
    """The V3-EXQ-638 cue-silent ROOT CAUSE as a unit: a perceived type with an
    EMPTY bank must NOT fire AND must record reason 'no_token' (not a silent 0)."""
    agent = _bridge_agent()  # fresh agent -> empty incentive bank
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_cue_recall_bridge_enabled=True)

    class _Env:
        resource_type_names = ("food", "water", "novelty")
    obs = {"resource_field_view_food": torch.ones(25),
           "resource_field_view_water": torch.zeros(25),
           "resource_field_view_novelty": torch.zeros(25)}
    diag = _new_cue_diag()
    fired = _maybe_cue_recall(agent, _Env(), obs, 0.9, cfg, diag=diag)
    assert fired == 0
    assert diag["cue_nonfire_reason_counts"].get("no_token") == 1
    assert diag["n_external_cues_seen"] == 1   # the cue WAS perceived
    assert diag["token_bank_size"] == 0        # but the bank was empty
    assert diag["n_cue_recall_fires"] == 0
    assert diag["best_prox_peak"] == 1.0


def test_c9_cue_diag_attributes_resource_field_absent():
    agent = _bridge_agent()
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_cue_recall_bridge_enabled=True)

    class _Env:
        resource_type_names = ("food", "water", "novelty")
    diag = _new_cue_diag()
    assert _maybe_cue_recall(agent, _Env(), {}, 0.5, cfg, diag=diag) == 0
    assert diag["cue_nonfire_reason_counts"].get("resource_field_absent") == 1
    assert diag["n_external_cues_seen"] == 0
    assert diag["drive_peak"] == 0.5


def test_c9_cue_diag_records_fire_and_strength():
    agent = _bridge_agent()
    z = torch.zeros(1, 32); z[0, 0] = 1.0
    agent.goal_state.incentive_bank.update(1, 0.5, z)  # bind a food token
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_cue_recall_bridge_enabled=True)

    class _Env:
        resource_type_names = ("food", "water", "novelty")
    obs = {"resource_field_view_food": torch.ones(25),
           "resource_field_view_water": torch.zeros(25),
           "resource_field_view_novelty": torch.zeros(25)}
    diag = _new_cue_diag()
    assert _maybe_cue_recall(agent, _Env(), obs, 0.9, cfg, diag=diag) == 1
    assert diag["n_cue_recall_fires"] == 1
    assert diag["n_token_matches"] == 1
    assert diag["matched_token_strength_peak"] > 0.0
    assert diag["token_bank_size"] == 1


def test_c9_exception_path_is_attributed_not_swallowed(monkeypatch):
    """The pre-fix `except: pass` made cue failures invisible. A thrown error
    must now surface as an 'exception:<Type>' reason, while still not breaking
    the episode loop (returns 0)."""
    import experiments.scaffolded_sd054_onboarding as sched_mod

    def _boom(env, obs):
        raise RuntimeError("synthetic cue-path failure")
    monkeypatch.setattr(sched_mod, "_strongest_perceived_type", _boom)

    agent = _bridge_agent()
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_cue_recall_bridge_enabled=True)

    class _Env:
        resource_type_names = ("food",)
    diag = _new_cue_diag()
    assert _maybe_cue_recall(agent, _Env(), {"resource_field_view_food": torch.ones(25)},
                             0.9, cfg, diag=diag) == 0
    assert diag["cue_nonfire_reason_counts"].get("exception:RuntimeError") == 1


def test_c9_stage0_binding_populates_bank():
    """The FORMATION FIX: with scaffold_stage0_bind_incentive_token ON, Stage-0
    forced feeding binds tokens to the strongest-perceived type, so the bank is
    NON-EMPTY entering P1/P2 (the empty-bank state that silenced 638's cue)."""
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_stage0_episode_budget=2,
        scaffold_steps_per_episode=25,
    )
    # Build the agent from the actual Stage-0 env dims (bridge on -> SD-049 views),
    # with the SD-057 bank/cue flags, so real stepping shape-matches.
    env = _build_env(cfg, "stage0")
    rc = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, world_dim=32, self_dim=32, alpha_world=0.9,
        z_goal_enabled=True, drive_weight=2.0,
        use_incentive_token_bank=True, use_cue_recall=True, cue_recall_gain=0.2,
    )
    rc.latent.use_resource_encoder = True
    agent = REEAgent(rc)

    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    res = sched.run_stage0_nursery(agent, torch.device("cpu"))
    assert res.aborted is False
    assert res.token_bank_size_end > 0, (
        "FORMATION FIX FAIL: Stage-0 forced feeding bound no incentive token -- "
        "the bank is still empty entering P1/P2 (the 638 cue-silent root cause)"
    )


# --- C9 cont.: n_cue_recall_fires aggregation fix (2026-06-04b) --------------
# The clean underlying fix for the V3-EXQ-638 measurement gap: run_p2 / run_p1
# now AGGREGATE the per-episode cue fires onto a top-level
# P2OnboardingMetrics.n_cue_recall_fires / P1OnboardingResult.n_cue_recall_fires
# so a consumer doing getattr(p2, "n_cue_recall_fires", 0) no longer silently
# reads 0 even when the cue fired. Contract: the new field equals
# cue_diag["n_cue_recall_fires"] (both count the same fires) under the bridge-on
# path and is 0 under bridge-off.


class _StubAgentForP2:
    """Minimal agent stand-in: run_p2's pre-loop setup only needs eval() (and a
    getattr-absent goal_state, so _apply_goal_seeding_calibration is a no-op).
    The episode loop is faked so no real stepping is required."""

    def eval(self):
        pass


_P2_EP_REQUIRED_KEYS = {
    "z_goal_norm_peak": 0.0,
    "approach_commit_steps": 0,
    "contact_steps": 0,
    "contact_rate": 0.0,
    "bridge_cue_fires": 0,
    "dacc_bias_nonzero_steps": 0,
    "episode_length": 10,
    "n_contact_refresh_updates": 0,
    "n_decay_only_updates": 0,
    "n_skipped_protected_updates": 0,
    "z_goal_norm_at_contact_peak": 0.0,
    "num_contact_events": 0,
}


def test_c9_p2_aggregates_cue_recall_fires_equals_cue_diag():
    """Bridge-on path: P2OnboardingMetrics.n_cue_recall_fires is the sum of the
    per-episode fires AND equals cue_diag["n_cue_recall_fires"]. A faked
    _eval_episode mirrors the real wiring -- it increments the shared cue_diag in
    lockstep with its per-episode return, exactly as _maybe_cue_recall does --
    so this exercises run_p2's aggregation + constructor wiring deterministically.
    """
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_p2_episode_budget=3,
    )
    sched = ScaffoldedSD054OnboardingScheduler(cfg)

    per_ep_fires = [4, 0, 7]
    calls = {"i": 0}

    def _fake_eval(agent, env, device, *, contact_gated, contact_threshold,
                   gating_threshold, cue_diag=None, post_cue_diag=None):
        n = per_ep_fires[calls["i"]]
        calls["i"] += 1
        if cue_diag is not None:  # lockstep with the per-episode return (real path)
            cue_diag["n_cue_recall_fires"] += n
        return {**_P2_EP_REQUIRED_KEYS, "n_cue_recall_fires": n}

    sched._eval_episode = _fake_eval
    p2 = sched.run_p2(_StubAgentForP2(), torch.device("cpu"))

    assert p2.n_cue_recall_fires == sum(per_ep_fires) == 11
    assert p2.n_cue_recall_fires == p2.cue_diag["n_cue_recall_fires"]
    # The gap this closes: getattr now finds the real total, not a silent 0.
    assert getattr(p2, "n_cue_recall_fires", 0) == 11


def test_c9_p2_cue_recall_fires_zero_when_bridge_off():
    """Bridge-off path: no cue ever fires, so the aggregate is 0 and still equals
    cue_diag["n_cue_recall_fires"] (which stays 0). Bit-identical readout."""
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_cue_recall_bridge_enabled=False,
        scaffold_p2_episode_budget=2,
    )
    sched = ScaffoldedSD054OnboardingScheduler(cfg)

    def _fake_eval(agent, env, device, *, contact_gated, contact_threshold,
                   gating_threshold, cue_diag=None, post_cue_diag=None):
        # Bridge off -> _maybe_cue_recall returns 0 and never touches cue_diag.
        return {**_P2_EP_REQUIRED_KEYS, "n_cue_recall_fires": 0}

    sched._eval_episode = _fake_eval
    p2 = sched.run_p2(_StubAgentForP2(), torch.device("cpu"))

    assert p2.n_cue_recall_fires == 0
    assert p2.n_cue_recall_fires == p2.cue_diag["n_cue_recall_fires"]


def test_c9_p1_p2_result_field_defaults_zero_on_master_off():
    """The aggregated field exists on both result dataclasses and defaults to 0;
    the master-off short-circuit returns results carrying 0 (no cue accounting)."""
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=False,
    )
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    p1 = sched.run_p1(None, torch.device("cpu"))
    p2 = sched.run_p2(None, torch.device("cpu"))
    assert p1.n_cue_recall_fires == 0
    assert p2.n_cue_recall_fires == 0


# ---------------------------------------------------------------------------
# C10: V3-EXQ-640 post-cue MEASUREMENT-ONLY instrumentation
# ---------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    _nearest_resource_manhattan,
    _opposite_action,
    _new_post_cue_diag,
    _finalize_post_cue_window,
)


def test_c10_post_cue_config_defaults_are_noop():
    """The instrumentation flag defaults OFF and run_p2 then surfaces an EMPTY
    post_cue_diag dict -> bit-identical readout for every existing consumer."""
    cfg = ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_post_cue_instrumentation is False
    assert cfg.scaffold_post_cue_window_steps == 4

    cfg2 = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_p2_episode_budget=2,
    )
    sched = ScaffoldedSD054OnboardingScheduler(cfg2)

    def _fake_eval(agent, env, device, *, contact_gated, contact_threshold,
                   gating_threshold, cue_diag=None, post_cue_diag=None):
        # Flag off -> run_p2 passes post_cue_diag=None; the instrumentation block
        # in the real _eval_episode is skipped. The fake asserts that contract.
        assert post_cue_diag is None
        return {**_P2_EP_REQUIRED_KEYS, "n_cue_recall_fires": 0}

    sched._eval_episode = _fake_eval
    p2 = sched.run_p2(_StubAgentForP2(), torch.device("cpu"))
    assert p2.post_cue_diag == {}


def test_c10_post_cue_diag_passed_when_flag_on():
    """With the flag ON, run_p2 builds a _new_post_cue_diag() accumulator, threads
    it through every _eval_episode call, and surfaces it on P2OnboardingMetrics."""
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_p2_episode_budget=3,
        scaffold_post_cue_instrumentation=True,
        scaffold_post_cue_window_steps=5,
    )
    sched = ScaffoldedSD054OnboardingScheduler(cfg)

    def _fake_eval(agent, env, device, *, contact_gated, contact_threshold,
                   gating_threshold, cue_diag=None, post_cue_diag=None):
        # Real _eval_episode mutates the shared accumulator in place; mirror that.
        assert post_cue_diag is not None
        post_cue_diag["n_cue_fire_steps"] += 2
        post_cue_diag["n_cue_windows"] += 2
        return {**_P2_EP_REQUIRED_KEYS, "n_cue_recall_fires": 2}

    sched._eval_episode = _fake_eval
    p2 = sched.run_p2(_StubAgentForP2(), torch.device("cpu"))
    assert p2.post_cue_diag["post_cue_window_steps"] == 5
    assert p2.post_cue_diag["n_cue_fire_steps"] == 6   # 2 * 3 episodes
    assert p2.post_cue_diag["n_cue_windows"] == 6


def test_c10_nearest_resource_manhattan():
    """Non-toroidal Manhattan distance to the nearest resource; None on empty."""
    env = SimpleNamespace(agent_x=2, agent_y=2,
                          resources=[[2, 5], [4, 4], [0, 1]])
    # nearest: [2,5] d=3 ; [4,4] d=4 ; [0,1] d=3 -> 3
    assert _nearest_resource_manhattan(env) == 3
    env_empty = SimpleNamespace(agent_x=0, agent_y=0, resources=[])
    assert _nearest_resource_manhattan(env_empty) is None
    env_none = SimpleNamespace(agent_x=0, agent_y=0, resources=None)
    assert _nearest_resource_manhattan(env_none) is None


def test_c10_opposite_action():
    """Spatial-inverse detection over a real env ACTIONS table (oscillation)."""
    env = CausalGridWorldV2(size=8)
    actions = env.ACTIONS
    # find a pair of opposite actions
    found = False
    for i in range(len(actions)):
        for j in range(len(actions)):
            ax, ay = actions[i]
            bx, by = actions[j]
            if ax == -bx and ay == -by and (ax, ay) != (0, 0):
                assert _opposite_action(env, i, j) is True
                assert _opposite_action(env, i, i) is False
                found = True
                break
        if found:
            break
    assert found, "env ACTIONS has no opposite pair (unexpected)"


def test_c10_finalize_window_accumulates():
    """_finalize_post_cue_window folds a window's flags into the accumulator:
    first-move-approach, any-approach, latency (only on improved), hazard,
    oscillation."""
    diag = _new_post_cue_diag(window_steps=4)
    # window A: immediate approach (latency 1), hazard, 2 reversals.
    diag_window_a = {"age": 4, "improved": True, "first_latency": 1,
                     "first_move_approach": True, "hazard": True, "osc": 2}
    # window B: improving on step 3, no hazard, 0 reversals, not first-move.
    diag_window_b = {"age": 4, "improved": True, "first_latency": 3,
                     "first_move_approach": False, "hazard": False, "osc": 0}
    # window C: never improved (no gradient-following), no hazard.
    diag_window_c = {"age": 4, "improved": False, "first_latency": 0,
                     "first_move_approach": False, "hazard": False, "osc": 1}
    for w in (diag_window_a, diag_window_b, diag_window_c):
        _finalize_post_cue_window(diag, w)
    assert diag["n_cue_windows"] == 3
    assert diag["n_windows_first_move_approach"] == 1          # only A
    assert diag["n_windows_with_approach_move"] == 2           # A, B
    assert diag["n_windows_improved"] == 2                     # A, B
    assert diag["sum_first_improving_latency"] == 1 + 3        # only improved windows
    assert diag["n_windows_with_hazard_interrupt"] == 1        # only A
    assert diag["sum_window_oscillations"] == 2 + 0 + 1


# ---------------------------------------------------------------------------
# C11: foraging-competence residual (GAP-2 reach-contact, 2026-06-05).
# (1) reconcile contact-gating with the GoalState seeding firing threshold;
# (2) graded P1 reef-spawn weaning; (3) consumption-event-gated G3 readiness.
# ---------------------------------------------------------------------------

from experiments.scaffolded_sd054_onboarding import (  # noqa: E402
    substrate_readiness_from_results,
    Stage0NurseryResult,
)


def test_c11_config_defaults_are_noop():
    """New foraging-competence knobs default to no-op (bit-identical OFF)."""
    cfg = ScaffoldedSD054OnboardingConfig()
    assert cfg.scaffold_auto_reconcile_gating_to_seeding is False
    assert cfg.scaffold_p1_reef_spawn_hold_fraction == 0.0


def test_c11_effective_gating_off_falls_back_to_static():
    """auto-reconcile OFF -> _reconciled_gating_threshold None and
    _effective_gating_threshold == the static _gating_threshold (bit-identical)."""
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_p2_contact_benefit_threshold=1e-6)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    agent = _build_goal_enabled_agent(cfg)
    assert sched._reconciled_gating_threshold(agent) is None
    assert sched._effective_gating_threshold(agent) == sched._gating_threshold() == 1e-6


def test_c11_reconciled_gating_threshold_derives_from_goalconfig():
    """auto-reconcile ON -> the gating floor is DERIVED from the live GoalConfig:
    benefit_threshold / (gain * (1 + drive_weight * drive_floor)). A wild benefit
    just above it clears the GoalState seeding gate; just below does not."""
    cfg = ScaffoldedSD054OnboardingConfig(
        scaffold_auto_reconcile_gating_to_seeding=True,
        # Reconciled to the 634c-validated seeded-arm calibration.
        scaffold_z_goal_seeding_gain=1.5,
        scaffold_benefit_threshold=0.02,
        scaffold_drive_floor=0.9,
    )
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    agent = _build_goal_enabled_agent(cfg)
    # The scaffold must first push its calibration onto the live GoalConfig.
    sched._apply_goal_seeding_calibration(agent)
    gc = agent.goal_state.config
    derived = sched._reconciled_gating_threshold(agent)
    expected = gc.benefit_threshold / (
        gc.z_goal_seeding_gain * (1.0 + gc.drive_weight * gc.drive_floor)
    )
    assert derived == pytest.approx(expected)
    assert sched._effective_gating_threshold(agent) == pytest.approx(expected)
    # A benefit just above the derived floor clears the GoalState firing gate at
    # steady-state drive (trace ~ drive_floor); just below does not.
    def _effective(b):
        return b * gc.z_goal_seeding_gain * (1.0 + gc.drive_weight * gc.drive_floor)
    assert _effective(derived * 1.01) > gc.benefit_threshold
    assert _effective(derived * 0.99) < gc.benefit_threshold


def test_c11_reconciled_threshold_none_without_goal_state():
    """No goal_state -> reconciled threshold None even with the flag on."""
    cfg = ScaffoldedSD054OnboardingConfig(scaffold_auto_reconcile_gating_to_seeding=True)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)

    class _NoGoal:
        goal_state = None
    assert sched._reconciled_gating_threshold(_NoGoal()) is None


def test_c11_build_env_p1_spawn_in_reef_half_param():
    """_build_env p1 with p1_spawn_in_reef_half=True widens spawn to the reef
    half; default False keeps the midline band (bit-identical legacy P1)."""
    cfg = ScaffoldedSD054OnboardingConfig()
    reef_rows, mid_rows = [], []
    for _ in range(40):
        env_reef = _build_env(cfg, "p1", anneal_t=0.0, p1_spawn_in_reef_half=True)
        env_reef.reset()
        reef_rows.append(env_reef.agent_x)
        env_mid = _build_env(cfg, "p1", anneal_t=0.0, p1_spawn_in_reef_half=False)
        env_mid.reset()
        mid_rows.append(env_mid.agent_x)
    # Reef-half spawn produces some rows >= 8; midline default never does.
    assert sum(1 for r in reef_rows if r >= 8) > 0
    assert all(r in {5, 6, 7} for r in mid_rows), (
        f"legacy P1 spawn must stay midline, got {sorted(set(mid_rows))}"
    )


def test_c11_reef_spawn_hold_default_all_midline():
    """run_p1 with reef_hold=0.0 (default) spawns at midline every episode ->
    n_reef_spawn_episodes == 0 (bit-identical to legacy P1)."""
    import torch
    cfg = _dw_cfg(scaffold_p1_episode_budget=4, scaffold_steps_per_episode=8)
    agent = _build_goal_enabled_agent(cfg)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    p1 = sched.run_p1(agent, torch.device("cpu"))
    assert p1.n_reef_spawn_episodes == 0


def test_c11_reef_spawn_hold_early_p1_spawns_in_reef():
    """run_p1 with reef_hold>0 spawns in the reef half for the early held
    fraction of P1 and at midline afterwards: 0 < n_reef_spawn < n_eps."""
    import torch
    n_eps = 6
    cfg = _dw_cfg(
        scaffold_p1_episode_budget=n_eps,
        scaffold_steps_per_episode=6,
        scaffold_p1_reef_spawn_hold_fraction=0.5,
    )
    agent = _build_goal_enabled_agent(cfg)
    sched = ScaffoldedSD054OnboardingScheduler(cfg)
    p1 = sched.run_p1(agent, torch.device("cpu"))
    assert 0 < p1.n_reef_spawn_episodes < n_eps


def _seed_results(stage0_peak, p1_pass, contact_rate, frozen_peak, contact_peak):
    """Build a (Stage0, P1, P2) per-seed result triple for the readiness helper."""
    s0 = Stage0NurseryResult(
        n_episodes=1, mean_forced_benefit=1.0, z_goal_norm_peak=stage0_peak,
        z_goal_formed=stage0_peak > 0.4, aborted=False,
    )
    p1 = P1OnboardingResult(
        n_episodes=1, median_last_window_episode_length=100.0,
        survival_gate_passed=p1_pass, final_hazard_food_attraction=0.7,
        final_mech295_min_drive_to_fire=0.01,
        final_mech307_conjunction_z_beta_threshold=0.3, aborted=False,
    )
    p2 = P2OnboardingMetrics(
        n_episodes=1, z_goal_norm_peak_per_episode=[frozen_peak],
        z_goal_norm_peak_max=frozen_peak, approach_commit_steps=0,
        approach_commit_rate=0.0, bridge_cue_fires=0, dacc_bias_nonzero_steps=0,
        mean_episode_length=100.0, per_episode=[], contact_rate=contact_rate,
        z_goal_norm_at_contact_peak=contact_peak,
    )
    return s0, p1, p2


def test_c11_substrate_readiness_uses_consumption_gated_g3():
    """The seed-42 artifact: a seed carries a frozen Stage-0 trace through a
    zero-contact P2 (frozen_peak high, contact_peak 0). The default readiness
    helper feeds the CONSUMPTION-GATED peak -> G3 fails; the legacy frozen-peak
    path would (wrongly) pass G3."""
    # 3 seeds: stage0 + survival + contact all pass; frozen peak passes G3 but
    # the consumption-gated peak is 0 on all -> g3 must fail under the default.
    triples = [_seed_results(0.45, True, 0.2, 0.44, 0.0) for _ in range(3)]
    stage0 = [t[0] for t in triples]
    p1 = [t[1] for t in triples]
    p2 = [t[2] for t in triples]

    gate = substrate_readiness_from_results(stage0, p1, p2)
    assert gate["g3_source"] == "z_goal_norm_at_contact_peak"
    assert gate["stage0_positive_control"] and gate["g1_survival"] and gate["g2_contact"]
    assert gate["g3_zgoal"] is False           # consumption-gated catches the artifact
    assert gate["substrate_gate_passed"] is False

    legacy = substrate_readiness_from_results(stage0, p1, p2, use_consumption_gated_g3=False)
    assert legacy["g3_source"] == "z_goal_norm_peak_max"
    assert legacy["g3_zgoal"] is True          # frozen peak masks the artifact


def test_c11_substrate_readiness_full_pass_on_genuine_contact():
    """When the consumption-gated peak is genuinely high (real foraging contact
    seeded z_goal) on >=2/3 seeds, the readiness gate passes."""
    triples = [_seed_results(0.45, True, 0.2, 0.44, 0.43) for _ in range(3)]
    gate = substrate_readiness_from_results(
        [t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples]
    )
    assert gate["substrate_gate_passed"] is True
    assert gate["g3_zgoal"] is True
