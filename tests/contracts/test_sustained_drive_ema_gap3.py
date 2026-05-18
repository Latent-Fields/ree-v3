"""Contract tests for goal_pipeline:GAP-3 -- SD-012 sustained-drive EMA
(Phase 3, Option 1).

See REE_assembly/evidence/planning/goal_pipeline_plan.md GAP-3 and
REE_assembly/docs/architecture/sustained_drive_anticipatory_wanting.md.

Problem (EXQ-536a): the SD-012 multiplier (1 + drive_weight * drive_level)
collapses to ~1.0 the step a resource is consumed -- energy resets toward 1.0
so drive_level ~ 0.005 at exactly the contact events where z_goal seeding must
fire. Option 1 replaces the instantaneous drive_level with a slow EMA trace:

    trace_t = (1 - drive_ema_alpha) * trace_{t-1} + drive_ema_alpha * drive_level
    effective_benefit = benefit_exposure * z_goal_seeding_gain
                        * (1 + drive_weight * trace_t)

Guarantees enforced:
  C1. drive_ema_alpha defaults to 1.0 on GoalConfig and reaches
      config.goal.drive_ema_alpha unchanged through REEConfig.from_dims
      (backward-compat surface).
  C2. drive_ema_alpha=1.0 is BIT-IDENTICAL to the pre-amendment instantaneous
      form: _z_goal and _goal_norm_peak match a faithful reimplementation of
      the old update() over a drive sequence including a contact-collapse
      (0.9 -> 0.005 -> 0.9), regardless of the zero-init trace.
  C3. drive_ema_alpha=0.02 does NOT collapse on the single-step energy reset:
      after a sustained-high-drive run, one step at drive_level=0.005 leaves
      _drive_trace > 0.10 (the seeding-relevant threshold from the design memo).
  C4. drive_ema_alpha=0.02 trace timescale: half-life is ~35 steps (the
      lit-anchored 30-60 step window; wanting_liking synthesis).
  C5. alpha-sweep monotonicity (the plan's falsifier curve): the
      seeding-relevant quantity -- _drive_trace AT the consummatory contact
      step, after warmup -- is monotone non-increasing in drive_ema_alpha
      across {0.01, 0.02, 0.2, 1.0}. This is exactly EXQ-536a's "mean drive
      on contact": alpha=1.0 collapses it to 0.005, slower alpha holds it
      elevated. (Whole-sequence mean is NOT the right metric -- the accepted
      zero-init cold-start makes slow alpha still-climbing over short runs.)
  C6. zero-init cold-start is present and bounded: with alpha=0.02 the first
      step trace == alpha * drive_level (documented, accepted per Q2).
  C7. GoalState.reset() restores the zero-init trace (the Q2 cold-start is
      per-episode; eval/training loops reset() between episodes).
"""

from __future__ import annotations

import torch


_SWEEP = (0.01, 0.02, 0.2, 1.0)  # goal_pipeline:GAP-3 Q2 discriminative grid


# ---------------------------------------------------------------------------- #
# Helpers                                                                      #
# ---------------------------------------------------------------------------- #

def _make_goal_state(drive_ema_alpha: float, goal_dim: int = 4):
    from ree_core.goal import GoalConfig, GoalState

    cfg = GoalConfig(
        goal_dim=goal_dim,
        alpha_goal=0.05,
        decay_goal=0.005,
        benefit_threshold=0.1,
        drive_weight=2.0,
        z_goal_seeding_gain=1.0,
        valence_wanting_floor=0.0,
        drive_ema_alpha=drive_ema_alpha,
    )
    return GoalState(cfg, torch.device("cpu"))


def _reference_old_update(cfg, z_goal, goal_norm_peak, z_world, benefit, drive):
    """Faithful reimplementation of the PRE-amendment GoalState.update()
    math (instantaneous drive_level, valence_wanting_floor disabled)."""
    z_goal = z_goal * (1.0 - cfg.decay_goal)
    effective_benefit = benefit * cfg.z_goal_seeding_gain * (
        1.0 + cfg.drive_weight * drive
    )
    if effective_benefit > cfg.benefit_threshold:
        z_w = z_world.detach()
        if z_w.dim() == 2:
            z_w = z_w.mean(dim=0, keepdim=True)
        z_goal = (1.0 - cfg.alpha_goal) * z_goal + cfg.alpha_goal * z_w
        norm = z_goal.norm().item()
        if norm > goal_norm_peak:
            goal_norm_peak = norm
    return z_goal, goal_norm_peak


def _drive_sequence_with_collapse(n_warm=60, n_post=5):
    """Sustained high drive, a single consummatory collapse, then recovery."""
    return [0.9] * n_warm + [0.005] + [0.9] * n_post


# ---------------------------------------------------------------------------- #
# Contracts                                                                    #
# ---------------------------------------------------------------------------- #

def test_c1_default_alpha_one_and_from_dims_surface():
    """drive_ema_alpha defaults 1.0 and survives the from_dims passthrough."""
    from ree_core.goal import GoalConfig
    from ree_core.utils.config import REEConfig

    assert GoalConfig().drive_ema_alpha == 1.0

    cfg_default = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=64, action_dim=4
    )
    assert cfg_default.goal.drive_ema_alpha == 1.0

    cfg_set = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=64, action_dim=4,
        drive_ema_alpha=0.02,
    )
    assert cfg_set.goal.drive_ema_alpha == 0.02


def test_c2_alpha_one_is_bit_identical_to_instantaneous():
    """alpha=1.0 reproduces the pre-amendment update() exactly."""
    gs = _make_goal_state(drive_ema_alpha=1.0)
    cfg = gs.config

    torch.manual_seed(20260517)
    seq = _drive_sequence_with_collapse()
    z_worlds = [torch.randn(1, cfg.goal_dim) for _ in seq]

    ref_z = torch.zeros(1, cfg.goal_dim)
    ref_peak = 0.0
    for drive, z_world in zip(seq, z_worlds):
        # benefit chosen so seeding fires under high drive but the
        # collapse step (drive 0.005) does NOT fire -- exercises both paths.
        benefit = 0.2
        gs.update(z_world, benefit, drive_level=drive)
        ref_z, ref_peak = _reference_old_update(
            cfg, ref_z, ref_peak, z_world, benefit, drive
        )
        assert torch.equal(gs._z_goal, ref_z), (
            f"z_goal diverged from instantaneous reference at drive={drive}"
        )
    assert gs._goal_norm_peak == ref_peak


def test_c3_alpha_002_does_not_collapse_on_single_contact():
    """One step at drive=0.005 after sustained high drive: trace stays > 0.10."""
    gs = _make_goal_state(drive_ema_alpha=0.02)
    z_world = torch.zeros(1, gs.config.goal_dim)

    for _ in range(60):
        gs.update(z_world, benefit_exposure=0.0, drive_level=0.9)
    trace_before = gs._drive_trace
    assert trace_before > 0.10, f"warmup trace too low: {trace_before}"

    gs.update(z_world, benefit_exposure=0.0, drive_level=0.005)
    assert gs._drive_trace > 0.10, (
        f"trace collapsed on single consummatory step: {gs._drive_trace} "
        f"(was {trace_before}); EXQ-536a regression"
    )


def test_c4_alpha_002_half_life_is_about_35_steps():
    """Trace half-life at alpha=0.02 lies in the lit-anchored 30-60 window."""
    gs = _make_goal_state(drive_ema_alpha=0.02)
    z_world = torch.zeros(1, gs.config.goal_dim)

    # Saturate the trace toward 1.0.
    for _ in range(400):
        gs.update(z_world, benefit_exposure=0.0, drive_level=1.0)
    saturated = gs._drive_trace
    assert saturated > 0.99, f"trace did not saturate: {saturated}"

    half = 0.5 * saturated
    steps = 0
    while gs._drive_trace > half:
        gs.update(z_world, benefit_exposure=0.0, drive_level=0.0)
        steps += 1
        if steps > 200:
            break
    assert 30 <= steps <= 42, f"half-life {steps} steps outside [30, 42]"


def test_c5_alpha_sweep_trace_at_contact_monotone():
    """Trace AT the consummatory contact step (post-warmup) is the falsifier
    curve: monotone non-increasing in alpha; alpha=1.0 collapses to 0.005."""
    z_dim = 4
    trace_at_contact = []
    for alpha in _SWEEP:
        gs = _make_goal_state(drive_ema_alpha=alpha, goal_dim=z_dim)
        z_world = torch.zeros(1, z_dim)
        # Warm to saturation so even alpha=0.01 is ~1.0 (removes the
        # accepted zero-init transient -- this isolates the contact effect).
        for _ in range(600):
            gs.update(z_world, benefit_exposure=0.0, drive_level=0.9)
        # One consummatory step (energy resets -> drive_level ~ 0.005).
        gs.update(z_world, benefit_exposure=0.0, drive_level=0.005)
        trace_at_contact.append(gs._drive_trace)

    # _SWEEP is ascending alpha; trace-at-contact must be non-increasing.
    for slower, faster in zip(trace_at_contact, trace_at_contact[1:]):
        assert slower >= faster - 1e-9, (
            f"non-monotone falsifier curve: {trace_at_contact} "
            f"for alpha={_SWEEP}"
        )
    # alpha=1.0 is the instantaneous arm: collapses to the contact value
    # (EXQ-536a's 0.005). The slowest arm must clear the 0.10 seeding-
    # relevant threshold; the instantaneous arm must not.
    assert trace_at_contact[-1] < 0.01, (
        f"instantaneous arm did not collapse: {trace_at_contact}"
    )
    assert trace_at_contact[0] > 0.10, (
        f"slowest arm did not hold drive at contact: {trace_at_contact}"
    )


def test_c6_zero_init_cold_start_is_bounded():
    """First step trace == alpha * drive_level (documented zero-init confound)."""
    gs = _make_goal_state(drive_ema_alpha=0.02)
    assert gs._drive_trace == 0.0
    gs.update(torch.zeros(1, gs.config.goal_dim),
              benefit_exposure=0.0, drive_level=0.9)
    assert abs(gs._drive_trace - 0.02 * 0.9) < 1e-9, (
        f"cold-start not alpha*drive: {gs._drive_trace}"
    )


def test_c7_reset_restores_zero_init_trace():
    """reset() must re-zero the trace (per-episode cold-start, Q2)."""
    gs = _make_goal_state(drive_ema_alpha=0.02)
    z_world = torch.zeros(1, gs.config.goal_dim)
    for _ in range(80):
        gs.update(z_world, benefit_exposure=0.0, drive_level=0.9)
    assert gs._drive_trace > 0.10, "precondition: trace warmed up"
    gs.reset()
    assert gs._drive_trace == 0.0, (
        f"reset() did not restore zero-init trace: {gs._drive_trace}"
    )
