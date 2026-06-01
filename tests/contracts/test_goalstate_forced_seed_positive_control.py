"""
Stage-0 positive-control contract for the goal/wanting/liking stream.

Purpose
-------
Institutionalise the positive control that V3-EXQ-626 lacked. 626 FAILed with
z_goal medians [0,0,0] across every arm NOT because the substrate cannot form a
goal, but because the bespoke experiment loop never called update_z_goal (the
only hook into GoalState.update). See
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-626_2026-06-01.md.

These tests exercise GoalState.update directly with forced ideal inputs (no agent
loop, no environment) and assert that drive-modulated benefit seeds a non-zero,
direction-stable z_goal that decays / floors correctly. If any of these fail, the
substrate gate itself has regressed (which 622 S0 PASS + V3-EXQ-582a PASS say it
has not) -- as opposed to a harness bug, which these tests make structurally
visible.

Acceptance (intake ladder Stage 0; goal_stream_repair_diagnostic_ladder_2026-06-01.md)
  A0.1  effective_benefit crosses benefit_threshold for forced benefit+drive.
  A0.2  goal_norm() >= 0.1 within <= 10 forced updates.
  A0.3  z_goal direction stable: cosine(z_goal, z_world_forced) >= 0.9 once seeded.
  A0.4  decay + valence_wanting_floor behave as documented.
  A0.5  negative control: a sub-threshold benefit pulse does NOT seed z_goal.

Run: /opt/local/bin/python3 -m pytest tests/contracts/test_goalstate_forced_seed_positive_control.py -q
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn.functional as F

from ree_core.goal import GoalConfig, GoalState

DEVICE = torch.device("cpu")
GOAL_DIM = 8


def _forced_z(dim=GOAL_DIM):
    """Fixed unit-norm world latent direction the goal should point toward."""
    z = torch.ones(1, dim)
    return z / z.norm()


def _make_state(
    benefit_threshold=0.1,
    drive_weight=2.0,
    drive_floor=0.9,
    drive_ema_alpha=1.0,
    valence_wanting_floor=0.0,
    alpha_goal=0.05,
    decay_goal=0.005,
):
    cfg = GoalConfig(
        goal_dim=GOAL_DIM,
        alpha_goal=alpha_goal,
        decay_goal=decay_goal,
        benefit_threshold=benefit_threshold,
        drive_weight=drive_weight,
        drive_floor=drive_floor,
        drive_ema_alpha=drive_ema_alpha,
        valence_wanting_floor=valence_wanting_floor,
        z_goal_enabled=True,
    )
    return GoalState(cfg, DEVICE)


def test_a0_1_effective_benefit_crosses_threshold():
    """A0.1: forced benefit + drive yields effective_benefit > benefit_threshold."""
    gs = _make_state()
    # Replicate the internal effective_benefit formula at drive_floor steady state.
    benefit = 0.2
    drive_level = 1.0
    # one update advances the EMA trace; with alpha=1.0, trace = max(drive,floor) = 1.0
    gs.update(_forced_z(), benefit_exposure=benefit, drive_level=drive_level)
    effective = (
        benefit
        * gs.config.z_goal_seeding_gain
        * (1.0 + gs.config.drive_weight * gs._drive_trace)
    )
    assert effective > gs.config.benefit_threshold, (
        f"A0.1 FAIL: effective_benefit={effective:.4f} <= "
        f"threshold={gs.config.benefit_threshold}"
    )
    print(f"A0.1 PASS: effective_benefit={effective:.4f} > {gs.config.benefit_threshold}")


def test_a0_2_goal_norm_becomes_nonzero_within_10_steps():
    """A0.2: goal_norm() reaches >= 0.1 within 10 forced updates."""
    gs = _make_state()
    z = _forced_z()
    crossed_at = None
    for step in range(1, 11):
        gs.update(z, benefit_exposure=0.2, drive_level=1.0)
        if gs.goal_norm() >= 0.1:
            crossed_at = step
            break
    assert crossed_at is not None, (
        f"A0.2 FAIL: goal_norm={gs.goal_norm():.4f} < 0.1 after 10 updates"
    )
    print(f"A0.2 PASS: goal_norm={gs.goal_norm():.4f} crossed 0.1 at step {crossed_at}")


def test_a0_3_direction_stable():
    """A0.3: seeded z_goal direction aligns with the forced world latent."""
    gs = _make_state()
    z = _forced_z()
    for _ in range(10):
        gs.update(z, benefit_exposure=0.2, drive_level=1.0)
    cos = F.cosine_similarity(gs.z_goal, z, dim=-1).item()
    assert cos >= 0.9, f"A0.3 FAIL: cosine(z_goal, z_world)={cos:.4f} < 0.9"
    print(f"A0.3 PASS: cosine(z_goal, z_world)={cos:.4f} >= 0.9")


def test_a0_4a_decays_without_benefit():
    """A0.4a: with benefit=0 and no floor, z_goal norm strictly decays."""
    gs = _make_state(valence_wanting_floor=0.0)
    z = _forced_z()
    for _ in range(10):
        gs.update(z, benefit_exposure=0.2, drive_level=1.0)
    seeded_norm = gs.goal_norm()
    assert seeded_norm >= 0.1, f"precondition: seeded_norm={seeded_norm:.4f}"
    for _ in range(20):
        gs.update(z, benefit_exposure=0.0, drive_level=0.0)
    decayed_norm = gs.goal_norm()
    assert decayed_norm < seeded_norm, (
        f"A0.4a FAIL: decayed={decayed_norm:.4f} not < seeded={seeded_norm:.4f}"
    )
    print(f"A0.4a PASS: norm decayed {seeded_norm:.4f} -> {decayed_norm:.4f} (benefit=0)")


def test_a0_4b_valence_wanting_floor_holds():
    """A0.4b: valence_wanting_floor prevents the norm from decaying below the floor."""
    floor = 0.05
    gs = _make_state(valence_wanting_floor=floor)
    z = _forced_z()
    for _ in range(10):
        gs.update(z, benefit_exposure=0.2, drive_level=1.0)
    # Drain with benefit=0 for many steps; floor must clamp the norm.
    min_norm = gs.goal_norm()
    for _ in range(200):
        gs.update(z, benefit_exposure=0.0, drive_level=0.0)
        min_norm = min(min_norm, gs.goal_norm())
    # decay (0.995) runs before the floor scale-up, so the post-update floor is
    # the clamp target; allow a one-step decay tolerance below `floor`.
    assert min_norm >= floor * (1.0 - gs.config.decay_goal) - 1e-6, (
        f"A0.4b FAIL: min_norm={min_norm:.5f} dropped below floor={floor}"
    )
    print(f"A0.4b PASS: min_norm={min_norm:.5f} held at floor~{floor}")


def test_a0_5_subthreshold_benefit_does_not_seed():
    """A0.5 (negative control): sub-threshold benefit leaves z_goal at zero."""
    # drive_floor=0.0 + drive_level=0.0 -> multiplier = 1.0, so effective == benefit.
    gs = _make_state(drive_floor=0.0)
    z = _forced_z()
    for _ in range(20):
        gs.update(z, benefit_exposure=0.05, drive_level=0.0)  # 0.05 < 0.1 threshold
    assert gs.goal_norm() < 1e-9, (
        f"A0.5 FAIL: z_goal seeded ({gs.goal_norm():.6f}) on sub-threshold benefit"
    )
    assert not gs.is_active(), "A0.5 FAIL: is_active() true with no supra-threshold benefit"
    print("A0.5 PASS: sub-threshold benefit did not seed z_goal (gate holds)")


if __name__ == "__main__":
    test_a0_1_effective_benefit_crosses_threshold()
    test_a0_2_goal_norm_becomes_nonzero_within_10_steps()
    test_a0_3_direction_stable()
    test_a0_4a_decays_without_benefit()
    test_a0_4b_valence_wanting_floor_holds()
    test_a0_5_subthreshold_benefit_does_not_seed()
    print("\nAll Stage-0 positive-control contracts PASS")
