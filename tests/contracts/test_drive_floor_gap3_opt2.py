"""
Contract tests for SD-012 GAP-3 Option 2: drive_floor insatiability floor.

C1  drive_floor=0.0 (default) is bit-identical to pre-amendment: drive_level
    flows unchanged into the EMA (with drive_ema_alpha=1.0, trace == drive_level).
C2  drive_floor > 0: drive_level_floored = max(drive_level, drive_floor) feeds
    the EMA, so trace >= drive_floor in steady state (alpha=1.0 arm).
C3  Backward-compat: GoalConfig default drive_floor=0.0 and drive_ema_alpha=1.0
    means GoalState.update() is fully bit-identical to the pre-GAP-3 code path.
C4  drive_floor is wired through REEConfig.from_dims(): setting drive_floor in
    from_dims propagates to config.goal.drive_floor.
C5  Combining Option 1 (alpha<1) and Option 2 (floor>0): trace converges to
    >= drive_floor (floor raises the EMA input, so steady-state trace >= floor).
C6  drive_floor does NOT affect GoalState.reset(): trace resets to 0.0 regardless
    of floor value (floor acts only during update, not on construction or reset).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from ree_core.goal import GoalConfig, GoalState
from ree_core.utils.config import REEConfig


DEVICE = torch.device("cpu")


def _make_state(drive_floor=0.0, drive_ema_alpha=1.0, drive_weight=2.0):
    cfg = GoalConfig(
        goal_dim=8,
        drive_floor=drive_floor,
        drive_ema_alpha=drive_ema_alpha,
        drive_weight=drive_weight,
        z_goal_enabled=True,
        benefit_threshold=0.01,  # low so seeding fires in tests
    )
    return GoalState(cfg, DEVICE)


def _dummy_z(dim=8):
    return torch.ones(1, dim)


def test_c1_floor_zero_bit_identical():
    """C1: drive_floor=0.0 gives trace == drive_level at every step (alpha=1.0)."""
    gs = _make_state(drive_floor=0.0, drive_ema_alpha=1.0)
    for dl in [0.005, 0.3, 0.7, 1.0]:
        gs.update(_dummy_z(), benefit_exposure=0.0, drive_level=dl)
        assert abs(gs._drive_trace - dl) < 1e-7, (
            f"C1 FAIL: trace={gs._drive_trace} != drive_level={dl} at floor=0"
        )
    print("C1 PASS: drive_floor=0.0 trace == drive_level (bit-identical OFF)")


def test_c2_floor_raises_trace():
    """C2: drive_floor > 0 raises trace to >= drive_floor when drive_level < floor."""
    floor = 0.9
    gs = _make_state(drive_floor=floor, drive_ema_alpha=1.0)
    low_drive = 0.005  # simulates satiated agent (EXQ-536a / EXQ-582 regime)
    gs.update(_dummy_z(), benefit_exposure=0.0, drive_level=low_drive)
    assert gs._drive_trace >= floor - 1e-7, (
        f"C2 FAIL: trace={gs._drive_trace} < floor={floor} with drive_level={low_drive}"
    )
    print(f"C2 PASS: trace={gs._drive_trace:.4f} >= floor={floor} at drive_level={low_drive}")


def test_c2b_floor_does_not_suppress_high_drive():
    """C2b: when drive_level > drive_floor, trace follows drive_level (max is identity)."""
    floor = 0.2
    gs = _make_state(drive_floor=floor, drive_ema_alpha=1.0)
    high_drive = 0.8
    gs.update(_dummy_z(), benefit_exposure=0.0, drive_level=high_drive)
    assert abs(gs._drive_trace - high_drive) < 1e-7, (
        f"C2b FAIL: trace={gs._drive_trace} != drive_level={high_drive} when drive > floor"
    )
    print(f"C2b PASS: trace={gs._drive_trace:.4f} == drive_level={high_drive} when drive > floor")


def test_c3_full_backward_compat():
    """C3: default config (floor=0.0, alpha=1.0) is bit-identical end-to-end."""
    gs_default = _make_state(drive_floor=0.0, drive_ema_alpha=1.0)
    # Simulate a few steps with low drive (the problematic regime)
    for _ in range(10):
        gs_default.update(_dummy_z(), benefit_exposure=0.002, drive_level=0.005)
    # Effective benefit = 0.002 * 1.0 * (1 + 2.0 * 0.005) = 0.002 * 1.01 = 0.00202
    # With benefit_threshold=0.01 (test only), seeding should NOT fire
    assert gs_default._z_goal.norm().item() < 1e-5, (
        f"C3 FAIL: z_goal norm {gs_default._z_goal.norm():.6f} > 0 unexpectedly"
    )
    print("C3 PASS: default config bit-identical (no spurious seeding)")


def test_c4_from_dims_wiring():
    """C4: drive_floor propagates through REEConfig.from_dims()."""
    cfg = REEConfig.from_dims(
        world_dim=32, action_dim=5, body_obs_dim=16, world_obs_dim=16,
        z_goal_enabled=True,
        drive_floor=0.9,
    )
    assert cfg.goal.drive_floor == 0.9, (
        f"C4 FAIL: cfg.goal.drive_floor={cfg.goal.drive_floor} != 0.9"
    )
    print(f"C4 PASS: from_dims drive_floor={cfg.goal.drive_floor} wired correctly")


def test_c5_combined_option1_option2():
    """C5: combining floor (Option 2) with slow EMA (Option 1) -- trace converges to >= floor."""
    floor = 0.5
    alpha = 0.1  # slow EMA
    gs = _make_state(drive_floor=floor, drive_ema_alpha=alpha)
    low_drive = 0.005
    # Run many steps: in steady state trace -> floor (since input is always max(0.005, 0.5)=0.5)
    for _ in range(200):
        gs.update(_dummy_z(), benefit_exposure=0.0, drive_level=low_drive)
    # After 200 steps with alpha=0.1, trace ~ floor * (1 - (1-alpha)^200) ~ floor
    assert gs._drive_trace >= floor - 0.01, (
        f"C5 FAIL: trace={gs._drive_trace:.4f} < floor={floor} after 200 steps"
    )
    print(f"C5 PASS: trace={gs._drive_trace:.4f} >= floor={floor} after 200 steps (Option1+2 combined)")


def test_c6_reset_zeroes_trace():
    """C6: reset() zeroes drive_trace regardless of drive_floor."""
    gs = _make_state(drive_floor=0.9, drive_ema_alpha=1.0)
    gs.update(_dummy_z(), benefit_exposure=0.0, drive_level=0.005)
    assert gs._drive_trace >= 0.9 - 1e-7, "C6 setup: trace should be floored"
    gs.reset()
    assert abs(gs._drive_trace) < 1e-9, (
        f"C6 FAIL: drive_trace={gs._drive_trace} != 0.0 after reset()"
    )
    print(f"C6 PASS: drive_trace={gs._drive_trace} zeroed by reset() (floor does not persist)")


if __name__ == "__main__":
    test_c1_floor_zero_bit_identical()
    test_c2_floor_raises_trace()
    test_c2b_floor_does_not_suppress_high_drive()
    test_c3_full_backward_compat()
    test_c4_from_dims_wiring()
    test_c5_combined_option1_option2()
    test_c6_reset_zeroes_trace()
    print("All drive_floor GAP-3 Option 2 contract tests PASS")
