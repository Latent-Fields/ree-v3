#!/opt/local/bin/python3
"""
Smoke test for infant_curriculum Phase 3 destabilizing pressure (IGW-20260601-023).

Verifies that Phase 3 env_kwargs include the expected environmental dynamics
features (multi_source_dynamics, interoceptive_noise, accelerated drift) that
provide post-crystallization destabilizing pressure for INV-074 testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add experiments/ to path for infant_curriculum import.
EXPERIMENTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS_DIR))

from infant_curriculum import InfantCurriculumScheduler


def test_phase3_destabilizing_pressure():
    """Phase 3 env_kwargs should include destabilizing dynamics."""
    sched = InfantCurriculumScheduler(grid_size=12)

    # Get Phase 3 kwargs directly.
    phase3_kwargs = sched.env_kwargs(phase=3)

    # Verify multi-source dynamics enabled.
    assert phase3_kwargs.get("multi_source_dynamics_enabled") is True, \
        "Phase 3 must enable multi_source_dynamics_enabled"
    assert phase3_kwargs.get("weather_field_enabled") is True, \
        "Phase 3 must enable weather_field_enabled"
    assert phase3_kwargs.get("transient_events_enabled") is True, \
        "Phase 3 must enable transient_events_enabled"
    assert phase3_kwargs.get("background_drift_enabled") is True, \
        "Phase 3 must enable background_drift_enabled"

    # Verify interoceptive noise enabled.
    assert phase3_kwargs.get("interoceptive_noise_enabled") is True, \
        "Phase 3 must enable interoceptive_noise_enabled"
    assert phase3_kwargs.get("interoceptive_noise_scale", 0) > 0, \
        "Phase 3 interoceptive_noise_scale must be > 0"

    # Verify accelerated drift.
    assert phase3_kwargs.get("env_drift_interval", 999) < 5, \
        "Phase 3 env_drift_interval should be accelerated (< 5)"
    assert phase3_kwargs.get("env_drift_prob", 0) >= 0.4, \
        "Phase 3 env_drift_prob should be elevated (>= 0.4)"

    print("PASS: Phase 3 destabilizing pressure kwargs verified")
    return True


def test_phase2_no_extra_dynamics():
    """Phase 2 should NOT have the Phase-3-specific dynamics."""
    sched = InfantCurriculumScheduler(grid_size=12)

    phase2_kwargs = sched.env_kwargs(phase=2)

    # Phase 2 should not have multi-source dynamics.
    assert phase2_kwargs.get("multi_source_dynamics_enabled") is not True, \
        "Phase 2 should not enable multi_source_dynamics (Phase 3 only)"
    assert phase2_kwargs.get("interoceptive_noise_enabled") is not True, \
        "Phase 2 should not enable interoceptive_noise (Phase 3 only)"

    print("PASS: Phase 2 correctly excludes Phase-3-only dynamics")
    return True


def test_phase_progression():
    """Verify scheduler progresses through phases with expected kwargs."""
    sched = InfantCurriculumScheduler(grid_size=12)

    # Initially Phase 0.
    assert sched.current_phase == 0
    phase0_kwargs = sched.env_kwargs()
    assert phase0_kwargs.get("microhabitat_enabled") is False
    assert phase0_kwargs.get("harm_gradient_enabled") is False

    # Advance to Phase 1 (requires ep >= 100).
    sched.update(episode=100, h_pos=1.0)
    if sched.current_phase >= 1:
        phase1_kwargs = sched.env_kwargs()
        assert phase1_kwargs.get("harm_gradient_enabled") is True
        assert phase1_kwargs.get("microhabitat_enabled") is False

    # Advance to Phase 2 (requires ep >= 500).
    sched.update(episode=500, z_goal_norm=0.35, benefit_contacts=10)
    if sched.current_phase >= 2:
        phase2_kwargs = sched.env_kwargs()
        assert phase2_kwargs.get("microhabitat_enabled") is True
        assert phase2_kwargs.get("multi_source_dynamics_enabled") is not True

    # Advance to Phase 3 (requires ep >= 2000).
    sched.update(episode=2000, residue_coverage_pct=0.20)
    assert sched.current_phase >= 3
    phase3_kwargs = sched.env_kwargs()
    assert phase3_kwargs.get("multi_source_dynamics_enabled") is True

    print("PASS: Phase progression and kwargs evolution verified")
    return True


if __name__ == "__main__":
    try:
        test_phase3_destabilizing_pressure()
        test_phase2_no_extra_dynamics()
        test_phase_progression()
        print("\nAll tests PASSED")
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        sys.exit(1)
