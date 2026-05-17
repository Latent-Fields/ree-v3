"""
infant_substrate:GAP-9 contract tests for InfantCurriculumScheduler.

Covers:
  C1  Fresh scheduler starts at phase 0, phase_changed=False.
  C2  Hard episode-count transitions (no telemetry): 0->1 at ep 100,
      1->2 at ep 500, 2->3 at ep 2000.
  C3  Telemetry gate blocks phase 0->1 when H_pos below threshold even
      after episode count passes.
  C4  Telemetry gate blocks phase 1->2 when z_goal_norm below threshold.
  C5  Phase only advances -- never retreats even when metrics drop.
  C6  env_kwargs correct per phase: OFF for phase 0; harm+transient for
      phase 1; +microhabitat for phase 2+.
  C7  config_overrides correct per phase (spot checks key fields).
  C8  phase_changed True only on the transition step, False otherwise.
  C9  Full walk 0->1->2->3 via hard counts; phase_summary keys present.
  C10 benefit_contacts window gate: phase 1->2 blocked when recent
      contacts below threshold.
"""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from infant_curriculum import (
    InfantCurriculumScheduler,
    PHASE_EP_MIN,
    H_POS_FRAC_OF_MAX,
    Z_GOAL_THRESHOLD,
    BENEFIT_CONTACTS_REQUIRED,
    RESIDUE_COVERAGE_THRESHOLD,
    BENEFIT_CONTACTS_WINDOW,
)


def make_sched(grid_size=12):
    return InfantCurriculumScheduler(grid_size=grid_size)


# ------------------------------------------------------------------
# C1: Fresh start
# ------------------------------------------------------------------

def test_c1_fresh_start():
    s = make_sched()
    assert s.current_phase == 0
    assert s.phase_changed is False
    assert s.episode == -1


# ------------------------------------------------------------------
# C2: Hard episode-count transitions (no telemetry)
# ------------------------------------------------------------------

def test_c2_hard_transition_0_to_1():
    s = make_sched()
    # Still phase 0 just before the boundary.
    p = s.update(99)
    assert p == 0, f"Expected phase 0 at ep 99, got {p}"
    assert not s.phase_changed
    # Crosses the boundary -- no telemetry means hard count governs.
    p = s.update(100)
    assert p == 1, f"Expected phase 1 at ep 100, got {p}"
    assert s.phase_changed


def test_c2_hard_transition_1_to_2():
    s = make_sched()
    # Advance past phase 0.
    s.update(100)
    assert s.current_phase == 1
    p = s.update(499)
    assert p == 1
    p = s.update(500)
    assert p == 2, f"Expected phase 2 at ep 500, got {p}"
    assert s.phase_changed


def test_c2_hard_transition_2_to_3():
    s = make_sched()
    s.update(100)   # -> 1
    s.update(500)   # -> 2
    assert s.current_phase == 2
    p = s.update(1999)
    assert p == 2
    p = s.update(2000)
    assert p == 3, f"Expected phase 3 at ep 2000, got {p}"
    assert s.phase_changed


# ------------------------------------------------------------------
# C3: H_pos gate blocks phase 0->1
# ------------------------------------------------------------------

def test_c3_h_pos_gate_blocks():
    s = make_sched(grid_size=12)
    h_max = math.log(12 ** 2)
    low_h = H_POS_FRAC_OF_MAX * h_max * 0.5  # well below threshold
    # Even at ep >= 100, low H_pos keeps us in phase 0.
    p = s.update(150, h_pos=low_h)
    assert p == 0, f"Expected phase 0 with low H_pos, got {p}"
    # Raise H_pos above threshold -> transitions.
    high_h = H_POS_FRAC_OF_MAX * h_max * 1.05
    p = s.update(151, h_pos=high_h)
    assert p == 1, f"Expected phase 1 after H_pos threshold met, got {p}"
    assert s.phase_changed


def test_c3_h_pos_gate_below_ep_min():
    """H_pos above threshold is irrelevant before ep 100."""
    s = make_sched(grid_size=12)
    h_max = math.log(12 ** 2)
    high_h = h_max  # maximum possible
    p = s.update(50, h_pos=high_h)
    assert p == 0, "Phase should stay 0 before ep 100 regardless of H_pos"


# ------------------------------------------------------------------
# C4: z_goal gate blocks phase 1->2
# ------------------------------------------------------------------

def test_c4_z_goal_gate_blocks():
    s = make_sched()
    s.update(100)  # -> phase 1
    # z_goal below threshold at ep 500.
    p = s.update(500, z_goal_norm=Z_GOAL_THRESHOLD * 0.5)
    assert p == 1, f"Expected phase 1 with low z_goal_norm, got {p}"
    # z_goal meets threshold -> transitions.
    p = s.update(501, z_goal_norm=Z_GOAL_THRESHOLD + 0.05)
    assert p == 2, f"Expected phase 2 after z_goal threshold met, got {p}"


# ------------------------------------------------------------------
# C5: Phase only advances, never retreats
# ------------------------------------------------------------------

def test_c5_no_retreat():
    s = make_sched()
    s.update(100)   # -> 1
    s.update(500)   # -> 2
    assert s.current_phase == 2
    # Provide metrics that would have kept us in phase 0 or 1.
    h_max = math.log(12 ** 2)
    low_h = 0.0
    p = s.update(501, h_pos=low_h, z_goal_norm=0.0, benefit_contacts=0,
                 residue_coverage_pct=0.0)
    assert p == 2, "Phase should never retreat"
    assert not s.phase_changed


# ------------------------------------------------------------------
# C6: env_kwargs correct per phase
# ------------------------------------------------------------------

def test_c6_env_kwargs_phase0():
    s = make_sched()
    kw = s.env_kwargs(phase=0)
    assert kw["harm_gradient_enabled"] is False
    assert kw["transient_benefit_enabled"] is False
    assert kw["microhabitat_enabled"] is False


def test_c6_env_kwargs_phase1():
    kw = InfantCurriculumScheduler().env_kwargs(phase=1)
    assert kw["harm_gradient_enabled"] is True
    assert kw["transient_benefit_enabled"] is True
    assert kw["microhabitat_enabled"] is False
    assert kw["harm_gradient_scale"] == 0.15


def test_c6_env_kwargs_phase2():
    kw = InfantCurriculumScheduler().env_kwargs(phase=2)
    assert kw["harm_gradient_enabled"] is True
    assert kw["transient_benefit_enabled"] is True
    assert kw["microhabitat_enabled"] is True
    assert kw["harm_gradient_scale"] == 0.30


def test_c6_env_kwargs_phase3_same_as_phase2():
    s = InfantCurriculumScheduler()
    assert s.env_kwargs(phase=3) == s.env_kwargs(phase=2), (
        "Phase 3 env_kwargs must equal phase 2 (agent-side config changes only)"
    )


# ------------------------------------------------------------------
# C7: config_overrides correct per phase
# ------------------------------------------------------------------

def test_c7_config_overrides():
    s = InfantCurriculumScheduler()
    ov0 = s.config_overrides(phase=0)
    ov1 = s.config_overrides(phase=1)
    ov2 = s.config_overrides(phase=2)
    ov3 = s.config_overrides(phase=3)
    # residue_scale_factor must increase phase over phase.
    assert ov0["residue_scale_factor"] == 0.0
    assert ov1["residue_scale_factor"] == 0.05
    assert ov2["residue_scale_factor"] == 0.10
    assert ov3["residue_scale_factor"] == 0.15
    # offline_integration_frequency must increase phase over phase.
    assert ov0["offline_integration_frequency"] < ov1["offline_integration_frequency"]
    assert ov1["offline_integration_frequency"] < ov2["offline_integration_frequency"]
    assert ov2["offline_integration_frequency"] < ov3["offline_integration_frequency"]
    # novelty_bonus_weight must be positive in all phases.
    for ov in (ov0, ov1, ov2, ov3):
        assert ov["novelty_bonus_weight"] > 0


# ------------------------------------------------------------------
# C8: phase_changed flag
# ------------------------------------------------------------------

def test_c8_phase_changed_flag():
    s = make_sched()
    # Not changed on a plain update before the boundary.
    s.update(50)
    assert not s.phase_changed
    # Changed exactly on transition.
    s.update(100)
    assert s.phase_changed
    # Not changed on subsequent steps within the same phase.
    s.update(101)
    assert not s.phase_changed
    s.update(200)
    assert not s.phase_changed


# ------------------------------------------------------------------
# C9: Full walk + phase_summary
# ------------------------------------------------------------------

def test_c9_full_walk_hard_counts():
    s = make_sched()
    phases_seen = []
    changes = []
    for ep in range(2100):
        s.update(ep)
        phases_seen.append(s.current_phase)
        if s.phase_changed:
            changes.append((ep, s.current_phase))
    # Three transitions must occur.
    assert len(changes) == 3, f"Expected 3 transitions, got {changes}"
    assert changes[0] == (100, 1)
    assert changes[1] == (500, 2)
    assert changes[2] == (2000, 3)
    # phase_summary contains required keys.
    summary = s.phase_summary()
    assert "current_phase" in summary
    assert "episode" in summary
    assert "phase_changed" in summary
    assert "benefit_window_sum" in summary


# ------------------------------------------------------------------
# C10: benefit_contacts window gate
# ------------------------------------------------------------------

def test_c10_benefit_contacts_gate():
    s = make_sched()
    s.update(100)  # -> phase 1
    # Provide z_goal above threshold but zero contacts -> stays in phase 1.
    p = s.update(500, z_goal_norm=Z_GOAL_THRESHOLD + 0.1, benefit_contacts=0)
    assert p == 1, f"Expected phase 1 when benefit_contacts=0, got {p}"
    # One more episode with sufficient contacts in window.
    for _ in range(BENEFIT_CONTACTS_REQUIRED):
        s.update(501, z_goal_norm=Z_GOAL_THRESHOLD + 0.1, benefit_contacts=1)
    p = s.update(502, z_goal_norm=Z_GOAL_THRESHOLD + 0.1, benefit_contacts=0)
    assert p == 2, f"Expected phase 2 after contacts accumulated, got {p}"
