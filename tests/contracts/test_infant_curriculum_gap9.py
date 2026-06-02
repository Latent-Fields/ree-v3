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


def test_c6_env_kwargs_phase3_enriches_phase2():
    """Phase 3 SUPERSETS Phase 2 and adds the documented env-dynamics keys.

    The 2026-06-01 IGW-20260601-023 landing (commit 15f32c82c7,
    test_bed_enrichment_crystallization_necessity) intentionally made Phase 3
    enable SD-047 multi_source_dynamics + SD-048 interoceptive_noise +
    accelerated env_drift as destabilizing pressure for the INV-074
    crystallization-necessity test. Phase 3 env_kwargs therefore legitimately
    differs from Phase 2: it carries every Phase-2 agent-side key unchanged AND
    adds the env-dynamics keys. The pre-IGW-023 contract (Phase 3 == Phase 2)
    is stale; this contract asserts the intended enrichment delta instead.
    """
    s = InfantCurriculumScheduler()
    kw2 = s.env_kwargs(phase=2)
    kw3 = s.env_kwargs(phase=3)

    # Phase 3 is a strict superset of Phase 2: every Phase-2 key is present in
    # Phase 3 with an identical value (agent-side curriculum settings carry
    # forward unchanged).
    for key, val in kw2.items():
        assert key in kw3, f"Phase 3 must carry forward Phase-2 key {key!r}"
        assert kw3[key] == val, (
            f"Phase 3 must preserve Phase-2 value for {key!r}: "
            f"expected {val!r}, got {kw3[key]!r}"
        )

    # Phase 3 adds the documented env-dynamics enrichment keys (SD-047
    # multi-source dynamics + SD-048 interoceptive noise + accelerated drift).
    enrichment_keys = {
        "multi_source_dynamics_enabled",
        "multi_source_intensity_scale",
        "weather_field_enabled",
        "transient_events_enabled",
        "background_drift_enabled",
        "n_drift_sources",
        "interoceptive_noise_enabled",
        "interoceptive_noise_scale",
        "env_drift_interval",
        "env_drift_prob",
    }
    added_keys = set(kw3) - set(kw2)
    assert added_keys == enrichment_keys, (
        "Phase 3 enrichment delta drifted from the IGW-023 contract: "
        f"expected added keys {sorted(enrichment_keys)}, got {sorted(added_keys)}"
    )

    # The enrichment substrates must actually be ON in Phase 3 (the whole point
    # of the destabilizing-pressure landing).
    assert kw3["multi_source_dynamics_enabled"] is True
    assert kw3["interoceptive_noise_enabled"] is True
    assert kw3["weather_field_enabled"] is True
    assert kw3["transient_events_enabled"] is True
    assert kw3["background_drift_enabled"] is True


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

def test_c11_h_pos_default_within_observed_band():
    """H_POS_FRAC_OF_MAX default must produce a threshold inside the observed
    rolling-mean H_pos band 0.03-1.08 (failure_autopsy_V3-EXQ-591_2026-05-27.md
    section 7). Pre-recalibration 0.70 -> 0.70 * ln(144) ~ 3.48 was outside
    the band and never crossed across 2000 episodes / 5 seeds. Post-recalibration
    2026-05-31: 0.20 -> 0.20 * ln(144) ~ 0.99 sits inside the band."""
    h_max = math.log(12 ** 2)
    threshold = H_POS_FRAC_OF_MAX * h_max
    OBSERVED_MIN, OBSERVED_MAX = 0.03, 1.08
    assert threshold <= OBSERVED_MAX, (
        f"H_POS_FRAC_OF_MAX={H_POS_FRAC_OF_MAX} yields threshold={threshold:.4f}, "
        f"above observed max H_pos {OBSERVED_MAX} from V3-EXQ-591 autopsy. "
        f"Threshold must sit inside the achievable band."
    )
    assert threshold > 0.0, (
        f"H_POS_FRAC_OF_MAX={H_POS_FRAC_OF_MAX} yields zero threshold; would "
        f"defeat the H_pos gate entirely."
    )


def test_c11_synthetic_p0_trajectory_advances_phase():
    """GAP-C prereq 3 smoke: a synthetic P0 H_pos trajectory drawn from the
    middle of the observed-band (~0.99 = 0.20 * ln(144)) must advance Phase
    0 -> 1 at the boundary tick. Regression guard against future recalibration
    drift back above the observed ceiling."""
    s = make_sched(grid_size=12)
    h_max = math.log(12 ** 2)
    # Simulate a P0 random-policy trajectory where rolling-mean H_pos saturates
    # near the top of the observed band (autopsy max 1.08; threshold at
    # 0.20 * ln(144) ~ 0.994). 1.05 sits inside the band with margin above the
    # threshold.
    p0_h_pos = 1.05
    # Ramp up to the boundary at ep 100; gate should NOT fire before then.
    for ep in (10, 50, 99):
        p = s.update(ep, h_pos=p0_h_pos)
        assert p == 0, f"Phase advanced before ep 100 (ep={ep}, p={p})"
    # Boundary tick at the achievable signal magnitude must advance the phase.
    p = s.update(100, h_pos=p0_h_pos)
    assert p == 1, (
        f"Phase 0->1 gate failed to fire at ep=100 with H_pos=0.99 (achievable "
        f"per V3-EXQ-591 autopsy band 0.03-1.08); H_POS_FRAC_OF_MAX="
        f"{H_POS_FRAC_OF_MAX} threshold={H_POS_FRAC_OF_MAX * h_max:.4f}"
    )
    assert s.phase_changed


def test_c11_synthetic_p0_trajectory_marginal_clearance():
    """The autopsy quotes 0.99 as 'reachable' at the 0.20 fraction-of-max
    calibration. Confirm the gate clears with H_pos exactly at 0.99 (no
    floating-point margin pathology) and stays blocked just below."""
    s_blocked = make_sched(grid_size=12)
    h_max = math.log(12 ** 2)
    threshold = H_POS_FRAC_OF_MAX * h_max  # ~ 0.9939 at 0.20 / size=12
    # Just below threshold: stay in Phase 0.
    p = s_blocked.update(150, h_pos=threshold * 0.99)
    assert p == 0, "Gate must block H_pos strictly below threshold."
    # At threshold (or one tiny epsilon above): advance.
    s_unblocked = make_sched(grid_size=12)
    p = s_unblocked.update(150, h_pos=threshold * 1.001)
    assert p == 1, (
        f"Gate must admit H_pos at or just above threshold {threshold:.4f}; "
        f"got phase {p} at H_pos={threshold * 1.001:.4f}"
    )


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
