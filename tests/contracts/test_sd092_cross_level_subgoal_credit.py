"""
SD-092 cross-level subgoal credit contracts (MECH-427 maintenance-direction /
MECH-428 formation-direction).

Purpose
-------
`GoalState` previously held exactly one persistent attractor (`_z_goal`). This
SD adds an optional second, PARENT (superordinate) attractor
(`_z_goal_parent`) and a discrete-event credit channel
(`credit_subgoal_attainment`) that pulls the parent toward an attained
subgoal's own representation -- the cross-LEVEL complement to MECH-217's
within-level `backward_credit_sweep` / `spread_reverse_replay_wanting`.

These tests exercise GoalState directly (no agent loop, no environment),
exactly in the style of test_goalstate_forced_seed_positive_control.py --
forced synthetic inputs, no torch grad, standalone.

Acceptance
  R1   Flag OFF: no parent tensor is allocated; credit_subgoal_attainment is a
       true no-op ({}); base _z_goal/goal_norm()/update()/reset() behavior is
       exactly the pre-existing single-level behavior (the critical
       regression to prevent -- goal.py is read by many consumers).
  R2   Flag ON, zero credit calls: parent_goal_norm() == 0.0 (nothing to
       credit yet).
  R3   Flag ON, MECH-428 bootstrap shape: repeated credit calls from a
       near-zero parent raise parent_goal_norm() measurably above the
       no-credit control, direction-aligned with the credited child
       representation.
  R4   Flag ON, MECH-427 maintenance shape: an ALREADY-SEEDED parent
       (Bandura & Schunk's "distal goal maintained through subgoal
       attainment") is reinforced further by an additional credit event,
       exceeding a matched no-further-credit control over the same number of
       decay ticks.
  R5   Credit gating: credit <= 0, or below subgoal_credit_min, applies no
       pull (n_subgoal_credits / parent_goal_norm unchanged, credit_applied
       == 0.0).
  R6   Parent decay: without credit events, parent_goal_norm() strictly
       decays across update() ticks (mirrors the base _z_goal decay).
  R7   Independence: crediting the parent never perturbs the child-level
       `_z_goal` / goal_norm() / goal_proximity() state.
  R8   reset() clears the parent attractor + credit counter (per-episode
       state), leaving the flag itself untouched.
  R9   state_dict()/load_state_dict() round-trip the parent attractor;
       loading a PRE-SD-092 checkpoint dict (missing the new keys) leaves the
       parent unallocated rather than raising.
  R10  with_injection() (the MECH-188 lightweight view) carries the parent
       state through without raising AttributeError.
  R11  Lazy allocation: flipping use_hierarchical_goal_credit on AFTER
       construction still allocates the parent attractor on first credit
       call.

Run: /opt/local/bin/python3 -m pytest tests/contracts/test_sd092_cross_level_subgoal_credit.py -q
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn.functional as F

from ree_core.goal import GoalConfig, GoalState

DEVICE = torch.device("cpu")
GOAL_DIM = 8


def _forced_z(dim=GOAL_DIM, seed_val=1.0):
    """Fixed unit-norm latent direction to credit toward."""
    z = torch.full((1, dim), seed_val)
    return z / z.norm()


def _make_state(use_hierarchical_goal_credit=False, **overrides):
    kwargs = dict(
        goal_dim=GOAL_DIM,
        alpha_goal=0.05,
        decay_goal=0.005,
        benefit_threshold=0.1,
        drive_weight=2.0,
        z_goal_enabled=True,
        use_hierarchical_goal_credit=use_hierarchical_goal_credit,
    )
    kwargs.update(overrides)
    cfg = GoalConfig(**kwargs)
    return GoalState(cfg, DEVICE)


# --------------------------------------------------------------------- #
# R1: flag OFF -- true no-op, base behavior untouched                    #
# --------------------------------------------------------------------- #

def test_r1_flag_off_no_parent_tensor_allocated():
    gs = _make_state(use_hierarchical_goal_credit=False)
    assert gs._z_goal_parent is None, "R1 FAIL: parent tensor allocated with flag off"


def test_r1_flag_off_credit_call_is_true_noop():
    gs = _make_state(use_hierarchical_goal_credit=False)
    result = gs.credit_subgoal_attainment(_forced_z(), credit=1.0)
    assert result == {}, f"R1 FAIL: expected {{}} no-op dict, got {result}"
    assert gs._z_goal_parent is None, "R1 FAIL: credit call allocated parent with flag off"
    assert gs.parent_goal_norm() == 0.0
    assert gs.parent_is_active() is False


def test_r1_flag_off_base_zgoal_behavior_unchanged():
    """Regression guard: base single-level seeding/decay is bit-identical to
    the pre-SD-092 behavior (values match test_goalstate_forced_seed_positive_control.py)."""
    gs = _make_state(use_hierarchical_goal_credit=False, drive_floor=0.9, drive_ema_alpha=1.0)
    z = _forced_z()
    for _ in range(10):
        gs.update(z, benefit_exposure=0.2, drive_level=1.0)
    assert gs.goal_norm() >= 0.1, f"R1 FAIL: goal_norm={gs.goal_norm():.4f} regressed"
    cos = F.cosine_similarity(gs.z_goal, z, dim=-1).item()
    assert cos >= 0.9, f"R1 FAIL: direction stability regressed, cos={cos:.4f}"
    seeded = gs.goal_norm()
    for _ in range(20):
        gs.update(z, benefit_exposure=0.0, drive_level=0.0)
    assert gs.goal_norm() < seeded, "R1 FAIL: base decay regressed"


def test_r1_flag_off_update_never_touches_parent_path():
    """update() must not enter the parent-decay branch at all when off --
    verified indirectly: no exception, and the attribute stays None across
    many ticks (would be caught immediately if the guard were dropped)."""
    gs = _make_state(use_hierarchical_goal_credit=False)
    z = _forced_z()
    for _ in range(50):
        gs.update(z, benefit_exposure=0.2, drive_level=1.0)
    assert gs._z_goal_parent is None


# --------------------------------------------------------------------- #
# R2: flag ON, no credit yet                                             #
# --------------------------------------------------------------------- #

def test_r2_flag_on_zero_credits_parent_norm_zero():
    gs = _make_state(use_hierarchical_goal_credit=True)
    assert gs._z_goal_parent is not None, "R2 FAIL: parent not allocated with flag on"
    assert gs.parent_goal_norm() == 0.0
    assert gs.parent_is_active() is False


# --------------------------------------------------------------------- #
# R3: MECH-428 bootstrap shape                                           #
# --------------------------------------------------------------------- #

def test_r3_bootstrap_raises_parent_norm_above_no_credit_control():
    z_child = _forced_z()

    gs_bootstrap = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(10):
        gs_bootstrap.credit_subgoal_attainment(z_child, credit=1.0)
    bootstrap_norm = gs_bootstrap.parent_goal_norm()

    gs_control = _make_state(use_hierarchical_goal_credit=True)
    # matched number of update() ticks (decay-only), NO credit calls
    for _ in range(10):
        gs_control.update(torch.zeros(1, GOAL_DIM), benefit_exposure=0.0, drive_level=0.0)
    control_norm = gs_control.parent_goal_norm()

    assert bootstrap_norm > control_norm, (
        f"R3 FAIL: bootstrap parent_norm={bootstrap_norm:.4f} not above "
        f"no-credit control={control_norm:.4f}"
    )
    assert control_norm == 0.0, "R3 FAIL: no-credit control should stay exactly at 0"
    assert bootstrap_norm > 0.1, f"R3 FAIL: bootstrap did not seed materially ({bootstrap_norm:.4f})"


def test_r3_bootstrap_direction_aligned_with_credited_child():
    z_child = _forced_z()
    gs = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(10):
        gs.credit_subgoal_attainment(z_child, credit=1.0)
    cos = F.cosine_similarity(gs.z_goal_parent, z_child, dim=-1).item()
    assert cos >= 0.9, f"R3 FAIL: parent direction not aligned with credited child, cos={cos:.4f}"


# --------------------------------------------------------------------- #
# R4: MECH-427 maintenance shape                                         #
# --------------------------------------------------------------------- #

def test_r4_maintenance_reinforces_already_seeded_parent():
    z_child = _forced_z()

    gs_reinforced = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(5):
        gs_reinforced.credit_subgoal_attainment(z_child, credit=1.0)
    seeded_norm = gs_reinforced.parent_goal_norm()
    assert seeded_norm > 0.0, "precondition: parent must already be seeded"

    gs_maintained = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(5):
        gs_maintained.credit_subgoal_attainment(z_child, credit=1.0)
    # one further maintenance credit event (the MECH-427 case)
    gs_maintained.credit_subgoal_attainment(z_child, credit=1.0)
    maintained_norm = gs_maintained.parent_goal_norm()

    gs_unmaintained = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(5):
        gs_unmaintained.credit_subgoal_attainment(z_child, credit=1.0)
    # matched tick with NO further credit -- only decay
    gs_unmaintained.update(torch.zeros(1, GOAL_DIM), benefit_exposure=0.0, drive_level=0.0)
    unmaintained_norm = gs_unmaintained.parent_goal_norm()

    assert maintained_norm > unmaintained_norm, (
        f"R4 FAIL: maintained parent_norm={maintained_norm:.4f} not above "
        f"unmaintained (decay-only) control={unmaintained_norm:.4f}"
    )


# --------------------------------------------------------------------- #
# R5: credit gating                                                      #
# --------------------------------------------------------------------- #

def test_r5_zero_or_negative_credit_is_noop():
    gs = _make_state(use_hierarchical_goal_credit=True)
    result = gs.credit_subgoal_attainment(_forced_z(), credit=0.0)
    assert result["credit_applied"] == 0.0
    assert gs.parent_goal_norm() == 0.0
    assert gs._n_subgoal_credits == 0

    result_neg = gs.credit_subgoal_attainment(_forced_z(), credit=-1.0)
    assert result_neg["credit_applied"] == 0.0
    assert gs.parent_goal_norm() == 0.0


def test_r5_below_subgoal_credit_min_is_noop():
    gs = _make_state(use_hierarchical_goal_credit=True, subgoal_credit_min=0.5)
    result = gs.credit_subgoal_attainment(_forced_z(), credit=0.3)
    assert result["credit_applied"] == 0.0, "R5 FAIL: sub-minimum credit should not apply"
    assert gs.parent_goal_norm() == 0.0

    result_ok = gs.credit_subgoal_attainment(_forced_z(), credit=0.6)
    assert result_ok["credit_applied"] > 0.0, "R5 FAIL: above-minimum credit should apply"
    assert gs.parent_goal_norm() > 0.0


# --------------------------------------------------------------------- #
# R6: parent decay                                                       #
# --------------------------------------------------------------------- #

def test_r6_parent_decays_without_credit_events():
    gs = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(5):
        gs.credit_subgoal_attainment(_forced_z(), credit=1.0)
    seeded = gs.parent_goal_norm()
    assert seeded > 0.0
    for _ in range(30):
        gs.update(torch.zeros(1, GOAL_DIM), benefit_exposure=0.0, drive_level=0.0)
    decayed = gs.parent_goal_norm()
    assert decayed < seeded, f"R6 FAIL: parent_norm did not decay ({seeded:.4f} -> {decayed:.4f})"


# --------------------------------------------------------------------- #
# R7: independence from child-level state                                #
# --------------------------------------------------------------------- #

def test_r7_credit_does_not_perturb_child_zgoal():
    gs = _make_state(use_hierarchical_goal_credit=True)
    z = _forced_z()
    # seed the child level normally
    for _ in range(10):
        gs.update(z, benefit_exposure=0.2, drive_level=1.0)
    child_norm_before = gs.goal_norm()
    child_vec_before = gs.z_goal.clone()

    # credit the parent repeatedly using a DIFFERENT child representation
    other_child = _forced_z(seed_val=-1.0)
    for _ in range(10):
        gs.credit_subgoal_attainment(other_child, credit=1.0)

    assert gs.goal_norm() == child_norm_before, "R7 FAIL: crediting parent changed child goal_norm"
    assert torch.equal(gs.z_goal, child_vec_before), "R7 FAIL: crediting parent mutated _z_goal tensor"


# --------------------------------------------------------------------- #
# R8: reset() clears parent state                                        #
# --------------------------------------------------------------------- #

def test_r8_reset_clears_parent_and_counters():
    gs = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(5):
        gs.credit_subgoal_attainment(_forced_z(), credit=1.0)
    assert gs.parent_goal_norm() > 0.0
    assert gs._n_subgoal_credits == 5

    gs.reset()
    assert gs.parent_goal_norm() == 0.0, "R8 FAIL: reset() did not clear parent attractor"
    assert gs._n_subgoal_credits == 0, "R8 FAIL: reset() did not clear credit counter"
    assert gs._parent_goal_norm_peak == 0.0, "R8 FAIL: reset() did not clear parent peak"
    # flag itself must survive reset (it's config, not per-episode state)
    assert gs.config.use_hierarchical_goal_credit is True


# --------------------------------------------------------------------- #
# R9: state_dict / load_state_dict round-trip + backward compat          #
# --------------------------------------------------------------------- #

def test_r9_state_dict_round_trip_preserves_parent():
    gs = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(5):
        gs.credit_subgoal_attainment(_forced_z(), credit=1.0)
    d = gs.state_dict()
    assert d["z_goal_parent"] is not None
    assert d["n_subgoal_credits"] == 5

    gs2 = _make_state(use_hierarchical_goal_credit=True)
    gs2.load_state_dict(d)
    assert gs2.parent_goal_norm() == gs.parent_goal_norm()
    assert gs2._n_subgoal_credits == 5


def test_r9_state_dict_flag_off_serializes_none():
    gs = _make_state(use_hierarchical_goal_credit=False)
    d = gs.state_dict()
    assert d["z_goal_parent"] is None


def test_r9_load_pre_sd092_checkpoint_leaves_parent_unallocated():
    """A checkpoint dict missing the new SD-092 keys entirely (the
    pre-SD-092 shape) must load without raising, with the parent left
    unallocated."""
    gs = _make_state(use_hierarchical_goal_credit=True)
    legacy_dict = {"z_goal": torch.zeros(1, GOAL_DIM), "goal_norm_peak": 0.0}
    gs.load_state_dict(legacy_dict)  # must not raise
    assert gs._z_goal_parent is None
    assert gs.parent_goal_norm() == 0.0
    assert gs._n_subgoal_credits == 0


# --------------------------------------------------------------------- #
# R10: with_injection carries parent state                               #
# --------------------------------------------------------------------- #

def test_r10_with_injection_carries_parent_state_without_raising():
    gs = _make_state(use_hierarchical_goal_credit=True)
    for _ in range(3):
        gs.credit_subgoal_attainment(_forced_z(), credit=1.0)
    injected = gs.with_injection(0.3)
    # must not raise AttributeError
    assert injected.parent_goal_norm() == gs.parent_goal_norm()
    assert injected._n_subgoal_credits == gs._n_subgoal_credits


def test_r10_with_injection_flag_off_still_works():
    gs = _make_state(use_hierarchical_goal_credit=False)
    injected = gs.with_injection(0.3)
    assert injected.parent_goal_norm() == 0.0


# --------------------------------------------------------------------- #
# R11: lazy allocation after construction                                #
# --------------------------------------------------------------------- #

def test_r11_lazy_allocation_when_flag_flipped_post_construction():
    gs = _make_state(use_hierarchical_goal_credit=False)
    assert gs._z_goal_parent is None
    # flip the flag on the live config object (mirrors a caller that
    # constructs GoalState once but toggles the ablation flag later)
    gs.config.use_hierarchical_goal_credit = True
    result = gs.credit_subgoal_attainment(_forced_z(), credit=1.0)
    assert gs._z_goal_parent is not None, "R11 FAIL: parent not lazily allocated"
    assert result["parent_goal_norm"] > 0.0


if __name__ == "__main__":
    test_r1_flag_off_no_parent_tensor_allocated()
    test_r1_flag_off_credit_call_is_true_noop()
    test_r1_flag_off_base_zgoal_behavior_unchanged()
    test_r1_flag_off_update_never_touches_parent_path()
    test_r2_flag_on_zero_credits_parent_norm_zero()
    test_r3_bootstrap_raises_parent_norm_above_no_credit_control()
    test_r3_bootstrap_direction_aligned_with_credited_child()
    test_r4_maintenance_reinforces_already_seeded_parent()
    test_r5_zero_or_negative_credit_is_noop()
    test_r5_below_subgoal_credit_min_is_noop()
    test_r6_parent_decays_without_credit_events()
    test_r7_credit_does_not_perturb_child_zgoal()
    test_r8_reset_clears_parent_and_counters()
    test_r9_state_dict_round_trip_preserves_parent()
    test_r9_state_dict_flag_off_serializes_none()
    test_r9_load_pre_sd092_checkpoint_leaves_parent_unallocated()
    test_r10_with_injection_carries_parent_state_without_raising()
    test_r10_with_injection_flag_off_still_works()
    test_r11_lazy_allocation_when_flag_flipped_post_construction()
    print("All SD-092 cross-level subgoal credit contracts PASS")
