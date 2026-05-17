"""
commitment_closure:GAP-3 -- CausalGridWorldV2 env extensions, primitives 1-3.

Contract tests for:
  Primitive 1: adaptive tolerance-band completion
  Primitive 2: counter-evidence injection hook (graded contingency degradation)
  Primitive 3: dual simultaneously-active resource cue

Spec (decision-complete 2026-05-16):
  REE_assembly/evidence/planning/causalgridworldv2_env_extensions_spec.md

C1  bit-identical OFF (all three) + frac=0.0 dynamics-identical + sentinel keys
C2  tolerance band: fires within / not outside; frac=0.0==exact; graded_exp; metric
C3  counter-evidence: persistent-only, monotone validity->floor, context-invariant,
    transition tag, duration, bit-identical when disabled
C4  dual-cue: SD-049 precondition fail-fast; waypoint+resource reserved; accounting
C5  spec section-5 integration smoke (all three in one episode)
"""

import numpy as np
import pytest

from ree_core.environment.causal_grid_world import CausalGridWorldV2

_BASE = dict(size=12, seed=7, num_hazards=3, num_resources=4,
             subgoal_mode=True, num_waypoints=3)


def _rollout(env, acts):
    """Deterministic rollout -> per-step (reward, done, obs-sum, agent, hazards,
    resources). Captures env *dynamics* for bit-identity comparison."""
    env.reset()
    tr = []
    for a in acts:
        o, r, d, i, _ = env.step(a)
        tr.append((
            round(float(r), 9), bool(d), round(float(np.asarray(o).sum()), 6),
            (env.agent_x, env.agent_y),
            sorted(tuple(h) for h in env.hazards),
            sorted(tuple(x) for x in env.resources),
        ))
        if d:
            env.reset()
    return tr


def _acts(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 4, n)]


# --------------------------------------------------------------------------- C1
def test_c1_bit_identical_off_dynamics():
    """Default env vs explicit-all-disabled: identical dynamics in lockstep."""
    a = CausalGridWorldV2(**_BASE)
    b = CausalGridWorldV2(**_BASE, completion_tolerance_enabled=False,
                          counter_evidence_enabled=False, dual_cue_enabled=False)
    assert _rollout(a, _acts()) == _rollout(b, _acts())


def test_c1_frac_zero_dynamics_identical_to_off():
    """completion_tolerance_enabled=True with frac=0.0 -> T=0 -> no dynamics
    change (only diagnostic info keys differ). Bit-identical env dynamics."""
    off = CausalGridWorldV2(**_BASE)
    f0 = CausalGridWorldV2(**_BASE, completion_tolerance_enabled=True,
                           completion_tolerance_frac=0.0)
    assert _rollout(off, _acts()) == _rollout(f0, _acts())


def test_c1_sentinel_keys_present_and_inert_when_off():
    env = CausalGridWorldV2(**_BASE)
    env.reset()
    _, _, _, info, _ = env.step(0)
    for k in (
        "completion_tolerance_enabled", "completion_tolerance_T",
        "completion_within_tolerance_this_tick", "completion_dist_to_target",
        "completion_tolerance_credit", "counter_evidence_enabled",
        "counter_evidence_event_count", "counter_evidence_injected_this_tick",
        "counter_evidence_target_validity", "counter_evidence_sustained_steps",
        "counter_evidence_last_step", "dual_cue_enabled", "dual_cue_n_active",
        "dual_cue_ticks_both_active", "dual_cue_consumed_tag_this_tick",
        "dual_cue_invalid_this_episode",
    ):
        assert k in info, "missing info key: " + k
    assert info["completion_tolerance_enabled"] is False
    assert info["completion_tolerance_T"] == -1
    assert info["completion_within_tolerance_this_tick"] is False
    assert info["completion_dist_to_target"] == -1
    assert info["completion_tolerance_credit"] == 0.0
    assert info["counter_evidence_enabled"] is False
    assert info["counter_evidence_event_count"] == 0
    assert info["counter_evidence_target_validity"] == 1.0
    assert info["dual_cue_enabled"] is False
    assert info["dual_cue_n_active"] == 0
    assert info["dual_cue_invalid_this_episode"] is False


# --------------------------------------------------------------------------- C2
def test_c2_tolerance_band_fires_within_not_outside():
    """OFF never reports within-tolerance; a real band reports many."""
    off = CausalGridWorldV2(**_BASE)
    band = CausalGridWorldV2(**_BASE, completion_tolerance_enabled=True,
                             completion_tolerance_frac=0.17)
    n_off = n_band = 0
    off.reset()
    band.reset()
    for a in _acts():
        _, _, do, io, _ = off.step(a)
        _, _, db, ib, _ = band.step(a)
        n_off += io["completion_within_tolerance_this_tick"]
        n_band += ib["completion_within_tolerance_this_tick"]
        if do:
            off.reset()
        if db:
            band.reset()
    assert n_off == 0
    assert n_band > 0


def test_c2_graded_exp_credit_formula_invariant():
    """On every within-tolerance tick the graded_exp credit equals
    exp(-dist / lambda) for the reported dist, and lies in (0, 1]. Tested as
    an invariant over a rollout so it does not depend on exact post-step
    geometry."""
    lam = 2.0
    g = CausalGridWorldV2(**_BASE, completion_tolerance_enabled=True,
                          completion_tolerance_frac=0.25,
                          completion_tolerance_kernel="graded_exp",
                          completion_tolerance_lambda=lam)
    g.reset()
    n_within = 0
    for a in _acts(400):
        _, _, d, i, _ = g.step(a)
        if i["completion_within_tolerance_this_tick"]:
            n_within += 1
            dist = i["completion_dist_to_target"]
            cred = i["completion_tolerance_credit"]
            assert 0.0 < cred <= 1.0
            assert cred == pytest.approx(float(np.exp(-dist / lam)),
                                         rel=1e-6)
        if d:
            g.reset()
    assert n_within > 0, "no within-tolerance ticks observed"


def test_c2_hard_kernel_full_credit_and_metric_validation():
    h = CausalGridWorldV2(**_BASE, completion_tolerance_enabled=True,
                          completion_tolerance_cells=2,
                          completion_tolerance_kernel="hard")
    h.reset()
    wp = h.waypoints[h._next_waypoint_idx]
    h.agent_x, h.agent_y = max(1, int(wp[0]) - 1), int(wp[1])
    _, _, _, info, _ = h.step(0)
    assert info["completion_within_tolerance_this_tick"] is True
    assert info["completion_tolerance_credit"] == 1.0
    with pytest.raises(ValueError):
        CausalGridWorldV2(**_BASE, completion_tolerance_enabled=True,
                          completion_tolerance_metric="euclidean")


# --------------------------------------------------------------------------- C3
def test_c3_counter_evidence_bit_identical_when_disabled():
    a = CausalGridWorldV2(**_BASE)
    b = CausalGridWorldV2(**_BASE, counter_evidence_enabled=False)
    assert _rollout(a, _acts()) == _rollout(b, _acts())


def test_c3_validity_monotone_to_floor_only_while_persistent():
    env = CausalGridWorldV2(size=10, seed=1, num_hazards=2, num_resources=3,
                            subgoal_mode=True, num_waypoints=3,
                            counter_evidence_enabled=True,
                            counter_evidence_interval=2,
                            counter_evidence_prob=1.0,
                            counter_evidence_degrade_step=0.25,
                            counter_evidence_degrade_floor=0.0)
    env.reset()
    # No persistent rule yet: validity stays intact, no events.
    for _ in range(6):
        _, _, _, i0, _ = env.step(0)
    assert i0["counter_evidence_target_validity"] == 1.0
    assert i0["counter_evidence_event_count"] == 0
    # Force a persistent rule_state; validity now degrades monotonically.
    env._sequence_in_progress = True
    seen = []
    for _ in range(12):
        _, _, _, i, _ = env.step(0)
        seen.append(i["counter_evidence_target_validity"])
    assert seen == sorted(seen, reverse=True)          # monotone non-increasing
    assert min(seen) == pytest.approx(0.0)             # reaches floor
    assert i["counter_evidence_event_count"] >= 1
    assert i["counter_evidence_sustained_steps"] >= 1


def test_c3_inject_method_touches_only_validity():
    """The degradation primitive itself mutates ONLY the committed target's
    outcome-validity -- grid, hazards, resources, agent position and the
    rule_state are all invariant. (Tested at the method level so normal
    env-caused hazard drift, which is independent, cannot confound it.)"""
    env = CausalGridWorldV2(size=10, seed=2, num_hazards=2, num_resources=3,
                            subgoal_mode=True, num_waypoints=3,
                            counter_evidence_enabled=True,
                            counter_evidence_degrade_step=0.2,
                            counter_evidence_degrade_floor=0.0)
    env.reset()
    env._sequence_in_progress = True
    grid0 = env.grid.copy()
    haz0 = sorted(tuple(h) for h in env.hazards)
    res0 = sorted(tuple(r) for r in env.resources)
    pos0 = (env.agent_x, env.agent_y)
    wp0 = env._next_waypoint_idx
    seq0 = env._sequence_in_progress
    seen = [env._counter_evidence_target_validity]
    for _ in range(8):
        changed = env._inject_counter_evidence()
        seen.append(env._counter_evidence_target_validity)
        assert np.array_equal(env.grid, grid0)
        assert sorted(tuple(h) for h in env.hazards) == haz0
        assert sorted(tuple(r) for r in env.resources) == res0
        assert (env.agent_x, env.agent_y) == pos0
        assert env._next_waypoint_idx == wp0
        assert env._sequence_in_progress == seq0
        assert isinstance(changed, bool)
    assert seen == sorted(seen, reverse=True)        # monotone non-increasing
    assert min(seen) == pytest.approx(0.0)           # reaches the floor
    # Returns False once pinned at the floor (no further change).
    assert env._inject_counter_evidence() is False


# --------------------------------------------------------------------------- C4
def test_c4_dual_cue_requires_sd049_fail_fast():
    with pytest.raises(ValueError):
        CausalGridWorldV2(size=10, seed=1, dual_cue_enabled=True)


def test_c4_waypoint_plus_resource_reserved_fail_fast():
    with pytest.raises(ValueError):
        CausalGridWorldV2(**_BASE, completion_tolerance_enabled=True,
                          completion_tolerance_targets="waypoint+resource")


def test_c4_dual_cue_type_tags_must_be_distinct():
    with pytest.raises(ValueError):
        CausalGridWorldV2(size=10, seed=1,
                          multi_resource_heterogeneity_enabled=True,
                          n_resource_types=2, dual_cue_enabled=True,
                          dual_cue_type_tags=(1, 1))


def test_c4_dual_cue_accounting_invariants():
    """With SD-049 on, the dual-cue info keys obey their invariants over a
    rollout: n_active in {0,1,2}, both-active count monotone within an episode,
    invalid flag boolean, default replace=False."""
    env = CausalGridWorldV2(size=12, seed=7, num_hazards=2, num_resources=6,
                            subgoal_mode=False,
                            multi_resource_heterogeneity_enabled=True,
                            n_resource_types=2, dual_cue_enabled=True,
                            dual_cue_min_active_ticks=10)
    assert env.dual_cue_replace_on_early_consume is False
    env.reset()
    prev = 0
    for a in _acts(200):
        _, _, d, i, _ = env.step(a)
        assert i["dual_cue_n_active"] in (0, 1, 2)
        assert isinstance(i["dual_cue_invalid_this_episode"], bool)
        assert i["dual_cue_ticks_both_active"] >= prev
        prev = i["dual_cue_ticks_both_active"]
        if d:
            env.reset()
            prev = 0


# --------------------------------------------------------------------------- C5
def test_c5_integration_smoke_all_three_in_one_episode():
    """Spec section 5: a single configuration exercising all three primitives
    produces within-tolerance completion, counter-evidence events while a rule
    is persistent, and a non-trivial both-active dual-cue window."""
    env = CausalGridWorldV2(size=12, seed=7, num_hazards=2, num_resources=6,
                            subgoal_mode=True, num_waypoints=3,
                            multi_resource_heterogeneity_enabled=True,
                            n_resource_types=2,
                            completion_tolerance_enabled=True,
                            completion_tolerance_frac=0.2,
                            completion_tolerance_kernel="hard",
                            counter_evidence_enabled=True,
                            counter_evidence_interval=3,
                            counter_evidence_prob=1.0,
                            counter_evidence_degrade_step=0.25,
                            dual_cue_enabled=True,
                            dual_cue_min_active_ticks=5)
    env.reset()
    env._sequence_in_progress = True  # ensure a persistent rule for the smoke
    within = 0
    ce_events = 0
    both_active = 0
    for a in _acts(300):
        _, _, d, i, _ = env.step(a)
        within += i["completion_within_tolerance_this_tick"]
        ce_events = max(ce_events, i["counter_evidence_event_count"])
        both_active = max(both_active, i["dual_cue_ticks_both_active"])
        if d:
            env.reset()
            env._sequence_in_progress = True
    assert within > 0, "tolerance-band completion never fired"
    assert ce_events > 0, "no counter-evidence events while rule persistent"
    assert both_active >= 5, "dual-cue both-active window below min"
