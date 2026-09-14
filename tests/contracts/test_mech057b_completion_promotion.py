"""Contracts for MECH-057b hippocampal sequence-completion verification and
trajectory-promotion policy (HippocampalModule.verify_sequence_completion /
promote_candidates), and its end-to-end wiring into propose_trajectories().

Fix for the V3-EXQ-672-series finding (failure_autopsy_V3-EXQ-672-series_
2026-06-15, CONFIRMED): the prior harness gate ranked candidates directly on
hippocampal._score_trajectory (ARC-007 STRICT terrain/residue cost) and
inherited the ARC-065 GAP-A candidate-pool collapse instead of measuring
completion at all. These contracts assert the new verification channel
(VisitationCounter-derived) is genuinely distinct and selective, not a
relabelled re-use of the terrain score.
"""

from __future__ import annotations

import torch


def _make_module(
    world_dim: int = 8,
    use_completion_promotion_gate: bool = False,
    completion_promotion_verification_floor: float = 0.3,
    completion_promotion_drop_fraction: float = 0.4,
    completion_promotion_min_candidates: int = 2,
    completion_verification_tau: float = 1.0,
):
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2Config, E2FastPredictor
    from ree_core.residue.field import ResidueConfig, ResidueField
    from ree_core.utils.config import HippocampalConfig

    cfg = HippocampalConfig(
        world_dim=world_dim,
        action_dim=4,
        action_object_dim=8,
        hidden_dim=32,
        horizon=4,
        num_candidates=16,
        num_cem_iterations=2,
        elite_fraction=0.25,
        use_completion_promotion_gate=use_completion_promotion_gate,
        completion_promotion_verification_floor=completion_promotion_verification_floor,
        completion_promotion_drop_fraction=completion_promotion_drop_fraction,
        completion_promotion_min_candidates=completion_promotion_min_candidates,
        completion_verification_tau=completion_verification_tau,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=8,
        world_dim=world_dim,
        action_dim=4,
        action_object_dim=8,
        hidden_dim=32,
    ))
    residue = ResidueField(ResidueConfig(
        world_dim=world_dim,
        hidden_dim=32,
        num_basis_functions=8,
    ))
    return HippocampalModule(cfg, e2=e2, residue_field=residue)


def _traj(world_points, world_dim=8, metadata=None):
    """Minimal Trajectory with a controlled world_states sequence.

    states/actions are dummy placeholders (irrelevant to the methods under
    test); only world_states is read by verify_sequence_completion.
    """
    from ree_core.predictors.e2_fast import Trajectory

    world_states = [
        torch.as_tensor(p, dtype=torch.float32).reshape(1, world_dim)
        for p in world_points
    ]
    states = [torch.zeros(1, 4) for _ in world_points]
    actions = torch.zeros(1, max(1, len(world_points) - 1), 4)
    return Trajectory(
        states=states,
        actions=actions,
        world_states=world_states,
        metadata=metadata,
    )


# --------------------------------------------------------------------- #
# Config defaults
# --------------------------------------------------------------------- #

def test_config_exposes_completion_promotion_gate_default_off():
    from ree_core.utils.config import HippocampalConfig, REEConfig

    cfg = HippocampalConfig()
    assert hasattr(cfg, "use_completion_promotion_gate")
    assert cfg.use_completion_promotion_gate is False
    assert cfg.completion_verification_tau == 1.0
    assert cfg.completion_promotion_verification_floor == 0.3
    assert cfg.completion_promotion_drop_fraction == 0.4
    assert cfg.completion_promotion_min_candidates == 2

    master = REEConfig.from_dims(
        body_obs_dim=4,
        world_obs_dim=8,
        action_dim=4,
        self_dim=8,
        world_dim=8,
    )
    assert master.hippocampal.use_completion_promotion_gate is False

    # Reachability: from_dims must actually forward the flag (not silently
    # drop it -- see [memory] reference-reeconfig-from-dims-silent-kwargs
    # and test_from_dims_flag_reachability.py).
    master_on = REEConfig.from_dims(
        body_obs_dim=4,
        world_obs_dim=8,
        action_dim=4,
        self_dim=8,
        world_dim=8,
        use_completion_promotion_gate=True,
        completion_verification_tau=2.0,
        completion_promotion_verification_floor=0.5,
        completion_promotion_drop_fraction=0.3,
        completion_promotion_min_candidates=4,
    )
    assert master_on.hippocampal.use_completion_promotion_gate is True
    assert master_on.hippocampal.completion_verification_tau == 2.0
    assert master_on.hippocampal.completion_promotion_verification_floor == 0.5
    assert master_on.hippocampal.completion_promotion_drop_fraction == 0.3
    assert master_on.hippocampal.completion_promotion_min_candidates == 4


# --------------------------------------------------------------------- #
# verify_sequence_completion: distinct channel from _score_trajectory
# --------------------------------------------------------------------- #

def test_verify_sequence_completion_unvisited_is_zero():
    module = _make_module()
    traj = _traj([[1.0] * 8, [2.0] * 8, [3.0] * 8])
    assert module.verify_sequence_completion(traj) == 0.0


def test_verify_sequence_completion_rises_with_visits_and_is_read_only():
    module = _make_module()
    point = [1.0] * 8
    traj = _traj([point, point, point])

    conf_before = module.verify_sequence_completion(traj)
    assert conf_before == 0.0

    # query() must not itself write -- repeated verification of the SAME
    # unvisited trajectory must stay at 0.0.
    conf_again = module.verify_sequence_completion(traj)
    assert conf_again == conf_before

    # Only update() (a real waking visit) raises confidence.
    for _ in range(5):
        module.visitation_counter.update(torch.as_tensor(point))
    conf_after = module.verify_sequence_completion(traj)
    assert conf_after > conf_before
    # tau=1.0, count=5 -> confidence = 5/6.
    assert abs(conf_after - (5.0 / 6.0)) < 1e-6


def test_verify_sequence_completion_is_min_over_waypoints_not_mean():
    module = _make_module()
    visited = [10.0] * 8
    unvisited = [-10.0] * 8
    for _ in range(20):
        module.visitation_counter.update(torch.as_tensor(visited))

    traj = _traj([visited, unvisited, visited])
    # The unvisited waypoint should pin the sequence confidence near 0,
    # even though two of three waypoints are heavily visited -- a mean
    # would read much higher and mask the weak link.
    assert module.verify_sequence_completion(traj) == 0.0


def test_verify_sequence_completion_distinct_from_score_trajectory():
    """The two channels must be able to disagree: a trajectory with a GOOD
    (low) terrain score but ZERO visitation must read completion
    confidence 0.0, proving this is not a relabelled _score_trajectory."""
    module = _make_module()
    traj = _traj([[0.0] * 8, [0.0] * 8])  # residue field default near origin
    terrain_score = module._score_trajectory(traj)
    completion_conf = module.verify_sequence_completion(traj)
    assert isinstance(terrain_score, torch.Tensor)
    assert completion_conf == 0.0  # never visited, regardless of terrain


# --------------------------------------------------------------------- #
# promote_candidates: gate wiring, selectivity, deadlock guard, exemptions
# --------------------------------------------------------------------- #

def test_promote_candidates_noop_when_gate_disabled():
    module = _make_module(use_completion_promotion_gate=False)
    trajs = [_traj([[float(i)] * 8]) for i in range(10)]
    promoted, diag = module.promote_candidates(trajs)
    assert promoted is trajs
    assert diag["use_completion_promotion_gate"] is False
    assert diag["completion_promotion_filtered_fraction"] == 0.0


def test_promote_candidates_selectively_drops_low_confidence():
    module = _make_module(
        use_completion_promotion_gate=True,
        completion_promotion_verification_floor=0.3,
        completion_promotion_drop_fraction=0.6,
        completion_promotion_min_candidates=2,
    )
    visited = [5.0] * 8
    for _ in range(20):
        module.visitation_counter.update(torch.as_tensor(visited))

    high_conf = [_traj([visited, visited]) for _ in range(4)]
    low_conf = [_traj([[100.0 + i] * 8, [100.0 + i] * 8]) for i in range(4)]
    pool = high_conf + low_conf

    promoted, diag = module.promote_candidates(pool)
    assert diag["use_completion_promotion_gate"] is True
    assert diag["completion_promotion_candidates_scored"] == 8
    assert diag["completion_promotion_confidence_spread"] > 0.0
    assert len(promoted) < len(pool)
    # Selectivity: every dropped candidate must come from the low-confidence
    # set, never the high-confidence one (this is the exact criterion
    # EVB-1722/EXP-0594's own acceptance check names: suppression must be
    # selective on completion, not on an arbitrary/noisy ranking). Identity
    # (id()), not == , because Trajectory is a tensor-bearing dataclass and
    # `in`/`==` on it raises (ambiguous multi-element tensor truth value).
    promoted_ids = {id(t) for t in promoted}
    high_conf_ids = {id(t) for t in high_conf}
    low_conf_ids = {id(t) for t in low_conf}
    dropped_ids = {id(t) for t in pool} - promoted_ids
    assert len(dropped_ids) > 0
    assert dropped_ids <= low_conf_ids
    assert high_conf_ids <= promoted_ids


def test_promote_candidates_deadlock_guard_when_pool_entirely_unvisited():
    module = _make_module(
        use_completion_promotion_gate=True,
        completion_promotion_verification_floor=0.3,
        completion_promotion_drop_fraction=1.0,
        completion_promotion_min_candidates=3,
    )
    # Every candidate reads confidence 0.0 (nothing ever visited) -- the
    # common early-training/early-episode regime. The gate must not
    # deadlock E3 on an empty pool.
    pool = [_traj([[float(i)] * 8]) for i in range(10)]
    promoted, diag = module.promote_candidates(pool)
    assert len(promoted) >= 3
    assert diag["completion_promotion_mean_confidence"] == 0.0


def test_promote_candidates_never_drops_exempt_source_candidates():
    module = _make_module(
        use_completion_promotion_gate=True,
        completion_promotion_verification_floor=0.9,  # near-impossible floor
        completion_promotion_drop_fraction=1.0,
        completion_promotion_min_candidates=1,
    )
    protected = _traj(
        [[0.0] * 8], metadata={"source": "action_class_scaffold"}
    )
    ordinary = [_traj([[float(i)] * 8]) for i in range(5)]
    pool = [protected] + ordinary

    promoted, _diag = module.promote_candidates(pool)
    assert protected in promoted


def test_promote_candidates_noop_when_pool_at_or_below_min_candidates():
    module = _make_module(
        use_completion_promotion_gate=True,
        completion_promotion_min_candidates=5,
    )
    pool = [_traj([[float(i)] * 8]) for i in range(5)]
    promoted, diag = module.promote_candidates(pool)
    assert promoted is pool
    assert diag["completion_promotion_filtered_fraction"] == 0.0


# --------------------------------------------------------------------- #
# End-to-end wiring: propose_trajectories()
# --------------------------------------------------------------------- #

def test_propose_trajectories_bit_identical_when_gate_disabled():
    torch.manual_seed(42)
    module_off = _make_module(use_completion_promotion_gate=False)
    candidates = module_off.propose_trajectories(
        z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8),
    )
    diag = module_off.get_last_propose_diagnostics()
    assert len(candidates) == module_off.config.num_candidates
    assert diag["use_completion_promotion_gate"] is False
    assert diag["completion_promotion_filtered_fraction"] == 0.0


def test_propose_trajectories_gate_enabled_end_to_end_no_deadlock():
    torch.manual_seed(42)
    module_on = _make_module(
        use_completion_promotion_gate=True,
        completion_promotion_min_candidates=2,
    )
    candidates = module_on.propose_trajectories(
        z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8),
    )
    diag = module_on.get_last_propose_diagnostics()
    assert diag["use_completion_promotion_gate"] is True
    # Fresh module -> visitation memory is empty -> deadlock guard must
    # still leave a usable pool for E3.
    assert len(candidates) >= 2
    assert "completion_promotion_filtered_fraction" in diag
