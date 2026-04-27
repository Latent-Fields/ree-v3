"""Contract tests for MECH-293 waking ghost-goal probe search.

MECH-293 extends HippocampalModule.propose_trajectories() with a minority
budget of CEM probes seeded around the highest-priority entries in the
MECH-292 ranked ghost-goal bank. Each ghost trajectory carries
hypothesis_tag=True and a metadata dict tagging its source anchor +
ghost_priority for downstream provenance.

Spec: REE_assembly/docs/architecture/mech_293_ghost_goal_probe_search.md.

Guarantees enforced here:
  C1. Module surface: HippocampalConfig has use_mech293_ghost_probes +
      4 sub-knobs; HippocampalModule has _propose_ghost_seeded,
      _mix_value_flat_with_ghost, get_last_propose_diagnostics.
  C2. Default backward-compat: use_mech293_ghost_probes=False;
      propose_trajectories returns value-flat candidates with no ghost
      branch; _last_propose_diagnostics stays empty.
  C3. Precondition gate: use_mech293_ghost_probes=True without
      use_mech292_ghost_bank=True raises ValueError at module construction.
  C4. Ghost branch fires when bank has entries: at least one trajectory
      in the returned list carries metadata["source"]=="mech293_ghost_probe"
      and hypothesis_tag=True; diagnostics report n_ghost_admitted >= 1.
  C5. Ghost branch silent on empty bank / None z_goal:
      mech293_n_ghost_admitted == 0 and no trajectory carries the ghost
      source tag.
  C6. Budget respected: with a bank larger than the configured budget,
      n_ghost_admitted == clamp(round(n*fraction), [min, max]) bounded
      by bank size.
  C7. Mix replace_lowest=True preserves total candidate count; ghost
      trajectories appear at the tail of the returned list.
  C8. Trajectory.hypothesis_tag and .metadata default to False / None
      for value-flat CEM proposals (backward-compat for existing
      consumers).
  C9. record_committed_trajectory strips hypothesis_tag/metadata even
      when the source proposal was a ghost probe (the executed trajectory
      IS real, regardless of its origin).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pytest
import torch


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #


def _make_hippocampal_with_bank(use_mech293=False, **overrides):
    """Construct a HippocampalModule with the full MECH-292 substrate
    chain wired (anchor_set + sd039_payload + bank), optionally with
    MECH-293 enabled. Returns (module, config) for further setup."""
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2FastPredictor, E2Config
    from ree_core.residue.field import ResidueField, ResidueConfig
    from ree_core.utils.config import (
        AnchorSetConfig,
        GhostGoalBankConfig,
        HippocampalConfig,
    )

    anchor_cfg = AnchorSetConfig(use_sd039_anchor_payload=True)
    bank_cfg = GhostGoalBankConfig(goal_match_floor=0.05)
    cfg = HippocampalConfig(
        world_dim=8,
        action_dim=4,
        action_object_dim=8,
        hidden_dim=32,
        horizon=4,
        num_candidates=8,
        num_cem_iterations=1,
        elite_fraction=0.25,
        use_anchor_sets=True,
        anchor_set=anchor_cfg,
        use_mech292_ghost_bank=True,
        ghost_goal_bank_config=bank_cfg,
        use_mech293_ghost_probes=use_mech293,
        mech293_ghost_fraction=0.25,
        mech293_min_ghost_candidates=1,
        mech293_max_ghost_candidates=4,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)

    e2_cfg = E2Config(
        self_dim=8, world_dim=8, action_dim=4,
        action_object_dim=8, hidden_dim=32,
    )
    e2 = E2FastPredictor(e2_cfg)
    rcfg = ResidueConfig(
        world_dim=8, hidden_dim=32, num_basis_functions=8,
    )
    rf = ResidueField(rcfg)

    module = HippocampalModule(cfg, e2=e2, residue_field=rf)
    return module, cfg


def _seed_anchor_with_payload(
    module, *, scale="fast", segment_id="0.0",
    z_world=None, z_goal_snapshot=None, wanting=0.5,
):
    """Install one anchor on a known (scale, segment_id, mixture) key
    with a populated SD-039 goal payload. Returns the Anchor."""
    from ree_core.hippocampal.anchor_set import AnchorGoalPayload

    if z_world is None:
        z_world = torch.zeros(1, 8)
    if z_goal_snapshot is None:
        z_goal_snapshot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    payload = AnchorGoalPayload(
        z_goal_snapshot=z_goal_snapshot.detach().clone(),
        wanting_strength=float(wanting),
        arousal_tag=0.3,
        last_vs=0.8,
        staleness_at_write=0.1,
        payload_written_step=0,
    )
    anchor = module.anchor_set.write_anchor(
        scale=scale,
        segment_id=segment_id,
        stream_mixture=("z_world",),
        z_world=z_world,
        goal_payload=payload,
    )
    return anchor


# ------------------------------------------------------------------ #
# C1: surface / symbol presence                                      #
# ------------------------------------------------------------------ #


def test_c1_module_surface():
    """HippocampalConfig + HippocampalModule expose the MECH-293 surface."""
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.utils.config import HippocampalConfig

    cfg = HippocampalConfig()
    # Master flag + 4 sub-knobs default to safe values.
    assert cfg.use_mech293_ghost_probes is False
    assert cfg.mech293_ghost_fraction == 0.2
    assert cfg.mech293_min_ghost_candidates == 1
    assert cfg.mech293_max_ghost_candidates == 8
    assert cfg.mech293_replace_lowest_ranked is True
    # Methods present on HippocampalModule.
    assert hasattr(HippocampalModule, "_propose_ghost_seeded")
    assert hasattr(HippocampalModule, "_mix_value_flat_with_ghost")
    assert hasattr(HippocampalModule, "get_last_propose_diagnostics")


# ------------------------------------------------------------------ #
# C2: default backward-compat                                        #
# ------------------------------------------------------------------ #


def test_c2_master_off_no_op():
    """Default-OFF: ghost branch never fires; no trajectory carries the
    ghost source tag; diagnostics empty."""
    module, cfg = _make_hippocampal_with_bank(use_mech293=False)
    z_world = torch.zeros(1, 8)
    z_self = torch.zeros(1, 8)
    z_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    candidates = module.propose_trajectories(
        z_world=z_world, z_self=z_self, current_z_goal=z_goal,
    )

    assert len(candidates) == cfg.num_candidates
    for traj in candidates:
        # Defaults from Trajectory dataclass extension.
        assert traj.hypothesis_tag is False
        assert traj.metadata is None
    # Diagnostics empty when ghost branch is off.
    assert module.get_last_propose_diagnostics() == {}


# ------------------------------------------------------------------ #
# C3: precondition gate                                              #
# ------------------------------------------------------------------ #


def test_c3_precondition_requires_mech292():
    """use_mech293_ghost_probes=True without use_mech292_ghost_bank=True
    raises ValueError at module construction (loud-not-silent)."""
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2Config, E2FastPredictor
    from ree_core.residue.field import ResidueConfig, ResidueField
    from ree_core.utils.config import (
        AnchorSetConfig, HippocampalConfig,
    )

    cfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=8,
        hidden_dim=32, horizon=4, num_candidates=8,
        num_cem_iterations=1, elite_fraction=0.25,
        use_anchor_sets=True,
        anchor_set=AnchorSetConfig(use_sd039_anchor_payload=True),
        use_mech292_ghost_bank=False,           # bank OFF
        use_mech293_ghost_probes=True,          # MECH-293 wants the bank
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=8, world_dim=8, action_dim=4,
        action_object_dim=8, hidden_dim=32,
    ))
    rf = ResidueField(ResidueConfig(
        world_dim=8, hidden_dim=32, num_basis_functions=8,
    ))

    with pytest.raises(ValueError, match="MECH-293 requires"):
        HippocampalModule(cfg, e2=e2, residue_field=rf)


# ------------------------------------------------------------------ #
# C4: ghost branch fires with non-empty bank                         #
# ------------------------------------------------------------------ #


def test_c4_ghost_branch_fires_with_bank():
    """Bank with at least one entry above floor -> >=1 ghost trajectory in
    the returned list, tagged with metadata["source"]=='mech293_ghost_probe'
    and hypothesis_tag=True."""
    module, cfg = _make_hippocampal_with_bank(use_mech293=True)

    z_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # Seed one inactive anchor whose stored z_goal_snapshot matches the
    # current z_goal direction. mark_inactive preserves payload, so the
    # default include_inactive=True bank picks it up.
    _seed_anchor_with_payload(
        module, scale="fast", segment_id="0.0",
        z_world=torch.tensor([[1.0] * 8]),
        z_goal_snapshot=z_goal.clone(),
    )
    module.anchor_set.mark_inactive(scale="fast", stream_mixture=("z_world",))

    z_world = torch.zeros(1, 8)
    z_self = torch.zeros(1, 8)
    candidates = module.propose_trajectories(
        z_world=z_world, z_self=z_self, current_z_goal=z_goal,
    )

    ghost_count = sum(
        1 for t in candidates
        if t.metadata is not None
        and t.metadata.get("source") == "mech293_ghost_probe"
    )
    assert ghost_count >= 1, "expected at least one ghost trajectory"

    # Each ghost trajectory carries hypothesis_tag=True and a populated
    # metadata dict including ghost_priority and goal_match.
    for t in candidates:
        if t.metadata and t.metadata.get("source") == "mech293_ghost_probe":
            assert t.hypothesis_tag is True
            assert "anchor_key" in t.metadata
            assert "ghost_priority" in t.metadata
            assert "goal_match" in t.metadata
            assert t.metadata["goal_match"] > 0.0

    diag = module.get_last_propose_diagnostics()
    assert diag.get("mech293_n_ghost_admitted", 0) >= 1
    assert diag.get("mech293_reason") == "ok"


# ------------------------------------------------------------------ #
# C5: ghost branch silent on empty bank or None z_goal               #
# ------------------------------------------------------------------ #


def test_c5_silent_on_no_z_goal():
    """current_z_goal=None -> no ghost trajectories, diagnostics report
    no_z_goal."""
    module, cfg = _make_hippocampal_with_bank(use_mech293=True)
    # Even with anchors present, None z_goal short-circuits.
    _seed_anchor_with_payload(module)

    z_world = torch.zeros(1, 8)
    z_self = torch.zeros(1, 8)
    candidates = module.propose_trajectories(
        z_world=z_world, z_self=z_self, current_z_goal=None,
    )
    ghosts = [
        t for t in candidates
        if t.metadata and t.metadata.get("source") == "mech293_ghost_probe"
    ]
    assert ghosts == []
    diag = module.get_last_propose_diagnostics()
    assert diag.get("mech293_n_ghost_admitted") == 0
    assert diag.get("mech293_reason") == "no_z_goal"


def test_c5_silent_on_empty_bank():
    """No anchors / all-below-floor -> no ghost trajectories, diagnostics
    report empty_bank."""
    module, cfg = _make_hippocampal_with_bank(use_mech293=True)
    # No anchors seeded -> bank is empty.
    z_world = torch.zeros(1, 8)
    z_self = torch.zeros(1, 8)
    z_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    candidates = module.propose_trajectories(
        z_world=z_world, z_self=z_self, current_z_goal=z_goal,
    )
    ghosts = [
        t for t in candidates
        if t.metadata and t.metadata.get("source") == "mech293_ghost_probe"
    ]
    assert ghosts == []
    diag = module.get_last_propose_diagnostics()
    assert diag.get("mech293_n_ghost_admitted") == 0
    assert diag.get("mech293_reason") == "empty_bank"


# ------------------------------------------------------------------ #
# C6: budget respected                                               #
# ------------------------------------------------------------------ #


def test_c6_budget_respected():
    """With a bank of >=4 entries and fraction=0.25, n=8 -> n_ghost==2
    (round(0.25*8)=2 in [1,4] clamps; bank larger than budget)."""
    module, cfg = _make_hippocampal_with_bank(use_mech293=True)
    z_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    # Seed 4 anchors all matching the current z_goal direction.
    for i in range(4):
        offset = torch.full((1, 8), float(i + 1) * 0.1)
        _seed_anchor_with_payload(
            module, scale="fast", segment_id=f"0.{i}",
            z_world=offset, z_goal_snapshot=z_goal.clone(),
        )
        module.anchor_set.mark_inactive(
            scale="fast", stream_mixture=("z_world",),
        )

    z_world = torch.zeros(1, 8)
    z_self = torch.zeros(1, 8)
    candidates = module.propose_trajectories(
        z_world=z_world, z_self=z_self, current_z_goal=z_goal,
        num_candidates=8,
    )
    ghost_count = sum(
        1 for t in candidates
        if t.metadata and t.metadata.get("source") == "mech293_ghost_probe"
    )
    diag = module.get_last_propose_diagnostics()
    assert ghost_count == 2, f"expected n_ghost=2, got {ghost_count}"
    assert diag["mech293_n_ghost_admitted"] == 2


def test_c6_budget_floor_clamp():
    """With fraction=0.0 + min_ghost=1 + bank with entries, the floor
    forces n_ghost==1 (min wins over the round-down)."""
    module, cfg = _make_hippocampal_with_bank(
        use_mech293=True,
        mech293_ghost_fraction=0.0,
        mech293_min_ghost_candidates=1,
    )
    z_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _seed_anchor_with_payload(module, z_goal_snapshot=z_goal.clone())
    module.anchor_set.mark_inactive(scale="fast", stream_mixture=("z_world",))

    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
        current_z_goal=z_goal,
        num_candidates=8,
    )
    diag = module.get_last_propose_diagnostics()
    assert diag["mech293_n_ghost_admitted"] == 1


# ------------------------------------------------------------------ #
# C7: mix preserves total candidate count + tail placement           #
# ------------------------------------------------------------------ #


def test_c7_mix_replace_lowest_preserves_count():
    """replace_lowest=True keeps len(candidates)==n; ghost candidates
    sit at the tail."""
    module, cfg = _make_hippocampal_with_bank(use_mech293=True)
    z_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _seed_anchor_with_payload(module, z_goal_snapshot=z_goal.clone())
    module.anchor_set.mark_inactive(scale="fast", stream_mixture=("z_world",))

    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
        current_z_goal=z_goal,
        num_candidates=8,
    )
    assert len(candidates) == 8
    # Tail placement: ghosts appear at the end of the list.
    diag = module.get_last_propose_diagnostics()
    n_ghost = diag["mech293_n_ghost_admitted"]
    assert n_ghost >= 1
    tail = candidates[-n_ghost:]
    for t in tail:
        assert t.metadata is not None
        assert t.metadata.get("source") == "mech293_ghost_probe"


def test_c7_mix_append_raises_count():
    """replace_lowest=False appends; total count grows by n_ghost."""
    module, cfg = _make_hippocampal_with_bank(
        use_mech293=True,
        mech293_replace_lowest_ranked=False,
    )
    z_goal = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    _seed_anchor_with_payload(module, z_goal_snapshot=z_goal.clone())
    module.anchor_set.mark_inactive(scale="fast", stream_mixture=("z_world",))

    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
        current_z_goal=z_goal,
        num_candidates=8,
    )
    diag = module.get_last_propose_diagnostics()
    n_ghost = diag["mech293_n_ghost_admitted"]
    assert n_ghost >= 1
    assert len(candidates) == 8 + n_ghost


# ------------------------------------------------------------------ #
# C8: Trajectory dataclass defaults are backward-compat              #
# ------------------------------------------------------------------ #


def test_c8_trajectory_defaults_backward_compat():
    """Pre-MECH-293 callers that construct Trajectory(states, actions, ...)
    without the new fields get hypothesis_tag=False and metadata=None."""
    from ree_core.predictors.e2_fast import Trajectory

    t = Trajectory(
        states=[torch.zeros(1, 4)],
        actions=torch.zeros(1, 1, 4),
    )
    assert t.hypothesis_tag is False
    assert t.metadata is None


# ------------------------------------------------------------------ #
# C9: record_committed_trajectory strips ghost provenance            #
# ------------------------------------------------------------------ #


def test_c9_record_committed_strips_hypothesis_tag():
    """Even if a ghost-probe trajectory becomes the executed committed
    trajectory, record_committed_trajectory stores it with
    hypothesis_tag=False and metadata=None (it IS real now)."""
    from ree_core.predictors.e2_fast import Trajectory

    module, cfg = _make_hippocampal_with_bank(
        use_mech293=True,
        use_backward_credit_sweep=True,  # gate for record_committed
    )
    src = Trajectory(
        states=[torch.zeros(1, 8), torch.zeros(1, 8)],
        actions=torch.zeros(1, 1, 4),
        world_states=[torch.zeros(1, 8), torch.zeros(1, 8)],
        action_objects=None,
        hypothesis_tag=True,
        metadata={"source": "mech293_ghost_probe"},
    )
    module.record_committed_trajectory(src)
    stored = module._committed_trajectory_buffer
    assert stored is not None
    assert stored.hypothesis_tag is False
    assert stored.metadata is None
