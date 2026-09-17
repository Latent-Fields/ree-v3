"""Contract tests for SD-097 -- typed, multi-relation possibility topology.

SD-097 stores TYPED DIRECTED RELATIONS between existing AnchorKeys
(ree_core/hippocampal/possibility_topology.py). It ships one relation,
`enables`, one explicit write path (observed succession at anchor remap),
and one live consumer (ghost-probe seeding, via GhostGoalBank.rank ->
HippocampalModule._propose_ghost_seeded).

Guarantees enforced here, in two directions:

OFF (bit-identical to pre-SD-097):
  O1. Nothing is constructed by default: module / anchor_set / bank all
      carry possibility_topology=None, and no entry carries
      relation_provenance.
  O2. rank() with a DISABLED topology attached AND edges present is
      element-for-element identical to rank() with no topology at all --
      same anchors, same order, exact-equal float priorities, identical
      components dicts, identical diagnostics keys (no sd097_* keys, no
      "relation" channel in component_sums).
  O3. The remap write path does not fire while disabled: no edge is
      recorded.
  O4. Ghost-probe SEEDING is unchanged: the seeded anchor_key set and
      per-probe goal_match are identical with the disabled topology
      attached. (This is the OFF path the live SD-098 falsifier reads.)

ON (the relation is demonstrably READ, not merely stored):
  N1. A successor below goal_match_floor is invisible to the bank. Add
      one `enables` edge from a floor-clearing parent and it becomes a
      ranked entry -- while a MATCHED CONTROL anchor, equally below the
      floor but with no edge, stays excluded. The edge is the only
      difference, so the consumer is reading the relation.
  N2. The admitted successor carries a positive "relation" component and
      relation_provenance naming its parent and edge, and rank()
      diagnostics report the expansion.
  N3. Behavioural difference at the SEEDING layer: propose_trajectories
      emits a ghost probe seeded from the successor's z_world ONLY when
      the edge exists.
  N4. Relation TYPE is read, not ignored: an edge of a different
      registered relation does not expand the `enables` consumer.
  N5. Write path: an observed family remap records
      `prior enables incoming` with provenance
      "anchor_remap_succession"; a repeat re-confirms rather than
      duplicating.

REGISTRY / EXTENSIBILITY:
  R1. `requires` and `is_part_of` are reachable with one
      register_relation() call and no schema change -- same store, same
      edge type, same queries. (They ship UNregistered on purpose: a
      relation with no consumer is the failure SD-097 was triaged for.)
  R2. Registry refuses an undescribed relation and a conflicting
      redefinition; add_edge on an unregistered relation raises.

SD-098 GUARD:
  S1. No stored node-type field anywhere: not on Anchor, not on
      TopologyEdge, not on a bank entry. Goal-ness stays a READ-TIME
      computation. A stored type field here is SD-098's comparison arm
      and would pre-empt that experiment, so this test is a tripwire,
      not a style check.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest
import torch

from ree_core.hippocampal.anchor_set import Anchor, AnchorGoalPayload
from ree_core.hippocampal.possibility_topology import (
    PROVENANCE_ANCHOR_REMAP,
    PossibilityTopology,
    PossibilityTopologyConfig,
    RelationSpec,
    RELATION_ENABLES,
    default_registry,
)


# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #

Z_GOAL = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
Z_ORTHOGONAL = torch.tensor([[0.0, 1.0, 0.0, 0.0]])


def _make_module(use_mech293: bool = False):
    """HippocampalModule with the full MECH-292 chain wired (anchor_set +
    SD-039 payloads + bank), optionally with MECH-293 ghost probes."""
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2FastPredictor, E2Config
    from ree_core.residue.field import ResidueField, ResidueConfig
    from ree_core.utils.config import (
        AnchorSetConfig,
        GhostGoalBankConfig,
        HippocampalConfig,
    )

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
        anchor_set=AnchorSetConfig(use_sd039_anchor_payload=True),
        use_mech292_ghost_bank=True,
        ghost_goal_bank_config=GhostGoalBankConfig(goal_match_floor=0.05),
        use_mech293_ghost_probes=use_mech293,
        mech293_ghost_fraction=0.25,
        mech293_min_ghost_candidates=1,
        mech293_max_ghost_candidates=4,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=8, world_dim=8, action_dim=4,
        action_object_dim=8, hidden_dim=32,
    ))
    rf = ResidueField(ResidueConfig(
        world_dim=8, hidden_dim=32, num_basis_functions=8,
    ))
    return HippocampalModule(cfg, e2=e2, residue_field=rf), cfg


def _seed(module, segment_id, z_world_fill, z_goal_snapshot, wanting=0.5):
    """Write one payload-bearing anchor on the 'fast' scale and leave it
    inactive (dual-trace preserved, so the bank's default
    include_inactive pool sees it)."""
    payload = AnchorGoalPayload(
        z_goal_snapshot=z_goal_snapshot.detach().clone(),
        wanting_strength=float(wanting),
        arousal_tag=0.3,
        last_vs=0.8,
        staleness_at_write=0.1,
        payload_written_step=0,
    )
    anchor = module.anchor_set.write_anchor(
        scale="fast",
        segment_id=segment_id,
        stream_mixture=("z_world",),
        z_world=torch.full((1, 8), float(z_world_fill)),
        goal_payload=payload,
    )
    module.anchor_set.mark_inactive(scale="fast", stream_mixture=("z_world",))
    return anchor


def _write_active(module, segment_id, z_world_fill, z_goal_snapshot):
    """Write one anchor and LEAVE IT ACTIVE, so the next write on the same
    (scale, stream_mixture) family is a real remap -- the event the SD-097
    write path listens to."""
    payload = AnchorGoalPayload(
        z_goal_snapshot=z_goal_snapshot.detach().clone(),
        wanting_strength=0.5,
        arousal_tag=0.3,
        last_vs=0.8,
        staleness_at_write=0.1,
        payload_written_step=0,
    )
    return module.anchor_set.write_anchor(
        scale="fast",
        segment_id=segment_id,
        stream_mixture=("z_world",),
        z_world=torch.full((1, 8), float(z_world_fill)),
        goal_payload=payload,
    )


def _seed_standard_pool(module):
    """One floor-clearing parent A, and two matched below-floor anchors
    B and C. B will get the edge; C is the control."""
    a = _seed(module, "0.0", 1.0, Z_GOAL)
    b = _seed(module, "0.1", 2.0, Z_ORTHOGONAL)
    c = _seed(module, "0.2", 3.0, Z_ORTHOGONAL)
    return a, b, c


def _rank_fingerprint(entries) -> List[Tuple[Any, ...]]:
    """Exact, comparable summary of a rank() result."""
    return [
        (
            e.anchor.key,
            e.ghost_priority,
            tuple(sorted(e.components.items())),
            e.relation_provenance,
        )
        for e in entries
    ]


def _ghost_seeds(module) -> List[Tuple[Any, float]]:
    """(anchor_key, goal_match) of every ghost probe in one deterministic
    propose_trajectories() call."""
    torch.manual_seed(1234)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8),
        z_self=torch.zeros(1, 8),
        current_z_goal=Z_GOAL,
    )
    return [
        (t.metadata["anchor_key"], float(t.metadata["goal_match"]))
        for t in candidates
        if t.metadata is not None
        and t.metadata.get("source") == "mech293_ghost_probe"
    ]


# ================================================================== #
# OFF: bit-identity                                                  #
# ================================================================== #

def test_o1_nothing_constructed_by_default():
    module, _ = _make_module(use_mech293=True)
    assert module.possibility_topology is None
    assert module.anchor_set.possibility_topology is None
    assert module.ghost_goal_bank.possibility_topology is None

    _seed_standard_pool(module)
    entries = module.rank_ghost_goals(Z_GOAL)
    assert entries, "expected the floor-clearing anchor to rank"
    for e in entries:
        assert e.relation_provenance is None
        assert "relation" not in e.components
    diag = module.ghost_goal_bank.get_diagnostics()
    assert [k for k in diag if k.startswith("sd097_")] == []
    assert "relation" not in diag["component_sums"]


def test_o2_disabled_topology_with_edges_is_bit_identical():
    """A topology attached but DISABLED, holding edges, must not perturb
    rank() by so much as a float bit."""
    module, _ = _make_module(use_mech293=True)
    a, b, c = _seed_standard_pool(module)

    baseline = _rank_fingerprint(module.rank_ghost_goals(Z_GOAL))
    baseline_diag = dict(module.ghost_goal_bank.get_diagnostics())

    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=False))
    topo.add_edge(a.key, b.key)
    topo.add_edge(a.key, c.key)
    module.attach_possibility_topology(topo)

    after = _rank_fingerprint(module.rank_ghost_goals(Z_GOAL))
    after_diag = dict(module.ghost_goal_bank.get_diagnostics())

    assert after == baseline
    assert sorted(after_diag.keys()) == sorted(baseline_diag.keys())
    assert after_diag == baseline_diag
    assert [k for k in after_diag if k.startswith("sd097_")] == []


def test_o3_remap_writer_silent_while_disabled():
    module, _ = _make_module()
    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=False))
    module.attach_possibility_topology(topo)

    _write_active(module, "0.0", 1.0, Z_GOAL)
    _write_active(module, "0.1", 2.0, Z_GOAL)
    assert topo.edge_count() == 0


def test_o4_ghost_probe_seeding_unchanged_when_disabled():
    """The OFF path the live SD-098 falsifier reads."""
    module, _ = _make_module(use_mech293=True)
    a, b, c = _seed_standard_pool(module)

    before = _ghost_seeds(module)
    assert before, "expected at least one ghost probe from the parent"

    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=False))
    topo.add_edge(a.key, b.key)
    module.attach_possibility_topology(topo)

    assert _ghost_seeds(module) == before


# ================================================================== #
# ON: the relation is READ                                           #
# ================================================================== #

def test_n1_edge_admits_below_floor_successor_control_stays_excluded():
    module, _ = _make_module()
    a, b, c = _seed_standard_pool(module)

    keys_before = {e.anchor.key for e in module.rank_ghost_goals(Z_GOAL)}
    assert a.key in keys_before
    assert b.key not in keys_before and c.key not in keys_before

    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=True))
    topo.add_edge(a.key, b.key)
    module.attach_possibility_topology(topo)

    keys_after = {e.anchor.key for e in module.rank_ghost_goals(Z_GOAL)}
    assert b.key in keys_after, (
        "the `enables` successor must be admitted -- if it is not, nothing "
        "reads the relation and the topology is inert"
    )
    # The matched control, equally below the floor but with no edge, is
    # still excluded: the EDGE is the only operative difference.
    assert c.key not in keys_after
    assert a.key in keys_after


def test_n2_relation_component_and_provenance_and_diagnostics():
    module, _ = _make_module()
    a, b, _c = _seed_standard_pool(module)
    topo = PossibilityTopology(PossibilityTopologyConfig(
        enabled=True, relation_weight=0.5
    ))
    topo.add_edge(a.key, b.key, weight=2.0)
    module.attach_possibility_topology(topo)

    entries = module.rank_ghost_goals(Z_GOAL)
    successor = [e for e in entries if e.anchor.key == b.key]
    assert len(successor) == 1
    entry = successor[0]

    assert entry.components["relation"] > 0.0
    prov = entry.relation_provenance
    assert prov is not None
    assert prov["relation"] == RELATION_ENABLES
    assert prov["parent_key"] == a.key
    assert prov["edge_weight"] == 2.0
    # relation_term == relation_weight * edge_weight * parent_goal_match
    parent = [e for e in entries if e.anchor.key == a.key][0]
    expected = 0.5 * 2.0 * float(prov["parent_goal_match"])
    assert entry.components["relation"] == pytest.approx(expected)
    assert float(prov["parent_goal_match"]) == pytest.approx(
        parent.components["goal_match"]
    )
    # The parent's own entry is untouched by the expansion.
    assert parent.relation_provenance is None
    assert "relation" not in parent.components

    diag = module.ghost_goal_bank.get_diagnostics()
    assert diag["sd097_relation"] == RELATION_ENABLES
    assert diag["sd097_n_relational_admitted"] == 1
    assert diag["sd097_n_edges_walked"] >= 1
    assert diag["sd097_topology_edges"] == 1
    assert "relation" in diag["component_sums"]


def test_n3_ghost_probe_seeding_changes_with_the_edge():
    """The behavioural difference: which regions the agent probes."""
    module, _ = _make_module(use_mech293=True)
    a, b, _c = _seed_standard_pool(module)

    before = _ghost_seeds(module)
    assert {k for k, _ in before} == {a.key}

    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=True))
    topo.add_edge(a.key, b.key)
    module.attach_possibility_topology(topo)

    after = _ghost_seeds(module)
    assert b.key in {k for k, _ in after}, (
        "ghost-probe seeding must reach the `enables` successor"
    )
    assert after != before
    # And the successor is seeded despite a goal_match of 0.0 -- it got
    # there through the relation, not through a direct match.
    assert dict(after)[b.key] == pytest.approx(0.0)


def test_n4_relation_type_is_respected_not_ignored():
    module, _ = _make_module()
    a, b, _c = _seed_standard_pool(module)
    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=True))
    topo.register_relation("requires", "dst cannot be reached without src")
    topo.add_edge(a.key, b.key, relation="requires")
    module.attach_possibility_topology(topo)

    keys = {e.anchor.key for e in module.rank_ghost_goals(Z_GOAL)}
    assert b.key not in keys, (
        "the `enables` consumer must not expand a `requires` edge"
    )
    assert topo.edge_count("requires") == 1


def test_n5_remap_write_path_records_observed_succession():
    module, _ = _make_module()
    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=True))
    module.attach_possibility_topology(topo)

    # Two successive writes on the SAME family: the second REMAPS the
    # first, which is the observed succession the writer records.
    first = _write_active(module, "0.0", 1.0, Z_GOAL)
    second = _write_active(module, "0.1", 2.0, Z_GOAL)
    assert first.active is False and second.active is True

    edge = topo.get_edge(first.key, second.key)
    assert edge is not None, "an observed family remap must record an edge"
    assert edge.relation == RELATION_ENABLES
    assert edge.provenance == PROVENANCE_ANCHOR_REMAP
    assert edge.observation_count == 1
    # Re-confirming the same succession updates, never duplicates.
    topo.note_transition(first.key, second.key, tick=9)
    assert topo.edge_count() == 1
    assert topo.get_edge(first.key, second.key).observation_count == 2
    # No self-loop is ever recorded.
    assert topo.note_transition(first.key, first.key) is None


def test_n6_admission_caps_are_enforced():
    module, _ = _make_module()
    a, b, c = _seed_standard_pool(module)
    topo = PossibilityTopology(PossibilityTopologyConfig(
        enabled=True, max_successors_per_parent=1
    ))
    topo.add_edge(a.key, b.key, weight=2.0)
    topo.add_edge(a.key, c.key, weight=1.0)
    module.attach_possibility_topology(topo)

    keys = {e.anchor.key for e in module.rank_ghost_goals(Z_GOAL)}
    # Strongest edge first, and only one per parent.
    assert b.key in keys and c.key not in keys
    assert module.ghost_goal_bank.get_diagnostics()[
        "sd097_n_relational_admitted"
    ] == 1


def test_n7_successor_outside_the_pool_is_counted_never_invented():
    module, _ = _make_module()
    a, _b, _c = _seed_standard_pool(module)
    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=True))
    topo.add_edge(a.key, ("fast", "9.9", ("z_world",)))
    module.attach_possibility_topology(topo)

    entries = module.rank_ghost_goals(Z_GOAL)
    assert {e.anchor.key for e in entries} == {a.key}
    diag = module.ghost_goal_bank.get_diagnostics()
    assert diag["sd097_n_successor_not_in_pool"] == 1
    assert diag["sd097_n_relational_admitted"] == 0


# ================================================================== #
# Registry / extensibility                                           #
# ================================================================== #

def test_r1_requires_and_is_part_of_reachable_without_schema_change():
    """One register_relation() call each; the store, the edge type and
    every query are unchanged."""
    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=True))
    assert topo.relations() == [RELATION_ENABLES]

    topo.register_relation("requires", "dst cannot be reached without src")
    topo.register_relation("is_part_of", "src is a component of dst")
    assert topo.relations() == ["enables", "is_part_of", "requires"]

    x = ("fast", "0.0", ("z_world",))
    y = ("fast", "0.1", ("z_world",))
    for rel in ("enables", "requires", "is_part_of"):
        edge = topo.add_edge(x, y, relation=rel)
        assert edge is not None and edge.relation == rel
        assert [e.dst for e in topo.successors(x, relation=rel)] == [y]
        assert [e.src for e in topo.predecessors(y, relation=rel)] == [x]
    assert topo.edge_count() == 3
    assert topo.stats()["edges_per_relation"] == {
        "enables": 1, "is_part_of": 1, "requires": 1,
    }


def test_r2_registry_refusals():
    topo = PossibilityTopology()
    with pytest.raises(KeyError):
        topo.add_edge(
            ("fast", "0.0", ("z_world",)),
            ("fast", "0.1", ("z_world",)),
            relation="requires",
        )
    with pytest.raises(ValueError):
        topo.register_relation("undescribed", "")
    topo.register_relation("requires", "dst cannot be reached without src")
    # Idempotent for an identical spec; refuses a conflicting one.
    topo.registry.register(
        RelationSpec(name="requires", description="dst cannot be reached without src")
    )
    with pytest.raises(ValueError):
        topo.register_relation("requires", "something else entirely")
    with pytest.raises(KeyError):
        topo.registry.get("is_part_of")


def test_r3_default_registry_ships_only_enables():
    assert default_registry().names() == [RELATION_ENABLES]


def test_r4_reset_follows_the_anchor_pool_by_default():
    module, _ = _make_module()
    topo = PossibilityTopology(PossibilityTopologyConfig(enabled=True))
    module.attach_possibility_topology(topo)
    _write_active(module, "0.0", 1.0, Z_GOAL)
    _write_active(module, "0.1", 2.0, Z_GOAL)
    assert topo.edge_count() == 1

    module.anchor_set.reset()
    assert topo.edge_count() == 0, (
        "segment_ids are recycled per episode; retained edges would alias"
    )


# ================================================================== #
# Config wiring (pre-verifies the pending config.py patch)           #
# ================================================================== #

def test_w1_flat_config_knobs_construct_and_attach_the_topology():
    """HippocampalModule builds the topology from FLAT HippocampalConfig
    knobs and attaches it to BOTH call sites.

    The knobs do not exist on HippocampalConfig yet (config.py was held by
    a concurrent session at build time), so they are set here the way the
    pending patch will set them. This test therefore verifies the wiring
    NOW and keeps verifying it once the fields land -- setattr on a
    dataclass instance and a real field are indistinguishable to the
    getattr reads in __init__. They are FLAT scalars on purpose:
    config.py cannot import PossibilityTopologyConfig (that module imports
    anchor_set, which imports config).
    """
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2FastPredictor, E2Config
    from ree_core.residue.field import ResidueField, ResidueConfig
    from ree_core.utils.config import (
        AnchorSetConfig, GhostGoalBankConfig, HippocampalConfig,
    )

    cfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=8, hidden_dim=32,
        horizon=4, num_candidates=8, num_cem_iterations=1,
        elite_fraction=0.25,
        use_anchor_sets=True,
        anchor_set=AnchorSetConfig(use_sd039_anchor_payload=True),
        use_mech292_ghost_bank=True,
        ghost_goal_bank_config=GhostGoalBankConfig(goal_match_floor=0.05),
    )
    cfg.use_possibility_topology = True
    cfg.possibility_topology_relation_weight = 0.25
    cfg.possibility_topology_max_successors_per_parent = 3
    cfg.possibility_topology_max_relational_admits = 5

    e2 = E2FastPredictor(E2Config(
        self_dim=8, world_dim=8, action_dim=4,
        action_object_dim=8, hidden_dim=32,
    ))
    rf = ResidueField(ResidueConfig(
        world_dim=8, hidden_dim=32, num_basis_functions=8,
    ))
    module = HippocampalModule(cfg, e2=e2, residue_field=rf)

    topo = module.possibility_topology
    assert topo is not None
    assert topo.config.enabled is True
    assert topo.config.relation_weight == 0.25
    assert topo.config.max_successors_per_parent == 3
    assert topo.config.max_relational_admits == 5
    assert topo.config.seed_relation == RELATION_ENABLES
    # Both call sites, not just one.
    assert module.anchor_set.possibility_topology is topo
    assert module.ghost_goal_bank.possibility_topology is topo


def test_w2_config_knob_without_anchor_sets_raises():
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2FastPredictor, E2Config
    from ree_core.residue.field import ResidueField, ResidueConfig
    from ree_core.utils.config import HippocampalConfig

    cfg = HippocampalConfig(
        world_dim=8, action_dim=4, action_object_dim=8, hidden_dim=32,
        horizon=4, num_candidates=8, num_cem_iterations=1,
        elite_fraction=0.25, use_anchor_sets=False,
    )
    cfg.use_possibility_topology = True
    e2 = E2FastPredictor(E2Config(
        self_dim=8, world_dim=8, action_dim=4,
        action_object_dim=8, hidden_dim=32,
    ))
    rf = ResidueField(ResidueConfig(
        world_dim=8, hidden_dim=32, num_basis_functions=8,
    ))
    with pytest.raises(ValueError):
        HippocampalModule(cfg, e2=e2, residue_field=rf)


# ================================================================== #
# SD-098 guard                                                       #
# ================================================================== #

def test_s1_no_stored_node_type_field_anywhere():
    """SD-098 tests whether goal-ness is computed at READ TIME. A stored
    node-type field is its comparison arm; SD-097 must not create one."""
    from ree_core.hippocampal.possibility_topology import TopologyEdge
    from ree_core.hippocampal.ghost_goal_bank import GhostGoalBankEntry

    banned = ("node_type", "goal_type", "possibility_type", "subgoal",
              "is_goal", "is_subgoal", "is_possibility")
    for cls in (Anchor, AnchorGoalPayload, TopologyEdge, GhostGoalBankEntry,
                PossibilityTopologyConfig):
        fields = set(getattr(cls, "__dataclass_fields__", {}).keys())
        for name in banned:
            assert name not in fields, (
                "%s.%s is a stored node-type field -- SD-097 stores "
                "RELATIONS only; a node type here pre-empts SD-098"
                % (cls.__name__, name)
            )
    # The relation registry is about EDGES; no relation may be a node tag.
    assert default_registry().names() == [RELATION_ENABLES]
