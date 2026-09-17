"""SD-097 -- typed, multi-relation possibility topology over AnchorKeys.

WHAT THIS IS
    A directed, typed edge store whose NODES are existing
    `AnchorKey = (scale, segment_id, stream_mixture)` tuples
    (ree_core/hippocampal/anchor_set.py) and whose EDGES carry a
    registered relation type. It ships exactly one relation, `enables`,
    with exactly one live consumer: ghost-probe seeding
    (GhostGoalBank.rank -> HippocampalModule._propose_ghost_seeded).

WHY AnchorKey AND NOT A NEW NODE OBJECT
    SD-097 as originally written proposed generalising "the existing
    parent/subgoal edge". That edge does not exist: goal.py couples
    _z_goal and _z_goal_parent by an EMA pull behind
    use_hierarchical_goal_credit -- two latent vectors, no node identity,
    no edge object. Three candidate node primitives existed (AnchorKey,
    SuperOrdinalGoalMemory slots, E2 action objects). AnchorKey is the
    only one that is ALREADY a hashable identity with a LIVE read-time
    consumer ranging over the same pool (GhostGoalBank ranks every anchor
    each call). Building on either of the other two would require
    inventing the consumer alongside the topology, i.e. inert substrate.

WHAT THIS DELIBERATELY IS NOT
    There is NO stored node-type field (no goal / subgoal / possibility
    tag on a node). SD-098 is a live falsifier testing whether goal-ness
    is computed at READ TIME against the existing GhostGoalBank under a
    mid-episode destination shift; a stored node-type field is precisely
    SD-098's comparison arm, so writing one here would pre-empt that
    experiment. This module stores RELATIONS between nodes and nothing
    about what a node IS. Every "is this a goal?" question stays a
    read-time computation in the bank, exactly as today.

    The build did not need a node-type field. Relational admission is
    keyed on the PARENT's read-time goal_match (computed, not stored)
    and on the edge; the successor's own nature is never consulted.

RELATION REGISTRY (extensibility without a schema change)
    The store is relation-generic: edges live in
    `relation -> src -> dst -> TopologyEdge` maps, and any registered
    name is storable and queryable. `enables` is registered by
    `default_registry()`. `requires` and `is_part_of` -- the first
    relations of the two ALTERNATIVE node primitives, which the user may
    still want -- are reachable with a single `register_relation()` call
    and no change to the store, the edge dataclass, or any query. That
    property is pinned by a contract test rather than asserted here.
    They are NOT pre-registered: a relation with no semantics and no
    consumer is the failure mode SD-097 was triaged for.

WRITE PATH (explicit, event-driven, NOT inferred from latents)
    One writer ships: OBSERVED SUCCESSION AT ANCHOR REMAP.
    `AnchorSet.write_anchor` already detects the case where anchor A was
    the active anchor of a (scale, stream_mixture) family and anchor B
    replaces it. That is an observed transition A -> B on the waking
    stream (MECH-094: write_anchor is only reached from
    HippocampalModule.tick_anchor_set via REEAgent.sense). It is recorded
    as `A enables B` with provenance "anchor_remap_succession" and a
    confirmation count. No similarity computation is involved anywhere --
    inferring edges from latent proximity would reintroduce the "it is
    not really an edge" problem one level up.

    `add_edge()` remains available for explicit external writers.

READ PATH / CONSUMER
    GhostGoalBank.rank() admits, in addition to the anchors that clear
    goal_match_floor on their own, the `enables` SUCCESSORS of those
    anchors -- waiving the floor for them, because their relevance
    arrives through the relation rather than through a direct goal-match.
    A successor is scored on its own four (or five) channels plus a
    RELATION channel `relation_weight * edge.weight * parent_goal_match`,
    then sorted into the same list. Since
    HippocampalModule._propose_ghost_seeded seeds its CEM probes from
    `rank()[:n_ghost]` anchors' z_world, an edge changes WHICH REGIONS
    THE AGENT PROBES. That is the behavioural difference.

    Depth is ONE hop. No transitive closure: a successor admitted by an
    edge does not itself expand.

NO-OP DEFAULT
    `PossibilityTopologyConfig.enabled` defaults to False, and a topology
    is only ever constructed when a caller asks for one. With no topology
    attached (the default everywhere), AnchorSet and GhostGoalBank take
    literally the same code paths as before -- no extra components keys,
    no extra diagnostics keys, no extra dict allocations in rank(). The
    SD-098 falsifier depends on that OFF path staying bit-identical.

Phased training: not applicable (no trainable parameters; pure bookkeeping).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ree_core.hippocampal.anchor_set import AnchorKey


# The one relation SD-097 ships with a consumer.
RELATION_ENABLES = "enables"

# Provenance string stamped on edges written by the anchor-remap succession
# writer (as opposed to an explicit external add_edge call).
PROVENANCE_ANCHOR_REMAP = "anchor_remap_succession"
PROVENANCE_EXPLICIT = "explicit"


@dataclass(frozen=True)
class RelationSpec:
    """Declaration of one relation type.

    Fields:
      name:        the relation key used in every store / query call.
      description: one line of semantics. Required -- an undescribed
                   relation is the "eight types with no semantics"
                   failure mode.
      directed:    documentation-only today (the store is always
                   directed); False marks a relation whose consumer is
                   expected to read it symmetrically.
      inverse_name: the name of the inverse relation when one exists
                   (e.g. a future "enabled_by"). Documentation-only; the
                   store does NOT auto-materialise inverses, since
                   predecessors() already answers the reverse query.
    """
    name: str
    description: str
    directed: bool = True
    inverse_name: Optional[str] = None


class RelationRegistry:
    """Mutable set of RelationSpecs. Per-instance, not a global."""

    def __init__(self, specs: Optional[List[RelationSpec]] = None) -> None:
        self._specs: Dict[str, RelationSpec] = {}
        for spec in (specs or []):
            self.register(spec)

    def register(self, spec: RelationSpec) -> RelationSpec:
        """Register a relation type. Idempotent for an identical spec;
        raises on a conflicting redefinition of the same name."""
        if not isinstance(spec, RelationSpec):
            raise TypeError("register() expects a RelationSpec")
        if not spec.name:
            raise ValueError("RelationSpec.name must be a non-empty string")
        if not spec.description:
            raise ValueError(
                "RelationSpec.description must be non-empty -- a relation "
                "with no stated semantics is not registerable"
            )
        existing = self._specs.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(
                "relation '%s' is already registered with a different spec"
                % spec.name
            )
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> RelationSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(
                "unknown relation '%s'; registered: %s"
                % (name, ", ".join(self.names()) or "(none)")
            )
        return spec

    def is_registered(self, name: str) -> bool:
        return name in self._specs

    def names(self) -> List[str]:
        return sorted(self._specs.keys())


def default_registry() -> RelationRegistry:
    """A fresh registry carrying only the relation SD-097 ships.

    `requires` and `is_part_of` are intentionally absent: each is
    reachable later with one register_relation() call and no schema
    change (see the module docstring and the extensibility contract
    test), and neither has a consumer today.
    """
    return RelationRegistry([
        RelationSpec(
            name=RELATION_ENABLES,
            description=(
                "src was occupied and dst became reachable from it -- an "
                "observed succession, read by ghost-probe seeding to admit "
                "dst as a probe target when src is goal-relevant"
            ),
            directed=True,
        ),
    ])


@dataclass
class TopologyEdge:
    """One typed directed edge between two AnchorKeys."""
    src: AnchorKey
    dst: AnchorKey
    relation: str
    weight: float = 1.0
    observation_count: int = 1
    created_at_tick: int = 0
    last_updated_tick: int = 0
    provenance: str = PROVENANCE_EXPLICIT


@dataclass
class PossibilityTopologyConfig:
    """SD-097 knobs. Every default is the OFF / minimal setting.

    enabled:
        Master switch. False -> the store accepts writes but no consumer
        reads it and the anchor-remap writer does not fire; with no
        topology attached at all (the default) nothing changes anywhere.
    seed_relation:
        Which relation the ghost-probe consumer expands. `enables` today;
        pointing it at a future `requires` needs no store change.
    relation_weight:
        Scale on the relational channel added to a successor's
        ghost_priority (relation_weight * edge.weight * parent_goal_match).
    max_successors_per_parent:
        Per-parent fan-out cap on relational admission (ordering is
        deterministic: strongest edge first, then key order).
    max_relational_admits:
        Global per-rank() cap, so a dense topology cannot swamp the
        directly-matching entries.
    write_on_anchor_remap:
        Whether AnchorSet records an `enables` edge on an observed
        family remap.
    max_edges:
        Hard cap on stored edges; new edges past the cap are refused and
        counted (stats()["n_refused_at_cap"]) rather than evicting an
        existing edge, so the store never silently loses a relation.
    reset_with_anchor_set:
        True (default) clears the topology when AnchorSet.reset() clears
        the anchor pool. Anchor segment_ids are per-episode counters
        ("outer.inner") and are RECYCLED across episodes, so retaining
        edges across a reset would alias a new episode's anchors onto an
        old episode's relations. Set False only in a setting where keys
        are known to be stable across the reset.
    """
    enabled: bool = False
    seed_relation: str = RELATION_ENABLES
    relation_weight: float = 0.5
    max_successors_per_parent: int = 2
    max_relational_admits: int = 8
    write_on_anchor_remap: bool = True
    max_edges: int = 4096
    reset_with_anchor_set: bool = True


class PossibilityTopology:
    """Typed directed relation store keyed by AnchorKey.

    Not a graph algorithms library: it stores edges, answers one-hop
    successor / predecessor queries, and counts what it holds. Depth,
    transitive closure and path search are deliberately absent -- the one
    consumer needs one hop, and an unread capability is the thing SD-097
    was triaged to avoid.
    """

    def __init__(
        self,
        config: Optional[PossibilityTopologyConfig] = None,
        registry: Optional[RelationRegistry] = None,
    ) -> None:
        self.config = config if config is not None else PossibilityTopologyConfig()
        self.registry = registry if registry is not None else default_registry()
        # relation -> src -> dst -> edge
        self._out: Dict[str, Dict[AnchorKey, Dict[AnchorKey, TopologyEdge]]] = {}
        # relation -> dst -> src -> edge (same edge objects, reverse index)
        self._in: Dict[str, Dict[AnchorKey, Dict[AnchorKey, TopologyEdge]]] = {}
        self._n_edges: int = 0
        self._n_refused_at_cap: int = 0
        self._n_updates: int = 0

    # ------------------------------------------------------------------ #
    # Registry passthrough                                               #
    # ------------------------------------------------------------------ #
    def register_relation(
        self,
        name: str,
        description: str,
        directed: bool = True,
        inverse_name: Optional[str] = None,
    ) -> RelationSpec:
        """Make a new relation type storable. No schema change required:
        the edge dataclass, the indexes and every query are relation-
        generic. This is the whole of what `requires` / `is_part_of`
        would need."""
        return self.registry.register(
            RelationSpec(
                name=name,
                description=description,
                directed=directed,
                inverse_name=inverse_name,
            )
        )

    def relations(self) -> List[str]:
        return self.registry.names()

    # ------------------------------------------------------------------ #
    # Write path                                                         #
    # ------------------------------------------------------------------ #
    def add_edge(
        self,
        src: AnchorKey,
        dst: AnchorKey,
        relation: str = RELATION_ENABLES,
        weight: Optional[float] = None,
        tick: int = 0,
        provenance: str = PROVENANCE_EXPLICIT,
    ) -> Optional[TopologyEdge]:
        """Record (or re-confirm) one typed directed edge.

        Re-confirming an existing (src, dst, relation) increments
        observation_count and refreshes last_updated_tick; an explicit
        `weight` overwrites the stored weight, and omitting it leaves the
        stored weight alone (so a repeat observation never silently
        rewrites a hand-set weight).

        Returns the edge, or None when the write was refused: a self-loop
        (src == dst) or the max_edges cap. Refusals are counted in
        stats(); an unregistered relation RAISES instead, because that is
        a caller bug rather than a capacity condition.
        """
        if not self.registry.is_registered(relation):
            raise KeyError(
                "unknown relation '%s'; register it first via "
                "register_relation()" % relation
            )
        src_key = _normalise_key(src)
        dst_key = _normalise_key(dst)
        if src_key == dst_key:
            return None

        out_rel = self._out.setdefault(relation, {})
        by_src = out_rel.setdefault(src_key, {})
        existing = by_src.get(dst_key)
        if existing is not None:
            existing.observation_count += 1
            existing.last_updated_tick = int(tick)
            if weight is not None:
                existing.weight = float(weight)
            self._n_updates += 1
            return existing

        if self.config.max_edges > 0 and self._n_edges >= self.config.max_edges:
            self._n_refused_at_cap += 1
            return None

        edge = TopologyEdge(
            src=src_key,
            dst=dst_key,
            relation=relation,
            weight=1.0 if weight is None else float(weight),
            observation_count=1,
            created_at_tick=int(tick),
            last_updated_tick=int(tick),
            provenance=str(provenance),
        )
        by_src[dst_key] = edge
        self._in.setdefault(relation, {}).setdefault(dst_key, {})[src_key] = edge
        self._n_edges += 1
        return edge

    def note_transition(
        self,
        src: AnchorKey,
        dst: AnchorKey,
        tick: int = 0,
    ) -> Optional[TopologyEdge]:
        """The anchor-remap succession writer.

        Called by AnchorSet.write_anchor when anchor `src` was the active
        anchor of a (scale, stream_mixture) family and `dst` replaces it:
        an OBSERVED transition on the waking stream. Silent no-op when
        the topology is disabled, so the live call site costs one boolean
        test when the lever is off.
        """
        if not self.config.enabled:
            return None
        return self.add_edge(
            src,
            dst,
            relation=RELATION_ENABLES,
            tick=tick,
            provenance=PROVENANCE_ANCHOR_REMAP,
        )

    # ------------------------------------------------------------------ #
    # Read path                                                          #
    # ------------------------------------------------------------------ #
    def successors(
        self,
        src: AnchorKey,
        relation: str = RELATION_ENABLES,
    ) -> List[TopologyEdge]:
        """One-hop outgoing edges, strongest first then by key.

        Deterministic ordering matters: the ghost-probe consumer takes
        the first `max_successors_per_parent` of this list, so the seed
        set must not depend on dict insertion history.
        """
        by_src = self._out.get(relation, {}).get(_normalise_key(src))
        if not by_src:
            return []
        return sorted(
            by_src.values(),
            key=lambda e: (-float(e.weight), -int(e.observation_count), _sort_key(e.dst)),
        )

    def predecessors(
        self,
        dst: AnchorKey,
        relation: str = RELATION_ENABLES,
    ) -> List[TopologyEdge]:
        """One-hop incoming edges. The reverse query a future `requires`
        consumer wants; present because it is the same index, not because
        anything reads it today."""
        by_dst = self._in.get(relation, {}).get(_normalise_key(dst))
        if not by_dst:
            return []
        return sorted(
            by_dst.values(),
            key=lambda e: (-float(e.weight), -int(e.observation_count), _sort_key(e.src)),
        )

    def get_edge(
        self,
        src: AnchorKey,
        dst: AnchorKey,
        relation: str = RELATION_ENABLES,
    ) -> Optional[TopologyEdge]:
        return self._out.get(relation, {}).get(
            _normalise_key(src), {}
        ).get(_normalise_key(dst))

    def has_edge(
        self,
        src: AnchorKey,
        dst: AnchorKey,
        relation: str = RELATION_ENABLES,
    ) -> bool:
        return self.get_edge(src, dst, relation) is not None

    def edge_count(self, relation: Optional[str] = None) -> int:
        if relation is None:
            return int(self._n_edges)
        return sum(
            len(by_dst) for by_dst in self._out.get(relation, {}).values()
        )

    def stats(self) -> Dict[str, Any]:
        """Diagnostics snapshot. Read-only; allocates a fresh dict."""
        return {
            "n_edges": int(self._n_edges),
            "n_refused_at_cap": int(self._n_refused_at_cap),
            "n_updates": int(self._n_updates),
            "relations": self.relations(),
            "edges_per_relation": {
                rel: self.edge_count(rel) for rel in sorted(self._out.keys())
            },
            "enabled": bool(self.config.enabled),
        }

    def reset(self) -> None:
        """Clear all edges. Counters for refusals / updates reset too."""
        self._out.clear()
        self._in.clear()
        self._n_edges = 0
        self._n_refused_at_cap = 0
        self._n_updates = 0


# ---------------------------------------------------------------------- #
# Key helpers                                                            #
# ---------------------------------------------------------------------- #
def _normalise_key(key: AnchorKey) -> AnchorKey:
    """Coerce an AnchorKey to its canonical hashable form.

    AnchorSet builds keys as (scale, segment_id, tuple(stream_mixture));
    a caller holding a list-valued mixture would otherwise produce an
    unhashable or non-equal key silently.
    """
    if not isinstance(key, tuple) or len(key) != 3:
        raise TypeError(
            "AnchorKey must be a 3-tuple (scale, segment_id, stream_mixture); "
            "got %r" % (type(key).__name__,)
        )
    scale, segment_id, mixture = key
    return (str(scale), str(segment_id), tuple(mixture))


def _sort_key(key: AnchorKey) -> Tuple[str, str, Tuple[str, ...]]:
    return (str(key[0]), str(key[1]), tuple(str(s) for s in key[2]))
