"""Preservation Increment-2 DE-RISKING SPIKE (not the increment itself).

Scope, deliberately: this file measures the per-store cost of mid-life
snapshot/resume and pins the failure modes that make a naive resume WRONG. It
adds NO capture/restore code to the substrate -- every restore helper here is
local to the test, so the spike costs the fleet nothing and can be deleted or
promoted wholesale once the Increment-2 go/no-go gate is decided
(REE_assembly/evidence/planning/preservation_snapshot_plan.md, "Increment 2").

Why a spike at all: `REEAgent` has no `state_dict`/`load_state_dict` override,
so the default `nn.Module.state_dict` captures registered Parameters/buffers
ONLY and silently drops every non-parametric store. Increment 2 is therefore
per-module contract work, and it must be costed before it is started.

S1  cheapest store round-trips bit-identically  (SuperOrdinalGoalMemory)
S2  ATTRIBUTE CENSUS -- the state-vs-telemetry fork, made mechanical
S3  the fork's cost: telemetry is NOT restored, by design, and that is pinned
S4  `centering` is the trap the fork hides: behaviour-bearing, config-derived,
    absent from state_dict -> a cross-config restore is silently wrong
S5  agent-level round-trip through two REEAgent instances (the real seam)
S6  WORST CASE A -- ResidueField EWC anchor: state_dict drops it SILENTLY and
    the resumed organism loses its critical-period write-protection with no
    error at all
S7  WORST CASE A, restore: a 4-key capture/restore dict is sufficient
S8  WORST CASE B -- GatedPolicy crystallization: lazy `expansion` makes load
    fail LOUDLY, and the `requires_grad=False` freeze is lost SILENTLY even
    when the load is forced through

The load-bearing assertions are S2, S4, S6 and S8: each pins a way a resume can
be wrong WITHOUT raising. S1/S5/S7 pin that the happy paths do work, which is
what makes the cost estimate in the plan doc real rather than assumed.
"""
from __future__ import annotations

import copy

import pytest
import torch

from ree_core.goal import GoalConfig, SuperOrdinalGoalMemory


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _store(context_dim: int = 4, **kw) -> SuperOrdinalGoalMemory:
    cfg = GoalConfig(goal_dim=4, use_super_ordinal_goal_anchors=True, **kw)
    return SuperOrdinalGoalMemory(
        cfg, context_dim=context_dim, device=torch.device("cpu")
    )


def _onehot(dim: int, i: int, eps: float = 0.0) -> torch.Tensor:
    """A [1, dim] near-one-hot row. `eps` perturbs it into a near-duplicate."""
    v = torch.zeros(1, dim, dtype=torch.float32)
    v[0, i] = 1.0 - eps
    v[0, (i + 1) % dim] = eps
    return v


def _probe(s: SuperOrdinalGoalMemory) -> torch.Tensor:
    """The query used to compare retrieval behaviour across a round-trip."""
    return _onehot(s.context_dim, 0)


def _mutate(s: SuperOrdinalGoalMemory) -> None:
    """Drive the store off its zero-init into a non-trivial live state.

    Dimension-aware on purpose: the standalone fixtures use a 4-d context but
    a real REEAgent builds this store at the live z_world/z_goal widths (S5),
    and a hardcoded width silently mis-sizes there.

    Deliberately exercises BOTH write branches (allocate a fresh slot, then
    reinforce a near-duplicate) so the round-trip is not vacuously comparing
    two zero tensors -- the failure mode a naive round-trip test has.
    """
    cd, gd = s.context_dim, s.goal_dim
    s.write(_onehot(cd, 0), _onehot(gd, 0, 0.1), salience=1.0,
            context_complexity=1.0)
    s.write(_onehot(cd, 1), _onehot(gd, 1, 0.2), salience=1.0,
            context_complexity=1.0)
    # near-duplicate of slot 0 -> reinforce rather than allocate
    s.write(_onehot(cd, 0, 0.01), _onehot(gd, 0, 0.3), salience=1.0,
            context_complexity=1.0)
    s.retrieve(_probe(s))
    s.note_seed()


# --------------------------------------------------------------------------- #
# S1 -- cheapest store round-trips bit-identically                             #
# --------------------------------------------------------------------------- #

def test_s1_super_ordinal_goal_memory_round_trips_bit_identically():
    src = _store()
    _mutate(src)
    assert src.n_occupied() >= 2, "mutation did not populate the store"
    assert src._strength.max().item() > 0.0

    sd = copy.deepcopy(src.state_dict())

    dst = _store()
    assert dst.n_occupied() == 0
    dst.load_state_dict(sd)

    for name in ("_keys", "_values", "_strength", "_occupied"):
        a, b = getattr(src, name), getattr(dst, name)
        assert torch.equal(a, b), f"{name} not bit-identical after round-trip"
    assert dst.write_enabled == src.write_enabled
    assert dst.n_occupied() == src.n_occupied()

    # Behavioural equality, not just field equality: the restored store must
    # answer the same query the same way (this is what "resume" actually means).
    q = _probe(src)
    ra, rb = src.retrieve(q), dst.retrieve(q)
    assert ra is not None and rb is not None
    assert torch.equal(ra[0], rb[0])
    assert ra[1] == rb[1]
    assert ra[2] == rb[2]


# --------------------------------------------------------------------------- #
# S2 -- the attribute census: the state-vs-telemetry fork made mechanical       #
# --------------------------------------------------------------------------- #

# Every instance attribute of SuperOrdinalGoalMemory, partitioned by the fork
# the plan doc names. A NEW attribute added to the store fails this test until
# someone classifies it -- which is the whole point: the silent-fidelity-bug
# risk in Increment 2 is an attribute being added and nobody deciding.
_RESTORED = {           # load-bearing, present in state_dict
    "_keys", "_values", "_strength", "_occupied", "write_enabled", "_baseline",
}
_CONFIG_DERIVED = {     # rebuilt by the constructor from config; see S4
    "config", "device", "context_dim", "goal_dim", "n_slots", "centering",
}
_TELEMETRY = {          # deliberately dropped; see S3
    "_n_writes", "_n_reinforce", "_n_allocate", "_n_seeds",
    "_last_complexity", "_last_salience", "_last_seed_match",
}


def test_s2_attribute_census_is_complete_and_disjoint():
    s = _store()
    _mutate(s)
    live = set(vars(s).keys())
    classified = _RESTORED | _CONFIG_DERIVED | _TELEMETRY

    unclassified = live - classified
    assert not unclassified, (
        "SuperOrdinalGoalMemory grew attribute(s) with no snapshot "
        f"classification: {sorted(unclassified)}. Decide restore-vs-telemetry "
        "and add it to the right set (preservation Increment 2)."
    )
    stale = classified - live
    assert not stale, f"census names attribute(s) that no longer exist: {sorted(stale)}"

    # partitions must not overlap
    assert not (_RESTORED & _CONFIG_DERIVED)
    assert not (_RESTORED & _TELEMETRY)
    assert not (_CONFIG_DERIVED & _TELEMETRY)


def test_s2b_state_dict_covers_exactly_the_restored_set():
    s = _store()
    keys = set(s.state_dict().keys())
    # state_dict strips the leading underscore; normalise before comparing.
    assert {k.lstrip("_") for k in _RESTORED} == keys, (
        "state_dict and the _RESTORED census have drifted -- one of them is "
        "wrong, and a mid-life resume would silently lose or invent state."
    )


# --------------------------------------------------------------------------- #
# S3 -- the fork's cost, pinned: telemetry does not survive a resume            #
# --------------------------------------------------------------------------- #

def test_s3_telemetry_is_not_restored_by_design():
    src = _store()
    _mutate(src)
    assert src._n_writes > 0 and src._n_seeds > 0
    assert src._last_salience >= 0.0

    dst = _store()
    dst.load_state_dict(copy.deepcopy(src.state_dict()))

    # A resumed organism's counters restart at zero. This is the DELIBERATE
    # side of the fork -- pinned so a later reader does not mistake it for a
    # bug, and so any analysis that reads these counters across a resume
    # boundary knows it must not.
    assert dst._n_writes == 0
    assert dst._n_reinforce == 0
    assert dst._n_allocate == 0
    assert dst._n_seeds == 0
    assert dst._last_complexity == -1.0
    assert dst._last_salience == -1.0
    assert dst._last_seed_match == -1.0


# --------------------------------------------------------------------------- #
# S4 -- the trap inside the fork: config-derived does NOT mean safe to drop     #
# --------------------------------------------------------------------------- #

def test_s4_centering_is_behaviour_bearing_and_absent_from_state_dict():
    """`centering` reads as config-derived (so state_dict omits it) but it
    GATES the cue-key arithmetic. Restoring a centered snapshot into an agent
    built with centering OFF is therefore silently wrong -- no key error, no
    shape error, just different retrieval geometry.

    This is the shape of the fidelity bug Increment 2 has to defend against
    ~10 times over, and it is why every store's restore needs a config-identity
    check alongside its round-trip contract.
    """
    on = _store(super_ordinal_cue_centering=True)
    assert on.centering is True
    _mutate(on)
    assert on._baseline is not None, "centering ON must seed a baseline"

    sd = copy.deepcopy(on.state_dict())
    assert "centering" not in sd

    off = _store(super_ordinal_cue_centering=False)
    assert off.centering is False
    off.load_state_dict(sd)          # accepted without complaint

    # The baseline tensor IS restored, but `centering` is not -- so the
    # restored store holds a live baseline it will never apply.
    assert off._baseline is not None
    assert off.centering is False

    q = _onehot(on.context_dim, 0, 0.05)
    m_on = on.retrieve(q)
    m_off = off.retrieve(q)
    assert m_on is not None and m_off is not None
    # Same anchor bank, same query, different match -- silently.
    assert m_on[1] != m_off[1], (
        "expected the centered and uncentered stores to disagree; if this "
        "ever passes trivially the fixture no longer exercises centering"
    )


# --------------------------------------------------------------------------- #
# S5 -- the real seam: agent -> snapshot -> fresh agent                        #
# --------------------------------------------------------------------------- #

def _agent_with_store():
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=17, world_obs_dim=250, action_dim=4, z_goal_enabled=True
    )
    cfg.goal.use_super_ordinal_goal_anchors = True
    a = REEAgent(cfg)
    return a


def test_s5_agent_level_round_trip():
    src = _agent_with_store()
    if src.super_ordinal_goal_memory is None:
        pytest.skip("super-ordinal store not constructed under this config")

    _mutate(src.super_ordinal_goal_memory)
    assert src.super_ordinal_goal_memory.n_occupied() >= 2

    sd = copy.deepcopy(src.super_ordinal_goal_memory.state_dict())

    dst = _agent_with_store()
    assert dst.super_ordinal_goal_memory is not None
    assert dst.super_ordinal_goal_memory.n_occupied() == 0

    # The Increment-2 "whole-organism walker" is exactly this line, x ~10
    # stores, plus env + RNG. Cost of THIS store, end to end: one line.
    dst.super_ordinal_goal_memory.load_state_dict(sd)

    for name in ("_keys", "_values", "_strength", "_occupied"):
        assert torch.equal(
            getattr(src.super_ordinal_goal_memory, name),
            getattr(dst.super_ordinal_goal_memory, name),
        ), f"{name} diverged across the agent-level round-trip"

    # nn.Module.state_dict on the AGENT does not carry this store at all --
    # the premise of Increment 2, asserted rather than assumed.
    agent_keys = set(src.state_dict().keys())
    assert not [k for k in agent_keys if "super_ordinal" in k], (
        "agent.state_dict() unexpectedly carries the super-ordinal store; if "
        "this fires, the per-store walker may no longer be necessary"
    )


# --------------------------------------------------------------------------- #
# S6/S7 -- WORST CASE A: ResidueField EWC anchor (capture exists, no loader)    #
# --------------------------------------------------------------------------- #

def _field_with_anchor():
    from ree_core.residue.field import ResidueConfig, ResidueField

    cfg = ResidueConfig(world_dim=4, num_basis_functions=4)
    cfg.ewc_enabled = True
    cfg.ewc_lambda = 1.0
    f = ResidueField(cfg)
    f.ewc_enabled = True
    f._ewc_lambda = 1.0
    with torch.no_grad():
        f.rbf_field.weights.copy_(torch.linspace(0.5, 2.0, f.rbf_field.weights.numel())
                                  .reshape(f.rbf_field.weights.shape))
        if hasattr(f.rbf_field, "active_mask"):
            f.rbf_field.active_mask.fill_(True)
    f.snapshot_ewc_anchor()
    # Move the live params off the anchor so the penalty is non-zero.
    with torch.no_grad():
        f.rbf_field.weights.add_(0.25)
        f.rbf_field.centers.add_(0.1)
    return f


def test_s6_ewc_anchor_is_dropped_silently_by_state_dict():
    """The worst case is not that the load CRASHES -- it is that it does not.

    A resumed organism silently loses its MECH-334 critical-period
    write-protection: `ewc_penalty()` returns exactly 0.0 because
    `_ewc_anchored` is False, so training carries on overwriting established
    basins with no error, no warning, and a plausible-looking loss curve.
    """
    src = _field_with_anchor()
    assert src.ewc_anchored is True
    p_src = float(src.ewc_penalty().detach())
    assert p_src > 0.0, "fixture failed to produce a live EWC penalty"

    sd = copy.deepcopy(src.state_dict())
    assert not [k for k in sd if "ewc" in k.lower()], (
        "an ewc_* key appeared in state_dict -- if the anchor became a "
        "registered buffer, this spike's worst case is solved and the "
        "Increment-2 cost estimate should be revised down"
    )

    from ree_core.residue.field import ResidueConfig, ResidueField
    cfg = ResidueConfig(world_dim=4, num_basis_functions=4)
    dst = ResidueField(cfg)
    dst.ewc_enabled = True
    dst._ewc_lambda = 1.0
    dst.load_state_dict(sd)          # succeeds, strict=True, no complaint

    assert dst.ewc_anchored is False
    assert float(dst.ewc_penalty().detach()) == 0.0, (
        "the silent-loss failure mode did not reproduce; re-derive the "
        "Increment-2 worst case before trusting the cost estimate"
    )


# The whole restore contract for this store, for cost-estimation purposes.
_EWC_KEYS = ("_ewc_anchored", "_ewc_centers_anchor", "_ewc_weights_anchor",
             "_ewc_fisher")


def _capture_ewc(f) -> dict:
    return {k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in ((k, getattr(f, k)) for k in _EWC_KEYS)}


def _restore_ewc(f, d: dict) -> None:
    for k in _EWC_KEYS:
        v = d[k]
        setattr(f, k, v.clone() if torch.is_tensor(v) else v)


def test_s7_ewc_anchor_capture_restore_is_four_keys():
    """Sizing the tail risk: the fix is a 4-key dict and two helpers.

    Kept LOCAL to the test on purpose -- the spike must not put new
    capture/restore code on the executable-code plane the fleet pulls before
    the Increment-2 go/no-go gate is decided.
    """
    src = _field_with_anchor()
    p_src = float(src.ewc_penalty().detach())
    assert p_src > 0.0

    from ree_core.residue.field import ResidueConfig, ResidueField
    cfg = ResidueConfig(world_dim=4, num_basis_functions=4)
    dst = ResidueField(cfg)
    dst.ewc_enabled = True
    dst._ewc_lambda = 1.0
    dst.load_state_dict(copy.deepcopy(src.state_dict()))
    _restore_ewc(dst, _capture_ewc(src))

    assert dst.ewc_anchored is True
    for k in _EWC_KEYS[1:]:
        assert torch.equal(getattr(src, k), getattr(dst, k)), f"{k} diverged"
    # Bit-equal penalty is the behavioural proof, not just field equality.
    assert float(dst.ewc_penalty().detach()) == p_src


def test_s7b_harm_history_is_dropped_too():
    """Same field, second dropped store -- `_harm_history` is a plain
    `List[Tensor]` alongside two registered buffers that DO survive. The
    asymmetry is invisible at the call site, which is why the per-store census
    (S2) rather than eyeballing is the Increment-2 method."""
    from ree_core.residue.field import ResidueConfig, ResidueField

    cfg = ResidueConfig(world_dim=4, num_basis_functions=4)
    src = ResidueField(cfg)
    src._harm_history.append(torch.ones(3))
    src.total_residue += 5.0

    sd = copy.deepcopy(src.state_dict())
    dst = ResidueField(cfg)
    dst.load_state_dict(sd)

    assert float(dst.total_residue) == float(src.total_residue)  # buffer: kept
    assert dst._harm_history == []                               # plain attr: lost
    assert len(src._harm_history) == 1


# --------------------------------------------------------------------------- #
# S8 -- WORST CASE B: GatedPolicy crystallization (lazy submodule)              #
# --------------------------------------------------------------------------- #

def _policy(crystallize_enabled: bool = True):
    from ree_core.policy.gated_policy import GatedPolicy, GatedPolicyConfig

    cfg = GatedPolicyConfig(use_gated_policy=True,
                            crystallize_enabled=crystallize_enabled)
    return GatedPolicy(world_dim=8, self_dim=4, harm_a_dim=4, config=cfg)


def test_s8_crystallized_policy_fails_loudly_then_silently():
    torch.manual_seed(0)
    src = _policy()
    info = src.crystallize()
    assert info["crystallized"] is True
    assert src.expansion is not None
    assert info["n_frozen_params"] > 0

    sd = copy.deepcopy(src.state_dict())
    assert [k for k in sd if k.startswith("expansion.")], (
        "expansion params missing from state_dict -- fixture did not crystallize"
    )

    # (a) LOUD: the lazy submodule does not exist on a fresh policy, so a
    #     strict load rejects the snapshot outright.
    dst = _policy()
    assert dst.expansion is None
    with pytest.raises(RuntimeError):
        dst.load_state_dict(sd)

    # (b) SILENT, and this is the dangerous half: force the load through by
    #     rebuilding the submodule first (the naive "fix"), and the parameter
    #     FREEZE -- the actual MECH-334 write-protect semantics -- is still
    #     gone, because requires_grad is not part of state_dict at all.
    dst2 = _policy()
    dst2.crystallize()               # rebuild `expansion` so the keys line up
    dst2.load_state_dict(sd)         # now succeeds
    assert dst2._crystallized is True

    frozen_src = [p.requires_grad for p in src.discriminator.parameters()]
    assert frozen_src and not any(frozen_src), "fixture: source is not frozen"

    fresh = _policy()
    assert all(p.requires_grad for p in fresh.discriminator.parameters()), (
        "a fresh policy is expected to be trainable; if not, the silent-thaw "
        "failure mode below is not being exercised"
    )
    # `requires_grad` survived here only because crystallize() was re-run by
    # hand. A restore that merely loads tensors -- the obvious implementation
    # -- leaves the heads trainable and the crystallization silently undone.
    naive = _policy()
    naive.expansion = copy.deepcopy(src.expansion)
    naive.load_state_dict(sd)
    assert naive._crystallized is False, (
        "_crystallized is a plain attr and must not survive a tensor-only load"
    )
    assert all(p.requires_grad for p in naive.discriminator.parameters()), (
        "expected the naive restore to leave the discriminator trainable -- "
        "this IS the silent failure; if it fires, re-derive worst case B"
    )
