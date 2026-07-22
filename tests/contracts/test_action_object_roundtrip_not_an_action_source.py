"""Contracts for the action-object round-trip defect and the CEM elite-pool floor.

Two independent substrate defects, confirmed 2026-07-22 while authoring
V3-EXQ-800/801 and reproduced independently here.

DEFECT 1 -- the a -> E2.action_object(a) -> action_object_decoder round trip is
not invertible, so `argmax(action_object_decoder(traj.get_action_object_sequence()
[:, 0, :]))` is NOT the candidate's action. It collapses to one constant class,
which makes any driver selecting through it inert under every manipulation of the
candidate set or its scores.

The decoder is NOT the degenerate part -- fed N(0,1) inputs it spans all classes.
The degeneracy is in its INPUT: E2's step-0 action-object embedding is very nearly
action-invariant, so the round trip discards the action. Pinning both halves
separately is the point of these tests: a future session that "fixes" this by
training the decoder would be treating the wrong component.

MEASURED (untrained module, action_dim=5, 32 candidates, seed 42):

  arm                     traj.actions[:,0]   re-decoded ao_0
  default (SP-CEM on)     2 classes {0,3}     1 class {3}
  SP-CEM off              1 class  {3}        1 class {3}
  action-class scaffold   5 classes {0..4}    1 class {3}

and on a TRAINED agent (CausalGridWorldV2 size=6, 40 warmup episodes, seed 42,
alpha_world=0.9, self_dim=world_dim=32): distinct re-decoded classes = min 1,
mean 1.00, max 1 of 5, over 121 / 121 / 679 ticks for default / SP-CEM /
scaffold respectively. Training does not repair it.

DEFECT 2 -- num_elite < 2 poisons the CEM with NaN. torch's default std() is
unbiased, so std over a single elite is NaN; it propagates through `+ 1e-6` and
through the SP-CEM `torch.clamp` floor, reaching E3 as
`RuntimeError: probability tensor contains either inf, nan or element < 0` from
`torch.multinomial`. Reached silently at num_candidates=8 with the DEFAULT
elite_fraction=0.2, since int(8 * 0.2) == 1.
"""

from __future__ import annotations

import pytest
import torch


def _make_module(action_dim: int = 5, num_candidates: int = 32, **kw):
    from ree_core.hippocampal.module import HippocampalModule
    from ree_core.predictors.e2_fast import E2Config, E2FastPredictor
    from ree_core.residue.field import ResidueConfig, ResidueField
    from ree_core.utils.config import HippocampalConfig

    cfg = HippocampalConfig(
        world_dim=8,
        action_dim=action_dim,
        action_object_dim=8,
        hidden_dim=32,
        horizon=4,
        num_candidates=num_candidates,
        num_cem_iterations=2,
        **kw,
    )
    e2 = E2FastPredictor(E2Config(
        self_dim=8,
        world_dim=8,
        action_dim=action_dim,
        action_object_dim=8,
        hidden_dim=32,
    ))
    residue = ResidueField(ResidueConfig(
        world_dim=8,
        hidden_dim=32,
        num_basis_functions=8,
    ))
    return HippocampalModule(cfg, e2=e2, residue_field=residue)


def _roundtrip_classes(module, candidates):
    out = []
    for traj in candidates:
        ao_seq = traj.get_action_object_sequence()
        if ao_seq is None or ao_seq.shape[1] == 0:
            continue
        logits = module.action_object_decoder(ao_seq[:, 0, :])
        out.append(int(logits.argmax(dim=-1).flatten()[0].item()))
    return out


# --------------------------------------------------------------------------
# DEFECT 1: the round trip is not an action source
# --------------------------------------------------------------------------

def test_decoder_itself_has_full_output_support():
    """The DECODER is not the degenerate component -- localises the defect.

    If this ever fails, the diagnosis in this module is wrong and the round-trip
    tests below need re-deriving before anything is built on them.
    """
    torch.manual_seed(42)
    module = _make_module()
    probe = torch.randn(512, module.config.action_object_dim)
    with torch.no_grad():
        classes = set(module.action_object_decoder(probe).argmax(dim=-1).tolist())
    assert len(classes) == module.config.action_dim, (
        f"decoder spans {len(classes)} of {module.config.action_dim} classes on "
        f"N(0,1) inputs; the round-trip collapse was attributed to its INPUT "
        f"(a near action-invariant action-object embedding), not to the decoder"
    )


def test_scaffold_candidates_dissociate_true_action_from_roundtrip():
    """The decisive case: 5 distinct CONSTRUCTED first actions, 1 re-decoded class.

    Action-class scaffold candidates are built with one distinct one-hot first
    action per class, bypassing the decoder entirely. So `traj.actions` spans
    every class by construction, while the round trip through their E2
    action-object embeddings still collapses.
    """
    torch.manual_seed(42)
    module = _make_module(use_action_class_scaffold_candidates=True)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8),
    )

    true_classes = {
        module.candidate_first_action_class(c) for c in candidates
    } - {None}
    round_classes = set(_roundtrip_classes(module, candidates))

    # The scaffold puts every class into the candidate set...
    assert true_classes == set(range(module.config.action_dim))
    # ...and the round trip throws that information away.
    assert len(round_classes) < len(true_classes), (
        "the a -> action_object -> decoder round trip recovered more action "
        "classes than the documented collapse. If E2's action-object embedding "
        "has become action-discriminative this is GOOD NEWS, but the "
        "deprecation of the round trip as a selection path (and the "
        "validate_experiments action_object_selection lint) were justified by "
        "it being inert -- re-derive both before relying on the decoder path, "
        "and revisit EXQ-196 / ARC-018."
    )


def test_candidate_first_action_class_reads_the_real_action():
    """The sanctioned accessor returns what the candidate actually executes."""
    torch.manual_seed(42)
    module = _make_module(use_action_class_scaffold_candidates=True)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8),
    )
    for traj in candidates:
        cls = module.candidate_first_action_class(traj)
        assert cls is not None
        assert cls == int(traj.actions[:, 0, :].argmax(dim=-1).flatten()[0].item())


def test_candidate_first_action_class_tolerates_actionless_input():
    module = _make_module()

    class _Empty:
        actions = None

    class _Degenerate:
        actions = torch.zeros(1, 0, 5)

    assert module.candidate_first_action_class(_Empty()) is None
    assert module.candidate_first_action_class(_Degenerate()) is None


def test_roundtrip_recovery_diagnostic_is_emitted_and_flags_the_collapse():
    """The defect must be visible in every run's diagnostics, not only in probes."""
    torch.manual_seed(42)
    module = _make_module(use_action_class_scaffold_candidates=True)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8),
    )
    diags = module.get_last_propose_diagnostics()
    assert "action_object_roundtrip_recovery" in diags

    rt = diags["action_object_roundtrip_recovery"]
    assert rt["n_scored"] == len(candidates)
    assert rt["true_unique_classes"] == module.config.action_dim
    # The signature of an inert selection path: the candidate set spans several
    # true classes but the round trip maps them all to one.
    assert rt["roundtrip_unique_classes"] == 1
    assert rt["recovery_rate"] < 1.0

    # recovery_rate ALONE is a trap and must not be read as a health score: most
    # CEM candidates were themselves PRODUCED by the decoder, so they agree with
    # it trivially and inflate it (measured 0.875 here). When the round trip is
    # constant, recovery_rate is exactly the share of candidates whose true class
    # happens to equal that one collapsed class -- it says nothing about
    # invertibility. Assert that identity rather than a level.
    collapsed = set(_roundtrip_classes(module, candidates))
    assert len(collapsed) == 1
    only = collapsed.pop()
    true_classes = [module.candidate_first_action_class(c) for c in candidates]
    expected = sum(1 for c in true_classes if c == only) / len(true_classes)
    assert rt["recovery_rate"] == pytest.approx(expected)


def test_roundtrip_recovery_handles_empty_candidate_list():
    module = _make_module()
    rt = module.action_object_roundtrip_recovery([])
    assert rt["n_scored"] == 0
    assert rt["recovery_rate"] is None


# --------------------------------------------------------------------------
# DEFECT 2: the CEM elite-pool floor
# --------------------------------------------------------------------------

def test_num_elite_floor_prevents_nan_poisoning_at_small_candidate_counts():
    """num_candidates=8 x default elite_fraction=0.2 -> int(1.6) == 1 -> NaN.

    Guards the exact configuration that was measured producing 1 finite
    candidate of 8, and that crashed `torch.multinomial` inside E3.
    """
    from ree_core.utils.config import HippocampalConfig

    # The trap is reachable from the SUBSTRATE DEFAULT, not from an exotic knob.
    assert int(8 * HippocampalConfig().elite_fraction) == 1

    torch.manual_seed(42)
    module = _make_module(num_candidates=8, elite_fraction=0.2)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8), num_candidates=8,
    )

    diags = module.get_last_propose_diagnostics()
    assert diags["cem_num_elite"] >= 2, (
        "CEM elite pool fell below 2; elite_ao_tensor.std(dim=0) is NaN over a "
        "single sample and poisons every subsequent candidate"
    )

    assert len(candidates) == 8
    for i, traj in enumerate(candidates):
        ao_seq = traj.get_action_object_sequence()
        assert ao_seq is not None, f"candidate {i} has no action-object sequence"
        assert torch.isfinite(ao_seq).all(), f"candidate {i} action-objects non-finite"
        assert torch.isfinite(traj.actions).all(), f"candidate {i} actions non-finite"


def test_num_elite_floor_is_capped_by_candidate_count():
    """The floor must never exceed n -- a 1-candidate pool cannot yield 2 elites."""
    torch.manual_seed(42)
    module = _make_module(num_candidates=1, elite_fraction=0.2)
    candidates = module.propose_trajectories(
        z_world=torch.zeros(1, 8), z_self=torch.zeros(1, 8), num_candidates=1,
    )
    diags = module.get_last_propose_diagnostics()
    assert diags["cem_num_elite"] == 1
    assert len(candidates) == 1
    # The NaN-safe std backstop must hold where the floor cannot.
    for traj in candidates:
        assert torch.isfinite(traj.actions).all()


def test_stack_std_is_nan_safe_for_a_single_sample():
    """Direct unit check of the backstop, independent of CEM wiring."""
    from ree_core.hippocampal.module import HippocampalModule

    single = torch.randn(1, 3, 4)
    # Establish that the hazard this guards is real in this torch build.
    assert torch.isnan(single.std(dim=0)).all()

    guarded = HippocampalModule._stack_std(single)
    assert torch.isfinite(guarded).all()
    assert torch.equal(guarded, torch.zeros_like(single[0]))

    pair = torch.randn(2, 3, 4)
    assert torch.allclose(HippocampalModule._stack_std(pair), pair.std(dim=0))
