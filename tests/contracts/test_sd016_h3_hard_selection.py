"""SD-016 H3 contract tests: hard/competitive selection operator.

GOV-FANOUT-1 / V3-EXQ-898 failure autopsy (2026-08-08): the soft, end-to-end
differentiable Path 3 tagger (test_sd016_cue_slot_tagger.py) reproduced the
exact uniform ln(16) attractor under C2's control even when correctly
rewarded, raising the hypothesis (H3) that a soft softmax gate cannot HOLD a
sparse, context-selective optimum regardless of what terrain_loss demands.
This module swaps the tagger's slot-SELECTION operator (only) for a
structurally competitive one -- gumbel (annealed straight-through
Gumbel-softmax) or topk (straight-through top-k) -- independent of the
downstream loss. These contracts pin: (1) default "soft" is bit-identical to
the pre-H3 Path 3 behaviour, (2) both hard modes produce a valid,
structurally sparse selection distribution, (3) gradient still reaches the
tagger through the straight-through estimator, (4) gumbel's anneal schedule
only advances in train mode and eval is deterministic, (5) an invalid
selection mode is rejected.
"""
import math

import torch
import torch.nn.functional as F

from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.utils.config import E1Config

WORLD_DIM = 32
SELF_DIM = 32
NUM_SLOTS = 16  # ContextMemory default


def _build(selection: str = "soft", **kw) -> E1DeepPredictor:
    cfg = E1Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        latent_dim=SELF_DIM + WORLD_DIM,
        hidden_dim=128,
        sd016_enabled=True,
        action_object_dim=16,
        sd016_cue_slot_tagger=True,
        sd016_cue_slot_tagger_selection=selection,
        **kw,
    )
    return E1DeepPredictor(cfg)


def _selection_entropy(e1: E1DeepPredictor, z_world: torch.Tensor) -> float:
    e1.extract_cue_context(z_world)
    w = e1._last_cue_slot_weights
    return -(w * w.clamp_min(1e-9).log()).sum(-1).mean().item()


def test_h1_default_selection_is_soft():
    """Not specifying sd016_cue_slot_tagger_selection defaults to 'soft'."""
    cfg = E1Config(
        self_dim=SELF_DIM, world_dim=WORLD_DIM, latent_dim=SELF_DIM + WORLD_DIM,
        hidden_dim=128, sd016_enabled=True, sd016_cue_slot_tagger=True,
    )
    assert cfg.sd016_cue_slot_tagger_selection == "soft"
    e1 = E1DeepPredictor(cfg)
    assert e1._sd016_cue_slot_tagger_selection == "soft"


def test_h2_soft_selection_bit_identical_to_manual_softmax():
    """selection='soft' produces EXACTLY F.softmax(logits/temp) -- no new
    code path is exercised for the pre-existing default behaviour."""
    torch.manual_seed(0)
    e1 = _build(selection="soft")
    z = torch.randn(8, WORLD_DIM)
    e1.eval()
    e1.extract_cue_context(z)
    w = e1._last_cue_slot_weights

    with torch.no_grad():
        expected_logits = e1.cue_slot_tagger(z)
        expected = F.softmax(expected_logits / 1.0, dim=-1)
    assert torch.equal(w, expected)


def test_h3_soft_selection_matches_pre_h3_entropy_profile():
    """selection='soft' reproduces the pre-H3 Path 3 entropy behaviour
    (test_c1/c3 of test_sd016_cue_slot_tagger.py): off the uniform saddle,
    a valid per-context distribution."""
    torch.manual_seed(0)
    e1 = _build(selection="soft")
    z = torch.randn(8, WORLD_DIM)
    ab, tw = e1.extract_cue_context(z)
    assert ab.shape == (8, 16) and tw.shape == (8, 2)
    w = e1._last_cue_slot_weights
    assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)
    ent = _selection_entropy(e1, z)
    assert ent <= math.log(NUM_SLOTS) + 1e-6


def test_h4_invalid_selection_mode_raises():
    """An unrecognised selection string fails loudly at construction, not
    silently at forward time."""
    cfg = E1Config(
        self_dim=SELF_DIM, world_dim=WORLD_DIM, latent_dim=SELF_DIM + WORLD_DIM,
        hidden_dim=128, sd016_enabled=True, sd016_cue_slot_tagger=True,
        sd016_cue_slot_tagger_selection="bogus",
    )
    try:
        E1DeepPredictor(cfg)
        assert False, "expected ValueError for invalid selection mode"
    except ValueError:
        pass


def test_h5_gumbel_selection_is_structurally_sparse_and_valid_distribution():
    """selection='gumbel': forward pass is (near-)one-hot per row, weights
    sum to 1, and gradient still reaches the tagger (straight-through)."""
    torch.manual_seed(0)
    e1 = _build(selection="gumbel", sd016_cue_slot_tagger_gumbel_anneal_steps=10)
    e1.train()
    z = torch.randn(8, WORLD_DIM)
    ab, tw = e1.extract_cue_context(z)
    w = e1._last_cue_slot_weights
    assert w.shape == (8, NUM_SLOTS)
    assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)
    # forward value of the ST estimator equals the hard one-hot: exactly one
    # slot per row should be (numerically) 1.0.
    assert torch.allclose(w.max(dim=-1).values, torch.ones(8), atol=1e-5)
    assert (w > 1e-6).sum(dim=-1).eq(1).all()

    F.mse_loss(tw, torch.rand_like(tw)).backward()
    g = e1.cue_slot_tagger[0].weight.grad
    assert g is not None and g.abs().sum().item() > 0.0


def test_h6_gumbel_anneal_only_advances_in_train_mode_and_eval_is_deterministic():
    """The annealed-temperature step counter advances only on training-mode
    forward calls; eval-mode calls are deterministic (no Gumbel noise, no
    schedule advance) so a validation/inference pass is reproducible."""
    torch.manual_seed(1)
    e1 = _build(selection="gumbel", sd016_cue_slot_tagger_gumbel_anneal_steps=5)
    e1.train()
    for _ in range(3):
        e1.extract_cue_context(torch.randn(4, WORLD_DIM))
    assert e1._sd016_gumbel_step == 3

    e1.eval()
    z = torch.randn(4, WORLD_DIM)
    ab1, tw1 = e1.extract_cue_context(z)
    step_after_first_eval = e1._sd016_gumbel_step
    ab2, tw2 = e1.extract_cue_context(z)
    assert e1._sd016_gumbel_step == step_after_first_eval == 3
    assert torch.equal(ab1, ab2)
    assert torch.equal(tw1, tw2)


def test_h7_topk_selects_exactly_k_slots_and_is_trainable():
    """selection='topk': forward pass keeps exactly k nonzero slots per row
    (renormalised to sum to 1), gradient still reaches the tagger."""
    torch.manual_seed(2)
    for k in (1, 2):
        e1 = _build(selection="topk", sd016_cue_slot_tagger_topk_k=k)
        z = torch.randn(8, WORLD_DIM)
        ab, tw = e1.extract_cue_context(z)
        w = e1._last_cue_slot_weights
        assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)
        nonzero = (w > 1e-8).sum(dim=-1)
        assert (nonzero == k).all(), f"k={k}: expected {k} nonzero slots, got {nonzero}"

        F.mse_loss(tw, torch.rand_like(tw)).backward()
        g = e1.cue_slot_tagger[0].weight.grad
        assert g is not None and g.abs().sum().item() > 0.0


def test_h8_topk_k_clamped_to_num_slots():
    """A k larger than num_slots is clamped rather than erroring."""
    torch.manual_seed(3)
    e1 = _build(selection="topk", sd016_cue_slot_tagger_topk_k=NUM_SLOTS + 5)
    z = torch.randn(4, WORLD_DIM)
    e1.extract_cue_context(z)
    w = e1._last_cue_slot_weights
    assert torch.allclose(w.sum(-1), torch.ones(4), atol=1e-5)
    assert (w > 1e-8).sum(dim=-1).eq(NUM_SLOTS).all()
