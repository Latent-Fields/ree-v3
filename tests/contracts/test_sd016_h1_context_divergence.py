"""SD-016 H1 contract tests: context-divergence auxiliary loss.

GOV-FANOUT-1 / V3-EXQ-907 (2026-08-09, CONFIRMED via
failure_autopsy_V3-EXQ-907_2026-08-10.json): the uniform-softmax saddle
every prior SD-016 selection mechanism hit was never given a training
signal that specifically rewards context-CONDITIONED divergence (as
distinct from population-level slot diversity, which V3-EXQ-418e/i showed
does not help). V3-EXQ-907's driver-only probe confirmed
-lambda*|mean_w_safe - mean_w_dang|_1 (computed directly on
agent.e1.cue_slot_tagger, on a fixed safe/dangerous z_world batch pair)
breaks the saddle at every lambda in {0.1, 0.5, 2.0} tested. This module
pins the PROMOTED substrate form (E1DeepPredictor.compute_context_
divergence_loss + sd016_context_divergence_weight), per the promotion the
V3-EXQ-907 driver's own docstring calls for. These contracts pin: (1) the
knob is no-op (zero tensor, no graph) at its default weight=0.0, (2) it is
no-op when the tagger is disabled regardless of weight, (3) a positive
weight produces a nonzero, correctly-signed loss with gradient reaching
the tagger, (4) the computation matches the V3-EXQ-907-validated formula
exactly, (5) it composes cleanly with the H3-recommended "gumbel"
selection mode (the validated production combination).
"""
import math

import torch
import torch.nn.functional as F

from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.utils.config import E1Config

WORLD_DIM = 32
SELF_DIM = 32
NUM_SLOTS = 16  # ContextMemory default


def _build(**kw) -> E1DeepPredictor:
    cfg = E1Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        latent_dim=SELF_DIM + WORLD_DIM,
        hidden_dim=128,
        sd016_enabled=True,
        action_object_dim=16,
        **kw,
    )
    return E1DeepPredictor(cfg)


def test_c1_default_weight_is_zero_and_noop():
    """Not specifying sd016_context_divergence_weight defaults to 0.0."""
    e1 = _build(sd016_cue_slot_tagger=True)
    assert e1.config.sd016_context_divergence_weight == 0.0
    z_safe = torch.randn(8, WORLD_DIM)
    z_dang = torch.randn(8, WORLD_DIM)
    loss = e1.compute_context_divergence_loss(z_safe, z_dang)
    assert loss.item() == 0.0
    assert loss.grad_fn is None, "off-state must build no autograd graph"


def test_c2_explicit_zero_weight_is_noop():
    """weight=0.0 passed explicitly behaves identically to the default."""
    e1 = _build(sd016_cue_slot_tagger=True, sd016_context_divergence_weight=0.0)
    z_safe = torch.randn(8, WORLD_DIM)
    z_dang = torch.randn(8, WORLD_DIM)
    loss = e1.compute_context_divergence_loss(z_safe, z_dang)
    assert loss.item() == 0.0
    assert loss.grad_fn is None


def test_c3_noop_when_tagger_disabled_even_with_positive_weight():
    """A positive weight has no effect when sd016_cue_slot_tagger=False --
    the knob is scoped to the tagger's slot-selection path only."""
    e1 = _build(sd016_cue_slot_tagger=False, sd016_context_divergence_weight=0.5)
    assert e1.cue_slot_tagger is None
    z_safe = torch.randn(8, WORLD_DIM)
    z_dang = torch.randn(8, WORLD_DIM)
    loss = e1.compute_context_divergence_loss(z_safe, z_dang)
    assert loss.item() == 0.0
    assert loss.grad_fn is None


def test_c4_positive_weight_produces_gradient_into_tagger():
    """weight>0 with the tagger enabled: nonzero loss, gradient reaches
    every cue_slot_tagger parameter (the H1 mechanism's whole point)."""
    torch.manual_seed(0)
    e1 = _build(sd016_cue_slot_tagger=True, sd016_context_divergence_weight=0.5)
    z_safe = torch.randn(16, WORLD_DIM)
    z_dang = torch.randn(16, WORLD_DIM) + 3.0  # shifted -> non-trivial divergence
    loss = e1.compute_context_divergence_loss(z_safe, z_dang)
    assert loss.grad_fn is not None
    loss.backward()
    for p in e1.cue_slot_tagger.parameters():
        assert p.grad is not None and p.grad.abs().sum().item() > 0.0


def test_c5_matches_v3_exq_907_validated_formula():
    """The promoted method must compute EXACTLY the V3-EXQ-907 driver's
    -weight*|mean_w_safe - mean_w_dang|_1 on
    softmax(cue_slot_tagger(z)/temperature) -- no drift during promotion."""
    torch.manual_seed(1)
    weight = 0.5
    e1 = _build(
        sd016_cue_slot_tagger=True,
        sd016_cue_slot_tagger_temperature=1.0,
        sd016_context_divergence_weight=weight,
    )
    z_safe = torch.randn(8, WORLD_DIM)
    z_dang = torch.randn(8, WORLD_DIM) + 2.0

    with torch.no_grad():
        logits_safe = e1.cue_slot_tagger(z_safe)
        logits_dang = e1.cue_slot_tagger(z_dang)
        w_safe = F.softmax(logits_safe / 1.0, dim=-1).mean(dim=0)
        w_dang = F.softmax(logits_dang / 1.0, dim=-1).mean(dim=0)
        expected = -weight * (w_safe - w_dang).abs().sum()

    actual = e1.compute_context_divergence_loss(z_safe, z_dang)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_c6_identical_batches_give_near_zero_divergence_loss():
    """Sanity check: safe and dangerous batches with the SAME content ->
    the tagger produces the same mean distribution for both -> divergence
    (and therefore the loss) is ~0, not an artifact of batch difference
    alone."""
    torch.manual_seed(2)
    e1 = _build(sd016_cue_slot_tagger=True, sd016_context_divergence_weight=0.5)
    z = torch.randn(8, WORLD_DIM)
    loss = e1.compute_context_divergence_loss(z, z.clone())
    assert abs(loss.item()) < 1e-5


def test_c7_composes_with_gumbel_selection_production_combination():
    """The validated production combination (V3-EXQ-907 H1 + V3-EXQ-908 H3,
    both CONFIRMED): tagger=True, selection='gumbel', context_divergence_
    weight>0. extract_cue_context still returns valid action_bias/
    terrain_weight (MECH-151/MECH-152), and compute_context_divergence_loss
    still trains the SAME tagger the gumbel selection reads from -- the two
    mechanisms are not wired to conflict."""
    torch.manual_seed(3)
    e1 = _build(
        sd016_cue_slot_tagger=True,
        sd016_cue_slot_tagger_selection="gumbel",
        sd016_cue_slot_tagger_gumbel_anneal_steps=10,
        sd016_context_divergence_weight=0.5,
    )
    e1.train()
    z = torch.randn(8, WORLD_DIM)
    action_bias, terrain_weight = e1.extract_cue_context(z)
    assert action_bias.shape == (8, 16)
    assert terrain_weight.shape == (8, 2)
    w = e1._last_cue_slot_weights
    assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)

    z_safe = torch.randn(8, WORLD_DIM)
    z_dang = torch.randn(8, WORLD_DIM) + 3.0
    div_loss = e1.compute_context_divergence_loss(z_safe, z_dang)
    assert div_loss.grad_fn is not None
    div_loss.backward()
    for p in e1.cue_slot_tagger.parameters():
        assert p.grad is not None


def test_c8_topk_selection_not_recommended_but_still_composes_mechanically():
    """H3 ELIMINATED straight-through top-k (constant_peaky_degenerate) --
    this is a mechanical composition check only, NOT an endorsement: the
    ctxdiv loss still trains the tagger regardless of which selection
    operator reads its logits at inference, since the two are independent
    code paths (compute_context_divergence_loss calls cue_slot_tagger
    directly; extract_cue_context's selection branch is a separate read).
    """
    torch.manual_seed(4)
    e1 = _build(
        sd016_cue_slot_tagger=True,
        sd016_cue_slot_tagger_selection="topk",
        sd016_cue_slot_tagger_topk_k=1,
        sd016_context_divergence_weight=0.5,
    )
    z_safe = torch.randn(8, WORLD_DIM)
    z_dang = torch.randn(8, WORLD_DIM) + 3.0
    loss = e1.compute_context_divergence_loss(z_safe, z_dang)
    assert loss.grad_fn is not None
