"""SD-016 Path 3 contract tests: feedforward cue->slot tagger.

V3-EXQ-418i established that the q.k soft-attention slot-selection inside
E1DeepPredictor.extract_cue_context is pinned at the uniform ln(num_slots)
saddle ("the attention bottleneck is categorically in query selectivity").
Path 3 replaces ONLY the slot-selection scores with a feedforward tagger
(z_world -> slot logits) that sits off the saddle, trained by the existing
cue_terrain_proj terrain_loss gradient. These contracts pin the interface
guarantees (bit-identical OFF, wired + trainable ON), not magnitudes.
"""
import math

import torch
import torch.nn.functional as F

from ree_core.predictors.e1_deep import E1DeepPredictor
from ree_core.utils.config import E1Config

WORLD_DIM = 32
SELF_DIM = 32
NUM_SLOTS = 16  # ContextMemory default


def _build(tagger: bool, temperature: float = 1.0) -> E1DeepPredictor:
    cfg = E1Config(
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        latent_dim=SELF_DIM + WORLD_DIM,
        hidden_dim=128,
        sd016_enabled=True,
        action_object_dim=16,
        sd016_cue_slot_tagger=tagger,
        sd016_cue_slot_tagger_temperature=temperature,
    )
    return E1DeepPredictor(cfg)


def _selection_entropy(e1: E1DeepPredictor, z_world: torch.Tensor) -> float:
    e1.extract_cue_context(z_world)
    w = e1._last_cue_slot_weights
    return -(w * w.clamp_min(1e-9).log()).sum(-1).mean().item()


def test_c1_default_off_is_legacy_uniform_saddle():
    """OFF: no tagger module; selection sits on the uniform ln(num_slots) saddle."""
    torch.manual_seed(0)
    e1 = _build(tagger=False)
    assert e1.cue_slot_tagger is None
    assert e1._sd016_cue_slot_tagger is False
    z = torch.randn(8, WORLD_DIM)
    ent = _selection_entropy(e1, z)
    # legacy q.k attention with small-init slots is the uniform saddle
    assert abs(ent - math.log(NUM_SLOTS)) < 1e-4


def test_c2_master_off_has_no_tagger_state():
    """sd016_enabled=False: Path-3 attributes default to inert with no tagger."""
    cfg = E1Config(self_dim=SELF_DIM, world_dim=WORLD_DIM,
                   latent_dim=SELF_DIM + WORLD_DIM, hidden_dim=128,
                   sd016_enabled=False)
    e1 = E1DeepPredictor(cfg)
    assert e1.cue_slot_tagger is None
    assert e1._sd016_cue_slot_tagger is False


def test_c3_on_builds_tagger_and_selection_is_valid_distribution():
    """ON: tagger built; selection weights are a valid per-context distribution."""
    torch.manual_seed(0)
    e1 = _build(tagger=True)
    assert e1.cue_slot_tagger is not None
    z = torch.randn(8, WORLD_DIM)
    ab, tw = e1.extract_cue_context(z)
    assert ab.shape == (8, 16) and tw.shape == (8, 2)
    w = e1._last_cue_slot_weights
    assert w.shape == (8, NUM_SLOTS)
    assert torch.allclose(w.sum(-1), torch.ones(8), atol=1e-5)
    # off the saddle (a fresh MLP is not exactly uniform) and context-varying
    ent = _selection_entropy(e1, z)
    assert ent <= math.log(NUM_SLOTS) + 1e-6
    assert w.std(0).mean().item() > 1e-3


def test_c4_tagger_receives_terrain_loss_gradient():
    """ON: terrain output backprops into the tagger parameters (trainable path)."""
    torch.manual_seed(0)
    e1 = _build(tagger=True)
    z = torch.randn(8, WORLD_DIM)
    _, tw = e1.extract_cue_context(z)
    F.mse_loss(tw, torch.rand_like(tw)).backward()
    g = e1.cue_slot_tagger[0].weight.grad
    assert g is not None and g.abs().sum().item() > 0.0


def test_c5_temperature_sharpens_selection():
    """Lower tagger temperature yields a sharper (lower-entropy) selection."""
    torch.manual_seed(0)
    z = torch.randn(8, WORLD_DIM)
    e_hot = _build(tagger=True, temperature=5.0)
    e_cold = _build(tagger=True, temperature=0.1)
    # copy weights so only temperature differs
    e_cold.cue_slot_tagger.load_state_dict(e_hot.cue_slot_tagger.state_dict())
    assert _selection_entropy(e_cold, z) <= _selection_entropy(e_hot, z) + 1e-6
