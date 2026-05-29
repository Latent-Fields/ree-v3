"""SD-056 contracts: E2 action-conditional divergence preservation via
auxiliary InfoNCE contrastive loss on world_forward.

C1: Default-OFF backward compatibility (E2Config defaults are no-op).
C2: from_dims surfaces the 4 SD-056 knobs and propagates to config.e2.
C3: cand_world_pairwise_dist diagnostic helper exists, returns non-negative
    scalar, and is 0.0 on K < 2 / single-class degenerate batches.
C4: world_forward_contrastive_loss with simulation_mode=True returns
    tensor(0.0) without advancing optimiser state (MECH-094 gate).
C5: world_forward_contrastive_loss respects min_batch_classes floor and
    returns 0.0 below it.
C6: world_forward_contrastive_loss returns non-negative cross-entropy
    scalar in the well-formed K=8 sibling-CEM-candidate case AND backward()
    succeeds (gradient flows through world_transition / world_action_encoder).
C7: After 200 SGD steps under contrastive loss on synthetic K=8 batches,
    cand_world_pairwise_dist rises from the random-init baseline by a
    positive margin (UC3 direction-of-change; the load-bearing PASS
    condition per the design memo). NOT pinned to a specific threshold
    here -- threshold calibration is V3-EXQ-NEW-1's job; this contract
    asserts only the direction.
"""

import pytest
import torch

from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.utils.config import E2Config, REEConfig


# ----------------------------- helpers ----------------------------------- #


def _make_e2_with_contrastive(
    self_dim=8,
    world_dim=16,
    action_dim=4,
    hidden_dim=32,
    enabled=True,
    weight=0.01,
    temperature=0.1,
    min_batch_classes=2,
):
    cfg = E2Config(
        self_dim=self_dim,
        world_dim=world_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    cfg.e2_action_contrastive_enabled = enabled
    cfg.e2_action_contrastive_weight = weight
    cfg.e2_action_contrastive_temperature = temperature
    cfg.e2_action_contrastive_min_batch_classes = min_batch_classes
    return E2FastPredictor(cfg)


def _make_sibling_candidate_batch(K=8, world_dim=16, action_dim=4, seed=0):
    """Sibling CEM batch: shared z_world_0, distinct first-action one-hots,
    targets are perturbed z_world_0 + per-action shift so the contrastive
    task is learnable (each action has a structurally different target)."""
    g = torch.Generator().manual_seed(seed)
    z_world_0 = torch.randn(world_dim, generator=g)
    # K actions: cycle through action classes so we cover the action_dim space
    classes = torch.arange(K, dtype=torch.long) % action_dim
    actions = torch.nn.functional.one_hot(classes, num_classes=action_dim).float()
    # Per-action target shift: action class c -> shift e_c in the first action_dim
    # coords of z_world. Magnitude 0.5 so the target is far enough from z_world_0
    # for the contrastive task to have signal.
    shifts = torch.zeros(K, world_dim)
    for i in range(K):
        shifts[i, int(classes[i])] = 0.5
    targets = z_world_0.unsqueeze(0).expand(K, -1) + shifts
    return z_world_0, actions, targets


# -------------------------- C1: default OFF ------------------------------ #


def test_c1_e2config_defaults_are_no_op():
    """SD-056 E2Config defaults preserve bit-identical pre-substrate behaviour."""
    cfg = E2Config()
    assert cfg.e2_action_contrastive_enabled is False
    assert cfg.e2_action_contrastive_weight == pytest.approx(0.01)
    assert cfg.e2_action_contrastive_temperature == pytest.approx(0.1)
    assert cfg.e2_action_contrastive_min_batch_classes == 2


# ---------------------- C2: from_dims propagation ------------------------ #


def test_c2_from_dims_surfaces_sd056_knobs_and_propagates():
    """REEConfig.from_dims kwargs land on config.e2.* with correct values."""
    cfg_default = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, harm_obs_dim=8, action_dim=4,
    )
    assert cfg_default.e2.e2_action_contrastive_enabled is False
    assert cfg_default.e2.e2_action_contrastive_weight == pytest.approx(0.01)
    assert cfg_default.e2.e2_action_contrastive_temperature == pytest.approx(0.1)
    assert cfg_default.e2.e2_action_contrastive_min_batch_classes == 2

    cfg_on = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, harm_obs_dim=8, action_dim=4,
        e2_action_contrastive_enabled=True,
        e2_action_contrastive_weight=0.05,
        e2_action_contrastive_temperature=0.2,
        e2_action_contrastive_min_batch_classes=3,
    )
    assert cfg_on.e2.e2_action_contrastive_enabled is True
    assert cfg_on.e2.e2_action_contrastive_weight == pytest.approx(0.05)
    assert cfg_on.e2.e2_action_contrastive_temperature == pytest.approx(0.2)
    assert cfg_on.e2.e2_action_contrastive_min_batch_classes == 3


# --------------- C3: cand_world_pairwise_dist diagnostic ---------------- #


def test_c3_cand_world_pairwise_dist_basic_properties():
    e2 = _make_e2_with_contrastive()
    z0, actions, _ = _make_sibling_candidate_batch(K=8, world_dim=16, action_dim=4)

    # Untrained random-init E2 may produce a tiny but non-zero dist (the
    # design memo's V3-EXQ-571 measurement of exactly 0.0 reflects the V3
    # production substrate's trained-to-collapse state, NOT a random-init
    # property). The contract asserts non-negativity and finite, not zero.
    dist = e2.cand_world_pairwise_dist(z0, actions)
    assert dist.dim() == 0
    assert torch.isfinite(dist)
    assert float(dist) >= 0.0


def test_c3_cand_world_pairwise_dist_K_lt_2_returns_zero():
    e2 = _make_e2_with_contrastive()
    z0 = torch.randn(16)
    actions = torch.nn.functional.one_hot(torch.tensor([0]), num_classes=4).float()
    dist = e2.cand_world_pairwise_dist(z0, actions)
    assert float(dist) == 0.0


def test_c3_cand_world_pairwise_dist_accepts_batched_z0():
    """z_world_0 can be [world_dim] or [1, world_dim] or [K, world_dim]."""
    e2 = _make_e2_with_contrastive()
    K = 4
    z0_1d = torch.randn(16)
    actions = torch.nn.functional.one_hot(
        torch.arange(K) % 4, num_classes=4
    ).float()
    d_1d = e2.cand_world_pairwise_dist(z0_1d, actions)
    d_2d = e2.cand_world_pairwise_dist(z0_1d.unsqueeze(0), actions)
    d_Kd = e2.cand_world_pairwise_dist(z0_1d.unsqueeze(0).expand(K, -1), actions)
    # All three forms produce identical predictions, hence identical pairwise
    # distance matrices.
    assert float(d_1d) == pytest.approx(float(d_2d), abs=1e-6)
    assert float(d_1d) == pytest.approx(float(d_Kd), abs=1e-6)


# ----------------- C4: MECH-094 simulation_mode gate -------------------- #


def test_c4_simulation_mode_returns_zero_and_blocks_grad():
    """world_forward_contrastive_loss simulation_mode=True -> tensor(0.0).
    Gradient should not flow through the helper when simulation_mode is on.
    Mirrors the SD-035 / MECH-279 / MECH-313 / MECH-314 / MECH-319 /
    MECH-320 / MECH-341 pattern."""
    e2 = _make_e2_with_contrastive()
    z0, actions, targets = _make_sibling_candidate_batch()

    loss = e2.world_forward_contrastive_loss(
        z0, actions, targets, simulation_mode=True
    )
    assert loss.dim() == 0
    assert float(loss) == 0.0
    # Loss returned with grad_fn=None when simulation_mode is on (constructed
    # via torch.zeros, not via the cross_entropy path).
    assert loss.requires_grad is False or loss.grad_fn is None

    # Compare against waking call to confirm the simulation branch short-circuits
    # before world_forward / cross_entropy.
    loss_waking = e2.world_forward_contrastive_loss(z0, actions, targets)
    assert float(loss_waking) > 0.0


# --------------- C5: min_batch_classes floor ---------------------------- #


def test_c5_min_batch_classes_floor_returns_zero_below_threshold():
    e2 = _make_e2_with_contrastive(min_batch_classes=3)
    # Build a K=4 batch with only 2 distinct first-action classes
    z0 = torch.randn(16)
    classes = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    actions = torch.nn.functional.one_hot(classes, num_classes=4).float()
    targets = z0.unsqueeze(0).expand(4, -1) + 0.1 * torch.randn(4, 16)
    loss = e2.world_forward_contrastive_loss(z0, actions, targets)
    assert float(loss) == 0.0

    # With min_batch_classes=2 the same batch now clears the floor
    e2_low = _make_e2_with_contrastive(min_batch_classes=2)
    loss_low = e2_low.world_forward_contrastive_loss(z0, actions, targets)
    assert float(loss_low) > 0.0


# ----------- C6: well-formed contrastive loss + backward() -------------- #


def test_c6_well_formed_contrastive_loss_and_backward():
    e2 = _make_e2_with_contrastive()
    z0, actions, targets = _make_sibling_candidate_batch(K=8)

    loss = e2.world_forward_contrastive_loss(z0, actions, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert float(loss) > 0.0

    # backward() must reach world_transition / world_action_encoder weights
    e2.zero_grad(set_to_none=True)
    loss.backward()
    wt_grad = e2.world_transition[0].weight.grad
    enc_grad = e2.world_action_encoder.weight.grad
    assert wt_grad is not None
    assert enc_grad is not None
    assert torch.isfinite(wt_grad).all()
    assert torch.isfinite(enc_grad).all()
    # At least one of the two has non-zero gradient (gradient does in fact
    # flow through the loss).
    assert (wt_grad.abs().sum() + enc_grad.abs().sum()) > 0.0


# ------- C7: UC3 direction-of-change after 200 SGD steps --------------- #


def test_c7_uc3_direction_of_change_after_200_sgd_steps():
    """After 200 SGD steps under contrastive loss on synthetic K=8 batches,
    cand_world_pairwise_dist rises from the random-init baseline. This is
    the design-memo UC3 direction-of-change PASS condition (magnitude is
    calibrated by V3-EXQ-NEW-1; only the direction is contracted here)."""
    torch.manual_seed(42)
    e2 = _make_e2_with_contrastive()

    # Baseline at random-init: average cand_world_pairwise_dist across a small
    # batch of distinct seeds so the baseline isn't a single fluke.
    def _avg_dist(model, n_batches=4, K=8):
        dists = []
        for s in range(n_batches):
            z0, actions, _ = _make_sibling_candidate_batch(
                K=K, world_dim=16, action_dim=4, seed=100 + s
            )
            with torch.no_grad():
                dists.append(float(model.cand_world_pairwise_dist(z0, actions)))
        return sum(dists) / len(dists)

    baseline = _avg_dist(e2)

    optimizer = torch.optim.Adam(e2.parameters(), lr=1e-3)
    for step in range(200):
        z0, actions, targets = _make_sibling_candidate_batch(
            K=8, world_dim=16, action_dim=4, seed=step
        )
        loss = e2.world_forward_contrastive_loss(z0, actions, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    trained = _avg_dist(e2)

    # Direction-of-change: trained > baseline by a positive margin. The
    # design-memo UC3 suggests >= 0.05 in normalised units; here we assert
    # the looser direction-of-change because the synthetic env is small and
    # the absolute magnitude depends on world_dim / action shift scale. The
    # full UC3 magnitude check lives in V3-EXQ-NEW-1.
    assert trained > baseline, (
        f"SD-056 UC3 direction-of-change FAILED: baseline={baseline:.6f}, "
        f"trained={trained:.6f}"
    )
    # Loose magnitude guard: trained should be at least 2x baseline OR
    # cleared the 0.01 minimum-observable threshold. This prevents the test
    # passing on a tiny random-walk improvement.
    assert (trained >= 2.0 * baseline) or (trained >= 0.01), (
        f"SD-056 UC3 magnitude floor FAILED: baseline={baseline:.6f}, "
        f"trained={trained:.6f}"
    )
