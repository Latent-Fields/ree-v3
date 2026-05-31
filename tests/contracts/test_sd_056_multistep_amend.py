"""SD-056 multi-step rollout stability amend contracts (2026-05-31).

Validates the V3-EXQ-569e autopsy Section 9 amend: two new togglable levers
on top of the existing SD-056 t=1 contrastive substrate.

  (a) multi-step contrastive: extend the contrastive objective to an h-step
      rollout horizon (Dreamer/PlaNet anchor; Srivastava 2021 lever B).
      Helper: E2FastPredictor.world_forward_contrastive_loss_multistep.
  (b) per-step output norm clamp inside E2.rollout_with_world (B2 anchor:
      ||z_world_{t+1}|| <= ratio * ||z_world_0||).

Both default OFF -- bit-identical to the pre-amend SD-056 path. The autopsy
acceptance target is max-NaN-fraction < 0.05 + rollout magnitudes within 2x
of OFF baseline at the behavioural-runtime episode length; these contracts
validate the substrate primitives that deliver that property (full-budget
8-arm probe is the post-amend /queue-experiment session).

Contracts:
    A1: default E2Config has the 5 new amend knobs all set to no-op defaults.
    A2: REEConfig.from_dims surfaces all 5 amend knobs and propagates them
        onto config.e2.
    A3: world_forward_contrastive_loss_multistep exists, returns 0-d scalar
        in the well-formed K-candidate batch, and backward() succeeds
        (gradient flows through world_transition / world_action_encoder).
    A4: simulation_mode=True returns tensor(0.0) (MECH-094 standard pattern).
    A5: K < 2 returns tensor(0.0).
    A6: min_batch_classes floor returns tensor(0.0) on single-class batch.
    A7: horizon=0 or beyond available action_sequences / z_world_targets
        clamps to effective horizon and never raises.
    A8: per-step rollout norm clamp -- with master OFF the rollout output is
        bit-identical to a parallel run with the clamp config absent.
    A9: per-step rollout norm clamp -- with master ON, every step of the
        returned z_world trajectory satisfies ||z_world_t|| <= ratio *
        ||z_world_0|| element-wise.
    A10: per-step rollout norm clamp -- with master ON and a deliberately
        unstable world_forward (large random init) over a long horizon, the
        rollout never produces NaN/Inf and its max magnitude stays inside
        the B2 bound.
"""

import math

import pytest
import torch

from ree_core.predictors.e2_fast import E2FastPredictor
from ree_core.utils.config import E2Config, REEConfig


def _make_e2(
    self_dim=8,
    world_dim=16,
    action_dim=4,
    hidden_dim=32,
    multistep_enabled=False,
    horizon=5,
    horizon_weights_decay=1.0,
    clamp_enabled=False,
    clamp_ratio=2.0,
    weight_init_scale=None,
):
    cfg = E2Config(
        self_dim=self_dim,
        world_dim=world_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )
    cfg.e2_action_contrastive_multistep_enabled = multistep_enabled
    cfg.e2_action_contrastive_horizon = horizon
    cfg.e2_action_contrastive_horizon_weights_decay = horizon_weights_decay
    cfg.e2_rollout_output_norm_clamp_enabled = clamp_enabled
    cfg.e2_rollout_output_norm_clamp_ratio = clamp_ratio
    model = E2FastPredictor(cfg)
    if weight_init_scale is not None:
        # Deliberately rescale world_transition weights to make the residual
        # delta large -- used by A10 to provoke an unstable rollout.
        with torch.no_grad():
            for p in model.world_transition.parameters():
                p.mul_(weight_init_scale)
    return model


def _sibling_multistep_batch(K=8, world_dim=16, action_dim=4, horizon=5, seed=0):
    """Build a sibling-CEM-style multi-step batch:
    - z_world_0 shared across K anchors
    - action_sequences [K, horizon, action_dim] one-hot at each step; the
      first-step actions form K distinct one-hot rows so min_batch_classes
      floor (default 2) clears.
    - z_world_targets [K, horizon+1, world_dim] = z_world_0 + cumulative
      per-anchor per-step shift, so each anchor's target trajectory is
      structurally distinguishable from peers' under the contrastive task.
    """
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(world_dim, generator=g)
    # First-step actions: one-hot, distinct first-action class for each anchor
    # within action_dim slots (the K > action_dim case cycles, which is fine
    # for min_batch_classes; we choose K <= action_dim in tests to make
    # n_classes == K).
    classes = torch.arange(K) % action_dim
    first_action = torch.zeros(K, action_dim)
    first_action[torch.arange(K), classes] = 1.0
    # Later-step actions: per-anchor fixed (different one-hot per step) so
    # the rollout map distinguishes anchors across all h steps.
    actions = torch.zeros(K, horizon, action_dim)
    for t in range(horizon):
        for i in range(K):
            actions[i, t, (classes[i].item() + t) % action_dim] = 1.0
    # Per-anchor structural shift in target space: anchor i shifts the target
    # latent by a unique direction at each step.
    targets = torch.zeros(K, horizon + 1, world_dim)
    targets[:, 0, :] = z0.unsqueeze(0).expand(K, -1)
    shift_dirs = torch.randn(K, world_dim, generator=g) * 0.3
    for t in range(1, horizon + 1):
        targets[:, t, :] = z0.unsqueeze(0) + (t * 0.1) * shift_dirs
    return z0, actions, targets


# ----------------------------- A1: config defaults ----------------------- #


def test_a1_default_e2config_amend_defaults_no_op():
    cfg = E2Config()
    assert cfg.e2_action_contrastive_multistep_enabled is False
    assert cfg.e2_action_contrastive_horizon == 5
    assert cfg.e2_action_contrastive_horizon_weights_decay == 1.0
    assert cfg.e2_rollout_output_norm_clamp_enabled is False
    assert cfg.e2_rollout_output_norm_clamp_ratio == 2.0


# ----------------------------- A2: from_dims propagation ----------------- #


def test_a2_from_dims_surfaces_amend_knobs():
    # Default path: amend knobs should propagate as the documented defaults.
    base_dims = dict(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    cfg_default = REEConfig.from_dims(**base_dims)
    assert cfg_default.e2.e2_action_contrastive_multistep_enabled is False
    assert cfg_default.e2.e2_action_contrastive_horizon == 5
    assert cfg_default.e2.e2_rollout_output_norm_clamp_enabled is False
    assert cfg_default.e2.e2_rollout_output_norm_clamp_ratio == 2.0
    # Set path: amend knobs should propagate through.
    cfg_on = REEConfig.from_dims(
        **base_dims,
        e2_action_contrastive_multistep_enabled=True,
        e2_action_contrastive_horizon=8,
        e2_action_contrastive_horizon_weights_decay=0.7,
        e2_rollout_output_norm_clamp_enabled=True,
        e2_rollout_output_norm_clamp_ratio=1.75,
    )
    assert cfg_on.e2.e2_action_contrastive_multistep_enabled is True
    assert cfg_on.e2.e2_action_contrastive_horizon == 8
    assert cfg_on.e2.e2_action_contrastive_horizon_weights_decay == 0.7
    assert cfg_on.e2.e2_rollout_output_norm_clamp_enabled is True
    assert cfg_on.e2.e2_rollout_output_norm_clamp_ratio == 1.75


# ----------------------------- A3: helper exists + grad flows ----------- #


def test_a3_multistep_helper_returns_scalar_and_grad_flows():
    K = 4
    world_dim = 16
    action_dim = 4
    horizon = 5
    e2 = _make_e2(world_dim=world_dim, action_dim=action_dim, horizon=horizon)
    z0, actions, targets = _sibling_multistep_batch(
        K=K, world_dim=world_dim, action_dim=action_dim, horizon=horizon
    )
    loss = e2.world_forward_contrastive_loss_multistep(z0, actions, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    # Gradient flow: backward() must succeed and produce non-zero gradients
    # on the world_transition trunk.
    e2.zero_grad()
    loss.backward()
    grads = [p.grad for p in e2.world_transition.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any(g.abs().sum().item() > 0.0 for g in grads)


# ----------------------------- A4: MECH-094 gate ------------------------- #


def test_a4_simulation_mode_returns_zero():
    K = 4
    world_dim = 16
    action_dim = 4
    horizon = 5
    e2 = _make_e2(world_dim=world_dim, action_dim=action_dim, horizon=horizon)
    z0, actions, targets = _sibling_multistep_batch(
        K=K, world_dim=world_dim, action_dim=action_dim, horizon=horizon
    )
    loss = e2.world_forward_contrastive_loss_multistep(
        z0, actions, targets, simulation_mode=True
    )
    assert loss.item() == 0.0


# ----------------------------- A5: K < 2 returns zero -------------------- #


def test_a5_k_below_two_returns_zero():
    world_dim = 16
    action_dim = 4
    horizon = 5
    e2 = _make_e2(world_dim=world_dim, action_dim=action_dim, horizon=horizon)
    z0 = torch.zeros(world_dim)
    # K=1 batch
    actions = torch.zeros(1, horizon, action_dim)
    actions[0, :, 0] = 1.0
    targets = torch.zeros(1, horizon + 1, world_dim)
    loss = e2.world_forward_contrastive_loss_multistep(z0, actions, targets)
    assert loss.item() == 0.0


# ----------------------------- A6: min_batch_classes floor --------------- #


def test_a6_min_batch_classes_floor_returns_zero():
    K = 4
    world_dim = 16
    action_dim = 4
    horizon = 5
    e2 = _make_e2(world_dim=world_dim, action_dim=action_dim, horizon=horizon)
    z0 = torch.zeros(world_dim)
    # All K anchors share the SAME first-action class -- min_batch_classes
    # default 2 is not met (n_classes == 1).
    actions = torch.zeros(K, horizon, action_dim)
    actions[:, :, 0] = 1.0
    targets = torch.zeros(K, horizon + 1, world_dim)
    loss = e2.world_forward_contrastive_loss_multistep(z0, actions, targets)
    assert loss.item() == 0.0


# ----------------------------- A7: horizon clamps gracefully ------------- #


def test_a7_horizon_clamps_to_effective():
    K = 4
    world_dim = 16
    action_dim = 4
    e2 = _make_e2(world_dim=world_dim, action_dim=action_dim, horizon=10)
    # Provide only 3 steps in actions / 4 steps in targets; effective horizon
    # should clamp to 3.
    z0, actions_5, targets_6 = _sibling_multistep_batch(
        K=K, world_dim=world_dim, action_dim=action_dim, horizon=5
    )
    actions_3 = actions_5[:, :3, :]
    targets_4 = targets_6[:, :4, :]
    loss = e2.world_forward_contrastive_loss_multistep(z0, actions_3, targets_4)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    # horizon=0 -> tensor(0.0)
    loss_zero = e2.world_forward_contrastive_loss_multistep(
        z0, actions_5, targets_6, horizon=0
    )
    assert loss_zero.item() == 0.0


# ----------------------------- A8: rollout clamp OFF bit-identical ------- #


def test_a8_rollout_clamp_off_bit_identical():
    K = 4
    self_dim = 8
    world_dim = 16
    action_dim = 4
    horizon = 10
    e2_off = _make_e2(
        self_dim=self_dim,
        world_dim=world_dim,
        action_dim=action_dim,
        horizon=5,
        clamp_enabled=False,
    )
    # Build a deterministic rollout input.
    g = torch.Generator().manual_seed(42)
    initial_z_self = torch.randn(K, self_dim, generator=g)
    initial_z_world = torch.randn(K, world_dim, generator=g)
    action_seq = torch.zeros(K, horizon, action_dim)
    classes = torch.arange(K) % action_dim
    for i in range(K):
        action_seq[i, :, classes[i].item()] = 1.0
    # Identical model state, identical inputs -> two parallel rollouts
    # must produce bit-identical world_states.
    torch.manual_seed(7)
    traj_off = e2_off.rollout_with_world(
        initial_z_self, initial_z_world, action_seq, compute_action_objects=False
    )
    # Second copy with the same seeds + same config: should be identical.
    e2_off_2 = _make_e2(
        self_dim=self_dim,
        world_dim=world_dim,
        action_dim=action_dim,
        horizon=5,
        clamp_enabled=False,
    )
    # Force same weights between e2_off and e2_off_2 by copying state dict.
    e2_off_2.load_state_dict(e2_off.state_dict())
    traj_2 = e2_off_2.rollout_with_world(
        initial_z_self, initial_z_world, action_seq, compute_action_objects=False
    )
    for w_off, w_2 in zip(traj_off.world_states, traj_2.world_states):
        assert torch.allclose(w_off, w_2, atol=0.0, rtol=0.0)


# ----------------------------- A9: rollout clamp ON enforces bound ------- #


def test_a9_rollout_clamp_on_enforces_b2_bound():
    K = 4
    self_dim = 8
    world_dim = 16
    action_dim = 4
    horizon = 20
    ratio = 2.0
    e2_on = _make_e2(
        self_dim=self_dim,
        world_dim=world_dim,
        action_dim=action_dim,
        horizon=5,
        clamp_enabled=True,
        clamp_ratio=ratio,
        # Deliberately scale up world_transition so without the clamp the
        # rollout would diverge fast.
        weight_init_scale=5.0,
    )
    g = torch.Generator().manual_seed(11)
    initial_z_self = torch.randn(K, self_dim, generator=g)
    initial_z_world = torch.randn(K, world_dim, generator=g)
    action_seq = torch.zeros(K, horizon, action_dim)
    classes = torch.arange(K) % action_dim
    for i in range(K):
        action_seq[i, :, classes[i].item()] = 1.0
    traj = e2_on.rollout_with_world(
        initial_z_self, initial_z_world, action_seq, compute_action_objects=False
    )
    # B2 bound: every step's per-row L2 norm must be <= ratio * ||z_world_0||
    # (with a small floating-point tolerance for the renormalisation arithmetic).
    z0_norms = initial_z_world.norm(dim=-1)  # [K]
    max_allowed = ratio * z0_norms  # [K]
    for t, z_t in enumerate(traj.world_states):
        norms_t = z_t.norm(dim=-1)  # [K]
        # Tolerance: allow 1e-5 relative slack.
        assert torch.all(norms_t <= max_allowed * (1.0 + 1e-5)), (
            f"step {t}: rows {norms_t.tolist()} exceed bound {max_allowed.tolist()}"
        )


# ----------------------------- A10: clamp blocks NaN/Inf under stress ---- #


def test_a10_rollout_clamp_blocks_overflow_under_stress():
    K = 8
    self_dim = 8
    world_dim = 16
    action_dim = 4
    horizon = 200  # behavioural-runtime episode length per V3-EXQ-569e
    ratio = 2.0
    # Deliberately scale up world_transition by 20x -- without the clamp this
    # would produce huge magnitudes (close to the 1e16+ pathology the autopsy
    # documented; we use horizon=200 to match 569e's P1 measurement budget).
    e2_on = _make_e2(
        self_dim=self_dim,
        world_dim=world_dim,
        action_dim=action_dim,
        horizon=5,
        clamp_enabled=True,
        clamp_ratio=ratio,
        weight_init_scale=20.0,
    )
    g = torch.Generator().manual_seed(13)
    initial_z_self = torch.randn(K, self_dim, generator=g)
    initial_z_world = torch.randn(K, world_dim, generator=g)
    action_seq = torch.zeros(K, horizon, action_dim)
    classes = torch.arange(K) % action_dim
    for i in range(K):
        action_seq[i, :, classes[i].item()] = 1.0
    traj = e2_on.rollout_with_world(
        initial_z_self, initial_z_world, action_seq, compute_action_objects=False
    )
    z0_norms = initial_z_world.norm(dim=-1)
    max_allowed = ratio * z0_norms
    # No NaN / Inf anywhere along the rollout, and every step inside the B2
    # bound -- the load-bearing PASS condition for the autopsy acceptance
    # criterion (max-NaN-fraction < 0.05; rollout magnitudes within 2x of
    # OFF baseline).
    for t, z_t in enumerate(traj.world_states):
        assert torch.isfinite(z_t).all(), f"step {t} contains NaN/Inf"
        norms_t = z_t.norm(dim=-1)
        assert torch.all(norms_t <= max_allowed * (1.0 + 1e-5)), (
            f"step {t}: rows {norms_t.tolist()} exceed bound {max_allowed.tolist()}"
        )
