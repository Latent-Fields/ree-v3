"""SD-070 contracts: z_world P0 anti-collapse encoder-training recipe.

Covers the properties the recipe's correctness rests on:
  C1  target derivation is right (presence, nearest-distance bucketing, absent-saturation)
  C2  the local-view index layout agrees with SplitEncoder's own constants
  C3  class balancing never lets an absent class dominate or NaN the loss
  C4  the anti-collapse penalty actually penalises collapse, and is well-defined at n<2
  C5  the trainer trains the z_world path AND NOTHING ELSE
  C6  SD-070 is bit-identical OFF by construction (no substrate surface changed)
  C7  the recipe measurably raises effective dimensionality on a collapse-inducing task
"""

import pytest
import torch
import torch.nn.functional as F

from ree_core.latent.zworld_p0 import (
    HAZARD_ENTITY_INDEX,
    LOCAL_VIEW_CELLS,
    LOCAL_VIEW_ENTITY_STRIDE,
    RESOURCE_ENTITY_INDEX,
    ZWorldP0Config,
    ZWorldP0Trainer,
    balanced_class_weights,
    chebyshev_offsets,
    entity_presence_mask,
    scene_structure_targets,
    variance_covariance_penalty,
)
from ree_core.latent.stack import SplitEncoder

WORLD_OBS_DIM = 275
LOCAL_VIEW_WIDTH = LOCAL_VIEW_CELLS * LOCAL_VIEW_ENTITY_STRIDE  # 175


def _obs_with(entity_index, cells, batch=1, dim=WORLD_OBS_DIM):
    """world_obs with `entity_index` occupying the given local-view cells."""
    o = torch.zeros(batch, dim)
    for c in cells:
        o[:, c * LOCAL_VIEW_ENTITY_STRIDE + entity_index] = 1.0
    return o


# --- C1 target derivation -------------------------------------------------------------
def test_c1_presence_flags_track_occupancy():
    t = scene_structure_targets(_obs_with(HAZARD_ENTITY_INDEX, [0]))
    assert int(t["hazard_present"][0]) == 1
    assert int(t["resource_present"][0]) == 0

    t = scene_structure_targets(torch.zeros(1, WORLD_OBS_DIM))
    assert int(t["hazard_present"][0]) == 0
    assert int(t["resource_present"][0]) == 0


def test_c1_distance_is_chebyshev_to_nearest():
    centre = (LOCAL_VIEW_CELLS - 1) // 2  # cell 12 == (2,2)
    # hazard on the agent's own cell -> distance 0
    t = scene_structure_targets(_obs_with(HAZARD_ENTITY_INDEX, [centre]))
    assert int(t["hazard_distance"][0]) == 0
    # corner cell (0,0) is Chebyshev distance 2 from centre
    t = scene_structure_targets(_obs_with(HAZARD_ENTITY_INDEX, [0]))
    assert int(t["hazard_distance"][0]) == 2
    # NEAREST wins when several are present
    t = scene_structure_targets(_obs_with(HAZARD_ENTITY_INDEX, [0, centre]))
    assert int(t["hazard_distance"][0]) == 0


def test_c1_absent_entity_saturates_to_last_bucket():
    """An absent entity must read as 'nothing near' (last bucket), never as distance 0 and
    never as a missing label -- an absent hazard is information, not an omission."""
    for nb in (2, 4, 6):
        t = scene_structure_targets(torch.zeros(1, WORLD_OBS_DIM), n_distance_buckets=nb)
        assert int(t["hazard_distance"][0]) == nb - 1
        assert int(t["resource_distance"][0]) == nb - 1


def test_c1_targets_are_in_range_and_batched():
    obs = torch.rand(9, WORLD_OBS_DIM).round()
    t = scene_structure_targets(obs, n_distance_buckets=4)
    for k, v in t.items():
        assert v.shape == (9,), k
        assert v.dtype == torch.long, k
        assert int(v.min()) >= 0 and int(v.max()) <= 3, k


def test_c1_accepts_unbatched_input():
    t = scene_structure_targets(torch.zeros(WORLD_OBS_DIM))
    assert t["hazard_present"].shape == (1,)


# --- C2 layout agreement --------------------------------------------------------------
def test_c2_hazard_indices_match_split_encoder():
    """The target derivation and SplitEncoder must read the SAME cells for hazard, or the
    encoder would be supervised on channels it does not receive."""
    obs = torch.arange(WORLD_OBS_DIM, dtype=torch.float32).unsqueeze(0)
    mine = entity_presence_mask(obs, HAZARD_ENTITY_INDEX)
    theirs = obs[:, SplitEncoder.HAZARD_INDICES]
    assert torch.equal(mine, theirs)


def test_c2_resource_and_hazard_channels_are_distinct():
    o = _obs_with(RESOURCE_ENTITY_INDEX, [0])
    t = scene_structure_targets(o)
    assert int(t["resource_present"][0]) == 1
    assert int(t["hazard_present"][0]) == 0


def test_c2_chebyshev_offsets_shape_and_range():
    offs = chebyshev_offsets()
    assert offs.shape == (LOCAL_VIEW_CELLS,)
    assert float(offs.min()) == 0.0
    assert float(offs.max()) == 2.0


# --- C3 class balancing ---------------------------------------------------------------
def test_c3_absent_class_gets_zero_weight():
    y = torch.tensor([0, 0, 0, 1])
    w = balanced_class_weights(y, 4)
    assert float(w[2]) == 0.0 and float(w[3]) == 0.0
    assert float(w[0]) > 0.0 and float(w[1]) > 0.0


def test_c3_rare_class_outweighs_common_class():
    y = torch.tensor([0] * 95 + [1] * 5)
    w = balanced_class_weights(y, 2)
    assert float(w[1]) > float(w[0])


def test_c3_expected_sample_weight_is_one():
    """The FREQUENCY-weighted mean weight must be 1.0 -- the expected weight of a randomly
    drawn sample. That is what keeps the weighted CE on the same scale as an unweighted
    one, so the other loss weights do not need retuning when the balance shifts. (The
    plain mean over present classes is not 1 and grows as the balance skews.)"""
    y = torch.tensor([0] * 90 + [1] * 9 + [2])
    w = balanced_class_weights(y, 3)
    counts = torch.bincount(y, minlength=3).float()
    expected = float((counts / counts.sum() * w).sum())
    assert expected == pytest.approx(1.0, abs=1e-5)


def test_c3_all_one_class_does_not_nan():
    w = balanced_class_weights(torch.zeros(10, dtype=torch.long), 3)
    assert torch.isfinite(w).all()
    loss = F.cross_entropy(torch.randn(10, 3), torch.zeros(10, dtype=torch.long), weight=w)
    assert torch.isfinite(loss)


def test_c3_empty_labels_returns_finite_zero_weights():
    w = balanced_class_weights(torch.zeros(0, dtype=torch.long), 3)
    assert torch.isfinite(w).all() and float(w.sum()) == 0.0


# --- C4 anti-collapse penalty ---------------------------------------------------------
def test_c4_collapsed_batch_penalised_more_than_spread_batch():
    torch.manual_seed(0)
    spread = torch.randn(64, 16)
    collapsed = torch.randn(64, 1) @ torch.ones(1, 16)  # rank 1: every dim identical
    v_s, c_s = variance_covariance_penalty(spread)
    v_c, c_c = variance_covariance_penalty(collapsed)
    assert float(c_c) > float(c_s), "covariance term must punish rank-1 collapse"
    assert float(v_c + c_c) > float(v_s + c_s)


def test_c4_variance_term_alone_cannot_detect_correlation():
    """Documents WHY the covariance term is not optional: unit-variance but perfectly
    correlated dimensions still occupy one effective dimension, and the variance hinge is
    blind to that. This is the measured reason w_cov is the participation-ratio lever."""
    z = torch.randn(128, 1).expand(128, 8).contiguous()
    # Normalise with the SAME (population) convention the penalty uses, so the variance
    # hinge is exactly satisfied and the test isolates the covariance term's role.
    z = z / (z.var(dim=0, unbiased=False, keepdim=True).sqrt() + 1e-8)
    v, c = variance_covariance_penalty(z, variance_gamma=1.0)
    assert float(v) == pytest.approx(0.0, abs=1e-3), "variance hinge sees nothing wrong"
    assert float(c) > 0.1, "covariance term is what sees the collapse"


def test_c4_below_unit_variance_is_penalised():
    z = torch.randn(64, 8) * 0.01
    v, _c = variance_covariance_penalty(z, variance_gamma=1.0)
    assert float(v) > 0.9


def test_c4_degenerate_batch_is_zero_not_nan():
    for n in (0, 1):
        v, c = variance_covariance_penalty(torch.randn(n, 4))
        assert torch.isfinite(v) and torch.isfinite(c)
        assert float(v) == 0.0 and float(c) == 0.0


def test_c4_rejects_non_2d_input():
    with pytest.raises(ValueError):
        variance_covariance_penalty(torch.randn(4, 5, 6))


def test_c4_penalty_is_differentiable():
    z = torch.randn(32, 8, requires_grad=True)
    v, c = variance_covariance_penalty(z)
    (v + c).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()


# --- C5 / C6 / C7 trainer contracts ---------------------------------------------------
def _tiny_stack(world_dim=16, body_obs_dim=8, world_obs_dim=WORLD_OBS_DIM):
    """A minimal object exposing the SplitEncoder surface the trainer requires."""
    class _Stack:
        pass
    s = _Stack()
    s.split_encoder = SplitEncoder(
        body_obs_dim=body_obs_dim,
        world_obs_dim=world_obs_dim,
        self_dim=8,
        world_dim=world_dim,
        topdown_dim=0,
        hidden_dim=32,
        use_resource_proximity_head=True,
    )
    return s


def _fill(trainer, n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        obs = torch.zeros(WORLD_OBS_DIM)
        cell = int(torch.randint(0, LOCAL_VIEW_CELLS, (1,), generator=g))
        ent = HAZARD_ENTITY_INDEX if i % 2 else RESOURCE_ENTITY_INDEX
        obs[cell * LOCAL_VIEW_ENTITY_STRIDE + ent] = 1.0
        obs[LOCAL_VIEW_WIDTH:] = torch.rand(WORLD_OBS_DIM - LOCAL_VIEW_WIDTH, generator=g)
        trainer.observe(obs, float(torch.rand(1, generator=g)))


def test_c5_trains_world_path_and_leaves_self_path_untouched():
    """The z_world P0 must not silently train z_self. If it did, every downstream
    self-stream result run after a P0 would be confounded by it."""
    stack = _tiny_stack()
    se = stack.split_encoder
    before_world = {n: p.clone() for n, p in se.world_encoder.named_parameters()}
    before_self = {n: p.clone() for n, p in se.self_encoder.named_parameters()}
    before_prec_self = se.self_precision_logit.clone()

    t = ZWorldP0Trainer(stack, ZWorldP0Config(epochs=2, batch_size=32, seed=1))
    _fill(t)
    t.train()

    assert any(not torch.equal(p, before_world[n])
               for n, p in se.world_encoder.named_parameters()), "world path did not move"
    for n, p in se.self_encoder.named_parameters():
        assert torch.equal(p, before_self[n]), "self_encoder was modified: %s" % n
    assert torch.equal(se.self_precision_logit, before_prec_self)


def test_c5_moves_world_precision_logit():
    stack = _tiny_stack()
    before = stack.split_encoder.world_precision_logit.clone()
    t = ZWorldP0Trainer(stack, ZWorldP0Config(epochs=2, batch_size=32, seed=1))
    _fill(t)
    t.train()
    assert not torch.equal(stack.split_encoder.world_precision_logit, before)


def test_c5_rejects_a_stack_without_a_split_encoder():
    with pytest.raises(ValueError):
        ZWorldP0Trainer(object(), ZWorldP0Config())


def test_c5_refuses_to_train_on_an_undersized_buffer():
    """Training on a handful of samples would produce meaningless batch statistics; the
    trainer must say so rather than return a confident-looking result."""
    t = ZWorldP0Trainer(_tiny_stack(), ZWorldP0Config(batch_size=32))
    _fill(t, n=4)
    with pytest.raises(ValueError):
        t.train()


def test_c5_buffer_does_not_alias_caller_tensors():
    """The env reuses observation tensors between steps; a buffer that aliased them would
    silently fill with copies of the final observation."""
    t = ZWorldP0Trainer(_tiny_stack(), ZWorldP0Config())
    obs = torch.zeros(WORLD_OBS_DIM)
    t.observe(obs, 0.5)
    obs[0] = 99.0
    assert float(t._obs[0][0]) == 0.0


def test_c5_reports_holdout_discriminativeness():
    """PR can be held up vacuously by regularisation alone, so the trainer must always
    report whether the encoder actually learned anything."""
    stack = _tiny_stack()
    t = ZWorldP0Trainer(stack, ZWorldP0Config(epochs=3, batch_size=32, seed=2))
    _fill(t, n=300)
    st = t.train()
    assert "holdout" in st
    assert st["n_buffered"] == 300
    assert st["used_reconstruction_head"] is True


def test_c5_reconstruction_head_is_disablable():
    stack = _tiny_stack()
    t = ZWorldP0Trainer(
        stack, ZWorldP0Config(epochs=2, batch_size=32, reconstruction_weight=0.0))
    _fill(t)
    st = t.train()
    assert st["used_reconstruction_head"] is False


def test_c5_tolerates_missing_proximity_targets():
    """resource_field_view is absent unless use_proxy_fields=True; a None target must skip
    the SD-018 leg rather than poison the loss with NaN."""
    stack = _tiny_stack()
    t = ZWorldP0Trainer(stack, ZWorldP0Config(epochs=2, batch_size=32))
    for i in range(200):
        obs = torch.zeros(WORLD_OBS_DIM)
        obs[(i % LOCAL_VIEW_CELLS) * LOCAL_VIEW_ENTITY_STRIDE + HAZARD_ENTITY_INDEX] = 1.0
        t.observe(obs, None)
    st = t.train()
    assert st["used_proximity_head"] is False
    assert st["final_loss"] is not None
    assert st["final_loss"] == st["final_loss"]  # not NaN


# --- C6 bit-identical OFF -------------------------------------------------------------
def test_c6_no_new_latent_stack_config_fields():
    """SD-070 adds NO config surface: the no-op guarantee is structural (nothing builds a
    trainer unless an experiment asks) rather than flag-based, so there is no flag that can
    be left in the wrong state. This test pins that design decision."""
    from ree_core.utils.config import LatentStackConfig
    fields = set(LatentStackConfig.__dataclass_fields__)
    assert not any("zworld_p0" in f or "sd070" in f for f in fields)


def test_c6_split_encoder_gains_no_grounding_heads():
    se = _tiny_stack().split_encoder
    assert not hasattr(se, "hazard_present_head")
    assert not hasattr(se, "grounding_heads")


def test_c6_agent_gains_no_sd070_method():
    from ree_core.agent import REEAgent
    assert not any(n.startswith("compute_world_grounding") for n in dir(REEAgent))


# --- C7 the recipe does what it claims ------------------------------------------------
def test_c7_recipe_raises_effective_dimensionality_over_a_collapsing_objective():
    """End-to-end property test, on synthetic data, of the whole point of SD-070.

    Trains the SAME encoder twice from the same init: once on a near-constant single-scalar
    objective (the shape of the old P0 -- a 95%-saturated CE plus one scalar regression),
    once with the SD-070 recipe. The first should collapse the representation; the second
    should retain substantially more effective dimensionality.
    """
    def participation_ratio(z):
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / float(max(z.shape[0] - 1, 1))
        e = torch.linalg.eigvalsh(cov).clamp(min=0.0)
        s1, s2 = float(e.sum()), float((e ** 2).sum())
        return (s1 * s1 / s2) if s2 > 1e-24 else 0.0

    torch.manual_seed(7)
    stack_a, stack_b = _tiny_stack(world_dim=32), _tiny_stack(world_dim=32)
    stack_b.split_encoder.load_state_dict(stack_a.split_encoder.state_dict())

    t = ZWorldP0Trainer(stack_a, ZWorldP0Config())
    _fill(t, n=400, seed=3)
    obs = torch.stack(t._obs)

    with torch.no_grad():
        z_init = t._z_world_path(obs)
    pr_init = participation_ratio(z_init)

    # (a) the collapsing objective: 95% saturated CE + one scalar
    se_a = stack_a.split_encoder
    head = torch.nn.Linear(32, 3)
    y = torch.zeros(obs.shape[0], dtype=torch.long)
    y[: max(obs.shape[0] // 20, 1)] = 1
    opt = torch.optim.Adam(
        list(se_a.world_encoder.parameters()) + [se_a.world_precision_logit]
        + list(head.parameters()), lr=1e-3)
    for _ in range(400):
        idx = torch.randint(0, obs.shape[0], (64,))
        z = se_a.world_encoder(obs[idx]) * torch.sigmoid(se_a.world_precision_logit)
        loss = F.cross_entropy(head(z), y[idx]) + F.mse_loss(z.mean(-1), torch.zeros(64))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pr_old = participation_ratio(
            se_a.world_encoder(obs) * torch.sigmoid(se_a.world_precision_logit))

    # (b) the SD-070 recipe from the same init
    t_b = ZWorldP0Trainer(stack_b, ZWorldP0Config(epochs=10, batch_size=64, seed=3))
    for o in t._obs:
        t_b.observe(o, 0.5)
    t_b.train()
    with torch.no_grad():
        pr_new = participation_ratio(t_b._z_world_path(obs))

    assert pr_old < pr_init, "control did not reproduce collapse (pr %.3f -> %.3f)" % (
        pr_init, pr_old)
    assert pr_new > pr_old * 1.5, (
        "SD-070 did not retain dimensionality over the collapsing objective "
        "(init %.3f, old-P0 %.3f, SD-070 %.3f)" % (pr_init, pr_old, pr_new))
    assert pr_new >= 2.0, "SD-070 fell below the absolute PR floor (%.3f)" % pr_new
