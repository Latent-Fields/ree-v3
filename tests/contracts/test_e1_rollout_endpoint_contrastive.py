"""
Contract: SD-e1-rollout-consistency-training ITEM 3 -- E1 rollout-endpoint
contrastive objective (e1_rollout_sequence_divergence_*).

Licensed by the confirmed V3-EXQ-976 autopsy (user decision Q1, 2026-09-02):
ITEM 2 (candidate 1, trajectory accuracy) made NO absolute progress on the
0.1 cr_ratio / 0.002 e1coe_score_var evaluator bars and DAMPS per-action
divergence growth at depth on 8/8 ON cells -- an accuracy objective trained
against observed intermediate states works AGAINST the per-action divergence
the evaluator needs, because the cheapest way to lower per-step MSE at depth
is to let the deep trajectory collapse toward a common, low-variance path.

This is the candidate ITEM 2's own landing note designed and deliberately
withheld ("why_not_contrastive" on E1Config.e1_rollout_consistency_enabled).
It constrains the ITERATED map under candidate action SEQUENCES -- what the
C3 evaluator actually consumes (40 sequence endpoints, not 40 single
actions) -- via a rollout-ENDPOINT InfoNCE: same-sequence predicted/observed
endpoints pulled together, distinct-sequence endpoints pushed apart. It
carries NO per-step MSE against intermediate observed states, so unlike
ITEM 2 it cannot be minimised by trajectory collapse.

This file pins:
  * the OFF path is bit-identical (no module or parameter is constructed;
    pure function of existing weights, same as ITEM 2 and SD-056's E2
    contrastive helpers);
  * the ON path is NOT VACUOUS -- finite, nonzero, and genuinely trains
    output_proj and the LSTM (the ~675x crush's location);
  * the ON path rewards ENDPOINT DIVERGENCE across distinct action
    sequences and specifically does NOT reward trajectory accuracy -- a
    collapsed-endpoint configuration (the ITEM 2 damping signature) is
    exactly what this loss penalises hardest, and its gradient concretely
    pushes each candidate's predicted endpoint toward its OWN observed
    target and away from the OTHER candidates' targets.

It does NOT assert that this objective clears the evaluator bars, or that it
beats ITEM 2 -- that is the validation experiment's job, the same boundary
the ITEM 1/2/residual-knob contracts hold.

Every assertion is upstream of any discrete quantizer -- on tensors, losses
and gradients, never on a committed action (see CLAUDE.md "Running the test
suite" on cross-machine-class torch.multinomial divergence).

See REE_assembly/docs/architecture/sd_e1_rollout_consistency_training.md.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.predictors.e1_deep import E1DeepPredictor  # noqa: E402
from ree_core.utils.config import E1Config, REEConfig    # noqa: E402

ACTION_DIM = 4
SELF_DIM = 32
WORLD_DIM = 32
TOTAL_DIM = SELF_DIM + WORLD_DIM


def _e1(**kw) -> E1DeepPredictor:
    """
    Build an E1 from a fixed seed and put it in EVAL mode.

    eval() is load-bearing for the same reason it is in the ITEM 1/2 files:
    transition_rnn is built with num_layers=3, so nn.LSTM applies
    dropout=0.1, and in train mode two identical calls differ by dropout
    noise alone -- which would let the non-vacuity tests pass on dropout
    noise while the contrastive gradient was inert.
    """
    torch.manual_seed(0)
    cfg = E1Config(self_dim=SELF_DIM, world_dim=WORLD_DIM,
                   latent_dim=TOTAL_DIM, **kw)
    e1 = E1DeepPredictor(cfg)
    e1.eval()
    return e1


def _onehot_seq(batch: int, horizon: int, action_idx, seed: int = 3) -> torch.Tensor:
    """
    Build a [batch, horizon, ACTION_DIM] one-hot action-sequence batch.

    action_idx: either a single int (every candidate/step gets that action --
    used for the OFF-identity check) or a [batch] sequence of ints (each
    candidate i gets a DISTINCT constant action across the horizon -- the K
    sibling-sequence shape this objective is designed for).
    """
    torch.manual_seed(seed)
    seq = torch.zeros(batch, horizon, ACTION_DIM)
    if isinstance(action_idx, int):
        seq[:, :, action_idx] = 1.0
    else:
        for i, idx in enumerate(action_idx):
            seq[i, :, idx] = 1.0
    return seq


def _from_dims(**kw) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=ACTION_DIM, **kw
    )


def _rollout(e1, init, horizon, actions):
    """Rollout through the module's own path, hidden state left undisturbed."""
    saved = e1._hidden_state
    e1.reset_hidden_state()
    with torch.no_grad():
        preds = e1.predict_long_horizon(init, horizon=horizon, actions=actions)
    e1._hidden_state = saved
    return preds


# --------------------------------------------------------------------------- #
# Defaults are no-op                                                          #
# --------------------------------------------------------------------------- #

def test_default_is_off():
    cfg = E1Config()
    assert cfg.e1_rollout_sequence_divergence_enabled is False
    assert cfg.e1_rollout_sequence_divergence_horizon == 5
    assert cfg.e1_rollout_sequence_divergence_temperature == 0.1
    assert cfg.e1_rollout_sequence_divergence_min_batch_classes == 2


def test_knob_adds_no_parameters():
    """
    No module and no parameter is constructed in either setting -- the
    objective is a pure function of existing weights, exactly like ITEM 2
    and SD-056's E2 contrastive helpers.
    """
    off = _e1()
    on = _e1(e1_rollout_sequence_divergence_enabled=True,
             e1_rollout_sequence_divergence_temperature=0.05)
    p_off = list(off.parameters())
    p_on = list(on.parameters())
    assert sum(p.numel() for p in p_off) == sum(p.numel() for p in p_on)
    assert all(torch.equal(a, b) for a, b in zip(p_off, p_on))


@pytest.mark.parametrize("action_conditioned", [False, True])
def test_enabling_the_flag_does_not_change_the_rollout(action_conditioned):
    """
    The flag gates an OBJECTIVE, not the forward path. Turning it on must not
    perturb prediction.
    """
    kw = dict(action_conditioned_transition=action_conditioned)
    off = _e1(**kw)
    on = _e1(e1_rollout_sequence_divergence_enabled=True, **kw)
    init = torch.randn(3, TOTAL_DIM)
    actions = _onehot_seq(3, 5, [0, 1, 2]) if action_conditioned else None
    a = _rollout(off, init, 5, actions)
    b = _rollout(on, init, 5, actions)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# ON: non-vacuous, real training signal                                       #
# --------------------------------------------------------------------------- #

def test_on_is_finite_and_nonzero_and_action_dependent():
    """
    Baseline non-vacuity. Uses ACTION_CONDITIONED=True: with the master
    switch off, predict_long_horizon ignores the action sequences entirely
    (ITEM 1's own contract), so every candidate's endpoint is identical by
    construction and the loss saturates to the degenerate log(K) with zero
    gradient -- a documented, expected null, not a bug. This test targets the
    informative regime.
    """
    e1 = _e1(action_conditioned_transition=True)
    torch.manual_seed(5)
    init = torch.randn(4, TOTAL_DIM)
    actions = _onehot_seq(4, 5, [0, 1, 2, 3])
    torch.manual_seed(7)
    targets = torch.randn(4, TOTAL_DIM)
    loss = e1.rollout_sequence_divergence_loss(init, actions, targets)
    assert torch.isfinite(loss)
    assert loss.item() != 0.0


def test_gradient_reaches_output_proj_and_the_lstm():
    """
    The ~675x crush ITEM 2's design doc names as the real target sits at the
    LSTM + output_proj stage. An objective that cannot deliver gradient there
    cannot possibly move it.
    """
    e1 = _e1(action_conditioned_transition=True)
    torch.manual_seed(5)
    init = torch.randn(4, TOTAL_DIM)
    actions = _onehot_seq(4, 5, [0, 1, 2, 3])
    torch.manual_seed(7)
    targets = torch.randn(4, TOTAL_DIM)
    e1.zero_grad()
    e1.rollout_sequence_divergence_loss(init, actions, targets).backward()
    assert e1.output_proj[0].weight.grad is not None
    assert e1.output_proj[0].weight.grad.abs().sum() > 0
    assert e1.transition_rnn.weight_ih_l0.grad is not None
    assert e1.transition_rnn.weight_ih_l0.grad.abs().sum() > 0


def test_off_action_conditioning_is_a_documented_null_not_an_error():
    """
    With action_conditioned_transition=False, every candidate's rollout is
    identical regardless of action sequence (all K candidates broadcast from
    ONE shared initial state -- load-bearing for this test, see below), so
    every logit is exactly equal by construction -> the loss is a finite,
    well-defined uniform log(K). This must not raise or silently do
    something else -- it is the documented "flag gates an objective, not the
    forward path" boundary this method deliberately does not enforce itself.

    The gradient that reaches the SHARED trunk (output_proj) is exactly zero
    in real arithmetic (see the method's own docstring for why: the pulls
    toward each candidate's own target sum to a pull toward their MEAN,
    which self-cancels once every candidate is an identical function of one
    shared input) -- but the true value is 0.0 only in exact arithmetic, so
    this is checked as "orders of magnitude below an informative gradient"
    rather than a fixed absolute tolerance, which would be float32-noise
    flaky in exactly the way CLAUDE.md's cross-machine-class note warns
    about for tighter-than-necessary checks.
    """
    e1_off = _e1(action_conditioned_transition=False)
    e1_on = _e1(action_conditioned_transition=True)
    torch.manual_seed(5)
    # ONE shared initial state, broadcast to all K candidates -- required for
    # the OFF collapse to actually hold. A per-candidate DISTINCT init state
    # (as most other tests in this file use, since it does not matter there)
    # would make the K endpoints differ even with actions ignored.
    init_shared = torch.randn(1, TOTAL_DIM)
    actions = _onehot_seq(4, 5, [0, 1, 2, 3])
    torch.manual_seed(7)
    targets = torch.randn(4, TOTAL_DIM)

    loss_off = e1_off.rollout_sequence_divergence_loss(init_shared, actions, targets)
    assert torch.isfinite(loss_off)
    assert loss_off.item() == pytest.approx(torch.log(torch.tensor(4.0)).item(), abs=1e-4)

    e1_off.zero_grad()
    e1_off.rollout_sequence_divergence_loss(init_shared, actions, targets).backward()
    off_grad = e1_off.output_proj[0].weight.grad.abs().sum().item()

    e1_on.zero_grad()
    e1_on.rollout_sequence_divergence_loss(init_shared, actions, targets).backward()
    on_grad = e1_on.output_proj[0].weight.grad.abs().sum().item()

    assert on_grad > 0.0
    assert off_grad < on_grad * 1e-3


# --------------------------------------------------------------------------- #
# ON rewards endpoint DIVERGENCE across distinct sequences, NOT accuracy      #
# --------------------------------------------------------------------------- #

def test_collapsed_endpoints_are_penalised_hardest():
    """
    THE load-bearing distinction from ITEM 2 (trajectory accuracy). Construct
    two candidates whose predicted endpoints are made to COLLAPSE onto the
    same point (mimicking the ITEM-2 damping signature), against two DISTINCT
    observed targets. The contrastive loss on the collapsed pair must be
    HIGHER than on a pair whose predictions are correctly separated toward
    their own targets -- the opposite of what a trajectory-accuracy (MSE-to-
    own-target) objective would necessarily show, since MSE-to-own-target
    does not penalise two predictions being close to EACH OTHER at all.
    """
    e1 = _e1(action_conditioned_transition=True)
    tau = e1.config.e1_rollout_sequence_divergence_temperature

    t0 = torch.tensor([[1.0] + [0.0] * (TOTAL_DIM - 1)])
    t1 = torch.tensor([[-1.0] + [0.0] * (TOTAL_DIM - 1)])
    targets = torch.cat([t0, t1], dim=0)  # [2, TOTAL_DIM], far apart

    # Collapsed: both predicted endpoints sit at the midpoint.
    mid = (t0 + t1) / 2.0
    collapsed_preds = torch.cat([mid, mid], dim=0)

    # Separated: each prediction sits exactly on its own target.
    separated_preds = targets.clone()

    def _ce(preds):
        diffs = preds.unsqueeze(0) - targets.unsqueeze(1)
        sq = diffs.pow(2).sum(dim=-1)
        logits = -sq / tau
        labels = torch.arange(2)
        return F.cross_entropy(logits, labels)

    collapsed_loss = _ce(collapsed_preds)
    separated_loss = _ce(separated_preds)
    assert collapsed_loss > separated_loss
    # Collapsed logits are exactly tied -> softmax is uniform -> CE == log(2).
    assert collapsed_loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item(), abs=1e-5)


def test_gradient_pushes_toward_own_target_and_away_from_others():
    """
    A concrete, sign-level test that the objective is genuinely CONTRASTIVE
    (relative structure across candidates) and not merely "close to a
    target" (which trajectory accuracy already provides). At a collapsed
    starting point, the gradient of the loss with respect to prediction i
    must point TOWARD candidate i's own target and AWAY from the other
    candidate's target -- i.e. -grad has positive dot product with
    (target_i - pred_i) and negative dot product with (target_j - pred_i)
    for j != i. A pure per-candidate accuracy objective (MSE-to-own-target)
    would produce the first property but says nothing about the second;
    this asserts BOTH, which only a contrastive term guarantees.
    """
    tau = 0.1
    t0 = torch.tensor([[1.0, 0.0]])
    t1 = torch.tensor([[-1.0, 0.0]])
    targets = torch.cat([t0, t1], dim=0)  # [2, 2]

    mid = torch.tensor([[0.0, 0.0]])
    preds = torch.cat([mid, mid], dim=0).clone().requires_grad_(True)

    diffs = preds.unsqueeze(0) - targets.unsqueeze(1)
    sq = diffs.pow(2).sum(dim=-1)
    logits = -sq / tau
    labels = torch.arange(2)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    grad0 = preds.grad[0]
    own_dir0 = (targets[0] - preds[0]).detach()
    other_dir0 = (targets[1] - preds[0]).detach()
    assert torch.dot(-grad0, own_dir0).item() > 0
    assert torch.dot(-grad0, other_dir0).item() < 0

    grad1 = preds.grad[1]
    own_dir1 = (targets[1] - preds[1]).detach()
    other_dir1 = (targets[0] - preds[1]).detach()
    assert torch.dot(-grad1, own_dir1).item() > 0
    assert torch.dot(-grad1, other_dir1).item() < 0


# --------------------------------------------------------------------------- #
# Degeneracy floor: distinct SEQUENCES, not distinct first actions            #
# --------------------------------------------------------------------------- #

def test_distinct_sequence_floor_counts_full_sequences_not_first_action():
    """
    Two candidates sharing the SAME first action but diverging on a later
    step must count as 2 distinct sequences (n_distinct=2), clearing a
    min_batch_classes=2 floor -- deliberately stricter than SD-056's first-
    action-only floor on E2.
    """
    e1 = _e1(action_conditioned_transition=True,
             e1_rollout_sequence_divergence_min_batch_classes=2)
    torch.manual_seed(5)
    init = torch.randn(2, TOTAL_DIM)
    actions = torch.zeros(2, 3, ACTION_DIM)
    actions[:, 0, 0] = 1.0   # both candidates: same first action
    actions[0, 1, 1] = 1.0   # candidate 0 diverges at step 1
    actions[1, 1, 2] = 1.0   # candidate 1 diverges differently
    actions[:, 2, 0] = 1.0
    torch.manual_seed(7)
    targets = torch.randn(2, TOTAL_DIM)
    loss = e1.rollout_sequence_divergence_loss(init, actions, targets)
    assert loss.item() != 0.0


def test_identical_sequences_are_degenerate_and_return_zero():
    """
    Two candidates with the IDENTICAL action sequence give n_distinct=1,
    below the min_batch_classes=2 floor -> the zero (grad-connected) return,
    not a NaN/garbage cross-entropy over a rank-deficient logit matrix.
    """
    e1 = _e1(action_conditioned_transition=True)
    torch.manual_seed(5)
    init = torch.randn(2, TOTAL_DIM)
    actions = _onehot_seq(2, 3, 1)  # both candidates: identical sequence
    torch.manual_seed(7)
    targets = torch.randn(2, TOTAL_DIM)
    loss = e1.rollout_sequence_divergence_loss(init, actions, targets)
    assert loss.item() == 0.0


def test_config_min_batch_classes_is_read_when_no_override_passed():
    """A stricter floor set on the config must actually gate, not just exist."""
    e1 = _e1(action_conditioned_transition=True,
             e1_rollout_sequence_divergence_min_batch_classes=3)
    torch.manual_seed(5)
    init = torch.randn(2, TOTAL_DIM)
    actions = _onehot_seq(2, 3, [0, 1])  # only 2 distinct sequences
    torch.manual_seed(7)
    targets = torch.randn(2, TOTAL_DIM)
    loss = e1.rollout_sequence_divergence_loss(init, actions, targets)
    assert loss.item() == 0.0  # 2 < config floor of 3


# --------------------------------------------------------------------------- #
# K < 2, horizon clamping, malformed shapes                                   #
# --------------------------------------------------------------------------- #

def test_single_candidate_returns_zero():
    e1 = _e1(action_conditioned_transition=True)
    init = torch.randn(1, TOTAL_DIM)
    actions = _onehot_seq(1, 3, 0)
    targets = torch.randn(1, TOTAL_DIM)
    loss = e1.rollout_sequence_divergence_loss(init, actions, targets)
    assert loss.item() == 0.0


def test_horizon_is_clamped_to_available_action_sequence():
    """A horizon longer than the supplied action sequence must not read past it."""
    e1 = _e1(action_conditioned_transition=True)
    torch.manual_seed(5)
    init = torch.randn(3, TOTAL_DIM)
    actions = _onehot_seq(3, 3, [0, 1, 2])  # only 3 steps available
    torch.manual_seed(7)
    targets = torch.randn(3, TOTAL_DIM)
    loss = e1.rollout_sequence_divergence_loss(init, actions, targets, horizon=50)
    assert torch.isfinite(loss)


def test_config_horizon_is_used_when_no_override_is_passed():
    torch.manual_seed(5)
    init = torch.randn(3, TOTAL_DIM)
    actions = _onehot_seq(3, 8, [0, 1, 2])
    e1_h2 = _e1(action_conditioned_transition=True,
                e1_rollout_sequence_divergence_horizon=2)
    e1_h6 = _e1(action_conditioned_transition=True,
                e1_rollout_sequence_divergence_horizon=6)
    torch.manual_seed(7)
    targets2 = torch.randn(3, TOTAL_DIM)
    torch.manual_seed(7)
    targets6 = torch.randn(3, TOTAL_DIM)
    a = e1_h2.rollout_sequence_divergence_loss(init, actions, targets2)
    b = e1_h6.rollout_sequence_divergence_loss(init, actions, targets6)
    # Different rollout depths over a stochastic-at-init model should not
    # coincide exactly.
    assert not torch.allclose(a, b)


@pytest.mark.parametrize("bad", ["actions_2d", "targets_batch_mismatch",
                                  "initial_state_bad_dim"])
def test_malformed_shapes_raise(bad):
    e1 = _e1(action_conditioned_transition=True)
    init = torch.randn(3, TOTAL_DIM)
    actions = _onehot_seq(3, 3, [0, 1, 2])
    targets = torch.randn(3, TOTAL_DIM)
    if bad == "actions_2d":
        with pytest.raises(ValueError):
            e1.rollout_sequence_divergence_loss(init, actions[:, 0, :], targets)
    elif bad == "targets_batch_mismatch":
        with pytest.raises(ValueError):
            e1.rollout_sequence_divergence_loss(init, actions, targets[:2])
    else:
        with pytest.raises(ValueError):
            e1.rollout_sequence_divergence_loss(
                torch.randn(3, 2, TOTAL_DIM), actions, targets
            )


# --------------------------------------------------------------------------- #
# Side-effect freedom and MECH-094                                            #
# --------------------------------------------------------------------------- #

def test_hidden_state_is_restored():
    e1 = _e1(action_conditioned_transition=True)
    torch.manual_seed(5)
    init = torch.randn(3, TOTAL_DIM)
    actions = _onehot_seq(3, 5, [0, 1, 2])
    with torch.no_grad():
        e1.predict_long_horizon(init, horizon=3, actions=actions)
    assert e1._hidden_state is not None
    before = tuple(t.clone() for t in e1._hidden_state)
    torch.manual_seed(7)
    targets = torch.randn(3, TOTAL_DIM)
    e1.rollout_sequence_divergence_loss(init, actions, targets)
    assert all(torch.equal(a, b) for a, b in zip(before, e1._hidden_state))


def test_simulation_mode_returns_zero():
    """MECH-094 gate: replay / DMN paths cannot recruit the objective."""
    e1 = _e1(action_conditioned_transition=True)
    torch.manual_seed(5)
    init = torch.randn(3, TOTAL_DIM)
    actions = _onehot_seq(3, 5, [0, 1, 2])
    torch.manual_seed(7)
    targets = torch.randn(3, TOTAL_DIM)
    loss = e1.rollout_sequence_divergence_loss(
        init, actions, targets, simulation_mode=True
    )
    assert loss.item() == 0.0


@pytest.mark.parametrize("kwargs", [
    dict(simulation_mode=True),
    dict(horizon=0),
])
def test_degenerate_returns_are_grad_connected(kwargs):
    """
    Same divergence from SD-056's plain torch.zeros(()) as ITEM 2, for the
    same reason: a caller doing (w * loss).backward() must not raise on an
    intermittent degenerate case.
    """
    e1 = _e1(action_conditioned_transition=True)
    torch.manual_seed(5)
    init = torch.randn(3, TOTAL_DIM)
    actions = _onehot_seq(3, 5, [0, 1, 2])
    torch.manual_seed(7)
    targets = torch.randn(3, TOTAL_DIM)
    loss = e1.rollout_sequence_divergence_loss(init, actions, targets, **kwargs)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()  # must not raise


# --------------------------------------------------------------------------- #
# from_dims reachability                                                      #
# --------------------------------------------------------------------------- #

def test_from_dims_reaches_e1config():
    cfg = _from_dims(
        e1_rollout_sequence_divergence_enabled=True,
        e1_rollout_sequence_divergence_weight=0.3,
        e1_rollout_sequence_divergence_horizon=7,
        e1_rollout_sequence_divergence_temperature=0.05,
        e1_rollout_sequence_divergence_min_batch_classes=3,
    )
    assert cfg.e1.e1_rollout_sequence_divergence_enabled is True
    assert cfg.e1.e1_rollout_sequence_divergence_weight == 0.3
    assert cfg.e1.e1_rollout_sequence_divergence_horizon == 7
    assert cfg.e1.e1_rollout_sequence_divergence_temperature == 0.05
    assert cfg.e1.e1_rollout_sequence_divergence_min_batch_classes == 3


def test_from_dims_defaults_are_off():
    cfg = _from_dims()
    assert cfg.e1.e1_rollout_sequence_divergence_enabled is False
    assert cfg.e1.e1_rollout_sequence_divergence_temperature == 0.1
    assert cfg.e1.e1_rollout_sequence_divergence_min_batch_classes == 2


def test_from_dims_value_reaches_the_built_predictor():
    cfg = _from_dims(
        action_conditioned_transition=True,
        e1_rollout_sequence_divergence_horizon=4,
        e1_rollout_sequence_divergence_temperature=0.05,
    )
    torch.manual_seed(0)
    e1 = E1DeepPredictor(cfg.e1)
    e1.eval()
    torch.manual_seed(5)
    init = torch.randn(3, TOTAL_DIM)
    actions = _onehot_seq(3, 8, [0, 1, 2])
    torch.manual_seed(7)
    targets = torch.randn(3, TOTAL_DIM)
    from_cfg = e1.rollout_sequence_divergence_loss(init, actions, targets)
    explicit = e1.rollout_sequence_divergence_loss(
        init, actions, targets, horizon=4, temperature=0.05
    )
    assert torch.allclose(from_cfg, explicit, atol=0.0, rtol=0.0)
