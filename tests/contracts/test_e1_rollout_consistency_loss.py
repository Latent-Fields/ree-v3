"""
Contract: SD-e1-rollout-consistency-training ITEM 2 -- E1 multi-step
rollout-consistency objective with per-step discounting.

This is synthesis candidate 1, the doc's ranked-STRONGEST ITEM 2 lever:
"Multi-step latent consistency over an action-conditioned transition
(TD-MPC-style) -- strongest template; transposes to E1's deterministic MSE
without reinterpretation."

WHY THE HELPER EXISTS AT ALL, given REEAgent.compute_prediction_loss already
rolls E1 out autoregressively and MSEs the whole trajectory. Two gaps, and this
file pins the first of them as the only behavioural axis added:

  (i)  compute_prediction_loss uses a FLAT F.mse_loss over the stacked rollout,
       weighting every horizon step equally. Under an autoregressive rollout
       deep-step error is larger by construction, so a flat mean lets the
       deepest steps dominate the gradient. decay < 1.0 is TD-MPC's actual
       form. `test_decay_one_reduces_exactly_to_the_flat_form` pins that at
       decay=1.0 this helper IS the flat form -- an exact algebraic identity,
       not a "the numbers are close" check -- so the discount is provably the
       only thing that differs.
  (ii) compute_prediction_loss is reachable only through the agent loop. Every
       driver in this SD's own lineage trains E1 directly and single-step
       teacher-forced (`F.mse_loss(e1_pred[:, 0, :], ...)` -- V3-EXQ-954:312,
       965:409, 968:431), so the multi-step objective has never once been
       exercised in the lineage that motivated this SD.

WHAT THIS FILE DELIBERATELY DOES NOT ASSERT: that a multi-step objective beats
a single-step one, or that any particular decay relieves the ~675x
LSTM+output_proj crush. That is the validation experiment's job, not the
contract's -- the same boundary the sibling residual-knob contract holds.

Every assertion is upstream of any discrete quantizer -- on tensors, losses and
gradients, never on a committed action. torch.multinomial returns different
categories on linux-x86_64 than on darwin-arm64 from a bit-identical
probability tensor at the same seed (CLAUDE.md, "Running the test suite"), so a
contract asserting an exact action would be machine-class flaky.

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

    eval() is load-bearing, not tidiness -- transition_rnn is built with
    num_layers=3 so nn.LSTM applies dropout=0.1, and in train mode two
    identical calls differ by dropout noise alone. That would break the exact
    flat-form identity below AND would let the non-vacuity tests pass on
    dropout noise while the discount term was inert.
    """
    torch.manual_seed(0)
    cfg = E1Config(self_dim=SELF_DIM, world_dim=WORLD_DIM,
                   latent_dim=TOTAL_DIM, **kw)
    e1 = E1DeepPredictor(cfg)
    e1.eval()
    return e1


def _batch(batch: int = 3, steps: int = 8, seed: int = 11):
    torch.manual_seed(seed)
    init = torch.randn(batch, TOTAL_DIM)
    targets = torch.randn(batch, steps, TOTAL_DIM)
    actions = torch.zeros(batch, steps, ACTION_DIM)
    actions[:, :, 2] = 1.0
    return init, targets, actions


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
# Defaults are no-op                                                           #
# --------------------------------------------------------------------------- #

def test_default_is_off():
    cfg = E1Config()
    assert cfg.e1_rollout_consistency_enabled is False
    assert cfg.e1_rollout_consistency_horizon_weights_decay == 1.0


def test_knob_adds_no_parameters():
    """
    No module and no parameter is constructed in either setting, so unlike
    action_encoder there is no construction-time RNG asymmetry to defend
    against. The objective is a pure function of existing weights.
    """
    off = _e1()
    on = _e1(e1_rollout_consistency_enabled=True,
             e1_rollout_consistency_horizon_weights_decay=0.5)
    p_off = list(off.parameters())
    p_on = list(on.parameters())
    assert sum(p.numel() for p in p_off) == sum(p.numel() for p in p_on)
    assert all(torch.equal(a, b) for a, b in zip(p_off, p_on))


@pytest.mark.parametrize("action_conditioned", [False, True])
def test_enabling_the_flag_does_not_change_the_rollout(action_conditioned):
    """
    The flag gates an OBJECTIVE, not the forward path. Turning it on must not
    perturb prediction -- otherwise an experiment's ON arm would differ from
    its OFF arm for a reason unrelated to the training objective under test.
    """
    kw = dict(action_conditioned_transition=action_conditioned)
    off = _e1(**kw)
    on = _e1(e1_rollout_consistency_enabled=True, **kw)
    init, _, actions = _batch()
    acts = actions if action_conditioned else None
    a = _rollout(off, init, 5, acts)
    b = _rollout(on, init, 5, acts)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# The discount is the ONLY behavioural axis added                              #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("action_conditioned", [False, True])
def test_decay_one_reduces_exactly_to_the_flat_form(action_conditioned):
    """
    THE load-bearing identity. At decay=1.0 the per-step weights are uniform,
    so the helper must equal F.mse_loss over the same rollout window.

    This is what makes the discount the only thing this helper adds over
    compute_prediction_loss's flat mean. A looser "the losses are close" check
    at a percent-scale tolerance would pass on an implementation that
    normalised by the wrong denominator, weighted from t=1 instead of t=0, or
    dropped a step -- every one of those is wrong by a FACTOR, so a float32-eps
    tolerance still catches them.

    NOT ASSERTED BIT-EXACTLY, on purpose. The helper reduces per-step
    (mean over batch and features, then a weighted mean over steps) because
    per-step weighting requires it; F.mse_loss reduces over all elements at
    once. Those are mathematically equal but have different float32 summation
    ORDERS, so they differ at the last ulp. Measured 2026-09-01 on this exact
    case: identical to the bit for the legacy branch and 6.4e-08 RELATIVE for
    the action-conditioned one -- right at float32 eps (~1.2e-07). An earlier
    revision of this test asserted atol=0.0/rtol=0.0 and passed on
    ree-worker-4 while failing on darwin-arm64, i.e. it was machine-class
    flaky in exactly the way CLAUDE.md's "Running the test suite" note warns
    about. rtol=1e-6 keeps ~15x headroom over the observed divergence while
    staying orders of magnitude tighter than any real defect.
    """
    e1 = _e1(action_conditioned_transition=action_conditioned)
    init, targets, actions = _batch()
    acts = actions if action_conditioned else None
    h = 5
    loss = e1.rollout_consistency_loss(init, targets, actions=acts, horizon=h)
    preds = _rollout(e1, init, h, acts)
    flat = F.mse_loss(preds[:, :h, :], targets[:, :h, :])
    assert torch.allclose(loss, flat, rtol=1e-6, atol=1e-12)


def test_decay_below_one_actually_changes_the_loss():
    """Non-vacuity: the discount must reach the number, not just the config."""
    e1 = _e1(action_conditioned_transition=True)
    init, targets, actions = _batch()
    flat = e1.rollout_consistency_loss(init, targets, actions=actions, horizon=5)
    disc = e1.rollout_consistency_loss(init, targets, actions=actions, horizon=5,
                                       horizon_weights_decay=0.5)
    assert not torch.allclose(flat, disc)


def test_decay_weights_earlier_steps_more():
    """
    Directional check on the discount's SIGN, against a hand-built target whose
    error is concentrated at the DEEP end of the rollout. Down-weighting deep
    steps must then lower the loss relative to uniform. A decay applied with an
    inverted exponent would pass the "changes the loss" test above and fail
    here.
    """
    e1 = _e1(action_conditioned_transition=True)
    init, _, actions = _batch()
    h = 5
    preds = _rollout(e1, init, h, actions)
    targets = preds.clone()
    targets[:, h - 1, :] += 10.0          # all error at the deepest step
    flat = e1.rollout_consistency_loss(init, targets, actions=actions, horizon=h)
    disc = e1.rollout_consistency_loss(init, targets, actions=actions, horizon=h,
                                       horizon_weights_decay=0.5)
    assert disc < flat


def test_config_decay_is_used_when_no_override_is_passed():
    """The knob must be read off the config, not only from the call argument."""
    init, targets, actions = _batch()
    e1_uniform = _e1(action_conditioned_transition=True)
    e1_disc = _e1(action_conditioned_transition=True,
                  e1_rollout_consistency_horizon_weights_decay=0.5)
    a = e1_uniform.rollout_consistency_loss(init, targets, actions=actions, horizon=5)
    b = e1_disc.rollout_consistency_loss(init, targets, actions=actions, horizon=5)
    assert not torch.allclose(a, b)


def test_config_horizon_is_used_when_no_override_is_passed():
    init, targets, actions = _batch()
    e1_h2 = _e1(action_conditioned_transition=True,
                e1_rollout_consistency_horizon=2)
    e1_h6 = _e1(action_conditioned_transition=True,
                e1_rollout_consistency_horizon=6)
    assert not torch.allclose(
        e1_h2.rollout_consistency_loss(init, targets, actions=actions),
        e1_h6.rollout_consistency_loss(init, targets, actions=actions),
    )


# --------------------------------------------------------------------------- #
# It is a real training signal                                                 #
# --------------------------------------------------------------------------- #

def test_gradient_reaches_output_proj_and_the_lstm():
    """
    The ~675x crush the design doc names as ITEM 2's real target sits at the
    LSTM + output_proj stage. An objective that cannot deliver gradient there
    cannot possibly move it, whatever else it measures.
    """
    e1 = _e1(action_conditioned_transition=True)
    init, targets, actions = _batch()
    e1.zero_grad()
    e1.rollout_consistency_loss(init, targets, actions=actions, horizon=5).backward()
    assert e1.output_proj[0].weight.grad is not None
    assert e1.output_proj[0].weight.grad.abs().sum() > 0
    assert e1.transition_rnn.weight_ih_l0.grad is not None
    assert e1.transition_rnn.weight_ih_l0.grad.abs().sum() > 0


def test_deep_steps_carry_gradient_not_only_the_first():
    """
    The whole point of a MULTI-step objective. Error placed only at a deep step
    must still produce gradient -- otherwise this is a single-step loss wearing
    a horizon argument, which is exactly the status quo it exists to change.
    """
    e1 = _e1(action_conditioned_transition=True)
    init, _, actions = _batch()
    h = 5
    preds = _rollout(e1, init, h, actions)
    targets = preds.clone()
    targets[:, h - 1, :] += 5.0           # deepest step only
    e1.zero_grad()
    e1.rollout_consistency_loss(init, targets, actions=actions, horizon=h).backward()
    assert e1.output_proj[0].weight.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# Side-effect freedom and the degenerate returns                               #
# --------------------------------------------------------------------------- #

def test_hidden_state_is_restored():
    """
    Training must not disturb inference-time recurrence -- the same save/restore
    contract REEAgent.compute_prediction_loss holds.
    """
    e1 = _e1(action_conditioned_transition=True)
    init, targets, actions = _batch()
    with torch.no_grad():                 # seed a non-None hidden state; no restore here
        e1.predict_long_horizon(init, horizon=3, actions=actions)
    assert e1._hidden_state is not None
    before = tuple(t.clone() for t in e1._hidden_state)
    e1.rollout_consistency_loss(init, targets, actions=actions, horizon=5)
    assert all(torch.equal(a, b) for a, b in zip(before, e1._hidden_state))


def test_simulation_mode_returns_zero():
    """MECH-094 gate: replay / DMN paths cannot recruit the objective."""
    e1 = _e1(action_conditioned_transition=True)
    init, targets, actions = _batch()
    loss = e1.rollout_consistency_loss(init, targets, actions=actions,
                                       horizon=5, simulation_mode=True)
    assert loss.item() == 0.0


@pytest.mark.parametrize("kwargs", [
    dict(horizon=5, simulation_mode=True),
    dict(horizon=0),
])
def test_degenerate_returns_are_grad_connected(kwargs):
    """
    DELIBERATE DIVERGENCE FROM SD-056's E2 helpers, which return a plain
    torch.zeros(()) with requires_grad=False. A caller doing
    `(w * loss).backward()` would then raise "does not require grad" -- and the
    degenerate case here is intermittent (a short target window), which is the
    worst kind of trap. This helper returns the grad-connected
    `next(self.parameters()).sum() * 0.0` idiom REEAgent.compute_prediction_loss
    already uses for the same reason.
    """
    e1 = _e1(action_conditioned_transition=True)
    init, targets, actions = _batch()
    loss = e1.rollout_consistency_loss(init, targets, actions=actions, **kwargs)
    assert loss.item() == 0.0
    assert loss.requires_grad
    loss.backward()                       # must not raise


def test_horizon_is_clamped_to_available_targets():
    """A horizon longer than the target window must not read past the end."""
    e1 = _e1(action_conditioned_transition=True)
    init, targets, actions = _batch(steps=3)
    loss = e1.rollout_consistency_loss(init, targets, actions=actions, horizon=50)
    assert torch.isfinite(loss)
    preds = _rollout(e1, init, 3, actions[:, :3, :])
    # rtol, not bit-identity -- same float32 reduction-order caveat as
    # test_decay_one_reduces_exactly_to_the_flat_form.
    assert torch.allclose(loss, F.mse_loss(preds, targets), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("decay", [0.0, -0.5])
def test_non_positive_decay_raises(decay):
    """
    Fail closed on a nonsensical discount. decay <= 0 makes the weight vector
    degenerate or sign-alternating and weights.sum() zero or negative, so the
    loss would come back NaN/inf SILENTLY rather than erroring -- the quiet
    failure class this substrate keeps being bitten by.
    """
    e1 = _e1(action_conditioned_transition=True)
    init, targets, actions = _batch()
    with pytest.raises(ValueError):
        e1.rollout_consistency_loss(init, targets, actions=actions, horizon=5,
                                    horizon_weights_decay=decay)


@pytest.mark.parametrize("bad", ["targets_2d", "batch_mismatch"])
def test_malformed_shapes_raise_rather_than_silently_broadcast(bad):
    """
    Fail closed. A [batch, total_dim] target silently broadcasting against a
    [batch, h, total_dim] rollout would train on a horizon-constant target and
    look correct at the shape level -- the same class of defect the ITEM 1
    buffer-alignment note warns about.
    """
    e1 = _e1(action_conditioned_transition=True)
    init, targets, actions = _batch()
    if bad == "targets_2d":
        arg = targets[:, 0, :]
    else:
        arg = targets[:1]
    with pytest.raises(ValueError):
        e1.rollout_consistency_loss(init, arg, actions=actions, horizon=5)


# --------------------------------------------------------------------------- #
# from_dims reachability                                                       #
# --------------------------------------------------------------------------- #

def test_from_dims_reaches_e1config():
    """
    REEConfig.from_dims swallows unknown kwargs silently, so a knob wired into
    the dataclass but not into from_dims is unreachable from every experiment
    that builds its config the normal way -- and fails as a NO-OP, not an error.
    """
    cfg = _from_dims(
        e1_rollout_consistency_enabled=True,
        e1_rollout_consistency_weight=0.25,
        e1_rollout_consistency_horizon=7,
        e1_rollout_consistency_horizon_weights_decay=0.5,
    )
    assert cfg.e1.e1_rollout_consistency_enabled is True
    assert cfg.e1.e1_rollout_consistency_weight == 0.25
    assert cfg.e1.e1_rollout_consistency_horizon == 7
    assert cfg.e1.e1_rollout_consistency_horizon_weights_decay == 0.5


def test_from_dims_defaults_are_off():
    cfg = _from_dims()
    assert cfg.e1.e1_rollout_consistency_enabled is False
    assert cfg.e1.e1_rollout_consistency_horizon_weights_decay == 1.0


def test_from_dims_value_reaches_the_built_predictor():
    cfg = _from_dims(
        action_conditioned_transition=True,
        e1_rollout_consistency_horizon=4,
        e1_rollout_consistency_horizon_weights_decay=0.5,
    )
    torch.manual_seed(0)
    e1 = E1DeepPredictor(cfg.e1)
    e1.eval()
    init, targets, actions = _batch()
    from_cfg = e1.rollout_consistency_loss(init, targets, actions=actions)
    explicit = e1.rollout_consistency_loss(init, targets, actions=actions,
                                           horizon=4, horizon_weights_decay=0.5)
    assert torch.allclose(from_cfg, explicit, atol=0.0, rtol=0.0)
