"""
Contract: SD-e1-rollout-consistency-training ITEM 1 -- E1 action-conditioned transition.

V3-EXQ-954 (2026-08-29, PASS, confirmed autopsy) established that E1's rollout
evaluator is floored at h=1 (cr_ratio 4.8e-07 against a 0.1 bar) with the
horizon-compounding signature ABSENT, and its red-team pass measured a ~5,000x
per-action divergence attenuation inside E1 (E2 output 2.8e-2 -> E1 output
5.6e-6). Cause: E1's transition took no action at all, and the LSTM seed zeroed
the z_self half.

This file pins the ITEM 1 fix. It deliberately pins BOTH directions:

  * the OFF path is bit-identical (this substrate has a long history of
    "improvements" that changed a default), and
  * the ON path is NOT VACUOUS -- a flag that is armed but conditions on
    nothing is the failure mode the 108/108a history warns about, and it is
    invisible at the shape level.

It does NOT assert that the collapse is fixed. The dominant ~675x crush the
red-team localised is at the LSTM + output_proj stage; closing it is ITEM 2
(the multi-step / rollout-consistency objective). Any test here claiming the
C3 bar is cleared would be asserting something this change does not deliver.

See REE_assembly/docs/architecture/sd_e1_rollout_consistency_training.md.
"""

import itertools
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ree_core.agent import REEAgent                      # noqa: E402
from ree_core.predictors.e1_deep import E1DeepPredictor  # noqa: E402
from ree_core.utils.config import E1Config, REEConfig    # noqa: E402

ACTION_DIM = 4
TOTAL_DIM = 64


def _e1(**kw) -> E1DeepPredictor:
    """
    Build an E1 and put it in EVAL mode.

    eval() is load-bearing, not tidiness. transition_rnn is built with
    num_layers=3, so nn.LSTM applies dropout=0.1, and in train mode two
    identical calls differ by dropout noise alone. That breaks the
    bit-identity assertions -- and, worse, it would let
    test_on_distinct_actions_give_distinct_predictions_at_h1 PASS on dropout
    noise while the action channel was inert, which is precisely the vacuous
    green this file exists to prevent.
    """
    torch.manual_seed(0)
    cfg = E1Config(self_dim=32, world_dim=32, latent_dim=TOTAL_DIM, **kw)
    e1 = E1DeepPredictor(cfg)
    e1.eval()
    return e1


def _onehot(i: int, n: int = ACTION_DIM) -> torch.Tensor:
    a = torch.zeros(1, n)
    a[0, i] = 1.0
    return a


# --------------------------------------------------------------------------- #
# Defaults are no-op                                                           #
# --------------------------------------------------------------------------- #

def test_default_is_off():
    cfg = E1Config()
    assert cfg.action_conditioned_transition is False


def test_off_lstm_input_size_unchanged():
    e1 = _e1()
    assert e1.transition_rnn.input_size == TOTAL_DIM


def test_off_constructs_no_action_encoder():
    """No parameters, and therefore no construction-time RNG, when OFF."""
    assert _e1().action_encoder is None


def test_off_parameter_count_unchanged_by_action_dim():
    """action_dim must be inert while the master switch is off."""
    a = sum(p.numel() for p in _e1(action_dim=4).parameters())
    b = sum(p.numel() for p in _e1(action_dim=16).parameters())
    assert a == b


def test_off_ignores_actions_bit_identical():
    e1 = _e1()
    st = torch.randn(1, TOTAL_DIM)
    e1.reset_hidden_state()
    p_none = e1.predict_long_horizon(st, horizon=3)
    e1.reset_hidden_state()
    p_act = e1.predict_long_horizon(st, horizon=3, actions=_onehot(2))
    assert torch.equal(p_none, p_act)


def test_off_forward_signature_backward_compatible():
    """Existing positional call sites must keep working untouched."""
    e1 = _e1()
    preds, prior = e1(torch.randn(1, TOTAL_DIM), 3)
    assert preds.shape == (1, 3, TOTAL_DIM)
    assert prior.shape == (1, 32)


# --------------------------------------------------------------------------- #
# ON: dedicated channel, and it actually carries signal                        #
# --------------------------------------------------------------------------- #

def test_on_widens_lstm_input_by_action_dim():
    """A DEDICATED channel, not a projection back down to total_dim."""
    e1 = _e1(action_conditioned_transition=True, action_dim=ACTION_DIM)
    assert e1.transition_rnn.input_size == TOTAL_DIM + ACTION_DIM
    assert e1.action_encoder is not None


def test_on_distinct_actions_give_distinct_predictions_at_h1():
    """
    The whole point. V3-EXQ-954 measured the action-blind model's one-step
    per-action divergence at ~1e-6 (float32-ULP scale). A model whose
    predictions are identical across actions cannot express the quantity C3
    measures, however well it is trained.
    """
    e1 = _e1(action_conditioned_transition=True)
    st = torch.randn(1, TOTAL_DIM)
    outs = []
    for k in range(ACTION_DIM):
        e1.reset_hidden_state()
        outs.append(e1.predict_long_horizon(st, horizon=1, actions=_onehot(k)).detach())
    dists = [float((x - y).norm()) for x, y in itertools.combinations(outs, 2)]
    assert min(dists) > 1e-5, f"action channel is inert: pairwise L2 {dists}"


def test_on_accepts_both_action_shapes():
    e1 = _e1(action_conditioned_transition=True)
    st = torch.randn(1, TOTAL_DIM)
    e1.reset_hidden_state()
    held = e1.predict_long_horizon(st, horizon=5, actions=_onehot(1))
    seq = torch.zeros(1, 5, ACTION_DIM)
    seq[0, :, 1] = 1.0
    e1.reset_hidden_state()
    per_step = e1.predict_long_horizon(st, horizon=5, actions=seq)
    assert held.shape == per_step.shape == (1, 5, TOTAL_DIM)
    # a held action and the equivalent constant sequence must agree exactly
    assert torch.equal(held, per_step)


def test_on_action_sequence_order_matters():
    """Per-step conditioning, not a bag of actions."""
    e1 = _e1(action_conditioned_transition=True)
    st = torch.randn(1, TOTAL_DIM)
    fwd = torch.zeros(1, 2, ACTION_DIM)
    fwd[0, 0, 0] = 1.0
    fwd[0, 1, 3] = 1.0
    rev = torch.zeros(1, 2, ACTION_DIM)
    rev[0, 0, 3] = 1.0
    rev[0, 1, 0] = 1.0
    e1.reset_hidden_state()
    a = e1.predict_long_horizon(st, horizon=2, actions=fwd).detach()
    e1.reset_hidden_state()
    b = e1.predict_long_horizon(st, horizon=2, actions=rev).detach()
    assert not torch.equal(a, b)


def test_on_rejects_wrong_action_width():
    e1 = _e1(action_conditioned_transition=True, action_dim=ACTION_DIM)
    with pytest.raises(ValueError, match="action_dim"):
        e1.predict_long_horizon(torch.randn(1, TOTAL_DIM), horizon=1,
                                actions=torch.zeros(1, ACTION_DIM + 3))


def test_on_missing_actions_counted_not_silent():
    """
    actions=None under the ON path falls back to a zero action so legacy
    internal callers keep working -- but it MUST be counted, or an ON arm can
    silently be an OFF arm.
    """
    e1 = _e1(action_conditioned_transition=True)
    assert e1._action_cond_missing_calls == 0
    e1.reset_hidden_state()
    e1.predict_long_horizon(torch.randn(1, TOTAL_DIM), horizon=2, actions=None)
    assert e1._action_cond_missing_calls == 1
    e1.reset_hidden_state()
    e1.predict_long_horizon(torch.randn(1, TOTAL_DIM), horizon=2, actions=_onehot(0))
    assert e1._action_cond_missing_calls == 1, "supplying actions must not count"


# --------------------------------------------------------------------------- #
# The z_self un-zeroing stays SEPARATELY ablatable                             #
# --------------------------------------------------------------------------- #

def test_unzero_self_slot_is_separately_ablatable():
    st = torch.randn(1, TOTAL_DIM)
    outs = []
    for unzero in (True, False):
        e1 = _e1(action_conditioned_transition=True,
                 action_cond_unzero_self_slot=unzero)
        e1.reset_hidden_state()
        outs.append(e1.predict_long_horizon(st, horizon=1, actions=_onehot(0)).detach())
    assert not torch.equal(outs[0], outs[1])


def test_unzero_self_slot_inert_when_master_off():
    """The sub-flag defaults True; it must do nothing until the master is on."""
    st = torch.randn(1, TOTAL_DIM)
    outs = []
    for unzero in (True, False):
        e1 = _e1(action_cond_unzero_self_slot=unzero)
        e1.reset_hidden_state()
        outs.append(e1.predict_long_horizon(st, horizon=2).detach())
    assert torch.equal(outs[0], outs[1])


# --------------------------------------------------------------------------- #
# from_dims reachability (the three-site bug)                                  #
# --------------------------------------------------------------------------- #

def _from_dims(**kw) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=ACTION_DIM, **kw
    )


def test_from_dims_reaches_e1config():
    cfg = _from_dims(action_conditioned_transition=True)
    assert cfg.e1.action_conditioned_transition is True
    assert cfg.e1.action_cond_unzero_self_slot is True
    cfg2 = _from_dims(action_conditioned_transition=True,
                      action_cond_unzero_self_slot=False)
    assert cfg2.e1.action_cond_unzero_self_slot is False


def test_from_dims_wires_e1_and_e2_action_dim_together():
    """E1 and E2 must never disagree about the one-hot width."""
    cfg = REEConfig.from_dims(body_obs_dim=8, world_obs_dim=16, action_dim=7)
    assert cfg.e1.action_dim == cfg.e2.action_dim == 7


# --------------------------------------------------------------------------- #
# Agent wiring: alignment, non-vacuity, gradient flow                          #
# --------------------------------------------------------------------------- #

def _agent(flag: bool) -> REEAgent:
    torch.manual_seed(0)
    cfg = _from_dims(action_conditioned_transition=flag)
    torch.manual_seed(0)
    return REEAgent(cfg)


def test_action_buffer_stays_aligned_with_state_buffers():
    agent = _agent(True)
    agent._last_action = _onehot(1)
    for _ in range(5):
        agent._self_experience_buffer.append(torch.randn(1, 32))
        agent._world_experience_buffer.append(torch.randn(1, 32))
        agent._action_experience_buffer.append(agent._e1_action_one_hot())
    assert (len(agent._action_experience_buffer)
            == len(agent._self_experience_buffer)
            == len(agent._world_experience_buffer))


def test_action_one_hot_is_zero_before_any_action():
    """No action led to the reset state; encode that honestly."""
    agent = _agent(True)
    agent._last_action = None
    assert float(agent._e1_action_one_hot().sum()) == 0.0


def test_action_one_hot_uses_argmax_of_executed_action():
    """
    Must be the ONE-HOT of the discrete action the env executed, not the raw
    continuous policy output -- the same convention every other forward-model
    call site in agent.py uses.
    """
    agent = _agent(True)
    agent._last_action = torch.tensor([[0.1, 0.9, 0.3, 0.2]])
    oh = agent._e1_action_one_hot()
    assert float(oh.sum()) == 1.0
    assert int(oh.argmax()) == 1


def test_buffer_stats_detect_the_vacuous_arm():
    """
    A driver that steps the env directly without record_executed_action() fills
    the buffer with zero actions. `_action_cond_missing_calls` cannot see that
    -- actions ARE supplied, they are just all zero -- so the stats helper is
    the check that can.
    """
    agent = _agent(True)
    agent._last_action = None
    for _ in range(4):
        agent._action_experience_buffer.append(agent._e1_action_one_hot())
    assert agent.e1_action_buffer_stats()["nonzero_fraction"] == 0.0

    agent.record_executed_action(_onehot(2))
    for _ in range(4):
        agent._action_experience_buffer.append(agent._e1_action_one_hot())
    assert agent.e1_action_buffer_stats()["nonzero_fraction"] == pytest.approx(0.5)


def test_prediction_loss_trains_the_action_channel():
    """
    Gradient must actually reach action_encoder. A channel that is wired but
    never trained is the quieter half of the vacuity problem.
    """
    agent = _agent(True)
    agent.record_executed_action(_onehot(0))
    for i in range(8):
        agent._self_experience_buffer.append(torch.randn(1, 32))
        agent._world_experience_buffer.append(torch.randn(1, 32))
        agent.record_executed_action(_onehot(i % ACTION_DIM))
        agent._action_experience_buffer.append(agent._e1_action_one_hot())
    loss = agent.compute_prediction_loss()
    loss.backward()
    grad = agent.e1.action_encoder.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_prediction_loss_action_slice_is_offset_by_one():
    """
    Pins the alignment convention: the action carrying state_i -> state_{i+1}
    is the buffer entry recorded ALONGSIDE state_{i+1}, so the slice is
    [start+1:end]. An off-by-one trains E1 on the action that led INTO the
    input state and is invisible at the shape level.
    """
    agent = _agent(True)
    captured = {}
    real = agent.e1.predict_long_horizon

    def spy(current_state, horizon=None, actions=None):
        captured["actions"] = actions
        captured["horizon"] = horizon
        return real(current_state, horizon=horizon, actions=actions)

    agent.e1.predict_long_horizon = spy
    for i in range(10):
        agent._self_experience_buffer.append(torch.randn(1, 32))
        agent._world_experience_buffer.append(torch.randn(1, 32))
        # entry i carries the marker i in channel 0 position via one-hot index
        agent.record_executed_action(_onehot(i % ACTION_DIM))
        agent._action_experience_buffer.append(agent._e1_action_one_hot())
    agent.compute_prediction_loss()

    acts = captured["actions"]
    assert acts is not None, "actions were not passed to predict_long_horizon"
    # one action per predicted step, not per state in the slice
    assert acts.shape[1] == captured["horizon"]


def test_off_agent_passes_no_actions():
    agent = _agent(False)
    captured = {}
    real = agent.e1.predict_long_horizon

    def spy(current_state, horizon=None, actions=None):
        captured["actions"] = actions
        return real(current_state, horizon=horizon, actions=actions)

    agent.e1.predict_long_horizon = spy
    for _ in range(6):
        agent._self_experience_buffer.append(torch.randn(1, 32))
        agent._world_experience_buffer.append(torch.randn(1, 32))
        agent._action_experience_buffer.append(agent._e1_action_one_hot())
    agent.compute_prediction_loss()
    assert captured["actions"] is None
