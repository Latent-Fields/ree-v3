"""
Contract: SD-e1-rollout-consistency-training -- E1 residual output_proj knob.

The design doc records a suspect for the dominant ~675x LSTM+output_proj crush
and PRE-REGISTERS the discrimination verbatim:

    "output_proj predicts the ABSOLUTE next state, where E2 uses a residual
     z + delta(z, a) parameterisation. If the item-1 ON arm still shows crushed
     per-action divergence at the E1 output, that parameterisation is the next
     thing to test."

V3-EXQ-965 (2026-08-30, confirmed autopsy) established that the item-1 ON arm
IS still crushed relative to the evaluator bar -- cr_ratio(h=1) 2.67e-03..
3.96e-03 against a 0.1 bar, i.e. 25-37x short -- so the branch is live. This
file pins the SUBSTRATE knob that makes the A/B runnable. It does NOT assert
that the residual form is better; that is the experiment, not the contract.

Pinned in both directions, per the ITEM 1 file's convention:

  * OFF is bit-identical to the pre-change absolute form, verified against a
    hand-rolled replication of the legacy loop rather than against a frozen
    magic number, and
  * ON is NOT VACUOUS -- the residual term actually reaches the output, in
    BOTH rollout branches (the action-conditioned one and the legacy one). A
    knob wired into only one branch would leave the discrimination half-armed
    and is invisible at the shape level.

Every assertion is upstream of any discrete quantizer -- on tensors, deltas
and magnitudes, never on a committed action sequence. torch.multinomial
returns different categories on linux-x86_64 than on darwin-arm64 from a
bit-identical probability tensor at the same seed (CLAUDE.md, "Running the
test suite"), so a contract asserting an exact action would be machine-class
flaky.

See REE_assembly/docs/architecture/sd_e1_rollout_consistency_training.md.
"""

import itertools
import sys
from pathlib import Path

import pytest
import torch

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
    identical calls differ by dropout noise alone. That would break the
    bit-identity assertions AND would let the non-vacuity tests pass on
    dropout noise while the residual term was inert.
    """
    torch.manual_seed(0)
    cfg = E1Config(self_dim=SELF_DIM, world_dim=WORLD_DIM,
                   latent_dim=TOTAL_DIM, **kw)
    e1 = E1DeepPredictor(cfg)
    e1.eval()
    return e1


def _onehot(i: int, n: int = ACTION_DIM) -> torch.Tensor:
    a = torch.zeros(1, n)
    a[0, i] = 1.0
    return a


def _seed_state(e1: E1DeepPredictor, state: torch.Tensor) -> torch.Tensor:
    """
    Recompute prior_full -- the tensor predict_long_horizon feeds the LSTM at
    step 0, and therefore the residual base at h=1.

    Deliberately reconstructed from the module's OWN submodules rather than
    read off a private attribute: that is what makes the h=1 algebraic identity
    below a real check on the rollout rather than a restatement of it.
    """
    batch = state.shape[0]
    context = e1.context_memory.read(state)
    prior = e1.prior_generator(torch.cat([state, context], dim=-1))
    if e1._action_conditioned and e1._action_cond_unzero_self_slot:
        prior_self = state[:, :SELF_DIM]
    else:
        prior_self = torch.zeros(batch, SELF_DIM, device=state.device)
    return torch.cat([prior_self, prior], dim=-1)


def _legacy_rollout(e1: E1DeepPredictor, state: torch.Tensor,
                    horizon: int) -> torch.Tensor:
    """
    Hand-rolled replication of the PRE-CHANGE absolute rollout (the legacy,
    non-action-conditioned branch), driven through the module's own
    submodules. This is the reference the OFF path must be bit-identical to.
    """
    batch = state.shape[0]
    prior_full = _seed_state(e1, state)
    h0 = torch.zeros(e1.config.num_layers, batch, e1.config.hidden_dim)
    c0 = torch.zeros(e1.config.num_layers, batch, e1.config.hidden_dim)
    hidden = (h0, c0)
    input_state = prior_full.unsqueeze(1)
    preds = []
    for _ in range(horizon):
        output, hidden = e1.transition_rnn(input_state, hidden)
        predicted = e1.output_proj(output.squeeze(1))
        preds.append(predicted)
        input_state = predicted.unsqueeze(1)
    return torch.stack(preds, dim=1)


# --------------------------------------------------------------------------- #
# Default is off, and off is a no-op                                           #
# --------------------------------------------------------------------------- #

def test_default_is_off():
    assert E1Config().output_proj_residual is False


def test_off_matches_hand_rolled_legacy_absolute_rollout():
    """
    Bit-identity against a replication of the pre-change loop, not against a
    frozen constant. The refactor that introduced the knob also replaced
    `input_state = predicted.unsqueeze(1)` with `state_i` + an unsqueeze at the
    call site; unsqueeze is a view, so this must hold exactly.
    """
    e1 = _e1()
    st = torch.randn(1, TOTAL_DIM)
    with torch.no_grad():
        e1.reset_hidden_state()
        got = e1.predict_long_horizon(st, horizon=4)
        e1.reset_hidden_state()
        want = _legacy_rollout(e1, st, horizon=4)
    assert torch.equal(got, want)


def test_knob_adds_no_parameters_and_consumes_no_construction_rng():
    """
    Unlike action_encoder, the residual form adds no module -- so parameter
    IDENTITY (not merely count) must be preserved across the flag from the same
    seed. A knob that perturbed the construction-time RNG stream would silently
    change every downstream initialisation and make an A/B uninterpretable.
    """
    off = _e1()
    on = _e1(output_proj_residual=True)
    off_p = dict(off.named_parameters())
    on_p = dict(on.named_parameters())
    assert set(off_p) == set(on_p)
    for name in off_p:
        assert torch.equal(off_p[name], on_p[name]), f"parameter {name} differs"


def test_off_is_unchanged_under_action_conditioning():
    """The OFF path must be bit-identical on the ITEM 1 branch too."""
    st = torch.randn(1, TOTAL_DIM)
    outs = []
    for _ in range(2):
        e1 = _e1(action_conditioned_transition=True)
        with torch.no_grad():
            e1.reset_hidden_state()
            outs.append(e1.predict_long_horizon(st, horizon=3,
                                                actions=_onehot(1)))
    assert torch.equal(outs[0], outs[1])


# --------------------------------------------------------------------------- #
# ON is not vacuous, in BOTH rollout branches                                  #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("action_conditioned", [False, True])
def test_on_changes_the_prediction_in_both_branches(action_conditioned):
    """
    Covering only one branch would leave the discrimination half-armed. The
    legacy branch is not dead code -- every non-ITEM-1 arm runs it.
    """
    st = torch.randn(1, TOTAL_DIM)
    kw = {"action_conditioned_transition": action_conditioned}
    acts = _onehot(0) if action_conditioned else None
    with torch.no_grad():
        off = _e1(**kw)
        off.reset_hidden_state()
        p_off = off.predict_long_horizon(st, horizon=3, actions=acts)
        on = _e1(output_proj_residual=True, **kw)
        on.reset_hidden_state()
        p_on = on.predict_long_horizon(st, horizon=3, actions=acts)
    assert not torch.equal(p_off, p_on)
    assert float((p_on - p_off).abs().max()) > 0.0


@pytest.mark.parametrize("action_conditioned", [False, True])
def test_on_h1_is_exactly_seed_plus_absolute_readout(action_conditioned):
    """
    The algebraic identity that says it is a RESIDUAL and not merely 'different'
    -- at h=1, ON minus OFF is exactly the LSTM seed state.

    This is the assertion that would catch a knob wired to the wrong base
    (e.g. current_state instead of state_i, or the post-goal-projection state),
    which every looser 'the numbers moved' check passes.
    """
    st = torch.randn(1, TOTAL_DIM)
    kw = {"action_conditioned_transition": action_conditioned}
    acts = _onehot(0) if action_conditioned else None
    with torch.no_grad():
        off = _e1(**kw)
        off.reset_hidden_state()
        p_off = off.predict_long_horizon(st, horizon=1, actions=acts)
        on = _e1(output_proj_residual=True, **kw)
        on.reset_hidden_state()
        p_on = on.predict_long_horizon(st, horizon=1, actions=acts)
        seed = _seed_state(off, st)
    assert torch.equal(p_on[:, 0, :], seed + p_off[:, 0, :])


def test_on_preserves_output_shape():
    e1 = _e1(output_proj_residual=True)
    with torch.no_grad():
        e1.reset_hidden_state()
        out = e1.predict_long_horizon(torch.randn(2, TOTAL_DIM), horizon=5)
    assert out.shape == (2, 5, TOTAL_DIM)


def test_on_reaches_forward_not_only_predict_long_horizon():
    """
    substrate_paths names BOTH e1_deep.py::forward and ::predict_long_horizon.
    forward() delegates, so one change covers both -- pin that it really does,
    rather than assuming the delegation.
    """
    st = torch.randn(1, TOTAL_DIM)
    with torch.no_grad():
        off = _e1()
        off.reset_hidden_state()
        p_off, _ = off(st, 3)
        on = _e1(output_proj_residual=True)
        on.reset_hidden_state()
        p_on, _ = on(st, 3)
    assert not torch.equal(p_off, p_on)


def test_on_keeps_per_action_divergence_alive():
    """
    Non-vacuity in the direction the experiment cares about: with ITEM 1 ON,
    distinct actions must still give distinct h=1 predictions under the
    residual form. Asserted as a pairwise-L2 FLOOR, not as an improvement --
    whether residual beats absolute is what the A/B measures, and a contract
    asserting the answer would pre-empt the experiment.
    """
    e1 = _e1(action_conditioned_transition=True, output_proj_residual=True)
    st = torch.randn(1, TOTAL_DIM)
    outs = []
    with torch.no_grad():
        for k in range(ACTION_DIM):
            e1.reset_hidden_state()
            outs.append(e1.predict_long_horizon(st, horizon=1, actions=_onehot(k)))
    dists = [float((x - y).norm()) for x, y in itertools.combinations(outs, 2)]
    assert min(dists) > 1e-5, f"action channel inert under residual: {dists}"


def test_on_gradient_reaches_output_proj():
    """
    A residual path can hide a dead branch: if the delta head stopped receiving
    gradient the rollout would still produce plausible numbers (it would just
    copy the seed forward). Pin that output_proj is still trained.
    """
    e1 = _e1(output_proj_residual=True)
    e1.train()
    preds = e1.predict_long_horizon(torch.randn(1, TOTAL_DIM), horizon=3)
    preds.pow(2).mean().backward()
    grad = e1.output_proj[-1].weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_on_is_independent_of_action_conditioning():
    """
    Deliberately NOT gated on the ITEM 1 master switch: it parameterises the
    state recurrence, not the action channel, and the A/B needs both knobs
    separately settable. This pins that independence, which is the opposite of
    action_cond_unzero_self_slot's inert-unless-master-on contract.
    """
    st = torch.randn(1, TOTAL_DIM)
    with torch.no_grad():
        off = _e1()
        off.reset_hidden_state()
        p_off = off.predict_long_horizon(st, horizon=2)
        on = _e1(output_proj_residual=True)
        on.reset_hidden_state()
        p_on = on.predict_long_horizon(st, horizon=2)
    assert not torch.equal(p_off, p_on), (
        "residual knob is inert while action_conditioned_transition is False"
    )


# --------------------------------------------------------------------------- #
# from_dims reachability (the three-site bug)                                  #
# --------------------------------------------------------------------------- #

def _from_dims(**kw) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=16, action_dim=ACTION_DIM, **kw
    )


def test_from_dims_reaches_e1config():
    """
    REEConfig.from_dims silently swallows unknown kwargs, so a knob wired into
    only the dataclass and the consumer reads as implemented and is
    unreachable. Verify the landed RUNTIME value.
    """
    assert _from_dims().e1.output_proj_residual is False
    assert _from_dims(output_proj_residual=True).e1.output_proj_residual is True


def test_from_dims_value_reaches_the_built_predictor():
    """The third site: config field -> from_dims -> the module that reads it."""
    cfg = _from_dims(output_proj_residual=True)
    torch.manual_seed(0)
    e1 = E1DeepPredictor(cfg.e1)
    assert e1._output_proj_residual is True

    cfg_off = _from_dims()
    torch.manual_seed(0)
    e1_off = E1DeepPredictor(cfg_off.e1)
    assert e1_off._output_proj_residual is False


def test_from_dims_residual_is_orthogonal_to_action_conditioning():
    cfg = _from_dims(output_proj_residual=True)
    assert cfg.e1.action_conditioned_transition is False
    cfg2 = _from_dims(action_conditioned_transition=True)
    assert cfg2.e1.output_proj_residual is False
