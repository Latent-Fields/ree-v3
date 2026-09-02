"""SD-018 AMEND contracts: directional resource-field regression head on z_world.

Routed by failure_autopsy_V3-EXQ-948_2026-08-25 (H-observation-interface CONFIRMED):
with SD-018's scalar proximity head already active, a PPO reader of z_world alone
forages 0.5 res/ep against the 1.0 floor while the same reader given the full 25-dim
resource_field_view (world_obs[225:250], already in z_world's OWN input) clears it
3/3. The amend adds SplitEncoder.resource_field_head (Linear(world_dim, 25) + Sigmoid)
so an auxiliary MSE against the whole field forces z_world to EXPOSE the directional
gradient it currently discards.

C1  OFF is byte-identical: default config constructs no head, state_dict keys are
    unchanged, resource_field_pred is None, the loss is a zero-with-grad, and the
    P0 trainer's field leg is skipped at its default weight 0.0.
C2  ON: head exists, prediction is [batch, 25] in [0, 1], and the loss gradient
    reaches world_encoder (the whole point -- z_world must change, not a head).
C3  from_dims plumbs all three knobs (the from_dims **kwargs swallow --
    [memory] reference-reeconfig-from-dims-silent-kwargs).
C4  The two RESOURCE_FIELD_SLICE constants (SplitEncoder, zworld_p0) agree with the
    CausalGridWorldV2 use_proxy_fields layout, and the P0 leg learns a decodable
    field from an obs that contains it (held-out r2 > 0 vs the mean predictor).
C5  Width mismatch is a loud error, not a silent broadcast.
"""
import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.latent.stack import SplitEncoder
from ree_core.latent.zworld_p0 import (
    RESOURCE_FIELD_SLICE, ZWorldP0Config, ZWorldP0Trainer,
)
from ree_core.utils.config import REEConfig

from tests.fixtures.seed_utils import set_all_seeds
from tests.fixtures.tiny_configs import make_tiny_config
from tests.fixtures.tiny_env import make_tiny_env


def _agent_and_obs(**overrides):
    set_all_seeds(0)
    env = make_tiny_env(seed=0)
    cfg = make_tiny_config(env, **overrides)
    agent = REEAgent(cfg)
    agent.reset()
    _flat, od = env.reset()
    body = od["body_state"].unsqueeze(0)
    world = od["world_state"].unsqueeze(0)
    return env, agent, od, body, world


# --- C1 -------------------------------------------------------------------------------
def test_c1_default_off_is_inert():
    env, agent, od, body, world = _agent_and_obs(use_resource_proximity_head=True)
    se = agent.latent_stack.split_encoder
    assert se.resource_field_head is None
    assert not any("resource_field_head" in k for k in agent.latent_stack.state_dict())
    with torch.no_grad():
        lat = agent.sense(obs_body=body, obs_world=world)
    assert lat.resource_field_pred is None
    assert lat.resource_prox_pred is not None  # the scalar head is untouched
    loss = agent.compute_resource_field_loss(torch.rand(25), lat)
    assert float(loss) == 0.0 and loss.requires_grad


def test_c1_off_forward_matches_a_stack_built_without_the_kwarg():
    """The new __init__ kwarg at its default must not perturb parameter creation order:
    two encoders built with and without naming it, same seed, are parameter-identical."""
    torch.manual_seed(3)
    a = SplitEncoder(body_obs_dim=8, world_obs_dim=250, self_dim=8, world_dim=16,
                     topdown_dim=0, hidden_dim=32, use_resource_proximity_head=True)
    torch.manual_seed(3)
    b = SplitEncoder(body_obs_dim=8, world_obs_dim=250, self_dim=8, world_dim=16,
                     topdown_dim=0, hidden_dim=32, use_resource_proximity_head=True,
                     use_resource_field_head=False)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb)
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    x_b, x_w = torch.rand(2, 8), torch.rand(2, 250)
    out_a, out_b = a(x_b, x_w), b(x_b, x_w)
    assert len(out_a) == 8 and out_a[7] is None and out_b[7] is None
    assert torch.equal(out_a[1], out_b[1])


def test_c1_p0_field_leg_skipped_at_default_weight():
    s = _stack_with_head()
    tr = ZWorldP0Trainer(s, ZWorldP0Config(epochs=1, batch_size=32, seed=1))
    for _ in range(64):
        tr.observe(torch.rand(250), 0.5)
    st = tr.train()
    assert st["used_resource_field_head"] is False
    assert "resource_field_holdout" not in st


# --- C2 -------------------------------------------------------------------------------
def test_c2_on_predicts_25_dim_field_and_grad_reaches_world_encoder():
    env, agent, od, body, world = _agent_and_obs(
        use_resource_proximity_head=True, use_resource_field_head=True)
    assert env.world_obs_dim >= RESOURCE_FIELD_SLICE.stop
    lat = agent.sense(obs_body=body, obs_world=world)
    p = lat.resource_field_pred
    assert p is not None and tuple(p.shape) == (1, 25)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
    tgt = od["resource_field_view"]
    assert tuple(tgt.reshape(-1).shape) == (25,)
    # the field really is the world_obs slice the encoder already sees
    assert torch.allclose(world[0, RESOURCE_FIELD_SLICE], tgt.reshape(-1).float())
    loss = agent.compute_resource_field_loss(tgt, lat)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    grads = [q.grad for q in agent.latent_stack.split_encoder.world_encoder.parameters()]
    assert all(g is not None and float(g.abs().sum()) > 0.0 for g in grads)


# --- C3 -------------------------------------------------------------------------------
def test_c3_from_dims_plumbs_all_three_knobs():
    cfg = REEConfig.from_dims(12, 250, 5, use_resource_field_head=True,
                              resource_field_weight=0.25, resource_field_dim=25)
    assert cfg.latent.use_resource_field_head is True
    assert cfg.latent.resource_field_weight == 0.25
    assert cfg.latent.resource_field_dim == 25
    d = REEConfig.from_dims(12, 250, 5)
    assert d.latent.use_resource_field_head is False
    assert d.latent.resource_field_weight == 0.5
    assert d.latent.resource_field_dim == 25


# --- C4 -------------------------------------------------------------------------------
def _stack_with_head(world_dim=16):
    class _S:
        pass
    s = _S()
    s.split_encoder = SplitEncoder(
        body_obs_dim=8, world_obs_dim=250, self_dim=8, world_dim=world_dim,
        topdown_dim=0, hidden_dim=32, use_resource_proximity_head=True,
        use_resource_field_head=True,
    )
    return s


def test_c4_slice_constants_agree_with_env_layout():
    assert SplitEncoder.RESOURCE_FIELD_SLICE == RESOURCE_FIELD_SLICE == slice(225, 250)
    env = make_tiny_env(seed=0)
    _flat, od = env.reset()
    assert torch.allclose(od["world_state"][RESOURCE_FIELD_SLICE],
                          od["resource_field_view"].reshape(-1).float())


def test_c4_p0_leg_learns_a_decodable_field():
    """A field that is a deterministic function of the obs must be recoverable through
    z_world: held-out r2 > 0 against the mean predictor after a short P0."""
    s = _stack_with_head(world_dim=32)
    tr = ZWorldP0Trainer(s, ZWorldP0Config(epochs=40, batch_size=32, seed=1,
                                           resource_field_weight=5.0))
    g = torch.Generator().manual_seed(0)
    for _ in range(400):
        obs = torch.zeros(250)
        cell = int(torch.randint(0, 25, (1,), generator=g))
        obs[cell * 7 + 2] = 1.0                     # resource entity in that local-view cell
        f = torch.zeros(25)
        f[cell] = 1.0                               # field peaks at the resource cell
        obs[RESOURCE_FIELD_SLICE] = f
        obs[175:225] = torch.rand(50, generator=g)  # contamination + hazard views: noise
        tr.observe(obs, float(f.max()))
    st = tr.train()
    assert st["used_resource_field_head"] is True
    ho = st["resource_field_holdout"]
    assert ho["mse"] < ho["mean_predictor_mse"], ho
    assert ho["r2"] is not None and ho["r2"] > 0.0, ho


# --- C5 -------------------------------------------------------------------------------
def test_c5_width_mismatch_is_loud():
    env, agent, od, body, world = _agent_and_obs(use_resource_field_head=True)
    lat = agent.sense(obs_body=body, obs_world=world)
    with pytest.raises(ValueError, match="resource_field_dim"):
        agent.compute_resource_field_loss(torch.rand(24), lat)


# --- C6: the DRIVER SEAM -------------------------------------------------------------
# C1-C5 pin the substrate. C6 pins the thing that makes the substrate REACHABLE from an
# experiment driver, which it was not between the amend landing (ree-v3 028a625) and
# 2026-09-02: ZWorldP0Config.resource_field_weight defaults to 0.0 (its scalar sibling
# proximity_weight defaults to 0.5), and _lib/zworld_p0_warmup.run_zworld_p0 -- the ONLY
# P0a path _train_all_on_agent uses -- built its config with no override. Since
# _train_all_on_agent also calls neither compute_resource_proximity_loss nor
# compute_resource_field_loss, the directional head received ZERO gradient steps from any
# driver in the x734/737/808/948 family. An "ON" arm would have differed from its OFF
# sibling only by an untrained randomly-initialised head that nothing reads: a manipulation
# that cannot reach the DV, i.e. a silently vacuous experiment rather than a failing one.
# Found while authoring the amend's own owed validation (V3-EXQ-978);
# chip-20260902-sd018-p0a-field-weight-seam.
import inspect  # noqa: E402

from experiments._lib.allon_training import _train_all_on_agent  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402


def _p0a(agent, env, **kw):
    return run_zworld_p0(agent, env, seed=0, episodes=6, steps_per_episode=40,
                         policy=RandomPolicy(0), label="c6", dry_run=True, **kw)


def test_c6_default_is_inert_field_leg_never_runs():
    """The seam's default must leave every pre-existing caller bit-identical: no weight
    passed -> the field leg does not run, even with the head present on the config."""
    env, agent, _od, _b, _w = _agent_and_obs(use_resource_proximity_head=True,
                                             use_resource_field_head=True)
    out = _p0a(agent, env)
    assert out["p0a_ran"] is True, out
    assert out["p0a_resource_field_weight"] == 0.0
    assert not out["p0a_used_resource_field_head"]
    assert out["p0a_resource_field_holdout"] is None
    # the scalar leg is untouched by the new parameter
    assert out["p0a_used_proximity_head"] is True


def test_c6_weight_plus_head_runs_the_leg_and_reports_the_holdout():
    """The seam's whole purpose: weight > 0 AND the head present -> the leg runs and the
    mechanism readout (held-out field MSE vs the constant-mean predictor) is recorded."""
    env, agent, _od, _b, _w = _agent_and_obs(use_resource_proximity_head=True,
                                             use_resource_field_head=True)
    out = _p0a(agent, env, resource_field_weight=0.5)
    assert out["p0a_ran"] is True, out
    assert out["p0a_resource_field_weight"] == 0.5
    assert out["p0a_used_resource_field_head"] is True
    ho = out["p0a_resource_field_holdout"]
    assert ho is not None and set(ho) >= {"mse", "mean_predictor_mse", "r2"}
    assert ho["mse"] == ho["mse"] and ho["mse"] >= 0.0          # finite, not NaN
    assert ho["mean_predictor_mse"] >= 0.0


def test_c6_weight_without_the_head_reports_false_not_a_silent_assumption():
    """Half-configured (weight but no head) must READ as not-run, so a driver cannot
    believe its ON arm was manipulated when it was not -- the exact failure this seam
    exists to make visible."""
    env, agent, _od, _b, _w = _agent_and_obs(use_resource_proximity_head=True)
    out = _p0a(agent, env, resource_field_weight=0.5)
    assert out["p0a_ran"] is True, out
    assert out["p0a_resource_field_weight"] == 0.5      # what was asked for
    assert not out["p0a_used_resource_field_head"]      # what actually happened
    assert out["p0a_resource_field_holdout"] is None


def test_c6_train_all_on_agent_exposes_and_defaults_the_seam():
    """The family trainer must carry the parameter through, defaulting OFF."""
    sig = inspect.signature(_train_all_on_agent)
    p = sig.parameters.get("zworld_p0_resource_field_weight")
    assert p is not None, "the family trainer must expose the P0a field-weight seam"
    assert p.default == 0.0, "default must keep every existing caller bit-identical"
    assert "resource_field_weight" in inspect.signature(run_zworld_p0).parameters
