"""Contracts for the SD-063 ONLINE head-training loop (ARC-065 GAP-A keystone).

WHAT THIS PINS. `mech314bc_percandidate_extension_staged_2026-08-08.md` landed
the MECH-314b per-candidate SLOT: agent._curiosity_per_candidate_uncertainty
builds `head.predictive_variance(z0_K, actions_K) -> [K]` and passes it as
`per_candidate_uncertainty`. That made 314b per-candidate-CAPABLE. It did NOT
make it LIVE: nothing ever TRAINED the head, so its predictive_variance was a
random-init near-uniform vector -- the 604a / 624a / 614d / 640a vacuous-channel
class. This module pins the training loop that closes that gap (section 6
follow-on 1) and, critically, pins the MEASUREMENT that tells trained from
untrained.

THE LOAD-BEARING FINDING (measured 2026-08-22, real CausalGridWorldV2 rollouts,
seeds 71/101/202, 4 episodes x 80 steps):

  The ARC-065 section-5 readiness gate `last_uncertainty_dev_range > 0` is
  NECESSARY BUT NOT SUFFICIENT -- an UNTRAINED head passes it, and passes it
  with a LARGER absolute range than a trained one. Training lowers the overall
  predicted spread (the world is more predictable than a random init assumes)
  while raising the RELATIVE differentiation. So:

      untrained : max/min across action classes ~1.2x   (near-uniform)
      trained   : max/min across action classes ~29x    (differentiated)

  yet the untrained ABSOLUTE range is the bigger of the two. Gating on the
  absolute range alone therefore admits exactly the vacuous channel the gate
  exists to refuse. `_last_pvar_relative_spread` is the discriminator.

  test_untrained_head_passes_the_absolute_range_gate is the NEGATIVE CONTROL
  that pins this: if a later change makes the absolute-range gate discriminate
  after all, that test fails and this docstring is what needs revisiting -- it
  must not be "fixed" by deleting the control.

Coverage:
  - config defaults inert: train_online False -> no optimizer, no replay, and
    observe_transition is a no-op (bit-identical to the pre-2026-08-22 head);
  - P0 warmup: no head update until warmup_steps transitions are observed;
  - P1: pinball loss falls and predictive_variance becomes CONDITIONAL,
    monotone in the true per-action noise scale;
  - the relative-spread discriminator separates trained from untrained where
    the absolute range does not (plus the negative control above);
  - SD-031 agency-residual guard survives train_step: no gradient reaches an
    encoder-side leaf, and the head shares no params with E2WorldForward;
  - simulation_mode never trains (an imagined tick has no observed next state);
  - agent wiring: OFF is bit-identical; ON trains across a real rollout and
    resets the prev-z_world cache per episode (no cross-episode transition).
"""

from __future__ import annotations

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e2_world import E2WorldConfig, E2WorldForward
from ree_core.predictors.e2_world_uncertainty import (
    E2WorldUncertaintyConfig,
    E2WorldUncertaintyHead,
)
from ree_core.utils.config import LatentStackConfig, REEConfig

WORLD_DIM = 16
ACTION_DIM = 4
# True per-action next-state noise scales -- deliberately spread over ~60x so a
# head that has learned the conditional structure is unambiguously separable
# from one that has not.
NOISE_BY_ACTION = [0.01, 0.05, 0.20, 0.60]


def _head(**kw) -> E2WorldUncertaintyHead:
    cfg = E2WorldUncertaintyConfig(
        use_e2_world_uncertainty=True,
        z_world_dim=WORLD_DIM,
        action_dim=ACTION_DIM,
        **kw,
    )
    return E2WorldUncertaintyHead(cfg)


def _onehot(idx: int) -> torch.Tensor:
    a = torch.zeros(1, ACTION_DIM)
    a[0, idx] = 1.0
    return a


def _heteroscedastic_step(z: torch.Tensor, a_idx: int) -> torch.Tensor:
    """Action-conditional next state: same drift, per-action noise scale."""
    return z * 0.9 + torch.randn_like(z) * NOISE_BY_ACTION[a_idx]


def _drive_head(head: E2WorldUncertaintyHead, steps: int) -> list:
    z = torch.randn(1, WORLD_DIM)
    losses = []
    for i in range(steps):
        a_idx = i % ACTION_DIM
        nxt = _heteroscedastic_step(z, a_idx)
        loss = head.observe_transition(z, _onehot(a_idx), nxt)
        if loss is not None:
            losses.append(loss)
        z = nxt
    return losses


def _pvar_over_action_classes(head: E2WorldUncertaintyHead) -> torch.Tensor:
    z = torch.randn(1, WORLD_DIM).expand(ACTION_DIM, -1)
    return head.predictive_variance(z, torch.eye(ACTION_DIM))


# ------------------------------------------------------------------ #
# Inert OFF surface                                                   #
# ------------------------------------------------------------------ #

def test_config_defaults_are_inert():
    cfg = E2WorldUncertaintyConfig(use_e2_world_uncertainty=True, z_world_dim=WORLD_DIM)
    assert cfg.train_online is False
    ls = LatentStackConfig()
    assert ls.use_e2_world_uncertainty_online_training is False
    assert ls.e2_world_uncertainty_warmup_steps == 200
    assert ls.e2_world_uncertainty_replay_capacity == 2048
    assert ls.e2_world_uncertainty_batch_size == 32
    assert ls.e2_world_uncertainty_ready_min_train_steps == 50


def test_train_online_off_allocates_nothing_and_never_updates():
    """OFF must not merely skip the update -- it must build no optimizer and no
    replay buffer, so the OFF path is bit-identical to the pre-landing head."""
    torch.manual_seed(0)
    head = _head()  # train_online defaults False
    _drive_head(head, 300)
    assert head.n_train_steps == 0
    assert head.n_observed_transitions == 0
    assert head._optimizer is None
    assert head._replay is None
    assert head.training_ready is False


def test_from_dims_surfaces_the_online_flags_off():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=54, action_dim=ACTION_DIM,
        self_dim=32, world_dim=32,
    )
    assert cfg.latent.use_e2_world_uncertainty_online_training is False
    assert cfg.latent.e2_world_uncertainty_warmup_steps == 200


# ------------------------------------------------------------------ #
# P0 warmup -> P1 update schedule                                     #
# ------------------------------------------------------------------ #

def test_p0_warmup_buffers_but_does_not_update():
    torch.manual_seed(0)
    head = _head(train_online=True, warmup_steps=100, batch_size=16)
    _drive_head(head, 100)
    assert head.n_observed_transitions == 100
    assert head.n_train_steps == 0, "no head update may occur during P0 warmup"
    assert head._replay is not None and len(head._replay) == 100
    # One more transition crosses the warmup boundary.
    _drive_head(head, 1)
    assert head.n_train_steps == 1


def test_replay_is_bounded_so_stale_encoder_transitions_age_out():
    torch.manual_seed(0)
    head = _head(train_online=True, warmup_steps=10, batch_size=8, replay_capacity=64)
    _drive_head(head, 500)
    assert len(head._replay) == 64
    assert head.n_observed_transitions == 500


def test_training_ready_tracks_the_configured_floor():
    torch.manual_seed(0)
    head = _head(train_online=True, warmup_steps=10, batch_size=8,
                 ready_min_train_steps=25)
    _drive_head(head, 30)          # ~20 updates
    assert head.training_ready is False
    _drive_head(head, 40)
    assert head.n_train_steps >= 25
    assert head.training_ready is True


# ------------------------------------------------------------------ #
# P1 actually learns the conditional structure                        #
# ------------------------------------------------------------------ #

def test_online_training_reduces_pinball_loss():
    torch.manual_seed(0)
    head = _head(train_online=True, warmup_steps=50, batch_size=32)
    losses = _drive_head(head, 3000)
    assert len(losses) > 500
    first = sum(losses[:50]) / 50.0
    last = sum(losses[-50:]) / 50.0
    assert last < first * 0.7, f"pinball loss did not fall: {first} -> {last}"


def test_online_training_makes_predictive_variance_conditional():
    """The property the state-blind EMA structurally cannot carry: predicted
    spread ORDERED by the true per-action noise scale."""
    torch.manual_seed(0)
    head = _head(train_online=True, warmup_steps=50, batch_size=32)
    _drive_head(head, 3000)
    pv = _pvar_over_action_classes(head)
    assert pv.shape == (ACTION_DIM,)
    ranks = torch.argsort(pv)
    assert list(ranks) == list(range(ACTION_DIM)), (
        f"predictive_variance not monotone in true noise scale: {pv.tolist()}"
    )
    assert float(pv.max() / pv.min()) > 5.0


# ------------------------------------------------------------------ #
# The readiness discriminator (the load-bearing finding)              #
# ------------------------------------------------------------------ #

def test_relative_spread_discriminates_trained_from_untrained():
    torch.manual_seed(0)
    trained = _head(train_online=True, warmup_steps=50, batch_size=32)
    _drive_head(trained, 3000)
    _pvar_over_action_classes(trained)
    trained_rel = trained._last_pvar_relative_spread

    torch.manual_seed(0)
    untrained = _head()
    _pvar_over_action_classes(untrained)
    untrained_rel = untrained._last_pvar_relative_spread

    assert untrained_rel < 0.5, (
        f"a random-init head should be near-uniform; got rel_spread={untrained_rel}"
    )
    assert trained_rel > untrained_rel * 3.0, (
        "relative spread must separate trained from untrained: "
        f"trained={trained_rel} untrained={untrained_rel}"
    )


def test_untrained_head_passes_the_absolute_range_gate():
    """NEGATIVE CONTROL -- do not delete when it starts to look redundant.

    This pins the reason the ARC-065 section-5 gate had to be corrected: an
    untrained head produces a strictly-positive cross-candidate range, so
    `last_uncertainty_dev_range > 0` alone admits a vacuous channel. If this
    ever fails, the absolute-range gate has become discriminating and the
    finding recorded in this module's docstring needs revisiting -- it must not
    be silenced by removing the control.
    """
    torch.manual_seed(0)
    untrained = _head()
    pv = _pvar_over_action_classes(untrained)
    assert float(pv.max() - pv.min()) > 0.0, (
        "untrained head produced an exactly-flat vector -- the finding this "
        "control pins no longer reproduces; re-read the module docstring"
    )
    # ...and yet it is near-uniform, which is the whole point.
    assert float(pv.max() / pv.min()) < 2.0


def test_get_state_reports_the_readiness_fields():
    torch.manual_seed(0)
    head = _head(train_online=True, warmup_steps=20, batch_size=8)
    _drive_head(head, 200)
    _pvar_over_action_classes(head)
    st = head.get_state()
    for key in (
        "e2_world_uncertainty_n_train_steps",
        "e2_world_uncertainty_n_observed",
        "e2_world_uncertainty_last_train_loss",
        "e2_world_uncertainty_replay_size",
        "e2_world_uncertainty_training_ready",
        "e2_world_uncertainty_last_pvar_mean",
        "e2_world_uncertainty_last_pvar_range",
        "e2_world_uncertainty_last_pvar_relative_spread",
    ):
        assert key in st, key
    assert st["e2_world_uncertainty_n_train_steps"] > 0


# ------------------------------------------------------------------ #
# SD-031 agency-residual guard + MECH-094-adjacent simulation gate     #
# ------------------------------------------------------------------ #

def test_train_step_leaves_no_gradient_on_an_encoder_side_leaf():
    """The SD-031 guard: train_step detaches inputs AND target internally, so a
    caller that forgets to detach still cannot leak gradient into the encoder."""
    torch.manual_seed(0)
    head = _head(train_online=True)
    encoder_leaf = torch.randn(1, WORLD_DIM, requires_grad=True)
    z_prev = encoder_leaf * 2.0            # attached to the encoder-side graph
    z_next = encoder_leaf * 3.0            # target also attached, deliberately
    loss = head.train_step(z_prev, _onehot(0), z_next)
    assert loss is not None
    assert encoder_leaf.grad is None, "gradient reached an encoder-side leaf"


def test_head_shares_no_parameters_with_e2_world_forward():
    head = _head(train_online=True)
    fwd = E2WorldForward(E2WorldConfig(
        use_e2_world_forward=True, z_world_dim=128, action_dim=ACTION_DIM))
    head_ids = {id(p) for p in head.parameters()}
    fwd_ids = {id(p) for p in fwd.parameters()}
    assert head_ids.isdisjoint(fwd_ids)


def test_simulation_mode_never_trains():
    torch.manual_seed(0)
    head = _head(train_online=True, warmup_steps=0, batch_size=4)
    z = torch.randn(1, WORLD_DIM)
    for i in range(200):
        head.observe_transition(z, _onehot(i % ACTION_DIM),
                                _heteroscedastic_step(z, i % ACTION_DIM),
                                simulation_mode=True)
    assert head.n_train_steps == 0
    assert head.n_observed_transitions == 0
    assert head.train_step(z, _onehot(0), z, simulation_mode=True) is None
    assert head.n_train_steps == 0


def test_train_step_refuses_mismatched_batch_rather_than_broadcasting():
    head = _head(train_online=True)
    z = torch.randn(4, WORLD_DIM)
    a = torch.zeros(3, ACTION_DIM); a[:, 0] = 1.0
    assert head.train_step(z, a, torch.randn(4, WORLD_DIM)) is None
    assert head.n_train_steps == 0


# ------------------------------------------------------------------ #
# Agent wiring                                                        #
# ------------------------------------------------------------------ #

def _agent_cfg(train_online: bool) -> REEConfig:
    env = CausalGridWorldV2()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32,
        reafference_action_dim=env.action_dim,
    )
    cfg.use_structured_curiosity = True
    cfg.curiosity_uncertainty_source = "e2_predictive_variance"
    cfg.latent.use_e2_world_uncertainty = True
    cfg.latent.use_e2_world_uncertainty_online_training = train_online
    cfg.latent.e2_world_uncertainty_warmup_steps = 20
    cfg.latent.e2_world_uncertainty_batch_size = 8
    return cfg


def _drive_agent(train_online: bool, episodes: int = 2, steps: int = 30):
    torch.manual_seed(71)
    env = CausalGridWorldV2()
    _, obs = env.reset()
    cfg = _agent_cfg(train_online)
    agent = REEAgent(cfg)
    world_dim = cfg.latent.world_dim
    for _ in range(episodes):
        _, obs = env.reset()
        agent.reset()
        assert agent._e2u_prev_z_world is None, (
            "reset() must clear the prev-z_world cache so the first tick of an "
            "episode never trains on a cross-episode transition"
        )
        for _ in range(steps):
            latent = agent.sense(obs["body_state"], obs["world_state"])
            ticks = agent.clock.advance()
            e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                        else torch.zeros(1, world_dim, device=agent.device))
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=0.0,
                drive_level=REEAgent.compute_drive_level(obs["body_state"]))
            action = agent.select_action(candidates, ticks)
            _f, harm, _d, _i, obs = env.step(int(action.argmax(dim=-1).item()))
            agent.update_residue(harm)
    return agent


def test_agent_off_path_never_trains_the_head():
    agent = _drive_agent(train_online=False)
    head = agent.e2_world_uncertainty
    assert head is not None
    assert head.n_train_steps == 0
    assert head.n_observed_transitions == 0
    assert head._optimizer is None


def test_agent_on_path_trains_the_head_across_a_real_rollout():
    agent = _drive_agent(train_online=True)
    head = agent.e2_world_uncertainty
    assert head is not None
    assert head.n_observed_transitions > 0
    assert head.n_train_steps > 0, (
        "the online loop did not fire on a real rollout -- the keystone is not live"
    )
    assert head.get_state()["e2_world_uncertainty_last_train_loss"] > 0.0


def test_agent_eval_mode_freezes_the_head_p2():
    """P2: a measurement phase must measure frozen weights."""
    torch.manual_seed(71)
    env = CausalGridWorldV2()
    _, obs = env.reset()
    agent = REEAgent(_agent_cfg(train_online=True))
    agent.eval()
    world_dim = agent.config.latent.world_dim
    for _ in range(40):
        latent = agent.sense(obs["body_state"], obs["world_state"])
        ticks = agent.clock.advance()
        e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=agent.device))
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        agent.update_z_goal(
            benefit_exposure=0.0,
            drive_level=REEAgent.compute_drive_level(obs["body_state"]))
        action = agent.select_action(candidates, ticks)
        _f, harm, _d, _i, obs = env.step(int(action.argmax(dim=-1).item()))
        agent.update_residue(harm)
    assert agent.e2_world_uncertainty.n_train_steps == 0
