"""Contract: ree_core.utils.grad_reach_guard is a working gradient-reach instrument.

The guard (a pure instrument; nothing in ree_core imports it) reports, per parameter
tensor, whether the optimizer(s) holding it actually train it. Spec: REE_assembly
evidence/planning/native_waking_trainer_design_20260925.md section 3 (0c0f5b76ec).

These tests drive the REAL agent (REEConfig.from_dims -> REEAgent) and the REAL recipes;
nothing whose training the guard asserts on is stubbed (CLAUDE.md "the test half").

  (a) CANARY 1 -- the V3-EXQ-1078-style naive recipe, Adam(agent.parameters()) on
      compute_prediction_loss + compute_e2_loss, is FAIL, and
      hippocampal.action_object_decoder is among the dead tensors. The status-quo check
      "some parameter in the optimizer moved" passes on the same run (pinned: that is the
      blind spot this instrument closes).
  (b) CANARY 2 -- the smallest recipe that reproduces an E2-self dead entry: the all-ON
      e2 optimizer and its SD-056 contrastive step, used verbatim from
      experiments/_lib/allon_training.py (Adam(agent.e2.parameters()) at the
      _train_all_on_agent optimizer build; _e2_contrastive_step), on a native-config
      agent. FAIL naming e2.self_transition and e2.self_action_encoder.
  (c) POSITIVE CONTROL -- the ree_core ZSelfP0Trainer (DR-13 config), observed through
      the global optimizer hook because it builds its optimizer inside train(): PASS.
  (d) VACUOUS -- zero steps, no optimizers, a too-short window, an all-allowlisted group:
      CANNOT_DETERMINE, and asserted != PASS; the same group over a full window PASSes.
  (e) ALLOWLIST -- a correctly scoped E1 group PASSes WITH the default allowlist and FAILs
      without it on exactly e1.context_memory.write_gate; a planted wrong entry FAILs as
      stale; in (a) the allowlisted write_gate does not fail while the non-allowlisted
      dead decoder does.
  (f) BYTE-NEUTRAL -- a training run observed by the guard ends with parameters and the
      global RNG state identical to the same run unobserved.

Budget: CPU, torch.set_num_threads(2), self_dim = world_dim = 16, a 6x6 grid.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils import grad_reach_guard as grg
from ree_core.utils.config import REEConfig

SEED = 42
DIM = 16
MIN_STEPS = 8
N_STEPS = 12

# V3-EXQ-1078 CFG_KW (experiments/v3_exq_1078_inv069_zself_coherence_unsettled.py), dims
# shrunk: DR-13 self-recurrence on, per-stream VS on.
DR13_KW = dict(alpha_world=0.9, use_self_recurrence=True, self_recurrence_e1_coupling=0.15,
               use_per_stream_vs=True)


@pytest.fixture(autouse=True)
def _threads():
    prev = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(prev)


def _seed_all(s: int = SEED) -> None:
    torch.manual_seed(s)
    random.seed(s)
    np.random.seed(s)


def _env(seed: int = SEED) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, size=6, num_hazards=1, num_resources=3,
                             hazard_harm=0.1, contaminated_harm=0.0)


def _agent(env: CausalGridWorldV2, **extra) -> REEAgent:
    cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
                              action_dim=env.action_dim, self_dim=DIM, world_dim=DIM, **extra)
    return REEAgent(cfg)


def _harness_train(agent, env, opt, loss_fn, n_steps: int = N_STEPS) -> None:
    """The V3-EXQ-1078 P0 inner loop: StepHarness tick, then step `opt` on `loss_fn`."""
    from experiments._harness import StepHarness
    harness = StepHarness(agent, env, train_mode=True, seed=SEED)
    agent.train()
    _, obs = env.reset()
    agent.reset()
    harness.reset()
    for _ in range(n_steps):
        r = harness.step(obs)
        obs = r.next_obs_dict
        loss = loss_fn()
        if loss.requires_grad:
            opt.zero_grad()
            loss.backward()
            opt.step()
        if r.done:
            _, obs = env.reset()
            agent.reset()
            harness.reset()


def _status_quo_moved(res: grg.GradReachResult) -> bool:
    """The check drivers use today: did ANY parameter in the optimizer move?"""
    return any(t.in_optimizer and t.moved for t in res.tensors)


# ----------------------------------------------------------------------------- (a) + (e)
def _naive_1078_result() -> grg.GradReachResult:
    _seed_all()
    env = _env()
    agent = _agent(env, **DR13_KW)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    return grg.check_grad_reach(
        lambda: _harness_train(agent, env, opt,
                               lambda: agent.compute_prediction_loss()
                               + agent.compute_e2_loss()),
        optimizers=[opt], named_parameters=agent.named_parameters(), min_steps=MIN_STEPS)


@pytest.fixture(scope="module")
def naive_1078_result():
    return _naive_1078_result()


def test_a_canary_naive_1078_recipe_fails_on_decoder(naive_1078_result):
    res = naive_1078_result
    assert res.status == grg.FAIL, res.summary()
    assert res.optimizer_steps and min(res.optimizer_steps.values()) >= MIN_STEPS
    dead = res.dead_names
    assert any(n.startswith("hippocampal.action_object_decoder.") for n in dead), res.summary()
    # Non-vacuity of the FAIL: the recipe does train something (E1 is reached), so this is
    # a partially-dead optimizer, not an optimizer that never stepped.
    reached = [t.name for t in res.by_status(grg.T_REACHED)]
    assert any(n.startswith("e1.transition_rnn.") for n in reached), res.summary()
    # The status-quo check is blind to it.
    assert _status_quo_moved(res) is True


def test_e_allowlist_suppresses_frozen_but_not_dead_in_naive_recipe(naive_1078_result):
    res = naive_1078_result
    wg = [t for t in res.tensors if t.name.startswith("e1.context_memory.write_gate.")]
    assert wg, "write_gate tensors missing from the agent -- allowlist test is vacuous"
    assert all(t.status == grg.T_ALLOWLISTED for t in wg), [(t.name, t.status) for t in wg]
    assert not any(n.startswith("e1.context_memory.write_gate.") for n in res.failing_names)
    # ...while a non-allowlisted dead module in the same optimizer still fails.
    assert any(n.startswith("hippocampal.action_object_decoder.") for n in res.failing_names)


# ----------------------------------------------------------------------------------- (b)
def test_b_canary_allon_e2_optimizer_fails_on_e2_self():
    from experiments._lib import allon_training as at
    _seed_all()
    env = _env()
    agent = _agent(env, alpha_world=0.3)  # the from_dims default, made explicit
    agent.eval()
    # Fill the recipe's (z_world_0, action, z_world_1) buffer from the agent's own
    # encoder, with uniformly random actions so the class-diverse sampler has classes.
    buf = []
    rng_act = random.Random(SEED)
    _, obs = env.reset()
    agent.reset()
    prev = None
    with torch.no_grad():
        for _ in range(40):
            latent = agent.sense(obs_body=obs["body_state"].float().unsqueeze(0),
                                 obs_world=obs["world_state"].float().unsqueeze(0))
            z = latent.z_world.detach().reshape(-1).clone()
            if prev is not None:
                buf.append((prev[0], prev[1], z))
            a = rng_act.randrange(env.action_dim)
            onehot = torch.zeros(env.action_dim)
            onehot[a] = 1.0
            prev = (z, onehot)
            _, _, done, _, obs = env.step(a)
            if done:
                _, obs = env.reset()
                agent.reset()
                prev = None
    assert len(buf) >= at.MIN_BUFFER_BEFORE_TRAIN

    e2_opt = torch.optim.Adam(agent.e2.parameters(), lr=at.E2_CONTRASTIVE_LR)
    sample_rng = random.Random(SEED)
    n_stepped = []

    def run():
        for _ in range(N_STEPS):
            if at._e2_contrastive_step(agent, buf, e2_opt, sample_rng) is not None:
                n_stepped.append(1)

    res = grg.check_grad_reach(run, optimizers=[e2_opt],
                               named_parameters=agent.named_parameters(),
                               min_steps=MIN_STEPS)
    assert sum(n_stepped) >= MIN_STEPS, "recipe did not step -- canary would be vacuous"
    assert res.status == grg.FAIL, res.summary()
    dead = res.dead_names
    assert any(n.startswith("e2.self_transition.") for n in dead), res.summary()
    assert any(n.startswith("e2.self_action_encoder.") for n in dead), res.summary()
    # The group does train its world head, and the status-quo check is blind again.
    assert any(t.name.startswith("e2.world_transition.")
               for t in res.by_status(grg.T_REACHED)), res.summary()
    assert _status_quo_moved(res) is True


# ----------------------------------------------------------------------------------- (c)
def test_c_positive_control_zself_p0_trainer_passes():
    from ree_core.latent.zself_p0 import ZSelfP0Config, ZSelfP0Trainer
    _seed_all()
    env = _env()
    agent = _agent(env, **DR13_KW)
    cfg = ZSelfP0Config(updates=10, batch_size=4, chunk_length=4, head_hidden=16, seed=SEED,
                        ridge_folds=2)
    trainer = ZSelfP0Trainer(agent, cfg)
    rng_act = random.Random(SEED)
    for _ep in range(4):
        _, obs = env.reset()
        for _t in range(10):
            a = rng_act.randrange(env.action_dim)
            trainer.observe(obs["body_state"], obs["world_state"], a)
            with torch.no_grad():
                _, _, done, _, obs = env.step(a)
            if done:
                break
        trainer.observe(obs["body_state"], obs["world_state"], None)
        trainer.end_episode()

    res = grg.check_grad_reach(trainer.train, optimizers=None,
                               named_parameters=agent.named_parameters(),
                               min_steps=MIN_STEPS)
    assert res.status == grg.PASS, res.summary()
    assert len(res.optimizer_steps) == 1 and list(res.optimizer_steps.values())[0] >= MIN_STEPS
    assert res.n_checked > 0
    held = {t.name for t in res.tensors if t.in_optimizer}
    assert any(n.startswith("body_obs_encoder.") for n in held), sorted(held)[:5]
    assert any(n.startswith("latent_stack.self_recurrence.") for n in held), sorted(held)[:5]
    # G5 (report-only): the recipe's gradient reaches world_obs_encoder, which its optimizer
    # excludes on purpose (the census ** case; an open user decision, design record D3).
    # Reported, and it does not turn the PASS into a FAIL.
    assert any(n.startswith("world_obs_encoder.") for n in res.leaked_names), res.summary()


# ----------------------------------------------------------------------------------- (d)
def _linear_steps(lin, opt, n):
    for _ in range(n):
        opt.zero_grad()
        lin(torch.randn(4, 3)).sum().backward()
        opt.step()


def test_d_vacuous_cases_are_cannot_determine_never_pass():
    _seed_all()
    lin = torch.nn.Linear(3, 1)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1)
    named = list(lin.named_parameters())

    cases = {
        "zero steps": grg.check_grad_reach(lambda: None, optimizers=[opt],
                                           named_parameters=named, allowlist={}),
        "no optimizers": grg.check_grad_reach(lambda: _linear_steps(lin, opt, MIN_STEPS),
                                              optimizers=[], named_parameters=named,
                                              allowlist={}),
        "global hook, nothing steps": grg.check_grad_reach(lambda: None, optimizers=None,
                                                           named_parameters=named,
                                                           allowlist={}),
        "window too short": grg.check_grad_reach(lambda: _linear_steps(lin, opt, 3),
                                                 optimizers=[opt], named_parameters=named,
                                                 allowlist={}, min_steps=MIN_STEPS),
    }
    # All tensors allowlisted and genuinely unreached: nothing is checked.
    other = torch.nn.Linear(3, 1)
    opt3 = torch.optim.SGD(other.parameters(), lr=0.1)

    def step_other_no_grad():
        for _ in range(MIN_STEPS):
            opt3.zero_grad()
            opt3.step()

    cases["every tensor allowlisted"] = grg.check_grad_reach(
        step_other_no_grad, optimizers=[opt3],
        named_parameters=[("other." + n, p) for n, p in other.named_parameters()],
        allowlist={"other": "frozen by design (test)"}, min_steps=MIN_STEPS)

    for label, res in cases.items():
        assert res.status == grg.CANNOT_DETERMINE, (label, res.summary())
        assert res.status != grg.PASS, label
        assert res.reasons, label

    # Non-degeneracy: the same group over a full window is a real PASS.
    full = grg.check_grad_reach(lambda: _linear_steps(lin, opt, MIN_STEPS), optimizers=[opt],
                                named_parameters=named, allowlist={}, min_steps=MIN_STEPS)
    assert full.status == grg.PASS, full.summary()


# ----------------------------------------------------------------------------------- (e)
def test_e_allowlist_is_necessary_specific_and_stale_checked():
    _seed_all()
    env = _env()
    agent = _agent(env, **DR13_KW)
    named = list(agent.named_parameters())
    e1_ids = {id(p) for p in agent.e1.parameters()}
    opt = torch.optim.Adam(agent.e1.parameters(), lr=1e-3)

    allow_stale = dict(grg.FROZEN_BY_DESIGN)
    allow_stale["e1.transition_rnn"] = "WRONG entry planted by this test"
    g_default = grg.GradReachGuard(named, min_steps=MIN_STEPS)
    g_none = grg.GradReachGuard(named, allowlist={}, min_steps=MIN_STEPS)
    g_stale = grg.GradReachGuard(named, allowlist=allow_stale, min_steps=MIN_STEPS)
    for g in (g_default, g_none, g_stale):
        g.attach(opt)
    try:
        _harness_train(agent, env, opt, lambda: agent.compute_prediction_loss())
    finally:
        for g in (g_default, g_none, g_stale):
            g.detach()
    r_default, r_none, r_stale = g_default.result(), g_none.result(), g_stale.result()

    assert r_default.status == grg.PASS, r_default.summary()
    assert r_default.n_checked > 0
    assert {t.status for t in r_default.tensors if t.in_optimizer} <= {
        grg.T_REACHED, grg.T_ALLOWLISTED}
    assert len([t for t in r_default.tensors if t.in_optimizer]) == len(e1_ids)

    assert r_none.status == grg.FAIL, r_none.summary()
    assert r_none.failing_names, r_none.summary()
    assert all(n.startswith("e1.context_memory.write_gate.") for n in r_none.failing_names), \
        r_none.failing_names

    assert r_stale.status == grg.FAIL, r_stale.summary()
    stale = [t.name for t in r_stale.by_status(grg.T_STALE_ALLOWLIST)]
    assert stale and all(n.startswith("e1.transition_rnn.") for n in stale), stale


# ----------------------------------------------------------------------------------- (f)
def _rng_training(observe: bool):
    torch.manual_seed(7)
    lin = torch.nn.Linear(5, 2)
    drop = torch.nn.Dropout(p=0.5)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)

    def run():
        for _ in range(MIN_STEPS):
            opt.zero_grad()
            drop(lin(torch.randn(6, 5))).pow(2).sum().backward()
            opt.step()

    if observe:
        res = grg.check_grad_reach(run, optimizers=[opt],
                                   named_parameters=lin.named_parameters(), allowlist={},
                                   min_steps=MIN_STEPS)
        res.summary()
    else:
        run()
    return [p.detach().clone() for p in lin.parameters()], torch.get_rng_state()


def test_f_guard_is_byte_neutral():
    p_bare, s_bare = _rng_training(observe=False)
    p_obs, s_obs = _rng_training(observe=True)
    assert torch.equal(s_bare, s_obs)
    assert all(torch.equal(a, b) for a, b in zip(p_bare, p_obs))


def test_summary_is_ascii(naive_1078_result):
    naive_1078_result.summary().encode("ascii")
