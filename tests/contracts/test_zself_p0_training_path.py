"""Contracts for sd_zself_training_path: the P0 objective that actually reaches z_self.

DR-13 shipped a dedicated self-recurrence on the premise that it "trains via the
EXISTING E1/E2 z_self prediction losses -- v1 adds no new loss". V3-EXQ-1078 measured
that premise FALSE on 3/3 seeds (`gru_param_max_delta = 0.0`,
`latent_stack_tensors_changed = 0/53`) and the causal-reach trace
(`REE_assembly/evidence/planning/zself_causal_reach_trace_20260924.md`) root-caused it:
every z_self the E1/E2 losses see is a detached copy (`agent.py:5855`, `:6294`,
`:11397`). `ree_core/latent/zself_p0.py` is the repair -- a phased body-forward-model P0
that runs its own forward passes over recorded observations.

THE CRITERIA ARE THE TRACE'S SECTION 9, AND THE OBVIOUS ONE IS NOT ENOUGH.
"Nonzero GRU delta AND a changed self_encoder param" is NECESSARY BUT NOT SUFFICIENT:
all four candidate objectives the trace prototyped pass it, INCLUDING the two that
COLLAPSE the self-state (effective rank to ~1.0, or the z norm shrunk 5x with the loss
at the trivial solution) -- and a collapsed z_self makes INV-069's coherence trivially
maximal and MECH-113's D_eff trivially minimal, so both would read as SUCCESS and be
vacuous. Hence C2-C4 below.

  C1  attributable change: the GRU AND self_encoder move under the new objective and
      NOT under the E1/E2 replay path (the V3-EXQ-1078 signature, pinned here so the
      C1 assertion is a contrast and not a bare fact); the world path is untouched
  C2  non-collapse: held-out effective rank and z norm not below the untrained baseline
  C3  held-out information gain on episodes the objective never trained on -- next-body
      AND a history target, the history target above the raw-observation ceiling
  C4  a native consumer responds: E1's prediction moves more under a z_self swap than
      it did before training
  C5  bit-identical OFF, and RNG-neutral ON
  C6  instrument hygiene -- the readouts cannot report a false negative quietly

SCOPE, STATED RATHER THAN OMITTED. There is deliberately NO behavioural-consequence
(D3) criterion here. The trace measured 0/68 E3 ticks and 0/12 whole episodes with any
action change under z_self intervention (z_world canary 4-30% in the same harness),
because no valuation consumer reads z_self. An INV-069 / MECH-113 retest on top of this
build measures self-state quality and E1 use ONLY. See GFLAG-0481.

generation:v4 -- off the V3 closure path; PROMOTES NOTHING.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.zself_p0 import (
    ZSelfP0Config,
    ZSelfP0Trainer,
    effective_rank,
    ridge_cv_r2,
    self_path_parameter_names,
)
from ree_core.utils.config import REEConfig

# The trace's config A, at the deployed dims. The information-gain criterion is a
# held-out ridge readout, so it needs enough episodes and enough self_dim to be
# meaningful: measured at size=9 / self_dim=24 / 16 episodes the next-body leg is
# within noise on one of two seeds, and at these values it separates cleanly on both
# (see the module docstring of zself_p0.py for the trace's own table).
ENV_KW = dict(size=10, num_hazards=2, num_resources=5,
              hazard_harm=0.1, contaminated_harm=0.0)
SELF_DIM = 32
REC_EPISODES = 20
REC_STEPS = 100
SEED = 42


def _build(seed: int = SEED, **flags):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KW)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=SELF_DIM,
        alpha_world=0.9,
        **flags,
    )
    return REEAgent(cfg), env


def _record(trainer, env, seed, episodes=REC_EPISODES, steps=REC_STEPS):
    """Random-policy rollout into the trainer's buffer. The agent is not driven."""
    rs = np.random.RandomState(seed)
    for _ep in range(episodes):
        _flat, od = env.reset()
        for _t in range(steps):
            a = int(rs.randint(0, env.action_dim))
            trainer.observe(od["body_state"], od["world_state"], a)
            with torch.no_grad():
                _flat, _harm, done, _info, od = env.step(a)
            if done:
                break
        trainer.observe(od["body_state"], od["world_state"], None)
        trainer.end_episode()


def _native_z(agent, body, world, action):
    """z_self/z_world over one recorded episode, through the native encode path.

    Written out here rather than reusing the trainer's private `_native_chain` so the
    consumer-response measurement in C4 does not depend on the module it is checking.
    """
    ls = agent.latent_stack
    prev = ls.init_state(1, body.device)
    zs, zw = [], []
    with torch.no_grad():
        for t in range(body.shape[0]):
            enc = torch.cat(
                [agent.body_obs_encoder(body[t].unsqueeze(0)),
                 agent.world_obs_encoder(world[t].unsqueeze(0))], dim=-1)
            pa = action[t - 1].unsqueeze(0) if t > 0 else None
            st = ls.encode(enc, prev, prev_action=pa)
            zs.append(st.z_self)
            zw.append(st.z_world)
            prev = st
    return torch.cat(zs, 0), torch.cat(zw, 0)


@pytest.fixture(scope="module")
def trained():
    """One trace-scale P0 run, shared by C1-C4.

    Returns (agent, pre_train_state_dict, stats, holdout_episodes). The pre-train state
    dict is the C4 baseline: the SAME agent before its first optimizer step, which is a
    tighter control than a separately-seeded build.
    """
    agent, env = _build(use_self_recurrence=True, self_recurrence_e1_coupling=0.15)
    trainer = ZSelfP0Trainer(
        agent, ZSelfP0Config(seed=SEED, updates=300, chunk_length=16,
                             batch_size=16, holdout_fraction=0.25))
    _record(trainer, env, SEED)
    pre = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    stats = trainer.train()
    # The holdout episodes the readout used, for the C4 consumer measurement.
    rng = random.Random(SEED)
    _tr_idx, hold_idx = trainer._split_episodes(rng)
    hold = [trainer._episodes[i] for i in hold_idx]
    return agent, pre, stats, hold


# --- C1 attributable change ------------------------------------------------------------
def test_c1_e1e2_replay_alone_leaves_the_self_path_frozen():
    """The V3-EXQ-1078 signature, pinned.

    This is the DEFECT, and it is still live: the E1/E2 losses read detached copies of
    z_self, so stepping `agent.parameters()` on them moves NOTHING on the z_self path.
    Without this test the C1 assertion below is a bare fact about a new module; with it,
    C1 is a contrast against the route the DR-13 doc claimed would work -- which is what
    makes it fail on pre-change code, where the only available route is this one.
    """
    agent, env = _build(use_self_recurrence=True)
    agent.train()
    names = set(self_path_parameter_names(agent.latent_stack))
    assert names, "self-path predicate matched nothing"
    snap = {n: p.detach().clone() for n, p in agent.latent_stack.named_parameters()}
    opt = torch.optim.Adam(agent.parameters(), lr=1e-3)
    _flat, od = env.reset()
    agent.reset()
    backwards = 0
    for _t in range(40):
        b, w = od["body_state"], od["world_state"]
        if b.dim() == 1:
            b = b.unsqueeze(0)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        a = agent.act_with_split_obs(b, w)
        loss = agent.compute_prediction_loss() + agent.compute_e2_loss()
        if getattr(loss, "requires_grad", False):
            opt.zero_grad()
            loss.backward()
            opt.step()
            backwards += 1
        _flat, _harm, done, _info, od = env.step(a)
        if done:
            break
    assert backwards > 0, "no E1/E2 backward ran -- the control did not exercise anything"
    moved = {n: float((p.detach() - snap[n]).abs().max())
             for n, p in agent.latent_stack.named_parameters() if n in names}
    assert max(moved.values()) == 0.0, (
        "E1/E2 replay moved the z_self path -- the V3-EXQ-1078 root cause has changed "
        "and this contract's premise needs re-deriving: %r"
        % ({n: v for n, v in moved.items() if v > 0.0},))


def test_c1_zself_p0_moves_the_gru_and_the_self_encoder(trained):
    _agent, _pre, stats, _hold = trained
    d = stats["param_delta"]
    assert stats["used_self_recurrence"] is True
    assert d["self_recurrence"] is not None and d["self_recurrence"] > 0.0, (
        "the DR-13 GRU did not move: %r" % (d,))
    assert d["self_encoder"] is not None and d["self_encoder"] > 0.0, (
        "self_encoder did not move: %r" % (d,))
    assert d["self_precision_logit"] > 0.0
    assert d["body_obs_encoder"] > 0.0


def test_c1_zself_p0_leaves_the_world_path_untouched(trained):
    """If the z_self P0 silently trained z_world, every world-stream result run after it
    would be confounded -- the mirror of SD-070's own C5."""
    agent, pre, _stats, _hold = trained
    live = agent.state_dict()
    moved = [
        k for k in pre
        if ("world_encoder" in k.split(".") or k.split(".")[-1] == "world_precision_logit")
        and not torch.equal(live[k], pre[k])
    ]
    assert not moved, "the z_self P0 moved world-path tensors: %r" % (moved,)


# --- C2 non-collapse --------------------------------------------------------------------
def test_c2_does_not_collapse_the_self_state(trained):
    """The criterion that separates the shipped objective from the two the trace
    rejected: (iii) anchor-matching drove effective rank to 1.02, and (iv) the live-tap
    E2 route shrank the z norm 5x to the trivial solution. Both move the GRU and
    self_encoder, so C1 alone cannot tell them from this one."""
    _agent, _pre, stats, _hold = trained
    before, after = stats["holdout_before"], stats["holdout_after"]
    for key in ("z_eff_rank", "z_norm_mean"):
        assert before[key] is not None and after[key] is not None, (
            "%s could not be computed -- the readout is broken, which is NOT the same "
            "as a failing score" % key)
    assert after["z_eff_rank"] >= before["z_eff_rank"], (
        "effective rank fell (collapse): %.3f -> %.3f"
        % (before["z_eff_rank"], after["z_eff_rank"]))
    assert after["z_norm_mean"] >= before["z_norm_mean"], (
        "z norm fell (collapse): %.3f -> %.3f"
        % (before["z_norm_mean"], after["z_norm_mean"]))


# --- C3 held-out information gain -------------------------------------------------------
def test_c3_gains_heldout_next_body_information(trained):
    _agent, _pre, stats, _hold = trained
    b, a = stats["holdout_before"], stats["holdout_after"]
    assert b["n_episodes"] >= 2 and b["n_rows"] > 0
    assert b["R2_next_body"] is not None and a["R2_next_body"] is not None
    assert a["R2_next_body"] > b["R2_next_body"], (
        "no held-out next-body gain: %.3f -> %.3f"
        % (b["R2_next_body"], a["R2_next_body"]))


def test_c3_gains_heldout_recurrent_history_information(trained):
    """The leg that makes the self-state TEMPORAL rather than a re-encoded snapshot.

    `ceiling_R2_action_history_from_raw_body` is how much of action(t-2) the raw
    instantaneous body observation already carries -- measured at -0.03 in the trace,
    i.e. essentially nothing. A z_self score above that ceiling is content the current
    frame does not contain, so it can only have come through the recurrence.
    """
    _agent, _pre, stats, _hold = trained
    b, a = stats["holdout_before"], stats["holdout_after"]
    assert b["R2_action_history"] is not None and a["R2_action_history"] is not None
    assert a["R2_action_history"] > b["R2_action_history"], (
        "no held-out history gain: %.3f -> %.3f"
        % (b["R2_action_history"], a["R2_action_history"]))
    ceiling = a["ceiling_R2_action_history_from_raw_body"]
    assert ceiling is not None
    assert a["R2_action_history"] > ceiling, (
        "the history score (%.3f) does not clear the raw-observation ceiling (%.3f), so "
        "it is not evidence of recurrent content" % (a["R2_action_history"], ceiling))


# --- C4 a native consumer responds -------------------------------------------------------
def _e1_swap_response(agent, episodes, n_ticks=40, seed=SEED):
    """Mean relative change in E1's next-step prediction under a z_self swap.

    E1 is the ONLY native consumer that responds to z_self at all (the trace's Section
    3a read-site table); E3 scores world rollouts only. The swap partner is another
    tick's real z_self, so this asks whether E1 distinguishes one self-state from a
    genuine alternative, not whether it reacts to an arbitrary vector.

    The E1 hidden state is saved and restored around the intervened forward pass, so the
    counterfactual costs the control nothing: the stream E1 actually carries forward is
    the unintervened one.

    Caveat recorded rather than hidden: a trained z_self also has a LARGER NORM, so part
    of any rise here is the perturbation being bigger in absolute terms. The test
    therefore reports the norms alongside (C2) rather than claiming the rise is purely
    representational.
    """
    rng = random.Random(seed)
    zs_all, zw_all = [], []
    for body, world, act in episodes:
        z_s, z_w = _native_z(agent, body, world, act)
        zs_all.append(z_s)
        zw_all.append(z_w)
    zs = torch.cat(zs_all, 0)
    zw = torch.cat(zw_all, 0)
    n = min(int(n_ticks), zs.shape[0] - 1)
    rels = []
    agent.e1.reset_hidden_state()
    with torch.no_grad():
        for t in range(n):
            inp = torch.cat([zs[t: t + 1], zw[t: t + 1]], dim=-1)
            saved = agent.e1._hidden_state
            ctrl, _ = agent.e1(inp)
            agent.e1._hidden_state = saved
            j = rng.randrange(zs.shape[0])
            while j == t:
                j = rng.randrange(zs.shape[0])
            swapped, _ = agent.e1(torch.cat([zs[j: j + 1], zw[t: t + 1]], dim=-1))
            agent.e1._hidden_state = saved
            c0, s0 = ctrl[:, 0, :], swapped[:, 0, :]
            rels.append(float((s0 - c0).norm() / (c0.norm() + 1e-9)))
            # commit the CONTROL tick, so E1 carries the unintervened stream forward
            _c, _ = agent.e1(inp)
    return float(np.mean(rels)) if rels else None


def test_c4_e1_responds_to_a_zself_swap_above_the_untrained_baseline(trained):
    agent, pre, _stats, hold = trained
    assert hold, "no holdout episodes to measure on"
    baseline_agent, _env = _build(use_self_recurrence=True,
                                  self_recurrence_e1_coupling=0.15)
    baseline_agent.load_state_dict(pre)
    baseline_agent.eval()
    agent.eval()
    before = _e1_swap_response(baseline_agent, hold)
    after = _e1_swap_response(agent, hold)
    assert before is not None and after is not None
    assert after > before, (
        "E1 does not respond to the trained self-state more than to the untrained one "
        "(%.4f -> %.4f); with no behavioural consumer, E1 is the ONLY native consumer "
        "this build can reach" % (before, after))


# --- C5 bit-identical OFF, RNG-neutral ON -------------------------------------------------
def test_c5_warmup_is_a_no_op_at_zero_episodes():
    from experiments._lib.zself_p0_warmup import run_zself_p0

    agent, env = _build(use_self_recurrence=True)
    before = {k: v.detach().clone() for k, v in agent.state_dict().items()}
    t_state = torch.get_rng_state()
    out = run_zself_p0(agent, env, seed=1, episodes=0, steps_per_episode=10,
                       policy=None)
    assert out["p0s_ran"] is False and out["p0s_reason"] == "episodes<=0"
    assert all(torch.equal(agent.state_dict()[k], v) for k, v in before.items())
    assert torch.equal(torch.get_rng_state(), t_state)


def test_c5_warmup_is_rng_neutral_when_it_runs():
    """Turning the warmup ON must not displace the global RNG stream, or an ON-vs-OFF
    comparison confounds 'z_self is now trained' with 'every later draw moved'."""
    from experiments._lib.capability_eval import RandomPolicy
    from experiments._lib.zself_p0_warmup import run_zself_p0

    agent, env = _build(use_self_recurrence=True)
    warm_env = CausalGridWorldV2(seed=SEED, **ENV_KW)
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)
    t_state, np_state, py_state = (
        torch.get_rng_state(), np.random.get_state(), random.getstate())
    out = run_zself_p0(agent, warm_env, seed=1, episodes=3, steps_per_episode=40,
                       policy=RandomPolicy(1), dry_run=True)
    assert out["p0s_ran"] is True, out
    assert torch.equal(torch.get_rng_state(), t_state)
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert random.getstate() == py_state


def test_c5_no_new_latent_stack_config_surface():
    """The no-op guarantee is STRUCTURAL -- nothing constructs a trainer unless asked,
    and the wrapper returns at episodes<=0 -- rather than a flag that can be left in the
    wrong state. Same design decision as SD-070; pinned so it is not quietly reversed."""
    from ree_core.utils.config import LatentStackConfig
    fields = set(LatentStackConfig.__dataclass_fields__)
    assert not any("zself_p0" in f for f in fields)


def test_c5_agent_gains_no_zself_p0_method():
    assert not any(n.startswith("compute_zself_p0") for n in dir(REEAgent))


# --- C6 instrument hygiene ----------------------------------------------------------------
def test_c6_self_path_predicate_finds_the_gru_from_either_root():
    """The predicate must match `self_recurrence.cell.weight_ih` (names read off the
    LatentStack) as well as `latent_stack.self_recurrence.cell.weight_ih` (names read
    off the agent). A '.self_recurrence.' SUBSTRING test matches only the second, which
    silently drops the GRU from the optimizer while everything else still trains -- the
    exact defect this module exists to repair, one level down. Measured during the
    build: `param_delta['self_recurrence'] == 0.0` with the substring spelling."""
    agent, _env = _build(use_self_recurrence=True)
    from_stack = self_path_parameter_names(agent.latent_stack)
    assert any("self_recurrence" in n.split(".") for n in from_stack), from_stack
    assert any("self_encoder" in n.split(".") for n in from_stack)
    assert not any("self_predictor" in n.split(".") for n in from_stack), (
        "self_predictor is E1/E2's head on z_self, not the encoder path")


def test_c6_refuses_when_the_predicate_would_drop_the_gru(monkeypatch):
    """The guard's own guard. If the name predicate ever stops matching the GRU, the run
    must REFUSE rather than train everything else and look healthy."""
    import ree_core.latent.zself_p0 as mod

    agent, _env = _build(use_self_recurrence=True)
    t = mod.ZSelfP0Trainer(agent, ZSelfP0Config())
    monkeypatch.setattr(
        mod, "self_path_parameter_names",
        lambda ls: [n for n, _ in ls.named_parameters()
                    if "self_recurrence" not in n.split(".")])
    with pytest.raises(ValueError, match="self_recurrence"):
        t.self_path_parameters()


def test_c6_gru_delta_is_none_not_zero_when_dr13_is_off():
    """`None` = the module is not present; `0.0` = present and did not move. Collapsing
    the two would make a DR-13-off run read as the V3-EXQ-1078 failure."""
    agent, env = _build(seed=7)   # use_self_recurrence defaults OFF
    assert agent.latent_stack.self_recurrence is None
    t = ZSelfP0Trainer(agent, ZSelfP0Config(seed=7, updates=3, chunk_length=4,
                                            batch_size=2, holdout_fraction=0.25))
    _record(t, env, 7, episodes=4, steps=20)
    stats = t.train()
    assert stats["used_self_recurrence"] is False
    assert stats["param_delta"]["self_recurrence"] is None
    assert stats["param_delta"]["self_encoder"] > 0.0


def test_c6_ridge_returns_none_when_it_cannot_be_computed():
    """A probe that could not run must not be indistinguishable from a probe that ran
    and found nothing."""
    x = torch.randn(10, 3)
    y = torch.randn(10, 2)
    assert ridge_cv_r2(x, y, [0] * 10) is None, "one group cannot be cross-validated"
    assert ridge_cv_r2(torch.randn(2, 3), torch.randn(2, 2), [0, 1]) is None
    const = torch.zeros(40, 2)
    assert ridge_cv_r2(torch.randn(40, 3), const, [i // 10 for i in range(40)]) is None


def test_c6_effective_rank_detects_collapse():
    torch.manual_seed(0)
    spread = torch.randn(200, 8)
    direction = torch.randn(1, 8)
    collapsed = torch.randn(200, 1) * direction
    assert effective_rank(collapsed) < 1.5
    assert effective_rank(spread) > 4.0
    assert effective_rank(torch.randn(1, 8)) == 0.0


def test_c6_trainer_refuses_an_undersized_buffer():
    agent, env = _build(seed=3, use_self_recurrence=True)
    t = ZSelfP0Trainer(agent, ZSelfP0Config(seed=3, chunk_length=16))
    with pytest.raises(ValueError, match="at least 2 buffered episodes"):
        t.train()
    _record(t, env, 3, episodes=3, steps=6)
    with pytest.raises(ValueError, match="chunk_length"):
        t.train()


def test_c6_observe_accepts_an_index_or_a_one_hot_and_rejects_a_wrong_width():
    agent, env = _build(seed=4, use_self_recurrence=True)
    t = ZSelfP0Trainer(agent, ZSelfP0Config())
    _flat, od = env.reset()
    t.observe(od["body_state"], od["world_state"], 2)
    assert int(t._cur_action[-1].argmax()) == 2
    oh = torch.zeros(env.action_dim)
    oh[1] = 1.0
    t.observe(od["body_state"], od["world_state"], oh)
    assert int(t._cur_action[-1].argmax()) == 1
    with pytest.raises(ValueError, match="expected"):
        t.observe(od["body_state"], od["world_state"], torch.zeros(env.action_dim + 3))
    with pytest.raises(ValueError, match="outside"):
        t.observe(od["body_state"], od["world_state"], env.action_dim + 5)


def test_c6_buffer_does_not_alias_caller_tensors():
    agent, env = _build(seed=5, use_self_recurrence=True)
    t = ZSelfP0Trainer(agent, ZSelfP0Config())
    _flat, od = env.reset()
    body = od["body_state"].clone().float()
    t.observe(body, od["world_state"], 0)
    body.zero_()
    assert float(t._cur_body[-1].abs().sum()) > 0.0


def test_c6_end_episode_truncates_at_an_unrecorded_action():
    """A gap in the action record would make `prev_action` at that step a lie, and
    SD-007 reafference correction reads it."""
    agent, env = _build(seed=6, use_self_recurrence=True)
    t = ZSelfP0Trainer(agent, ZSelfP0Config())
    _flat, od = env.reset()
    for i in range(5):
        t.observe(od["body_state"], od["world_state"], None if i == 2 else 0)
    t.end_episode()
    assert t.n_episodes == 1
    body, _world, act = t._episodes[0]
    assert act.shape[0] == 2 and body.shape[0] == 3


def test_c6_action_dim_resolution_refuses_rather_than_defaulting_to_zero():
    """A zero width silently disables the one-hot check in observe(), and the objective
    would then condition on a scalar index whose magnitude is meaningless."""
    agent, _env = _build(seed=8, use_self_recurrence=True)

    class _Cfg:
        pass

    broken = _Cfg()
    broken.latent = agent.config.latent
    with pytest.raises(ValueError, match="action_dim"):
        ZSelfP0Trainer._resolve_action_dim(broken)
