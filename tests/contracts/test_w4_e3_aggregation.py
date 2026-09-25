"""Contracts for W4 E3 aggregation (default OFF; branch integration/coupled-loop-repair).

Plan of record: REE_assembly evidence/planning/coupled_loop_repair_campaign_plan.md section 3
W4. Selection: probe N3 proper (n3_e3_aggregation_probe_20260925.md, results 9608f3117a):
DISC_0.5 -- the simplest SD-081-compatible aggregation passing member gate (a) (4/5 seeds).
Build record: evidence/planning/w4_e3_aggregation_build_20260925.md.
Build: ree_core/predictors/e3_selector.py ``score_trajectory`` ->
``_score_trajectory_discounted``, knobs ``use_e3_discounted_aggregation`` /
``e3_aggregation_gamma`` on E3Config (+ REEConfig.from_dims signature and assignment).

  W4-01 OFF: defaults False / 0.5; OFF, the planned read is byte-identical to the full-depth
        read of the same scorer and does not read gamma; a default-config agent and an
        explicit-OFF agent with a different gamma act identically.
  W4-02 ON == N3's DISC_0.5 re-scoring (numerical parity) on a fixed RECORDED native pool
        (N3 s532, state 0, 32 candidates, the REAL W3 head). The oracle below is the probe's
        own ``depth_scores`` + ``aggregate`` copied verbatim. gamma = 1 telescopes to the full
        read exactly; gamma outside (0, 1] raises.
  W4-03 SD-081 depth contrast: the HABIT read (depth max(2, dualsystem_habit_depth), inside
        select()'s arbitration) is byte-identical ON vs OFF; the PLANNED read ON reads beyond
        the habit depth (a perturbation at world step 5 moves it, not the habit read) and
        its ranking of the recorded native pool is not the habit ranking (the contrast is
        THIN without a residue field -- measured, not gated; see that test's docstring).
  W4-04 a REDUCED member gate (a) on the recorded N3 s532 dataset: mean Spearman(J_pred,
        J_true) over the scaffold pool, REAL W3 head minus the fixed-permutation twin,
        > 0.15 with the W4 aggregation ON; the full-horizon read (pre-W4 behaviour) FAILS
        the same bar on the same data (canary that the dataset discriminates). Uses the I1
        instrument ``e3_choice_quality``.
  W4-05 knobs plumb through REEConfig.from_dims to agent.e3.config; an ON agent acts.

NO gate (c) contract: N3 found (c) structurally non-discriminating in both forms tried and
its re-specification is HELD FOR THE USER. W4 does not yet count as gate-passed.

Fixture ``tests/fixtures/w4_n3_s532.pt`` (dumped by REE_assembly
evidence/planning/probes/w4/w4_dump_fixture.py, which re-runs N3's seed-532 protocol
verbatim): the REAL and SHUF W3 member heads (world_transition + world_action_encoder), the
probe agent's harm_eval_head, and per probe state z0, the pool[0] action sequence the
scaffold pool is built from, and the true next z_world per action class; plus state 0's
native pool actions. Rollouts are re-done here with ree_core's own E2 and scored by
ree_core's own E3 (residue_field None: phi = 0; the probe agent carried a residue field).
Evidence domain of W4-04: D1 on recorded data (the ranking the native scorer produces), a
reduced form of N3's D2 reading. CPU-small (world_dim 32, deployed).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments._lib import coupled_acceptance as CA
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e2_fast import E2FastPredictor, Trajectory
from ree_core.predictors.e3_selector import E3Config, E3TrajectorySelector
from ree_core.utils.config import REEConfig

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "w4_n3_s532.pt"
A_ENV = 5


# ------------------------------------------------------------------ fixture + builders
@pytest.fixture(scope="module")
def fx():
    return torch.load(FIXTURE, weights_only=True)


def _cfg(fx):
    m = fx["meta"]
    return REEConfig.from_dims(body_obs_dim=m["body_obs_dim"], world_obs_dim=m["world_obs_dim"],
                               action_dim=A_ENV)


def _e2(fx, head):
    e2 = E2FastPredictor(_cfg(fx).e2)
    e2.world_transition.load_state_dict(fx["heads"][head]["t"])
    e2.world_action_encoder.load_state_dict(fx["heads"][head]["a"])
    return e2.eval()


def _e3(fx, **kw):
    cfg = _cfg(fx).e3
    for k, v in kw.items():
        setattr(cfg, k, v)
    e3 = E3TrajectorySelector(cfg)
    e3.harm_eval_head.load_state_dict(fx["harm_eval_head"])
    return e3.eval()


def _batch(trajs):
    L = len(trajs[0].world_states)
    return Trajectory(states=[torch.cat([t.states[k] for t in trajs]) for k in range(len(trajs[0].states))],
                      actions=torch.cat([t.actions for t in trajs]),
                      world_states=[torch.cat([t.world_states[k] for t in trajs]) for k in range(L)])


@torch.no_grad()
def _roll(e2, fx, i, actions):
    zs = torch.zeros(actions.shape[0], fx["meta"]["self_dim"])
    z0 = fx["z0"][i:i + 1].expand(actions.shape[0], -1)
    return e2.rollout_with_world(zs, z0, actions, compute_action_objects=False)


def _scaffold(e2, fx, i):
    base = fx["base_actions"][i:i + 1]
    acts = []
    for c in range(A_ENV):
        a = base.clone()
        a[:, 0, :] = 0.0
        a[:, 0, c] = 1.0
        acts.append(a)
    return _roll(e2, fx, i, torch.cat(acts))


def _native_pool(fx):
    return _roll(_e2(fx, "REAL"), fx, 0, fx["native_pool_actions_state0"])


@torch.no_grad()
def _jtrue(e3, fx, i):
    zs = torch.zeros(A_ENV, fx["meta"]["self_dim"])
    tt = Trajectory(states=[zs, zs], actions=torch.eye(A_ENV).unsqueeze(1),
                    world_states=[fx["z0"][i:i + 1].expand(A_ENV, -1), fx["z_true"][i]])
    return e3.score_trajectory(tt).reshape(-1).double().numpy()


# ---- N3 proper's re-scoring, VERBATIM from REE_assembly probes/n3/n3_probe.py (9608f3117a)
@torch.no_grad()
def _probe_depth_scores(e3, bt):
    Lmax = len(bt.world_states)
    prev = e3._score_depth_limit
    out = {}
    try:
        for L in list(range(2, Lmax + 1)) + [None]:
            e3._score_depth_limit = L
            out[L] = e3.score_trajectory(bt).detach().reshape(-1).numpy().astype(np.float64)
    finally:
        e3._score_depth_limit = prev
    return out, Lmax


def _probe_disc(J, Lmax, g):
    base = J[2]
    inc = {d: J[d + 1] - J[d] for d in range(2, Lmax)}
    return base + sum((g ** (d - 1)) * inc[d] for d in inc)
# ---- end verbatim


def _on(e3, bt, gamma=0.5):
    e3.config.use_e3_discounted_aggregation = True
    e3.config.e3_aggregation_gamma = gamma
    try:
        with torch.no_grad():
            return e3.score_trajectory(bt)
    finally:
        e3.config.use_e3_discounted_aggregation = False


# ------------------------------------------------------------------ W4-01 OFF
def test_w401_defaults_off():
    c = E3Config()
    assert c.use_e3_discounted_aggregation is False
    assert c.e3_aggregation_gamma == 0.5
    rc = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=40, action_dim=5)
    assert rc.e3.use_e3_discounted_aggregation is False
    assert rc.e3.e3_aggregation_gamma == 0.5


def test_w401_off_planned_read_is_full_read_and_ignores_gamma(fx):
    e3 = _e3(fx)
    bt = _native_pool(fx)
    Lmax = len(bt.world_states)
    with torch.no_grad():
        off = e3.score_trajectory(bt)
        e3.config.e3_aggregation_gamma = 0.9  # not read while OFF
        off_g = e3.score_trajectory(bt)
        e3._score_depth_limit = Lmax
        full_explicit = e3.score_trajectory(bt)
        e3._score_depth_limit = None
    assert torch.equal(off, off_g)
    assert torch.equal(off, full_explicit)


def _run_agent(seed, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, max_episode_steps=200, seed=seed)
    cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
                              action_dim=env.action_dim, **kw)
    agent = REEAgent(cfg)
    agent.eval()
    _f, od = env.reset()
    agent.reset()
    acts = []
    with torch.no_grad():
        for _ in range(12):
            body, world = od["body_state"], od["world_state"]
            body = body.unsqueeze(0) if body.dim() == 1 else body
            world = world.unsqueeze(0) if world.dim() == 1 else world
            a = agent.act_with_split_obs(body, world)
            c = int(a.reshape(-1).argmax())
            acts.append(c)
            _f, _h, done, _i, od = env.step(c)
            if done:
                _f, od = env.reset()
                agent.reset()
    return acts, agent


def test_w401_off_agent_identical_to_default():
    a0, _ = _run_agent(11)
    a1, ag = _run_agent(11, use_e3_discounted_aggregation=False, e3_aggregation_gamma=0.7)
    assert ag.e3.config.e3_aggregation_gamma == 0.7
    assert a0 == a1


# ------------------------------------------------------------------ W4-02 parity
def test_w402_on_equals_n3_disc05_rescoring_on_recorded_pool(fx):
    e3 = _e3(fx)
    bt = _native_pool(fx)
    assert bt.actions.shape[0] == 32 and len(bt.world_states) == 31
    J, Lmax = _probe_depth_scores(e3, bt)
    oracle = _probe_disc(J, Lmax, 0.5)
    full = J[None]
    on = _on(e3, bt).reshape(-1).double().numpy()
    scale = max(1.0, float(np.abs(oracle).max()))
    assert float(np.abs(on - oracle).max()) / scale < 1e-6
    # not vacuous: DISC_0.5 is a different ranking from the full read on this pool
    assert float(np.abs(oracle - full).max()) / scale > 1e-2
    assert int(np.argmin(oracle)) != int(np.argmin(full)) or \
        CA.spearman(oracle, full) < 0.95


def test_w402_gamma_one_telescopes_to_full_and_bad_gamma_raises(fx):
    e3 = _e3(fx)
    bt = _native_pool(fx)
    with torch.no_grad():
        full = e3.score_trajectory(bt)
    assert torch.equal(_on(e3, bt, gamma=1.0), full)
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            _on(e3, bt, gamma=bad)
    assert e3._score_depth_limit is None  # the try/finally restored it


def test_w402_on_is_differentiable_and_restores_depth_limit(fx):
    e3 = _e3(fx)
    bt = _native_pool(fx)
    ws = [w.clone().requires_grad_(True) for w in bt.world_states]
    bt2 = Trajectory(states=bt.states, actions=bt.actions, world_states=ws)
    e3.config.use_e3_discounted_aggregation = True
    s = e3.score_trajectory(bt2).sum()
    s.backward()
    assert ws[1].grad is not None and float(ws[1].grad.abs().sum()) > 0
    assert e3._score_depth_limit is None


# ------------------------------------------------------------------ W4-03 SD-081 contrast
def _spy(e3):
    calls = []
    orig = e3.score_trajectory

    def spy(traj, **kw):
        out = orig(traj, **kw)
        calls.append((e3._score_depth_limit, out.detach().clone()))
        return out
    e3.score_trajectory = spy
    return calls


def _select_with_dualsystem(fx, on):
    e3 = _e3(fx, use_dualsystem_arbitration=True)
    e3._running_variance = 0.0
    if on:
        e3.config.use_e3_discounted_aggregation = True
    bt = _native_pool(fx)
    cands = [Trajectory(states=[s[k:k + 1] for s in bt.states], actions=bt.actions[k:k + 1],
                        world_states=[w[k:k + 1] for w in bt.world_states]) for k in range(8)]
    calls = _spy(e3)
    with torch.no_grad():
        res = e3.select(cands, temperature=1.0, habit_uncertainty=0.5, habit_uncertainty_source="test")
    return calls, res, e3


def test_w403_habit_read_unchanged_on_vs_off(fx):
    habit_depth = max(2, int(E3Config().dualsystem_habit_depth))
    c_off, _r0, e_off = _select_with_dualsystem(fx, on=False)
    c_on, _r1, e_on = _select_with_dualsystem(fx, on=True)
    h_off = [o for d, o in c_off if d == habit_depth]
    # ON: the habit loop's calls are the depth-2 calls made while the arbitration runs, i.e.
    # the LAST 8 depth-2 calls (the planned reads' own depth-2 calls come first).
    h_on = [o for d, o in c_on if d == habit_depth][-8:]
    assert len(h_off) == 8 and len(h_on) == 8
    for a, b in zip(h_off, h_on):
        assert torch.equal(a, b)
    assert e_on.last_arbitration is not None and not e_on.last_arbitration["degenerate"]
    # and the planned vector did change (ON is not a no-op on the planned side)
    assert e_on.last_arbitration["planned_score_range"] != e_off.last_arbitration["planned_score_range"]


def test_w403_planned_reads_beyond_habit_depth(fx):
    e3 = _e3(fx)
    habit_depth = max(2, int(e3.config.dualsystem_habit_depth))
    bt = _native_pool(fx)
    ws = [w.clone() for w in bt.world_states]
    ws[5][0] = ws[5][0] + 0.5  # candidate 0, world step 5 (> habit depth)
    bt_p = Trajectory(states=bt.states, actions=bt.actions, world_states=ws)
    on0, on1 = _on(e3, bt), _on(e3, bt_p)
    assert float((on0[0] - on1[0]).abs()) > 1e-4
    assert torch.equal(on0[1:], on1[1:])
    with torch.no_grad():
        e3._score_depth_limit = habit_depth
        h0, h1 = e3.score_trajectory(bt), e3.score_trajectory(bt_p)
        e3._score_depth_limit = None
    assert torch.equal(h0, h1)


def test_w403_planned_ranking_is_not_the_habit_ranking(fx):
    """SD-081 P1 needs planned depth > habit depth; W4-03b above shows the reach. Here: the
    planned ranking of the recorded native pool is not the habit ranking.

    Measured (build record sec 4), NOT gated: with residue_field None (this standalone E3)
    the contrast is THIN -- Spearman(DISC_0.5, habit) 0.94 with the same argmin on this pool,
    and the scaffold-pool pick differs from the habit pick at only 2/41 states. N3's 40-65%
    pick disagreement was measured with the probe agent's residue field (Phi_R ~ 39 J units
    here), i.e. the deep contrast lives mostly in the residue term. Gamma 0.5 has effective
    depth ~2 by design; a larger contrast is a gamma / residue question, not this contract.
    """
    e3 = _e3(fx)
    habit_depth = max(2, int(e3.config.dualsystem_habit_depth))
    bt = _native_pool(fx)
    on = _on(e3, bt).reshape(-1).double().numpy()
    with torch.no_grad():
        e3._score_depth_limit = habit_depth
        h = e3.score_trajectory(bt).reshape(-1).double().numpy()
        e3._score_depth_limit = None
    rho = CA.spearman(on, h)
    assert rho is not None and rho < 0.99, rho
    # the planned read is closer to the habit read than FULL is (effective depth ~2) --
    # pinned so a regression to the full read (rho 0.62 here) is visible.
    with torch.no_grad():
        full = e3.score_trajectory(bt).reshape(-1).double().numpy()
    assert rho > CA.spearman(full, h)


# ------------------------------------------------------------------ W4-04 reduced gate (a)
def _gate_a(fx, on):
    e3 = _e3(fx)
    rho = {}
    n_inf = {}
    for head in ("REAL", "SHUF"):
        e2 = _e2(fx, head)
        preds, truths = [], []
        for i in range(fx["meta"]["n_states"]):
            bt = _scaffold(e2, fx, i)
            if on:
                p = _on(e3, bt)
            else:
                with torch.no_grad():
                    p = e3.score_trajectory(bt)
            preds.append(p.reshape(-1).double().numpy())
            truths.append(_jtrue(e3, fx, i))
        q = CA.e3_choice_quality({"agg": preds}, truths, truth_lower_is_better=True)
        rho[head] = q["arms"]["agg"]["spearman_mean"]
        n_inf[head] = q["n_informative"]
    return rho, n_inf


def test_w404_reduced_gate_a_on_passes_full_read_fails(fx):
    rho_on, n_on = _gate_a(fx, on=True)
    rho_full, _ = _gate_a(fx, on=False)
    assert min(n_on.values()) >= 10
    d_on = rho_on["REAL"] - rho_on["SHUF"]
    d_full = rho_full["REAL"] - rho_full["SHUF"]
    # the W4 aggregation separates the correctly mapped head from the permuted twin ...
    assert d_on > 0.15, (rho_on, d_on)
    # ... where the pre-W4 full-horizon read does not (dataset-discrimination canary).
    assert d_full <= 0.15, (rho_full, d_full)


# ------------------------------------------------------------------ W4-05 plumbing
def test_w405_knobs_plumb_and_on_agent_acts():
    acts, ag = _run_agent(13, use_e3_discounted_aggregation=True, e3_aggregation_gamma=0.5)
    assert ag.e3.config.use_e3_discounted_aggregation is True
    assert ag.e3.config.e3_aggregation_gamma == 0.5
    assert len(acts) == 12 and all(0 <= c < 5 for c in acts)
