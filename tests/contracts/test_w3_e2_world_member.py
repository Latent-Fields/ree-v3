"""Contracts for the W3 E2-world waking-trainer member (default OFF; branch integration/coupled-loop-repair).

Plan of record: REE_assembly evidence/planning/coupled_loop_repair_campaign_plan.md section 3
W3 (and A1 draft sec 5.4 E7 / sec 6.7 O15). Build record:
evidence/planning/w3_e2_world_member_build_20260925.md. Build: ree_core/utils/waking_trainer.py
``E2WorldMember`` (+ ``WakingTrainer.on_sense`` / ``set_e2_world_source``, the per-member
``grad_clip`` / ``updates_per_step`` hooks), the raw-obs hook in ``REEAgent.sense`` and the
``waking_trainer_e2_world_*`` knobs.

  W3-01 OFF: knob default False; nothing constructed; a default-config rollout never reaches
        the trainer or its sense hook; an ON trainer without the knob holds the T1 member set.
  W3-02 recording is inert: e2_world ON with a never-firing cadence (raw obs recorded, no
        update) is byte-identical to OFF over a multi-episode rollout.
  W3-03 every e2_world update (re-encode included) leaves global torch/numpy/python RNG
        identical; the spy is not blind to a leaky update.
  W3-04 reach (D1): group == e2.world_transition + e2.world_action_encoder; guard PASS; every
        tensor moved; NO leak (the re-encode runs under no_grad, so no gradient reaches the
        encoder).
  W3-05 not blind: a disconnected loss RAISES (guard armed) / fails the moved-check (disarmed).
  W3-06 FROZEN: the retained babbling set survives N on-policy steps (on-policy buffer
        evicting throughout) byte-identical; the comparison is not blind.
  W3-07 replay mix: pure retained before the on-policy buffer holds a batch, then exactly
        round(B * 0.25) retained per batch; pure on-policy with no retained set.
  W3-08 re-encode: (a) episode-start windows reproduce the live z_world exactly (<= 1e-5);
        (b) truncated windows at the auto window are within 1e-3 of it, and a window of 0 is
        not (canary); (c) re-encoding tracks the CURRENT read path (an encoder change
        invalidates the cache) and a frozen encoder is served from the cache.
  W3-09 A1 O15: export_retained -> append_external round-trips byte-identically into a fresh
        agent (source 'external'), re-encodes to the same z and trains; schedule_external
        releases each record at its logged step index.
  W3-10 the L2R bar (plan W3 gate (a), disc4_h1 >= 0.47 and k == 10) on a small FIXED
        dataset: W2a babbling + a monostrategy on-policy stream with 25% retained replay
        meets it; the shuffled-action twin (FIXED class relabelling of the retained
        actions, plan Decision log 14:19Z (3)) does NOT.
  W3-11 knobs plumb through REEConfig.from_dims; bad values raise; a TRAIN-mode rollout
        with the member ON has no autograd error.

Scope: reach, isolation, buffer semantics and the member-level L2R bar (D1). No D2/D3 claim.
CPU-small: 8x8 grid, world_dim = self_dim = 32 (deployed). W3-10 is the slow one (~30 s).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments._harness import StepHarness
from experiments._lib import coupled_acceptance as CA
from ree_core.agent import REEAgent
from ree_core.developmental.structured_babbling import StructuredBabbler
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils import waking_trainer as wt_mod
from ree_core.utils.config import REEConfig
from ree_core.utils.grad_reach_guard import FAIL, PASS

SEED = 7
TICKS = 40
EP_LEN = 15
DIM = 32

W3_ON = dict(waking_trainer_enabled=True, waking_trainer_every_k=1,
             waking_trainer_batch_size=4, waking_trainer_guard_min_steps=8,
             waking_trainer_e2_world_enabled=True, waking_trainer_e2_world_batch_size=4)


def _env(seed: int = SEED, size: int = 8) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=size, num_hazards=3, num_resources=2, hazard_harm=0.5,
                             seed=seed)


def _cfg(alpha_world: float = 0.3, **flags) -> REEConfig:
    env = _env()
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=DIM, world_dim=DIM, alpha_world=alpha_world,
        **flags)


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _rollout(cfg: REEConfig, ticks: int = TICKS, seed: int = SEED, train_mode: bool = False,
             ep_len: int = EP_LEN):
    _seed_all(seed)
    env = _env(seed)
    agent = REEAgent(cfg)
    harness = StepHarness(agent, env, train_mode=train_mode, seed=seed)
    actions, zworld, episodes = [], [], 0
    while len(actions) < ticks:
        episodes += 1
        for r in harness.run_episode(max_steps=min(ep_len, ticks - len(actions))):
            actions.append(int(r.action.argmax().item()))
            zworld.append(r.latent.z_world.detach().clone())
    return {
        "agent": agent, "actions": actions, "zworld": zworld, "episodes": episodes,
        "state": {k: v.detach().clone() for k, v in agent.state_dict().items()},
        "torch_rng": torch.get_rng_state().clone(),
        "np_rng": np.random.get_state(),
        "py_rng": random.getstate(),
    }


def _differences(a, b):
    diffs = []
    if a["actions"] != b["actions"]:
        diffs.append("actions")
    if len(a["zworld"]) != len(b["zworld"]) or any(
            not torch.equal(x, y) for x, y in zip(a["zworld"], b["zworld"])):
        diffs.append("z_world")
    if set(a["state"]) != set(b["state"]):
        diffs.append("state_dict keys")
    else:
        diffs += ["param:" + k for k in a["state"] if not torch.equal(a["state"][k], b["state"][k])]
    if not torch.equal(a["torch_rng"], b["torch_rng"]):
        diffs.append("torch_rng")
    na, nb = a["np_rng"], b["np_rng"]
    if na[0] != nb[0] or not np.array_equal(na[1], nb[1]) or tuple(na[2:]) != tuple(nb[2:]):
        diffs.append("np_rng")
    if a["py_rng"] != b["py_rng"]:
        diffs.append("py_rng")
    return diffs


def _obs_args(obs):
    return dict(obs_harm=obs.get("harm_obs"), obs_harm_a=obs.get("harm_obs_a"),
                obs_harm_history=obs.get("harm_history"))


def _drive(agent, n, policy, env_seed0, ep_len=50, update=True):
    """Scripted waking loop: sense -> executed action -> env.step -> waking-trainer step."""
    k = env_seed0
    env = _env(k)
    _f, obs = env.reset()
    agent.reset()
    t = 0
    for _ in range(n):
        agent.sense(torch.as_tensor(obs["body_state"]).float(),
                    torch.as_tensor(obs["world_state"]).float(), **_obs_args(obs))
        c = int(policy())
        agent.record_executed_action(F.one_hot(torch.tensor([c]), 5).float())
        _f, h, done, _i, obs = env.step(c)
        t += 1
        if update:
            agent.update_residue(float(h))
        else:
            agent.waking_trainer.on_waking_step(float(h))
        if done or t >= ep_len:
            k += 1
            env = _env(k)
            _f, obs = env.reset()
            agent.reset()
            t = 0


def _content(rec):
    """The stored (non-cache) content of one record, for byte comparisons."""
    obs = [tuple(None if x is None else x.clone() for x in o) for o in rec["obs"]]
    pa = [None if x is None else x.clone() for x in rec["prev_a"]]
    return obs, pa, rec["a"].clone(), rec["step"], rec["source"], rec["seg_start"]


def _same_content(x, y):
    def eq(u, v):
        if u is None or v is None:
            return u is None and v is None
        return torch.equal(u, v)
    return (len(x[0]) == len(y[0]) and all(
        all(eq(u, v) for u, v in zip(o1, o2)) for o1, o2 in zip(x[0], y[0]))
        and len(x[1]) == len(y[1]) and all(eq(u, v) for u, v in zip(x[1], y[1]))
        and torch.equal(x[2], y[2]) and x[3:] == y[3:])


# --- W3-01 / W3-02: default OFF, recording inert -------------------------------------------

def test_w3_01_off_nothing_constructed_and_sense_hook_absent(monkeypatch):
    assert REEConfig().waking_trainer_e2_world_enabled is False

    def _boom(*a, **k):
        raise AssertionError("W3 path touched at default config")

    monkeypatch.setattr(wt_mod.WakingTrainer, "__init__", _boom)
    monkeypatch.setattr(wt_mod.WakingTrainer, "on_sense", _boom)
    monkeypatch.setattr(wt_mod.E2WorldMember, "__init__", _boom)
    off = _rollout(_cfg())
    assert off["episodes"] >= 3 and off["agent"].waking_trainer is None


def test_w3_01b_on_trainer_without_knob_keeps_t1_member_set(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("E2WorldMember constructed with its knob OFF")

    monkeypatch.setattr(wt_mod.E2WorldMember, "__init__", _boom)
    tr = REEAgent(_cfg(waking_trainer_enabled=True)).waking_trainer
    assert list(tr.members) == ["harm_eval"]
    with pytest.raises(ValueError):
        tr.set_e2_world_source("babble")


def test_w3_02_recording_raw_obs_is_byte_identical_to_off():
    off = _rollout(_cfg())
    on = _rollout(_cfg(**dict(W3_ON, waking_trainer_every_k=10 ** 6)))
    tr = on["agent"].waking_trainer
    m = tr.members["e2_world"]
    assert all(v == 0 for v in tr.steps.values())
    assert len(m._on_policy) == TICKS - on["episodes"]        # it recorded, per episode pairs
    assert m._retained == []
    assert _differences(off, on) == []


# --- W3-03: RNG neutrality -------------------------------------------------------------------

def _rng_spy(monkeypatch):
    seen = {}
    orig = wt_mod.WakingTrainer._update

    def _spy(self, name, member):
        before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        out = orig(self, name, member)
        after = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        seen.setdefault(name, []).append(
            torch.equal(before[0], after[0])
            and np.array_equal(before[1][1], after[1][1])
            and before[1][2:] == after[1][2:]
            and before[2] == after[2])
        return out

    monkeypatch.setattr(wt_mod.WakingTrainer, "_update", _spy)
    return seen


def test_w3_03_every_e2_world_update_is_rng_neutral(monkeypatch):
    seen = _rng_spy(monkeypatch)
    on = _rollout(_cfg(**W3_ON))
    tr = on["agent"].waking_trainer
    assert tr.steps["e2_world"] >= 8
    assert tr.members["e2_world"].n_reencoded > 0              # the re-encode really ran
    assert seen.get("e2_world") and all(seen["e2_world"])


def test_w3_03b_rng_spy_is_not_blind(monkeypatch):
    orig_loss = wt_mod.E2WorldMember.loss

    def _leaky(self, agent):
        out = orig_loss(self, agent)
        np.random.rand()                                        # a GLOBAL numpy draw
        return out

    monkeypatch.setattr(wt_mod.E2WorldMember, "loss", _leaky)
    orig_update = wt_mod.WakingTrainer._update

    def _unguarded(self, name, member):                         # bypass the trainer's restore
        before = np.random.get_state()
        out = orig_update(self, name, member)
        np.random.set_state(before)
        np.random.rand()
        return out

    monkeypatch.setattr(wt_mod.WakingTrainer, "_update", _unguarded)
    a = _rollout(_cfg(**W3_ON), ticks=12)
    b = _rollout(_cfg(**dict(W3_ON, waking_trainer_every_k=10 ** 6)), ticks=12)
    assert "np_rng" in _differences(a, b)


# --- W3-04 / W3-05: reach, not blind --------------------------------------------------------

def _run_with_group_snapshot(cfg, train_mode=False):
    snaps = {}
    orig_register = wt_mod.WakingTrainer.register

    def _register(self, member):
        snaps[member.name] = {n: p.detach().clone() for n, p in member.named_parameters()}
        orig_register(self, member)

    wt_mod.WakingTrainer.register = _register
    try:
        out = _rollout(cfg, train_mode=train_mode)
    finally:
        wt_mod.WakingTrainer.register = orig_register
    return out, snaps


def test_w3_04_group_reached_moved_guard_pass_no_leak():
    out, snaps = _run_with_group_snapshot(_cfg(**W3_ON))
    agent = out["agent"]
    tr = agent.waking_trainer
    expected = sorted(
        ["e2.world_transition." + n for n, _ in agent.e2.world_transition.named_parameters()]
        + ["e2.world_action_encoder." + n
           for n, _ in agent.e2.world_action_encoder.named_parameters()])
    assert sorted(tr.group_names("e2_world")) == expected and len(expected) == 6
    res = tr.guard_results["e2_world"]
    assert res.status == PASS, res.summary()
    assert res.n_checked == len(expected) and res.leaked_names == []
    moved = {n: not torch.equal(p.detach(), snaps["e2_world"][n])
             for n, p in tr.members["e2_world"].named_parameters()}
    assert all(moved.values()), moved


def _disconnected(self, agent):
    batch = self.replay_batch()
    if batch is None:
        return None
    z0, acts, z1 = batch
    leaf = torch.zeros((), requires_grad=True)
    return F.mse_loss(agent.e2.world_forward(z0, acts).detach(), z1) + leaf * 0.0


def test_w3_05a_guard_raises_on_a_disconnected_member(monkeypatch):
    monkeypatch.setattr(wt_mod.E2WorldMember, "loss", _disconnected)
    with pytest.raises(wt_mod.WakingTrainerReachError) as exc:
        _rollout(_cfg(**W3_ON))
    msg = str(exc.value)
    assert "'e2_world'" in msg and "FAILED" in msg and FAIL in msg


def test_w3_05b_moved_check_fails_on_a_disconnected_member(monkeypatch):
    monkeypatch.setattr(wt_mod.E2WorldMember, "loss", _disconnected)
    out, snaps = _run_with_group_snapshot(_cfg(**dict(W3_ON, waking_trainer_guard_min_steps=0)))
    tr = out["agent"].waking_trainer
    assert tr.steps["e2_world"] >= 8
    moved = [not torch.equal(p.detach(), snaps["e2_world"][n])
             for n, p in tr.members["e2_world"].named_parameters()]
    assert not any(moved), "the moved-check would PASS a disconnected e2_world member"


# --- W3-06 / W3-07: FROZEN retained set and the replay mix ----------------------------------

def _babble_agent(n_babble=60, buffer_max=10, batch=4, **extra):
    _seed_all(SEED)
    agent = REEAgent(_cfg(**dict(W3_ON, waking_trainer_buffer_max=buffer_max,
                                 waking_trainer_e2_world_batch_size=batch, **extra)))
    tr = agent.waking_trainer
    bab = StructuredBabbler(5, 4, seed=SEED)
    tr.set_e2_world_source("babble")
    _drive(agent, n_babble, bab.next_class, env_seed0=100)
    tr.set_e2_world_source("on_policy")
    return agent, tr, tr.members["e2_world"]


def test_w3_06_frozen_retained_set_survives_on_policy_steps_byte_identical():
    agent, tr, m = _babble_agent()
    assert len(m._retained) >= 50 and all(r["source"] == "babble" for r in m._retained)
    before = [_content(r) for r in m._retained]
    first_on_policy = []
    g = np.random.default_rng(1)

    def mono():
        return 2 if g.random() < 0.9 else int(g.integers(0, 5))

    for block in range(4):                                     # 4 x 30 on-policy steps
        _drive(agent, 30, mono, env_seed0=200 + block)
        first_on_policy.append(m._on_policy[0]["step"])
    assert len(m._on_policy) == 10                             # deque full, evicting
    assert first_on_policy == sorted(first_on_policy) and len(set(first_on_policy)) == 4
    assert tr.steps["e2_world"] >= 100 and m.n_drawn_retained > 0
    after = [_content(r) for r in m._retained]
    assert len(after) == len(before)
    assert all(_same_content(x, y) for x, y in zip(before, after))
    # Not blind: a one-element in-place change of one retained tensor is detected.
    m._retained[3]["obs"][-1][1][0, 0] += 1.0
    assert not _same_content(before[3], _content(m._retained[3]))


def test_w3_06b_retained_capacity_counts_drops_never_evicts():
    agent, tr, m = _babble_agent(n_babble=40, waking_trainer_e2_world_retained_max=20)
    assert len(m._retained) == 20 and m.retained_dropped > 0
    assert [r["step"] for r in m._retained] == sorted(r["step"] for r in m._retained)


def test_w3_07_replay_mix_is_25_percent_once_both_buffers_fill():
    _seed_all(SEED)
    agent = REEAgent(_cfg(**dict(W3_ON, waking_trainer_every_k=10 ** 6,
                                 waking_trainer_e2_world_batch_size=16)))
    tr = agent.waking_trainer
    m = tr.members["e2_world"]
    tr.set_e2_world_source("babble")
    _drive(agent, 40, StructuredBabbler(5, 4, seed=1).next_class, env_seed0=10)
    for _ in range(5):
        tr._update("e2_world", m)
    assert (m.n_drawn_retained, m.n_drawn_on_policy) == (80, 0)   # pure retained epoch
    tr.set_e2_world_source("on_policy")
    _drive(agent, 40, lambda: 1, env_seed0=20)
    r0, o0 = m.n_drawn_retained, m.n_drawn_on_policy
    for _ in range(20):
        tr._update("e2_world", m)
    dr, do = m.n_drawn_retained - r0, m.n_drawn_on_policy - o0
    assert dr + do == 20 * 16 and abs(dr / (dr + do) - 0.25) < 1e-9
    # no retained set -> pure on-policy
    _seed_all(SEED)
    agent2 = REEAgent(_cfg(**dict(W3_ON, waking_trainer_every_k=10 ** 6,
                                  waking_trainer_e2_world_batch_size=16)))
    m2 = agent2.waking_trainer.members["e2_world"]
    _drive(agent2, 40, lambda: 1, env_seed0=20)
    for _ in range(5):
        agent2.waking_trainer._update("e2_world", m2)
    assert (m2.n_drawn_retained, m2.n_drawn_on_policy) == (0, 80)


# --- W3-08: re-encode fidelity and tracking --------------------------------------------------

def _live_vs_reencoded(m, recs):
    for r in recs:
        r.pop("_zc", None)
    z0, z1 = m._reencode(recs)
    lz1 = torch.cat([r["z_live"][1] for r in recs])
    lz0 = torch.cat([r["z_live"][0] for r in recs])
    return torch.maximum((z0 - lz0).abs().max(dim=1).values, (z1 - lz1).abs().max(dim=1).values)


def test_w3_08a_b_reencode_reproduces_the_live_latent():
    _seed_all(SEED)
    agent = REEAgent(_cfg(**dict(W3_ON, waking_trainer_every_k=10 ** 6)))
    m = agent.waking_trainer.members["e2_world"]
    assert m.reencode_window == wt_mod.auto_reencode_window(0.3) == 26
    g = np.random.default_rng(3)
    _drive(agent, 120, lambda: int(g.integers(0, 5)), env_seed0=30, ep_len=60)
    recs = list(m._on_policy)
    seg = torch.tensor([r["seg_start"] for r in recs])
    assert seg.any() and (~seg).any()
    d = _live_vs_reencoded(m, recs)
    assert float(d[seg].max()) <= 1e-5
    assert float(d[~seg].max()) <= 1e-3
    # canary: a zero-step warm-up window (no EMA history) is visibly wrong on the same data
    m.reencode_window = 0
    short = []
    for r in recs:
        if not r["seg_start"]:
            rr = dict(r)
            rr["obs"], rr["prev_a"] = r["obs"][-2:], (None,) + tuple(r["prev_a"][-1:])
            short.append(rr)
    assert float(_live_vs_reencoded(m, short).max()) > 1e-3


def test_w3_08c_reencode_tracks_the_current_read_path_and_caches_when_frozen():
    agent, tr, m = _babble_agent(n_babble=30)
    recs = m._retained[:8]
    z0a, _ = m._reencode(recs)
    n_enc = m.n_reencoded
    z0b, _ = m._reencode(recs)
    assert torch.equal(z0a, z0b) and m.n_reencoded == n_enc and m.n_cache_hits >= 8
    with torch.no_grad():
        next(agent.world_obs_encoder.parameters()).mul_(1.5)   # the read path changes
    z0c, _ = m._reencode(recs)
    assert m.n_reencoded == n_enc + 8
    assert not torch.allclose(z0a, z0c)
    for r in recs:
        r.pop("_zc", None)
    z0d, _ = m._reencode(recs)
    assert torch.equal(z0c, z0d)                               # fresh == invalidated-cache path


# --- W3-09: A1 O15 external transitions ------------------------------------------------------

def test_w3_09_export_append_external_round_trip_and_training():
    donor, _tr, dm = _babble_agent(n_babble=40)
    log = dm.export_retained()
    assert log and all("_zc" not in r and isinstance(r["step"], int) for r in log)
    assert [r["step"] for r in log] == sorted(r["step"] for r in log)
    _seed_all(SEED)
    recv = REEAgent(_cfg(**dict(W3_ON, waking_trainer_every_k=10 ** 6)))
    rm = recv.waking_trainer.members["e2_world"]
    assert rm.append_external(log) == len(log)
    assert all(r["source"] == "external" for r in rm._retained)
    assert all(_same_content(_content(dict(a, source="external")), _content(b))
               for a, b in zip(log, rm._retained))
    # same seed -> same encoder weights -> the receiving member re-encodes to the donor's z
    zd, _ = dm._reencode([dict(r) for r in log[:10]])
    zr, _ = rm._reencode(rm._retained[:10])
    assert torch.allclose(zd, zr, atol=1e-6)
    before = {n: p.detach().clone() for n, p in rm.named_parameters()}
    for _ in range(3):
        recv.waking_trainer._update("e2_world", rm)
    assert rm.n_drawn_retained == 3 * rm.batch_size
    assert any(not torch.equal(p.detach(), before[n]) for n, p in rm.named_parameters())


def test_w3_09b_schedule_external_releases_at_the_logged_step():
    donor, _tr, dm = _babble_agent(n_babble=40)
    log = dm.export_retained()
    _seed_all(SEED)
    recv = REEAgent(_cfg(**dict(W3_ON, waking_trainer_every_k=10 ** 6)))
    rm = recv.waking_trainer.members["e2_world"]
    rm.schedule_external(log)
    assert rm._retained == []
    released = []
    _drive(recv, 40, lambda: 4, env_seed0=500)
    for r in rm._retained:
        released.append(r["step"])
    assert released == [r["step"] for r in log if r["step"] <= rm.n_observed]
    assert len(released) >= len(log) - 2
    with pytest.raises(ValueError):
        rm.schedule_external([dict(log[0], step=None)])


# --- W3-10: the L2R bar on a small fixed dataset ---------------------------------------------

L2R_BAR = 0.47
PERM = [1, 2, 3, 4, 0]


def _l2r_head(shuffle: bool):
    """W2a babbling (2400) -> 3000 member updates -> 1200 monostrategy on-policy steps with
    8 member updates per step at the 25% retained mix; returns the held-out discrimination."""
    S = 3
    _seed_all(S)
    agent = REEAgent(_cfg(alpha_world=0.9, **dict(
        W3_ON, waking_trainer_every_k=10 ** 9, waking_trainer_batch_size=16,
        waking_trainer_e2_world_batch_size=32)))
    agent.eval()
    tr = agent.waking_trainer
    m = tr.members["e2_world"]

    def env_(k):
        return _env(S * 1000 + k)

    def loop(n, policy, k0, ups):
        k = k0
        env = env_(k)
        _f, obs = env.reset()
        agent.reset()
        t = 0
        for _ in range(n):
            agent.sense(torch.as_tensor(obs["body_state"]).float(),
                        torch.as_tensor(obs["world_state"]).float(), **_obs_args(obs))
            c = int(policy())
            agent.record_executed_action(F.one_hot(torch.tensor([c]), 5).float())
            _f, h, done, _i, obs = env.step(c)
            t += 1
            agent.update_residue(float(h))
            if ups and len(m._on_policy) >= m.batch_size:
                for _u in range(ups):
                    tr._update("e2_world", m)
            if done or t >= 50:
                k += 1
                env = env_(k)
                _f, obs = env.reset()
                agent.reset()
                t = 0

    tr.set_e2_world_source("babble")
    loop(2400, StructuredBabbler(5, 4, seed=S).next_class, 0, 0)
    if shuffle:
        for r in m._retained:
            r["a"] = F.one_hot(torch.tensor([PERM[int(r["a"].argmax())]]), 5).float()
    for _ in range(3000):
        tr._update("e2_world", m)
    tr.set_e2_world_source("on_policy")
    g = np.random.default_rng(S + 5)
    c0 = int(g.integers(0, 4))
    loop(1200, lambda: c0 if g.random() < 0.9 else int(g.integers(0, 5)), 100, 8)
    te = CA.collect_uniform_random_episodes(agent, env_(900), 1500, seed=S + 991)
    res = CA.action_discrimination(CA.e2_world_predictor(agent.e2), te, action_dim=5,
                                   max_starts=150, seed=S, bar_disc4_h1=L2R_BAR)
    return res, tr, m


def test_w3_10_l2r_bar_met_by_member_and_not_by_shuffled_twin():
    real, tr, m = _l2r_head(shuffle=False)
    assert tr.guard_results["e2_world"].status == PASS
    assert m.n_drawn_retained > 0 and m.n_drawn_on_policy > 0
    assert real["verdict"] == PASS, real
    assert real["disc4_h1"] >= L2R_BAR and real["k"] == 10, real
    twin, _tr, _m = _l2r_head(shuffle=True)
    assert twin["verdict"] == FAIL, twin
    assert not (twin["disc4_h1"] >= L2R_BAR and twin["k"] == 10), twin


# --- W3-11: knobs, bad values, retained graph -------------------------------------------------

def test_w3_11_knobs_plumb_through_from_dims_and_validate():
    cfg = _cfg(waking_trainer_enabled=True, waking_trainer_e2_world_enabled=True,
               waking_trainer_e2_world_lr=2e-4, waking_trainer_e2_world_batch_size=8,
               waking_trainer_e2_world_replay_frac=0.5, waking_trainer_e2_world_retained_max=77,
               waking_trainer_e2_world_reencode_window=5,
               waking_trainer_e2_world_replay_latent="stored",
               waking_trainer_e2_world_objective="infonce",
               waking_trainer_e2_world_grad_clip=0.5,
               waking_trainer_e2_world_updates_per_step=3)
    m = REEAgent(cfg).waking_trainer.members["e2_world"]
    tr_opt = REEAgent(cfg).waking_trainer.optimizers["e2_world"]
    assert tr_opt.param_groups[0]["lr"] == 2e-4
    assert (m.batch_size, m.replay_frac, m._retained_max, m.reencode_window, m.replay_latent,
            m.objective, m.grad_clip, m.updates_per_step) == (
        8, 0.5, 77, 5, "stored", "infonce", 0.5, 3)
    for bad in (dict(waking_trainer_e2_world_objective="l1"),
                dict(waking_trainer_e2_world_replay_latent="z"),
                dict(waking_trainer_e2_world_replay_frac=1.5)):
        with pytest.raises(ValueError):
            REEAgent(_cfg(waking_trainer_enabled=True, waking_trainer_e2_world_enabled=True,
                          **bad))


def test_w3_11b_infonce_and_stored_modes_train():
    on = _rollout(_cfg(**dict(W3_ON, waking_trainer_e2_world_objective="infonce",
                              waking_trainer_e2_world_replay_latent="stored")))
    tr = on["agent"].waking_trainer
    assert tr.guard_results["e2_world"].status == PASS
    assert tr.members["e2_world"].n_reencoded == 0             # stored mode never re-encodes


def test_w3_11c_train_mode_rollout_with_member_on_has_no_autograd_error():
    on = _rollout(_cfg(**dict(W3_ON, waking_trainer_e1_enabled=True,
                              waking_trainer_e2_self_enabled=True)), train_mode=True)
    tr = on["agent"].waking_trainer
    assert {n: r.status for n, r in tr.guard_results.items()} == {
        "harm_eval": PASS, "e1": PASS, "e2_self": PASS, "e2_world": PASS}
