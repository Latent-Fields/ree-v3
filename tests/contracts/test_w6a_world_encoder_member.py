"""Contracts for the W6a world-encoder waking-trainer member (default OFF; branch integration/coupled-loop-repair).

Plan of record: REE_assembly evidence/planning/coupled_loop_repair_campaign_plan.md section 3
W-trainer (W6a, user decision Q4c; buffer-staleness requirement P8). Build record:
evidence/planning/w6a_world_encoder_member_build_20260925.md. Build:
ree_core/utils/waking_trainer_world_encoder.py ``WorldEncoderMember`` (SD-070 P0a objective
trained through the LIVE sense path, world_obs_encoder -> latent_stack.encode, the ZSelfP0
``_native_chain`` pattern), its registration + the ``clip_parameters`` /
``restore_outside_grads`` hooks in ree_core/utils/waking_trainer.py, and the
``waking_trainer_world_encoder_*`` knobs.

  W6a-01 OFF: knob default False; nothing constructed at defaults; an ON trainer without the
         knob holds exactly the C1 member set and never constructs the member.
  W6a-02 recording is inert: the member ON with a never-firing cadence (raw obs recorded,
         heads built, no update) is byte-identical to OFF over a multi-episode rollout --
         including the global RNG streams (head construction draws nothing global).
  W6a-03 reach (D1): group == world_obs_encoder + split_encoder.world_encoder +
         world_precision_logit + the trainer heads; guard PASS; every tensor moved; every
         update RNG-neutral; the G5 leak set is exactly the depth stack + z_self path (the
         deferred families) and their .grad is restored after each update.
  W6a-04 live sense path: under the member the SENSE read path moves (world_obs_encoder AND
         split_encoder.world_encoder) and the next sense() of a fixed observation changes;
         with the knob OFF (trainer ON, same rollout) none of those tensors moves.
  W6a-05 the chain IS the live latent: re-running a record's window through the member's
         chain reproduces the live sensed z_world (<= 1e-5 at episode-start windows,
         <= 1e-3 at the auto window; a 1-tick window is NOT within 1e-3 -- canary).
  W6a-06 W3 interaction: with the W3 E2-world member ON (replay_latent "reencode"), a W6a
         step invalidates W3's re-encode cache (the next re-encode is fresh and differs);
         with no W6a step the cache serves; the "stored" replay path returns the SAME
         stored z before and after a W6a step (stale by construction -- the N2 arm).
  W6a-07 not blind: a detached (disconnected) loss RAISES (guard armed) / fails the moved
         check (disarmed); the OLD defect -- SD-070's direct world_encoder(world_state)
         path, which bypasses world_obs_encoder -- FAILS the guard naming world_obs_encoder.
  W6a-08 knobs plumb through REEConfig.from_dims; a TRAIN-mode rollout with every trainer
         member ON (e1, e2_self, codec, e2_world, world_encoder) has no autograd error.

Scope: reach, isolation and the re-encode interaction (D1). No claim that the trained
encoder helps any consumer (that is probe N2 / A1). CPU-small: 8x8 grid, world_dim =
self_dim = 32 (deployed).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments._harness import StepHarness
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils import waking_trainer as wt_mod
from ree_core.utils import waking_trainer_world_encoder as wenc_mod
from ree_core.utils.config import REEConfig
from ree_core.utils.grad_reach_guard import FAIL, PASS

SEED = 11
TICKS = 40
EP_LEN = 15
DIM = 32

W6A_ON = dict(waking_trainer_enabled=True, waking_trainer_every_k=1,
              waking_trainer_batch_size=4, waking_trainer_guard_min_steps=8,
              waking_trainer_world_encoder_enabled=True,
              waking_trainer_world_encoder_batch_size=6)

# Families the sensed z_world reaches but the member deliberately does NOT step: the depth
# stack (gradient-reach census 940c690c9dd #13, DEFERRED) and, via top-down, the z_self
# path (owned by sd_zself_training_path).
LEAK_FAMILIES = (
    "latent_stack.beta_encoder.", "latent_stack.theta_encoder.", "latent_stack.delta_encoder.",
    "latent_stack.delta_to_theta.", "latent_stack.theta_to_beta.", "latent_stack.beta_to_split.",
    "latent_stack.split_encoder.world_topdown.", "latent_stack.split_encoder.self_topdown.",
    "latent_stack.split_encoder.self_encoder.", "latent_stack.split_encoder.self_precision_logit",
    "body_obs_encoder.",
)


def _env(seed: int = SEED, size: int = 8, hazards: int = 3) -> CausalGridWorldV2:
    if hazards == 0:     # SD-094: no self-contamination either, so episodes run long
        return CausalGridWorldV2(size=size, num_hazards=0, num_resources=2, hazard_harm=0.5,
                                 seed=seed, contamination_spread=0.0)
    return CausalGridWorldV2(size=size, num_hazards=hazards, num_resources=2, hazard_harm=0.5,
                             seed=seed)


def _cfg(alpha_world: float = 0.9, **flags) -> REEConfig:
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


def _drive(agent, n, policy, env_seed0, ep_len=30, update=False, live=None, hazards=3):
    """Scripted waking loop: sense -> executed action -> env.step -> trainer observe.
    ``update=False`` records only (the trainer's observe path, no optimizer step).
    ``live`` (a dict) collects the live sensed z_world per member step index."""
    k = env_seed0
    env = _env(k, hazards=hazards)
    _f, obs = env.reset()
    agent.reset()
    agent.notify_env_reset() if hasattr(agent, "notify_env_reset") else None
    t = 0
    tr = agent.waking_trainer
    for _ in range(n):
        lat = agent.sense(torch.as_tensor(obs["body_state"]).float(),
                          torch.as_tensor(obs["world_state"]).float(), **_obs_args(obs))
        c = int(policy())
        agent.record_executed_action(F.one_hot(torch.tensor([c]), 5).float())
        _f, h, done, _i, obs = env.step(c)
        t += 1
        if update:
            agent.update_residue(float(h))
        else:
            for m in tr.members.values():
                m.observe(agent, float(h))
        if live is not None and "world_encoder" in tr.members:
            live[tr.members["world_encoder"].n_observed] = lat.z_world.detach().clone()
        if done or t >= ep_len:
            k += 1
            env = _env(k, hazards=hazards)
            _f, obs = env.reset()
            agent.reset()
            agent.notify_env_reset() if hasattr(agent, "notify_env_reset") else None
            t = 0


def _cycle(seq):
    it = iter(int(x) for x in seq)
    return lambda: next(it)


# --- W6a-01 / W6a-02: default OFF, recording inert ------------------------------------------

def test_w6a_01_off_nothing_constructed(monkeypatch):
    assert REEConfig().waking_trainer_world_encoder_enabled is False

    def _boom(*a, **k):
        raise AssertionError("W6a member constructed at default config")

    monkeypatch.setattr(wenc_mod.WorldEncoderMember, "__init__", _boom)
    off = _rollout(_cfg())
    assert off["episodes"] >= 3 and off["agent"].waking_trainer is None
    tr = REEAgent(_cfg(waking_trainer_enabled=True)).waking_trainer
    assert list(tr.members) == ["harm_eval"]


def test_w6a_02_recording_is_byte_identical_to_off():
    off = _rollout(_cfg())
    on = _rollout(_cfg(**dict(W6A_ON, waking_trainer_every_k=10 ** 6)))
    tr = on["agent"].waking_trainer
    m = tr.members["world_encoder"]
    assert all(v == 0 for v in tr.steps.values())
    assert m.n_observed == TICKS and len(m._buf) == TICKS         # it recorded every tick
    assert _differences(off, on) == []


# --- W6a-03: reach, RNG neutrality, leak set -------------------------------------------------

def _run_with_group_snapshot(cfg, train_mode=False, ticks=TICKS):
    snaps = {}
    orig_register = wt_mod.WakingTrainer.register

    def _register(self, member):
        snaps[member.name] = {n: p.detach().clone() for n, p in member.named_parameters()}
        orig_register(self, member)

    wt_mod.WakingTrainer.register = _register
    try:
        out = _rollout(cfg, train_mode=train_mode, ticks=ticks)
    finally:
        wt_mod.WakingTrainer.register = orig_register
    return out, snaps


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


def test_w6a_03_group_reached_moved_guard_pass_rng_neutral_leak_pinned(monkeypatch):
    seen = _rng_spy(monkeypatch)
    out, snaps = _run_with_group_snapshot(_cfg(**W6A_ON))
    agent = out["agent"]
    tr = agent.waking_trainer
    m = tr.members["world_encoder"]
    se = agent.latent_stack.split_encoder
    enc_expected = sorted(
        ["world_obs_encoder." + n for n, _ in agent.world_obs_encoder.named_parameters()]
        + ["latent_stack.split_encoder.world_encoder." + n
           for n, _ in se.world_encoder.named_parameters()]
        + ["latent_stack.split_encoder.world_precision_logit"])
    assert sorted(n for n, _ in m.encoder_named_parameters()) == enc_expected
    assert len(enc_expected) == 7
    names = tr.group_names("world_encoder")
    assert len(names) == 7 + 10                                   # + 4 grounding + recon heads
    assert tr.steps["world_encoder"] >= 8
    res = tr.guard_results["world_encoder"]
    assert res.status == PASS, res.summary()
    assert res.n_checked == 17
    moved = {n: not torch.equal(p.detach(), snaps["world_encoder"][n])
             for n, p in m.named_parameters()}
    assert all(moved.values()), moved
    assert seen.get("world_encoder") and all(seen["world_encoder"])
    assert m.n_chain_encodes > 0
    leaks = res.leaked_names
    assert leaks and all(n.startswith(LEAK_FAMILIES) for n in leaks), leaks
    assert any(n.startswith("latent_stack.beta_encoder.") for n in leaks)
    # restored: no gradient is left accumulating on an un-stepped tensor
    own = {id(p) for _, p in m.named_parameters()}
    stale = [n for n, p in agent.named_parameters() if id(p) not in own and p.grad is not None]
    assert stale == [], stale


def test_w6a_03b_outside_grads_are_restored_not_contaminated():
    _seed_all(SEED)
    agent = REEAgent(_cfg(**dict(W6A_ON, waking_trainer_every_k=10 ** 6)))
    tr = agent.waking_trainer
    m = tr.members["world_encoder"]
    _drive(agent, 20, _cycle([0, 1, 2, 3, 4] * 4), env_seed0=40)
    beta_w = agent.latent_stack.beta_encoder.encoder[0].weight
    sentinel = torch.full_like(beta_w, 0.5)
    beta_w.grad = sentinel.clone()                                 # a driver's pending grad
    assert tr._update("world_encoder", m) is not None
    assert torch.equal(beta_w.grad, sentinel)                      # untouched by W6a's backward
    assert agent.body_obs_encoder[0].weight.grad is None


# --- W6a-04: the live sense path moves; OFF it does not ----------------------------------------

def _sense_probe(agent, obs):
    with torch.no_grad():
        enc = torch.cat([agent.body_obs_encoder(torch.as_tensor(obs["body_state"]).float()[None]),
                         agent.world_obs_encoder(torch.as_tensor(obs["world_state"]).float()[None])],
                        dim=-1)
        return agent.latent_stack.encode(
            enc, agent.latent_stack.init_state(batch_size=1, device=agent.device)).z_world.clone()


def _sense_path_tensors(agent):
    se = agent.latent_stack.split_encoder
    d = {"world_obs_encoder." + n: p for n, p in agent.world_obs_encoder.named_parameters()}
    d.update({"world_encoder." + n: p for n, p in se.world_encoder.named_parameters()})
    d["world_precision_logit"] = se.world_precision_logit
    return {k: v.detach().clone() for k, v in d.items()}


def test_w6a_04_sense_path_moves_under_member_and_not_when_off():
    _f, probe_obs = _env(99).reset()
    res = {}
    for label, flags in (("on", W6A_ON), ("off", dict(waking_trainer_enabled=True,
                                                     waking_trainer_batch_size=4))):
        _seed_all(SEED)
        agent = REEAgent(_cfg(**flags))
        before_t = _sense_path_tensors(agent)
        before_z = _sense_probe(agent, probe_obs)
        StepHarness(agent, _env(SEED), train_mode=False, seed=SEED).run_episode(max_steps=30)
        after_t = _sense_path_tensors(agent)
        res[label] = ({k: not torch.equal(before_t[k], after_t[k]) for k in before_t},
                      float((_sense_probe(agent, probe_obs) - before_z).abs().max()))
    moved_on, dz_on = res["on"]
    moved_off, dz_off = res["off"]
    assert all(moved_on.values()), moved_on
    assert moved_on["world_obs_encoder.0.weight"]                 # the pre-projection itself
    assert dz_on > 1e-4
    assert not any(moved_off.values()), moved_off
    assert dz_off == 0.0


# --- W6a-05: the chain reproduces the live latent ----------------------------------------------

def _chain_vs_live(alpha, window):
    _seed_all(SEED)
    agent = REEAgent(_cfg(alpha_world=alpha, **dict(
        W6A_ON, waking_trainer_every_k=10 ** 6, waking_trainer_world_encoder_window=window)))
    m = agent.waking_trainer.members["world_encoder"]
    live = {}
    # hazard-free env so episodes outlast the 26-tick auto window
    _drive(agent, 80, _cycle([2, 2, 1, 0, 3, 4] * 14), env_seed0=60, ep_len=40, live=live,
           hazards=0)
    full = [r for r in m._buf if len(r["obs"]) < m.window + 1]   # episode-start windows
    trunc = [r for r in m._buf if len(r["obs"]) == m.window + 1]
    out = []
    for recs in (full, trunc):
        if not recs:
            out.append(None)
            continue
        with torch.no_grad():
            z = m.sensed_z_world(recs)
        lz = torch.cat([live[r["step"]] for r in recs])
        out.append(float((z - lz).abs().max()))
    return m.window, out


def test_w6a_05_chain_reproduces_the_live_sensed_latent():
    w, (full, trunc) = _chain_vs_live(0.3, 0)
    assert w == 26
    assert full is not None and full <= 1e-5, full
    assert trunc is not None and trunc <= 1e-3, trunc
    # canary: a 1-tick window at alpha 0.3 is NOT the live latent
    w1, (_f1, trunc1) = _chain_vs_live(0.3, 1)
    assert w1 == 1 and trunc1 is not None and trunc1 > 1e-3, trunc1


# --- W6a-06: interaction with the W3 re-encode cache --------------------------------------------

def _w3_w6a_agent(replay_latent="reencode"):
    _seed_all(SEED)
    agent = REEAgent(_cfg(**dict(
        W6A_ON, waking_trainer_every_k=10 ** 6, waking_trainer_e2_world_enabled=True,
        waking_trainer_e2_world_batch_size=4,
        waking_trainer_e2_world_replay_latent=replay_latent)))
    tr = agent.waking_trainer
    _drive(agent, 24, _cycle([0, 1, 2, 3, 4] * 5), env_seed0=80)
    return agent, tr, tr.members["e2_world"], tr.members["world_encoder"]


def test_w6a_06_w6a_step_invalidates_w3_reencode_cache():
    agent, tr, m3, m6 = _w3_w6a_agent()
    assert list(tr.members)[-1] == "world_encoder"                # steps AFTER e2_world
    recs = list(m3._on_policy)[:6]
    z0a, z1a = m3._reencode(recs)
    n0, h0 = m3.n_reencoded, m3.n_cache_hits
    z0b, _ = m3._reencode(recs)                                    # frozen: served from cache
    assert (m3.n_reencoded, m3.n_cache_hits) == (n0, h0 + len(recs))
    assert torch.equal(z0a, z0b)
    key0 = m3.read_path_key()
    assert tr._update("world_encoder", m6) is not None
    assert m3.read_path_key() != key0
    z0c, z1c = m3._reencode(recs)                                  # fresh after the W6a step
    assert m3.n_reencoded == n0 + len(recs) and m3.n_cache_hits == h0 + len(recs)
    assert float((z0c - z0a).abs().max()) > 1e-6
    assert float((z1c - z1a).abs().max()) > 1e-6


def test_w6a_06b_stored_replay_does_not_track_the_encoder():
    agent, tr, m3, m6 = _w3_w6a_agent(replay_latent="stored")
    recs = list(m3._on_policy)[:6]
    stored0 = torch.cat([r["z_live"][0] for r in recs]).clone()
    fresh0, _ = m3._reencode(recs)
    assert tr._update("world_encoder", m6) is not None
    assert torch.equal(torch.cat([r["z_live"][0] for r in recs]), stored0)   # stale by design
    fresh1, _ = m3._reencode(recs)
    assert float((fresh1 - fresh0).abs().max()) > 1e-6            # only re-encode tracks it


# --- W6a-07: not blind ---------------------------------------------------------------------------

def _disconnected(self, agent):
    if not self.ready():
        return None
    ii = torch.randint(0, len(self._buf), (self.batch_size,)).tolist()
    recs = [self._buf[i] for i in ii]
    z = self.sensed_z_world(recs).detach()
    leaf = torch.zeros((), requires_grad=True)
    return self.objective(z, recs).detach() + leaf * 0.0


def test_w6a_07a_guard_raises_on_a_disconnected_member(monkeypatch):
    monkeypatch.setattr(wenc_mod.WorldEncoderMember, "loss", _disconnected)
    with pytest.raises(wt_mod.WakingTrainerReachError) as exc:
        _rollout(_cfg(**W6A_ON))
    msg = str(exc.value)
    assert "'world_encoder'" in msg and "FAILED" in msg and FAIL in msg


def test_w6a_07b_moved_check_fails_on_a_disconnected_member(monkeypatch):
    monkeypatch.setattr(wenc_mod.WorldEncoderMember, "loss", _disconnected)
    out, snaps = _run_with_group_snapshot(_cfg(**dict(W6A_ON, waking_trainer_guard_min_steps=0)))
    tr = out["agent"].waking_trainer
    assert tr.steps["world_encoder"] >= 8
    moved = [not torch.equal(p.detach(), snaps["world_encoder"][n])
             for n, p in tr.members["world_encoder"].encoder_named_parameters()]
    assert not any(moved), "the moved-check would PASS a disconnected world_encoder member"


def _direct_path(self, recs):
    """The OLD defect: SD-070's ZWorldP0Trainer._z_world_path on RAW world_state, which
    bypasses world_obs_encoder (SD-ZWORLD-SENSE-PATH-PARITY)."""
    se = self._agent.latent_stack.split_encoder
    w = torch.cat([r["obs"][-1][1] for r in recs], dim=0)
    return se.world_encoder(w) * torch.sigmoid(se.world_precision_logit).unsqueeze(0)


def test_w6a_07c_old_direct_path_fails_the_guard_on_world_obs_encoder(monkeypatch):
    monkeypatch.setattr(wenc_mod.WorldEncoderMember, "sensed_z_world", _direct_path)
    with pytest.raises(wt_mod.WakingTrainerReachError) as exc:
        _rollout(_cfg(**W6A_ON))
    assert "world_obs_encoder" in str(exc.value)


# --- W6a-08: knobs, TRAIN mode with every member ON ---------------------------------------------

def test_w6a_08_knobs_plumb_and_train_mode_all_members_on():
    cfg = _cfg(**dict(W6A_ON, waking_trainer_world_encoder_lr=5e-4,
                      waking_trainer_world_encoder_window=3,
                      waking_trainer_world_encoder_grad_clip=0.0,
                      waking_trainer_world_encoder_updates_per_step=2))
    assert cfg.waking_trainer_world_encoder_enabled is True
    assert cfg.waking_trainer_world_encoder_batch_size == 6
    assert cfg.waking_trainer_world_encoder_lr == 5e-4
    assert cfg.waking_trainer_world_encoder_window == 3
    _seed_all(SEED)
    m = REEAgent(cfg).waking_trainer.members["world_encoder"]
    assert (m.lr, m.window, m.grad_clip, m.updates_per_step) == (5e-4, 3, None, 2)
    all_on = _cfg(**dict(W6A_ON, waking_trainer_e1_enabled=True,
                         waking_trainer_e2_self_enabled=True,
                         waking_trainer_codec_enabled=True,
                         waking_trainer_e2_world_enabled=True,
                         waking_trainer_e2_world_batch_size=4))
    out = _rollout(all_on, train_mode=True, ticks=30)
    tr = out["agent"].waking_trainer
    assert list(tr.members)[-1] == "world_encoder"
    assert tr.steps["world_encoder"] >= 8 and tr.guard_results["world_encoder"].status == PASS
