"""Contracts for the native WakingTrainer skeleton + its harm_eval member (default OFF).

Design record: REE_assembly evidence/planning/native_waking_trainer_design_20260925.md
(0c0f5b76ec), sections 2 (i), 3 and 4a (M2/M3). Build: ree_core/utils/waking_trainer.py,
wired from REEAgent.__init__ (built only when ``waking_trainer_enabled``) and stepped from
REEAgent.update_residue() via ``_waking_trainer_step``.

  W1  OFF builds nothing: default config -> ``agent.waking_trainer is None`` and the
      trainer class is never constructed.
  W2  OFF is byte-identical to the PRE-CHANGE code path: a default-config StepHarness
      rollout equals the same rollout with ``_waking_trainer_step`` replaced by the
      pre-change behaviour (no trainer call at all) -- actions, per-tick z_world, the
      full state_dict and the torch / numpy / python RNG states after N ticks.
  W2b The W2 comparison is not blind: it DETECTS a default-OFF hook that draws one
      torch random number (the defect it exists to catch).
  W3  ON is RNG-neutral: (i) with a cadence that never steps, the ON rollout is
      byte-identical to OFF (recording replay samples draws and mutates nothing);
      (ii) with K=1 the global torch / numpy / python RNG states are identical before
      and after every trainer update.
  W4  ON reach (D1): the harm_eval group is exactly ``e3.harm_eval_head``'s tensors, its
      parameters move, and the gradient-reach guard returns PASS on the group.
  W4b W4 is not blind: with the loss disconnected (zero-sentinel, or a detached
      prediction) and the guard disarmed, W4's own "parameters moved" check FAILS.
  W5  Guard wiring (Q4d): with a disconnected harm_eval loss and the guard armed, the
      guard FAILs and the trainer RAISES, naming harm_eval_head.
  W6  The knobs reach the agent through ``REEConfig.from_dims`` (from_dims silently
      swallows unknown kwargs; see mech307_from_dims_unreachable_2026-08-07).

Scope: reach, not quality or influence. Nothing here claims the trained head changes E3's
selection or behaviour (that would be D2/D3 and is not measured).
Kept CPU-small: 8x8 grid, world_dim = self_dim = 32 (the deployed value), <= 24 ticks.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from experiments._harness import StepHarness
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils import waking_trainer as wt_mod
from ree_core.utils.config import REEConfig
from ree_core.utils.grad_reach_guard import FAIL, PASS

SEED = 7
TICKS = 20
DIM = 32


def _env(seed: int = SEED) -> CausalGridWorldV2:
    # hazard_harm > 0 so real harm events land in the replay buffer.
    return CausalGridWorldV2(size=8, num_hazards=3, num_resources=2, hazard_harm=0.5,
                             seed=seed)


def _cfg(**flags) -> REEConfig:
    env = _env()
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=DIM, world_dim=DIM, **flags)


def _rollout(cfg: REEConfig, ticks: int = TICKS, seed: int = SEED, train_mode: bool = False):
    """Fixed-seed StepHarness rollout over as many episodes as needed for ``ticks``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = _env(seed)
    agent = REEAgent(cfg)
    harness = StepHarness(agent, env, train_mode=train_mode, seed=seed)
    actions, zworld = [], []
    while len(actions) < ticks:
        for r in harness.run_episode(max_steps=ticks - len(actions)):
            actions.append(int(r.action.argmax().item()))
            zworld.append(r.latent.z_world.detach().clone())
    return {
        "agent": agent,
        "actions": actions,
        "zworld": zworld,
        "state": {k: v.detach().clone() for k, v in agent.state_dict().items()},
        "torch_rng": torch.get_rng_state().clone(),
        "np_rng": np.random.get_state(),
        "py_rng": random.getstate(),
    }


def _differences(a, b):
    """Every way two rollouts differ (empty list == byte-identical)."""
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


def _pre_change_step(self, harm_signal, hypothesis_tag):
    # The pre-change update_residue made no trainer call and added no metrics.
    return {}


# --- W1 / W2 / W2b: default OFF --------------------------------------------------------

def test_w1_off_builds_nothing(monkeypatch):
    assert REEConfig().waking_trainer_enabled is False
    cfg = _cfg()
    assert cfg.waking_trainer_enabled is False

    def _boom(*a, **k):
        raise AssertionError("WakingTrainer constructed at default config")

    monkeypatch.setattr(wt_mod.WakingTrainer, "__init__", _boom)
    agent = REEAgent(cfg)
    assert agent.waking_trainer is None
    # The trainer is a plain object: nothing trainer-owned may appear among parameters.
    assert not any("waking" in n for n, _ in agent.named_parameters())


def test_w2_off_rollout_byte_identical_to_pre_change(monkeypatch):
    off = _rollout(_cfg())
    with monkeypatch.context() as m:
        m.setattr(REEAgent, "_waking_trainer_step", _pre_change_step)
        ref = _rollout(_cfg())
    assert len(off["actions"]) == TICKS
    assert _differences(off, ref) == []


def test_w2b_off_identity_check_detects_an_rng_draw(monkeypatch):
    """Run the W2 comparison against the defect it guards: an OFF hook that draws RNG."""
    ref = _rollout(_cfg())

    def _leaky(self, harm_signal, hypothesis_tag):
        torch.rand(1)
        return {}

    with monkeypatch.context() as m:
        m.setattr(REEAgent, "_waking_trainer_step", _leaky)
        leaky = _rollout(_cfg())
    assert "torch_rng" in _differences(ref, leaky)


# --- W3: ON is RNG-neutral -------------------------------------------------------------

def test_w3i_on_without_updates_is_byte_identical_to_off():
    off = _rollout(_cfg())
    on = _rollout(_cfg(waking_trainer_enabled=True, waking_trainer_every_k=10 ** 6))
    tr = on["agent"].waking_trainer
    assert tr is not None and tr.ticks == TICKS and tr.steps["harm_eval"] == 0
    assert len(tr.members["harm_eval"]._buf) == TICKS  # it recorded, and drew nothing
    assert _differences(off, on) == []


def test_w3ii_every_update_is_rng_neutral(monkeypatch):
    seen = []
    orig = wt_mod.WakingTrainer._update

    def _spy(self, name, member):
        before = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        out = orig(self, name, member)
        after = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
        seen.append(torch.equal(before[0], after[0])
                    and np.array_equal(before[1][1], after[1][1])
                    and before[1][2:] == after[1][2:]
                    and before[2] == after[2])
        return out

    monkeypatch.setattr(wt_mod.WakingTrainer, "_update", _spy)
    on = _rollout(_cfg(waking_trainer_enabled=True, waking_trainer_every_k=1,
                       waking_trainer_batch_size=4))
    assert on["agent"].waking_trainer.steps["harm_eval"] >= 8  # updates really ran
    assert seen and all(seen)


# --- W4 / W4b / W5: reach and guard ----------------------------------------------------

def _harm_eval_snapshot(agent):
    return {n: p.detach().clone() for n, p in agent.e3.harm_eval_head.named_parameters()}


def _harm_eval_moved(agent, snap):
    return {n: not torch.equal(p.detach(), snap[n])
            for n, p in agent.e3.harm_eval_head.named_parameters()}


def _run_on_from_fresh_snapshot(cfg):
    """ON rollout, returning (agent, harm_eval snapshot taken right after construction)."""
    snaps = {}
    orig_init = wt_mod.WakingTrainer.__init__

    def _init(self, agent, config, members=None):
        snaps["pre"] = _harm_eval_snapshot(agent)
        orig_init(self, agent, config, members)

    wt_mod.WakingTrainer.__init__ = _init
    try:
        out = _rollout(cfg)
    finally:
        wt_mod.WakingTrainer.__init__ = orig_init
    return out["agent"], snaps["pre"]


ON_FLAGS = dict(waking_trainer_enabled=True, waking_trainer_every_k=1,
                waking_trainer_batch_size=4, waking_trainer_guard_min_steps=8)


def test_w4_on_harm_eval_reached_moved_and_guard_pass():
    agent, pre = _run_on_from_fresh_snapshot(_cfg(**ON_FLAGS))
    tr = agent.waking_trainer
    expected = sorted("e3.harm_eval_head." + n for n, _ in agent.e3.harm_eval_head.named_parameters())
    assert sorted(tr.group_names("harm_eval")) == expected and len(expected) == 4
    assert tr.steps["harm_eval"] >= 8
    moved = _harm_eval_moved(agent, pre)
    assert all(moved.values()), moved
    res = tr.guard_results["harm_eval"]
    assert res.status == PASS, res.summary()
    assert res.n_checked == 4
    assert tr.report()["harm_eval"]["guard"] == PASS


def _zero_sentinel_loss(self, agent):
    if not self.ready():
        return None
    zw = torch.cat([z for z, _ in list(self._buf)[: self.batch_size]], dim=0)
    return self._harm_eval(zw).sum() * 0.0          # grad exactly 0 everywhere


def _detached_loss(self, agent):
    if not self.ready():
        return None
    zw = torch.cat([z for z, _ in list(self._buf)[: self.batch_size]], dim=0)
    leaf = torch.zeros((), requires_grad=True)      # makes .backward() legal
    return self._harm_eval(zw).detach().pow(2).mean() + leaf * 0.0   # grad None


@pytest.mark.parametrize("broken", [_zero_sentinel_loss, _detached_loss],
                         ids=["zero_sentinel", "detached"])
def test_w4b_reach_check_is_not_blind_to_a_disconnected_loss(monkeypatch, broken):
    monkeypatch.setattr(wt_mod.HarmEvalMember, "loss", broken)
    flags = dict(ON_FLAGS, waking_trainer_guard_min_steps=0)  # guard disarmed
    agent, pre = _run_on_from_fresh_snapshot(_cfg(**flags))
    assert agent.waking_trainer.steps["harm_eval"] >= 8  # the optimizer did step
    moved = _harm_eval_moved(agent, pre)
    assert not any(moved.values()), "W4's moved-check would PASS a disconnected loss"


@pytest.mark.parametrize("broken", [_zero_sentinel_loss, _detached_loss],
                         ids=["zero_sentinel", "detached"])
def test_w5_guard_fails_and_trainer_raises_on_disconnected_loss(monkeypatch, broken):
    monkeypatch.setattr(wt_mod.HarmEvalMember, "loss", broken)
    with pytest.raises(wt_mod.WakingTrainerReachError) as exc:
        _rollout(_cfg(**ON_FLAGS))
    msg = str(exc.value)
    assert "FAILED" in msg and "e3.harm_eval_head" in msg and FAIL in msg


def test_w5b_guard_allowlist_is_the_explicit_frozen_by_design_table():
    agent = REEAgent(_cfg(**ON_FLAGS))
    from ree_core.utils.grad_reach_guard import FROZEN_BY_DESIGN
    guard = agent.waking_trainer._guards["harm_eval"]
    assert guard is not None and guard.allowlist == dict(FROZEN_BY_DESIGN)
    # No allowlisted prefix overlaps the harm_eval group (G3 would call it stale).
    for n in agent.waking_trainer.group_names("harm_eval"):
        assert not any(n == p or n.startswith(p + ".") for p in FROZEN_BY_DESIGN)


# --- W6: knobs reach the agent through from_dims ---------------------------------------

def test_w6_knobs_plumb_through_from_dims():
    cfg = _cfg(waking_trainer_enabled=True, waking_trainer_every_k=3,
               waking_trainer_harm_eval_lr=5e-4, waking_trainer_batch_size=6,
               waking_trainer_buffer_max=50, waking_trainer_guard_min_steps=5,
               waking_trainer_seed=11)
    assert (cfg.waking_trainer_enabled, cfg.waking_trainer_every_k,
            cfg.waking_trainer_harm_eval_lr, cfg.waking_trainer_batch_size,
            cfg.waking_trainer_buffer_max, cfg.waking_trainer_guard_min_steps,
            cfg.waking_trainer_seed) == (True, 3, 5e-4, 6, 50, 5, 11)
    tr = REEAgent(cfg).waking_trainer
    m = tr.members["harm_eval"]
    assert (tr.every_k, tr.guard_min_steps, m.batch_size, m._buf.maxlen) == (3, 5, 6, 50)
    assert tr.optimizers["harm_eval"].param_groups[0]["lr"] == 5e-4
