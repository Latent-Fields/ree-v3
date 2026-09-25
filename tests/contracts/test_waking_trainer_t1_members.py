"""Contracts for the WakingTrainer T1 members: E1Member and E2SelfMember (default OFF).

Plan of record: REE_assembly evidence/planning/coupled_loop_repair_campaign_plan.md,
section 3 W-trainer (T1). Design: evidence/planning/native_waking_trainer_design_20260925.md
section 1a rows 1-3 and section 2 (i) point 3. Build: ree_core/utils/waking_trainer.py
(``E1Member`` over the native ``compute_prediction_loss``; ``E2SelfMember`` over the native
``compute_e2_loss`` with transitions it records itself), knobs
``waking_trainer_e1_enabled`` / ``waking_trainer_e2_self_enabled`` (+ ``_lr``), and a
one-line env-boundary hook in ``REEAgent.notify_env_reset``.

  T1a OFF: the knobs default False; an ON trainer with them OFF holds exactly the C1
      member set (harm_eval) and never constructs E1Member / E2SelfMember.
  T1b OFF byte-identity across episode boundaries: a default-config multi-episode rollout
      equals the same rollout with ``notify_env_reset`` replaced by its pre-change body.
  T2  ON RNG neutrality: (i) with a never-firing cadence and both members ON, a
      multi-episode rollout is byte-identical to OFF (recording draws nothing, changes
      nothing); (ii) every update of every member leaves global torch / numpy / python
      RNG identical.
  T3  E1 reach (D1): group == every trainable ``agent.e1`` tensor; guard PASS with no
      CANNOT_DETERMINE; the FROZEN_BY_DESIGN ``write_gate`` tensors are in the group and
      classified allowlisted (G3 checked, not skipped); every other tensor moved; no leak.
  T4  E2-self reach (D1): group == ``e2.self_transition`` + ``e2.self_action_encoder``;
      guard PASS; all moved; no leak.
  T5  Not blind: with either member's loss disconnected, (a) guard armed -> the trainer
      RAISES naming the group; (b) guard disarmed -> T3/T4's moved-check FAILS.
  T6  E2-self records its OWN transitions correctly: over a multi-episode rollout its
      buffer equals, entry for entry, what the harness passed to ``record_transition``
      (no cross-episode pair); the comparison is not blind (a one-step misalignment does
      not match); every recorded tensor is detached; the loss call leaves the agent's
      own ``_e2_transition_buffer`` untouched.
  T6b ``notify_env_reset`` alone (a driver that resets the env without ``agent.reset``)
      drops the pending pair; without it the pair is recorded.
  T7  Retained graph: a TRAIN-mode (grad-enabled) rollout with all members ON completes
      with no autograd error.
  T8  The four knobs reach the agent through ``REEConfig.from_dims``.

Scope: reach and isolation, not quality or influence (no D2/D3 claim).
CPU-small: 8x8 grid, world_dim = self_dim = 32 (the deployed value), <= 40 ticks.
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
from ree_core.utils.grad_reach_guard import FAIL, FROZEN_BY_DESIGN, PASS, T_ALLOWLISTED

SEED = 7
TICKS = 40
EP_LEN = 15          # -> 3 episodes in 40 ticks, so env boundaries are exercised
DIM = 32

T1_ON = dict(waking_trainer_enabled=True, waking_trainer_every_k=1,
             waking_trainer_batch_size=4, waking_trainer_guard_min_steps=8,
             waking_trainer_e1_enabled=True, waking_trainer_e2_self_enabled=True)


def _env(seed: int = SEED) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=3, num_resources=2, hazard_harm=0.5,
                             seed=seed)


def _cfg(**flags) -> REEConfig:
    env = _env()
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=DIM, world_dim=DIM, alpha_world=0.3, **flags)


def _rollout(cfg: REEConfig, ticks: int = TICKS, seed: int = SEED, train_mode: bool = False,
             on_agent=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = _env(seed)
    agent = REEAgent(cfg)
    if on_agent is not None:
        on_agent(agent)
    harness = StepHarness(agent, env, train_mode=train_mode, seed=seed)
    actions, zworld, episodes = [], [], 0
    while len(actions) < ticks:
        episodes += 1
        for r in harness.run_episode(max_steps=min(EP_LEN, ticks - len(actions))):
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


def _snapshot(named):
    return {n: p.detach().clone() for n, p in named}


def _moved(named, snap):
    return {n: not torch.equal(p.detach(), snap[n]) for n, p in named}


def _run_with_group_snapshots(cfg, train_mode=False):
    """ON rollout; also returns each member group's params snapshotted at construction."""
    snaps = {}
    orig_register = wt_mod.WakingTrainer.register

    def _register(self, member):
        snaps[member.name] = _snapshot(member.named_parameters())
        orig_register(self, member)

    wt_mod.WakingTrainer.register = _register
    try:
        out = _rollout(cfg, train_mode=train_mode)
    finally:
        wt_mod.WakingTrainer.register = orig_register
    return out, snaps


# --- T1a / T1b: default OFF ------------------------------------------------------------

def test_t1a_knobs_default_off_and_on_trainer_keeps_c1_member_set(monkeypatch):
    base = REEConfig()
    assert (base.waking_trainer_e1_enabled, base.waking_trainer_e2_self_enabled) == (False, False)

    def _boom(*a, **k):
        raise AssertionError("T1 member constructed with its knob OFF")

    monkeypatch.setattr(wt_mod.E1Member, "__init__", _boom)
    monkeypatch.setattr(wt_mod.E2SelfMember, "__init__", _boom)
    tr = REEAgent(_cfg(waking_trainer_enabled=True)).waking_trainer
    assert list(tr.members) == ["harm_eval"]


def _pre_change_notify_env_reset(self):
    # notify_env_reset as it was before T1 (no waking-trainer hook).
    if self.observation_reliability is not None:
        self.observation_reliability.on_episode_reset()
    if self.replay_provenance is not None:
        self._provenance_env_reset_pending = True


def test_t1b_off_multi_episode_rollout_byte_identical_to_pre_change(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("WakingTrainer touched at default config")

    monkeypatch.setattr(wt_mod.WakingTrainer, "__init__", _boom)
    monkeypatch.setattr(wt_mod.WakingTrainer, "on_env_reset", _boom)
    off = _rollout(_cfg())
    assert off["episodes"] >= 3 and off["agent"].waking_trainer is None
    with monkeypatch.context() as m:
        m.setattr(REEAgent, "notify_env_reset", _pre_change_notify_env_reset)
        ref = _rollout(_cfg())
    assert _differences(off, ref) == []


# --- T2: ON RNG neutrality --------------------------------------------------------------

def test_t2i_on_without_updates_is_byte_identical_to_off():
    off = _rollout(_cfg())
    on = _rollout(_cfg(**dict(T1_ON, waking_trainer_every_k=10 ** 6)))
    tr = on["agent"].waking_trainer
    assert set(tr.members) == {"harm_eval", "e1", "e2_self"}
    assert all(v == 0 for v in tr.steps.values())
    assert len(tr.members["e2_self"]._buf) == TICKS - on["episodes"]  # it recorded
    assert _differences(off, on) == []


def test_t2ii_every_member_update_is_rng_neutral(monkeypatch):
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
    on = _rollout(_cfg(**T1_ON))
    tr = on["agent"].waking_trainer
    for name in ("e1", "e2_self"):
        assert tr.steps[name] >= 8, (name, tr.steps)      # the updates really ran
        assert seen.get(name) and all(seen[name]), name


def test_t2iii_rng_spy_is_not_blind(monkeypatch):
    """The T2ii comparison detects an update that draws from the GLOBAL RNG."""
    def _leaky_update(self, name, member):
        torch.rand(1)
        return None

    monkeypatch.setattr(wt_mod.WakingTrainer, "_update", _leaky_update)
    a = _rollout(_cfg(**T1_ON), ticks=12)
    b = _rollout(_cfg(**dict(T1_ON, waking_trainer_every_k=10 ** 6)), ticks=12)
    assert "torch_rng" in _differences(a, b)


# --- T3 / T4: reach -------------------------------------------------------------------

def _allowlisted(name):
    return any(name == p or name.startswith(p + ".") for p in FROZEN_BY_DESIGN)


def test_t3_e1_group_reached_moved_guard_pass():
    out, snaps = _run_with_group_snapshots(_cfg(**T1_ON))
    agent = out["agent"]
    tr = agent.waking_trainer
    ids = {id(p) for p in agent.e1.parameters() if p.requires_grad}
    expected = sorted(n for n, p in agent.named_parameters() if id(p) in ids)
    assert sorted(tr.group_names("e1")) == expected and len(expected) > 10
    res = tr.guard_results["e1"]
    assert res.status == PASS, res.summary()
    allow = [n for n in expected if _allowlisted(n)]
    assert allow and all("context_memory.write_gate" in n for n in allow)
    assert sorted(t.name for t in res.by_status(T_ALLOWLISTED)) == sorted(allow)
    assert res.n_checked == len(expected) - len(allow)
    assert res.leaked_names == []
    moved = _moved(tr.members["e1"].named_parameters(), snaps["e1"])
    assert all(moved[n] for n in expected if n not in allow), moved
    assert not any(moved[n] for n in allow)


def test_t4_e2_self_group_reached_moved_guard_pass():
    out, snaps = _run_with_group_snapshots(_cfg(**T1_ON))
    agent = out["agent"]
    tr = agent.waking_trainer
    expected = sorted(
        ["e2.self_transition." + n for n, _ in agent.e2.self_transition.named_parameters()]
        + ["e2.self_action_encoder." + n
           for n, _ in agent.e2.self_action_encoder.named_parameters()])
    assert sorted(tr.group_names("e2_self")) == expected and len(expected) >= 4
    assert tr.steps["e2_self"] >= 8
    res = tr.guard_results["e2_self"]
    assert res.status == PASS, res.summary()
    assert res.n_checked == len(expected) and res.leaked_names == []
    moved = _moved(tr.members["e2_self"].named_parameters(), snaps["e2_self"])
    assert all(moved.values()), moved


# --- T5: not blind ----------------------------------------------------------------------

def _e1_zero_sentinel(self, agent):
    if not self.ready():
        return None
    return agent.compute_prediction_loss() * 0.0                 # grad exactly 0


def _e2_detached(self, agent):
    if not self.ready():
        return None
    leaf = torch.zeros((), requires_grad=True)
    z, a, z1 = zip(*list(self._buf)[: self.batch_size])
    pred = agent.e2.predict_next_self(torch.cat(z), torch.cat(a)).detach()
    return ((pred - torch.cat(z1)) ** 2).mean() + leaf * 0.0     # grad None


BROKEN = [("e1", wt_mod.E1Member, _e1_zero_sentinel),
          ("e2_self", wt_mod.E2SelfMember, _e2_detached)]


@pytest.mark.parametrize("group,cls,broken", BROKEN, ids=[b[0] for b in BROKEN])
def test_t5a_guard_raises_on_a_disconnected_member(monkeypatch, group, cls, broken):
    monkeypatch.setattr(cls, "loss", broken)
    with pytest.raises(wt_mod.WakingTrainerReachError) as exc:
        _rollout(_cfg(**T1_ON))
    msg = str(exc.value)
    assert ("'%s'" % group) in msg and "FAILED" in msg and FAIL in msg


@pytest.mark.parametrize("group,cls,broken", BROKEN, ids=[b[0] for b in BROKEN])
def test_t5b_moved_check_fails_on_a_disconnected_member(monkeypatch, group, cls, broken):
    monkeypatch.setattr(cls, "loss", broken)
    out, snaps = _run_with_group_snapshots(_cfg(**dict(T1_ON, waking_trainer_guard_min_steps=0)))
    tr = out["agent"].waking_trainer
    assert tr.steps[group] >= 8                                     # the optimizer stepped
    moved = _moved(tr.members[group].named_parameters(), snaps[group])
    assert not any(moved.values()), "the moved-check would PASS a disconnected %s" % group


# --- T6 / T6b: E2-self records its own transitions -------------------------------------

def test_t6_e2_self_own_transitions_match_harness_record_transition():
    on = _rollout(_cfg(**dict(T1_ON, waking_trainer_every_k=10 ** 6)))
    agent = on["agent"]
    own = list(agent.waking_trainer.members["e2_self"]._buf)
    ref = list(agent._e2_transition_buffer)       # filled by StepHarness.record_transition
    assert on["episodes"] >= 3
    assert len(own) == len(ref) == TICKS - on["episodes"]   # no cross-episode pair

    def _same(x, y):
        return (torch.equal(x[0], y[0]) and torch.equal(x[1].reshape(-1), y[1].reshape(-1))
                and torch.equal(x[2], y[2]))

    assert all(_same(x, y) for x, y in zip(own, ref))
    # Not blind: a one-step misalignment of the same streams does not match.
    assert not all(_same(x, y) for x, y in zip(own[1:], ref[:-1]))
    for trip in own:
        assert all(t.grad_fn is None and not t.requires_grad for t in trip)


def test_t6_loss_call_leaves_the_agents_own_buffer_untouched():
    on = _rollout(_cfg(**T1_ON))
    agent = on["agent"]
    before = agent._e2_transition_buffer
    snap = [tuple(t.clone() for t in trip) for trip in before]
    member = agent.waking_trainer.members["e2_self"]
    assert member.ready()
    assert member.loss(agent) is not None
    assert agent._e2_transition_buffer is before
    assert len(before) == len(snap) and all(
        all(torch.equal(a, b) for a, b in zip(x, y)) for x, y in zip(before, snap))


def test_t6b_notify_env_reset_drops_the_pending_pair():
    def _two_calls(reset_between):
        agent = REEAgent(_cfg(**dict(T1_ON, waking_trainer_every_k=10 ** 6)))
        tr = agent.waking_trainer
        m = tr.members["e2_self"]
        agent._current_latent = agent.latent_stack.init_state(batch_size=1, device=agent.device)
        agent._last_action = torch.nn.functional.one_hot(torch.tensor([1]), 5).float()
        tr.on_waking_step(0.0)
        if reset_between:
            agent.notify_env_reset()          # env reset WITHOUT agent.reset()
        tr.on_waking_step(0.0)
        return len(m._buf)

    assert _two_calls(reset_between=False) == 1
    assert _two_calls(reset_between=True) == 0


# --- T7: retained graph -----------------------------------------------------------------

def test_t7_train_mode_rollout_with_all_members_on_has_no_autograd_error():
    on = _rollout(_cfg(**T1_ON), train_mode=True)
    tr = on["agent"].waking_trainer
    assert all(tr.steps[n] >= 8 for n in ("harm_eval", "e1", "e2_self")), tr.steps
    assert {n: r.status for n, r in tr.guard_results.items()} == {
        "harm_eval": PASS, "e1": PASS, "e2_self": PASS}


# --- T8: knobs through from_dims --------------------------------------------------------

def test_t8_knobs_plumb_through_from_dims():
    cfg = _cfg(waking_trainer_enabled=True, waking_trainer_e1_enabled=True,
               waking_trainer_e1_lr=2e-4, waking_trainer_e2_self_enabled=True,
               waking_trainer_e2_self_lr=3e-4)
    assert (cfg.waking_trainer_e1_enabled, cfg.waking_trainer_e1_lr,
            cfg.waking_trainer_e2_self_enabled, cfg.waking_trainer_e2_self_lr) == (
        True, 2e-4, True, 3e-4)
    tr = REEAgent(cfg).waking_trainer
    assert tr.optimizers["e1"].param_groups[0]["lr"] == 2e-4
    assert tr.optimizers["e2_self"].param_groups[0]["lr"] == 3e-4
