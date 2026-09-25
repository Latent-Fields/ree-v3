"""Contracts for the W1 codec parts (1)-(3) (coupled-loop-repair campaign, default OFF).

Plan: REE_assembly evidence/planning/coupled_loop_repair_campaign_plan.md sec 3 W1 (member
gate (a)-(e)); build record evidence/planning/w1_codec_build_20260925.md. The three coupled
defects are from action_decoder_training_causal_probe_20260925.md (a369f411ff8):

  (1) the codec is untied (no loss reaches e2.action_object_head or
      hippocampal.action_object_decoder)    -> CodecMember, waking_trainer_codec_enabled
  (2) raw decoder logits are the rollout action (decoded norm 50 -> ~370-1080 over 3 CEM
      iterations once the decoder is trained) -> use_codec_bounded_decode
  (3) iteration-0 samples at unit scale, ~12x outside the encoder image
                                            -> use_codec_iter0_image_match

  K1  knobs: defaults False (5 knobs), from_dims round trip (it swallows unknown kwargs).
  K2  OFF: _decode_action_objects returns the raw decoder output, iteration 0 reads
      terrain_prior with ao_std = 1, the new helpers are never called, and an ON trainer
      holds no codec group; the default pool, its world states and the RNG state after the
      call equal those of a tree whose new helpers raise if touched.
  K3  bounded decode: forward values are an exact one-hot of the logits' argmax and the
      gradient still reaches every decoder tensor (straight-through).
  K4  ON is RNG-count neutral: knobs ON consume the same global RNG draws as OFF.
  GA  gate (a): guard PASS on the codec group, which is exactly the encoder + decoder
      tensors, and both halves move.  GA' not blind: a decoder-only loss (codes detached,
      trace candidate 1) FAILs the guard and the trainer raises naming the encoder.
  GB  gate (b): after member training on a small fixed dataset (160 z_world states sensed
      from uniform-random play), held-out round trip >= 0.95 on EVERY class.  GB' not
      blind: the untrained codec FAILs the same instrument.
  GC  gate (c): with the trained codec and both knobs ON, the median decoded norm stays in
      [0.5, 2] x the one-hot norm over the 3 CEM iterations with no growth (I1
      codec_ranges).  GC' the same trained codec with the knobs OFF (pre-build) FAILs:
      decoded norm grows iteration on iteration.
  GD  gate (d): iteration-0 median O-norm within [0.5, 2] x the encoder-image median, on
      the untrained and the trained codec.  GD' OFF (pre-build) FAILs on both.
  P   probes behind the three PROBED entries in tests/test_flag_inertness.py: each knob ON
      changes what it controls.

Gate (e) is NOT a contract (needs W3/W4; its consumer leg is reported-until-W5). Evidence
domain of GA-GD: D1 (the codec round-trips and is well-conditioned in the native CEM loop),
never evidence that proposals become choice-relevant.
Kept CPU-small: 8x8 grid, world_dim = self_dim = 32 (the deployed value), action_dim 5,
num_candidates 32, horizon 30 (from_dims), 3 CEM iterations; the member trains once per
module (1000 steps, ~1 s).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments._lib import coupled_acceptance as C
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils import waking_trainer as wt_mod
from ree_core.utils.config import HippocampalConfig, REEConfig
from ree_core.utils.grad_reach_guard import PASS

DIM = 32
SEED = 0
TRAIN_STEPS = 1000
KNOBS_H = ("use_codec_bounded_decode", "use_codec_iter0_image_match")


def _env(seed: int = SEED) -> CausalGridWorldV2:
    return CausalGridWorldV2(size=8, num_hazards=2, num_resources=3, seed=seed)


def _agent(seed: int = SEED, **flags) -> REEAgent:
    env = _env(seed)
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
                              action_dim=env.action_dim, self_dim=DIM, world_dim=DIM, **flags)
    agent = REEAgent(cfg)
    agent.eval()
    return agent


def _set(agent: REEAgent, on: bool) -> None:
    for k in KNOBS_H:
        setattr(agent.hippocampal.config, k, bool(on))


def _states(agent: REEAgent, seed: int = 1, n_steps: int = 240) -> torch.Tensor:
    env = _env(seed)
    eps = C.collect_uniform_random_episodes(agent, env, n_steps, seed=seed,
                                            classes=list(range(env.action_dim)))
    return torch.cat([e["z"] for e in eps], dim=0)


def _latents(Z: torch.Tensor, lo: int, n: int):
    return [(Z[i:i + 1], torch.zeros(1, DIM)) for i in range(lo, lo + n)]


@pytest.fixture(scope="module")
def codec():
    """One agent, its fixed dataset, and its codec BEFORE and AFTER member training."""
    torch.set_num_threads(2)
    np.random.seed(SEED)
    agent = _agent(waking_trainer_enabled=True, waking_trainer_codec_enabled=True)
    Z = _states(agent)
    train, held = Z[:160], Z[160:220]
    lat = _latents(Z, 220, 8)
    wt = agent.waking_trainer
    member = wt.members["codec"]
    enc0 = {n: p.detach().clone() for n, p in member.named_parameters()}
    pre = {}
    pre["rt"] = C.codec_roundtrip_accuracy(agent, list(held), bar_per_class=0.95)
    pre["img"] = C.encoder_image_norm(agent.e2, list(held[:20]), agent.e2.config.action_dim)
    _set(agent, False)
    pre["off"] = C.codec_ranges(C.cem_codec_trace(agent, lat, seed=3), pre["img"])
    _set(agent, True)
    pre["on"] = C.codec_ranges(C.cem_codec_trace(agent, lat, seed=3), pre["img"])
    member.add_states(list(train))
    losses = [wt._update("codec", member) for _ in range(TRAIN_STEPS)]
    post = {"losses": losses, "moved": {n: float((p.detach() - enc0[n]).abs().max())
                                        for n, p in member.named_parameters()}}
    post["rt"] = C.codec_roundtrip_accuracy(agent, list(held), bar_per_class=0.95)
    post["img"] = C.encoder_image_norm(agent.e2, list(held[:20]), agent.e2.config.action_dim)
    _set(agent, False)
    post["off_trace"] = C.cem_codec_trace(agent, lat, seed=3)
    post["off"] = C.codec_ranges(post["off_trace"], post["img"])
    _set(agent, True)
    post["on_trace"] = C.cem_codec_trace(agent, lat, seed=3)
    post["on"] = C.codec_ranges(post["on_trace"], post["img"])
    _set(agent, False)
    return {"agent": agent, "Z": Z, "pre": pre, "post": post}


# ------------------------------------------------------------------------------ K1 knobs
def test_k1_knobs_default_off_and_reach_through_from_dims():
    h = HippocampalConfig()
    assert h.use_codec_bounded_decode is False and h.use_codec_iter0_image_match is False
    base = _agent()
    assert base.config.hippocampal.use_codec_bounded_decode is False
    assert base.config.hippocampal.use_codec_iter0_image_match is False
    assert base.config.waking_trainer_codec_enabled is False
    env = _env()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=DIM, world_dim=DIM,
        use_codec_bounded_decode=True, use_codec_iter0_image_match=True,
        waking_trainer_codec_enabled=True, waking_trainer_codec_lr=3e-3,
        waking_trainer_codec_code_l2=0.05)
    assert cfg.hippocampal.use_codec_bounded_decode is True
    assert cfg.hippocampal.use_codec_iter0_image_match is True
    assert cfg.waking_trainer_codec_enabled is True
    assert cfg.waking_trainer_codec_lr == 3e-3 and cfg.waking_trainer_codec_code_l2 == 0.05


# ------------------------------------------------------------------------------ K2 OFF
def _pool_fingerprint(agent: REEAgent, Z: torch.Tensor, n_states: int = 3):
    out = []
    for j in range(n_states):
        torch.manual_seed(100 + j)
        pool = agent.hippocampal.propose_trajectories(Z[j:j + 1], z_self=torch.zeros(1, DIM))
        out.append((torch.stack([t.actions for t in pool]).detach(),
                    torch.stack([torch.stack(t.world_states) for t in pool]).detach(),
                    torch.get_rng_state().clone()))
    return out


def test_k2_off_is_the_pre_build_codec_path(monkeypatch):
    agent = _agent()
    Z = _states(agent, n_steps=12)
    hip = agent.hippocampal
    x = torch.randn(4, 3, hip.config.action_object_dim)
    y = hip._decode_action_objects(x)
    assert torch.equal(y, hip.action_object_decoder(x.reshape(12, -1)).reshape(4, 3, -1)), \
        "OFF decode must return the raw decoder output"
    ref = _pool_fingerprint(agent, Z)
    calls = {"terrain": 0}
    orig_terrain = hip._get_terrain_action_object_mean

    def terrain_spy(*a, **k):
        calls["terrain"] += 1
        return orig_terrain(*a, **k)

    def boom(*a, **k):
        raise AssertionError("a W1 helper ran on the OFF path")

    monkeypatch.setattr(hip, "_get_terrain_action_object_mean", terrain_spy)
    monkeypatch.setattr(hip, "_encoder_image_init", boom)
    monkeypatch.setattr(type(hip), "_bounded_onehot", staticmethod(boom))
    got = _pool_fingerprint(agent, Z)
    assert calls["terrain"] == 3, "OFF iteration 0 reads terrain_prior once per call"
    for (a0, w0, r0), (a1, w1, r1) in zip(ref, got):
        assert torch.equal(a0, a1) and torch.equal(w0, w1) and torch.equal(r0, r1)
    trainer_on = _agent(waking_trainer_enabled=True)
    assert list(trainer_on.waking_trainer.members) == ["harm_eval"], \
        "codec knob OFF -> an ON trainer holds no codec group"


# ------------------------------------------------------------------------------ K3 bounded decode
def test_k3_bounded_decode_is_exact_onehot_with_straight_through_gradient():
    agent = _agent(use_codec_bounded_decode=True)
    hip = agent.hippocampal
    x = torch.randn(8, 5, hip.config.action_object_dim) * 40.0  # far outside any image
    y = hip._decode_action_objects(x)
    raw = hip.action_object_decoder(x.reshape(40, -1)).reshape(8, 5, -1)
    assert torch.equal(y.detach().sum(-1), torch.ones(8, 5))
    assert torch.equal(y.detach().argmax(-1), raw.detach().argmax(-1))
    assert set(torch.unique(y.detach()).tolist()) <= {0.0, 1.0}
    assert float(y.detach().norm(dim=-1).max()) == 1.0
    (y * torch.randn_like(y)).sum().backward()
    for n, p in hip.action_object_decoder.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, n


# ------------------------------------------------------------------------------ K4 RNG count
def test_k4_on_consumes_the_same_rng_draws_as_off():
    off, on = _agent(), _agent(use_codec_bounded_decode=True, use_codec_iter0_image_match=True)
    Z = _states(off, n_steps=12)
    for j in range(3):
        torch.manual_seed(7 + j)
        off.hippocampal.propose_trajectories(Z[j:j + 1], z_self=torch.zeros(1, DIM))
        s_off = torch.get_rng_state().clone()
        torch.manual_seed(7 + j)
        on.hippocampal.propose_trajectories(Z[j:j + 1], z_self=torch.zeros(1, DIM))
        assert torch.equal(s_off, torch.get_rng_state()), "ON must draw exactly what OFF draws"


# ------------------------------------------------------------------------------ gate (a)
def test_ga_guard_pass_on_codec_group(codec):
    agent = codec["agent"]
    wt = agent.waking_trainer
    names = wt.group_names("codec")
    enc = {n for n, _ in agent.named_parameters() if n.startswith("e2.action_object_head.")}
    dec = {n for n, _ in agent.named_parameters()
           if n.startswith("hippocampal.action_object_decoder.")}
    assert enc and dec and set(names) == enc | dec, names
    assert wt.guard_results["codec"].status == PASS
    moved = codec["post"]["moved"]
    assert all(v > 0.0 for v in moved.values()), moved
    assert all(np.isfinite(l) for l in codec["post"]["losses"])


def test_ga_not_blind_decoder_only_loss_fails_the_guard(monkeypatch):
    from ree_core.utils import waking_trainer_codec as codec_mod
    agent = _agent(waking_trainer_enabled=True, waking_trainer_codec_enabled=True)
    member = agent.waking_trainer.members["codec"]

    def decoder_only(self, zw):
        A = self.action_dim
        codes = self._action_object(zw.repeat_interleave(A, dim=0),
                                    torch.eye(A).repeat(zw.shape[0], 1)).detach()
        return torch.nn.functional.cross_entropy(
            self._decoder(codes), torch.arange(A).repeat(zw.shape[0]))

    monkeypatch.setattr(codec_mod.CodecMember, "batch_loss", decoder_only)
    member.add_states([torch.randn(DIM) for _ in range(32)])
    with pytest.raises(wt_mod.WakingTrainerReachError, match="action_object_head"):
        for _ in range(agent.config.waking_trainer_guard_min_steps):
            agent.waking_trainer._update("codec", member)


# ------------------------------------------------------------------------------ gate (b)
def test_gb_heldout_roundtrip_every_class(codec):
    rt = codec["post"]["rt"]
    assert rt["verdict"] == C.PASS and rt["min_class"] >= 0.95, rt
    assert rt["n_states"] == 60 and len(rt["per_class"]) == 5


def test_gb_not_blind_untrained_codec_fails(codec):
    rt = codec["pre"]["rt"]
    assert rt["verdict"] == C.FAIL and rt["min_class"] < 0.95, rt


# ------------------------------------------------------------------------------ gate (c)
def test_gc_decoded_norm_bounded_without_growth(codec):
    r = codec["post"]["on"]
    assert r["gate_c_pass"], r
    tr = codec["post"]["on_trace"]
    assert tr["num_cem_iterations"] == 3 and len(tr["per_iteration"]) == 3
    assert all(0.5 <= d <= 2.0 for d in r["decoded_ratio_by_iteration"])
    assert all(g <= 1.0 + 1e-6 for g in tr["decoded_growth_per_iteration"])


def test_gc_not_blind_pre_build_decode_grows(codec):
    r, tr = codec["post"]["off"], codec["post"]["off_trace"]
    assert not r["gate_c_pass"], r
    assert tr["decoded_growth_max"] > 1.5, tr["decoded_growth_per_iteration"]


# ------------------------------------------------------------------------------ gate (d)
def test_gd_iteration0_matches_encoder_image(codec):
    for phase in ("pre", "post"):
        r = codec[phase]["on"]
        assert r["gate_d_pass"] and 0.5 <= r["iter0_range_ratio"] <= 2.0, (phase, r)
    assert codec["post"]["on"]["verdict"] == C.PASS


def test_gd_not_blind_pre_build_iteration0_out_of_range(codec):
    for phase in ("pre", "post"):
        r = codec[phase]["off"]
        assert not r["gate_d_pass"] and r["iter0_range_ratio"] > 2.0, (phase, r)


# ------------------------------------------------------------------------------ P probes
def test_p_each_knob_on_changes_what_it_controls(codec):
    agent, Z = codec["agent"], codec["Z"]
    hip = agent.hippocampal
    base = _pool_fingerprint(agent, Z, n_states=2)
    for knob in KNOBS_H:
        setattr(hip.config, knob, True)
        try:
            got = _pool_fingerprint(agent, Z, n_states=2)
        finally:
            setattr(hip.config, knob, False)
        assert any(not torch.equal(a0, a1) for (a0, _, _), (a1, _, _) in zip(base, got)), knob
    # waking_trainer_codec_enabled: the trained group moved and round-trips (GA/GB above);
    # here, its registration is the knob's only effect on construction.
    assert "codec" in agent.waking_trainer.members
