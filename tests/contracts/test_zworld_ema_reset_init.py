"""Contracts for use_zworld_ema_reset_init: initialise the SD-008 z_world EMA
from the first observation at episode reset instead of from zeros.

Defect (REE_assembly/evidence/planning/zworld_near_collapse_rootcause_20260926.md,
U4, 1294f6edf5c): LatentStack.init_state() zeroes z_world at reset, SD-007
reafference correction is skipped at t=0, so the EMA blend in encode()
degenerates on the first tick after every reset to

    z_world(t0) = alpha_world * raw(t0) + (1 - alpha_world) * 0

i.e. ||z_world(t0)|| = alpha_world * ||raw(t0)||, exactly. At the deployed
alpha_world=0.3 that manufactures a 0-1st-percentile ||z_world|| spike on 100%
of reset ticks (21.9x enrichment) out of an unremarkable raw encode.

Fix (default OFF): when LatentStackConfig.use_zworld_ema_reset_init is True, the
first encode() after reset (prev_state.timestamp == 0) sets the EMA state to the
instantaneous encode: z_world(t0) = raw(t0). Ticks t>=1 blend exactly as before.

Pinned here: the knob exists at every config site and defaults OFF; OFF is
bit-identical to the default config and keeps the legacy alpha-scaled reset tick;
ON makes ||z(t0)|| == ||raw(t0)|| at every reset (agent level, several resets);
ON leaves the t>=1 blend unchanged in form; ON also covers the MECH-157
mode-routing branch; and a reduced U4 readout -- with ON, reset ticks are no
longer over-represented in the 0-1st-percentile ||z_world|| bucket.
Assertions stay upstream of any sampled action (the rollout uses scripted
actions from a local generator and never touches the global RNG).
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import LatentStack
from ree_core.utils.config import LatentStackConfig, REEConfig


FLAG = "use_zworld_ema_reset_init"


def _stack(seed: int = 0, **kw) -> LatentStack:
    torch.manual_seed(seed)
    return LatentStack(LatentStackConfig(**kw))


def _obs_seq(stack: LatentStack, n: int, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    d = stack.config.body_obs_dim + stack.config.world_obs_dim
    return [torch.randn(1, d, generator=g) for _ in range(n)]


def _run_stack(stack: LatentStack, obs_seq, reset_every: int = 0):
    """Encode a fixed observation sequence; optional periodic reset (prev=None)."""
    states = []
    prev = None
    for i, o in enumerate(obs_seq):
        if reset_every and i % reset_every == 0:
            prev = None
        with torch.no_grad():
            st = stack.encode(o, prev)
        states.append(st)
        prev = st.detach()
    return states


def _digest(states) -> str:
    h = hashlib.sha256()
    for st in states:
        for t in (st.z_world, st.z_self, st.z_beta, st.z_theta, st.z_delta):
            h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# C0: the knob exists at every config site and defaults OFF
# ---------------------------------------------------------------------------

def test_c0_default_off_and_from_dims_lands_knob():
    assert getattr(LatentStackConfig(), FLAG) is False
    env = CausalGridWorldV2(size=6, num_hazards=1, num_resources=1, seed=1)
    base = dict(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, alpha_world=0.3,
    )
    assert getattr(REEConfig.from_dims(**base).latent, FLAG) is False
    assert getattr(REEConfig.from_dims(**base, use_zworld_ema_reset_init=True).latent, FLAG) is True


# ---------------------------------------------------------------------------
# C1: OFF is bit-identical and keeps the legacy alpha-scaled reset tick
# ---------------------------------------------------------------------------

def test_c1_off_bit_identical_to_default():
    for alpha in (0.3, 0.9):
        a = _stack(seed=5, alpha_world=alpha)
        b = _stack(seed=5, alpha_world=alpha, use_zworld_ema_reset_init=False)
        seq = _obs_seq(a, 24)
        assert _digest(_run_stack(a, seq, reset_every=8)) == _digest(_run_stack(b, seq, reset_every=8))


def test_c1_off_keeps_legacy_reset_scaling():
    s = _stack(seed=2, alpha_world=0.3, use_zworld_ema_reset_init=False)
    st = _run_stack(s, _obs_seq(s, 1))[0]
    ratio = float(st.z_world.norm() / st.z_world_raw.norm())
    assert abs(ratio - 0.3) < 1e-6


# ---------------------------------------------------------------------------
# C2: ON -> ||z(t0)|| == ||raw(t0)|| at every reset (agent level)
# ---------------------------------------------------------------------------

def _agent_rollout(flag: bool, alpha: float, seed: int, n_episodes: int, ep_len: int):
    """Sense-only rollout with scripted actions (local generator; global RNG
    is seeded once for weight init only, never drawn from by the driver)."""
    torch.manual_seed(seed)
    env = CausalGridWorldV2(size=8, num_hazards=3, num_resources=2, hazard_harm=0.5, seed=seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32,
        alpha_world=alpha, use_zworld_ema_reset_init=flag,
    )
    agent = REEAgent(cfg)
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(n_episodes):
        env = CausalGridWorldV2(
            size=8, num_hazards=3, num_resources=2, hazard_harm=0.5, seed=1000 * seed + ep
        )
        _flat, od = env.reset()
        agent.reset()
        for t in range(ep_len):
            with torch.no_grad():
                lat = agent.sense(
                    torch.as_tensor(od["body_state"]).float().unsqueeze(0),
                    torch.as_tensor(od["world_state"]).float().unsqueeze(0),
                )
            rows.append(dict(
                t=t,
                zw=float(lat.z_world.norm()),
                zwr=float(lat.z_world_raw.norm()),
                z=lat.z_world.detach().clone(),
                raw=lat.z_world_raw.detach().clone(),
            ))
            _flat, _h, done, _info, od = env.step(int(rng.integers(env.action_dim)))
            if done:
                break
    return rows


def test_c2_on_reset_tick_norm_equals_raw_norm():
    for alpha in (0.3, 0.9):
        rows = _agent_rollout(True, alpha, seed=3, n_episodes=5, ep_len=6)
        resets = [r for r in rows if r["t"] == 0]
        assert len(resets) == 5
        for r in resets:
            assert abs(r["zw"] - r["zwr"]) < 1e-6
            assert torch.allclose(r["z"], r["raw"], atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# C3: ON leaves the t>=1 blend unchanged in form
# ---------------------------------------------------------------------------

def test_c3_on_t_ge_1_blend_is_the_legacy_ema():
    alpha = 0.3
    s = _stack(seed=4, alpha_world=alpha, use_zworld_ema_reset_init=True)
    states = _run_stack(s, _obs_seq(s, 6))
    for t in range(1, len(states)):
        expect = alpha * states[t].z_world_raw + (1 - alpha) * states[t - 1].z_world
        assert torch.allclose(states[t].z_world, expect, atol=1e-6, rtol=0.0)


def test_c3_on_matches_off_from_same_nonreset_prev():
    on = _stack(seed=6, alpha_world=0.3, use_zworld_ema_reset_init=True)
    off = _stack(seed=6, alpha_world=0.3, use_zworld_ema_reset_init=False)
    seq = _obs_seq(on, 3)
    with torch.no_grad():
        prev = off.encode(seq[0]).detach()  # timestamp 1: not a reset state
        assert prev.timestamp == 1
        a = on.encode(seq[1], prev)
        b = off.encode(seq[1], prev)
    assert torch.equal(a.z_world, b.z_world)
    assert torch.equal(a.z_self, b.z_self)


def test_c3_on_covers_mode_routing_branch():
    s = _stack(
        seed=8, alpha_world=0.3, use_zworld_ema_reset_init=True,
        use_mode_precision_routing=True,
    )
    (o,) = _obs_seq(s, 1)
    with torch.no_grad():
        st = s.encode(o, None, operating_mode={"internal_replay": 1.0})
    # no E2 anchor supplied -> no generative pull; reset tick = instantaneous encode
    assert torch.allclose(st.z_world, st.z_world_raw, atol=1e-6, rtol=0.0)
    assert st.mode_precision_diag["alpha_eff"] == 1.0


# ---------------------------------------------------------------------------
# C4: reduced U4 readout -- reset ticks no longer dominate the collapse bucket
# ---------------------------------------------------------------------------

def _reset_enrichment(rows) -> float:
    zw = torch.tensor([r["zw"] for r in rows])
    p1 = float(torch.quantile(zw, 0.01))
    bucket = [r for r in rows if r["zw"] <= p1]
    base = sum(1 for r in rows if r["t"] == 0) / len(rows)
    in_bucket = sum(1 for r in bucket if r["t"] == 0) / len(bucket)
    return in_bucket / base


def test_c4_u4_reset_enrichment_below_3x_when_on():
    kw = dict(alpha=0.3, seed=106, n_episodes=10, ep_len=30)
    off = _reset_enrichment(_agent_rollout(False, **kw))
    on = _reset_enrichment(_agent_rollout(True, **kw))
    assert off >= 3.0, f"readout not sensitive: OFF enrichment {off:.2f}"
    assert on < 3.0, f"ON reset enrichment {on:.2f} >= 3x"
