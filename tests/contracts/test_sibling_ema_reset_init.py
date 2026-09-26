"""Contracts for the sibling-EMA reset-init knobs: initialise the z_self,
shared-latent (z_beta / z_theta / z_delta) and SD-036 harm-stream EMAs from the
first encode at episode reset instead of from init_state()'s zeros.

Defect (sibling of use_zworld_ema_reset_init, REE_assembly
evidence/planning/zworld_ema_reset_init_build_20260926.md sec 4): init_state()
zeroes every one of these streams, and encode() then blends the reset tick
against that hard-zero prior:

    x(t0) = alpha_x * raw_x(t0) + (1 - alpha_x) * 0

so ||x(t0)|| = alpha_x * ||raw_x(t0)||, exactly -- alpha_self (default 0.3) for
z_self, the hard-coded alpha_shared = 0.3 for z_beta/z_theta/z_delta, and
gaba_state_alpha_z_harm_s (default 0.5) for the SD-036 z_harm blend when
gaba_harm_state_recurrence is on and harm_dim > 0.

Fix (default OFF, one knob per family, independent):
    use_zself_ema_reset_init   -- z_self legacy EMA
    use_shared_ema_reset_init  -- z_beta / z_theta / z_delta
    use_zharm_ema_reset_init   -- SD-036 harm-stream blend
When ON, the first encode() after reset (prev_state.timestamp == 0) sets that
family's EMA state to its instantaneous encode. Ticks t>=1 blend exactly as
before. use_zworld_ema_reset_init is untouched.

"raw" is read INDEPENDENTLY of the EMA arithmetic: forward hooks capture the
encoder module outputs inside the same encode() call (split_encoder -> z_self,
the last beta_encoder / theta_encoder call -> z_beta / z_theta, delta_encoder
-> z_delta, harm_encoder -> SD-010 z_harm). No assertion is on a sampled
action; the rollout uses scripted actions from a local generator.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import LatentStack
from ree_core.utils.config import LatentStackConfig, REEConfig


FLAGS = (
    "use_zself_ema_reset_init",
    "use_shared_ema_reset_init",
    "use_zharm_ema_reset_init",
)
SHARED = ("z_beta", "z_theta", "z_delta")
ALPHA_SHARED = 0.3  # hard-coded in LatentStack.encode()

# SD-036 harm config: lateral head harm_dim == SD-010 z_harm_dim so init_state's
# zero z_harm has the blend's shape (the case in which the zero prior is live).
# The MECH-099 lateral head indexes the env's hazard/contamination channels, so
# this config uses the real CausalGridWorldV2 observation dims.
_ENV = CausalGridWorldV2(size=8, num_hazards=3, num_resources=2, seed=1)
HARM_KW = dict(
    body_obs_dim=_ENV.body_obs_dim, world_obs_dim=_ENV.world_obs_dim,
    harm_dim=32, z_harm_dim=32, use_harm_stream=True,
    gaba_harm_state_recurrence=True,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stack(seed: int = 0, knobs=(), **kw) -> LatentStack:
    """Build a stack; knobs are set by attribute so that on a substrate lacking
    them the contract fails behaviourally rather than at construction."""
    cfg = LatentStackConfig(**kw)
    for k in knobs:
        setattr(cfg, k, True)
    torch.manual_seed(seed)
    return LatentStack(cfg)


class _Raw:
    """Forward hooks recording the instantaneous (pre-EMA) encoder outputs of
    the most recent encode() call."""

    def __init__(self, stack: LatentStack):
        self.v = {}
        hs = [
            stack.split_encoder.register_forward_hook(self._grab("z_self", 0)),
            stack.beta_encoder.register_forward_hook(self._grab("z_beta", 0)),
            stack.theta_encoder.register_forward_hook(self._grab("z_theta", 0)),
            stack.delta_encoder.register_forward_hook(self._grab("z_delta", 0)),
        ]
        if stack.harm_encoder is not None:
            hs.append(stack.harm_encoder.register_forward_hook(self._grab("z_harm", None)))
        self.handles = hs

    def _grab(self, name, idx):
        def hook(_m, _inp, out):
            o = out if idx is None else out[idx]
            self.v[name] = o.detach().clone()  # last call wins (second pass)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def _obs_seq(stack: LatentStack, n: int, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    d = stack.config.body_obs_dim + stack.config.world_obs_dim
    return [torch.randn(1, d, generator=g) for _ in range(n)]


def _harm_seq(stack: LatentStack, n: int, seed: int = 11):
    g = torch.Generator().manual_seed(seed)
    d = getattr(stack.config, "harm_obs_dim", 51)
    return [torch.rand(1, d, generator=g) for _ in range(n)]


def _run(stack: LatentStack, obs_seq, harm_seq=None, reset_every: int = 0):
    """Encode a fixed observation sequence (optional periodic reset, prev=None).
    Returns (states, raws) with raws[i] the hooked instantaneous encodes."""
    rec = _Raw(stack)
    states, raws = [], []
    prev = None
    try:
        for i, o in enumerate(obs_seq):
            if reset_every and i % reset_every == 0:
                prev = None
            ho = harm_seq[i] if harm_seq is not None else None
            with torch.no_grad():
                st = stack.encode(o, prev, harm_obs=ho)
            states.append(st)
            raws.append(dict(rec.v))
            prev = st.detach()
    finally:
        rec.close()
    return states, raws


def _digest(states) -> str:
    h = hashlib.sha256()
    for st in states:
        for t in (st.z_world, st.z_self, st.z_beta, st.z_theta, st.z_delta, st.z_harm):
            if t is not None:
                h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def _close(a: torch.Tensor, b: torch.Tensor) -> None:
    assert abs(float(a.norm()) - float(b.norm())) < 1e-6
    assert torch.allclose(a, b, atol=1e-6, rtol=0.0)


# ---------------------------------------------------------------------------
# C0: every knob exists at every config site and defaults OFF
# ---------------------------------------------------------------------------

def test_c0_default_off_and_from_dims_lands_each_knob():
    env = CausalGridWorldV2(size=6, num_hazards=1, num_resources=1, seed=1)
    base = dict(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, alpha_world=0.3,
    )
    default = REEConfig.from_dims(**base).latent
    for f in FLAGS:
        assert getattr(LatentStackConfig(), f) is False
        assert getattr(default, f) is False
        on = REEConfig.from_dims(**base, **{f: True}).latent
        assert getattr(on, f) is True
        # independence: turning one family on leaves the others OFF
        for g in FLAGS:
            if g != f:
                assert getattr(on, g) is False
    # the existing z_world knob is a separate switch
    assert REEConfig.from_dims(**base, use_zself_ema_reset_init=True).latent.use_zworld_ema_reset_init is False


# ---------------------------------------------------------------------------
# C1: OFF is bit-identical to the default config and keeps the legacy
#     alpha-scaled reset tick for every family
# ---------------------------------------------------------------------------

def test_c1_off_bit_identical_to_default():
    off_kw = {f: False for f in FLAGS}
    for kw in (dict(alpha_self=0.3), dict(alpha_self=0.9), dict(alpha_self=0.3, **HARM_KW)):
        a = _stack(seed=5, **kw)
        b = _stack(seed=5, **kw, **off_kw)
        seq = _obs_seq(a, 24)
        hs = _harm_seq(a, 24) if "use_harm_stream" in kw else None
        da = _digest(_run(a, seq, hs, reset_every=8)[0])
        db = _digest(_run(b, seq, hs, reset_every=8)[0])
        assert da == db


def test_c1_off_keeps_legacy_reset_scaling():
    s = _stack(seed=2, alpha_self=0.3, **HARM_KW)
    states, raws = _run(s, _obs_seq(s, 1), _harm_seq(s, 1))
    st, raw = states[0], raws[0]
    _close(st.z_self, 0.3 * raw["z_self"])
    for n in SHARED:
        _close(getattr(st, n), ALPHA_SHARED * raw[n])
    _close(st.z_harm, 0.5 * raw["z_harm"])


# ---------------------------------------------------------------------------
# C2: ON -> ||x(t0)|| == ||raw_x(t0)|| for each family, at every reset
# ---------------------------------------------------------------------------

def _agent_rollout(knob, seed: int, n_episodes: int, ep_len: int, alpha_self: float = 0.3):
    """Sense-only agent rollout with scripted actions (local generator; the
    global RNG is seeded once for weight init only)."""
    torch.manual_seed(seed)
    env = CausalGridWorldV2(size=8, num_hazards=3, num_resources=2, hazard_harm=0.5, seed=seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32,
        alpha_world=0.3, alpha_self=alpha_self,
    )
    if knob is not None:
        setattr(cfg.latent, knob, True)
    agent = REEAgent(cfg)
    rec = _Raw(agent.latent_stack)
    rng = np.random.default_rng(seed)
    rows = []
    try:
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
                rows.append((t, lat.detach(), dict(rec.v)))
                _flat, _h, done, _info, od = env.step(int(rng.integers(env.action_dim)))
                if done:
                    break
    finally:
        rec.close()
    return rows


def test_c2_zself_on_reset_tick_equals_raw():
    for alpha in (0.3, 0.9):
        rows = _agent_rollout("use_zself_ema_reset_init", seed=3, n_episodes=5, ep_len=4, alpha_self=alpha)
        resets = [(lat, raw) for t, lat, raw in rows if t == 0]
        assert len(resets) == 5
        for lat, raw in resets:
            _close(lat.z_self, raw["z_self"])
            # independence: the shared latents keep the legacy scaled reset tick
            _close(lat.z_beta, ALPHA_SHARED * raw["z_beta"])


def test_c2_shared_on_reset_tick_equals_raw():
    rows = _agent_rollout("use_shared_ema_reset_init", seed=4, n_episodes=5, ep_len=4)
    resets = [(lat, raw) for t, lat, raw in rows if t == 0]
    assert len(resets) == 5
    for lat, raw in resets:
        for n in SHARED:
            _close(getattr(lat, n), raw[n])
        _close(lat.z_self, 0.3 * raw["z_self"])  # independence


def test_c2_zharm_on_reset_tick_equals_raw():
    s = _stack(seed=9, knobs=("use_zharm_ema_reset_init",), **HARM_KW)
    states, raws = _run(s, _obs_seq(s, 12), _harm_seq(s, 12), reset_every=4)
    for i in (0, 4, 8):
        assert states[i].timestamp == 1
        _close(states[i].z_harm, raws[i]["z_harm"])
        _close(states[i].z_self, 0.3 * raws[i]["z_self"])  # independence


# ---------------------------------------------------------------------------
# C3: ON leaves the t>=1 blend unchanged -- the whole ON trajectory is
#     x(0) = raw(0), x(t) = alpha*raw(t) + (1-alpha)*x(t-1)
# ---------------------------------------------------------------------------

def test_c3_on_trajectory_is_reset_init_then_legacy_ema():
    s = _stack(seed=4, knobs=FLAGS, alpha_self=0.3, **HARM_KW)
    states, raws = _run(s, _obs_seq(s, 6), _harm_seq(s, 6))
    fams = [("z_self", 0.3), ("z_harm", 0.5)] + [(n, ALPHA_SHARED) for n in SHARED]
    for name, a in fams:
        expect = raws[0][name]
        _close(getattr(states[0], name), expect)
        for t in range(1, len(states)):
            expect = a * raws[t][name] + (1 - a) * getattr(states[t - 1], name)
            _close(getattr(states[t], name), expect)


def test_c3_on_matches_off_from_same_nonreset_prev():
    on = _stack(seed=6, alpha_self=0.3, **HARM_KW, **{f: True for f in FLAGS})
    off = _stack(seed=6, alpha_self=0.3, **HARM_KW, **{f: False for f in FLAGS})
    seq, hs = _obs_seq(on, 2), _harm_seq(on, 2)
    with torch.no_grad():
        prev = off.encode(seq[0], harm_obs=hs[0]).detach()  # timestamp 1: not a reset state
        assert prev.timestamp == 1
        a = on.encode(seq[1], prev, harm_obs=hs[1])
        b = off.encode(seq[1], prev, harm_obs=hs[1])
    for n in ("z_world", "z_self", "z_beta", "z_theta", "z_delta", "z_harm"):
        assert torch.equal(getattr(a, n), getattr(b, n)), n
