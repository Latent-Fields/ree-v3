"""Contracts for MECH-157 option A: mode-conditioned precision routing on z_world.

User decision 2026-09-25 (rec-20260925-6231be2b), design in
REE_assembly/evidence/planning/claim_synthesis_MECH-157_20260925.md as revised
by its red-team (claim_synthesis_MECH-157-039_redteam_20260925.md):

  * SENSORY GAIN  = per-mode ABSOLUTE alpha_world (external_task >= 0.9 so
    SD-008 holds in the external regime).
  * GENERATIVE DRIVE = per-mode coupling g_m pulling z_world toward the E2
    forward prediction E2.world_forward(z_world_prev, a_prev) -- the SELF-1
    self_e1_anchor pattern on the world side.

    z = (1-g) * (alpha*z_obs + (1-alpha)*z_prev) + g * z_pred

What is pinned here: OFF is bit-identical even when a mode and an anchor are
supplied; ON with no mode is the legacy blend; ON external_task equals a plain
alpha_world=0.9 stack; the update formula is exact; a mode change changes BOTH
the z_world update and the E2 pull (the reach contract the chip asks for); the
anchor magnitude bound holds; from_dims() lands every knob; and the agent-level
plumbing (override / coordinator mode -> E2 anchor -> encode) runs end-to-end.
Assertions stay upstream of the sampled action (z_world, diag), never on an
exact committed action -- torch.multinomial is not portable across machine
classes.
"""

from __future__ import annotations

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import LatentStack
from ree_core.utils.config import LatentStackConfig, REEConfig


REPLAY = {"internal_replay": 1.0}
EXTERNAL = {"external_task": 1.0}


def _stack(seed: int = 0, **kw) -> LatentStack:
    torch.manual_seed(seed)
    return LatentStack(LatentStackConfig(**kw))


def _obs(stack: LatentStack, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, stack.config.body_obs_dim + stack.config.world_obs_dim, generator=g)


def _prev(stack: LatentStack, seed: int = 3):
    """A non-trivial previous state from a plain (legacy) encode."""
    return stack.encode(_obs(stack, seed))


def _anchor(stack: LatentStack, seed: int = 11, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return scale * torch.randn(1, stack.config.world_dim, generator=g)


# ---------------------------------------------------------------------------
# C1: OFF (default) is bit-identical, even with a mode and an anchor supplied
# ---------------------------------------------------------------------------

def test_c1_default_is_off():
    cfg = LatentStackConfig()
    assert cfg.use_mode_precision_routing is False
    assert cfg.mode_alpha_world["external_task"] >= 0.9  # SD-008 floor in external


def test_c1_off_ignores_mode_and_anchor():
    s = _stack()
    prev = _prev(s)
    obs = _obs(s)
    plain = s.encode(obs, prev)
    fed = s.encode(obs, prev, operating_mode=REPLAY, world_e2_anchor=_anchor(s))
    assert torch.equal(plain.z_world, fed.z_world)
    assert torch.equal(plain.z_self, fed.z_self)
    assert fed.mode_precision_diag is None


# ---------------------------------------------------------------------------
# C2: ON with no mode supplied -> the legacy (mode-unconditioned) blend exactly
# ---------------------------------------------------------------------------

def test_c2_on_without_mode_is_legacy():
    off = _stack(0)
    on = _stack(0, use_mode_precision_routing=True)
    prev = _prev(off)
    obs = _obs(off)
    a = off.encode(obs, prev)
    b = on.encode(obs, prev, operating_mode=None, world_e2_anchor=_anchor(off))
    assert torch.equal(a.z_world, b.z_world)
    assert b.mode_precision_diag is None


# ---------------------------------------------------------------------------
# C3: ON + external_task == a plain alpha_world=0.9 stack (absolute alpha)
# ---------------------------------------------------------------------------

def test_c3_external_task_is_absolute_alpha_09():
    ref = _stack(0, alpha_world=0.9)
    on = _stack(0, use_mode_precision_routing=True)  # base alpha_world stays 0.3
    prev = _prev(ref)
    obs = _obs(ref)
    a = ref.encode(obs, prev)
    b = on.encode(obs, prev, operating_mode=EXTERNAL, world_e2_anchor=_anchor(ref))
    assert torch.allclose(a.z_world, b.z_world, atol=1e-7)
    d = b.mode_precision_diag
    assert d["active"] is True
    assert abs(d["alpha_eff"] - 0.9) < 1e-9
    assert d["gen_coupling"] == 0.0 and d["anchor_present"] is False


# ---------------------------------------------------------------------------
# C4: exact update formula, and a mode change moves BOTH terms (reach)
# ---------------------------------------------------------------------------

def test_c4_update_formula_exact():
    alpha, g = 0.2, 0.6  # shipped internal_replay defaults
    ref = _stack(0, alpha_world=alpha)
    on = _stack(0, use_mode_precision_routing=True, mode_world_e2_anchor_norm_cap=0.0)
    prev = _prev(ref)
    obs = _obs(ref)
    anchor = _anchor(ref)
    legacy = ref.encode(obs, prev).z_world
    got = on.encode(obs, prev, operating_mode=REPLAY, world_e2_anchor=anchor)
    expected = (1.0 - g) * legacy + g * anchor
    assert torch.allclose(got.z_world, expected, atol=1e-6)
    d = got.mode_precision_diag
    assert d["anchor_present"] is True
    assert abs(d["obs_weight"] - alpha * (1 - g)) < 1e-9
    assert abs(d["prior_weight"] - (1 - alpha) * (1 - g)) < 1e-9
    assert abs(d["pred_weight"] - g) < 1e-9


def test_c4_mode_change_changes_update_and_pull():
    on = _stack(0, use_mode_precision_routing=True)
    prev = _prev(on)
    obs = _obs(on)
    anchor = _anchor(on)
    ext = on.encode(obs, prev, operating_mode=EXTERNAL, world_e2_anchor=anchor)
    rep = on.encode(obs, prev, operating_mode=REPLAY, world_e2_anchor=anchor)
    # z_world update differs by mode ...
    assert not torch.allclose(ext.z_world, rep.z_world)
    # ... through both scalars: sensory gain AND the E2 pull.
    assert ext.mode_precision_diag["alpha_eff"] > rep.mode_precision_diag["alpha_eff"]
    assert ext.mode_precision_diag["pred_weight"] == 0.0
    assert rep.mode_precision_diag["pred_weight"] > 0.0
    # The pull is live: the same replay mode with no anchor lands elsewhere.
    rep_no_anchor = on.encode(obs, prev, operating_mode=REPLAY, world_e2_anchor=None)
    assert not torch.allclose(rep.z_world, rep_no_anchor.z_world)
    assert rep_no_anchor.mode_precision_diag["anchor_present"] is False


def test_c4_soft_mixture_is_weighted_sum():
    on = _stack(0, use_mode_precision_routing=True)
    prev = _prev(on)
    d = on.encode(
        _obs(on), prev,
        operating_mode={"external_task": 0.5, "internal_planning": 0.5},
        world_e2_anchor=_anchor(on),
    ).mode_precision_diag
    assert abs(d["alpha_eff"] - 0.5 * (0.9 + 0.5)) < 1e-9
    assert abs(d["gen_coupling"] - 0.5 * (0.0 + 0.3)) < 1e-9


# ---------------------------------------------------------------------------
# C5: anchor magnitude bound + non-finite / ill-shaped anchors are refused
# ---------------------------------------------------------------------------

def test_c5_anchor_norm_cap_bounds_pull():
    on = _stack(0, use_mode_precision_routing=True, mode_world_e2_anchor_norm_cap=1.0)
    prev = _prev(on)
    obs = _obs(on)
    huge = _anchor(on, scale=1.0e6)
    st = on.encode(obs, prev, operating_mode=REPLAY, world_e2_anchor=huge)
    assert st.mode_precision_diag["anchor_capped"] is True
    assert torch.isfinite(st.z_world).all()
    # Bounded: z_world stays on the scale of the observation, not the anchor.
    assert float(st.z_world.detach().norm()) < 1.0e3


def test_c5_nonfinite_or_misshaped_anchor_is_ignored():
    on = _stack(0, use_mode_precision_routing=True)
    prev = _prev(on)
    obs = _obs(on)
    base = on.encode(obs, prev, operating_mode=REPLAY, world_e2_anchor=None)
    bad = torch.full((1, on.config.world_dim), float("nan"))
    st = on.encode(obs, prev, operating_mode=REPLAY, world_e2_anchor=bad)
    assert torch.equal(st.z_world, base.z_world)
    assert st.mode_precision_diag["anchor_present"] is False
    wrong = torch.randn(1, on.config.world_dim + 1)
    st2 = on.encode(obs, prev, operating_mode=REPLAY, world_e2_anchor=wrong)
    assert torch.equal(st2.z_world, base.z_world)


# ---------------------------------------------------------------------------
# C6: from_dims() lands every knob (it silently swallows unknown kwargs)
# ---------------------------------------------------------------------------

def test_c6_from_dims_lands_all_knobs():
    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4, alpha_world=0.9,
        use_mode_precision_routing=True,
        mode_alpha_world={"external_task": 0.95, "internal_replay": 0.1},
        mode_world_e2_coupling={"internal_replay": 0.7},
        mode_world_e2_anchor_norm_cap=0.5,
    )
    lat = cfg.latent
    assert lat.use_mode_precision_routing is True
    assert lat.mode_alpha_world == {"external_task": 0.95, "internal_replay": 0.1}
    assert lat.mode_world_e2_coupling == {"internal_replay": 0.7}
    assert lat.mode_world_e2_anchor_norm_cap == 0.5
    default = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4, alpha_world=0.9)
    assert default.latent.use_mode_precision_routing is False
    assert default.latent.mode_alpha_world == LatentStackConfig().mode_alpha_world


# ---------------------------------------------------------------------------
# C7: agent-level plumbing (mode source -> E2 anchor -> encode), end to end
# ---------------------------------------------------------------------------

def _agent(seed: int = 0, **flags):
    torch.manual_seed(seed)
    env = CausalGridWorldV2(seed=seed, size=6, num_hazards=2, num_resources=2,
                            use_proxy_fields=True)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=16, world_dim=16, alpha_world=0.9, **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent, env


def _run(agent, env, override, n=8, seed=0):
    torch.manual_seed(seed)
    _flat, od = env.reset()
    agent.reset()
    agent.mode_precision_routing_override = override
    zs = []
    agent._m157_test_mode_before_last = None
    for _ in range(n):
        if agent.salience is not None:
            agent._m157_test_mode_before_last = dict(agent.salience.operating_mode)
        b, w = od["body_state"], od["world_state"]
        if b.dim() == 1:
            b, w = b.unsqueeze(0), w.unsqueeze(0)
        with torch.no_grad():
            a = agent.act_with_split_obs(b, w)
        zs.append(agent._current_latent.z_world.detach().clone())
        _flat, _h, done, _info, od = env.step(a)
        if done:
            _flat, od = env.reset()
    return torch.cat(zs), agent._current_latent.mode_precision_diag


def test_c7_agent_off_override_is_inert():
    agent, env = _agent()
    z_plain, d_plain = _run(agent, env, override=None)
    agent2, env2 = _agent()
    z_over, d_over = _run(agent2, env2, override=REPLAY)
    assert d_plain is None and d_over is None
    assert torch.equal(z_plain, z_over)


def test_c7_agent_on_mode_reaches_z_world_and_e2_pull():
    agent, env = _agent(use_mode_precision_routing=True)
    z_ext, d_ext = _run(agent, env, override=EXTERNAL)
    agent2, env2 = _agent(use_mode_precision_routing=True)
    z_rep, d_rep = _run(agent2, env2, override=REPLAY)
    assert d_ext is not None and d_ext["active"] is True
    assert d_ext["pred_weight"] == 0.0
    # Replay: the E2 anchor was computed from (z_world_prev, a_prev) and applied.
    assert d_rep["anchor_present"] is True and d_rep["pred_weight"] > 0.0
    assert d_rep["dist_to_pred"] is not None
    # A mode change changes the live z_world trajectory.
    assert not torch.allclose(z_ext, z_rep)
    assert torch.isfinite(z_rep).all()


def test_c7_agent_on_reads_coordinator_mode_when_no_override():
    agent, env = _agent(use_mode_precision_routing=True, use_salience_coordinator=True)
    assert agent.salience is not None
    _z, d = _run(agent, env, override=None)
    assert d is not None and d["active"] is True
    om = agent._m157_test_mode_before_last
    expected_alpha = sum(
        float(p) * agent.config.latent.mode_alpha_world.get(m, agent.config.latent.alpha_world)
        for m, p in om.items()
    ) / sum(float(p) for p in om.values())
    # The coordinator is read at its PREVIOUS tick: compare against the
    # vector it held going into the last sense().
    assert abs(d["alpha_eff"] - expected_alpha) < 1e-6
