"""Contract tests for SD-PP-4: provenance-conditioned consolidation gain.

Contract: REE_assembly/docs/architecture/precision_provenance_substrate_spec.md
section 5. Eleven contracts, numbered as the spec numbers them.

  T1  bounds -- every gain lies in [gain_min, gain_max].
  T2  monotone in evidence precision at matched pe (the Kalman factor K).
  T3  ANTI-SELF-SEALING -- a reliably-falsified HIGH-confidence prediction is
      never protected by having been confident: its gain is >= the gain of the
      identical contradiction held at LOW historical precision. Historical
      precision enters only through r_i, and r_i >= 1 always.
  T4  noisy contradiction -- large pe that is fully explained by evidence noise
      (noise_gain * evidence_variance_z >= pe) collapses to gain_min.
  T5  residual_only = clip(gain_max*sqrt(res/v_ref)) from the CURRENT residual only; global is constant.
  T6  provenance_nohist <= provenance rowwise (r >= 1 is the whole difference).
  T7  missing packet -> 1.0, and counted in n_missing.
  T8  config validation.
  T9  weighted_row_loss -- gradient flows through the loss only, never the gain.
  T10 consolidator liveness -- module_step_scale MOVES the displacement
      (strictly increasing in the scale at a pinned seed), and the no-kwargs
      path is unchanged (key-identical metrics, and scale 1.0 is a numerical
      no-op). This is the MECH-572 liveness pin: without the hook the fresh
      per-call Adam holds displacement at its bound regardless of the loss
      weighting.
  T11 reduction="none" rows mean equals reduction="mean".

The packets are duck-typed SimpleNamespace objects on purpose: SD-PP-3's
ReplayProvenancePacket is a sibling build and the rule must not depend on it.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.sleep.cross_module_consolidation import (  # noqa: E402
    CrossModuleConsolidator,
    CrossModuleConsolidatorConfig,
)
from ree_core.sleep.provenance_gain import (  # noqa: E402
    GAIN_MODES,
    ProvenanceGainConfig,
    compute_provenance_gains,
    weighted_row_loss,
)

BASE_METRIC_KEYS = {
    "n_updates",
    "n_traces",
    "n_cross_module_traces",
    "cross_module_replay_share",
    "interleaved",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _packet(
    evidence_precision_z: float,
    evidence_variance_z: float,
    pe: float,
    surprise: float,
    has_prev: bool = True,
):
    """A duck-typed stand-in for SD-PP-3's ReplayProvenancePacket."""
    return types.SimpleNamespace(
        evidence_precision_z=float(evidence_precision_z),
        evidence_variance_z=float(evidence_variance_z),
        pe=float(pe),
        surprise=float(surprise),
        has_prev=bool(has_prev),
    )


# ----------------------------------------------------------------------
# T1 bounds
# ----------------------------------------------------------------------
def test_t1_gains_bounded():
    cfg = ProvenanceGainConfig(mode="provenance")
    packets = [
        _packet(1e-6, 1e2, 1e-9, 1e-9),      # nothing at all -> floor
        _packet(1e6, 1e-9, 1e6, 1e9),        # everything at once -> ceiling
        _packet(10.0, 1e-3, 0.05, 2.0),      # ordinary
        _packet(0.5, 5e-3, 0.02, 0.4),       # weak
    ]
    gains, diag = compute_provenance_gains(packets, pi_cur=10.0,
                                           per_row_loss=None, config=cfg)
    assert gains.dtype == torch.float32
    assert gains.requires_grad is False
    assert tuple(gains.shape) == (4,)
    assert bool(torch.isfinite(gains).all())
    # float32 storage, so the bounds hold to float32 resolution.
    assert float(gains.min()) >= cfg.gain_min - 1e-7
    assert float(gains.max()) <= cfg.gain_max + 1e-7
    # The extremes really are reached, so the bound is not vacuous.
    assert float(gains[0]) == pytest.approx(cfg.gain_min)
    assert float(gains[1]) == pytest.approx(cfg.gain_max)
    # Diagnostics contract.
    for key in ("gain_mean", "gain_min", "gain_max", "gain_sd", "n_missing",
                "k_mean", "m_mean", "r_mean", "r_max", "surprise_max",
                "pi_cur", "mode"):
        assert key in diag, key
    assert diag["n_missing"] == 0.0
    assert diag["mode"] == float(GAIN_MODES.index("provenance"))
    assert diag["pi_cur"] == 10.0
    assert diag["gain_min"] == pytest.approx(float(gains.min()))
    assert diag["gain_max"] == pytest.approx(float(gains.max()))


# ----------------------------------------------------------------------
# T2 monotone in evidence precision at matched pe
# ----------------------------------------------------------------------
def test_t2_monotone_in_evidence_precision():
    cfg = ProvenanceGainConfig(mode="provenance")
    # pe, evidence_variance_z and surprise held FIXED; only the evidence
    # precision (the Kalman numerator) moves.
    packets = [_packet(ev, 1e-4, 0.002, 1.0) for ev in (1.0, 10.0, 100.0)]
    gains, diag = compute_provenance_gains(packets, pi_cur=10.0,
                                           per_row_loss=None, config=cfg)
    g = [float(x) for x in gains]
    assert g[0] < g[1] < g[2], g
    # unclipped, so this is the rule and not the bound
    assert cfg.gain_min < g[0] and g[2] < cfg.gain_max
    # K = ev / (ev + pi_cur)
    assert diag["k_mean"] == pytest.approx(
        ((1.0 / 11.0) + (10.0 / 20.0) + (100.0 / 110.0)) / 3.0
    )
    # And higher CURRENT precision needs stronger evidence for the same change.
    hi_pi, _ = compute_provenance_gains(packets, pi_cur=1000.0,
                                        per_row_loss=None, config=cfg)
    assert float(hi_pi[2]) < g[2]


# ----------------------------------------------------------------------
# T3 anti-self-sealing
# ----------------------------------------------------------------------
def test_t3_high_historical_precision_is_not_protective():
    cfg = ProvenanceGainConfig(mode="provenance")
    ev_prec, ev_var, pe = 50.0, 1e-4, 0.003
    # The SAME contradiction (same evidence, same pe) read under a high vs a
    # low historical precision. surprise = pi_hist * innovation, so pi_hist
    # shows up only as a larger surprise. pe is chosen to leave the row in the
    # UNCLIPPED band -- at the clip the comparison would be vacuous.
    innovation = pe - cfg.noise_gain * ev_var
    low = _packet(ev_prec, ev_var, pe, surprise=180.0 * innovation)
    high = _packet(ev_prec, ev_var, pe, surprise=970.0 * innovation)
    gains, diag = compute_provenance_gains([low, high], pi_cur=5.0,
                                           per_row_loss=None, config=cfg)
    assert float(gains[0]) < cfg.gain_max, "fixture must stay off the clip"
    assert float(gains[1]) < cfg.gain_max, "fixture must stay off the clip"
    assert float(gains[1]) >= float(gains[0])
    assert float(gains[1]) > float(gains[0]), "reopen factor must be live here"
    # r is never a shrinking multiplier, in either row.
    assert diag["r_max"] >= 1.0
    # Structural version of the same claim: sweep pi_hist upward and confirm
    # the gain never falls.
    sweep = [
        _packet(ev_prec, ev_var, pe, surprise=pi_hist * innovation)
        for pi_hist in (1.0, 10.0, 100.0, 500.0, 1e3, 1e5)
    ]
    swept, _ = compute_provenance_gains(sweep, pi_cur=5.0,
                                        per_row_loss=None, config=cfg)
    vals = [float(x) for x in swept]
    assert all(b >= a for a, b in zip(vals, vals[1:])), vals
    # reopen_max caps it rather than letting confidence run away
    assert max(vals) <= cfg.gain_max


# ----------------------------------------------------------------------
# T4 noisy contradiction -> gain_min
# ----------------------------------------------------------------------
def test_t4_noisy_contradiction_gets_gain_min():
    cfg = ProvenanceGainConfig(mode="provenance")
    # pe is LARGE, but the evidence channel is so unreliable that
    # noise_gain * evidence_variance_z fully accounts for it -> no epistemic
    # innovation -> m == 0 -> gain clipped to the floor.
    noisy = _packet(
        evidence_precision_z=1.0 / 5.0,
        evidence_variance_z=5.0,
        pe=4.0,                       # < noise_gain (2.0) * 5.0 == 10.0
        surprise=50.0,                # even a huge surprise cannot rescue it
    )
    gains, diag = compute_provenance_gains([noisy], pi_cur=1.0,
                                           per_row_loss=None, config=cfg)
    assert float(gains[0]) == pytest.approx(cfg.gain_min)
    assert diag["m_mean"] == 0.0
    # gain_sd on a single row is the population sd -> 0.0, never nan
    assert diag["gain_sd"] == 0.0
    # The SAME pe with reliable evidence is not floored.
    clean = _packet(1.0 / 1e-4, 1e-4, 4.0, 50.0)
    clean_gains, _ = compute_provenance_gains([clean], pi_cur=1.0,
                                              per_row_loss=None, config=cfg)
    assert float(clean_gains[0]) > cfg.gain_min


# ----------------------------------------------------------------------
# T5 residual_only / global control modes
# ----------------------------------------------------------------------
def test_t5_control_modes_budget():
    losses = torch.tensor([0.5, 1.5, 2.0, 4.0])
    residual = torch.tensor([1e-6, 1e-4, 1e-2, 1.0])
    packets = [_packet(10.0, 1e-4, 0.01, 1.0) for _ in range(4)]

    # residual_only: g = clip(gain_max * sqrt(res / v_ref), gain_min, gain_max);
    # reads ONLY the current residual (no packet, no precision, no global_scale).
    cfg_res = ProvenanceGainConfig(mode="residual_only", global_scale=0.25)
    g_res, d_res = compute_provenance_gains(packets, pi_cur=10.0,
                                            per_row_loss=losses,
                                            config=cfg_res,
                                            per_row_residual=residual)
    expected = torch.clamp(
        cfg_res.gain_max * torch.sqrt(residual / cfg_res.v_ref),
        cfg_res.gain_min, cfg_res.gain_max)
    assert torch.allclose(g_res, expected.to(torch.float32), atol=1e-6)
    assert float(g_res[0]) == pytest.approx(cfg_res.gain_min)   # floors
    assert float(g_res[3]) == pytest.approx(cfg_res.gain_max)   # caps
    assert d_res["mode"] == float(GAIN_MODES.index("residual_only"))
    # rule diagnostics are nan: this mode never evaluates the rule
    assert math.isnan(d_res["k_mean"]) and math.isnan(d_res["r_mean"])
    # packets are not consulted: None packets give the same gains
    g_res_np, _ = compute_provenance_gains(None, pi_cur=10.0,
                                           per_row_loss=losses, config=cfg_res,
                                           per_row_residual=residual)
    assert torch.equal(g_res, g_res_np)

    cfg_glob = ProvenanceGainConfig(mode="global", global_scale=0.7)
    g_glob, d_glob = compute_provenance_gains(packets, pi_cur=10.0,
                                              per_row_loss=losses,
                                              config=cfg_glob)
    assert torch.allclose(g_glob, torch.full((4,), 0.7))
    assert d_glob["gain_sd"] == pytest.approx(0.0)
    assert d_glob["mode"] == float(GAIN_MODES.index("global"))

    # residual_only NEEDS the current residual
    with pytest.raises(ValueError):
        compute_provenance_gains(packets, pi_cur=10.0, per_row_loss=losses,
                                 config=cfg_res)


# ----------------------------------------------------------------------
# T6 provenance_nohist <= provenance, rowwise
# ----------------------------------------------------------------------
def test_t6_nohist_never_exceeds_provenance():
    packets = [
        _packet(10.0, 1e-4, 0.004, s)
        for s in (0.01, 0.5, 1.0, 2.0, 10.0, 1e3)
    ]
    full, _ = compute_provenance_gains(
        packets, pi_cur=10.0, per_row_loss=None,
        config=ProvenanceGainConfig(mode="provenance"))
    nohist, d_nohist = compute_provenance_gains(
        packets, pi_cur=10.0, per_row_loss=None,
        config=ProvenanceGainConfig(mode="provenance_nohist"))
    assert bool((nohist <= full + 1e-7).all()), (nohist, full)
    # strictly below somewhere -- otherwise the contrast arm is degenerate
    assert bool((nohist < full - 1e-7).any())
    # nohist pins r == 1 exactly
    assert d_nohist["r_mean"] == pytest.approx(1.0)
    assert d_nohist["r_max"] == pytest.approx(1.0)
    # surprise <= 1 -> ln <= 0 -> r == 1, so those rows must MATCH
    for i, s in enumerate((0.01, 0.5, 1.0)):
        assert float(nohist[i]) == pytest.approx(float(full[i])), s


# ----------------------------------------------------------------------
# T7 missing packet
# ----------------------------------------------------------------------
def test_t7_missing_packet_gets_unit_gain_and_is_counted():
    cfg = ProvenanceGainConfig(mode="provenance")
    packets = [
        _packet(10.0, 1e-4, 0.004, 2.0),                    # fine
        None,                                                # no packet
        _packet(10.0, 1e-4, float("nan"), float("nan"),
                has_prev=False),                             # episode start
        _packet(10.0, 1e-4, float("nan"), 2.0),              # nan pe
        _packet(float("nan"), 1e-4, 0.004, 2.0),             # nan evidence
    ]
    gains, diag = compute_provenance_gains(packets, pi_cur=10.0,
                                           per_row_loss=None, config=cfg)
    assert diag["n_missing"] == 4.0
    for i in (1, 2, 3, 4):
        assert float(gains[i]) == 1.0, i
    assert float(gains[0]) != 1.0
    assert bool(torch.isfinite(gains).all())
    # k/m/r means are over the NON-missing rows only
    assert not math.isnan(diag["k_mean"])
    # all-missing -> nan rule diagnostics, unit gains
    all_missing, d2 = compute_provenance_gains([None, None], pi_cur=1.0,
                                               per_row_loss=None, config=cfg)
    assert torch.allclose(all_missing, torch.ones(2))
    assert d2["n_missing"] == 2.0
    for key in ("k_mean", "m_mean", "r_mean", "r_max", "surprise_max"):
        assert math.isnan(d2[key]), key
    # `global` is the information-free control: it never inspects a packet,
    # so it reports no missing rows.
    _g, d3 = compute_provenance_gains(
        [None, None], pi_cur=1.0, per_row_loss=None,
        config=ProvenanceGainConfig(mode="global"))
    assert d3["n_missing"] == 0.0


# ----------------------------------------------------------------------
# T8 config validation
# ----------------------------------------------------------------------
def test_t8_config_validation():
    ProvenanceGainConfig()  # defaults must be valid
    assert ProvenanceGainConfig().use_provenance_conditioned_consolidation_gain is False
    assert ProvenanceGainConfig().mode == "provenance"
    with pytest.raises(ValueError):
        ProvenanceGainConfig(mode="nonsense")
    with pytest.raises(ValueError):
        ProvenanceGainConfig(gain_min=0.0)
    with pytest.raises(ValueError):
        ProvenanceGainConfig(gain_min=-1.0)
    with pytest.raises(ValueError):
        ProvenanceGainConfig(gain_min=3.0, gain_max=2.0)
    with pytest.raises(ValueError):
        ProvenanceGainConfig(v_ref=0.0)
    with pytest.raises(ValueError):
        ProvenanceGainConfig(reopen_max=0.5)
    # gain_min == gain_max is legal (a fully pinned gain)
    ProvenanceGainConfig(gain_min=1.0, gain_max=1.0)


# ----------------------------------------------------------------------
# T9 weighted_row_loss gradient path
# ----------------------------------------------------------------------
def test_t9_weighted_row_loss_gradient_only_through_loss():
    rows = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    gains = torch.tensor([0.5, 1.0, 2.0, 0.25], requires_grad=True)
    out = weighted_row_loss(rows, gains)
    assert out.dim() == 0
    g = gains.detach()
    assert float(out) == pytest.approx(
        float((g * rows.detach()).sum() / g.sum())
    )
    out.backward()
    assert gains.grad is None, "gradient must NOT flow into the gain"
    assert rows.grad is not None
    # d/dl_i = g_i / sum(g)
    expected = g / g.sum()
    assert torch.allclose(rows.grad, expected, atol=1e-7)
    # unit gains reduce to the plain mean
    flat = weighted_row_loss(rows, torch.ones(4))
    assert float(flat) == pytest.approx(float(rows.detach().mean()))
    # degenerate (all-zero) gains fall back to the mean rather than nan
    zero = weighted_row_loss(rows, torch.zeros(4))
    assert float(zero) == pytest.approx(float(rows.detach().mean()))
    with pytest.raises(ValueError):
        weighted_row_loss(rows, torch.ones(3))


# ----------------------------------------------------------------------
# T10 consolidator: step scale moves the displacement; OFF is unchanged
# ----------------------------------------------------------------------
def _fixed_problem(seed: int = 1234):
    """A deterministic module + loss closure -- no RNG inside consolidate()."""
    torch.manual_seed(seed)
    mod = nn.Linear(4, 3)
    inp = torch.randn(6, 4)
    target = torch.randn(6, 3)
    init = [p.detach().clone() for p in mod.parameters()]

    def loss_fn():
        return ((mod(inp) - target) ** 2).mean()

    return mod, loss_fn, init


def _restore(mod, init):
    with torch.no_grad():
        for p, p0 in zip(mod.parameters(), init):
            p.copy_(p0)


def _max_delta(mod, init):
    return max(
        float((p.detach() - p0).abs().max().item())
        for p, p0 in zip(mod.parameters(), init)
    )


def _run(mod, loss_fn, init, **kwargs):
    _restore(mod, init)
    c = CrossModuleConsolidator(
        CrossModuleConsolidatorConfig(schedule="interleaved", n_steps=3, lr=1e-3)
    )
    metrics = c.consolidate(
        module_losses={"m": loss_fn},
        module_params={"m": list(mod.parameters())},
        **kwargs,
    )
    return c, metrics, [p.detach().clone() for p in mod.parameters()]


def test_t10a_step_scale_moves_displacement():
    mod, loss_fn, init = _fixed_problem()
    deltas = []
    for s in (0.1, 0.5, 1.0, 2.0):
        _c, metrics, _params = _run(
            mod, loss_fn, init,
            module_step_scale={"m": (lambda s=s: s)},
        )
        assert metrics["updates_m"] == 3.0
        assert metrics["step_scale_mean_m"] == pytest.approx(s)
        assert metrics["step_scale_min_m"] == pytest.approx(s)
        assert metrics["step_scale_max_m"] == pytest.approx(s)
        deltas.append(_max_delta(mod, init))
    assert all(b > a for a, b in zip(deltas, deltas[1:])), deltas
    # MECH-572 liveness: the whole point is that this is NOT pinned.
    assert deltas[-1] > 5.0 * deltas[0]


def test_t10b_off_path_unchanged():
    mod, loss_fn, init = _fixed_problem()

    # (i) no kwargs -> exactly the pre-existing metric keys, nothing added.
    _c_off, m_off, params_off = _run(mod, loss_fn, init)
    assert set(m_off.keys()) == BASE_METRIC_KEYS | {"updates_m"}
    assert not any(k.startswith("step_scale_") for k in m_off)
    assert _c_off.last_step_trace == []

    # (ii) an explicit scale of 1.0 must be a numerical no-op, so the OFF path
    #      is pinned against the ON path at unity rather than against itself.
    _c_one, m_one, params_one = _run(
        mod, loss_fn, init, module_step_scale={"m": lambda: 1.0}
    )
    for a, b in zip(params_off, params_one):
        assert torch.equal(a, b), "scale 1.0 must not change the arithmetic"
    for key in BASE_METRIC_KEYS | {"updates_m"}:
        assert m_off[key] == m_one[key]
    assert m_one["step_scale_mean_m"] == pytest.approx(1.0)

    # (iii) a module with NO entry in the mapping is untouched and emits no key.
    mod2, loss2, init2 = _fixed_problem(seed=99)
    _restore(mod, init)
    c = CrossModuleConsolidator(
        CrossModuleConsolidatorConfig(schedule="interleaved", n_steps=3, lr=1e-3)
    )
    m_mixed = c.consolidate(
        module_losses={"m": loss_fn, "other": loss2},
        module_params={"m": list(mod.parameters()),
                       "other": list(mod2.parameters())},
        module_step_scale={"m": lambda: 2.0},
    )
    assert "step_scale_mean_m" in m_mixed
    assert not any(k.endswith("_other") and k.startswith("step_scale")
                   for k in m_mixed)


def test_t10c_record_trace():
    mod, loss_fn, init = _fixed_problem()
    c, metrics, _p = _run(
        mod, loss_fn, init,
        module_step_scale={"m": lambda: 1.5},
        record_trace=True,
    )
    trace = c.last_step_trace
    assert len(trace) == 3
    assert [e["step"] for e in trace] == [0, 1, 2]
    for entry in trace:
        assert set(entry.keys()) == {
            "module", "step", "loss", "step_scale", "grad_norm"
        }
        assert entry["module"] == "m"
        assert entry["step_scale"] == pytest.approx(1.5)
        assert entry["grad_norm"] > 0.0
        assert math.isfinite(entry["loss"])
    # the trace is a copy -- mutating it cannot corrupt the consolidator
    trace[0]["loss"] = -1.0
    assert c.last_step_trace[0]["loss"] != -1.0
    # cleared at the start of the next call, and unscaled steps report 1.0
    c2, _m2, _p2 = _run(mod, loss_fn, init, record_trace=True)
    assert len(c2.last_step_trace) == 3
    assert all(e["step_scale"] == 1.0 for e in c2.last_step_trace)
    # and a trace-free call leaves the list empty
    c3, _m3, _p3 = _run(mod, loss_fn, init)
    assert c3.last_step_trace == []


def test_t10d_step_scale_is_read_after_the_loss_closure():
    """ORDERING CONTRACT: the callable is invoked AFTER loss_fn().

    The live caller computes its gain as a SIDE EFFECT of the loss closure
    (the per-row gains need that step's replay draw), so reading the callable
    before the closure would hand back a stale -- on the first step, an
    uninitialised -- scale.
    """
    mod, _unused, init = _fixed_problem()
    inp = torch.randn(6, 4)
    target = torch.randn(6, 3)
    order = []
    box = {"scale": None}

    def loss_fn():
        order.append("loss")
        box["scale"] = 0.5 + 0.5 * len(order)   # only knowable here
        return ((mod(inp) - target) ** 2).mean()

    def scale_fn():
        order.append("scale")
        assert box["scale"] is not None, "closure must have run first"
        return box["scale"]

    _restore(mod, init)
    c = CrossModuleConsolidator(
        CrossModuleConsolidatorConfig(schedule="interleaved", n_steps=3, lr=1e-3)
    )
    metrics = c.consolidate(
        module_losses={"m": loss_fn},
        module_params={"m": list(mod.parameters())},
        module_step_scale={"m": scale_fn},
        record_trace=True,
    )
    assert order == ["loss", "scale", "loss", "scale", "loss", "scale"]
    assert metrics["step_scale_min_m"] == pytest.approx(1.0)
    assert metrics["step_scale_max_m"] == pytest.approx(3.0)
    assert metrics["step_scale_mean_m"] == pytest.approx(2.0)
    assert [e["step_scale"] for e in c.last_step_trace] == [1.0, 2.0, 3.0]


# ----------------------------------------------------------------------
# T11 reduction="none" rows mean equals reduction="mean"
# ----------------------------------------------------------------------
def _build(seed: int = 7, **flags):
    from ree_core.environment.causal_grid_world import CausalGridWorldV2
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1, use_proxy_fields=True
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, od = env.reset()
    b = od["body_state"]
    w = od["world_state"]
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if w.dim() == 1:
        w = w.unsqueeze(0)
    return agent, b, w


def test_t11_reduction_none_matches_mean():
    agent, _b, _w = _build()
    agent.e2.eval()
    torch.manual_seed(3)
    K, world_dim, action_dim = 4, 16, 4
    z0 = torch.randn(1, world_dim)
    actions = torch.eye(action_dim)[:K]
    targets = torch.randn(K, world_dim)

    with torch.no_grad():
        rows = agent.e2.world_forward_contrastive_loss(
            z0, actions, targets, min_batch_classes=1, reduction="none"
        )
        mean = agent.e2.world_forward_contrastive_loss(
            z0, actions, targets, min_batch_classes=1, reduction="mean"
        )
    assert tuple(rows.shape) == (K,)
    assert mean.dim() == 0
    # MEASURED (darwin-arm64, torch 2.10.0, 2026-09-22): NOT bitwise.
    # `reduction="none"` + torch.mean sums the [K] rows in a different order
    # from F.cross_entropy's own fused mean, so the two agree to ~7e-8
    # RELATIVE (2.4e-7 absolute on a CE of ~3.64) rather than exactly. The
    # contract is therefore stated as a relative tolerance; the bitwise leg is
    # asserted only when the platform happens to deliver it.
    bitwise = bool(torch.equal(rows.mean(), mean))
    assert torch.allclose(rows.mean(), mean, rtol=1e-6, atol=1e-6), (
        float(rows.mean()), float(mean)
    )
    assert abs(float(rows.mean()) - float(mean)) <= 1e-6 * abs(float(mean))
    if bitwise:
        assert torch.equal(rows.mean(), mean)

    # Default is "mean" -- the pre-existing call shape is unchanged.
    with torch.no_grad():
        default = agent.e2.world_forward_contrastive_loss(
            z0, actions, targets, min_batch_classes=1
        )
    assert torch.equal(default, mean)

    # Degenerate early-returns give a 0-d zero under EITHER reduction.
    single_action = torch.eye(action_dim)[:1].repeat(K, 1)
    with torch.no_grad():
        degenerate = agent.e2.world_forward_contrastive_loss(
            z0, single_action, targets, min_batch_classes=2, reduction="none"
        )
        sim = agent.e2.world_forward_contrastive_loss(
            z0, actions, targets, min_batch_classes=1,
            simulation_mode=True, reduction="none"
        )
        tiny = agent.e2.world_forward_contrastive_loss(
            z0, actions[:1], targets[:1], min_batch_classes=1, reduction="none"
        )
    for t in (degenerate, sim, tiny):
        assert t.dim() == 0 and float(t) == 0.0

    # And an invalid reduction is rejected.
    with pytest.raises(ValueError):
        agent.e2.world_forward_contrastive_loss(
            z0, actions, targets, min_batch_classes=1, reduction="sum"
        )


# The gains are weights for a real loss -- pin the two halves composing.
def test_t11b_gains_compose_with_per_row_loss():
    agent, _b, _w = _build()
    agent.e2.eval()
    torch.manual_seed(5)
    K, world_dim, action_dim = 4, 16, 4
    z0 = torch.randn(1, world_dim)
    actions = torch.eye(action_dim)[:K]
    targets = torch.randn(K, world_dim)
    rows = agent.e2.world_forward_contrastive_loss(
        z0, actions, targets, min_batch_classes=1, reduction="none"
    )
    packets = [
        _packet(ev, 1e-4, pe, s)
        for ev, pe, s in ((1.0, 0.002, 0.5), (10.0, 0.01, 2.0),
                          (100.0, 0.02, 20.0), (5.0, 0.001, 1.0))
    ]
    gains, diag = compute_provenance_gains(
        packets, pi_cur=10.0, per_row_loss=rows,
        config=ProvenanceGainConfig(mode="provenance"),
    )
    total = weighted_row_loss(rows, gains)
    assert total.requires_grad
    total.backward()
    assert any(p.grad is not None for p in agent.e2.parameters())
    assert math.isfinite(diag["gain_mean"])
