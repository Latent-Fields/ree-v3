"""Contracts for dacc-pe-scale-normalisation (IGW-20260925-219, MECH-268).

Design: REE_assembly/docs/architecture/dacc_pe_scale_normalisation.md.
User decision dec-20260923T185804-MECH-268 option 2: keep
dacc_saturation_strength at 0.3 and normalise dacc_pe at its producer so the
fixed f_sat floor stops drifting against the training-dependent pe scale.

Every assertion reads values from the module under test
(DACCAdaptiveControl / REEConfig / REEAgent); nothing here restates the
transform as a literal it then checks against itself, except the closed-form
steady state target * (1 + prec_norm), which is the design's contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

from ree_core.cingulate.dacc import DACCAdaptiveControl, DACCConfig

DIM = 4
PREC_CAPPED = 5000.0  # precision / dacc_precision_scale(500) = 10 -> capped at 3.0
GAIN_CAPPED = 4.0


def _vec(norm: float) -> torch.Tensor:
    v = torch.zeros(DIM)
    v[0] = norm
    return v


def _step(dacc: DACCAdaptiveControl, pe_u: float, pred: bool = False,
          precision: float = PREC_CAPPED, outcome_class=None) -> dict:
    """One forward whose raw affective quantity is exactly pe_u."""
    if pred:
        z, zp = _vec(pe_u + 1.0), _vec(1.0)
    else:
        z, zp = _vec(pe_u), None
    return dacc(
        z_harm_a=z,
        z_harm_a_pred=zp,
        candidate_payoffs=torch.tensor([0.1, 0.2, 0.3]),
        candidate_effort=torch.tensor([1.0, 1.0, 1.0]),
        candidate_action_classes=[0, 1, 2],
        precision=precision,
        current_outcome_class=outcome_class,
    )


def _run(cfg: DACCConfig, stream, **kw):
    d = DACCAdaptiveControl(cfg)
    return d, [_step(d, x, **kw) for x in stream]


def _stream(level: float, n: int = 3000):
    # Deterministic mild within-regime structure around `level`.
    g = torch.Generator().manual_seed(7)
    jitter = torch.rand(n, generator=g) * 0.2 + 0.9
    return [level * float(j) for j in jitter]


# 1. OFF is bit-identical ---------------------------------------------------

def test_off_bundle_identical_to_default_and_buffers_untouched():
    stream = _stream(1.3, 200)
    d_def, b_def = _run(DACCConfig(), stream)
    d_off, b_off = _run(DACCConfig(dacc_pe_norm_enabled=False,
                                   dacc_pe_norm_target=9.0,
                                   dacc_pe_norm_floor=7.0), stream)
    for a, b in zip(b_def, b_off):
        assert set(a) == set(b)
        assert "pe_norm_scale" not in a and "pe_prenorm_unsaturated" not in a
        assert a["pe"] == b["pe"]
        assert a["pe_unsaturated"] == b["pe_unsaturated"]
        assert a["foraging_value"] == b["foraging_value"]
    # Raw OFF path: pe == pe_u * gain exactly.
    assert b_off[0]["pe"] == pytest.approx(stream[0] * GAIN_CAPPED, rel=1e-6)
    assert d_off.pe_norm_updates == [0, 0]
    assert d_off.pe_norm_scale == [0.0, 0.0]


# 2/3. Scale invariance and steady state ------------------------------------

@pytest.mark.parametrize("k", [0.5, 3.6, 16.0])
def test_scale_invariance(k):
    base = _stream(1.0)
    cfg = DACCConfig(dacc_pe_norm_enabled=True)
    _, b1 = _run(cfg, base)
    _, bk = _run(cfg, [k * x for x in base])
    tail1 = [b["pe"] for b in b1[-500:]]
    tailk = [b["pe"] for b in bk[-500:]]
    for a, b in zip(tail1, tailk):
        assert b == pytest.approx(a, rel=1e-6)


@pytest.mark.parametrize("level", [0.25, 1.0, 4.0])
def test_steady_state_is_target_times_precision_gain(level):
    cfg = DACCConfig(dacc_pe_norm_enabled=True)
    _, bs = _run(cfg, [level] * 400)
    assert bs[-1]["pe"] == pytest.approx(cfg.dacc_pe_norm_target * GAIN_CAPPED, rel=1e-6)
    # MECH-258 precision gain survives: low precision -> gain ~1.
    _, bl = _run(cfg, [level] * 400, precision=0.0)
    assert bl[-1]["pe"] == pytest.approx(cfg.dacc_pe_norm_target, rel=1e-6)


def test_warmup_is_exact_cumulative_mean():
    cfg = DACCConfig(dacc_pe_norm_enabled=True, dacc_pe_norm_alpha=0.001)
    d, _ = _run(cfg, [1.0, 2.0, 3.0, 6.0])
    assert d.pe_norm_scale[0] == pytest.approx(3.0)
    assert d.pe_norm_updates == [4, 0]


# 4/5. Saturation and cap are not undone ------------------------------------

def test_saturation_not_undone_and_scale_reads_pre_saturation():
    stream = [1.7] * 300
    on = DACCConfig(dacc_pe_norm_enabled=True, dacc_saturation_enabled=True)
    d_sat = DACCAdaptiveControl(on)
    for _ in range(on.dacc_saturation_window):
        d_sat.record_outcome(1)
    b_sat = [_step(d_sat, x, outcome_class=1) for x in stream]
    d_ref, b_ref = _run(DACCConfig(dacc_pe_norm_enabled=True), stream)
    assert d_sat.pe_norm_scale == d_ref.pe_norm_scale
    w, g, s = on.dacc_saturation_window, on.dacc_saturation_grace, on.dacc_saturation_strength
    floor = 1.0 / (1.0 + s * (w - g))
    assert b_sat[-1]["saturation_factor"] == pytest.approx(floor)
    assert b_sat[-1]["pe"] == pytest.approx(b_ref[-1]["pe"] * floor, rel=1e-9)


def test_cap_not_undone_and_scale_reads_pre_cap():
    stream = [2.0] * 300
    d_cap, b_cap = _run(DACCConfig(dacc_pe_norm_enabled=True, dacc_pe_cap=0.3), stream)
    d_ref, _ = _run(DACCConfig(dacc_pe_norm_enabled=True), stream)
    assert d_cap.pe_norm_scale == d_ref.pe_norm_scale
    assert b_cap[-1]["pe"] == pytest.approx(0.3)


# 6. Floor ------------------------------------------------------------------

def test_floor_prevents_amplifying_a_quiet_stream():
    cfg = DACCConfig(dacc_pe_norm_enabled=True)
    _, bs = _run(cfg, [0.01] * 400)
    expected = cfg.dacc_pe_norm_target * 0.01 / cfg.dacc_pe_norm_floor * GAIN_CAPPED
    assert bs[-1]["pe"] == pytest.approx(expected, rel=1e-6)
    assert bs[-1]["pe"] < 0.1 * cfg.dacc_pe_norm_target * GAIN_CAPPED


# 7. Per-statistic scales (the red-team blocking fix) -----------------------

def test_all_nopred_stream_updates_the_nopred_scale():
    # The V3-EXQ-1089 shape: no E2HarmAForward, so every tick has pred None.
    cfg = DACCConfig(dacc_pe_norm_enabled=True)
    d, bs = _run(cfg, [1.4] * 50)
    assert d.pe_norm_updates == [50, 0]
    assert bs[-1]["pe_norm_updates"] == 50
    assert d.pe_norm_scale[0] == pytest.approx(1.4)


def test_mixed_stream_keeps_scales_separate():
    d = DACCAdaptiveControl(DACCConfig(dacc_pe_norm_enabled=True))
    for _ in range(20):
        _step(d, 5.0, pred=False)
        _step(d, 0.5, pred=True)
    assert d.pe_norm_updates == [20, 20]
    assert d.pe_norm_scale[0] == pytest.approx(5.0)
    assert d.pe_norm_scale[1] == pytest.approx(0.5, rel=1e-5)


# 8. Persistence, loading, freeze -------------------------------------------

def test_resets_do_not_clear_scale():
    d, _ = _run(DACCConfig(dacc_pe_norm_enabled=True), [2.0] * 30)
    before = d.pe_norm_scale, d.pe_norm_updates
    d.reset()
    d.reset_episode_pe()
    d.reset_outcome_history()
    assert (d.pe_norm_scale, d.pe_norm_updates) == before


def test_state_dict_roundtrip_and_cross_flag_strict_loads():
    d_on, _ = _run(DACCConfig(dacc_pe_norm_enabled=True), [2.0] * 30)
    sd = d_on.state_dict()
    fresh_on = DACCAdaptiveControl(DACCConfig(dacc_pe_norm_enabled=True))
    fresh_on.load_state_dict(sd, strict=True)
    assert fresh_on.pe_norm_scale == d_on.pe_norm_scale
    assert fresh_on.pe_norm_updates == d_on.pe_norm_updates
    # ON snapshot -> OFF module (a normaliser-OFF arm off a shared snapshot).
    DACCAdaptiveControl(DACCConfig()).load_state_dict(sd, strict=True)
    # Pre-build checkpoint (no estimator keys) -> ON module.
    legacy = {k: v for k, v in sd.items() if not k.startswith("_pe_norm")}
    assert legacy == {}  # the dACC module had no state before this build
    fresh_on.load_state_dict(legacy, strict=True)
    assert fresh_on.pe_norm_updates == d_on.pe_norm_updates  # filled from self


def test_freeze_stops_updates_but_still_applies_scale():
    d, _ = _run(DACCConfig(dacc_pe_norm_enabled=True), [2.0] * 30)
    d.freeze_pe_norm()
    frozen = d.pe_norm_scale, d.pe_norm_updates
    b = _step(d, 8.0)
    assert (d.pe_norm_scale, d.pe_norm_updates) == frozen
    assert b["pe"] == pytest.approx(0.5 * 8.0 / 2.0 * GAIN_CAPPED)
    d.freeze_pe_norm(False)
    _step(d, 8.0)
    assert d.pe_norm_updates[0] == frozen[1][0] + 1


# 9. Three-site plumbing (from_dims swallows unknown kwargs) ----------------

def test_from_dims_reaches_agent_dacc_config():
    from ree_core.utils.config import REEConfig
    from ree_core.agent import REEAgent

    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=5,
        use_dacc=True, dacc_pe_norm_enabled=True, dacc_pe_norm_target=0.7,
        dacc_pe_norm_alpha=0.002, dacc_pe_norm_floor=0.2,
    )
    assert cfg.dacc_pe_norm_enabled is True
    agent = REEAgent(cfg)
    dc = agent.dacc.config
    assert (dc.dacc_pe_norm_enabled, dc.dacc_pe_norm_target,
            dc.dacc_pe_norm_alpha, dc.dacc_pe_norm_floor) == (True, 0.7, 0.002, 0.2)
    off = REEAgent(REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250,
                                       action_dim=5, use_dacc=True))
    assert off.dacc.config.dacc_pe_norm_enabled is False


# 10. The failure record: one release rung for every 1089 seed --------------

def test_1089_seed_spread_collapses_to_one_release_rung():
    # V3-EXQ-1089 trained seeds' unsaturated pe p50 (precision capped on all 5).
    seeds_pe_unsat = {43: 2.12, 45: 3.21, 42: 4.18, 44: 5.61}
    cfg = DACCConfig(dacc_pe_norm_enabled=True)
    w, g, s = cfg.dacc_saturation_window, cfg.dacc_saturation_grace, cfg.dacc_saturation_strength
    rungs = [1.0 / (1.0 + s * max(0, n - g)) for n in range(w + 1)]
    critical = 1.0  # trained-1089 margin (external_task_bias; foraging/difficulty ~0 there)

    raw_first_release, norm_first_release = set(), set()
    for pe_unsat in seeds_pe_unsat.values():
        pe_u = pe_unsat / GAIN_CAPPED
        _, bs = _run(cfg, _stream(pe_u))
        steady = sum(b["pe"] for b in bs[-500:]) / 500
        raw_first_release.add(next((n for n, r in enumerate(rungs) if pe_unsat * r < critical), None))
        norm_first_release.add(next((n for n, r in enumerate(rungs) if steady * r < critical), None))
    assert len(raw_first_release) > 1          # the defect: seed-dependent (incl. never)
    assert len(norm_first_release) == 1        # the fix: one boundary for every seed
    assert None not in norm_first_release      # and the s=0.3 floor does release


# Liveness (skill Step 5 b2): a REAL rollout, not a hand-built stream --------

def _live_salience_trace(norm_on: bool, seed: int = 7, steps: int = 60):
    from _harness import StepHarness
    from ree_core.agent import REEAgent
    from ree_core.environment.causal_grid_world import CausalGridWorldV2
    from ree_core.utils.config import REEConfig

    torch.manual_seed(seed)
    env = CausalGridWorldV2(seed=seed, size=5, num_hazards=2, num_resources=1,
                            use_proxy_fields=True)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=16, world_dim=16,
        use_dacc=True, use_affective_harm_stream=True,
        use_salience_coordinator=True,
        dacc_pe_norm_enabled=norm_on,
    )
    agent = REEAgent(cfg)
    trace = []

    def on_step(_r):
        if agent._dacc_last_bundle is not None and agent.salience is not None:
            trace.append((float(agent.salience._input_signals.get("dacc_pe", 0.0)),
                          dict(agent._dacc_last_bundle)))

    harness = StepHarness(agent, env, train_mode=False, seed=seed)
    with torch.no_grad():
        harness.run_episode(max_steps=steps, on_step=on_step)
    return agent, trace


def test_liveness_live_agent_on_changes_salience_input():
    agent_off, off = _live_salience_trace(False)
    agent_on, on = _live_salience_trace(True)
    assert off and on, "no dACC bundle reached the salience coordinator"
    # The live consumer reads the normalised value.
    for sal_pe, bundle in on:
        assert sal_pe == pytest.approx(bundle["pe"])
    # The estimator actually moved on the live path (no E2HarmAForward here,
    # so it is the no-prediction statistic that updates).
    assert agent_on.dacc.pe_norm_updates[0] > 0
    assert agent_off.dacc.pe_norm_updates == [0, 0]
    # ON and OFF present a different dacc_pe to the coordinator.
    n = min(len(off), len(on))
    assert any(abs(off[i][0] - on[i][0]) > 1e-6 for i in range(n)), "INERT"
    last = on[-1][1]
    assert last["pe_norm_updates"] == agent_on.dacc.pe_norm_updates[0]
    assert last["pe_norm_scale"] >= agent_on.dacc.config.dacc_pe_norm_floor
