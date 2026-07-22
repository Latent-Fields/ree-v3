"""SD-081 / MECH-477 -- dual-system uncertainty arbitration contracts.

Pins the four properties the falsifier depends on:

  1. OFF is bit-identical and writes no arbitration state.
  2. The flag is reachable through from_dims (the three-wiring-site hazard, and
     specifically the E3Config-vs-REEConfig level that this build tripped over
     during authoring).
  3. The HABIT pathway carries a real ranking -- i.e. depth 1 is unreachable.
     This is the defect that made V3-EXQ-786a's recruitment DV degenerate.
  4. The arbitration weight VARIES WITH measured uncertainty, which MECH-477
     makes the mandatory manipulation check: without it a null is a readiness
     failure that scores nothing rather than a refutation.
"""

import math

import numpy as np
import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e3_selector import _ARB_STD_FLOOR
from ree_core.utils.config import E3Config, REEConfig


def _obs(d, k):
    v = d.get(k)
    if v is None:
        return None
    v = v.float()
    return v.unsqueeze(0) if v.dim() == 1 else v


def _drive(arb_on, steps=200, seed=0, curiosity=0.5, **e3_over):
    """Run the agent forward, returning (actions, arbitration records on E3 ticks)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = CausalGridWorldV2(seed=123, size=10, num_hazards=3, num_resources=5)
    _f, od = env.reset()
    body, world = _obs(od, "body_state"), _obs(od, "world_state")
    kw = dict(
        body_obs_dim=body.shape[-1],
        world_obs_dim=world.shape[-1],
        action_dim=int(env.action_dim),
    )
    if arb_on:
        kw["use_dualsystem_arbitration"] = True
    cfg = REEConfig.from_dims(**kw)
    for k, v in e3_over.items():
        setattr(cfg.e3, k, v)
    if curiosity:
        cfg.hippocampal.curiosity_weight = curiosity
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = REEAgent(cfg)
    agent.reset()

    acts, arb = [], []
    for _ in range(steps):
        body, world = _obs(od, "body_state"), _obs(od, "world_state")
        if body is None or world is None:
            break
        lat = agent.sense(
            obs_body=body,
            obs_world=world,
            obs_harm=_obs(od, "harm_obs"),
            obs_harm_a=_obs(od, "harm_obs_a"),
            obs_harm_history=_obs(od, "harm_history"),
        )
        ticks = agent.clock.advance()
        wdim = lat.z_world.shape[-1]
        e1 = (
            agent._e1_tick(lat)
            if ticks.get("e1_tick")
            else torch.zeros(1, wdim, device=agent.device)
        )
        cands = agent.generate_trajectories(lat, e1, ticks)
        a = agent.select_action(cands, ticks)
        rec = getattr(agent.e3, "last_arbitration", None)
        if ticks.get("e3_tick") and rec is not None:
            arb.append(dict(rec))
        ai = int(a[0].argmax().item()) if a is not None and torch.isfinite(a).all() else 0
        acts.append(ai)
        _f, _h, done, info, od = env.step(ai)
        with torch.no_grad():
            agent.update_residue(
                harm_signal=float(info.get("harm_signal", 0.0)),
                world_delta=None,
                hypothesis_tag=False,
                owned=True,
            )
        if done:
            _f, od = env.reset()
    return acts, arb


def test_sd081_defaults_off():
    """Default config leaves the arbitrator off -- version-layering doctrine."""
    assert E3Config().use_dualsystem_arbitration is False
    assert REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=4
    ).e3.use_dualsystem_arbitration is False


def test_sd081_flag_reachable_through_from_dims():
    """The three-wiring-site hazard, pinned at the level that actually bit.

    from_dims silently swallows unknown kwargs, so a missing signature entry
    leaves the flag unreachable with no error. This build ALSO tripped the
    one-level-down variant: E3Selector.config IS the E3Config, so a
    REEConfig-level field reads as a missing attribute in the selector and
    defaults to False. Both are pinned here.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=8,
        action_dim=4,
        use_dualsystem_arbitration=True,
        dualsystem_arbitration_gain=3.0,
        dualsystem_arbitration_bias=0.25,
        dualsystem_uncertainty_ema_alpha=0.2,
        dualsystem_habit_depth=3,
    )
    assert cfg.e3.use_dualsystem_arbitration is True
    assert cfg.e3.dualsystem_arbitration_gain == 3.0
    assert cfg.e3.dualsystem_arbitration_bias == 0.25
    assert cfg.e3.dualsystem_uncertainty_ema_alpha == 0.2
    assert cfg.e3.dualsystem_habit_depth == 3
    # The selector reads self.config, which is the E3Config -- not REEConfig.
    assert getattr(cfg.e3, "use_dualsystem_arbitration", False) is True


def test_sd081_off_is_bit_identical_and_writes_nothing():
    """OFF: reproducible action stream, and last_arbitration never written."""
    a1, arb1 = _drive(False)
    a2, arb2 = _drive(False)
    assert a1 == a2
    assert arb1 == [] and arb2 == []


def test_sd081_on_changes_selection_and_records_the_paired_series():
    """ON: the arbitrator is live, non-degenerate, and reallocates control."""
    a_off, _ = _drive(False)
    a_on, arb = _drive(True)
    assert arb, "arbitrator never ran with the flag ON"
    live = [r for r in arb if not r["degenerate"]]
    assert live, f"every tick degenerate: {sorted({r['degeneracy_reason'] for r in arb})}"
    assert a_on != a_off, "arbitration did not change the committed action stream"
    for r in live:
        assert 0.0 < r["w_planned"] < 1.0
        assert r["habit_uncertainty_source"] in ("familiarity", "e1_novelty_ema")
        # RANGE, not magnitude (V3-EXQ-643): a uniform per-tick offset has large
        # magnitude and ~0 range, so a magnitude check passes on an arbitrary ranking.
        assert r["habit_score_range"] > _ARB_STD_FLOOR
        assert r["planned_score_range"] > _ARB_STD_FLOOR


def test_sd081_weight_varies_with_measured_uncertainty():
    """MECH-477's MANDATORY manipulation check, pinned as a contract.

    The falsifier must be able to show the arbitration weight varies with
    measured uncertainty. If w were constant, a null in the ON arm would be a
    readiness failure scoring nothing rather than a refutation of the claim --
    so a substrate that cannot produce a varying w is not falsifier-ready.
    """
    _a, arb = _drive(True)
    live = [r for r in arb if not r["degenerate"]]
    assert len(live) >= 5, "too few live arbitration ticks to assess variation"
    ws = [r["w_planned"] for r in live]
    us = [r["u_habit_norm"] - r["u_planned_norm"] for r in live]
    assert len(set(round(w, 9) for w in ws)) > 1, "arbitration weight is constant"
    # w is a strictly monotone (sigmoid) function of the relative-uncertainty
    # difference, so the two must agree in rank ORDER exactly.
    ru = np.argsort(np.argsort(np.asarray(us, dtype=float)))
    rw = np.argsort(np.argsort(np.asarray(ws, dtype=float)))
    assert np.array_equal(ru, rw), "w is not monotone in relative uncertainty"


def test_sd081_habit_depth_floor_blocks_the_786a_degeneracy():
    """Depth 1 must be unreachable by config -- the V3-EXQ-786a defect.

    Index 0 of the z_world sequence is the CURRENT state, shared by every
    candidate (they all start where the agent is). A depth-1 score vector is
    therefore constant, with cross-candidate range exactly 0.0 and no ranking to
    contribute. V3-EXQ-786a's recruitment DV used exactly that read, and its
    Spearman was consequently computed against an arbitrary tie-break
    permutation -- yielding noise centred near 1.0, i.e. a FLAT response by
    construction. The max(2, ...) floor in _arbitrate_dual_system makes the
    degenerate depth unreachable even if a config sets it.
    """
    _a, arb = _drive(True, dualsystem_habit_depth=1)
    live = [r for r in arb if not r["degenerate"]]
    assert live, "depth floor did not rescue the habit pathway"
    for r in live:
        assert r["habit_score_range"] > _ARB_STD_FLOOR


def test_sd081_depth_limit_is_always_restored():
    """_score_depth_limit must never leak past the habit pass.

    A leaked depth limit would silently make every subsequent full-horizon score
    myopic -- a substrate-wide corruption with no error.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    env = CausalGridWorldV2(seed=123, size=10, num_hazards=3, num_resources=5)
    _f, od = env.reset()
    body, world = _obs(od, "body_state"), _obs(od, "world_state")
    cfg = REEConfig.from_dims(
        body_obs_dim=body.shape[-1],
        world_obs_dim=world.shape[-1],
        action_dim=int(env.action_dim),
        use_dualsystem_arbitration=True,
    )
    cfg.hippocampal.curiosity_weight = 0.5
    agent = REEAgent(cfg)
    agent.reset()
    for _ in range(120):
        body, world = _obs(od, "body_state"), _obs(od, "world_state")
        if body is None or world is None:
            break
        lat = agent.sense(obs_body=body, obs_world=world)
        ticks = agent.clock.advance()
        e1 = (
            agent._e1_tick(lat)
            if ticks.get("e1_tick")
            else torch.zeros(1, lat.z_world.shape[-1], device=agent.device)
        )
        cands = agent.generate_trajectories(lat, e1, ticks)
        agent.select_action(cands, ticks)
        assert agent.e3._score_depth_limit is None, "depth limit leaked out of the habit pass"
        _f, _h, done, info, od = env.step(0)
        if done:
            _f, od = env.reset()
