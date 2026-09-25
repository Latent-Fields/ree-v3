"""Contracts for the SD-032b candidate effort proxy (sd032b-candidate-effort-proxy).

THE DEFECT THIS GUARDS (mech_268_dacc_saturation_form.md, Consumer A, 2026-09-18).
REEAgent.select_action passed dACC candidate_effort = c.actions.shape[1] -- the
physical rollout horizon, identical for every candidate. control_required * effort
was therefore a uniform (argmin-invariant) shift and the Croxson harm_interaction
term was identically zero, so nothing pe-dependent could ever move E3 selection
(measured: effort spread > 0 on 0/120 ticks).

The fix adds REEConfig.dacc_candidate_effort_source: "horizon" (default,
bit-identical legacy) | "harm_a_forward" (E2_harm_a rolled over each candidate's
own action sequence; effort_k = mean_t ||z_harm_a_pred_t||).

Assertions are UPSTREAM of the discrete action quantizer (effort vectors, the
dACC bundle terms) -- never a committed action (torch.multinomial is not portable
across machine classes). Candidates come from the agent's own
generate_trajectories on a real env observation, not a hand-built batch.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


def _build(source=None, e2ha=True, **extra):
    torch.manual_seed(3)
    env = CausalGridWorldV2(seed=5, size=8, num_hazards=8, num_resources=3,
                            num_waypoints=0)
    _, obs = env.reset()
    kw = dict(
        body_obs_dim=obs["body_state"].shape[-1],
        world_obs_dim=obs["world_state"].shape[-1],
        action_dim=4,
        alpha_world=0.3,
        use_dacc=True,
        use_affective_harm_stream=True,
        use_e2_harm_a=e2ha,
        dacc_weight=1.0,
    )
    if source is not None:
        kw["dacc_candidate_effort_source"] = source
    kw.update(extra)
    agent = REEAgent(REEConfig.from_dims(**kw))
    agent.reset()
    return agent, env, obs


def _tick(agent, obs):
    """One real sense -> generate -> select tick; returns the candidates."""
    wd = agent.config.latent.world_dim
    with torch.no_grad():
        lat = agent.sense(obs["body_state"], obs["world_state"],
                          obs_harm_a=obs.get("harm_obs_a"))
        ticks = agent.clock.advance()
        e1p = (agent._e1_tick(lat) if ticks.get("e1_tick")
               else torch.zeros(1, wd))
        cands = agent.generate_trajectories(lat, e1p, ticks)
        agent.select_action(cands, ticks)
    return cands


# ---- (1) config surface: default, from_dims threading, validation ------------

def test_default_is_legacy_horizon():
    cfg = REEConfig()
    assert cfg.dacc_candidate_effort_source == "horizon"
    assert cfg.dacc_effort_rollout_steps == 0


def test_from_dims_threads_both_knobs():
    """from_dims silently swallows unknown kwargs (MECH-307 class) -- pin that
    both knobs actually reach the built config."""
    agent, _, _ = _build("harm_a_forward", dacc_effort_rollout_steps=4)
    assert agent.config.dacc_candidate_effort_source == "harm_a_forward"
    assert agent.config.dacc_effort_rollout_steps == 4


def test_from_dims_rejects_unknown_source():
    with pytest.raises(ValueError):
        REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4,
                            dacc_candidate_effort_source="trajectory_length")


# ---- (2) legacy path unchanged -----------------------------------------------

def test_horizon_source_is_constant_rollout_length():
    agent, _, obs = _build()  # default
    cands = _tick(agent, obs)
    eff = agent._dacc_last_effort
    assert agent._dacc_last_effort_source == "horizon"
    assert eff is not None and eff.numel() == len(cands)
    assert torch.all(eff == float(cands[0].actions.shape[1]))
    # ... and so the Croxson interaction is exactly zero (the defect, preserved
    # bit-identically on the default path).
    assert float(agent._dacc_last_bundle["harm_interaction"].abs().max()) == 0.0


# ---- (3) the new proxy is live on real candidates ------------------------------

def test_harm_a_forward_effort_varies_across_candidates():
    """Effort spread > 0 on every tick; and wherever the payoff (E3's previous
    score) itself has spread, the Croxson interaction is no longer zero. The
    payoff is flat on the first few ticks of an episode (measured: 8 ticks on
    this seed), so the interaction check is conditioned on payoff spread."""
    agent, env, obs = _build("harm_a_forward")
    spreads = []
    inter_where_payoff_varies = []
    for _ in range(20):
        cands = _tick(agent, obs)
        assert agent._dacc_last_effort_source == "harm_a_forward"
        eff = agent._dacc_last_effort
        assert eff.shape == (len(cands),)
        assert torch.isfinite(eff).all()
        spreads.append(float(eff.max() - eff.min()))
        b = agent._dacc_last_bundle
        pay = b["mode_ev"] + b["effort_term"]
        if float(pay.max() - pay.min()) > 1e-6:
            inter_where_payoff_varies.append(
                float(b["harm_interaction"].abs().max())
            )
        _, _, done, _, obs = env.step(0)
        if done:
            _, obs = env.reset()
    assert all(s > 0.0 for s in spreads), f"effort spread pinned at 0: {spreads}"
    assert inter_where_payoff_varies, "no tick with payoff spread -- probe is degenerate"
    assert all(x > 0.0 for x in inter_where_payoff_varies), (
        "harm_interaction zero despite payoff and effort spread"
    )


def test_harm_a_forward_matches_manual_rollout():
    """Reads the effort from the module under test and recomputes it from
    E2_harm_a directly -- the formula is effort_k = mean_t ||z_pred_t||."""
    agent, _, obs = _build("harm_a_forward", dacc_effort_rollout_steps=3)
    cands = _tick(agent, obs)
    eff, src = agent._dacc_candidate_effort(cands, torch.float32)
    assert src == "harm_a_forward"
    with torch.no_grad():
        z = agent._current_latent.z_harm_a.detach().reshape(1, -1)
        z = z.expand(len(cands), -1)
        acts = torch.stack([c.actions[0, :3, :] for c in cands], dim=0)
        norms = []
        for t in range(3):
            z = agent.e2_harm_a(z, acts[:, t, :])
            norms.append(z.norm(dim=-1))
        expected = torch.stack(norms, 0).mean(0)
    assert torch.allclose(eff, expected, atol=1e-6)


def test_rollout_steps_changes_the_cost():
    agent, _, obs = _build("harm_a_forward")
    cands = _tick(agent, obs)
    full, _ = agent._dacc_candidate_effort(cands, torch.float32)
    agent.config.dacc_effort_rollout_steps = 1
    one, _ = agent._dacc_candidate_effort(cands, torch.float32)
    assert not torch.allclose(full, one)


def test_effort_term_now_depends_on_candidate():
    """The EVC control term is no longer a uniform shift: effort_term spread > 0,
    so control_required (hence pe / f_sat) can reorder candidates."""
    agent, _, obs = _build("harm_a_forward", dacc_effort_cost=10.0)
    _tick(agent, obs)
    et = agent._dacc_last_bundle["effort_term"]
    assert float(et.max() - et.min()) > 0.0


# ---- (4) fallback when E2_harm_a is absent -------------------------------------

def test_falls_back_to_horizon_without_e2_harm_a():
    agent, _, obs = _build("harm_a_forward", e2ha=False)
    assert agent.e2_harm_a is None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cands = _tick(agent, obs)
    assert agent._dacc_last_effort_source == "horizon"
    assert torch.all(agent._dacc_last_effort == float(cands[0].actions.shape[1]))
    assert any("falling back" in str(x.message) for x in w)
