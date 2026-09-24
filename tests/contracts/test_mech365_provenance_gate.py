"""MECH-365 contracts: the committed_vs_imagined label (Trajectory.hypothesis_tag)
travels on imagined replay output, and the one-way commit-status gate at the
REM replay -> MECH-217 consolidation boundary honours it.

Pins the three default-OFF/no-op knobs landed 2026-09-24 (HippocampalConfig):
  mech365_suppress_replay_provenance_stamp (default False = replay() output stamped)
  rem_route_forward_replay_to_consolidation (default False = as shipped)
  mech365_provenance_lesion (default "off")

test_replay_output_carries_imagined_label is the regression latch for the
pre-2026-09-24 defect: HippocampalModule.replay() documented "all content
carries hypothesis_tag=True" but returned the Trajectory dataclass default
(False). It FAILS on that tree.

Assertions are on write counts / valence values (upstream of any sampled
action), never on a committed action sequence.
"""
from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.predictors.e2_fast import Trajectory
from ree_core.residue.field import VALENCE_WANTING
from ree_core.utils.config import REEConfig


def _env(seed=7):
    return CausalGridWorldV2(
        seed=seed, size=8, num_hazards=0, num_resources=1, resource_benefit=1.0,
        proximity_harm_scale=0.0, proximity_benefit_scale=0.0, env_drift_prob=0.0,
        use_proxy_fields=False, resource_respawn_on_consume=True,
    )


def _agent(env, seed=7, **kw):
    torch.manual_seed(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32, alpha_world=0.9,
        replay_diversity_enabled=True, sws_enabled=True, sws_consolidation_steps=3,
        rem_enabled=True, rem_attribution_steps=10, **kw,
    )
    cfg.residue.num_basis_functions = 128
    cfg.residue.kernel_bandwidth = 0.03
    return REEAgent(cfg)


def _theta_recent(agent, n=4):
    wd = agent.config.latent.world_dim
    return torch.randn(n, 1, wd)


# --------------------------------------------------------------------------- #
# E1: the label travels on imagined replay output                              #
# --------------------------------------------------------------------------- #

def test_replay_output_carries_imagined_label():
    env = _env()
    agent = _agent(env)
    trajs = agent.hippocampal.replay(_theta_recent(agent), num_replay_steps=3)
    assert len(trajs) == 3
    assert all(t.hypothesis_tag is True for t in trajs), (
        "HippocampalModule.replay() output must carry hypothesis_tag=True "
        "(MECH-094 / MECH-365): replay is imagined content"
    )


def test_suppress_flag_reproduces_unlabelled_output_canary():
    env = _env()
    agent = _agent(env, mech365_suppress_replay_provenance_stamp=True)
    trajs = agent.hippocampal.replay(_theta_recent(agent), num_replay_steps=2)
    assert all(t.hypothesis_tag is False for t in trajs)


def test_reverse_replay_of_real_experience_is_untagged():
    env = _env()
    agent = _agent(env)
    wd = agent.config.latent.world_dim
    src = Trajectory(
        states=[torch.zeros(1, 32) for _ in range(3)],
        actions=torch.zeros(1, 3, env.action_dim),
        world_states=[torch.randn(1, wd) for _ in range(3)],
    )
    rev = agent.hippocampal.reverse_replay(src)
    assert rev.hypothesis_tag is False and rev.is_reverse is True


# --------------------------------------------------------------------------- #
# E3: gate honours the label; the lesion drops it at the boundary only         #
# --------------------------------------------------------------------------- #

def _field_with_wanting(agent, wd):
    """Three active centers; positive wanting at the terminus center."""
    rf = agent.residue_field
    pts = [torch.full((1, wd), float(i)) * 0.5 for i in range(3)]
    for p in pts:
        rf.rbf_field.add_residue(p, 1e-3)
    rf.update_valence(pts[0], VALENCE_WANTING, 2.0, hypothesis_tag=False)
    return pts


def _wanting(agent):
    rbf = agent.residue_field.rbf_field
    return rbf.valence_vecs[rbf.active_mask.bool()][:, VALENCE_WANTING].clone()


def _traj(pts, tag):
    return Trajectory(
        states=[torch.zeros(1, 32) for _ in pts],
        actions=torch.zeros(1, len(pts), 5),
        world_states=list(pts),
        hypothesis_tag=tag,
    )


@pytest.mark.parametrize("lesion,tag,expect_write", [
    ("off", True, False),
    ("off", False, True),
    ("drop_at_consolidation", True, True),
    ("drop_at_consolidation", False, True),
    ("sham_real_only", True, False),
    ("sham_real_only", False, True),
])
def test_spread_gate_honours_label_and_lesion_semantics(lesion, tag, expect_write):
    env = _env()
    agent = _agent(env, use_offline_wanting_spread=True, mech365_provenance_lesion=lesion)
    wd = agent.config.latent.world_dim
    pts = _field_with_wanting(agent, wd)
    before = _wanting(agent)
    traj = _traj(pts, tag)
    out = agent.hippocampal.spread_reverse_replay_wanting(traj)
    after = _wanting(agent)
    assert out["n_steps_spread"] == len(pts) - 1
    if expect_write:
        assert out["n_steps_accepted"] == len(pts) - 1
        assert out["n_steps_refused_provenance"] == 0
        assert out["accepted_mass"] > 0.0
        assert not torch.equal(before, after)
    else:
        assert out["n_steps_accepted"] == 0
        assert out["n_steps_refused_provenance"] == len(pts) - 1
        assert out["accepted_mass"] == 0.0
        assert torch.equal(before, after)
    # Boundary lesion never mutates the sender's label.
    assert traj.hypothesis_tag is tag
    assert out["mech365_lesion_override"] is (lesion == "drop_at_consolidation" and tag)
    assert out["mech365_sham_override"] is (lesion == "sham_real_only" and not tag)


def test_invalid_lesion_value_is_refused_loudly():
    env = _env()
    with pytest.raises(ValueError):
        _agent(env, mech365_provenance_lesion="drop")


def test_knobs_reachable_through_from_dims_and_default_noop():
    env = _env()
    a = _agent(env)
    h = a.config.hippocampal
    assert h.mech365_suppress_replay_provenance_stamp is False
    assert h.rem_route_forward_replay_to_consolidation is False
    assert h.mech365_provenance_lesion == "off"
    b = _agent(env, rem_route_forward_replay_to_consolidation=True,
               mech365_provenance_lesion="sham_real_only",
               mech365_suppress_replay_provenance_stamp=True)
    hb = b.config.hippocampal
    assert hb.rem_route_forward_replay_to_consolidation is True
    assert hb.mech365_provenance_lesion == "sham_real_only"
    assert hb.mech365_suppress_replay_provenance_stamp is True


# --------------------------------------------------------------------------- #
# E2: REM routing -- live, selective, and bit-identical OFF                    #
# --------------------------------------------------------------------------- #

def _one_hot(i, n):
    a = torch.zeros(1, n)
    a[0, int(i)] = 1.0
    return a


def _sense(agent, obs):
    lat = agent.sense(obs["body_state"], obs["world_state"],
                      obs_harm=obs.get("harm_obs"), obs_harm_a=obs.get("harm_obs_a"),
                      obs_harm_history=obs.get("harm_history"))
    if agent.clock.advance().get("e1_tick", False):
        agent._e1_tick(lat)
    return lat


def _run(n_cycles=3, **kw):
    """Scripted walk-to-resource + sleep cycles (the V3-EXQ-842/1085 harness)."""
    env = _env(11)
    agent = _agent(env, seed=11, use_offline_wanting_spread=True, **kw)
    rng = torch.Generator().manual_seed(11)
    totals = {}
    for c in range(n_cycles):
        _f, obs = env.reset_to(agent_pos=(1, 1), hazard_positions=[], resource_positions=[(6, 6)])
        agent.reset()
        agent.e1.reset_hidden_state()
        for _ in range(25):
            lat = _sense(agent, obs)
            agent.residue_field.rbf_field.add_residue(lat.z_world, 1e-3)
            ax, ay = env.agent_x, env.agent_y
            if torch.rand(1, generator=rng).item() < 0.1:
                act = int(torch.randint(0, 4, (1,), generator=rng).item())
            elif ax != 6:
                act = 1 if ax < 6 else 0
            else:
                act = 3 if ay < 6 else 2
            agent._record_exploration_action(_one_hot(act, env.action_dim))
            _f, harm, done, _i, obs = env.step(_one_hot(act, env.action_dim))
            if harm > 0.0 or done:
                break
        lat = _sense(agent, obs)
        agent.residue_field.rbf_field.add_residue(lat.z_world, 1e-3)
        if c == 0:
            agent.residue_field.update_valence(lat.z_world, VALENCE_WANTING, 2.0,
                                               hypothesis_tag=False)
        m = agent.run_sleep_cycle()
        for k, v in m.items():
            totals[k] = totals.get(k, 0.0) + float(v)
        agent.reset()
    return totals, _wanting(agent)


def test_rem_routing_default_off_is_bit_identical_and_emits_no_new_keys():
    t_off, w_off = _run()
    assert not any(k.startswith("rem_fwd_") or k.startswith("rem_rev_") for k in t_off)
    t_gate, w_gate = _run(rem_route_forward_replay_to_consolidation=True)
    # Gate intact: forward (imagined) replay reaches the writer and is REFUSED...
    assert t_gate["rem_fwd_spread_n_presented"] > 0
    assert t_gate["rem_fwd_spread_n_refused"] > 0
    assert t_gate["rem_fwd_spread_n_accepted"] == 0
    assert t_gate["rem_fwd_spread_accepted_mass"] == 0.0
    # ...so committed history is bit-identical to the unrouted substrate, and
    # the read (scoring) path is unchanged.
    assert torch.equal(w_off, w_gate)
    assert t_gate["rem_n_rollouts"] == t_off["rem_n_rollouts"]
    assert t_gate["rem_mean_harm_terrain"] == pytest.approx(t_off["rem_mean_harm_terrain"], abs=1e-9)


def test_rem_boundary_lesion_and_source_canary_contaminate_committed_history():
    _t, w_gate = _run(rem_route_forward_replay_to_consolidation=True)
    t_les, w_les = _run(rem_route_forward_replay_to_consolidation=True,
                        mech365_provenance_lesion="drop_at_consolidation")
    t_can, w_can = _run(rem_route_forward_replay_to_consolidation=True,
                        mech365_suppress_replay_provenance_stamp=True)
    t_sham, w_sham = _run(rem_route_forward_replay_to_consolidation=True,
                          mech365_provenance_lesion="sham_real_only")
    assert t_les["rem_fwd_spread_accepted_mass"] > 0.0
    assert t_les["rem_fwd_spread_n_lesion_overrides"] > 0
    assert not torch.equal(w_les, w_gate)
    assert t_can["rem_fwd_spread_accepted_mass"] > 0.0
    assert not torch.equal(w_can, w_gate)
    assert t_sham["rem_rev_spread_n_sham_overrides"] > 0
    assert torch.equal(w_sham, w_gate)
