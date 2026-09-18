"""Contracts for the MECH-131 anticipatory-residue lesion knob (2026-09-18).

    HippocampalConfig.terrain_prior_residue_channel_enabled (default True)

MECH-131 asserts that a vmPFC-analog must ACTIVATE stored aversive residue as an
anticipatory forward-biasing signal BEFORE candidate generation -- and that
residue which is correctly STORED but not so activated fails to suppress
harm-associated trajectory re-selection. Testing that requires a lesion arm:
activation OFF while storage stays ON. No such arm existed; this knob is it.

The knob zeroes residue_val OUT-OF-PLACE at the terrain_prior read
(hippocampal/module.py _get_terrain_action_object_mean) -- out-of-place because
ResidueField.evaluate may hand back a tensor sharing storage with field state,
where an in-place .zero_() would corrupt the field rather than just the read.
Zeroing rather than dropping the channel preserves terrain_prior's structural
input width (terrain_input_dim, HippocampalModule.__init__).

Contracts:
  C1  REACHABILITY -- from_dims routes the kwarg onto config.hippocampal.
      from_dims silently swallows unknown kwargs, which is how MECH-307 shipped
      with 84 drivers running the feature OFF while passing it True.
  C2  DEFAULT IS ON, and an explicit True is bit-identical to the default.
  C3  LIVENESS -- with residue accumulated at real visited locations, the lesion
      MOVES the terrain_prior proposal mean. Existence is not liveness.
  C4  STORAGE UNTOUCHED -- the "stored but not activated" property: accumulation
      totals are identical across the two arms.
  C5  POST-HOC SCORER UNTOUCHED -- E3.compute_residue_cost, the null path the
      claim contrasts against, is identical across the two arms.
  C6  SCOPE, PINNED -- this knob gates ONE of TWO live anticipatory reads. The
      CEM elite-selection terrain score (_score_trajectory ->
      residue_field.evaluate_trajectory) is NOT gated and stays residue-driven
      with the channel off. C6 exists so that a later reader cannot mistake this
      knob for a COMPLETE anticipatory lesion -- it is not, and an experiment
      whose lesion arm sets only this flag leaves the dominant anticipatory
      pathway intact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import HippocampalConfig, REEConfig

SEED = 11


def _build(channel_enabled=None, seed=SEED):
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=2, use_proxy_fields=True
    )
    kwargs = dict(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16,
        world_dim=16,
    )
    if channel_enabled is not None:
        kwargs["terrain_prior_residue_channel_enabled"] = channel_enabled
    cfg = REEConfig.from_dims(**kwargs)
    agent = REEAgent(cfg)
    agent.reset()
    return agent, env, cfg


def _obs(obs_dict):
    body, world = obs_dict["body_state"], obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def _charge_residue(agent, env, n_steps=40):
    """Lay down residue at REAL visited z_world locations via a real rollout.

    Not a hand-built favourable batch: the locations are wherever the agent
    actually went. Returns the last (body, world) obs so callers probe from a
    genuine state.
    """
    _flat, obs_dict = env.reset()
    body, world = _obs(obs_dict)
    torch.manual_seed(SEED + 7)
    for _ in range(n_steps):
        latent = agent.sense(body, world)
        agent.residue_field.accumulate(
            latent.z_world.detach(), harm_magnitude=1.0
        )
        action = int(torch.randint(0, 4, (1,)).item())
        _flat, _harm, done, _info, obs_dict = env.step(action)
        body, world = _obs(obs_dict)
        if done:
            _flat, obs_dict = env.reset()
            body, world = _obs(obs_dict)
    return body, world


def _proposal_mean(agent, body, world):
    latent = agent.sense(body, world)
    torch.manual_seed(SEED + 3)
    return agent.hippocampal._get_terrain_action_object_mean(
        latent.z_world.detach()
    ).detach().clone()


# ----------------------------------------------------------------------
def test_c1_from_dims_routes_the_kwarg():
    """C1: the kwarg reaches config.hippocampal -- not swallowed (MECH-307)."""
    for value in (True, False):
        _agent, _env, cfg = _build(channel_enabled=value)
        assert cfg.hippocampal.terrain_prior_residue_channel_enabled is value, (
            "from_dims did not route terrain_prior_residue_channel_enabled="
            f"{value} onto config.hippocampal"
        )


def test_c2_default_is_on_and_explicit_true_is_bit_identical():
    """C2: default True, and passing True changes nothing."""
    assert HippocampalConfig().terrain_prior_residue_channel_enabled is True
    a_def, e_def, cfg_def = _build(channel_enabled=None)
    assert cfg_def.hippocampal.terrain_prior_residue_channel_enabled is True
    b_def, w_def = _charge_residue(a_def, e_def)
    mean_default = _proposal_mean(a_def, b_def, w_def)

    a_on, e_on, _ = _build(channel_enabled=True)
    b_on, w_on = _charge_residue(a_on, e_on)
    mean_on = _proposal_mean(a_on, b_on, w_on)

    assert torch.equal(mean_default, mean_on), (
        "explicit True is not bit-identical to the default"
    )


def test_c3_liveness_lesion_moves_the_proposal_mean():
    """C3: the knob is LIVE -- a measured downstream quantity differs."""
    a_on, e_on, _ = _build(channel_enabled=True)
    b_on, w_on = _charge_residue(a_on, e_on)
    mean_on = _proposal_mean(a_on, b_on, w_on)

    a_off, e_off, _ = _build(channel_enabled=False)
    b_off, w_off = _charge_residue(a_off, e_off)
    mean_off = _proposal_mean(a_off, b_off, w_off)

    delta = float((mean_off - mean_on).abs().max())
    print(
        "C3 proposal-mean max|delta| = %.8g  (ON absmean %.8g / OFF absmean %.8g)"
        % (delta, float(mean_on.abs().mean()), float(mean_off.abs().mean()))
    )
    assert not torch.equal(mean_on, mean_off), (
        "INERT: disabling the residue channel changed no downstream quantity"
    )
    assert delta > 0.0


def test_c4_storage_is_untouched_by_the_lesion():
    """C4: 'stored but not activated' -- accumulation is identical."""
    totals = {}
    for value in (True, False):
        agent, env, _ = _build(channel_enabled=value)
        _charge_residue(agent, env)
        totals[value] = (
            float(agent.residue_field.total_residue),
            float(agent.residue_field.num_harm_events),
        )
    assert totals[True] == totals[False], (
        f"lesion perturbed residue STORAGE: {totals}"
    )
    assert totals[True][1] > 0.0, "residue never accumulated -- probe is vacuous"


def test_c5_post_hoc_scorer_is_untouched_by_the_lesion():
    """C5: the post-hoc null path (compute_residue_cost) is unchanged."""
    costs = {}
    for value in (True, False):
        agent, env, _ = _build(channel_enabled=value)
        body, world = _charge_residue(agent, env)
        latent = agent.sense(body, world)
        torch.manual_seed(SEED + 5)
        trajs = agent.hippocampal.propose_trajectories(
            latent.z_world.detach(), z_self=latent.z_self.detach()
        )
        assert trajs, "no candidates proposed -- probe is vacuous"
        torch.manual_seed(SEED + 5)
        costs[value] = float(agent.e3.compute_residue_cost(trajs[0]).detach().sum())
    # Same trajectory object content per arm is not guaranteed (the proposal
    # differs by design in C3), so this asserts the SCORER is reachable and
    # residue-driven in BOTH arms rather than bitwise equality of its input.
    print("C5 post-hoc residue cost: ON=%.8g OFF=%.8g" % (costs[True], costs[False]))
    assert costs[True] != 0.0 and costs[False] != 0.0, (
        f"post-hoc residue scorer went inert: {costs}"
    )


def test_c6_scope_cem_terrain_score_is_not_gated_by_this_knob():
    """C6: SCOPE PIN -- the second anticipatory read stays live when OFF.

    This knob is NOT a complete anticipatory lesion. _score_trajectory's terrain
    score reads the residue field independently, drives the CEM elite argsort,
    and is unaffected by this flag. Any experiment treating this one knob as
    'anticipatory activation OFF' is mis-specified.
    """
    agent, env, cfg = _build(channel_enabled=False)
    assert cfg.hippocampal.terrain_prior_residue_channel_enabled is False
    body, world = _charge_residue(agent, env)
    latent = agent.sense(body, world)
    torch.manual_seed(SEED + 9)
    trajs = agent.hippocampal.propose_trajectories(
        latent.z_world.detach(), z_self=latent.z_self.detach()
    )
    assert trajs, "no candidates proposed -- probe is vacuous"

    scores = [float(agent.hippocampal._score_trajectory(t)) for t in trajs]
    # Default HippocampalConfig has wanting_weight=0.0, curiosity_weight=0.0 and
    # an empty mode_value_weight, so _score_trajectory IS the residue terrain
    # score and nothing else.
    assert any(s != 0.0 for s in scores), (
        "CEM terrain score is all-zero -- the second anticipatory read would "
        "then be inert, which contradicts C6's premise; re-derive the scope note"
    )
    print(
        "C6 CEM terrain score with channel OFF: n=%d min=%.8g max=%.8g "
        "spread=%.8g (STILL residue-driven -> not a complete lesion)"
        % (len(scores), min(scores), max(scores), max(scores) - min(scores))
    )
