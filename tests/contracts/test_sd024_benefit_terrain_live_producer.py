"""Contract tests for the SD-024 benefit-terrain LIVE-PATH PRODUCER (2026-07-20).

Why this file exists
--------------------
SD-024 landed 2026-07-16 with 13 contracts in
test_sd024_da_modulated_rbf_density.py. Every one of them exercises ResidueField
directly -- they call rf.accumulate_benefit(...) themselves and then assert on the
resulting field. That is a valid IN-VITRO validation of the allocation mechanism,
and it is also precisely why the following went unnoticed for four days:

  ResidueField.accumulate_benefit had NO CALLER anywhere in ree_core/.

Its only two write sites into benefit_rbf_field (field.py:673, :682) live inside
that one method, and nothing in the agent loop invoked it -- agent.py called
update_valence, accumulate, accumulate_safety and evaluate_safety, but never
accumulate_benefit. The consequences, measured on darwin-arm64 over a real
warmup_train loop with curiosity_weight=0.5:

  - benefit_rbf_field.active_mask.sum() == 0 and num_benefit_events == 0.0, even
    with benefit_terrain_enabled=True AND use_da_modulated_rbf_density=True;
  - RBFLayer.compute_local_density early-returns zeros on an empty active mask
    (field.py:273), so HippocampalModule.compute_representational_density
    returned exactly 0.0;
  - therefore in _curiosity_bonus (hippocampal/module.py:870-885),
    novelty = density * (1 - familiarity) = 0, and the returned bonus was
    curiosity_weight * 0 = 0 -- measured 0.0 on all 14432 live calls.
    The use_curiosity_familiarity True/False ablation was BIT-IDENTICAL,
    confirming familiarity was not the binding constraint.

So the SD-025 curiosity drive contributed exactly zero to CEM trajectory scoring
in every live agent run. The gap was a missing PRODUCER, not a broken mechanism.

The contracts below therefore assert at the level the in-vitro suite could not:
through the real REEAgent API that experiment drivers actually call, against a
real CausalGridWorldV2 episode loop -- never by calling accumulate_benefit
directly. A test that populates the terrain itself cannot detect a missing
producer; that is the whole lesson here.

Contracts:
  C1  Bit-identical OFF -- default config leaves the live producer disabled, and
      a real episode loop with real reward contacts populates nothing.
  C2  THE MISSING CONTRACT -- with benefit_terrain_enabled + the live producer on,
      a real episode loop DOES populate the terrain (active centers > 0,
      num_benefit_events > 0).
  C3  Downstream liveness -- once populated by the live path,
      compute_representational_density (the SD-025 curiosity hook) reads > 0 at
      the visited location. This is the exact quantity that measured 0.0.
  C4  MECH-094 gate -- a hypothesis_tag=True latent (replay / DMN tick) does not
      build benefit terrain via the live path.
  C5  Consummatory threshold -- sub-threshold benefit_exposure (proximity
      gradient, not consumption) does not write.
  C6  familiarity_bandwidth default is pinned at 0.20, not the degenerate 1.0.
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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_agent(seed: int = 11, terrain=False, live_producer=False, da=False):
    """Real REEAgent + CausalGridWorldV2, configured per the flags under test."""
    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=2,
        use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16, world_dim=16,
    )
    # Residue-terrain flags are not from_dims kwargs; set them on the nested
    # ResidueConfig before REEAgent.__init__ builds the ResidueField from it.
    cfg.residue.benefit_terrain_enabled = terrain
    cfg.residue.benefit_terrain_live_producer = live_producer
    cfg.residue.use_da_modulated_rbf_density = da
    if da:
        cfg.residue.da_allocation_scale = 4.0
    agent = REEAgent(cfg)
    agent.reset()
    return agent, env


def _obs(obs_dict):
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def _run_episode(agent, env, n_steps=24, benefit=0.8, drive=0.7):
    """A real episode loop: env.step + agent.sense every tick, and the reward-
    contact hook driven exactly as experiment drivers drive it.

    benefit_exposure is supplied explicitly rather than harvested from a
    stochastic resource collision so the contract is deterministic -- what is
    under test is whether the LIVE AGENT PATH lays down terrain when a reward
    contact occurs, not whether a short random rollout happens to collide with
    a resource. update_z_goal is the real driver-facing API (the same one
    v3_exq_432 / 540f / 540g call); nothing here touches accumulate_benefit.
    """
    _flat, obs_dict = env.reset()
    body, world = _obs(obs_dict)
    contacts = 0
    for t in range(n_steps):
        agent.sense(body, world)
        action = int(torch.randint(0, 4, (1,)).item())
        # CausalGridWorldV2.step -> (flat_obs, harm_signal, done, info, obs_dict)
        _flat, _harm, done, _info, obs_dict = env.step(action)
        body, world = _obs(obs_dict)
        # Reward contact every third tick.
        if t % 3 == 0:
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)
            contacts += 1
        if done:
            _flat, obs_dict = env.reset()
            body, world = _obs(obs_dict)
    return contacts


def _terrain_state(agent):
    rf = agent.residue_field
    n_active = int(rf.benefit_rbf_field.active_mask.sum().item())
    n_events = float(rf.num_benefit_events)
    return n_active, n_events


# ----------------------------------------------------------------------
# C1 bit-identical OFF
# ----------------------------------------------------------------------
def test_c1_default_config_leaves_live_producer_off():
    """Default config: the producer flag is False and a real episode loop with
    real reward contacts populates nothing. This is the backward-compat
    guarantee -- every pre-2026-07-20 config must be unaffected."""
    from ree_core.utils.config import ResidueConfig

    assert ResidueConfig().benefit_terrain_live_producer is False, (
        "live producer must default OFF (backward compat)"
    )

    # Terrain built but producer off -- the state every existing config is in.
    agent, env = _build_agent(terrain=True, live_producer=False)
    contacts = _run_episode(agent, env)
    assert contacts > 0, "test harness must actually deliver reward contacts"

    n_active, n_events = _terrain_state(agent)
    assert n_active == 0, (
        f"producer OFF must leave the benefit terrain empty, got {n_active} "
        "active centers"
    )
    assert n_events == 0.0, (
        f"producer OFF must record no benefit events, got {n_events}"
    )


# ----------------------------------------------------------------------
# C2 the contract whose absence hid the defect
# ----------------------------------------------------------------------
def test_c2_live_episode_loop_populates_benefit_terrain():
    """THE MISSING CONTRACT. With the terrain and the live producer enabled, a
    real episode loop must populate benefit_rbf_field. Before 2026-07-20 this
    assertion failed at 0 active centers on every configuration, because
    accumulate_benefit had no caller in ree_core/."""
    agent, env = _build_agent(terrain=True, live_producer=True)
    contacts = _run_episode(agent, env)

    n_active, n_events = _terrain_state(agent)
    assert n_active > 0, (
        "live episode loop must populate the benefit terrain when the producer "
        f"is enabled; got {n_active} active centers after {contacts} reward "
        "contacts (this is the exact 2026-07-20 defect signature)"
    )
    assert n_events > 0.0, (
        f"num_benefit_events must advance on the live path, got {n_events}"
    )


def test_c2b_da_cluster_allocation_reaches_the_live_path():
    """The SD-024 DA cluster path is reachable from the live producer: with the
    master switch on, the same number of reward contacts allocates strictly more
    centers than the single-center default path."""
    agent_plain, env_plain = _build_agent(seed=5, terrain=True, live_producer=True, da=False)
    _run_episode(agent_plain, env_plain)
    n_plain, _ = _terrain_state(agent_plain)

    agent_da, env_da = _build_agent(seed=5, terrain=True, live_producer=True, da=True)
    _run_episode(agent_da, env_da)
    n_da, _ = _terrain_state(agent_da)

    assert n_plain > 0 and n_da > 0, "both arms must populate"
    assert n_da > n_plain, (
        "DA-modulated allocation must expand representation on the live path: "
        f"da={n_da} centers vs plain={n_plain}"
    )


# ----------------------------------------------------------------------
# C3 the downstream quantity that measured exactly 0.0
# ----------------------------------------------------------------------
def test_c3_representational_density_nonzero_after_live_population():
    """compute_representational_density -- the SD-025 curiosity hook -- must read
    > 0 at a visited location once the live path has populated the terrain. This
    is the quantity that returned exactly 0.0 on all 14432 live calls."""
    agent, env = _build_agent(terrain=True, live_producer=True, da=True)
    _run_episode(agent, env)

    assert agent.hippocampal is not None, "hippocampal module required for C3"
    z = agent._current_latent.z_world
    density = agent.hippocampal.compute_representational_density(z)
    assert float(density.max().item()) > 0.0, (
        "representational density must be non-zero at a live-visited location "
        "once the producer has run; 0.0 here reproduces the defect"
    )


# ----------------------------------------------------------------------
# C4 MECH-094
# ----------------------------------------------------------------------
def test_c4_hypothesis_tag_blocks_live_accumulation():
    """MECH-094: a replay / DMN tick must not build benefit terrain. The live
    producer reads hypothesis_tag off the current latent rather than hardcoding
    False, so a simulation-tagged latent is refused at the call site as well as
    inside accumulate_benefit."""
    agent, env = _build_agent(terrain=True, live_producer=True)
    _flat, obs_dict = env.reset()
    body, world = _obs(obs_dict)
    agent.sense(body, world)

    agent._current_latent.hypothesis_tag = True
    for _ in range(6):
        agent.update_z_goal(benefit_exposure=0.9, drive_level=0.8)

    n_active, n_events = _terrain_state(agent)
    assert n_active == 0 and n_events == 0.0, (
        "MECH-094: hypothesis_tag=True must not create benefit terrain, got "
        f"{n_active} centers / {n_events} events"
    )

    # And the same agent DOES accumulate once the tag clears -- proving the
    # block above is the tag, not an inert wiring path.
    agent._current_latent.hypothesis_tag = False
    agent.update_z_goal(benefit_exposure=0.9, drive_level=0.8)
    n_active_after, _ = _terrain_state(agent)
    assert n_active_after > 0, (
        "waking tick on the same agent must accumulate (guards against the "
        "C4 pass being vacuous)"
    )


# ----------------------------------------------------------------------
# C5 consummatory threshold
# ----------------------------------------------------------------------
def test_c5_subthreshold_benefit_does_not_write():
    """Below benefit_live_producer_threshold the signal is a proximity gradient,
    not consumption, and must not lay down terrain -- the same convention
    update_liking() applies via liking_threshold."""
    agent, env = _build_agent(terrain=True, live_producer=True)
    thresh = float(agent.config.residue.benefit_live_producer_threshold)
    _flat, obs_dict = env.reset()
    body, world = _obs(obs_dict)
    agent.sense(body, world)

    for _ in range(8):
        agent.update_z_goal(benefit_exposure=thresh * 0.5, drive_level=0.8)
    n_active, _ = _terrain_state(agent)
    assert n_active == 0, (
        f"sub-threshold benefit must not write, got {n_active} centers"
    )

    agent.update_z_goal(benefit_exposure=thresh, drive_level=0.8)
    n_active_at, _ = _terrain_state(agent)
    assert n_active_at > 0, "at-threshold benefit must write (gate is >=)"


# ----------------------------------------------------------------------
# C6 pin the familiarity bandwidth default
# ----------------------------------------------------------------------
def test_c6_familiarity_bandwidth_default_is_not_degenerate():
    """familiarity_bandwidth was 1.0, which V3-EXQ-786a measured at exactly
    +0.000 effect -- the degenerate case. FamiliarityTracker.query is a CLAMPED
    SUM over anchors, and the same constant is the association threshold in
    update(), so at 1.0 the ~3 active anchors' near-unit weights pin the clamp
    and the (1 - familiarity) novelty discount collapses. Pinned at the sweep
    peak (0.20 -> +0.171) so it cannot silently drift back."""
    default_bw = HippocampalConfig().familiarity_bandwidth
    assert abs(default_bw - 0.20) < 1e-9, (
        f"familiarity_bandwidth default must be 0.20, got {default_bw}"
    )
    assert default_bw < 1.0, "1.0 is the degenerate saturating value"
