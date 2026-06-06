"""Contract tests for MECH-314a Phase 2 (Candidate 5A).

Architecture doc:
REE_assembly/docs/architecture/mech_314a_phase2_novelty_source_design.md
section 6 (recommendation + "What candidate 5A specifically commits to").

Phase 2 adds, on top of the Phase-1 ResidueField-only 314a novelty source:
  - a rolling z_world visitation buffer (collections.deque on REEAgent,
    appended in sense() after update_per_stream_vs, MECH-094-gated), used as
    the "visitation" novelty comparison set;
  - a first-action one-hot augmentation leg (substrate-robustness bypass)
    with never / auto / always engagement policies.

Four contracts (the chip-mandated coverage):

  C1: bit-identical OFF. Default config (novelty_source="residue",
      use_first_action_onehot=False, first_action_augmentation_policy=
      "never") reproduces the Phase-1 single-call _compute_novelty path
      exactly; passing visitation_source / first_action_onehots is ignored.
      Default agent has curiosity=None and no visitation buffer.

  C2: the visitation buffer accumulates ONLY on waking ticks (MECH-094
      gate): waking sense() ticks append z_world; a hypothesis_tag=True
      (simulation/replay) latent does not.

  C3: per-candidate spread > 0 on harm-free runs with visitation ON +
      SD-056 ON (diverse candidate z_world). The Phase-1 residue source is
      silent on a harm-free episode (empty ResidueField); the visitation
      source produces a non-zero per-candidate novelty spread.

  C4: augmentation engages when the un-augmented per-candidate spread falls
      below threshold for N consecutive ticks (auto policy), and disengages
      when the spread recovers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.policy import StructuredCuriosity, StructuredCuriosityConfig
from ree_core.utils.config import REEConfig


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
class _RBF:
    """Stand-in rbf_field with `n_active` active centers (or zero)."""

    def __init__(self, world_dim: int, n_active: int, total: int = 4):
        self.centers = torch.randn(total, world_dim)
        self.active_mask = torch.tensor(
            [True] * n_active + [False] * (total - n_active)
        )


class _Residue:
    def __init__(self, world_dim: int, n_active: int):
        self.rbf_field = _RBF(world_dim, n_active)


def _module(**overrides) -> StructuredCuriosity:
    cfg = StructuredCuriosityConfig(use_structured_curiosity=True, **overrides)
    return StructuredCuriosity(cfg)


def _build_visitation_agent(seed: int = 7, **flags):
    """Small REEAgent with curiosity ON and the visitation novelty source."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=5, num_hazards=1, num_resources=1,
        use_proxy_fields=True,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        self_dim=16, world_dim=16,
        use_structured_curiosity=True,
        **flags,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return agent, body, world


# ----------------------------------------------------------------------
# C1 bit-identical OFF
# ----------------------------------------------------------------------
def test_c1_default_module_ignores_phase2_inputs():
    """Default (residue / never) module: passing visitation_source and
    first_action_onehots does not change the output -- bit-identical to the
    Phase-1 single-call _compute_novelty path."""
    torch.manual_seed(0)
    K, world_dim = 6, 16
    summaries = torch.randn(K, world_dim) * 2.0
    res = _Residue(world_dim, n_active=3)
    buf = [torch.randn(world_dim) for _ in range(20)]
    oh = torch.zeros(K, 4)
    for i in range(K):
        oh[i, i % 4] = 1.0

    mod = _module()  # all Phase-2 defaults
    bias_plain = mod.compute_score_bias(summaries, residue_field=res, e3=None)
    bias_with_phase2 = mod.compute_score_bias(
        summaries,
        residue_field=res,
        e3=None,
        visitation_source=buf,
        first_action_onehots=oh,
    )
    assert torch.allclose(bias_plain, bias_with_phase2), (
        "residue/never defaults must ignore Phase-2 inputs (bit-identical OFF)"
    )
    # And it must equal the direct Phase-1 novelty arithmetic.
    nov = mod._compute_novelty(summaries, res)
    expected = torch.clamp(
        -mod.config.curiosity_novelty_weight * nov,
        min=-mod.config.curiosity_bias_scale,
        max=mod.config.curiosity_bias_scale,
    )
    assert torch.allclose(bias_plain, expected)


def test_c1_default_agent_no_buffer():
    """Default REEConfig: no curiosity module, no visitation buffer."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    env = CausalGridWorldV2(seed=1, size=5, num_hazards=1, num_resources=1,
                            use_proxy_fields=True)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=4, self_dim=16, world_dim=16,
    )
    agent = REEAgent(cfg)
    assert agent.curiosity is None
    assert agent._zworld_visitation_buffer is None


def test_c1_curiosity_on_residue_default_has_no_buffer():
    """Curiosity ON but novelty_source default 'residue': still no buffer
    allocated (bit-identical to Phase-1 curiosity)."""
    agent, _b, _w = _build_visitation_agent(curiosity_novelty_source="residue")
    assert agent.curiosity is not None
    assert agent._zworld_visitation_buffer is None


# ----------------------------------------------------------------------
# C2 visitation buffer accumulates only on waking ticks (MECH-094 gate)
# ----------------------------------------------------------------------
def test_c2_buffer_accumulates_on_waking_ticks():
    agent, body, world = _build_visitation_agent(curiosity_novelty_source="visitation")
    assert agent._zworld_visitation_buffer is not None
    assert len(agent._zworld_visitation_buffer) == 0
    with torch.no_grad():
        for _ in range(4):
            agent.sense(obs_body=body, obs_world=world)
    assert len(agent._zworld_visitation_buffer) == 4, (
        "waking sense() ticks must append z_world to the visitation buffer"
    )


def test_c2_mech094_gate_blocks_simulation_writes():
    """A hypothesis_tag=True latent (simulation/replay) must NOT append."""
    agent, body, world = _build_visitation_agent(curiosity_novelty_source="visitation")
    with torch.no_grad():
        agent.sense(obs_body=body, obs_world=world)  # one waking write
    assert len(agent._zworld_visitation_buffer) == 1

    # Force the next encode to return a simulation-tagged latent.
    real_encode = agent.latent_stack.encode

    def _sim_encode(*args, **kwargs):
        latent = real_encode(*args, **kwargs)
        latent.hypothesis_tag = True
        return latent

    agent.latent_stack.encode = _sim_encode  # type: ignore[assignment]
    with torch.no_grad():
        agent.sense(obs_body=body, obs_world=world)
    assert len(agent._zworld_visitation_buffer) == 1, (
        "hypothesis_tag=True ticks must not append (MECH-094 gate)"
    )


def test_c2_buffer_respects_maxlen():
    agent, body, world = _build_visitation_agent(
        curiosity_novelty_source="visitation", curiosity_visitation_buffer_len=3,
    )
    assert agent._zworld_visitation_buffer.maxlen == 3
    with torch.no_grad():
        for _ in range(6):
            agent.sense(obs_body=body, obs_world=world)
    assert len(agent._zworld_visitation_buffer) == 3


# ----------------------------------------------------------------------
# C3 per-candidate spread > 0 with visitation ON (SD-056 ON), harm-free
# ----------------------------------------------------------------------
def test_c3_visitation_lifts_spread_when_residue_empty():
    torch.manual_seed(3)
    K, world_dim = 8, 16
    # SD-056 ON: candidate z_world summaries genuinely diverge.
    summaries = torch.randn(K, world_dim)
    # Harm-free episode: ResidueField has NO active centers.
    empty_res = _Residue(world_dim, n_active=0)
    buf = [torch.randn(world_dim) for _ in range(32)]

    # Phase-1 residue source on a harm-free episode is silent (zero bias).
    mod_res = _module(novelty_source="residue")
    bias_res = mod_res.compute_score_bias(summaries, residue_field=empty_res, e3=None)
    assert torch.allclose(bias_res, torch.zeros(K)), (
        "residue source must be silent on a harm-free (empty-residue) episode"
    )

    # Visitation source produces a non-zero per-candidate novelty spread.
    mod_vis = _module(novelty_source="visitation")
    bias_vis = mod_vis.compute_score_bias(
        summaries, residue_field=empty_res, e3=None, visitation_source=buf,
    )
    assert mod_vis._last_novelty_source_used == "visitation"
    spread = float(bias_vis.max() - bias_vis.min())
    assert spread > 0.0, (
        f"visitation source must yield non-zero per-candidate spread, got {spread}"
    )


def test_c3_auto_source_falls_back_to_residue_when_buffer_empty():
    torch.manual_seed(4)
    K, world_dim = 6, 16
    summaries = torch.randn(K, world_dim) * 2.0
    res = _Residue(world_dim, n_active=2)
    mod_auto = _module(novelty_source="auto")
    # Empty buffer -> auto falls back to residue.
    bias_auto = mod_auto.compute_score_bias(
        summaries, residue_field=res, e3=None, visitation_source=[],
    )
    assert mod_auto._last_novelty_source_used == "residue"
    mod_res = _module(novelty_source="residue")
    bias_res = mod_res.compute_score_bias(summaries, residue_field=res, e3=None)
    assert torch.allclose(bias_auto, bias_res)


# ----------------------------------------------------------------------
# C4 augmentation engages after N consecutive below-threshold ticks
# ----------------------------------------------------------------------
def test_c4_auto_augmentation_engages_after_n_ticks():
    world_dim = 16
    K = 8
    N = 3
    # Collapsed candidate summaries -> per-candidate spread == 0.
    collapsed = torch.ones(K, world_dim) * 0.5
    buf = [torch.randn(world_dim) for _ in range(20)]
    oh = torch.zeros(K, 4)
    for i in range(K):
        oh[i, i % 4] = 1.0

    mod = _module(
        novelty_source="visitation",
        use_first_action_onehot=True,
        first_action_augmentation_policy="auto",
        min_spread_consecutive_ticks=N,
        min_spread_threshold=0.01,
    )
    engaged_history = []
    for _ in range(N + 2):
        mod.compute_score_bias(
            collapsed, residue_field=None, e3=None,
            visitation_source=buf, first_action_onehots=oh,
        )
        engaged_history.append(mod._last_augmentation_engaged)
    # Not engaged for the first N-1 ticks; engaged from tick N onward.
    assert engaged_history[: N - 1] == [False] * (N - 1)
    assert all(engaged_history[N - 1:]), (
        f"augmentation must engage from the Nth consecutive low-spread tick: "
        f"{engaged_history}"
    )


def test_c4_auto_augmentation_disengages_on_recovery():
    world_dim = 16
    K = 8
    buf = [torch.randn(world_dim) for _ in range(20)]
    oh = torch.zeros(K, 4)
    for i in range(K):
        oh[i, i % 4] = 1.0
    mod = _module(
        novelty_source="visitation",
        use_first_action_onehot=True,
        first_action_augmentation_policy="auto",
        min_spread_consecutive_ticks=2,
        min_spread_threshold=0.01,
    )
    collapsed = torch.ones(K, world_dim) * 0.5
    diverse = torch.randn(K, world_dim) * 3.0
    # Engage via collapse.
    for _ in range(2):
        mod.compute_score_bias(collapsed, residue_field=None, e3=None,
                               visitation_source=buf, first_action_onehots=oh)
    assert mod._last_augmentation_engaged is True
    # Recover -> disengage.
    mod.compute_score_bias(diverse, residue_field=None, e3=None,
                           visitation_source=buf, first_action_onehots=oh)
    assert mod._last_augmentation_engaged is False
    assert mod._below_threshold_streak == 0


def test_c4_never_policy_never_augments():
    world_dim = 16
    K = 8
    collapsed = torch.ones(K, world_dim) * 0.5
    buf = [torch.randn(world_dim) for _ in range(20)]
    oh = torch.zeros(K, 4)
    for i in range(K):
        oh[i, i % 4] = 1.0
    mod = _module(
        novelty_source="visitation",
        use_first_action_onehot=True,
        first_action_augmentation_policy="never",
    )
    for _ in range(6):
        mod.compute_score_bias(collapsed, residue_field=None, e3=None,
                               visitation_source=buf, first_action_onehots=oh)
        assert mod._last_augmentation_engaged is False


# ----------------------------------------------------------------------
# MECH-094 module gate (defensive) + config validation
# ----------------------------------------------------------------------
def test_simulation_mode_returns_zeros_with_phase2_on():
    K, world_dim = 8, 16
    summaries = torch.randn(K, world_dim)
    buf = [torch.randn(world_dim) for _ in range(10)]
    oh = torch.zeros(K, 4)
    for i in range(K):
        oh[i, i % 4] = 1.0
    mod = _module(
        novelty_source="visitation",
        use_first_action_onehot=True,
        first_action_augmentation_policy="always",
    )
    bias = mod.compute_score_bias(
        summaries, residue_field=None, e3=None, simulation_mode=True,
        visitation_source=buf, first_action_onehots=oh,
    )
    assert torch.allclose(bias, torch.zeros(K))
    assert mod._last_n_simulation_skips == 1


def test_invalid_novelty_source_raises():
    with pytest.raises(ValueError):
        _module(novelty_source="bogus")


def test_invalid_augmentation_policy_raises():
    with pytest.raises(ValueError):
        _module(first_action_augmentation_policy="sometimes")
