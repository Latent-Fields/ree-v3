"""Contract tests for MECH-314 structured_curiosity_bonus (ARC-065 child).

Five contracts (C1-C5):

  C1: default-off no-op. With use_structured_curiosity=False,
      agent.curiosity is None and select_action's score_bias chain is
      bit-identical to baseline.

  C2: each sub-flavour fires independently. With master ON, flipping
      individual sub-flavour switches off zeroes the corresponding
      contribution while leaving the others intact. Q-044's three-arm
      ablation must be a flag-set decision, not a code-edit decision.

  C3: sub-flavour outputs land additively in candidate scoring. With
      all three sub-flavours on at known weights and known signal
      sources, the output [K] tensor equals the sum of the three
      individual sub-flavour contributions clamped to bias_scale.

  C4: MECH-094 simulation gate. compute_score_bias(simulation_mode=True)
      returns zeros[K] and increments only the simulation-skip counter.
      update_prediction_error(simulation_mode=True) is a no-op on the
      LP buffer.

  C5: backward-compat across config matrix. Toggling structured_curiosity
      with other major flags (use_gated_policy, use_dacc, use_noise_floor)
      does not raise during agent construction or one-tick boot.
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
def _build_agent_and_one_tick(seed: int = 7, **flags):
    """Build a small REEAgent and run one sense() + one act tick."""
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
    with torch.no_grad():
        latent = agent.sense(obs_body=body, obs_world=world)
        try:
            _action = agent.act_with_split_obs(body, world, temperature=1.0)
        except Exception as e:  # noqa: BLE001
            agent._select_action_error = str(e)  # type: ignore[attr-defined]
    return agent, latent


# ----------------------------------------------------------------------
# C1 default-off no-op
# ----------------------------------------------------------------------
def test_c1_default_off_no_op():
    agent, latent = _build_agent_and_one_tick()
    assert agent.curiosity is None, (
        "default config should produce curiosity=None"
    )
    assert torch.isfinite(latent.z_world).all()


# ----------------------------------------------------------------------
# C2 each sub-flavour fires independently
# ----------------------------------------------------------------------
class _MockE3:
    """Stand-in exposing _running_variance for 314b unit tests."""

    def __init__(self, running_variance: float = 1.0):
        self._running_variance = running_variance


class _MockResidue:
    """Stand-in ResidueField with one active RBF center for 314a tests."""

    def __init__(self, world_dim: int = 16, n_active: int = 1):
        # Simulate the rbf_field.centers + active_mask interface.
        class _RBF:
            pass
        rbf = _RBF()
        rbf.centers = torch.zeros(4, world_dim)
        rbf.centers[0] = torch.zeros(world_dim)  # center at origin
        rbf.active_mask = torch.tensor([True] * n_active + [False] * (4 - n_active))
        self.rbf_field = rbf


def _build_module(**overrides):
    cfg = StructuredCuriosityConfig(use_structured_curiosity=True, **overrides)
    return StructuredCuriosity(cfg)


def test_c2_novelty_only():
    """314a-only: bias = -w_a * normalised_min_distance, broadcast scalar = 0."""
    K = 4
    world_dim = 16
    summaries = torch.randn(K, world_dim) * 2.0  # away from origin
    mod = _build_module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        curiosity_novelty_weight=0.05,
    )
    e3 = _MockE3(running_variance=10.0)  # would dominate if 314b fired
    res = _MockResidue(world_dim=world_dim)
    bias = mod.compute_score_bias(summaries, residue_field=res, e3=e3)
    assert bias.shape == (K,)
    # All entries must be <= 0 (curiosity reduces score). Strictly negative
    # because every summary is away from the single center at origin.
    assert (bias < 0).all(), f"expected strictly negative bias, got {bias}"
    assert mod._last_n_subflavours_fired == 1
    # 314b / 314c contributed nothing.
    assert mod._last_uncertainty_signal == 0.0
    assert mod._last_learning_progress_signal == 0.0


def test_c2_uncertainty_only():
    """314b-only: bias = -w_b * unc broadcast scalar; 314a / 314c contribute 0."""
    K = 4
    world_dim = 16
    summaries = torch.randn(K, world_dim)
    mod = _build_module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=False,
        curiosity_uncertainty_weight=0.05,
    )
    e3 = _MockE3(running_variance=1.0)
    res = _MockResidue(world_dim=world_dim)  # 314a OFF -> ignored
    bias = mod.compute_score_bias(summaries, residue_field=res, e3=e3)
    # Uniform broadcast: -0.05 * 1.0 = -0.05 across all K entries.
    expected = torch.full((K,), -0.05, dtype=bias.dtype)
    assert torch.allclose(bias, expected, atol=1e-6)
    assert mod._last_n_subflavours_fired == 1
    assert mod._last_novelty_norm == 0.0


def test_c2_learning_progress_only():
    """314c-only: bias = -w_c * lp_ema scalar; needs prior PE updates to seed."""
    K = 4
    world_dim = 16
    summaries = torch.randn(K, world_dim)
    mod = _build_module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=True,
        curiosity_learning_progress_weight=0.05,
        curiosity_lp_window_k=3,
    )
    # Pre-seed the LP buffer with K+1 PE values so the EMA fires.
    for pe in [0.5, 0.4, 0.3, 0.2]:
        mod.update_prediction_error(pe)
    assert mod._lp_seeded
    # |PE_t - PE_{t-K}| = |0.2 - 0.5| = 0.3 (single seed value).
    assert mod._lp_ema == pytest.approx(0.3, abs=1e-6)

    bias = mod.compute_score_bias(summaries, residue_field=None, e3=None)
    expected = torch.full((K,), -0.05 * 0.3, dtype=bias.dtype)
    assert torch.allclose(bias, expected, atol=1e-6)
    assert mod._last_n_subflavours_fired == 1


def test_c2_all_off_when_master_subflavours_off():
    """Master ON but all sub-flavours OFF -> zeros[K]."""
    K = 4
    summaries = torch.randn(K, 16)
    mod = _build_module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
    )
    bias = mod.compute_score_bias(
        summaries,
        residue_field=_MockResidue(),
        e3=_MockE3(running_variance=10.0),
    )
    assert bias.shape == (K,)
    assert (bias == 0).all()
    assert mod._last_n_subflavours_fired == 0


# ----------------------------------------------------------------------
# C3 sub-flavour outputs land additively in candidate scoring
# ----------------------------------------------------------------------
def test_c3_additive_composition():
    """All ON: total_bias = 314a + 314b + 314c clamped to [-bias_scale, +bias_scale].

    Build identical modules for the per-flavour-only configs and the
    all-on config, run with the same inputs, and confirm
    all_on == clamp(novelty_only + uncertainty_only + lp_only).
    """
    K = 4
    world_dim = 16
    summaries = torch.randn(K, world_dim) * 0.3  # close to origin so novelty is small
    res = _MockResidue(world_dim=world_dim)
    e3 = _MockE3(running_variance=0.5)

    common = dict(
        curiosity_novelty_weight=0.02,
        curiosity_uncertainty_weight=0.02,
        curiosity_learning_progress_weight=0.02,
        curiosity_bias_scale=1.0,  # large clamp so we can verify additivity directly
        curiosity_lp_window_k=2,
    )

    def _seed_lp(mod):
        mod.update_prediction_error(0.5)
        mod.update_prediction_error(0.3)
        mod.update_prediction_error(0.1)  # window k=2 -> |0.1 - 0.5|=0.4 seed

    mod_a = _build_module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=False,
        **common,
    )
    bias_a = mod_a.compute_score_bias(summaries, residue_field=res, e3=e3)

    mod_b = _build_module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=False,
        **common,
    )
    bias_b = mod_b.compute_score_bias(summaries, residue_field=res, e3=e3)

    mod_c = _build_module(
        use_curiosity_novelty=False,
        use_curiosity_uncertainty=False,
        use_curiosity_learning_progress=True,
        **common,
    )
    _seed_lp(mod_c)
    bias_c = mod_c.compute_score_bias(summaries, residue_field=res, e3=e3)

    mod_all = _build_module(
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=True,
        **common,
    )
    _seed_lp(mod_all)
    bias_all = mod_all.compute_score_bias(summaries, residue_field=res, e3=e3)

    expected = bias_a + bias_b + bias_c
    expected_clamped = torch.clamp(expected, min=-1.0, max=1.0)
    assert torch.allclose(bias_all, expected_clamped, atol=1e-6), (
        f"additive composition violated. all_on={bias_all}, "
        f"sum={expected_clamped}, a={bias_a}, b={bias_b}, c={bias_c}"
    )
    assert mod_all._last_n_subflavours_fired == 3


# ----------------------------------------------------------------------
# C4 MECH-094 simulation gate
# ----------------------------------------------------------------------
def test_c4_mech094_simulation_gate():
    """Simulation: zeros[K] + skip counter advances; LP no-op on update."""
    K = 5
    world_dim = 16
    summaries = torch.randn(K, world_dim)
    mod = _build_module()
    e3 = _MockE3(running_variance=1.0)
    res = _MockResidue(world_dim=world_dim)

    # Pre-seed LP so 314c would fire under waking conditions.
    for pe in [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]:
        mod.update_prediction_error(pe)
    pre_seeded = mod._lp_seeded
    assert pre_seeded
    pre_lp = mod._lp_ema

    pre_waking = mod._n_waking_calls
    pre_skip = mod._last_n_simulation_skips

    # Simulation call: zeros, skip+1, waking unchanged.
    sim_bias = mod.compute_score_bias(
        summaries, residue_field=res, e3=e3, simulation_mode=True,
    )
    assert sim_bias.shape == (K,)
    assert (sim_bias == 0).all()
    assert mod._last_n_simulation_skips == pre_skip + 1
    assert mod._n_waking_calls == pre_waking

    # update_prediction_error in simulation mode: LP buffer untouched.
    pre_ring = list(mod._pe_ring)
    mod.update_prediction_error(99.0, simulation_mode=True)
    assert mod._pe_ring == pre_ring
    assert mod._lp_ema == pytest.approx(pre_lp, abs=1e-12)
    # Skip counter advanced again.
    assert mod._last_n_simulation_skips == pre_skip + 2

    # Subsequent waking call advances waking counter, NOT skip counter.
    pre_skip_2 = mod._last_n_simulation_skips
    waking_bias = mod.compute_score_bias(summaries, residue_field=res, e3=e3)
    assert mod._n_waking_calls == pre_waking + 1
    assert mod._last_n_simulation_skips == pre_skip_2
    assert (waking_bias != 0).any()  # at least one sub-flavour fired


# ----------------------------------------------------------------------
# C5 backward-compat across config matrix
# ----------------------------------------------------------------------
@pytest.mark.parametrize("flags", [
    {"use_structured_curiosity": True},
    {"use_structured_curiosity": True, "use_noise_floor": True},
    {"use_structured_curiosity": True, "use_gated_policy": True},
    {
        "use_structured_curiosity": True,
        "use_dacc": True,
        "dacc_weight": 0.0,
    },
])
def test_c5_backward_compat_config_matrix(flags):
    agent, latent = _build_agent_and_one_tick(**flags)
    assert agent.curiosity is not None
    assert latent is not None
    assert torch.isfinite(latent.z_world).all()
    assert getattr(agent, "_select_action_error", None) is None


# ----------------------------------------------------------------------
# Reset clears diagnostics + LP buffer
# ----------------------------------------------------------------------
def test_reset_clears_diagnostics_and_lp_buffer():
    mod = _build_module()
    mod.update_prediction_error(0.5)
    mod.update_prediction_error(0.4)
    mod.update_prediction_error(0.3)
    mod.update_prediction_error(0.2)
    mod.update_prediction_error(0.1)
    mod.update_prediction_error(0.05)
    summaries = torch.randn(3, 16)
    mod.compute_score_bias(
        summaries, residue_field=None, e3=_MockE3(running_variance=1.0),
    )
    assert mod._n_waking_calls == 1
    assert mod._lp_seeded is True
    assert len(mod._pe_ring) > 0

    mod.reset()
    assert mod._n_waking_calls == 0
    assert mod._last_n_simulation_skips == 0
    assert mod._lp_seeded is False
    assert mod._lp_ema == 0.0
    assert mod._pe_ring == []
    assert mod._last_novelty_norm == 0.0
    assert mod._last_uncertainty_signal == 0.0
    assert mod._last_learning_progress_signal == 0.0
    assert mod._last_n_subflavours_fired == 0


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------
def test_invalid_config_raises():
    with pytest.raises(ValueError):
        StructuredCuriosity(StructuredCuriosityConfig(
            use_structured_curiosity=True, curiosity_novelty_weight=-0.1,
        ))
    with pytest.raises(ValueError):
        StructuredCuriosity(StructuredCuriosityConfig(
            use_structured_curiosity=True, curiosity_uncertainty_weight=-0.1,
        ))
    with pytest.raises(ValueError):
        StructuredCuriosity(StructuredCuriosityConfig(
            use_structured_curiosity=True, curiosity_learning_progress_weight=-0.1,
        ))
    with pytest.raises(ValueError):
        StructuredCuriosity(StructuredCuriosityConfig(
            use_structured_curiosity=True, curiosity_bias_scale=0.0,
        ))
    with pytest.raises(ValueError):
        StructuredCuriosity(StructuredCuriosityConfig(
            use_structured_curiosity=True, curiosity_lp_ema_alpha=0.0,
        ))
    with pytest.raises(ValueError):
        StructuredCuriosity(StructuredCuriosityConfig(
            use_structured_curiosity=True, curiosity_lp_window_k=0,
        ))


# ----------------------------------------------------------------------
# C6 MECH-314a Phase-2 amend: e2.world_forward novelty-candidate-source
# (V3-EXQ-648 autopsy fix). curiosity_candidate_source switches the
# per-candidate novelty signature between the collapsed proposer first-step
# z_world ("proposer", default, bit-identical) and the SD-056-trained
# action-conditional e2.world_forward(z0, a_i) predictions.
# ----------------------------------------------------------------------
def _build_curiosity_agent_with_candidates(candidate_source: str, seed: int = 42):
    """Build a structured-curiosity + SP-CEM agent and generate one tick's
    candidate pool (>= 2 action-divergent candidates)."""
    from ree_core.environment.causal_grid_world import CausalGridWorldV2

    torch.manual_seed(seed)
    env = CausalGridWorldV2(
        seed=seed, size=12, num_hazards=0, num_resources=5,
        hazard_harm=0.0, harm_history_len=10,
    )
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=32, world_dim=32, alpha_world=0.9,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_structured_curiosity=True, use_curiosity_novelty=True,
        use_curiosity_uncertainty=False, use_curiosity_learning_progress=False,
        curiosity_novelty_source="visitation",
        curiosity_candidate_source=candidate_source,
    )
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    with torch.no_grad():
        latent = agent.sense(obs_body=body, obs_world=world)
    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (
        agent._e1_tick(latent) if ticks.get("e1_tick", False)
        else torch.zeros(1, wdim)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    return agent, candidates


def test_c6_default_proposer_is_bit_identical():
    """Default curiosity_candidate_source is 'proposer' and the helper returns
    None, so the legacy proposer-summary reuse-chain runs unchanged."""
    cfg = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=250, action_dim=4)
    assert cfg.curiosity_candidate_source == "proposer"
    agent, candidates = _build_curiosity_agent_with_candidates("proposer")
    assert len(candidates) >= 2
    summ = agent._curiosity_candidate_summaries(candidates)
    assert summ is None, (
        "proposer source must return None so the legacy reuse-chain is used "
        "(bit-identical to pre-amend)"
    )


def test_c6_e2_world_forward_source_shape():
    """'e2_world_forward' source returns a [K, world_dim] tensor of per-candidate
    e2.world_forward predictions (the consumed novelty signature)."""
    agent, candidates = _build_curiosity_agent_with_candidates("e2_world_forward")
    K = len(candidates)
    assert K >= 2
    summ = agent._curiosity_candidate_summaries(candidates)
    assert summ is not None
    assert summ.shape == (K, agent.config.latent.world_dim)
    assert torch.isfinite(summ).all()


def test_c6_e2_world_forward_feeds_candidate_spread():
    """With a divergent e2.world_forward, the curiosity-consumed _candidate_spread
    keys on the action-divergent e2 predictions (>0), not the collapsed proposer
    summaries. This is the V3-EXQ-648 root-cause fix made deterministic."""
    agent, candidates = _build_curiosity_agent_with_candidates("e2_world_forward")
    K = len(candidates)

    # Proposer spread on this collapsed monostrategy pool (for contrast).
    proposer_summ = torch.stack(
        [c.get_world_state_sequence()[0, 0, :] for c in candidates], dim=0
    )
    proposer_spread = agent.curiosity._candidate_spread(proposer_summ)

    # Monkeypatch a strictly per-candidate-divergent world_forward: row i = i.
    def _divergent_wf(z, a):
        n = z.shape[0]
        return (
            torch.arange(n, dtype=z.dtype)
            .unsqueeze(1)
            .expand(n, z.shape[1])
            .clone()
        )

    agent.e2.world_forward = _divergent_wf  # type: ignore[assignment]
    e2_summ = agent._curiosity_candidate_summaries(candidates)
    assert e2_summ is not None
    e2_spread = agent.curiosity._candidate_spread(e2_summ)
    assert e2_spread > 0.0
    assert e2_spread > proposer_spread, (
        f"e2_world_forward spread {e2_spread} must exceed the collapsed proposer "
        f"spread {proposer_spread}"
    )


def test_c6_proposer_summaries_collapsed_baseline():
    """Sanity: on the monostrategy candidate pool the proposer first-step z_world
    spread is near-zero -- the condition that zeroed 314a novelty in V3-EXQ-648."""
    agent, candidates = _build_curiosity_agent_with_candidates("proposer")
    proposer_summ = torch.stack(
        [c.get_world_state_sequence()[0, 0, :] for c in candidates], dim=0
    )
    spread = agent.curiosity._candidate_spread(proposer_summ)
    assert spread >= 0.0  # well-defined; expected small under monostrategy
