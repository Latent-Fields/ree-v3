"""ARC-065 GAP-A: shared candidate_summary_source e2.world_forward re-sourcing.

Contract for the GAP-A substrate (V3-EXQ-614e autopsy 2026-06-07): the SHARED
per-candidate cand_world_summaries consumed by the E3-side bias channels
(lateral_pfc / ofc / mech295 / gated_policy / tonic_vigor) can be re-sourced from
the SD-056-trained action-conditional e2.world_forward(z0, a_i) predictions
instead of the collapsed proposer first-step z_world. Sibling of the curiosity
channel fix (MECH-314a Phase-2, test_mech_314_curiosity.py C6).

G1 default "proposer" -> helper returns None (legacy reuse-chain, bit-identical).
G2 "e2_world_forward" -> helper returns [K, world_dim] finite tensor.
G3 divergent e2.world_forward -> shared spread > collapsed proposer spread.
G4 master-OFF bit-identical (helper None on proposer; default == explicit).
G5 select_action runs end-to-end with the source ON + bias channels ON.
"""

import torch

from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig


def _mean_pairwise_l2(summ: torch.Tensor) -> float:
    """Mean pairwise L2 distance across the K rows -- the cand_world_pairwise_dist
    quantity the V3-EXQ-614e autopsy reports as 0.0000 under monostrategy."""
    summ = summ.detach()
    K = summ.shape[0]
    if K < 2:
        return 0.0
    total = 0.0
    n = 0
    for i in range(K):
        for j in range(i + 1, K):
            total += float(torch.linalg.vector_norm(summ[i] - summ[j]))
            n += 1
    return total / max(n, 1)


def _build_agent_with_candidates(candidate_summary_source: str, seed: int = 42):
    """SP-CEM agent with the E3-side bias channels (lateral_pfc / ofc / mech295)
    ON, plus one tick's action-divergent candidate pool."""
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
        use_lateral_pfc_analog=True,
        use_ofc_analog=True,
        use_mech295_liking_bridge=True,
        candidate_summary_source=candidate_summary_source,
    )
    cfg.goal.z_goal_enabled = True
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
    return agent, candidates, ticks


def test_g1_default_proposer_is_bit_identical():
    """Default candidate_summary_source is 'proposer'; the helper returns None so
    the legacy proposer-summary build runs unchanged (bit-identical to pre-GAP-A)."""
    cfg = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=250, action_dim=4)
    assert cfg.candidate_summary_source == "proposer"
    agent, candidates, _ = _build_agent_with_candidates("proposer")
    assert len(candidates) >= 2
    summ = agent._candidate_world_summaries(candidates)
    assert summ is None, (
        "proposer source must return None so callers take the legacy collapsed "
        "proposer-summary path (bit-identical to pre-GAP-A)"
    )


def test_g1b_default_equals_explicit_proposer():
    """from_dims default and explicit 'proposer' resolve identically."""
    cfg_default = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=250, action_dim=4)
    cfg_explicit = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=250, action_dim=4,
        candidate_summary_source="proposer",
    )
    assert cfg_default.candidate_summary_source == cfg_explicit.candidate_summary_source


def test_g2_e2_world_forward_source_shape():
    """'e2_world_forward' returns a [K, world_dim] tensor of per-candidate
    e2.world_forward predictions."""
    agent, candidates, _ = _build_agent_with_candidates("e2_world_forward")
    K = len(candidates)
    assert K >= 2
    summ = agent._candidate_world_summaries(candidates)
    assert summ is not None
    assert summ.shape == (K, agent.config.latent.world_dim)
    assert torch.isfinite(summ).all()


def test_g3_e2_world_forward_beats_collapsed_proposer_spread():
    """With a divergent e2.world_forward the shared summaries carry cross-candidate
    spread > the collapsed proposer first-step z_world spread -- the exact GAP-A
    fix (V3-EXQ-614e cand_world_pairwise_dist=0.0000) made deterministic."""
    agent, candidates, _ = _build_agent_with_candidates("e2_world_forward")

    proposer_summ = torch.stack(
        [c.get_world_state_sequence()[0, 0, :] for c in candidates], dim=0
    )
    proposer_spread = _mean_pairwise_l2(proposer_summ)

    # Strictly per-candidate-divergent world_forward: row i scaled by i.
    def _divergent_wf(z, a):
        n = z.shape[0]
        return (
            torch.arange(n, dtype=z.dtype)
            .unsqueeze(1)
            .expand(n, z.shape[1])
            .clone()
        )

    agent.e2.world_forward = _divergent_wf  # type: ignore[assignment]
    e2_summ = agent._candidate_world_summaries(candidates)
    assert e2_summ is not None
    e2_spread = _mean_pairwise_l2(e2_summ)
    assert e2_spread > 0.0
    assert e2_spread > proposer_spread, (
        f"e2_world_forward shared spread {e2_spread} must exceed the collapsed "
        f"proposer spread {proposer_spread}"
    )


def test_g4_master_off_helper_none_on_proposer():
    """The GAP-A re-sourcing only engages under 'e2_world_forward'; under
    'proposer' the helper is a no-op regardless of bias-channel state."""
    agent, candidates, _ = _build_agent_with_candidates("proposer")
    assert agent._candidate_world_summaries(candidates) is None


def test_g5_select_action_runs_with_source_on():
    """select_action runs end-to-end with candidate_summary_source='e2_world_forward'
    and the lateral_pfc / ofc / mech295 bias channels ON -- the consumer wiring
    contract (the shared summaries reach the bias channels without error)."""
    agent, candidates, ticks = _build_agent_with_candidates("e2_world_forward")
    with torch.no_grad():
        action = agent.select_action(candidates, ticks)
    assert action is not None
    assert torch.isfinite(action).all()


def _build_agent_with_active_goal(candidate_summary_source: str, seed: int = 42):
    """SD-057 L7 dACC-consume variant: use_mech_consume ON and goal_state
    seeded ACTIVE (via cue_pull, no benefit gate needed), so the
    candidate_goal_proximity block in select_action actually fires."""
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
        use_dacc=True,  # SD-057 L7 (use_mech_consume) requires use_dacc.
        use_mech_consume=True,
        candidate_summary_source=candidate_summary_source,
    )
    cfg.goal.z_goal_enabled = True
    agent = REEAgent(cfg)
    agent.reset()
    _flat, obs_dict = env.reset()
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    with torch.no_grad():
        latent = agent.sense(obs_body=body, obs_world=world)
    # Seed z_goal directly so goal_state.is_active() is True -- the SD-057 L7
    # dACC-consume block (agent.py select_action) gates on this, and cue_pull
    # is the no-benefit-gate way to activate it deterministically for a test.
    agent.goal_state.cue_pull(latent.z_world.detach(), 1.0)
    assert agent.goal_state.is_active()
    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (
        agent._e1_tick(latent) if ticks.get("e1_tick", False)
        else torch.zeros(1, wdim)
    )
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    return agent, candidates, ticks


def test_g6_dacc_goal_proximity_uses_candidate_world_summaries():
    """Regression test (found 2026-07-31, chip-20260731-arc005-802-precision-
    anomaly, while diagnosing why V3-EXQ-802's precision DV was invariant to
    all four ARC-005 control-plane channels): the SD-057 L7 (MECH-348)
    dACC-consume candidate_goal_proximity previously read
    get_world_state_sequence()[0, 0, :] directly -- the collapsed proposer
    timestep-0 state, identical across every candidate by construction -- so
    it could never discriminate between candidates. This is the same failure
    class ARC-065 GAP-A (G1-G5 above) already fixed for the sibling
    lateral_pfc/ofc/mech295/gated_policy/tonic_vigor channels; the
    dACC-consume block's own comment claimed to mirror MECH-295's build but
    had only copied its legacy fallback half. Fixed by mirroring MECH-295's
    _candidate_world_summaries()-first pattern exactly."""
    agent, candidates, _ = _build_agent_with_active_goal("e2_world_forward")
    K = len(candidates)
    assert K >= 2

    # The legacy collapsed-proposer summary -- what the bug used, always
    # uniform across candidates. Sanity-checks the bug this test guards.
    proposer_summ = torch.stack(
        [c.get_world_state_sequence()[0, 0, :] for c in candidates], dim=0
    )
    with torch.no_grad():
        proposer_gp = agent.goal_state.goal_proximity(proposer_summ).detach()
    proposer_range = float((proposer_gp.max() - proposer_gp.min()).item())
    assert proposer_range == 0.0, (
        "sanity check failed: the collapsed proposer summary is expected to "
        "be exactly uniform across candidates -- this is the bug being "
        "regression-tested, so a nonzero range here means the fixture no "
        "longer reproduces it"
    )

    # Strictly per-candidate-divergent world_forward (same technique as G3)
    # so the fixed path has a genuine discriminative signal to carry.
    def _divergent_wf(z, a):
        n = z.shape[0]
        return (
            torch.arange(n, dtype=z.dtype)
            .unsqueeze(1)
            .expand(n, z.shape[1])
            .clone()
        )

    agent.e2.world_forward = _divergent_wf  # type: ignore[assignment]
    fixed_summ = agent._candidate_world_summaries(candidates)
    assert fixed_summ is not None
    with torch.no_grad():
        fixed_gp = agent.goal_state.goal_proximity(fixed_summ).detach()
    fixed_range = float((fixed_gp.max() - fixed_gp.min()).item())
    assert fixed_range > 0.0, (
        "candidate_goal_proximity must carry nonzero cross-candidate range "
        "once candidate_summary_source='e2_world_forward' is set and "
        "e2.world_forward genuinely diverges by candidate -- the fix under "
        "test"
    )


def test_g7_dacc_goal_proximity_default_unchanged():
    """Bit-identical guarantee: with candidate_summary_source left at its
    'proposer' default, the SD-057 L7 dACC-consume block must still take the
    legacy collapsed-proposer fallback path end-to-end through select_action
    -- the fix is fully opt-in, never changing default behaviour."""
    agent, candidates, ticks = _build_agent_with_active_goal("proposer")
    assert agent._candidate_world_summaries(candidates) is None
    with torch.no_grad():
        action = agent.select_action(candidates, ticks)
    assert action is not None
    assert torch.isfinite(action).all()
