"""SD-082 AMEND (failure_autopsy_V3-EXQ-822c_2026-08-29): candidate_summary_source
"proposer_post_action" + the LateralPFCAnalog centering-degeneracy guard.

This is the missing regression coverage for the substrate landed on
integration/sd082-percandidate-summary (ree-v3 3aa45dea40). The landing commit's
own CLAUDE.md entry describes "28 new contracts in
tests/contracts/test_sd082_candidate_summary_post_action_amend.py" but no such
file existed in the commit -- this file supplies that coverage before the branch
is merged to main, per the umbrella verify-and-land chip.

Root cause under test (V3-EXQ-822c): trajectory.world_states[:, 0, :] is the
rollout's SHARED initial z_world seed (E2FastPredictor.rollout_with_world seeds
world_states=[initial_z_world]), bit-identical across every candidate by
construction. SD-082's compute_bias() centering step (subtract the cross-candidate
mean) annihilates that constant to float32 cancellation noise. The fix:
candidate_summary_source="proposer_post_action" reads the POST-ACTION world_states
(t>=1, mean over horizon) instead, which differ per candidate's own action
sequence; and a diagnostic-only degeneracy guard on LateralPFCAnalog.compute_bias
flags (never alters) the pre-fix degenerate case.
"""

from __future__ import annotations

from typing import List

import torch

from ree_core.agent import REEAgent
from ree_core.pfc.lateral_pfc_analog import LateralPFCAnalog, LateralPFCConfig
from ree_core.predictors.e2_fast import Trajectory
from ree_core.utils.config import REEConfig


RULE_DIM = 16
WORLD_DIM = 32
K = 6
BIAS_SCALE = 0.1


# ---------------------------------------------------------------------------
# Helpers: synthetic Trajectory fixtures (no full agent needed for the
# _proposer_post_action_summaries / compute_bias unit-level tests).
# ---------------------------------------------------------------------------


def _make_trajectory(
    world_states_list: List[torch.Tensor], action_dim: int = 4, horizon: int = 3,
    self_dim: int = WORLD_DIM,
) -> Trajectory:
    """Trajectory with the given world_states (a LIST of [1, world_dim] tensors,
    one per timestep -- matches Trajectory.get_world_state_sequence()'s
    torch.stack(self.world_states, dim=1) contract). states (z_self) is a
    required field on Trajectory; filled with a matching-length placeholder
    list since these tests do not exercise the z_self path."""
    actions = torch.zeros(1, horizon, action_dim)
    states = [torch.zeros(1, self_dim) for _ in range(len(world_states_list))]
    return Trajectory(
        actions=actions,
        world_states=world_states_list,
        states=states,
        action_objects=None,
    )


def _collapsed_proposer_candidates(
    n: int = K, horizon: int = 3, world_dim: int = WORLD_DIM, seed: int = 0
) -> List[Trajectory]:
    """K candidates sharing an identical t=0 seed (the collapse SD-082 fixes)
    but genuinely differing in their post-action (t>=1) states -- the realistic
    rollout shape: shared initial z_world, action-divergent continuation."""
    g = torch.Generator().manual_seed(seed)
    shared_seed = torch.randn(1, world_dim, generator=g)
    cands = []
    for i in range(n):
        # Each candidate's post-action states differ (action-divergent rollout).
        post = [
            torch.randn(1, world_dim, generator=g) + float(i) * 0.5
            for _ in range(horizon)
        ]
        ws_list = [shared_seed.clone()] + post  # length horizon+1
        cands.append(_make_trajectory(ws_list, horizon=horizon))
    return cands


def _zero_horizon_candidates(n: int = K, world_dim: int = WORLD_DIM, seed: int = 0) -> List[Trajectory]:
    """K candidates with a degenerate zero-horizon rollout (T=1, only the seed
    state, no post-action state at all)."""
    g = torch.Generator().manual_seed(seed)
    cands = []
    for i in range(n):
        ws = torch.randn(1, world_dim, generator=g) + float(i) * 0.3
        cands.append(_make_trajectory([ws], horizon=0))
    return cands


def _no_world_states_candidates(n: int = K, action_dim: int = 4, horizon: int = 3) -> List[Trajectory]:
    """K candidates whose world_states is None (the None-fallback branch)."""
    cands = []
    for _ in range(n):
        actions = torch.zeros(1, horizon, action_dim)
        states = [torch.zeros(1, WORLD_DIM) for _ in range(horizon + 1)]
        cands.append(
            Trajectory(actions=actions, world_states=None, states=states, action_objects=None)
        )
    return cands


def _bare_agent(candidate_summary_source: str = "proposer") -> REEAgent:
    """Minimal REEAgent with no candidate-summary-consuming bias channels on --
    just enough to exercise _candidate_world_summaries / _proposer_post_action_summaries
    in isolation."""
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=WORLD_DIM, action_dim=4,
        self_dim=WORLD_DIM, world_dim=WORLD_DIM,
        candidate_summary_source=candidate_summary_source,
    )
    agent = REEAgent(cfg)
    agent.reset()
    return agent


# ---------------------------------------------------------------------------
# A. candidate_summary_source config reachability
# ---------------------------------------------------------------------------


def test_a1_default_source_is_proposer():
    cfg = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=WORLD_DIM, action_dim=4)
    assert cfg.candidate_summary_source == "proposer"


def test_a2_proposer_post_action_reachable_through_from_dims():
    cfg = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=WORLD_DIM, action_dim=4,
        candidate_summary_source="proposer_post_action",
    )
    assert cfg.candidate_summary_source == "proposer_post_action"


def test_a3_absent_default_is_bit_identical_to_explicit_proposer():
    cfg_default = REEConfig.from_dims(body_obs_dim=10, world_obs_dim=WORLD_DIM, action_dim=4)
    cfg_explicit = REEConfig.from_dims(
        body_obs_dim=10, world_obs_dim=WORLD_DIM, action_dim=4,
        candidate_summary_source="proposer",
    )
    assert cfg_default.candidate_summary_source == cfg_explicit.candidate_summary_source == "proposer"


# ---------------------------------------------------------------------------
# B. _proposer_post_action_summaries direct unit tests
# ---------------------------------------------------------------------------


def test_b1_returns_none_on_empty_candidates():
    agent = _bare_agent("proposer_post_action")
    assert agent._proposer_post_action_summaries([]) is None


def test_b2_shape_and_finite():
    agent = _bare_agent("proposer_post_action")
    cands = _collapsed_proposer_candidates()
    summ = agent._proposer_post_action_summaries(cands)
    assert summ is not None
    assert summ.shape == (K, WORLD_DIM)
    assert torch.isfinite(summ).all()


def test_b3_post_action_summary_is_candidate_discriminating():
    """The core fix: unlike world_states[0] (shared seed, uniform across
    candidates), the post-action mean genuinely differs per candidate."""
    agent = _bare_agent("proposer_post_action")
    cands = _collapsed_proposer_candidates()

    # Sanity: the t=0 seed really is bit-identical across candidates (the bug
    # this fix addresses -- confirms the fixture reproduces it).
    seed_states = torch.stack([c.get_world_state_sequence()[0, 0, :] for c in cands], dim=0)
    seed_range = float((seed_states.max(dim=0).values - seed_states.min(dim=0).values).abs().max())
    assert seed_range == 0.0, "fixture must reproduce the shared-seed collapse"

    summ = agent._proposer_post_action_summaries(cands)
    assert summ is not None
    # Post-action summaries must NOT collapse to a single point.
    pairwise_range = float((summ.max(dim=0).values - summ.min(dim=0).values).abs().max())
    assert pairwise_range > 0.0, "post-action summaries must discriminate candidates"


def test_b4_post_action_summary_matches_manual_mean():
    agent = _bare_agent("proposer_post_action")
    cands = _collapsed_proposer_candidates()
    summ = agent._proposer_post_action_summaries(cands)
    for i, c in enumerate(cands):
        ws = c.get_world_state_sequence()
        expected = ws[0, 1:, :].mean(dim=0)
        assert torch.allclose(summ[i], expected, atol=1e-6)


def test_b5_zero_horizon_fallback_uses_seed_state():
    """Degenerate zero-horizon rollout (T=1): falls back to world_states[0,0,:],
    identical to the 'proposer' default in that edge case."""
    agent = _bare_agent("proposer_post_action")
    cands = _zero_horizon_candidates()
    summ = agent._proposer_post_action_summaries(cands)
    assert summ is not None
    for i, c in enumerate(cands):
        expected = c.get_world_state_sequence()[0, 0, :]
        assert torch.allclose(summ[i], expected, atol=1e-6)


def test_b6_none_world_states_falls_back_to_current_latent():
    """When a candidate lacks world_states entirely, falls back to
    self._current_latent.z_world (mirrors the e2_world_forward branch's
    None-fallback contract). REEAgent.reset() seeds _current_latent to a
    zero-valued LatentState (via latent_stack.init_state), so it is never
    actually None post-reset -- the helper's own 'or return None' branch is
    reachable only pre-construction/pre-reset, which this test does not
    exercise directly (see test_b1 for the true-None empty-candidates case)."""
    agent = _bare_agent("proposer_post_action")
    cands = _no_world_states_candidates()
    assert agent._current_latent is not None
    summ = agent._proposer_post_action_summaries(cands)
    assert summ is not None
    assert summ.shape == (K, WORLD_DIM)
    expected_row = agent._current_latent.z_world[0].detach()
    for i in range(K):
        assert torch.allclose(summ[i], expected_row, atol=1e-6)


# ---------------------------------------------------------------------------
# C. _candidate_world_summaries dispatch
# ---------------------------------------------------------------------------


def test_c1_dispatch_routes_proposer_post_action_to_helper():
    agent = _bare_agent("proposer_post_action")
    cands = _collapsed_proposer_candidates()
    dispatched = agent._candidate_world_summaries(cands)
    direct = agent._proposer_post_action_summaries(cands)
    assert dispatched is not None
    assert torch.allclose(dispatched, direct, atol=1e-6)


def test_c2_dispatch_default_proposer_returns_none():
    """Backward-compat: default 'proposer' source is untouched by this amend --
    _candidate_world_summaries still returns None so callers take the legacy
    manual ws[0, 0, :] fallback path."""
    agent = _bare_agent("proposer")
    cands = _collapsed_proposer_candidates()
    assert agent._candidate_world_summaries(cands) is None


def test_c3_dispatch_e2_world_forward_unaffected():
    """The pre-existing e2_world_forward branch is untouched -- dispatch order
    checks proposer_post_action first, then falls through to the existing
    e2_world_forward / None logic exactly as before."""
    agent = _bare_agent("e2_world_forward")
    cands = _collapsed_proposer_candidates()
    # e2 and _current_latent are both present post-construction/reset (e2 is
    # always built by REEAgent.__init__; _current_latent is seeded to a
    # zero-valued LatentState by reset()), so the e2_world_forward branch runs
    # its real path and returns a [K, world_dim] tensor -- exactly the
    # pre-amend behaviour, untouched by the proposer_post_action dispatch
    # added ahead of it.
    assert agent._current_latent is not None
    summ = agent._candidate_world_summaries(cands)
    assert summ is not None
    assert summ.shape == (K, WORLD_DIM)


# ---------------------------------------------------------------------------
# D. LateralPFCAnalog centering-degeneracy guard
# ---------------------------------------------------------------------------


def _make_head(consumer: bool = True, degeneracy_floor: float = 1e-4) -> LateralPFCAnalog:
    # train_rule_bias_head=True (matches test_sd082_rule_readout_consumer.py's
    # own _make_head): with the default False, rule_bias_head's last Linear is
    # zeroed at init so compute_bias always returns exactly zero regardless of
    # input -- correct for the OFF-by-default landing guarantee, but it makes
    # the bias-dependence tests below (D3/E1/E2/E3) vacuous, since a bias that
    # is always zero trivially "propagates" nothing either way.
    torch.manual_seed(7)
    cfg = LateralPFCConfig(
        use_lateral_pfc_analog=True,
        rule_dim=RULE_DIM,
        hidden_dim=32,
        bias_scale=BIAS_SCALE,
        train_rule_bias_head=True,
        rule_readout_consumer=consumer,
        readout_init_scale=0.25,
        candidate_summary_degeneracy_floor=degeneracy_floor,
    )
    return LateralPFCAnalog(delta_dim=8, world_dim=WORLD_DIM, config=cfg)


def _collapsed_summaries(seed: int = 0) -> torch.Tensor:
    """K candidate summaries that are (near-)constant across candidates --
    exactly what world_states[:, 0, :] produces pre-fix (V3-EXQ-822c)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, WORLD_DIM, generator=g) * 3.0
    return base.expand(K, -1).clone()


def _differentiated_summaries(seed: int = 0) -> torch.Tensor:
    """K genuinely differentiated candidate summaries -- what the
    proposer_post_action fix produces."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K, WORLD_DIM, generator=g) * 3.0


def test_d1_guard_fires_on_degenerate_summaries():
    lp = _make_head(consumer=True)
    lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 0.5)
    with torch.no_grad():
        lp.compute_bias(_collapsed_summaries())
    state = lp.get_state()
    assert state["candidate_summary_degenerate"] is True
    # Post-centering norm must be far below the pre-centering norm (float32
    # cancellation-noise regime), per the failure record's own measurement.
    assert state["candidate_summary_norm_post_centering"] <= (
        1e-4 * state["candidate_summary_norm_pre_centering"]
    )


def test_d2_guard_does_not_fire_on_differentiated_summaries():
    lp = _make_head(consumer=True)
    lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 0.5)
    with torch.no_grad():
        lp.compute_bias(_differentiated_summaries())
    state = lp.get_state()
    assert state["candidate_summary_degenerate"] is False
    # A genuinely differentiated summary must retain substantial norm after
    # centering (nowhere near the degeneracy floor).
    assert state["candidate_summary_norm_post_centering"] > (
        1e-2 * state["candidate_summary_norm_pre_centering"]
    )


def test_d3_guard_never_changes_returned_bias():
    """Diagnostic-only guarantee: whether the guard fires or not, compute_bias's
    returned value is identical to a build with the flag artificially disabled
    at the call site (i.e. the guard block is side-effect-free on `bias`)."""
    lp_a = _make_head(consumer=True)
    lp_a.rule_state.copy_(torch.zeros(1, RULE_DIM))
    lp_b = _make_head(consumer=True)
    lp_b.rule_state.copy_(torch.zeros(1, RULE_DIM))
    # Same weights (same seed=7 in _make_head), same rule_state -> same input.
    summ = _collapsed_summaries()
    with torch.no_grad():
        bias_a = lp_a.compute_bias(summ).clone()
    with torch.no_grad():
        bias_b = lp_b.compute_bias(summ).clone()
    assert torch.allclose(bias_a, bias_b)
    # Degenerate flag fired but bias values are still finite and bounded.
    assert lp_a.get_state()["candidate_summary_degenerate"] is True
    assert torch.isfinite(bias_a).all()
    assert float(bias_a.abs().max()) <= BIAS_SCALE + 1e-6


def test_d4_guard_inactive_when_consumer_off():
    """rule_readout_consumer=False: no centering happens at all (legacy hard
    clamp on raw summary), so the guard fields stay at their init defaults."""
    lp = _make_head(consumer=False)
    lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 0.5)
    with torch.no_grad():
        lp.compute_bias(_collapsed_summaries())
    state = lp.get_state()
    assert state["candidate_summary_degenerate"] is False
    assert state["candidate_summary_norm_pre_centering"] == 0.0
    assert state["candidate_summary_norm_post_centering"] == 0.0


def test_d5_guard_inactive_with_single_candidate():
    """k < 2: no cross-candidate mean to subtract, so centering (and thus the
    guard) never engages -- raw summary passed through untouched."""
    lp = _make_head(consumer=True)
    lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 0.5)
    single = torch.randn(1, WORLD_DIM) * 3.0
    with torch.no_grad():
        lp.compute_bias(single)
    state = lp.get_state()
    assert state["candidate_summary_degenerate"] is False
    assert state["candidate_summary_norm_pre_centering"] == 0.0


def test_d6_reset_clears_degeneracy_guard_fields():
    lp = _make_head(consumer=True)
    lp.rule_state.copy_(torch.randn(1, RULE_DIM) * 0.5)
    with torch.no_grad():
        lp.compute_bias(_collapsed_summaries())
    assert lp.get_state()["candidate_summary_degenerate"] is True
    lp.reset()
    state = lp.get_state()
    assert state["candidate_summary_degenerate"] is False
    assert state["candidate_summary_norm_pre_centering"] == 0.0
    assert state["candidate_summary_norm_post_centering"] == 0.0


def test_d7_degeneracy_floor_is_configurable():
    """A lenient floor (e.g. 10.0, guaranteed to exceed any realistic
    post/pre ratio) always flags degenerate; the strict default does not
    on genuinely differentiated input -- confirms the threshold is read from
    config, not hardcoded."""
    lp_lenient = _make_head(consumer=True, degeneracy_floor=10.0)
    lp_lenient.rule_state.copy_(torch.zeros(1, RULE_DIM))
    with torch.no_grad():
        lp_lenient.compute_bias(_differentiated_summaries())
    assert lp_lenient.get_state()["candidate_summary_degenerate"] is True

    lp_strict = _make_head(consumer=True, degeneracy_floor=1e-4)
    lp_strict.rule_state.copy_(torch.zeros(1, RULE_DIM))
    with torch.no_grad():
        lp_strict.compute_bias(_differentiated_summaries())
    assert lp_strict.get_state()["candidate_summary_degenerate"] is False


# ---------------------------------------------------------------------------
# E. End-to-end: proposer_post_action feeding LateralPFCAnalog resolves the
#    V3-EXQ-822c "carries ZERO candidate-discriminating information" defect.
#
# The V3-EXQ-822c failure signature is NOT that the bias is uniformly zero
# (a rule_state ablation, |bias(rule_state) - bias(0)|, was already measured
# NONZERO and clearing the readiness floor pre-fix -- 0.001662, per the
# failure record). The corrupting defect is that this apparently-healthy
# propagation number carries ZERO CANDIDATE-DISCRIMINATING information: with
# collapsed (constant-across-candidates) summaries, centering drives the
# world-summary component of every candidate's head input to (near-)zero, so
# every candidate receives the SAME bias value (driven only by rule_state,
# which is candidate-invariant) -- an authentic-looking but meaningless
# number. The correct DV is therefore the CROSS-CANDIDATE BIAS RANGE
# (max - min over K), not a rule_state on/off delta.
# ---------------------------------------------------------------------------


def _bias_range(lp: LateralPFCAnalog, summ: torch.Tensor, rule_seed: int = 0) -> float:
    g = torch.Generator().manual_seed(rule_seed)
    lp.rule_state.copy_(torch.randn(1, RULE_DIM, generator=g) * 1.0)
    with torch.no_grad():
        bias = lp.compute_bias(summ).clone()
    return float((bias.max() - bias.min()).item())


def test_e1_collapsed_summary_reproduces_the_822c_zero_discrimination():
    """Sanity: the pre-fix input shape (collapsed proposer summary) still
    produces a near-zero cross-candidate bias range on the consumer-enabled
    head -- confirms this test file's fixtures reproduce the failure record's
    'carries ZERO candidate-discriminating information' signature, and that
    the degeneracy guard (Section D) correctly flags the same input.

    NOT an exact-equality check: `x - mean(x, x, ..., x)` for K bit-identical
    rows is float32 CANCELLATION NOISE (the failure record's own term), not a
    mathematical zero -- summing K identical values then dividing is not
    guaranteed to round back to exactly the original value on every BLAS/CPU,
    so the post-centering residual (and thus the bias range after passing
    through the head) can be a tiny nonzero number that differs by platform.
    An exact `== 0.0` assertion here previously passed locally (Mac,
    darwin-arm64) and failed on a Linux cloud worker for exactly this reason.
    The threshold below (1e-3, the same 'non-vacuity floor' scale used
    elsewhere in this failure lineage for prop_delta) is the meaningful
    contrast: near-zero vs. the ~0.17 range test_e2 measures for genuinely
    differentiated input, three orders of magnitude apart."""
    lp = _make_head(consumer=True)
    rng = _bias_range(lp, _collapsed_summaries())
    assert rng < 1e-3, (
        f"collapsed (constant) summaries must produce a near-zero "
        f"cross-candidate bias range (float32 cancellation noise, not a "
        f"real signal) -- got {rng}"
    )
    assert lp.get_state()["candidate_summary_degenerate"] is True


def test_e2_proposer_post_action_summary_restores_discrimination():
    """The fix: feeding a genuinely differentiated (proposer_post_action-style)
    summary into the SAME consumer-enabled head produces a nonzero
    cross-candidate bias range -- clears V3-EXQ-822c's zero-discrimination
    failure signature."""
    lp = _make_head(consumer=True)
    rng = _bias_range(lp, _differentiated_summaries())
    assert rng > 1e-4, (
        f"differentiated summaries must produce a non-trivial "
        f"cross-candidate bias range once the degeneracy is resolved -- "
        f"got {rng}"
    )
    assert lp.get_state()["candidate_summary_degenerate"] is False


def test_e3_agent_level_proposer_post_action_summary_discriminates():
    """Wires _proposer_post_action_summaries' actual output (not a synthetic
    fixture) through LateralPFCAnalog.compute_bias and confirms both (a) the
    degeneracy guard does not flag it, and (b) the resulting bias genuinely
    discriminates across candidates -- the full fix, end to end at the unit
    level."""
    agent = _bare_agent("proposer_post_action")
    cands = _collapsed_proposer_candidates()
    summ = agent._proposer_post_action_summaries(cands)
    assert summ is not None

    lp = _make_head(consumer=True)
    rng = _bias_range(lp, summ)
    assert lp.get_state()["candidate_summary_degenerate"] is False
    assert rng > 0.0, "proposer_post_action summaries must discriminate candidates"
