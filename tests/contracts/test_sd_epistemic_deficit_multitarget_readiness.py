"""sd_epistemic_deficit_multitarget_readiness -- contracts.

Substrate-readiness contracts for the multi-target regime that unblocks
MECH-482's ORNT-2 validation. Source: the ratified V3-EXQ-964 autopsy
(REE_assembly evidence/planning/failure_autopsy_V3-EXQ-964_2026-08-30.json),
whose C2 ("the downstream consumer can change the committed action") was
STRUCTURALLY UNSATISFIABLE because the accumulator held n_targets == 1 on all
three seeds with all 32 candidates matching that one target, so
EpistemicDeficitAccumulator.readout returned a CONSTANT vector and a constant
cannot move an argmax.

Every contract here is written so it FAILS against the pre-readiness substrate
and PASSES after it -- the OFF arm of each pair reproduces the 964 collapse
signature deliberately, as the negative control. Roughly half the file is
those negative controls: a readiness knob that cannot be shown to change the
collapse is not evidence of anything.

The four causes the contracts pin, in the order they bind:

  R1  match_radius's 1.0 default exceeds the ENTIRE reachable z_world manifold
      (measured max pairwise L2 0.41 on the 964 config), so every location
      falls inside one target at ANY episode length. -> match_radius_mode.
  R2  a matched target RE-CENTERS onto the latest z, making it a random walk
      that absorbs the manifold even under a correct radius. -> center_update.
  R3  UPDATE keys targets on the ENCODER's z_world while READOUT matches
      e2.world_forward PREDICTIONS -- two clouds separated by a systematic
      centroid offset ~5.6x their own internal spread, so no single radius can
      both separate targets and match candidates. -> target_frame.
  R4  a HARD nearest-match threshold is a STEP function of candidate position,
      so it saturates in BOTH directions (all-match -> one constant; all-miss
      -> zero, also constant). -> readout_mode.

R3 and R4 are NOT in the autopsy -- they were found by measuring this
substrate while building the fix, and each is independently sufficient to keep
the readout constant, so a build addressing only the autopsy's named cause
(the per-episode clear) would still have produced a constant readout.
"""

from __future__ import annotations

import pytest
import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.policy.epistemic_deficit import (
    EpistemicDeficitAccumulator,
    EpistemicDeficitConfig,
)
from ree_core.utils.config import REEConfig

WORLD_DIM = 16
SEED = 71


# ------------------------------------------------------------------ #
# Config surface                                                      #
# ------------------------------------------------------------------ #


def test_readiness_config_defaults_reproduce_pre_readiness_behaviour():
    """Every readiness knob defaults to the pre-964 behaviour EXACTLY."""
    cfg = EpistemicDeficitConfig()
    assert cfg.match_radius_mode == "absolute"
    assert cfg.center_update == "replace"
    assert cfg.persist_targets_across_episodes is False
    assert cfg.require_differentiated_readout is False
    assert cfg.readout_mode == "hard_match"
    # The two inert-unless-selected numerics still carry their documented
    # values, so a caller flipping only the mode gets the stated behaviour.
    assert cfg.match_radius_relative_frac == pytest.approx(0.5)
    assert cfg.center_ema_beta == pytest.approx(0.1)


def test_readiness_config_rejects_invalid_modes():
    with pytest.raises(ValueError, match="match_radius_mode"):
        EpistemicDeficitAccumulator(
            EpistemicDeficitConfig(match_radius_mode="nope")
        )
    with pytest.raises(ValueError, match="center_update"):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(center_update="nope"))
    with pytest.raises(ValueError, match="readout_mode"):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(readout_mode="nope"))
    with pytest.raises(ValueError, match="match_radius_relative_frac"):
        EpistemicDeficitAccumulator(
            EpistemicDeficitConfig(match_radius_relative_frac=0.0)
        )
    with pytest.raises(ValueError, match="center_ema_beta"):
        EpistemicDeficitAccumulator(EpistemicDeficitConfig(center_ema_beta=0.0))


def test_every_readiness_knob_is_reachable_through_from_dims():
    """from_dims swallows unknown kwargs -- a knob wired at two of three sites
    fails open and SILENTLY. Pin all three sites for all seven knobs."""
    cfg = REEConfig.from_dims(
        body_obs_dim=8,
        world_obs_dim=8,
        action_dim=4,
        self_dim=8,
        world_dim=8,
        reafference_action_dim=4,
        epistemic_deficit_match_radius_mode="relative",
        epistemic_deficit_match_radius_relative_frac=0.25,
        epistemic_deficit_center_update="ema",
        epistemic_deficit_center_ema_beta=0.3,
        epistemic_deficit_persist_targets_across_episodes=True,
        epistemic_deficit_require_differentiated_readout=True,
        epistemic_deficit_target_frame="predicted",
        epistemic_deficit_readout_mode="rbf_weighted",
    )
    assert cfg.epistemic_deficit_match_radius_mode == "relative"
    assert cfg.epistemic_deficit_match_radius_relative_frac == pytest.approx(0.25)
    assert cfg.epistemic_deficit_center_update == "ema"
    assert cfg.epistemic_deficit_center_ema_beta == pytest.approx(0.3)
    assert cfg.epistemic_deficit_persist_targets_across_episodes is True
    assert cfg.epistemic_deficit_require_differentiated_readout is True
    assert cfg.epistemic_deficit_target_frame == "predicted"
    assert cfg.epistemic_deficit_readout_mode == "rbf_weighted"


def test_from_dims_defaults_are_the_pre_readiness_values():
    cfg = REEConfig.from_dims(
        body_obs_dim=8, world_obs_dim=8, action_dim=4, self_dim=8, world_dim=8,
        reafference_action_dim=4,
    )
    assert cfg.epistemic_deficit_match_radius_mode == "absolute"
    assert cfg.epistemic_deficit_center_update == "replace"
    assert cfg.epistemic_deficit_persist_targets_across_episodes is False
    assert cfg.epistemic_deficit_require_differentiated_readout is False
    assert cfg.epistemic_deficit_target_frame == "realized"
    assert cfg.epistemic_deficit_readout_mode == "hard_match"


# ------------------------------------------------------------------ #
# R1 -- the absolute radius exceeds the whole manifold                #
# ------------------------------------------------------------------ #


def _walk(acc: EpistemicDeficitAccumulator, n: int = 40, scale: float = 0.05):
    """Feed a random walk on a manifold whose diameter is ~= `scale`.

    Mirrors the measured 964 geometry: max pairwise L2 0.41 against the 1.0
    absolute default. `scale` here is deliberately far below match_radius=1.0.
    """
    g = torch.Generator().manual_seed(SEED)
    z = torch.zeros(8)
    for i in range(n):
        z = z + torch.randn(8, generator=g) * scale
        acc.update(z, 0.5 + 0.01 * i, 0.3, 0.2)


def test_absolute_radius_collapses_to_one_target_the_964_signature():
    """NEGATIVE CONTROL: reproduce the 964 collapse on the defaults."""
    acc = EpistemicDeficitAccumulator()  # absolute radius 1.0
    _walk(acc)
    assert acc.get_state()["n_targets"] == 1


def test_relative_radius_reaches_a_multi_target_regime():
    acc = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(
            match_radius_mode="relative", match_radius_relative_frac=0.5
        )
    )
    _walk(acc)
    state = acc.get_state()
    assert state["n_targets"] >= 2, (
        "relative radius must reach n_targets >= 2 -- the exact precondition "
        "V3-EXQ-964's C2 needs and could never satisfy"
    )
    # It must be self-calibrating, not a smaller hardcoded constant.
    assert 0.0 < state["last_effective_match_radius"] < 1.0
    assert state["zworld_scale_dev"] > 0.0


def test_relative_radius_self_calibrates_to_the_observed_scale():
    """A 10x larger manifold gets a ~10x larger effective radius -- the
    property a fixed absolute threshold structurally cannot have."""
    small = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(match_radius_mode="relative")
    )
    big = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(match_radius_mode="relative")
    )
    _walk(small, scale=0.05)
    _walk(big, scale=0.5)
    r_small = small.get_state()["last_effective_match_radius"]
    r_big = big.get_state()["last_effective_match_radius"]
    assert r_big > 5.0 * r_small


# ------------------------------------------------------------------ #
# R2 -- re-centering makes a matched target a random walk             #
# ------------------------------------------------------------------ #


def test_replace_center_update_re_centers_onto_the_latest_observation():
    """NEGATIVE CONTROL: the pre-readiness `replace` behaviour, pinned."""
    acc = EpistemicDeficitAccumulator()  # replace + absolute 1.0
    a = torch.zeros(4)
    b = torch.tensor([0.4, 0.0, 0.0, 0.0])
    acc.update(a, 0.5, 0.0, 0.0)
    acc.update(b, 0.5, 0.0, 0.0)
    assert acc.get_state()["n_targets"] == 1
    assert torch.allclose(acc._targets[0]["center"], b)


def test_ema_center_update_anchors_the_target_near_where_it_was_observed():
    acc = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(center_update="ema", center_ema_beta=0.1)
    )
    a = torch.zeros(4)
    b = torch.tensor([0.4, 0.0, 0.0, 0.0])
    acc.update(a, 0.5, 0.0, 0.0)
    acc.update(b, 0.5, 0.0, 0.0)
    center = acc._targets[0]["center"]
    # 0.1 of the way from a toward b, not all the way.
    assert torch.allclose(center, 0.1 * b, atol=1e-6)
    assert not torch.allclose(center, b)


# ------------------------------------------------------------------ #
# R4 -- the hard threshold saturates in BOTH directions               #
# ------------------------------------------------------------------ #


def _two_targets(cfg: EpistemicDeficitConfig) -> EpistemicDeficitAccumulator:
    acc = EpistemicDeficitAccumulator(cfg)
    acc._targets = [
        {"center": torch.zeros(4), "deficit": 0.2},
        {"center": torch.tensor([1.0, 0.0, 0.0, 0.0]), "deficit": 0.9},
    ]
    return acc


def test_hard_match_saturates_when_the_radius_swallows_every_candidate():
    """NEGATIVE CONTROL, arm A of the 964 collapse: a wide radius makes every
    candidate match the SAME target, so the readout is a CONSTANT vector."""
    acc = _two_targets(EpistemicDeficitConfig(match_radius=10.0))
    cands = torch.zeros(5, 4) + torch.linspace(0, 0.01, 5).unsqueeze(1)
    out = acc.readout(cands)
    assert out is not None
    assert float(out.max() - out.min()) == pytest.approx(0.0, abs=1e-12)
    assert acc.get_state()["last_readout_deficit_range"] == pytest.approx(0.0)


def test_hard_match_saturates_when_the_radius_excludes_every_candidate():
    """NEGATIVE CONTROL, arm B: the OTHER saturation direction. A radius small
    enough to separate targets can leave every candidate unmatched -- also a
    constant (all zeros). This is why tuning the radius alone cannot work."""
    acc = _two_targets(EpistemicDeficitConfig(match_radius=1e-4))
    cands = torch.zeros(5, 4) + 0.5
    out = acc.readout(cands)
    assert out is not None
    assert float(out.max() - out.min()) == pytest.approx(0.0, abs=1e-12)


def test_rbf_weighted_readout_differentiates_tightly_clustered_candidates():
    """The fix: a graded read is CONTINUOUS in candidate position, so it
    differentiates whenever candidates differ at all -- no knife-edge radius."""
    acc = _two_targets(
        EpistemicDeficitConfig(match_radius=10.0, readout_mode="rbf_weighted")
    )
    cands = torch.zeros(5, 4) + torch.linspace(0, 0.01, 5).unsqueeze(1)
    out = acc.readout(cands)
    assert out is not None
    assert float(out.max() - out.min()) > 0.0
    assert acc.get_state()["last_readout_deficit_range"] > 0.0
    # Monotone: a candidate nearer the high-deficit target reads higher.
    assert bool(torch.all(out[1:] >= out[:-1]))


def test_rbf_weighted_readout_still_returns_none_with_no_targets():
    acc = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(readout_mode="rbf_weighted")
    )
    assert acc.readout(torch.zeros(4, 4)) is None


# ------------------------------------------------------------------ #
# The runtime differentiation gate (autopsy learning #4)              #
# ------------------------------------------------------------------ #


def test_differentiation_gate_off_by_default_returns_the_constant():
    acc = _two_targets(EpistemicDeficitConfig(match_radius=10.0))
    out = acc.readout(torch.zeros(5, 4))
    assert out is not None  # default: return the provably-inert constant


def test_differentiation_gate_refuses_a_constant_readout_and_self_reports():
    acc = _two_targets(
        EpistemicDeficitConfig(
            match_radius=10.0, require_differentiated_readout=True
        )
    )
    out = acc.readout(torch.zeros(5, 4))
    assert out is None, (
        "a constant lp_vec is provably argmax-inert (subtracting a constant "
        "from every candidate score cannot move an argmax) -- the runtime "
        "structural-unsatisfiability check the 964 autopsy asks for"
    )
    state = acc.get_state()
    assert state["n_undifferentiated_readouts"] == 1
    assert state["n_vacuous_readouts"] == 1
    assert state["last_readout_vacuous"] is True


def test_differentiation_gate_admits_a_differentiated_readout():
    acc = _two_targets(
        EpistemicDeficitConfig(
            match_radius=10.0,
            readout_mode="rbf_weighted",
            require_differentiated_readout=True,
        )
    )
    cands = torch.zeros(5, 4) + torch.linspace(0, 0.5, 5).unsqueeze(1)
    out = acc.readout(cands)
    assert out is not None
    assert acc.get_state()["n_undifferentiated_readouts"] == 0


# ------------------------------------------------------------------ #
# Cross-episode persistence                                           #
# ------------------------------------------------------------------ #


def test_reset_clears_targets_by_default():
    """NEGATIVE CONTROL: the pre-readiness per-episode clear, pinned."""
    acc = EpistemicDeficitAccumulator()
    acc.update(torch.zeros(4), 0.5, 0.0, 0.0)
    assert acc.get_state()["n_targets"] == 1
    acc.reset()
    assert acc.get_state()["n_targets"] == 0


def test_persist_targets_across_episodes_keeps_targets_and_scale():
    acc = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(persist_targets_across_episodes=True)
    )
    acc.update(torch.zeros(4), 0.5, 0.0, 0.0)
    acc.update(torch.ones(4) * 5.0, 0.5, 0.0, 0.0)
    before = acc.get_state()
    acc.reset()
    after = acc.get_state()
    assert after["n_targets"] == before["n_targets"] >= 1
    assert after["zworld_scale_n"] == before["zworld_scale_n"]
    # Diagnostics are still per-episode, unchanged either way.
    assert after["n_updates"] == 0
    assert after["n_readouts"] == 0


# ------------------------------------------------------------------ #
# MECH-094 -- unchanged by the readiness knobs                        #
# ------------------------------------------------------------------ #


def test_simulation_mode_is_still_a_noop_under_every_readiness_knob():
    acc = EpistemicDeficitAccumulator(
        EpistemicDeficitConfig(
            match_radius_mode="relative",
            center_update="ema",
            persist_targets_across_episodes=True,
            readout_mode="rbf_weighted",
        )
    )
    acc.update(torch.zeros(4), 0.5, 0.0, 0.0, simulation_mode=True)
    state = acc.get_state()
    assert state["n_targets"] == 0
    assert state["n_updates"] == 0
    assert state["zworld_scale_n"] == 0
    assert state["last_n_simulation_skips"] == 1


# ------------------------------------------------------------------ #
# R3 + end-to-end -- the frames, and the whole chain on a real agent   #
# ------------------------------------------------------------------ #


def _cfg(readiness: bool) -> REEConfig:
    env = CausalGridWorldV2()
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
    )
    cfg.use_structured_curiosity = True
    cfg.use_curiosity_novelty = False
    cfg.use_curiosity_uncertainty = False
    cfg.use_curiosity_learning_progress = True
    cfg.curiosity_learning_progress_source = "epistemic_deficit"
    cfg.latent.use_e2_world_uncertainty = True
    cfg.latent.use_e2_world_uncertainty_online_training = True
    cfg.latent.e2_world_uncertainty_warmup_steps = 20
    cfg.latent.e2_world_uncertainty_batch_size = 8
    if readiness:
        cfg.epistemic_deficit_match_radius_mode = "relative"
        cfg.epistemic_deficit_match_radius_relative_frac = 0.5
        cfg.epistemic_deficit_center_update = "ema"
        cfg.epistemic_deficit_persist_targets_across_episodes = True
        cfg.epistemic_deficit_target_frame = "predicted"
        cfg.epistemic_deficit_readout_mode = "rbf_weighted"
    return cfg


def _drive(readiness: bool, episodes: int = 2, steps: int = 40):
    torch.manual_seed(SEED)
    env = CausalGridWorldV2()
    agent = REEAgent(_cfg(readiness))
    n_diff = 0
    n_read = 0
    for _ in range(episodes):
        _, obs = env.reset()
        agent.reset()
        for _ in range(steps):
            latent = agent.sense(obs["body_state"], obs["world_state"])
            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent)
                if ticks.get("e1_tick")
                else torch.zeros(1, WORLD_DIM, device=agent.device)
            )
            cands = agent.generate_trajectories(latent, e1_prior, ticks)
            agent.update_z_goal(
                benefit_exposure=0.0,
                drive_level=REEAgent.compute_drive_level(obs["body_state"]),
            )
            action = agent.select_action(cands, ticks)
            st = agent.epistemic_deficit.get_state()
            if st["n_readouts"] > 0:
                n_read += 1
                if st["last_readout_deficit_range"] > 0.0:
                    n_diff += 1
            _f, harm, _d, _i, obs = env.step(int(action.argmax(dim=-1).item()))
            agent.update_residue(harm)
    return agent, n_diff, n_read


def test_target_frame_is_reachable_and_defaults_to_realized():
    agent_off = REEAgent(_cfg(False))
    assert agent_off.config.epistemic_deficit_target_frame == "realized"
    agent_on = REEAgent(_cfg(True))
    assert agent_on.config.epistemic_deficit_target_frame == "predicted"


def test_end_to_end_off_arm_reproduces_the_964_collapse():
    """NEGATIVE CONTROL, and the load-bearing one: on a REAL agent the
    pre-readiness defaults must still produce n_targets == 1 and a readout
    whose cross-candidate range is EXACTLY zero -- the V3-EXQ-964 signature.

    Without this the ON arm proves nothing: two green arms would be equally
    consistent with 'the fix works' and 'the collapse was never reproduced'.
    """
    agent, n_diff, n_read = _drive(readiness=False)
    state = agent.epistemic_deficit.get_state()
    assert state["n_targets"] == 1
    assert n_read > 0, "the readout path must actually be exercised"
    assert n_diff == 0
    assert state["last_readout_deficit_range"] == pytest.approx(0.0)


def test_end_to_end_on_arm_reaches_a_differentiated_multi_target_readout():
    """The C2 precondition the autopsy names: n_targets >= 2 AND a
    per-candidate readout that is not constant."""
    agent, n_diff, n_read = _drive(readiness=True)
    state = agent.epistemic_deficit.get_state()
    assert state["max_n_targets"] >= 2
    assert n_read > 0
    assert n_diff > 0
    assert state["readout_mode"] == "rbf_weighted"


def test_end_to_end_on_arm_lifts_structured_curiosity_lp_dev_range():
    """The autopsy's own named readiness statistic.

    _last_lp_dev_range = max(lp_contrib) - min(lp_contrib) is 'exactly the
    number that decides whether the per-candidate read can differentiate'. It
    is already computed and exposed by StructuredCuriosity.get_state(); the
    964 recording gap was the DRIVER never recording it. Pin that it moves off
    zero, since that is what makes C2 satisfiable downstream.
    """
    agent_on, _, _ = _drive(readiness=True)
    assert agent_on.curiosity.get_state()["last_lp_dev_range"] > 0.0
