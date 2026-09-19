"""Contracts for the SD-011 P0h affective-harm-encoder warmup (experiments/_lib/zharm_a_p0_warmup.py).

WHAT THIS DEFENDS. The defect being fixed is silent by construction, exactly like its z_world
sibling: the three optimizer groups `_train_all_on_agent` builds (e2, lPFC bias head, OFC
devaluation head) cover no AffectiveHarmEncoder parameter, so `z_harm_a` is a frozen random
projection for the whole run with no error and no warning. An experiment whose DV reads
`z_harm_a` on that path has been measuring a random projection. Measured 2026-09-18; see
`REE_assembly/evidence/planning/sd086_zharma_readout_precondition_staged_20260918.md` sec 4b.

The three properties the chip that commissioned this build named, and where they are pinned:

    "params covered when ON"                 -> C1
    "bit-identical behaviour when OFF"       -> C2 (fixed seed, RNG + params + result block)
    "encoder weights change after N steps"   -> C3

C0 pins the DEFECT ITSELF rather than the fix: it asserts the affective encoder is still
disjoint from the three legacy optimizer groups. That is deliberate -- if someone later covers
the encoder from one of those groups instead, this test fails and forces the substrate_queue
record to be updated rather than silently becoming wrong.

C4 pins that every half-configured shape REFUSES loudly instead of running a zero-gradient
warmup and reporting success -- `compute_harm_accum_loss` returns a zero loss whenever
`harm_history_len <= 0`, which is correct for its per-tick callers and silently fatal here.

C6 pins RNG neutrality as a SHARED object with the z_world stage. That is not style: an
unguarded warmup shifts every subsequent draw in P0b/P1, so an ON-vs-OFF contrast would
confound "the encoder is now trained" with "the RNG stream moved".

Assertions are on losses, weight deltas and RNG state -- never on a sampled action
(`torch.multinomial` is not portable across machine classes; CLAUDE.md "Running the test suite").

ASCII-only (repo rule).
"""

import numpy as np
import pytest
import torch

import experiments._lib.allon_training as allon
import experiments._lib.zharm_a_p0_warmup as zh
import experiments._lib.zworld_p0_warmup as zw
from experiments._lib.capability_eval import RandomPolicy
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

HARM_HISTORY_LEN = 8
STEPS = 20


def _make_env(seed: int, harm_history_len: int = HARM_HISTORY_LEN) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, use_proxy_fields=True, harm_history_len=harm_history_len,
    )


def _make_agent(seed: int, harm_history_len: int = HARM_HISTORY_LEN,
                affective: bool = True) -> REEAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = _make_env(seed, harm_history_len)
    _flat, obs = env.reset()
    cfg = REEConfig.from_dims(
        body_obs_dim=obs["body_state"].shape[-1],
        world_obs_dim=obs["world_state"].shape[-1],
        action_dim=env.action_dim,
    )
    cfg.latent.use_affective_harm_stream = bool(affective)
    cfg.latent.harm_history_len = int(harm_history_len)
    if obs.get("harm_obs_a") is not None:
        cfg.latent.harm_obs_a_dim = obs["harm_obs_a"].shape[-1]
    return REEAgent(cfg)


def _rng_fingerprint():
    return (
        torch.get_rng_state().clone(),
        np.random.get_state()[1].copy(),
    )


def _rng_equal(a, b) -> bool:
    return bool(torch.equal(a[0], b[0])) and bool(np.array_equal(a[1], b[1]))


# --------------------------------------------------------------------------------------
# C0 -- the defect itself, still true.
# --------------------------------------------------------------------------------------

def test_c0_legacy_optimizer_groups_cover_no_affective_encoder_param():
    """The e2 / lPFC-bias / OFC-devaluation groups `_train_all_on_agent` builds are DISJOINT
    from the affective encoder. This is the 2026-09-18 measurement, pinned so that a later
    change which covers the encoder from one of those groups cannot silently invalidate the
    substrate_queue record without failing a test."""
    agent = _make_agent(0)
    legacy = set()
    legacy |= {id(p) for p in agent.e2.parameters()}
    if getattr(agent, "lateral_pfc", None) is not None:
        legacy |= {id(p) for p in agent.lateral_pfc.bias_head_parameters()}
    if getattr(agent, "ofc", None) is not None:
        legacy |= {id(p) for p in agent.ofc.devaluation_bias_head_parameters()}

    affective = zh.affective_encoder_parameters(agent)
    assert affective, "affective encoder should exist in this fixture"
    covered = [p for p in affective if id(p) in legacy]
    assert covered == [], (
        "an affective-encoder parameter is now covered by a legacy optimizer group; the "
        "sd_zharm_a_warmup_optimizer_group substrate_queue entry must be re-derived"
    )


# --------------------------------------------------------------------------------------
# C1 -- params covered when ON.
# --------------------------------------------------------------------------------------

def test_c1_optimizer_group_covers_encoder_and_aux_head():
    agent = _make_agent(0)
    enc = agent.latent_stack.affective_harm_encoder
    expected = {id(p) for p in enc.parameters()}
    got = {id(p) for p in zh.affective_encoder_parameters(agent)}
    assert got == expected and got, "P0h group must be exactly affective_harm_encoder.parameters()"
    # The aux head is the ONLY gradient source: without it the loss short-circuits to zero.
    assert enc.harm_accum_head is not None
    head_ids = {id(p) for p in enc.harm_accum_head.parameters()}
    assert head_ids and head_ids <= got, "harm_accum_head params must be in the P0h group"


# --------------------------------------------------------------------------------------
# C2 -- OFF is bit-identical at a fixed seed.
# --------------------------------------------------------------------------------------

def test_c2_off_makes_no_rng_draw_and_moves_no_parameter():
    agent = _make_agent(0)
    before = zh.encoder_weight_snapshot(agent)
    all_before = [p.detach().clone() for p in agent.parameters()]
    rng_before = _rng_fingerprint()

    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=0,
                            steps_per_episode=STEPS, policy=RandomPolicy(0), label="c2")

    assert out["p0h_ran"] is False and out["p0h_reason"] == "episodes<=0"
    assert _rng_equal(rng_before, _rng_fingerprint()), "OFF must draw no RNG"
    assert zh.encoder_weight_delta(before, zh.encoder_weight_snapshot(agent)) == (0, 0.0)
    for b, p in zip(all_before, agent.parameters()):
        assert torch.equal(b, p), "OFF must move no agent parameter at all"


def test_c2b_train_all_on_agent_default_is_off():
    """The wiring default is 0, so every existing caller stays on the prior path."""
    import inspect
    sig = inspect.signature(allon._train_all_on_agent)
    assert sig.parameters["zharm_a_p0_episodes"].default == 0
    assert sig.parameters["zharm_a_p0_env"].default is None
    assert sig.parameters["zharm_a_p0_config"].default is None


def test_c2c_train_all_on_agent_refuses_opt_in_without_a_dedicated_env():
    """Reusing train_env would shift the layout sequence P0b/P1 then see, which is the
    confound the dedicated-env rule exists to prevent -- so it raises rather than warns."""
    agent = _make_agent(0)
    with pytest.raises(ValueError, match="requires zharm_a_p0_env"):
        allon._train_all_on_agent(
            agent, _make_env(0), seed=0, p0_episodes=0, p1_episodes=0,
            steps_per_episode=1, rung_id="c2c", total_denominator=1,
            zharm_a_p0_episodes=4,
        )


# --------------------------------------------------------------------------------------
# C3 -- ON moves the encoder, restores the RNG, and generalises.
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def on_run():
    agent = _make_agent(0)
    before = zh.encoder_weight_snapshot(agent)
    ema_before = getattr(agent, "_harm_obs_ema", None)
    rng_before = _rng_fingerprint()
    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=6,
                            steps_per_episode=STEPS, policy=RandomPolicy(0), label="c3")
    return agent, before, ema_before, rng_before, out


def test_c3_on_moves_the_encoder_weights(on_run):
    agent, before, _ema, _rng, out = on_run
    assert out["p0h_ran"] is True, out.get("p0h_reason")
    assert out["p0h_n_steps"] > 0
    n_changed, max_abs = zh.encoder_weight_delta(before, zh.encoder_weight_snapshot(agent))
    assert n_changed == out["p0h_n_encoder_tensors"], (
        "every affective-encoder tensor must move; %d of %d did"
        % (n_changed, out["p0h_n_encoder_tensors"])
    )
    assert max_abs > 0.0
    assert out["p0h_encoder_tensors_changed"] == n_changed
    assert out["p0h_encoder_max_abs_delta"] == pytest.approx(max_abs)


def test_c3b_on_restores_the_global_rng_streams(on_run):
    _agent, _before, _ema, rng_before, _out = on_run
    assert _rng_equal(rng_before, _rng_fingerprint()), (
        "P0h must be RNG-neutral; otherwise an ON/OFF contrast confounds 'encoder trained' "
        "with 'RNG stream moved'"
    )


def test_c3c_on_restores_the_sd020_expected_harm_tracker(on_run):
    agent, _before, ema_before, _rng, _out = on_run
    if ema_before is None:
        pytest.skip("agent has no _harm_obs_ema attribute")
    assert getattr(agent, "_harm_obs_ema") == ema_before


def test_c3d_training_loss_falls_and_the_holdout_follows(on_run):
    """The weight delta alone can be satisfied vacuously (an encoder can move a long way and
    fit only the training order). The held-out EPISODES are what say the signal generalises."""
    _agent, _before, _ema, _rng, out = on_run
    assert out["p0h_final_epoch_mean_loss"] < out["p0h_first_epoch_mean_loss"]
    assert out["p0h_n_holdout"] > 0 and out["p0h_n_holdout_episodes"] > 0
    assert out["p0h_holdout_loss_post"] < out["p0h_holdout_loss_pre"]
    assert out["p0h_holdout_loss_drop"] > 0.0


def test_c3f_holdout_is_scored_against_a_constant_mean_predictor(on_run):
    """A falling loss curve is NOT evidence the encoder learned the signal.

    `harm_accum_head` ends in a Sigmoid and starts near 0.5, so if the target sits near a small
    constant every other readout in the block (epoch loss down, held-out loss down, all tensors
    moved) is satisfied by a model that learned only the offset. Measured on this substrate
    2026-09-18: `accumulated_harm` has mean ~0.028 and std ~0.004 over a 600-tick random
    rollout, and the trained head scores WORSE than a constant-mean predictor on held-out
    episodes. The lift is what makes that visible, so it must always be reported."""
    _agent, _before, _ema, _rng, out = on_run
    blk = out["p0h_holdout_vs_constant"]
    assert blk["basis"] == "sd011_accumulated_harm"
    assert blk["lift"] is not None
    assert blk["head_mse"] is not None and blk["const_mse"] is not None
    # The target's own dispersion has to be on the record: a near-zero std IS the explanation
    # for a negative lift, and without it a reader cannot tell a weak encoder from a
    # near-degenerate target.
    assert blk["target_std"] is not None and blk["target_mean"] is not None


def test_c3h_effective_target_recovery_is_exact_on_the_sd011_path():
    """`recover_effective_target` inverts the agent's own loss (pred forced to 0, so
    loss = weight * target**2). On the SD-011 path the target is the buffered
    `accumulated_harm` scalar, so the recovery must reproduce it EXACTLY.

    This is what makes the recovery safe to use as a scoring basis instead of re-deriving the
    target here: if a future change makes either target SIGNED, or alters the loss shape away
    from `weight * mse`, the square root silently starts returning magnitudes -- and this fails
    rather than the readout quietly becoming wrong."""
    agent = _make_agent(0)
    assert bool(getattr(agent.config, "harm_surprise_pe_enabled", False)) is False
    for value in (0.0, 0.0137, 0.25, 0.9):
        got = zh.recover_effective_target(agent, value)
        assert got == pytest.approx(value, abs=1e-6), (
            "SD-011 recovery %r != buffered target %r" % (got, value)
        )


def test_c3i_effective_target_recovery_tracks_the_sd020_pe_formula():
    """On the SD-020 path the head regresses `|actual - expected| * precision_norm`, NOT the
    buffered scalar -- so the recovery must reproduce THAT, walking the expected-harm EMA in
    temporal order exactly as `compute_harm_accum_loss` advances it.

    Pinned independently of C3h because the two paths fail differently: C3h would still pass if
    the PE branch were broken, and this is the branch the 2026-09-18 option-B measurement reads.
    """
    agent = _make_agent(0)
    agent.config.harm_surprise_pe_enabled = True
    alpha = float(getattr(agent.config, "harm_obs_ema_alpha", 0.1))
    prec_norm = min(float(agent.e3.current_precision) / 500.0, 3.0)

    ema = float(getattr(agent, "_harm_obs_ema", 0.0))
    for value in (0.05, 0.05, 0.4, 0.4, 0.0):
        ema = (1.0 - alpha) * ema + alpha * value      # advanced BEFORE the PE, per the agent
        expected = abs(value - ema) * prec_norm
        got = zh.recover_effective_target(agent, value)
        assert got == pytest.approx(expected, abs=1e-9, rel=1e-5), (
            "SD-020 recovery %r != |actual-expected|*precision_norm %r" % (got, expected)
        )


def test_c3j_the_pe_path_is_SCORED_not_reported_as_unavailable():
    """REGRESSION PIN for the 2026-09-18 option-B fix. The first cut returned `lift: None` on
    the SD-020 path, which made `p0h_readiness_met` False BY CONSTRUCTION there rather than by
    measurement -- i.e. the instrument could not answer the question it was built to answer."""
    agent = _make_agent(0)
    agent.config.harm_surprise_pe_enabled = True
    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=6,
                            steps_per_episode=STEPS, policy=RandomPolicy(0), label="c3j")
    assert out["p0h_ran"] is True, out.get("p0h_reason")
    assert out["p0h_target"] == "sd020_precision_weighted_pe"
    blk = out["p0h_holdout_vs_constant"]
    assert blk["basis"] == "sd020_precision_weighted_pe"
    assert blk["lift"] is not None, "the PE path must be SCORED, not reported as unavailable"
    assert blk["head_mse"] is not None and blk["const_mse"] is not None
    assert isinstance(out["p0h_readiness_met"], bool)


def test_c3g_readiness_is_the_conjunction_of_both_halves(on_run):
    """Readiness requires the gradient path to have REACHED the encoder (weight delta) AND the
    result to beat a constant (lift). Either alone is satisfiable by a failure mode: a zero
    delta is the defect this module fixes; a non-positive lift is a constant-fit."""
    _agent, _before, _ema, _rng, out = on_run
    lift = out["p0h_holdout_vs_constant"]["lift"]
    expected = bool(out["p0h_encoder_tensors_changed"] > 0 and lift is not None and lift > 0.0)
    assert out["p0h_readiness_met"] is expected
    assert "holdout_lift_over_constant_mean" in out["p0h_readiness_basis"]


def test_c3e_target_name_reports_which_supervision_actually_ran(on_run):
    """SD-011 EMA vs SD-020 precision-weighted PE is the caller's existing
    `harm_surprise_pe_enabled` flag, not a choice this stage makes -- but the manifest must
    say which one ran, or a reader cannot tell what was trained."""
    _agent, _before, _ema, _rng, out = on_run
    assert out["p0h_target"] == "sd011_accumulated_harm_ema"
    assert out["p0h_harm_surprise_pe_enabled"] is False


# --------------------------------------------------------------------------------------
# C3k-C3n -- the 2026-09-19 two-arm re-specification levers (user decision OPTION H).
# Both default to the prior behaviour; each isolates one measured cause of the readiness
# failure. These pin the LEVERS, not the scientific result -- the measurement itself lives in
# docs/substrate/SD-011-p0h-affective-encoder-warmup.md and the substrate_queue entry.
# --------------------------------------------------------------------------------------

def test_c3k_lever_defaults():
    """Arm F stays OFF by default. Arm E's explicit PIN also stays off -- but since
    2026-09-19 (user decision OPTION I) a default FLOOR of 0.1 is ON, which is the one
    setting here that changes behaviour for a caller who opts into P0h."""
    cfg = zh.ZHarmAP0Config()
    assert cfg.target_source == "accumulated_harm", "arm F must be OFF by default"
    assert cfg.p0_precision_norm is None, "the explicit PIN must stay off by default"
    assert cfg.p0_precision_norm_floor == pytest.approx(0.1), (
        "the P0 precision floor is the adopted default (OPTION I, mid-plateau on the pin sweep)"
    )


def test_c3o_the_floor_is_a_floor_not_a_pin():
    """A FLOOR only lifts an agent that is BELOW it; it never drags a higher one down.

    This is the whole point of the SD-020 finding: the ARC-016 coupling is correct at RUNTIME
    and wrong at P0. A pin would discard a genuine trained-agent precision; a floor rescues
    only the P0 case, where precision has not yet had a chance to exist."""
    r = zh.resolve_p0_precision_norm
    # below the floor -> lifted
    assert r(0.004, None, 0.1) == pytest.approx(0.1)
    # already above it -> untouched (None = no mutation at all)
    assert r(0.19, None, 0.1) is None
    assert r(3.0, None, 0.1) is None
    # exactly at it -> untouched
    assert r(0.1, None, 0.1) is None
    # an explicit pin wins outright, in BOTH directions
    assert r(0.004, 1.0, 0.1) == pytest.approx(1.0)
    assert r(3.0, 0.02, 0.1) == pytest.approx(0.02)
    # floor disabled -> pre-2026-09-19 behaviour exactly
    assert r(0.004, None, None) is None


def test_c3p_the_default_floor_actually_reaches_the_stage():
    """End-to-end, not just the resolver: a default-config PE run must report the floor as the
    source and apply it, and must still restore E3's running variance afterwards."""
    agent = _make_agent(0)
    agent.config.harm_surprise_pe_enabled = True
    before_var = agent.e3._running_variance
    baseline = zh.current_precision_norm(agent)
    assert baseline < 0.1, "fixture assumption: an untrained agent sits below the floor"

    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=6, steps_per_episode=STEPS,
                            policy=RandomPolicy(0), label="c3p")
    assert out["p0h_ran"] is True, out.get("p0h_reason")
    assert out["p0h_precision_norm_requested"] is None
    assert out["p0h_precision_norm_floor"] == pytest.approx(0.1)
    assert out["p0h_precision_norm_source"] == "floor"
    assert out["p0h_precision_norm_applied"] == pytest.approx(0.1, rel=1e-6)
    assert agent.e3._running_variance == before_var, "the floor LEAKED out of the stage"


def test_c3q_disabling_the_floor_restores_the_pre_option_i_path():
    agent = _make_agent(0)
    agent.config.harm_surprise_pe_enabled = True
    baseline = zh.current_precision_norm(agent)
    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=4, steps_per_episode=STEPS,
                            policy=RandomPolicy(0), label="c3q",
                            config=zh.ZHarmAP0Config(seed=0, p0_precision_norm_floor=None))
    assert out["p0h_precision_norm_source"] == "agent_untouched"
    assert out["p0h_precision_norm_applied"] == pytest.approx(baseline, rel=1e-9)


def test_c3l_arm_e_pins_precision_norm_and_restores_it():
    """Arm E pins the ARC-016 factor FOR THE STAGE ONLY. The restore is the load-bearing half:
    an unrestored pin would silently change E3's running variance for every phase that follows,
    turning a P0 diagnostic into a whole-run manipulation."""
    agent = _make_agent(0)
    agent.config.harm_surprise_pe_enabled = True
    before_var = agent.e3._running_variance
    before_norm = zh.current_precision_norm(agent)
    assert before_norm == pytest.approx(min(2.0 / 500.0, 3.0), rel=1e-3)

    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=6, steps_per_episode=STEPS,
                            policy=RandomPolicy(0), label="c3l",
                            config=zh.ZHarmAP0Config(seed=0, p0_precision_norm=1.0))
    assert out["p0h_ran"] is True, out.get("p0h_reason")
    assert out["p0h_precision_norm_requested"] == 1.0
    assert out["p0h_precision_norm_applied"] == pytest.approx(1.0, rel=1e-6)
    assert out["p0h_precision_norm_baseline"] == pytest.approx(before_norm, rel=1e-9)
    assert agent.e3._running_variance == before_var, "the pin LEAKED out of the stage"
    assert zh.current_precision_norm(agent) == pytest.approx(before_norm, rel=1e-9)


def test_c3m_arm_e_is_reported_inert_when_the_pe_branch_is_off():
    """Only the SD-020 branch reads precision, so a pin with PE off changes nothing. It must
    read as INERT rather than looking applied -- a caller that half-configured its arm should
    see a False here, not assume it was manipulated."""
    agent = _make_agent(0)
    assert bool(getattr(agent.config, "harm_surprise_pe_enabled", False)) is False
    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=4, steps_per_episode=STEPS,
                            policy=RandomPolicy(0), label="c3m",
                            config=zh.ZHarmAP0Config(seed=0, p0_precision_norm=1.0))
    assert out["p0h_precision_override_inert"] is True
    # The DEFAULT floor is inert on this path for the same reason, and must say so rather than
    # look applied -- otherwise the OPTION I default reads as active on every SD-011 run.
    out2 = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=4, steps_per_episode=STEPS,
                             policy=RandomPolicy(0), label="c3m2")
    assert out2["p0h_precision_norm_source"] == "floor"
    assert out2["p0h_precision_override_inert"] is True


def test_c3n_arm_f_reads_the_per_tick_scalar_and_rejects_an_unknown_source():
    """Arm F swaps the supervision scalar for the PER-TICK `harm_exposure` the env already
    emits at `harm_obs[-1]` -- no env change, no new channel. An unknown source raises rather
    than silently falling back, because a silent fallback here would train one arm on the other
    arm's target and report the wrong label."""
    env = _make_env(0)
    _flat, obs = env.reset()
    assert zh._target_scalar(obs, "accumulated_harm") == pytest.approx(
        float(obs["accumulated_harm"]))
    assert zh._target_scalar(obs, "harm_exposure") == pytest.approx(
        float(obs["harm_obs"].reshape(-1)[-1].item()))
    with pytest.raises(ValueError, match="unknown target_source"):
        zh._target_scalar(obs, "not_a_channel")

    agent = _make_agent(0)
    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=6, steps_per_episode=STEPS,
                            policy=RandomPolicy(0), label="c3n",
                            config=zh.ZHarmAP0Config(seed=0, target_source="harm_exposure"))
    assert out["p0h_ran"] is True, out.get("p0h_reason")
    assert out["p0h_target_source"] == "harm_exposure"
    assert out["p0h_target"] == "sd011_per_tick_harm_exposure"
    assert out["p0h_holdout_vs_constant"]["lift"] is not None


# --------------------------------------------------------------------------------------
# C4 -- half-configured shapes refuse loudly rather than training on a zero gradient.
# --------------------------------------------------------------------------------------

def test_c4a_refuses_when_the_affective_stream_is_off():
    agent = _make_agent(0, affective=False)
    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=2,
                            steps_per_episode=5, policy=RandomPolicy(0), label="c4a")
    assert out["p0h_ran"] is False
    assert "affective_harm_encoder absent" in out["p0h_reason"]


def test_c4b_refuses_when_harm_history_is_disabled_on_the_agent():
    """With harm_history_len=0 there is no harm_accum_head, compute_harm_accum_loss returns a
    zero loss, and a naive stage would report a clean run having stepped on nothing."""
    agent = _make_agent(0, harm_history_len=0)
    out = zh.run_zharm_a_p0(agent, _make_env(0), seed=0, episodes=2,
                            steps_per_episode=5, policy=RandomPolicy(0), label="c4b")
    assert out["p0h_ran"] is False
    assert "harm_accum_head absent" in out["p0h_reason"]


def test_c4c_refuses_when_the_env_emits_no_harm_history():
    agent = _make_agent(0)
    out = zh.run_zharm_a_p0(agent, _make_env(0, harm_history_len=0), seed=0, episodes=2,
                            steps_per_episode=5, policy=RandomPolicy(0), label="c4c")
    assert out["p0h_ran"] is False
    assert "warmup_env.harm_history_len" in out["p0h_reason"]


def test_c4d_weight_delta_raises_on_a_snapshot_mismatch():
    """A silent zero here would read identically to the defect this module exists to fix."""
    with pytest.raises(ValueError, match="snapshot length mismatch"):
        zh.encoder_weight_delta([torch.zeros(2)], [])


# --------------------------------------------------------------------------------------
# C5/C6 -- wiring and shared RNG neutrality.
# --------------------------------------------------------------------------------------

def test_c5_stage_is_wired_into_the_shared_trainer_and_reported():
    src = allon._train_all_on_agent.__doc__ or ""
    del src  # the docstring is not the contract; the call site and the result key are.
    import inspect
    body = inspect.getsource(allon._train_all_on_agent)
    assert "run_zharm_a_p0(" in body, "P0h stage must be called by the shared trainer"
    assert '"zharm_a_p0": zharm_a_p0_stats' in body, (
        "the stage's diagnostic block must reach the returned dict, or a manifest cannot say "
        "whether the encoder was trained"
    )
    # ORDERING: P0h must run BEFORE the P0b e2 warmup optimizers are built, because z_harm_a
    # feeds E3 commit gating on every tick of P0b and P1.
    assert body.index("run_zharm_a_p0(") < body.index("e2_opt = torch.optim.Adam")


def test_c6_rng_neutrality_is_the_same_object_as_the_zworld_stage():
    assert zh._rng_neutral is zw._rng_neutral, (
        "the two P0 stages must not be able to drift on RNG neutrality"
    )


def test_c7_module_output_is_ascii_only():
    import pathlib
    path = pathlib.Path(zh.__file__)
    text = path.read_text(encoding="utf-8")
    bad = [i + 1 for i, line in enumerate(text.splitlines())
           if any(ord(c) > 127 for c in line)]
    assert bad == [], "non-ASCII on lines %r (repo rule: ASCII-only)" % (bad,)
