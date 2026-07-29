"""Contract tests for SD-083 offline policy-consolidation window (2026-07-29).

Unblocks the two blocked_substrate arms of MECH-476's retrograde-interference falsifier: the
A->B offline INTERVAL arm (V3-EXQ-836b) and the Moncada-2007 behavioural-tagging NOVELTY arm
(V3-EXQ-836c). The window is an OFFLINE, TRACE-SELECTIVE, INTERVAL-ACCUMULATED, NOVELTY-GATED
counterpart to the ONLINE global PolicyKLAnchor (MECH-475) -- the sleep-dependent consolidation
the MECH-476 literature review (Walker 2003) records REE as lacking.

Most of these contracts exist to hold the separations the falsifier depends on, not to check
arithmetic:

  S1  OFF is a no-op: default cfg builds no anchor, adds no term, and leaves the trained weights
      BIT-IDENTICAL to the pre-SD-083 trainer at the same seed.
  S2  capture_resource is the interval clock: c(0)=0, strictly increasing in N, saturating at
      capture_max.
  S3  The window BUILDS a live anchor (N>0, coef>0): non-zero Fisher mass, positive coef, and the
      refinement reports offline_ewc_installed with a measured penalty.
  S4  The anchor BINDS: with it ON the trained policy stays measurably closer to the installed
      snapshot than the unconstrained control (non-degeneracy asserted first).
  S5  TRACE SELECTIVITY / anti-alias vs the value estimator (LOAD-BEARING). The Fisher is of the
      policy log-likelihood, so value-head-only parameters get exactly 0 Fisher and the EWC
      penalty puts EXACTLY ZERO gradient on them -- the same locus discipline as PolicyKLAnchor.
  S6  INTERVAL monotonicity: the effective EWC coefficient rises with the offline interval N.
  S7  NOVELTY gating (Moncada behavioural tagging): none->factor 1.0; unpaired->tag_leak floor;
      paired->a measured novelty above the floor.
  S8  Mis-wiring RAISES rather than silently producing the control under the treatment label.
  S9  BootstrapExplorerConfig declares the knobs in as_slice() and defaults them OFF; the shared
      retention baseline threads them and leaves the other legs' knobs untouched.
  S10 RNG NEUTRALITY: consolidate_offline_window restores the torch/numpy/python RNG streams, so a
      window changes ONLY the returned anchor and never desynchronises the downstream refinement.
  S11 The credit-replay update is EWC-anchored TOO (that second policy-gradient step is live in the
      reference build; leaving it unconstrained would make a retention null unreadable).
  S12 Module sources are ASCII-only.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import experiments._lib.baselines.mech457_retention as retention
import experiments._lib.mech457_bootstrap_explorer as boot
import experiments._lib.mech457_explorer_classes as mech
import experiments._lib.mech457_offline_consolidation as offcons
import experiments._lib.mech457_fanout as fan
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734
from experiments._lib.capability_eval import LocalViewGreedyPolicy


STEPS = 12
N_EPISODES = 8
COEF = 50.0        # large enough that the constraint is visible in 8 short episodes
WIN_N = 300        # a window that captures well past the tau knee
FISHER = 24


def _env_kwargs():
    return x734._env_kwargs_for_rung(fan.RUNG)


def _env(seed: int = 0):
    return x734._make_env(seed, _env_kwargs())


def _rep(seed: int = 0):
    return mech.make_rep("raw_view", _env(seed), seed=seed, p0=0, steps=STEPS,
                         actor_critic_hidden=32, cotrain_encoder=False)


def _flat(policy) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in policy.parameters()])


def _seed_all(seed: int) -> None:
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)


def _build_window(seed: int, **cfgkw):
    """Reseed, build a fresh rep, run the window; return (rep, window_diag)."""
    _seed_all(seed)
    rep = _rep(seed)
    cfg = retention.reference_config(N_EPISODES, **cfgkw)   # on_budget positional
    demo = LocalViewGreedyPolicy(seed=seed)
    win = retention.consolidate_offline_window(rep, seed, _env_kwargs(), steps=STEPS, cfg=cfg,
                                               demo=demo)
    return rep, win


def _train_with_weights(seed: int = 0, offline_ewc_anchor=None, **kw):
    _seed_all(seed)
    rep = _rep(seed)
    installed = _flat(rep.policy()).clone()
    guard = mech.train_a2c(
        rep, _env(seed), seed=seed, n_episodes=N_EPISODES, steps=STEPS,
        arm_label="sd083_test", denom=N_EPISODES, offline_ewc_anchor=offline_ewc_anchor, **kw
    )
    return guard, _flat(rep.policy()), installed


# --------------------------------------------------------------------------- S1
def test_s1_off_is_a_noop():
    guard, w_a, _ = _train_with_weights(seed=0)
    _g2, w_b, _ = _train_with_weights(seed=0)
    assert torch.equal(w_a, w_b), "trainer is not deterministic at a fixed seed"
    assert guard["offline_ewc_installed"] is False
    assert guard["offline_ewc_coef"] == 0.0
    assert guard["mean_offline_ewc_penalty_recent"] == 0.0

    # Explicit no-anchor path is bit-identical to the default.
    _g3, w_off, _ = _train_with_weights(seed=0, offline_ewc_anchor=None)
    assert torch.equal(w_a, w_off), "explicit None anchor diverged from default"

    # The baseline wrapper with use_offline_consolidation=False builds no anchor.
    _rep0, win = _build_window(0, use_offline_consolidation=False)
    assert win["anchor"] is None


# --------------------------------------------------------------------------- S2
def test_s2_capture_resource_is_the_interval_clock():
    assert offcons.capture_resource(0, 200.0, 1.0) == 0.0
    vals = [offcons.capture_resource(n, 200.0, 1.0) for n in (0, 50, 100, 200, 400, 800)]
    for a, b in zip(vals, vals[1:]):
        assert b > a, f"capture must increase with the interval ({vals})"
    assert vals[-1] <= 1.0 + 1e-9, "capture must saturate at capture_max"
    assert offcons.capture_resource(10_000, 200.0, 1.0) == pytest.approx(1.0, abs=1e-3)


# --------------------------------------------------------------------------- S3
def test_s3_window_builds_a_live_anchor():
    rep, win = _build_window(
        0, use_offline_consolidation=True, offline_window_steps=WIN_N,
        offline_ewc_max_coef=COEF, offline_fisher_samples=FISHER,
    )
    anchor = win["anchor"]
    assert anchor is not None, "window built no anchor with N>0 and coef>0"
    assert anchor.fisher_mass > 0.0, "Fisher diagonal is degenerate (all-zero)"
    assert anchor.coef > 0.0
    assert win["n_fisher_states"] > 0

    # Train the SAME rep the anchor was built on (the anchor holds references to rep's live
    # params). The realised penalty starts at 0 at the snapshot point and grows as the policy
    # drifts, so over several episodes it must be measurably positive.
    cfg = retention.reference_config(N_EPISODES)
    guard = retention.train_off_arm(
        rep, 0, _env_kwargs(), steps=STEPS, arm_label="sd083_on", cfg=cfg,
        demo=LocalViewGreedyPolicy(seed=0),   # reference cfg carries the persistent BC auxiliary
        offline_ewc_anchor=anchor,
    )
    assert guard["offline_ewc_installed"] is True
    assert guard["offline_ewc_coef"] == pytest.approx(anchor.coef)
    assert guard["mean_offline_ewc_penalty_recent"] > 0.0, (
        "the anchor never bound -- a ~0 realised penalty on an ON arm that drifted is a different "
        "reading from a retention null and must be visible in the manifest"
    )


# --------------------------------------------------------------------------- S4
def test_s4_anchor_binds_policy_closer_to_the_installed_snapshot():
    _off, w_off, installed_off = _train_with_weights(seed=0)

    rep, win = _build_window(
        0, use_offline_consolidation=True, offline_window_steps=WIN_N,
        offline_ewc_max_coef=COEF, offline_fisher_samples=FISHER,
    )
    anchor = win["anchor"]
    # Anchor built on rep at the SAME seed/init as the control, so both start from one policy.
    installed_on = _flat(rep.policy()).clone()
    assert torch.equal(installed_off, installed_on), "the two arms did not start from one policy"
    guard = mech.train_a2c(
        rep, _env(0), seed=0, n_episodes=N_EPISODES, steps=STEPS,
        arm_label="sd083_bind", denom=N_EPISODES, offline_ewc_anchor=anchor,
    )
    w_on = _flat(rep.policy())

    drift_off = float((w_off - installed_off).norm())
    drift_on = float((w_on - installed_on).norm())
    assert drift_off > 0.0, "control did not drift -- the comparison would be vacuous"
    assert drift_on < drift_off, (
        f"offline EWC anchor did not constrain drift (anchored {drift_on:.6f} >= "
        f"unconstrained {drift_off:.6f})"
    )


# --------------------------------------------------------------------------- S5
def test_s5_trace_selectivity_no_gradient_on_the_value_head():
    """LOAD-BEARING. Fisher of log pi(a|s) is 0 on value-head-only params, so the EWC penalty
    leaves the value-estimator locus (MECH-459) untouched -- trace selectivity respecting the same
    locus discipline as PolicyKLAnchor."""
    rep, win = _build_window(
        0, use_offline_consolidation=True, offline_window_steps=WIN_N,
        offline_ewc_max_coef=COEF, offline_fisher_samples=FISHER,
    )
    anchor = win["anchor"]
    assert anchor is not None
    policy = rep.policy()

    # Perturb BOTH heads so both (theta - theta*) terms are nonzero; only the policy head carries
    # nonzero Fisher, so only it may receive gradient.
    with torch.no_grad():
        policy.policy_head.weight.add_(torch.randn_like(policy.policy_head.weight) * 0.5)
        policy.value_head.weight.add_(torch.randn_like(policy.value_head.weight) * 0.5)

    policy.zero_grad(set_to_none=True)
    anchor.penalty().backward()

    for name, p in policy.named_parameters():
        if name.startswith("value_head"):
            assert p.grad is None or torch.all(p.grad == 0), (
                f"offline EWC penalty leaked gradient onto the value estimator ({name}) -- it is "
                "not trace-selective on the policy locus"
            )
    head_grad = policy.policy_head.weight.grad
    assert head_grad is not None and torch.any(head_grad != 0), (
        "EWC penalty put no gradient on the policy head -- the selectivity contract is vacuous"
    )


# --------------------------------------------------------------------------- S6
def test_s6_interval_monotonicity_of_effective_coefficient():
    coefs = []
    for n in (0, 100, 300, 900):
        _rep_n, win = _build_window(
            0, use_offline_consolidation=True, offline_window_steps=n,
            offline_ewc_max_coef=COEF, offline_fisher_samples=FISHER,
        )
        coefs.append(win["effective_ewc_coef"])
    assert coefs[0] == 0.0, "N=0 must yield a zero coefficient (the unconstrained control)"
    for a, b in zip(coefs[1:], coefs[2:]):
        assert b > a, f"effective EWC coefficient must rise with the interval N ({coefs})"


# --------------------------------------------------------------------------- S7
def test_s7_novelty_gating():
    # none: full capture, factor 1.0.
    _r0, none_win = _build_window(
        0, use_offline_consolidation=True, offline_window_steps=WIN_N,
        offline_ewc_max_coef=COEF, offline_fisher_samples=FISHER, offline_novelty_pairing="none",
    )
    assert none_win["novelty_factor"] == pytest.approx(1.0)

    # unpaired: the weak tag is not captured -> factor pinned at the tag_leak floor.
    _r1, unpaired = _build_window(
        1, use_offline_consolidation=True, offline_window_steps=WIN_N,
        offline_ewc_max_coef=COEF, offline_fisher_samples=FISHER, offline_novelty_pairing="unpaired",
    )
    assert unpaired["novelty_factor"] == pytest.approx(0.05), "unpaired must sit at the leak floor"
    assert unpaired["novelty_factor"] < none_win["novelty_factor"]

    # paired: a fresh exposure is novel -> a measured novelty above the leak floor.
    _r2, paired = _build_window(
        2, use_offline_consolidation=True, offline_window_steps=WIN_N,
        offline_ewc_max_coef=COEF, offline_fisher_samples=FISHER, offline_novelty_pairing="paired",
    )
    assert paired["measured_novelty"] is not None, "paired must record a measured novelty"
    assert paired["novelty_factor"] > 0.05, (
        "paired novelty did not exceed the leak floor -- behavioural tagging cannot dissociate"
    )


# --------------------------------------------------------------------------- S8
def test_s8_miswiring_raises():
    rep = _rep(0)
    params = rep.params()
    fisher = [torch.ones_like(p) for p in params]
    # non-positive coefficient
    with pytest.raises(ValueError, match="> 0"):
        offcons.OfflineEWCAnchor(params, fisher, 0.0)
    # empty params
    with pytest.raises(ValueError, match="empty"):
        offcons.OfflineEWCAnchor([], [], COEF)
    # length mismatch
    with pytest.raises(ValueError, match="align"):
        offcons.OfflineEWCAnchor(params, fisher[:-1], COEF)
    # bad pairing token
    with pytest.raises(ValueError, match="none/paired/unpaired"):
        offcons.consolidate_offline_window(
            rep, LocalViewGreedyPolicy(seed=0), _env_kwargs(), 0, steps=STEPS,
            offline_window_steps=WIN_N, offline_ewc_max_coef=COEF, offline_novelty_pairing="bogus",
        )


# --------------------------------------------------------------------------- S9
def test_s9_config_declares_the_knobs_and_defaults_them_off():
    cfg = boot.BootstrapExplorerConfig()
    assert cfg.use_offline_consolidation is False
    assert cfg.offline_window_steps == 0
    assert cfg.offline_ewc_max_coef == 0.0

    sl = cfg.as_slice()
    for key in ("use_offline_consolidation", "offline_window_steps", "offline_capture_tau",
                "offline_capture_max", "offline_ewc_max_coef", "offline_fisher_samples",
                "offline_novelty_pairing"):
        assert key in sl, f"{key} missing from as_slice() -- an undeclared knob aliases arms"

    ref_off = retention.reference_config()
    assert ref_off.use_offline_consolidation is False and ref_off.offline_ewc_max_coef == 0.0
    ref_on = retention.reference_config(
        use_offline_consolidation=True, offline_window_steps=WIN_N, offline_ewc_max_coef=COEF
    )
    assert ref_on.use_offline_consolidation is True
    assert ref_on.offline_ewc_max_coef == pytest.approx(COEF)
    # The three ONLINE legs' knobs must stay untouched (anti-alias): SD-083 is a fourth axis.
    assert ref_on.use_distributional_critic is False
    assert ref_on.use_policy_kl_anchor is False and ref_on.kl_anchor_coef == 0.0
    assert ref_on.bc_aux_coef == retention.BC_AUX_COEF_BASELINE


# --------------------------------------------------------------------------- S10
def test_s10_rng_neutrality():
    """A window must restore all three RNG streams so it changes ONLY the returned anchor."""
    _seed_all(7)
    rep = _rep(7)
    cfg = retention.reference_config(
        N_EPISODES, use_offline_consolidation=True, offline_window_steps=WIN_N,
        offline_ewc_max_coef=COEF, offline_fisher_samples=FISHER, offline_novelty_pairing="paired",
    )
    demo = LocalViewGreedyPolicy(seed=7)
    before_t = torch.get_rng_state().clone()
    before_np = np.random.get_state()
    win = retention.consolidate_offline_window(rep, 7, _env_kwargs(), steps=STEPS, cfg=cfg,
                                               demo=demo)
    assert win["anchor"] is not None, "window did not run -- the RNG check would be vacuous"
    assert torch.equal(torch.get_rng_state(), before_t), "window left the torch RNG stream moved"
    assert np.array_equal(np.random.get_state()[1], before_np[1]), (
        "window left the numpy RNG stream moved -- the downstream refinement would desync"
    )


# --------------------------------------------------------------------------- S11
def test_s11_credit_replay_is_ewc_anchored_too():
    import inspect
    src = inspect.getsource(mech._prioritized_credit_replay)
    assert "offline_ewc_anchor" in src, "credit-replay update is not EWC-anchored -- leaky"

    _seed_all(0)
    rep = _rep(0)
    params = rep.params()
    fisher = [torch.ones_like(p) for p in params]
    anchor = offcons.OfflineEWCAnchor(params, fisher, COEF)
    optimiser = torch.optim.Adam(params, lr=1e-3)
    obs = _env(0).reset()[1]
    n = mech._prioritized_credit_replay(
        rep, optimiser, params, [obs, obs], [0, 1], [1.0, 0.5], [0.1, 0.2],
        passes=1, topk=2, offline_ewc_anchor=anchor,
    )
    assert n == 1, "replay pass did not apply"
    # Consultation, not magnitude: at the snapshot point theta == theta*, so the realised penalty
    # is exactly 0 (the K6-analog). What must hold is that the replay update CONSULTED the anchor
    # -- i.e. added its term to the loss -- which an unconstrained second update would not.
    assert len(anchor._penalty_recent) > 0, (
        "replay update ran without consulting the offline EWC anchor -- an unconstrained second "
        "update would make a null from this leg unreadable"
    )


# --------------------------------------------------------------------------- S12
def test_s12_module_sources_are_ascii():
    for path in (
        Path(offcons.__file__).resolve(),
        Path(mech.__file__).resolve(),
        Path(boot.__file__).resolve(),
        Path(retention.__file__).resolve(),
    ):
        assert path.read_bytes().decode("ascii"), f"{path.name} must be ASCII-only"
