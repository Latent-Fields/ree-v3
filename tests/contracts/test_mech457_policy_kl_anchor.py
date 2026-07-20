"""Contract tests for mech457_policy_kl_anchor (2026-07-19).

A trust-region / KL penalty pinning the policy to a FROZEN SNAPSHOT of the INSTALLED policy,
unblocking H-retention-consolidation (competence_floor). Motivated by the measured V3-EXQ-780
raw_view reading: post-BC competence 20.933 eroded to 11.667 under unconstrained RL refinement
with 3/3 seeds having taken the install -- an acquired policy with no consolidation pathway.

This is the UPDATE-CONSTRAINT leg of a three-way retention portfolio whose whole value depends
on the three legs being separately readable. Most of the contracts below exist to hold that
separation, not to check arithmetic:

  K1  OFF is a no-op: defaults take no snapshot, add no term, and leave the trained weights
      BIT-IDENTICAL to the pre-change trainer at the same seed.
  K2  The anchor BINDS: with the anchor ON the trained policy stays measurably closer to the
      installed snapshot than the unconstrained control does. Non-degeneracy is asserted first
      (the control must actually drift), so a broken trainer cannot pass this vacuously.
  K3  ANTI-ALIAS vs mech457_distributional_critic (LOAD-BEARING). The KL term puts EXACTLY ZERO
      gradient on the value head. The symmetric mirror of that build's contract C2, which
      asserted its CE loss puts no gradient on the policy head.
  K4  ANTI-ALIAS vs mech457_bc_aux_schedule (LOAD-BEARING). The anchor targets the INSTALLED
      POLICY, never the demonstrator: it functions with bc_demo=None, and its reference logits
      track the snapshot rather than any expert.
  K5  The snapshot is FROZEN: reference parameters are bit-identical before and after training
      while the live policy moves, and no snapshot parameter is in the optimiser's param list.
  K6  KL(pi || pi) == 0 at the snapshot point, so an anchored arm starts unpenalised and the
      penalty is a pure function of DRIFT.
  K7  Mis-wiring RAISES rather than silently producing the control wearing the treatment label:
      switch without weight, weight without switch, non-positive coefficient, absent policy.
  K8  The realised KL is REPORTED (mean_policy_kl_to_anchor_recent), so a manifest can verify
      the anchor bound rather than assuming it; keys are present even when OFF.
  K9  The credit-replay update is anchored TOO. That second policy-gradient step is live in the
      retention reference build (credit_replay=True); leaving it unconstrained would make a null
      from this leg unreadable -- "anchoring does not preserve competence" would be
      indistinguishable from "the unanchored replay update drifted the policy anyway".
  K10 BootstrapExplorerConfig declares both knobs in as_slice() (fingerprint hygiene) and
      defaults them OFF; the shared retention baseline threads them.
  K11 Module sources are ASCII-only (repo runtime-string rule).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import experiments._lib.baselines.mech457_retention as retention
import experiments._lib.mech457_bootstrap_explorer as boot
import experiments._lib.mech457_explorer_classes as mech
import experiments._lib.mech457_fanout as fan
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734


STEPS = 12
N_EPISODES = 8
COEF = 5.0          # large enough that the constraint is visible in 8 short episodes


def _env(seed: int = 0):
    return x734._make_env(seed, x734._env_kwargs_for_rung(fan.RUNG))


def _rep(seed: int = 0):
    return mech.make_rep("raw_view", _env(seed), seed=seed, p0=0, steps=STEPS,
                         actor_critic_hidden=32, cotrain_encoder=False)


def _flat(policy) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in policy.parameters()])


def _train_with_weights(seed: int = 0, **kw):
    """One short raw-view A2C run -> (guard, trained weights, snapshot-at-entry weights).

    Reseeds all three global streams first so two calls at the same seed are comparable; the
    trainer's rollout draws from the global torch stream, so without this the comparisons below
    would be measuring leftover RNG rather than the anchor.
    """
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)
    rep = _rep(seed)
    installed = _flat(rep.policy()).clone()
    guard = mech.train_a2c(
        rep, _env(seed), seed=seed, n_episodes=N_EPISODES, steps=STEPS,
        arm_label="kl_anchor_test", denom=N_EPISODES, **kw
    )
    return guard, _flat(rep.policy()), installed


# --------------------------------------------------------------------------- K1
def test_k1_off_is_a_noop():
    """Defaults must reproduce the pre-change trainer bit-identically.

    Asserts on WEIGHTS rather than guard readouts, following the trajectory-probe precedent: a
    handful of rolling-window means over 8 episodes can coincide across two divergent runs,
    which would make this a vacuous pass.
    """
    guard, w_a, _ = _train_with_weights(seed=0)
    _guard_b, w_b, _ = _train_with_weights(seed=0)
    assert torch.equal(w_a, w_b), "trainer is not deterministic at a fixed seed"
    assert guard["policy_kl_anchor_installed"] is False
    assert guard["policy_kl_anchor_coef"] == 0.0
    assert guard["mean_policy_kl_to_anchor_recent"] == 0.0

    explicit_off, w_off, _ = _train_with_weights(
        seed=0, use_policy_kl_anchor=False, kl_anchor_coef=0.0
    )
    assert torch.equal(w_a, w_off), "explicit OFF diverged from default OFF"


# --------------------------------------------------------------------------- K2
def test_k2_anchor_binds_policy_closer_to_the_installed_snapshot():
    """The anchor must actually CONSTRAIN, not merely be wired in.

    Non-degeneracy is asserted BEFORE the comparison: if the unconstrained control did not drift
    from its install, 'the anchored arm drifted less' would be true of a trainer that does
    nothing at all.
    """
    _off, w_off, installed_off = _train_with_weights(seed=0)
    _on, w_on, installed_on = _train_with_weights(
        seed=0, use_policy_kl_anchor=True, kl_anchor_coef=COEF
    )
    assert torch.equal(installed_off, installed_on), "the two arms did not start from one policy"

    drift_off = float((w_off - installed_off).norm())
    drift_on = float((w_on - installed_on).norm())

    assert drift_off > 0.0, "control did not drift -- the comparison would be vacuous"
    assert drift_on < drift_off, (
        "KL anchor did not constrain drift from the installed policy "
        f"(anchored {drift_on:.6f} >= unconstrained {drift_off:.6f})"
    )


# --------------------------------------------------------------------------- K3
def test_k3_antialias_kl_term_puts_no_gradient_on_the_value_head():
    """LOAD-BEARING anti-alias vs mech457_distributional_critic (H-retention-critic).

    The retention portfolio only reads if each leg moves ONE locus. This leg is the update
    CONSTRAINT; the value ESTIMATOR belongs to the sibling build. The exact mirror of that
    build's contract C2 (its CE loss puts no gradient on the policy head).
    """
    torch.manual_seed(0)
    rep = _rep(0)
    policy = rep.policy()
    anchor = mech.PolicyKLAnchor(policy, COEF)

    # DRIFT THE LIVE POLICY FIRST. At the snapshot point pi == pi_ref, so KL sits at its exact
    # minimum and its gradient is identically zero EVERYWHERE -- including on the policy head.
    # Probing there would make the value-head assertion below pass vacuously (as it did on the
    # first run of this contract). Perturbing is what gives the anti-alias check teeth: it is
    # the only regime in which a value-head leak could actually appear.
    with torch.no_grad():
        policy.policy_head.weight.add_(torch.randn_like(policy.policy_head.weight) * 0.5)

    state = rep.encode(_env(0).reset()[1])
    step = rep.step(state, deterministic=False)
    penalty = anchor.penalty(
        step.logits.reshape(1, -1), anchor.ref_logits(rep, state).reshape(1, -1)
    )
    policy.zero_grad(set_to_none=True)
    penalty.backward()

    for name, p in policy.named_parameters():
        if name.startswith("value_head"):
            assert p.grad is None or torch.all(p.grad == 0), (
                f"KL anchor leaked gradient onto the value estimator ({name}) -- this aliases "
                "the update-constraint leg with the value-estimator leg"
            )

    # Non-degeneracy: the term must reach the policy head, else 'no value gradient' is trivial.
    head_grad = policy.policy_head.weight.grad
    assert head_grad is not None and torch.any(head_grad != 0), (
        "KL term put no gradient on the policy head -- the contract above is vacuous"
    )


# --------------------------------------------------------------------------- K4
def test_k4_antialias_anchor_targets_the_installed_policy_not_the_demonstrator():
    """LOAD-BEARING anti-alias vs mech457_bc_aux_schedule (H-retention-auxiliary-decay).

    Anchoring to the demonstrator would make this leg a restatement of the auxiliary leg. Two
    independent witnesses: the anchor runs with NO demonstrator at all, and its reference logits
    are reproduced exactly by a copy of the INSTALLED POLICY.
    """
    guard, _w, _installed = _train_with_weights(
        seed=0, use_policy_kl_anchor=True, kl_anchor_coef=COEF, bc_demo=None, bc_aux_coef=0.0
    )
    assert guard["policy_kl_anchor_installed"] is True

    torch.manual_seed(0)
    rep = _rep(0)
    policy = rep.policy()
    anchor = mech.PolicyKLAnchor(policy, COEF)
    state = rep.encode(_env(0).reset()[1])

    with torch.no_grad():
        expected, _v, _phi, _psi = policy(rep.z_detached(state))
    assert torch.equal(anchor.ref_logits(rep, state), expected.reshape(-1)), (
        "reference logits are not the installed policy's own logits"
    )


# --------------------------------------------------------------------------- K5
def test_k5_snapshot_is_frozen_and_outside_the_optimiser():
    torch.manual_seed(0)
    rep = _rep(0)
    anchor = mech.PolicyKLAnchor(rep.policy(), COEF)
    ref_before = _flat(anchor.ref).clone()

    assert all(not p.requires_grad for p in anchor.ref.parameters()), "snapshot is trainable"

    trainable = {id(p) for p in rep.params()}
    assert not any(id(p) in trainable for p in anchor.ref.parameters()), (
        "snapshot parameters are in the optimiser's param list -- the anchor would chase itself"
    )

    guard = mech.train_a2c(
        rep, _env(0), seed=0, n_episodes=N_EPISODES, steps=STEPS,
        arm_label="frozen_test", denom=N_EPISODES,
        use_policy_kl_anchor=True, kl_anchor_coef=COEF,
    )
    assert guard["policy_kl_anchor_installed"] is True
    assert torch.equal(_flat(anchor.ref), ref_before), "snapshot drifted during training"


# --------------------------------------------------------------------------- K6
def test_k6_kl_is_zero_at_the_snapshot_point():
    """An anchored arm starts unpenalised: the penalty is a pure function of DRIFT."""
    torch.manual_seed(0)
    rep = _rep(0)
    anchor = mech.PolicyKLAnchor(rep.policy(), COEF)
    state = rep.encode(_env(0).reset()[1])
    step = rep.step(state, deterministic=False)
    kl = anchor.kl(step.logits.reshape(1, -1), anchor.ref_logits(rep, state).reshape(1, -1))
    assert float(kl.detach().abs()) < 1e-6, f"KL(pi || pi) should be 0, got {float(kl.detach())}"

    # The penalty is therefore a pure function of DRIFT: it is not merely small at the snapshot
    # point but stationary there, so an anchored arm is unpenalised until it starts to move.
    # (This is why contract K3 must perturb the policy before probing gradients.)
    kl.backward()
    head_grad = rep.policy().policy_head.weight.grad
    assert head_grad is not None and torch.all(head_grad.abs() < 1e-6), (
        "KL gradient is nonzero at zero drift -- the anchor would penalise a policy that has "
        "not moved, biasing the arm rather than constraining it"
    )


# --------------------------------------------------------------------------- K7
def test_k7_miswiring_raises():
    """A mis-wired anchor must fail LOUDLY, never yield the control under the treatment label."""
    with pytest.raises(ValueError, match="together"):
        _train_with_weights(seed=0, use_policy_kl_anchor=True, kl_anchor_coef=0.0)
    with pytest.raises(ValueError, match="together"):
        _train_with_weights(seed=0, use_policy_kl_anchor=False, kl_anchor_coef=COEF)
    with pytest.raises(ValueError, match="must be > 0"):
        mech.PolicyKLAnchor(_rep(0).policy(), 0.0)
    with pytest.raises(ValueError, match="returned None"):
        mech.PolicyKLAnchor(None, COEF)


# --------------------------------------------------------------------------- K8
def test_k8_realised_kl_is_reported():
    """The measured KL is what lets a manifest verify the anchor BOUND rather than assume it."""
    off, _w, _i = _train_with_weights(seed=0)
    on, _w2, _i2 = _train_with_weights(seed=0, use_policy_kl_anchor=True, kl_anchor_coef=COEF)

    for key in (
        "policy_kl_anchor_installed", "policy_kl_anchor_coef", "mean_policy_kl_to_anchor_recent"
    ):
        assert key in off and key in on, f"{key} must be emitted unconditionally"

    assert on["policy_kl_anchor_coef"] == pytest.approx(COEF)
    assert on["mean_policy_kl_to_anchor_recent"] > 0.0, (
        "realised KL is ~0 on an anchored arm -- the policy never left the snapshot, which is a "
        "different reading from a retention null and must be visible in the manifest"
    )


# --------------------------------------------------------------------------- K9
def test_k9_credit_replay_update_is_anchored_too():
    """The replay step is a SECOND policy-gradient update, live in the reference build.

    Asserted structurally (the anchor reaches the replay path and records KL from it) rather
    than by end-to-end drift, because replay fires only on reward-bearing episodes and a short
    test run may see none -- which would make a drift-based assertion silently vacuous.
    """
    import inspect
    src = inspect.getsource(mech._prioritized_credit_replay)
    assert "kl_anchor" in src, "credit-replay update is not anchored -- the constraint is leaky"

    torch.manual_seed(0)
    rep = _rep(0)
    anchor = mech.PolicyKLAnchor(rep.policy(), COEF)
    optimiser = torch.optim.Adam(rep.params(), lr=1e-3)
    obs = _env(0).reset()[1]
    n = mech._prioritized_credit_replay(
        rep, optimiser, rep.params(), [obs, obs], [0, 1], [1.0, 0.5], [0.1, 0.2],
        passes=1, topk=2, kl_anchor=anchor,
    )
    assert n == 1, "replay pass did not apply"
    assert len(anchor.kl_recent) > 0, (
        "replay update ran without consulting the anchor -- an unconstrained second update "
        "would make a null from this leg unreadable"
    )


# --------------------------------------------------------------------------- K10
def test_k10_config_declares_the_knobs_and_defaults_them_off():
    cfg = boot.BootstrapExplorerConfig()
    assert cfg.use_policy_kl_anchor is False
    assert cfg.kl_anchor_coef == 0.0

    sl = cfg.as_slice()
    for key in ("use_policy_kl_anchor", "kl_anchor_coef"):
        assert key in sl, (
            f"{key} missing from as_slice() -- an undeclared knob lets two materially "
            "different arms share one arm fingerprint"
        )
    assert sl["use_policy_kl_anchor"] is False
    assert sl["kl_anchor_coef"] == 0.0

    # The shared retention baseline threads them, and its default stays the OFF control.
    ref_off = retention.reference_config()
    assert ref_off.use_policy_kl_anchor is False and ref_off.kl_anchor_coef == 0.0
    ref_on = retention.reference_config(use_policy_kl_anchor=True, kl_anchor_coef=COEF)
    assert ref_on.use_policy_kl_anchor is True
    assert ref_on.kl_anchor_coef == pytest.approx(COEF)
    # The OTHER two legs' knobs must remain untouched by this one (three-way anti-alias).
    assert ref_on.use_distributional_critic is False
    assert ref_on.bc_aux_coef == retention.BC_AUX_COEF_BASELINE
    assert ref_on.bc_aux_coef_end is None


# --------------------------------------------------------------------------- K11
def test_k11_module_sources_are_ascii():
    for name in ("mech457_explorer_classes.py", "mech457_bootstrap_explorer.py"):
        path = Path(mech.__file__).resolve().parent / name
        assert path.read_bytes().decode("ascii"), f"{name} must be ASCII-only"
    base = Path(retention.__file__).resolve()
    assert base.read_bytes().decode("ascii"), "mech457_retention.py must be ASCII-only"
