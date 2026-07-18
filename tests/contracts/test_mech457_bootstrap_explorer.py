"""Contract tests for the MECH-457 competence bootstrap explorer (2026-07-16).

The composed floor->competent build (experiments/_lib/mech457_bootstrap_explorer): RND
success-independent drive + first-class actor-critic + prioritized backward credit-replay
converter + a NEW developmental intrinsic-coefficient/entropy anneal + increased budget.
See the module docstring and failure_autopsy_MECH-457-fanout-755_2026-07-15 for the diagnosis.

Contracts:
  C1  linear_anneal is correct: constant when frac<=0 or start==end; linear from start to end
      over the first `frac` of episodes; holds at v_end after the cutoff; monotone.
  C2  make_off_config is the no-op 751 RND plateau: constant coefficient (schedule == coef for
      every episode), no credit-replay, plateau budget, RND on.
  C3  make_on_config is the composed bootstrap: coefficient anneals DOWN (start>end), entropy
      anneals down, credit-replay on, budget strictly larger than the plateau.
  C4  train_a2c schedule hooks are no-op by default and mutually exclusive with mode_gate
      (schedule anneal vs the utility-gate 755 refuted).
  C5  Activation smoke: the full composition trains end-to-end on a real raw 5x5 env for a few
      episodes with credit-replay + anneal + RND all live, returning finite guard metrics.
  C6  The module source is ASCII-only (repo runtime-string rule).
"""

from pathlib import Path

import pytest

import experiments._lib.mech457_bootstrap_explorer as boot
import experiments._lib.mech457_explorer_classes as mech
import experiments._lib.mech457_fanout as fan
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734


# --------------------------------------------------------------------------- C1
def test_c1_linear_anneal_constant_and_ramp():
    # frac <= 0 -> constant v_start regardless of ep.
    assert boot.linear_anneal(1.0, 0.0, 0.0, 0, 100) == 1.0
    assert boot.linear_anneal(1.0, 0.0, 0.0, 99, 100) == 1.0
    # start == end -> constant.
    for ep in (0, 50, 99):
        assert boot.linear_anneal(0.7, 0.7, 0.6, ep, 100) == pytest.approx(0.7)
    # Linear ramp over the first frac of episodes, then hold at v_end.
    n = 100
    frac = 0.5           # cutoff = 50 episodes
    assert boot.linear_anneal(1.0, 0.0, frac, 0, n) == pytest.approx(1.0)
    assert boot.linear_anneal(1.0, 0.0, frac, 25, n) == pytest.approx(0.5)   # halfway
    assert boot.linear_anneal(1.0, 0.0, frac, 50, n) == pytest.approx(0.0)   # at cutoff
    assert boot.linear_anneal(1.0, 0.0, frac, 90, n) == pytest.approx(0.0)   # held past cutoff
    # Monotone decreasing across the ramp.
    vals = [boot.linear_anneal(1.0, 0.05, 0.6, ep, n) for ep in range(n)]
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(n - 1))
    assert vals[0] == pytest.approx(1.0)
    assert vals[-1] == pytest.approx(0.05)


# --------------------------------------------------------------------------- C2
def test_c2_off_config_is_noop_plateau():
    cfg = boot.make_off_config()
    assert cfg.use_rnd is True
    assert cfg.credit_replay is False
    assert cfg.n_episodes == fan.RL_EPISODES
    # Constant coefficient across all episodes (no anneal) -> reproduces the 751 plateau.
    coef = [
        boot.linear_anneal(cfg.intrinsic_coef_start, cfg.intrinsic_coef_end,
                           cfg.anneal_fraction, ep, cfg.n_episodes)
        for ep in (0, cfg.n_episodes // 2, cfg.n_episodes - 1)
    ]
    assert all(c == pytest.approx(mech.INTRINSIC_COEF) for c in coef)
    # Budget override still yields a constant-coef arm.
    cfg2 = boot.make_off_config(n_episodes=250)
    assert cfg2.n_episodes == 250 and cfg2.credit_replay is False


# --------------------------------------------------------------------------- C3
def test_c3_on_config_is_composed_bootstrap():
    cfg = boot.make_on_config()
    assert cfg.use_rnd is True
    assert cfg.credit_replay is True
    # Coefficient and entropy anneal DOWN (explore -> exploit consolidation).
    assert cfg.intrinsic_coef_start > cfg.intrinsic_coef_end
    assert cfg.entropy_beta_start >= cfg.entropy_beta_end
    assert 0.0 < cfg.anneal_fraction <= 1.0
    # Increased budget (capacity to convert) strictly above the plateau budget.
    assert cfg.n_episodes > fan.RL_EPISODES
    # The end coefficient shifts weight onto the extrinsic forage reward without zeroing coverage.
    assert 0.0 <= cfg.intrinsic_coef_end < cfg.intrinsic_coef_start


# --------------------------------------------------------------------------- C4
def test_c4_train_a2c_schedule_hooks_default_and_exclusive():
    import inspect
    sig = inspect.signature(mech.train_a2c)
    assert "coef_schedule" in sig.parameters
    assert "entropy_schedule" in sig.parameters
    assert sig.parameters["coef_schedule"].default is None
    assert sig.parameters["entropy_schedule"].default is None
    # mode_gate + schedule is rejected (schedule anneal vs utility-gate anneal).
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    env = x734._make_env(42, env_kwargs)
    rep = mech.make_rep("raw_view", env, seed=42, p0=0, steps=8)
    with pytest.raises(ValueError):
        mech.train_a2c(
            rep, env, seed=42, n_episodes=1, steps=4, arm_label="x", denom=1,
            mode_gate=mech.ModeGate(), coef_schedule=(lambda ep, n: 1.0),
        )


# --------------------------------------------------------------------------- C5
def test_c5_bootstrap_explorer_activation_smoke():
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    env = x734._make_env(43, env_kwargs)
    rep = mech.make_rep("raw_view", env, seed=43, p0=0, steps=15)
    cfg = boot.BootstrapExplorerConfig(
        use_rnd=True,
        intrinsic_coef_start=1.0, intrinsic_coef_end=0.05, anneal_fraction=0.6,
        entropy_beta_start=0.10, entropy_beta_end=0.03,
        credit_replay=True, n_episodes=4,
    )
    guard = boot.train_bootstrap_explorer(
        rep, env, seed=43, steps=15, arm_label="smoke_on", cfg=cfg,
    )
    assert set(guard) >= {
        "mean_train_forage_recent", "mean_intrinsic_reward_recent",
        "n_return_episodes", "n_credit_replay_passes",
    }
    for k in ("mean_train_forage_recent", "mean_intrinsic_reward_recent"):
        v = float(guard[k])
        assert v == v and abs(v) < 1e6   # finite (not NaN/inf)
    assert guard["n_credit_replay_passes"] >= 0


# --------------------------------------------------------------------------- C6
def test_c6_module_source_is_ascii():
    src = Path(boot.__file__).read_text(encoding="utf-8")
    non_ascii = [(i, ch) for i, ch in enumerate(src) if ord(ch) > 127]
    assert not non_ascii, f"non-ASCII characters in bootstrap-explorer source: {non_ascii[:5]}"


# ===========================================================================
# MECH-457 capacity-side amend (2026-07-16, failure_autopsy_V3-EXQ-765). Three knobs on ONE
# build: (a) capacity (hidden + budget), (b) reliability (warm-start + credit passes/topk),
# (c) integration (z_world detached). All OFF-preserving no-op defaults.
# ===========================================================================


# --------------------------------------------------------------------------- C7
def test_c7_warm_then_anneal_holds_then_anneals_and_reduces_to_linear():
    n = 100
    # warm_frac == 0 reduces EXACTLY to linear_anneal(v_start, v_end, anneal_frac, ...).
    for ep in (0, 10, 25, 50, 75, 99):
        assert boot.warm_then_anneal(1.0, 0.0, 0.0, 0.5, ep, n) == pytest.approx(
            boot.linear_anneal(1.0, 0.0, 0.5, ep, n)
        )
    # anneal_frac <= 0 or start == end -> constant v_start (no-op OFF path).
    for ep in (0, 50, 99):
        assert boot.warm_then_anneal(1.0, 0.05, 0.2, 0.0, ep, n) == pytest.approx(1.0)
        assert boot.warm_then_anneal(0.7, 0.7, 0.2, 0.6, ep, n) == pytest.approx(0.7)
    # Warm-start: hold v_start over the first warm_frac, then anneal over the next anneal_frac.
    warm, anneal = 0.2, 0.5           # hold [0,20), anneal [20,70), hold v_end after.
    assert boot.warm_then_anneal(1.0, 0.0, warm, anneal, 0, n) == pytest.approx(1.0)
    assert boot.warm_then_anneal(1.0, 0.0, warm, anneal, 19, n) == pytest.approx(1.0)  # warm hold
    assert boot.warm_then_anneal(1.0, 0.0, warm, anneal, 20, n) == pytest.approx(1.0)  # anneal t=0
    assert boot.warm_then_anneal(1.0, 0.0, warm, anneal, 45, n) == pytest.approx(0.5)  # halfway
    assert boot.warm_then_anneal(1.0, 0.0, warm, anneal, 70, n) == pytest.approx(0.0)  # anneal end
    assert boot.warm_then_anneal(1.0, 0.0, warm, anneal, 99, n) == pytest.approx(0.0)  # held
    # Monotone non-increasing across the schedule.
    vals = [boot.warm_then_anneal(1.0, 0.05, warm, anneal, ep, n) for ep in range(n)]
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(n - 1))


# --------------------------------------------------------------------------- C8
def test_c8_off_config_is_capacity_neutral_bit_identical():
    """OFF must reproduce the 751/765 plateau arm byte-identical: the three capacity knobs are
    all no-op (128-wide, z_world cotrain, no warm-start, default credit passes/topk)."""
    cfg = boot.make_off_config()
    assert cfg.actor_critic_hidden == fan.ACTOR_CRITIC_HIDDEN == 128
    assert cfg.cotrain_encoder is True          # z_world co-shaped (the 5.22 plateau reference)
    assert cfg.warm_start_fraction == 0.0
    assert cfg.credit_replay_passes == mech.CREDIT_REPLAY_PASSES
    assert cfg.credit_topk == mech.CREDIT_TOPK
    # With anneal_fraction 0, warm-start cannot perturb the (constant) coefficient.
    for ep in (0, cfg.n_episodes // 2, cfg.n_episodes - 1):
        assert boot.warm_then_anneal(
            cfg.intrinsic_coef_start, cfg.intrinsic_coef_end,
            cfg.warm_start_fraction, cfg.anneal_fraction, ep, cfg.n_episodes,
        ) == pytest.approx(mech.INTRINSIC_COEF)


# --------------------------------------------------------------------------- C9
def test_c9_on_config_carries_all_three_capacity_knobs():
    cfg = boot.make_on_config()
    # (a) capacity: wider trunk + larger budget than the plateau.
    assert cfg.actor_critic_hidden > fan.ACTOR_CRITIC_HIDDEN
    assert cfg.n_episodes > fan.RL_EPISODES
    assert boot.ON_BUDGET_MULTIPLIER >= 4
    # (b) reliability: a full-explore warm-start + raised credit passes/topk.
    assert cfg.warm_start_fraction > 0.0
    assert cfg.credit_replay is True
    assert cfg.credit_replay_passes > mech.CREDIT_REPLAY_PASSES
    assert cfg.credit_topk > mech.CREDIT_TOPK
    # (c) integration: z_world path DETACHED (frozen prediction-trained encoder, Stooke 2021).
    assert cfg.cotrain_encoder is False
    # as_slice declares every new field (arm_fingerprint / manifest).
    sl = cfg.as_slice()
    for k in ("warm_start_fraction", "credit_replay_passes", "credit_topk",
              "actor_critic_hidden", "cotrain_encoder"):
        assert k in sl


# --------------------------------------------------------------------------- C10
def test_c10_make_rep_capacity_and_detach_wiring():
    """make_rep threads capacity (raw trunk width) + the z_world co-shape-vs-frozen mode; the
    detached z_world rep excludes the encoder params from its optimizer group."""
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    env = x734._make_env(42, env_kwargs)
    # Raw view: wider trunk actually enlarges the actor-critic first layer.
    r_small = mech.make_rep("raw_view", env, seed=42, p0=0, steps=8, actor_critic_hidden=128)
    r_big = mech.make_rep("raw_view", env, seed=42, p0=0, steps=8, actor_critic_hidden=256)
    n_small = sum(p.numel() for p in r_small.params())
    n_big = sum(p.numel() for p in r_big.params())
    assert n_big > n_small
    # z_world: detached mode drops the encoder params; cotrain keeps them.
    z_cotrain = mech.make_rep("z_world", env, seed=42, p0=0, steps=6,
                              cotrain_encoder=True, actor_critic_hidden=128)
    z_detach = mech.make_rep("z_world", env, seed=42, p0=0, steps=6,
                             cotrain_encoder=False, actor_critic_hidden=128)
    n_cotrain = sum(p.numel() for p in z_cotrain.params())
    n_detach = sum(p.numel() for p in z_detach.params())
    assert n_detach < n_cotrain          # encoder params excluded under detach
    assert z_detach._cotrain is False and z_cotrain._cotrain is True


# --------------------------------------------------------------------------- C11
def test_c11_capacity_on_config_trains_end_to_end_raw():
    """The full capacity-amend ON composition (warm-start + anneal + raised credit passes/topk +
    wider trunk + budget) trains end-to-end on a real raw 5x5 env, returning finite metrics."""
    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    env = x734._make_env(44, env_kwargs)
    cfg = boot.make_on_config()
    cfg.n_episodes = 5                   # tiny budget for the smoke (keep the composition intact)
    rep = mech.make_rep("raw_view", env, seed=44, p0=0, steps=15,
                        actor_critic_hidden=cfg.actor_critic_hidden,
                        cotrain_encoder=cfg.cotrain_encoder)
    guard = boot.train_bootstrap_explorer(rep, env, seed=44, steps=15, arm_label="cap_on", cfg=cfg)
    for k in ("mean_train_forage_recent", "mean_intrinsic_reward_recent"):
        v = float(guard[k])
        assert v == v and abs(v) < 1e6
    assert guard["n_credit_replay_passes"] >= 0


# ===========================================================================
# mech457_bc_aux_schedule (2026-07-18, MECH-457 retention portfolio). Makes the BC-auxiliary
# PERSISTENCE sweepable (constant / annealed / off) so H-retention-auxiliary-decay can read a
# competence half-life. No-op default: bc_aux_schedule=None -> constant bc_aux_coef.
# ===========================================================================


# --------------------------------------------------------------------------- C12
def test_c12_bc_aux_schedule_defaults_are_noop():
    """OFF path: no schedule on train_a2c, no anneal fields on the config, declared in as_slice."""
    import inspect
    assert inspect.signature(mech.train_a2c).parameters["bc_aux_schedule"].default is None
    cfg = boot.make_off_config()
    assert cfg.bc_aux_coef == 0.0
    assert cfg.bc_aux_coef_end is None
    assert cfg.bc_aux_anneal_fraction == 0.0
    s = cfg.as_slice()
    # Declared in the config_slice -- a varyable knob absent from the slice would let two
    # materially different arms share one fingerprint.
    assert s["bc_aux_coef_end"] is None
    assert s["bc_aux_anneal_fraction"] == 0.0


# --------------------------------------------------------------------------- C13
def test_c13_bc_aux_schedule_three_cells_are_distinct():
    """constant / annealed / off must yield distinguishable realised coefficient trajectories."""
    n = 100
    constant = [boot.linear_anneal(0.5, 0.5, 0.0, ep, n) for ep in range(n)]
    annealed = [boot.linear_anneal(0.5, 0.0, 0.5, ep, n) for ep in range(n)]
    assert constant[0] == constant[-1] == 0.5
    assert annealed[0] == 0.5 and annealed[-1] == 0.0
    assert annealed[0] > annealed[n // 4] > annealed[n // 2 - 1]
    assert annealed != constant


# --------------------------------------------------------------------------- C14
def test_c14_bc_aux_schedule_equals_float_path_when_constant():
    """A schedule returning c must be BIT-IDENTICAL to passing the float c.

    This is the regression guard for the GUARD fix: the auxiliary's `if` reads the effective
    per-episode coefficient, not the constant. An annealed cell passes bc_aux_coef=0.0 with a
    nonzero schedule -- a guard reading the constant would silently produce an OFF arm labelled
    ANNEALED, i.e. a degenerate arm read as a scientific verdict.
    """
    import random
    import numpy as np
    import torch
    from experiments._lib.capability_eval import LocalViewGreedyPolicy

    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)

    def train(**kw):
        torch.manual_seed(7); np.random.seed(7); random.seed(7)
        env = x734._make_env(7, env_kwargs)
        rep = mech.make_rep("raw_view", env, seed=7, p0=0, steps=12)
        mech.train_a2c(rep, env, seed=7, n_episodes=4, steps=12, arm_label="c14", denom=4,
                       bc_demo=LocalViewGreedyPolicy(seed=7), **kw)
        return torch.cat([p.detach().reshape(-1) for p in rep.params()])

    w_float = train(bc_aux_coef=0.3)
    w_sched = train(bc_aux_coef=0.0, bc_aux_schedule=lambda ep, n: 0.3)
    assert float((w_float - w_sched).norm()) == 0.0


# --------------------------------------------------------------------------- C15
def test_c15_bc_aux_schedule_reports_realised_trajectory():
    """The guard dict must report the realised first/last coefficient so a manifest can VERIFY
    the schedule moved rather than assuming it did."""
    from experiments._lib.capability_eval import LocalViewGreedyPolicy

    env_kwargs = x734._env_kwargs_for_rung(fan.RUNG)
    env = x734._make_env(45, env_kwargs)
    rep = mech.make_rep("raw_view", env, seed=45, p0=0, steps=12)
    cfg = boot.BootstrapExplorerConfig(
        use_rnd=True, n_episodes=6,
        bc_aux_coef=0.5, bc_aux_coef_end=0.0, bc_aux_anneal_fraction=0.5,
    )
    guard = boot.train_bootstrap_explorer(
        rep, env, seed=45, steps=12, arm_label="c15", cfg=cfg,
        bc_demo=LocalViewGreedyPolicy(seed=45),
    )
    assert guard["bc_aux_coef_first"] == 0.5
    assert guard["bc_aux_coef_last"] == 0.0


# --------------------------------------------------------------------------- C16
def test_c16_bc_aux_nonzero_requires_demonstrator_including_ramp_up():
    """Precondition reads max(start, end) -- a ramp-UP cell would slip past a start-only check --
    and fires BEFORE any module is constructed."""
    for kw in (dict(bc_aux_coef=0.5), dict(bc_aux_coef=0.0, bc_aux_coef_end=0.4)):
        cfg = boot.BootstrapExplorerConfig(use_rnd=True, n_episodes=2, **kw)
        with pytest.raises(ValueError, match="bc_demo"):
            # rep=None: if the precondition is correctly ordered first, it raises ValueError
            # before ever touching rep.feature_dim.
            boot.train_bootstrap_explorer(
                None, None, seed=0, steps=2, arm_label="c16", cfg=cfg, bc_demo=None,
            )


# --------------------------------------------------------------------------- C17
def test_c17_explorer_classes_source_is_ascii():
    src = Path(mech.__file__).read_text(encoding="utf-8")
    non_ascii = [(i, ch) for i, ch in enumerate(src) if ord(ch) > 127]
    assert not non_ascii, f"non-ASCII in explorer-classes source: {non_ascii[:5]}"
