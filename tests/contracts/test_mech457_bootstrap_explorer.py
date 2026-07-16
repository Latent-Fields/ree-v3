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
