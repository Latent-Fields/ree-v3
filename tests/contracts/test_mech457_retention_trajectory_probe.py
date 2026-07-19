"""Contract tests for mech457_retention_trajectory_probe (2026-07-19).

A non-perturbing mid-training COMPETENCE PROBE for the MECH-457 trainer. The four
competence_floor retention legs must record the post-installation competence TRAJECTORY rather
than terminal competence (mech457_retention_portfolio_2026-07-18 sec 53); terminal-only
measurement is precisely what kept the retention deficit invisible for ten legs, with
V3-EXQ-780 the worked failure. Before this build train_a2c had no observation hook at all, and
the driver-side workaround (chunking the budget and evaluating between segments) was unfaithful
on two counts: the coef/entropy/bc_aux schedules are computed from the LOCAL loop index, and the
Adam optimiser / reward normaliser / novelty counter are constructed inside the call.

Lives in its OWN file rather than extending test_mech457_bootstrap_explorer.py: that file's
C18-C18f belong to the concurrently-developed untrained-encoder guard, and a shared numbering
sequence across two live sessions collides. Matches the test_mech457_distributional_critic.py
precedent.

Contracts:
  T1  Probe defaults are a no-op: no probe params -> empty competence_trajectory, and the guard
      key is present regardless so consumers read one stable key.
  T2  MEASUREMENT NEUTRALITY (LOAD-BEARING). For a fixed seed, probe ON reproduces probe OFF
      bit-identically on every training readout. This is the contract that makes the trajectory
      trustworthy: it is what distinguishes an instrument from an intervention, and it is what
      keeps a future edit from quietly making the probe interact with the experiment.
  T3  Cadence: the trajectory has n_episodes // probe_every entries at exactly the expected
      episode numbers, and episode numbers are 1-indexed.
  T4  A half-wired probe RAISES rather than silently yielding an empty trajectory (either
      direction), and a non-positive cadence RAISES.
  T5  The probe never receives or touches the training env.
  T6  BootstrapExplorerConfig declares the cadence in as_slice() (fingerprint hygiene) and
      defaults it to None; train_bootstrap_explorer enforces both-or-neither.
  T7  Module sources are ASCII-only (repo runtime-string rule).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import experiments._lib.mech457_bootstrap_explorer as boot
import experiments._lib.mech457_explorer_classes as mech
import experiments._lib.mech457_fanout as fan
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734


STEPS = 12
N_EPISODES = 8


def _env(seed: int = 0):
    return x734._make_env(seed, x734._env_kwargs_for_rung(fan.RUNG))


def _rep(seed: int = 0):
    return mech.make_rep("raw_view", _env(seed), seed=seed, p0=0, steps=STEPS,
                         actor_critic_hidden=32, cotrain_encoder=False)


def _train(seed: int = 0, **kw):
    """One short raw-view A2C run. Every stochastic component is reseeded first so two calls
    with the same seed are comparable -- without this the neutrality check (T2) would be
    measuring leftover global RNG rather than the probe."""
    guard, _weights = _train_with_weights(seed, **kw)
    return guard


def _train_with_weights(seed: int = 0, **kw):
    """As _train, but also returns the flattened trained policy parameters.

    The WEIGHTS are the sensitive neutrality detector, not the guard readouts. The guard's
    rolling-window means aggregate 8 episodes into a handful of floats, and two genuinely
    divergent runs can coincide on them -- verified: with the RNG restore deliberately removed,
    a guard-readout comparison still passed while the weights differed. Comparing weights is
    what gives T2 teeth (mutation-checked in both directions).
    """
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)
    rep = _rep(seed)
    guard = mech.train_a2c(
        rep, _env(seed), seed=seed, n_episodes=N_EPISODES, steps=STEPS,
        arm_label="probe_test", denom=N_EPISODES, **kw
    )
    weights = torch.cat([p.detach().reshape(-1) for p in rep.policy().parameters()])
    return guard, weights


# --------------------------------------------------------------------------- T1
def test_t1_probe_defaults_are_a_noop():
    guard = _train(seed=0)
    assert "competence_trajectory" in guard, "guard key must be present even when unprobed"
    assert guard["competence_trajectory"] == []


# --------------------------------------------------------------------------- T2
def test_t2_measurement_neutrality_probe_on_equals_probe_off():
    """LOAD-BEARING. The probe may only ADD outputs, never perturb.

    The probe body is deliberately RNG-HUNGRY -- it burns all three global streams (torch,
    numpy, python random). This is not a contrived worry: the training rollout DOES draw from
    the global torch stream, so an unrestored probe genuinely desynchronises training. Verified
    by mutation in both directions -- with the restore removed the trained weights diverge, with
    it in place they are bit-identical.

    Asserts on WEIGHTS, not on the guard's aggregate means: under the same mutation a
    guard-readout comparison still passed (8 episodes of rolling-window means can coincide
    across two divergent runs), which would have made this contract a vacuous pass.
    """
    import random as _random

    def rng_hungry_probe(ep):
        torch.rand(4)
        np.random.rand(4)
        _random.random()
        return {"foraging_competence": 1.0}

    off, w_off = _train_with_weights(seed=0)
    on, w_on = _train_with_weights(seed=0, probe_every=2, probe_fn=rng_hungry_probe)

    assert len(on["competence_trajectory"]) == N_EPISODES // 2

    # Non-degeneracy: the comparison is only meaningful if training actually moved the weights.
    assert not torch.allclose(w_off, torch.zeros_like(w_off)), "weights are trivially zero"
    assert w_off.numel() > 0

    assert torch.equal(w_off, w_on), (
        "probe perturbed training: trained policy weights differ between probe-on and "
        f"probe-off (max abs delta {float((w_off - w_on).abs().max())})"
    )
    for key, off_value in off.items():
        if key == "competence_trajectory":
            continue
        assert on[key] == off_value, (
            f"probe perturbed the training readout {key!r}: "
            f"probe-off={off_value!r} probe-on={on[key]!r}"
        )


# --------------------------------------------------------------------------- T3
def test_t3_cadence_and_episode_numbering():
    seen = []

    def probe(ep):
        seen.append(ep)
        return {"foraging_competence": float(ep)}

    guard = _train(seed=1, probe_every=3, probe_fn=probe)
    traj = guard["competence_trajectory"]

    assert len(traj) == N_EPISODES // 3
    # 1-indexed episode numbers at exact multiples of the cadence.
    assert [row["episode"] for row in traj] == [3, 6]
    assert seen == [3, 6], "probe_fn must receive the same 1-indexed episode it is stamped with"
    assert [row["foraging_competence"] for row in traj] == [3.0, 6.0]


# --------------------------------------------------------------------------- T4
def test_t4_half_wired_probe_raises_both_directions():
    with pytest.raises(ValueError, match="must be supplied together"):
        _train(seed=0, probe_every=2)
    with pytest.raises(ValueError, match="must be supplied together"):
        _train(seed=0, probe_fn=lambda ep: {"foraging_competence": 0.0})


@pytest.mark.parametrize("bad", [0, -1])
def test_t4b_non_positive_cadence_raises(bad):
    with pytest.raises(ValueError, match="positive episode cadence"):
        _train(seed=0, probe_every=bad, probe_fn=lambda ep: {"foraging_competence": 0.0})


# --------------------------------------------------------------------------- T5
def test_t5_probe_never_receives_the_training_env():
    """probe_fn takes only an episode index. The probe cannot reach the training env through
    the hook, so it cannot clobber its episode state -- the contract is enforced by the
    signature rather than by convention."""
    received = []

    def probe(*args, **kwargs):
        received.append((args, kwargs))
        return {"foraging_competence": 0.0}

    _train(seed=0, probe_every=4, probe_fn=probe)
    assert received, "probe should have fired"
    for args, kwargs in received:
        assert len(args) == 1 and isinstance(args[0], int)
        assert kwargs == {}


# --------------------------------------------------------------------------- T6
def test_t6_config_declares_cadence_and_defaults_off():
    cfg = boot.BootstrapExplorerConfig()
    assert cfg.retention_probe_every is None
    assert cfg.as_slice()["retention_probe_every"] is None

    cfg_on = boot.BootstrapExplorerConfig(retention_probe_every=25)
    assert cfg_on.as_slice()["retention_probe_every"] == 25
    # A varyable knob absent from the slice would let a probed and an unprobed cell collide on
    # one arm fingerprint despite carrying different artifacts.
    assert boot.BootstrapExplorerConfig().as_slice() != cfg_on.as_slice()


def test_t6b_bootstrap_explorer_enforces_both_or_neither():
    rep = _rep(0)
    cfg = boot.BootstrapExplorerConfig(n_episodes=2, retention_probe_every=1)
    with pytest.raises(ValueError, match="must be supplied together"):
        boot.train_bootstrap_explorer(rep, _env(0), seed=0, steps=STEPS,
                                      arm_label="x", cfg=cfg, denom=2)


def test_t6c_bootstrap_explorer_passes_probe_through():
    seen = []
    rep = _rep(0)
    cfg = boot.BootstrapExplorerConfig(n_episodes=4, retention_probe_every=2, use_rnd=False)
    guard = boot.train_bootstrap_explorer(
        rep, _env(0), seed=0, steps=STEPS, arm_label="x", cfg=cfg, denom=4,
        probe_fn=lambda ep: seen.append(ep) or {"foraging_competence": 2.0},
    )
    assert seen == [2, 4]
    assert [r["episode"] for r in guard["competence_trajectory"]] == [2, 4]


# --------------------------------------------------------------------------- T7
def test_t7_module_sources_are_ascii():
    for name in ("mech457_explorer_classes.py", "mech457_bootstrap_explorer.py"):
        path = Path(mech.__file__).resolve().parent / name
        raw = path.read_bytes()
        assert raw.decode("ascii"), f"{name} must be ASCII-only"
