"""Contract tests for mech457_approach_extinction (MECH-457 H-consummation-binding leg 4, 2026-07-25).

Completes the DRIVE half of competence_floor retention leg 4. The mech457_consummatory_act env
node (2026-07-25) made contact AFFORD rather than EFFECT consumption and emits
info["on_consumable_resource"], but nothing consumed that signal: V3-EXQ-781's appetitive
approach primitive (mech.resource_proximity) reads obs_dict, which does not carry the flag, so
the drive could not extinguish on contact. This node threads the info flag into train_a2c's
approach block behind a default-OFF knob (approach_extinguishes_on_contact) so an appetitive
approach drive TERMINATES on arrival and hands off to the distinct CONSUME act. That
extinguish-and-hand-off is leg 4's treatment arm; 781's non-extinguishing terminal drive is the
control.

Lives in its OWN file (matches the test_mech457_retention_trajectory_probe.py /
test_mech457_distributional_critic.py precedent -- test_mech457_bootstrap_explorer.py's C18*
numbering belongs to a concurrent guard).

Contracts:
  E1  DEFAULT NO-OP (LOAD-BEARING). With a live approach drive on a consummatory env, passing
      approach_extinguishes_on_contact=False reproduces NOT passing it bit-identically on the
      trained policy weights -- the knob is inert at its default, so no existing 781-lineage
      caller changes behaviour.
  E2  EXTINCTION FIRES. On a consummatory env where contact demonstrably occurs, the ON knob
      zeros the approach reward on contact ticks and therefore changes the learned policy vs
      OFF. Non-degeneracy: the OFF drive must have fired (approach reward > 0) and contact must
      have been reached, else the comparison is vacuous.
  E3  HALF-WIRED IS AN ERROR (no drive). approach_extinguishes_on_contact=True with no
      approach_drive RAISES -- extinction with no drive is the control wearing the treatment
      label.
  E4  HALF-WIRED IS AN ERROR (non-consummatory env). approach_extinguishes_on_contact=True on an
      env built without consummatory_act_enabled RAISES -- on_consumable_resource is always
      False there, so extinction would silently never fire.
  E5  Config declares the knob in as_slice() (fingerprint hygiene), defaults it to False, and
      train_bootstrap_explorer enforces the config-level half-wired guard (extinction set but
      use_approach_primitive False raises through the same train_a2c check).
  E6  Module sources are ASCII-only (repo runtime-string rule).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

import experiments._lib.mech457_bootstrap_explorer as boot
import experiments._lib.mech457_explorer_classes as mech
import experiments._lib.mech457_fanout as fan
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734


STEPS = 20
N_EPISODES = 30
_EK = x734._env_kwargs_for_rung(fan.RUNG)


def _env(seed: int, consummatory: bool):
    kw = dict(_EK)
    if consummatory:
        kw["consummatory_act_enabled"] = True
    return x734._make_env(seed, kw)


def _rep(env, seed: int):
    return mech.make_rep("raw_view", env, seed=seed, p0=0, steps=STEPS,
                         actor_critic_hidden=32, cotrain_encoder=False)


def _reseed(seed: int):
    import random as _random
    torch.manual_seed(seed)
    np.random.seed(seed)
    _random.seed(seed)


def _train(seed: int, env, drive, coef: float, **kw):
    """One short raw-view A2C run, reseeded so two same-seed calls are bit-comparable."""
    _reseed(seed)
    rep = _rep(env, seed)
    guard = mech.train_a2c(
        rep, env, seed=seed, n_episodes=N_EPISODES, steps=STEPS,
        arm_label="extinction_test", denom=N_EPISODES,
        approach_drive=drive, approach_coef=coef, **kw
    )
    weights = torch.cat([p.detach().reshape(-1) for p in rep.policy().parameters()])
    return guard, weights


# --------------------------------------------------------------------------- E1
def test_e1_default_off_is_byte_identical():
    """The knob is inert at its default: passing False reproduces omitting it, on WEIGHTS."""
    g_absent, w_absent = _train(0, _env(0, consummatory=True), mech.resource_proximity, 1.0)
    g_false, w_false = _train(0, _env(0, consummatory=True), mech.resource_proximity, 1.0,
                              approach_extinguishes_on_contact=False)
    assert w_absent.numel() > 0
    assert not torch.allclose(w_absent, torch.zeros_like(w_absent)), "weights trivially zero"
    assert torch.equal(w_absent, w_false), (
        "approach_extinguishes_on_contact=False perturbed training relative to omitting it "
        f"(max abs delta {float((w_absent - w_false).abs().max())})"
    )


# --------------------------------------------------------------------------- E2
def test_e2_extinction_fires_and_changes_policy():
    """ON zeros the approach reward on contact ticks -> the learned policy diverges from OFF.

    Non-degeneracy: a recording drive on the OFF run proves the drive fired (reward > 0) and
    that at least one contact tick occurred, so the ON/OFF weight comparison is not vacuous.
    """
    seed = 0
    env_off = _env(seed, consummatory=True)
    contacts = []

    def recording_drive(obs_dict, _env=env_off):
        # Close over the SAME env passed to train_a2c: _on_consumable_resource is updated in
        # env.step() before train_a2c reads the approach drive, so it mirrors the info flag.
        contacts.append(bool(getattr(_env, "_on_consumable_resource", False)))
        return mech.resource_proximity(obs_dict)

    g_off, w_off = _train(seed, env_off, recording_drive, 1.0,
                          approach_extinguishes_on_contact=False)
    g_on, w_on = _train(seed, _env(seed, consummatory=True), mech.resource_proximity, 1.0,
                        approach_extinguishes_on_contact=True)

    assert float(g_off["mean_approach_reward_recent"]) > 0.0, (
        "OFF drive never fired (approach reward 0) -- resource_field_view absent or no contact; "
        "extinction test is vacuous"
    )
    assert any(contacts), "no contact tick occurred in the fixture -- extinction never exercised"
    assert not torch.equal(w_off, w_on), (
        "extinction ON did not change the learned policy vs OFF despite contact occurring "
        "-- the on_consumable_resource extinction branch is not firing"
    )


# --------------------------------------------------------------------------- E3
def test_e3_extinction_without_drive_raises():
    with pytest.raises(ValueError, match="requires an approach_drive"):
        _train(0, _env(0, consummatory=True), None, 0.0,
               approach_extinguishes_on_contact=True)


# --------------------------------------------------------------------------- E4
def test_e4_extinction_without_consummatory_env_raises():
    with pytest.raises(ValueError, match="consummatory_act_enabled"):
        _train(0, _env(0, consummatory=False), mech.resource_proximity, 1.0,
               approach_extinguishes_on_contact=True)


# --------------------------------------------------------------------------- E5
def test_e5_config_declares_knob_and_defaults_false():
    cfg = boot.BootstrapExplorerConfig()
    assert cfg.approach_extinguishes_on_contact is False
    sl = cfg.as_slice()
    assert "approach_extinguishes_on_contact" in sl
    assert sl["approach_extinguishes_on_contact"] is False


def test_e5b_train_bootstrap_explorer_enforces_config_half_wired():
    """A cfg with extinction set but no approach primitive raises through the train_a2c guard."""
    cfg = boot.BootstrapExplorerConfig(
        n_episodes=N_EPISODES, actor_critic_hidden=32, cotrain_encoder=False,
        use_approach_primitive=False, approach_extinguishes_on_contact=True,
    )
    env = _env(0, consummatory=True)
    _reseed(0)
    rep = _rep(env, 0)
    with pytest.raises(ValueError, match="requires an approach_drive"):
        boot.train_bootstrap_explorer(
            rep, env, seed=0, steps=STEPS, arm_label="extinction_cfg_test",
            cfg=cfg, denom=N_EPISODES,
        )


# --------------------------------------------------------------------------- E6
def test_e6_module_sources_ascii_only():
    for mod in (mech, boot):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        bad = [(i + 1, ln) for i, ln in enumerate(src.splitlines())
               if any(ord(c) > 127 for c in ln)]
        assert not bad, f"{mod.__file__} has non-ASCII lines: {bad[:3]}"
