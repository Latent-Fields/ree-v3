"""Contract tests for the MECH-204 F1 cold-start guard.

Flag: SerotoninConfig.precision_zero_point_require_waking (default False).

THE DEFECT THIS GUARDS (measured IGW-20260915-243, real StepHarness loop,
CausalGridWorldV2 size 8, K=1, recal step 0.25; V3-EXQ-541c shows the same
seed at cycle-1 target 2.148):

REEAgent.enter_rem_mode() calls SerotoninModule.enter_rem(current_precision=
self.e3.current_precision) unconditionally, and enter_rem() seeded
_persistent_zero_point (the MECH-204 F1 cross-cycle EMA) from whatever
precision it was handed, with no gate on whether any waking experience had
occurred since the last capture. The canonical driver start pattern
(env.reset(); agent.reset() before any waking tick -- also
StepHarness.run_episode) makes SleepLoopManager fire a cycle with ZERO waking
ticks, so the cold-start anchor was E3 precision_init (rv=0.5 -> precision
2.0): a sentinel reflecting no experience at all. The F1 EMA (alpha=0.1) then
carried that sentinel for ~30+ cycles, so the WRITEBACK consumer pushed an
already-calibrated rv AWAY from calibration (realized z_world PE variance
~0.0039 -> recalibrated 0.0121).

A SECOND instance of the same predicate, found while building this guard
(2026-09-18) and covered by the same fix: a cycle issues more than one
enter_rem() call, and the later call re-reads the SAME precision with no
intervening waking tick, double-counting one observation into the EMA. The
traced 8-episode probe showed 17 enter_rem calls -- one pre-waking sentinel
plus two per cycle -- of which the guard suppresses 9 and admits 8, exactly
one genuine capture per cycle.

Guarantees enforced:
  C1. OFF is bit-identical: with the flag False the _persistent_zero_point
      sequence equals the explicitly-computed legacy EMA sequence EXACTLY,
      including the pre-waking cold-start anchor on the sentinel.
  C2. ON, a pre-waking REM entry does not seed the EMA, and the diagnostic
      side effects (precision_at_rem_entry, tonic, phase) are preserved.
  C3. ON, the first REAL capture becomes the cold-start anchor exactly --
      NOT an EMA blend against the sentinel.
  C4. ON, the counter is zeroed on capture, so an immediate second entry
      with no intervening tick does not update (the double-count instance).
  C5. ON, the guard re-arms across reset() (new episode).
  C6. Consumer level, ON: a zero-waking-tick cycle leaves the WRITEBACK
      consumer with no target, so it emits mech204_recalibration_fired=0.0
      and NO mech204_recalibration_target key.
  C7. The knob survives REEConfig.from_dims(), which silently swallows
      unknown kwargs (memory reference-reeconfig-from_dims-silent-kwargs).
  C8. sense() is the waking-tick producer, and it is gated on the flag so
      the default path makes no new call.

MACHINE-SAFETY (CLAUDE.md "Running the test suite"): every assertion here is
module-level SCALAR arithmetic with no RNG and no torch sampling, so exact
float asserts are legitimate. Deliberately NOT written as an agent-level
comparison against a recorded constant: torch.multinomial diverges
linux-x86_64 vs darwin-arm64 and an action trajectory (hence a precision
trajectory) would diverge with it.
"""

from __future__ import annotations

import inspect

import pytest

from ree_core.neuromodulation.serotonin import SerotoninConfig, SerotoninModule


def _mod(require_waking: bool, alpha: float = 0.1) -> SerotoninModule:
    return SerotoninModule(
        SerotoninConfig(
            tonic_5ht_enabled=True,
            precision_zero_point_ema_alpha=alpha,
            precision_zero_point_require_waking=require_waking,
        )
    )


def test_c1_off_is_bit_identical_legacy_ema():
    """Flag OFF reproduces the legacy sequence exactly, sentinel included."""
    assert SerotoninConfig().precision_zero_point_require_waking is False, (
        "the guard must default OFF"
    )
    sero = _mod(False)
    alpha = 0.1

    # Pre-waking entry: legacy cold-starts the anchor on the sentinel.
    sero.enter_rem(2.0)
    assert sero._persistent_zero_point == 2.0
    sero.exit_sleep()

    expected = 2.0
    for captured in (255.0, 300.0, 120.0):
        for _ in range(5):
            sero.serotonin_step(benefit_exposure=0.0)
        sero.enter_rem(captured)
        expected = (1.0 - alpha) * expected + alpha * captured
        assert sero._persistent_zero_point == expected
        sero.exit_sleep()

    # The first two values of that legacy sequence, spelled out, so this
    # contract fails loudly if the EMA form itself is ever changed.
    assert 0.9 * 2.0 + 0.1 * 255.0 == pytest.approx(27.3)


def test_c2_on_pre_waking_entry_does_not_seed():
    """ON: zero waking ticks -> no seed, but diagnostics still captured."""
    sero = _mod(True)
    sero.enter_rem(2.0)

    assert sero._persistent_zero_point is None, (
        "pre-waking REM entry must not anchor the F1 EMA on precision_init"
    )
    assert sero.compute_recalibration_target() == 0.0
    # Diagnostic continuity and the ordinary REM side effects are preserved.
    assert sero._precision_at_rem_entry == 2.0
    assert sero._tonic_5ht == 0.0
    assert sero.phase == "rem"


def test_c3_on_first_real_capture_is_the_anchor():
    """ON: the anchor is the first REAL capture, not an EMA vs the sentinel."""
    sero = _mod(True)
    sero.enter_rem(2.0)          # skipped per C2
    sero.exit_sleep()
    sero.serotonin_step(benefit_exposure=0.0)
    sero.enter_rem(255.0)

    assert sero._persistent_zero_point == 255.0
    # 27.3 is the MEASURED defective cycle-2 value, so this encodes the actual
    # bug rather than restating the new code.
    assert sero._persistent_zero_point != pytest.approx(27.3, rel=1e-2)
    assert sero.compute_recalibration_target() == 255.0


def test_c4_on_counter_zeroed_on_capture_blocks_double_count():
    """ON: a second entry with no intervening tick must not update."""
    sero = _mod(True)
    sero.exit_sleep()
    sero.serotonin_step(benefit_exposure=0.0)
    sero.enter_rem(255.0)
    assert sero._persistent_zero_point == 255.0

    # Same cycle, no waking tick in between: the double-count instance.
    sero.enter_rem(400.0)
    assert sero._persistent_zero_point == 255.0, (
        "a re-read with no intervening waking tick must not enter the EMA"
    )


def test_c5_on_guard_rearms_across_reset():
    """ON: a new episode starts with no waking experience."""
    sero = _mod(True)
    sero.exit_sleep()
    sero.serotonin_step(benefit_exposure=0.0)
    sero.enter_rem(255.0)
    assert sero._persistent_zero_point == 255.0

    sero.reset()
    sero.enter_rem(2.0)
    assert sero._persistent_zero_point == 255.0, (
        "an episode-boundary entry before any tick must not capture"
    )
    assert sero._persistent_zero_point is not None


def test_c6_consumer_sees_no_target_on_zero_waking_tick_cycle():
    """ON at the consumer: no target -> fired=0.0 and no target key."""
    sero = _mod(True)
    sero.enter_rem(2.0)
    target = sero.compute_recalibration_target()

    writeback_metrics: dict[str, float] = {}
    # The consumer's own predicate, verbatim from
    # ree_core/sleep/phase_manager.py (`if target > 0.0:`).
    if target > 0.0:
        writeback_metrics["mech204_recalibration_target"] = target
        writeback_metrics["mech204_recalibration_fired"] = 1.0
    else:
        writeback_metrics["mech204_recalibration_fired"] = 0.0

    assert writeback_metrics["mech204_recalibration_fired"] == 0.0
    assert "mech204_recalibration_target" not in writeback_metrics

    # And after real waking experience the consumer does fire.
    sero.exit_sleep()
    sero.serotonin_step(benefit_exposure=0.0)
    sero.enter_rem(255.0)
    assert sero.compute_recalibration_target() > 0.0


def test_c7_knob_survives_from_dims():
    """from_dims silently swallows unknown kwargs -- pin all three sites."""
    from ree_core.utils.config import REEConfig

    cfg_off = REEConfig.from_dims(body_obs_dim=12, world_obs_dim=250, action_dim=4)
    assert cfg_off.serotonin.precision_zero_point_require_waking is False

    cfg_on = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        precision_zero_point_require_waking=True,
    )
    assert cfg_on.serotonin.precision_zero_point_require_waking is True, (
        "from_dims must propagate the knob to config.serotonin -- a knob added "
        "only to the dataclass is a silent no-op"
    )


def test_c8_sense_is_the_gated_waking_tick_producer():
    """The producer must be sense() (universal) and gated on the flag.

    serotonin_step() is NOT a usable sole producer: it is called by experiment
    drivers, never from inside ree_core, and StepHarness never calls it -- a
    counter fed only from there would stay at 0 forever on the canonical loop
    and make this guard a permanent kill switch for MECH-204 recalibration.
    """
    from ree_core.agent import REEAgent

    src = inspect.getsource(REEAgent.sense)
    assert "note_waking_tick" in src, (
        "REEAgent.sense() must drive SerotoninModule.note_waking_tick()"
    )
    assert "precision_zero_point_require_waking" in src, (
        "the producer call must be gated on the flag so OFF makes no new call"
    )

    # And the tick itself is waking-only: a sleep-phase call is a no-op.
    sero = _mod(True)
    sero.enter_rem(2.0)
    assert sero.phase == "rem"
    sero.note_waking_tick()
    sero.enter_rem(2.0)
    assert sero._persistent_zero_point is None, (
        "a tick recorded during REM must not satisfy the waking guard"
    )
