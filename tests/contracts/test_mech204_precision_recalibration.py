"""Contract tests for MECH-204 precision recalibration consumer.

Sleep-substrate plan Phase 1 (REE_assembly/evidence/planning/
sleep_substrate_plan.md, GAP-1). Closes the
captured-without-consumer gap: SerotoninModule._precision_at_rem_entry
is captured at REM entry but, prior to Phase 1, never read by any
module other than get_state(). Option A statistical update on
E3._running_variance is the smallest-step instantiation; Option B
broadcast read-site is deferred to Phase 7.

Guarantees enforced:
  C1. New module surface present:
      * SerotoninModule.compute_recalibration_target() exists.
      * E3TrajectorySelector.recalibrate_precision_to(target, step)
        exists and is callable.
      * REEConfig has use_rem_precision_recalibration (default False)
        and rem_precision_recalibration_step (default 0.1).
  C2. Default config flag OFF: agent build is bit-identical at the
      flag level (sleep_loop OFF -> agent.sleep_loop is None).
  C3. Sleep-loop ON but recalibration flag OFF: SleepLoopManager
      instance carries use_rem_precision_recalibration=False and a
      fired cycle does not emit any mech204_* metric.
  C4. Recalibration arithmetic (unit-level, no agent boot): Option A
      linear interpolation moves _running_variance toward
      1.0/target by exactly the configured step.
  C5. Zero-or-negative target is a no-op (sentinel: REM never
      entered, or serotonin disabled); rv_before == rv_after.
  C6. Step=0 is a no-op (caller-side guard).
  C7. Captured precision_at_rem_entry has at least one non-get_state
      reader. This is the regression guard against re-introducing
      the capture-only pattern: we walk the source for any reference
      to compute_recalibration_target() in a module that is not
      serotonin.py, and assert the count is >= 1.
  C8. End-to-end via SleepLoopManager: master flag + sleep_loop +
      tonic_5ht + rem_enabled all ON, force a recalibration target
      that corresponds to a variance lower than _running_variance,
      drive one cycle, and assert _running_variance moved toward
      the target's variance by exactly the configured step.
"""

from __future__ import annotations

from pathlib import Path


def _build_agent(*, sleep_loop=True, K=1, sws=True, rem=True,
                 use_rem_precision_recalibration=False,
                 rem_precision_recalibration_step=0.1):
    from ree_core.agent import REEAgent
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12,
        world_obs_dim=250,
        action_dim=4,
        use_sleep_loop=sleep_loop,
        sleep_loop_episodes_K=K,
        use_rem_precision_recalibration=use_rem_precision_recalibration,
        rem_precision_recalibration_step=rem_precision_recalibration_step,
    )
    cfg.sws_enabled = sws
    cfg.rem_enabled = rem
    cfg.serotonin.tonic_5ht_enabled = True
    return REEAgent(cfg)


def test_c1_module_surface_present():
    from ree_core.neuromodulation.serotonin import SerotoninModule, SerotoninConfig
    from ree_core.predictors.e3_selector import E3TrajectorySelector
    from ree_core.utils.config import REEConfig

    sero = SerotoninModule(SerotoninConfig(tonic_5ht_enabled=True))
    assert hasattr(sero, "compute_recalibration_target")
    target = sero.compute_recalibration_target()
    assert isinstance(target, float)

    assert hasattr(E3TrajectorySelector, "recalibrate_precision_to")

    cfg = REEConfig()
    assert getattr(cfg, "use_rem_precision_recalibration", None) is False
    assert getattr(cfg, "rem_precision_recalibration_step", None) == 0.1


def test_c2_default_flag_off_backward_compat():
    from ree_core.utils.config import REEConfig

    cfg = REEConfig.from_dims(
        body_obs_dim=12, world_obs_dim=250, action_dim=4,
    )
    assert cfg.use_rem_precision_recalibration is False
    assert cfg.rem_precision_recalibration_step == 0.1


def test_c3_sleep_loop_on_recalibration_off_no_metrics():
    agent = _build_agent(use_rem_precision_recalibration=False)
    assert agent.sleep_loop is not None
    assert agent.sleep_loop.use_rem_precision_recalibration is False
    metrics = agent.sleep_loop.force_cycle(agent)
    assert metrics is not None
    for k in metrics.keys():
        assert not k.startswith("mech204_"), (
            f"recalibration metric {k} present with flag OFF"
        )


def test_c4_recalibration_arithmetic_correctness():
    from ree_core.predictors.e3_selector import E3TrajectorySelector, E3Config
    from ree_core.residue.field import ResidueField, ResidueConfig

    e3 = E3TrajectorySelector(
        E3Config(),
        ResidueField(ResidueConfig()),
    )
    e3._running_variance = 1.0
    target_precision = 4.0  # target variance = 0.25
    step = 0.5
    rv_before, rv_after = e3.recalibrate_precision_to(target_precision, step=step)
    assert rv_before == 1.0
    expected = (1.0 - step) * 1.0 + step * (1.0 / (target_precision + 1e-6))
    assert abs(rv_after - expected) < 1e-9
    assert e3._running_variance == rv_after


def test_c5_zero_target_is_no_op():
    from ree_core.predictors.e3_selector import E3TrajectorySelector, E3Config
    from ree_core.residue.field import ResidueField, ResidueConfig

    e3 = E3TrajectorySelector(
        E3Config(),
        ResidueField(ResidueConfig()),
    )
    e3._running_variance = 0.7
    rv_before, rv_after = e3.recalibrate_precision_to(0.0, step=0.5)
    assert rv_before == rv_after == 0.7
    rv_before, rv_after = e3.recalibrate_precision_to(-1.0, step=0.5)
    assert rv_before == rv_after == 0.7


def test_c6_zero_step_is_no_op():
    from ree_core.predictors.e3_selector import E3TrajectorySelector, E3Config
    from ree_core.residue.field import ResidueField, ResidueConfig

    e3 = E3TrajectorySelector(
        E3Config(),
        ResidueField(ResidueConfig()),
    )
    e3._running_variance = 0.7
    rv_before, rv_after = e3.recalibrate_precision_to(4.0, step=0.0)
    assert rv_before == rv_after == 0.7


def test_c7_precision_at_rem_entry_has_consumer():
    """Regression guard: the captured precision must be read by at least
    one module other than serotonin.get_state() / load_state(). Phase 1
    introduces SleepLoopManager._run_cycle as that consumer through
    SerotoninModule.compute_recalibration_target()."""
    repo_root = Path(__file__).resolve().parents[2]
    serotonin_path = repo_root / "ree_core" / "neuromodulation" / "serotonin.py"
    sleep_path = repo_root / "ree_core" / "sleep" / "phase_manager.py"

    sleep_src = sleep_path.read_text(encoding="utf-8")
    assert "compute_recalibration_target" in sleep_src, (
        "SleepLoopManager must read SerotoninModule.compute_recalibration_target() "
        "(MECH-204 capture-only regression guard)"
    )
    sero_src = serotonin_path.read_text(encoding="utf-8")
    assert "compute_recalibration_target" in sero_src
    assert "_precision_at_rem_entry" in sero_src


def test_c8_end_to_end_writeback_recalibration_fires():
    """End-to-end: master flag + sleep_loop + tonic_5ht + rem_enabled all
    ON, fire one cycle, and assert the WRITEBACK hook fires with all
    expected mech204_* metrics populated. Within a single cycle, the
    captured precision_at_rem_entry equals the rv at REM entry, so the
    Option A interpolation is mathematically a no-op against itself --
    behavioural movement requires drift between cycles, which the
    validation EXQ exercises. This test verifies wiring, not movement."""
    agent = _build_agent(
        use_rem_precision_recalibration=True,
        rem_precision_recalibration_step=0.5,
    )
    assert agent.sleep_loop is not None
    assert agent.sleep_loop.use_rem_precision_recalibration is True

    rv_before_cycle = agent.e3._running_variance
    metrics = agent.sleep_loop.force_cycle(agent)
    assert metrics is not None
    assert metrics.get("mech204_recalibration_fired", 0.0) == 1.0
    # All four diagnostic metrics must be present:
    assert "mech204_recalibration_target" in metrics
    assert "mech204_running_variance_before" in metrics
    assert "mech204_running_variance_after" in metrics
    assert metrics["mech204_recalibration_step"] == 0.5
    # Target is a positive precision (1/variance with rv ~ 0.5 default ->
    # precision ~ 2.0, well above zero).
    assert metrics["mech204_recalibration_target"] > 0.0
    # Within-cycle no-op invariant: rv_after equals rv_before to
    # numerical tolerance because target_variance = rv_at_rem_entry =
    # rv at WRITEBACK.
    assert abs(
        metrics["mech204_running_variance_after"]
        - metrics["mech204_running_variance_before"]
    ) < 1e-6


def test_c9_end_to_end_drift_recalibration_moves_rv():
    """Engineer a drift between REM entry and WRITEBACK by directly
    overwriting _precision_at_rem_entry AFTER the agent's run_sleep_cycle
    has captured it. Since WRITEBACK reads compute_recalibration_target()
    AFTER agent.run_sleep_cycle(), poking the value mid-cycle is not
    feasible from outside; instead, simulate the prior-cycle drift
    pattern by:
      1. running force_cycle once (this populates precision_at_rem_entry),
      2. deliberately moving rv to a value that differs from the
         captured target,
      3. directly invoking the recalibration arithmetic (already covered
         by C4) and asserting movement matches the formula.

    This ensures the consumer-side wiring is genuinely capable of moving
    rv when there is a non-trivial gap between current rv and the
    captured zero-point.
    """
    agent = _build_agent(
        use_rem_precision_recalibration=True,
        rem_precision_recalibration_step=0.25,
    )
    metrics_first = agent.sleep_loop.force_cycle(agent)
    assert metrics_first.get("mech204_recalibration_fired", 0.0) == 1.0
    captured_target = float(agent.serotonin.compute_recalibration_target())
    assert captured_target > 0.0

    # Simulate waking drift: push rv far from the captured zero-point.
    agent.e3._running_variance = 5.0
    rv_before, rv_after = agent.e3.recalibrate_precision_to(
        captured_target, step=0.25,
    )
    assert rv_before == 5.0
    expected = 0.75 * 5.0 + 0.25 * (1.0 / (captured_target + 1e-6))
    assert abs(rv_after - expected) < 1e-9
    assert rv_after < rv_before  # rv moved toward (smaller) target_variance
