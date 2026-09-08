## SD-032c: cingulate.aic_analog_salience_urgency -- IMPLEMENTED (2026-04-19)
- SD-032c: cingulate.aic_analog_salience_urgency -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/aic_analog.py (AICAnalog, AICConfig).
  Anterior-insula-analog interoceptive-salience / urgency-interrupt module.
  NOT the affective-pain consumer (that is SD-032b); this is the mode-switch
  trigger source AND the descending pain-modulation gate. Subsumes SD-021:
  the raw beta_gate.is_elevated check in agent.sense() is replaced by a
  drive-aware, operating-mode-aware gain function.
  Inputs (per sense() tick):
    z_harm_a_norm  (SD-011 affective stream)
    drive_level    (SD-012 GoalState._last_drive_level)
    beta_gate_elevated (MECH-090 committed-state signal)
    operating_mode (SD-032a coordinator, previous tick; None -> treat
                    p_external_task=1.0, preserves SD-032c function even
                    without coordinator)
    extra_salient  (optional; unexpected z_goal drop, reward-surprise,
                    irreversibility; default no-op via aic_extra_weight=0)
  Outputs (stored on the module, cached in agent._aic_last_tick):
    aic_salience   -- fed to SalienceCoordinator.update_signal("aic_salience",
                      ...) BEFORE coordinator.tick() each select_action cycle
                      (drives MECH-259 urgency-trigger).
    harm_s_gain    -- multiplier on z_harm in sense(), replacing the raw
                      SD-021 beta_gate check when use_aic_analog=True.
                      harm_s_gain < 1.0 only when committed AND the agent is
                      not depleted (drive_protect=1.0 default).
    urgency_signal -- diagnostic threshold crossing on aic_salience.
  Computation:
    baseline <- (1-alpha)*baseline + alpha * z_harm_a_norm  (EMA interoceptive
                                                             baseline)
    urgency  = max(0, (z_harm_a_norm - baseline) / (baseline + eps))
    aic_salience = urgency * (1 + drive_coupling * drive_level)
                 + aic_extra_weight * sum(extra_salient)
    drive_protect = max(0, 1 - drive_protect_weight * drive_level)
    harm_s_gain = clip_[0,1] ( 1 - base_attenuation * p_external *
                               float(beta_gate_elevated) * drive_protect )
  Config: REEConfig.use_aic_analog (bool, default False).
    Sub-knobs: aic_baseline_alpha (0.02, ~50-step window),
    aic_drive_coupling (1.0 -- MUST be non-zero for falsification signature),
    aic_urgency_threshold (1.0, diagnostic only),
    aic_base_attenuation (0.5, matches legacy descending_attenuation_factor),
    aic_drive_protect_weight (1.0; alterable-configuration knob flagged by
                              SD-032c spec: +1 preserve depleted signal,
                              0 drive-independent, -1 opposite-sign),
    aic_extra_weight (0.0, reserved for extra salient-event signals).
  Falsification signature (spec): same z_harm_a -> different mode-switch
    behaviour in depleted vs well-resourced agents. Both aic_salience AND
    harm_s_gain depend on drive_level -- this is the ONLY V3 substrate that
    makes the dependence structural. EXQ-325a FAIL (DESCENDING ==
    CONTROL bit-identical under raw beta_gate check) resolves when the AIC
    path replaces the raw check -- the descending branch becomes a
    genuinely different function of state.
  Data flow: encode() -> z_harm_a, z_harm -> aic.tick(z_harm_a_norm,
    drive_level, beta_gate_elevated, operating_mode_prev) -> aic_salience
    cached + harm_s_gain applied to z_harm if harm_descending_mod_enabled.
    select_action() injects aic_salience into coordinator via
    update_signal("aic_salience", ...) BEFORE coordinator.tick() so MECH-259
    trigger sees it on the current cycle. One-step lag on operating_mode
    read is biologically plausible (AIC->dACC->SAL is a circuit).
  Backward compatible: use_aic_analog=False by default. Legacy SD-021 raw
    beta_gate check retained behind the same harm_descending_mod_enabled
    flag -- selected only when use_aic_analog=False. With both flags off,
    existing experiments unchanged. The old descending_attenuation_factor
    config is still consumed by the legacy path.
  Biological basis: Craig 2009 AIC as interoceptive-salience hub with
    autonomic and motor efferents; Menon & Uddin 2010 salience-network
    coupling; Basbaum 1984 + Keltner 2006 ACC/AIC -> PAG descending
    inhibitory pathway.
  MECH-094: not applicable (waking observation stream, not replay content).
  Phased training: not applicable (non-trainable arithmetic, single EMA).
  Validation experiment: V3-EXQ-325b queued (3-condition x 2-drive-regime
    retest of EXQ-325a; supersedes EXQ-325a; acceptance criteria include
    drive-dependence contrast which the prior metric could not measure).
  See SD-032c, SD-032a, SD-032b, SD-021, MECH-259, MECH-261, SD-032 parent.
