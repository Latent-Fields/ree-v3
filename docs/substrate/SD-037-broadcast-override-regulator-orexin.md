## SD-037: Broadcast Override Regulator (orexin-analog) (2026-04-25)
- SD-037: regulators.broadcast_override -- IMPLEMENTED 2026-04-25.
  Module: ree_core/regulators/broadcast_override.py (BroadcastOverrideRegulator,
  BroadcastOverrideConfig). Third regulatory layer of the V3 control stack
  alongside 5-HT goal-pipeline gain (MECH-186/187/188) and SD-036
  GABAergic cross-stream decay. Orexinergic (hypocretin) hub analog: scalar
  override_signal in [0, 1] driven by SD-012 drive_level + sustained-threat
  rolling-window magnitude over z_harm, EMA-smoothed.
  Computation:
    sustained_threat = clip_[0,1]( rolling_mean(z_harm.norm, window) /
                                   sustained_threat_threshold )
    drive_input      = clip_[0,1]( drive_level )
    raw              = sigmoid( drive_weight*drive_input
                              + harm_weight*sustained_threat
                              - recruitment_threshold )
    override_signal  = clip_[0,1]( (1-decay_rate)*prev + decay_rate*raw )
  Consumed at three sites:
    PAG freeze-gate (MECH-279): exit_threshold scaled by
      (1 + alpha_override * override_signal). Strong override raises
      the bar for entering / staying in committed-freeze (orexin ->
      arousal / escape-from-freeze; Carter et al. 2009 LH -> PAG).
      PAGFreezeGateConfig.alpha_override (default 0.0; agent wires
      override_alpha_pag when both flags on). Override_signal passed
      explicitly per-tick into PAGFreezeGate.tick().
    SalienceCoordinator (SD-032a): update_signal("override_signal", ...)
      injection biases operating-mode aggregate toward external_task
      (registered affinity_weights["override_signal"] =
      {"external_task": override_salience_reweight_alpha}). MECH-261
      generalises MECH-094 here -- registry is the gating point.
    GoalState (SD-012): drive -> z_goal seeding amplified by
      effective_drive *= (1 + (override_goal_seeding_gain - 1) *
      override_signal). Implements "drive becomes action-orienting only
      when override system has recruited" semantic. Default gain 2.0
      means saturated override doubles the seeding multiplier.
  Config: REEConfig.use_broadcast_override (bool, default False).
    Sub-knobs: override_recruitment_threshold (0.5),
    override_alpha_pag (0.5; PAG exit-threshold scaling),
    override_salience_reweight_alpha (0.3; SalienceCoordinator affinity),
    override_drive_weight (1.0), override_harm_weight (1.0),
    override_sustained_threat_window (12 ticks),
    override_sustained_threat_threshold (0.4),
    override_decay_rate (0.05; ~20-tick EMA),
    override_goal_seeding_gain (2.0).
  Defaults are biologically defensible per orexin kinetics lit-pull
  (Mileykovskiy et al. 2005 LH burst firing 5-15 Hz on threat;
  Lee et al. 2005 LHA orexin neuron arousal-correlated activity;
  Karnani et al. 2020 sleep/wake state transitions; Johnson et al. 2012
  PAG-projecting orexin escape behaviours). Two flagged for sweep:
  recruitment_threshold and alpha_pag at low end.
  Agent wiring (REEAgent):
    __init__ -- after PAG instantiation: if use_broadcast_override is on,
      construct BroadcastOverrideConfig from sub-knobs and instantiate
      BroadcastOverrideRegulator. PAG freeze-gate config receives
      alpha_override = override_alpha_pag when both flags are on
      (else 0.0 -- no-op). SalienceCoordinator (if present) gets
      affinity_weights["override_signal"] registered.
    sense() -- after SD-036 GABAergic decay tick: if broadcast_override
      is not None, tick(drive_level=goal_state._last_drive_level,
      z_harm_norm=z_harm.norm, simulation_mode=hypothesis_tag).
      One-step latency on drive_level read is intentional: the
      goal_state value reflects the previous tick's effective_drive,
      which is the post-pACC-bias drive. No double counting.
    select_action() -- before salience.tick(): inject
      update_signal("override_signal", broadcast_override.override_signal).
      PAG.tick() receives override_signal explicitly each cycle so
      exit_threshold scaling responds on the same tick.
    update_z_goal() -- after pacc.effective_drive: amplify effective_drive
      by (1 + (override_goal_seeding_gain - 1) * override_signal),
      clipped to [0, 1].
    reset() -- per episode: broadcast_override.reset() clears threat
      window, EMA state, and diagnostics.
  Backward compatible: use_broadcast_override=False by default;
    agent.broadcast_override is None; PAG receives alpha_override=0.0;
    salience signal slot is no-op; goal seeding multiplier=1.0. 95/95
    contracts PASS with flag OFF (bit-identical to pre-SD-037 HEAD,
    2026-04-25).
  MECH-282 (LPB interoceptive routing) -- IMPLEMENTED 2026-05-21.
    Module: ree_core/regulators/lpb_interoceptive_routing.py (LPBInteroceptiveRouter).
    Config: REEConfig.use_lpb_interoceptive_routing (bool, default False),
      lpb_intero_z_dim (16), lpb_drive_weight, lpb_resource_weight.
    Data flow: harm_obs resource slice zeroed before HarmEncoder -> z_harm (external);
      drive_level + harm_obs_a resource EMA -> z_harm_intero (non-trainable broadcast).
    SD-037 coupling: when both flags on, BroadcastOverrideRegulator.tick uses
      lpb_split_recruitment (intero drives override; external drives PAG freeze proxy).
    Backward compatible: flag OFF -> agent.lpb_router is None; z_harm_intero None.
    Validation experiment: V3-EXQ-600 queued (3-arm substrate diagnostic).
    See MECH-282, SD-037 design doc section MECH-282.

  Activation smoke (2026-04-25):
    Flag OFF: agent.broadcast_override is None.
    Flag ON: regulator instantiates with default config; one tick at
      drive=0.9, harm=0.6 produces override_signal=0.040 (sigmoid raw
      ~0.81 EMA-smoothed at decay_rate=0.05). 50 ticks of sustained
      load climb to 0.7431.
    MECH-094: simulation_mode=True returns cached signal unchanged
      (no threat-window advance, no EMA update).
    PAG with both flags on: alpha_override=0.5 wired correctly.
  No trainable parameters. Pure scalar arithmetic. No phased training.
  Biological basis: orexinergic (hypocretin) hub in lateral
    hypothalamus (LH). Persistent depletion (SD-012) plus sustained
    nociceptive signal (z_harm window) recruits LH orexin neurons;
    broad projections (PAG, BLA, LC, VTA, mPFC) gate downstream
    arousal / escape / motivational systems. Lit-pull synthesis:
    REE_assembly/evidence/literature/targeted_review_orexin_kinetics/
    synthesis.md.
  Failure-mode predictions (validation EXQ acceptance criteria):
    PWS-hyperphagia analog: saturated override (chronic high
      drive + harm) -> >=2x approach-commit rate vs balanced arm.
    Narcolepsy/cataplexy analog: lost override (regulator OFF
      under threat) -> <30% approach-commit vs balanced arm.
    Catatonic lock-in (V3-EXQ-471): with SD-036 + SD-037 ON, the
      orexin-analog raises PAG exit_threshold under sustained
      drive+harm so freeze releases instead of persisting.
  MECH-094: simulation_mode argument on tick(); when True, cached
    signal returned unchanged and no state advances. Replay / DMN
    content cannot recruit the override system.
  Validation experiment: V3-EXQ-483b PASS substrate-readiness
    (2026-05-08; override_mean>0.30, PAG release ratio 1.875x).
    V3-EXQ-483/483a FAIL behavioural (approach_commit=0 all arms).
    Tier-1 evidence: V3-EXQ-483c queued (GAP-4 fishtank factorial).
    MECH-282 not in SD-037 landing scope; MECH-286 landed separately 2026-05-21.
  Design doc: REE_assembly/docs/architecture/sd_037_broadcast_override_regulator.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_orexin_kinetics/
  See SD-037, SD-036, MECH-279, SD-012, SD-032a, MECH-261, MECH-094.
