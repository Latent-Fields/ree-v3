## MECH-295 Drive -> Liking-Stream -> Approach Cue Bridge (2026-04-26)
- MECH-295 weak-reading bridge: regulators.mech295_liking_bridge -- IMPLEMENTED 2026-04-26.
  Module: ree_core/regulators/mech295_liking_bridge.py (MECH295LikingBridge,
  MECH295LikingBridgeConfig, MECH295LikingBridgeOutput). Wires the missing
  link between SD-012 drive amplification, the SD-014/SD-015 liking-stream
  substrate, and E3 / BG action selection. Without this bridge, drive
  amplification produces a passive z_goal latent without behavioural
  consequence (the EXQ-483 catatonic-lock signature: override_signal climbs
  to mean 0.563, PAG release ratio ON_ON / ON_OFF = 1.69, but
  approach_commit = 0.0 across all four arms).
  Two integration sites:
    (a) update_z_goal() -- after the existing SD-012 / SD-037 effective_drive
        computation and GoalState.update() call: when bridge is active and
        goal_state.is_active(), call
        bridge.compute_anticipatory_liking_write(effective_drive,
        goal_state.goal_norm()) -> non-zero scalar -> ResidueField.update_valence
        at the goal location (z_goal latent, NOT current z_world), component
        VALENCE_LIKING. This is the anticipatory cue-side pulse, distinct
        from the existing consummatory-contact write in update_liking().
    (b) select_action() -- after lateral_pfc + ofc score_bias composition,
        before e3.select(): build per-candidate first-step z_world
        summaries (reuse cand_world_summaries when lateral_pfc / ofc are on),
        compute per-candidate goal_proximity via GoalState.goal_proximity,
        call bridge.compute_approach_cue_score_bias(effective_drive,
        proximities) -> NEGATIVE [K] tensor (E3 lower-is-better, so liking
        favours approach by reducing the score), composed additively with
        existing dacc_score_bias.
  Weak-necessity reading commitment: baseline liking-stream activation is
  sufficient. Cue-side gain is a function of drive * goal_proximity --
  the "is the bridge intact?" surface, not drive * residue.liking which
  would be the level-coupled strong reading. Setting
  mech295_liking_to_approach_cue_gain=0.0 is the SEVERED-BRIDGE arm of
  the falsifiable test: drive elevated AND write side intact AND cue
  side severed -> approach_commit predicted to collapse.
  Config: REEConfig.use_mech295_liking_bridge (bool, default False).
    Sub-knobs: mech295_drive_to_liking_gain (float, 1.0; 0 disables write
    side), mech295_liking_to_approach_cue_gain (float, 0.5; 0 severs cue
    side), mech295_min_drive_to_fire (float, 0.1; drive floor below which
    bridge is silent), mech295_min_z_goal_norm_to_fire (float, 0.05; goal
    norm floor below which bridge does not fire). All wired through
    REEConfig.from_dims().
  Backward compatible: use_mech295_liking_bridge=False by default;
    agent.mech295_bridge is None; both integration sites are no-ops.
    154/154 contracts + 7/7 preflight PASS with flag OFF (bit-identical
    to pre-MECH-295 HEAD, 2026-04-26).
  Activation smoke (2026-04-26): default agent + flag ON + drive_weight=2.0
    + cfg.goal.z_goal_enabled=True + min_z_goal_norm_to_fire=0.001 + 30 ticks
    forced drive=0.8 + benefit=0.4 -> n_write_fires=30, n_cue_fires=4,
    final goal_norm=0.333. Severed-bridge arm (cue gain=0) -> bias_max_abs
    exactly 0.0 with write side still firing.
  No trainable parameters. Pure scalar arithmetic + per-candidate proximity
  read. No phased training needed.
  Biological basis: NAc shell hedonic hotspot (Pecina & Berridge 2005,
    Castro & Berridge 2014), ventral pallidum (Smith Berridge & Aldridge
    2011 -- the strongest direct mechanistic anchor: VP single-unit
    recording shows drive change recodes palatability before cue firing),
    Berridge & Kringelbach 2015 architectural articulation, Dickinson &
    Balleine 1994 foundational behavioural devaluation requires outcome
    re-experience. Strong-vs-weak necessity not arbitrated by literature;
    Pecina 2003 DAT-knockdown finding (more wanting, unchanged liking) is
    compatible only with the weak reading. Bridge commits to the WEAK
    reading provisionally per claims.yaml MECH-295. Lit-pull synthesis:
    REE_assembly/evidence/literature/targeted_review_mech295_liking_approach_bridge/
  MECH-094: simulation_mode argument honoured at both compute methods;
    when True, write returns 0.0 and cue returns zero score_bias and
    counters do not advance.
  Validation experiment: V3-EXQ-493 queued (six-part diagnostic: UC1
    module-importable, UC2 master-OFF no-op, UC3 30-tick env loop write
    fires, UC4 cue side produces monotone-negative bias, UC5 SEVERED-BRIDGE
    COLLAPSE -- cue gain=0 produces zero bias even at elevated drive +
    write intact, UC6 MECH-094 simulation gate). All 6 PASS via --dry-run
    smoke 2026-04-26. Behavioural EXQ-483-style approach_commit recovery
    deferred to a successor (combined-cluster after V3-EXQ-490 lands).
  Design doc: REE_assembly/docs/architecture/mech_295_drive_liking_approach_bridge.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_mech295_liking_approach_bridge/
  See MECH-295, SD-012 (homeostatic drive input), SD-014 (valence vector
    substrate -- VALENCE_LIKING component), SD-015 (z_resource encoder
    upstream of GoalState.update), MECH-117 (existing wanting/liking
    dissociation in REE benefit_eval_head vs z_goal_latent), ARC-036
    (hedonic hotspot anatomical substrate -- prerequisite), MECH-094
    (call-site scoping + simulation_mode argument), SD-037 (broadcast
    override, drives effective_drive that the bridge consumes), MECH-269b
    (complementary candidate cause for EXQ-483 wired-but-inert; Q-040
    factorial points evidence at this bridge if MECH-269b alone fails to
    recover approach_commit).
