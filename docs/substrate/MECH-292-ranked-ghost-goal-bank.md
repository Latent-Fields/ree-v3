## MECH-292 Ranked Ghost-Goal Bank (2026-04-27)
- MECH-292: hippocampal.unresolved_goal_ghost_bank -- IMPLEMENTED 2026-04-27.
  Module: ree_core/hippocampal/ghost_goal_bank.py (GhostGoalBank,
  GhostGoalBankConfig, GhostGoalBankEntry). First downstream consumer of the
  SD-039 dual-trace anchor goal-snapshot payload (substrate landed 2026-04-26;
  population layer landed 2026-04-27 with V3-EXQ-494 6/6 PASS). Pure-
  arithmetic, non-trainable derived view over the existing AnchorSet.all_anchors()
  pool. The bank does NOT own state beyond a small per-call diagnostics cache;
  the anchor pool itself remains the source of truth. Per spec
  (REE_assembly/docs/architecture/mech_292_ghost_goal_bank.md): "Implementation
  is intentionally a derived view, not a persistent store: SD-039 already
  preserves the per-anchor payload; MECH-292 just arranges the existing data."
  Ranking formula (per anchor a clearing goal_match_floor):
    wanting        = a.goal_payload.wanting_strength
    goal_match     = a.goal_match(current_z_goal)             [SD-039 cosine]
    staleness      = staleness_accumulator.snapshot()[(scale, segment_id)]
                     when accumulator present, else
                     clip_[0,1]((current_tick - last_accessed) * staleness_proxy_rate)
    recoverability = clip_[0,1](a.goal_payload.last_vs)
                     when last_vs is not None, else
                     default_recoverability_when_unknown
    ghost_priority = w_w*wanting + w_m*goal_match + w_s*staleness + w_r*recoverability
  goal_match_floor (default 0.05) is the architectural rumination guard:
  anchors with no payload OR with goal_match below the floor are invisible
  to the bank entirely. Pure low-V_s chasing is excluded by construction.
  Default pool: include_inactive=True, include_active=False, scale=None
  (all scales). MECH-293 ghost-goal probes work primarily over inactive
  traces; diagnostic / replay-prioritisation consumers may flip include_active
  on. top_k caps the returned list at 32 by default.
  Config: HippocampalConfig.use_mech292_ghost_bank (bool, default False).
    Nested: HippocampalConfig.ghost_goal_bank_config (GhostGoalBankConfig,
    default factory). REEConfig.from_dims surfaces 6 sub-knobs:
    mech292_wanting_weight (1.0), mech292_goal_match_weight (1.0),
    mech292_staleness_weight (0.5), mech292_recoverability_weight (0.5),
    mech292_goal_match_floor (0.05), mech292_top_k (32). Other GhostGoalBankConfig
    fields (default_recoverability_when_unknown, include_inactive,
    include_active, scale, staleness_proxy_rate) are not surfaced through
    from_dims; set on the nested config directly when needed.
  HippocampalModule wiring (after staleness_accumulator block):
    instantiate GhostGoalBank when use_mech292_ghost_bank is on; raise
    ValueError if anchor_set is None (use_anchor_sets must be True) OR if
    anchor_set.config.use_sd039_anchor_payload is False (otherwise every
    anchor scores goal_match=0.0 and the bank degenerates to empty).
    Public API: rank_ghost_goals(current_z_goal) -> List[GhostGoalBankEntry]
    (returns [] when bank is None or current_z_goal is None);
    reset_ghost_goal_bank() per-episode reset of diagnostics cache (anchor
    pool is reset separately by reset_anchor_set()).
  Agent wiring (REEAgent.reset()): reset_ghost_goal_bank() called on
    episode boundary when use_mech292_ghost_bank is on. No agent.sense /
    select_action wiring -- MECH-293 will be the first behavioural consumer.
  Backward compatible: use_mech292_ghost_bank=False by default;
    hippocampal.ghost_goal_bank is None; rank_ghost_goals returns [].
    164/164 contracts + 7/7 preflight PASS with master OFF (bit-identical
    to pre-MECH-292 HEAD). Smoke (master ON + SD-039 ON + 60-tick episode +
    forced fast-scale boundaries every 8 ticks): 6 inactive anchors with
    populated payload, 6 admitted entries, max_priority 1.609,
    monotone-decreasing.
  No trainable parameters. Pure scalar arithmetic + cosine via
    Anchor.goal_match (SD-039 helper). No phased training needed.
  Falsifiable signature (per spec, behavioural validation deferred): in a
    reward-relocation or blocked-corridor task, anchors from the now-
    obstructed but still-valued path should rank above equally stale but
    goal-irrelevant anchors. Substrate-level dissociation (UC4 of V3-EXQ-496)
    confirmed: Phase A goal-inactive anchors all below floor; Phase B
    goal-active anchors admitted with goal_match component dominant on top
    entry.
  MECH-094: substrate-side scope. Bank reads payloads whose provenance was
    set by the SD-039 population layer (sense() always passes
    simulation_mode=False, so source anchors carry waking-stream provenance).
    The bank itself has no write path -- nothing to gate. Inherits whatever
    provenance the source anchors carry.
  Validation experiment: V3-EXQ-496 queued (5 sub-tests UC1-UC5 covering
    module / config / method exposure, master OFF no-op, ranking_fires,
    goal_irrelevant_excluded, component_breakdown_consistent). Mac
    2026-04-27: 5/5 PASS (39s). Behavioural validation lives in V3-EXQ-495
    (V3 full-completion gate) once MECH-293 wires propose_trajectories()
    to consume the bank.
  Design doc: REE_assembly/docs/architecture/mech_292_ghost_goal_bank.md
  See MECH-292 (parent claim), SD-039 (dual-trace payload substrate),
  MECH-216 (predictive wanting -- wanting_strength source), MECH-230
  (z_goal latent -- cosine query target), MECH-269 Phase 2 (ii) (anchor
  substrate), MECH-269 Phase 1/2 (iii) (per-stream / per-region V_s for
  last_vs), MECH-284 (region staleness accumulator), MECH-293 (downstream
  consumer -- waking ghost-goal probe search), ARC-060 (hybrid field+bank
  architectural framing).
