## SD-057 phase-2: L6 cue-recall + L7 dACC object-discriminative readout (2026-06-04)
- SD-057 phase-2: drive.object_bound_incentive_salience (L6 MECH-347 + L7
  MECH-348) -- IMPLEMENTED 2026-06-04. Completes the GAP-7 closure map on top of
  the SD-057 v1 (L2-L3-L4) bank landed earlier the same day. Both no-op-default,
  bit-identical OFF, no trained parameters (no phased training).
  Modules:
    ree_core/goal.py: GoalConfig gains use_cue_recall (False) + cue_recall_gain
      (0.05) + cue_recall_min_proximity (0.0); new GoalState.cue_pull(z_object,
      strength) -- a directional z_goal nudge with NO benefit gate and NO token
      revaluation (distinct from update(), which is benefit-gated + EMA-revalues).
    ree_core/agent.py: new REEAgent.cue_recall_wanting(cue_type, drive_level,
      simulation_mode) -- L6 primitive; retrieves the bank token for cue_type,
      computes drive-specific wanting amplitude, and cue_pulls z_goal toward
      z_object[cue_type] by cue_recall_gain*clamp(amp). L7: select_action dACC
      block computes per-candidate goal_proximity (to the object-bound z_goal,
      reusing the MECH-295 first-step z_world summary pattern) under
      use_mech_consume and passes candidate_goal_proximity into self.dacc(...).
      __init__ adds two loud-not-silent preconditions: use_mech_consume requires
      use_dacc; use_cue_recall requires use_incentive_token_bank. DACCConfig
      construction forwards dacc_goal_readout_weight.
    ree_core/cingulate/dacc.py: DACCConfig.dacc_goal_readout_weight (0.0);
      DACCAdaptiveControl.forward gains candidate_goal_proximity kwarg -> bundle
      "goal_readout"; DACCtoE3Adapter restructured so the goal-readout term
      (bias -= dacc_goal_readout_weight * goal_readout; proximity high -> favoured,
      REE lower-is-better) is added INDEPENDENTLY of dacc_weight (so a
      goal-conditioned consumer works even when the legacy dACC bias is off),
      after the dacc_bias clamp; skipped (bit-identical) when weight 0 / readout None.
    ree_core/utils/config.py: from_dims surfaces use_cue_recall / cue_recall_gain
      / cue_recall_min_proximity (GoalConfig) + use_mech_consume +
      dacc_goal_readout_weight (REEConfig); REEConfig dataclass fields added.
    experiments/_harness.py: StepHarness auto cue-perception -- when use_cue_recall,
      derive the strongest-perceived resource type from the SD-049 per-type
      proximity field views (resource_field_view_<name>, argmax over types) and,
      if >= cue_recall_min_proximity, call cue_recall_wanting each step
      (best-effort, try/except; bit-identical no-op when off / bank absent / env
      emits no per-type views). The forced-cue diagnostic calls the primitive
      directly.
  L6 data flow: perceived cue type k (no benefit) -> bank token[k] -> wanting[k]
    = base_value[k]*(1+kappa*per_axis_drive[k]) -> GoalState.cue_pull(z_object[k],
    cue_recall_gain*clamp(wanting)) -> z_goal moves toward k -> E3 goal_proximity
    + MECH-295 approach bias (existing, unchanged) raise pre-consummatory approach
    toward k. Identity-matched + drive-specific (specific PIT).
  L7 data flow: select_action -> per-candidate goal_proximity to object-bound
    z_goal -> dacc(candidate_goal_proximity=...) -> bundle["goal_readout"] ->
    DACCtoE3Adapter bias -= dacc_goal_readout_weight*goal_readout -> dACC->E3
    score_bias becomes object-discriminative.
  Backward compatible: use_cue_recall=False + use_mech_consume=False +
    cue_recall_gain unused + dacc_goal_readout_weight=0.0 by default. Full
    contracts: SD-057 v1 6/6, phase-2 new test_sd_057_phase2_cue_recall_consume.py
    5/5, dACC 5/5, step-harness 7/7, 750/757 full (7 pre-existing local-git-env
    runner-conflict-recovery fails "Not a valid object name master", unrelated) +
    7/7 preflight. Unit + agent smoke 2026-06-04: cue_recall fires + moves z_goal,
    sim_mode no-op, preconditions raise, dACC OFF bit-identical / ON favours
    high-proximity candidate.
  MECH-094: cue_recall_wanting(simulation_mode=True) is a no-op (replay must not
    move z_goal via a cue); dACC readout is waking-only. No replay write surface.
  Phased training: N/A (pure-arithmetic nudge + scalar bias; no learned params).
  Biological basis: Berridge 2009 cue-triggered wanting; Corbit/Balleine 2005/2011
    specific PIT; Schultz 1997/98 DA-transfer-to-cue (L6); Balleine & O'Doherty
    2010 goal-conditioned approach_commit (L7).
  Validation experiment: V3-EXQ-637 forced-cue diagnostic (claim_ids=[],
    decoupled from GAP-2 like V3-EXQ-636) queued via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md
    (Phase-2 section). Plan-of-record: goal_pipeline_plan.md GAP-7.
  See SD-057 (v1 parent entry above), MECH-347 (L6) / MECH-348 (L7), MECH-344/345/346
    (v1 L2-L4), MECH-295 (approach bridge; L6 downstream), SD-032b/dACC (L7 host),
    SD-049 (per-type tag + per-axis drive + proximity views), MECH-229/MECH-117/ARC-030
    (downstream; stay v3_pending), MECH-094 (call-site scoping).
