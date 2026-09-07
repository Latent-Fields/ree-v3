## SD-058 / MECH-357: instrumental-avoidance acquisition (ilPFC-analog freeze-suppression + avoidance action pathway) (2026-06-07)
- SD-058 (architecture) / MECH-357 (mechanism): defensive_action.instrumental_avoidance_acquisition
  -- IMPLEMENTED 2026-06-07 (substrate; v3_pending until the Stage-H validation
  EXQ PASSes). Closes the scaffolded_sd054_onboarding Stage-H / P1 survival-leg
  gap (V3-EXQ-603g G_H 0/3; goal_pipeline GAP-2). Routed by
  failure_autopsy_V3-EXQ-603g-624c-651a_2026-06-07 + the lit verdict
  evidence/literature/targeted_review_hazard_avoidance_learning/SYNTHESIS.md
  (SD-035 x3 + MECH-279 + SD-054): the fix is STRUCTURAL, not budgetary. REE has
  the Pavlovian/defensive REACTION side (SD-035 amygdala salience + MECH-279 PAG
  freeze) but lacked the instrumental-ACQUISITION side. Moscarello & LeDoux 2013:
  active avoidance learning is the resolution of a Pavlovian-instrumental
  conflict -- learning to avoid REQUIRES the infralimbic PFC to SUPPRESS
  CeA-driven freezing (ilPFC lesion -> more freezing, less avoidance). A
  freeze-only substrate freezes instead of learning to avoid -- the 603g G_H 0/3
  signature.
  Module: ree_core/pfc/infralimbic_avoidance_gate.py (InstrumentalAvoidanceGate +
  InstrumentalAvoidanceGateConfig + InstrumentalAvoidanceGateOutput). Pure-
  arithmetic regulator (no nn.Module, no trained params, no gradient flow);
  sibling to the SD-035 CeA/BLA, MECH-279 PAG, MECH-313 NoiseFloor, MECH-320
  TonicVigor pattern. Lives in ree_core/pfc/ alongside lateral_pfc_analog.py
  (dlPFC) + ofc_analog.py -- infralimbic is the third PFC subdivision analogue.
  Three pieces, all behind use_instrumental_avoidance (default False, bit-identical):
    (a) INSTRUMENTAL-AVOIDANCE ACTION pathway: a per-candidate E3 score-bias
        composed last in the dacc_score_bias chain (after MECH-320 vigor +
        MECH-353 assert) that, under retained threat (z_harm_a), PENALISES the
        no-op/freeze class proportional to effective_efficacy * threat_scale --
        releasing the instrumental action. Does NOT compute the escape direction
        (E3's existing harm gradient ranks the directed candidates; ARC-007-
        strict-compatible). compute_action_bias(z_harm_a_norm, action_classes,
        noop_class) -> [K].
    (b) ilPFC FREEZE-SUPPRESSION gate: at the MECH-279 application site in
        select_action, when freeze_active AND should_suppress_freeze(z_harm_a_norm)
        (effective_efficacy * threat_scale >= suppression_threshold), the no-op
        override is SKIPPED so the agent takes its selected instrumental action.
        Inert when use_pag_freeze_gate=False; the action-pathway half (a) still
        operates (freezing is a passive no-op the bias penalises).
    (c) AVOIDANCE-EFFICACY LEARNING (eligibility trace; the acquisition): a scalar
        avoidance_efficacy in [0,1] starting at initial_efficacy (0.0 = freeze-
        default). update(z_harm_a_norm, action_was_directed) in sense() (one-tick
        lag): a directed action under threat that DROPS z_harm_a credits efficacy
        (EMA toward 1); freezing/failed-avoidance under threat decays it. The
        gradual developmental acquisition (Debiec & Sullivan 2017 / Thompson 2008).
        PERSISTS across episodes within a stage -- reset() clears ONLY the
        within-episode threat trace, NOT the learned efficacy.
  PROTECTIVE-SCAFFOLD anneal (the curriculum; SECONDARY): effective_efficacy =
  max(avoidance_efficacy, scaffold_floor). The Stage-H curriculum
  (scaffolded_sd054_onboarding) sets a high scaffold_floor early and anneals it
  down as the learned efficacy takes over (maternal-buffering / Turchetta 2020
  reset-curriculum analogue). Budget escalation is the explicitly-secondary lever.
  Config (REEConfig + from_dims, all no-op default): use_instrumental_avoidance
  (False), avoidance_learn_rate (0.05), avoidance_leak_rate (0.02),
  avoidance_initial_efficacy (0.0), avoidance_scaffold_floor (0.0),
  avoidance_threat_floor (0.1), avoidance_threat_ref (0.5),
  avoidance_efficacy_reward_floor (1e-4), avoidance_action_bias_gain (0.1),
  avoidance_bias_scale (0.1), avoidance_suppression_threshold (0.5),
  avoidance_noop_class (0).
  Agent wiring (ree_core/agent.py): instantiate self.instrumental_avoidance when
  the master flag is on; _ia_last_action_directed cache; update() in sense()
  after _update_blocked_agency (reads new_latent.z_harm_a; MECH-094 no-op under
  hypothesis_tag); action-bias composed after the MECH-353 ba_bias block;
  freeze-suppression at the MECH-279 site; directed-action cache after
  self._last_action = action; reset() clears the within-episode trace + cache
  (preserves learned efficacy).
  Curriculum wiring (experiments/scaffolded_sd054_onboarding.py): run_hazard_avoidance
  (Stage-H) protective-scaffold anneal under scaffold_avoidance_driver_enabled
  (default False) + scaffold_avoidance_scaffold_floor_start (0.8) /
  scaffold_avoidance_scaffold_floor_end (0.0); HazardAvoidanceResult gains
  avoidance_driver_enabled + avoidance_gate_state so the manifest can confirm
  acquisition (efficacy rose, freeze suppressed) rather than survival-by-chance.
  LOAD-BEARING PREREQUISITE (found 2026-06-07): the legacy scaffold called
  sense(body, world) with NO harm args, so z_harm_a was None across the WHOLE
  curriculum -- leaving MECH-279 PAG, SD-035 amygdala AND the SD-058/MECH-357 gate
  all INERT (they key on z_harm_a). New scaffold flag scaffold_feed_harm_stream
  (default False -> bit-identical) + module helper _sense_with_optional_harm feed
  the env harm_obs + harm_obs_a into sense() so z_harm_a is populated (~0.34 in
  Stage-H). The avoidance-driver experiments set it True; without it the gate has
  no threat signal to learn from.
  DISTINCT from the reflexive escape-from-freeze levers (do NOT read as a
  duplicate): SD-037 override_signal (orexin) raises the PAG exit threshold and
  MECH-281 lowers the MECH-091 urgency-interrupt -- BOTH reflexive threat/arousal-
  driven escape. MECH-357's ilPFC suppression is gated by LEARNED avoidance-
  efficacy (eligibility trace), bootstrapped by the protective-scaffold floor --
  the acquisition mechanism (Moscarello & LeDoux), not a reflex.
  Backward compatible: use_instrumental_avoidance=False by default ->
  agent.instrumental_avoidance is None; sense update + select_action bias +
  freeze-suppression all skipped -> bit-identical. Stage-H driver gated by
  scaffold_avoidance_driver_enabled AND requires the gate. 912 contracts + 7/7
  preflight PASS; 7 new contracts in
  tests/contracts/test_mech_357_instrumental_avoidance.py + 4 C13 in
  tests/contracts/test_scaffolded_sd054_onboarding.py. Bit-identical OFF verified
  (default vs use_instrumental_avoidance=True at zero efficacy/floor: identical
  action stream, no RNG, zero bias).
  Phased training: N/A (pure-arithmetic regulator; no learned parameters; the
  "learning" is the eligibility-trace EMA, not an encoder head). MECH-094: both
  compute methods + update() no-op under simulation_mode (replay must not credit
  avoidance or suppress freeze on imagined outcomes). Evidence-staleness: NOT
  triggered (no-op-default flag; no dependent claim's measured mechanism changed).
  Validation experiment: V3-EXQ-603h substrate-readiness diagnostic (claim_ids=[],
  queued via /queue-experiment). LITERAL Moscarello & LeDoux lesion-vs-intact,
  2-arm, 3 seeds: BOTH arms have MECH-279 PAG (tuned to z_harm_a~0.34) + the fed
  harm stream; ARM_LESION (PAG, no gate -> freezes) vs ARM_INTACT (PAG + ilPFC
  gate + driver -> suppresses freeze + acquires avoidance). PRIMARY: G_H_INTACT
  >= 2/3 AND G_H_INTACT_frac > G_H_LESION_frac. Non-vacuity preconditions: PAG
  freezes on LESION (pag_n_commits>0) AND the gate engages+suppresses on INTACT
  (n_credit+n_decay>0 AND n_freeze_suppressed>0), else substrate_not_ready_requeue.
  Dry-run engages the full chain (PAG freezes 14, gate suppresses 8, readiness
  met; primary not met at dry scale as expected). PASS unblocks the goal_pipeline
  GAP-2 retest cohort + the pending_retest_after_substrate claims ARC-060 /
  MECH-320 / ARC-068 / SD-054-readiness. substrate_queue scaffolded_sd054_onboarding
  ready STAYS false until then.
  Design doc: REE_assembly/docs/architecture/sd_058_instrumental_avoidance_acquisition.md
  See SD-035 (amygdala BLA/CeA -- defensive-reaction salience side this builds the
  acquisition side onto), MECH-279 (PAG freeze-gate -- the freeze output the ilPFC
  gate suppresses), SD-011 (z_harm_a -- threat signal), MECH-320 / MECH-314
  (sibling policy-layer score-bias regulators), MECH-353 (blocked-agency assert
  pole -- distinct; SD-058 resolves freeze-vs-avoid under threat), MECH-094 (call-
  site scoping), scaffolded_sd054_onboarding (Stage-H curriculum driver),
  goal_pipeline:GAP-2 (the survival leg this closes), V3-EXQ-603g (the FAIL this
  addresses).
