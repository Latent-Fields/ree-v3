## scaffolded_sd054_onboarding AMEND: Stage-H harm-pathway training (603i nav/survival-competence ceiling) (2026-06-09)
- scaffolded_sd054_onboarding harm-pathway-training amend -- IMPLEMENTED 2026-06-09.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change -- the block calls EXISTING ree_core harm heads).
  Routed by failure_autopsy_V3-EXQ-603i_2026-06-08 (PRIMARY: nav/survival-competence
  ceiling; ARM_NAV_CONTROL spawn-in-reef G_H=0.0) + the 603g/624c/651a cluster autopsy
  ("deeper than budget") + lit verdict targeted_review_hazard_avoidance_learning.
  ROOT CAUSE (code trace + empirical probe 2026-06-09, NOT budget): across the ENTIRE
  curriculum (Stage-0/0b/P0/Stage-H/P1/P2), _train_episode optimizes ONLY E1
  (compute_prediction_loss, LSTM MSE) + E2.world_transition/world_action_encoder. The
  hazard-avoidance VALUATION pathway is never in any optimizer:
  E3.harm_eval_head(z_world) -- the harm cost that scores EVERY candidate trajectory
  in E3.select (e3_selector.py:419/564) -- is a near-constant ~0.523 (random init;
  measured range [0.522,0.524] over 300 states), the HarmEncoder z_harm output is
  constant (zero variance vs proximity), and `_harm_signal` from env.step is DISCARDED.
  So the agent navigates a RANDOM harm landscape and dies even handed the reef refuge
  (probe: Stage-H survival slope -0.94 steps/ep == no learning; 24/25 episodes die
  early; median last-window 23 vs gate 75). More budget cannot train a head that is
  not in the loss. This is why SD-058/MECH-357 (IA gate) + SD-059/MECH-358 (bridge)
  were "engaged but insufficient": they bias action selection within whatever harm
  landscape E3 hands them, and that landscape is noise.
  THE FIX (user-confirmed FULL scope + encoder co-train): train the existing-but-
  untrained harm pathway during P0 + Stage-H, supervised by the env hazard-proximity
  label (harm_obs centre, SD-010/SD-018) + accumulated-harm scalar (SD-011). Four
  independently-toggleable terms (Q-044 / MECH-314a-style ablatability):
    (1) harm_eval(z_world): E3.harm_eval_head + the z_world encoder -- the proximity
        MSE backprops INTO latent_stack (SD-018 semantics) so z_world becomes
        hazard-discriminative (head-only fails on the flat z_world the probe measured).
        LOAD-BEARING: makes the trajectory-rollout harm landscape (E2.world_forward ->
        harm_eval_head) predictive.
    (2) z_harm sensory: HarmEncoder + E3.harm_eval_z_harm_head (SD-010) on the same label.
    (3) z_harm_a affective: AffectiveHarmEncoder via compute_harm_accum_loss (SD-011) --
        gives MECH-279 PAG + SD-058 IA gate + SD-059 bridge a TRAINED threat signal.
    (4) E2_harm_s forward: E2HarmSForward (ARC-033) on FROZEN (detached) z_harm_s for
        multi-step harm lookahead.
  Config (ScaffoldedSD054OnboardingConfig, all no-op default): scaffold_train_harm_pathway
  (False, master) + scaffold_train_harm_eval_head / _z_harm_sensory / _z_harm_affective /
  _e2_harm_s_forward (True; consulted only when master on) + scaffold_harm_pathway_lr
  (1e-3) + scaffold_harm_pathway_in_p0 (True) + scaffold_harm_s_buf_max (2000).
  New module helpers: _hazard_proximity_target / _accumulated_harm_target (target
  extraction from obs_dict), _harm_pathway_params (deduped param union per enabled
  term), _harm_pathway_step (one co-train step on the pristine post-sense latent + a
  detached-buffer E2_harm_s step), _measure_harm_discriminativeness (post-training
  NON-VACUITY readout). Scheduler._make_harm_pathway builds the optimizer/buffer/diag;
  run_p0 (when in_p0) + run_hazard_avoidance thread train_harm + harm_opt + harm_s_buf
  + harm_diag through _train_episode; run_hazard_avoidance also runs the
  discriminativeness probe at the end. HazardAvoidanceResult / P0OnboardingResult gain
  harm_pathway_enabled + harm_pathway_diag (+ harm_discriminativeness on Stage-H).
  Terms 2 + 4 require the agent built with use_harm_stream=True (sensory z_harm) +
  use_e2_harm_s_forward=True; inert no-op (correctly skipped) when z_harm / e2_harm_s
  are absent -- the 603k validation config enables both so all four terms engage.
  Backward compatible: master OFF -> train_harm False, harm_opt None, harm block + the
  Stage-H discriminativeness probe skipped -> bit-identical. 97/97 scaffold contracts
  (91 prior + 6 new C15) + 7/7 preflight PASS. Activation smoke 2026-06-09 (ARM_NAV_CONTROL
  spawn-in-reef, harm pathway ON, reduced budget): survival_gate_passed=True,
  median_last_window=80.0 (vs OFF probe 23.0, gate 75); survival slope +0.19 (vs OFF
  -0.94); early deaths 9/25 (vs OFF 24/25); harm_eval loss trains (n_train_steps
  586 P0 + 1511 Stage-H). The headline (G_H 0 -> survival_passed) confirms the fix.
  Phased training: the curriculum IS phased; harm_eval/z_harm proximity heads co-train
  with their encoder (single SD-010/SD-018 regression, no collapse risk); E2_harm_s
  trains on detached z_harm_s (ARC-033 P1 phasing). MECH-094: N/A -- waking training
  stream; no simulation/replay write surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (harm pathway off), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  Validation experiment: V3-EXQ-603k substrate-readiness diagnostic (claim_ids=[];
  HARM_OFF vs HARM_ON ablation on the 603i-INTACT base + nav-competence reef spawn;
  use_harm_stream=True + use_e2_harm_s_forward=True so all four terms engage). Acceptance
  per the 603i failure-record target: Stage-H G_H >= 2/3 (median last-window >= 75) with
  nav-to-safety handed AND a non-vacuity gate (harm_eval discriminativeness lifts above
  the flat baseline) AND the HARM_OFF arm reproduces G_H ~ 0. PASS unblocks the
  escape-affordance-bridge retest (the bridge can finally be scored once survival clears)
  + the GAP-2 survival-leg cohort. substrate_queue.ready STAYS false until 603k clears.
  Design doc: REE_assembly/evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md
  (Amend 2026-06-09 section). Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603i_2026-06-08.{md,json}.
  See scaffolded_sd054_onboarding (parent + prior amends incl Stage-H curriculum
  decomposition), SD-058/MECH-357 (IA gate -- bias side; this trains the valuation it
  navigates), SD-059/MECH-358 (escape bridge -- blocked on this survival leg),
  SD-010/SD-018 (z_world/z_harm hazard-proximity supervision), SD-011 (z_harm_a
  accumulated-harm), ARC-033 (E2_harm_s forward), MECH-279 (PAG freeze; keys on the
  now-trained z_harm_a), goal_pipeline:GAP-2 (the survival leg this closes),
  V3-EXQ-603i (the FAIL this addresses), V3-EXQ-603k (validation), MECH-094 (N/A).
