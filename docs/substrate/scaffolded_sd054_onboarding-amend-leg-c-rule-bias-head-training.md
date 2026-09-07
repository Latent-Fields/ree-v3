## scaffolded_sd054_onboarding AMEND: Leg C rule_bias_head training (commitment_closure:GAP-4) (2026-06-16)
- scaffolded_sd054_onboarding rule-bias-training amend -- IMPLEMENTED 2026-06-16.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change -- the block calls the EXISTING SD-033a/ARC-062 GAP-D
  lateral_pfc.compute_bias / bias_head_parameters substrate, landed 2026-05-17).
  commitment_closure:GAP-4; the third leg of the SD-034 commitment-closure-control-plane
  (Legs A env-completion-hook + B de-commit-refractory landed 2026-06-12; Leg C was
  marked "experiment-side" and never actually built). Routed by the confirmed
  failure_autopsy_SD-034-closure-control-plane-d_2026-06-13.
  ROOT CAUSE (code-confirmed by the autopsy): the V3-EXQ-460d/468d *d retests set
  lateral_pfc_train_rule_bias_head=True (un-zeroing the GAP-D head's last Linear) but
  NEVER added it to any optimizer -- grep for optim|Adam|.backward(|bias_head_parameters
  returned ZERO matches in either script beyond the config mention. So the head stayed
  at random init: the rule_state handed to the SD-034 ClosureOperator carried no
  task-shaped magnitude -> the automatic rule-stability detector stayed inert and the
  closure-coupled de-commit had no net authority over the MECH-090 latch (460d C2_beta_release
  / C4 FAIL: ON latch occupancy >= OFF on seeds 43/44) and the agent committed-without-beta
  on 2/3 seeds (468d total_beta_elevated=0). NOT a falsification of SD-034/MECH-261.
  THE FIX (no-op-default; bit-identical OFF): a scaffold_train_rule_bias_head leg that
  trains agent.lateral_pfc.bias_head_parameters() during P1 (goal-unfrozen, ecological-
  contact, commitment forms) via the V3-EXQ-598b outcome-coupled E3-gradient REINFORCE
  pattern -- mirroring the existing scaffold_train_harm_pathway leg. Episode-level (not
  per-step like the harm pathway): run_p1 builds the optimizer + a persistent runtime
  (outcome buffer + EMA return baseline) via Scheduler._make_rule_bias_pathway; each
  _train_episode records a (candidate_features=world_states[1] of the leading n_probe
  candidates, selected-candidate index) snapshot every N steps and accumulates the
  episode return (-harm); at episode end _rule_bias_episode_update takes one Adam step --
  advantage = ep_return - EMA baseline; bias = lateral_pfc.compute_bias(candidate_features)
  recomputed (gradient flows into the head); loss = mean(-adv * log_softmax(-bias/T)[sel]);
  grad-clip 1.0; step. New module helpers: _rule_bias_params (the trainable-head guard),
  _new_rule_bias_diag, _build_rule_bias_snap, _selected_candidate_index,
  _rule_bias_reinforce_loss (the 598b _lpfc_reinforce_loss, parameterised off cfg),
  _rule_bias_episode_update.
  REQUIRES the agent built with use_lateral_pfc_analog=True AND
  lateral_pfc_train_rule_bias_head=True (the GAP-D un-zero flag); with the head
  zeroed-and-frozen OR no lateral_pfc, _rule_bias_params returns [] -> optimizer None ->
  the leg is a clean inert no-op (a misconfig is surfaced as rule_bias_pathway_enabled=False
  on the P1 manifest, never silently trains the baseline-OFF head).
  Config (ScaffoldedSD054OnboardingConfig, all no-op default -> bit-identical OFF; NOT
  surfaced through REEConfig.from_dims, matching the scaffold_train_harm_pathway / SD-054
  env-only scheduler-config precedent -- the 460e experiment sets them on the scheduler
  config directly): scaffold_train_rule_bias_head (False, master) + scaffold_rule_bias_lr
  (5e-4) + _batch_size (32) + _record_every_n_steps (4) + _outcome_buf_max (512) +
  _n_probe_candidates (8) + _policy_temperature (1.0) + _adv_min_threshold (0.005) +
  _ema_decay (0.9) (the 598b constants). P1OnboardingResult gains rule_bias_pathway_enabled
  + rule_bias_diag (counters + live per-candidate |bias| samples for the non-vacuity gate).
  Backward compatible: scaffold_train_rule_bias_head=False by default -> _make_rule_bias_pathway
  returns (None,None,None); the per-step snap collection + per-episode REINFORCE are skipped;
  the _train_episode early-return-on-done was refactored to for/else + single post-loop return
  (value-identical). 109/109 scaffolded contracts (102 prior + 7 new C17: config no-op /
  pathway off->None / on->Adam over the head params / inert-when-zeroed-or-no-lpfc /
  snap+selection-match / REINFORCE loss gradient reaches the head [the load-bearing inversion
  of the 460d gap] / run_p1 trains ON-vs-bit-identical-OFF) + 7/7 preflight PASS;
  v3_exq_460d --dry-run unchanged (leg off). Activation smoke 2026-06-16 (run_p1, tiny scale,
  leg ON): rule_bias_head last-Linear max|dW|=0.0015 > 0 (head TRAINS -- the 460d bug
  inverted), 3 REINFORCE steps over 57 snaps, mean per-candidate |bias|=0.039 (non-trivial,
  vs the ~0 the untrained 460d head produced); leg OFF -> max|dW|=0.0 exactly (bit-identical).
  Phased training: correctly phased by construction -- P0 warms the encoder (goal frozen),
  the rule_bias_head trains in P1 AFTER warmup (the 598b P0->P1 discipline). No new encoder
  head; the head is the existing GAP-D substrate. MECH-094: N/A -- the REINFORCE training is
  on the waking P1 training loop (no simulation/replay write surface; lateral_pfc.update keeps
  its existing MECH-319 simulation gate). Evidence-staleness (Step 8.5): NOT triggered --
  no-op-default flag; every existing experiment uses the default (leg off), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  ML/AI engineering notes: outcome-coupled REINFORCE on a policy bias head carries the
  standard high-variance-gradient hazard -- mitigated exactly as 598b: EMA return baseline
  (advantage = return - baseline), an absolute advantage-floor skip (near-baseline episodes
  carry no signal), grad-norm clip 1.0, and an episodic outcome buffer for batched replay.
  The compute_bias +/-bias_scale clamp can saturate the gradient at random init (the SD-033b
  GAP-8 clamp note); the per-candidate variation + small bias_scale keep some candidates
  in-band (598b-proven for this exact head).
  Validation experiment: V3-EXQ-460e (supersedes 460d), queued via /queue-experiment -- the
  closure-control-plane re-run with scaffold_train_rule_bias_head=True +
  lateral_pfc_train_rule_bias_head=True + a non-cap-pinned ON<OFF latch-occupancy-drop DV for
  C2_beta_release + a beta-engagement non-vacuity gate + a rule_bias-magnitude readiness gate
  (rule_bias_diag mean |bias| > floor, else substrate_not_ready_requeue). Acceptance per the
  autopsy failure record: ON<OFF de-commit on a non-cap-pinned statistic on >=2/3 seeds with
  beta-engagement met. substrate_queue commitment-closure-control-plane ready STAYS false
  until 460e scores a contributory PASS.
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md
  (Leg C rule-bias-training amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_SD-034-closure-control-plane-d_2026-06-13.{md,json}.
  See SD-034 (parent closure operator + commitment-closure-control-plane Legs A/B),
  SD-033a GAP-D / ARC-062 GAP-D lateral_pfc_train_rule_bias_head (the substrate this trains;
  landed 2026-05-17), V3-EXQ-598b (the REINFORCE pattern mirrored), scaffold_train_harm_pathway
  (the sibling scaffold training leg whose structure this mirrors), MECH-090 (the latch the
  de-commit authority acts on), MECH-260 (No-Go; 460d supports), MECH-261 (mode-conditioning;
  unexercised), MECH-268 (dACC PE; 468d), V3-EXQ-460d/468d (the FAILs this addresses),
  V3-EXQ-460e (validation), MECH-094 (N/A).
