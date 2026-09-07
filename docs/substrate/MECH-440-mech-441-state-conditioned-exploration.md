## MECH-440 / MECH-441: state-conditioned exploration -- propagating selection-head weight noise (NoisyNet) + model-disagreement directed curiosity (RND/Plan2Explore) (2026-06-27)
- MECH-440: selection.state_conditioned_self_annealing_noise_floor -- IMPLEMENTED 2026-06-27
  (substrate; stays candidate / substrate_ceiling / v3_pending -- PROMOTES NOTHING; falsifier queued,
  not scored). Module ree_core/policy/noisy_selection_head.py (NoisySelectionHead /
  NoisySelectionHeadConfig). The mechanistic refinement of MECH-313's NON-PROPAGATING temperature
  floor (V3-EXQ-687 r1a_entropy_only_artefact): factorised-Gaussian WEIGHT NOISE built per-candidate
  from each candidate's first-action vector and added into _modulatory_accum BEFORE the within-eligible
  argmin (and into the ARC-110 segregated-loop `final`), so it PROPAGATES into the committed action.
  mu FROZEN at 0 (pure noise injector -- isolates the falsifier). State-conditioned via the action
  activation; SELF-ANNEALS via a LOCAL confidence EMA on (1 - F-gap_norm). Lazy-built in e3_selector
  (action_dim from the candidate width). Config (E3Config): use_noisy_selection_head (default False;
  bit-identical OFF), noisy_selection_sigma_init (default 0.0 -> exactly-zero output -> bit-identical
  even ON), noisy_selection_weight, noisy_selection_anneal{,_floor,_ema_alpha}. Data flow: candidate
  first-actions -> NoisySelectionHead -> per-candidate bias -> _modulatory_accum + scores -> committed
  argmin. MECH-094: zero perturbation on simulation ticks. Phased training: NOT required (sigma is a
  buffer annealed by a local rule, not gradient-trained -- ARC-106 divergence #2). Diagnostics:
  noisy_selection_active / noisy_selection_bias_range (+ get_state()). Smoke-tested: sigma=0
  bit-identical; sigma>0 flips the committed within-eligible argmin across resamples (propagation
  confirmed); finiteness + EMA guards prevent any nan injection.
  Validation experiment: <EXQ pending Step 8 -- the corrected 4-arm tonic-noise falsifier on the 569i
  top-k + MECH-448 demotion stack, with loop-seg OFF vs ON arms to disambiguate injection-site from the
  single-arena/ARC-110 axis>.
- MECH-441: selection.model_disagreement_directed_curiosity -- IMPLEMENTED 2026-06-27 (substrate;
  stays candidate / substrate_ceiling / v3_pending -- PROMOTES NOTHING; FALSIFIER HELD, see below).
  Module ree_core/policy/model_disagreement.py (ModelDisagreementEnsemble / ModelDisagreementConfig):
  a standalone SMALL K-head ensemble of ResidualHarmForward delta-predictors over (z_world, action);
  per-candidate cross-head VARIANCE feeds E3 selection as a propagating per-candidate curiosity BONUS
  (a cost reduction, lower=preferred) added into _modulatory_accum -- unlike the V3-EXQ-590a broadcast
  EMA, this is per-candidate so it can change the committed argmin (WITHIN the F-eligible set; safety
  inherited). Self-annealing is INTRINSIC (variance -> 0 as the heads train). Built at the agent level
  (self.disagreement_ensemble) only when use_model_disagreement_curiosity AND n_disagreement_heads>=2;
  per-candidate disagreement computed in select_action and passed via the version-layering-guarded
  model_disagreement_per_candidate kwarg (default V3 path never sends it). Config: E3Config.
  use_model_disagreement_curiosity (default False), model_disagreement_weight (default 0.0),
  model_disagreement_mode/scale; LatentStackConfig.n_disagreement_heads (default 0 -> not built),
  disagreement_bootstrap_mask_prob, disagreement_learning_rate. Data flow: z_world + candidate
  first-actions -> K-head ensemble cross-head variance -> per-candidate bonus -> _modulatory_accum ->
  committed argmin. Phased training REQUIRED: train each head on the FROZEN z_world target
  (disagreement_ensemble.train_step(); P1 stop-gradient). MECH-094: waking-only no_grad read.
  Smoke-tested: ensemble disagreement supra-floor; per-candidate bonus steers the committed
  within-eligible argmin; sim-gated; train_step runs. FALSIFIER HELD (blocked_substrate) gated on
  ARC-110 validation V3-EXQ-707 returning contributory -- failure_autopsy_704b-706b-conversion-ceiling_
  2026-06-27 found the single-arena collapse (not the curiosity channel; 706b proved the channel works)
  is the binding constraint, so a MECH-441 run before ARC-110 is validated would re-derive the arena
  ceiling (vacuous FAIL). User decision 2026-06-27.
- Both: ARC-106 divergences logged in the grounding ledger -- (1) per-parameter sigma is one level
  below biology's systems-level tonic/phasic LC-NE mode gate; (2) sigma self-anneals via REE's LOCAL
  confidence EMA, not NoisyNet's RL gradient (REE does not backprop through E3 selection). Decision of
  record: REE_assembly/evidence/decisions/cpkt_tonic_exploration_noise_build_decision_2026-06-27.md.
  Design-of-record: REE_assembly/docs/architecture/state_conditioned_exploration_noise_floor.md
  (#mech-440 / #mech-441). See ARC-065 (parents), MECH-313 (440 refines), MECH-314 (441 refines),
  ARC-110 (the multi-arena substrate both build on top of), MECH-439 (the F-dominance ceiling),
  MECH-448 / 569i top-k (the SOTA conversion stack the falsifiers run on).
