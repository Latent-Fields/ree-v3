## MECH-451: finer-channel-granularity E3 selection-gating (the cheap V3 rung BETWEEN ARC-108's single global w_chan and ARC-110's V4 segregated loops; explode the compressed score_bias blend into separately-learnable per-head channels) (2026-06-24)
- MECH-451: selection.intermediate_channel_granularity -- IMPLEMENTED 2026-06-24 (substrate;
  MECH-451 stays candidate / substrate_conditional / implementation_phase=v3 -- this PROMOTES
  NOTHING; EXP-0391 is the validation falsifier, NOT queued in this build session). The cheap
  V3-tractable rung the sd_v4_loop_segregation "Sequencing" section + ARC-110's notes say must be
  EXHAUSTED FIRST: if finer channels convert non-motor influence to committed action, the
  F-dominance conversion ceiling (MECH-439) is REPRESENTATIONAL COMPRESSION and the expensive V4
  ARC-110 loop-segregation build is PRE-EMPTED. Routed by failure_autopsy_V3-EXQ-700b_2026-06-24
  (the V3-EXQ-700 lineage could not validly test learned-gating conversion on the single arena;
  the V4 escalation + this V3 rung were opened concurrently). Second worked application of ARC-106
  (reuse-the-mechanism / parallel buffer / no-op-default + the cargo-cult ablation guard).
  PROBLEM: ARC-108 added the first LEARNING afferent to the ARC-107 arbitration layer -- a SINGLE
  global learned w_chan over the modulatory channels feeding the E3 _modulatory_accum site. But at
  that site "score_bias" is ALREADY the COMPRESSED dACC+lPFC+OFC+MECH-295+MECH-320+gated_policy
  blend, summed UPSTREAM in agent.py before reaching select() (the ARC-108 comment names "a finer
  per-head channel split" as the documented follow-on, out of step-1 scope). A learner that can
  only re-weight a pre-compressed blend cannot dissociate the control functions compression fused.
  MECH-451 IS that follow-on.
  THE BUILD (strict ADDITIVE extension behind a no-op-default master flag; the ARC-108 w_chan path
  is BYTE-IDENTICAL -- ARC-106 G2 reuse-the-mechanism, parallel buffer, zero risk to the V3-frozen
  substrate):
    (1) FINER REGISTRY (ree_core/predictors/e3_selector.py): the single ARC-108 "score_bias" slot
      exploded into _FCG_CHANNEL_NAMES = ("ofc","dacc","lpfc","vigour","liking","gated_policy",
      "residual","mech341","route"). The six FINER_NAMED_CHANNELS map onto the existing per-head
      biases (OFC<-SD-033b, dACC<-SD-032b adapter, lateral-PFC<-SD-033a, vigour<-MECH-320,
      liking<-MECH-295, gated_policy<-ARC-062); mech341/route preserved unchanged. "residual" =
      score_bias - sum(named present) (computed by SUBTRACTION in select()) captures everything
      ELSE summed into score_bias (MECH-314 curiosity / MECH-353 blocked-agency / SD-058 avoidance
      / SD-059 escape / any future term) -> the decomposition is EXHAUSTIVE so sum(finer) ==
      score_bias EXACTLY (the bit-identical-at-init guarantee in the authority/shortlist path).
    (2) PARALLEL LEARNED BUFFER w_chan_finer (register_buffer, NOT nn.Parameter) + _fcg_elig_trace
      + _fcg_pending/_fcg_last_delta/_fcg_n_updates, sized to the finer registry. Init at
      _LCG_W_INIT = ln(e-1) so softplus(w_chan_finer[c]) == 1.0 -> reproduces the compressed blend
      EXACTLY at init (bit-identical even when ON, until the weights train apart). V-hat_t SHARED
      with the ARC-108 _lcg_value_baseline (the two gating modes are mutually exclusive -- A1 vs A2
      arms). The ARC-108 w_chan / _lcg_elig_trace are UNTOUCHED.
    (3) REGISTRY-AGNOSTIC machinery: select() picks the active (_FCG vs _LCG) registry + buffer via
      _fcg = use_finer_channel_gating. The score_bias add-site registers one _lcg_term per present
      finer channel + the residual (finer) instead of the single "score_bias" term (legacy); the
      mech341/route add-sites use the active index. scores = raw + summed score_bias in BOTH modes
      (the authority-OFF selection path + the scores tensor are unchanged); only the recomposed
      _modulatory_accum the authority/shortlist consumes differs. The recompose
      (sum_c softplus(w_buf[c])*term), the eligibility (_fcg_elig_trace[c] += |term[selected]|),
      and the three-factor update (Delta w_chan_finer[c] = eta*learn_signal*elig_c*asym, ONE shared
      signed RPE delta_t with ARC-108) all ride the active buffer. As the finer weights diverge
      under per-channel credit, _modulatory_accum becomes a per-candidate vector != the uniform sum
      -- the conversion MECH-451 tests.
    (4) AGENT WIRING (ree_core/agent.py select_action): _fcg_channels dict (None unless the flag is
      on) captures each finer constituent's un-summed [K] bias at its existing add-site (the SAME
      tensors already summed into dacc_score_bias: dacc post-e3_gate / gp_bias / lpfc_bias /
      ofc_bias / m295_bias / tv_bias); passed as score_bias_channels=... into e3.select() ONLY when
      the flag is on (version-layering guard -- the default V3 path never sends the kwarg, so an
      older e3.select cannot raise; same doctrine as the DR-12 / Go-No-Go guards).
    (4-DEFECT, FIXED 2026-06-28 -- ARC-110 V3-EXQ-707 autopsy) The _fcg_channels builder gate read
      the TOP-LEVEL self.config.use_finer_channel_gating, which is NEVER set anywhere in ree_core
      (always False), while the consumer gate at the e3.select() call site reads
      config.e3.use_finer_channel_gating. Net: _fcg_channels was ALWAYS None -> the named channels
      (ofc/dacc/lpfc/vigour/liking/gated_policy) NEVER reached the selector; only the lumped
      residual/mech341/route did. So EVERY finer-channel experiment (704/704b/706*/707/708) ran with
      MECH-451's named decomposition DEAD -- A2_FINER_CHANNELS was only a residual/mech341/route
      3-way split, NOT the named decomposition the claim asserts. Fixed agent.py:5238 to read
      config.e3.use_finer_channel_gating (matching the consumer). Regression guard
      tests/test_arc110_loop_segregation.py::TestFinerChannelsReachSelector (fails pre-fix, passes
      post-fix). MECH-451 is effectively UNTESTED (pending_retest_after_substrate). NB a genuine
      retest is itself gated on the named bias heads carrying per-candidate range -- they currently
      emit per-candidate-FLAT output (OFC input range 0.028 -> output 0.0; MECH-191 phasic gap),
      see REE_assembly sd_v4_loop_segregation.md "VALIDATION + DEFECT 2026-06-28".
  Config (REEConfig + from_dims + E3Config, no-op default -> bit-identical OFF):
  use_finer_channel_gating (False). REUSES the ARC-108 learning knobs (learned_channel_gating_eta /
  _elig_decay / _value_baseline_beta / _asym_potentiation / _asym_depression / _rpe_mode) so
  A1_GLOBAL_WCHAN vs A2_FINER differ ONLY in channel granularity (EXP-0391's single-variable design).
  Backward compatible: use_finer_channel_gating=False by default -> score_bias_channels=None ->
  legacy single "score_bias" term -> bit-identical; the A1 use_learned_channel_gating path is
  unchanged. preflight 8/8 + 1274 contracts PASS (the only 3 failures are the documented pre-existing
  flakes -- control_vector C4 [order-dependent] + 2 runner_fail_branch [local-git-env]; the runner
  fail CONFIRMED failing identically with the e3/agent/config edits stashed). 12 new contracts in
  tests/contracts/test_mech451_finer_channel_gating.py (C1 config no-op + from_dims + softplus-unity
  init at the finer registry size / C2 finer-ON-at-init EXACT-equal scores+selection to legacy +
  ARC-108 w_chan untouched / C3 w_chan_finer MOVES under a non-flat delta_t while w_chan stays at
  init / C4 MECH-094 simulation no-op / C5 residual exhaustiveness -- only some named channels
  supplied, residual absorbs, recompose still reproduces score_bias / C6 dissociation -- different
  per-channel eligibility drives the weights APART [range>0] while identical-eligibility channels
  move together [the degenerate re-labelled-blend the noise guard catches] / C7 envelope intact --
  a finer weight cannot re-admit an F-excluded candidate). Agent-level activation smoke (real
  REEAgent via from_dims, CausalGridWorldV2, dacc+lpfc+ofc+vigour+authority on): bit-identical
  action stream finer-OFF vs finer-ON-at-init; w_chan_finer (9,) present; the three-factor update
  fired on waking ticks (_fcg_n_updates=5); ARC-108 w_chan untouched.
  Phased training: N/A (local non-backprop three-factor rule; reuses the already-trained valuation
  heads R_t = benefit_eval - harm_eval; no encoder head, no collapse risk). MECH-094: waking-only --
  eligibility recorded only on a non-simulation select(); the finer three-factor update gated on a
  pending waking finer trace (a replay/DMN tick leaves _fcg_pending False -> no write). Inherited
  from the ARC-108 path. Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every
  existing experiment uses the default (finer off), so no dependent claim's measured mechanism
  changed. KEEP all evidence.
  ARC-106 cargo-cult guard (built into EXP-0391, NOT the substrate): the degeneracy hazard is "the
  finer channels move IDENTICALLY = the compressed blend re-labelled." (a) the non-degeneracy
  readiness gate (dissociable cross-channel w_chan_finer variance via fcg_w_chan_finer_range/_std +
  a divergent GAP-A candidate pool) self-routes substrate_not_ready_requeue if unmet; (b) the
  load-bearing ablation = A1_GLOBAL_WCHAN (collapse-to-blend = one global w_chan over the sum) -- if
  A1 reproduces A2's lift the decomposition is NOT load-bearing; A1 IS an EXP-0391 arm.
  GOVERNANCE: PROMOTES NOTHING. MECH-451 stays candidate / substrate_conditional /
  implementation_phase=v3; ARC-108 / MECH-450 / MECH-439 / ARC-110 untouched. claims.yaml carries
  only an implementation_note.
  Validation experiment: EXP-0391 (manual_proposals.v1.json) -> a V3-EXQ-700-sibling on the
  GAP-A-ready foraging substrate (arms A0_ENVELOPE_ONLY / A1_GLOBAL_WCHAN / A2_FINER_CHANNELS /
  ARM_NOISE, settling W_lat OFF, landed arithmetic envelope a matched constant, SD-056-trained
  e2.world_forward + ARC-065 GAP-A candidate_summary_source=e2_world_forward divergent-pool
  precondition, PRIMARY DV = committed-action-class entropy). Queued via /queue-experiment (separate
  session). A2 lift beyond A1 -> representational compression -> pre-empts the V4 ARC-110 build; A2
  finer-weights-move-but-no-lift -> positive evidence FOR ARC-110.
  Design doc: REE_assembly/docs/architecture/mech_451_finer_channel_granularity.md.
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-700b_2026-06-24.{md,json}.
  See MECH-451 (this claim), ARC-108 (the learned-gating machinery this reuses; the single global
  w_chan over the compressed blend = the A1 reference), MECH-450 (the coupled settling step; W_lat
  OFF in this slice), ARC-110 / sd_v4_loop_segregation (the V4 per-loop build this is the cheap rung
  BEFORE; pre-empted on PASS, routed-to on no-lift), MECH-439 (the F-dominance conversion ceiling
  under test), ARC-107 (BG-constitution arbitration layer the learning composes inside),
  modulatory-bias-selection-authority / MECH-448 / MECH-449 (the F-bounded eligible set the
  re-weighted _modulatory_accum is arbitrated within; safety inherited), SD-033a/b / SD-032b /
  MECH-320 / MECH-295 / ARC-062 (the per-head biases the finer channels decompose), ARC-106
  (grounding framework; second worked application), V3-EXQ-700/700a/700b (the lineage that could not
  test conversion on the single arena), EXP-0391 (validation), MECH-094 (waking-only call-site scoping).
