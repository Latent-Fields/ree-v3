## ARC-110: parallel segregated cortico-BG-thalamic loops (motor / associative / limbic) + S2 in-layer null + ARC-109 D1/D2 split + MECH-452 loop-local traces (2026-06-27)
- ARC-110: selection.parallel_segregated_loops -- IMPLEMENTED 2026-06-27 (substrate; ARC-110 +
  ARC-109 + MECH-452 stay candidate / substrate_conditional / implementation_phase=v3 -- this
  PROMOTES NOTHING; the validation falsifier is queued, not yet scored). The full BG-thalamo-cortical
  loop substrate the conversion-ceiling lineage converged on from FOUR angles (V3-EXQ-700b/700c
  learned-gating+settling; V3-EXQ-704b finer-channel; V3-EXQ-706b double-gated curiosity), all of
  which showed the V3 single E3 foraging arena structurally denies (a) committed-action-class
  conversion from non-motor channels and (b) a valid same-layer null. Gate CLEARED by V3-EXQ-704b
  FAIL-to-convert (the pre-registered positive-evidence-FOR-ARC-110 outcome: compression is NOT the
  binding constraint). Cluster autopsy: REE_assembly/evidence/planning/
  failure_autopsy_704b-706b-conversion-ceiling_2026-06-27.{md,json}. Design-of-record:
  REE_assembly/docs/architecture/sd_v4_loop_segregation.md. Third worked application of ARC-106
  (reuse-the-mechanism / parallel buffers / no-op-default + load-bearing-vs-decorative ablation).
  THE BUILD (strict additive extension behind no-op-default flags; every existing path -- ARC-108
  w_chan, MECH-451 finer, MECH-450 settling, single-arena argmin -- is BYTE-IDENTICAL OFF):
    (1) SEGREGATED LOOPS (ree_core/predictors/e3_selector.py _segregated_loop_arbitrate, gated on
      config.e3.use_loop_segregation): REPLACES the single-arena within-eligible argmin. The MOTOR
      loop is F (raw_scores); the modulatory channels partition into the ASSOCIATIVE (dACC + lPFC,
      _LOOP_DEFAULT_CHANNEL_MAP) and LIMBIC (OFC + liking + vigour) loops via the active channel
      registry (finer or base). Each non-motor loop accumulates its OWN channel subset, optionally
      settles (MECH-450 W_lat) and splits into D1/D2 (ARC-109); cross-loop arbitration runs AFTER via
      Haber's ascending dopamine spiral (limbic -> assoc -> motor: weighted sum of per-loop
      preferences with loop_segregation_spiral_gain_assoc / _limbic / _motor_authority). Each loop's
      preference is NORMALISED (loop_segregation_normalize, zscore default) BEFORE arbitration -- this
      strips F's raw-magnitude advantage and is the conversion mechanism (F dominates only the motor
      loop). Runs STRICTLY within the F + MECH-449 Go/No-Go eligible set, so a non-motor loop can FLIP
      the within-eligible winner but can NEVER re-admit a suppressed candidate (safety inherited from
      the envelope, orthogonal-to-F guarantee). Meaningful only WITH finer-channel gating on (the base
      registry has one score_bias lump); the validation runs the full stack.
    (2) S2 IN-LAYER SAME-LAYER NULL (_loop_inlayer_null, gated on loop_segregation_noise_on /
      loop_segregation_noise_alpha): replaces each NON-motor loop accumulator with a magnitude-matched
      random-structure (gaussian) perturbation -- range == alpha * the real loop range -- injected at
      the SAME eligibility/settling field the loops settle on (NOT policy softmax temperature, the
      decoupled 700-lineage null). So noise_verified_lifting becomes a MEANINGFUL non-vacuity
      precondition -- the substrate-level fix for the measurement ROOT the 704b/706b autopsies named.
      Motor (F) is never nulled (the thing conversion is tested against). Selection-only (no memory
      write; MECH-094 not engaged).
    (3) ARC-109 D1/D2 POPULATION SPLIT (_d1_d2_split, gated on use_d1_d2_population_split /
      d1_da_gain / d2_da_gain): each loop's accumulator (COST) decomposes into a Go/D1 population
      (relu(-accum)) potentiated by DA (LTP) and a No-Go/D2 population (relu(+accum)) depressed by DA
      (LTD); net cost = D2 - D1; da = the shared ARC-108 value baseline V-hat_t (tanh-squashed). At
      da==0 net == accum EXACTLY (bit-identical; the dissociation is earned only once da != 0). The
      two populations make approach-avoidance CONFLICT (both high) dissociable from indifference (both
      low) -- the representational distinction the additive scalar destroys (the OCD/Parkinson/
      dyskinesia CSTC axis substrate, ARC-106 EARNS). Diagnostic loop_d1_d2_conflict_signal.
    (4) MECH-452 LOOP-LOCAL ELIGIBILITY TRACES (eligibility-recording site, gated on
      use_loop_local_eligibility_traces): credit a channel for the shared signed-RPE delta_t ONLY if
      its loop's within-loop winner matched the committed action (the loop "voted for" the outcome),
      so credit stays loop-local under one broadcast dopamine signal. Diagnostic
      loop_local_credited_channels.
  Config: E3Config.use_loop_segregation + loop_segregation_channel_map / _default_loop /
  _spiral_gain_assoc / _spiral_gain_limbic / _motor_authority / _normalize / _noise_on / _noise_alpha
  + use_d1_d2_population_split / d1_da_gain / d2_da_gain + use_loop_local_eligibility_traces (all
  no-op default; threaded through REEConfig.from_dims). Default False -> the legacy single-arena
  within-eligible path runs UNCHANGED (verified: V3-EXQ-704b finer-channel dry-run ran clean through
  the modified selector, 4/4 arms, exit 0).
  Backward compatible: disabled by default; existing experiments unaffected.
  Biological basis: functional translation of the Alexander/DeLong/Strick parallel cortico-BG-thalamic
  loop organisation integrated by Haber's ascending striato-nigro-striatal dopamine spiral (ARC-106
  L1; NOT anatomical mimicry -- each loop carries a divergence-ledger row + per-loop ablation
  falsifier; the non-degeneracy guard = live cross-loop variance, a loop pinned to the motor winner is
  a vacuous split -> substrate_not_ready_requeue).
  Phased training required: no (reuses already-trained valuation heads; all learned objects ride the
  existing ARC-108 LOCAL three-factor update, not autograd). MECH-094: all learning writes inherit the
  existing simulation_mode=False waking gate.
  C2 RELEASE 2026-06-28 (per-named-channel range-preserving routing -- unblocks C2 limbic load-bearing):
    new no-op-default flag E3Config.use_named_channel_routing. The named cortical bias HEADS emit
    per-candidate-FLAT output (MECH-191 phasic gap), so under per-loop zscore the limbic channels were
    inert and ARM_DROP_LIMBIC was byte-identical to A1_LOOPS (V3-EXQ-707, C2 untestable). When the flag
    is on (with loop seg + finer gating), each named channel's loop-arbitration term is sourced from its
    per-candidate REPRESENTATION -- agent.select_action captures ofc/lpfc -> cand_world_summaries [K,D],
    liking -> goal-proximity [K], vigour -> first-action one-hots [K,A], dacc -> payoff/effort [K,2],
    gated_policy -> summaries [K,D] -> project_channel_range (range-preserving, the SAME GAP-A path that
    keeps `route` phasic) -> score_bias_channel_routed kwarg -> e3_selector.select builds a
    loop_term_override -> _segregated_loop_arbitrate substitutes the routed term for the flat scalar in
    the LOOP ACCUMULATION ONLY. The _lcg_terms eligibility traces, the authority/shortlist
    _modulatory_accum recompose, and the F/score commit path are UNCHANGED -> bit-identical OFF. New
    diagnostics loop_named_channel_routed_ranges / loop_limbic_routed_max_range expose the per-named-channel
    routed per-candidate range for the C2 non-degeneracy gate. Selection-only (MECH-094 not engaged).
    Regression guard tests/test_arc110_loop_segregation.py::TestNamedChannelRoutingC2Release (limbic range
    > 0 + DROP != A1 once routed terms carry range) + ::TestRoutedRepsReachSelectorThroughAgent (plumbing).
  Validation experiment: V3-EXQ-707 -> superseded by V3-EXQ-707a (in-layer-null gate fix) ->
    V3-EXQ-707b (v3_exq_707b_arc110_loop_segregation_c2_release, the C2 release validation: enables
    use_named_channel_routing + limbic input modules, adds the named_channel_routing_live precondition
    before scoring C2). See /queue-experiment.
  Non-degeneracy diagnostics: loop_committed_neq_motor_winner, loop_cross_loop_winner_disagreement,
  loop_assoc_pref_range / loop_limbic_pref_range, loop_d1_d2_conflict_signal, loop_local_credited_channels,
  loop_named_channel_routed_ranges / loop_limbic_routed_max_range (C2 release).
  See ARC-110, ARC-109, MECH-452 (built here); MECH-439 (the F-dominance conversion ceiling under
  test), ARC-108 / MECH-450 / MECH-451 (the within-loop machinery reused), ARC-107 (BG constitution),
  MECH-448 / MECH-449 (the F-bounded eligible set arbitration runs within; safety inherited), ARC-106
  (grounding framework; third worked application), V3-EXQ-700b/704b/706b (the lineage that could not
  test conversion / a valid null on the single arena), MECH-094 (waking-only call-site scoping).
