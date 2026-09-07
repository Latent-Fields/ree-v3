## modulatory-bias-selection-authority AMEND: gain/contrast + shortlist-then-modulate conversion (569g/682, 2026-06-15)
- modulatory-bias-selection-authority CONVERSION amend -- IMPLEMENTED 2026-06-15.
  Modules: ree_core/predictors/e3_selector.py (authority block normalize_basis
  branch + shortlist-then-modulate selection block + 3 diagnostics keys),
  ree_core/utils/config.py (3 E3Config fields + from_dims passthrough). Routed by
  the confirmed failure_autopsy_V3-EXQ-569g_2026-06-14 (CORRECTED) + the GAP-A node's
  682-gated route, after V3-EXQ-682 LANDED PASS 2026-06-15 (no_collapse_reproduced).
  GATE CHECK: 682 (the in-arm route-range collapse diagnostic, claim_ids=[]) confirmed
  the 06-10 route-range amend SOLVED REACH -- ARM_1 applies in-arm
  modulatory_channel_route_range ~0.20 at the live select tick, summaries_none_frac 0.0,
  and ALL FOUR collapse causes (cause_i live summary re-collapse / cause_ii
  project_channel_range / cause_iii wiring / cause_iv applied-zero) RULED OUT, including
  on seed 43 (the seed where 569g's CONVERSION failed). So there is NO residual upstream
  re-collapse to fix -> proceed directly to the conversion amend (the clean Branch A).
  ROOT CAUSE (569g, the residual one link past route-range): the gap-relative ADDITIVE
  authority at gain 0.5 (modrange = 0.5*raw_score_range) is subdominant to the
  F-dominated primary (88-89% of E3 variance, V3-EXQ-571), so the routed range flips
  only near-tie OUTLIERS, not near-decisive winners (569g committed entropy 1/3 seeds
  strict-above a temperature-matched control; 662 TV>0 but no entropy lift). Upstream
  magnitude sweeps (667/640a byte-identical) cannot fix it -- the authority
  range-renormalizes its input -- so only authority GAIN and arbitration STRUCTURE are
  live levers.
  TWO no-op-default conversion levers (bit-identical OFF):
    (a) GAIN/CONTRAST NORMALIZATION -- E3Config.modulatory_authority_normalize_basis:
      "range" (default, bit-identical legacy: target = gain*raw_score_range, scaled by
      the modulatory RANGE -- outlier-sensitive, near-tie-only) vs "std" (target =
      gain*raw_score_STD, scaled by the modulatory STD -- robust to outliers, anchoring
      authority to the TYPICAL primary spread so the structured channel competes against
      NEAR-DECISIVE candidates). modulatory_authority_gain stays sweepable. SAFETY NOTE:
      additive gain >= 1.0 breaks the safety bound (modulatory can override a
      clearly-harmful rejection); keep gain < 1.0 on the additive path for the shipped
      config, or use lever (b).
    (b) SHORTLIST-THEN-MODULATE -- use_modulatory_shortlist_then_modulate +
      modulatory_shortlist_margin (0.25): F (raw primary scores) filters to a near-tie
      set (candidates within margin*raw_score_range of the best raw score), then the
      modulatory channel (_modulatory_accum) ARBITRATES the winner WITHIN that set (argmin
      committed / softmax-sampled uncommitted). The structured channel is load-bearing
      WITHOUT out-magnitude-ing F, and SAFETY is preserved at any internal strength
      (clearly-harmful candidates outside the shortlist are never selectable). Takes
      precedence over the additive-authority rescale + argmin/stratified selection at the
      selection site when enabled.
  Diagnostics on last_score_diagnostics: modulatory_authority_normalize_basis,
  modulatory_shortlist_active, modulatory_shortlist_size (the near-tie-set size; the
  readiness EXQ reads these).
  Backward compatible: basis="range" + shortlist OFF by default -> bit-identical to the
  pre-conversion-amend authority path. 1036 contracts + 7/7 preflight PASS; 4 new
  contracts in tests/contracts/test_e3_score_bias_candidate_support.py (conversion-amend
  OFF bit-identical / std-basis distinct scale_factor / shortlist restricts to the F
  near-tie set + safety: worst-primary never selected even with overwhelming pull /
  shortlist arbitrates by the modulatory channel within the set). Bit-identical OFF
  verified (default == explicit-range/shortlist-off action stream, atol 1e-12).
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters). MECH-094:
  N/A (waking committed-selection path; no replay write surface; unchanged).
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flags; every existing
  experiment uses the defaults, so no dependent claim's measured mechanism changed.
  KEEP all evidence.
  GOVERNANCE: ARC-065 / MECH-341 / ARC-062 / MECH-309 / MECH-294 NEITHER promoted NOR
  weakened; ARC-065 stays provisional / non_contributory / pending_retest_after_substrate;
  MECH-341 untouched. claims.yaml NOT modified (substrate-only amend; the amend record
  lands in substrate_queue.json + the GAP-A plan node).
  Validation experiment: V3-EXQ-684 claim-free readiness sweep (gain x {range,std} x
  shortlist vs a VERIFIED-LIFTING matched-noise control; committed-action entropy must
  MOVE with channel range AND beat the control, not merely reach the accumulator -- 569g's
  ARM_2 temperature control under-lifted, so the sweep first verifies the noise control
  actually raises entropy). On PASS, V3-EXQ-569h (the GAP-A falsifier with an IN-ARM
  applied-route-range non-vacuity gate) is queued with the winning conversion config --
  it is GATED on the readiness sweep landing, NOT pre-queued on a guessed config.
  Design doc: REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
  (CONVERSION amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-569g_2026-06-14.{md,json}.
  Diagnostic: REE_assembly/evidence/experiments/v3_exq_682_gapa_inarm_routerange_collapse_diagnostic_20260615T032040Z_v3.json.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (modulatory-bias-selection-authority entry, current_pending_amend).
  Closure node: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
  (behavioral_diversity_isolation:GAP-A).
  See modulatory-bias-selection-authority (parent substrate; gap-relative 2026-06-03 +
  643a float32 fix 2026-06-06 + route-range 2026-06-10 above), V3-EXQ-569g (the FAIL this
  amend addresses), V3-EXQ-682 (the in-arm diagnostic that cleared the gate), V3-EXQ-571
  (F-dominated primary 88-89%), ARC-065 GAP-A (closure node), MECH-341 / ARC-062 /
  MECH-309 / MECH-294 (one shared conversion fix; xref), V3-EXQ-684 (readiness validation)
  / V3-EXQ-569h (GAP-A falsifier, gated), MECH-090 (admission gate; unchanged), MECH-094 (N/A).
