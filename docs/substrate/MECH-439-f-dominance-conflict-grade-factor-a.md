## MECH-439: F-dominance conflict-grade -- Factor A conflict-graded shortlist width + Factor B gap-scaled commit-T (2026-06-18)
- MECH-439 conflict-grade levers: ethics_engine_3.f_dominance_conflict_grade -- IMPLEMENTED
  2026-06-18 (substrate; MECH-439 stays candidate -- this PROMOTES NOTHING. A PASS moves it
  toward supports; a preconditions-met FAIL routes to the synthesis-doc V4 directions, NOT a
  dead end). The conversion-ceiling campaign's V3 fix for the live root (F-dominance). Routed
  by REE_assembly/evidence/planning/conversion_ceiling_phase0_synthesis_2026-06-18.md (Phase 0
  four-root + Phase 1 fork-resolution + Phase 1 backfill).
  ROOT (V3-EXQ-571): the primary harm/goal score F carries ~88-89%% of E3 committed-selection
  variance, UNCHANGED by the full diversity stack, so every diversity channel reaches the E3
  accumulator but cannot move the F-dominated committed argmax. The standalone 569i top-k PASS
  (2/3 seeds) only thinly cleared (0.711 vs proposer 0.650). Phase 1 resolved the fork toward
  STRUCTURAL conflict-grading (NOT F-variance rebalancing -- the rebalance lever = std-basis
  additive authority already FAILED 569h 1/3; the structural bound = top-k shortlist PASSED
  569i 2/3). The backfill found the two V3-tractable levers are TWO RENDERINGS OF ONE PRINCIPLE
  -- the BG hyperdirect conflict-grade: grade the committed decision by the normalized top-F
  gap -- and should be tested TOGETHER as a 2-factor experiment.
  Module: ree_core/predictors/e3_selector.py (E3TrajectorySelector.select +
  _gap_scaled_commit_pick helper + 6 diagnostics keys), ree_core/utils/config.py (E3Config 5
  fields + REEConfig.from_dims). One shared quantity gap_norm in [0,1] computed once from
  raw_scores (the F-best to F-second-best gap / raw_score_range; None when < 2 candidates ->
  both factors no-op) drives BOTH levers.
  FACTOR A (conflict-graded shortlist width; the safety-hard primary). The existing top_k
  shortlist block (mode='top_k') gets, under modulatory_shortlist_conflict_graded, a graded k:
  k = clamp(round(k_max - (k_max-1)*gap_norm), 1, K). Near-ties (gap_norm ~ 0) -> k = k_max
  (wider eligible set / slower commit, the STN threshold-raise); a decisive F-gap (~1) -> k = 1
  (fast commit on the F-winner). F gates ELIGIBILITY only; it is ABSENT from the within-set
  arbitration (the routed modulatory channel argmin/sample picks inside the eligible set).
  SAFETY: because the eligible set is the k F-best, a clearly-harmful candidate (large F-gap
  above the best) is never admitted. Flags: modulatory_shortlist_conflict_graded (False) +
  modulatory_shortlist_k_max (6). Default False -> the fixed modulatory_shortlist_k path is
  bit-identical.
  FACTOR B (gap-scaled entropy-regularized commit; the complement). The committed selection --
  otherwise a HARD argmin over the F-dominated scores (or the routed modulatory channel within
  a shortlist) -- becomes, under use_gap_scaled_commit_temperature, multinomial(softmax(-q /
  T_eff)) over the eligible set, with T_eff = base_temperature + gap_scaled_commit_entropy_alpha
  * (1 - gap_norm). Near-ties -> hotter (softer argmax); a decisive gap -> cold (T_eff -> base,
  preserves the decisive F-winner). q is the routed modulatory channel within an active Factor-A
  shortlist (F-bounded eligible set = the safety guarantee), else the F-dominated scores
  restricted to an F-eligibility envelope (candidates within gap_scaled_commit_harm_floor *
  raw_score_range of the best raw score) so a hot commit-T in a near-tie can NEVER softmax-
  promote a clearly-harmful candidate -- the SAFETY GATE the backfill flagged. This softens the
  COMMITTED hard-argmin (where the monostrategy lives); distinct from MECH-313 tonic-noise
  (gap-blind, pre-select) and from the existing uncommitted multinomial. A FLAT (gap-blind)
  commit-T reduces to the 569g temperature control that under-lifted -- the (1 - gap_norm)
  gap-scaling is LOAD-BEARING (asserted by contract). Flags: use_gap_scaled_commit_temperature
  (False) + gap_scaled_commit_entropy_alpha (1.0) + gap_scaled_commit_harm_floor (0.25). Default
  False -> the hard argmin path is bit-identical.
  Backward compatible: both factors default no-op -> the legacy fixed-k top_k argmin path is
  bit-identical (verified). Contracts: tests/contracts/test_e3_conflict_graded_conversion.py
  (12: OFF bit-identical to legacy fixed-k argmin + default flags; Factor-A width clamps to
  k_max at near-tie + to 1 at decisive gap + monotone non-increasing in gap + within [1,K];
  F absent from Factor-A within-set arbitration; Factor-A safety -- clearly-harmful never
  admitted; Factor-B T_eff monotone + gap-scaling load-bearing; Factor-B cold preserves the
  decisive winner + hot softens a near-tie; Factor-B standalone harm-floor safety gate excludes
  a harmful candidate even when its composed score is lowest; Factor-B standalone OFF == hard
  argmin). Preflight 7/7. Full contract suite green.
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters; no gradient
  flow). MECH-094: N/A (waking committed-selection path; no replay/memory write surface).
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default levers; every existing
  experiment uses the defaults, so no dependent claim's measured mechanism changed. KEEP all
  evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-439 stays candidate; ARC-065 / MECH-341 / ARC-062 /
  MECH-309 / MECH-294 untouched. claims.yaml NOT modified (substrate-only amend).
  Validation experiment: V3-EXQ-689 (2-factor 2x2 discriminating falsifier on the GAP-A-ready
  foraging substrate; trained e2.world_forward via SD-056 + ARC-065 GAP-A
  candidate_summary_source=e2_world_forward so the eligible set is genuinely diverse -- the
  non-vacuity precondition; vacuous top_k over a class-uniform pool self-routes
  substrate_not_ready_requeue, NEVER a false weakens). 2x2: Factor A (fixed-k=3 vs conflict-
  graded k) x Factor B (hard argmin vs gap-scaled commit-T), CRF stack constant. PRIMARY:
  committed_action_class_entropy strict-above BOTH collapsed-proposer and matched-noise controls
  on >=2/3 seeds. LOAD-BEARING PRE-REGISTERED FALSIFIER (shared by both factors): bin ticks by
  top-F-gap and regress committed entropy on gap -- the lift MUST correlate with per-tick F-gap;
  uniform lift = the grading adds nothing over a bigger fixed shortlist / hotter flat softmax.
  Non-vacuity self-route: k AND T_eff actually vary across ticks; eligible set diverse.
  claim_ids=[MECH-439] (this is its first falsifier). substrate-side ready stays FALSE until
  the falsifier scores. V3 fallback if both gap-graded margins stay thin =
  rank_preserving_F_to_eligibility_demotion (F removed from the final argmin entirely, used only
  as a graded eligibility envelope) -- next step, NOT built now.
  Synthesis doc: REE_assembly/evidence/planning/conversion_ceiling_phase0_synthesis_2026-06-18.md
  See MECH-439 (this claim; first falsifier), modulatory-bias-selection-authority (parent; the
  top_k shortlist Factor A grades + the authority Factor B composes with), V3-EXQ-571 (F
  monopoly 88-89%), V3-EXQ-569i (top-k PASS 2/3, thin) / 569h (std-basis FAIL 1/3), ARC-065
  GAP-A candidate_summary_source=e2_world_forward + SD-056 (the divergent-pool non-vacuity
  precondition), MECH-341 / ARC-062 / MECH-309 / MECH-294 (the diversity channels that drown at
  the F-dominated argmax), MECH-313 (tonic-noise; distinct -- gap-blind, pre-select), MECH-090
  (admission gate; unchanged), MECH-094 (N/A).
