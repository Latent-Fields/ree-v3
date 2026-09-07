## MECH-448 / ARC-107: rank-preserving F->eligibility demotion (LEAD lever of the basal-ganglia E3-selector constitution) (2026-06-20)
- MECH-448: ethics_engine_3.rank_preserving_f_to_eligibility_demotion -- IMPLEMENTED
  2026-06-20 (substrate; MECH-448 stays candidate -- this PROMOTES NOTHING. The
  689a-successor falsifier is the sequenced next step, NOT queued here). THE FIRST major
  worked application of ARC-106 (brain-like construction: grounding ladder + divergence
  ledger + load-bearing-vs-decorative ablation falsifier + psychiatric failure mode).
  Routed by the user-adjudicated failure_autopsy_V3-EXQ-689a_2026-06-20 Step-8 decision
  ("elevate the constitutional build"; the conflict-grade near-tie parametric family --
  MECH-447 -- is exhausted). Design note:
  REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md s3.1.
  PRECONDITION (build branch): V3-EXQ-689c (Factor-B-alone isolation retest) was PENDING /
  not-landed in the coordinator queue at build time (status pending, pinned ree-cloud-3,
  no results row), so no gap-CONCENTRATED parametric win existed to shrink scope -> PROCEED
  per the design note s5.2/s5.3 branch. The no-op-default lever commits nothing about scope;
  the scope-sensitive falsifier will incorporate 689c when it lands.
  PROBLEM (V3-EXQ-571): F (the primary harm/goal score) monopolises ~88-89%% of E3
  committed-selection variance, unmoved by the full diversity stack -- every diversity
  channel drowns at the F-dominated committed argmin. The required fix is constitutional:
  a signal's STRENGTH must be necessary but not sufficient; it needs lawful ACCESS to
  committed action. MECH-448 = F decides who is ELIGIBLE, not who wins.
  THE LEVER (no-op default; bit-identical OFF):
    Module: ree_core/predictors/e3_selector.py (new helper
    E3TrajectorySelector._f_eligibility_envelope + a "f_demotion" eligibility branch in the
    EXISTING shortlist-then-modulate block + 5 diagnostics), ree_core/utils/config.py
    (E3Config 3 fields + from_dims passthrough).
    Graded eligibility envelope (divisive-normalisation analog, lower-is-better F):
      merit[i] = clamp(raw_scores.max() - raw_scores[i], min=0)   # best=highest
      pooled   = f_eligibility_dn_sigma + merit.sum()
      elig[i]  = merit[i] / pooled                                # share of competing field
      eligible = { i : elig[i] >= f_eligibility_envelope_floor }  # ABSOLUTE share floor
    The absolute share floor is LOAD-BEARING: a fraction-of-max threshold cancels the pooled
    term and degenerates to the margin shortlist. With an absolute floor, a decisive F-winner
    commands most of the merit share so others fall below the floor (NARROW envelope), while a
    near-tie spreads the share (WIDE envelope) -- the BG hyperdirect conflict-grade emerging
    from the field structure, NOT a hard top-k count (which is env-conditional: 569i works only
    on the reef-bipartite guarantee; V3-EXQ-684 margin admits a near-whole state-stable set).
    elig is monotone in merit -> monotone in -F, so the eligible set is an F-RANK PREFIX
    (rank-preserving). The EXISTING _modulatory_accum within-eligible arbitration (reused, NOT
    duplicated -- ARC-106 guardrail 2) then picks the committed action (argmin committed /
    softmax uncommitted) with F REMOVED from the final argmin.
    Fallback: flat F (range ~0) or a genuine N-way tie whose per-candidate share is below the
    floor -> WIDE (all eligible) = correct low-conflict behaviour (and reported excluded_count
    == 0 = the non-degeneracy signal the falsifier checks against a divergent pool).
  Config (E3Config + REEConfig.from_dims, all no-op default; bit-identical OFF):
    use_f_eligibility_demotion (False, master), f_eligibility_envelope_floor (0.30; absolute
    DN-share floor), f_eligibility_dn_sigma (0.0; DN semi-saturation / global tightness).
    Requires a modulatory channel (_modulatory_accum not None, i.e. score_bias / MECH-341
    bonus / route bias) -- with no modulation there is nothing to demote F to, so the block is
    skipped (legacy F argmin, bit-identical); this is the falsifier's non-vacuity precondition.
  Diagnostics (last_score_diagnostics): f_eligibility_demotion_active, f_eligibility_envelope_size,
    f_eligibility_excluded_count (NON-DEGENERACY: >0 = the envelope actually excluded, not all-admit),
    f_eligibility_winner_neq_f_argmin (F demoted at commit), f_eligibility_rank_preserving (eligible
    set is an F-rank prefix; every eligible cost <= every excluded cost -- tie-robust).
  ARC-106 DIVERGENCE LEDGER (LOAD-BEARING): canonical divisive normalisation (Carandini & Heeger
    2012; value DN, Louie/Khaw/Glimcher 2013) is ORDER-PRESERVING + POOLED-SYMMETRIC. REE demotes
    ONLY F and removes it from the commit argmin -- rank-ALTERING at COMMIT -- which EXCEEDS
    canonical DN (the QD/MAP-Elites justification, CDQ-003). Must be lit-anchored (the concurrent
    targeted_review_connectome_mech_439 grounding extension) + falsifier-validated.
  SAFETY: a clearly-harmful candidate has near-zero merit -> near-zero share -> below floor ->
    excluded; no global disinhibition (the envelope is itself the F-bound). Contract-verified
    (an overwhelming modulatory pull toward an excluded harmful candidate never selects it).
  Psychiatric failure mode (ARC-106 mandate): envelope too wide / F removed without bounded No-Go
    -> disinhibition / impulsivity (mania, OCD-spectrum loss of inhibitory braking); envelope too
    tight -> bradykinesia / avolition (the current nothing-but-F-converts failure).
  Backward compatible: use_f_eligibility_demotion=False by default -> the f_demotion branch is
    never entered (the shortlist block guard is OR'd with the master flag; the legacy margin/top_k
    bodies are unchanged); bit-identical OFF. 10/10 new contracts in
    tests/contracts/test_mech_448_f_eligibility_demotion.py PASS; 8/8 preflight + 172/172
    E3-related contracts (e3/selector/mech_439/mech_341/score_bias/candidate_support/modulatory/
    arc065/dr12/mech090) PASS unchanged. Pure-envelope unit check: near-tie wider (2) than
    decisive (1), harmful outlier excluded (safety), exact 4-tie wide fallback (excluded 0).
    Full-agent activation smoke (MECH-341 modulatory channel, 20-tick loop): OFF 0/20 ticks
    demotion-active, ON 20/20 ticks demotion-active with winner!=F-argmin on every tick (F removed
    at commit); excluded=0 on the toy near-flat-F substrate (the divergent-pool exclusion the
    falsifier requires is contract-proven, not a smoke artifact).
  Phased training: N/A (pure-arithmetic selection rule; no learned parameters; no gradient flow).
    MECH-094: N/A (waking committed-selection path; no replay/memory write surface).
    Evidence-staleness (Step 8.5): NOT triggered -- no-op-default lever; every existing experiment
    uses the default (lever off), so no dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-448 stays candidate; ARC-107 / MECH-447 / MECH-449 /
    MECH-439 / Q-078 untouched. claims.yaml NOT modified (substrate-only build; co-claimed by a
    concurrent governance-689a-apply session at build time). substrate_queue f_dominance_conversion_ceiling
    build-rung amend DEFERRED -- live conflict with governance-689a-apply (re-checked at write time).
  Validation experiment: NOT queued here. The 689a-successor falsifier on the demotion lever
    (acceptance criteria = design note section 4; committed-class entropy reaches the proposer
    ceiling on >=2/3 seeds AND order preserved on the numerators AND no harmful class disinhibited;
    NON-DEGENERACY excluded_count>0 on a divergent pool) is the sequenced next /queue-experiment
    step, run on the GAP-A-ready foraging substrate (SD-056-trained e2.world_forward + ARC-065
    GAP-A candidate_summary_source=e2_world_forward = the divergent-pool non-vacuity precondition).
  Design doc: REE_assembly/docs/architecture/mech_448_f_eligibility_demotion.md
  Design note: REE_assembly/evidence/planning/arc_107_selector_constitution_design_2026-06-20.md
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-689a_2026-06-20.md
  See MECH-448 (this claim), ARC-107 (architecture), ARC-106 (grounding framework -- first worked
    application), modulatory-bias-selection-authority (the shortlist-then-modulate _modulatory_accum
    arbitration reused), MECH-439 (F-dominance root), MECH-447 (conflict-grade near-tie family;
    exhausted), MECH-449 (Go/No-Go constitution; follow-on, double-gated), Q-078 (umbrella),
    V3-EXQ-571 (F monopoly 88-89%%), V3-EXQ-689a (the autopsy that routed this), V3-EXQ-684 / 569i
    (margin-vs-top_k shortlist evidence), MECH-094 (N/A).
