## MECH-341 Amend: stratified_within_class_temperature + A-vs-B partial-redundancy probe (2026-06-01)
- MECH-341 amend -- IMPLEMENTED 2026-06-01. Routed by
  failure_autopsy_V3-EXQ-616_2026-05-31.md Sections 7 + 10 (contingent-on-
  614b-FAIL-C1 path: 614b FAILed C1 R2.c with B_only Rung-1 majority=False
  AND C2 necessity_delta 0.087 nats below the pre-amend 0.1 threshold;
  C3 ALL_ON Rung-1 PASSed with mean entropy 0.800 nats -- highest of any
  614-lineage run, positive substrate-readiness for the SD-056 amend at
  the behavioural-runtime horizon). The 616 autopsy's named contingent
  amend is "stratified_temperature default + A-vs-B partial-redundancy
  probe"; this session lands the within-class proportional sampling lever
  + names the A-vs-B factorial via the existing independent flags.
  Two-part amend:
    Part (a) within-class proportional sampling. Extends
      ree_core/predictors/e3_score_diversity.py E3ScoreDiversity.stratified_select
      with a new togglable lever: when stratified_within_class_temperature
      is set (Optional[float], default None = legacy argmin bit-identical),
      sample within each first-action class via
      softmax(-class_scores / T) before the across-class softmax step.
      Decoupled from the existing `stratified_temperature` (which controls
      across-class softmax sampling) so the A-vs-B probe can dissociate
      Layer B within-class sub-axis from across-class sub-axis. Decoupling
      rationale: collapsing both into one knob would make Q-054 sweep
      results uninterpretable -- a single-temperature dial cannot tell
      whether the lift came from within- or across-class sampling.
    Part (b) A-vs-B partial-redundancy probe lever. The autopsy's
      "config flag" requirement is satisfied by the existing independent
      master flags use_support_preserving_cem (Layer A: CEM proposal
      diversity, ARC-065 SP-CEM child) and use_e3_score_diversity
      (Layer B: E3 score-layer preservation, MECH-341). These compose
      to a complete factorial: A_only / B_only / BOTH / NEITHER. No new
      config flag is added (avoids redundancy and dead code); the
      validation experiment V3-EXQ-614c uses these flags directly to
      construct the probe grid. The amend's contribution is naming the
      probe pattern, pinning acceptance criteria, and recording the
      composition convention in substrate_queue.json amend_history.
  Config: E3ScoreDiversityConfig gains stratified_within_class_temperature
    (Optional[float], default None). REEConfig gains
    e3_diversity_stratified_within_class_temperature (Optional[float],
    default None) surfaced through REEConfig.from_dims and propagated to
    config.e3_diversity_stratified_within_class_temperature.
    build_e3_score_diversity_from_ree_config reads the new flat field via
    getattr with default None (bit-identical when the flat REEConfig field
    is absent).
  Diagnostics on E3ScoreDiversity.get_state(): three new keys
    mech341_n_within_class_sampled, mech341_last_within_class_sampled,
    mech341_last_within_class_temperature. V3-EXQ-614c reads these for
    acceptance criteria.
  Backward compatible: with default None, the new branch in
    stratified_select is short-circuited and within-class selection falls
    through to legacy argmin (bit-identical to pre-amend MECH-341).
    With use_e3_score_diversity=False (master OFF), the entire MECH-341
    block at the e3_selector.py call sites is skipped. 655/655 contracts
    (645 prior + 10 new MECH-341 amend contracts in
    tests/contracts/test_mech_341_stratified_temperature_amend.py) + 7/7
    preflight PASS with master OFF and amend OFF.
  Activation smoke (2026-06-01, verified by C2/C3/C4 contracts):
    T=1e-4 -> within-class collapses to argmin (sharpening matches legacy);
    T=1.0 -> stochastic non-deterministic within-class selection
    (n_within_class_sampled advances each call);
    T=1e4 -> uniform-within-class (all members surface across many trials).
  No trainable parameters. No phased training (pure-arithmetic regulator
  extension; same as the original MECH-341 retune). MECH-094 preserved
  by the existing simulation_mode argument on stratified_select -- the
  new branch sits AFTER the simulation_mode short-circuit, so replay
  paths never enter within-class sampling.
  Architectural choice (single-knob vs decoupled): the 616 autopsy's
    "stratified_temperature default" hint could have been read as
    repurposing the existing parameter. That reading was rejected because
    the existing `stratified_temperature` controls across-class softmax
    (line 269 of e3_score_diversity.py) with default 1.0; the V3-EXQ-611c
    retune defaults were calibrated against this exact knob, and any
    semantic shift would break the bit-identical OFF guarantee for callers
    that explicitly set it today. The decoupled-knob design preserves
    backward compat AND lets V3-EXQ-614c sweep within-class temperature
    in isolation while across-class temperature stays at the 611c
    default (1.0).
  Cross-plan impact: same amend transitively unblocks
    arc_062_rule_apprehension:GAP-B (V3-EXQ-543l successor cohort) under
    shared SD-056-amended substrate. Layer B within-class sub-axis
    contributes to ARC-062 head-input candidate-pool diversity if Q-054
    Rung 1 PASS routes via within-class lift. Flagged in V3-EXQ-614c
    queue rationale.
  Constraint preserved: NO flip of use_differentiable_cem default
    (safety note in substrate_queue.json scaffolded_sd054_onboarding /
    SD-055 entry; default OFF remains the baseline 569a/614b
    falsifiers measure against).
  Validation experiment: V3-EXQ-614c queued (4-arm sweep
    stratified_within_class_temperature in {None=legacy, 0.5, 1.0, 2.0}
    on SD-056-amended baseline; all other levers held at the V3-EXQ-614b
    config that produced ALL_ON 0.800 nats; 3 seeds; 30 P0 + 60 P1 ep;
    200 steps/ep; ~3-4 h on Mac). Acceptance: C1 ARM_LEGACY reproduces
    614b ARM_2 ALL_ON within 10% (regression guard); C2 at least one of
    {0.5, 1.0, 2.0} produces mean_selected_class_entropy_nats >= 0.800
    nats on majority of seeds (within-class lift); C3 frac_pre_ge2 > 0.3
    on majority of seeds (substrate-readiness check). PASS = C1 AND
    (C2 OR C3-only-with-no-regression). Cross-link to
    behavioral_diversity_isolation_plan.md GAP-B + arc_062_rule_apprehension:GAP-B.
  Design doc: REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
    (new "Amend 2026-06-01" section).
  Plan-of-record: REE_assembly/evidence/planning/substrate_queue.json
    MECH-341 entry amend_history (mirroring SD-056 multi-step pattern at
    2026-05-31T11:25Z) and
    REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
    GAP-B node governance_2026_06_01.
  See MECH-341 (parent claim), failure_autopsy_V3-EXQ-616_2026-05-31
    (routing autopsy; Sections 7 + 10), failure_autopsy_V3-EXQ-614b_2026-05-31
    (corroborating cluster-member autopsy on the prior 3-arm behavioural
    falsifier; FAILed C1 structurally), ARC-065 (parent diversity-pathway
    architectural commitment), Q-054 (minimum trajectory-class diversity
    floor; the V3-EXQ-614c sweep is a Q-054 instantiation at the within-
    class sub-axis), arc_062_rule_apprehension:GAP-B (cross-plan
    beneficiary), SD-055 use_differentiable_cem (NOT flipped; safety note
    preserved), MECH-094 (simulation_mode argument; existing gate
    preserved -- new branch sits after the gate).
