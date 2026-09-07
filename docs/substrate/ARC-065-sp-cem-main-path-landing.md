## ARC-065 SP-CEM Main-Path Landing (2026-05-17)
- ARC-065 hippocampal-trajectory-sampling child (support-preserving + stratified
  CEM, "SP-CEM") — LANDED AS MAIN-PATH DEFAULT 2026-05-17. The substrate itself
  was already implemented (HippocampalModule._support_preserving_elite_indices,
  hippocampal/module.py); this change flips its config defaults so the main agent
  action path runs SP-CEM instead of the legacy collapsing CEM.
  Config (6 defaults flipped, in BOTH the HippocampalConfig dataclass AND the
  REEConfig.from_dims() signature, because from_dims assigns its params onto
  config.hippocampal.* unconditionally — flipping only the dataclass would be
  silently reverted for from_dims-built agents):
    use_support_preserving_cem            False -> True
    support_preserving_stratified_elites  False -> True
    support_preserving_ao_std_floor       0.0   -> 0.2
  (support_preserving_min_first_action_classes stays 2; normalize_score_bias_to_e3_range
  untouched — it was NOT in the validated combination.)
  This is an INTENTIONAL non-no-op default change (the one deliberate departure
  from the implement-substrate no-op-default rule): the legacy collapsing CEM
  produced the monostrategy that left SD-029, ARC-062 Rung 2, goal_pipeline
  GAP-2/4 and self_attribution GAP-1/2/3 non_contributory. Every experiment that
  builds config WITHOUT explicitly setting these flags now gets SP-CEM.
  Bit-identical legacy opt-out: explicitly pass use_support_preserving_cem=False,
  support_preserving_stratified_elites=False, support_preserving_ao_std_floor=0.0.
  EXQ-567 ARM_0 and all existing ablation/control arms already pin these
  explicitly, so they are unaffected; only default-reliant scripts change (the
  intended effect).
  Evidence basis: V3-EXQ-567 PASS 2026-05-15 (ARM_1 vs ARM_0:
  selected_action_entropy 0.0124 -> 0.4965, candidate support 1.007 -> 2.810;
  claim ARC-065). Corroborated by V3-EXQ-573 + V3-EXQ-568. Confound-free per the
  V3-EXQ-543e failure autopsy (2026-05-17: H1/H2 FALSE, H3 confirmed; SP-CEM
  delivers ~4.95 unique first-action classes). Phased training: N/A (selection/
  sampling change, no encoder/learning). MECH-094: N/A (no simulation/replay/
  memory write). Smoke: /tmp/smoke_spcem_mainpath.py 4/4 (bare from_dims agent
  runs SP-CEM via StepHarness, gate on every candidate step, no crash, opt-out
  intact); experiments/v3_exq_567_..._sp_cem.py --dry-run still PASS (both
  explicit-off ARM_0 and explicit-on ARM_1 end-to-end). Validation experiment:
  V3-EXQ-583 (experiments/v3_exq_583_spcem_mainpath_default_wiring.py,
  experiment_type v3_exq_583_spcem_mainpath_default_wiring) 3-arm default-wiring
  equivalence queued, DIAGNOSTIC claim_ids=[] (ARM_default == ARM_explicit_on
  within 1e-9 + exact dict match, both >> ARM_explicit_off; dry-run PASS both
  criteria). claims.yaml: ARC-065 carries an
  implementation_note; NOT promoted (promotion is Rung-1 matched-entropy governance,
  gated on V3-EXQ-569). The ARC-062 GatedPolicy head-input one-hot augmentation is
  a SEPARATE follow-on (V3-EXQ-543f), out of scope here. See ARC-065, ARC-062,
  SD-029, MECH-269, MECH-309, behavioral_diversity_acceptance_criteria.md.
