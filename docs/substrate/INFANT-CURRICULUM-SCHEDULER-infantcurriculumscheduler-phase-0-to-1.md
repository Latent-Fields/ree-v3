## InfantCurriculumScheduler Phase 0->1 H_pos Floor Recalibration (2026-05-31)
- experiments.infant_curriculum.H_POS_FRAC_OF_MAX default-value recalibration --
  IMPLEMENTED 2026-05-31. behavioral_diversity_isolation:GAP-C prereq (3) per
  failure_autopsy_V3-EXQ-591_2026-05-27 section 7. Module:
  ree-v3/experiments/infant_curriculum.py (experiment-harness helper, NOT a
  ree_core substrate scheduler -- lives alongside StepHarness per the file's
  own docstring; matches the existing precedent of curriculum / scheduler
  helpers being experiments/ harness modules).
  Single module-level-constant change: H_POS_FRAC_OF_MAX 0.70 -> 0.20.
  Phase 0 -> 1 advancement gate at _try_phase_0_to_1 (line 253-264) checks
  h_pos < H_POS_FRAC_OF_MAX * ln(grid_size**2); with size=12 the legacy
  threshold was 0.70 * ln(144) ~= 3.48. V3-EXQ-591 (2026-05-27, ARC-046) ran
  the scheduler with random-policy stepping across 5 seeds * 2000 episodes and
  measured observed rolling-mean H_pos band 0.03-1.08 -- the legacy threshold
  was structurally unreachable in every seed of every arm; the curriculum
  never advanced past Phase 0 across the entire validation run. New 0.20
  calibration yields threshold 0.20 * ln(144) ~= 0.99, sitting inside the
  observed band with ~9% margin at the upper end (autopsy: "probe data implies
  ~0.20 of max ~= 0.99 is reachable"). This is an INTENTIONAL non-no-op
  default change, the same exception class as the MECH-307 default-value
  recalibration (2026-05-12 -- mech295_min_drive_to_fire / mech307_conjunction_z_beta_threshold)
  and the ARC-065 SP-CEM main-path landing (2026-05-17 -- six flags flipped on
  HippocampalConfig + REEConfig.from_dims). Rationale: the legacy default sits
  above the substrate ceiling and prevents the gate from ever firing under
  realistic substrate output magnitudes; preserving it would falsely lock the
  curriculum at Phase 0 in every downstream experiment. Bit-identical opt-out
  for any caller that explicitly imports H_POS_FRAC_OF_MAX and overrides to
  0.70 (rare -- module-level constants are used as defaults, not threaded
  through caller config).
  Path (b) alternative-gate (z_goal-norm / residue-coverage replacement) per
  autopsy section 7 NOT taken in this session: z_goal collapses to ~1e-7
  across all V3-EXQ-591 arms (blocked on goal_pipeline:GAP-4 prereq (2)) and
  residue_coverage saturates to 1.0 trivially per autopsy section 7 ("degenerate
  as a discrimination criterion"). Path (a) is the only data-supported
  substrate-side fix until prereq (2) clears. Once prereq (2) lands, path (b)
  becomes architecturally viable and can be considered as a follow-on
  hardening pass.
  No new flags, no new dataclass fields, no REEConfig changes. Pure module-
  level-constant adjustment; matches the MECH-307 2026-05-12 default-tweaks
  session pattern (same one-line-equivalent change shape).
  Phased training: N/A (curriculum-harness helper; no learned parameters; no
  gradient flow; no encoder head).
  MECH-094: N/A (waking-stream curriculum scheduler driven by per-episode
  measurement-side telemetry; the scheduler has no simulation / replay
  invocation site).
  ML/AI engineering notes: standard curriculum-learning threshold-recalibration
  practice (Bengio et al. 2009 automated curriculum learning) -- thresholds
  set by initial design intent must be revised when measurement reveals the
  substrate ceiling sits below them. No additional engineering hazard.
  Contract tests: tests/contracts/test_infant_curriculum_gap9.py extended with
  3 new C11-prefix contracts -- (a) C11_h_pos_default_within_observed_band
  (regression guard: default H_POS_FRAC_OF_MAX must yield a threshold strictly
  inside the observed band [0.03, 1.08]; pre-recalibration value would fail
  this contract), (b) C11_synthetic_p0_trajectory_advances_phase (substrate-
  level smoke: a synthetic P0 trajectory at observed-band-top H_pos=1.05 must
  advance Phase 0 -> 1 at episode 100 boundary), (c) C11_synthetic_p0_trajectory_marginal_clearance
  (clearance arithmetic: H_pos at 0.99 * threshold blocks; at 1.001 * threshold
  admits; floating-point boundary). 19/19 contracts in the GAP-9 file PASS;
  593/593 full ree-v3 contracts + 7/7 preflight PASS (regression-clean
  2026-05-31).
  Backward compatible behaviour with the previous (broken) default is
  intentionally NOT preserved -- the previous default never produced a
  firing gate, so no caller can have a legitimate dependency on the
  Phase 0 lock-in.
  Downstream beneficiaries:
    GAP-C / V3-EXQ-603d / V3-EXQ-591b -- behavioural cluster validation
      (blocked_pending_substrate until both prereq (2) goal_pipeline:GAP-4
      and prereq (3) -- this -- clear; this session lands prereq (3)).
    ARC-046 / V3-EXQ-591b -- ARC-046 itself is blocked on prereq (2);
      landing (3) is necessary but not sufficient.
    infant_substrate plan (REE_assembly/evidence/planning/infant_substrate_plan.md)
      -- the underlying curriculum's ability to walk Phase 0 -> Phase 3 at
      all depends on this exit-signal being clearable.
    DEV-NEED-004 gate experiments (tier-1: V3-EXQ-587 GAP-10 etc.) --
      transitively dependent on the curriculum being walkable.
  Validation experiment: the behavioural validation gate is V3-EXQ-603d /
  591b (per autopsy section 8), which is also blocked on prereq (2) and
  therefore NOT queued in this session. Contract tests serve as the
  substrate-readiness gate at the unit level. No separate substrate-readiness
  EXQ queued; the unit tests verify the recalibration delivers a clearable
  gate at observed-band signal magnitudes, which is the substrate-level
  acceptance criterion. Behavioural validation lands as V3-EXQ-603d / 591b
  when prereq (2) is cleared by goal_pipeline:GAP-4 / V3-EXQ-490g cohort.
  Plan-of-record: REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md
    GAP-C node (substrate_landed_2026_05_31 annotation added; status remains
    blocked_pending_substrate because prereq (2) is still the load-bearing
    blocker; once prereq (2) clears, prereq (3) is no longer the gate).
  Design doc: REE_assembly/docs/architecture/infant_substrate_expansion.md
    Section 6.1 Phase 0 exit condition updated with the recalibration note.
  Cross-link: IGW-20260531-009.
  See ARC-046 (parent claim; behavioural validation owner), GAP-C
    (behavioral_diversity_isolation:GAP-C closure node), MECH-307 (2026-05-12
    default-value recalibration -- same exception class as this default flip;
    sister GAP-1 prereq cleared 2026-05-15 by V3-EXQ-540g PASS), ARC-065
    SP-CEM main-path landing (2026-05-17 -- another intentional non-no-op
    default-flip precedent), SD-017 (sleep / scheduler family the curriculum
    coordinates with via on_phase3_entry hook), goal_pipeline:GAP-4 (prereq
    (2) z_goal-collapse blocker that gates the behavioural retest 591b;
    V3-EXQ-490g cohort), V3-EXQ-603a/b/c (the 603 family the recalibration
    unblocks), V3-EXQ-591 (the autopsy that named the gap), failure_autopsy_V3-EXQ-591_2026-05-27
    section 7 (the routing document for this implement-substrate session),
    behavioral_diversity_isolation_plan.md (closure-plan doc; GAP-C node
    annotated), MECH-094 (simulation gate; not applicable -- waking-stream
    scheduler).
