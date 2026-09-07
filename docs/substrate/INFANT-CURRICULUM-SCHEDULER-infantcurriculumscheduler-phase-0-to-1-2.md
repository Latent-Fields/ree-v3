## InfantCurriculumScheduler Phase 0->1 crossing-count criterion (V3-EXQ-591f; GAP-14 c-2) (2026-06-19)
- experiments.infant_curriculum.InfantCurriculumScheduler Phase 0->1 crossing-count
  advancement criterion -- IMPLEMENTED 2026-06-19. infant_substrate:GAP-14 defect
  (c-2) gate-over-permissiveness, resolved at the criterion level by V3-EXQ-591f PASS
  (2026-06-15; recommended_criterion=crossing_count) and wired here via
  /implement-substrate. Module: ree-v3/experiments/infant_curriculum.py
  (experiment-harness helper, NOT a ree_core substrate scheduler -- lives alongside
  StepHarness; same precedent as the 2026-05-31 H_POS_FRAC_OF_MAX recalibration).
  Two new constructor kwargs (NOT REEConfig / from_dims -- the scheduler is
  instantiated directly; matches on_phase3_entry precedent), both no-op default:
    phase_0to1_use_crossing_count (bool, default False) -- master switch.
    phase_0to1_crossing_count_min (int, default 3 = PHASE_01_CROSSING_COUNT_MIN,
      the 591f CROSSING_COUNT_MIN).
  Criterion (when ON): _try_phase_0_to_1 accumulates self._phase01_crossing_count =
  the number of post-PHASE_EP_MIN[1] (>=100) episodes whose h_pos crosses the SPIKE
  bar (H_POS_FRAC_OF_MAX * ln(grid_cells) ~= 0.994 at grid=12) and advances Phase
  0->1 once the count reaches phase_0to1_crossing_count_min. Mirrors the 591f offline
  _advance_crossing_count(seq, spike_threshold, ep_min, min_count) replay EXACTLY
  (contract test asserts online==offline on a genuine-explorer and a seed-45-like
  sequence). The no-telemetry path (h_pos is None) is UNCHANGED under both criteria
  (hard episode-count minimum alone governs -- the crossing-count gate has no signal
  to count). Crossings before ep_min are never counted (the gate returns early below
  PHASE_EP_MIN[1], matching the range(ep_min, len(seq)) replay).
  WHY crossing-count: 591f swept four sustained-level candidates (EMA-of-level,
  window-mean, EMA-with-hold, crossing-count) on the diversity-armed 591c/591d
  reachability traces; only crossing-count>=3 DISCRIMINATED -- it ADMITS genuine
  explorers (seeds 42/43/44 cross the spike bar 7/6/36x post-ep_min) and REJECTS the
  seed-45 false-advancer (one transient h_pos_max=1.453 spike but only 2 crossings).
  591d (single-episode K-of-N) and 591e (EMA-of-level@0.20, spike-vulnerable) failed
  to discriminate. The single-episode SPIKE gate (the legacy default) advanced seed 45
  on its lone spike at ep 142 = the over-permissiveness 591f fixed.
  Backward compatible: phase_0to1_use_crossing_count=False by default -> the legacy
  single-episode SPIKE gate runs (advance on the first post-ep_min crossing); the
  crossing counter is never accumulated; bit-identical to the pre-591f scheduler.
  Every existing caller (V3-EXQ-591/591b-f, 586, 610e, 669, 667 ...) constructs
  InfantCurriculumScheduler(grid_size=...) with no new kwargs -> unaffected. 27/27
  test_infant_curriculum_gap9.py contracts (19 prior + 8 new C12: default-off
  single-episode gate / requires-min-crossings / rejects-2-crossing-false-advancer /
  admits-genuine-explorer / crossings-before-ep_min-ignored / no-telemetry-hard-count
  / custom-min / matches-591f-offline-replay) + 8/8 preflight PASS.
  phase_summary() now exposes phase_0to1_use_crossing_count + phase01_crossing_count.
  Phased training: N/A (curriculum-harness helper; no learned parameters; no gradient
  flow). MECH-094: N/A (waking-stream per-episode scheduler driven by measurement-side
  telemetry; no simulation/replay invocation site). Evidence-staleness (Step 8.5): NOT
  triggered -- no-op-default flag; every existing experiment uses the default (legacy
  single-episode gate), so no dependent claim's measured mechanism changed. KEEP all
  evidence.
  GOVERNANCE: PROMOTES NOTHING. GAP-14 STAYS blocked_pending_substrate: this lands the
  (c-2) crossing-count wiring but GAP-14 also requires (c-1) seed-46 exploration-strength
  collapse to resolve (still blocked on modulatory-bias-selection-authority; Q-043/667
  magnitude sweep exhausted, 667a not yet queued). ARC-046 / DEV-NEED-008 stay
  candidate / blocked; claims.yaml NOT modified. The full curriculum-vs-flat
  EXQ-ISEF-005 (V3-EXQ-591 successor) stays blocked until BOTH legs land.
  Validation: this is the WIRING of an already-validated criterion (591f PASS is the
  evidence), not a new mechanism -- the substrate-readiness gate is the contract suite
  (online==591f-offline replay). No new EXQ queued here; the next experiment is the
  EXQ-ISEF-005 successor, which stays blocked on (c-1).
  Plan node: REE_assembly/evidence/planning/infant_substrate_plan.md
  (infant_substrate:GAP-14). 591f manifest: REE_assembly/evidence/experiments/
  v3_exq_591f_isef005_phase01_gate_criterion_20260615T115131Z_v3.json.
  See infant_substrate:GAP-14 (closure node; c-1 + c-2 legs), V3-EXQ-591f (the PASS
  this wires; recommended_criterion=crossing_count), V3-EXQ-591d/591e (the K-of-N /
  EMA-of-level candidates that failed to discriminate), the 2026-05-31
  H_POS_FRAC_OF_MAX recalibration above (prior GAP-14 prereq (c) work on the same
  gate), ARC-046 / DEV-NEED-008 (gated claims; unchanged), Q-043 / V3-EXQ-667 /
  modulatory-bias-selection-authority (the orthogonal c-1 exploration-strength thread),
  MECH-094 (N/A -- waking-stream scheduler).
