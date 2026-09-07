## MECH-090 Commit-Entry Predicate: R-c single-gate readiness conjunction (2026-05-28)
- MECH-090 (commit-entry predicate amendment): control_plane.beta_gate.commit_entry_readiness_conjunction
  -- IMPLEMENTED 2026-05-28. Closes commitment_closure_plan.md GAP-4 at the
  substrate-readiness level (behavioural validation pending V3-EXQ-592b PASS).
  Module: ree_core/heartbeat/beta_gate.py (BetaGate.should_admit_elevation +
  __init__ kwargs use_commit_readiness_gate / commit_readiness_floor /
  commit_readiness_strict_single_candidate; get_state / reset extended with
  mech090_n_elevation_admitted / _blocked / _single_candidate /
  _last_readiness_score_margin diagnostics). ree_core/agent.py: BetaGate
  construction at REEAgent.__init__ forwards the three knobs via getattr
  fallback (config.heartbeat.use_commit_readiness_gate etc., default False)
  so the from_dims signature is unchanged. The two beta_gate.elevate() call
  sites in REEAgent.select_action (bistable branch + legacy branch) compute
  _readiness_margin and _n_candidates once from result.scores (REE
  lower-is-better -> margin = sorted(scores)[1] - sorted(scores)[0]) and
  guard the elevate call with should_admit_elevation. Bistable branch:
  gate consulted only on the not-yet-elevated transition tick. Legacy
  branch: gate consulted every committed tick (legacy semantic is per-tick
  re-evaluation); gate block treats the tick as effectively uncommitted
  (releases any prior elevation, single-stage by design).
  Reading: R-c single-gate conjunction (synthesis-strongest) per
  REE_assembly/evidence/literature/targeted_review_connectome_mech_090/
  synthesis.md (commit 9e68c5ca8a, 2026-05-28). Anchored on Cisek &
  Kalaska 2010 (affordance-competition), Hanes & Schall 1996 (FEF
  accumulator-to-threshold), Roesch / Calu / Schoenbaum 2007 (dopaminergic
  readiness signal). R-a (rv-only is correct) not defensible post-pass;
  R-b (rv-only entry + downstream propagation gate) retained as fallback
  if validation fails. Tandetnik 2021 (frontal-lesion dissociation) is
  R-b's anchor and is preserved as the fallback architecture.
  Config (HeartbeatConfig, ree_core/utils/config.py): use_commit_readiness_gate
  (bool, default False; bit-identical OFF master), commit_readiness_floor
  (float, default 0.05 -- small relative to EXQ-608 mean_top2_class_gap range
  0.27-1.96; Q-053-style calibration is a follow-on, not a precondition for
  landing), commit_readiness_strict_single_candidate (bool, default False;
  permissive single-candidate handling, strict-mode is diagnostic-only).
  All knobs NOT surfaced through REEConfig.from_dims (matches the existing
  beta_gate_bistable precedent -- callers set config.heartbeat.use_commit_readiness_gate
  directly; keeps from_dims signature unchanged to avoid concurrent-session
  conflict with the MECH-341 retune signature).
  Data flow: e3.select() -> E3SelectionResult.scores [K] + .committed bool ->
  agent.py: if result.committed and use_commit_readiness_gate:
  _readiness_margin = sorted(scores)[1] - sorted(scores)[0],
  _n_candidates = K -> bistable branch (if not is_elevated) or legacy branch
  (every committed tick) -> beta_gate.should_admit_elevation(margin,
  n_candidates) -> elevate() admitted iff margin >= floor.
  Diagnostics on BetaGate.get_state(): mech090_n_elevation_admitted /
  _blocked / _single_candidate / _last_readiness_score_margin all reset
  per-episode in BetaGate.reset(). V3-EXQ-592b reads these for the
  acceptance criteria.
  Backward compatible: use_commit_readiness_gate=False by default;
  should_admit_elevation returns True unconditionally without incrementing
  counters; agent.py skips the margin computation block entirely when the
  master flag is off (committed=True alone never enters the readiness branch).
  506/506 contracts PASS with master OFF (regression-clean 2026-05-28). 7
  unit tests on the BetaGate primitive (default no-op, gate ON admit / block,
  single-candidate permissive / strict, reset clears, backward-compat
  elevate/propagate/release) all PASS.
  MECH-094: N/A. The gate is a control-state-transition predicate at waking
  action selection; reads E3 scores; writes only the beta-elevation event,
  not memory content. No simulation-write surface. The match the SD-035 /
  MECH-279 / MECH-313 / MECH-314 / MECH-319 / MECH-320 / MECH-341 pattern.
  Phased training: N/A (pure arithmetic regulator; no learned parameters;
  no gradient flow).
  Validation experiment: V3-EXQ-592b queued as 2-arm diagnostic. ARM_0 GATED:
  use_commit_readiness_gate=True, floor=0.05, same env+seed (42) as
  V3-EXQ-592. Acceptance: total_committed_steps == 0 AND
  mech090_n_elevation_blocked >= 1 AND running_variance < commitment_threshold
  at some point during the run (confirming the gate is the load-bearing
  block). ARM_1 GATED_FORCED_READY: same gate config with experiment-side
  score_bias injection forcing margin >= 0.10 by construction. Acceptance:
  total_committed_steps > 0 AND mech090_n_elevation_admitted >= 1
  (confirming the gate does not permanently lock out commitment when
  readiness clears). Joint PASS = commitment_closure:GAP-4 partial -> done.
  Design doc: REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md
  Predecessor synthesis: REE_assembly/evidence/literature/targeted_review_connectome_mech_090/synthesis.md
  See MECH-090 (parent claim), MECH-091 (urgency interrupt; orthogonal release-side
  override; unaffected), ARC-028 + MECH-105 (hippocampal-BetaGate completion
  coupling; release side; unaffected), SD-034 / MECH-266 / MECH-267 / MECH-268
  (downstream behavioural arms; transitively unblocked via GAP-4),
  commitment_closure:GAP-4 (the closure-plan gap this amendment resolves),
  Cisek & Kalaska 2010 + Hanes & Schall 1996 + Roesch / Calu / Schoenbaum 2007
  (literature anchors R1/R2/R3), Tandetnik 2021 (R-b fallback anchor),
  MECH-094 (call-site scoping; not applicable).
