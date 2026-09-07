## DR-12 (self_model_v4:SELF-4): E2 forward-PE -> E3 trajectory-scoring confidence down-weight (FIRST V4 SUBSTRATE BUILD, 2026-06-17)
- DR-12: ethics_engine_3.pe_conditioned_confidence_weighting -- IMPLEMENTED 2026-06-17.
  THE FIRST-EVER V4 substrate build (generation:v4, off the V3 critical path; PROMOTES
  NOTHING in V3). User-APPROVED via self_model_v4_plan.md SELF-4 graduation_decision_2026_06_16.
  Modules: ree_core/predictors/e3_selector.py (E3TrajectorySelector._pe_confidence_penalty
  helper + score_trajectory penalty term + select() per-candidate threading + 4 diagnostics),
  ree_core/utils/config.py (E3Config 4 fields + REEConfig.from_dims passthrough),
  ree_core/agent.py (_injected_e2_forward_pe attr + set_injected_e2_forward_pe() seam +
  select_action passthrough).
  Problem (v4_spec V4-2 DR-12): E3.score_trajectory scores purely from the E2-rolled-out
  world_states, trusting them as if E2 were reliable everywhere; high E2 forward-PE in a
  trajectory's region does NOT discount that trajectory. E3 already CONSUMES two
  PE-magnitude signals for its own dynamics (_running_variance, ARC-016 world-forward PE;
  _novelty_ema, MECH-111 E1 PE) -- DR-12 adds an E2-forward-PE confidence down-weight
  ALONGSIDE them. A NEW lever on EXISTING machinery; no learned parameters; no stateful
  z_self substrate (keys off PE magnitude present in V3 today).
  THE LEVER (no-op default; bit-identical OFF): in score_trajectory (score is a COST,
  lower-is-better), when use_pe_confidence_weighting AND a per-trajectory e2_forward_pe is
  supplied AND pe_confidence_weight != 0.0:
    score = score + pe_confidence_weight * penalty(e2_forward_pe)
  penalty monotone non-decreasing in PE magnitude (clamped >=0): mode "linear"
  (penalty=pe) | "saturating" (penalty = 1 - exp(-pe/pe_confidence_scale) in [0,1)).
  Threaded PER-CANDIDATE via select(e2_forward_pe_per_candidate=[K]) so a varying PE can
  change the committed argmin -- a UNIFORM scalar is argmin-invariant (the V3-EXQ-571
  deleted-broadcast lesson; C3 contract pins this). Diagnostics on last_score_diagnostics:
  pe_confidence_active, pe_confidence_weight, e2_forward_pe_range (the pilot's non-vacuity
  gate -- a flat PE cannot change selection), pe_confidence_penalty_range.
  Config (E3Config + REEConfig.from_dims, all no-op default): use_pe_confidence_weighting
  (False, master), pe_confidence_weight (0.0), pe_confidence_mode ("linear"),
  pe_confidence_scale (1.0). Default OFF -> the penalty block is skipped entirely ->
  bit-identical.
  Per-candidate PE source (v1 scope, user-confirmed AskUserQuestion 2026-06-17):
  CALLER-SUPPLIED. The lever consumes a per-candidate PE passed into select(); the DR-12
  pilot (V4-EXQ-001) is a CONTROLLED substrate-readiness probe (assigns known high/low
  per-candidate PE, tests ON-vs-OFF selection divergence). REEAgent.select_action plumbs
  an optional injected per-candidate PE (agent._injected_e2_forward_pe via
  set_injected_e2_forward_pe(); default None -> bit-identical) so the lever is reachable
  from the waking loop. DOCUMENTED FOLLOW-ON (NOT v1): an ecological region-PE auto-source
  (extend the existing global _running_variance EMA into a region-keyed E2-forward-PE map
  looked up per-trajectory) -- the only piece that adds new state, deferred to keep v1 a
  "lever on existing machinery."
  Backward compatible: use_pe_confidence_weighting=False by default. preflight 7/7 +
  from_dims activation smoke PASS; 8 new contracts in tests/contracts/test_dr12_pe_confidence.py
  (C1 OFF/weight-0/no-PE all bit-identical / C2 high-PE-on-primary-best flips selection /
  C3 uniform-PE argmin-invariant / C4 linear monotone == weight*pe + saturating bounded
  by weight / C5 negative-PE clamped no-reward). Full contract suite re-run for the OFF
  bit-identical guarantee.
  Phased training: N/A (no learned parameters; pure arithmetic on an existing PE magnitude).
  MECH-094: N/A -- waking action-selection scoring; no replay/memory write surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default lever; every existing
  experiment uses the default (lever off), so no dependent claim's measured mechanism
  changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-215 (the claim DR-10+DR-12 unblock) stays candidate /
  implementation_phase=v4 / unchanged -- DR-12 alone does not unblock it (DR-10 + experiments
  remain), so v3_pending/status untouched; claims.yaml NOT modified.
  PRECEDENT (first V4 experiment): the DR-12 pilot V4-EXQ-001 sets a NEW V4 architecture_epoch
  (per v4_spec.md:267) + V4 run_id suffix + assigns owner_exq to the SELF-4 node WHEN queued.
  Verified the generation-aware consumers keep a generation:v4 node with an owner_exq OUT of
  the V3 closure %: check_closure_drift.py:497 skips non-v3 plans; generate_closure_snapshot.py
  + serve.py read_closure segment v4 into a separate roadmap rollup.
  Validation experiment: V4-EXQ-001 (DR-12 pilot) queued via /queue-experiment -- the
  controlled probe; FALSIFIER = if PE-conditioned weighting does NOT change selection in
  high-PE regions vs the unconditional-trust baseline, DR-12 buys nothing and the wiring is
  inert (pre-registered non-vacuity gate + inert-wiring off-ramp). unblocks_claims=MECH-215.
  Design doc: REE_assembly/docs/architecture/dr12_pe_conditioned_e3_confidence.md.
  Plan node: REE_assembly/evidence/planning/self_model_v4_plan.md (self_model_v4:SELF-4).
  See MECH-215 (unblocked; E2 self-transition-accuracy half), ARC-016 (E3 dynamic precision
  -- extended to the E2 stream), MECH-111 (sibling E1-novelty PE), SD-056 (trained E2
  forward divergence -- the source of a meaningful forward-PE), DR-10/SELF-3 (the
  z_self-in-E3 half of the MECH-215 unblock), MECH-094 (N/A).
