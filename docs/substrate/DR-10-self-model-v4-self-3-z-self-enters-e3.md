## DR-10 (self_model_v4:SELF-3): z_self enters E3 trajectory viability scoring (2026-07-01)
- DR-10: ethics_engine_3.self_viability_weighting -- IMPLEMENTED 2026-07-01.
  The E3-scoring half of the MECH-215 unblock, built on the same-day DR-13 stateful z_self
  (SELF-1, validated V4-EXQ-002 PASS). generation:v4, off the V3 critical path; PROMOTES
  NOTHING in V3. User-approved graduation (AskUserQuestion 2026-07-01, caller-supplied v1).
  Modules: ree_core/predictors/e3_selector.py (E3TrajectorySelector._self_viability_penalty
  helper + score_trajectory penalty term + select() per-candidate threading + 4 diagnostics),
  ree_core/utils/config.py (E3Config 4 fields + REEConfig.from_dims passthrough),
  ree_core/agent.py (_injected_self_viability attr + set_injected_self_viability() seam +
  select_action version-layering-guarded passthrough).
  Problem (v4_spec V4-2 DR-10): E3.score_trajectory scores purely over z_world (F/M/goal);
  there is NO z_self term in viability, so a trajectory scores identically whether the agent
  is fresh or depleted/damaged. DR-10 makes bodily capacity/affect/damage state (read from
  the DR-13 stateful z_self) gate which trajectories are viable for THIS agent. Sibling lever
  to DR-12 on EXISTING machinery; no learned parameters; needs a STABLE z_self subject
  (hence the DR-13/SELF-1 dependency).
  THE LEVER (no-op default; bit-identical OFF): in score_trajectory (score is a COST,
  lower-is-better), when use_self_viability_weighting AND a per-trajectory self_viability is
  supplied AND self_viability_weight != 0.0:
    score = score + self_viability_weight * penalty(self_viability)
  penalty monotone non-decreasing in the cost (clamped >=0): mode "linear" (penalty=sv) |
  "saturating" (penalty = 1 - exp(-sv/self_viability_scale) in [0,1)). Threaded PER-CANDIDATE
  via select(self_viability_per_candidate=[K]) so a varying cost can change the committed
  argmin -- a UNIFORM scalar is argmin-invariant (V3-EXQ-571 lesson; C3 contract pins this).
  Diagnostics on last_score_diagnostics: self_viability_active, self_viability_weight,
  self_viability_range (the pilot's non-vacuity gate), self_viability_penalty_range.
  Config (E3Config + REEConfig.from_dims, all no-op default): use_self_viability_weighting
  (False, master), self_viability_weight (0.0), self_viability_mode ("linear"),
  self_viability_scale (1.0). Default OFF -> the penalty block is skipped -> bit-identical.
  Per-candidate self-viability source (v1 scope, user-confirmed AskUserQuestion 2026-07-01):
  CALLER/AGENT-SUPPLIED, derived from the DR-13 stateful z_self. REEAgent.select_action plumbs
  an optional injected per-candidate self-viability (agent._injected_self_viability via
  set_injected_self_viability(); default None -> bit-identical; version-layering guard so the
  default V3 path never sends the kwarg). DOCUMENTED FOLLOW-ON (NOT v1): an ecological
  z_self-derived auto-source (allostatic z_self-deviation x per-candidate demand, or a learned
  z_self->viability head needing phased training + SELF-2's per-candidate self-transition).
  Backward compatible: disabled by default; existing experiments unaffected. preflight/full
  suite green + 8 new contracts in tests/contracts/test_dr10_z_self_viability.py (C1 OFF /
  weight-0 / no-signal all bit-identical / C2 differential flips selection / C3 uniform
  argmin-invariant / C4 linear==cost + saturating bounded-monotone + negative clamped).
  Phased training: N/A (no learned parameters; pure arithmetic on a supplied per-candidate cost).
  MECH-094: N/A -- waking action-selection scoring; no replay/memory write surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default lever; every existing
  experiment uses the default (OFF). KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-215 (unblocked by DR-10 + DR-12) and ARC-081 stay
  candidate / implementation_phase=v4; DR-10 is the z_self-viability half -- experiments +
  the ecological source remain, so v3_pending/status untouched; claims.yaml gets
  implementation_notes only.
  Validation experiment: V4-EXQ-003 (DR-10 pilot) queued via /queue-experiment -- controlled
  caller-supplied self-viability probe; FALSIFIER = if a decisive per-candidate self-viability
  does NOT change selection vs OFF, DR-10 buys nothing (pre-registered non-vacuity + decisiveness
  gates + inert off-ramp). unblocks_claims=MECH-215/ARC-081.
  Design doc: REE_assembly/docs/architecture/dr10_z_self_in_e3_viability.md.
  Plan node: REE_assembly/evidence/planning/self_model_v4_plan.md (self_model_v4:SELF-3).
  See DR-13/SELF-1 (the stateful z_self subject), DR-12/SELF-4 (sibling lever), MECH-215 +
  ARC-081 (unblocked halves), ARC-016 (E3 dynamic precision precedent), MECH-094 (N/A).
