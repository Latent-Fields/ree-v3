## ARC-108 JOB-2: dopaminergic control-plane DRIVER pair -- rho_t maintenance ramp + habenula negative-delta_t de-commit (the driver of the commit/maintain/de-commit machinery REE built but never gave its neuromodulator) (2026-06-22)
- ARC-108 (JOB 2): control_plane.dopaminergic_commit_maintain_decommit_driver -- IMPLEMENTED
  2026-06-22 (substrate; ARC-108 stays candidate / substrate_conditional / implementation_phase=v3 --
  this PROMOTES NOTHING. The sec-7.2 L0/L1/L2 control-plane falsifier is a SEPARATE /queue-experiment
  chip, sequenced after this build). The control-plane half of the unified dopamine substrate
  (JOB-1 = learned selection-gating; JOB-2 = the driver of the commit/maintain/de-commit control
  plane). Built on the MECH-450 base (ree-v3 9839492) after serializing behind that concurrent
  ARC-108 JOB-1 step-2 session (shared post_action_update RPE block). User-ratified V3 pull-forward
  (claims.yaml 045c4b73df). Design-of-record:
  REE_assembly/evidence/planning/unified_dopamine_substrate_design_2026-06-22.md secs 3-6/10;
  doc REE_assembly/docs/architecture/arc_108_job2_control_plane.md.
  PROBLEM (assembly-map A.6): REE built the commit/maintain/de-commit MACHINERY (MECH-090 beta-gate
  latch, the natural-commit latch-hold, the SD-034 closure operator, the de-commit refractory) and
  ran it off arithmetic readiness signals with NO dopaminergic driver. Two deviations: (B6 / 460h)
  maintenance is a FLAT hold -- the latch-hold re-asserts beta UNCONDITIONALLY each tick, no
  intrinsic decay, so it monopolises ~2400 steps; (rung-6) the de-commit fires on a refractory
  CLOCK, not on outcome content, so it is non-dissociable from the commit it releases. Biology: a DA
  ramp scaled to goal-proximity x value peaks-then-declines so it CANNOT monopolise (Howe 2013,
  Mohebi 2019); de-commit is lateral-habenula -> negative RPE, a content-driven "worse than expected"
  trigger (Matsumoto 2007, Hong 2011, Sosa 2021).
  THE PAIR (COMPOSE with MECH-090/342/SD-034 -- keep gate + operator + refractory as safety plumbing;
  REPLACE only the flat maintenance DRIVER, ADD the de-commit DRIVER; no parallel module, ARC-106 G2;
  both no-op-default -> bit-identical OFF; waking-only MECH-094):
    (c) rho_t MAINTENANCE RAMP -- ree_core/policy/rho_maintenance_ramp.py (RhoMaintenanceRamp,
      pure-arithmetic, no params). rho_t = goal_proximity(z_world) x value, formed from quantities
      REE already has (GoalState.goal_proximity in [0,1] x the benefit valuation feeding F,
      E3.benefit_eval_head clamped >= 0; built by REEAgent._compute_rho_t). The ramp tracks a running
      proximity PEAK and SELF-LIMITS (returns release) once rho_t has declined from the peak by
      >= release_margin*peak (or below hold_floor), after an onset grace. WIRING: at the
      natural-commit latch-hold RE-ASSERTION site (REEAgent.select_action), when
      use_rho_maintenance_ramp is on the UNCONDITIONAL re-assert is REPLACED by a ramp-gated one --
      the ramp self-limit is ADDED as a yield condition; ALL existing latch-hold yields are kept
      (refractory / MECH-091 threat / rung-6 release / max-ticks = the safety plumbing). The ramp
      only decides WHEN the hold ends. This is the STRUCTURAL B6 fix: a flat hold never crosses the
      decline test (no decline term) so it never self-limits (the 460h monopoly); a proximity-scaled
      rho_t peaks-then-declines by construction. PRECONDITION (loud ValueError): requires
      use_natural_commit_latch_hold=True (the hold this ramp drives).
    (d) HABENULA negative-delta_t DE-COMMIT -- ree_core/governance/closure_operator.py
      (ClosureOperator.habenula_tick + ClosureOperatorConfig.habenula_abort_enabled /
      habenula_delta_threshold + _n_habenula_aborts). The SAME signed delta_t = R_t - V-hat_t the
      ARC-108 JOB-1 slice computes in e3_selector.post_action_update (REUSED, not recomputed) is a
      new INTERNAL-scalar abort input: when delta_t < habenula_delta_threshold ("worse than
      expected") AND beta is elevated, habenula_tick fires the SAME 5-part _fire the operator
      already runs (beta release + No-Go + residue discharge + salience + PE cap) + the de-commit
      refractory. ADDED ALONGSIDE the rule-stability detector (tick) and the refractory-timer release
      -- the operator/refractory/No-Go machinery is NOT replaced. Content-driven (fires on outcome
      valence, not the clock) and DISSOCIABLE from the latch's refractory state -- the D3 property the
      rung-6 timer lineage could not produce. Internal scalar only (the routed GPi->habenula efferent
      drain stays V4). WIRING: REEAgent.update_residue reads e3_metrics["habenula_delta_t"] after
      post_action_update and routes it into closure_operator.habenula_tick; on a fire it tears down
      the committed program (beta released, e3._committed_trajectory=None, hold disarmed). Requires
      use_closure_operator=True (forwarded onto ClosureOperatorConfig via the closure_decommit_hold_ticks
      getattr-fallback pattern).
  delta_t REUSE plumbing: e3_selector.post_action_update computes delta_t + advances the shared
  V-hat_t whenever JOB-1 learned gating (or MECH-450 W_lat) OR use_habenula_decommit is on (broadened
  from the JOB-1/W_lat-only gate; the JOB-1 path is bit-identical), and emits habenula_delta_t.
  use_habenula_decommit is mirrored onto E3Config.use_habenula_decommit (so post_action_update, which
  reads the E3Config, computes the RPE) AND REEConfig (agent wiring + operator), both from one
  from_dims param.
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF): use_rho_maintenance_ramp
  (False, master; requires the latch-hold) + rho_hold_floor (0.05) + rho_release_margin (0.5) +
  rho_onset_grace_ticks (3); use_habenula_decommit (False, master; requires use_closure_operator) +
  habenula_decommit_delta_threshold (0.0 = fire on any negative RPE). E3Config gains the mirror
  use_habenula_decommit (False).
  Backward compatible: both masters default False -> agent.rho_maintenance_ramp is None (the
  latch-hold's flat re-assert is unchanged) and ClosureOperatorConfig.habenula_abort_enabled False
  (habenula_tick no-ops); post_action_update emits no habenula_delta_t; bit-identical OFF (verified:
  default == explicit-False). 8/8 new contracts in tests/contracts/test_arc108_job2_control_plane.py
  (C1 config defaults + bit-identical OFF / C2 rho ramp peaks-then-declines self-limit where a FLAT
  rho never self-limits + floor release / C3 rho MECH-094 sim no-op / C4 habenula_tick fires the
  SD-034 closure on a negative delta_t below threshold while beta elevated + no-ops disabled/beta-
  down/delta-above-threshold/hypothesis_tag + threshold respected / C5 agent preconditions + operator
  forwarding / C6 LOAD-BEARING agent-level: a DECLINING rho_t self-limits the hold where the flat
  hold ramp-OFF keeps re-asserting against churn / C7 e3 delta_t reuse: JOB-2-ON emits habenula_delta_t
  + advances shared V-hat, JOB-2-OFF emits none + JOB-1 w_chan untouched / C8 agent habenula de-commit
  end-to-end: a committed+elevated agent whose realised outcome is worse-than-expected fires the abort,
  releases beta, tears down the program) + 8/8 preflight + full contract suite 1240 passed (the 3
  failures -- control_vector C4 + 2 runner_fail_branch -- are the documented pre-existing flakes,
  CONFIRMED failing identically on a clean stash of these changes; control_vector C4 passes in
  isolation = the known order-dependent flake).
  Phased training: N/A (pure-arithmetic regulators + scalar reuse of the already-formed RPE; no
  learned parameters). MECH-094: the rho ramp's tick(simulation_mode=True) never self-limits and does
  not advance the peak; habenula_tick(hypothesis_tag=True) is a no-op; delta_t is computed only on the
  waking update_residue path. Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flags;
  every existing experiment uses the defaults, so no dependent claim's measured mechanism changed.
  KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. ARC-108 stays candidate / substrate_conditional /
  implementation_phase=v3; MECH-450 / ARC-107 / MECH-448 / MECH-449 / MECH-090 / MECH-342 / SD-034 /
  MECH-439 untouched. claims.yaml carries only an implementation_note on ARC-108 (DEFERRED behind the
  live governance-cycle session that holds claims.yaml; PROMOTES NOTHING, no flag/status change).
  Validation experiment: NOT queued here -- the sec-7.2 control-plane L0/L1/L2 falsifier
  (ramp-releases-where-flat-latch-monopolises D1, release content-driven not re-parameterised-timer D2,
  habenula de-commit dissociable D3, on the closure_exclusive_decommit_eval substrate, the 460k
  successor) is a SEPARATE /queue-experiment chip.
  Design doc: REE_assembly/docs/architecture/arc_108_job2_control_plane.md.
  See ARC-108 JOB-1 (docs/architecture/dopamine_into_gating.md; the shared delta_t / V-hat_t this
  reuses), MECH-450 (ARC-108 JOB-1 step-2 -- the maintenance ramp's selection-side twin, design sec 5;
  the base this built on), MECH-090 (beta-gate latch the ramp re-asserts), MECH-342 (maintenance-
  release sibling; degraded-readiness face), SD-034 (closure operator the habenula aborts),
  natural_commit_occupancy_release.md (the latch-hold the ramp drives + the closure_exclusive_decommit_eval
  substrate the falsifier runs on), MECH-439 (F-dominance front -- JOB-2 attacks the duration/de-commit
  face while JOB-1 / MECH-448 attack selection), ARC-109 (D1/D2 split -- V4), ARC-106 (grounding
  framework), MECH-094 (waking-only call-site scoping).