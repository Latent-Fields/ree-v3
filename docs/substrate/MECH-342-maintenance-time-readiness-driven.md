## MECH-342: Maintenance-time readiness-driven commitment-release coupling (B3b) (2026-06-02)
- MECH-342: control_plane.commit_maintenance_release -- IMPLEMENTED 2026-06-02.
  Release-side complement to the MECH-090 commit-entry R-c readiness
  conjunction (which the V3-EXQ-592f autopsy + MECH-090 release-path audit +
  motor-cessation lit-pull established is ADMISSION-ONLY by design). The SAME
  two R-c readiness signals MECH-090 AND-composes to ADMIT a commitment here
  drive a graded, bounded-accumulation RELEASE of an already-elevated beta
  latch when they degrade mid-commitment. Closes the V3-EXQ-592f reach gap
  (predicates fire under forced beta-elevated state but produce zero
  state-occupancy suppression + zero decommit transitions). Routed by
  REE_assembly/evidence/planning/mech090_release_path_audit_2026-06-02.md
  (B1 ruled out: none of ARC-028/MECH-105 completion, MECH-091 urgency,
  V_s commit-release, SD-034 closure covers degraded-readiness mid-commitment)
  + targeted_review_mech_090_release_motor_cessation/SYNTHESIS.md (verdict B3b).
  Module: ree-v3/ree_core/policy/commit_maintenance_release.py
    (CommitMaintenanceRelease + CommitMaintenanceReleaseConfig). Pure-arithmetic
    regulator (no nn.Module, no learned params); sibling to commit_readiness.py
    (the admission-side R-c signal) in the policy package.
  Accumulator dynamics (per maintenance tick, only while beta is elevated):
    deficit_d = clip((score_margin_floor - score_margin)/score_margin_floor, 0, 1)
                (within-tick decisiveness axis; 0 when healthy OR no signal)
    deficit_n = clip((nav_floor - nav_competence)/nav_floor, 0, 1)
                (across-tick nav_competence axis; 0 when no signal)
    combined  = max(deficit_d, deficit_n)   (OR-composition; De Morgan dual of
                the MECH-090 AND admission; conflict-graded by the worse axis)
    if combined>0:  pressure += accumulation_rate*combined   (drift-to-bound)
    elif recovered: pressure  = max(0, pressure - leak_rate)  (reengagement)
    else:           pressure  unchanged                       (dead-band hold)
    fire = pressure >= release_bound                          (resets pressure on fire)
  On fire: beta_gate.release() + _committed_step_idx=0 + _committed_anchor_keys=None
    + e3._committed_trajectory=None (so the decommit is observable to the 592f
    probe). Release branch in REEAgent.select_action sits immediately after the
    MECH-091 urgency block (mirrors that template; DO NOT modify MECH-091).
    Reads decisiveness from self.e3.last_scores (available every tick; probe
    sets it directly) and nav from self.commit_readiness.get_readiness() (None
    -> nav axis inert, decisiveness alone drives). Pressure reset to 0 at commit
    ENTRY (both bistable + legacy elevate sites, legacy guarded on the genuine
    not-elevated->elevated transition) so each program accumulates independently.
  Config (REEConfig + from_dims, all default no-op): use_maintenance_release
    (False), maintenance_release_score_margin_floor (0.05),
    maintenance_release_score_margin_reengage (0.10),
    maintenance_release_nav_floor (0.3), maintenance_release_nav_reengage (0.5),
    maintenance_release_accumulation_rate (0.2), maintenance_release_leak_rate
    (0.1), maintenance_release_bound (1.0), maintenance_release_pressure_cap
    (1.5). Floors mirror the MECH-090 admission floors.
  BINDING DESIGN CONSTRAINTS (lit-pull B3b, vs autopsy's naive option-(a) sketch):
    (1) GRADED/ONLINE, not a one-shot Schmitt flag -- drift-to-a-release-bound,
        conflict-scaled by deficit magnitude (Resulaj 2009 bounded accumulation +
        Cavanagh/Frank 2011 conflict-graded STN threshold; Brandstaetter/Klinger
        contested-phase). A single below-floor tick does NOT release.
    (2) TARGETED + HYSTERETIC with reengagement -- releases only the active beta
        latch + committed trajectory (Falasconi/Arber 2025 movement-specific vs
        Wessel 2022 non-selective); separate reengage levels create a hysteresis
        band; recovery leaks pressure back (guards the premature-abort pole).
  Distinct from (falsifiable): MECH-090 admission predicate (entry-only, AND;
    this is maintenance, OR); MECH-091 (z_harm threat -- this fires with z_harm_a
    BELOW threshold); ARC-028 completion (options GOOD -- this fires when
    completion_signal LOW); MECH-269b/V_s (schema staleness -- this fires with a
    STABLE schema); MECH-340 ghost-goal (goal-appraisal timescale on the
    ghost-goal bank -- this is the active beta latch at motor-program timescale).
  Backward compatible: use_maintenance_release=False by default ->
    agent.maintenance_release is None; the select_action release branch is
    skipped entirely. 685/685 contracts + 7/7 preflight PASS with master OFF
    (bit-identical; was 685 before, +15 new MECH-342 contracts in
    tests/contracts/test_mech_342_commit_maintenance_release.py = 700 total).
  Activation smoke (2026-06-02): regulator unit dynamics (graded accumulation,
    fire after ~5 sustained-max-deficit ticks, ~10-11 at half deficit,
    reengagement leak, dead-band hold, OR-composition, MECH-094 sim no-op,
    config validation); agent-loop (default OFF -> None; OFF degraded ->
    reproduces 592f gap, beta stays elevated, no release; ON degraded(nav) ->
    releases at tick 4 + e3 pointer cleared + 1 fire; ON healthy -> no false
    abort).
  MECH-094: tick(simulation_mode=True) returns False without advancing pressure
    (replay must not abort a committed motor program). Match SD-035 / MECH-279 /
    MECH-313 / MECH-320 / commit_readiness pattern.
  Phased training: N/A (pure-arithmetic regulator; no learned parameters).
  Validation experiment: V3-EXQ-592g (queued via /queue-experiment) -- 592f-style
    controlled state-machine probe with use_maintenance_release=True; verifies
    state-occupancy suppression + decommit transition under sustained degraded
    readiness (zero in 592f) + no false abort under healthy readiness. The 592f
    manifest's pending_retest_after_substrate stays TRUE until 592g validates.
  Does NOT reopen the MECH-090 admission axis (V3-EXQ-592d 4-arm validator stays
    live; lit/exp decoupled -- this lit-pull does not weight MECH-090 confidence).
  Design doc: REE_assembly/docs/architecture/mech_342_commit_maintenance_release.md
  Audit + lit-pull: REE_assembly/evidence/planning/mech090_release_path_audit_2026-06-02.md,
    REE_assembly/evidence/literature/targeted_review_mech_090_release_motor_cessation/SYNTHESIS.md,
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-592f_2026-06-02.md.
  See MECH-090 (parent; commit-entry R-c admission), commit_readiness.py
    (admission-side sibling signal), MECH-091 (sibling release pathway; threat
    axis; template mirrored), ARC-028/MECH-105 (completion release; opposite
    regime), MECH-269b/MECH-284 (V_s release; schema axis), SD-034 (closure;
    rule-stability axis), MECH-340/ARC-079/Q-053 (goal-level disengagement),
    MECH-094 (simulation-mode call-site scoping).
