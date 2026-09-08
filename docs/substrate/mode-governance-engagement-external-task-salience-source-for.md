## mode-governance-engagement: external_task salience source for SalienceCoordinator (2026-06-13)
- mode-governance-engagement -- IMPLEMENTED 2026-06-13 (substrate; MECH-266 stays
  provisional / SD-032a stays stable -- PROMOTES NOTHING until the 464d/467d retest
  runs). The external_task salience SOURCE the SD-032a SalienceCoordinator lacked on the
  603n foraging substrate. Routed by the confirmed
  failure_autopsy_SD-034-closure-cluster-ext_2026-06-12 (sub-cluster B: V3-EXQ-464c +
  467c) via the substrate_queue mode-governance-engagement entry minted by the 2026-06-13
  AM governance cycle.
  ROOT CAUSE (code-confirmed in salience_coordinator.py): external_task gets only
  external_task_bias (1.0) + drive_level (affinity weight 1.0), while dacc_pe /
  dacc_foraging / dacc_difficulty all push internal_planning. On the foraging substrate
  drive_level ~ 0.016 (540c probe), so on tick 1 the argmax flips to internal_planning and
  the agent settles there for the episode -> fraction_in_external_task = 0.0 on both arms /
  all seeds, and the 464c/467c eval loops count one episode-initial settle per episode
  (n_switches == n_episodes), so MECH-266's exit-rail had no contested mode to bind and the
  n_switches>=1 non-vacuity gate passed VACUOUSLY.
  THE FIX (no-op-default; bit-identical OFF; mirrors the SD-035 CeA / SD-037 override
  signal-injection pattern exactly -- the SalienceCoordinator class is UNCHANGED, it
  already accepts arbitrary named signals):
    Module: ree_core/agent.py (registration at __init__ + injection at the salience tick
      site in select_action), ree_core/utils/config.py (6 no-op-default flags + from_dims).
    Registration (REEAgent.__init__, gated on use_external_task_drive + salience present):
      affinity_weights["external_task_drive"] = {"external_task": external_task_drive_affinity_weight}
      salience_weights["external_task_drive"] = external_task_drive_salience_weight
      -- registered in BOTH so external_task can win the mode argmax (affinity) AND a switch
      INTO external_task can fire the MECH-259 trigger (salience aggregate).
    Injection (select_action, BEFORE coord.tick(), alongside the aic/cea/override injections):
      engagement = goal_active ? clip(commit_w*float(beta_gate.is_elevated)
                                      + prox_w*float(goal_state.goal_proximity(z_world)), 0, 1) : 0
      coord.update_signal("external_task_drive", engagement)
    The engagement is DYNAMIC by design (gated on an active goal, graded by committed
    pursuit x proximity), so it RELEASES toward internal_planning during deliberation /
    between-goals / just-consumed -- producing GENUINE mode competition, NOT the 464b
    "100% external_task, 0 switches" saturation degeneracy (the opposite failure the
    2026-06-04 MECH-266 evidence_quality_note recorded).
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
    use_external_task_drive (False, master), external_task_drive_affinity_weight (1.0),
    external_task_drive_salience_weight (1.0), external_task_drive_commit_weight (1.0),
    external_task_drive_proximity_weight (1.0), external_task_drive_require_goal_active (True).
  Backward compatible: use_external_task_drive=False by default -> no slot registered, no
    injection, "external_task_drive" never enters the coordinator's _input_signals (tick
    reads 0) -> bit-identical. 7/7 preflight + 1031 contracts (1026 prior + 5 new in
    tests/contracts/test_mech266_external_task_drive.py: C1 OFF no-slot + bit-identical
    action stream / C2 ON registers BOTH affinity+salience slots / C3 coordinator math --
    drive raises external_task probability AND salience_aggregate over an
    internal_planning-pushed baseline / C4 agent injects engagement>0 on a goal-active
    agent + monotone (drive never reduces external_task occupancy) / C5 goal-inactive ->
    injected engagement 0, the release path) PASS. v3_exq_464c --dry-run unchanged
    (drive OFF -> reproduces the prior sym_frac=0.0 / asym_frac=0.0 substrate-ceiling
    signature).
  Phased training: N/A (non-trainable arithmetic signal injection; no learned parameters).
    MECH-094: waking-only by call-site scoping (select_action), as with the neighbouring
    AIC / CeA / override injections. Evidence-staleness (Step 8.5): NOT triggered --
    no-op-default flag; every existing experiment uses the default (drive off), so no
    dependent claim's measured mechanism changed. KEEP all evidence.
  depends_on (unresolved at landing): scaffolded_sd054_onboarding nav-competence (Stage-H).
    The substrate + contract tests land regardless (user-directed); the VALIDATION may
    self-route substrate_not_ready if the agent does not survive/forage long enough, which
    the 603n contact guard + the restated occupancy gate handle cleanly.
  Validation experiments: V3-EXQ-464d (competing-goals) + V3-EXQ-467d (mode-stickiness
    dose-response) -- successors (NEW letter, NOT supersede) of 464c/467c with
    use_external_task_drive=True AND the readiness gate RE-STATED as
    min_across_arms(fraction_in_external_task) > floor (~0.1) replacing the n_switches>=1
    non-vacuity gate, so the asymmetric exit-rail finally has a contested mode to bind.
    claim_ids=[MECH-266, SD-032a]; experiment_purpose=evidence. Queued via /queue-experiment.
    GOVERNANCE: MECH-266 stays provisional / SD-032a stays stable; claims.yaml carries only
    an implementation_note + the pending_retest_after_substrate flag added this cycle (no
    flag/confidence/promotion change). substrate_queue mode-governance-engagement ready
    STAYS false until the retest clears the occupancy gate.
  Design doc: REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
    (mode-governance-engagement section). Substrate_queue:
    REE_assembly/evidence/planning/substrate_queue.json (mode-governance-engagement).
    Autopsy: REE_assembly/evidence/planning/failure_autopsy_SD-034-closure-cluster-ext_2026-06-12.{md,json}.
  See MECH-266 (asymmetric mode hysteresis -- the exit-rail this unblocks), SD-032a
    (SalienceCoordinator -- the mode register the drive feeds), MECH-259 (switch threshold),
    SD-035 CeA / SD-037 override (the affinity+salience injection pattern this mirrors),
    SD-012 drive_level (the inadequate external_task driver this complements), MECH-295
    (goal-pursuit / approach bridge -- adjacent goal machinery), V3-EXQ-464c/467c (the FAILs
    this addresses), V3-EXQ-464d/467d (validation), MECH-094 (call-site scoping).
