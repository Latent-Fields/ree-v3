## MECH-340 Persistence / Efficacy Gate (2026-05-21)
- MECH-340: hippocampal.persistence_efficacy_gate -- IMPLEMENTED 2026-05-21.
  Module: ree_core/hippocampal/ghost_goal_bank.py (PersistenceAppraisal,
  _persistence_license, rank exclusion). ARC-079 / Q-053 front-runner:
  persistence of an entry as an active MECH-293 re-probe target is gated;
  disengagement is the default when license < persistence_floor.
  Data flow: optional PersistenceAppraisal passed into
  GhostGoalBank.rank() / HippocampalModule.rank_ghost_goals() /
  _propose_ghost_seeded():
    license = clip_[0,1](control_efficacy * (1 - goal_unattainability))
    exclude anchor when license < persistence_floor (SD-039 trace preserved)
  Config: GhostGoalBankConfig.use_persistence_efficacy_gate (default False),
    persistence_floor (0.05),
    persistence_default_when_appraisal_missing (1.0).
  Q-053 agent wiring (2026-05-21): REEAgent._compute_persistence_appraisal()
    maps prior hippocampal completion + E3 commitment -> control_efficacy;
    1 - goal_proximity -> goal_unattainability (one-shot; not staleness /
    failure). persistence_appraisal_compute.py + HippocampalConfig block;
    threaded through propose_trajectories / MECH-293 ghost branch.
    Reengagement-coupled disengagement STATE still deferred.
  Backward compatible: gate off -> rank() ignores appraisal, bit-identical.
  MECH-094: read-only; no write path.
  Validation: V3-EXQ-607 queued (diagnostic, priority 10;
    experiments/v3_exq_607_mech340_persistence_efficacy_gate_validation.py;
    5 sub-tests; contracts 4/4 + dry-run PASS).
  See MECH-340, ARC-079, ARC-078, MECH-292, MECH-293, Q-053.
