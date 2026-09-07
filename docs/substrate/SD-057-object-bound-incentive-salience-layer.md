## SD-057: Object-bound incentive-salience layer (GAP-7 L2-L3-L4) (2026-06-04)
- SD-057: drive.object_bound_incentive_salience -- IMPLEMENTED 2026-06-04
  (v1 = L2+L3+L4 core; L6 cue-recall + L7 dACC-wiring deferred to a phase-2
  pass within this SD). Closes the goal_pipeline:GAP-7 middle layer: the goal
  stream wrote a SINGLE z_goal attractor overwritten on every contact
  (wanting target == liking target always; L9 wanting!=liking dissoc stuck at
  0.0, V3-EXQ-514l). SD-057 inserts a per-object incentive layer between the
  benefit pulse and z_goal.
  Modules: ree_core/goal.py (new IncentiveTokenBank class + 5 GoalConfig
  fields + GoalState.incentive_bank member, instantiated when the master flag
  is set, reset() clears it); ree_core/agent.py (update_z_goal gains
  resource_type: Optional[int]=None kwarg + the L2-bind / L3-decay /
  L4-seed-redirect block before goal_state.update); ree_core/utils/config.py
  (from_dims surfaces all 5 knobs); experiments/_harness.py (StepHarness
  forwards obs_dict["resource_type_at_agent"] to update_z_goal).
  Mechanism (L0->L4):
    L2 MECH-344 (bind): on contact, benefit binds to the SD-049 per-type tag k
      (resource_type_at_agent / sd049_consumed_type_tag_this_tick, 1..n) ->
      IncentiveTokenBank.update(k, benefit, z_resource). The associative
      object->benefit node (BLA-analog; lit Cardinal/Everitt 2002).
    L3 MECH-345 (token): per-type bank entry holds base_value[k] (slow-decay,
      revaluable EMA of received benefit) + z_object[k] (stored z_resource
      identity embedding). Wanting at recall = base_value[k] *
      (1 + incentive_drive_kappa_weight * per_axis_drive[k]) -- the Zhang 2009
      V = r*kappa(drive) multiplier RELOCATED from the GoalState seeding gate
      onto the stored per-object value. Per-axis drive (SD-049 hunger/thirst/
      curiosity) makes wanting drive-specific / identity-matched (specific
      PIT; Corbit/Balleine 2005/2011).
    L4 MECH-346 (pointer; MECH-230 amend): z_goal seeded FROM the most-wanted
      object's embedding (k* = argmax_k wanting[k] -> seed_latent =
      z_object[k*]) instead of the raw last-contacted z_resource. GoalState
      firing gate (benefit/drive threshold) UNCHANGED -- only the seed SOURCE
      changes. So liking target (last-contacted) and wanting target (z_goal ->
      most-wanted) can DIFFER (e.g. ate food while thirsty -> z_goal points at
      water), making the L9 dissociation structurally expressible.
  Config (GoalConfig + from_dims; all no-op defaults, bit-identical OFF):
    use_incentive_token_bank (False) -- master switch.
    incentive_decay (0.005) -- per-object base_value slow decay per update().
    incentive_value_alpha (0.1) -- revaluation EMA rate on contact.
    incentive_drive_kappa_weight (2.0) -- relocated drive_weight for value x
      kappa(drive) at recall.
    incentive_use_per_axis_drive (True) -- per-axis (drive-specific) wanting.
  Backward compatible: use_incentive_token_bank=False by default ->
    GoalState.incentive_bank is None; update_z_goal takes the legacy
    single-attractor path; resource_type defaults None (2-arg callers
    unaffected). 747/751 contracts PASS (4 pre-existing local-git-env runner
    failures "Not a valid object name master", unrelated) + 7/7 preflight.
    Unit + agent-level smoke 2026-06-04: bank binds two types, most-wanted is
    drive-specific (food when hungry / water when thirsty) and differs from
    the last-contacted type; bank OFF bit-identical legacy seeding.
  Phased training: NOT required (the bank is a stateful EMA dict -- no trained
    parameters; reuses the already-trained SD-015/SD-049 z_resource encoder).
    A learned-affordance-embedding upgrade WOULD need P0/P1/P2.
  MECH-094: the bank updates only on WAKING contact via update_z_goal; no
    simulation/replay write surface, so hypothesis_tag does not apply
    (guardrail noted in the SD doc if a future revision writes during replay).
  use_resource_encoder (SD-015) must be set on cfg.latent directly (not via
    from_dims) for z_resource to populate -- SD-057's L2 bind requires it.
  Validation: behavioural L9 (wanting!=liking fraction >= 0.6, identity-probe
    > 0.6, per-axis ANOVA p<0.01) is GATED on goal_pipeline:GAP-2 supplying
    foraging contact; SD-057's own validation is a forced-contact MECHANISM
    diagnostic (two types at opposing drives, bank ON vs OFF, wanting_target
    != liking_target > 0 with bank ON / = 0 OFF) decoupled from GAP-2 --
    mirrors how V3-EXQ-626b decoupled the L1 positive control. Queued via
    /queue-experiment (claim_ids=[], diagnostic).
  Contracts: tests/contracts/test_sd_057_incentive_token_bank.py (C1 default-
    off no bank + legacy seeding / C2 L2 bind + tag-0 skip / C3 L3 per-axis
    drive-specific wanting + most_wanted / C4 revaluation + decay / C5 [1,n]
    shape robustness / C6 reset clears bank); test_step_harness_contract.py
    signature + kwargs pins updated in lockstep for the new resource_type arg.
  Design doc: REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md
  Plan-of-record: REE_assembly/evidence/planning/goal_pipeline_plan.md GAP-7.
  See SD-057, MECH-344 (L2 bind) / MECH-345 (L3 token) / MECH-346 (L4 pointer;
    MECH-230 amend), MECH-229 / MECH-117 / ARC-030 (unblocked, stay v3_pending
    until validated), SD-049 (per-type tag + per-axis drive) / SD-015
    (z_resource encoder) / SD-012 / MECH-306 (reused substrate), MECH-295
    (approach bridge; downstream consumer), MECH-292/293 (ghost-goal bank;
    inactive-anchor precedent), MECH-347 (L6) / MECH-348 (L7)
    (phase-2, landed 2026-06-04 -- see entry below), MECH-094 (call-site scoping).
