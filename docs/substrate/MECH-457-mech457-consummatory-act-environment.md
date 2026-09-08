## mech457_consummatory_act: environment.consummatory_act -- IMPLEMENTED (2026-07-25)
- mech457_consummatory_act: environment.consummatory_act -- IMPLEMENTED 2026-07-25.
  ree_core/environment/causal_grid_world.py. Adds a distinct no-move CONSUME action (index 5,
  class attr CONSUME_ACTION) so that entering a resource cell AFFORDS rather than EFFECTS
  consumption: an approach drive can extinguish on contact and hand off to a separate
  consummatory act, the dissociation V3-EXQ-781's non-extinguishing terminal drive could not
  express (leg 4 of the MECH-457 retention portfolio).
  Config: CausalGridWorldV2(consummatory_act_enabled=False) -- a direct env constructor kwarg
  (NOT a REEConfig field; the env is built by the experiment driver). Set True to enable.
  Data flow: consummatory_act_enabled -> action_dim returns len(ACTIONS)+1 (== 6) so every actor
  head sizing from env.action_dim grows 5 -> 6 with no further wiring -> step() dispatches the
  CONSUME action explicitly (NOT via _action_map, so the world-rule-shift permutation can never
  turn it into a movement) -> on move-onto-resource the resource branch sets
  transition_type="resource_contact", zero benefit reward, no drive restore, resource RETAINED
  (on_consumable_resource info flag set) -> the CONSUME action while standing on a resource cell
  effects consumption via the shared helper _consume_resource_at(cx, cy).
  Shared code path: the 118-line benefit/removal/per-axis-drive-restore/respawn/field-recompute
  block was factored out of the legacy inline resource branch into _consume_resource_at, called
  by BOTH the legacy auto-consume-on-entry path (OFF) and the CONSUME action (ON) -- consumption
  is the SAME operation whichever way it is reached; only the TIMING differs. Reward binds to the
  ACT: contact yields 0 reward, CONSUME delivers it.
  Backward compatible: consummatory_act_enabled defaults False -> action_dim == 5,
  auto-consume-on-entry, observation_dim unchanged; byte-identical to the pre-change env. No
  running experiment is affected.
  BLAST RADIUS: enabling the flag grows action_dim 5 -> 6, re-keying every actor head and
  BUSTING all cached arm fingerprints for consummatory-ON lineages (reuse correctly refuses
  across the change). Pre-change lineages keep the 5-action space and valid fingerprints because
  the flag defaults OFF.
  Minor limitation: the body_state last-action one-hot (body[5..8]) already aliases stay(4)->slot0;
  CONSUME(5) aliases the same way. A dedicated slot would change body_obs_dim and break every
  existing observation_dim, so it is left aliased -- the agent selects CONSUME via a distinct
  policy-head logit (action_dim grew), which is what the leg needs.
  Phased training required: no (no head trained). MECH-094: not applicable (no simulation/replay).
  7 new contracts C1-C7: tests/contracts/test_mech457_consummatory_act.py (action_dim 5/6; OFF
  auto-consume; ON contact-affords; ON CONSUME effects; CONSUME-off-resource == stay; un-consumed
  departure restores the grid marker; consumption path-independent OFF-entry vs ON-CONSUME).
  Consumption refactor regression-covered by the SD-049/MECH-307/SD-057/SD-037 consumption
  contracts (70 pass on the hub, ree-v3 base 120efac).
  Unblocks H-consummation-binding (the LAST open competence_floor retention leg) as a
  /queue-experiment target; the behavioural experiment is NOT queued in this build pass.
  CORRECTION (2026-07-25): the design doc's claim that "the drive half was already built" via
  goal.py refers to the HOMEOSTATIC per-axis drive, which is NOT 781's approach primitive and is
  NOT live in the mech457 bootstrap-explorer path; the drive-side extinction wiring the leg-4
  treatment arm actually needs is built separately below (mech457_approach_extinction).
  MECH-457 stays candidate/v3_pending; INV-088 unchanged; this build promotes and demotes nothing.
  See REE_assembly/docs/architecture/sd_mech457_consummatory_act.md and
  REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md +
  REE_assembly/evidence/planning/competence_floor_reposing_2026-07-25.md.
