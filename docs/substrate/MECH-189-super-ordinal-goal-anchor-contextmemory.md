## MECH-189: Super-ordinal goal-anchor ContextMemory writes substrate (infant_substrate:GAP-11) (2026-06-09)
- MECH-189: development.super_ordinal_goal_formation -- IMPLEMENTED 2026-06-09
  (substrate; v3_pending until the validation EXQ PASSes). Closes the
  infant_substrate:GAP-11 / DEV-NEED-006 substrate gap routed via
  /implement-substrate: the "ContextMemory writes substrate" the V3-EXQ-588 FAIL
  (failure_autopsy_V3-EXQ-588_2026-05-19, non_contributory for MECH-189) was
  deferred to. 588 measured the within-episode GoalState attractor (MECH-112 /
  DEV-NEED-006), NOT the child-phase ContextMemory super-ordinal write path the
  claim describes -- and that path did not exist (developmental_needs_register.md:136:
  "MECH-189 needs cue-indexed persistent goal-anchor writes and an adult z_goal
  seeding readout before it can be gated"). Do NOT re-queue V3-EXQ-588; the
  MECH-189 retest is a 588 successor with a NEW letter.
  ROOT-CAUSE FRAMING: the IncentiveTokenBank (SD-057) is per-object + per-episode
  (GoalState.reset() clears it); the ghost-goal bank (MECH-292) is over
  hippocampal anchors; GoalState resets z_goal each episode. No CROSS-EPISODE
  super-ordinal z_goal store existed -- the persistence that distinguishes a
  childhood-formed goal hierarchy from an episodic z_goal.
  Module: ree_core/goal.py (new SuperOrdinalGoalMemory class + 11 GoalConfig
  fields, sibling to IncentiveTokenBank). AGENT-owned (REEAgent.super_ordinal_goal_memory),
  NOT reset by per-episode agent.reset() -- cross-episode persistence is the
  point. Pure stateful tensor store + cosine arithmetic; no nn.Module, no
  trainable parameters, no gradient flow; no phased training.
  Store: N cue-indexed slots, each (key = z_world context [world_dim], value =
  z_goal anchor [goal_dim], strength). Cue-indexed retrieval by cosine match.
  WRITE (child phase only, wired at agent.update_z_goal AFTER GoalState.update):
  the current z_goal is written keyed on the current z_world context iff the
  MECH-189 conjunction holds -- (a) high salience: salience = benefit_exposure *
  (1 + drive_weight * effective_drive) >= super_ordinal_salience_threshold (the
  "large benefit spike"; routine contacts do not clear it, transient-benefit
  patches do); AND (b) high contextual complexity >= super_ordinal_complexity_threshold.
  The complexity signal is PLUGGABLE (super_ordinal_complexity_mode) because it is
  the DEV-NEED-024 open question ("what contextual-complexity threshold triggers a
  write") -- to be adjudicated by the validation EXQ + a follow-on lit-pull, NOT
  hard-coded: "novelty" (default, self-contained: complexity = 1 - max cosine to
  occupied anchor keys; empty store -> 1.0 bootstraps, covered contexts -> low ->
  no write = selective neoteny/adult stability) or "external" (caller-supplied, so
  an experiment can drive it from E1 cue-context entropy / PE without coupling the
  substrate to those channels). ALLOCATE-vs-REINFORCE (gate (b) governs FORMATION
  only): gate (a) salience is required for any write; a contact within
  super_ordinal_merge_similarity of an existing anchor REINFORCES it (EMA blend
  toward the CURRENT z_goal, raise strength) on salience alone REGARDLESS of
  complexity -- a recurring high-salience context strengthens its meta-goal toward
  the matured z_goal; gate (b) complexity governs only ALLOCATION of a NEW anchor
  (empty slot, else weakest). (Without this split the anchor freezes at the tiny
  z_goal captured at its first contact -- surfaced + fixed by the V3-EXQ-588c
  smoke test: anchor norm 0.019 frozen -> 0.373 matured.) Writes permitted only
  while write_enabled -- the curriculum freezes them at child->adult via
  REEAgent.set_super_ordinal_write_enabled(False) (MECH-334 on_phase3_entry
  precedent).
  READ (adult z_goal seeding readout, wired at the TOP of agent.update_z_goal):
  when the live z_goal norm < super_ordinal_seed_below_norm (default 0.4, matching
  the DEV-NEED-006 gate), retrieve the best-matching anchor for the current
  z_world; if match >= super_ordinal_seed_match_threshold, pull z_goal toward it
  via GoalState.cue_pull (no benefit pulse) by super_ordinal_seed_strength -- the
  "stored anchors bias adult z_goal seeding across novel episodes" readout.
  Config (GoalConfig + REEConfig.from_dims, all no-op default -> bit-identical OFF):
  use_super_ordinal_goal_anchors (False, master), super_ordinal_n_slots (16),
  super_ordinal_salience_threshold (0.5), super_ordinal_complexity_mode ("novelty"),
  super_ordinal_complexity_threshold (0.3), super_ordinal_merge_similarity (0.8),
  super_ordinal_write_alpha (0.3), super_ordinal_seed_below_norm (0.4),
  super_ordinal_seed_match_threshold (0.3), super_ordinal_seed_strength (0.1).
  Agent methods: set_super_ordinal_write_enabled(bool) (curriculum freeze hook),
  reset_super_ordinal_anchors() (developmental-stage clear; NOT per-episode).
  Backward compatible: use_super_ordinal_goal_anchors=False by default ->
  agent.super_ordinal_goal_memory is None; both update_z_goal hooks skipped ->
  bit-identical (verified: default vs explicit-False z_goal identical; store stays
  None after update). 985/985 contracts + 7/7 preflight PASS; 8 new contracts in
  tests/contracts/test_mech_189_super_ordinal_goal_anchors.py (C1 default-OFF +
  agent no-op / C2 write conjunction / C3 reinforce-vs-allocate / C4 retrieve +
  complexity / C5 freeze + MECH-094 sim no-op / C6 cross-episode persistence +
  reset_anchors / C7 agent forms anchor on forced high-salience update_z_goal /
  C8 agent adult-seeding pulls a fresh sub-floor z_goal toward a frozen childhood
  anchor in a matching context).
  MECH-094: waking-only; SuperOrdinalGoalMemory.write(simulation_mode=True) is a
  no-op (replay/DMN must not form super-ordinal anchors). Evidence-staleness
  (Step 8.5): NOT triggered -- no-op-default flag; every existing experiment uses
  the default (store off), so no dependent claim's measured mechanism changed.
  KEEP all evidence.
  GOVERNANCE: MECH-189 NEITHER promoted NOR weakened; stays candidate / confidence
  0.0; claims.yaml carries only an implementation_note. The illusory conflict_ratio
  from the 588 does_not_support stands as already adjudicated by the autopsy.
  Validation experiment: V3-EXQ-588c (NEW letter, supersedes the 588 chain's MECH-189
  framing; claim_ids=["MECH-189"], experiment_purpose=diagnostic, supersedes
  v3_exq_588_isef002...) -- child-phase forced-feed write across episodes -> freeze
  via set_super_ordinal_write_enabled(False) -> adult episodes with fresh sub-floor
  z_goal (goal_state.reset()) seed z_goal from anchors WITHOUT a benefit pulse;
  ARM_ON vs ARM_OFF x 3 seeds. LOAD-BEARING acceptance is DISCRIMINATION (ARM_ON adult
  median z_goal > DISCRIM_FLOOR 0.1 AND > per-seed ARM_OFF + 0.1 on >=2/3 seeds) -- the
  substrate question; the 588 / DEV-NEED-006 0.4 crossing is ADVISORY (reported, not
  gating) because the matured-z_goal anchor norm ceilings at ~0.37 on the
  untrained-encoder readiness harness (forced-feed z_world-EMA asymptote ~0.9*||z_world||),
  so 0.4 is regime-bound -- a near-miss with strong discrimination validates the substrate
  and routes the absolute gate to a trained-encoder evidence successor, NOT a failure.
  Readiness/non-vacuity (else substrate_not_ready_requeue): ARM_ON anchors form
  (n_occupied>0), seeding fires (n_seeds>0), positive-control adult z_goal>0. Dry-run PASS
  (C1 discrimination 1.0; ON 0.28 vs OFF 0.0). Queued via /queue-experiment.
  Design doc: REE_assembly/docs/architecture/mech_189_super_ordinal_goal_anchors.md
  Closure node: REE_assembly/evidence/planning/infant_substrate_plan.md
  (infant_substrate:GAP-11).
  See MECH-189 (this claim), SD-057 IncentiveTokenBank (distinct per-object/
  per-episode store; cue_pull reused for the seed), MECH-292/293 ghost-goal bank
  (distinct hippocampal-anchor store), GoalState (episodic z_goal attractor;
  unchanged), MECH-112/116/117 (GoalState claims), INV-037/038 (stored-vs-active),
  INV-041/055/056 (childhood prerequisite / selective neoteny), SD-016
  (cue-indexed ContextMemory semantics), MECH-329 (wanting-before-liking child of
  MECH-189), MECH-334 (crystallization freeze-hook precedent), V3-EXQ-588 (the FAIL
  this addresses), DEV-NEED-006 / DEV-NEED-024, MECH-094 (call-site scoping).
