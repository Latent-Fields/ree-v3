## SD-049-PHASE-2 density-preserving spawn: per-type resource density held constant across arms (V3-EXQ-693a) (2026-07-20)
- SD-049-PHASE-2 density-preserving spawn -- IMPLEMENTED 2026-07-20 (substrate;
  env-only, no encoder / consumer change).
  Module: ree_core/environment/causal_grid_world.py (SD-049 spawn branch in reset()).
  Config: CausalGridWorldV2.sd049_preserve_per_type_density (default False =
    bit-identical to the pre-amend split budget; True to enable).
  Problem it fixes: the default spawn path draws a FIXED budget of num_resources
    cells and then SPLITS it across the active types
    (`n_to_spawn = min(self.num_resources, len(forage_pool))`), so an n-type arm
    has ~n-fold lower per-type density than a 1-type arm. Two consequences:
    (1) CONFOUND -- the SD-049 4-arm substrate gradient (ARM_0..ARM_3) varies
        per-type density as well as heterogeneity, so C_GR's ARM_2-ARM_0 lift is
        not attributable to identity alone. (Consistent with the pre-registered
        693a watch item: C_GR margin near-zero, NEGATIVE in ARM_3 -- the arm with
        the most types, hence the sparsest per type.)
    (2) CONTACT CEILING -- the mechanism behind V3-EXQ-693a's ARM_2
        behav_contact_rate 0.0099-0.0188 against a CONSUMPTION_FLOOR of 0.02.
        Measured on the default path at num_resources=5: 3 types -> by_type
        [2, 1, 2]; 5 types -> [2, 0, 1, 0, 2] (two types spawn ZERO cells).
  Data flow: reset() SD-049 spawn branch -> when the flag is on, num_resources is
    read as a PER-ACTIVE-TYPE count (`desired = num_resources * len(active_types)`)
    -> spawn budget scales with the types actually introduced by the curriculum,
    so per-type density stays constant as types come online and the arms differ
    only in heterogeneity -> obs_dict diagnostics.
    Scaling is on len(active_types), NOT n_resource_types, so a type still behind
    a resource_introduction_schedule gate does not inflate the budget.
  Diagnostics (obs_dict, always present): sd049_preserve_per_type_density and
    sd049_density_budget_truncated. The truncated flag is True when the forage
    pool capped the scaled budget, i.e. per-type density was NOT actually held
    constant -- any experiment relying on constant density MUST check it, because
    a silent truncation reproduces exactly the confound the flag exists to remove.
    Cleared per episode so the no-active-types edge case cannot leave a stale read.
  Backward compatible: default False; the budget expression and RNG draw sequence
    are unchanged on the default path, so existing experiments are bit-identical.
  Phased training required: no (env-only; no encoder head added).
  MECH-094: N/A (no simulation / replay / memory write introduced).
  Activation smoke (2026-07-20, all PASS):
    (a) default path holds the fixed total 5 at 1/2/3/5 types (split preserved);
    (b) flag on -> totals 5/10/15/25 at 1/2/3/5 types, per-type mean exactly 5.00;
    (c) truncation flag fires on a size-8 grid with desired=200 (total 32);
    (d) both diagnostics present in obs_dict on ON and OFF paths.
  Validation experiment: NOT YET QUEUED -- see the open item in the session report.
    The natural owner is a V3-EXQ-693b re-issue of 693a with the flag ON.
  Design doc: REE_assembly/docs/architecture/sd_049_multi_resource_heterogeneity.md
    (density-preserving spawn section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-693a_2026-06-21.{md,json}.
    Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json (SD-049-PHASE-2).
  Scope note: this lands the ENV half of the 693a autopsy's recommended amend. The
    curriculum-calibration half (scaffolded_sd054_onboarding Stage-H / hazard-stage
    survival, failing 2/3 seeds) lives in ree-v3/experiments/ and therefore belongs
    to /queue-experiment, not /implement-substrate; it was chipped as a separate
    session 2026-07-20. Keeping the two separable is deliberate -- it lets a retest
    attribute hazard-survival failure to resource starvation or rule it out.
  See SD-049 Phase 1 (the spawn path amended), SD-049 Phase 2 (the encoder this
    unblocks), goal_pipeline:GAP-2 (cleared 2026-06-15 on the 514 harness; this is
    the 4-arm-fork residual it did not cover), MECH-229 / MECH-230 / SD-015 /
    MECH-436 (untouched), V3-EXQ-693a (the FAIL this amend addresses).
