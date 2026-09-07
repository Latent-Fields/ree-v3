## scaffolded_sd054_onboarding AMEND: opt-in STRICT goal isolation (2026-07-27)

- scaffold strict goal isolation -- IMPLEMENTED 2026-07-27.
  Module: `experiments/scaffolded_sd054_onboarding.py` (harness layer; NO ree_core /
  goal.py / e3_selector.py / claims.yaml change).
  Config: `ScaffoldedSD054OnboardingConfig.scaffold_strict_goal_isolation`
  (default **False**; set True to enable).

  THE GAP IT CLOSES. `_set_goal_pipeline_frozen(agent, frozen=True)` silences the goal
  WRITE paths only -- `use_mech295_liking_bridge` and `use_mech307_conjunction`. The two
  goal READ paths are gated independently and stayed LIVE inside every stage the
  scheduler calls "frozen" (Stage-0b, P0, Stage-H):
    * E3 -- `e3_selector.score_trajectory` subtracts `goal_weight * goal_proximity`
      under `E3Config.goal_weight > 0.0` AND `goal_state.is_active()`. The trap is that
      `REEConfig.from_dims` defaults `goal_weight` to **1.0**, NOT the `E3Config`
      dataclass default of 0.0 -- so it is live for essentially every scaffold-built
      agent, including the 73 importers that never mention `goal_weight`.
    * E1 -- `GoalConfig.e1_goal_conditioned` (default True).
  Measured (460c dry-run config, 3 seeds, `e3_score_decomp_enabled`): `goal_active_frac`
  **1.000** in all three frozen stages, and removing the E3 goal term counterfactually
  moves the cost-argmin candidate on **39% / 19% / 38%** of Stage-0b / P0 / Stage-H
  ticks. So a future experiment needing a genuinely goal-free Stage-H could not get one.

  DATA FLOW (strict ON, freeze): `_set_goal_pipeline_frozen(strict=True)` ->
  `_enter_strict_goal_isolation(agent)` -> saves the prior values on the agent under
  `_scaffold_strict_goal_isolation_saved`, then sets `agent.e3.config.goal_weight = 0.0`
  (the E3 term's own `> 0.0` gate then fails, so the subtraction is SKIPPED, not scaled
  to zero) and `agent.config.goal.e1_goal_conditioned = False` (`sense()` passes
  `z_goal=None` into E1 -- the identical path E1 already takes whenever the goal is
  inactive, so nothing is reshaped). On unfreeze, `_exit_strict_goal_isolation` restores
  the **saved** values -- never a hardcoded 1.0/True, because an experiment may set a
  non-default `goal_weight`. Both writes are per-tick config reads, so the mutation is
  immediate and fully reversible; z_goal, `GoalState` and all weights are untouched.
  Restore keys off the SAVED STATE, not the caller's `strict` flag, so an unfreeze that
  forgets `strict=True` cannot leave the read paths silenced for the rest of the
  curriculum; entry is idempotent (a double-freeze cannot save the zeroed value over the
  real one).
  All five call sites (`run_stage0_nursery`, `run_stage0b_consolidation`, `run_p0`,
  `run_hazard_avoidance`, `run_p1`) thread `strict=self.cfg.scaffold_strict_goal_isolation`,
  so no stage can end up half-isolated.

  Backward compatible: **default False is bit-identical.** VERIFIED, not asserted -- a
  full 460c `--dry-run` curriculum (all 7 stages, seed 42) was run on `ree-cloud-2` from
  two clean checkouts of the same base commit differing ONLY in this file, and the full
  stdout, the complete result dict and the terminal torch/random RNG states matched
  byte-for-byte; a same-tree control pair was run alongside to establish that the
  comparison can detect a difference at all. NOTE for anyone repeating this: the 460c
  driver seeds only torch + numpy, while `ree_core/hippocampal/module.py` also draws
  from **stdlib `random`** (`random.choice` / `random.random()` in the exploration
  path), so an unseeded repeat of an IDENTICAL tree diverges from Stage-0 onward -- seed
  `random` in the harness or the A/B is meaningless.

  DO NOT flip the default. Widening the freeze unconditionally would change E3 selection
  in three stages for all 78 scaffold importers and break comparability with every
  landed scaffold run; that option was considered and deliberately declined in the
  triage (section 4). A run with the knob True is not comparable to one with it False --
  the comparison must be made deliberately, as an ON-vs-OFF arm pair inside one
  experiment. Whether silencing the term actually changes learned avoidance is
  **UNMEASURED**: the 38% argmin-flip figure is a selection counterfactual, not a DV.
  Phased training: N/A (no encoder head added). MECH-094: N/A (no simulation/replay
  content).
  Contracts: `tests/contracts/test_frozen_z_goal_scaffold_family.py` --
  `test_goal_pipeline_freeze_does_not_touch_the_read_paths` AMENDED (its AST write-set
  equality is now a behavioural default-path assertion, which is strictly stronger; the
  helper's own inline write set is still pinned to exactly the two MECH flags, the
  strict writes living in the two new helpers), plus 5 NEW tests: default-off, both read
  paths silenced under strict, saved-prior restore (non-default 0.37), idempotent
  freeze + strict-free unfreeze, and all-five-call-sites-thread-the-knob.
  Triage: `REE_assembly/evidence/planning/scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md`.
  See SD-054, MECH-295 / MECH-307 (the write paths this does not replace), MECH-112 /
  MECH-117 (the E3 goal term), SD-058 / MECH-357 and SD-059 / MECH-358 (Stage-H
  avoidance consumers that a strict-isolation arm would probe).
