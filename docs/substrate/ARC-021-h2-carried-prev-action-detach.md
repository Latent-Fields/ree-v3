## ARC-021 H2 carried prev_action detach (2026-09-24)

- ARC-021 (H2 leg): agent.carried_prev_action_detach -- IMPLEMENTED 2026-09-24.
  `ree_core/agent.py` `REEAgent.sense()` (the `latent_stack.encode(..., prev_action=...)` call).
  Config: `REEConfig.detach_carried_prev_action` (default `False`; set `True` to enable).
  Reachable via `REEConfig.from_dims(detach_carried_prev_action=True)` (three-site pattern:
  field, from_dims signature, from_dims assignment) or by setting the attribute on a
  `REEConfig.large(...)` result before `REEAgent(cfg)`.
  Problem: `select_action` stores the emitted action in `_last_action` UNDETACHED
  (`requires_grad=True` every step), and the next `sense()` passes it as `prev_action` into
  SD-007 reafference `correct_z_world`. Step t's `z_world` therefore carried step t-1's whole
  selection graph (E1 prior -> `ContextMemory.read`, E2, E3). A single-optimizer driver that
  backprops an undetached `z_world` loss (the ARC-021 H2 MERGED arm,
  `experiments/v3_spark_arc021_three_loop_scale.py`) walks back across step t-1's Adam step and
  crashes on its second step ("modified by an inplace operation", [16,256] AsStridedBackward0).
  The originally suspected mutator, `ContextMemory.write`'s `.data` write
  (`ree_core/predictors/e1_deep.py`), was measured NOT to be the cause (it does not bump the
  version counter and is not called in the probe config); e1_deep.py is untouched.
  Data flow: `_last_action` -> (`.detach()` when flag on) -> `LatentStack.encode(prev_action)`
  -> SD-007 reafference -> `LatentState.z_world`. Forward values are bit-identical ON vs OFF;
  only the cross-step gradient path is cut. Every other `_last_action` consumer already detaches.
  Backward compatible: disabled by default; existing experiments unaffected (gradients included).
  Liveness: at `REEConfig.large`, MERGED arm 0/40 -> 40/40 clean steps with the flag ON;
  `e1.context_memory.memory` keeps receiving gradient (it remains a real merged-objective
  participant). Phased training required: no. MECH-094: not applicable.
  Contract: `tests/contracts/test_arc021_h2_merged_optimizer_backward.py` (flag ON runs clean
  + memory gets grad -- FAILS on the pre-change tree; flag OFF still crashes; forward
  bit-identical; default off).
  Reproducer: `experiments/_scratch/arc021_h2_merged_optimizer_runnability_probe.py`
  (docstring corrected 2026-09-24).
  Validation: OWED -- the ARC-021 H2 merged-vs-separate driver run must set
  `detach_carried_prev_action=True` in BOTH arms (the SEPARATE arm is unaffected by it, but the
  arms must differ only by the optimizer topology). Not queued by this build.
  See ARC-021, GFLAG-0229, REE_assembly/evidence/planning/arc021_h2_leg_blocked_substrate_merged_arm_crash_20260908.md.
