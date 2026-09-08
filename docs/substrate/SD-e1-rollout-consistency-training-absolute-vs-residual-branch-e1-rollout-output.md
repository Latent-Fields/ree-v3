## SD-e1-rollout-consistency-training ABSOLUTE-VS-RESIDUAL BRANCH: e1.rollout.output_proj_residual -- IMPLEMENTED (2026-09-01)
- SD-e1-rollout-consistency-training ABSOLUTE-VS-RESIDUAL BRANCH: e1.rollout.output_proj_residual --
  IMPLEMENTED 2026-09-01 (ree-v3 `b40139b3ed`). `E1DeepPredictor.predict_long_horizon` in
  `ree_core/predictors/e1_deep.py`. Full design:
  `REE_assembly/docs/architecture/sd_e1_rollout_consistency_training.md`.
  Config: `E1Config.output_proj_residual` (default `False`; set `True` to enable).
  Data flow: unchanged except the per-step read-out -- `transition_rnn` -> `output_proj` ->
  `predicted = state_i + output_proj(...)` instead of `predicted = output_proj(...)`, in BOTH
  rollout branches (the ITEM 1 action-conditioned one and the legacy one). `forward()` delegates
  to `predict_long_horizon`, so the one change covers both of this SD's `substrate_paths`
  (`::forward` and `::predict_long_horizon`); a contract pins that delegation rather than
  assuming it.
  Backward compatible: disabled by default. Unlike `action_encoder` this adds NO module and NO
  parameter in either setting, so construction-time RNG consumption is identical both ways --
  verified by loading the pre-change `e1_deep.py` alongside the new one from the same seed:
  identical parameters and bit-identical rollouts (max abs diff 0.0) across three shapes (legacy
  branch; ITEM 1 ON with a held action; ITEM 1 ON with `action_cond_unzero_self_slot=False`).
  V3-EXQ-965 driver `--dry-run` after the change: PASS, `missing_action_calls=0`.
  Phased training required: no -- no new head, no new parameter.
  MECH-094: not applicable -- no new content-to-memory write path.

  WHY THIS IS NOT ITEM 2, AND WHY IT IS NOT A GUESS. This is the design doc's OWN PRE-REGISTERED
  branch, quoted in the ITEM 1 entry immediately above ("`output_proj` predicts the ABSOLUTE next
  state where E2 uses a residual `z + delta(z, a)` parameterisation; if the ON arm still shows
  crushed divergence, that is the next thing to test"). V3-EXQ-965 (2026-08-30, confirmed autopsy)
  validated ITEM 1 and measured the ON arm still 25-37x short of the 0.1 `cr_ratio(h=1)` bar and
  5-7 orders below the 0.002 `e1coe_score_var` bar, so the branch condition is MET. The form is
  copied from `e2_fast.py`'s `self_forward` / `world_forward` -- no scaling, gate, or extra module
  invented. This cheap discrimination was taken BEFORE the ITEM 2 build, per this lineage's own
  recorded lesson that a 49-second probe re-scoped ITEM 1 -- and it RETURNED A NULL (V3-EXQ-968,
  below), after which ITEM 2's candidate-1 substrate landed the same day (next entry).

  DELIBERATELY INDEPENDENT OF `action_conditioned_transition`. It parameterises the state
  recurrence, not the action channel, and the discrimination it exists to enable is an A/B ON the
  ITEM 1 ON arm -- so both knobs must be separately settable. This is the OPPOSITE contract to
  `action_cond_unzero_self_slot`, which is inert unless its master switch is on; both directions
  are pinned by tests so neither can be "tidied" into the other.
  Contracts: `tests/contracts/test_e1_output_proj_residual.py` (16, all pass). Same `eval()`-mode
  harness discipline as the ITEM 1 file, and for the same reason. It pins OFF against a
  HAND-ROLLED REPLICATION of the legacy loop rather than a frozen constant, parameter identity
  across the flag, and the h=1 algebraic identity `ON == seed + OFF` -- that last one is what
  catches a knob wired to the wrong residual base, which every looser "the numbers moved" check
  passes. It deliberately does NOT assert that residual beats absolute; that is the experiment.
  Validation experiment: DONE -- V3-EXQ-968 (2026-09-01,
  `v3_exq_968_sd_e1_output_proj_residual_ab_20260901T162647Z_v3`, PASS, diagnostic,
  `evidence_direction: non_contributory`). Result: `residual_no_material_difference`.
  `cr_ratio(h=1)` absolute -> residual was seed42 2.673e-03 -> 5.909e-03 (2.21x) and seed123
  2.717e-03 -> 9.227e-04 (0.34x). DO NOT COMPRESS THIS INTO A DIRECTION: the seeds disagree in
  SIGN and NEITHER approaches the pre-registered `lift_factor_abs_floor` of 3.0 (absolute-arm
  cross-seed noise ratio 1.016); `residual_materially_exceeds` and `residual_materially_below`
  are both false on both seeds. It is not "residual is worse" and not "residual is better".
  All five readiness preconditions were met, so the comparison is real, not vacuous. The knob
  stays default-off as a CHARACTERISED NULL, not a recommendation -- and the ~675x
  LSTM+output_proj crush is therefore not a mere parameterisation artefact.
  DOES NOT unblock INV-088 / MECH-135. Both keep `pending_retest_after_substrate: true`: the bars
  are still missed by 25-37x (`cr_ratio`) and 5-7 orders (`e1coe_score_var`), so a retest must NOT
  be queued off the back of this build.
  OPEN DECISION recorded, not taken: `E1Config.action_cond_unzero_self_slot` still defaults `True`
  despite V3-EXQ-965 returning a NULL for it (C_both <= B_action on 3 of 4 h=1 comparisons). Inert
  today because its master switch defaults `False`, but enabling ITEM 1 silently enables a second
  unvalidated change alongside the one under test. Flipping a shipped default is a behaviour
  change and is owed a user decision.
