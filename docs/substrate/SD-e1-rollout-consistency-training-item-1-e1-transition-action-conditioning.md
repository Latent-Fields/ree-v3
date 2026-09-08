## SD-e1-rollout-consistency-training ITEM 1: e1.transition.action_conditioning -- IMPLEMENTED (2026-08-29)
- SD-e1-rollout-consistency-training ITEM 1: e1.transition.action_conditioning --
  IMPLEMENTED 2026-08-29. `E1DeepPredictor` in `ree_core/predictors/e1_deep.py`;
  agent wiring in `ree_core/agent.py`. Full design:
  `REE_assembly/docs/architecture/sd_e1_rollout_consistency_training.md`.
  Config: `E1Config.action_conditioned_transition` (default `False`; set `True` to enable),
  `E1Config.action_dim` (default 4, wired from the same `from_dims(action_dim=...)` that feeds
  `E2Config.action_dim`, so E1 and E2 cannot disagree about the one-hot width),
  `E1Config.action_cond_unzero_self_slot` (default `True`, INERT unless the master switch is on).
  Data flow: env action -> one-hot -> `REEAgent._action_experience_buffer` -> `E1.action_encoder`
  (`Linear(action_dim, action_dim)`, E2's convention) -> per-step LSTM input
  `cat([state_i, a_enc_i])` -> `transition_rnn` -> `output_proj` -> predictions.
  Backward compatible: disabled by default; with the flag off no `action_encoder` is constructed,
  `transition_rnn.input_size` is unchanged, no construction-time RNG is consumed for the new path,
  and `predict_long_horizon` takes the pre-existing branch verbatim. Verified by running the
  V3-EXQ-954 driver `--dry-run` after the change: it reproduces the action-blindness signature
  unchanged (`cr_ratio` ~1e-07 at h=1).
  Biological basis: biological forward models are action-conditioned (efference copy); the online
  call site conditions on the action just executed, which is the efference-copy reading and the
  only action available at that point in the loop.
  Phased training required: no -- no new head trains on a moving latent target; the action encoder
  trains under the existing `compute_prediction_loss` MSE (contract asserts gradient actually
  reaches it). MECH-094: not applicable -- no new content-to-memory write path.

  WHY THE WORK ORDER INVERTED. `substrate_queue.json`'s original hint said "add a multi-step /
  rollout-consistency term to E1's training objective". V3-EXQ-954 (2026-08-29, PASS, confirmed
  autopsy `failure_autopsy_V3-EXQ-954_2026-08-29.md`) measured `cr_ratio` FLOORED AT h=1
  (4.8e-07 / 5.4e-07 against a 0.1 bar) with a flat depth profile and a healthy horizon-matched
  denominator -- the action-blindness signature, the compounding signature ABSENT. Its red-team
  pass measured a ~5,000x per-action divergence attenuation inside E1 (E2 output 2.8e-2 -> E1
  output 5.6e-6). All five lit-pull training-objective fixes presuppose an action-conditioned
  transition, so applied to E1 as it stood they would have made one trajectory more
  self-consistent while leaving the forty candidates exactly as indistinguishable.

  THIS IS THE INTERFACE FIX ONLY -- DO NOT READ IT AS "THE COLLAPSE IS FIXED". The dominant crush
  the red-team localised is ~675x at the LSTM + output_proj stage, and closing it is ITEM 2 (the
  multi-step / rollout-consistency objective, still `pending_implementation`). Measured here at
  untrained init, eval mode, h=1: mean per-action pairwise L2 is EXACTLY 0 with the flag off
  (mathematically exact action-blindness) and 2.6e-04 with it on -- signal where there was none,
  but still ~100x under E2's trained 2.8e-2. A related suspect recorded but NOT acted on:
  `output_proj` predicts the ABSOLUTE next state where E2 uses a residual `z + delta(z, a)`
  parameterisation; if the ON arm still shows crushed divergence, that is the next thing to test.

  TWO VACUITY TRAPS, both closed by instrumentation rather than by hope (the 108/108a history is
  a record of plausible-looking vacuous verdicts):
  (1) `actions=None` under the ON path falls back to a ZERO action so legacy internal callers
      keep working, and increments `E1DeepPredictor._action_cond_missing_calls`.
  (2) A driver that steps the env DIRECTLY without `select_action()` -- which the whole 108/954
      lineage does, driving with `random.randint` -- leaves `_last_action` None, so the buffer
      fills with zero actions. The counter in (1) CANNOT see this (actions are supplied, they are
      just all zero). Such a driver must call `REEAgent.record_executed_action(action)`, and must
      assert on `REEAgent.e1_action_buffer_stats()["nonzero_fraction"]` before believing its own
      arm.
  Buffer alignment (subtle): `_action_experience_buffer` entry i is the most recent executed
  action at the time state i was recorded, so the action carrying state_i -> state_{i+1} is entry
  i+1 and `compute_prediction_loss` slices `[start+1:end]`. These buffers advance at E1's TICK
  cadence, not per env step -- an inherited assumption `compute_prediction_loss` already made
  about consecutive states, not a new one, but an experiment needing exact per-step conditioning
  should run E1 at tick-every-step.
  Contracts: `tests/contracts/test_e1_action_conditioned_transition.py` (23, all pass). Note the
  harness runs E1 in `eval()` mode deliberately: `num_layers=3` means `nn.LSTM` applies
  `dropout=0.1`, and in train mode the distinctness test would PASS ON DROPOUT NOISE with the
  action channel inert -- caught during authoring, and exactly the vacuous green these tests
  exist to prevent.
  Validation experiment: EXQ pending (see substrate_queue.json).
  See INV-088 and MECH-135 (both `pending_retest_after_substrate: true`, gated on this entry),
  SD-056 (E2 action-conditional divergence preservation -- the same interface standard on E2),
  MECH-116 (`goal_input_proj`, the in-file precedent for optional conditioning, whose
  project-back-down form was deliberately NOT copied here).
