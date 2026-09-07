## SD-102 / MECH-482: policy.epistemic_deficit_accumulator -- IMPLEMENTED (2026-08-29)
- SD-102: policy.epistemic_deficit_accumulator -- IMPLEMENTED 2026-08-29. Full design:
  `REE_assembly/docs/architecture/sd_102_epistemic_deficit_accumulator.md`.
  `ree_core/policy/epistemic_deficit.py` (`EpistemicDeficitAccumulator` / `EpistemicDeficitConfig`).
  Fills the `per_candidate_learning_progress` slot MECH-314c's Phase-2 per-candidate extension
  reserved (ree-v3 `c0e0ce8`, 2026-08-08) but left unfilled -- the honest per-candidate 314c source
  is MECH-482 itself, per that landing's own design doc.
  Config: `REEConfig.curiosity_learning_progress_source` {broadcast, epistemic_deficit} -- EXISTING
  reserved enum value, this landing fills the "epistemic_deficit" branch (default stays
  "broadcast", bit-identical -- `self.epistemic_deficit` stays `None`, same None-when-off pattern
  as `self.e2_world_uncertainty` / `self.curiosity`). New: `epistemic_deficit_max_targets` (16),
  `epistemic_deficit_match_radius` (1.0, matches ResidueField's default RBF bandwidth),
  `epistemic_deficit_ema_alpha` (0.1), `epistemic_deficit_uncertainty_weight` /
  `_disagreement_weight` / `_persistent_pe_weight` (1.0 each).
  Data flow: two-phase, mirroring ResidueField's persistent spatially-indexed accumulator (the
  existing precedent for this shape). UPDATE (post-hoc, `REEAgent._update_epistemic_deficit`,
  called from `sense()` right after `_train_e2_world_uncertainty`, same realized-transition cache
  cadence): `(z_world_prev, action_taken, z_world_now)` -> `e2.world_forward` point prediction +
  `E2WorldUncertaintyHead.forward` median quantile prediction + `predictive_variance` ->
  `deficit_input = w_u*uncertainty + w_d*disagreement + w_pe*persistent_pe` -> nearest existing
  target within `match_radius` EMA-updated, or a new target allocated (lowest-deficit target
  evicted at `max_targets` capacity). READOUT (pre-hoc, `REEAgent._curiosity_per_candidate_
  learning_progress`, called from the same `select_action()` site as MECH-314b): this tick's K
  candidate `e2.world_forward` first-step predictions -> nearest-target lookup (read-only) -> `[K]`
  persistent deficit vector -> `StructuredCuriosity.compute_score_bias(per_candidate_learning_
  progress=...)`.
  Candidate inputs (conservative subset per claims.yaml MECH-482 notes): candidate-specific
  predictive uncertainty (SD-063 head, same source 314b's Phase-2 path reads), persistent
  prediction error (REALIZED `||z_world_now - e2.world_forward(...)||`, distinct from 314c's
  `_lp_ema` rate-of-change EMA), predictive-system disagreement (`||e2.world_forward(...) -
  head.forward(...)[median]||` -- two independently-parameterized predictors, MSE-trained vs
  pinball-trained, sharing no parameters). MECH-441's `ModelDisagreementEnsemble` was considered
  and REJECTED as the disagreement source (separate not-built-by-default claim, undeclared
  cross-claim dependency). NOT implemented: failed-replay-resolution / competence-blocking-
  uncertainty inputs (no memory/replay visibility at this integration point) and the full
  `importance x uncertainty x expected_resolvability x persistence` multiplicative formula from
  the claim's title (no `importance` / `expected_resolvability` signal exists in the substrate;
  manufacturing one would be the vacuous-channel risk the 314bc design doc warns against) -- v1
  scope is a persistence-weighted ADDITIVE combination of the three available proxies.
  Readiness gate (binding, corrected ARC-065 gate per `mech314bc_percandidate_extension_staged_
  2026-08-08.md` section 5): READOUT refuses (returns `None` -> Phase-1 broadcast fallback) unless
  the K-candidate batch `predictive_variance` read yields `e2_world_uncertainty_last_pvar_
  relative_spread > 0` this tick -- absolute range alone is necessary but NOT sufficient. A
  refusal calls `accumulator.mark_vacuous_readout()` (self-report, observable via `get_state()`)
  rather than silently reading as "no deficit anywhere." UPDATE is not gated on this (accumulates
  unconditionally every waking tick, mirrors `update_prediction_error`'s always-on cadence).
  MECH-094: `update()` takes `simulation_mode` (no-op, mirrors `update_prediction_error`); no
  memory/replay write surface (waking online read/accumulate, same posture as the SD-063 head).
  Phased training: N/A -- pure arithmetic, no learned parameters, no `nn.Module` (same posture as
  `StructuredCuriosity`).
  Episode lifecycle: `reset()` clears all persistent targets, called alongside
  `StructuredCuriosity.reset()`'s own per-episode 314c LP-EMA clear (MECH-482 is architecturally
  314c's genuine source, inherits the same episode-scoping convention).
  Backward compatible: disabled by default (`curiosity_learning_progress_source="broadcast"`).
  Contracts: `tests/contracts/test_mech_482_epistemic_deficit_accumulator.py` (18 tests, all
  green) -- accumulator unit tests (config validation, target match/create/evict, EMA persistence,
  MECH-094 no-op, readout matching semantics, reset, diagnostics) + agent wiring (OFF path never
  instantiates the accumulator; ON path accumulates across a real rollout; readiness gate refuses
  and self-reports on a single-candidate tick; per-episode reset clears both the accumulator and
  its one-tick-lag prev-z_world cache; simulation_mode does not update). Full existing MECH-314 /
  SD-063 contract suites re-run green (75 tests) confirming no regression.
  Validation experiment: V3-EXQ-964 queued (`/queue-experiment`, `EXPERIMENT_PURPOSE=diagnostic`
  -- substrate readiness, not MECH-482's own claim hypothesis). 2-arm yoked pair (314c source
  broadcast vs epistemic_deficit, 314a/314b OFF on both arms to isolate the factor), SD-063 head
  trained identically on both arms. C1 (load-bearing): accumulator becomes live (`n_targets>0` AND
  `n_updates>0`). C2 (load-bearing): downstream consumer can diverge (yoked divergence vs the
  broadcast reference `> 0` on at least one seed). `--dry-run` smoke: all three checks OK
  (accumulator live, updates fire, self-yoked instrument control `== 0`).
  Governance posture: MECH-482's own `claims.yaml` entry gets an `implementation_note` only (this
  landing does not resolve its non-degeneracy precondition's confirming/falsifying signature,
  which needs the validation experiment's result). Downstream claims naming MECH-482 in
  `depends_on` (MECH-483/Q-089/ARC-121/MECH-485/MECH-487/MECH-493) each still have OTHER unmet
  dependencies, so none had `v3_pending` flipped by this landing. ORNT-2's `status` flip
  (open -> in_progress) is left to `/governance` per this chip's brief, not applied here.
  See MECH-314/314a/314b/314c (parent + siblings), ARC-065 (parent architectural slot), SD-063 (the
  uncertainty-head keystone this reads from), MECH-483/Q-089 (downstream, ORNT-3/ORNT-4).
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

- SD-e1-rollout-consistency-training ITEM 2: e1.transition.rollout_consistency --
  IMPLEMENTED 2026-09-01. `E1DeepPredictor.rollout_consistency_loss` in
  `ree_core/predictors/e1_deep.py`. Full design:
  `REE_assembly/docs/architecture/sd_e1_rollout_consistency_training.md`.
  Config: `E1Config.e1_rollout_consistency_enabled` (default `False`; master switch),
  `_weight` (1.0, caller-side scaling -- the helper returns the UNWEIGHTED horizon-mean),
  `_horizon` (5), `_horizon_weights_decay` (1.0 = uniform; <1.0 = TD-MPC discounting).
  Data flow: `initial_state` + `actions` -> `predict_long_horizon` (autoregressive rollout) ->
  per-step MSE against the OBSERVED latent trajectory -> `L = sum_t (decay**t * L_t) / sum_t
  decay**t`.
  Backward compatible: disabled by default, and the flag gates an OBJECTIVE not the forward
  path -- turning it on does not perturb prediction at all (contract-pinned, both action-
  conditioned and legacy). Adds NO module and NO parameter in either setting, so
  construction-time RNG is identical both ways -- verified against the pre-change `e1_deep.py`
  from the same seed: identical parameters and bit-identical rollouts (max abs diff 0.0) across
  legacy / ITEM 1 ON with a held action / ITEM 1 ON + `output_proj_residual`.
  Phased training required: no -- no new head trains on a moving latent target.
  MECH-094: carried anyway as a `simulation_mode` gate returning zero, matching SD-056's helper
  convention -- replay / DMN paths cannot recruit the objective.

  WHAT THIS ADDS OVER `REEAgent.compute_prediction_loss`, WHICH NEARLY STOPPED THIS BUILD. That
  method ALREADY rolls E1 out autoregressively to `prediction_horizon` and MSEs the whole
  trajectory, and post-ITEM-1 already supplies the executed action sequence -- so the multi-step
  FORM was present on the agent-loop path and candidate 1 is NOT the greenfield build the ranked
  list implies. Two real gaps: (i) `F.mse_loss` weights every horizon step EQUALLY, and under an
  autoregressive rollout deep-step error is larger by construction, so a flat mean lets the
  deepest steps dominate the gradient; `decay < 1.0` is TD-MPC's actual form, and at `decay=1.0`
  the helper reduces to the flat form to within float32 reduction-order error (the helper reduces per-step then weights; F.mse_loss reduces over all elements at once -- mathematically equal, different summation order, measured 6.4e-08 RELATIVE at worst and bit-identical on the legacy branch), so the discount is the only axis
  added. The contract pins that at `rtol=1e-6`, NOT bit-exactly -- an earlier revision
  asserted bit-identity and passed on ree-worker-4 while failing on darwin-arm64, the
  machine-class flakiness CLAUDE.md's test-suite note warns about. (ii) `compute_prediction_loss` is reachable only
  through the agent loop, and EVERY driver in this SD's own lineage bypasses it, training E1
  single-step teacher-forced -- `F.mse_loss(e1_pred[:, 0, :], ...)` at V3-EXQ-954:312, 965:409,
  968:431. So the multi-step objective has never once been exercised in the lineage that
  motivated this SD. The design doc's defect (a), "E1 is trained at `horizon=1`", is true of
  those DRIVERS, not of the substrate.

  `compute_prediction_loss` is DELIBERATELY NOT REWIRED to use the discount -- that path is
  depended on by several hundred experiments and there is no consumer yet. Agent-loop wiring is
  held until the validation experiment reports.

  WHY CANDIDATE 1 AND NOT A ROLLOUT-ENDPOINT CONTRASTIVE. A contrastive over candidate action
  SEQUENCES (`e1_rollout_sequence_divergence_*`) was designed and deliberately NOT built. It is
  genuinely DISTINCT from the synthesis's de-prioritised #5 "contrastive next-state" -- #5
  constrains the one-step transition, this would constrain the ITERATED MAP, which is what the C3
  evaluator consumes (it scores 40 SEQUENCES) -- so #5's "TD-MPC went the other way" does not
  settle it. It was rejected on #5's OTHER objection: "no long-horizon anchor found" applies to a
  long-horizon contrastive with MORE force, not less. The in-repo precedent is not a warrant
  either: E2's SD-056 multi-step contrastive amend (`e2_fast.py`, 2026-05-31) has
  `e2_action_contrastive_multistep_enabled` true in ZERO runs across the whole evidence corpus
  (measured 2026-09-01) -- built, never validated. The remaining argument for preferring it (that
  an accuracy objective "already trains" the crushed weights so would not move them) was
  INTUITION, not measurement, and is recorded as such: no experiment in this lineage has ever
  trained E1 multi-step at all. Candidate 1 is both the doc's ranked-strongest and the untried
  one; a null on it narrows ITEM 2's target and buys the contrastive with evidence.
  Contracts: `tests/contracts/test_e1_rollout_consistency_loss.py` (24, all pass; 63 pass across
  all three E1 contract files). Same `eval()`-mode discipline as its siblings and for the same
  reason -- `transition_rnn` has `dropout=0.1` at `num_layers=3`, and a first A/B pass reported
  spurious ~2.7e-03 diffs from dropout alone before the harness was corrected. It pins the
  flat-form identity at `decay=1.0` (float32-eps tolerance, not bit-identity), the discount's SIGN (against error concentrated at the deep
  end, so an inverted exponent fails), that gradient reaches BOTH `output_proj` and
  `transition_rnn` (the ~675x crush's location -- an objective that cannot deliver gradient there
  cannot move it), that deep-step-only error still produces gradient (otherwise this is a
  single-step loss wearing a horizon argument), hidden-state save/restore, the MECH-094 gate,
  grad-connected degenerate returns, horizon clamping, fail-closed shape validation, and all
  three `from_dims` sites. It deliberately does NOT assert that multi-step beats single-step.
  Validation experiment: OWED, chipped as follow-on. It must actually TRAIN with
  `rollout_consistency_loss` rather than merely enable the flag -- enabling it changes nothing on
  its own, BY DESIGN -- and should carry `decay=1.0` as the flat-form control.
  DOES NOT unblock INV-088 / MECH-135. Both keep `pending_retest_after_substrate: true`: as of
  V3-EXQ-965 the bars are still missed by 25-37x (`cr_ratio`) and 5-7 orders
  (`e1coe_score_var`), and nothing here has moved them yet. Do NOT queue a retest off this build.
