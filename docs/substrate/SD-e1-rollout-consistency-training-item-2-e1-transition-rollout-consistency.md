## SD-e1-rollout-consistency-training ITEM 2: e1.transition.rollout_consistency -- IMPLEMENTED (2026-09-01)
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
