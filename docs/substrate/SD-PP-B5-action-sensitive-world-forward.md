## SD-PP-B5: Action-Sensitive world_forward + Action-Sensitivity Readiness Gate (2026-09-22)
- SD-PP-B5: predictors.e2_fast.world_forward_action_sensitivity -- IMPLEMENTED
  2026-09-22 (substrate; validation experiment pending). Two pieces, both default
  OFF / no-op.

  **Why.** V3-EXQ-1073 measured on seeds 42/123/456 that the converged
  `e2.world_forward` head (P0 3600 recon-only steps, conv_rel_drop 0.997-0.999)
  DOES NOT READ ITS ACTION: skill vs the copy-the-input predictor
  -0.071/+0.227/-0.007 (MECH-573), and its pre-sleep MSE on an INVERTED-action-map
  battery is 0.760/0.901/0.881x its MSE on the original rule -- the inverted rule
  is EASIER. An action-map inversion is therefore not a contradiction, and no
  anti-self-sealing or reopening test can be posed on this head at all (the run
  self-routed `confidently_wrong_condition_unposeable_on_this_head`). The cause is
  structural: `world_forward` is a residual head (`z + delta`) whose action enters
  through one linear layer, and on CausalGridWorldV2 a one-cell move compresses to
  `identity_predictor_mse ~1e-5`, so driving `delta -> 0` scores well and the
  action encoder receives almost no gradient. Independently reproduced during this
  build: a fresh 120-row rollout gave identity_mse 1.17e-05.

  **(1) Instrument.** `experiments/_lib/action_sensitivity_gate.py` --
  `identity_predictor_mse()`, `skill_vs_identity()`, `action_shuffle_ratio()`,
  `battery_pair_ratio()`, `readiness_verdict()`, `check_canary()`,
  `format_verdict()`. Lets any consolidation experiment assert FROM MEASURED
  OUTPUT that its head reads its action before posing a contradiction. This was
  pre-specified in SD-031's Validation P1 ("Identity-collapse check: r2 on
  action-shuffled control must drop substantially", 2026-04-18) and never built;
  `_identity_predictor_mse` existed only as copy-pasted private functions in
  v3_exq_1063 and v3_exq_1073, which this promotes to shared code.

  NEGATIVE INSTRUMENT (CLAUDE.md): it authorises STARTING work on a negative, so
  it carries all three remedies -- a structural three-valued `status`
  (`ready` / `action_blind` / `cannot_determine`, never a bool, propagating into
  `to_dict()`); a `CANARY_V3_EXQ_1073` known-baseline replay; and printed
  denominators (`n_rows`, `n_distinct_actions`). The monostrategy case is why the
  third value is load-bearing: a battery with ONE distinct action makes an
  action-map change a no-op and returns ratio exactly 1.0, which a bool would read
  as `action_blind` when the truth is that the battery cannot test the question.

  **(2) Remedy.** `E2FastPredictor.compute_world_interventional_loss()` -- SD-013's
  contrastive margin loss `max(0, margin - ||wf(z,a) - wf(z,a_cf)||_2)`, ported to
  the head V3-EXQ-1073 actually trains. `E2WorldForward` (e2_world.py) already has
  this loss but hard-asserts `world_dim >= 128` and 1073 ran WORLD_DIM=16, so it
  was not constructible at that operating point.
  Config: `E2Config.use_world_interventional` (default False),
  `world_interventional_fraction` (0.3), `world_interventional_margin` (0.1),
  threaded through all THREE `from_dims` sites (MECH-307 shape).
  Data flow: z_world (detached) -> world_forward(a_actual) vs world_forward(a_cf)
  -> margin loss -> P0/P1 optimiser -> world_action_encoder gradient.

  NOT the SD-056 InfoNCE form (`world_forward_contrastive_loss`, same file): that
  is a CONFIRMED P0 destabiliser (V3-EXQ-701b ablation, carried by 798a) because it
  competes with reconstruction throughout training. The margin form goes silent
  once predictions are >= margin apart.

  **Known property, documented not fixed:** an EXACTLY action-invariant head is a
  stationary point of this loss -- zero distance is a minimum of ||d||, so every
  smooth function of ||d|| is flat there and no epsilon or squared-margin variant
  escapes it. Measure-zero under random init; both reference implementations
  (e2_harm_s.py, e2_world.py) share it. Pinned by
  `test_margin_loss_gradient_vanishes_only_at_exact_collapse`.

  Liveness (seed 42, real 120-row rollout, 5 distinct actions): the term raised the
  gradient reaching `world_action_encoder` from 0.0021 to 0.1445 (69x) and the loss
  from 0.00733 to 0.3644. Not inert.

  Backward compatible: disabled by default; `world_forward`'s forward pass is
  bit-identical with the flag at default (pinned by
  `test_world_forward_unchanged_when_feature_off`).
  Phased training required: yes -- P1 trains on `.detach()`ed z_world, same
  stop-gradient discipline as SD-013 / SD-031.
  MECH-094: not applicable (waking forward model, not replay content).
  Biological basis: Scholkopf et al. 2021 -- causal identifiability requires
  interventional, not merely observational, data.
  Contracts: `tests/contracts/test_action_sensitivity_gate.py` (25 tests,
  including three blind-spot measurements that construct the defect and assert the
  guard fails on it).
  Design doc:
  REE_assembly/docs/architecture/sd_pp_b5_action_sensitive_world_forward.md
  Validation experiment: V3-EXQ-1075 (queued 2026-09-23, live in the coordinator
  DB; ree-v3 9e17cb644f). It gates on the ACTION-SENSITIVITY axis only -- the
  inverted-rule battery ratio, where its OFF arm reproduces V3-EXQ-1073
  (0.833/0.879 vs 0.760/0.901). The READABILITY axis (skill vs the trivial
  predictor, MECH-573) is RECORDED NOT GATED (user decision rec-20260923-cb59ede6):
  it is encoder-bound and the driver's OFF arm does NOT reproduce 1073 on it
  (-5.79/-2.24 vs -0.071/+0.227, cause unidentified). So B5's ENCODER half stays
  OPEN: if V3-EXQ-1075 returns FAIL-a the route is SD-018-amend / SD-009 / SD-106
  as a new substrate entry, per the user's 2026-09-22 scope decision.
  See MECH-572, MECH-573, MECH-574, INV-063, SD-013, SD-031, SD-056.
