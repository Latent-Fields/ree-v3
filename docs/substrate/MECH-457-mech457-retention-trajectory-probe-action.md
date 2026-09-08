## mech457_retention_trajectory_probe: action_learning.mid_training_competence_probe -- IMPLEMENTED (2026-07-19)
- mech457_retention_trajectory_probe: action_learning.mid_training_competence_probe -- IMPLEMENTED
  2026-07-19. experiments/_lib/mech457_explorer_classes.py (train_a2c) +
  experiments/_lib/mech457_bootstrap_explorer.py (BootstrapExplorerConfig, train_bootstrap_explorer).
  The MEASUREMENT prerequisite shared by ALL FOUR competence_floor retention legs: the portfolio
  (sec 53) requires every leg to record the post-installation competence TRAJECTORY, and before
  this build train_a2c had NO observation hook -- all 16 optional params are reward/loss-shaping
  hooks, and the guard returned rolling-window means only. All 18 mech457 scripts measured
  terminal-only, which is what kept the retention deficit invisible for ten legs (V3-EXQ-780:
  raw_view 20.933 post-BC eroded to 11.667, scored a null).
  Config: train_a2c(probe_every=None, probe_fn=None) -- Optional[int] cadence +
  Optional[Callable[[int],Dict]]; BootstrapExplorerConfig.retention_probe_every (default None),
  declared in as_slice(); train_bootstrap_explorer(probe_fn=None) passthrough.
  Data flow: driver probe closure (fresh env + rep.eval_policy + capability_eval.evaluate_seed) ->
  train_bootstrap_explorer -> train_a2c -> fires at the episode boundary AFTER the optimiser step,
  credit-replay sweep, RND update and deque appends -> guard["competence_trajectory"] =
  [{"episode": int, "foraging_competence": float}, ...].
  Backward compatible: both hook params default None -> byte-identical OFF. competence_trajectory
  is emitted unconditionally (empty when unprobed), matching the bc_aux_coef_first/_last precedent.
  WHY A HOOK AND NOT DRIVER-SIDE CHUNKING (the load-bearing design finding): segmenting the budget
  and evaluating between train_a2c calls is unfaithful twice over -- (a) coef/entropy/bc_aux
  schedules take the LOCAL loop index and warm_then_anneal/linear_anneal derive their cutoffs from
  the local n_episodes, so N segments restart the anneal N times, and for
  H-retention-auxiliary-decay the bc_aux anneal IS the independent variable; (b) the Adam
  optimiser, _RunningStd reward normaliser, novelty_counter and deques are constructed inside the
  call, so each segment resets the very learning dynamics whose erosion is the DV.
  RNG ISOLATION IS LOAD-BEARING, NOT DEFENSIVE: train_a2c snapshots/restores torch + global numpy +
  python random around each probe call. Initially assessed as belt-and-braces (the env owns a
  per-instance default_rng; ActorCriticEvalPolicy is deterministic argmax) -- that assessment was
  WRONG and was corrected by mutation testing: the training rollout itself draws from the global
  torch stream, so an unrestored probe desynchronises training.
  HALF-WIRED IS AN ERROR: probe_every without probe_fn (or vice versa) RAISES, as does a
  non-positive cadence -- a half-wired probe returns an empty trajectory indistinguishable from a
  genuinely flat one (the degenerate-arm-as-verdict failure the distributional critic guards).
  ANTI-ALIAS: instrumentation only -- no update rule, loss term, schedule or value estimator
  changes, so it cannot contaminate the three-way retention anti-alias (value estimator =
  mech457_distributional_critic / update constraint = mech457_policy_kl_anchor / auxiliary
  persistence = mech457_bc_aux_schedule). Contract T2 enforces this mechanically.
  SAME-STATISTIC REQUIREMENT for consumers: unshaped foraging_competence via evaluate_seed (the
  statistic post_bc_foraging_competence uses, and what the bands 1.0/5.22/13.05/20.933/32.72/48.05
  are denominated in) -- NOT the shaped mean_train_forage_recent, which would put the half-life on
  a different statistic from the criterion (the V3-EXQ-643 mismatch class).
  Phased training required: no (no head trained). MECH-094: not applicable.
  Fingerprint: edits experiments/_lib/**, bound into substrate_hash, so pre-change baseline arm
  fingerprints are correctly refused for reuse -- expected, not a regression. as_slice() gains one
  declared key because a probed and an unprobed cell are not interchangeable ARTIFACTS even though
  the probe cannot change the learned result.
  11 new contracts T1-T7: tests/contracts/test_mech457_retention_trajectory_probe.py (own file --
  test_mech457_bootstrap_explorer.py's C18-C18f belong to the concurrent untrained-encoder guard;
  matches the test_mech457_distributional_critic.py precedent). T2 (measurement neutrality) is
  load-bearing and MUTATION-CHECKED BOTH WAYS: the first draft compared the guard's aggregate means
  and passed even with the RNG restore removed -- a vacuous pass, since 8 episodes of rolling-window
  means can coincide across divergent runs. It now asserts on trained policy WEIGHTS, which do fail
  under that mutation, plus a non-degeneracy assertion.
  Unblocks H-retention-critic + H-retention-auxiliary-decay for queueing as a pair (both
  manipulations already built: 8e88ffc, 9a8dbae). MECH-457 stays candidate/v3_pending; INV-088
  unchanged; this build promotes and demotes nothing.
  Validation experiment: V3-EXQ-788 (v3_exq_788_mech457_retention_critic.py, H-retention-critic)
  + V3-EXQ-789 (v3_exq_789_mech457_retention_auxiliary_decay.py, H-retention-auxiliary-decay),
  queued as a PAIR 2026-07-19 (ree-v3 5a7bcf9). Both wire retention_probe_every/probe_fn and read
  guard["competence_trajectory"] -- they are the consuming validation of this probe. Audited
  against the four measurement_requirement constraints: trajectory probe wired, explicit
  succeeded_then_decayed branches, substrate_not_ready_requeue self-route,
  post_bc_foraging_competence genuinely CONSUMED (not merely declared -- the V3-EXQ-780 failure),
  same-statistic evaluate_seed/foraging_competence. Anti-alias holds both ways: 789 pins
  use_distributional_critic False on all arms, 788 does not touch the bc_aux schedule.
  See REE_assembly/docs/architecture/sd_mech457_retention_trajectory_probe.md and
  REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md.
