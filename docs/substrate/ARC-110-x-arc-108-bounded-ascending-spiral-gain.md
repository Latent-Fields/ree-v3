## ARC-110 x ARC-108: BOUNDED ascending-spiral gain -- target-PARITY controller (V3-EXQ-711 runaway repair) (2026-07-04)
- ARC-110 x ARC-108: selection.cross_loop_ascending_parity_controller -- IMPLEMENTED 2026-07-04
  (substrate; PROMOTES NOTHING -- MECH-439/ARC-108/ARC-110 stay candidate + pending_retest_after_substrate
  until the successor falsifier converts them). ree_core/predictors/e3_selector.py (_parity_forward_gain +
  the W_cross forward assembly + the diagnostic w_eff block + the M_cross post_action_update clamp) +
  ree_core/utils/config.py. Repairs the RUNAWAY the confirmed failure_autopsy_V3-EXQ-711_2026-07-04 found
  in the RAW-scalar ascending gain above: at raw scalar 20x-forward x 5x-plasticity the plastic ascending
  M_cross entries compounded through the positive-feedback plastic loop and RAN AWAY (M_cross range peak
  4897.8 vs the un-gained ~0.02-0.12; w_eff[limbic] peak 10-2274x w_eff[motor] across the 3 divergent
  seeds) -- a limbic-loop MONOPOLY that merely replaces the F/motor-pinning, not a fair parity win, and
  committed-class entropy FELL below baseline on 2/3 divergent seeds. The 709->711 pattern showed the raw
  scalar has NO stable parity regime (sub-threshold 709 never wins; runaway 711 monopolizes): the mechanism
  was MISSING A CONTROLLER. This replaces the unbounded multiply with actuator-saturated setpoint control:
    1. FORWARD parity-ceiling (_parity_forward_gain in the W_cross assembly): a per-step ascending gain in
       [0, parity_forward_gain] SOLVED so the limbic effective column weight w_eff[limbic] is LIFTED toward
       but HARD-CAPPED at parity_ceiling_ratio * w_eff[motor]. The motor column carries no strict-upper-tri
       entry -> w_eff[motor] is gain-invariant -> the fixed parity reference. Bounds the
       w_eff[limbic]/w_eff[motor] RATIO -> a FAIR within-eligible reorder, never a monopoly. Applied via the
       same _ascending_gain_matrix so the map stays LINEAR and bit-identical-at-init (M_cross==0 ->
       W_cross==I for any gain).
    2. MATURATION bounded loop (post_action_update): the ascending three-factor update is scaled by the
       BOUNDED parity_plasticity_gain, then the ascending (upper-tri) M_cross entries are clamped to
       [-m_cross_clamp, m_cross_clamp] -- an anti-windup clamp that stops the plastic positive-feedback loop
       from running away (the second 711 runaway source).
  Config: E3Config.use_ascending_parity_controller (default False, master switch) +
  loop_segregation_parity_forward_gain (default 1.0, lift strength) + loop_segregation_parity_ceiling_ratio
  (default 0.0 = disabled, the w_eff[limbic]<=ratio*w_eff[motor] cap) + loop_segregation_parity_plasticity_gain
  (default 1.0, bounded maturation rate) + loop_segregation_m_cross_clamp (default 0.0 = disabled, the
  ascending |M_cross| bound). Threaded through REEConfig.from_dims. Requires
  use_learned_cross_loop_arbitration (hence use_loop_segregation) on to act. TAKES PRECEDENCE over
  use_ascending_spiral_gain when both are on (the raw path is retained ONLY for 709/711 reproducibility).
  Master switch False -> BIT-IDENTICAL OFF; sub-params default inert (forward_gain 1.0 = no lift,
  ceiling_ratio 0.0 / m_cross_clamp 0.0 = disabled) so the successor's ON arm configures them explicitly.
  Data flow: M_cross -> (forward) g=_parity_forward_gain(M) -> G(g) .* M_cross -> W_cross @
  [motor_z;assoc_z;limbic_z] -> final -> commit; (learning) three-factor delta -> G_plast .* delta ->
  M_cross.add_ -> clamp ascending entries to [-clamp, clamp]. The w_eff / limbic_ge_motor diagnostics use
  the SAME parity-gained W_cross. New diagnostics: loop_ascending_parity_controller_active /
  _parity_forward_gain_applied / _parity_ceiling_ratio (and loop_ascending_spiral_gain_active now reports
  False whenever the controller is active, so a saturation guard can read which path drove selection).
  Safety: unchanged -- arbitration stays STRICTLY within the F+MECH-448/449 eligible set; the controller
  reorders within-eligible candidates and can NEVER re-admit a No-Go-suppressed one. F still fully owns the
  MOTOR loop. MECH-094: the maturation clamp rides the existing waking-only M_cross update (a simulation
  tick forms no delta_t, writes no M_cross); no new encoder / no autograd -> phased training NOT required
  by the substrate (the successor still phases P0/P1/P2 because the cross-loop M_cross LEARNS, same as 711).
  ML/AI note (Layer 7): actuator-saturated setpoint control -- output saturation = the parity ceiling,
  integrator clamp = the M_cross anti-windup clamp; the standard fix for an unbounded gain on a
  positive-feedback plastic loop that diverges. Biologically compatible: Haber's spiral is a graded, BOUNDED
  modulation held in parity by tonic-DA homeostasis + striatal lateral inhibition -- the raw scalar had the
  symbol without that bounding dependency.
  Contract tests: tests/contracts/test_ascending_parity_controller.py (9 contracts: byte-identical OFF
  across 12 seeds; at-init identity under large params; inert ON (fwd 1.0 + ceil 0.0); parity ceiling
  bounds w_eff[limbic]/w_eff[motor] where the raw scalar runs to 144x while motor stays invariant;
  maturation clamp bounds ascending M_cross where it otherwise runs to ~7.7e5; parity win not monopoly --
  limbic_ge_motor True while staying under the ceiling; from_dims plumbing; safety within the eligible set;
  precedence over the raw scalar). Both ascending suites green (18 contracts). Backward-compat: the raw
  test_ascending_spiral_gain.py 8 contracts still pass bit-for-bit (raw path untouched).
  Validation experiment: V3-EXQ-713 queued (Step 8) -- a NEW-EXQ successor falsifier (OFF vs bounded-ON on
  the GAP-A reef-bipartite substrate; a saturation-guarded limbic_loop_can_win gate requiring a parity BAND
  win + a w_eff/M_cross ceiling; C1 committed-class entropy strict-above the un-gained baseline on >=2/3
  divergent seeds; non-vacuity self-route substrate_not_ready_requeue, never a false weakens). This is a
  redesign of a DIFFERENT mechanism (the controller), NOT a raw-gain-magnitude re-letter -- the
  re-derive brake in the 711 autopsy explicitly REFUSES a same-claim raw-gain re-queue.
  Biology + ARC-106 divergence ledger: REE_assembly/docs/architecture/learned_cross_loop_arbitration.md
  (Addendum: ascending-spiral gain -> Sub-addendum: bounded parity controller). See ARC-110 / ARC-108
  (owning coupling), MECH-439 (the F-dominance conversion ceiling), use_ascending_spiral_gain (the raw path
  this bounds), use_learned_cross_loop_arbitration (the matrix this gains), ARC-106 (grounding framework).

- mech457_competence_bootstrap_explorer: action_learning.competence_bootstrap_explorer -- IMPLEMENTED 2026-07-16.
  Composition primitive (experiments/_lib, NOT ree_core; no SD-NNN number) closing the MECH-457
  floor->competent action-learning gap. Module: experiments/_lib/mech457_bootstrap_explorer.py
  (BootstrapExplorerConfig + linear_anneal + train_bootstrap_explorer + make_off_config/make_on_config).
  Plus two OPTIONAL no-op-default hooks on experiments/_lib/mech457_explorer_classes.train_a2c:
  coef_schedule / entropy_schedule (per-episode fn(ep, n_episodes); default None -> constant, byte-identical
  to the 752-756 callers; mutually exclusive with the utility-gate mode_gate 755 refuted).
  WHY: the GOV-FANOUT-1 discrimination is CLOSED (751-756) -- no single mechanism and no pairwise
  combination clears the 1.0 foraging floor toward the 48.05 local-view ceiling. The wall is ONE
  structural property with two joined halves (failure_autopsy_MECH-457-fanout-755_2026-07-15): (1)
  cold-start/success-dependence -- credit/return/curriculum/pair (752/753/754/756) amplify signal derived
  from prior success -> collapse sub-floor from ~0; only success-INDEPENDENT RND (751, 5.22) and BC (748,
  32.72) break the floor; (2) capacity-to-convert -- even a competent explore/exploit gate (755) adds
  nothing; RND reaches only ~11% of the ceiling. Every fanout converter ran on the SPARSE base (nothing to
  convert). The build composes ONLY landed pieces (honours the SYNTHESIS.md duplication objection): RND
  success-independent dense drive (mech457_explorer_classes.RNDModule = ARC-065/MECH-314) + first-class
  actor-critic (ree_core/action_learning/actor_critic.ActorCriticPolicy via RepAgent, z_world cotrain AND
  raw 5x5) + prioritized backward credit-replay (_prioritized_credit_replay, NOW fed by RND-generated
  successes) + THE NEW PRIMITIVE: a DEVELOPMENTAL intrinsic-coef/entropy anneal (linear_anneal;
  coef_start->coef_end over anneal_fraction; NOT the critic-utility ModeGate 755 refuted -- LC-NE
  explore/exploit consolidation as an ontogenetic schedule) + increased budget (n_episodes above the 1000
  that plateaus RND).
  Config: BootstrapExplorerConfig default = OFF = 751 RND plateau (use_rnd True, constant coef 1.0,
  anneal_fraction 0.0, credit_replay False, n_episodes 1000). make_on_config: coef 1.0->0.05, entropy
  0.10->0.03 over 60% of training, credit-replay on, 3x budget.
  Data flow: env obs -> RepAgent.encode (z_world cotrain OR raw 5x5) -> ActorCriticPolicy.select -> env.step
  -> shaped reward (harm + FORAGE_BONUS + count_novelty + coef_t*RND) -> GAE + A2C backward + credit-replay
  -> unshaped foraging_competence eval @D3.
  Backward compatible: new file imported by nobody yet + no-op-default train_a2c hooks -> bit-identical OFF;
  full pytest tests/contracts + tests/preflight 1428 passed (0 regressions) + 6 new bootstrap-explorer
  contracts (tests/contracts/test_mech457_bootstrap_explorer.py). MECH-094 N/A; phased training N/A.
  Validation experiment: V3-EXQ-765 queued (Step 8) -- diagnostic, OFF (RND plateau) vs ON (composed
  bootstrap) x {z_world, raw} for foraging_competence @D3 lift >= 7.83 above the 5.22 plateau (target
  ~13.05, toward BC 32.72); reuse-eligible per representation. Design doc:
  REE_assembly/docs/architecture/sd_mech457_competence_bootstrap_explorer.md. See MECH-457 (candidate/
  v3_pending), INV-088 (candidate/pending_substrate_reconfirmation), the 751-756 fanout + autopsies, and
  SYNTHESIS.md (targeted_review_action_learning_bootstrap_class_choice).
  CAPACITY-SIDE AMEND 2026-07-16 (routed from failure_autopsy_V3-EXQ-765): the 765 post-build retest RAN and
  FAILED (bootstrap_explorer_plateaus_capacity_gap_remains) -- the DRIVE half works on raw (ON 6.48 vs OFF
  0.62, +5.87) but the actor-critic plateaus at ~13% of the 48.05 ceiling, clears neither rep's 13.05 lift
  target, seed variance is large (15.9/3.05/0.5), and z_world cotrain is DESTRUCTIVE (ON 0.35 < OFF 5.22).
  The autopsy routed ONE capacity-side build (three JOINT knobs, NOT a discrimination fanout), now applied:
  (a) CAPACITY -- ON actor_critic_hidden 128->256 (BootstrapExplorerConfig.actor_critic_hidden, threaded
  through make_rep -> RawViewRep / ZWorldRep -> fan.make_rawview_ac / make_zworld_agent -> x742.
  _make_actor_critic_agent(hidden=)) + ON_BUDGET_MULTIPLIER 3->5; (b) RELIABILITY -- ON warm_start_fraction
  0.2 (warm_then_anneal holds full-explore coef/entropy before the anneal) + credit_replay_passes 3->6 /
  credit_topk 32->64 (threaded through train_a2c -> _prioritized_credit_replay); (c) INTEGRATION -- ON
  cotrain_encoder False = z_world DETACHED (ZWorldRep trains the policy on the FROZEN prediction-trained
  encoder, excluding encoder params from the optimizer; agent.actor_critic_step detaches z; Stooke 2021).
  ALL five new config fields are OFF-preserving no-op defaults (hidden 128, cotrain True, warm 0.0, passes 3,
  topk 32) -> OFF arms bit-identical to 765's (drift-guard reproduces ~5.22 z_world / ~0.6 raw). 5 new
  contracts (C7-C11: warm_then_anneal, OFF capacity-neutral, ON carries all three knobs, make_rep capacity/
  detach wiring, capacity-ON end-to-end); 11 bootstrap-explorer contracts pass. Validation experiment:
  V3-EXQ-769 (supersedes 765) queued -- diagnostic, same OFF-vs-ON x {z_world,raw} lift criterion, cloud
  (ree-cloud-2). Re-derive brake HELD-BUT-SANCTIONED (named-capacity post-build retest under a NEW EXQ
  number is the one exception the 765 autopsy names; not a single-axis probe). MECH-457 stays candidate/
  v3_pending. Driver: experiments/v3_exq_769_mech457_bootstrap_explorer_capacity.py.

- SD-MECH457-DISTRIBUTIONAL-CRITIC: action_learning.distributional_value -- IMPLEMENTED 2026-07-18.
  New module ree_core/action_learning/distributional_value.py (symlog/symexp + ValueBins: bin support,
  two-hot / HL-Gauss target projection, expectation decode, cross-entropy), plus a third critic form on
  ActorCriticPolicy alongside the plain and successor-feature heads.
  Config: REEConfig.actor_critic_use_distributional_critic (default False -- the scalar nn.Linear(hidden,1)
  head, byte-identical; set True to enable), with actor_critic_n_value_bins 41 / actor_critic_value_bin_limit
  10.0 (symlog space) / actor_critic_value_bin_sigma 0.75 (HL-Gauss sigma in bin widths; 0.0 -> pure two-hot).
  Also BootstrapExplorerConfig.use_distributional_critic (default False), declared in as_slice() so the flag
  lands in the arm fingerprint config_slice, applied at REP CONSTRUCTION via make_rep like actor_critic_hidden /
  cotrain_encoder.
  Data flow: GAE return -> ValueBins.project (symlog, clamp, HL-Gauss) -> cross-entropy against
  value_head(h)[B,n_bins] -> ValueBins.decode (softmax, E[support], symexp) -> ActorCriticStep.value (SCALAR)
  -> GAE / bootstrap / credit-replay TD priority / eval, all unchanged. ActorCriticStep gains value_logits
  (None on the scalar and SF paths). The four _lib scalar-MSE sites now go through one dispatch,
  fan.critic_value_loss(policy, value_logits_t, value_t, ret_t), which falls back to the identical
  AC_VALUE_COEF * 0.5 * (V-G)^2 when the critic is scalar: mech457_fanout.py (raw-view A2C, z_world shaped
  A2C) + mech457_explorer_classes.py (train_a2c, _prioritized_credit_replay). The H-curriculum
  goal-conditioned trainer keeps its scalar critic by design (different mechanism leg).
  Backward compatible: disabled by default; verified by a full --dry-run of
  experiments/v3_exq_780_mech457_bc_prior_discrimination.py (six arms, both reps) and by contracts C1/C6.
  Biological basis: Dabney et al. 2020 -- dopaminergic populations carry heterogeneous reversal points that
  jointly encode a DISTRIBUTION over reward, not a single mean; the scalar head was the simplification.
  ML statement (engineering counsel only): symlog two-hot bins (Hafner 2023, DreamerV3); HL-Gauss target
  (Farebrother 2024, "Stop Regressing").
  Phased training required: no (the critic head trains inside the existing RL phase on the same returns;
  no encoder head on a moving latent target). MECH-094: not applicable (no memory writes).
  ANTI-ALIAS (load-bearing): VALUE ESTIMATOR ONLY. policy_loss, log-prob, entropy bonus, advantage weighting,
  BC auxiliary and the credit-replay policy term are byte-identical on both branches -- the update-constraint
  locus belongs to the sibling substrate_queue entry mech457_policy_kl_anchor (H-retention-consolidation), and
  a leg changing both would make neither readable. Contracts C2/C2b assert the trunk + policy head are
  bit-identical at the same seed and that the CE loss puts no gradient on the policy head.
  DO NOT substitute use_sf_critic for this: it is hard-wired False on every 457 path, has no
  BootstrapExplorerConfig field, and its psi-Bellman + reward-regression losses exist only in
  v3_exq_742:357-372 -- flipping the flag leaves reward_w zero-init and V_SF identically 0, i.e. an UNTRAINED
  critic read as an alternative one. critic_loss() raises on the scalar path so a mis-wired ON arm fails loudly.
  Motivation (measured): V3-EXQ-782 R-(b) -- shared CTRL critic flat and uninformed, std(V)/std(G)=0.041 vs a
  0.25 collapse threshold, pre-reward-vs-far separation 0.016 vs a 0.25 floor; V3-EXQ-780 raw_view reached
  20.933 and RL eroded it to 11.667.
  Validation experiment: NOT QUEUED -- per GOV-FANOUT-1 nothing is queued until at least two of the four
  MECH-457 retention legs are buildable. Unblocks hypothesis H-retention-critic (competence_floor question,
  evidence/planning/hypothesis_space_registry.v1.json). MECH-457 stays candidate/v3_pending; this build
  promotes and demotes nothing.
  12 new contracts: tests/contracts/test_mech457_distributional_critic.py.
  See REE_assembly/docs/architecture/sd_mech457_distributional_critic.md and
  REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md.
- mech457_bc_aux_schedule: action_learning.bc_auxiliary_persistence_schedule -- IMPLEMENTED 2026-07-18.
  experiments/_lib/mech457_explorer_classes.py (train_a2c) + experiments/_lib/mech457_bootstrap_explorer.py
  (BootstrapExplorerConfig, train_bootstrap_explorer). Makes the BC-auxiliary PERSISTENCE sweepable so
  H-retention-auxiliary-decay can read a competence half-life.
  Config: train_a2c(bc_aux_schedule=None) -- Optional[Callable[[int,int],float]], default None -> constant
  bc_aux_coef; BootstrapExplorerConfig.bc_aux_coef_end (default None) + .bc_aux_anneal_fraction (default 0.0),
  both declared in as_slice(). Three sweep cells: constant (end=None), annealed (end<start over the first
  bc_aux_anneal_fraction of episodes), off (bc_aux_coef=0.0).
  Data flow: cfg.bc_aux_coef/_end/_anneal_fraction -> linear_anneal closure -> train_a2c bc_aux_schedule ->
  per-episode bc_coef_eff (resolved beside beta_eff/coef_eff) -> BC auxiliary guard AND weight -> episode loss.
  Backward compatible: OFF defaults byte-identical; full suite 1649 passed.
  THE GUARD IS PART OF THE FEATURE, not incidental: bc_coef_eff drives the auxiliary's `if`, not only its
  weight. An annealed cell passes bc_aux_coef=0.0 with a nonzero schedule, so the pre-existing
  `bc_aux_coef > 0.0` guard would have silently produced an OFF arm labelled ANNEALED -- a degenerate arm read
  as a scientific verdict. Contract C14 is the regression: a schedule returning c is asserted BIT-IDENTICAL to
  the float c.
  ANTI-ALIAS (load-bearing): this leg owns the bc_aux_coef axis. The schedule uses linear_anneal, NOT
  warm_then_anneal -- the latter is parameterised by the SHARED warm_start_fraction and would couple BC
  persistence to the exploration anneal, confounding the leg's single intervention. For the same reason
  bc_aux_schedule is deliberately NOT under the mode_gate mutual exclusion, which arbitrates two competing
  answers to EXPLORATION scheduling only. The sibling mech457_policy_kl_anchor (H-retention-consolidation)
  MUST NOT be operationalised through the BC auxiliary: anchoring via bc_aux_coef anchors to the
  DEMONSTRATOR rather than the installed policy snapshot, aliasing the two legs directly.
  Precondition: the bc_demo requirement reads max(bc_aux_coef, bc_aux_coef_end) -- a ramp-UP cell slips past a
  start-only check -- and runs BEFORE module construction, so a misconfigured arm fails on its config rather
  than partway through allocation (contract C16).
  Trajectory reporting: the guard dict carries bc_aux_coef_first/_last so a manifest can VERIFY the schedule
  moved; an annealed arm whose schedule silently stayed flat is otherwise indistinguishable from a constant one.
  Phased training required: no (reweights an existing CE term against a fixed demonstrator; no encoder head on
  a moving latent target). MECH-094: not applicable (no simulation/replay memory writes).
  Fingerprint: edits experiments/_lib/**, bound into substrate_hash, so pre-change baseline arm fingerprints are
  correctly refused for reuse -- expected, not a regression. as_slice() gains two declared keys for the same
  reason: a varyable knob absent from the config_slice would let two materially different arms share a
  fingerprint.
  Validation experiment: NOT QUEUED -- per GOV-FANOUT-1 nothing is queued until at least two of the four
  MECH-457 retention legs are buildable. That threshold is NOW MET (this build + mech457_distributional_critic,
  same day), but queueing remains a separate decision. Unblocks hypothesis H-retention-auxiliary-decay
  (competence_floor question, evidence/planning/hypothesis_space_registry.v1.json). MECH-457 stays
  candidate/v3_pending; this build promotes and demotes nothing.
  6 new contracts C12-C17: tests/contracts/test_mech457_bootstrap_explorer.py (17 pass).
  See REE_assembly/docs/architecture/sd_mech457_bc_aux_schedule.md and
  REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md.
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
- mech457_consummatory_act: environment.consummatory_act -- IMPLEMENTED 2026-07-25.
  ree_core/environment/causal_grid_world.py. Adds a distinct no-move CONSUME action (index 5,
  class attr CONSUME_ACTION) so that entering a resource cell AFFORDS rather than EFFECTS
  consumption: an approach drive can extinguish on contact and hand off to a separate
  consummatory act, the dissociation V3-EXQ-781's non-extinguishing terminal drive could not
  express (leg 4 of the MECH-457 retention portfolio).
  Config: CausalGridWorldV2(consummatory_act_enabled=False) -- a direct env constructor kwarg
  (NOT a REEConfig field; the env is built by the experiment driver). Set True to enable.
  Data flow: consummatory_act_enabled -> action_dim returns len(ACTIONS)+1 (== 6) so every actor
  head sizing from env.action_dim grows 5 -> 6 with no further wiring -> step() dispatches the
  CONSUME action explicitly (NOT via _action_map, so the world-rule-shift permutation can never
  turn it into a movement) -> on move-onto-resource the resource branch sets
  transition_type="resource_contact", zero benefit reward, no drive restore, resource RETAINED
  (on_consumable_resource info flag set) -> the CONSUME action while standing on a resource cell
  effects consumption via the shared helper _consume_resource_at(cx, cy).
  Shared code path: the 118-line benefit/removal/per-axis-drive-restore/respawn/field-recompute
  block was factored out of the legacy inline resource branch into _consume_resource_at, called
  by BOTH the legacy auto-consume-on-entry path (OFF) and the CONSUME action (ON) -- consumption
  is the SAME operation whichever way it is reached; only the TIMING differs. Reward binds to the
  ACT: contact yields 0 reward, CONSUME delivers it.
  Backward compatible: consummatory_act_enabled defaults False -> action_dim == 5,
  auto-consume-on-entry, observation_dim unchanged; byte-identical to the pre-change env. No
  running experiment is affected.
  BLAST RADIUS: enabling the flag grows action_dim 5 -> 6, re-keying every actor head and
  BUSTING all cached arm fingerprints for consummatory-ON lineages (reuse correctly refuses
  across the change). Pre-change lineages keep the 5-action space and valid fingerprints because
  the flag defaults OFF.
  Minor limitation: the body_state last-action one-hot (body[5..8]) already aliases stay(4)->slot0;
  CONSUME(5) aliases the same way. A dedicated slot would change body_obs_dim and break every
  existing observation_dim, so it is left aliased -- the agent selects CONSUME via a distinct
  policy-head logit (action_dim grew), which is what the leg needs.
  Phased training required: no (no head trained). MECH-094: not applicable (no simulation/replay).
  7 new contracts C1-C7: tests/contracts/test_mech457_consummatory_act.py (action_dim 5/6; OFF
  auto-consume; ON contact-affords; ON CONSUME effects; CONSUME-off-resource == stay; un-consumed
  departure restores the grid marker; consumption path-independent OFF-entry vs ON-CONSUME).
  Consumption refactor regression-covered by the SD-049/MECH-307/SD-057/SD-037 consumption
  contracts (70 pass on the hub, ree-v3 base 120efac).
  Unblocks H-consummation-binding (the LAST open competence_floor retention leg) as a
  /queue-experiment target; the behavioural experiment is NOT queued in this build pass.
  CORRECTION (2026-07-25): the design doc's claim that "the drive half was already built" via
  goal.py refers to the HOMEOSTATIC per-axis drive, which is NOT 781's approach primitive and is
  NOT live in the mech457 bootstrap-explorer path; the drive-side extinction wiring the leg-4
  treatment arm actually needs is built separately below (mech457_approach_extinction).
  MECH-457 stays candidate/v3_pending; INV-088 unchanged; this build promotes and demotes nothing.
  See REE_assembly/docs/architecture/sd_mech457_consummatory_act.md and
  REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md +
  REE_assembly/evidence/planning/competence_floor_reposing_2026-07-25.md.
- mech457_approach_extinction: experiments/_lib approach-drive extinction-on-contact -- IMPLEMENTED
  2026-07-25. experiments/_lib/mech457_explorer_classes.py (train_a2c) +
  experiments/_lib/mech457_bootstrap_explorer.py (BootstrapExplorerConfig, train_bootstrap_explorer).
  Completes the DRIVE half of competence_floor retention leg 4 (H-consummation-binding). The
  mech457_consummatory_act ENV node (2026-07-25) made contact AFFORD rather than EFFECT
  consumption and emits info["on_consumable_resource"], but NOTHING consumed that signal:
  V3-EXQ-781's appetitive approach primitive (mech.resource_proximity) reads obs_dict, which does
  not carry the flag, and train_a2c never threaded info to the approach hook -- so the drive could
  not extinguish on contact. goal.py's homeostatic drive (drive_ema_alpha/drive_floor/
  per_axis_restoration_fraction) is a DIFFERENT system, does not reference on_consumable_resource,
  and is not imported anywhere in this path (GOAL_DIM=2 here is spatial nav to a target cell).
  Config: BootstrapExplorerConfig.approach_extinguishes_on_contact (bool, default False;
  declared in as_slice()). The env's consummatory_act_enabled is a DIRECT ENV CONSTRUCTOR kwarg
  set by the driver in env_kwargs (NOT a config field -- this config does not build the env).
  Data flow: cfg.approach_extinguishes_on_contact -> train_bootstrap_explorer -> train_a2c: on
  each tick where info["on_consumable_resource"] is True the approach reward is zeroed, so the
  appetitive drive terminates on arrival and hands off to the distinct CONSUME act (the treatment
  arm); 781's non-extinguishing terminal drive is the control.
  HALF-WIRED IS AN ERROR (train_a2c raises): approach_extinguishes_on_contact=True requires (1) an
  approach_drive (use_approach_primitive=True) -- extinction with no drive is the control wearing
  the treatment label; (2) the env built with consummatory_act_enabled=True -- else
  on_consumable_resource is always False and extinction silently never fires.
  Backward compatible: default False -> the extinction branch never fires, byte-identical to the
  pre-change non-extinguishing drive (contract E1 asserts weight-identity; 788 dry-run clean).
  Fingerprint: edits experiments/_lib/**, so pre-change baseline arm fingerprints are correctly
  refused for reuse across the change (expected, not a regression -- same as the consummatory_act
  and retention_probe nodes); as_slice() gains one declared key because an extinguishing and a
  non-extinguishing cell are not interchangeable ARTIFACTS.
  Phased training required: no (no head trained). MECH-094: not applicable (no simulation/replay).
  7 new contracts E1-E6: tests/contracts/test_mech457_approach_extinction.py (default no-op
  byte-identical; extinction fires and changes policy on a consummatory env with proven contact;
  both half-wired guards raise; as_slice declares + defaults False; config-level half-wired via
  train_bootstrap_explorer; ASCII-only). All 7 pass locally (3.72s).
  Companion build (same day): consummatory-aware reference/demonstrator policies (below) -- the
  retention (BC-install) framing of leg 4 needs them so the install can take in the consummatory
  env.
  Validation experiment: V3-EXQ-821 (H-consummation-binding leg 4, retention/BC-install framing)
  queued via /queue-experiment -- the behavioural leg IS the validation.
  MECH-457 stays candidate/v3_pending; INV-088 unchanged; this build promotes and demotes nothing.
  See REE_assembly/docs/architecture/sd_mech457_approach_extinction.md,
  REE_assembly/docs/architecture/sd_mech457_consummatory_act.md, and
  REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md (leg 4).
- consummatory_aware_reference_policies: experiments/_lib/capability_eval.py -- IMPLEMENTED
  2026-07-25. OraclePolicy + LocalViewGreedyPolicy select the distinct CONSUME action while
  standing on a resource cell in consummatory mode, via the shared helper
  _consummatory_consume_action(env) (gated on env.consummatory_act_enabled; reads the env's
  _on_consumable_resource "should consume now" flag, falling back to _agent_on_resource_cell()).
  WHY: with mech457_consummatory_act, arrival on a resource cell only AFFORDS consumption -- the
  benefit needs a CONSUME action. The hand-coded greedy policies otherwise return "stay" on the
  target cell and forage 0 in the consummatory env, which would leave the readiness anchors
  reading the env as unsolvable and a BC install cloned from the LocalViewGreedyPolicy
  demonstrator unable to take. This is what makes the RETENTION (BC-install) framing of leg 4
  runnable in the consummatory env (the demonstrator-free 781 framing does not need it).
  Measured: in the D3 consummatory env local_view_greedy forages ~13.5/ep, greedy_oracle ~13.0
  (vs ~17.9 / ~18.7 non-consummatory -- CONSUME costs one step per resource), random_walk ~0.33.
  So the consummatory achievable ceiling / install band is ~13, NOT the non-consummatory 20.933;
  the leg records the LIVE consummatory anchors as denominators. RandomPolicy is deliberately NOT
  made consummatory-aware (it is the floor).
  Backward compatible: default OFF (helper returns None when consummatory_act_enabled is False)
  -> byte-identical; the pre-change 5-action env has no CONSUME action. capability_eval.py is in
  experiments/_lib/**, so this busts arm fingerprints fleet-wide for FUTURE runs (expected;
  runs are byte-identical, only substrate_hash changes).
  4 contracts CG1-CG4: tests/contracts/test_consummatory_aware_policies.py (non-consummatory
  no-op + unchanged foraging; consummatory greedy policies forage >= floor; helper semantics;
  ASCII). All 4 pass locally (1.41s).
  Phased training required: no. MECH-094: not applicable.
  See REE_assembly/docs/architecture/sd_mech457_approach_extinction.md (companion build section).
