## mech457_competence_bootstrap_explorer: action_learning.competence_bootstrap_explorer -- IMPLEMENTED (2026-07-16)
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
