## SD-MECH457-POLICY-KL-ANCHOR: mech457 policy trust-region anchor -- IMPLEMENTED (2026-07-19)
- SD-MECH457-POLICY-KL-ANCHOR: mech457 policy trust-region anchor -- IMPLEMENTED 2026-07-19.
  New `PolicyKLAnchor` in experiments/_lib/mech457_explorer_classes.py: a KL penalty pinning the
  policy to a FROZEN DEEP-COPY SNAPSHOT of itself taken at entry to `train_a2c`. Because callers
  install the BC prior BEFORE that call (baselines/mech457_retention.install_bc_prior ->
  train_off_arm), "at entry" IS the post-install checkpoint -- which is why the snapshot needs no
  explicit checkpoint argument.
  Config: BootstrapExplorerConfig.use_policy_kl_anchor (default False) + .kl_anchor_coef (default
  0.0), BOTH declared in as_slice() so the knobs land in the arm fingerprint config_slice; also
  threaded through baselines/mech457_retention.reference_config(). Applied inside
  train_bootstrap_explorer (an UPDATE-rule knob), NOT at rep construction like the critic swap.
  Data flow: rep.z_detached(state) -- on BOTH reps precisely the tensor ActorCriticPolicy.select()
  consumes -> frozen snapshot forward under no_grad -> ref logits -> coef * KL(pi || pi_ref),
  gradient through the LIVE logits only -> added to the episode loss AND to the credit-replay
  loss. Direction per the substrate_queue implementation_hint.
  Backward compatible: disabled by default; contract K1 asserts default-OFF trains BIT-IDENTICAL
  weights, verified end-to-end on the raw_view retention path (BC-install -> RL refine).
  ANTI-ALIAS (both load-bearing, both structural rather than conventional):
    (1) vs mech457_bc_aux_schedule / H-retention-auxiliary-decay -- anchors to the INSTALLED
        POLICY, never the demonstrator. The class never sees bc_demo and works with bc_demo=None
        (contract K4). Anchoring via bc_aux_coef would anchor to the demonstrator and alias.
    (2) vs mech457_distributional_critic / H-retention-critic -- the KL term is a function of
        `logits` alone and puts EXACTLY ZERO gradient on value_head/value_bins (contract K3); the
        fan.critic_value_loss dispatch is byte-identical on both branches. HONEST LIMIT: the
        trunk is shared, so the critic's INPUT FEATURES do move. That is the exact mirror of the
        sibling build's own situation (its CE loss moves the trunk too), whose contract C2
        asserted only that the CE loss puts no gradient on the policy HEAD.
  CREDIT-REPLAY IS ANCHORED TOO, and this is scientific rather than stylistic: the reference
  retention build runs credit_replay=True, so `_prioritized_credit_replay` applies a SECOND
  policy-gradient update per episode. Anchoring only the main loss would leave the constraint
  leaky and make a null from this leg unreadable -- "anchoring does not preserve competence" and
  "the unanchored replay update drifted the policy anyway" would be indistinguishable. The
  penalty sits INSIDE the CREDIT_LR_SCALE parenthesis so the constraint scales with the update it
  constrains (contract K9).
  THE PENALTY IS STATIONARY AT ZERO DRIFT, not merely small: KL(pi || pi) = 0 is the minimum, so
  its gradient vanishes there and an anchored arm is unpenalised until it starts to move
  (contract K6). Discovered by contract K3 failing its own non-degeneracy check on first run --
  a gradient probe taken AT the snapshot point passes vacuously, so K3 perturbs the policy first.
  Guard dict gains policy_kl_anchor_installed / policy_kl_anchor_coef /
  mean_policy_kl_to_anchor_recent, emitted unconditionally. The MEASURED KL is what lets a
  manifest verify the anchor actually BOUND rather than assuming it (same reasoning as
  bc_aux_coef_first/_last): a ~0 realised KL on an arm labelled anchored means the policy never
  tried to leave the snapshot, which is a different reading from a retention null.
  Mis-wiring RAISES (contract K7): switch without weight, weight without switch, non-positive
  coefficient, or rep.policy() returning None -- each would otherwise yield an arm that IS the
  control while labelled the treatment.
  Z_WORLD COTRAIN CAVEAT (documented, not defended against): under cotrain_encoder=True the
  encoder moves, so pi_ref is evaluated on a drifting input and the anchor is only approximate.
  The retention reference build is raw_view with cotrain_encoder=False, on which it is exact.
  NOT BUILT, deliberately: adaptive coefficient control / a target_kl trust-region radius. A
  second moving part would confound the leg's single declared intervention.
  Motivation (measured): V3-EXQ-780 raw_view post-BC competence 20.933 eroded to 11.667 under
  unconstrained RL refinement, 3/3 seeds having taken the install.
  ML statement (engineering counsel only): KL-penalty trust region (Schulman 2015/2017 TRPO/PPO);
  the biology is the consolidation/protection pathway the portfolio names, not the ML framing.
  Phased training required: no (no encoder head on a moving latent target). MECH-094: not
  applicable (no memory writes).
  Validation experiment: NOT QUEUED -- queueing the leg is governed by GOV-FANOUT-1 and routed
  through /queue-experiment. Unblocks hypothesis H-retention-consolidation (competence_floor
  question, evidence/planning/hypothesis_space_registry.v1.json), completing the three-leg
  retention portfolio alongside V3-EXQ-788 / V3-EXQ-789. MECH-457 stays candidate/v3_pending;
  this build promotes and demotes nothing.
  11 new contracts: tests/contracts/test_mech457_policy_kl_anchor.py.
  See REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md and the
  substrate_queue node mech457_policy_kl_anchor.
