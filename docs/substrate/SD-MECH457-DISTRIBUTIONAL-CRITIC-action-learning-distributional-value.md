## SD-MECH457-DISTRIBUTIONAL-CRITIC: action_learning.distributional_value -- IMPLEMENTED (2026-07-18)
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
