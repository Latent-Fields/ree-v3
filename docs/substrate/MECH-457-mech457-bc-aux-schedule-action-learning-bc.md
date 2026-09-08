## mech457_bc_aux_schedule: action_learning.bc_auxiliary_persistence_schedule -- IMPLEMENTED (2026-07-18)
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
