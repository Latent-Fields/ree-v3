## scaffolded_sd054_onboarding AMEND: harm-pathway training STABILIZATION (decoupled encoder LR + LR warmup; 603p seed-fragility) (2026-06-16)
- scaffolded_sd054_onboarding harm-pathway-stabilization amend -- IMPLEMENTED 2026-06-16.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). Routed by the confirmed failure_autopsy_V3-EXQ-603p_2026-06-15
  (Branch B, user-confirmed) + the GAP-C node's BLOCKED-ON-HARM-PATHWAY-STABILIZATION-AMEND
  route. behavioral_diversity_isolation:GAP-C; ISOLATED harm-valuation subsystem (no GAP-A overlap).
  ROOT CAUSE (V3-EXQ-603p, claim-free Stage-H base-harm-landscape diagnostic): the base harm
  landscape (E3.harm_eval_head(z_world)) forms a discriminative head (harm_eval_range >= 0.02)
  on only 1/3 seeds at the EASIEST regime (proximity_harm=0.10), and tripling the global
  harm-pathway LR to 3e-3 COLLAPSES it to ~1e-23 on all three seeds. The 2026-06-09 harm-pathway
  optimizer puts the harm params in a SINGLE Adam group at scaffold_harm_pathway_lr (1e-3): the
  same LR co-trains the latent_stack ENCODER (the SD-018 proximity MSE backprops into it so
  z_world becomes hazard-discriminative) AND the harm HEADS. Raising that one LR drives the
  encoder to the trivial constant-z_world solution (range -> 0, the 3x-LR collapse), while 1e-3
  leaves most seeds under-converged. The mechanism is RIGHT (where the landscape forms, prox_corr
  0.44-0.83); convergence/seed-robustness is the gap. NOT a budget tweak and -- per the autopsy --
  NOT a global-LR raise (it collapses the landscape).
  THE FIX (two no-op-default levers; bit-identical OFF; stabilize WITHOUT raising the global LR):
    (1) scaffold_harm_pathway_encoder_lr (Optional[float], default None): when set, the latent_stack
      ENCODER params get their own Adam param group at this (typically LOWER) LR while the harm
      heads + E2_harm_s keep scaffold_harm_pathway_lr -- the encoder moves gently (escaping the
      collapse-to-constant basin) while the heads still extract the proximity mapping at the base
      rate. None -> single Adam group at scaffold_harm_pathway_lr (bit-identical to the pre-amend
      flat optimizer). New helper _harm_pathway_param_groups builds the two disjoint groups
      (shared encoder-first dedup so no param appears twice).
    (2) scaffold_harm_pathway_warmup_steps (int, default 0): linear LR warmup over the first N
      harm-pathway training steps -- scales every param group's LR from base/N up to base, then
      holds at base, easing the early-training basin where the encoder is most prone to the
      collapse (gradient stabilization). 0 -> no scaling (bit-identical).
  Both levers cover the autopsy's primary prescriptions (lower [encoder] LR with the heads still
  training at base + gradient stabilization); the "more training steps" candidate stays available
  via the existing budget knobs (scaffold_p0_episode_budget / scaffold_hazard_stage_episode_budget).
  Optimizer construction (_make_harm_pathway): enc_lr=None keeps the EXACT legacy
  optim.Adam(params, lr=base); enc_lr set uses optim.Adam(param_groups). Either way the per-group
  base LRs are stashed on harm_opt._harm_base_lrs for the warmup scaling in _harm_pathway_step
  (applied per param group right before the existing grad-clip + step; n_train_steps is the 0-based
  step index).
  Backward compatible: scaffold_harm_pathway_encoder_lr=None + scaffold_harm_pathway_warmup_steps=0
  by default -> single-group Adam at base LR, no LR scaling -> bit-identical to the 2026-06-09
  harm-pathway optimizer. 102/102 scaffolded contracts (97 prior + 5 new C16: stabilization config
  defaults no-op / encoder_lr=None single legacy group / encoder_lr decouples into two disjoint
  groups at distinct LRs / warmup ramps base/N -> base then holds, driven through real
  _harm_pathway_step calls / warmup=0 leaves LR at base) + 7/7 preflight PASS.
  Phased training: N/A (the curriculum IS phased; this changes only the harm-pathway optimizer's
  param-group LRs + a warmup schedule -- no new encoder head, no new latent target, no collapse
  risk introduced). MECH-094: N/A (waking goal-pipeline onboarding; no simulation/replay write
  surface). Evidence-staleness (Step 8.5): NOT triggered -- no-op-default levers; every existing
  experiment uses the defaults (single-group LR, no warmup), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  In-session validation (proof-of-fix probe; scripts/_validate_603q_harm_amend.py replicates the
  603p positive-control cell via 603p's own config builders): the levers run END-TO-END in the
  real 603p pipeline and the harm head learns the correct proximity mapping (prox_corr positive
  even at reduced budget). The full-scale >=2/3-seed seed-robustness confirmation at
  proximity_harm=0.10 is carried as V3-EXQ-603q's FIRST self-routing non-vacuity precondition
  (the cloud establishes it; 603q self-routes substrate_not_ready_requeue if the base does not
  clear >=2/3, never a false bridge verdict) -- per the GAP-C durable 603q spec.
  Validation experiment: V3-EXQ-603q (the corrected SD-059/MECH-358 escape-affordance-bridge
  EVIDENCE re-run, bridge ON vs base) runs on the now-stabilized base with the two levers ON +
  the base-harm-landscape >=2/3 discriminativeness as a self-routing precondition. See Step 8.
  Design doc: REE_assembly/docs/architecture/sd_054_scaffolded_onboarding_substrate_design.md
  (Amend 2026-06-16 section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603p_2026-06-15.md.
  See scaffolded_sd054_onboarding (parent + the 2026-06-09 Stage-H harm-pathway-training amend
  this stabilizes), SD-018 / SD-010 / SD-011 / ARC-033 (the harm-proximity supervision targets),
  SD-059 / MECH-358 (the escape-affordance bridge blocked on a discriminative base harm landscape),
  MECH-313 / MECH-260 / Q-045 (the GAP-C tonic-noise cluster gated downstream of 603q),
  behavioral_diversity_isolation:GAP-C (closure node), V3-EXQ-603p (the FAIL this amend addresses),
  V3-EXQ-603k (the narrow 603p-superseded probe whose "VALIDATED 2026-06-09" status was over-stated),
  V3-EXQ-603q (validation/evidence), MECH-094 (N/A).
