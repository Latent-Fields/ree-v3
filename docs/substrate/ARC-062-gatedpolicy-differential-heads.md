## ARC-062 GatedPolicy differential-heads robustness fix (2026-05-18)
- ARC-062: policy.gated_policy two-head reparameterization -- IMPLEMENTED
  2026-05-18. ree_core/policy/gated_policy.py. Motivated by the V3-EXQ-543h
  failure autopsy + cross-machine 543g replication (REE_assembly/evidence/
  planning/failure_autopsy_V3-EXQ-543h_2026-05-18.{md,json}): the same 543g
  config landed gating-ACTIVE on host-A but INERT (n_inert_gating_seeds=3,
  TV<0.05) on cloud-3 AND cloud-4 -- head_0==head_1 collapse is the common
  cross-machine attractor; differentiation was a rare lucky-basin escape.
  Root cause: under outcome-coupled REINFORCE the inert state (head_0==head_1,
  w irrelevant) is a flat equilibrium; head_init_bias_offset is softmax-
  invariant (behaviorally invisible) and the removed head_div term was
  satisfiable by softmax-canceling anti-symmetric offsets.
  Fix: when use_differential_heads=True the two heads are SYNTHESIZED as a
  shared trunk plus a candidate-axis-norm-pinned differential:
    base(x), delta(x) MLPs ; delta_hat = differential_bias_scale *
    delta / (||delta||_K + 1e-8) ; head_0 = base + delta_hat ;
    head_1 = base - delta_hat ; composed = base + (2w-1)*delta_hat.
  Why it makes collapse a non-equilibrium: (route 1) delta==0 is structurally
  unreachable -- delta_hat depends ONLY on delta's DIRECTION (scale-invariant
  normalization), so the loss gradient w.r.t. delta's magnitude is exactly
  zero; gradient descent never drives ||delta||->0 and the nonzero delta
  last-bias init keeps it off zero from step 0. (route 2) at w=0.5,
  d(gated)/dw = head_0-head_1 = 2*delta_hat != 0 by the norm pin, so REINFORCE
  gets a non-vanishing gradient to move w off 0.5 whenever the two pinned
  modes differ in return -- which the pin guarantees. Not the removed head_div
  term: delta_hat is a unit-norm direction over candidates added to a shared
  base, so a nonzero differential necessarily changes the candidate ranking at
  the gating extremes and cannot be softmax-canceled.
  Config: GatedPolicyConfig.use_differential_heads (default False -> two
  independent heads, bit-identical pre-fix path) and .differential_bias_scale
  (default 0.1, mirrors bias_scale; only read on the True path).
  Wiring path (how experiments toggle it): agent.py builds GatedPolicyConfig
  with use_differential_heads=getattr(config,"gated_policy_use_differential_
  heads",False) and differential_bias_scale=getattr(config,"gated_policy_
  differential_bias_scale",0.1) -- same getattr style as
  gated_policy_use_first_action_onehot / crystallize_at_phase3, bit-identical
  when the flat REEConfig attr is absent. An experiment sets
  config.gated_policy_use_differential_heads=True AFTER REEConfig.from_dims()
  and BEFORE REEAgent(config) (single clean build; no rebuild / RNG offset).
  crystallize() freezes (base,delta,discriminator) instead of
  (head_0,head_1,discriminator) when the flag is on -- MECH-334 write-protect
  semantics identical in both configs; expansion device/dtype follows base[0].
  When the flag is on, self.head_0/self.head_1 are None (no external consumer
  touches the modules; downstream reads GatedPolicyOutput.head_*_bias, which
  are the synthesized base +/- delta_hat). get_state() reports
  use_differential_heads for the 543i manifest.
  Backward compatible: default False; every existing experiment runs unchanged
  (verified: v3_exq_543h --dry-run flag-off path). Activation smoke: delta_hat
  L2-over-K == differential_bias_scale exactly; crystallize freezes base+delta
  +disc; heads structurally differ.
  Phased training: N/A (architecture-only; no new encoder/learning signal --
  the P1 loss is deliberately UNCHANGED so V3-EXQ-543i is a clean single-
  variable test of structure-vs-MECH-309). MECH-094: N/A (no simulation/
  replay/memory write; forward simulation_mode path unchanged).
  Validation experiment: V3-EXQ-543i (supersedes V3-EXQ-543g + V3-EXQ-543h),
  queued via /queue-experiment -- same 2x2(x2) design + identical P1 loss,
  only new factor use_differential_heads; acceptance n_inert_gating_seeds==0
  across all seeds AND >=2 machines AND C2/C3 context-discrimination pass.
  Decisive either way: escape -> ARC-062 weak reading viable (MECH-309 holds
  only for unstructured parametric policies); still collapses -> MECH-309
  strong confirmation -> ARC-063/V4 distributed CandidateRule field.
  claims.yaml: ARC-062 + MECH-333 carry an implementation_note only; NO flag/
  confidence/promotion change (governance gated on V3-EXQ-543i). The MECH-309-
  support reading of the 543 cluster is a parked governance follow-on
  (workstream A; not applied this session). See ARC-062, ARC-063, MECH-309,
  MECH-333, MECH-334, INV-074, SD-054, rule_apprehension_layer.md.
