## SD-055: Differentiable CEM Selection Approximation (2026-05-15)
- SD-055: hippocampal.differentiable_cem_selection -- IMPLEMENTED 2026-05-15.
  Module: ree_core/hippocampal/module.py (post-elite refit block).
  Config: HippocampalConfig.use_differentiable_cem (bool, default False);
    HippocampalConfig.differentiable_cem_temperature (float, default 1.0);
    REEConfig.from_dims(use_differentiable_cem=..., differentiable_cem_temperature=...).
  Data flow: E2 rollouts score candidates -> softmax(-score/T) weights over ALL
    candidate ao sequences -> differentiable ao_mean (and ao_std) -> downstream
    HippocampalModule consumers. Legacy path (flag False): argsort elite fraction +
    indexed mean unchanged (bit-identical default).
  Motivation: EXP-0155 / EXQ-449 zero gradient through CEM argmax severs SD-016
    cue_action_proj learning (ARC-072 gap 2 diagnostic).
  Backward compatible: use_differentiable_cem=False by default; existing experiments
    unaffected. smoke_sd055_differentiable_cem.py 4/4 PASS (grad_max ~260 smoke,
    EXQ-568 grad_max=372).
  MECH-094: not applicable (waking CEM selection, not replay content).
  Phased training: not an encoder head; no P0/P1/P2 latent-target phasing required
    for the substrate switch itself. Behavioural experiments that train cue_action_proj
    should enable the flag explicitly.
  Validation experiment: V3-EXQ-568 PASS 20260515T204931Z (substrate-readiness,
    evidence_direction=non_contributory). Does NOT validate cue-conditioned behavioural
    divergence on goal-rich env.
  Design doc: REE_assembly/docs/architecture/sd_055_differentiable_cem_selection.md
  See SD-016, ARC-072, MECH-326, EXP-0155, developmental_bootstrapping_hippo_retrieval.md.

