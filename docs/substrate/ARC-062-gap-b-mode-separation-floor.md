## ARC-062 GAP-B mode-separation floor (2026-05-20)
- Follow-on to differential-heads (V3-EXQ-543i autopsy): at discriminator w~0.5
  the composed gated bias is base only -- delta_hat cancels in
  base + (2w-1)*delta_hat, so REINFORCE cannot train differentiation.
- Fix: GatedPolicyConfig.mode_separation_floor (default 0.0, bit-identical OFF).
  Composed bias becomes w*h0 + (1-w)*h1 + floor*(h0-h1). With differential
  heads this injects a non-cancelable mode contrast even when w=0.5.
- Optional P1 aux: p1_w_deviation_aux_weight penalizes w near 0.5 during
  outcome-coupled training (gated_policy.p1_training_auxiliary_loss).
- REEConfig: gated_policy_mode_separation_floor,
  gated_policy_p1_w_deviation_aux_weight (default 0).
- Validation: V3-EXQ-543k (supersedes 543i; same 12-arm design + floor/aux on
  gated arms; K_IDENTICAL_RUNS=3 basin-stability gate; manifest hostname).
