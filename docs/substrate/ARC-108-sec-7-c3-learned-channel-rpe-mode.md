## ARC-108 sec-7 C3: learned_channel_rpe_mode signed/unsigned ablation flag (unblocks V3-EXQ-700 C3 arm; the signed-RPE-is-load-bearing falsifier knob) (2026-06-22)
- ARC-108 sec-7 C3: selection.learned_channel_rpe_mode -- IMPLEMENTED 2026-06-22
  (substrate; ARC-108 stays candidate / substrate_conditional / implementation_phase=v3 --
  this PROMOTES NOTHING. The C3 ablation arm itself is a SEPARATE /queue-experiment session
  gated on this flag landing). The runtime knob the V3-EXQ-700 ARC-108 sec-7 learned-gating
  2x2 DEFERRED: C3 ("swap the signed delta_t for the unsigned ARC-016 variance -- it must
  FAIL to convert; if unsigned converts just as well, route back to ARC-016; falsifies
  divergence B5") had NO runtime knob because delta_t = R_t - V-hat_t was hard-wired in the
  JOB-1/W_lat three-factor update block. Design-of-record:
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md sec 5.2 C3.
  THE FLAG (no-op default; bit-identical OFF): E3Config.learned_channel_rpe_mode
  (Literal["signed","unsigned"], default "signed"), surfaced through REEConfig.from_dims onto
  config.e3. In ree_core/predictors/e3_selector.py post_action_update (the shared JOB-1
  w_chan + MECH-450 W_lat three-factor block), a single learn_signal is formed:
    "signed" (default) -> learn_signal = delta_t (the signed R_t - V-hat_t) -- BYTE-IDENTICAL
      to the original substrate (learn_signal == delta_t, learn_asym == the old asym, the
      add_() and _lcg_last_delta/_wlat_last_delta unchanged).
    "unsigned" -> learn_signal = abs(self._running_variance), the UNSIGNED ARC-016
      prediction-error magnitude (always >= 0; divergence B5's precision signal), substituted
      for delta_t in BOTH the w_chan AND W_lat updates -- removing the directional
      potentiate-vs-depress credit. learn_asym is computed from learn_signal, so under unsigned
      it is fixed at potentiation (learn_signal >= 0 always); the structural reason an unsigned
      signal CANNOT produce opposite-sign w_chan moves.
  SCOPE: only the LEARNING updates (w_chan + W_lat) change. The signed delta_t itself is kept
  intact for the JOB-2 habenula negative-delta_t de-commit (it reads metrics["habenula_delta_t"]
  = delta_t) and the V-hat_t baseline EMA (_lcg_value_baseline tracks R_t signed). So the C3
  ablation isolates the SELECTION teaching signal without touching the control-plane driver.
  Config: REEConfig.from_dims(learned_channel_rpe_mode="signed"|"unsigned") -> config.e3.
  Backward compatible: default "signed" -> learn_signal == delta_t exactly -> bit-identical to
  the pre-flag substrate for every existing run. 10/10 ARC-108 contracts (9 prior + new C7) +
  the MECH-450 / ARC-108 JOB-2 / e3-cluster (mech_448 / mech_449 / conflict-grade / score-bias /
  modulatory / DR-12 / ARC-065 GAP-A) 108-test cluster + 8/8 preflight PASS. C7
  (test_arc108_learned_channel_gating.py, mirror of the C5 signed potentiate-vs-depress
  contract): under learned_channel_rpe_mode="unsigned" a good outcome (R_t=+0.8) and a bad
  outcome (R_t=-0.8) move the voting channel's w_chan in the SAME direction
  (dw_pos * dw_neg > 0) -- the opposite-sign credit C5 shows for the signed delta_t is
  structurally impossible. (A non-zero realised state drives a positive ARC-016 running
  variance; the constant valuation-head stubs make R_t independent of it, so only the unsigned
  magnitude varies.) from_dims wiring smoke: explicit "unsigned" reaches config.e3; default
  "signed".
  Phased training: N/A (no learned parameters added; reuses the JOB-1 local three-factor rule
  and the already-trained valuation heads). MECH-094: unchanged -- the substitution lives
  inside the existing waking-only post_action_update gate (a replay/DMN tick forms no
  learn_signal and writes no w_chan/W_lat). Evidence-staleness (Step 8.5): NOT triggered --
  no-op-default flag; every existing experiment uses the default ("signed"), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. ARC-108 stays candidate / substrate_conditional /
  implementation_phase=v3; MECH-450 / MECH-439 / ARC-016 / ARC-107 untouched. claims.yaml
  carries only an implementation_note on ARC-108.
  Validation experiment: NOT queued here -- the C3 ablation arm (an A1-config arm +
  learned_channel_rpe_mode="unsigned") is added to a V3-EXQ-700 sibling/successor in a SEPARATE
  /queue-experiment session, with the pre-registered acceptance that the unsigned arm must FAIL
  to convert while A1 (signed) converts; if unsigned converts just as well, the signed-RPE claim
  is refuted and the mechanism collapses to a precision re-weighting (route back to ARC-016, do
  NOT mint a learning claim).
  Design-of-record: REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md
  (sec 5.2 C3). Doc: REE_assembly/docs/architecture/dopamine_into_gating.md.
  See ARC-108 JOB-1 step-1 (the w_chan + signed-RPE delta_t this ablates; landed 2026-06-22),
  MECH-450 (ARC-108 JOB-1 step-2 -- the W_lat update the flag also covers), ARC-016 (the
  unsigned variance the ablation substitutes -- divergence B5; the route-back target if unsigned
  converts), MECH-439 (the F-dominance conversion ceiling the 2x2 attacks), V3-EXQ-700 (the
  selection 2x2 that deferred C3), MECH-094 (waking-only call-site scoping).
