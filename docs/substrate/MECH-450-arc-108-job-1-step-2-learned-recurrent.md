## MECH-450 (ARC-108 JOB-1 step-2): learned recurrent-settling step + learned lateral-inhibition W_lat (factor 2 of the learned-gating 2x2; B1 + B3-blend repair) (2026-06-22)
- MECH-450 (ARC-108 JOB-1 step-2): selection.minimal_recurrent_settling_step --
  IMPLEMENTED 2026-06-22 (substrate; MECH-450 stays candidate / substrate_conditional /
  implementation_phase=v3 -- this PROMOTES NOTHING. The sec-7 selection 2x2 falsifier is a
  separate /queue-experiment chip). The SECOND factor of the learned-gating 2x2 (JOB-1
  step-1 w_chan landed earlier 2026-06-22, ree-v3 ae907b5), coupled to it and sharing the
  SAME signed-RPE delta_t / V-hat_t / D1-D2 asym. Design-of-record:
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md sec 4 +
  REE_assembly/docs/architecture/dopamine_into_gating.md.
  WHAT LANDS (all in ree-v3/ree_core/predictors/e3_selector.py -- extend the selector, NO
  parallel module, ARC-106 G2):
    (1) A bounded recurrent LATERAL-INHIBITION SETTLING step over the F-bounded eligible set,
      run at the within-eligible arbitration site (on mod_eligible =
      _modulatory_accum[eligible_idx]) BEFORE the commit, in new helper
      E3TrajectorySelector._lateral_settle:
        accum = mod_eligible
        for r in range(R):                       # R = learned_settling_rounds, default 3
            a       = softmax(-accum / T)         # support over eligible (low cost -> high)
            a_class = onehot.T @ a                # [C] per-action-class aggregated support
            accum   = accum + onehot @ (W_lat @ a_class)   # learned lateral inhibition
        commit = argmin(settled accum) (committed) / sample(softmax(-settled accum/T)) (uncommitted)
      Fixes divergence B1 (one-shot argmin -> recurrent settling) AND B3-blend (additive
      _modulatory_accum blend -> competitive winner-take-most) together. The settling
      transforms ONLY the eligible subset and the commit reads the SETTLED accum (argmin /
      gap-scaled / softmax-sample all read it).
    (2) W_lat = the LEARNED lateral-inhibition matrix over candidate first-action CLASSES
      (a stable [C,C] object: the per-candidate set is variable-size with no stable identity,
      so the inhibition is parametrised by action class -- the BG surround-inhibition between
      competing motor programs, Mink 1996, the opponency MECH-449 grounds). register_buffer
      (NOT nn.Parameter -- the three-factor plasticity is a LOCAL update, never an
      optimizer/autograd target; rides device + state_dict), init 0 -> the settling step is a
      no-op -> BIT-IDENTICAL OFF and bit-identical at init. C = learned_settling_n_action_classes
      (default 8; first-action class clamped into range).
    (3) W_lat learned by the SAME three-factor rule as w_chan, off ONE shared signed RPE:
      post_action_update computes delta_t = R_t - V-hat_t ONCE (R_t = benefit_eval - harm_eval
      at the realised state from the already-trained heads, detached; V-hat_t slow EMA) and
      applies Delta W_lat = eta_w * delta_t * asym(delta_t) * coact_trace, where coact_trace is
      the decayed Hebbian co-activation (outer product) of the per-round settling-step class
      activations recorded in _lateral_settle. One dopaminergic RPE drives both w_chan and W_lat
      (the post_action_update block was restructured to compute delta_t / V-hat_t / asym once and
      branch to each lever -- the w_chan-only path stays byte-identical).
  Composes INSIDE the F-bounded MECH-448/449 eligible set (raw scores / F untouched), so a
  learned W_lat can never re-admit a No-Go-excluded candidate -- safety inherited from the
  envelope, exactly as for w_chan. Waking-only (MECH-094): the settling is gated on
  not simulation_mode (no settling, no co-activation trace on a replay/DMN tick), and the
  three-factor W_lat update fires only on a pending waking trace; agent.reset() clears the
  within-episode settling trace via the extended clear_learned_channel_eligibility (W_lat and
  V-hat_t persist across episodes). Diagnostics on last_score_diagnostics: learned_settling_active
  + learned_settling_round_delta (the L2 cross-round movement of the eligible accumulator -- the
  NON-DEGENERACY signal the falsifier checks); post_action_update metrics gain wlat_delta_t /
  wlat_range.
  Config (E3Config + REEConfig.from_dims, all no-op default -> bit-identical OFF):
  use_learned_settling_step (False, master) + learned_settling_rounds (3, R) +
  learned_settling_temperature (1.0, T) + learned_settling_eta (0.01, W_lat learning rate) +
  learned_settling_elig_decay (0.9, cross-tick co-activation decay) +
  learned_settling_n_action_classes (8, the W_lat dimension, clamped). Reuses the step-1
  delta_t / V-hat_t / learned_channel_asym_potentiation/_depression (one shared signed RPE).
  Backward compatible: use_learned_settling_step=False by default -> the settling block is
  skipped, the within-eligible arbitration runs the legacy argmin/sample on the unsettled accum,
  W_lat stays zero -> bit-identical. 10/10 new contracts in
  tests/contracts/test_mech450_learned_settling_step.py (C1 config defaults + from_dims +
  W_lat zero-init / C2 OFF == ON-at-init EXACT scores+selection + settling no-op at init [round_delta
  0] + OFF writes no W_lat / C3 W_lat MOVES under a non-flat delta_t when ON [w_chan OFF -- the
  settling learns independently] / stays at init when OFF / C4 simulation tick writes no W_lat /
  C5 non-degeneracy -- a non-zero W_lat MOVES the field across rounds [round_delta > 0], no-op at
  init / C6 envelope intact -- a strong W_lat cannot re-admit an F-excluded candidate / C7 shared
  delta_t -- both w_chan and W_lat move off init under one RPE) + 9/9 JOB-1 + 8/8 preflight + 74
  e3-cluster + 1232/1235 full contract suite (the 3 fails are the documented pre-existing
  runner-fail-branch / control_vector flakes -- CONFIRMED failing identically with the e3_selector +
  config edits stashed, zero e3 overlap). Selector activation smoke: W_lat range 0.0 -> 4.23 over 13
  updates with a non-flat delta_t (last 0.7623); settling round_delta 6.16 (live cross-round
  movement); OFF == ON-at-init bit-identical (exact scores + selection). Agent env-loop smoke (real
  CausalGridWorldV2, top-k shortlist eligible set + MECH-341 channel): the settling engaged 60/60
  ticks ON, 0 OFF (wiring engages end-to-end).
  Phased training: N/A (pure-arithmetic settling + a local three-factor plasticity rule; no learned
  parameters in the optimizer/autograd sense, no new encoder head). MECH-094: waking-only by the
  simulation_mode gate + the pending-trace coupling. Evidence-staleness (Step 8.5): NOT triggered --
  no-op-default flag; every existing experiment uses the default (settling off), so no dependent
  claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-450 stays candidate / substrate_conditional /
  implementation_phase=v3; ARC-108 / ARC-107 / MECH-448 / MECH-449 / MECH-439 untouched. claims.yaml
  carries only the MECH-450 implementation_note (NOT-built -> BUILT); claims.json byte-identical
  (implementation_note not projected); validate_claims --strict exit 0.
  Validation experiment: NOT queued here -- the sec-7 selection 2x2 falsifier (learned-w_chan x
  learned-W_lat on the GAP-A divergent pool: committed-class entropy strict-above the envelope-only
  arm AND a matched-noise control, growing with training, with the signed-vs-unsigned-RPE ablation)
  is now fully runnable (both factors built) and is a separate /queue-experiment chip.
  Design doc: REE_assembly/docs/architecture/dopamine_into_gating.md (JOB-1 steps 1+2). Design-of-record:
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md (sec 4) +
  unified_dopamine_substrate_design_2026-06-22.md.
  See MECH-450 (this claim; factor 2 of the 2x2), ARC-108 JOB-1 step-1 (the coupled w_chan + signed-RPE
  delta_t this shares; landed earlier 2026-06-22), ARC-107 (the BG-constitution umbrella whose one-shot
  pallidal readout this replaces with a bounded recurrent settling competition), MECH-448 / MECH-449
  (the F-bounded eligible set the settling composes inside; safety inherited), MECH-439 (the F-dominance
  conversion ceiling -- a hard argmin returns the F-winner; a settling competition can flip the
  attractor), ARC-106 (grounding framework; G2 reuse-before-duplicate -- extend e3_selector, no parallel
  module), ARC-109 (D1/D2 population split -- deferred V4), MECH-094 (waking-only call-site scoping).
