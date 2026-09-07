## ARC-108 JOB-1 step-1: learned dopamine-gated E3 selection (signed-RPE w_chan over the modulatory channels; the next MECH-439 attack, learned-not-arithmetic) (2026-06-22)
- ARC-108 (JOB 1 step 1): selection.unified_dopamine_substrate_learned_gating --
  IMPLEMENTED 2026-06-22 (substrate; ARC-108 stays candidate / substrate_conditional /
  implementation_phase=v3 -- this PROMOTES NOTHING. The sec-7 selection falsifier + the
  MECH-450 settling step + the JOB-2 control-plane pair are SEPARATE downstream chips,
  NOT queued here). THE FIRST learned object in the ARC-107 arbitration layer (which was
  pure arithmetic with no learned parameters -- the A.4 learning gap and the predicted
  root of F-dominance, MECH-439). User-ratified V3 pull-forward (claims.yaml master
  045c4b73df). Design-of-record: dopamine_into_gating_design_2026-06-22.md secs 2-4 +
  unified_dopamine_substrate_design_2026-06-22.md sec 1/10. Second worked instance of
  ARC-106 (the learning rule is grounded at FUNCTION -- three-factor Hebbian x signed
  RPE with D1-LTP/D2-LTD asymmetry -- and EARNS the dopaminergic psychiatric axis).
  WHAT LANDS (all in ree_core/predictors/e3_selector.py -- extend the selector, NO
  parallel module, ARC-106 G2):
    (1) A single LEARNED per-channel selection-weight vector w_chan over the modulatory
      channels that feed the select() _modulatory_accum composition site. At that site
      the genuinely-separable constituents are the three add-terms already tracked there
      -- score_bias (the composed dACC+lPFC+OFC+MECH-295+MECH-314+MECH-320 chain, summed
      UPSTREAM in agent.py), the MECH-341 entropy bonus, and the route bias -- so the
      minimal channel registry is _LCG_CHANNEL_NAMES=("score_bias","mech341","route")
      (C=3), indexed by name (a channel absent on a tick simply does not contribute, so
      w_chan stays a stable learned object). A finer per-head channel split (dACC vs
      lPFC vs ...) is a documented follow-on -- those are summed upstream before reaching
      select(), so splitting them needs either un-summed biases threaded into select() or
      w_chan moved upstream; out of step-1 scope per "compose at the _modulatory_accum
      site / extend e3_selector".
      Composition: _modulatory_accum = sum_c softplus(w_chan[c]) * channel_bias_c (was
      the unweighted sum). w_chan is a register_buffer (NOT nn.Parameter -- the
      three-factor plasticity is a LOCAL update, never touched by an optimizer/autograd),
      init w_chan[c]=ln(e-1) so softplus(w_chan[c])==1.0 EXACTLY in float32 -> the
      recompose reproduces the unweighted accumulator bit-for-bit (1.0*x==x, matching add
      order). Only _modulatory_accum is re-weighted -- raw scores / F (the MECH-448
      envelope + the commit decision) are UNTOUCHED, so learning composes strictly INSIDE
      the F-bounded MECH-448/449 eligible set the within-eligible shortlist-argmin + the
      authority rescale consume, and a learned weight can NEVER re-admit a No-Go-suppressed
      candidate (safety inherited from the envelope; contract C6).
    (2) Signed RPE delta_t = R_t - V-hat_t, formed in post_action_update(actual_z_world)
      (the waking post-step hook). R_t = (benefit_eval_head - harm_eval_head)(actual_z_world)
      -- realised outcome valence from the ALREADY-TRAINED valuation heads (reuse, detached;
      NO new encoder, NO phased training). V-hat_t = a slow EMA leaky-integrator baseline
      (scalar). delta_t is SIGNED by construction and is explicitly NOT the unsigned ARC-016
      prediction-error VARIANCE (e3._running_variance) -- divergence B5; the two are kept
      separate (an unsigned magnitude cannot supply the directional Go-up/No-Go-down credit
      a learned gate needs). Contract C5 makes the signedness load-bearing (a positive
      delta_t POTENTIATES the voting channel; a negative delta_t DEPRESSES it -- opposite-sign
      w_chan changes an unsigned substitution could not produce).
    (3) Three-factor update Delta w_chan[c] = eta * delta_t * eligibility_c * asym(delta_t),
      applied in-place under no_grad in post_action_update. eligibility_c =
      |channel_bias_c[selected]| recorded after selected_idx in select() into a decayed
      last-K-ticks Hebbian co-activation trace. asym renders the D1-LTP/D2-LTD asymmetry as
      a single asymmetric gain (potentiation on delta_t>=0 faster than depression on
      delta_t<0) -- the V3 single-vector rendering; the D1/D2 opponent-population split is
      ARC-109 (deferred V4).
    (4) MECH-450 recurrent-settling step (learned W_lat) is the SECOND factor and is OFF in
      this build (W_lat==0): the integration point is reserved (a marked comment at the
      _modulatory_accum recompose) but NOT enabled -- it is the next chip.
  Waking-only gate (MECH-094 / divergence B5 contract): select() gains a keyword-only
  simulation_mode (default False, backward-compatible); eligibility is recorded ONLY when
  not simulation_mode, and post_action_update updates w_chan ONLY when a fresh waking
  eligibility trace is pending. A replay/DMN tick forms no delta_t and writes no w_chan
  (contract C4). Per-episode: agent.reset() calls e3.clear_learned_channel_eligibility()
  to clear the within-episode credit window (the trace + pending flag); w_chan and V-hat_t
  PERSIST across episodes as the learned state.
  Config (E3Config + REEConfig.from_dims, all no-op default -> bit-identical OFF):
  use_learned_channel_gating (False, master) + learned_channel_gating_eta (0.01) +
  learned_channel_gating_elig_decay (0.9) + learned_channel_value_baseline_beta (0.05) +
  learned_channel_asym_potentiation (1.0) + learned_channel_asym_depression (0.5).
  Backward compatible: use_learned_channel_gating=False by default -> the recompose +
  eligibility-recording + post-action update blocks are skipped -> bit-identical. 9/9 new
  contracts in tests/contracts/test_arc108_learned_channel_gating.py (C1 config defaults +
  from_dims + softplus-unity init / C2 OFF == ON-at-init EXACT scores+selection + OFF writes
  no w_chan / C3 w_chan MOVES under a non-flat delta_t when ON, stays at init when OFF /
  C4 simulation tick writes no w_chan / C5 signed-RPE potentiate-vs-depress load-bearing /
  C6 envelope intact -- learned weight cannot re-admit an F-excluded candidate) + 8/8
  preflight + 168 e3-cluster contracts PASS. Agent-level activation smoke (real env, MECH-341
  channel ON): w_chan[mech341] moved 0.54132->0.54632 over 13 updates with non-flat delta_t;
  OFF stayed exactly at init; first action ON==OFF bit-identical at init.
  Phased training: N/A -- the three-factor rule is a local (non-backprop) update; R_t reuses
  the already-trained valuation heads. MECH-094: waking-only by the simulation_mode gate +
  the pending-eligibility coupling. Evidence-staleness (Step 8.5): NOT triggered -- no-op-default
  flag; every existing experiment uses the default (gating off), so no dependent claim's
  measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. ARC-108 stays candidate / substrate_conditional /
  implementation_phase=v3; MECH-450 / ARC-107 / MECH-448 / MECH-449 / MECH-439 / ARC-016
  untouched. claims.yaml carries only an implementation_note on ARC-108 + MECH-450.
  Validation experiment: NOT queued here -- the sec-7 selection 2x2 falsifier (learned-w_chan
  x learned-W_lat on the GAP-A divergent pool: committed-class entropy strict-above the
  envelope-only arm AND a matched-noise control, GROWING with training, with the
  signed-vs-unsigned-RPE ablation arm) is a SEPARATE /queue-experiment chip, sequenced after
  the MECH-450 settling step lands (factor 2 of the 2x2).
  Design doc: REE_assembly/docs/architecture/dopamine_into_gating.md. Design-of-record:
  REE_assembly/evidence/planning/dopamine_into_gating_design_2026-06-22.md (secs 2-4) +
  unified_dopamine_substrate_design_2026-06-22.md (sec 1/10).
  See ARC-108 (this claim; JOB-1 step-1 slice), ARC-107 (the BG-constitution umbrella whose
  arithmetic envelope this adds the missing LEARNING afferent to), MECH-448 / MECH-449 (the
  F-bounded eligible set learning composes INSIDE; safety inherited), MECH-450 (the coupled
  settling step -- factor 2, OFF here), MECH-439 (the F-dominance conversion ceiling this
  attacks by learned re-weighting), ARC-016 (the unsigned variance kept SEPARATE -- divergence
  B5), ARC-106 (grounding framework; second worked application), ARC-109 (the D1/D2 population
  split -- deferred V4), MECH-094 (waking-only call-site scoping).
