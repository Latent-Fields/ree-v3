## SD-033b GAP-8 DECOUPLE: separate OFC devaluation_bias_head (clamp-starved devalued range; failure_autopsy V3-EXQ-485l) (2026-06-22)
- SD-033b GAP-8 devaluation-head decouple: pfc.ofc_analog.devaluation_bias_head --
  IMPLEMENTED 2026-06-22 (substrate; SD-033b / MECH-263 stay candidate /
  substrate_conditional / pending_retest_after_substrate -- this PROMOTES NOTHING;
  the 485-lineage behavioural retest is GATED behind this build and queued
  separately). Routed by the user-confirmed failure_autopsy_V3-EXQ-485l_2026-06-22
  (re-derive brake FIRED -> implement-substrate; a plain 485m gain-tweak REFUSED).
  ROOT CAUSE (code-confirmed): the single shared OFCAnalog.state_bias_head under the
  +/-ofc_bias_scale clamp has NO feasible gain band. MECH-449 (Go/No-Go constitution)
  is BUILT + validated (689g PASS) but its viability No-Go input is starved: the
  devaluation re-ranking driver must produce a differentiated DEVALUED cross-candidate
  range above the 0.05 readout floor while the SAME head + SAME +/-0.5 clamp also
  carries the C2 high-threat discrimination range. 485k gain 4.0 SATURATED the clamp
  (devalued range 0.0); 485l gain 1.5 UNDERSHOT it (0.031 < 0.05). The bias VECTOR
  inverts cleanly (l2 1.83, cosine -0.716, C1b PASS 2/3) -- direction is RIGHT -- but
  the clamp compresses the MAGNITUDE below floor, so MECH-449 engaged only 1/3 (< 2/3
  gate) and the devaluation behavioural DV could not register. The residual is
  STRUCTURAL SCALE, not a tunable gain on the shared clamped head.
  THE FIX (no-op default; bit-identical OFF): a SECOND output head decoupling the
  devaluation re-ranking from the C2 discrimination readout.
    Module: ree_core/pfc/ofc_analog.py -- OFCConfig.use_devaluation_head (bool,
      default False) + devaluation_bias_scale (float, default 2.0) +
      train_devaluation_head (bool, default False). When use_devaluation_head, OFCAnalog
      builds devaluation_bias_head (same Linear(state_dim+world_dim -> hidden -> 1)
      shape as state_bias_head), sharing the [state_code, per-candidate summary] input
      but clamped to +/-devaluation_bias_scale -- an INDEPENDENT dynamic range. New
      method compute_devaluation_bias() (clamp +/-devaluation_bias_scale; returns zeros
      when the head is off) + devaluation_bias_head_parameters() (empty iter when off) +
      get_state() reports the new fields + _last_devaluation_bias_abs_mean diagnostic.
      Last Linear zeroed unless train_devaluation_head (mirror of the GAP-8
      state_bias_head zeroing) so the head is bit-identical until deliberately trained.
    Config: REEConfig.use_ofc_devaluation_head (False) + ofc_devaluation_bias_scale
      (2.0) + ofc_train_devaluation_head (False) + from_dims signature + assignment.
    Agent: the OFCConfig build site (agent.py) forwards all three via getattr fallback
      (absent flat attr -> bit-identical). NO runtime composition change -- select_action
      does NOT add the devaluation bias; the head is read by the experiment directly
      (compute_devaluation_bias -> _build_viability_nogo), so the action stream is
      bit-identical.
  WHY this is the structural fix (autopsy a+b+c in one): the C2 discrimination head
  keeps its +/-ofc_bias_scale clamp untouched (magnitude no longer traded against C1),
  while the devaluation head's larger independent clamp lets the in-band re-ranking gain
  produce a supra-floor differentiated devalued range WITHOUT saturating (decouple = a;
  independent clamp = rescale = b); the No-Go viability mapping re-derives from the
  decoupled head's range (c). It is a structural decouple, NOT a gain change on the
  shared clamped head -- the lettered-iteration loop the re-derive brake refuses.
  Backward compatible: use_ofc_devaluation_head=False by default -> devaluation_bias_head
  is None, compute_devaluation_bias returns zeros, no params -> bit-identical (every
  existing experiment reads only compute_bias). 8/8 preflight + OFC/boot/config contracts
  PASS; 6 new contracts in tests/contracts/test_sd_033b_gap8_devaluation_head.py (C1
  default-off no-op / C2 trainable decoupled head clears the 0.05 floor at its own clamp
  while C2 head stays clamped to bias_scale untraded / C3 dev-only optimizer trains the
  dev head + freezes state_bias_head / C4 untrained head zeroed + get_state fields / C5
  from_dims wiring / C6 inert when use_ofc_analog off). Module smoke: trainable decoupled
  head range 0.63 >> 0.05 floor at clamp 2.0; C2 head |max| 0.10 <= bias_scale untraded;
  dev-only gradient leaves C2 frozen.
  Phased training: N/A at the substrate level (a second readout head on the existing
  state_code, same E3-gradient path as the GAP-8 state_bias_head -- no new latent
  target, no collapse risk). The 485-lineage retest trains the head in P1 via the
  established REINFORCE-over-candidates pattern (frozen-encoder), exactly as the
  state_bias_head GAP-8 arm does. MECH-094: N/A -- compute_bias / compute_devaluation_bias
  run on the waking select_action / experiment path; no replay/memory write surface.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (no second head), so no dependent claim's measured
  mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. SD-033b / MECH-263 stay candidate / substrate_conditional /
  pending_retest_after_substrate; MECH-448 / MECH-449 unweakened (both engaged correctly
  within scope). claims.yaml carries only an implementation_note (no flag/confidence/
  status change).
  Validation experiment: a 485-lineage behavioural retest (NEW letter, supersedes 485l;
  claim_ids=[SD-033b, MECH-263]) GATED behind this build -- arms use_ofc_devaluation_head
  + ofc_train_devaluation_head + the in-band devalued re-ranking driver on the decoupled
  head, reads the devalued range + No-Go viability from compute_devaluation_bias, and
  re-states the readiness gate on the decoupled head's supra-floor differentiated range.
  Queued via /queue-experiment in a separate session per the chip scope; do NOT re-author
  485l.
  Design doc: REE_assembly/docs/architecture/sd_033b_ofc_analog.md (GAP-8 decouple note).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-485l_2026-06-22.md.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling failure record).
  See SD-033b (parent), SD-033b GAP-8 trainable state_bias_head (the head this decouples
  from; landed 2026-06-09), MECH-449 (Go/No-Go constitution -- the No-Go this supra-floor
  range feeds; built + validated 689g, input-starved here), MECH-448 (rank-preserving
  F->eligibility demotion; converts the C2 discrimination signature), MECH-263 (the
  devaluation + task-role signatures the retest validates), commitment_closure:GAP-8 (the
  closure-plan gap), f_dominance_conversion_ceiling (the substrate lineage), V3-EXQ-485k
  (saturate) / V3-EXQ-485l (undershoot -- the FAIL this addresses), MECH-094 (N/A).
