# ree-v3

Active V3 implementation substrate for REE. Primary development target as of 2026-03-18 following formal V3 transition triggered by V2 hard-stop criteria.

## Why V3

V2 experiments (EXQ-014–028) produced three architectural failures requiring new substrate:

- **EXQ-027 FAIL**: E2 cannot discriminate agent-caused harm in z_gamma → z_self/z_world split required (SD-005)
- **EXQ-023 FAIL**: E1/E2 timescale not separable in shared latent → SD-005 needed
- **EXQ-021 FAIL**: Hippocampal map navigation fails without action-object backbone (SD-004)

## Core Design Decisions

**Founding substrate decisions (co-designed):**
- **SD-004**: Action objects as hippocampal map backbone — E2 produces compressed action-object representations that HippocampalModule maps over, enabling planning horizons far beyond raw state space.
- **SD-005**: Self/World latent split — z_gamma replaced by z_self (E2 domain: motor-sensory, proprioceptive) and z_world (E3/Hippocampus domain: residue, moral attribution, causal footprint). SD-004 and SD-005 must be co-designed.
- **SD-006**: Asynchronous multi-rate loop execution — E1, E2, E3 run at characteristic rates rather than lockstep.

**V3 substrate refinements (all implemented):**
- **SD-007**: Perspective-corrected z_world via ReafferencePredictor (implemented 2026-03-18, validated EXQ-027).
- **SD-008**: alpha_world >= 0.9 in LatentStackConfig — prevents EMA suppression of event responses (validated EXQ-040).
- **SD-009**: Event-contrastive CE auxiliary loss for z_world encoder — forces hazard vs empty discrimination (validated EXQ-020).
- **SD-010**: Harm stream separated as dedicated pathway (z_harm) — prerequisite for SD-003 counterfactual (validated EXQ-056c/058b/059c).
- **SD-011**: Dual nociceptive streams (z_harm_s + z_harm_a) — sensory-discriminative and affective-motivational streams (implemented 2026-03-30, validated EXQ-178b). Extended with harm history FIFO input (2026-04-08).
- **SD-012**: Homeostatic drive modulation for z_goal seeding — drive_weight scales benefit_exposure by depletion level (implemented 2026-04-02).
- **SD-013**: E2_harm_s interventional training — counterfactual margin loss for identifiable causal attribution (implemented 2026-04-10).
- **SD-014**: Hippocampal valence vector node recording — 4-component valence vector [wanting, liking, harm_discriminative, surprise] in RBFLayer and ResidueField (implemented 2026-04-04). Prerequisite for ARC-036 and replay prioritisation.
- **SD-015**: Resource indicator encoder (ResourceEncoder, z_resource) — location-invariant goal seeding for MECH-112 structured goal representation (implemented 2026-04-10).
- **SD-017**: Minimal sleep-phase infrastructure — run_sws_schema_pass(), run_rem_attribution_pass(), run_sleep_cycle() as first-class REEAgent methods (implemented 2026-04-09).
- **SD-018**: Resource proximity supervision — auxiliary Sigmoid head on z_world forcing proximity representation; prerequisite for benefit/goal pathway (implemented 2026-04-07).
- **SD-019**: Affective nonredundancy constraint — cosine penalty preventing z_harm_a from collapsing to monotone transform of z_harm_s (implemented 2026-04-10).
- **SD-020**: Affective harm surprise PE — z_harm_a encodes unexpected threat escalation (AIC analog) rather than raw magnitude (implemented 2026-04-10).
- **SD-021**: Descending pain modulation — commitment-gated z_harm_s attenuation via pgACC->PAG->RVM analog (implemented 2026-04-10).
- **SD-022**: Directional limb damage — 4-directional body damage state providing causal independence between z_harm_s and z_harm_a (implemented 2026-04-09).
- **SD-023**: Environmental gradient texture — Landmark A/B Gaussian fields in CausalGridWorldV2 for predictive cue structure (implemented 2026-04-09).
- **ARC-028 + MECH-105**: HippocampalModule completion signal + BetaGate coupling — implements Lisman & Grace 2005 subiculum->NAc->VP->VTA dopamine loop (implemented 2026-04-04).
- **ARC-033**: E2_harm_s forward model (ResidualHarmForward) — counterfactual harm-stream pipeline for SD-003 self-attribution (implemented 2026-04-09).
- **MECH-090**: Bistable beta gate — latches on commitment entry; hippocampal completion signal as release trigger (implemented 2026-04-10). Layer 1: trajectory stepping through committed_trajectory.actions[idx] via _committed_step_idx counter (implemented 2026-04-15).
- **MECH-091 Layer 2**: Urgency interrupt — when beta elevated and z_harm_a.norm() exceeds urgency_interrupt_threshold (E3Config, default 0.8), gate releases and step counter resets (implemented 2026-04-15).
- **MECH-120**: SHY synaptic homeostasis wiring — enter_sws_mode() calls shy_normalise() when shy_enabled=True (wired 2026-04-08).
- **MECH-203 + MECH-204**: Serotonergic sleep substrate — SerotoninModule with tonic_5ht state, benefit salience tagging, REM zero-point hook (implemented 2026-04-07).
- **MECH-205**: Surprise-gated replay write path — PE EMA tracking with pe_ema_alpha config, write count diagnostic (fixed 2026-04-09).
- **MECH-216**: E1 predictive wanting / schema readout — schema_readout_head on LSTM hidden state; schema_salience -> VALENCE_WANTING write (implemented 2026-04-09).
- **SD-016**: Frontal cue-indexed integration — E1 queries ContextMemory via z_world; cue_action_proj provides affordance bias for E2; cue_terrain_proj provides (w_harm, w_goal) terrain precision weights for E3 (implemented 2026-04-16).
- **SD-032 cluster (cingulate integration substrate, all implemented 2026-04-19):**
  - **SD-032b** dACC/aMCC-analog adaptive control (Croxson/Shenhav/Kolling bundle driving E3 score_bias via DACCtoE3Adapter shim) — first substrate, resolves the EXQ-395 z_harm_a wiring gap.
  - **SD-032a** salience-network coordinator (soft operating_mode vector + MECH-259 Schmitt-trigger switch threshold + MECH-261 dict-keyed write-gate registry).
  - **SD-032c** AIC-analog interoceptive salience / urgency-interrupt (subsumes SD-021: harm_s_gain is jointly drive- and mode-aware).
  - **SD-032d** PCC-analog metastability scalar (modulates MECH-259 effective_threshold by drive_level, success EMA, time-since-offline).
  - **SD-032e** pACC-analog slow-EMA autonomic coupling (drive_bias write-back from z_harm_a, MECH-094 hypothesis_tag gated).
- **SD-033a**: Lateral-PFC-analog / MECH-261 primary consumer — gate-modulated EMA rule_state with zeroed-last-Linear bias head; V3-EXQ-456 landing PASS (implemented 2026-04-20).
- **SD-034**: Governance closure operator — 5-part "done" token (MECH-090 beta release + MECH-260 targeted No-Go + rule-domain residue discharge + closure_event affinity bias + MECH-268 pe reset) gated by mode-conditioning predicate over SD-033a write_gate (implemented 2026-04-20).
- **MECH-267**: Mode-conditioned hippocampal proposals — operating_mode threads through HippocampalModule.propose_trajectories with per-mode CEM-noise multipliers (implemented 2026-04-20).
- **MECH-268**: dACC conflict saturation — outcome-history FIFO + f_sat attenuation layered atop SD-034 pe_cap and MECH-258 precision-weighted PE; closure_event resets the buffer (implemented 2026-04-21).
- **MECH-266**: Asymmetric per-mode hysteresis — Schmitt-trigger per-mode enter/exit rails over SD-032a's MECH-259 symmetric switch threshold; empty-dict defaults preserve legacy behaviour (implemented 2026-04-21).
- **SD-029**: Balanced hazard-event curriculum — scheduled external hazard injection in CausalGridWorldV2 to preserve per-seed n_self/n_ext >= 20 for C3/C4 comparator tests (implemented 2026-04-21).
- **SD-035**: Amygdala analogue — BLA + CeA peer modules (MECH-046 CeA mode-prior + MECH-074a/b/c/d BLA encoding gain, content-selective retrieval bias, fast prime, PE-spike remap); non-trainable arithmetic, BLA hippocampal consumer wiring for retrieval/remap deferred to first-pass post V3-EXQ-474 (implemented 2026-04-21).
- **SD-036**: GABAergic cross-stream decay regulator — broadly-projecting tonic decay across registered latent streams (z_harm / z_harm_a / z_beta by default); global gaba_tone knob models benzo / withdrawal regimes; out-of-place decay (preserves autograd version tracking for aux losses) (implemented 2026-04-22).
- **MECH-279**: PAG freeze-gate — committed-freeze substrate keyed on duration * magnitude of z_harm_a; exit threshold scales with SD-036 gaba_tone, so the same GABAergic system gates entry (PAG freeze-cell commitment) and exit; action-class no-op injection during freeze (implemented 2026-04-22).
- **MECH-269 Phase 1**: Hippocampal per-stream verisimilitude — HippocampalModule maintains V_s[stream] EMA using an identity-prediction proxy; foundation substrate for the V_s invalidation runtime that downstream consumers (MECH-287 broadcast trigger, MECH-284 staleness accumulator) will query (implemented 2026-04-22).
- **MECH-288**: Event segmenter — two-scale boundary detector (fast pe_threshold on z_world/z_self; slow BOCPD-Gaussian on z_goal) emitting BoundaryEvents with nested outer.inner segment IDs; force_boundary API for scripted injection; queued on HippocampalModule for downstream consumers (implemented 2026-04-22).
- **MECH-287**: Invalidation trigger — BoundaryEvent subscriber re-emitting graded BroadcastEvents (strength = posterior * gain, NO binary thresholding); phasic/tonic guardrail (Aston-Jones & Cohen 2005) via rolling posterior mean; verdict-3 option-c collapse of the biological two-stage comparator to a subscriber (implemented 2026-04-22).
- **MECH-269 Phase 2 ii**: AnchorSet — scale-tagged hippocampal anchor store keyed on (scale, segment_id, stream_mixture); dual-trace preservation (Bouton 2004) -- remap marks outgoing anchor INACTIVE not erased; k=5 consecutive-below hysteresis on V_s_anchor; FIFO soft-cap per scale; BoundaryEvent consumer (implemented 2026-04-22).
- **MECH-269 Phase 2 iii T4**: Per-region V_s readout — per_region_vs[(scale, segment_id)][stream] keyed on active AnchorSet regions; MECH-287 broadcast reset path (drop + mark_inactive); peek-not-drain on broadcast queue preserves events for Phase 3 staleness accumulator consumers (implemented 2026-04-22).

SD-004 and SD-005 are interdependent: action objects require z_world to exist; z_world's separation from z_self is what makes action-object attribution meaningful.

## Current Status (2026-04-23)

525 total completions (runner_status.json last_updated 2026-04-22T01:11Z; next indexer rebuild refreshes the count after the 2026-04-22 substrate wave): 108 PASS / 234 FAIL / 62 ERROR / 121 UNKNOWN across EXQ-001 through EXQ-474 plus lettered iterations. SD-004 through SD-023 implemented plus SD-016, ARC-033, MECH-090 (bistable + Layer 1 stepping) + MECH-091 Layer 2 urgency interrupt, MECH-120, MECH-203/204, MECH-205, MECH-216, the **SD-032 cingulate cluster (a/b/c/d/e, all landed 2026-04-19)**, **SD-033a lateral-PFC-analog (MECH-261 primary consumer, landed 2026-04-20, V3-EXQ-456 PASS)**, the **SD-034 governance closure-operator / MECH-267 mode-conditioned hippocampal / MECH-268 dACC conflict-saturation cluster (landed 2026-04-20 through 2026-04-21, all landing-diagnostic smokes PASS)**, the **MECH-266 asymmetric per-mode hysteresis extension on SalienceCoordinator (landed 2026-04-21)**, the **SD-029 curriculum-level balanced hazard-event support in CausalGridWorldV2 (landed 2026-04-21)**, the **SD-035 amygdala-analog peer modules (BLA + CeA, landed 2026-04-21; V3-EXQ-473 CeA mode-prior PASS, V3-EXQ-474 BLA encoding+remap PASS)**, and the **2026-04-22 V_s invalidation runtime wave: SD-036 GABAergic cross-stream decay regulator + MECH-279 PAG freeze-gate, MECH-269 Phase 1 per-stream V_s foundation, MECH-288 event segmenter Phase 2, MECH-287 invalidation trigger Phase 2 iv, MECH-269 Phase 2 ii AnchorSet substrate with dual-trace preservation + k=5 hysteresis, MECH-269 Phase 2 iii T4 per-region V_s readout -- all landing via contract tests (85/85 PASS) and activation smokes; end-to-end validation (V3-EXQ-476 combined-cluster re-run of EXQ-475) planned but not yet queued**. **SD-003 superseded 2026-04-18** by MECH-256 + SD-029 + MECH-257; SD-030/SD-031 deferred to V4. SD-033 PFC subdivision cluster registered 2026-04-19 (SD-033 + SD-033a-e + MECH-262/263/264/265); SD-033a + SD-034 + MECH-267/268 + MECH-266 implemented per the `sd033_governance_plan.md` anchor (2026-04-20 GAP MEMO: "REE-V3 is not missing cognition, it is missing governance."). SD-033e frontopolar-analog V4-reserved stub landed 2026-04-21 (FrontopolarConfig + FrontopolarAnalog skeleton with MECH-264 counterfactual-value and MECH-265 relative-importance heads raising NotImplementedError when enabled). MECH-270/271 (ephaptic substrate, MECH-094 as routing signature), MECH-272/273/274 (state-gated anchor/probe routing, sleep-dependent self-model aggregation, V4 other-attribution aggregation), MECH-284 (staleness accumulator Phase 3 consumer of per-region V_s), and MECH-275/276/277/278 + ARC-059 (scientist-agent developmental ordering cluster refining ARC-019) registered in claims.yaml 2026-04-21 -- all v3_pending, substrate not yet started (MECH-269 anchor selection core moved to Implemented across Phase 1/2 ii/2 iii T4 on 2026-04-22). Governance cycle 2026-04-19T21 promoted MECH-094 candidate->provisional, applied 12 `hold_pending_v3_substrate` decisions for the SD-032 cluster and dependents, and reclassified EXQ-395 / EXQ-418a / EXQ-430 as non_contributory substrate-gap symptoms. Regression suite PRs 1-5 landed: preflight + contracts + deferred changed layers, `/api/regression/preflight` serve.py endpoint, explorer preflight badge, and pre-commit contracts hook; contracts suite 85/85 PASS after the V_s invalidation runtime contract additions (MECH-269 Phase 1 + MECH-288 + MECH-287 + MECH-269 Phase 2 ii + MECH-269 Phase 2 iii T4). **3 experiments queued -- all claimed:** V3-EXQ-447 (SD-032d deterministic validation, ree-cloud-2), V3-EXQ-451 (Q-034 hazard/resource threshold retest, EWIN-PC), V3-EXQ-445a (SD-032b full-pipeline fix, EWIN-PC). Pending review queue last regenerated 2026-04-22T23:12:38Z with 10 items (all UNKNOWN due to stale index; next indexer rebuild resolves); governance-cycle pass pending for SD-032 behavioural FAILs, the SD-035 / MECH-266 landings, and the V_s invalidation runtime substrate wave. Current focus: SD-032 cluster behavioural follow-through (V3-EXQ-445a is the decisive test after V3-EXQ-445 / 445b / 445c all FAILed), SD-003 successor track (V3-EXQ-433a FAIL, V3-EXQ-452 diagnostic FAIL), ARC-007 path-memory track (V3-EXQ-397c claimed), SD-035 first-pass hippocampal consumer wiring for BLA retrieval_bias / remap_signal (deferred until V3-EXQ-474 confirms behavioural signature), and queuing V3-EXQ-476 to exercise the full V_s invalidation circuit end-to-end.

Full specification with substrate status table: [`docs/ree-v3-spec.md`](docs/ree-v3-spec.md)

## Experiment Tagging

- `run_id` must end in `_v3`
- `architecture_epoch` must be `"ree_hybrid_guardrails_v1"`
- Results go to `REE_assembly/evidence/experiments/`

## Governance

From `REE_assembly/` root:

```bash
bash scripts/governance.sh
```

Or manually:
```bash
python evidence/experiments/scripts/build_experiment_indexes.py
python scripts/generate_pending_review.py
```

## License

Apache License 2.0 (see `LICENSE`).

## Citation

- Cite this repository using `CITATION.cff`.
- For canonical architectural attribution, cite Daniel Golden's REE specification in `https://github.com/Latent-Fields/REE_assembly/`.
