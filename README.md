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

SD-004 and SD-005 are interdependent: action objects require z_world to exist; z_world's separation from z_self is what makes action-object attribution meaningful.

## Current Status (2026-04-21)

~715 V3 runs indexed (post indexer rebuild 2026-04-20T19:49Z, with ~9 further V3 manifests written since rebuild covering SD-033a / SD-034 / MECH-267 landings and the SD-032 cluster behavioural follow-through). SD-004 through SD-023 implemented plus SD-016, ARC-033, MECH-090 (bistable + Layer 1 stepping) + MECH-091 Layer 2 urgency interrupt, MECH-120, MECH-203/204, MECH-205, MECH-216, the **SD-032 cingulate cluster (a/b/c/d/e, all landed 2026-04-19)**, **SD-033a lateral-PFC-analog (MECH-261 primary consumer, landed 2026-04-20, V3-EXQ-456 PASS)**, and the **SD-034 governance closure-operator / MECH-267 mode-conditioned hippocampal / MECH-268 dACC conflict-saturation cluster (landed 2026-04-20 through 2026-04-21, all landing-diagnostic smoke PASS)**. **SD-003 superseded 2026-04-18** by MECH-256 + SD-029 + MECH-257; SD-030/SD-031 deferred to V4. SD-033 PFC subdivision cluster registered 2026-04-19 (SD-033 + SD-033a-e + MECH-262/263/264/265); SD-033a + SD-034 + MECH-267/268 implemented per the `sd033_governance_plan.md` anchor (2026-04-20 GAP MEMO: "REE-V3 is not missing cognition, it is missing governance."). Governance cycle 2026-04-19T21 promoted MECH-094 candidate->provisional, applied 12 `hold_pending_v3_substrate` decisions for the SD-032 cluster and dependents, and reclassified EXQ-395 / EXQ-418a / EXQ-430 as non_contributory substrate-gap symptoms. Regression suite PRs 1-5 landed: preflight + contracts + deferred changed layers, `/api/regression/preflight` serve.py endpoint, explorer preflight badge, and pre-commit contracts hook. 14 experiments queued (4 claimed: V3-EXQ-447 SD-032d ree-cloud-2; V3-EXQ-451 Q-034 retest EWIN-PC; V3-EXQ-445a SD-032b full-pipeline fix EWIN-PC; V3-EXQ-397c ARC-007 harder-env DLAPTOP-4); pending queue includes SD-034 / MECH-268 landing smokes (V3-EXQ-460, 463, 466, 468), SD-033a landing (V3-EXQ-456), SD-016 forward-path probe (V3-EXQ-449a), MECH-091 / MECH-104 discriminative pairs (V3-EXQ-133, V3-EXQ-126), and SD-032b variants (V3-EXQ-445b/c). Pending review queue last regenerated 2026-04-20T05:50Z; fresh rebuild pending after this cycle's wave of results. Current focus: SD-032 cluster behavioural follow-through (EXQ-445a fixes monostrategy + terrain inversion), SD-003 successor track (EXQ-433a FAIL, EXQ-452 diagnostic FAIL), ARC-007 path-memory track (EXQ-397c claimed), SD-033 governance-cluster landing validation.

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
