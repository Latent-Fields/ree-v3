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
- **MECH-284 Phase 3**: V_s residual schema-staleness accumulator — region-indexed (scale, segment_id) staleness with per-tick leak, attribution_mode=equal/stream_overlap, staleness_clip; MECH-287 broadcast integration via tick_anchor_set peek-not-drain; lookup_by_anchor_key getter (implemented 2026-04-24).
- **MECH-269 online hysteresis swap**: AnchorSet.tick_hysteresis accepts an optional staleness_lookup callable; V_s_anchor = V_s(r) - staleness[r]; orthogonal flag use_mech284_hysteresis (default OFF preserves Phase 2 internal proxy); allows Phase 3 staleness signal to drive online anchor-reset without invalidating the legacy proxy (implemented 2026-04-24).
- **MECH-290**: Backward trajectory credit sweep — Foster & Wilson 2006 reverse replay; record_committed_trajectory at BetaGate elevation; backward_credit_sweep at completion-signal release; per-step credit = outcome_quality * gamma^(T-1-t) -> ResidueField.update_valence(VALENCE_WANTING); reset on episode boundary (implemented 2026-04-24).
- **SD-037**: Broadcast override regulator (orexin/hypocretin analog) — scalar override_signal in [0,1] driven by SD-012 drive_level + sustained-threat rolling-window over z_harm; consumed at PAG freeze-gate (raises exit_threshold), SalienceCoordinator (external_task affinity), and GoalState (drive-amplified z_goal seeding). MECH-094 simulation_mode gate (implemented 2026-04-25).
- **Sleep Aggregation Cluster Phase A**: SleepLoopManager scaffolding — wraps existing SD-017 surface; SleepPhase enum + SleepCycleState dataclass; episode-end notify with K-cycle drive (implemented 2026-04-25).
- **MECH-285 (Sleep Phase B)**: SleepReplaySampler — softmax(staleness/temperature) priority draws from AnchorSet.all_with_dual_trace() (Bouton 2004 dual-trace preserved); StalenessAccumulator.snapshot() frozen at SLEEP_ENTRY (implemented 2026-04-25).
- **MECH-272 (Sleep Phase C)**: RoutingGate — state-conditioned anchor/probe channel weights flipping across SWS_ANALOG / REM_ANALOG / WAKING rows; per-draw RoutedEvents surfaced in cycle metrics (implemented 2026-04-25).
- **MECH-275 (Sleep Phase D)**: BayesianAggregator — per-domain per-region Gaussian posteriors with conjugate update; probe-channel-gated by RoutedEvent.probe_channel * probe_gain; snapshot+decay contract (snapshot frozen pre-PHASE_SWITCH; decay_factor multiplies live variance per cycle) (implemented 2026-04-25).
- **MECH-273 (Sleep Phase E)**: SelfModelAggregator + StalenessAccumulator.partial_decay — subclass of MECH-275 specialised on SD-003 causal_sig posterior; offline_gradient_pass over E2_harm_s.parameters() at waking_lr * offline_lr_scale; replayed regions decayed multiplicatively (not zeroed). 150/150 contracts (143 contracts + 7 preflight) PASS with all flags OFF (implemented 2026-04-25).
- **SD-016 Path 1**: ContextMemory.compute_diversification_loss — auxiliary mean-squared-off-diagonal-cosine loss on normalised slot vectors; provides gradient pressure for slot symmetry-breaking missing in EXQ-418d 4-arm writepath ablation. sd016_diversification_weight default 0.0 (backward compatible). Smoke verified slot_div climbs 0.2->0.5->1.0 (implemented 2026-04-25).

- **SD-033b**: OFC-analog (MECH-261 second consumer) — gate-modulated EMA `state_code` with eff_eta = update_eta * write_gate("sd_033b"); zeroed-last-Linear bias head so initial bias is exactly zero; per-mode gate weights external_task=1.0 / internal_planning=0.5 / internal_replay=0.05 / offline_consolidation=0.3. Behavioural MECH-263 signatures (devaluation sensitivity, same-sensory / different-task-role discrimination) deferred to environment-extension EXQs (implemented 2026-04-26).
- **MECH-269b**: Symmetric V_s gating on E1/E2 cortical rollouts — read-side consumer of MECH-269 Phase 1 `per_stream_vs` at the E1 `_e1_tick` site and the per-tick E2_harm_a forward call site; held substitution swaps current latent for snapshot when V_s[s] < per-side threshold; 0.4-0.5 dead-band Schmitt-trigger hysteresis; precondition `use_per_stream_vs=True`. Q-040 factorial validation (V3-EXQ-490) queued (implemented 2026-04-26).
- **MECH-295 weak-reading bridge**: drive -> liking-stream -> approach_cue substrate — anticipatory liking-stream pulse at z_goal location via `update_z_goal` -> `ResidueField.update_valence(VALENCE_LIKING)`; per-candidate negative score_bias via `select_action`; severed-bridge falsification arm at cue gain=0; weak-necessity reading committed provisionally per the lit-pull synthesis. V3-EXQ-493 substrate-landing diagnostic (6 sub-tests including UC5 SEVERED-BRIDGE COLLAPSE) queued (implemented 2026-04-26).
- **SD-039 substrate**: Dual-trace anchor goal-snapshot payload — `AnchorGoalPayload` dataclass (z_goal_snapshot + wanting_strength + arousal_tag + last_vs + staleness_at_write + payload_written_step); refresh-on-invalidate semantic preserves payload across `mark_inactive`; `Anchor.goal_match` cosine helper + `AnchorSet.query_by_goal_match` active+inactive dual-trace getter for the MECH-292 consumer (substrate landed 2026-04-26).
- **SD-039 population layer**: Module-level write-site wiring — REEAgent.sense() builds `AnchorGoalPayload` once per tick from GoalState (z_goal_snapshot), ResidueField VALENCE_WANTING (wanting_strength), BLA arousal_tag, mean(per_stream_vs) (last_vs), max staleness_accumulator (staleness_at_write); threaded through `HippocampalModule.tick_anchor_set` + `apply_invalidation_broadcasts_to_regions`. MECH-094 simulation-mode gate. V3-EXQ-494 6/6 PASS (implemented 2026-04-27).
- **MECH-292**: Ranked ghost-goal bank — pure-arithmetic derived view over the SD-039 dual-trace anchor pool; `ghost_priority = w_w*wanting + w_m*goal_match + w_s*staleness + w_r*recoverability`; `goal_match_floor=0.05` rumination guard. Raises ValueError if `use_anchor_sets` / `use_sd039_anchor_payload` are off. V3-EXQ-496 5/5 PASS (implemented 2026-04-27).
- **MECH-293**: Waking ghost-goal probe search — `HippocampalModule.propose_trajectories` extended with a minority-budget ghost branch consuming MECH-292; `mech293_ghost_fraction=0.2` default; `Trajectory.hypothesis_tag` + `metadata` carried through CEM rollout, stripped at `record_committed_trajectory`; ARC-007 strict preserved (no value head; goal-match enters via MECH-292's external ranking). V3-EXQ-497 5/5 PASS (implemented 2026-04-27).
- **ARC-054 V3 form**: D_V trajectory selection promoted v4 -> v3 in synaptic-EMA form (rollout-horizon synaptic EMA over V_s readout, no TCL substrate dependency at V3); V4 form (phase-coherent V(t) integration via ARC-053 + MECH-225/226/228) remains v4-by-design. V3-EXQ-491 validation queued (V3 form promoted 2026-04-26).
- **MECH-271 V3 substrate plan**: Hypothesis tag as downstream routing committed for V3 in synaptic form (discrete routing table + audit hook for confabulation-vs-psychosis dissociation); V4 ephaptic-field-strength routing remains v4-by-design. V3-EXQ-492 routing 4-arm queued behind the MECH-269b lock release (plan committed 2026-04-26).

SD-004 and SD-005 are interdependent: action objects require z_world to exist; z_world's separation from z_self is what makes action-object attribution meaningful.

## Current Status (2026-04-28)

**561 runner-side completions** (per `runner_status.json` 2026-04-27T08:04Z read: 109 PASS / 242 FAIL / 66 ERROR / 144 UNKNOWN). +10 vs the 2026-04-26 read covering the 2026-04-27 substrate wave: SD-039 module-level write-site population layer (V3-EXQ-494 6/6 PASS); MECH-292 ranked ghost-goal bank (V3-EXQ-496 5/5 PASS); MECH-293 waking ghost-goal probe search (V3-EXQ-497 5/5 PASS) — all three clear MECH-163 dual-system substrate prerequisites. Plus V3-EXQ-484 / 485 / 493 PASS recovery after the 2026-04-27 /diagnose-errors run_id naming-bug fix (`f"{experiment_type}_{ts}_v3"` patched in source scripts; manifests renamed in place; sync_v3_results.py picks them up cleanly).

SD-004 through SD-023 implemented plus SD-016, ARC-033, MECH-090 (bistable + Layer 1 stepping) + MECH-091 Layer 2 urgency interrupt, MECH-120, MECH-203/204, MECH-205, MECH-216, the **SD-032 cingulate cluster (a/b/c/d/e, all landed 2026-04-19)**, **SD-033a lateral-PFC-analog (MECH-261 primary consumer, landed 2026-04-20, V3-EXQ-456 PASS)**, the **SD-034 governance closure-operator / MECH-267 mode-conditioned hippocampal / MECH-268 dACC conflict-saturation cluster (landed 2026-04-20 through 2026-04-21)**, the **MECH-266 asymmetric per-mode hysteresis extension on SalienceCoordinator (landed 2026-04-21)**, the **SD-029 curriculum-level balanced hazard-event support in CausalGridWorldV2 (landed 2026-04-21)**, the **SD-035 amygdala-analog peer modules (BLA + CeA, landed 2026-04-21; V3-EXQ-473/474 PASS)**, the **2026-04-22 V_s invalidation runtime wave (SD-036 GABAergic cross-stream decay + MECH-279 PAG freeze-gate; MECH-269 Phase 1; MECH-288 event segmenter; MECH-287 invalidation trigger; MECH-269 Phase 2 ii AnchorSet; MECH-269 Phase 2 iii T4 per-region V_s)**, the **2026-04-24 Phase 3 wave: MECH-284 staleness accumulator + MECH-269 online hysteresis swap + MECH-290 backward trajectory credit sweep**, the **2026-04-25 substrate wave: SD-037 BroadcastOverrideRegulator + Sleep Aggregation Cluster Phases A-E + SD-016 Path 1 ContextMemory diversification loss**, the **2026-04-26 substrate wave: SD-039 substrate + SD-033b OFC-analog + MECH-269b symmetric V_s gating + MECH-295 weak-reading liking bridge + ARC-054 v4 -> v3 promotion + MECH-271 V3 substrate plan**, and the **2026-04-27 substrate wave: SD-039 module-level write-site population + MECH-292 ranked ghost-goal bank + MECH-293 waking ghost-goal probe search** (the MECH-163 V3-full-completion-gate substrate prerequisites cleared end-to-end). **SD-003 superseded 2026-04-18** by MECH-256 + SD-029 + MECH-257; SD-030/SD-031 deferred to V4. SD-033e frontopolar-analog V4-reserved stub landed 2026-04-21. MECH-270 (ephaptic substrate of V_s) + MECH-274 + MECH-276/277/278 + ARC-059 (scientist-agent developmental ordering cluster) remain v3_pending. Regression suite PRs 1-5 remain live; contracts suite at **183/183 PASS + 7/7 preflight PASS** with all flags OFF after the 2026-04-27 wave -- bit-identical-when-OFF guarantee preserved.

**Queue (`experiment_queue.json` 2026-04-28): 5 items pending**, all unclaimed. **V3-EXQ-495** (MECH-163 V3 full-completion gate -- VTA / hippocampally-planned arm) is the headline run: 3 conditions (HABIT / PLANNED / ABLATED) × 2 paradigms (A_DETOUR / B_NOVEL_CONTEXT) × 7 seeds; acceptance C2 = PLANNED-HABIT benefit-post-block gap >= 0.30 in detour, >= 4/7 seeds (THE V3-full-completion criterion); estimated ~25h on Mac / ~40h on ree-cloud-1; machine_affinity=any. Substrate prerequisites all cleared 2026-04-27 (SD-039 / MECH-292 / MECH-293). **V3-EXQ-490b** (MECH-269b VsRolloutGate substrate-readiness probe; Q-040a precondition; supersedes V3-EXQ-490a) is a smoke-only threshold-override probe -- /diagnose-errors 2026-04-27 root-cause: V3-EXQ-490 + 490a both FAILed c1 (vs_gate_total_held=0) because Phase 1 V_s identity-prediction proxy stays near 0.9-1.0 under aligned latents; V3-EXQ-490b raises vs_gate thresholds to 0.85 (snapshot_refresh 0.95) so the gate fires; PASS confirms substrate wiring (Q-040a precondition) but does NOT test MECH-269b's stale-stream-discrimination hypothesis (Q-040b stays gated on Phase 2 forward-predictor V_s OR a substrate change wiring `staleness_accumulator` into `VsRolloutGate.gate()`). claim_ids=['Q-040'] only. Three new diagnostics queued 2026-04-28: **V3-EXQ-498** (OCD Layer 1 closure-threshold sweep on V3 monostrategy; SD-034 parameter diagnostic; 4-arm sweep × 3 seeds; PASS = at least one of {LOOSE, VERY_LOOSE} produces entropy(arm) > entropy(DEFAULT) + 0.1 in >= 2/3 seeds; FAIL rules out Layer 1 and licenses Layer 2 / Layer 3 escalation; ~60 min). **V3-EXQ-418f** (SD-016 attention-uniformity diagnostic probe; localises EXQ-418d/418e ln(16) uniform-rail bottleneck before committing to a fix; ~15 min). **V3-EXQ-418g** (SD-016 Path 4 query-selectivity-first 4-arm with new learnable temperature + attention-entropy loss substrate hooks landed 2026-04-28; B0_off / B1_sel_only / B2_div_only / B3_sel_plus_div × 3 seeds; ~90 min).

Pending review queue regenerated 2026-04-28T04:18:29Z carries **15 items** -- 12 PASS (V3-EXQ-484/485/493 across multiple machine/timestamp runs indexed after the 2026-04-27 run_id naming-bug fix) and 3 runner-only UNKNOWN entries for the same queue IDs awaiting next governance walk (the queue grew from 6 at the 2026-04-27T14:47:47Z regen because each per-machine/per-timestamp PASS now indexes as a distinct run; same three queue IDs underneath). The 2026-04-27T14:11 governance cycle walked 9 indexed pending + 4 runner-only and applied: SD-039 / MECH-292 / MECH-293 substrate-readiness PASS clusters preserved as `hold_pending_v3_substrate` pending behavioural validation; V3-EXQ-433d SD-029 / MECH-256 reclassified `non_contributory`; V3-EXQ-418e SD-016 keeps `does_not_support` on path-1 div_weight=0.5; V3-EXQ-490 MECH-269b/Q-040 (×2 runs) reclassified `non_contributory` (sub-diagnostic c1 gate-firing precondition FAILed -- resolved with V3-EXQ-490b smoke-threshold override).

**Current focus:** V3-EXQ-495 is the V3-full-completion gate; queueing and running it is a deliberate runtime-budget decision (~25h on Mac). V3-EXQ-490b is the smaller upstream factorial for Q-040a. EXQ-483 wired-but-inert pattern remains an open thread for the SD-037 / MECH-269b / MECH-295 cluster; V3-EXQ-484/485/493 have all cleared as substrate-readiness PASSes, validating SD-033a / SD-033b / MECH-295 substrate landings; behavioural recovery of approach_commit awaits the combined-cluster successor EXQ. Open promotion blockers documented in claims.yaml: MECH-294 within-cycle-vs-cross-cycle binding (Kay 2020 challenge); MECH-295 strong-vs-weak liking-bridge necessity (weak reading committed provisionally).

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
