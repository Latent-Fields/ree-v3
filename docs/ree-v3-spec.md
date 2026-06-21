# ree-v3 Repository Specification

**Created:** 2026-03-16
**Last updated:** 2026-06-21
**Status:** Living specification — launch doc updated with current V3 state
**Repo name:** `ree-v3`
**Governance epoch:** `ree_hybrid_guardrails_v1` (same as V2 — epoch is per-architecture not per-repo)
**Run ID suffix for governance:** `_v3`

---

## 0. Current V3 State (2026-06-20)

This section supersedes the original launch snapshot. Sections 7 (initial experiment queue),
10 (CLAUDE.md content), and 11 (Build Order) are historical — they document what was planned
at V3 launch, not current state. The authoritative session guide is `ree-v3/CLAUDE.md`.

### Substrate Implementation Status

| SD | Subject | Status |
|---|---|---|
| SD-004 | Action objects as hippocampal map backbone | Implemented |
| SD-005 | z_self / z_world latent split | Implemented |
| SD-006 | Asynchronous multi-rate loop (phase 1: time-multiplexed) | Implemented |
| SD-007 | Perspective-corrected z_world (ReafferencePredictor) | Implemented 2026-03-18 |
| SD-008 | alpha_world >= 0.9 in LatentStackConfig | Implemented (EXQ-040 validated) |
| SD-009 | Event-contrastive CE auxiliary loss for z_world encoder | Implemented (EXQ-020 PASS) |
| SD-010 | Harm stream separated as dedicated pathway (z_harm) | Implemented (EXQ-056c/058b/059c PASS) |
| SD-011 | Dual nociceptive streams: z_harm_s + z_harm_a | Implemented (2026-03-30; EXQ-178b PASS) |
| SD-011 second source | Affective harm history input (AffectiveHarmEncoder extended with FIFO) | Implemented 2026-04-08 |
| SD-012 | Homeostatic drive modulation for z_goal seeding | Implemented (2026-04-02) |
| SD-013 | E2_harm_s interventional training (counterfactual margin loss) | Implemented 2026-04-10 |
| SD-014 | Hippocampal valence vector node recording (4-component) | Implemented (2026-04-04) |
| SD-015 | Resource indicator encoder (ResourceEncoder, z_resource) | Implemented 2026-04-10 |
| SD-016 | Frontal cue-indexed integration (E1 z_world->ContextMemory query; cue_action_proj + cue_terrain_proj) | Implemented 2026-04-16 |
| SD-017 | Minimal sleep-phase infrastructure: SWS + REM passes | Implemented 2026-04-09 |
| SD-018 | Resource proximity supervision (aux head on z_world) | Implemented 2026-04-07 |
| SD-019 | Harm stream affective nonredundancy constraint | Implemented 2026-04-10 |
| SD-020 | Affective harm surprise prediction error (AIC analog) | Implemented 2026-04-10 |
| SD-021 | Descending pain modulation (pgACC->PAG->RVM, commitment-gated z_harm_s attenuation) | Implemented 2026-04-10 |
| SD-022 | Body directional limb damage (4-directional; C-fiber / A-delta independence) | Implemented 2026-04-09 |
| SD-023 | Environmental gradient texture (Landmark A/B fields in CausalGridWorldV2) | Implemented 2026-04-09 |
| ARC-028 + MECH-105 | HippocampalModule completion signal + BetaGate coupling | Implemented (2026-04-04) |
| ARC-033 | E2_harm_s forward model (ResidualHarmForward counterfactual pipeline) | Implemented 2026-04-09 |
| SD-011/SD-012 E3 Integration | z_harm_a urgency gating + affective amplification wired through full agent loop into E3 | Implemented (2026-04-05) |
| MECH-090 | Bistable beta gate (latch on commit entry; hippocampal completion as release) | Implemented 2026-04-10 |
| MECH-090 Layer 1 | Trajectory stepping: REEAgent steps through committed_trajectory.actions[idx] across E3 ticks (via _committed_step_idx counter) instead of repeating actions[0] | Implemented 2026-04-15 |
| MECH-091 Layer 2 | Urgency interrupt: when beta elevated and z_harm_a.norm() > urgency_interrupt_threshold (E3Config, default 0.8), gate releases and _committed_step_idx resets | Implemented 2026-04-15 |
| MECH-120 | SHY synaptic homeostasis wiring (enter_sws_mode -> shy_normalise) | Wired 2026-04-08 |
| MECH-203 + MECH-204 | Serotonergic sleep substrate (SerotoninModule, tonic_5ht, REM zero-point) | Implemented 2026-04-07 |
| MECH-205 | Surprise-gated replay write path (PE EMA -> VALENCE_SURPRISE, write count diagnostic) | Fixed 2026-04-09 |
| MECH-216 | E1 predictive wanting / schema readout head (schema_salience -> VALENCE_WANTING) | Implemented 2026-04-09 |
| SD-032b | dACC/aMCC-analog adaptive control + MECH-258 precision-weighted pain PE + MECH-260 bias suppression | Implemented 2026-04-19 |
| SD-032a | Salience-network coordinator (operating_mode soft vector + MECH-259 switch threshold + MECH-261 write-gate registry) | Implemented 2026-04-19 |
| SD-032c | AIC-analog interoceptive salience / urgency-interrupt (subsumes SD-021 descending modulation; harm_s_gain is drive-aware + mode-aware) | Implemented 2026-04-19 |
| SD-032d | PCC-analog metastability scalar (modulates MECH-259 effective_threshold by drive_level, success EMA, time-since-offline) | Implemented 2026-04-19 |
| SD-032e | pACC-analog autonomic coupling (slow-EMA drive_bias write-back from z_harm_a, MECH-094 hypothesis_tag gated) | Implemented 2026-04-19 |
| SD-033a | Lateral-PFC-analog (MECH-261 primary consumer; gate-modulated EMA rule_state + zeroed-last-Linear bias head) | Implemented 2026-04-20 |
| SD-034 | Governance closure operator (5-part "done" token: MECH-090 release + MECH-260 No-Go + residue discharge + closure_event + MECH-268 pe reset) | Implemented 2026-04-20 |
| MECH-267 | Mode-conditioned hippocampal proposals (operating_mode threads through HippocampalModule; per-mode CEM-noise multipliers) | Implemented 2026-04-20 |
| MECH-268 | dACC conflict saturation (outcome-history FIFO + f_sat attenuation; closure_event resets buffer) | Implemented 2026-04-21 |
| MECH-266 | Asymmetric per-mode hysteresis on SalienceCoordinator (Schmitt-trigger per-mode enter/exit rails over MECH-259 symmetric threshold; empty-dict default preserves legacy) | Implemented 2026-04-21 |
| SD-029 | Curriculum-level balanced hazard-event support (CausalGridWorldV2 scheduled_external_hazard_* injection; enables per-seed n_ext >= 20 for C3/C4 comparator tests) | Implemented 2026-04-21 |
| SD-035 | Amygdala analogue -- BLA + CeA peer modules: MECH-046 CeA mode-prior + MECH-074a/b/c/d BLA encoding gain, retrieval bias, fast prime, PE-spike remap (non-trainable arithmetic; hippocampal consumer wiring for BLA retrieval/remap deferred) | Implemented 2026-04-21 |
| SD-033e stub | Frontopolar-analog V4-reserved stub in ree_core/pfc/ (FrontopolarConfig + FrontopolarAnalog skeleton with MECH-264 counterfactual-value + MECH-265 relative-importance heads; raises NotImplementedError when enabled until design doc lands) | Stub landed 2026-04-21 |
| SD-036 | GABAergic cross-stream decay regulator (broadly-projecting tonic decay across registered latent streams; out-of-place exp decay; benzo/withdrawal gaba_tone knob; simulation_mode gated) | Implemented 2026-04-22 |
| MECH-279 | PAG freeze-gate (committed-freeze substrate; duration*magnitude entry, exit threshold = theta * gaba_tone, action-class no-op injection; simulation-gated) | Implemented 2026-04-22 |
| MECH-269 base (Phase 1) | Per-stream verisimilitude V_s (HippocampalModule.update_per_stream_vs; identity-proxy EMA over registered streams; foundation for Phase 2/3 invalidation runtime) | Implemented 2026-04-22 |
| MECH-288 | Event segmenter Phase 2 (two-scale: pe_threshold fast + BOCPD-Gaussian slow; emits BoundaryEvents with outer.inner segment IDs; force_boundary API for scripted injection) | Implemented 2026-04-22 |
| MECH-287 | Invalidation trigger Phase 2 iv (BoundaryEvent subscriber -> BroadcastEvent emitter; graded posterior*gain; phasic/tonic guardrail over rolling window; verdict-3 option-c subscriber collapse) | Implemented 2026-04-22 |
| MECH-269 AnchorSet (Phase 2 ii) | Scale-tagged hippocampal anchor store with dual-trace preservation (Bouton 2004) + k=5 consecutive-below hysteresis on V_s_anchor; BoundaryEvent consumer; FIFO soft-cap per scale | Implemented 2026-04-22 |
| MECH-269 per-region V_s (Phase 2 iii T4) | Per-region V_s readout keyed on active AnchorSet regions (scale, segment_id); MECH-287 broadcast reset path (drop + mark_inactive); dual consumption via tick_anchor_set + apply_invalidation_broadcasts_to_regions | Implemented 2026-04-22 |
| MECH-284 (Phase 3) | V_s residual schema-staleness accumulator (region-indexed (scale, segment_id); MECH-287 broadcast integration with attribution_mode equal/stream_overlap; per-tick leak; staleness_clip; lookup_by_anchor_key getter) | Implemented 2026-04-24 |
| MECH-269 online hysteresis swap | AnchorSet.tick_hysteresis accepts optional staleness_lookup -> V_s_anchor = V_s(r) - staleness[r] (orthogonal flag use_mech284_hysteresis; default OFF preserves Phase 2 internal proxy) | Implemented 2026-04-24 |
| MECH-290 | Backward trajectory credit sweep (Foster & Wilson 2006 reverse replay): record_committed_trajectory at BetaGate elevation; backward_credit_sweep at completion-signal release; credit = outcome_quality * gamma^(T-1-t) -> ResidueField.update_valence(VALENCE_WANTING); reset on episode boundary | Implemented 2026-04-24 |
| SD-037 | Broadcast override regulator (orexin-analog): override_signal in [0,1] driven by SD-012 drive_level + sustained-threat rolling-window over z_harm; consumed by PAG freeze-gate (exit-threshold scaling), SalienceCoordinator (external_task affinity), GoalState (z_goal-seeding amplification); biological-defaults via orexin-kinetics lit-pull | Implemented 2026-04-25 |
| Sleep Phase A | ree_core/sleep/ scaffold: SleepPhase enum (6 phases) + SleepCycleState + SleepLoopManager wrapping SD-017 surface; master flag use_sleep_loop + sleep_loop_episodes_K + sleep_loop_require_passes; manager.notify_episode_end() at REEAgent.reset() before per-episode resets | Implemented 2026-04-25 |
| MECH-285 (Sleep Phase B) | SleepReplaySampler: at SLEEP_ENTRY freezes StalenessAccumulator.snapshot(), draws N seeds from AnchorSet.all_with_dual_trace() with softmax(staleness/temperature) priority; uniform-fallback when no accumulator; AnchorSet.all_with_dual_trace() alias added | Implemented 2026-04-25 |
| MECH-272 (Sleep Phase C) | RoutingGate: state-conditioned channel weights {anchor_channel, probe_channel} flipping across SWS_ANALOG / REM_ANALOG / WAKING per design-doc table; weights set at SLEEP_ENTRY (SWS) and PHASE_SWITCH (REM); routed counts surfaced as mech272_* diagnostics | Implemented 2026-04-25 |
| MECH-275 (Sleep Phase D) | BayesianAggregator: per-domain per-region conjugate Gaussian posteriors; probe-channel-gated update on each routed draw; snapshot+decay contract (snapshot fires at PHASE_SWITCH BEFORE REM phase set so SWS-only posteriors are captured for Phase E consumption); place-domain default with (scale, segment_id) region key matching MECH-284 | Implemented 2026-04-25 |
| MECH-273 (Sleep Phase E) | SelfModelAggregator (subclass of MECH-275 specialised on SD-003 causal_sig posterior); offline_gradient_pass on E2_harm_s reading SWS-only snapshot; StalenessAccumulator.partial_decay(replayed_regions) for region-scoped decay; full sleep cluster contracts at 150/150 (143 contracts + 7 preflight) all PASS with all flags OFF | Implemented 2026-04-25 |
| SD-016 Path 1 | ContextMemory.compute_diversification_loss (mean squared off-diagonal cosine similarity over normalized slot vectors) wired into REEAgent.compute_prediction_loss with sd016_diversification_weight config; explicit gradient pressure for slot diversification after EXQ-418d FAILed across all 4 write-path arms (read/write gradients alone cannot break ContextMemory slot symmetry) | Implemented 2026-04-25 |
| SD-033b | OFC-analog (MECH-261 second consumer; gate-modulated EMA state_code [1, state_dim] with eff_eta = update_eta * write_gate("sd_033b"); zeroed-last-Linear bias head -> initial bias exactly zero; per-mode gate weights external_task=1.0 / internal_planning=0.5 / internal_replay=0.05 / offline_consolidation=0.3; MECH-263 functional signatures via env-extension EXQs deferred) | Implemented 2026-04-26 |
| MECH-269b | Symmetric V_s gating on E1/E2 cortical rollouts (read-side consumer of MECH-269 Phase 1 per_stream_vs at the E1 _e1_tick site and the per-tick E2_harm_a forward call site; held substitution swaps current latent for snapshot when V_s[s] < per-side threshold; 0.4-0.5 dead-band lightweight Schmitt-trigger hysteresis; precondition use_per_stream_vs=True) | Implemented 2026-04-26 |
| MECH-295 weak-reading bridge | drive -> liking-stream -> approach_cue substrate (anticipatory liking-stream pulse at z_goal location via update_z_goal -> ResidueField.update_valence VALENCE_LIKING; per-candidate negative score_bias via select_action; severed-bridge falsification arm at cue gain=0; weak-necessity reading committed provisionally per the lit-pull synthesis) | Implemented 2026-04-26 |
| SD-039 substrate | Dual-trace anchor goal-snapshot payload (AnchorGoalPayload dataclass: z_goal_snapshot + wanting_strength + arousal_tag + last_vs + staleness_at_write + payload_written_step; refresh-on-invalidate semantic preserves payload across mark_inactive; Anchor.goal_match cosine helper + AnchorSet.query_by_goal_match active+inactive dual-trace getter for MECH-292 consumer) | Substrate landed 2026-04-26 |
| SD-039 population layer | Module-level write-site population: REEAgent.sense() builds AnchorGoalPayload once per tick from GoalState (z_goal_snapshot), ResidueField VALENCE_WANTING (wanting_strength), BLA arousal_tag, mean(per_stream_vs) (last_vs), max staleness (staleness_at_write); threaded into HippocampalModule.tick_anchor_set + apply_invalidation_broadcasts_to_regions; MECH-094 simulation-mode gate at build_goal_payload | Implemented 2026-04-27 (V3-EXQ-494 6/6 PASS) |
| MECH-292 | Ranked ghost-goal bank (pure-arithmetic derived view over SD-039 dual-trace anchor pool; ghost_priority = w_w*wanting + w_m*goal_match + w_s*staleness + w_r*recoverability; goal_match_floor=0.05 rumination guard; consumer of SD-039 payloads via Anchor.goal_match; raises ValueError if use_anchor_sets / use_sd039_anchor_payload off) | Implemented 2026-04-27 (V3-EXQ-496 5/5 PASS) |
| MECH-293 | Waking ghost-goal probe search (HippocampalModule.propose_trajectories minority-budget ghost branch consuming MECH-292 ranked bank; mech293_ghost_fraction=0.2 default; Trajectory.hypothesis_tag + metadata stripped at record_committed_trajectory; current_z_goal threaded from REEAgent._e3_tick; ARC-007-strict preserved — no value head) | Implemented 2026-04-27 (V3-EXQ-497 5/5 PASS) |
| ARC-054 V3 form | D_V trajectory selection promoted v4 -> v3 in synaptic-EMA form (rollout-horizon synaptic EMA over V_s readout; no TCL substrate dependency at V3); V4 form (phase-coherent V(t) integration via ARC-053 + MECH-225/226/228) remains v4-by-design; V3-EXQ-491 validation queued | V3 form promoted 2026-04-26 |
| MECH-271 V3 substrate plan | Hypothesis tag as downstream routing committed for V3 in synaptic form (discrete routing table + audit hook for confabulation-vs-psychosis dissociation); V4 ephaptic-field-strength routing remains v4-by-design; V3-EXQ-492 routing 4-arm queued behind MECH-269b lock release | Plan committed 2026-04-26 |
| SD-047 | Multi-source environmental dynamics (3 concurrent stochastic sources at distinct scales: AR(1) weather field + Poisson transient hazards + mobile drift sources; 4-arm noise-sweep lever; substrate-ceiling unblock for MECH-095 TPJ agency-detection comparator; bit-identical OFF + per-source ablation; activation smoke ARM_2 calibration ratio 1.95:1 within 1:1-2:1 target band; lit-anchored sd_047 lit_conf=0.841 across 18 PubMed entries) | Implemented 2026-05-03 (V3-EXQ-509 PASS 7/7); SD-047 candidate -> provisional, v3_pending removed |
| SD-048 | Body interoceptive noise dynamics (3 concurrent agent-independent body-state noise sources on harm_obs_a readout: autonomic Gaussian + sensitisation Poisson + fatigue AR(1); Level 2 counterpart to SD-047 at body-state layer; substrate-ceiling unblock for ARC-058 / ARC-033 arbitration; bit-identical OFF + per-source ablation) | Implemented 2026-05-03 (V3-EXQ-511 6/7 with C1b sub-threshold at scale=0.25; per SD-doc interpretation grid the partial-PASS is calibration-off-but-architecture-holds; governance reclassified non_contributory; SD-048 stays candidate / v3_pending pending V3-EXQ-512 ARC-058 behavioural successor) |
| SD-049 Phase 1 | Multi-resource heterogeneity environment (3 qualitatively distinct resource types incl non-homeostatic novelty channel + per-axis homeostatic drive vector parallel to legacy agent_energy + curriculum-introduction hook keyed on cross-episode _global_step; world_obs_dim 250 -> 325 default 3-type; benefit profiles sigmoidal_saturating / sharp_saturation / novelty_decay; substrate-roadmap H-priority #2; lit-anchored sd_049 lit_conf=0.898 across 5 PubMed entries) | Implemented 2026-05-03 (V3-EXQ-513 PASS 13/13 incl curriculum gates); v3_pending stays true pending Phase 2 behavioural validation |
| SD-049 Phase 2 | Hybrid identity-aware z_resource encoder (Option C from 2026-05-04 lit-pull verdict, lit_conf 0.78; shared trunk MLP -> 32-dim z_resource + identity-classifier head Linear(z_resource_dim, n_resource_types) supervised by cross-entropy on env-emitted obs_dict["sd049_consumed_type_tag_this_tick"]; biology anchored to Ballesta-Padoa-Schioppa 2019 OFC labeled-line + Quiroga 2005 sparse readouts + Schapiro 2017 hybrid CLS; phased training P0 joint -> P1 freeze head -> P2 evaluate; identity_logits exposed as separate LatentState field so z_resource shape preserved at 32 for GoalState seeding; backward-compat default OFF) | Implemented 2026-05-04 (substrate-side; phased-training behavioural validation V3-EXQ-514 family ongoing -- 514a/c/f/g status mixed; SD-049-PHASE-3 SD-032 consumer cascade migration deferred per goal_pipeline_plan.md GAP-5) |
| SD-050 / MECH-302 | Suffering-derivative comparator (z_harm_a downward-derivative threshold-crossing fires same downstream pipeline as goal-completion: MECH-057a beta-gate release + MECH-094 categorical VALENCE_LIKING tag write; polarity set at input, not parallel circuitry; non-trainable; sense() ticks comparator + select_action() consumes event flag; bit-identical OFF; MECH-094 simulation_mode gate) | Implemented 2026-05-04 (V3-EXQ-515 PASS 7/7 comparator logic; V3-EXQ-516 agent-loop integration diagnostic queued); MECH-302 / MECH-303 registered candidate / v3_pending |
| SD-019a | harm_unpleasantness_channel (third-tier z_harm_un EMA between fast z_harm_s and slow z_harm_a; 5-step rise at alpha=0.2; AIC + E3 short-horizon urgency redirect when use_harm_un; SD-021 descending modulation deliberately does NOT attenuate z_harm_un so controllability parity matches Loffler 2018; non-trainable EMA; bit-identical OFF) | Implemented 2026-05-04 (V3-EXQ-518 dry-run PASS; queued as 4-arm diagnostic with 9 acceptance criteria UC0a-b/UC1a-d/UC2a-b/UC3) |
| SD-051 / MECH-304 | Conditioned safety store (cue-specific predictive substrate: ConditionedSafetyStore non-trainable EMA prototype of z_world at MECH-302 event ticks + per-step decay forgetting; cosine similarity -> sigmoid -> commitment-release gate when beta elevated; pure arithmetic; new ree_core/safety/ package; V4-deferred items: approach attractor + contrastive cue-specific learning) | Implemented 2026-05-04 (V3-EXQ-519 substrate-readiness queued; integration C6 surfaced upstream MECH-302 event-source dependency now under test in V3-EXQ-517a/b) |
| SD-052 / MECH-303 | Contextual passive safety terrain (slow vmPFC/hippocampal-analog: ResidueField extended with safety_terrain_rbf_field + accumulate_safety + evaluate_safety; per-step accumulation when z_harm_a.norm() < harm_threshold AND not hypothesis_tag; commitment release when mean evaluate_safety >= release_threshold; same RBF pattern as ARC-030/MECH-117 benefit_terrain but separate field; bit-identical OFF; MECH-094 simulation gate) | Implemented 2026-05-04 (V3-EXQ-520 4-arm substrate-readiness diagnostic dry-run PASS 10/10) |
| SD-054 reef enrichment substrate | Reef enrichment / monostrategy-breaking behavioral diversity substrate in CausalGridWorldV2: corner-adjacent Manhattan-radius reef safe zones (hazards excluded; 5x5 reef_field_view scent gradient appended to world_state, world_obs_dim 250->275) + food-attracted hazard drift bias (probability hazard_food_attraction). Two behavioral attractors -- "flee to reef" vs "forage" -- to break the single fixed route. (Renamed from SD-050 to SD-054 on 2026-05-08 to disambiguate from the suffering-derivative comparator; substrate-readiness vs substrate-purpose-validation distinction added the same day -- trained-agent retest table V3-EXQ-433e/f / 523/a/b reclassified non_contributory because monomodal policy cannot exercise the substrate-purpose acceptance criteria; SD-054 v3_pending flipped to true pending the rule-apprehension cluster MECH-309 / ARC-062 / ARC-063 substrate landings.) | Substrate implemented 2026-05-04 (V3-EXQ-521 substrate-readiness PASS 7/7; V3-EXQ-522 monostrategy-breaking PASS zone_transitions=48.9); SD-054 + MECH-309 candidate / v3_pending registered 2026-05-08 alongside ARC-062 (V3 weak rule-apprehension reading) and ARC-063 (V4 strong rule-apprehension reading) per docs/architecture/rule_apprehension_layer.md |
| MECH-204 sleep substrate Phase 1 | Precision recalibration consumer (sleep_substrate_plan.md GAP-1): SerotoninModule.compute_recalibration_target() returns the captured precision_at_rem_entry zero-point reference (returns 0.0 when disabled / no REM entered). E3TrajectorySelector.recalibrate_precision_to(target, step) applies Option A linear interpolation new_rv = (1-step)*rv + step*(1.0/(target+1e-6)). WRITEBACK-phase sibling step in SleepLoopManager._run_cycle runs independently of MECH-273 self-model gradient; gated on use_rem_precision_recalibration AND rem_enabled AND serotonin.enabled. Cycle metrics emit mech204_recalibration_fired / target / before / after / step. New REEConfig fields use_rem_precision_recalibration (default False, bit-identical OFF) + rem_precision_recalibration_step (default 0.1 per plan-of-record Q1). Companion EXP-0171 step-size sweep gated on V3-EXQ-541 PASS. | Implemented 2026-05-08 (contract suite test_mech204_precision_recalibration.py 9/9 PASS covering C1 surface, C2 default-OFF, C3 sleep_loop-ON / recalibration-OFF no-metrics, C4 arithmetic, C5/C6 zero-target / zero-step no-op, C7 capture-only regression guard, C8 WRITEBACK firing end-to-end, C9 drift movement; V3-EXQ-541 validation FAIL 2026-05-08T23:43Z verdict at runner level pending governance review -- result_summary records "verdict: PASS" with FAIL outcome flag; manifest at evidence/experiments/v3_exq_541_mech204_precision_recalibration_consumer_20260508T234302Z_v3.json) |
| MECH-307 anticipatory affect conjunction architecture | First-line conjunction-fix proposal vs SD-014 6-channel amendment fallback (~40 lines of code, NOT a new VALENCE channel): excitement and dread emerge as derived states from a four-gap fix to existing channels (signed VALENCE_SURPRISE + MECH-216 z_beta coupling + anticipatory VALENCE_LIKING write + write-at-predicted-location). Lit anchors from targeted_review_excitement_5th_valence_channel (lit_conf 0.77, 9 entries). Cross-tags MECH-111 because the same wiring gaps likely substrate-confound the EXQ-141 curiosity-drive failure. Falsifiable 4-arm experiment in docs/architecture/anticipatory_affect_conjunction_vs_dual_channel.md. Consumer-side Path B extension: MECH295LikingBridge.compute_conjunction_score_bias() reads SD-014 valence + z_beta arousal at per-candidate predicted-imminent locations and applies a negative approach bias when the four-way conjunction holds; new flag use_mech307_consumer_conjunction_read (default False; bit-identical OFF). | Registered candidate / v3_pending 2026-05-08; consumer-conjunction bridge extension landed 2026-05-08 (test_mech307_consumer_conjunction.py 8/8 PASS); V3-EXQ-539 4-arm substrate-readiness FAIL governance-applied hold_pending_v3_substrate; V3-EXQ-540 3-arm gap decomposition queued; **Option-b Gap-1 substrate landed 2026-05-11** per user override (split VALENCE_POSITIVE_SURPRISE + VALENCE_NEGATIVE_SURPRISE channels, VALENCE_DIM 4 -> 6; legacy unsigned-magnitude write to VALENCE_SURPRISE preserved for backward compat; convenience master `use_mech307_conjunction` propagates split + multichannel + predicted-location subflags; SD-014 6-channel amendment retained as registered fallback). V3-EXQ-540a behavioural validation FAIL 2026-05-11 (C1 substrate dissociation PASS across all arms; C2 consumer-conjunction read FAIL -- bridge predicate never fired even with all substrate counters populated). V3-EXQ-540b consumer-side threshold sweep FAILed 2026-05-12 (conj_fire_rate=0 across all four arms including 0.01 floor); V3-EXQ-540c MECH-307 read-site probe ERRORed (SIGTERM during 2026-05-12T06:10Z runner-restart); V3-EXQ-540d re-queue PASS 2026-05-12T06:29Z diagnosed the gap (drive_level max 0.030 / mean 0.016 never crossed 0.1 floor at 1087 bridge calls / 34784 candidate reads; z_beta_arousal max 0.545 below 0.6 floor; conjunction predicate would have fired on 94.66% of reads at half-tier thresholds). **MECH-307 default-value recalibration landed 2026-05-12**: `mech295_min_drive_to_fire` 0.1 -> 0.01 and `mech307_conjunction_z_beta_threshold` 0.6 -> 0.3 across the 4+3 declaration sites (bridge dataclass + REEConfig + REEConfig.from_dims + REEAgent getattr fallback). 314/314 contracts + 7/7 preflight PASS with new defaults. V3-EXQ-540e (3-arm decomposition under the new defaults; DLAPTOP-4.local, 90 min, supersedes 540d) queued same session; dry-run smoke 2026-05-12T06:39Z PASS at 6 ep / 1 seed with ARM_2_full `conj_fire_rate=0.155` (cleared the 0.10 floor for the first time since substrate landed); ARM_0_off and ARM_1_split_only correctly silent. PASS on 540e clears goal_pipeline:GAP-1 v3_pending behavioural-validation gate and unblocks GAP-2 SD-049 Phase 2 V3-EXQ-514 behavioural validation. Deferred follow-on (separate session): Option-b semantic fix at `mech295_liking_bridge.py:343` (currently reads `v[:, 3]` legacy unsigned-magnitude rather than `v[:, 4]` VALENCE_POSITIVE_SURPRISE under Option-b semantics; design-doc fidelity bug, not a behavioural blocker) |
| ARC-062 Phase 1 (gated_policy_heads) | rule_apprehension.gated_policy_heads -- V3 weak-reading instantiation of the rule-apprehension architectural slot identified by MECH-309 (logical-necessity claim: monomodal collapse without a non-Bayesian rule-creator at the policy layer). Module ree_core/policy/gated_policy.py (GatedPolicy + GatedPolicyConfig + GatedPolicyOutput): N=2 scoring heads sharing E3 candidate features (symmetry-broken init on heads' last-Linear bias so heads differentiate from step 0 under any training pressure) + 3-stream context discriminator on (z_world, z_self, z_harm_a) per Pull A SYNTHESIS R1 verdict (Miller & Cohen 2001 + Rigotti 2013 + Mitchell 2016 macaque MD insular cluster) at score_bias level per R3 verdict. disc_init_scale=0.1 keeps sigmoid output near 0.5 at init; bias clamped to [-bias_scale, +bias_scale]. n_heads=2 substrate-constrained per Pull A R2 verdict; n_heads != 2 raises ValueError. NO connection to SD-033a in Phase 1 -- that wiring is Phase 3 (closes commitment_closure_plan.md GAP-1). REEAgent.select_action composes gated_policy_score_bias additively into dacc_score_bias before MECH-295 block. MECH-094 simulation_mode=True returns zeros and increments only skip counter. Plan-of-record: REE_assembly/evidence/planning/arc_062_rule_apprehension_plan.md (GAP-A done; GAP-B Phase 2 monomodal-collapse falsifier on SD-054 reef + hazard_food_attraction substrate queued). | Implemented 2026-05-09 (V3-EXQ-542 5/5 PASS UC1-UC5 substrate readiness on Mac 2026-05-09T20:22Z; runner outcome flag ERROR with manifest verdict PASS per the substrate-readiness pattern). Phase 2 GAP-B falsifier V3-EXQ-543 PASS 2026-05-09T21:45Z on Mac (3 seeds x 2 arms; ARM_0 use_gated_policy=False vs ARM_1c use_gated_policy=True with full 3-stream discriminator at ARM_1_med density on SD-054 reef). 5/5 contract tests test_gated_policy.py (C1 default-off no-op / C2 backward-compat flag-on / C3 discriminator output in [0,1] across 64 diverse latents with bias_scale clamp / C4 heads' OUTPUTS diverge >5x under anti-symmetric SGD / C5 simulation_mode skip counter). Full ree-v3 suite 249/249 PASS (244 prior + 5 new). Phase 3 GAP-C optimizer-side wiring (gated_policy bias-head into E3 optimizer + discriminator output threaded into SD-033a `LateralPFCAnalog.update()` source vector) attempted in V3-EXQ-543b (2026-05-10 Phase 3 falsifier on Mac, 14407s) but FAILED -- closes commitment_closure_plan.md GAP-1 remains blocked pending Phase 3 design refinement. |
| MECH-313 (ARC-065 child) | policy.stochastic_noise_floor_lc_ne_tonic_analog -- pure-arithmetic regulator (sibling to MECH-314 / MECH-318 / MECH-319) instantiating the LC-NE tonic complement to MECH-104 phasic spike. Single primitive `noise_floor.compute_effective_temperature(baseline_temperature, simulation_mode)` returns `max(baseline_T + noise_floor_alpha, noise_floor_min_temperature)`; SAC-entropy-bonus analog (Haarnoja 2018) on E3 softmax temperature. Distinct from MECH-260 dACC anti-recency (state-dependent); Q-045 falsifies whether they collapse. Phase-1 instantiation choice = SEPARATE module at the e3.select() call site rather than per-head temperature inside GatedPolicy (revisit at Q-045 4-arm ablation). MECH-094: simulation_mode=True returns baseline temperature unchanged + increments skip counter only. | Implemented 2026-05-10 (V3-EXQ-544 substrate-readiness 5/5 PASS UC1-UC5 smoke; runner outcome flag ERROR with manifest verdict PASS per the substrate-readiness false-ERROR stdout pattern fixed mid-day; 11 contract tests in tests/contracts/test_mech_313_noise_floor.py PASS). Behavioural validation deferred to Q-045 4-arm ablation AFTER MECH-314 also lands. |
| MECH-314 (ARC-065 child) + MECH-314a/b/c sub-flavours | policy.structured_curiosity_bonus_parent + 3 sub-flavours (314a striatal novelty per-candidate min-distance from candidate's first-step z_world to nearest active ResidueField RBF center; 314b frontopolar uncertainty broadcast scalar from e3._running_variance; 314c learning progress EMA of `|PE_t - PE_{t-K}|`). Pure-arithmetic, no learned parameters; sibling to MECH-313 NoiseFloor in the ree_core.policy package. Composed AFTER MECH-295 liking-bridge block and BEFORE MECH-313 noise_floor temperature lift (curiosity affects scores; noise floor affects temperature; orthogonal). Per Pull 1 R3 verdict the three sub-flavours are independently togglable so Q-044's three-arm ablation is a flag-set decision. Phase 1 honest-scoping caveat: 314a is genuinely per-candidate; 314b and 314c are state-dependent global scalars broadcast across [K]; per-candidate refinement deferred to Phase 2 follow-on. MECH-094: simulation_mode=True returns zeros[K] + increments skip counter only. | Implemented 2026-05-10 (V3-EXQ-545 substrate-readiness 5/5 PASS UC1-UC5 smoke; ran twice on Mac + cloud-2 via multi-machine race; 13 contract tests in tests/contracts/test_mech_314_curiosity.py PASS; 273/273 contracts + 7/7 preflight PASS with master OFF). Behavioural validation deferred to Q-044 three-arm ablation AFTER MECH-318 / MECH-319 sibling absorption-check sessions complete. |
| MECH-319 (arc_062 GAP-K) | policy.arbitration.simulation_mode_write_gating_substrate -- substrate-level instantiation of MECH-094 at the rule-arbitration layer per Pull 3 SYNTHESIS R1 GENUINE-NOVELTY-CONFIRMED + Pull 4 R3 KEEP-AS-IS verdicts. Pure-arithmetic regulator (sibling to GABAergicDecayRegulator and BroadcastOverrideRegulator) in the regulators package. Single primitive `gate.effective_simulation_mode(simulation_mode, site) -> bool` with truth-table semantics: master OFF identity; master ON + admit_writes=False blocks sim writes; master ON + admit_writes=True (V3-EXQ-543c-successor falsifier control) admits sim writes. Two existing arbitration-write call sites in REEAgent.select_action() consult the gate when instantiated: GatedPolicy block (replace literal simulation_mode=False) + LateralPFCAnalog block (skip update() when blocked, compute_bias still runs). MECH-094 NOT modified per KEEP-AS-IS verdict. Construction raises ValueError on admit_writes=True without master ON (loud-not-silent guard). Per-site diagnostic counters on {gated_policy, lateral_pfc, default}. | Implemented 2026-05-10 (V3-EXQ-546 substrate-readiness 6/6 PASS UC1-UC5 + UC3b precondition; ran twice on Mac + cloud-2 via multi-machine race; 15 contract tests in tests/contracts/test_mech_319_simulation_mode_rule_gate.py PASS; 288/288 contract + preflight tests PASS with master OFF). claims.yaml MECH-319 candidate -> candidate_substrate_landed; v3_pending: true retained pending V3-EXQ-543c-successor falsifier with admit_writes=True. arc_062 GAP-K closed (registered -> substrate_landed). |
| MECH-320 (ARC-066 child) | action.tonic_vigor_coupling_score_bias -- first child mechanism for ARC-066 (the non_deficit_action_drives architectural family). Pure-arithmetic regulator (sister to MECH-313 NoiseFloor + MECH-314 StructuredCuriosity in ree_core.policy). Composed AFTER MECH-314 curiosity (orthogonal axis: curiosity rewards novelty/uncertainty/LP at candidate level; vigor biases on action-vs-no-op axis) and BEFORE MECH-313 noise_floor. Algorithm: `v_t = max(0, slow EWMA over realised E3 score) * gate_energy * gate_drive * gate_pe`; `bias[i] = -w_action*v_t` on action classes / `+w_passive*v_t` on noop class (additive primary; multiplicative gain falsifiable secondary via tonic_vigor_form="multiplicative"). TARGET-FREE: bias applies regardless of whether any z_goal is currently active, closing the "well-fed-safe-familiar agent has no positive gradient to act" gap that ARC-066 registered. Defaults: half_life=100 (long-window per R4), w_action=w_passive=0.1, bias_scale=0.1. MECH-094: simulation_mode=True on either compute_score_bias or update_score_receipt returns zeros + increments skip counter only. | Implemented 2026-05-10 (V3-EXQ-547 substrate-readiness 6/6 PASS UC1-UC6 on cloud-2 2026-05-10T20:56Z; 28 contract tests in tests/contracts/test_mech_320_tonic_vigor.py PASS; 309/309 contracts + 7/7 preflight PASS with master OFF). claims.yaml MECH-320 candidate -> candidate_substrate_landed. Behavioural validation 3-arm discriminative pair (baseline / additive / multiplicative on a well-fed-safe-familiar substrate) deferred to a separate /queue-experiment session. |
| ARC-066 / ARC-067 / ARC-068 cluster (non_deficit_action_drives family) | Three architectural-slot claims registered 2026-05-10: ARC-066 tonic_vigor_coupling (capacity -> action bias); ARC-067 idle_aversion_boredom (sustained low-engagement is aversive); ARC-068 opportunity_cost_no_op_penalty (waiting carries cost). Family principle: behaviour comes from surplus capacity AND from deficits, not deficits alone. ARC-066 lit-pull lit_conf 0.789 supports (LC-NE substrate REJECTED, mesolimbic DA-vigor LOAD-BEARING per Niv 2007 + Salamone & Correa 2012 + Beierholm 2013). ARC-067 lit-pull lit_conf 0.85 supports. ARC-068 lit-pull lit_conf 0.806 supports-direction-dominant (R1 SEPARATE-AT-ARCHITECTURE-VIA-KERNEL not via substrate -- ARC-068 anchors on long-window historical EMA, SD-032b on current-environmental scalar; R3 ARC-066 + ARC-068 collapse LICENSED at implementation layer per Niv 2007 mathematical symmetry but slot-level separation preserved for psychiatric failure-mode dissociation). | Cluster registered 2026-05-10 (3 architectural-slot claims + umbrella architecture doc REE_assembly/docs/architecture/non_deficit_action_drives.md + 3 lit-pulls landing under targeted_review_arc_066_tonic_vigor / targeted_review_arc_067_boredom / targeted_review_arc_068_opportunity_cost). MECH-320 (ARC-066 first child mechanism) substrate landed same day. ARC-067 / ARC-068 child-MECH design completed 2026-05-16: two-child split for ARC-067 (MECH-330 idle_aversion_acute_restlessness_accumulator + MECH-331 idle_aversion_chronic_anhedonic_flatness_substrate); ARC-068 collapses into MECH-320 per ARC-068 lit-pull R3 verdict (Niv 2007 mathematical symmetry -- MECH-320 w_passive term IS the ARC-068 implementation). Biology-before-formal-definitions gate now fully clear for the family. |
| ARC-069 / ARC-070 / ARC-071 cluster (policy_primitive_granularity family) | Three architectural-slot claims registered 2026-05-10: ARC-069 parent (policy_hierarchy_dynamic_regranularisation -- the unit of policy operated on is itself dynamic, not fixed); ARC-070 decomposition-on-prediction-failure (zoom in / re-segment when an imagined chunk fails to ground); ARC-071 composition-via-repeated-grounding (zoom out / chunking when a sequence has been grounded enough times to be treated atomically). ARC-070 lit-pull lit_conf 0.88 supports (R2 LOAD-BEARING SHARED SUBSTRATE -- ARC-070 implemented as bidirectional extension of MECH-288 event_segmenter, not a new module; Schacter 2008 constructive-episodic-simulation core network supplies the empirical anchor). ARC-071 lit-pull lit_conf 0.848 supports (R3 LOAD-BEARING -- CONFIRMED ARC-071 IS the missing transition mechanism in MECH-163 dual_goal_directed_systems, MECH-163 depends_on extended +ARC-071 the same day; R6 SAFETY-CRITICAL escalation -- biology does NOT cleanly gate chunking write path against replay/imagined sequences per Albouy 2013, ARC-071's pre-registered MECH-094 hypothesis_tag=False strict-gating MORE CONSERVATIVE than biology, governance decision pending). | Cluster registered 2026-05-10 (3 architectural-slot claims + umbrella architecture doc REE_assembly/docs/architecture/policy_primitive_granularity.md + 2 lit-pulls under targeted_review_arc_070_decomposition / targeted_review_arc_071_composition). ARC-070 R2 reframe landed (MECH-288 cross-reference rewritten from observation-side analog to bidirectional consumer; event_segmenter.py needs input_stream label for MECH-094 hypothesis_tag-conditional dispatch). MECH-321 (ARC-070 first child mechanism, policy.decomposition_via_event_segmenter) registered candidate / v3_pending the same day with R1-R5 verdicts folded into functional_restatement; depends_on ARC-070 + MECH-288 + MECH-269 + MECH-094. ARC-071 child-MECH design deferred until R6 governance decision lands. |
| ARC-062 GAP-B/C/D | rule_apprehension wiring continuation: GAP-B head-input first-action one-hot augmentation (bypasses E2 world-forward compression that flattened SP-CEM first-action diversity to 0.22% of z_world before the z_world-only heads -- gated_policy_use_first_action_onehot, default OFF); GAP-C discriminator-output -> SD-033a rule_state source vector (lateral_pfc_use_discriminator_source, default OFF) with agent.py gated_policy block reordered before lateral_pfc so gating_weight is available; GAP-D SD-033a rule_bias_head made optionally trainable (lateral_pfc_train_rule_bias_head, default OFF -- last Linear no longer zeroed when True). All three bit-identical OFF; 484/484 contracts PASS. | Implemented 2026-05-17 (ree-v3 15ca95e); validation EXQ deferred until V3-EXQ-543f/h returns a contributory ARC-062 result |
| ARC-065 SP-CEM main-path default | Support-preserving + stratified CEM ("SP-CEM") flipped to the main-agent action-path default (6 HippocampalConfig + REEConfig.from_dims defaults: use_support_preserving_cem False->True, support_preserving_stratified_elites False->True, support_preserving_ao_std_floor 0.0->0.2). INTENTIONAL non-no-op default change (the one deliberate departure from the implement-substrate no-op rule) -- the legacy collapsing CEM produced the monostrategy that left SD-029 / ARC-062 Rung 2 / goal_pipeline GAP-2/4 / self_attribution GAP-1/2/3 non_contributory. Bit-identical legacy opt-out by explicitly pinning the three flags. Evidence: V3-EXQ-567 PASS 2026-05-15 (selected_action_entropy 0.0124 -> 0.4965). | Main-path default landed 2026-05-17; V3-EXQ-583 3-arm default-wiring equivalence PASS 2026-05-17T09:25Z (ARM_default == ARM_explicit_on within 1e-9, both >> ARM_explicit_off); claims.yaml ARC-065 implementation_note (NOT promoted -- promotion is Rung-1 matched-entropy governance gated on V3-EXQ-569) |
| INV-074 / MECH-333 / MECH-334 (ARC-075) | Phase-3 plasticity-injection crystallization + EWC residue write-protect. GatedPolicy.crystallize() freezes head_0/head_1/discriminator + adds a fresh plastic expansion MLP (zero-init last Linear so output bit-identical at the transition instant; forward = frozen_gated(x) + expansion(x.detach())); ResidueField.snapshot_ewc_anchor() + ewc_penalty() write-protect established basins (Kirkpatrick 2017 EWC, NOT a hard freeze); InfantCurriculumScheduler on_phase3_entry fire-once hook. REEConfig.crystallize_at_phase3 default OFF (bit-identical; 484/484 contracts PASS). Nikishin 2023 plasticity injection + Kirkpatrick 2017 EWC. | Implemented 2026-05-17 (ree-v3 f8b93e3); validation V3-EXQ-543h 2x2x2 (use_gated_policy x use_dacc x crystallize_at_phase3, supersedes V3-EXQ-543g) queued |
| ARC-062 GatedPolicy differential-heads robustness fix | policy.gated_policy two-head reparameterization. Motivated by the V3-EXQ-543h failure autopsy + cross-machine 543g replication (same config landed gating-ACTIVE on host-A but INERT on cloud-3 AND cloud-4; head_0==head_1 collapse is the common cross-machine attractor). When gated_policy_use_differential_heads=True the two heads are SYNTHESIZED as a shared trunk plus a candidate-axis-norm-pinned differential (base +/- delta_hat; delta_hat = differential_bias_scale * delta / (||delta||_K + 1e-8)) so head collapse is a non-equilibrium: delta==0 is structurally unreachable (scale-invariant normalization zeroes the magnitude gradient) and at w=0.5 d(gated)/dw = 2*delta_hat != 0 by the norm pin. crystallize() freezes (base, delta, discriminator). Config GatedPolicyConfig.use_differential_heads (default False -> two independent heads, bit-identical pre-fix path) + .differential_bias_scale (default 0.1). | Implemented 2026-05-18; validation V3-EXQ-543i (supersedes 543g+543h) FAILed branch e (MECH-309 supports / ARC-062 weakens; all 4 diff-ON gated arms 3/3 inert) on a SINGLE machine (Mac); 2026-05-18 governance cycle marked the 543f x4 / 543g / 543h x2 cluster evidence_direction=superseded by 543i + epistemic_category=substrate_ceiling; V3-EXQ-543j byte-identical cross-machine confirmation (pinned ree-cloud-4) queued -- ARC-062/MECH-309 governance gated on it |
| SD-055 | hippocampal.differentiable_cem_selection -- softmax(-score/T)-weighted differentiable ao_mean/ao_std over all CEM candidates (legacy argsort-elite path bit-identical default); restores gradient through the CEM argmax severed for SD-016 cue_action_proj (EXP-0155 / EXQ-449). HippocampalConfig.use_differentiable_cem default False. | Implemented 2026-05-15 (V3-EXQ-568 PASS substrate-readiness, grad_max=372; non_contributory -- does not validate cue-conditioned behavioural divergence) |
| MECH-339 | hippocampal.composite_retrieval_cue_outshining_gate -- ARC-078 Constraint 1: GhostGoalBank composite cue adds an arousal-tag context channel to the z_goal-cosine match, combined by an outshining gate (a strong direct goal_match suppresses the context channel, Smith & Vela 2001). GhostGoalBankConfig.use_composite_cue_outshining default False; context_weight default 0.0 -- bit-identical OFF. | Implemented 2026-05-19 (V3-EXQ-594 diagnostic queued, smoke 4/4) |
| ARC-062 GAP-B mode-separation floor | gated_policy.mode_separation_floor -- composed bias becomes w*h0 + (1-w)*h1 + floor*(h0-h1) so a non-cancelable mode contrast survives at discriminator w~0.5 (the V3-EXQ-543i autopsy gap where delta_hat cancels in base + (2w-1)*delta_hat); optional p1_w_deviation_aux penalizes w near 0.5. GatedPolicyConfig.mode_separation_floor default 0.0 -- bit-identical OFF. | Implemented 2026-05-20; validation V3-EXQ-543k (supersedes 543i; 12-arm + floor/aux on gated arms, K=3 basin-stability gate) re-queued with force_rerun, in flight |
| MECH-282 | regulators.lpb_interoceptive_routing -- LPBInteroceptiveRouter splits harm into z_harm (external; resource slice zeroed before HarmEncoder) and a non-trainable z_harm_intero broadcast (drive_level + harm_obs_a resource EMA); SD-037 coupling routes intero -> override, external -> PAG freeze proxy when both flags on. REEConfig.use_lpb_interoceptive_routing default False. | Implemented 2026-05-21 (V3-EXQ-600 3-arm substrate diagnostic queued) |
| MECH-286 | sleep.override_gated_state_transition -- wake-stability axis of SD-037: wake->offline transition in SleepLoopManager gated by a joint permit (override_signal below threshold AND max region-staleness above recruit AND z_harm_a.norm() below tonic threshold); blocked entry resets episodes_since_sleep. REEConfig.use_mech286_sleep_onset_gate default False -- preserves deterministic K-episode firing. | Implemented 2026-05-21 (V3-EXQ-599 3-arm substrate diagnostic queued) |
| MECH-340 (+ Q-053 wiring) | hippocampal.persistence_efficacy_gate -- ARC-079 / Q-053 front-runner: GhostGoalBank entry persistence as a MECH-293 re-probe target is gated; disengagement is the default when license = control_efficacy * (1 - goal_unattainability) < persistence_floor (SD-039 trace preserved). Q-053 agent wiring (2026-05-21) maps prior hippocampal completion + E3 commitment -> control_efficacy and one-shot 1 - goal_proximity -> goal_unattainability. GhostGoalBankConfig.use_persistence_efficacy_gate default False. | Implemented 2026-05-21 (V3-EXQ-607 diagnostic queued; contracts 8/8 + dry-run PASS) |
| MECH-341 (ARC-065 Layer-B child) | ethics_engine_3.scoring_trajectory_class_diversity_preservation -- new ree_core/predictors/e3_score_diversity.py module (E3ScoreDiversity + E3ScoreDiversityConfig + E3ScoreDiversityDiagnostics + build_from_ree_config). Layer-B (post-CEM scoring) diversity-preservation substrate triggered by V3-EXQ-608 P2 (2026-05-26T02:58Z) majority R2a_e3_collapse_confirmed_large_gap on the diversity-cluster isolation plan. Two togglable sub-flavours under one master, MECH-314a/b/c-style: Option 1 entropy_bonus (per-candidate positive bias proportional to the candidate's first-action class frequency, composed AFTER the dACC / lateral_pfc / ofc / mech295 / curiosity / tonic_vigor score_bias chain and BEFORE last_scores / softmax) + Option 2 stratified_select (partition by first-action class, argmin within class, softmax-sample across class-representatives at stratified_temperature; replaces argmin in the committed-path selection at e3_selector.py:811-820; falls through when fewer than min_classes_for_stratification unique classes are present). Pure-arithmetic regulator (no nn.Module inheritance, no learned parameters); sibling to MECH-313 NoiseFloor + MECH-314 StructuredCuriosity + MECH-320 TonicVigor. REEConfig.use_e3_score_diversity master + sub-knobs (use_e3_diversity_entropy_bonus, use_e3_diversity_stratified_select, e3_diversity_entropy_lambda 0.05, e3_diversity_entropy_bias_scale 0.1, e3_diversity_stratified_temperature 1.0, e3_diversity_min_classes_for_stratification 2) all default bit-identical OFF. 506/506 contracts + 7/7 preflight PASS with master OFF (regression-clean 2026-05-27). **Retune landed 2026-05-28 (ree-v3 e02e77f)**: stratified_select call-site expanded from committed-only to BOTH committed and uncommitted branches (V3-EXQ-611 ARM_2 measured n_stratified_fired=0 because the committed branch was never entered during validation; bit-identical when score_diversity is None or sub-flag is False). V3-EXQ-611b 6-arm parameter sweep queued (3 option groups x 2 entropy_bias_scale values 1.0/2.0; runner-claimed DLAPTOP-4.local 2026-05-28T17:26Z). | Implemented 2026-05-27 (ree-v3 547faa3); retune 2026-05-28 (ree-v3 e02e77f); design doc REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md; V3-EXQ-611 substrate-readiness FAIL 2026-05-27T13:02Z (call-site / scale gaps motivated retune); V3-EXQ-611b retune validation in flight at this snapshot |
| MECH-090 R-c commit-entry readiness conjunction | control_plane.beta_gate.commit_entry_readiness_conjunction -- BetaGate.should_admit_elevation predicate added (margin = sorted(scores)[1] - sorted(scores)[0]; admits iff margin >= commit_readiness_floor). Resolves commitment_closure_plan.md GAP-4 at the substrate-readiness level after V3-EXQ-592 seed 42 (2026-05-21) showed the legacy rv-only commit-entry predicate is satisfiable by degenerate trivial-predictability (rv=2.7e-5 with nav_competence=0.0). Reading R-c single-gate conjunction (synthesis-strongest) per REE_assembly/evidence/literature/targeted_review_connectome_mech_090/synthesis.md (commit 9e68c5ca8a, 28 entries), anchored on Cisek & Kalaska 2010 affordance-competition + Hanes & Schall 1996 FEF accumulator-to-threshold + Roesch / Calu / Schoenbaum 2007 dopaminergic readiness signal. R-a (rv-only is correct) not defensible post-pass; R-b (rv-only entry + downstream propagation gate, Tandetnik 2021) retained as fallback if validation fails. Knobs on HeartbeatConfig (NOT surfaced through REEConfig.from_dims to avoid concurrent-session signature conflict with MECH-341 retune): use_commit_readiness_gate (default False; bit-identical OFF master), commit_readiness_floor (0.05; Q-053-style calibration is a follow-on), commit_readiness_strict_single_candidate (False; permissive single-candidate handling). Per-episode diagnostics on BetaGate.get_state: mech090_n_elevation_admitted / _blocked / _single_candidate / _last_readiness_score_margin. MECH-094 N/A (control-state-transition predicate; no simulation-write surface). 506/506 contracts PASS with master OFF + 7 unit tests on the BetaGate primitive PASS. | Implemented 2026-05-28 (per ree-v3 CLAUDE.md MECH-090 section); validation V3-EXQ-592b queued as 2-arm diagnostic (ARM_0 GATED at floor=0.05 expecting total_committed_steps=0 + mech090_n_elevation_blocked >= 1; ARM_1 GATED_FORCED_READY with experiment-side score_bias injection forcing margin >= 0.10 expecting total_committed_steps > 0 + mech090_n_elevation_admitted >= 1); joint PASS clears commitment_closure:GAP-4 partial -> done |
| MECH-090 R-c continuation (nav_competence axis) | control_plane.beta_gate.commit_entry_readiness_conjunction.nav_competence -- pass 2 of 2 for commitment_closure:GAP-4. The 2026-05-28 landing covered the WITHIN-TICK DECISIVENESS axis (per-candidate score margin -- Hanes & Schall 1996); this pass adds the ACROSS-TICK MOTOR-PROGRAM READINESS axis (Cisek & Kalaska 2010 affordance-preparation + Roesch / Calu / Schoenbaum 2007 dopaminergic readiness). New module ree_core/policy/commit_readiness.py (CommitReadiness + CommitReadinessConfig); pure-arithmetic, no nn.Module, no learned params (sibling pattern to MECH-313 NoiseFloor / MECH-320 TonicVigor). Maintains a [0,1] readiness EMA over per-tick outcome signals plus an explicit notify_outcome(value) harness-push seam; initial value 1.0 fail-open. Both R-c axes AND-compose at both BetaGate elevate sites (bistable + legacy). REEConfig knobs: use_mech090_readiness_conjunction (default False; auto-arms use_commit_readiness via OR-only resolver), mech090_readiness_floor (default 0.3 -- V3-EXQ-592 seed 42 nav_competence=0.0 clearly fails to clear), commit_readiness_window (20), commit_readiness_ema_alpha (0.1; ~10-tick half-life), commit_readiness_initial (1.0). MECH-094 standard simulation_mode pattern. Per-tick outcome-signal source Phase 1: harness pushes via notify_outcome (committed_mode_curriculum.py wired); Phase 2 follow-on (separate session) wires env-emitted "mech090_readiness_outcome" key in agent.sense(). | Implemented 2026-05-29 (V3-EXQ-592b grid extended to 4 arms ARM_0 baseline / ARM_2 GATED_NAV_COMP_ON / ARM_3 GATED_BOTH_ON / ARM_4 BOTH_GATES_OFF_HARNESS_FORCES_READY for orthogonal-axis falsifier; 523/523 contracts + 7/7 preflight PASS with both R-c master flags OFF, including 17 new MECH-090 R-c-nav-competence contracts) |
| SD-056 | E2 action-conditional divergence preservation (contrastive next-state) -- substrate-level fix for the V3-EXQ-571 root-cause finding (2026-05-25): under reconstruction-shaped training, `E2.world_forward` fitted the action contribution to zero (`cand_world_pairwise_dist=0.0000` across K=8 candidates differing only in first action), collapsing per-candidate signal to every downstream consumer of `cand_world_summaries`. Same root cause as 2026-05-17 ARC-062 GAP-B autopsy; the GAP-B fix (gated_policy_use_first_action_onehot) was scoped only to GatedPolicy. SD-056 is the architecturally-faithful generalisation: fix the predictor's training objective rather than bypass it at each consumer. Adds InfoNCE-style auxiliary loss on `world_forward`: positive (z_world_0, a_i) -> predicted z_world_1[i], negatives drawn from in-batch sibling CEM candidates (K-1 distractors), temperature 0.1 (standard literature value). Two new helpers on E2FastPredictor: `cand_world_pairwise_dist` (headline substrate-readiness diagnostic; named by the 2026-05-28 lit-pull SYNTHESIS verdict 3 as a methodological gap worth publishing as standalone novel measurement) and `world_forward_contrastive_loss` (returns unweighted CE; caller multiplies by `config.e2.e2_action_contrastive_weight` before adding to L_E2). Scope: applies to `world_forward` only, not `predict_next_self` (z_self is not the collapse site). Biology anchors: cerebellar internal model (Tanaka 2020) + prefrontal counterfactual rollout (Miyamoto 2023) + vestibular cerebellum corollary discharge (Cullen 2023) all preserve action-specificity via dedicated structural mechanisms. ML/AI anchors: Srivastava 2021 contrastive RSSM, Saanum / Dayan / Schulz 2024 PLSM failure diagnosis, InfoNCE foundation. Config knobs (E2Config + REEConfig.from_dims mirror): e2_action_contrastive_enabled (default False; bit-identical OFF), e2_action_contrastive_weight (0.01), e2_action_contrastive_temperature (0.1), e2_action_contrastive_min_batch_classes (2). MECH-094 standard simulation_mode kwarg. 539/539 contracts + 7/7 preflight PASS with master OFF. | Implemented 2026-05-29 (V3-EXQ-NEW-1 substrate-readiness diagnostic UC1-UC5 queued; behavioural validation V3-EXQ-569a matched-entropy FP-2 falsifier on the fixed substrate queued separately per plan-of-record sequencing -- 569a hit a self-anchored-targets NaN bug 2026-05-29 and superseded by V3-EXQ-569b with observation-anchored targets via /diagnose-errors) |
| SD-022 scheduled-injection extension (MECH-302 unblock) | environment.scheduled_limb_damage_curriculum -- env-side curriculum that periodically injects damage directly into `self.limb_damage` independent of agent action or hazard contact, supplying detectable damage->heal trajectories so the MECH-302 SufferingDerivativeComparator (SD-050) has reliable suffering signals regardless of a trained avoidance policy. Triggered by failure_autopsy_V3-EXQ-517b_2026-05-30: three FAIL discriminative-pair attempts (V3-EXQ-517 / 517a / 517b, 2026-05-04..06) ruled out parameter tuning -- trained avoidance policy filters out hazard-contact -> heal trajectories the comparator needs. Architecturally orthogonal to SD-029 scheduled_external_hazard: SD-029 relocates a hazard adjacent to the agent (still requires agent contact); SD-022 scheduled-injection bypasses contact entirely (allostatic / externally-imposed tissue insult). Five new env-only kwargs (NOT surfaced through REEConfig.from_dims; matches SD-022 / SD-023 / SD-029 / SD-047 / SD-048 / SD-049 / SD-054 precedent): scheduled_limb_damage_enabled (False), _interval (50), _prob (0.5), _magnitude (0.4), _limb_selection ("random" or "all"). Preconditions: enabled=True requires limb_damage_enabled=True (loud-not-silent ValueError). Always-present info dict tags. ML/AI anchor: Bengio 2009 automated curriculum learning -- stochastic gate + random limb selection mitigate schedule-prediction degenerate solutions. MECH-094 N/A (env observation stream). 565/565 contracts + 7/7 preflight PASS. | Implemented 2026-05-30; V3-EXQ-517c PASS 2026-05-30T12:45Z (2/3 seeds ARM_A 160.3 events/seed; 3/3 seeds ARM_B 0 events/seed); cleared MECH-302 + MECH-303 v3_pending gates (IGW-021 at 17:16Z) and lifted gate (c) for MECH-304 V3-EXQ-519 conditioned-inhibition. |
| SD-037 consumer-cascade (MECH-281 motor-coupling axis amend) | regulators.broadcast_override.consumer_cascade -- amend session (NOT a fresh SD landing) triggered by V3-EXQ-483d FAIL (2026-05-29) substrate-ceiling diagnosis: with GoalState seeding + PAG freeze-gate consumers already wired (2026-04-25) but SalienceCoordinator slot dormant in the validation env and PFC/BLA/CeA/beta-gate sites unwired, override_signal had nowhere to land where it would move goal_norm_peak against the MECH-295 bridge baseline. Four additional consumer sites wired (all gated by 0.0-default scalar gains -- bit-identical OFF): (i) LateralPFCAnalog (SD-033a) eff_eta scaled by `1 + override_eta_gain * override_signal` (orexin-recruited state accelerates rule_state EMA); (ii) BLAAnalog (SD-035) encoding_gain scaled by `1 + override_encoding_gain * override_signal` (Roozendaal 2011 orexin -> NE/amygdala enhanced LTP); (iii) CeAAnalog (SD-035) mode_prior + fast_prime scaled by `1 + override_amplitude_gain * override_signal` (re-clipped to mode_prior_log_odds_max); (iv) BetaGate / MECH-091 urgency_interrupt_threshold attenuated by `max(0, 1 - override_beta_interrupt_gain * override_signal)` (orexin escape-from-freeze on motor side, parallel to PAG alpha_override). Four new REEConfig + from_dims knobs: override_pfc_eta_gain, override_bla_encoding_gain, override_cea_amplitude_gain, override_beta_interrupt_gain (all default 0.0). 556/556 contracts + 13 new MECH-281 contracts PASS with master OFF. | Implemented 2026-05-30; validation V3-EXQ-483e queued (4-arm successor under 483 lineage; claim_ids=[SD-037, MECH-280, MECH-281]; re-runs 483d ARM config with use_salience_coordinator=True + all four consumer-cascade gains>0 + PAG-engaging env via SD-036+MECH-279 freeze-engaging substrate). |
| InfantCurriculumScheduler Phase 0->1 H_pos floor recalibration | curriculum.infant.phase0_to_1_exit_signal_recalibration -- closes behavioral_diversity_isolation:GAP-C prereq (3) Phase 0->1 exit signal recalibration. Sibling work to the scaffolded_sd054_onboarding substrate; together they close prereqs (2) and (3) for the goal-pipeline default-config z_goal generation gate. | Implemented 2026-05-31 (earlier in the day before scaffolded_sd054_onboarding); 645/645 contracts + 7/7 preflight PASS. |
| SD-056 multi-step rollout stability amend | e2.world_forward.multi_step_contrastive -- amend to the 2026-05-29 SD-056 landing: multi-step contrastive horizon h=5 (vs h=1) + per-step output norm clamp ratio=2.0 to prevent multi-step rollout drift; resolves 569a self-anchored-targets NaN cluster + provides substrate-readiness pre-conditions for the 614 lineage behavioural validation chain. | Implemented 2026-05-31 (ree-v3 d327b89); V3-EXQ-617 substrate-readiness PASS 2026-05-31T11:31Z confirming amend stability + V3-EXQ-614a/b multi-arm rollout integrity (zero NaN/Inf across 162k steps at ARM_2 ALL_ON; 614 lineage entropy 0.684 -> 0.800 nats post-amend) |
| scaffolded_sd054_onboarding substrate | curriculum.scaffolded_sd054_onboarding -- closes behavioral_diversity_isolation:GAP-C prereq (2) Cluster B substrate-uniform z_goal-zero family addressed by V3-EXQ-490g-cohort autopsy 2026-05-29 (Cluster B / V3-EXQ-603c). Three-phase scheduler at ree-v3/experiments/scaffolded_sd054_onboarding.py (NEW; experiment-harness layer alongside infant_curriculum.py + committed_mode_curriculum.py precedent; ree_core/ otherwise UNTOUCHED). P0: frozen goal pipeline (use_mech295_liking_bridge=False + use_mech307_conjunction=False runtime mutation) + reef-half spawn admissibility (new env kwarg reef_bipartite_agent_spawn_in_reef_half default False on CausalGridWorldV2; pool predicate widened to agent_band OR reef_half when True) + sub-target proximity_harm_scale 0.05 + relaxed hazard density; E1+E2 training over 30 episodes at 200 steps/episode. P1: linear anneal hazard_food_attraction 0.0->0.7, proximity_harm_scale 0.05->0.1, mech295_min_drive_to_fire 1.0->0.01, mech307_conjunction_z_beta_threshold 0.6->0.3; spawn admissibility narrows back to midline band; goal pipeline UNFROZEN; end-of-P1 survival gate (median episode length >= 75 over last 10 episodes, Fix D retained from V3-EXQ-603c). P2: target env config (hazard_food_attraction=0.7, proximity_harm_scale=0.1, num_hazards=4, num_resources=5; matches V3-EXQ-603b GAP-4 Tier-1 measurement env); policy frozen (no optimizer steps); measures z_goal_norm_peak, approach_commit_rate, bridge_cue_fires, dacc_bias_nonzero_steps per episode. Master switch use_scaffolded_sd054_onboarding_scheduler default False on ScaffoldedSD054OnboardingConfig (NOT surfaced through REEConfig.from_dims, matches committed_mode_curriculum / infant_curriculum precedent); 14 phase-config knobs match memo Config Surface table. 645/645 contracts + 17 new scaffolded_sd054_onboarding contracts PASS; bit-identical OFF guarantee verified. MECH-094: N/A (waking-stream env + agent state; no simulation / replay write surface). MECH-302 + MECH-303 v3_pending cleared 2026-05-30 by V3-EXQ-517c PASS sit upstream of this substrate; scaffolded_sd054_onboarding addresses the 583-uniform monomodal-V_s monostrategy tail signature across 483c / 524a / 603 lineage / 540a-e / 590a / 591 / 598 / 598b. | Implemented 2026-05-31 (ree-v3 main 28ebd3d); V3-EXQ-621 ERROR (runner sentinel misclass; manifest recovered from cloud-3) superseded by V3-EXQ-621a PASS 2026-05-31T23:09Z with emit_outcome + P1 survival diagnostics (per-cell p1_episode_lengths[] + verdict lines); V3-EXQ-622 staged goal-stream S0-S3 decomposing 621 z_goal failure queued same evening |
| MECH-341 amend (stratified_within_class_temperature) | ethics_engine_3.scoring_trajectory_class_diversity_preservation.within_class_temperature -- amend (NOT supersede) of the 2026-05-29 SD-056 t=1 substrate landing; routed by failure_autopsy_V3-EXQ-616_2026-05-31 Sections 7 + 10 contingent-on-614b-FAIL-C1 path. Two-part amend: (a) within-class proportional sampling lever via new GhostGoalBankConfig-style field E3ScoreDiversityConfig.stratified_within_class_temperature (Optional[float], default None = legacy argmin bit-identical) so the A-vs-B probe can dissociate Layer B within-class sub-axis from across-class sub-axis (decoupling avoids un-interpretable single-knob conflation); (b) A-vs-B partial-redundancy probe lever NAMED via the existing independent master flags use_support_preserving_cem (Layer A) + use_e3_score_diversity (Layer B) which compose to a complete factorial -- no new config flag is added. Within-class sampling activates when T is set: softmax(-class_scores / T) per first-action class before across-class softmax. REEConfig.e3_diversity_stratified_within_class_temperature surfaced through from_dims (default None). Three new diagnostics on E3ScoreDiversity.get_state(): mech341_n_within_class_sampled / mech341_last_within_class_sampled / mech341_last_within_class_temperature. 655/655 contracts (645 prior + 10 new amend contracts) + 7/7 preflight PASS bit-identical. | Implemented 2026-06-01; V3-EXQ-614c queued (4-arm sweep stratified_within_class_temperature in {None=legacy, 0.5, 1.0, 2.0}; SD-056-amended baseline; ~3-4h Mac). V3-EXQ-614c FAILed instrumentation-defect 2026-06-01T12:45Z -> reclassified non_contributory; V3-EXQ-614d corrected-harness re-run queued (pending DLAPTOP-4.local at snapshot time) |
| MECH-090 R-c continuation Phase-2 (env-source) | control_plane.beta_gate.commit_entry_readiness_conjunction.nav_competence.env_source -- named Phase-2 follow-on to the 2026-05-29 R-c continuation landing: the 2026-05-29 pass wired the consumer + notify_outcome seam but grep-verified ZERO callers ever pushed via the seam (committed_mode_curriculum.py computes nav_competence but never pushes), so in any ecological run the readiness EMA sat pinned fail-open at 1.0 and the across-tick axis added no signal (exactly why V3-EXQ-063a left it OFF). This pass adds the env source. CausalGridWorldV2 gains env-only kwarg mech090_readiness_outcome_enabled (default False; NOT in REEConfig.from_dims, matches SD-022 / SD-023 / SD-029 / SD-047 / SD-048 / SD-049 / SD-054 precedent) emitting info[mech090_readiness_outcome]=clip(1.0 - mean(limb_damage), 0, 1) -- a Cisek-Kalaska affordance-preparation / motor-program-readiness scalar that degrades on SD-022 limb damage and recovers on heal. REEAgent.sense() gains mech090_readiness_outcome arg forwarding the value into commit_readiness.update(). ABSENT-WHEN-DISABLED (no always-present sentinel): default-OFF emits no key, agent reads None, EMA un-advanced. CommitReadiness module UNCHANGED (its None-sentinel + simulation_mode gate already supported this). 719/719 contracts + 7/7 preflight PASS; integration smoke (aggressive all-limb scheduled injection magnitude 0.5) drives readiness EMA to 0.001 below floor 0.3 and recovers on heal. claims.yaml MECH-090 additive note (NO flag/confidence/status change); substrate_dependencies landing mechanism_changing=false (axis never exercised by any prior experiment -> no stale MECH-090 / ARC-029 evidence). | Implemented 2026-06-02 (ree-v3 main fa026a0 substrate + 60d1a90 V3-EXQ-630 queue + e9e1b2b doc-id-fix; REE_assembly master b23ad1a125 design + 6be3673781 doc-id-fix); V3-EXQ-630 queued ecological 3-arm (OFF / GATED_NAV_COMP_ON / GATED_BOTH_ON) on SD-022 scheduled-injection env; claim_ids=[ARC-029, MECH-090]; C3-FAIL routes to /failure-autopsy NOT /diagnose-errors (R-c gate suppressing commitment in env = substantively different ARC-029 verdict). 630 was claimed and dropped from the queue snapshot mid-day during the fleet outage recovery. |
| MECH-342 | control_plane.commit_maintenance_release -- release-side complement to the MECH-090 commit-entry R-c admission predicate (which the V3-EXQ-592f autopsy + MECH-090 release-path audit + motor-cessation lit-pull established is ADMISSION-ONLY by design). Same two R-c readiness signals MECH-090 AND-composes to ADMIT drive a graded bounded-accumulation RELEASE of an already-elevated beta latch when they degrade mid-commitment. Closes V3-EXQ-592f reach gap (predicates fire under forced beta-elevated state but produce zero state-occupancy suppression + zero decommit transitions). Routed by REE_assembly/evidence/planning/mech090_release_path_audit_2026-06-02 (B1 ruled out: none of ARC-028/MECH-105 completion, MECH-091 urgency, V_s commit-release, SD-034 closure covers degraded-readiness mid-commitment) + targeted_review_mech_090_release_motor_cessation/SYNTHESIS.md verdict B3b. Module: ree-v3/ree_core/policy/commit_maintenance_release.py (CommitMaintenanceRelease + CommitMaintenanceReleaseConfig); pure-arithmetic regulator, sibling to commit_readiness.py. Accumulator dynamics per maintenance tick (only while beta is elevated): deficit_d (decisiveness axis) + deficit_n (nav_competence axis) -> combined=max() OR-composition (De Morgan dual of MECH-090 AND admission, conflict-graded by worse axis) -> if combined>0: pressure += accumulation_rate*combined (drift-to-bound, Resulaj 2009; conflict-scaled, Cavanagh/Frank 2011); elif recovered: pressure=max(0, pressure-leak_rate) (reengagement); else: hold (dead-band). fire = pressure >= release_bound. On fire: beta_gate.release() + reset _committed_step_idx + clear _committed_anchor_keys + clear e3._committed_trajectory. BINDING CONSTRAINTS preserved: (1) graded/online not Schmitt flag; (2) targeted+hysteretic with reengagement (Falasconi 2025 movement-specific vs Wessel 2022 non-selective). Distinct (falsifiable) from MECH-090 admission (entry-only AND), MECH-091 (z_harm threat; MECH-342 fires with z_harm_a BELOW threshold), ARC-028 completion (options GOOD; MECH-342 fires when completion LOW), MECH-269b/V_s (schema staleness; MECH-342 fires with STABLE schema), MECH-340 ghost-goal (goal-appraisal timescale; MECH-342 is active beta latch at motor-program timescale). Config (REEConfig + from_dims, all default no-op): use_maintenance_release (False) + 8 floor/reengage/rate/bound knobs. 700 contracts (685 + 15 new MECH-342) + 7/7 preflight PASS; bit-identical OFF verified. | Implemented 2026-06-02 (ree-v3 main 780d12f + REE_assembly master 625e218779); V3-EXQ-592g PASS 2026-06-02 (validation probe: all six criteria met -- C1 baseline occupancy 1.0; C2 score-margin decommit=2 suppression 0.4; C3 nav-competence decommit=2 suppression 0.6; C4 conjunction strictly-positive max_drop 0.6; C5 no false abort; C6 no-vacuity). DISPOSITION: MECH-342 stays candidate / v3_pending (592g is diagnostic; v3-pending gate forbids promotion regardless of evidence count; promotion needs an ecological evidence-grade run). V3-EXQ-629 ecological MECH-342 evidence run queued same day (3-arm shared-trained-weights ON/OFF; was claimed and dropped from queue snapshot mid-day during fleet outage recovery). substrate_queue MECH-342 status -> implemented_validated_v3_exq_592g (ready=false, current_blocker=null). |
| scaffolded_sd054_onboarding amend (update_z_goal wiring) | curriculum.scaffolded_sd054_onboarding.update_z_goal_wiring -- amend (folds V3-EXQ-603d + 625b failure records) on the 2026-05-31 scaffolded_sd054_onboarding substrate. Root cause (confirmed): neither _train_episode nor _eval_episode called agent.update_z_goal -> GoalState.update never reached -> z_goal stayed zero-init every step of every arm; V3-EXQ-603d C4 z_goal=0 SUBSTRATE_FAILURE was a 626-class Class-1 harness/wiring artifact LIVING IN THE SUBSTRATE MODULE, NOT a substrate ceiling. New helper _benefit_and_drive(obs_body) -> (benefit_exposure, drive_level) mirroring experiments/goal_stream_stages_sd054.py; _train_episode gains seed_goal kwarg (True for run_p1; False for run_p0 preserves the warm-up goal-frozen-by-design scope per AskUserQuestion 2026-06-02); _eval_episode (P2) calls update_z_goal after each env.step. TWO-PART FIX (load-bearing): wiring alone is INSUFFICIENT -- 603d's config built the agent WITHOUT z_goal_enabled=True (from_dims default False) -> goal_state was None -> update_z_goal early-returns. The V3-EXQ-603e successor MUST set z_goal_enabled=True + drive_weight=2.0 (matching working reference V3-EXQ-622). Stage-0 positive control: two new contracts (C6 P2 z_goal_norm_peak > 0.0 under forced inputs; C6 update_z_goal called-in-P1-not-P0) make a z_goal=0 scheduler structurally unshippable. 19/19 scaffolded contracts + 7/7 preflight + 665/665 full contract suite PASS (1 pre-existing unrelated infant_curriculum_gap9 C6 stale assertion). | Implemented 2026-06-02 (ree-v3 main deb24cc + d09af0e; REE_assembly master 36b0130ecf); V3-EXQ-603e queued (priority 250, supersedes V3-EXQ-603d, EXPERIMENT_PURPOSE=diagnostic, P0/P1=100/50 restored budget, z_goal_enabled=True + drive_weight=2.0); claimed by DLAPTOP-4.local at the snapshot time. P0 positive control is the adjudicating bit between harness-bug and object-binding abstraction-gap hypotheses. |
| scaffolded_sd054_onboarding amend (nursery / feeding scaffold) | curriculum.scaffolded_sd054_onboarding.nursery_feeding_scaffold -- second amend (2026-06-03) routed by failure_autopsy_V3-EXQ-603e-626a-622_2026-06-03 concluding the update_z_goal-wiring amend is necessary-but-insufficient: V3-EXQ-603e showed z_goal=0 ecologically across 15 cells because 2/3 seeds never reach foraging competence + the hard P2 env (hazard_food_attraction=0.7) starves benefit_exposure even for survivors. Infant REE needs a nursery + feeding time before mature autonomous goal formation can be fairly tested. Five additive levers (all default no-op; bit-identical OFF): (1) FORCED-BENEFIT STAGE-0 nursery method run_stage0_nursery (forced supra-threshold benefit + drive into update_z_goal in dense hazard-free reef-refuge env; positive control "goal stream lights when fed" decoupled from survival skill); (2) scaffold_p1_anneal_hold_fraction lever (staged withdrawal of assistance); (3) explicit STAGE_PLAN module + stage_plan() helper; (4) P2 MEASUREMENT GUARD scaffold_p2_hazard_food_attraction_guard override + contact-rate readout distinguishing "infant never fed" from "goal-formation failure despite contact"; (5) SUBSTRATE-GATE + five-way INTERPRETATION-BRANCH helpers (substrate_not_engaged / fed_but_no_goal / goal_formed_diversity_inert / goal_formed_mechanisms_load_bearing / goal_formed_behaviour_random_harmful). 731 contracts (19 prior scaffolded + 12 new C6) + 7/7 preflight PASS with master OFF. | Implemented 2026-06-03 (ree-v3 main); V3-EXQ-603f post-substrate re-issue authored but DEFERRED -- the runtime readiness gates (Stage-0 z_goal>0.4 on >=2/3 seeds, P1 survival >=2/3, P2 contact >0 on >=2/3) require a full-budget readiness run first; substrate_queue.ready stays FALSE until V3-EXQ-634 / 634b / 634c clear |
| scaffolded_sd054_onboarding amend (developmental-window / protected-goal consolidation) | curriculum.scaffolded_sd054_onboarding.developmental_window -- third amend (2026-06-03b) routed by the V3-EXQ-634 design-error review. Root cause (verified in code): GoalState.update (ree_core/goal.py:173) ALWAYS decays the persistent z_goal attractor (z_goal *= 1-decay_goal) BEFORE the benefit-gated pull AND REEAgent.reset does NOT reset goal_state, so the prior scaffold called update_z_goal every step (incl. UNFED steps) in P1/P2 -> each unfed step is a pure decay-only washout. Three additive levers (all default no-op; bit-identical OFF): (1) Stage-0b PROTECTED CONSOLIDATION window (new run_stage0b_consolidation; E1/E2 training open but update_z_goal NOT called so the z_goal attractor cannot be washed out by decay-only updating; retention_gate >= 0.75 of Stage-0 baseline); (2) CONTACT-GATED P1/P2 updates (when scaffold_contact_gated_goal_updates is set, _train_episode and _eval_episode call update_z_goal ONLY on validated contact steps; decay_only reserved for mature tests); (3) goal-write-mode constants + per-phase diagnostics (n_contact_refresh_updates / n_decay_only_updates / n_skipped_protected_updates) so manifests distinguish goal loss due to no-contact vs decay-only washout vs failed-formation-despite-contact. New C7 contract group; 739/739 contracts + 7/7 preflight PASS with master OFF; bit-identical legacy path verified. | Implemented 2026-06-03 (ree-v3 main); V3-EXQ-634b corrected nursery readiness (developmental-window flags ON) queued; G0b retention 3/3 PASSed (consolidation amend VALIDATED) but exposed seeding-magnitude / threshold mismatch downstream (G3 anti-correlated with foraging on seed 42) -> routed to next amend |
| scaffolded_sd054_onboarding amend (seeding-calibration + consumption-gated G3) | curriculum.scaffolded_sd054_onboarding.seeding_calibration -- fourth amend (2026-06-03c) routed by failure_autopsy_V3-EXQ-634b. Consolidation half VALIDATED (G0b retention 3/3, n_decay_only_updates=0) but exposed benefit-magnitude / threshold mismatch (verified in code): contact-gating skipped only benefit <= contact_threshold (1e-6) but GoalState.update seeds only when effective_benefit > benefit_threshold (0.1); natural wild benefit ~0.03 stays sub-threshold, so the band (1e-6, ~0.1-effective) DECAYED the consolidated trace during real foraging instead of refreshing it (the forced 1.0 nursery feed was the only supra-threshold input). Three coupled fixes (all default no-op; bit-identical OFF): (1) DECOUPLED CONTACT-GATING THRESHOLD (scaffold_contact_gating_benefit_threshold sentinel < 0 -> reuses readout threshold; >= 0 separates the gating floor from the contact-RATE readout so sub-seeding whiffs are PROTECTED not decay-only updated); (2) GOAL-SEEDING MAGNITUDE PROPAGATION (Optional scaffold_z_goal_seeding_gain / scaffold_benefit_threshold / scaffold_drive_floor knobs propagated onto agent.goal_state.config at the top of each seeding-capable stage -- GoalConfig owns the magnitudes; scaffold propagates so V3-EXQ-634c sweep can vary them through the scaffold's own config surface); (3) CONSUMPTION-EVENT-GATED G3 READOUT (P2OnboardingMetrics gains z_goal_norm_at_contact_peak + num_contact_events; G3 reads at genuine seeding events so a z_goal=0-at-contact read is no longer masked by the carried forced-feed nursery trace). New C8 contract group (6 contracts); 744/744 contracts + 7/7 preflight PASS with master OFF + amend OFF; bit-identical legacy path verified. | Implemented 2026-06-03 (ree-v3 main 15053a3); V3-EXQ-634c multi-arm sweep over seeding magnitudes + strengthened budgets + matched contact-gating floor queued (claimed DLAPTOP-4.local 2026-06-03T22:04Z); ready stays false; do NOT queue 603f until 634c clears a consumption-event-gated gate |
| modulatory-bias-selection-authority | ethics_engine_3.modulatory_bias_selection_authority -- gap-relative E3.select authority for modulatory / diversity score-bias channels. Root cause (604a / 624a / 614d cluster autopsy): fixed small modulatory magnitudes (~0.05-0.1) added to primary scores whose raw_score_range was much larger never changed the argmin -- 604a curiosity_bias=0.0 every arm, 624a vigor action_density byte-identical ON==OFF, 614d within-class temperature -> committed-class entropy byte-identical across T=0.5/1.0/2.0. Approach (b) gap-relative scaling (user-confirmed AskUserQuestion 2026-06-03): rescale composed modulatory bias so range(mod) == modulatory_authority_gain * raw_score_range; takes precedence over legacy normalize_score_bias_to_e3_range when on. Sibling stratified-across-class normalization in e3_score_diversity.stratified_select normalizes class-representative scores to UNIT range before the stratified_temperature softmax (614d C2 fix -- absolute class-rep gap no longer collapses committed-class selection). SAFETY: primary scores NOT modified -> commit-threshold / running_variance / softmax-temperature / urgency-interrupt / MECH-090 admission semantics unchanged; gain=0.5 < 1.0 keeps modulatory competitive in near-ties but subdominant when the primary harm/goal gap exceeds gain*range. Config (REEConfig + from_dims + E3Config, all default no-op): use_modulatory_selection_authority + modulatory_authority_gain (0.5) + modulatory_authority_min_range_floor (1e-6). NECESSARY-BUT-NOT-SUFFICIENT for the curiosity lever: 624a/614d are pure drowning (fixed directly); 604a had curiosity_bias=0.0 (genuinely zero -- MECH-314a no active residue centers + 314b/c broadcast-by-design), so scaling zero is still zero -- the validation EXQ guards curiosity_bias_abs_mean > 0 before testing curiosity. 734/734 contracts + 7/7 preflight PASS with flag OFF (regression-clean under two pytest-randomly orderings). MECH-094: pure arithmetic on the waking committed-selection path; no replay write surface. | Implemented 2026-06-03 (ree-v3 main); V3-EXQ-635 substrate-readiness PASS 2026-06-03 (WITHIN_CLASS lever lift +0.446, harm down, 19 authority-normalized ticks); concurrency note: clearing the substrate gate during 614d review auto-spawned IGW-024 for this substrate; both sessions converged on the identical design and the joint working-tree implementation was landed from the interactive session (igw-024 stood down, empty worktree). PASS unblocks per-claim EVIDENCE retests of MECH-314 / MECH-320 / MECH-341 + the MECH-343 hypothesis |
| SD-057 v1 (object-bound incentive-salience L2+L3+L4) | drive.object_bound_incentive_salience -- closes the goal_pipeline:GAP-7 middle layer (the goal stream wrote a SINGLE z_goal attractor overwritten on every contact; wanting target == liking target always; L9 wanting!=liking dissoc stuck at 0.0, V3-EXQ-514l). Inserts a per-object incentive layer between the benefit pulse and z_goal via new IncentiveTokenBank class on GoalState (instantiated when master flag set). L2 MECH-344 BIND: on contact, benefit binds to the SD-049 per-type tag k -> bank.update(k, benefit, z_resource) (Cardinal/Everitt 2002 BLA-analog). L3 MECH-345 TOKEN: per-type bank entry holds base_value[k] (slow-decay revaluable EMA of received benefit) + z_object[k] (stored z_resource identity embedding); wanting at recall = base_value[k] * (1 + incentive_drive_kappa_weight * per_axis_drive[k]) -- Zhang 2009 V = r*kappa(drive) multiplier RELOCATED from GoalState seeding gate onto the stored per-object value (specific PIT; Corbit/Balleine 2005/2011). L4 MECH-346 POINTER (MECH-230 amend): z_goal seeded FROM the most-wanted object's embedding (k* = argmax wanting -> seed_latent = z_object[k*]) instead of the raw last-contacted z_resource; firing gate UNCHANGED -- only the seed SOURCE changes; liking target (last-contacted) and wanting target (z_goal -> most-wanted) can DIFFER. Config (GoalConfig + from_dims; all no-op default, bit-identical OFF): use_incentive_token_bank (False), incentive_decay (0.005), incentive_value_alpha (0.1), incentive_drive_kappa_weight (2.0), incentive_use_per_axis_drive (True). MECH-094 N/A (waking contact only; no replay write surface). 747/751 contracts (4 pre-existing local-git-env runner artifacts unrelated) + 7/7 preflight PASS. | Implemented 2026-06-04 (ree-v3 main 53f6427 substrate + f297c1d V3-EXQ-636 queue; REE_assembly master 852c51c005 audit + 1f12a8e60f claims + 07d59e24fd plan-of-record); V3-EXQ-636 forced-contact mechanism diagnostic (claim_ids=[], decoupled from GAP-2) queued + ingested into coordinator /queue/active; PASS 2026-06-04 (bank binds >=2 identities, wanting!=liking 5/6, OFF control 0). MECH-229 / MECH-117 / ARC-030 stay v3_pending (unblocked-not-validated). |
| SD-057 phase-2 (L6 cue-recall + L7 dACC object-discriminative readout) | drive.object_bound_incentive_salience.phase_2 -- completes the GAP-7 L0-L9 closure map on top of SD-057 v1. L6 MECH-347 CUE-RECALL: new GoalState.cue_pull(z_object, strength) directional z_goal nudge with NO benefit gate and NO token revaluation; new REEAgent.cue_recall_wanting(cue_type, drive_level) primitive retrieves the bank token for cue_type and cue_pulls z_goal toward z_object[cue_type] by cue_recall_gain * clamp(amp) -- identity-matched + drive-specific (Schultz 1997/98 DA-transfer-to-cue). L7 MECH-348 dACC OBJECT-DISCRIMINATIVE READOUT: select_action dACC block computes per-candidate goal_proximity (to the object-bound z_goal, reusing the MECH-295 first-step z_world summary pattern) under use_mech_consume and passes candidate_goal_proximity into self.dacc(...). DACCConfig.dacc_goal_readout_weight (0.0) -- goal-readout term added INDEPENDENTLY of dacc_weight (so a goal-conditioned consumer works even when the legacy dACC bias is off, after the dacc_bias clamp; skipped/bit-identical when weight 0 / readout None). REEAgent precondition: use_mech_consume requires use_dacc; use_cue_recall requires use_incentive_token_bank (loud-not-silent at __init__). StepHarness auto cue-perception: when use_cue_recall, derive the strongest-perceived resource type from SD-049 per-type proximity field views (argmax over resource_field_view_<name>) and call cue_recall_wanting each step (best-effort try/except; bit-identical no-op when off / bank absent / env emits no per-type views). 750/757 contracts + 7/7 preflight PASS; smoke 2026-06-04 (cue fires + moves z_goal; sim_mode no-op; preconditions raise; dACC OFF bit-identical / ON favours high-proximity candidate). | Implemented 2026-06-04 (ree-v3 main 24f31e5 substrate + 4073dcb V3-EXQ-637 queue; REE_assembly master e79ef7207e claims + 44c4333870 plan node); V3-EXQ-637 forced-cue diagnostic (claim_ids=[]) PASS 2026-06-04 (cue fires no-benefit, identity-matched cos~1.0, readout reaches dACC len32). MECH-229 / MECH-117 / ARC-030 stay v3_pending pending the GAP-2-gated L9 wanting!=liking behavioural retest |
| ARC-063 v1 distributed CandidateRule field (GAP-B non-Bayesian rule-creator) | policy.rule_apprehension_layer.candidate_rule_field -- the non-Bayesian rule-CREATOR resolving arc_062_rule_apprehension:GAP-B (MECH-309: "trainers weight rules they do not invent"). Mint-then-weight over a subspace-partitioned field: CREATION is a non-gradient structural mint event (MECH-349 CREATE on recurring (context->action-object) regularities >= crf_mint_recurrence_threshold times); WEIGHTING is eligibility-trace credit on existing rule availability. New module ree_core/policy/candidate_rule_field.py (CandidateRuleField + CandidateRule unit = rule_embedding [rule_dim] + context_tag [world_dim] + availability + eligibility); pure-arithmetic regulator (no nn.Module, no trained parameters, no gradient flow). Five faces: MECH-349 CREATE (mint distinct slot on recurrence; optional ARC-062 discriminator seed); MECH-350 REPRESENT (pinned-distinct unit-vector slot directions -- deterministic seeded basis -> distinct minted rules occupy distinct subspace directions; the anti-monomodal geometry; Weber 2023 / Wallis 2001); MECH-351 GATE (tolerance-gated availability with theta = crf_tolerance_floor + crf_tolerance_conflict_gain * n_competing -- Frank 2006 conflict-graded threshold; availability != selection); MECH-338 SELECT (cue-driven context-bound retrieval cosine(context, context_tag) >= crf_context_match_threshold); MECH-352 CREDIT (eligibility-trace credit raise-on-success / lower-on-exception with slow decay -- Brzosko 2015 / Kovach 2012). GAP-B wiring: LateralPFCAnalog gains use_candidate_rule_source + crf_source kwarg; when supplied, crf_source REPLACES the legacy delta_proj/world_proj EMA source -- the literal 598b trainable_not_monomodal fix. Precondition: use_candidate_rule_field requires use_lateral_pfc_analog. MECH-094 standard simulation_mode + MECH-319 _lpfc_skip gate. Config (REEConfig + from_dims; all no-op default, bit-identical OFF): use_candidate_rule_field (False), crf_n_slots (16), crf_rule_dim (16), crf_mint_recurrence_threshold (3), crf_tolerance_floor (0.3), crf_tolerance_conflict_gain (1.0), crf_availability_alpha (0.1), crf_availability_decay (0.005), crf_eligibility_window (20), crf_context_match_threshold (0.5), crf_seed_from_arc062 (True). 775/782 contracts + 7/7 preflight PASS (7 pre-existing local-git runner-recovery artifacts unrelated). | Implemented 2026-06-04 (ree-v3 main 175a24f substrate + V3-EXQ-639 queue; REE_assembly master 13f4c8436e -- registers MECH-349/350/351/352 + ARC-063 implementation_note + sleep V4->V3 scope correction in doc+title); V3-EXQ-639 readiness diagnostic (claim_ids=[], C1-C4+UC5) PASS 2026-06-04 (C1-C4 + UC5; field wired, OFF bit-identical, MECH-094 sim-gate). Children candidate + v3_pending; substrate landing does NOT promote -- ARC-062 GAP-B behavioural diversity re-run on field-ON substrate is the governance-weighting successor (queued separately). |
| ARC-080 + ARC-081/082/083 object-representation umbrella + pillars | architectural_commitment.object_representation_primitive -- thin umbrella (Option A spine) registering object identity = cross-cutting representational primitive across the three previously-disconnected lineages (dormant ARC-006/045 entity-files / developmental ARC-059/MECH-278 with the object definition explicitly bypassed in V3 / live resource-bound SD-015->049->057). Object = token-bound identity persisting across time / perceptual gaps. ARC-080 architectural_commitment, candidate, v3_pending, implementation_phase v4; depends_on ARC-006 + MECH-278 + ARC-059 + SD-015/049/057. Four pillars as thin architectural_commitment children: ARC-081 self-as-object (V4 self-model + interoceptive grounding); ARC-082 tools / affordances (V3-or-V4 boundary); ARC-083 others-as-object (V4 social layer; the prerequisite for shared joys/sorrows). DOC + GOVERNANCE only -- no substrate code, no experiments, no promotion, no V3 behaviour change. New design doc docs/architecture/arc_080_object_representation_primitive.md (coherence-map: three disconnected lineages; three overlapping per-item stores [type-tag bank | anchor ghost bank | entity-token]; type-vs-anchor-vs-token first design fork -- live work is TYPE-level, true permanence/tools/self/other need TOKEN-instance). Cross-references threaded through entities_and_binding.md, sd_057, sd_039, developmental_needs_register.md. Biology grounding via concurrent L1 lit-pull (object files & feature binding: Kahneman/Treisman/Gibbs 1992; Treisman & Gelade 1980; Olsen 2012; Pylyshyn FINST 1988; Scholl 1999) + L2 lit-pull (object permanence: Baillargeon 1985 drawbridge VOE; Spelke 1990 core-knowledge; Kellman & Spelke 1983; Xu & Carey 1996 token-vs-type at 10 vs 12 mo; Diamond 1985 A-not-B). lit_conf parallel signal (ARC-080 0.857, ARC-006 0.871, MECH-045 0.868). | Registered 2026-06-04 (REE_assembly master 075ebbe76d umbrella + 4 pillars; lit-pulls d93508e19d + 078b4cea12). Off V3-closure critical path: GAP-7 / scaffolded_sd054_onboarding / SD-057 resource-bound behaviour untouched. V3/V4 boundary: V3 BEGIN (live SD-057 type-tag substrate continues; spine documentation only); V4 CUTOVER (token-instance permanence + tools + self + social pillars activate). |
| SD-055 re-registered (claims.yaml entry restored) | hippocampal.differentiable_cem_selection -- the SD-055 differentiable_cem_selection claim entry was silently dropped from claims.yaml on 2026-05-15 by commit d3fb89c64d (auto-sync conflict re-apply). The substrate implementation in ree-v3 was not affected. Entry restored 2026-06-04 from registration commit 59ca45d6c4 and updated to reflect subsequent substrate-ready / default-flip status. claims.json rebuilt. | Re-registered 2026-06-04 (REE_assembly master); substrate implementation unchanged (still implemented 2026-05-15; V3-EXQ-568 PASS substrate-readiness with grad_max=372) |
| scaffolded_sd054_onboarding amend (SD-057 cue-recall bridge into nursery curriculum) | curriculum.scaffolded_sd054_onboarding.cue_recall_bridge -- fifth amend (2026-06-04) integrates the SD-057 L6 cue-recall + L2 bank-token-binding into scaffolded_sd054_onboarding as a candidate lever on the GAP-2 foraging-CONTACT axis (NOT survival). Hypothesis (wean-to-wild): nursery forced-feed already builds z_goal but has no path from a nursery-built goal to APPROACHING a resource the agent can SEE but has not contacted; SD-057 cue-recall is that path (Pavlovian-instrumental transfer / sign-tracking). Changes (all behind scaffold_cue_recall_bridge_enabled, default False, bit-identical OFF): (1) _build_env spreads multi_resource_heterogeneity_enabled + n_resource_types + per_axis_drive_enabled into ALL 4 phase env constructors when the bridge is on (the scaffold's envs previously did NOT enable SD-049, so they emitted no per-type tags / proximity views / per-axis drive -- the central gap); (2) _train_episode (P1) + _eval_episode (P2) pass resource_type into agent.update_z_goal so the bank binds per-object tokens (L2) AND fire _maybe_cue_recall each step (L6); n_cue_recall_fires surfaced in P2OnboardingMetrics. REQUIRES the caller to build the agent with use_incentive_token_bank=True + use_cue_recall=True + use_resource_encoder=True. 55/55 scaffold contracts (42 prior + 13 new C9) + 7/7 preflight. Honest scope: targets the CONTACT axis only; does NOT fix survival. | Implemented 2026-06-04 (ree-v3 main 5fa7be3 substrate + V3-EXQ-638 queue; REE_assembly master). V3-EXQ-638 cue ON vs OFF contact-rate ablation queued; FAILed 2026-06-04 (C1 cue fires ON = 0 across all seeds) -- root cause IncentiveTokenBank empty entering P1/P2 because Stage-0 forced feed passed rt=_contacted_resource_type (~always None, decoupled from typed contact). Routed to formation-fix amend below. |
| scaffolded_sd054_onboarding amend (cue-recall FORMATION fix + diagnostics) | curriculum.scaffolded_sd054_onboarding.cue_recall_formation_fix -- sixth amend (2026-06-04b) routed by the V3-EXQ-638 cue-silent autopsy. Root cause (code-confirmed): IncentiveTokenBank empty entering P1/P2 because Stage-0 forced feed passed rt=_contacted_resource_type (~always None) -> bank.update never bound -> cue_recall_wanting returns 0 (no_token) -> cue_fires=0. Compounded by a bare `except: pass` in _maybe_cue_recall that made cue_fires=0 undiagnosable. Two no-op-default changes (both bit-identical OFF): (A) INSTRUMENTATION -- _maybe_cue_recall gains an optional cue_diag accumulator; every non-fire is attributed to a reason (no_token / resource_field_absent / proximity_below_threshold / bank_none / amp_zero_or_zobject_none / exception:<Type>); the `except: pass` is replaced with `exception:<Type>`; surfaced on P1OnboardingResult.cue_diag + P2OnboardingMetrics.cue_diag + Stage0NurseryResult.token_bank_size_end. (B) FORMATION FIX -- new flag scaffold_stage0_bind_incentive_token (default False); when on, Stage-0 forced feeding binds the token to the STRONGEST-PERCEIVED resource type each step (new shared helper _strongest_perceived_type factored from the cue logic so formation and recall use IDENTICAL perception). 62/62 scaffold contracts (55 prior + 7 new C9) + 7/7 preflight PASS. Smoke 2026-06-04 (bridge + bind ON, 2-ep Stage-0 + 2-ep P1): Stage-0 token_bank_size_end=2 (was 0); P1 cue fires 34x (n_token_matches=34, empty nonfire reasons) -- cue_fires=0 -> 34 purely from populating the bank. | Implemented 2026-06-04 (ree-v3 main a9ef0be); V3-EXQ-638a re-issue with scaffold_stage0_bind_incentive_token=True queued via /queue-experiment; the interoceptive need-gating layer + V3-EXQ-638b (OFF / EXTERNAL_ONLY / INTEROCEPTIVE+EXTERNAL arms) is a SEPARATE later pass pending 638a confirming the bank-empty reason dominated |
| scaffolded_sd054_onboarding amend (n_cue_recall_fires aggregation fix) | curriculum.scaffolded_sd054_onboarding.cue_fires_aggregation_fix -- seventh amend (2026-06-04c); the clean underlying fix for the V3-EXQ-638 measurement gap surfaced while validating V3-EXQ-638a. _eval_episode RETURNS a per-episode n_cue_recall_fires and _train_episode ACCUMULATES cue fires into goal_write_diag["n_cue_recall_fires"], but run_p2 never aggregated the per-episode value onto P2OnboardingMetrics and run_p1 never surfaced a total on P1OnboardingResult -- so any consumer doing getattr(p2, "n_cue_recall_fires", 0) silently read 0 EVEN WHEN THE CUE FIRED (directly observed: V3-EXQ-638a smoke fired the cue 30x in P2 while getattr returned 0). Fix (no-op-default; bit-identical when the cue bridge is off -> 0): new aggregated field n_cue_recall_fires: int = 0 on BOTH P2OnboardingMetrics and P1OnboardingResult; run_p2 sums the per-episode ep_metrics.get("n_cue_recall_fires", 0) across episodes; run_p1 surfaces goal_write_diag.get("n_cue_recall_fires", 0). Contract: the new top-level field EQUALS cue_diag["n_cue_recall_fires"] (both count the same fires); 0 when the bridge is off. cue_diag unchanged. 65/65 scaffold contracts (62 prior + 3 new C9) + 7/7 preflight PASS. | Implemented 2026-06-04 (ree-v3 main 636128a). Pure aggregation; original V3-EXQ-638 C1 measurement bug fixed at the source (future consumers reading getattr will no longer trip on it) |
| SD-016 Path 3 (feedforward cue->slot tagger) | e1.cue_slot_tagger -- replaces ONLY the slot-SELECTION scores in `E1DeepPredictor.extract_cue_context` (the z_world-only q.k attention that V3-EXQ-418i diagnosed as pinned at the uniform ln(num_slots) saddle); slot-CONTENT path (value_proj -> output_proj -> cue_context) + cue_action_proj (449a z_world concat retained) + cue_terrain_proj UNTOUCHED. New `cue_slot_tagger = Linear(world_dim, hidden) -> ReLU -> Linear(hidden, num_slots)` MLP; non-uniform logits from step 0 -> sits OFF the saddle so the existing terrain_loss gradient flows back into it and shapes contextual selectivity. No new supervised target invented -- better-conditioned replacement for the saddle-stuck attention; same gradient source. Read-only diagnostic `_last_cue_slot_weights` cached for validation experiments. Config (E1Config + REEConfig.from_dims; all no-op defaults): `sd016_cue_slot_tagger` (False; requires sd016_enabled=True), `sd016_cue_slot_tagger_hidden` (32), `sd016_cue_slot_tagger_temperature` (1.0). Off path bit-identical (selection entropy == ln(16) exactly). Honest scope: restores RETRIEVAL selectivity; full action_bias_div >= 0.05 behavioural propagation also depends on cue_action_proj (separate SD-055 differentiable-CEM / ARC-065 concern). MECH-094 N/A (waking E1 query). 7/7 preflight + 5/5 new contracts in tests/contracts/test_sd016_cue_slot_tagger.py PASS. | Implemented 2026-06-05 (ree-v3 main 88695ed substrate + V3-EXQ-418m validation; REE_assembly master design doc Path 3 section appended). V3-EXQ-418m substrate-readiness diagnostic (claim_ids=[]; PRIMARY acceptance = mean selection entropy < 2.5 vs the pinned ln(16)=2.773 with the tagger ON) queued via /queue-experiment. |
| MECH-353 / MECH-354 / MECH-355 (affect-stream cluster) | claims.yaml claim registration only -- three proto-feeling streams that the 2026-06-05 affect-stream lit-pulls (relief/safety/soothing + fatigue/suffering + boundary/agency) confirmed are distinct and NOT collapsible into existing harm primitives. **MECH-353** (blocked_agency / control-failure `z_block`; V3 candidate, v3_pending; dep SD-029 + MECH-112 + MECH-342 + ARC-016 + MECH-320 + SD-011 + SD-019b): detector = SD-029 comparator on action-outcome channel, antecedent = frustrative-non-reward expected-minus-realised with NO noxious input, smallest form = integrated comparator-mismatch + external-attribution gate + capacity gate; consumers = assert (MECH-320 vigor) -> decommit (MECH-342) gated by ARC-016 -> withdraw only at capacity-collapse handoff to z_harm_a (V3-tractable). **MECH-354** (effort/fatigue stop-recover two-bound accumulator; V3 candidate, v3_pending; dep SD-012 + SD-048 + MECH-342 + ARC-078 + SD-017 + SD-011, NOT SD-011): SD-012 homeostatic side, SD-048 interoceptive host, Meyniel-2013 two-bound (hysteretic) leaky cost-evidence accumulator F += Se*effort / F -= Sr; STOP at upper bound, recover to lower; fast within-task accumulator + slow Process-S sleep-pressure variant whose recover phase is OFFLINE (SD-017); wires via MECH-342 release actuator (deficit_f OR-composed with execution-readiness deficit) + ARC-078/ARC-079 cost-side persistence gate (cost/benefit, not aversive). **MECH-355** (soothing autonomic state-gain modulator; V4-social, candidate, substrate_conditional, promote/demote suppressed; dep MECH-219 + SD-012 + SD-032e + SD-011, NOT 302/303/304/112): DECAY-ACCELERATION update rule (multiplier on MECH-219 z_harm_a recovery_rate + SD-032e drive_bias leak; onset/sensory streams untouched) + optional default-OFF/ablatable gain-reduction secondary face on SD-032e accumulation WRITE only; multiplicative-on-existing-state -> zero effect on a calm agent -> soothing != sedation. NO substrate code; NO experimental evidence yet; lit-grounded only (mirrors the MECH-302 precedent). | Registered 2026-06-05 (REE_assembly master 0a9dda6b99 claims.yaml + 7d89ffd0ba affect_primitives.md Extension Register consolidation + 9d47b3a945 blocked_agency row + 045dac6b9d MECH-303/304 reuniens/remote-recent enrichment + 7c6a1f0b55 MECH-355 design pass + ea53570ec9 MECH-354 design pass). All three stay candidate; MECH-355 substrate_conditional. Validation EXQs gated: MECH-353 has a V3 discriminative experiment proposed (smallest blocked-action env, harm + goal held constant, measure z_block rise + assert/persist distinct from withdraw, dissociation from z_harm_a under matched controllability); MECH-354 has a GATED /queue-experiment plan (cue-authority 638b/640a routes the gating multipliers, so the BUILD stays gated); MECH-355 substrate build deferred to V4-social. |
| MECH-314a Phase-2 amend (e2.world_forward novelty-candidate-source) | policy.structured_curiosity_bonus.e2_world_forward_novelty_source -- routed by failure_autopsy_V3-EXQ-648_2026-06-07 precondition_unmet (proposer-derived `cand_world_summaries` spread <0.01 under monostrategy -> curiosity_bias_range=0 in every arm even though the SD-056-trained e2.world_forward predictions carry spread ~0.1147). Adds a new `REEAgent._curiosity_candidate_summaries(candidates)` helper consulted FIRST in the curiosity block; cur_summaries = `e2.world_forward(z0.expand(K,-1), first_actions_K)` when `curiosity_candidate_source="e2_world_forward"` (Literal default "proposer" = legacy bit-identical reuse-chain). Both 314a per-candidate RBF novelty AND the auto-augmentation `_candidate_spread` now key on the action-divergent representation. Bit-identical OFF; 877/877 contracts + 7/7 preflight PASS; 4 new C6 contracts. MECH-094 preserved (no_grad on the waking path). | Implemented 2026-06-07 (ree-v3 main). V3-EXQ-648a substrate-readiness validation queued (supersedes V3-EXQ-648) with cand_world_pairwise_dist readiness precondition; PASS gates V3-EXQ-590b + the section-8 MECH-314a/MECH-314/ARC-065 governance updates. |
| ARC-065 GAP-A (shared cand_world_summaries e2.world_forward source) | policy.candidate_pool_per_candidate_signal_preservation.shared_channel -- routed by failure_autopsy_V3-EXQ-614e_2026-06-07. Shared-channel sibling of the MECH-314a Phase-2 curiosity amend (landed earlier the same day): that pass fixed ONLY the curiosity-channel consumed representation; this pass extends the identical e2.world_forward re-sourcing to the SHARED per-candidate `cand_world_summaries` consumed by ALL the other E3-side bias channels (lateral_pfc / ofc / mech295 / gated_policy / tonic_vigor). New `REEAgent._candidate_world_summaries(candidates)` helper consulted FIRST at all five `cand_world_summaries` fresh-build sites (gated_policy block, lateral_pfc fallback, ofc fallback, mech295 fallback, and via reuse-chain the tonic_vigor anchor). Config: `REEConfig.candidate_summary_source: Literal["proposer","e2_world_forward"] = "proposer"` (default; bit-identical). Kept SEPARATE from `curiosity_candidate_source` so the two compose without perturbing each other. 889/889 contracts (883 prior + 6 new) + 7/7 preflight PASS; new contracts in `tests/contracts/test_arc065_gapa_candidate_summary_source.py` (G1-G5 covering proposer-helper-None bit-identical, e2_world_forward shape, divergent-world_forward shared spread > collapsed proposer spread, master-off, select_action end-to-end). | Implemented 2026-06-07 (ree-v3 main). V3-EXQ-649 substrate-readiness diagnostic queued with `cand_world_pairwise_dist` readiness precondition + shared-bias-channel per-candidate range readout. PASS unblocks the MECH-341 committed-class diversity re-test (within-class REPRESENTATIVE diversity readout, NOT committed-class entropy per Learning #2 of 614e autopsy). |
| scaffolded_sd054_onboarding AMEND (curriculum decomposition / isolated Stage-H) | curriculum.scaffolded_sd054_onboarding.curriculum_decomposition -- routed by failure_autopsy_V3-EXQ-603f_2026-06-07. 603f PROVED the goal-formation + ecological-seeding chain is SOUND (seed 44 foraged + seeded z_goal ecologically at z_goal_norm_at_contact_peak 0.450 > 0.4) but P1 SURVIVAL leg was 0/3 -- P1 couples goal-pipeline unfreeze + hazard wean simultaneously and the agent cannot acquire both at once; P0 trains only in the safe reef. Adds an isolated hazard-avoidance Stage-H between P0 and P1: new `run_hazard_avoidance` method + HazardAvoidanceResult dataclass; goal pipeline FROZEN (seed_goal=False -> update_z_goal never called), trains E1+E2 in a hazards-with-randomly-drifting-no-food-attraction env at midline spawn (~survival/avoidance signal). Curriculum becomes Stage-0 -> Stage-0b -> P0 -> Stage-H -> P1 -> P2. All behind `scaffold_hazard_stage_enabled` (False default; bit-identical). 85/85 scaffold contracts (79 + 6 new C12) + 7/7 preflight PASS. | Implemented 2026-06-07 (ree-v3 main). V3-EXQ-603g substrate-readiness queued (copy of 603f with Stage-H inserted); 603g FAILed P1 G1 (substrate-readiness fail, gate engaged but insufficient) routing to SD-058/MECH-357 instrumental-avoidance and SD-059/MECH-358 escape-affordance bridge below. |
| ControlVector logging (rec-B telemetry) | telemetry.control_vector_logging -- read-only, default-OFF telemetry making value / effort / opportunity-cost-of-time / vigor separately inspectable each E3 tick. Exposes the ARC-068-vs-MECH-320 collapse (opportunity cost and vigor are both `w*v_t` for the SAME MECH-320 v_t scalar -- ARC-068 is registered but unbuilt). Recommendation B (logging only); causal first-class opportunity-cost split (C) + full four-axis controller (D) DEFERRED post-green-board, gated on ARC-068 lit-pull and MECH-320 regaining selection authority. Modules: REEConfig.use_control_vector_logging (default False) + bundle gains control_required + effort_term; E3Selector stores `last_raw_scores`; REEAgent `_assemble_control_vector()` writes `_last_control_vector` after e3.select. Bit-identical OFF (contract C4); 889 contracts + 4 new ControlVector contracts; activation smoke v_t=0.5 (forced floor; v_raw=-1.75 EXQ-624a sign/scale issue now visible). | Implemented 2026-06-07 (ree-v3 main). Stage-B C_time<->G_vigor collapse-correlation diagnostic queued via /queue-experiment (claim_ids=[]; pre-registered rho ~ 1.0). |
| SD-058 / MECH-357 (instrumental-avoidance acquisition; ilPFC-analog) | defensive_action.instrumental_avoidance_acquisition -- closes the scaffolded_sd054_onboarding Stage-H / P1 survival-leg gap (V3-EXQ-603g G_H 0/3; goal_pipeline:GAP-2). The instrumental-ACQUISITION side REE was missing: REE had the Pavlovian/defensive REACTION side (SD-035 BLA/CeA salience + MECH-279 PAG freeze) but no learned avoidance. Per Moscarello & LeDoux 2013, active avoidance is the resolution of a Pavlovian-instrumental conflict requiring infralimbic PFC to SUPPRESS CeA-driven freezing; a freeze-only substrate freezes instead of learning to avoid. New module `ree_core/pfc/infralimbic_avoidance_gate.py` (`InstrumentalAvoidanceGate` + config + output; pure-arithmetic regulator, sibling to SD-035/MECH-279/MECH-313/MECH-320/MECH-342). Three pieces (all behind `use_instrumental_avoidance` default False): (a) per-candidate ASSERT action-pathway score-bias penalising the no-op/freeze class proportional to `effective_efficacy * threat_scale`; (b) ilPFC freeze-suppression gate at the MECH-279 application site (skips the no-op override when `effective_efficacy * threat_scale >= suppression_threshold`); (c) eligibility-trace avoidance-efficacy learning (scalar in [0,1] starting 0; directed action under threat that DROPS z_harm_a credits efficacy via EMA toward 1; freezing/failed-avoidance decays it). PROTECTIVE-SCAFFOLD anneal: `effective_efficacy = max(avoidance_efficacy, scaffold_floor)` with Stage-H curriculum starting high then annealing down (maternal-buffering / Turchetta 2020). Curriculum wiring in `scaffolded_sd054_onboarding.run_hazard_avoidance` under `scaffold_avoidance_driver_enabled` (False default) + `scaffold_avoidance_scaffold_floor_start` (0.8) / `..._floor_end` (0.0); LOAD-BEARING PREREQUISITE FOUND 2026-06-07: legacy scaffold called sense(body, world) with NO harm args so z_harm_a was None across the entire curriculum -- new flag `scaffold_feed_harm_stream` (False default) feeds env harm_obs + harm_obs_a so PAG/SD-035/SD-058 actually see threat (~0.34 in Stage-H). 912 contracts + 7/7 preflight PASS; 7 new MECH-357 contracts + 4 C13 in test_scaffolded_sd054_onboarding.py. MECH-094 no-op for compute methods + update under simulation_mode. | Implemented 2026-06-07 (ree-v3 main). v3_pending until V3-EXQ-603h Stage-H validation PASSes. V3-EXQ-603h FAILed adjudicated engaged-but-insufficient (gate engaged + suppressed PAG freeze on all INTACT seeds, readiness_met=true, but G_H_INTACT 0/3 not > LESION; scalar avoidance_efficacy decoupled from survival -- seed-43 inversion). Failure_autopsy_V3-EXQ-603h_2026-06-08 confirmed AND clean adjudicated; SD-058/MECH-357 stay candidate / v3_pending UNWEAKENED (claim_ids=[]). Discovered dependency = relief/safety escape-affordance bridge (SD-059 below). |
| SD-059 / MECH-358 (escape-affordance bridge: relief/safety -> directed escape) | defensive_action.escape_affordance_bridge -- closes the V3-EXQ-603h directed-escape gap. SD-058 suppressed the MECH-279 freeze but `avoidance_efficacy` is a GLOBAL SCALAR that only penalises the no-op class; `compute_action_bias` by design "does NOT compute the escape direction". 603h: gate suppressed freeze on all INTACT seeds but G_H_INTACT=0/3; seed-43 reached scalar efficacy 0.633 yet survived WORST (11.0) -- agent un-froze without acquiring a DIRECTED escape. Per Moscarello & LeDoux 2013: active avoidance also needs the LA/BA->NAcc relief/safety action-credit half. REE owned relief (MECH-302/SD-050) + safety (MECH-303/304/SD-052/SD-051) but they were UNWIRED to avoidance. New module `ree_core/pfc/escape_affordance_bridge.py` (`EscapeAffordanceBridge` + config + output; pure-arithmetic, sibling to SD-058 in `ree_core/pfc/`). Extends MECH-357's scalar `avoidance_efficacy` into a per-FIRST-ACTION-CLASS credit table (the minimal V3 rendering of `escape_affordance[action]` -- directed escape direction in the discrete action space). Two togglable halves (so the 4-arm validation dissociates): RELIEF half (directed action under threat that DROPS z_harm_a credits `relief_affordance[action_class]` toward 1) + SAFETY half (directed action after which threat is absent credits `safety_affordance[action_class]`). Approach bonus under FUTURE threat: per-candidate NEGATIVE (favoured) score-bias toward each candidate whose first-action class carries combined affordance credit; the no-op/freeze class never gets a bonus. THREE guards: bias_scale clamp + threat-context gate (exactly zero when safe) + per-tick leak (no pathological habit loop). DISTINCT from reflexive escape (SD-037 orexin / MECH-281 urgency) and from the generic relief/safety rows (MECH-302/303/304 fire on the CURRENT state); this is learned-efficacy-gated DIRECTED approach binding an action to relief/safety for future-threat use. Bit-identical OFF; 8 new SD-059 contracts in test_sd_059_escape_affordance_bridge.py PASS. MECH-094 no-op under simulation_mode. | Implemented 2026-06-08 (ree-v3 main 6c856a5 post-603i E2 escape-affordance linker reuse/readout over E2). v3_pending until the 4-arm validation EXQ PASSes. V3-EXQ-603i (ARM_BASE_IA_ONLY / ARM_RELIEF_BRIDGE / ARM_SAFETY_BRIDGE / ARM_RELIEF_SAFETY_BRIDGE + nav-competence positive control) FAILed precondition_unmet -- the diagnostic self-route flagged for adjudication via failure-autopsy; substrate_queue CREATE escape-affordance-bridge (priority 1) carried over to the post-603i successor scaffolds below. |
| MECH-189 super-ordinal goal-anchor ContextMemory writes substrate | development.super_ordinal_goal_anchor_writes -- closes infant_substrate:GAP-11. New `SuperOrdinalGoalMemory` in `ree_core/goal.py` (agent-owned + cue-indexed `key=z_world context / value=z_goal anchor`; NOT reset per episode). Write hook at `agent.update_z_goal` (child phase) gated on the MECH-189 conjunction `(high-salience drive-modulated benefit) AND (high contextual complexity)`; complexity policy pluggable via `super_ordinal_complexity_mode` (default novelty-vs-anchors; recurring high-salience contexts REINFORCE their existing anchor on salience alone). Read hook seeds the adult z_goal via `GoalState.cue_pull` from the retrieved anchor when z_goal sits below floor. 985/985 contracts + 8 new MECH-189 contracts + 7 preflight PASS; bit-identical OFF. | Substrate landed 2026-06-09 (ree-v3 main c7ac035; REE_assembly master f212523d80 + design doc `docs/architecture/mech_189_super_ordinal_goal_anchors.md`). V3-EXQ-588c readiness diagnostic queued (LOAD-BEARING C1 = ADULT z_goal seeding DISCRIMINATION ARM_ON vs ARM_OFF). MECH-189 stays candidate (substrate landing -- implementation_note only). |
| MECH-294 multi-content theta-burst packet + compose-coherence amend | binding.theta_burst_packet -- per-cycle joint binding of {goal, action, risk, state} into a `ThetaPacket`. Compose path reads within-cycle co-binding via `ThetaPacket.currency_coherence()` (fraction of co-temporally-current streams: joint ~1.0 / alternation ~0.25 / shuffled 0.0) so the binding MODE causally reaches E3 behaviour rather than collapsing onto action-only similarity (the 657a coherence-metric autopsy fork). Parameter-free, no trained head, bit-identical OFF; per-candidate RANKING stays in-space action cosine (no cross-semantic-space comparison). Master `theta_packet_compose_into_e3_bias` still default False; new no-op flag `theta_packet_compose_use_joint_coherence` (default True; False = legacy action-only ablation). 8/8 packet contracts + 7/7 preflight PASS. | Substrate landed 2026-06-09 (ree-v3 main 53d4c75 + amend af97b23; REE_assembly master 4c680c03f5 -- claims.yaml MECH-294 implementation_note + `docs/architecture/mech_294_*.md` section 10 update). V3-EXQ-661 substrate-readiness 5-arm diagnostic queued (TV-margin readouts across OFF / JOINT / ALTERNATION / SHUFFLED / ALT-coh-OFF; non-vacuity gate). MECH-294 stays candidate / v3_pending. |
| SD-033b GAP-8 OFC trainable state_bias_head | commitment_closure:GAP-8 mirror of SD-033a GAP-D. `OFCConfig.train_state_bias_head` (default False -> last Linear zeroed, bit-identical OFF) + `bias_head_parameters()` + REEConfig `ofc_train_state_bias_head` (from_dims + agent.py OFC build-site getattr). When True the OFC `state_bias_head` last Linear keeps random init so it trains via E3-gradient REINFORCE, unblocking the deferred behavioural arm. SUBSTRATE-CONSTRAINT recorded: OFC reads only z_world + z_harm (no appetitive / drive input) so the behavioural readout must be AVERSIVE devaluation (`ofc_harm_dim > 0`). | Substrate landed 2026-06-09 (ree-v3 main 382db2c + V3-EXQ-485d queue commit 8839724; REE_assembly master e7dc4f7152 + GAP-8 closure-plan node c5460ebb9f + SD-033b evidence_quality_note c8546ae0ff). V3-EXQ-485d 2-arm substrate-readiness diagnostic queued (frozen vs trainable; load-bearing C2 on head weight-delta-from-init under SP-CEM diversity; PASS unblocks the full GAP-8 behavioural arm, NOT yet queued). SD-033b stays candidate / v3_pending; this row only confirms the trainable arm is wired. |
| Post-603i successor scaffolds (NOT validated substrate) | THREE post-603i successor options inspired by the 603i FAIL, scaffolded behind feature flags as forward-looking experiments only -- NOT replacements for the SD-059/MECH-358 active bridge and NOT changes to V3-EXQ-603i. **(a) Trainable escape-affordance learner** (`trainable_escape_affordance_learner`, module `ree_core/pfc/trainable_escape_affordance_learner.py`): local PyTorch relief/safety heads (shared trunk + action embedding + AdamW) trained on continuous relief targets, response-produced safety targets, extinction targets. Off by default (`use_trainable_escape_affordance_learner`); 23 contracts + diff-clean. **(b) Trainable relief/safety heads upgrade** (ree-v3 main 58535af): upgrades the learner from scalar/prototype tables to actual trainable PyTorch heads with lazy model+AdamW, detached compact state/context + action embedding, optimizer/no-op/simulation/hypothesis guards, prediction-based bounded threat-gated bias, reset persistence, diagnostics. 27 contracts. **(c) E2 escape-affordance linker** (`ree_core/pfc/e2_escape_affordance_linker.py`, ree-v3 main 6c856a5): post-603i REUSE/READOUT over E2 (cerebellar-analog) `E2.world_forward` per user mid-session correction -- NOT a duplicate forward predictor. Reads DETACHED E2 action-consequence features for the executed (prev_z_world, action) pair into escape-affordance viability readouts (harm_delta / threat_termination / safety_transition / refuge_reachability / survival_step) + hippocampal-style per-action viability index (readout only, no trajectory gen/reward/selection) + a bounded threat-gated E3 bias behind `use_e2_escape_linker_e3_bias`. Reuses E2; does NOT duplicate a forward predictor. 949 contracts + 7 preflight + boot matrix PASS. | Scaffolded 2026-06-08 (ree-v3 main 7a0a417 + 58535af + 6c856a5). All three are NOT validated substrate and do not change governance state for SD-058/MECH-357 or SD-059/MECH-358. V3-EXQ-653 (E2 escape-affordance linker readiness microdiagnostic; claim-free, forced-choice 4-action probe over the linker; 4 arms x 3 seeds; readiness gates G0-G8) queued via /queue-experiment as the validation gate; PASS routes back to a 603-lineage full behavioural bridge re-test; FAIL routes to failure-autopsy on this readiness diagnostic. |
| MECH-219 / SD-019b (affective-harm hysteretic integrator; `z_harm_suffering`) | affect.affective_harm_hysteretic_integration -- tier-3 slow controllability-gated SUFFERING load state built on top of the SD-019a tier-2 `z_harm_un` EMA. Pure-arithmetic regulator in `ree_core/affect/harm_suffering_accumulator.py` owning the scalar suffering state `s_t`; the agent builds the `z_harm_suffering` vector in the `z_harm_un` direction at magnitude `s_t`. Dynamics: `u_t = ||z_harm_un|| (+ body_damage_weight*||z_harm_a||)`; `drive_t = (1 - escapability) * u_t (+ pe_gain * unsigned_PE)`; asymmetric `s_t = s_{t-1} + alpha*(drive_t - s_{t-1})` with `alpha_rise >> alpha_fall` (the hysteresis); optional Schmitt bistable latch. Escapability is PLUGGABLE (`constant` / `avoidance_efficacy` reading SD-058 `effective_efficacy()` / `external` validation seam); never sourced from MECH-353 capacity_belief (avoids the `z_harm_a -> capacity -> z_harm_a` loop). Per-consumer redirect flags (AIC / PAG / MECH-091 wired in v1 via `||z_harm_suffering||`; dACC / pACC defined but unwired pending an R^2 measurement against `z_harm_a`). Body-damage fold-in preserves SD-022 evidence. Precondition: requires `use_harm_un=True`. Bit-identical OFF + bit-identical under default `escapability_mode=constant=1.0` (drive collapses to zero). 11/11 new contracts + 7/7 preflight + full contract suite green. | Substrate landed 2026-06-10 (ree-v3 main; design doc `docs/architecture/mech_219_hysteretic_integrator.md` flipped plan-of-record -> IMPLEMENTED). SD-019b stays candidate / v3_pending until the controllability-dissociation falsifier (escapable vs inescapable matched-nociception) PASSes -- queued as a separate /queue-experiment session. |
| SD-034 commitment-closure-control-plane AMEND (env-completion hook + de-commit hold) | governance.closure_operator.commitment_closure_control_plane_amend -- the behavioural-authority amend the SD-034 ClosureOperator lacked on the 603n foraging-competent substrate. Leg A `use_closure_env_completion_hook` routes env `sequence_complete` -> `REEAgent.notify_env_completion(action_class, z_world)` -> `closure_operator.emit_closure` (closes the *c-cohort gap: env emitted completions but nothing routed them into emit_closure; the agent relied solely on the automatic rule_state-stability detector). Leg B `closure_decommit_hold_ticks` installs `BetaGate.apply_refractory(n)` on every closure fire so a closure-coupled release survives >1 tick -- measurable latch-occupancy drop instead of immediate re-commit. Leg C (experiment-side; not substrate): the *d retests set the landed SD-033a GAP-D `lateral_pfc_train_rule_bias_head` so the detector has a magnitude-bearing rule_state, gate readiness on `n_closures>0` reachable, and read de-commitment on a non-cap-pinned statistic. Both substrate legs no-op-default + bit-identical OFF; 1014 contracts + 7/7 preflight PASS. Routed by the confirmed `failure_autopsy_SD-034-closure-cluster_2026-06-12` (V3-EXQ-460c + V3-EXQ-468c). NO flag/confidence change; SD-034 provisional holds, MECH-261 stable, MECH-260 stays candidate (the 460c n_closures=0/nogo=0 was a positive-negative downstream of the unrouted env hook). | Substrate landed 2026-06-12 (ree-v3 main 6fdb111 + REE_assembly master cc49f120a4); V3-EXQ-460d + V3-EXQ-468d validation retests queued (claim_ids SD-034 / MECH-260 / MECH-261 / MECH-268 / MECH-090; readiness gate on `n_closures>0` reachable on >=2/3 completion-engaged seeds -> else substrate_not_ready_requeue, never a false weakens). PASS unblocks the broader SD-034 closure-cluster pending_retest cohort. |
| MECH-423 super-additivity readiness substrate (R2 + R3 + R1) | super_additivity.cross_model_integration_readiness -- the three readouts the EXP-0380 cross-model super-additivity acceptance_checks require, all no-op-default + bit-identical OFF + contract-tested. R2 (iterative-inference convergence on `LatentStack.encode`; generalises the legacy two-pass amortized recognition into a predictive-coding settling loop, tracks `inference_convergence` plain-float dict per-round `rel_delta` + early-stop at `inference_convergence_rel_tol`). R3 (module-tagged interleaved E1<->E2 cross-module consolidation in `SleepLoopManager` / MECH-121; a flag-gated hook runs the default E1+E2 loss set in the offline pass and surfaces `n_updates` + `n_cross_module_traces` + `cross_module_replay_share` + `updates_<name>`; `interleaved` runs one step per module per trace, `blocked` is the catastrophic-interference control). R1 (`shared_latent_gradient_probe` reusable utility -- per-module `d(loss)/d(z_shared)` cosine-coupling probe). MECH-094 simulation-gating on the replay/consolidation path; SAME explicit exception the e2_harm_s writeback uses (per-module optimisers constructed locally over only named-module params). ARC-004 (shared L-space inference machinery) + MECH-121 (NREM consolidation cluster) carry an `implementation_note` only -- PROMOTES NOTHING. MECH-423 stays candidate / v3_pending. | Substrate landed 2026-06-12 (ree-v3 main aff371f + 78fe92c + REE_assembly master 6964f47bf8; design doc `docs/architecture/mech_423_superadditivity_readiness_substrate.md` flipped plan-of-record -> IMPLEMENTED). EXP-0380 flipped `blocked_substrate` -> `proposed`. V3-EXQ-676 substrate-readiness validation queued + ingested into coordinator `/queue/active`. PASS confirms the readiness readouts are non-vacuous on a trained substrate; EXP-0380 super-additivity ablation is the SEPARATE /queue-experiment session this unblocks. |
| SD-034 commitment-closure-control-plane BETA-ENGAGEMENT amend (couple closure -> beta elevation) | governance.closure_operator.commitment_closure_control_plane_beta_engagement_amend -- no-op-default `REEConfig.use_closure_commit_beta_coupling` couples the closure-plane commit (`e3._committed_trajectory is not None`) to the bistable BetaGate elevate path so beta occupancy tracks closure commitment on every seed where one forms, making the Leg-B de-commit refractory's ON<OFF latch-occupancy drop measurable. Closes the V3-EXQ-460e commit-without-beta dissociation (seeds 42/43 committed 2415/2019 steps but `total_beta_elevated=0`; closures fire 7/6 -> coupling resolves it on every seed where a closure forms). `BetaGate.note_closure_coupled_elevation()` + `_n_closure_coupled_elevations` diagnostic; full `should_admit_elevation` AND `_readiness_admits` conjunction preserved (composes cleanly with MECH-090 R-c). Bit-identical OFF; 5/5 new contracts + 7/7 preflight + 1084 contracts pass. PROMOTES NOTHING -- SD-034 provisional / MECH-260 candidate / MECH-261 stable all stay `non_contributory` + `pending_retest_after_substrate` until V3-EXQ-460f scores a contributory PASS. | Substrate landed 2026-06-17 (ree-v3 main f4ceea4 + ccb36db; REE_assembly master bf0e530; design doc `docs/architecture/sd_034_governance_closure_operator.md` + substrate_queue `commitment-closure-control-plane` entry). V3-EXQ-460f (de-commit retest on non-cap-pinned ON<OFF latch-occupancy DV) + V3-EXQ-468e (MECH-090 commit-entry conjunction under the trained head) queued TOGETHER on the amended substrate, both arming `beta_gate_bistable=True` + `use_closure_commit_beta_coupling=True` + Leg-A env hook + Leg-B hold + Leg-C `scaffold_train_rule_bias_head`. Substrate_queue ready stays false. |
| DR-12 / `self_model_v4:SELF-4` (E2 forward-PE -> E3 trajectory-scoring confidence down-weight; FIRST V4 SUBSTRATE BUILD) | ethics_engine_3.pe_conditioned_confidence_weighting -- the FIRST-EVER V4 substrate build (`generation:v4`, OFF the V3 critical path; PROMOTES NOTHING in V3). User-approved via `self_model_v4_plan.md` SELF-4 `graduation_decision_2026_06_16`. A new lever on existing machinery (no learned parameters, no z_self state): in `E3TrajectorySelector.score_trajectory()`, when `use_pe_confidence_weighting=True` and a per-trajectory `e2_forward_pe` is supplied and `pe_confidence_weight != 0.0`, the score (a COST; lower-is-better) is increased by `pe_confidence_weight * penalty(e2_forward_pe)`. Penalty modes `linear` (penalty = pe) / `saturating` (penalty = 1 - exp(-pe/scale) in [0,1)). Threaded PER-CANDIDATE via `select(e2_forward_pe_per_candidate=[K])` so a varying PE can change the committed argmin (a UNIFORM scalar is argmin-invariant; the V3-EXQ-571 deleted-broadcast lesson; C3 contract pins this). Diagnostics: `pe_confidence_active`, `pe_confidence_weight`, `e2_forward_pe_range` (the non-vacuity gate), `pe_confidence_penalty_range`. Config (E3Config + REEConfig.from_dims, all no-op default): `use_pe_confidence_weighting` (False, master) + `pe_confidence_weight` (0.0) + `pe_confidence_mode` ("linear") + `pe_confidence_scale` (1.0). Per-candidate PE source in v1 is CALLER-SUPPLIED via `agent.set_injected_e2_forward_pe()` (the DR-12 pilot is a controlled probe; ecological region-PE auto-source is documented as a separate follow-on). Precedents set: `architecture_epoch = ree_self_model_v1`, `run_id` suffix `_v4`, `V4-EXQ-NNN` namespace (`validate_queue.py` widened to `V<gen>-EXQ`), `owner_exq=V4-EXQ-001` on SELF-4; `check_closure_drift.py` confirmed to skip `generation:v4` plans so V3 closure % stays clean. 8 DR-12 contracts + full suite pass; bit-identical OFF guaranteed. | Substrate landed 2026-06-17 (ree-v3 main f5eba3b + 394ccf4 + c4fc5bd; REE_assembly master 08f92b0d3e; design doc `docs/architecture/dr12_pe_conditioned_e3_confidence.md`; plan node `evidence/planning/self_model_v4_plan.md` SELF-4). V4-EXQ-001 (DR-12 pilot falsifier) RAN PASS on ree-cloud-1 (`dr12_pe_conditioning_changes_selection`; 3/3 diff-flipped, 3/3 uniform-inert; manifest `v4_exq_001_dr12_pe_conditioned_confidence_falsifier_20260617T105251Z_v4` 405ce947, `architecture_epoch=ree_self_model_v1`, on origin/master via the first push 394ccf4). MECH-215 (claim DR-10+DR-12 unblock) stays `candidate` / `implementation_phase=v4` -- DR-12 alone doesn't unblock it (DR-10 + experiments remain). First V4 manifest pending governance review. |
| modulatory-bias-selection-authority AMEND (route upstream-channel range into the bias the authority rescales) | ethics_engine_3.modulatory_bias_selection_authority.route_range_amend -- routed by `failure_autopsy_569f-661-654a_2026-06-10`. The 2026-06-03/06-06 authority rescales the composed `_modulatory_accum`; the 569f/661/654a cluster showed that a channel whose REPRESENTATION carries cross-candidate range still does not move committed action when that range is flattened by the consuming bias head (e.g. zeroed-last-layer SD-033a/b heads) before reaching the accumulator. Fix: new `project_channel_range` helper + `channel_route_bias` param on `e3_selector.select` that folds a unit-normalised per-candidate projection of the channel's representation (SVD on a detached copy of `cand_world_summaries`; identity for already-per-candidate biases) into BOTH scores and `_modulatory_accum` BEFORE the authority's range computation. Source selectable via `modulatory_channel_route_source`: `cand_world_summary` (the 569f world-summary case; reads the ARC-065 GAP-A `e2_world_forward` helper) / `curiosity` / `gated_policy` / `mech295` / `coherence`. P0 readiness diagnostic `modulatory_channel_route_range` lets a retest assert the bias itself carries cross-candidate range BEFORE behavioural scoring (prevents self-routing a false negative). Parameter-free, no trained head, no phased training, bit-identical OFF; default `none`. New contracts cover project_channel_range identity / range-preservation / OFF bit-identical / ON P0 range + scores-reach-rescaled-accumulator. HONEST SCOPE: routing makes the channel range REACH + MOVE the committed argmax (the readiness property), NOT necessarily move it beneficially -- that's the channel's own trained-head retest. | Substrate landed 2026-06-10 (ree-v3 main; design doc `docs/architecture/modulatory_bias_selection_authority.md` route-range amend section + `substrate_queue.json` modulatory-bias-selection-authority entry with 3 failure records 569f/661/654a applied 2026-06-10 governance). V3-EXQ-663 substrate-readiness diagnostic ran 2026-06-10 (reviewed in the 16:24Z + 17:23Z governance cycles); per-claim behavioural retests of ARC-065 / MECH-294 / ARC-062 / MECH-309 / MECH-341 / MECH-314 / MECH-320 stay candidate / v3_pending until queued separately. |
| modulatory-bias-selection-authority AMEND (TOP-K shortlist mode; small fixed F-best eligible set with rotating membership) | ethics_engine_3.modulatory_bias_selection_authority.top_k_shortlist_amend -- routed by `failure_autopsy_V3-EXQ-569h_2026-06-16`. The 2026-06-15 margin shortlist admitted ~7 of ~8 candidates (a state-STABLE near-whole eligible set) so within-set argmin collapsed to the channel's global favourite. New `modulatory_shortlist_mode='top_k'` + `modulatory_shortlist_k=3` lever uses a SMALL fixed top-k by primary score whose MEMBERSHIP ROTATES with state; within-set argmin of the routed `_modulatory_accum` (committed) / softmax-multinomial (uncommitted) selects the winner. SAFETY preserved at any internal strength -- only the k F-best are eligible, so a clearly-harmful candidate is never selectable. Bit-identical OFF (mode default `margin`); 22/22 conversion + shortlist contracts pass (18 prior + 4 new top-k). | Substrate landed 2026-06-16 (ree-v3 main; design doc `docs/architecture/modulatory_bias_selection_authority.md` TOP-K shortlist amend section). V3-EXQ-569i (the gated GAP-A behavioural falsifier on the now-validated TOP-K conversion + e2_world_forward pool divergence) PASSed 2/3 seeds on the 569 lineage (thin clearance over collapsed-proposer + matched-noise controls); the **V3-EXQ-689a gap-blind controls falsifier** rides on top of 569i + ARC-065 GAP-A. PROMOTES NOTHING -- ARC-065 stays provisional / substrate_ceiling / pending_retest_after_substrate. |
| MECH-439 F-dominance conflict-grade levers (Factor A conflict-graded shortlist width + Factor B gap-scaled commit-T) | ethics_engine_3.f_dominance_conflict_grade -- the conversion-ceiling campaign's V3 fix for the live root (F-dominance). V3-EXQ-571 established the primary harm/goal F monopolises ~88-89% of E3 committed-selection variance, unchanged by the full diversity stack -- so every diversity channel reaches the E3 accumulator but cannot move the F-dominated committed argmax. ONE shared quantity `gap_norm` in [0,1] computed once from raw_scores (top-F to second-best gap / raw_score_range) drives BOTH new levers. **Factor A** (`modulatory_shortlist_conflict_graded`) grades the top-k shortlist width: `k = round(k_max - (k_max-1)*gap_norm)` clamped to [1,K] -- near-ties widen to `k_max`, decisive F-gaps shrink to k=1; F gates ELIGIBILITY only, ABSENT from the within-set arbitration the routed modulatory channel performs. **Factor B** (`use_gap_scaled_commit_temperature`) softens the committed hard-argmin into multinomial(softmax(-q/T_eff)) with `T_eff = base + alpha*(1 - gap_norm)` -- near-ties hot, decisive gaps cold; F-bounded eligible set guarantees safety at any internal strength. Both no-op default + bit-identical OFF; 12 new contracts pass (within-set arbitration safety, gap-scaling monotonicity, dead-band hold, gap-blind reduces to hot flat softmax control). NO encoder head, no phased training; pure-arithmetic. | Substrate landed 2026-06-18 (ree-v3 main; design doc `docs/architecture/modulatory_bias_selection_authority.md` MECH-439 conflict-grade section; synthesis `evidence/planning/conversion_ceiling_phase0_synthesis_2026-06-18.md`). **V3-EXQ-689a** queued 2026-06-19 (`MECH-439 conflict-grade GAP-BLIND control-arm falsifier`; supersedes V3-EXQ-689 which self-routed substrate_not_ready_requeue with gap_spread=0 -- the redesign LEADS with gap-BLIND control arms ARM_FIXED_KMAX / ARM_FIXED_HOT_T so the `gap-concentrated vs uniform lift` becomes an arm contrast that sidesteps the uncomputable per-gap-bin regression). MECH-439 stays candidate; this is its first falsifier. |
| MECH-294 AMEND (per-candidate co-binding coherence; cross-candidate-range rendering so the route-range authority + 569i top-k can carve) | latent.multi_content_theta_packet.per_candidate_coherence_amend -- routed by V3-EXQ-661 + the substrate-ceiling-lifted triage 2026-06-19 (verdict (b) MECH-294 STILL CEILINGED). Prior `currency_coherence()` was a SCALAR; the 2026-06-09 compose path's `bias = -scale * coherence * cosine(cand, action_proposal)` produced a per-candidate PATTERN identical across binding modes (seal sets `action_proposal = _win_action` regardless of mode) and only scaled magnitude -- erased by the route-range authority's unit-range normalisation (joint == alternation; shuffled = 0). Fix (no-op-default, bit-identical OFF): `seal()` now binds, per V_s-gated content stream, the action_proposal CO-BOUND WITH that stream this cycle (`action_refs` + `coherence_weights`); new `compose_per_candidate_coherence` produces a mode-distinct per-candidate PATTERN (JOINT full range / ALTERNATION live + held-prior weighted distinct pattern (smoke joint-vs-alt cosine 0.82, NOT a uniform scaling) / SHUFFLED none ~0). Composes via the existing `coherence` modulatory_channel_route_source so the per-candidate range is the routed bias the authority rescales + the 569i top-k shortlists. Flags `theta_packet_compose_per_candidate_coherence` (False default) + `theta_packet_coherence_hold_weight` (0.5). Pure-arithmetic; no learned parameters; 7 new contracts. PROMOTES NOTHING -- MECH-294 stays candidate / substrate_ceiling / v3_pending; the 2026-04-26 governance hold stands until the behavioural falsifier on this channel PASSes. | Substrate landed 2026-06-19 (ree-v3 main a154664; REE_assembly master 33ce8424b1; design doc `docs/architecture/modulatory_bias_selection_authority.md`). NO behavioural EXQ queued in this pass (the MECH-294 falsifier on this channel is gated on candidate first-action diversity AND is a separate `/queue-experiment` step). |
| SD-034 commitment-closure-control-plane DE-COMMIT-AUTHORITY MAGNITUDE amend (committed-run-scaled Leg-B refractory) | governance.closure_operator.commitment_closure_control_plane_de_commit_authority_magnitude_amend -- routed by `failure_autopsy_V3-EXQ-460f_2026-06-18`. The 2026-06-17 beta-engagement amend ran (all 4 readiness gates cleared; C2 de-commit DV ran for the first time -- PASS seed 42 -33.5%, FAIL 2/3) but on strong-natural-commit seeds the closure->beta coupling was INERT and the Leg-B 5-tick refractory was SWAMPED by 530-560 natural-commit elevated steps. NEW levers `closure_decommit_hold_scale_with_run` (default 0.0) + `closure_decommit_hold_max_ticks` (0 = uncapped) scale the de-commit hold by committed-run length captured at fire BEFORE `release()` (which resets the counter), clamped to `max_ticks`. BetaGate gains `_committed_run_length` counter + `committed_run_length` property + `sd034_committed_run_length` get_state key; ClosureOperator `_fire()` installs the scaled hold. Both flags no-op default + bit-identical OFF; 6 new contracts pass. PROMOTES NOTHING -- SD-034 provisional, MECH-260 candidate, MECH-261 stable. | Substrate landed 2026-06-19 (ree-v3 main; design doc `docs/architecture/sd_034_governance_closure_operator.md`). V3-EXQ-460g queued + ran 2026-06-19 (de-commit retest arming the magnitude lever on the beta-engagement-amended substrate) -- adjudicated FAIL/non_contributory (the de-commit MAGNITUDE lever was SELF-DEFEATING: scaled refractory at 60-tick cap pins beta + suppresses its own coupling certifier `sd034_n_closure_coupled_elevations` (the 460f-prescribed non-vacuity gate)). |
| SD-034 commitment-closure-control-plane REFRACTORY-INDEPENDENT commit-intent counter amend (decouple the de-commit lever from its non-vacuity metric) | governance.closure_operator.commitment_closure_control_plane_refractory_independent_amend -- routed by `failure_autopsy_V3-EXQ-460g_2026-06-19`. The 460f-prescribed coupling non-vacuity gate keys on `sd034_n_closure_coupled_elevations`, counted only on the bistable elevate transition -- but the 460g committed-run-scaled de-commit-MAGNITUDE lever apply_refractory cap 60 blocks re-elevation, so scaling de-commit authority UP suppresses its own coupling certifier (counter collapsed 36 -> 0 on seed 42). New `BetaGate.note_closure_commit_intent()` + `_n_closure_commit_intent` counter + `sd034_n_closure_commit_intent` get_state key, called BEFORE the elevate gate so the closure-plane commit INTENT is certified every tick a closure-coupled commitment forms WITHOUT a natural running_variance crossing -- regardless of whether the latch is already elevated OR refractory-blocked. The new counter is the MECH-445 coupling-engagement certifier the MECH-446 magnitude lever cannot zero. Bit-identical OFF (rides `use_closure_commit_beta_coupling`); 3 new contracts pass. | Substrate landed 2026-06-19 (ree-v3 main; design doc `docs/architecture/sd_034_governance_closure_operator.md`). **V3-EXQ-460h** queued (supersedes 460g; the new MECH-446 de-commit-magnitude scored DV gates non-vacuity on `sd034_n_closure_commit_intent > 0`, NOT the refractory-suppressed coupled-elevations counter; `claim_ids=[MECH-446]`). Companion `/claim-synthesis` 2026-06-19 DECOMPOSED SD-034 (REE_assembly master 6a35087fd6): SD-034 narrowed + new claims **MECH-445** closure->beta coupling engagement + **MECH-446** de-commit-authority magnitude registered. SD-034 closure cluster `pending_retest` reroute stands. |
| SD-049-PHASE-2 BOUNDED kappa raise + deeper standing spread amend | substrate.sd_049_phase_2.bounded_kappa_amend -- routed by `failure_autopsy_V3-EXQ-514s_2026-06-18`. 514s established lever (b) standing-differential-depletion WORKED (enriched_spread 1.0 / mean_drive_spread_max 0.211 vs 514q's equalised ~0.006); residual shortfall is lever (a) kappa-SCALE on argmax-flip-gated WL metric. Calibration only -- NO ree_core default change, NO new flag (the 2026-06-17 levers `incentive_drive_kappa_scale` + `per_axis_restoration_fraction` both bit-identical OFF). Bounded operating point validated against the live `wanting()` formula: kappa_scale=12.0 (eff_kappa=24) flips a moderate 1.0-vs-0.6 base_value gap under realistic standing spread (~0.25) but HOLDS a clearly-larger 10x gap (1.0-vs-0.10) -- drive carves near-ties without overriding decisive base_value. Two new invariant guard contracts: **C7** OFF-floor-hard-zero (kappa multiplies ONLY the drive term -> zero drive => wanting == base_value at any kappa_scale; bank-disabled WL floor stays hard-zero) + **C8** bounded / MECH-229-leg-(a)-intact (kappa_scale=12 does NOT flip a clearly-larger 10x base_value gap). 9/9 SD-049-PHASE-2 contracts pass. | Substrate landed 2026-06-19 (ree-v3 main bc2d084; REE_assembly master 218bb78bdc; design doc `docs/architecture/sd_049_multi_resource_heterogeneity.md` bounded-raise amend section + `substrate_queue.json` SD-049-PHASE-2). **V3-EXQ-514t** queued (supersedes V3-EXQ-514s; arms kappa_scale=12.0 + restoration_fraction=0.15 only on the WL-scoring env; pre-registers natural drive-coupled delta >= max(k*pstdev,0.15) on >=2/3 seeds as the MECH-436 `substrate_ceiling -> supports` criterion). MECH-229 leg (a) wanting!=liking + MECH-436 unchanged status. |
| SD-061: difficulty-gated proposal-entropy regulator (stuck-state detector + transient CEM proposal-widening; MECH-343 blocker part 2 / Q-056) | control_plane.difficulty_gated_proposal_entropy -- the substrate MECH-343 was substrate_conditional on (blocker part 1 = modulatory-bias-selection-authority, NOW implemented via 569i top-k). Two coupled no-op-default modules: (1) `ree_core/cingulate/stuck_state_detector.py` integrates goal-progress stall + E3 first-action score margin + committed-action-class lock-in + dACC choice_difficulty into a graded `stuck_score` in [0,1] guarded by goal salience (the stuck-WITH-goal distinction); asymmetric EMA (rise >> fall) -> fast rise, slow decay; (2) `ree_core/policy/difficulty_gated_proposal_entropy.py` maps `stuck_score` to a transient PROPOSAL-layer gain: `extra_candidates = round(candidate_widen_max * s)` + `temperature_gain = 1 + temperature_gain_max * s`. Identity at s=0. `_e3_tick` applies the gain to `HippocampalModule.propose_trajectories` (`num_candidates += extra`; `differentiable_cem_temperature *= gain`, transient, restored in finally). Scoring + commitment (MECH-090/342) + selection authority (569i top-k / MECH-341) UNTOUCHED -- a hard problem widens proposals, not behaviour. Pure-arithmetic; no learned parameters; 8 new contracts pass. **PROMOTES NOTHING** -- MECH-343 stays candidate / substrate_conditional / v3_pending; claims.yaml registers SD-061 as a new substrate. | Substrate landed 2026-06-19 (ree-v3 main; design doc `docs/architecture/sd_061_difficulty_gated_proposal_entropy.md`; triage `evidence/planning/q054_q055_q056_buildability_triage_2026-06-19.md`). Q-056 falsifier (3-arm off / stuck-gated / always-high) queued as a separate /queue-experiment session once a substrate-readiness diagnostic PASSes. |
| ETHICS-PERIMETER Phase 0 (new `epistemic_category = governance_rule` + 20 SENT-* / GOV-* standing-gate claims + node ethical metadata carry-forward) | governance.ethics_perimeter -- consolidates the 11 ethics thought intakes (sentience/welfare register + creation-ethics + pre-meaning valley + anti-retrospective-justification + assembly-routing + responsible-release + external-framework + health-data + security-misuse + AI-welfare-crosswalk + ethics-process-translation) into REE's claim-governed loop. Spine: ethical agency != moral patienthood -> welfare is a SEPARATE governance dimension from epistemic confidence, progressively binding by version, no capability release without care release. Schema: new `epistemic_category=governance_rule` added to `validate_claims.py` VALID_EPISTEMIC_CATEGORIES + `build_experiment_indexes.py` EPISTEMIC_CATEGORIES (+ resolver / gating docstrings) + CLAUDE.md table row -- it is non-`standard` so the indexer suppresses promote/demote/narrow automatically (conflict alerts may still fire). New `claim_type=governance_rule` (outside SUBSTRATE_CLAIM_TYPES so it does NOT enter the substrate-status map). Registered 20 standing-gate claims in claims.yaml (844 total): SENT-0..15 (welfare boundary statement, indicator matrix, welfare budget, combination gate, welfare-preserving design, denial audit, external review, creation ethics, minimal necessary suffering, care obligation, pre-meaning valley, anti-retro-justification, refusal channel, assembly routing, responsible release, AI-welfare crosswalk) + GOV-EXT-1 (external framework crosswalk EU AI Act / CoE / NIST RMF / ISO 42001+23894 / OECD) + GOV-HEALTH-1 (clinical-use prohibition + DPIA/Belmont/Helsinki/CIOMS/WHO/MDR/3Rs) + GOV-SEC-1 (OWASP GenAI / MITRE ATLAS misuse gate) + GOV-PROC-1 (ethics-as-process). Each: candidate / `blocks_v3_green_board=false` / progressive `binds_at_version` v3..v6. CARRY-FORWARD: GOV-PROC-1 §2 implemented via per-node `ethical_metadata` field set (welfare_relevance / applicable_ethics_gates / requires_welfare_review / forbidden_combinations), 50 welfare-relevant nodes tagged across 16 generation:v4/v5/v6 plans; V3 nodes untouched. **NON-BLOCKING for V3 green-board 2026-07-19**; Phases 1-3 (V3 boundary statement + docs/governance registers + release/legal/security docs) deferred to the plan. | Substrate landed 2026-06-19 (REE_assembly master 7023c94bab + 2012e7a8cb; plan-of-record `evidence/planning/ethics_perimeter_plan.md`; closure scripts unchanged -- V3 80.4% / drift 0/7/0 IDENTICAL before/after the node-metadata tagging). GOV-HEALTH-1 public-repo data audit ran CLEAN 2026-06-19 (zero patient-identifiable data in any of the 9 public Latent-Fields REE repos; verification doc `docs/governance/public_repo_data_audit_2026-06.md`). |

SD-003 (two-pass counterfactual self-attribution) was **superseded 2026-04-18** after 28
accumulated FAILs across its two-pass counterfactual architecture. The successor layer is:
- **MECH-256** (general single-pass forward-model comparator, stream-agnostic; Frith/Shergill/
  Haggard/Blakemore biology)
- **SD-029** (concrete z_harm_s instantiation of MECH-256; event-conditioned test queued as
  V3-EXQ-433, next-up priority)
- **MECH-257** (dual-function single-substrate E2: comparator vs evaluator, controller-gated;
  single-substrate-with-gated-readout favoured over dual-substrate per Diba/Buzsaki 2007 +
  Dragoi/Tonegawa 2011)
- **SD-030** (z_self per-stream comparator, V4-deferred) and **SD-031** (z_world, V4-deferred)

EXQ-030b's original validation (world_forward_r2=0.947, attribution_gap=0.035) remains a valid
world-pipeline result but does not transfer to the z_harm_s topology. Architecture doc:
`REE_assembly/docs/architecture/self_attribution_per_stream.md`.

### Experiment Status

- **2026-06-21T01:10Z nightly read (scheduled `/update-docs`).** ARC-107 BG-selector
  constitution build day. `evidence/experiments/` flat top-level now holds
  **458 `v3_exq_*.json` manifests + 1 `v4_exq_*.json`** on disk (frontier through
  V3-EXQ-696 + V4-EXQ-001; +10 since the 2026-06-20 read). Legacy fleet
  `runner_status.json` snapshot held at the **840 dedup completion records** (V3
  subset 811 = PASS 267 / FAIL 424 / ERROR 87 / UNKNOWN 32 / INCONCLUSIVE 1) -- the
  per-machine Phase-3 cards under `runner_heartbeats/` + `runner_status/` lead the
  legacy single-file tally. **Pending review queue (regenerated 2026-06-20T21:45Z):
  1 item** -- V3-EXQ-696 SD-033b / MECH-263 OFC outcome-prediction ceiling retest
  (FAIL 2026-06-20T21:29Z; freshly queued via the 2026-06-20 `/governance` ceiling-audit
  may-have-lifted routing alongside V3-EXQ-695). **Currently queued
  (`experiment_queue.json` items[]): 6 items, 5 claimed + 1 pending** -- (a)
  **V3-EXQ-460i** rung-6 commit/release-DURATION lever de-commit falsifier
  (graded natural-commit-occupancy release tested via 460h disjoint-certifier
  recovery; claim_ids=[MECH-446, MECH-445]; cloud-2; priority 410); (b) **V3-EXQ-689c**
  MECH-439 Factor-B-alone (gap-scaled commit-T) isolation retest (the 689a
  decomposition Factor-B sub-lever check; kept queued per the 2026-06-20
  biology-fidelity steer as PURE NO-AUTHORITY confirmatory data; cloud-4; priority
  410); (c) **V3-EXQ-689d** MECH-448 / ARC-107 rank-preserving F->eligibility
  demotion falsifier (the 689a-successor and constitutional LEAD lever; on the
  GAP-A-ready foraging substrate; claim_ids=[MECH-448]; DLAPTOP-4.local;
  priority 410); (d) **V3-EXQ-693a** SD-049 Phase-2 4-arm substrate-gradient
  measurement re-issue (ports the working 514t info-tag WL-scoring harness so R3
  WL-fireable non-vacuity guard fires; supersedes 693; claim_ids=[SD-049, SD-015];
  cloud-1; priority 80); (e) **V3-EXQ-695** ARC-062 / MECH-309 modulatory-bias
  monomodal-collapse retest on the implemented modulatory / rule channel (569i
  top-k; route-range P0 gate; supersedes 654g; claim_ids=[MECH-309, ARC-062];
  cloud-3; priority 50); (f) **V3-EXQ-696** SD-033b / MECH-263 OFC-analog
  outcome-prediction substrate-ceiling retest (3-arm trained vs OFF vs matched-noise;
  route-range P0 gate; FAILed 2026-06-20T21:29Z, pending governance review;
  cloud-3; priority 50). **Substrate / governance landings since the
  2026-06-20T01:10Z spec sync (one major architectural pivot):** (1) **MECH-448 /
  ARC-107: rank-preserving F->eligibility demotion (LEAD lever of the
  basal-ganglia E3-selector constitution) landed** (ree-v3 main 4c9b3c9;
  REE_assembly master 5bdf97c105) -- the user-adjudicated 689a failure-autopsy's
  "elevate the constitutional build" branch (the conflict-grade near-tie
  parametric family MECH-447 is exhausted). New `_f_eligibility_envelope` graded
  divisive-normalisation envelope (rank-preserving, absolute share floor 0.30
  default; `f_demotion` mode of the shortlist-then-modulate block reuses the
  existing within-eligible `_modulatory_accum` arbitration with F REMOVED from the
  final argmin); 3 E3Config flags `use_f_eligibility_demotion` /
  `f_eligibility_envelope_floor` / `f_eligibility_dn_sigma` no-op-default,
  bit-identical OFF; 10 new contracts + 8 preflight + 1158 full contracts PASS;
  THE FIRST major worked application of ARC-106 (brain-like construction:
  divergence ledger anchors the rank-ALTERING-at-commit departure from canonical
  DN; psychiatric-failure-mode column: envelope-too-wide -> impulsivity/mania,
  envelope-too-tight -> bradykinesia/avolition). PROMOTES NOTHING (MECH-448
  stays candidate). V3-EXQ-689d (claim_ids=[MECH-448]) is the falsifier.
  (2) **Commit/release-DURATION lever: graded natural-commit-occupancy release
  landed PARALLEL to MECH-448** (ree-v3 main ab2c1a9; REE_assembly master
  fd98756f87) -- rung-6 of `f_dominance_conversion_ceiling`; BG-3 SYNTHESIS
  divergence D1 says biology times the commitment hold with a GRADED BG/pallidal
  urgency (Thura/Cisek 2022) and/or maintenance co-extensive with the executing
  action (Jin 2014), NOT a fixed refractory clock. New
  `ree_core/policy/natural_commit_urgency.py` regulator (reuses
  `BetaGate.committed_run_length`; NO parallel latch -- ARC-106 G2 reuse-before-
  duplicate guardrail); two D1-faithful modes (gap-scaled urgency LOAD-BEARING;
  action-extent Jin sibling); both togglable so the V3-EXQ-460i falsifier
  discriminates which lifts. NOT touched: e3_selector (clean separation from
  MECH-448), claims.yaml (PROMOTES NOTHING). 10 new contracts + 1169 full PASS;
  agent smoke OFF 0/20 vs ON 20/20 ticks demote-at-commit with bit-identical OFF.
  (3) **ARC-107 + MECH-449 + Q-078 elevated** by the 689a autopsy (the BG-constitution
  spine + Go/No-Go follow-on (double-gated) + Q-078 umbrella); MECH-439
  non_contributory + substrate_ceiling + pending_retest_after_substrate;
  MECH-447 weakened+split; SD-034 / MECH-260 / MECH-261 / ARC-062 / MECH-309 stay
  candidate / substrate_ceiling / pending_retest. (4) **Targeted lit-pulls grounding
  ARC-107 + BG-3** -- 5 new PubMed anchors in
  `targeted_review_connectome_mech_439/` (D1/D2 Go/No-Go opponency Kravitz 2010;
  Mink 1996 focused-selection + surround inhibition; Chevalier-Deniau 1990
  disinhibition-as-permission-to-commit; Hikosaka 2000 SNr->SC permission gate;
  Maia/Frank 2011 Go/No-Go imbalance -> Parkinson's/OCD/Tourette's/ADHD -- the
  psychiatric-failure-mode column ARC-106 mandates) + new dir
  `targeted_review_commit_release_duration_latch/` (5 anchors for BG-3
  commit/release-DURATION: Resulaj/Shadlen 2009 change-of-mind; Jin/Tecuapetla/Costa
  2014 BG start/stop+sustained maintenance; Thura/Cisek 2022 graded BG/pallidal
  urgency D1 anchor; Loh/Rolls/Deco 2007 one-stability-parameter -> both psychiatric
  poles; Seif 2025 catatonia=No-Go over-pressure over-maintenance pole). Both
  reach ARC-106 grounding-ladder L2 with divergence ledgers + two-poled psychiatric
  columns. (5) **MECH-442 decision packet biology-fidelity-governs-build steer
  recorded** (REE_assembly master feb5184) -- Section 7 ARC-106 anti-shortcut
  corollary: do not let a cheap parametric near-tie lever substitute for / skip
  the biologically-faithful build, even when it moves a metric; V3-EXQ-689c stripped
  of decision authority (kept as confirmatory data only). (6) **closure-link-validator
  wired into governance.sh** -- new `scripts/check_closure_links.py` flags truly-
  dangling `depends_on` / `cross_plan_link` refs so closure-map edge typos surface
  automatically instead of silently dropping in `closure.html`. (7) **Goblin tale
  episode "The Blow That Lit the Seam" landed** in `ree-paper/fantasy/` + the
  public `REE_assembly/docs/ree_for_my_parents.md` mirror -- chronicle-first /
  creative-second (Part-13 rule), failure-honest, constitution unratified, gate
  left shut, names no soul; provenance file
  `goblin_chronicle_2026-06-20_boss_gate_constitution.md`. (8) **proposal-gen skip
  resolved/oodomain/derivational/governance_rule claims** in
  `build_experiment_indexes.py` -- Q-020/Q-035/Q-079 (and the new SENT/GOV
  governance_rule claims) no longer mint experimental EXP-* proposals. **Bottleneck
  (continuation):** **MECH-439 F-dominance conversion ceiling is the live root
  choke**; the parametric near-tie family is exhausted and the response is the
  ARC-107 BG-constitution build (MECH-448 lead lever now in flight via V3-EXQ-689d
  on the GAP-A-ready substrate, with the commit/release-DURATION lever in parallel
  via V3-EXQ-460i). Constitutional path PRIMARY; V3-EXQ-689c kept as cheap
  confirmatory data on the dispreferred parametric path. The 695 / 696 ceiling-audit
  retests are the may-have-lifted side bets (696 FAIL on cloud-3 confirms the OFC
  channel is still substrate-ceiling'd at this conversion-ceiling stage; 695 in
  flight). **ETHICS-PERIMETER Phase 0 datum** stays on the record (Phases 1-3
  deferred; NON-BLOCKING for V3 green-board 2026-07-19).

- **2026-06-20T01:10Z nightly read (scheduled `/update-docs`).**
  Heavy-substrate day; the live root choke is now explicit. `evidence/experiments/`
  flat top-level holds **448 `v3_exq_*.json` manifests + 1 `v4_exq_*.json`** on disk
  (frontier through V3-EXQ-693 + V4-EXQ-001). Legacy fleet `runner_status.json`
  unchanged at the **840 dedup completion records** snapshot from earlier in June
  (under Phase 3 the per-machine cards under `runner_heartbeats/` + `runner_status/`
  lead the legacy single-file tally). **Pending review queue (regenerated
  2026-06-19T21:50:12Z): 0 items** -- the 2026-06-19T20:13Z + 21:41Z governance cycles
  closed everything pending. **Currently queued (`experiment_queue.json` items[]):
  6 items** -- (a) **V3-EXQ-689a** MECH-439 conflict-grade GAP-BLIND control-arm
  falsifier (gap-blind controls `ARM_FIXED_KMAX` + `ARM_FIXED_HOT_T` on the
  569i-validated top-k shortlist; supersedes 689; claim_ids=[MECH-439]; priority
  400; first falsifier of MECH-439); (b) **V3-EXQ-460h** SD-034 de-commit retest
  for MECH-446 on the refractory-INDEPENDENT commit-intent gate (supersedes 460g;
  priority 250); (c) **V3-EXQ-693** SD-049 Phase-2 4-arm substrate-gradient
  behavioural validation (V3-EXQ-514l successor; full curriculum on the 603n
  substrate; claim_ids=[SD-049, SD-015]; priority 240); (d) **V3-EXQ-690** Q-054
  ARC-062 minimum trajectory-class diversity floor sweep (priority 230); (e)
  **V3-EXQ-691** Q-055 SD-017 sleep consolidation diversity persistence
  (priority 225); (f) **V3-EXQ-514t** SD-049-PHASE-2 MECH-436 drive-coupling retest
  on the BOUNDED kappa raise (priority 200). **Substrate / governance landings
  since the 2026-06-19T01:10Z spec sync (a heavy substrate day):** (1) **MECH-439
  F-dominance conflict-grade levers landed** (ree-v3 main; design doc
  `docs/architecture/modulatory_bias_selection_authority.md` MECH-439 section +
  synthesis `evidence/planning/conversion_ceiling_phase0_synthesis_2026-06-18.md`)
  -- the conversion-ceiling campaign's V3 fix for the live root: Factor A
  conflict-graded shortlist width + Factor B gap-scaled commit-T, both bit-identical
  OFF, both keying off one shared `gap_norm`; 12 new contracts pass. **V3-EXQ-689a
  is the first MECH-439 falsifier** (689 self-routed substrate_not_ready_requeue
  with gap_spread=0; the redesign LEADS with gap-BLIND control arms so
  gap-concentrated-vs-uniform is an arm contrast sidestepping the uncomputable
  per-gap-bin regression). (2) **modulatory-bias-selection-authority TOP-K shortlist
  amend documented** (2026-06-16 ree-v3 main; the 7th GAP-A autopsy routing) --
  `modulatory_shortlist_mode='top_k'` + `modulatory_shortlist_k=3` lever uses a
  SMALL fixed top-k by primary score whose membership ROTATES with state. **V3-EXQ-569i**
  PASSed 2/3 seeds (thin clearance over collapsed-proposer + matched-noise);
  ARC-065 stays provisional / substrate_ceiling / pending_retest. (3) **MECH-294
  per-candidate co-binding coherence amend landed** (ree-v3 main a154664;
  REE_assembly master 33ce8424b1) -- closes V3-EXQ-661 byte-identical-committed-dist
  by rendering the scalar coherence as a cross-candidate-range per-candidate bias
  so it ROUTES into the route-range authority + 569i top-k can carve. MECH-294
  stays candidate / substrate_ceiling / v3_pending. (4) **SD-034 closure-control-plane
  DOUBLE amend landed (2026-06-19)**: (4a) de-commit-AUTHORITY MAGNITUDE
  (committed-run-scaled Leg-B refractory) -- routed by `failure_autopsy_V3-EXQ-460f_2026-06-18`;
  V3-EXQ-460g RAN FAIL 2026-06-19, autopsy adjudicated it self-defeating (scaled
  refractory pins beta + suppresses its own coupling certifier); (4b) refractory-INDEPENDENT
  commit-intent counter -- routed by `failure_autopsy_V3-EXQ-460g_2026-06-19`;
  new `sd034_n_closure_commit_intent` get_state key + counter called BEFORE the
  elevate gate so the closure-plane commit INTENT is certified every tick a
  closure-coupled commitment forms regardless of refractory. (5) **`/claim-synthesis`
  SD-034 closure cluster DECOMPOSITION landed** (REE_assembly master 6a35087fd6) --
  7-autopsy granularity-debt route: SD-034 NARROWED + new claims **MECH-445**
  closure->beta coupling engagement + **MECH-446** de-commit-authority magnitude
  registered; coupling-measurability-under-refractory REFUSED as a measurement
  property (the 460h fix, not a claim). **V3-EXQ-460h** queued with claim_ids=[MECH-446]
  and a non-vacuity gate on `sd034_n_closure_commit_intent > 0` (NOT the
  refractory-suppressed coupled-elevations counter). (6) **SD-049-PHASE-2 BOUNDED
  kappa raise amend landed** (ree-v3 main bc2d084; REE_assembly master 218bb78bdc)
  -- CALIBRATION only, no new flag, no ree_core default change (kappa_scale 6.0
  -> 12.0 and per_axis_restoration_fraction 0.3 -> 0.15 only on the WL-scoring env
  in the 514t experiment config); two new invariant guard contracts C7 OFF-floor-hard-zero
  + C8 bounded / MECH-229 leg-(a)-intact. **V3-EXQ-514t** queued (supersedes 514s).
  (7) **SD-061 difficulty-gated proposal-entropy regulator landed** (ree-v3 main) --
  closes the MECH-343 substrate-conditional blocker part 2 (Q-056): stuck-state
  detector + transient CEM proposal-widening (extra_candidates +
  differentiable_cem_temperature gain, transient + restored in finally); scoring +
  commitment + selection authority UNTOUCHED -- a hard problem widens proposals,
  not behaviour. PROMOTES NOTHING (MECH-343 stays candidate / substrate_conditional
  / v3_pending). (8) **ETHICS-PERIMETER Phase 0 landed** (REE_assembly master
  7023c94bab + 2012e7a8cb) -- new `epistemic_category = governance_rule` +
  `claim_type = governance_rule` in `validate_claims.py` / `build_experiment_indexes.py`
  + CLAUDE.md table row, 20 standing-gate claims registered (SENT-0..15 welfare
  register + GOV-EXT-1 external framework crosswalk + GOV-HEALTH-1 clinical-use
  prohibition + GOV-SEC-1 misuse gate + GOV-PROC-1 ethics-as-process; all candidate
  / `blocks_v3_green_board=false` / progressive `binds_at_version` v3..v6); per-node
  `ethical_metadata` carry-forward (50 welfare-relevant nodes tagged across 16
  generation:v4/v5/v6 plans). V3 80.4% / drift 0/7/0 IDENTICAL before/after. NEW
  governance dimension separated from epistemic confidence; **NON-BLOCKING for V3
  green-board 2026-07-19**. (9) **GOV-HEALTH-1 public-repo data audit CLEAN**
  (REE_assembly master 13ba5e5e72 + d0e31c3486) -- first verification of the
  ethics-perimeter "no patient-identifiable data in public repos" bright line.
  All 9 public Latent-Fields REE repos scanned; ZERO patient-identifiable data,
  clinical records, real-names+health-context, referral/service data, or secrets;
  visibility-correction landed (5 PUBLIC / 4 PRIVATE; corrected the prior
  unverified assertion that all Latent-Fields repos were public). (10) **Crystallization
  harness guards landed** (ree-v3 main 10dee76; `experiments/_metrics.py`) -- three
  durable MECH-334/INV-074 crystallization-necessity guards extracted from the
  655 inline preflight: `assert_policy_trained` + `assert_ewc_penalty_in_loss` +
  `assert_true_negative_arm0` so the next MECH-334 retest (copy-modify of 655)
  cannot silently regress to the 610c-f no-op (13 fire-on-deliberately-broken
  contracts + 6 happy-path; HarnessGuardError separates wiring bug from scientific
  self-route). (11) **InfantCurriculumScheduler Phase 0->1 crossing-count criterion
  wired** (ree-v3 main) -- closes `infant_substrate:GAP-14` defect (c-2)
  gate-over-permissiveness at the criterion level (591f PASS evidence;
  recommended_criterion=crossing_count). New flag
  `phase_0to1_use_crossing_count` (default False = legacy single-episode SPIKE
  gate) + `phase_0to1_crossing_count_min` (3 = 591f CROSSING_COUNT_MIN); mirrors
  the 591f offline replay EXACTLY (contract verified). (12) **Convergence Demand
  Pipeline HANDOFF-REACTIVATE node executed** (REE_assembly master 1014ee1f35;
  REE_convergence master 4043cce) -- first batch since 2026-02-24, three June
  promotion packets validated gate-ready + adjudicated (the tool only mirrors
  receipts and does NOT write claims; CDQ-001..003 reconciled IN_FLIGHT ->
  COMPLETED + 3 receipts landed both sides). PROCESS-lane closure 72.3% -> 92.3%;
  V3/V4/V5/V6/governance % UNCHANGED. (13) **CDQ-005 MECH-442/443/444
  decide-to-build packets landed** (REE_assembly master f861372e70 + 4f17018435)
  -- MECH-442 (QD-archive single-elite archive selection) verdict (b) DESIGN GAP
  -> CDQ-006 descriptor-variant query filed in REE_convergence; MECH-443
  (priority_weighted_replay_write_selection) verdict (b) DESIGN GAP -> BUILDABLE
  NOW on the landed MECH-319 write primitive; MECH-444
  (staleness_gated_target_refresh_on_replay_write) verdict (c) DON'T-BUILD-yet
  pending a not-yet-existing target-recompute-and-compare primitive. All three
  CDQ-005 entries carry `ceiling_decision: deferred` markers + revisit triggers.
  (14) **ARC-046 V3-EXQ-591 evidence-record drift fix** (REE_assembly master
  75933a8215) -- first-run manifest `evidence_direction: does_not_support` ->
  `non_contributory` per the 2026-05-27 autopsy override (the nested flat manifest
  was outside the indexer's flat-overlay merge path); ARC-046 weakens 1 -> 0,
  exp_conf 0.274 -> 0.0 (substrate_ceiling, NOT weakened). (15) **MECH-439
  F-dominance front stood up** (REE_assembly master 88b13aaf3a) -- closure node
  `behavioral_diversity_isolation:GAP-I` minted (MECH-439 in scope_claims; owner_exq
  V3-EXQ-689a; load-bearing; full fallback ladder); `substrate_queue.json`
  entry `f_dominance_conversion_ceiling` (IGW blocked_by linkage + F-variance
  trajectory + 5-rung action ladder; unblocks 11+ claims); living "Campaign Status
  & Action Ladder" section in the conversion_ceiling synthesis doc. Three
  structurally-different channels (V3-EXQ-654g CRF rule_state / V3-EXQ-485h OFC
  bias / V3-EXQ-569g/682 modulatory bias) all converge on MECH-439 as the single
  shared blocker. (16) **3 confirmed governance cycles** walked V3-EXQ-485h (OFC
  behavioural; non_contributory) + V3-EXQ-460g (de-commit magnitude; non_contributory)
  + V3-EXQ-689 (MECH-439 substrate_not_ready_requeue) + V3-EXQ-654g (ARC-062 GAP-B;
  non_contributory). 485g and 654f -> superseded. SD-034 / MECH-261 / MECH-260 /
  ARC-062 / MECH-309 stay candidate / substrate_ceiling / pending_retest. (17)
  **INV-074 `pending_retest_after_substrate` reconcile** (REE_assembly master
  ee1c3f180d) -- cleared the contradiction between the 2026-06-13 user-decision
  to STOP the 610 cascade (INV-074 substrate_ceiling accepted) and the pending_retest
  flag; future test redirected to the V4 plasticity_neuromodulation_v4 plan, NOT
  a V3 substrate retest. (18) **V3-EXQ-625e queued** (SD-037 axis-b P1b RECALIBRATED;
  supersedes 625d) and **V3-EXQ-693 queued** (SD-049 Phase-2 4-arm substrate-gradient
  validation, the still-owed 514l successor on the 603n-amended substrate).
  **Bottleneck (continuation):** **MECH-439 F-dominance conversion ceiling is the
  live root choke** -- 3 structurally-different behavioural channels (V3-EXQ-654g
  CRF / V3-EXQ-485h OFC / V3-EXQ-569g/682 modulatory) all converge on it; V3-EXQ-689a
  (the gap-blind controls falsifier on the 569i-validated top-k shortlist) is the
  load-bearing falsifier. On PASS -> 11+ downstream claims unblock per the
  f_dominance_conversion_ceiling substrate_queue entry; on FAIL -> routes to
  fallback ladder (rank-preserving F-to-eligibility demotion / commit-T schedules).
  **SD-034 closure-control-plane decomposition + V3-EXQ-460h** is the second
  in-flight gate (MECH-446 de-commit magnitude on the refractory-independent
  commit-intent gate). **SD-049-PHASE-2 V3-EXQ-514t** is the MECH-436 bounded
  kappa retest. **ETHICS-PERIMETER Phase 0 datum** is on the record; Phases 1-3
  deferred (V3 boundary statement + docs/governance registers + release/legal/security
  docs). **V4-EXQ-001 PASS** the first V4 datapoint stands established;
  MECH-215 stays candidate / `implementation_phase=v4`.

- **2026-06-19T01:10Z nightly read (scheduled `/update-docs`).**
  Queue + pending-review churn day with no new substrate landings.
  Yesterday's two queued items (V3-EXQ-468e SD-034
  commitment-vs-contradiction de-commit-hold retest on the
  BETA-ENGAGEMENT-amended substrate; V3-EXQ-514s MECH-436
  drive-coupling retest on the SD-049-PHASE-2 kappa-scale + standing-
  differential-depletion substrate) both ran to completion and
  **FAILed** -- neither the SD-034 closure-control-plane Legs A/B/C
  + beta-engagement amend bundle nor the SD-049-PHASE-2 kappa-scale
  amend produced the pre-registered ON<OFF / `substrate_ceiling ->
  supports` lift this cycle; the 2026-06-18 governance cycle's
  prior closure-control-plane adjudication (468e flagged for
  `/failure-autopsy`; 460f autopsied at 460f to `non_contributory`)
  stands as the load-bearing reading. **Pending review queue
  (regenerated 2026-06-18T08:15:03Z) reads 5 items, all FAIL** --
  (a) `V3-EXQ-468e` (SD-034 / MECH-268 / MECH-090 commit-vs-
  contradiction de-commit-hold retest on the beta-engagement-amended
  substrate; FAIL, flagged for `/failure-autopsy` adjudication per
  the 2026-06-18 governance cycle note); (b-d) `V3-EXQ-688` x3
  (`mech044_hippocampal_relational_binding`; FAIL with diagnostic
  self-route `substrate_not_ready_requeue` adjudicated
  `precondition_unmet` on all three timestamps -- the diagnostic-
  adjudication gate blocks the self-route from driving any
  governance action until a `/failure-autopsy` adjudicates whether
  the precondition gap is substrate or wiring); (e) `V3-EXQ-514s`
  (MECH-436 drive-coupling retest on the SD-049-PHASE-2 kappa-scale
  amend; FAIL, supersedes V3-EXQ-514r; routes back to the kappa /
  standing-differential-depletion calibration line per the pre-
  registered 514r-autopsy off-ramp). 3 diagnostic self-route flags
  carried forward from the 688 cluster. **Currently queued
  (`experiment_queue.json` items[]): 0 items** -- both yesterday's
  queued items drained without successors yet queued; the active
  /failure-autopsy work on 468e (TASK_CLAIMS `failure-autopsy-468e`)
  and the 688 cluster adjudication are the next steps before any
  re-queue. **Substrate / governance landings since the 2026-06-18T01:10Z
  spec sync** are infrastructure-only (no claim-touching substrate
  edits): the queue-experiment skill **audit R7 hardening** landed
  (REE_assembly master 8a4571f) -- the skill now wires the
  `POST /queue/add` coordinator ingress to the live `/queue/add`
  endpoint so queued items reach the coordinator DB without manual
  re-curl steps (a follow-on to the 2026-06-17 IGW R6/R8 reaper
  hardening; closes the queue-experiment session's audit residual);
  the **IGW R6+R8 reaper hardening session closed** (REE_assembly
  master 1aff9ac) -- the WORKSPACE_STATE Recent Work + claim entry
  documents the test green at 50/50 and confirms the generator-
  timeout fallback to the last-good workset is in production; the
  **failure-autopsy 468e session opened** (TASK_CLAIMS
  `failure-autopsy-468e`) per the 2026-06-18T0804Z governance cycle
  routing note flagging 468e for adjudication. `evidence/experiments/`
  flat top-level: **~440 manifests on disk** (frontier through
  V3-EXQ-688 + V4-EXQ-001 the first-ever V4 manifest; the 18->19
  delta absorbs 468e + 688 x3 + 514s + the 2026-06-18 governance
  pending_review walk). Legacy fleet `runner_status.json` unchanged
  at the **840 dedup completion records** snapshot from earlier in
  June (Phase-3 per-machine cards under `runner_heartbeats/` +
  `runner_status/` lead the legacy single-file tally;
  outcome-breakdown unchanged on the legacy file). **Bottleneck
  (continuation):** the **SD-034 closure-control-plane Legs A/B/C +
  beta-engagement V3-EXQ-468e adjudication** is the live in-flight
  call -- the autopsy will decide whether 468e is genuine
  weakens-on-substrate-ceiling or another commit-without-beta /
  precondition_unmet case; **the V3-EXQ-688 MECH-044 hippocampal
  relational binding precondition-unmet cluster** needs the same
  diagnostic-adjudication gate cleared before any 688-successor can
  be queued; **SD-049-PHASE-2 MECH-436 drive-coupling FAIL at
  V3-EXQ-514s** routes back to the kappa / standing-differential-
  depletion calibration line (the 514s-successor stays unqueued
  pending the routing call); **bdiv GAP-A conversion-architecture
  frontier paced by V3-EXQ-569i** (TOP-K shortlist falsifier, still
  in flight on the coordinator) remains the master conversion-
  ceiling choke; **Stage-H harm-pathway stabilization 603q** is the
  only blocking item left on the goal_pipeline GAP-2 survival-leg
  cohort. **V4-EXQ-001 PASS** (2026-06-18 governance walk applied;
  MECH-215 stays candidate / `implementation_phase=v4` -- DR-12
  cleared, MECH-215 awaits DR-10 + experiments) -- the first V4
  datapoint sits established in the record.

- **2026-06-18T01:10Z nightly read (scheduled `/update-docs`).**
  `evidence/experiments/` flat top-level holds **431
  `v3_exq_*.json` manifests** on disk (latest letter frontier
  through V3-EXQ-687 with V4-EXQ-001 the first V4 manifest;
  day-over-day delta over the 2026-06-17 read absorbs the
  /governance walk of 514r/460e/V4-EXQ-001 + the SD-034 closure
  commitment-beta-coupling amend + the version-layering guards
  landing + the crash-before-manifest observability fix + the
  arc_062 GAP-B 654f re-queue + the SD-049-PHASE-2 514s queueing
  + the convergence-demand-pipeline First Action #1 +
  Q-045/MECH-313/MECH-260 4-arm tonic-noise ablation V3-EXQ-687).
  Legacy fleet `runner_status.json` now at **840 dedup completion
  records** (283 PASS / 437 FAIL / 87 ERROR / 32 UNKNOWN / 1
  INCONCLUSIVE; V3 subset 805 = 263 PASS / 424 FAIL / 85 ERROR /
  32 UNKNOWN / 1 INCONCLUSIVE; Phase-3 per-machine cards under
  `runner_heartbeats/` + `runner_status/` lead the legacy
  single-file tally). 52 entries date-stamped 2026-06; most recent
  completed = V3-EXQ-652 PASS 2026-06-08T05:56Z (the file lags the
  Phase-3 DB, which carries the more recent landings via the
  per-machine cards).

  **Pending review queue (regenerated 2026-06-17T23:49:23Z) reads
  2 items** -- (a) `V3-EXQ-625d` (sd_037 axis_b P1b joint
  composite, FAIL with diagnostic self-route
  `substrate_not_ready_requeue` flagged `precondition_unmet`;
  awaiting `/failure-autopsy` adjudication before the self-route
  drives any governance action); (b) `V4-EXQ-001` (the
  **first-ever V4 manifest** -- DR-12 PE-conditioned confidence
  pilot falsifier; PASS unclaimed manifest with `architecture_epoch
  = ree_self_model_v1`; pending governance review). The active
  governance cycle (`governance-20260617T2349Z`) is the named
  reviewer for both. **Currently queued (`experiment_queue.json`
  items[]): 2 items** -- (a) `V3-EXQ-468e` (SD-034 / MECH-268 /
  MECH-090 commitment-vs-contradiction de-commit-hold retest on
  the BETA-ENGAGEMENT-amended substrate
  `use_closure_commit_beta_coupling`; supersedes V3-EXQ-468d;
  claimed by `DLAPTOP-4.local` at 2026-06-17T18:00:06Z; priority
  344; 3 seeds); (b) `V3-EXQ-514s` (MECH-436 drive-coupling retest
  on the SD-049-PHASE-2 kappa-scale + standing differential-
  depletion substrate; supersedes V3-EXQ-514r; claimed by
  `ree-cloud-3` at 2026-06-17T16:24:32Z; priority 340; 6 seeds).
  Other coordinator-side items queued by the 2026-06-17 evening
  cycle but already drained or in-flight: `V3-EXQ-460f` (the
  matching closure-control-plane de-commit retest paired with
  468e), `V3-EXQ-654f` (arc_062 GAP-B behavioural falsifier
  recovery from the silently-stalled 654e crash-before-manifest;
  supersedes 654e), `V3-EXQ-687` (Q-045 / MECH-313 / MECH-260
  4-arm tonic-noise / exploration-floor ablation on the now-
  survival-competent scaffolded_sd054_onboarding substrate).
  Substrate / governance landings since the 2026-06-17T07:30Z
  spec sync: (1) **SD-034 commitment-closure-control-plane
  BETA-ENGAGEMENT amend landed** (ree-v3 main f4ceea4 + ccb36db;
  REE_assembly master bf0e530) -- no-op-default
  `REEConfig.use_closure_commit_beta_coupling` couples the
  closure-plane commit (`e3._committed_trajectory is not None`)
  to the bistable BetaGate elevate path so beta occupancy tracks
  closure commitment on every seed where one forms, making the
  Leg-B de-commit refractory's ON<OFF latch-occupancy drop
  measurable; resolves the 460e commit-without-beta dissociation
  (seeds 42/43 committed 2415/2019 steps but
  `total_beta_elevated=0`); 5/5 new contracts +
  full suite green; 460f + 468e queued together on the amended
  substrate. PROMOTES NOTHING -- SD-034 provisional / MECH-260
  candidate / MECH-261 stable all stay `non_contributory` +
  `pending_retest_after_substrate` until 460f scores. (2)
  **V3-primacy version-layering guards landed**
  (ree-v3 main 703cbe0; REE_assembly master f274b45516) -- a new
  `ree_core/version_layering.py` registry of generation-tagged
  master flags + Guard A (DR-12 call-site conditional in
  `agent.py` -- skewed/older checkouts can no longer
  TypeError-crash on the e2_forward_pe_per_candidate kwarg, the
  exact V3-EXQ-654e fix), Guard B (runner preflight V3-parity
  smoke that refuses-to-claim on a default-path
  `select_action` raise + self-heals on the next consistent
  pull), Guard C (no-op-default contract asserting every gen
  flag defaults OFF and the default forward/select path runs);
  motivated by the 654e DR-12 unconditional call-site skew
  crash-burn. Bit-identical OFF guaranteed. (3) **Runner
  crash-before-manifest observability fix landed**
  (ree-v3 main f235eb7; REE_assembly master 9f38330d07) -- a
  manifest-less code crash (`result==ERROR`, `output_file`
  empty -- the FAIL/ERROR-class twin of the fixed UNKNOWN
  silent-drop) now mints a scoring-neutral synthetic ERROR
  manifest with `outcome=ERROR`, `claim_ids=[]`,
  `evidence_direction=non_contributory`, traceback tail, and
  ships it via `_report_result_and_align` BEFORE queue
  removal; `generate_pending_review.py` surfaces it under a
  new "Needs diagnosis (ERROR manifests -> /diagnose-errors)"
  section. Closes the crash-before-manifest invisibility hole
  documented in memory (`reference_phantom_completion_crash_before_manifest.md`).
  (4) **arc_062 GAP-B 654f re-queue** (ree-v3 main 5d3a192) --
  recovers the silently-stalled 654e falsifier as a verbatim
  copy (only identity changed: EXPERIMENT_TYPE/QUEUE_ID->654f,
  SUPERSEDES->V3-EXQ-654e; science unchanged), now safe because
  the cross-checkout DR-12 / ARC-063 kwarg skew that crashed
  654e is resolved on `origin/main`. (5) **SD-049-PHASE-2
  drive-coupling amend landed + V3-EXQ-514s queued** (ree-v3
  main b78101f + e737103; REE_assembly master eadd3bdb57) --
  two no-op-default levers (kappa-scale
  `incentive_drive_kappa_scale` + standing differential
  depletion `per_axis_restoration_fraction`, both bit-identical
  OFF, kappa LOAD-BEARING) routed by the V3-EXQ-514r overshoot
  disambiguator; 514s retest pre-registers a natural drive-
  coupled delta >= max(k*pstdev,0.15) on >=2/3 seeds as the
  MECH-436 substrate_ceiling -> supports criterion. (6)
  **DR-12 / `self_model_v4:SELF-4` first V4 substrate build
  landed + V4-EXQ-001 pilot RAN PASS** (ree-v3 f5eba3b +
  394ccf4 + c4fc5bd; REE_assembly 08f92b0d3e) -- the
  FIRST-EVER V4 substrate build: a no-op-default
  E2-forward-PE -> E3 confidence down-weight lever in
  `e3_selector.score_trajectory()` (per-candidate via
  `select()`; bit-identical OFF; 8 DR-12 contracts + full
  suite pass). Precedents set: `architecture_epoch =
  ree_self_model_v1`, `run_id` suffix `_v4`,
  `V4-EXQ-NNN` namespace (`validate_queue.py` widened
  V<gen>-EXQ), `owner_exq=V4-EXQ-001` on SELF-4; verified
  `generation:v4` keeps V3 closure % clean. V4-EXQ-001
  already ran on ree-cloud-1 and PASSed (`dr12_pe_conditioning_changes_selection`;
  3/3 diff-flipped, 3/3 uniform-inert; manifest 405ce947);
  PROMOTES NOTHING (MECH-215 stays candidate /
  `implementation_phase=v4`; only DR-12 cleared, MECH-215
  awaits DR-10 + experiments). (7) **Convergence Demand
  Pipeline First Action #1 landed** (REE_assembly master
  a99f1eb225) -- new `evidence/planning/convergence_demand_queue.v1.json`
  mirrors the plan Section-4 demand-queue table verbatim (5
  rows CDQ-001..CDQ-005 with `generation` + `blocks_v3_critical_path: false`
  per row), and `/governance` cadence gains a new
  non-interactive Step 6b (read-only over
  `closure_status.md` + `closure_drift.md`, appending
  demand-queue rows for newly verdicted mechanism-unclear /
  substrate_ceiling nodes). Plan status flipped
  PROPOSED -> ACTIVE. Tooling + cadence only -- registers
  NO claims, runs NO intake; V3 closure denominator
  UNTOUCHED. Bottleneck (continuation): **SD-034
  beta-engagement V3-EXQ-460f + MECH-090 conjunction
  V3-EXQ-468e are the live arms** that will resolve
  whether the closure-control-plane (Legs A/B/C + the
  beta-engagement amend) delivers a non-cap-pinned
  ON<OFF latch-occupancy drop with beta-engagement met,
  on the trained Leg-C `rule_bias_head`; **bdiv GAP-A
  conversion-architecture frontier paced by V3-EXQ-569i**
  (the TOP-K shortlist falsifier on the landed amend) and
  **SD-049-PHASE-2 V3-EXQ-514s** is the MECH-436 substrate
  -ceiling -> supports gate; the **Stage-H harm-pathway
  stabilization 603q** is the only blocking item left on
  the goal_pipeline GAP-2 survival-leg cohort (V3-EXQ-687
  Q-045 4-arm tonic-noise ablation rides on top of it).
  V4-EXQ-001 PASS is the first V4 datapoint -- pending
  governance review.

- **2026-06-17T07:30Z nightly read.** `evidence/experiments/` flat
  top-level holds **~421 `v3_exq_*.json` manifests** on disk (latest
  letter frontier through V3-EXQ-685; the day-over-day delta over the
  2026-06-16 read absorbs the 514q FAIL + the 569h/654d autopsies +
  the V3-EXQ-460e (Leg-C scaffold) + V3-EXQ-569i (TOP-K shortlist
  falsifier) + V3-EXQ-514r (MECH-229 disambiguator) queuings + the
  evening governance cycle still in flight). Legacy fleet
  `runner_status.json` unchanged at the 2026-06-09T06:00Z snapshot of
  840 dedup completion records (Phase-3 per-machine cards under
  `runner_heartbeats/` + `runner_status/` lead the legacy single-file
  tally).

  **Pending review queue (regenerated 2026-06-16T22:00:39Z) reads 1
  item** -- V3-EXQ-514q (MECH-229 drive-coupled wanting/liking
  weakens; the 514p-successor under the SD-of-delta + FLOOR=0.15
  effect-size gate). Active governance cycle
  (governance-20260616T2159Z) opened ~9.5h ago and is the named
  reviewer; expected to walk this single item alongside the
  per-claim weakens-vs-genuine adjudication. **Currently queued
  (`experiment_queue.json` items[]): 1 item** -- V3-EXQ-460e
  (SD-034 commitment_closure:GAP-4 Leg C, the trained
  rule_bias_head behavioural re-run on the freshly-landed scaffold;
  supersedes V3-EXQ-460d; claimed by ree-cloud-1 at 2026-06-17T04:20:33Z;
  priority 325; 3 seeds). Multiple coordinator-side items queued by
  the 2026-06-16 evening pass ride the DB beyond the file snapshot:
  V3-EXQ-569h GAP-A behavioural-diversity falsifier (3-arm
  matched-entropy, in-arm route-range gate, ARM_STD_G2 = std-basis +
  gain 2.0 + e2_world_forward conversion config; FAILed 1/3 seeds 2026-06-16
  -> 7th-autopsy adjudicated genuine conversion-ceiling; routed to
  /implement-substrate amend modulatory-bias-selection-authority TOP-K
  shortlist), V3-EXQ-569i (the TOP-K shortlist conversion-architecture
  falsifier, queued same evening on the landed amend; ARC-065 claim_ids;
  awaiting fleet claim), V3-EXQ-514r (MECH-229 drive-coupling
  disambiguator -- n=5 overshoot arm + OFF wanting==liking floor +
  recalibrated readiness gate; supersedes V3-EXQ-514q on the pre-registered
  routing; priority 320), V3-EXQ-654d / V3-EXQ-684a / V3-EXQ-680d /
  V3-EXQ-591f / V3-EXQ-685 (from the 2026-06-16 morning + day-cycle
  paths; awaiting fleet claim or in flight). V3-EXQ-654d (the de-collapsed
  GAP-B falsifier) FAILed 2026-06-16 with the same crf-availability-maintenance
  gate-lockout signature -> autopsy routed to /implement-substrate amend
  on a CRF maintenance-theta cap (the GAP-B fault is independent of the
  GAP-A context-collapse signal it was originally co-gated on; corrects
  the prior cohort framing). Substrate / governance landings since the
  2026-06-15T01:10Z spec
  sync: (1) **Morning governance cycle baked the indexer flat-merge
  shift + delivered MECH-229 SUBSTRATE CEILING LIFTED** (REE_assembly
  master be7261d9ca, 04:00Z; 874-file regen). MECH-229 V3-EXQ-514o
  PASS adjudicated: `substrate_ceiling -> standard`,
  `pending_retest_after_substrate -> false`, `narrow_supports -> false`,
  `exp_conf 0.868 -> confirmed_established`. **goal_pipeline:GAP-2 +
  GAP-7 CLOSED** on the terminal L9 wanting!=liking dissociation (the
  decisive substrate-readiness PASS the SD-049 / SD-057 / MECH-229
  closure chain was gated on). MECH-057b 672b genuine non-degenerate
  weakens recorded (conflict-hold stands; exp_conf 0.325; first counted
  experiment). MECH-423 680b -> superseded(by 680c) + 680c ->
  non_contributory (both autopsied, NOT weakened); 591d/666c/682 claim-
  free diagnostics reviewed. Pending review 7 -> 0; closure drift 0/0;
  overall 78.3%. crf-availability-maintenance flipped `ready=True`
  on 666c PASS (4d29688 substrate gate clear). (2) **V3-EXQ-682 GAP-A
  in-arm route-range PASS no_collapse_reproduced** (ree-v3 manifest
  20260615T032040Z) -- ARM_1_E2WF in-arm applied_route_range 0.204, all
  four 569g collapse causes RULED OUT incl seed 43 (the 569g failure
  seed). Confirmed Branch A: REACH solved, residual is genuine
  CONVERSION ceiling (not an upstream re-collapse). (3) **GAP-A 569g
  CONVERSION amend landed + 684 -> 684a HYBRID readiness sweep
  designed** (ree-v3 main 1acc343 + 37ff9b6; REE_assembly master
  012fca3a62 + 74112b99e9) -- two no-op-default conversion levers
  on the e3_selector authority block: (a) `modulatory_authority_normalize_basis`
  std vs range (anchor authority on the typical primary spread so the
  structured channel competes against near-decisive candidates), and
  (b) `use_modulatory_shortlist_then_modulate` + `modulatory_shortlist_margin`
  (F filters to a near-tie set, modulatory arbitrates the winner
  within -- safety preserved at any internal strength). 684 ran a
  6-arm readiness sweep on the live stack; 684a is the routed HYBRID
  redesign (ARM_STD_G2 forward, matched-noise-at-proposer turned
  NEGATIVE control, committed-layer metric-can-move non-vacuity guard
  added). 569h is GATED on 684a readiness PASS, NOT pre-queued on a
  guessed config. (4) **V3-EXQ-603p Stage-H harm-pathway autopsied
  -> /implement-substrate Branch B** (REE_assembly master d1a3a8e1c6)
  -- 603p was the diagnose-first locator for 603o's one failing
  readiness gate (`harm_landscape_discriminative_on_base`). Positive
  control (proximity_harm=0.10, harm_lr=1e-3) cleared the >=0.02
  range gate on only 1/3 seeds; ALL standard cells clear 1/3 (a
  different seed each cell, seed 44 never); 3x-LR rescue COLLAPSES the
  landscape to ~1e-23 on all seeds. Confirms GAP-C Branch B: the
  scaffolded_sd054_onboarding harm-VALUATION pathway training (603k)
  is SEED-FRAGILE -- not a regime-difficulty result. Substrate_queue
  amend slot opened on harm-pathway stability (>=2/3 seeds at the easy
  regime; more steps / lower LR / grad-stabilisation / seed-robust
  init, NOT raising LR). Do **NOT** queue V3-EXQ-603q (its located-
  parameter premise is false). (5) **V3-EXQ-591d/591e gate-criterion
  cluster autopsied + 591f robust sweep queued** (REE_assembly master
  86633d2a9f + 82f5950b99 + ree-v3 main 770a778) -- 591d FAIL adjudicated:
  neither K-of-N nor EMA-of-LEVEL@0.20 discriminates because both reuse
  the single-episode spike threshold 0.994 as the bar for a SUSTAINED
  statistic while genuine explorers' mean h_pos sits below it. 591e
  EMA-only re-admitted seed-45 (the causal one-way-latching EMA spikes
  over 0.20 on a late burst). 591f is the user-chosen ROBUST sustained-
  level sweep over four candidates (EMA-of-level / window-mean(W=20) /
  EMA-with-hold(H=5) / crossing-count>=3 of 0.994); PASS if ANY admits
  42/43/44 AND rejects 45. The parallel session's 591e queue entry was
  HELD via the coordinator so the flawed run does not execute.
  (6) **arm_reuse_fingerprint:P1-auto mint+consume cycle queued
  via V3-EXQ-685** (ree-v3 main 51906a9; REE_assembly master 7632783
  plan-doc) -- the gating owner V3-EXQ-610g is dead (655 resolved
  fork-(b) 2026-06-13: INV-074 substrate_ceiling, 610 cascade STOPPED
  by user decision), so the automated index-HIT will not arise
  naturally. 685 is a minimal purpose-built consumer: False-mode
  (`include_driver_script_in_hash=False`) mint of one canonical-baseline
  OFF cell + a distinct-driver consumer that automated-HITs it via
  `try_reuse_cell` (`reused_from_run_id` stamped) AND flips back to a
  fresh run on a one-byte config flip (section-9.5 step-6 acceptance).
  P1-auto status STAYS blocked until 685 runs+PASSes on the fleet.
  Bottleneck: **bdiv GAP-A is now a single sweep away from a verdict**
  -- V3-EXQ-684a (HYBRID readiness sweep with the load-bearing
  committed-entropy strict-above-LEGACY gate) is the gate that turns
  the landed gain/contrast + shortlist amend into a falsifier-ready
  config; on PASS, V3-EXQ-569h queues with the winning conversion
  config and the master choke gating 7 downstream nodes (entire sd_037
  axis-b chain + entire self_attribution chain) resolves. arc_062
  GAP-B (V3-EXQ-654c on the now-validated crf-availability-maintenance
  substrate) is the 2nd choke; MECH-423 super-additivity awaits 680d
  on the stabilised integrated co-training + NaN-guarded readiness
  cosine. The Stage-H harm-pathway substrate amend is the only
  blocking item on the goal_pipeline GAP-2 survival-leg cohort now
  that the MECH-229 ceiling is lifted.

- **2026-06-15T01:10Z nightly read.** `evidence/experiments/` flat
  top-level holds **382 `*_v3.json` manifests** on disk (latest letter
  frontier through V3-EXQ-682; the headline tally is a slight day-over-
  day adjustment of the prior ~395 estimate after a 2026-06-14
  governance-cycle reconcile and the 680/680a supersession sweep);
  legacy fleet `runner_status.json` unchanged at the 2026-06-09T06:00Z
  snapshot of 840 dedup completion records (Phase 3 per-machine cards
  under `runner_heartbeats/` + `runner_status/` lead the legacy
  single-file tally). **Pending review queue (regenerated
  2026-06-14T22:16:17Z) reads 1 item** -- V3-EXQ-680b MECH-423
  cross-model super-additivity (FAIL/inconclusive/additive_below_margin;
  delta_pair [+1.57, +1.24, -1.19], hardened margin 2.46 -> NOT
  super-additive); user-routed to `/failure-autopsy` which adjudicated
  the -1.19 seed as numerical-instability (NOT negative-transfer
  weakens) and routed to V3-EXQ-680c (supersedes 680b) with four
  robustness fixes: grad-clip world/affordance/proxy heads + shared
  encoder, R2 score floor at -1.0, re-wire the R1 cosine probe onto
  shared-encoder params (the 680b probe was structurally pinned at 0.0),
  n=5 seeds with 3/5 majority gate. **Currently queued
  (`experiment_queue.json` items[]): 5 items** -- V3-EXQ-514o
  (MECH-229 SD-049 Phase-2 wanting!=liking dissociation; sources the
  consumed type tag from `info[sd049_consumed_type_tag_this_tick]` per
  the 681 C4 fix; priority 310, ree-cloud-4), V3-EXQ-672b (MECH-057b
  trajectory-promotion-gate, ree-cloud-1), V3-EXQ-682 (GAP-A in-arm
  route-range collapse diagnostic; user-directed diagnose-first over a
  4th blind /implement-substrate amend; priority 290,
  `DLAPTOP-4.local`), V3-EXQ-603p (SD-059/MECH-358 base
  harm-landscape discriminativeness diagnostic on the 603o
  substrate_not_ready re-queue; ree-cloud-2), V3-EXQ-666c
  (arc_062:GAP-B CRF-readiness fraction-gated re-run with pre-gap
  e2ctx differentiation + N_EPISODES 100->200; supersedes 666b;
  ree-cloud-3). 680c is queued post-spec read at priority 315 and
  appears in the coordinator `/queue/active` tick. Substrate /
  governance landings since the 2026-06-14T01:10Z spec sync:
  (1) **Super-additivity PASS gate hardened** (ree-v3, EXQ versioning
  680a -> 680b -> 680c) -- the original `pstdev(iso_scores)` noise
  scale collapsed toward 0 on a reproducible substrate and let a
  trivially-small consistent positive delta look super-additive;
  hardened to `max(SUPERADD_SD_MULT * pstdev(delta_pair),
  SUPERADD_MIN_EFFECT_FLOOR=0.02)` so noise scales on the SD of the
  DELTA itself with an absolute effect-size floor (Daniel's memory
  rule: SD-of-delta + absolute floor, never SD of baseline level).
  680a manifest superseded; 680b confirmed the hardened gate flips the
  weak-criterion PASS to inconclusive on identical data. (2) **Indexer
  hardened: flat-manifest governance corrections now authoritative
  over the stale runs/ pack copy** (REE_assembly 074ab9401e) -- the
  indexer scored ONLY `runs/<run_id>/manifest.json` (the pack), so
  `/failure-autopsy` corrections written only to the flat sibling
  were silently ignored -> recurring MECH-171/057b/180 mis-fires.
  `_scan_runs` now auto-merges the flat sibling onto the pack, gated
  on `_is_annotated` (flat wins ONLY when it carries a note/reason the
  pack lacks; preserves pack-authoritative v3_exq_150-series). Surfaced
  ~20 claims to re-score; the late-evening governance cycle baked them
  in and surfaced no new promote/demote recs at the gate. (3) **Late-
  evening governance cycle** (REE_assembly a2d949d506; 864-file
  holistic regen) -- baked the flat-authoritative indexer 20-claim
  shift; user-directed promote **MECH-314a candidate_substrate_landed
  -> provisional** (exp_conf 0.759 > 0.62 gate, v3_pending cleared
  2026-06-08 via 604c). 5-item pending walk routed V3-EXQ-680b to
  `/failure-autopsy`; 679/681 PASS diagnostics marked reviewed.
  (4) **Plan-doc reconciles** -- sd_037_axis_b:P1b
  blocked_pending_substrate -> upstream_blocked (substrate BUILD half
  met; committed-action diversity half NOT met, downstream of GAP-A);
  behavioral_diversity_isolation:GAP-A/B/C drift absorbed (569f+569g
  FAIL/non_contributory, 603l autopsy-applied, 603n PASS), GAP-A/B
  reverse `cross_plan_link` to sd_037_axis_b:P1b added for
  bidirectional discoverability; MECH-341 ratified
  candidate -> provisional via claims.yaml-direct (80f4fcf250) +
  closure-node + decision_state reconcile (the derive-only gotcha
  noted in memory). (5) **GAP-A 569g autopsy correction** -- the prior
  20:34Z autopsy misread arm_results[0/1/2] as "all three falsifier
  arms applied route_range=0.0"; the corrected reading found ARM_1
  applied route_range 0.18 in-arm but committed entropy stayed
  bit-identical = genuine shared CONVERSION ceiling (gap-relative
  additive authority at gain 0.5; subdominant to F-dominated primary
  88-89%). Routing: implement-substrate amend the gain/contrast lever
  AFTER V3-EXQ-682 confirms the in-arm route-range reach. (6)
  **Critical-path synthesis doc** (REE_assembly 9d5f2a6760) --
  consolidated the five 2026-06-14 V3-closure critical-path drill-down
  sessions: bdiv GAP-A (V3-EXQ-682 -> /implement-substrate gain
  amend -> 569h falsifier) is the **master choke** gating 7 downstream
  nodes (entire sd_037 axis-b chain + entire self_attribution chain);
  arc_062 GAP-B (666c -> 654c) is the 2nd choke (4 nodes);
  goal_pipeline GAP-2 (514o) the only direct claim-closer in flight.
  Theme: every path's latest FAIL was an instrumentation defect
  (read-timing/post-gap/manifest-misread/marginal-guard), not a
  falsification -> diagnose-first is the correct default while the
  substrate is marginal. (7) **Heartbeat active-claim guard broadened
  to docs/claims/** (ree-v3 f61afb9 / REE_assembly badac028da) -- the
  per-minute autostash cycle was confirmed to transiently sweep
  uncommitted claims.yaml edits in a 2026-06-14 IGW window
  (ABM-1/Q-060); guard now skips the heartbeat push when any active
  TASK_CLAIMS entry lists a path under `evidence/` OR `docs/claims/`.
  Bottleneck: **bdiv GAP-A (V3-EXQ-682 diagnostic ->
  /implement-substrate gain/contrast amend -> V3-EXQ-569h falsifier)
  is the master choke** for the V3 closure frontier; arc_062 GAP-B
  (666c readiness -> 654c falsifier) is the 2nd choke; MECH-423
  cross-model super-additivity awaits V3-EXQ-680c on the hardened
  delta-scaled+floored margin (the substantive evidence path).

- **2026-06-14T01:10Z nightly read.** `evidence/experiments/` flat
  top-level holds **~395 `v3_exq_*.json` manifests** on disk (latest
  letter frontier through V3-EXQ-677); legacy fleet `runner_status.json`
  unchanged at the **2026-06-09T06:00Z snapshot of 840 dedup completion
  records (283 PASS / 437 FAIL / 87 ERROR / 32 UNKNOWN / 1 INCONCLUSIVE)**
  -- under Phase 3 the per-machine cards under `runner_heartbeats/` +
  `runner_status/` lead the legacy single-file tally. **Pending review
  queue (regenerated 2026-06-13T22:12:00Z) reads 2 items** -- both held
  by the evening governance cycle for `/failure-autopsy` per user
  directive (no `evidence_direction` assigned, stay pending): V3-EXQ-677
  MECH-180 novelty-sleep-upregulation probe (FAIL); V3-EXQ-672a
  MECH-057b trajectory-promotion-gate (FAIL). **Currently queued
  (`experiment_queue.json` items[]): 0 items** -- the 2026-06-13 IGW
  queue-6 Step-2.5 triage of 6 ready proposals (EXP-0143/MECH-423,
  EXP-0190/MECH-044, EXP-0191/MECH-048, EXP-0194/MECH-191,
  EXP-0195/MECH-270, INV-074 retest) found 0 cleanly queueable as
  evidence experiments: EXP-0143/MECH-423 deferred (proper experiment
  is EXP-0380 3-arm ablation gated on integration-readiness lit-pull;
  the R1/R2/R3 readiness substrate landed 2026-06-12 but EXP-0380
  itself stays a dedicated session); EXP-0190/MECH-044 needs bespoke
  relational-task design (no clean grid-world surface); EXP-0191/MECH-048
  switching-pressure half empirically substrate-blocked (smoke v3_exq_678
  wrote then deleted -- 0 mode switches across default + aggressive
  dACC-boost configs on a converged warmup; entropy half tautological;
  fix = mode-governance-engagement substrate); EXP-0194/MECH-191 skipped
  (active parallel IGW-220 worktree); EXP-0195/MECH-270 skipped (ephaptic
  lit-pull territory, "not an implementation requirement"); INV-074
  substrate_ceiling confirmed 2026-06-13T20:04Z by V3-EXQ-655 (7th
  iteration, decisive substrate-incapacity). Substrate / governance
  landings since the 2026-06-13T01:10Z spec sync: (1) **IGW plan-reconcile
  sweep + close-out** (REE_assembly master 6530f3c822 / d4d8c08990 /
  c9fd874, 2026-06-13) -- 7 stale-status IGW (plan reconcile) nodes
  flipped open -> done across the 2026-06-13 cycle as their architectural-
  commitment claims are already registered in claims.yaml: ABM-1 (Q-060
  decision -- distinct autobiographical-event memory type vs tag/indexing
  convention), GDL-1 (single-vs-multi-slot fork on SD-046), OBJ-ABS-1
  (substrate-gated; V3 atomic-only vocabulary missing), DRV-1 (SD-060 +
  MECH-111 drive-axes-as-first-class-register), OBJ-1 (ARC-080
  type-vs-token-vs-anchor primitive), ARR-1 (ARC-102), GRAM-1 (ARC-100 /
  INV-003 / INV-007), LANG-1 (INV-003 / ARC-009), LTI-1 (INV-007 /
  ARC-104), INF-1 (ARC-091), EMP-1/EMP-2 fast_empathy
  (ARC-094/ARC-095). Workset regenerated; ready (plan reconcile) lane
  trimmed. (2) **Closing-out lit-pull sprint -- 5 closures + 3 partner
  claims registered** (REE_assembly master c347e38b9e / 72c784be94 /
  381d9d05ba / d63faaf658 / d71360b679 / 53f5f2ddfa / 57d7af5bea /
  d42cfdaba6 / 652402209b, 2026-06-13) -- DMN-8 (developmental DMN /
  ARC-090, MECH-380/381/382/383 -- replaces analogy anchors with
  neuroscience mechanism; raises ARC-090 0.752, MECH-380/381 0.74,
  MECH-382 0.71, MECH-383 0.76); SELF-8 (self-model integration L1
  body-ownership + L2 agency / efference-copy + L3 interoceptive --
  SD-030 0.831, MECH-215 0.828, INV-064 0.725, ARC-081 0.718,
  MECH-214 0.71); INF-7 (belief-state inference L1-L5 + failure-mode
  register mapped onto MECH-126 -- Q-070 0.859, ARC-091 0.866); GDL-8
  (frontopolar deliberation -- first-ever lit grounding for the
  previously-ZERO-lit SD-046/SD-027/SD-028/MECH-254 cluster; SD-046
  0.822); AE-9 (per-candidate affect; MECH-359/360/364/369 + ARC-088);
  HPL-9 (Sinclair 2021 ACh/PE settles MECH-207 directionality 0.56
  -> 0.75 supports; closes the lit-pull DONE); ABM-9 FINISH
  (autobiographical-store + MECH-366 0.55 -> 0.83 + MECH-429 0.735).
  3 proposal-first partner claims registered: **MECH-432** dACC<->FPC
  reciprocal arbitration loop (mechanism_hypothesis, candidate, v4,
  substrate_conditional); **MECH-433** LC-NE-analog explore/exploit
  gain-control governing MECH-264 disengagement (NAMES the missing
  NE-analog substrate as a genuine gap); **Q-077** discrete-slot vs
  flexible-precision-resource open_question on SD-046 (what_would_answer
  = escalating-load step-function-vs-power-law degradation); **MECH-434**
  epistemic-commitment-timing axis (epistemic-freezing
  <-> anti-epistemic-panic; depends_on ARC-091/MECH-385/MECH-388/MECH-061/
  MECH-090/Q-044, distinct_from MECH-126; lit_conf 0.865; PROMOTES
  NOTHING -- v4/substrate_conditional); **AE-10** roadmap node
  (slow-modulator decommit-friction substrate gap; MECH-369 amended with
  Aston-Jones&Cohen 2005 LC-NE adaptive-gain grounding). claims.yaml
  802 -> 815/816 across the day. (3) **Backward-traceability gate fix**
  (REE_assembly master f8d71de093, 2026-06-13) -- added DEV-NEED-048
  register row for MECH-428 subgoal_bootstrapped_goal_seeding (the one
  developmental claim flagged untraced by
  `scripts/check_backward_traceability.py` via the 'curriculum'
  keyword); checker now 71 developmental claims, 0 missing -> 
  `governance.sh` Step 4b / G2 no longer halts (SKIP_TRACEABILITY=1
  no longer needed). (4) **Evening governance cycle (2026-06-13)** --
  851-file holistic governance regen + MECH-171 out_of_domain mis-fire
  note (REE_assembly master 81f671004e / 71f58ab); 2 pending FAILs
  (V3-EXQ-677 MECH-180 / V3-EXQ-672a MECH-057b) HELD for
  `/failure-autopsy` per user, no `evidence_direction` assigned, stay
  pending. MECH-171 keep candidate (mis-fire note); Q-054 keep broad;
  MECH-057b/SD-033b + 31 V3-pending recs accepted keep-held. Closure
  drift clean; substrate_queue unchanged (no new evidence/autopsies).
  Excluded IGW-owned `inter_governance_workset.*` edits. No claim
  promotions/demotions applied this cycle. (5) **Indexer reads runs/
  pack not flat JSON note** (memory) -- recorded that
  `build_experiment_indexes.py:546` reads `runs/<run_id>/manifest.json`,
  NOT the flat `evidence/experiments/*.json`, so failure-autopsy
  corrections (`non_contributory`/`non_degenerate`) written only to the
  flat manifest are silently ignored -> MECH-171/057b mis-fires recur
  (does_not_support is NOT in the scoring-exclude set); fix is to
  propagate corrections into the pack copy. Recorded as a flag on the
  prior evening governance cycle. Bottleneck: the **SD-034 commitment-
  closure-control-plane validation pair (V3-EXQ-460d + V3-EXQ-468d)**
  remains the in-flight gate for the cohort-wide closure-cluster
  pending_retest reroute; on the spec read no items are queued in
  `experiment_queue.json items[]`, so the next experimental signal
  depends on coordinator-side claims (which the runner heartbeats /
  per-machine cards drive) rather than the local queue file. V3-EXQ-655
  INV-074 task-distribution-shift returned a 7th-iteration decisive
  substrate-incapacity 2026-06-13T20:04Z (substrate_ceiling confirmed).
  V3-EXQ-460d / 468d / 669a remain in flight on the cloud workers.
  The two FAILs held for autopsy (677/672a) are the next adjudication
  decisions. The R1/R2/R3 MECH-423 readiness substrate landed 2026-06-12
  so EXP-0380 cross-model super-additivity remains the queued readiness
  check pending a dedicated session.

- **2026-06-13T01:10Z nightly read.** `evidence/experiments/`
  `claim_evidence.v1.json` carries **2816 unique V3 `run_id` records**
  in the per-claim matrix (the entry expansion includes per-claim rows
  for multi-claim manifests; the de-duplicated headline count tracked
  by yesterday's spec read was ~990). Legacy fleet
  `runner_status.json` totals **840 dedup completion records (283 PASS
  / 437 FAIL / 87 ERROR / 32 UNKNOWN / 1 INCONCLUSIVE,
  last_updated 2026-06-09T06:00Z)** -- the per-machine cards under
  `runner_heartbeats/` + `runner_status/` lead the legacy single-file
  tally under Phase 3. **Pending review queue (regenerated
  2026-06-12T22:37Z) reads 6 items** -- 5 FAIL siblings of the SD-034
  closure-cluster cohort (V3-EXQ-461c MECH-090 / SD-033a / SD-034;
  V3-EXQ-464c MECH-266 / SD-032a; V3-EXQ-466c SD-034 / MECH-094;
  V3-EXQ-467c MECH-266 / SD-032a; V3-EXQ-629b MECH-342) + 1 runner-only
  ERROR row V3-EXQ-669 (already superseded by V3-EXQ-669a re-queue);
  the morning 2026-06-13 governance cycle will apply the
  `failure_autopsy_SD-034-closure-cluster-ext_2026-06-12` extension's
  per-row routing (all 5 FAILs adjudicated `non_contributory` /
  `substrate_ceiling` / `pending_retest_after_substrate` across three
  substrate axes: closure-control-plane / mode-governance-engagement /
  scaffolded_sd054_onboarding nav-competence). **Currently queued
  (`experiment_queue.json` items[]): 4 items, all claimed** --
  V3-EXQ-655 INV-074 task-distribution-shift Phase-3 redesign
  (ree-cloud-1; supersedes V3-EXQ-610f; one of two scored-evidence
  gates in flight); V3-EXQ-669a MECH-329 / MECH-189 wanting-before-
  liking developmental goal-seeding (ree-cloud-2; supersedes the 669
  ERROR; rewrites the act/train loop onto the canonical StepHarness +
  exercises the MECH-189 super-ordinal goal-anchor substrate); V3-EXQ-460d
  SD-034 commitment-closure-control-plane retest (ree-cloud-3; supersedes
  V3-EXQ-460c; validates the env-completion hook seam + closure_decommit_hold
  + lateral_pfc_train_rule_bias_head landed 2026-06-12); V3-EXQ-468d SD-034
  de-commit hold retest on the non-cap-pinned committed-fraction DV
  (ree-cloud-4; supersedes V3-EXQ-468c). Substrate / governance
  landings since the 2026-06-12T01:10Z spec sync: (1) **SD-034
  commitment-closure-control-plane amend** (ree-v3 main 6fdb111 +
  REE_assembly master cc49f120a4, 2026-06-12) -- two-leg behavioural-
  authority amend the SD-034 ClosureOperator lacked. Leg A
  `use_closure_env_completion_hook` routes env
  `sequence_complete` -> `REEAgent.notify_env_completion` ->
  `closure_operator.emit_closure` (the *c-cohort gap: env emitted
  completions but nothing routed them into emit_closure). Leg B
  `closure_decommit_hold_ticks` installs a `BetaGate.apply_refractory`
  window on every closure fire so a closure-coupled release survives
  >1 tick -- measurable latch-occupancy drop instead of immediate
  re-commit. Both no-op-default; bit-identical OFF. 1014 contracts
  (1008 prior + 6 new) + 7/7 preflight PASS. V3-EXQ-460d + V3-EXQ-468d
  validation diagnostics queued (claim_ids carry SD-034 / MECH-260 /
  MECH-261 / MECH-268 / MECH-090; readiness gate on n_closures>0
  reachable on >=2/3 completion-engaged seeds -> else
  substrate_not_ready_requeue, never a false weakens). SD-034 stays
  provisional, MECH-261 stable; PROMOTES NOTHING. (2) **MECH-423
  super-additivity readiness substrate** (ree-v3 main aff371f +
  78fe92c + REE_assembly master 6964f47bf8, 2026-06-12) -- the three
  readiness readouts the EXP-0380 cross-model super-additivity
  acceptance_checks require, all no-op-default + bit-identical OFF +
  contract-tested. R2 (iterative-inference convergence on
  `LatentStack.encode`; generalises the legacy two-pass amortized
  recognition into a predictive-coding settling loop, tracks
  `inference_convergence` with per-round `rel_delta` + early-stop at
  `rel_tol`); R3 (module-tagged interleaved E1<->E2 cross-module
  consolidation in `SleepLoopManager` / MECH-121; a flag-gated hook
  runs the default E1+E2 loss set in the offline pass and surfaces
  `cross_module_replay_share` + `n_updates` + `interleaved`); R1
  (`shared_latent_gradient_probe` utility -- per-module
  `d(loss)/d(z_shared)` cosine-coupling probe). 12 new MECH-423
  contracts; ARC-004 + MECH-121 carry an `implementation_note` only
  (no flag / confidence change -- PROMOTES NOTHING). EXP-0380 flipped
  `blocked_substrate` -> `proposed`. V3-EXQ-676 substrate-readiness
  validation queued + ingested into the coordinator
  `/queue/active`. (3) **SD-034 closure-cluster cohort extension
  autopsy** (REE_assembly master 7ba8211038, 2026-06-12) -- the 5
  sibling FAILs above adjudicated `non_contributory` /
  `substrate_ceiling` / `pending_retest_after_substrate` across three
  substrate axes (A closure-control-plane: 461c + 466c -> parent
  amend pattern; B mode-governance-never-engages: 464c + 467c ->
  CREATE new `mode-governance-engagement` substrate gap; C
  readiness-requeue: 629b MECH-342 -> amend `scaffolded_sd054_onboarding`
  nav-competence ceiling with `narrow_supports_flag`). NO demotion
  SD-034 / MECH-090 / SD-033a; NO weaken MECH-266 / SD-032a /
  MECH-342 / MECH-094. (4) **V3-EXQ-674 MECH-087 cross-plane
  non-rescue PASS** (governance applied 2026-06-12 PM cycle, commit
  0ebc83dc3d) -- the literal 2x2 (degrade serotonin-terrain vs
  degrade dopamine-selection) x (rescue-with-dopamine-gain vs
  rescue-with-terrain-adjustment) on the E3 softmax selection stage
  produced the axis-specific rescue asymmetry. MECH-087 receives
  supports; out_of_domain receptor-subtype claims (PHARM-015..022)
  stay parked. (5) **Receptor-subtype intervention layer + 4 lit-pulls
  + reap cycle** (REE_assembly master 525949ab84 .. 1d6d28d9cd,
  2026-06-12) -- new `docs/architecture/receptor_subtype_intervention_layer.md`
  (rung between MECH-claims and PHARM predictions);
  `pharmacological_predictions.v1.json` PHARM-015..022 registered
  out_of_domain (5-HT1A / 5-HT2A; M4 / xanomeline; mu / kappa /
  buprenorphine; NMDA / ketamine; GABA-A subunit; D2 vs D3; orexin
  OX1R / OX2R); 4 lit-pulls (`targeted_review_receptor_subtype_layer`,
  `targeted_review_mech_423_integration_prerequisites` raising
  MECH-423 lit_conf 0 -> 0.848, `targeted_review_relational_harm_love_as_care` /
  `targeted_review_guilt_repair_moral_emotion` /
  `targeted_review_perceptual_manifold_adaptors` /
  `targeted_review_inv_043_caregiver_development` closing biology-
  grounding debts on RHM-6 / ETH-8 / PA-7 / LOVE-7 V4-V5 plans);
  reap Q-075 (ARC-086 receptor-grounding open question) + Q-076 +
  MECH-425 (histaminergic retrieval accessibility gating; 803-claim
  registry). All candidate / substrate_conditional / v4-v5; PROMOTES
  NOTHING on the V3 substrate axis. (6) **Architecture-view zoom-
  accuracy pass** (REE_assembly master e77983b5af / c56042b9a7 /
  d21c16f919, 2026-06-12) -- arch canvas + explorer claims-graph
  over-sensitive wheel zoom fixed (delta-proportional factor +
  deltaMode normalisation + per-event clamp); accuracy audit of
  10 architectural nodes (ResidueField valence 4->6; LatentStack
  shared `z_beta -> z_theta -> z_delta`); added FrontalCue (SD-016 /
  ARC-041) + GhostGoalBank (MECH-292 / MECH-339 / MECH-340) as
  first-class nodes with detail panels. `ree_architecture.html`
  bumped to 2026-06-12.3; `explorer.html` to EXPLORER_VERSION
  2026-06-12.2. (7) **Closure-drift `behavioral_diversity_isolation:GAP-B`
  reconcile** (REE_assembly master 2403768d62, 2026-06-12) -- the
  residual `stale_since_review` flag cleared by advancing owner_exq's
  leading id 660 -> 660b (lineage frontier; convention-consistent
  with the 514m->514n / 485e->485f advances); standing GAP-B evidence
  unchanged (predecessor V3-EXQ-660 base PASS supports MECH-341;
  660b retired non_contributory). Bottleneck (continuation): the
  **SD-034 commitment-closure-control-plane validation pair
  (V3-EXQ-460d + V3-EXQ-468d)** is the in-flight gate for the
  cohort-wide closure-cluster pending_retest reroute; PASS on either
  confirms the amend resolves the *c-cohort's "no behavioural
  authority" pattern. V3-EXQ-655 INV-074 task-distribution-shift +
  V3-EXQ-669a MECH-329 / MECH-189 wanting-before-liking are the
  other two scored-evidence gates in flight; V3-EXQ-676 MECH-423
  substrate-readiness is the EXP-0380 unblock readiness check.

- **2026-06-12T01:10Z nightly read.** `evidence/experiments/` flat
  top-level holds **~373 `v3_exq_*.json` manifests** on disk; **990
  unique V3 `run_id` records in `claim_evidence.v1.json` (356 PASS /
  634 FAIL)** at the most recent generation. **Pending review queue
  reads 1 item** -- V3-EXQ-667 (Q-043 magnitude sweep; autopsy
  completed and routed to a substrate_queue amend on
  modulatory-bias-selection-authority + 667a retest, but the manifest
  itself was not walked by the closing 2026-06-11 governance cycle
  because it landed after the snapshot; deferred-to-next-cycle).
  **Currently queued (`experiment_queue.json` items[]): 3 items, all
  claimed** -- V3-EXQ-655 INV-074 task-distribution-shift Phase-3
  redesign (ree-cloud-1; supersedes V3-EXQ-610f); V3-EXQ-672 MECH-057b
  trajectory-promotion-gate first genuine experiment (ree-cloud-4;
  EXP-0152 IGW-204 dispatch; ARM_0_NO_GATE vs ARM_1_COMPLETION_GATE on
  hippocampal trajectory completion filtering); V3-EXQ-673 MECH-171
  vicious-cycle sleep disruption (pending, MECH-171; EXP-0153 IGW-203
  dispatch; three-arm ARM_A_HEALTHY / ARM_B_EARLY / ARM_C_LATE).
  Substrate / governance landings since the 2026-06-11T01:10Z spec
  sync (a heavy autopsy + governance day with five confirmed
  failure-autopsies + four governance cycles): (1) **ARC-063 CRF
  mature-pool dynamics amend** (ree-v3 main 7e2e0ef, 2026-06-11) --
  closes V3-EXQ-654b GAP-B maturity. Master flag
  `crf_mature_pool_dynamics` routes a bundle of recalibrated knobs:
  tolerance-conflict deadlock (theta=1.3 > 1.0 max availability when
  >=2 rules match -> never co-fire) fixed by `mature_tolerance_floor`
  0.15 + `mature_tolerance_conflict_gain` 0.25 (theta(1..3)=
  0.40/0.65/0.90 < 1.0); 5x slower availability_decay; asymmetric
  availability_alpha_negative; mint_protection_ticks 30;
  mint_block_threshold 0.8 decoupled from retrieval threshold 0.5.
  Companion flag `crf_context_from_e2_world_forward` re-sources the
  CRF mint/match context from `e2.world_forward(z_world,
  prev_action_onehot)` (action-regime-separated; mirrors ARC-065
  GAP-A re-sourcing). (2) **ARC-063 crf-availability-maintenance
  amend** (ree-v3 main 1d04e51, 2026-06-11) -- closes the V3-EXQ-666
  successor gap. Master flag `crf_availability_maintenance` removes
  the per-tick silence decay in `credit()` (Mongillo 2008 /
  Stokes 2015 / Lundqvist 2018 activity-silent synaptic maintenance;
  minted differentiated rules HOLD availability across context-absent
  ticks; eligibility-gated negative-outcome credit + retirement
  UNTOUCHED so a consistently-bad rule still erodes);
  `maintenance_floor` 0.45 at mint; new readiness readout
  `crf_n_maintained_reactivatable` + `crf_maintained_pairwise_dist` +
  `crf_frac_maintained` retires the wrong-axis
  `crf_frac_active >= 0.30` target. V3-EXQ-666a / V3-EXQ-666b
  CRF-readiness validation queued (666b re-states the gate on the
  isolating `crf_frac_maintained` after 666a's count-floor measurement
  defect). (3) **Indexer non-degeneracy scoring net** (REE_assembly
  master 974e1305fd, 2026-06-11) -- manifest `non_degenerate: false` or
  `non_degenerate_per_claim: {...}` excludes the entry from confidence
  / conflict scoring (parallel to `superseded`/`stale_substrate`); new
  `ree-v3/experiments/_metrics.py` `check_degeneracy()` +
  `p0_readiness_gate()` helpers + `validate_experiments.py` + both
  `/queue-experiment` mirrors require producer self-report.
  Bit-identical OFF; 0 false-exclusions on the live evidence matrix.
  (4) **Epistemic stance derived view shown / believed / asked**
  (REE_assembly master 3da6340726, 2026-06-11) -- `build_claims_json.py`
  derives `epistemic_stance` per claim from `exp_conf >= 0.62`
  (shown=71) + `epistemic_category in {answer_state, derivational,
  out_of_domain}` (asked=74) + remainder (believed=653);
  `validate_claims.py` warns on `what_would_answer:` missing in the
  asked bucket (74 hits warn-only). View-only; claims.yaml content
  UNTOUCHED. (5) **Five confirmed failure autopsies + four governance
  cycles applied**: V3-EXQ-514m MECH-229 vacuous FAIL ->
  non_contributory + reroute V3-EXQ-514n (SD-057 wanting!=liking on
  most-wanted z_goal pointer vs last-consumed type); V3-EXQ-485e
  substrate_not_ready_requeue -> non_contributory + reroute V3-EXQ-485f
  (SD-033b trained-OFC retest, ofc_bias_scale 0.1->0.5 +
  e2_action_contrastive in P0 + absolute between_context_tv floor);
  V3-EXQ-660b MECH-341 graded sub-axis weakens=measurement_test_design_defect
  (latent nested-manifest scoring bug fix landed; retire
  graded-in-pool-size falsifier); V3-EXQ-666a CRF availability-
  maintenance count-floor differentiation-clearable ->
  measurement_test_design_defect (substrate_queue amend on
  crf-availability-maintenance; ready stays FALSE pending V3-EXQ-666b
  PASS on the re-gated isolating-fraction diagnostic); V3-EXQ-667 Q-043
  magnitude sweep (MECH-313 + MECH-314 jointly 1x/2x/4x/8x on the
  591c InfantCurriculumScheduler Phase-0 probe) confirmed
  substrate_not_ready_requeue -- 4/5 seeds byte-identical on EVERY
  per-cell statistic across the full 8x joint knob scaling -> the swept
  ARC-065 knobs have ZERO authority over the committed E3 argmin
  (convergent with 604a/624a/614d/660b/569f/661/654a); substrate_queue
  amend on modulatory-bias-selection-authority adding 667 as a
  blocked-retest failure record (ready stays FALSE), re-issue
  V3-EXQ-667a GATED on the route-range amend + ARC-065 GAP-A
  readiness_range>=0.05 positive-control clearing; Q-043 stays open /
  pending_retest_after_substrate (exempt from exp_conf gating); NO
  `/claim-synthesis` (clean calibration question, single known blocker;
  lineage 591/591b/591c->667 is convergent substrate engineering). NO
  claim promotions/demotions applied this cycle. (6) **infant_substrate
  Phase 0->1 H_pos floor recalibration** (ree-v3 main, 2026-06-11) --
  `H_POS_FRAC_OF_MAX` 0.70 -> 0.20 to sit inside the V3-EXQ-591
  observed band 0.03-1.08 (threshold drops from ~3.48 -- structurally
  unreachable -- to ~0.99 with ~9% upper-margin). INTENTIONAL non-no-op
  default change behind the exception precedent of MECH-307 default-
  tweaks (2026-05-12) + ARC-065 SP-CEM main-path landing (2026-05-17).
  (7) **`/thought-digestion` first bounded run** (REE_assembly master
  24852aa0b8, 2026-06-11) -- digested 5 asked claims
  (Q-021 / Q-015 / Q-007 / Q-023 / Q-019) with falsification
  conditions; minted EXP-0356 (Q-019 reframed-testable). 10 SKILL.md
  refinements applied to both `.claude/` + `.agents/` mirrors.
  Bottleneck (continuation): the **selection-authority frontier paced
  by the `modulatory-bias-selection-authority` route-range amend's
  behavioural validation reaching meaningful magnitudes** is the
  cross-cutting blocker. V3-EXQ-667's 4/5 byte-identical seeds across
  an 8x joint knob scaling on MECH-313 / MECH-314 adds a third
  high-confidence failure record to the substrate_queue entry,
  confirming the amend's honest scope ("routing makes the range REACH
  and MOVE the committed argmax (the readiness property), NOT
  necessarily move it beneficially"). V3-EXQ-667a will be the first
  retest gated on the route-range amend + ARC-065 GAP-A
  readiness_range>=0.05 clearing. The **ARC-063 CandidateRule field's
  V3-EXQ-666b validation** is the second in-flight readiness gate
  (re-states the load-bearing readiness on the isolating
  `crf_frac_maintained` statistic now that 666a confirmed maintenance
  WORKS on the right metric but the gate was on the wrong one).
  V3-EXQ-655 + V3-EXQ-672 are the two scored-evidence gates in flight.

- **2026-06-11T01:10Z nightly read.** `evidence/experiments/` flat
  top-level holds **~1053 `v3_exq_*.json` manifests** by recursive count
  (~997 unique V3 `run_id` records in `claim_evidence.v1.json` at
  generation 2026-06-11T00:03Z: 357 PASS / 629 FAIL / 11 SOURCE per
  unique-run-id status tally). `runner_status.json` fleet totals carry
  840 dedup completion records (283 PASS / 437 FAIL / 87 ERROR /
  32 UNKNOWN / 1 INCONCLUSIVE, last_updated 2026-06-09T06:00Z; the
  Phase-3 coordinator now writes per-machine telemetry under
  `runner_heartbeats/` + `runner_status/` so the legacy single-file
  tally lags the per-machine cards). **Pending review queue
  (regenerated 2026-06-11T00:11Z) reads 1 item** -- V3-EXQ-591c FAIL,
  the curriculum Phase-0->1 readiness DIAGNOSTIC RE-RUN under the armed
  diversity stack (claim-free; user-flagged for `/failure-autopsy`).
  V3-EXQ-665 curriculum affective-fishtank showcase re-run was reviewed
  in the 2026-06-11T00:13Z governance cycle (non_contributory PASS with
  `vacuous_pass` flag on non-load-bearing channels; load-bearing core_channels
  PASS non-vacuously). **Currently queued (`experiment_queue.json` items[]):
  3 items, all claimed** -- V3-EXQ-603o SD-059/MECH-358 escape-affordance
  bridge BEHAVIOURAL retest REDESIGN (claimed ree-cloud-3; supersedes
  V3-EXQ-603l; carries the headroom + continuous-mean-survival
  primary-DV fixes from the 603l autopsy); V3-EXQ-660a MECH-341 GAP-B
  GRADED CEM pool-size confirmation (claimed ree-cloud-4; K in
  {16,32,64,128} successor to 660 since 660's within-class temperature
  sweep was byte-identical -- temperature has zero leverage at
  near-degenerate within-class scores, so K is the viable graded axis);
  V3-EXQ-655 INV-074 crystallization-necessity TASK-DISTRIBUTION-SHIFT
  Phase-3 redesign (claimed ree-cloud-1; supersedes V3-EXQ-610f).
  Substrate / governance landings since the 2026-06-10T01:10Z spec sync
  (a heavy closure-roadmap + governance day): (1) **MECH-219 / SD-019b
  affective-harm hysteretic integrator substrate** (ree-v3 main,
  2026-06-10) -- tier-3 slow controllability-gated `z_harm_suffering`
  load state on top of SD-019a's tier-2 `z_harm_un` EMA; pluggable
  escapability (constant / SD-058 `effective_efficacy()` /
  external-seam); per-consumer redirect (AIC + PAG + MECH-091 wired in
  v1; dACC/pACC defined but unwired). SD-019b stays candidate /
  v3_pending pending the controllability-dissociation falsifier.
  (2) **modulatory-bias-selection-authority route-range AMEND** (ree-v3
  main + REE_assembly master, 2026-06-10) -- routed by
  `failure_autopsy_569f-661-654a_2026-06-10`. The authority now folds an
  SVD-projected per-candidate channel-range bias into the accumulator
  BEFORE rescaling, so a channel's representational range REACHES + MOVES
  the committed argmax (the readiness property the cluster proved missing
  for world-summary / coherence / rule_state channels). V3-EXQ-663
  substrate-readiness diagnostic landed clean in the 2026-06-11T00:13Z
  governance cycle. (3) **V3 closure roadmap V4/V5/V6 segmented partition
  completed** (REE_assembly master, 2026-06-10..06-11) -- three
  multi-agent workflow waves seeded the V4 individual-mind tier
  (12 generation:v4 plans + 27 reaped candidate claims; +5 v4->v5
  reassignments in wave 2), the V5 social-mind tier (6 generation:v5
  plans + 23 candidate claims), and the V6 linguistic-mind tier
  (5 generation:v6 plans + 16 candidate claims + 8 dedup mappings into
  existing language cluster). V3 closure % stays **75.9% (75 nodes /
  11 plans, byte-identical)** -- the closure map is generation-segmented
  so V4/V5/V6 nodes never dilute V3. Total roadmap now **25 plans /
  174 nodes across v3 / v4 / v5 / v6 / deferred** (the
  deferred-by-commitment bucket landed 2026-06-11 with 4 reversal-
  triggered nodes for TCL/phase ARC-053 cluster + ARC-084 signed coupling +
  attention map; excluded from every % by construction). `closure.html`
  now exposes the full V3/V4/V5/V6/deferred generation switch
  (`CLOSURE_VERSION 2026-06-11.1`). (4) **Total claim count 759 -> 798**
  (39 new candidate claims from the V4/V5/V6 waves; all
  substrate_conditional / v3_pending / implementation_phase v4-v6 so the
  governance Decision Queue produced 0 new actionable rec rows). (5) **Two
  governance cycles run 2026-06-11T00:13Z and post-failure-autopsy
  closure** -- pending_review walked from 2 -> 0 -> 1 (591c flagged for
  autopsy and adjudicated under user-confirmed routing: ARC-065 substrate
  load-bearing but UNDER-POWERED at default magnitudes for the
  collapse-prone seed tail; primary follow-on = `/queue-experiment`
  Q-043 magnitude sweep + substrate_queue ARC-065 amend with the 591b+591c
  default-magnitude-insufficiency failure_record + ARC-046 K-of-N/EMA
  gate-robustness follow-on noted). (6) **MECH-073 promoted candidate ->
  provisional** (REE_assembly master, 2026-06-10) -- targeted single-claim
  governance promotion driven by Phase-3 lit/exp decoupling: experimental
  record CLEAN (EXQ-375 3/3 PASS; `genuine_exp_direction_counts` 3 supports
  / 0 weakens; exp_conf 0.757 above the 0.62 bar). Lit-only conflict_note
  resolved by Q-020 (2026-03-16); under Phase-3 lit signal does not gate
  promotion. Sole genuinely-promotable claim among the 26
  hold_candidate_resolve_conflict set; the other 25 are substrate-held
  (gated on 603m, SD-016, commitment substrate) not conflict-resolvable.
  Bottleneck (continuation): the **selection-authority frontier paced by
  V3-EXQ-660a (MECH-341 within-class-rep diversity via CEM pool-size) +
  V3-EXQ-603o (SD-059/MECH-358 escape-affordance bridge re-test with
  continuous-mean-survival primary DV)** is the cross-cutting blocker;
  V3-EXQ-655 (INV-074 task-distribution-shift) is the third in-flight
  gate. The **infant-curriculum Phase 0->1 advancement gate** continues
  under the 591c autopsy routing (ARC-065 default-magnitude under-power
  is the load-bearing reading, NOT a gate-threshold defect).

- **2026-06-10T01:10Z nightly read.** `evidence/experiments/` flat
  top-level holds **346 `v3_exq_*.json` manifests** (recursive count
  ~1053 incl. nested per-run dirs). Per-machine fleet
  `runner_status/*.json` cards across DLAPTOP-4.local + Daniel-PC +
  EWIN-PC + ree-cloud-1..4 + ree-worker-3 aggregate to **848 unique
  V3 queue_ids completed** after dedup (286 PASS / 429 FAIL / 91 ERROR
  / 41 UNKNOWN / 1 INCONCLUSIVE; 1632 total completion records
  pre-dedup). **Pending review queue (regenerated 2026-06-09T21:09:04Z)
  reads 0 items** -- 603i diagnostic-self-route `precondition_unmet`
  flag previously listed there was adjudicated and the autopsy routed
  successor work onto the 603-lineage behavioural bridge re-test
  programme. **Currently queued (`experiment_queue.json` items[]): 3
  items, all claimed** -- V3-EXQ-603l SD-059 / MECH-358
  escape-affordance bridge BEHAVIOURAL re-test (claimed ree-cloud-2;
  scored-evidence successor to V3-EXQ-603i, runs against the
  trainable-relief / safety-half head substrate landed 2026-06-09);
  V3-EXQ-655 INV-074 crystallization-necessity task-distribution-shift
  Phase-3 redesign (claimed ree-cloud-1; supersedes V3-EXQ-610f);
  V3-EXQ-660 MECH-341 committed-class diversity RETEST on the GAP-A-
  ready / authority-ready stack using a WITHIN-CLASS-REPRESENTATIVE-
  DIVERSITY readout (claimed DLAPTOP-4.local; arms GAP-A
  `candidate_summary_source=e2_world_forward` + modulatory selection
  authority gain 0.5 + SP-CEM + MECH-341 + SD-056). Substrate /
  governance landings since the 2026-06-09T01:10Z spec sync (a heavy
  governance + thought-intake day): (1) **MECH-189 super-ordinal
  goal-anchor ContextMemory writes substrate** (ree-v3 main c7ac035 +
  REE_assembly master f212523d80, 2026-06-09) -- closes
  infant_substrate:GAP-11. New `SuperOrdinalGoalMemory` in
  `ree_core/goal.py` (agent-owned, NOT reset per episode, cue-indexed
  with key=z_world context / value=z_goal anchor); write hook at
  `agent.update_z_goal` (child phase) gated on the MECH-189
  conjunction (high-salience drive-modulated benefit AND high
  contextual complexity, pluggable `super_ordinal_complexity_mode`
  default novelty-vs-anchors); read hook seeds adult z_goal via
  GoalState.cue_pull from the retrieved anchor when z_goal is below
  floor. Recurring high-salience contexts REINFORCE their anchor on
  salience alone toward the matured z_goal. 985 contracts + 8 new
  MECH-189 contracts + 7 preflight PASS; bit-identical OFF.
  V3-EXQ-588c readiness diagnostic queued (LOAD-BEARING C1 = ADULT
  z_goal seeding DISCRIMINATION ARM_ON vs ARM_OFF). (2) **MECH-294
  multi-content theta-burst packet substrate + compose-coherence
  amend** (ree-v3 main 53d4c75 + af97b23, REE_assembly master
  4c680c03f5, 2026-06-09) -- per-cycle joint binding of
  {goal, action, risk, state} into `ThetaPacket`; compose path now
  reads within-cycle co-binding coherence via
  `currency_coherence()` so the binding mode (joint / alternation /
  shuffled) causally reaches E3 behaviour rather than collapsing onto
  action-only similarity. Parameter-free, no trained head, no-op
  default. V3-EXQ-661 substrate-readiness 5-arm diagnostic queued
  (TV-margin readouts + non-vacuity gate). (3) **SD-033b OFC trainable
  state_bias_head substrate + V3-EXQ-485d readiness diagnostic** --
  closes the SD-033a GAP-D mirror leg of commitment_closure:GAP-8;
  `OFCConfig.train_state_bias_head` (default False -> last Linear
  zeroed) + `bias_head_parameters()` + REEAgent build-site wiring;
  V3-EXQ-485d 2-arm diagnostic queued (frozen vs trainable; load-
  bearing C2 on head weight-delta-from-init under SP-CEM diversity).
  SD-033b stays candidate / v3_pending; PASS unblocks the
  behavioural-evidence arm. (4) **ARC-063 cross-episode rule-
  persistence flag** (ree-v3 main, 2026-06-09) -- amend for V3-EXQ-654
  GAP-B maturity. (5) **MECH-295 RE-SCOPED + goal_pipeline:GAP-4
  CLOSED** (REE_assembly master, 2026-06-09 governance decision) --
  MECH-295 narrowed to the MODULATORY reading (behavioural-necessity
  terminally FALSIFIED by V3-EXQ-490j severed-bridge baseline; the
  MODULATORY-SUFFICIENCY behavioural retest is retained as an OPTIONAL
  non-GAP-blocking future test). `goal_pipeline:GAP-4` closed.
  closure_status.md done 44 -> 45, overall_progress 72.4 -> 73.2%.
  (6) **self_attribution GAP-1 / GAP-2 / GAP-3 + infant_substrate
  GAP-13 re-adjudicated** (REE_assembly master, 2026-06-09) --
  collapsed onto the SHARED selection-authority frontier paced by
  V3-EXQ-660 (MECH-341) + V3-EXQ-643a authority + V3-EXQ-604c, since
  ARC-065 SP-CEM became the main-path default 2026-05-17 and the
  V3-EXQ-543l sleep-substrate run + V3-EXQ-614e CEM-pool collapse
  autopsy together prove the SP-CEM lever alone does not move the
  monomodal-equilibrium frontier. No code, no claim re-weighting --
  resume_conditions repointed. (7) **IGW generator suppresses V4+
  scoped claims from the proposal lane by PHASE/VERSION** (REE_assembly
  master 3c3f37f2ff, 2026-06-09) -- generator now skips
  `implementation_phase: v4|v5` and version-band-v4+ claims independent
  of `epistemic_category`. Removes 7 plain-v4 proposal leaks (SD-033e
  / MECH-264 / MECH-265 / MECH-129 / MECH-130). (8) **Heavy thought-
  intake + REAP cycle** (REE_assembly master, 2026-06-09) -- six
  separate intake sessions registered V4 / post-V5 claim clusters
  (INV-077 evaluation-channel-integrity + Q-069 + EXT-008 meta-agent
  failure mode; ARC-087 / MECH-372 / Q-065 sense-specific perceptual
  manifolds; ARC-089 / MECH-374 / Q-066 spintronic-memristive
  cognifold substrate; ARC-085 / MECH-365 / MECH-366 / Q-060
  competitive-interactions cluster; MECH-359 / MECH-360 / MECH-361
  candidate-differentiated affective gradients; MECH-362 + Q-057 CA3
  developmental sparsification; MECH-364 + Q-059 laughter
  load-release) -- all candidate / substrate_conditional /
  implementation_phase v4 / v4_v5 (off the GAP-7 / V3 critical path).
  (9) **infant_substrate:GAP-14 prereq-c InfantCurriculumScheduler
  Phase 0->1 advancement gate retune** (in-flight under the active
  21:48Z claim) -- queues V3-EXQ-591b readiness diagnostic. Bottleneck
  (continuation): the **selection-authority frontier paced by
  V3-EXQ-660 + V3-EXQ-604c** is now the cross-cutting blocker for
  self_attribution / behavioral_diversity_isolation:GAP-B /
  arc_062:GAP-B / sd_037_axis_b; the **603-lineage survival /
  escape-affordance evidence loop** (V3-EXQ-603l) remains the second
  in-flight gate; **V3-EXQ-485d / V3-EXQ-588c / V3-EXQ-661** are the
  three new substrate-readiness diagnostics seeded by today's
  substrate landings. Cluster-registration discipline: the 2026-06-09
  thought-intake cycle confirmed the user's standing rule that
  genuinely-new intake content is REGISTERED into `claims.yaml` (as
  candidate / substrate_conditional / version-scoped claims) in the
  same pass that authors the home doc, NOT left as "future-registration"
  prose.

- **2026-06-09T01:10Z nightly read.** `evidence/experiments/` flat
  top-level holds **334 `v3_exq_*.json` manifests** (recursive count
  ~994 incl. nested per-run dirs). Per-machine fleet `runner_status/*.json`
  carries cards for DLAPTOP-4.local, Daniel-PC, EWIN-PC, ree-cloud-1..4,
  and ree-worker-3. **Pending review queue (regenerated 2026-06-08T22:05Z)
  reads 1 item** -- V3-EXQ-603i SD-059/MECH-358 escape-affordance bridge
  validation FAIL with self-route label `substrate_not_ready_requeue`
  flagged `precondition_unmet` (the diagnostic adjudication gate -- the
  self-route's premise did not hold; awaits `/failure-autopsy` to
  adjudicate the engaged-but-insufficient reading). **Currently queued
  (`experiment_queue.json` items[]): 1 item** -- V3-EXQ-640b SD-057 /
  MECH-346 / MECH-347 cue-authority lineage CLEAN EVIDENCE retest
  (claimed DLAPTOP-4.local 2026-06-08T06:46Z; re-runs the 640a scaffold
  cue-recall gain sweep now that modulatory-bias-selection-authority is
  VALIDATED via V3-EXQ-643a PASS; promotes 640a -> EVIDENCE; PRIMARY
  acceptance C_LIFT_PRIMARY = post_cue_approach_lift > 0 in the decisive
  cell ARM_CUE_g5_k10). V3-EXQ-653 E2 escape-affordance linker readiness
  microdiagnostic (4-arm forced-choice probe; readiness gates G0-G8; PASS
  routes back to full 603-lineage behavioural bridge re-test) was queued
  via /queue-experiment in the active 22:17Z claim and is being prepared
  for the runner. Substrate / governance landings since the 2026-06-06T01:10Z
  spec sync: (1) **MECH-314a Phase-2 + ARC-065 GAP-A `e2.world_forward` source
  amends** (ree-v3 main 2026-06-07) -- routed by V3-EXQ-648 / V3-EXQ-614e
  autopsies; the curiosity channel + the SHARED `cand_world_summaries`
  consumed by lateral_pfc/ofc/mech295/gated_policy/tonic_vigor now optionally
  re-source from SD-056-trained `e2.world_forward(z0, a_i)` predictions
  (cross-candidate spread ~0.1147) instead of the collapsed proposer first-step
  z_world (<0.01 under monostrategy). Bit-identical OFF; V3-EXQ-648a / V3-EXQ-649
  substrate-readiness validations queued with `cand_world_pairwise_dist`
  readiness preconditions. (2) **scaffolded_sd054_onboarding AMEND curriculum
  decomposition Stage-H** (ree-v3 main 2026-06-07) -- routed by
  failure_autopsy_V3-EXQ-603f. Stage-H isolated hazard-avoidance training
  inserted between P0 (safe goal-frozen warm-up) and P1 (combined wean), so
  the agent acquires hazard navigation BEFORE P1 throws it at hazards. 85/85
  scaffold contracts. (3) **ControlVector logging** (ree-v3 main 2026-06-07) --
  read-only default-OFF telemetry exposing the ARC-068-vs-MECH-320 collapse
  (opportunity cost + vigor both = `w*v_t` for the SAME MECH-320 v_t scalar);
  recommendation B logging only. (4) **SD-058 / MECH-357 instrumental-avoidance
  acquisition** (ree-v3 main 2026-06-07) -- new `ree_core/pfc/infralimbic_avoidance_gate.py`;
  per-Moscarello&LeDoux active avoidance is the resolution of a Pavlovian-instrumental
  conflict requiring ilPFC to SUPPRESS CeA-driven freezing. Three pieces (action-bias
  + freeze-suppression gate + eligibility-trace efficacy learning) under
  `use_instrumental_avoidance`; protective-scaffold anneal in Stage-H. LOAD-BEARING
  PREREQUISITE: legacy scaffold called sense() with NO harm args -- new
  `scaffold_feed_harm_stream` feeds env harm streams so PAG/SD-035/SD-058 actually
  see threat. 912 contracts. V3-EXQ-603h FAILed engaged-but-insufficient -- gate
  engaged + suppressed PAG freeze on all INTACT seeds but G_H_INTACT 0/3; routed
  to SD-059 below. (5) **SD-059 / MECH-358 escape-affordance bridge**
  (ree-v3 main 2026-06-08) -- closes the 603h directed-escape gap. Extends
  MECH-357's scalar avoidance_efficacy into a per-first-action-class credit table
  (relief half + safety half); per-candidate negative score-bias toward action
  classes carrying combined affordance credit under FUTURE threat. Distinct from
  reflexive SD-037/MECH-281 escape and from the unconditioned MECH-302/303/304
  rows. V3-EXQ-603i validation FAILed precondition_unmet (in pending_review).
  (6) **Three post-603i successor scaffolds** (ree-v3 main 7a0a417 + 58535af +
  6c856a5; trainable escape-affordance learner + trainable relief/safety heads
  upgrade + E2 escape-affordance linker reuse/readout over E2.world_forward) --
  NOT validated substrate, NOT changes to 603i; V3-EXQ-653 readiness microdiagnostic
  queued as the validation gate for the E2 linker scaffold; PASS routes back
  to a full 603-lineage behavioural bridge re-test. (7) **Runner ETA fix**
  (ree-v3 b559b5a, 2026-06-07) -- multi-stage experiments without PASS/FAIL
  per-cell verdict lines now use the stable median-per-cell ETA path instead
  of the divergent live-extrapolation fallback (640b "all over the place" symptom
  fixed; takes effect on next runner restart). Bottleneck (continuation): the
  **GAP-2 / GAP-7 ecological-evidence loop** remains primary. SD-058 +
  SD-059 are the latest substrate-side answers to the 603-lineage survival /
  escape leg, with the V3-EXQ-603i adjudication + V3-EXQ-653 E2 linker readiness
  + V3-EXQ-640b cue-authority CLEAN EVIDENCE retest as the in-flight gates.

- **2026-06-06T01:10Z nightly read.** `evidence/experiments/` flat
  top-level holds **285 `v3_exq_*.json` manifests**. Per-machine fleet
  `runner_status/*.json` carries the DLAPTOP-4.local card at 610
  completed queue_ids plus Daniel-PC 28, EWIN-PC 77, ree-cloud-1..4 at
  254 / 199 / 154 / 150, ree-worker-3 at 133. **Pending review queue
  (regenerated 2026-06-05T14:55Z) reads 0 items.** **Currently queued
  (`experiment_queue.json` items[]): 3 items** -- V3-EXQ-640a SD-057
  cue-AUTHORITY GAIN SWEEP (claimed ree-cloud-2 2026-06-05T15:20Z;
  measurement successor to V3-EXQ-640; 2-axis factorial
  `cue_recall_gain` x `incentive_drive_kappa_weight`, 7 conditions x
  3 seeds; gates the planned V3-EXQ-638b interoceptive need-gating
  substrate); V3-EXQ-610f INV-074 / MECH-333 / MECH-334
  crystallization-necessity TRUE-NEGATIVE-CONTROL retest (claimed
  ree-cloud-1 2026-06-05T14:58Z; supersedes V3-EXQ-610e confounded
  control; strips noise_floor + E3 score-diversity + dACC anti-recency
  in ARM_0, phase-dependent entropy bonus 0.02 in phases 0-2 vs sweep
  {0, 0.005, 0.02} in phase 3, ARM_4 floor-on for MECH-341/313 contrast);
  V3-EXQ-641 coherence-ablation (claimed DLAPTOP-4.local 2026-06-05T18:59Z;
  paired A/B selectors over identical hippocampal-rollout candidate
  pools settling the shared binding + path-integral intakes' discriminator
  C(tau) non-reducibility to E(tau); gap-relative coherence authority
  + random-C control + perturbation rebinding arm). Substrate /
  governance landings since the 2026-06-05T19:38Z spec sync:
  (1) **SD-016 Path 3 feedforward cue->slot tagger** (ree-v3 main
  88695ed) -- replaces ONLY the slot-SELECTION scores in
  `E1DeepPredictor.extract_cue_context` (the saddle-stuck q.k attention
  V3-EXQ-418i diagnosed at the uniform ln(num_slots) saddle) with a
  fresh feedforward MLP `Linear -> ReLU -> Linear`; slot-CONTENT path +
  cue_action_proj + cue_terrain_proj UNTOUCHED; no new supervised
  target invented -- random MLP sits OFF the saddle so the existing
  terrain_loss gradient flows into it from step 0. Cached
  `_last_cue_slot_weights` read-only diagnostic + 5 new contracts in
  tests/contracts/test_sd016_cue_slot_tagger.py. V3-EXQ-418m
  substrate-readiness diagnostic (PRIMARY mean selection entropy < 2.5
  vs pinned ln(16) ~ 2.773 with the tagger ON) queued via /queue-experiment.
  (2) **MECH-353 / MECH-354 / MECH-355 mint** (REE_assembly master
  0a9dda6b99 claims.yaml + 7d89ffd0ba affect_primitives consolidation +
  9d47b3a945 blocked_agency row + 045dac6b9d MECH-303/304
  reuniens/remote-recent enrichment + 7c6a1f0b55 MECH-355 design pass +
  ea53570ec9 MECH-354 design pass) -- three proto-feeling-stream
  candidates that the 2026-06-05 affect-stream lit-pulls confirmed are
  distinct from existing harm primitives. MECH-353 z_block /
  blocked-agency (V3 candidate, v3_pending; SD-029 comparator over
  action-outcome channel with external-attribution + capacity gates;
  consumers: assert MECH-320 -> decommit MECH-342 gated by ARC-016 ->
  withdraw at capacity collapse). MECH-354 fatigue stop-recover (V3
  candidate, v3_pending; SD-012 side, SD-048 host; Meyniel two-bound
  accumulator wiring via MECH-342 release actuator). MECH-355 soothing
  autonomic state-gain modulator (V4-social, substrate_conditional;
  DECAY-ACCELERATION on MECH-219 z_harm_a recovery_rate + SD-032e
  drive_bias leak; multiplicative-on-existing-state so soothing != sedation).
  All three doc + claims-only (no substrate code). (3) **MECH-303 /
  MECH-304 reuniens-thalamic-relay + remote/recent enrichment**
  (REE_assembly master 045dac6b9d) -- amends the safety cluster with
  the midline-thalamic relay (nucleus reuniens -> BLA) + remote-vs-recent
  time-since-encoding dependence surfaced by the Silva 2021 lit anchor;
  amend not new claim (option (i) -- enrichment + named candidate
  third-sub-mechanism flag instead of minting on a single rodent
  anchor). (4) **goal_pipeline:GAP-7 frontmatter correction**
  (REE_assembly master f798bd1b80) -- closure-map node frontmatter
  rewritten to reflect that L2-L3-L4 + L6-L7 substrate already landed
  2026-06-04 as SD-057 (MECH-344..348), not an unbuilt placeholder;
  resume_condition repointed at the real remaining work (L9
  wanting!=liking acceptance, GAP-2-gated + in-flight 637/640a
  validation). (5) **SD-037 axis (a) + axis (b) closure-plan
  frontmatter** (REE_assembly master 45f70df66d) -- both SD-037
  plan-of-record docs gain real `closure_plan:` frontmatter so the
  Explorer Closure tab renders them as live cards (axis (a) 100%
  concluded-negatively per V3-EXQ-620 zero-distribution; axis (b)
  P1b blocked_pending_substrate against the behavioural-diversity
  cluster). (6) **Thought-intake sweep cleared 0 unprocessed** (REE_assembly
  master 6326680a24 + 56f97b5184 + d962a67c5b + 7e8d785412) -- 8
  remaining sweep-unprocessed PARTIALs handled (4 mark-only +
  4 forward-content intakes incl. orienting drive, therapy-bridge,
  cross-version V4/V5 missing-bits, grammar-LLMs as V5 mining
  scaffold); 12 incorporated-but-unmarked thoughts backfilled.
  Bottleneck unchanged: the **ecological-evidence v3_pending lift
  requirement** for the GAP-2 / GAP-7 / cue-authority loop -- 638a
  validated the formation half (cue fires + bank populated) but
  failed the C3 contact-lift, autopsy routed cue-to-action AUTHORITY
  as the smallest next step (640 / 640a / then 638b), and V3-EXQ-640a
  is the in-flight measurement gate. The MECH-353 / MECH-354 / MECH-355
  cluster registration is a doc-only sweep that captures the affect
  primitives the harm-stream substrate misses without diverting
  compute or critical path.

- **2026-06-04T01:10Z nightly read.** `evidence/experiments/` contains
  **994 v3_exq_* manifests** (recursive count incl. nested per-run dirs;
  289 flat top-level v3_exq_* manifests). Per-machine fleet
  `runner_status/*.json` aggregates to **779 unique V3 queue_ids
  completed** across all 8 workers after dedup (242 PASS / 406 FAIL /
  89 ERROR / 41 UNKNOWN / 1 INCONCLUSIVE; 1598 total completion records
  before dedup, reflecting cross-machine retries + smoke runs). The
  DLAPTOP-4.local card alone carries 610 completed queue_ids. Per
  CLAUDE.md "Phase 3 note" the central `runner_status.json` was
  decoupled on 2026-05-29; per-machine `runner_status/<host>.json`
  files are the authoritative surface. **Pending review queue
  (regenerated 2026-06-03T19:58Z) reads 0 items** -- all walked in
  today's two governance cycles (16:59Z + 19:57Z). **Currently queued
  (`experiment_queue.json` items[]): 7 items** -- V3-EXQ-634c
  scaffolded_sd054_onboarding seeding-calibration 4-arm readiness
  diagnostic (claimed DLAPTOP-4.local 2026-06-03T22:04Z; supersedes
  634b after consolidation amend VALIDATED but G3
  anti-correlated-with-foraging exposed a new contact-gating /
  seeding-firing-threshold mismatch); V3-EXQ-610e INV-074 / MECH-333 /
  MECH-334 crystallization-necessity retest with three-prescription
  harness fix (claimed ree-cloud-1 2026-06-03T21:19Z; supersedes 610d
  near-uniform untrained-policy no-op signature; verifies real REINFORCE
  + stepped expansion-parameter optimizer + EWC penalty in Phase-3 loss
  via mandatory startup assertion); V3-EXQ-463b MECH-268 dACC conflict
  saturation (claimed ree-cloud-4); V3-EXQ-464b MECH-266 competing
  goals (claimed ree-cloud-2); V3-EXQ-466b SD-034 satisficing residue
  discharge (claimed ree-cloud-3); V3-EXQ-467b MECH-266 mode stickiness
  (pending); V3-EXQ-468b SD-034 + MECH-268 commitment vs contradiction
  (pending). Five of the seven are the GAP-4 OCD behavioural *b cohort
  queued late in the session against the GAP-3-landed CausalGridWorldV2
  env extensions + GAP-11 committed_mode_curriculum. Substrate /
  governance landings since the 2026-06-03T01:10Z snapshot:
  (1) **scaffolded_sd054_onboarding nursery / feeding scaffold amend**
  (forced-benefit Stage-0 nursery + survival levers + P2 measurement
  guard; ree-v3 main) -- routed by V3-EXQ-603e-626a-622 failure-autopsy
  concluding the update_z_goal-wiring amend was necessary-but-insufficient
  (2/3 seeds never reach foraging competence; P2 hazard env starves
  benefit_exposure). Adds infant-nursery forced-feed positive control,
  curriculum hold-fraction lever, P2 hazard_food_attraction guard, and
  five-way interpretation-grid + substrate-gate helper. (2)
  **scaffolded_sd054_onboarding developmental-window amend** (Stage-0b
  protected consolidation + contact-gated P1/P2; ree-v3 main) -- routed
  by V3-EXQ-634 design-error review concluding GoalState.update
  ALWAYS-decays the persistent z_goal attractor before the benefit-gated
  pull (washing out the Stage-0 trace across UNFED steps). Adds Stage-0b
  retention gate + contact_gated_goal_updates path so decay-only is
  reserved for mature tests, not the nursery gate. (3)
  **scaffolded_sd054_onboarding seeding-calibration amend** (decoupled
  contact-gating threshold + GoalConfig seeding-magnitude propagation +
  consumption-event-gated G3 readout; ree-v3 main 15053a3) -- routed by
  V3-EXQ-634b autopsy: consolidation half VALIDATED (G0b retention 3/3
  + n_decay_only=0) but G3 anti-correlated-with-foraging revealed
  contact-gating decoupled from seeding firing threshold. (4)
  **modulatory-bias-selection-authority substrate** (gap-relative
  E3.select authority; ree-v3 main) -- routed by 604a / 624a / 614d
  cluster autopsy diagnosing fixed-small modulatory magnitudes drowned
  by primary-score range. Rescales composed modulatory bias so
  range(mod) == gain * raw_score_range; gain=0.5 keeps modulatory
  competitive in near-ties but subdominant when primary harm/goal gap
  exceeds gain*range. Necessary-but-not-sufficient for the curiosity
  lever (604a had curiosity_bias=0.0 genuinely). V3-EXQ-635 substrate-
  readiness PASS (within-class lever lift +0.446 across 19
  authority-normalized ticks; validates substrate). (5) **MECH-306
  promoted candidate -> provisional** (REE_assembly master 11c043ea79)
  -- V3-EXQ-627 sustained_drive_trace_validation PASS (exp_conf=0.773 >
  0.62 gate) satisfies the v3_pending gate; one substrate-conditional
  hold cleared. (6) **Governance evening cycle** (REE_assembly master
  8c85f06e5a) -- 6 reviews closed (2 PASS, 4 FAIL); 4 user-approved
  failure-autopsy dispositions applied (514l / 632 / 634 / 610c all
  routed to non_contributory + epistemic_category substrate_ceiling +
  pending_retest_after_substrate). MECH-094 evidence corroborated by
  V3-EXQ-633 PASS (3/3, supports). (7) **commitment_closure:GAP-8
  SD-033b behavioural validation** -- audit confirmed V3-EXQ-485b /
  485c never ran (not silent-drop); authored + smoke-tested + queued
  (ree-v3 main 9f45b0f). GAP-8 status node blocked -> in-progress.
  (8) **goal_pipeline:GAP-7 incentive-salience ratified into
  plan-of-record** (REE_assembly master db72095d46) -- L0-L9 closure
  map embedded in goal_pipeline_plan.md; L1 forced-seed positive-control
  V3-EXQ-626b queued (ree-v3 main ab55916). (9) **commitment_closure:GAP-4
  OCD behavioural *b cohort** -- 7 scripts authored, smoke-tested, and
  queued at priority 290 against the SD-033b (OFC), SD-034 (closure
  operator), MECH-266 (asymmetric hysteresis), MECH-268 (dACC conflict
  saturation), and MECH-090 (commit entry) substrates. Substrate side
  resolved 2026-06-02 (MECH-090 + MECH-342 validated by 592g). (10)
  **closure-drift checker enhancements** (REE_assembly master
  3133d10723) -- check_closure_drift.py gains lineage-advanced +
  claims-reclassified-since signals so GAP-2-class stale-since gaps
  surface instead of hiding. 0 drifted post-walk. (11) **Brain-map
  visualization rebuild** -- coronal MRI backdrop, full region re-drape
  onto anatomy, three-plane linked view (sagittal + coronal + axial)
  generated from a single region table. Multiple user-feedback
  iterations across the day. Bottleneck (continuation from yesterday's
  framing): the **ecological-evidence v3_pending lift requirement**
  remains the dominant blocker, but specific axes advanced today --
  (a) the scaffolded_sd054_onboarding substrate has now been amended
  THREE times in one day (wiring -> nursery scaffold ->
  developmental-window -> seeding-calibration); V3-EXQ-634c is the
  adjudicating bit on runtime readiness; (b) the
  modulatory-bias-selection-authority substrate now lets MECH-314 /
  MECH-320 / MECH-341 modulatory levers actually influence the
  committed argmin (gap-relative scaling), so pending behavioural
  re-runs on these claims can produce non-vacuous evidence; (c)
  MECH-306 has cleared the V3-pending gate and is now provisional.
  **V3-EXQ-610e (INV-074 crystallization-necessity)** remains the
  adjudicating bit between the INV-074 plasticity-injection closure
  prediction vs the prior 610c/d untrained-policy no-op artefact --
  610e wires REINFORCE policy training + stepped expansion-parameter
  optimizer + EWC penalty into the Phase-3 loss, each verified by a
  mandatory startup assertion.

- **2026-06-03T01:10Z nightly read.** `evidence/experiments/`
  contains **908 v3_exq_* manifests** on disk; per-machine
  `runner_status/DLAPTOP-4.local.json` carries 613 completed queue_ids
  (the local Mac's contribution to fleet totals). The central runner_status
  was decoupled in the 2026-05-29 Phase-3 cutover -- per-machine status
  files under `evidence/experiments/runner_status/` are the authoritative
  surface now (DLAPTOP-4.local + Daniel-PC + EWIN-PC + ree-cloud-1 through
  ree-cloud-4 + ree-worker-3). **Pending review queue (regenerated
  2026-06-02T16:55Z) reads 2 unclaimed manifests** -- V3-EXQ-626a (goal-pipeline
  developmental-window FAIL) and V3-EXQ-625c (SD-037 axis-b Phase 1b
  dynamic-crossings FAIL, non_contributory). **Currently queued
  (experiment_queue.json items[]): 2 items** -- V3-EXQ-603e
  Q-045/MECH-313/MECH-260 scaffolded_sd054_onboarding 5-arm at restored
  budget (claimed DLAPTOP-4.local 2026-06-02T06:46Z; supersedes 603d;
  z_goal_enabled=True + drive_weight=2.0 fix verified pre-run); V3-EXQ-614d
  MECH-341 within-class temperature corrected-harness 4-arm sweep
  (pending DLAPTOP-4.local; supersedes 614c instrumentation-defect FAIL).
  Substrate / governance landings since the 2026-06-02T01:10Z snapshot:
  (1) **MECH-090 R-c continuation Phase-2 env-source follow-on** (ree-v3
  main fa026a0 + 60d1a90; REE_assembly master b23ad1a125 + 6be3673781) --
  closes the named Phase-2 follow-on from the 2026-05-29 R-c continuation:
  CausalGridWorldV2 emits info[mech090_readiness_outcome] = clip(1 -
  mean(limb_damage), 0, 1) under env-only kwarg
  mech090_readiness_outcome_enabled (default False, ABSENT-WHEN-DISABLED);
  REEAgent.sense forwards into commit_readiness.update; agent UNCHANGED
  consumer + seam (the 2026-05-29 pass) now has a real source. Validation
  V3-EXQ-630 ecological across-tick 3-arm queued (was claimed and dropped
  from snapshot during fleet outage recovery). 719/719 contracts + 7/7
  preflight PASS. (2) **MECH-342 maintenance-time commitment-release coupling
  (B3b)** (ree-v3 main 780d12f + REE_assembly master 625e218779) --
  release-side complement to MECH-090 admission predicate; closes V3-EXQ-592f
  reach gap. Pure-arithmetic regulator commit_maintenance_release.py;
  OR-composition of decisiveness + nav_competence deficits, drift-to-bound
  + reengagement leak; distinct from MECH-090 / MECH-091 / ARC-028 /
  MECH-269b / MECH-340. V3-EXQ-592g PASS validation 2026-06-02 (all 6
  criteria); V3-EXQ-629 ecological evidence run queued (claimed + dropped
  mid-day fleet outage). MECH-342 stays candidate / v3_pending (592g
  diagnostic). (3) **scaffolded_sd054_onboarding AMEND** (update_z_goal
  wiring + Stage-0 positive control; ree-v3 main deb24cc + d09af0e;
  REE_assembly master 36b0130ecf) -- root-cause fix for V3-EXQ-603d
  Class-1 harness/wiring artifact (scheduler never called update_z_goal
  -> z_goal zero-init every step every arm). TWO-PART FIX: wiring +
  z_goal_enabled=True+drive_weight=2.0 config. V3-EXQ-603e successor
  queued. (4) **MECH-341 stratified_within_class_temperature amend**
  (2026-06-01; ree-v3) -- (a) within-class proportional sampling lever
  to decouple Layer B within-class sub-axis from across-class sub-axis;
  (b) A-vs-B partial-redundancy probe naming via existing independent
  flags. 655/655 contracts. V3-EXQ-614c queued -> FAILed instrumentation-
  defect -> 614d corrected re-run queued. (5) **Pre-governance disposition
  V3-EXQ-592f + V3-EXQ-592g** (REE_assembly 01144f9bf6) -- 592f re-tagged
  does_not_support -> non_contributory + epistemic_category substrate_ceiling
  + cleared pending_retest_after_substrate (reach gap closed by MECH-342);
  MECH-090 unchanged (release capability lives on dependent MECH-342); 592g
  reviewed PASS, MECH-342 validation_note added, substrate_queue MECH-342
  status -> implemented_validated_v3_exq_592g. pending_review = 0 indexed
  after walk. (6) **Cross-fleet experiment wave** (ree-v3 main 34e6369 +
  829f6b1 + b7fae0a + e9a0b87 + others) -- queued V3-EXQ-627 MECH-306
  sustained_drive_trace 2-arm evidence; V3-EXQ-604a MECH-314 curiosity
  validation on SD-056 substrate (supersedes 604); V3-EXQ-628 MECH-319
  replay/caller_sim falsifier evidence; V3-EXQ-629 MECH-342 ecological;
  V3-EXQ-630 ARC-029 across-tick. All claimed and dropped from queue
  snapshot mid-day during fleet outage recovery; will resurface as
  results / heartbeats on origin via the phase3-* writers as workers
  complete or release. (7) **Fleet outage diagnosed + recovered**
  (2026-06-02T23:50Z) -- hub-writer wedge (not migration), all 4 cloud
  runners crash-looping at startup since ~22:05-23:06Z due to
  hub-writer wedge causing frozen queue snapshot to keep V3-EXQ-610c
  pending despite FAIL+completed status (test_queue_integrity FAIL ->
  start-limit-hit); non-destructive recovery via backup+clean tree fix.
  Cloud-2/3/4 cut over to PHASE3_COMMANDS_VIA_COORDINATOR=1 same
  recovery; hub runner intentionally stopped pending operator decision.
  (8) **Plan-doc updates** -- commitment_closure:GAP-4 owner_exq advanced
  to V3-EXQ-629 (ecological MECH-342 evidence) with stale-631 correction
  (REE_assembly master 194400a994); arc_062_rule_apprehension:GAP-K
  owner_exq repointed to V3-EXQ-628 MECH-319 evidence falsifier
  (REE_assembly master 6e23af6fc3). (9) **Epoch stale-evidence
  bookkeeping closed** (REE_assembly master 2be6faafd7) -- 483c
  supersession recorded + 17 B.3 nested-manifest stale flags written;
  GOVERNANCE CONSEQUENCE: 483c was the sole genuine exp entry for
  SD-037 / MECH-280 / MECH-281, so exp_conf for all three -> 0.0;
  SD-037 retains other active streams (483b / 620b / 625c). Bottleneck
  (updated framing): the dominant blocker has shifted from the
  scaffolded_sd054_onboarding harness/wiring gap (now fixed; V3-EXQ-603e
  in flight) to the **ecological-evidence v3_pending lift requirement**
  -- multiple substrates (MECH-090 R-c admission + R-c continuation +
  MECH-342 release-side; MECH-319 simulation-mode rule gate;
  MECH-306/314 score-bias contributors) now have substrate-readiness
  PASS + diagnostic validation in hand, but the V3-pending governance
  gate forbids promotion regardless of evidence count until ecological
  evidence-grade runs PASS. The five ecological evidence runs queued
  today (604a / 627 / 628 / 629 / 630) form the next governance batch.
  **V3-EXQ-603e (Q-045/MECH-313/MECH-260 on hook-fixed substrate)**
  remains the adjudicating bit between scaffolded_sd054_onboarding
  harness-bug-fixed vs deeper z_goal-formation regression; P0 positive
  control is the adjudicator.
- **2026-06-02T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **794 cumulative
  completions (+8 since 2026-06-01T01:10Z read)** (195 PASS / 319 FAIL
  / 87 ERROR / 193 UNKNOWN; deltas: PASS +3, FAIL +5, ERROR +0, UNKNOWN
  +0); last_updated 2026-06-01T23:40:21Z -- ~1h35m fresh at this read.
  Today's eight returns: **V3-EXQ-603d FAIL 2026-06-01T09:53Z** on the
  scaffolded_sd054_onboarding 4-arm 5th-iter behavioural retest
  (Q-045/MECH-313/MECH-260; failure-autopsy load-bearing finding:
  ScaffoldedSD054OnboardingScheduler._train_episode never calls
  agent.update_z_goal -> z_goal stayed zero-init every step of every arm;
  C4 SUBSTRATE_FAILURE is a Class-1 harness/wiring artifact LIVING IN THE
  SUBSTRATE MODULE, NOT a substrate-ceiling falsification; routed to
  /implement-substrate amend on scaffolded_sd054_onboarding + 603e re-issue
  at restored budget; all three claims stay pending_retest_after_substrate /
  non_contributory); **V3-EXQ-625 PASS 2026-06-01T11:28Z** on the SD-037
  axis-b SD-029 curriculum overlay -- but headline PASS was vacuous (script
  defined PASS=ran-cleanly + n>0 distributions decoupled from the substrate
  acceptance gate, acceptance_pass=false with z_harm_a_norm IDENTICALLY 0.0
  across 1027 ticks / 3 seeds because the gap4 build_config path routed
  through REEConfig.goal_stream() WITHOUT SD-011 affective-harm-stream
  flags; same artifact as V3-EXQ-620; /diagnose-errors landed
  goal_pipeline_tier1.build_config guarded opt-in + queued 620b/625b
  successors with stream ON; 620/625 marked superseded by 620b/625b);
  **V3-EXQ-514k FAIL 2026-06-01T11:53Z** on SD-049 / SD-015 ecological
  wanting / liking dissociation (confounded by GAP-2 SP-CEM + missing
  object-bound substrate per failure_autopsy_V3-EXQ-626 routing; dissoc=0.0;
  SD-049 / SD-015 weakens -> non_contributory; MECH-229/230 stay
  non_contributory); **V3-EXQ-614c FAIL 2026-06-01T12:45Z** on MECH-341
  within-class temperature sweep (instrumentation defect: C2 vacuous --
  ARM_1/2/3 bit-identical per seed because reported
  selected_class_entropy_nats measured at experiment's own score-layer
  argmin upstream of the within-class temperature lever; C1 mis-specified
  per-seed band vs cross-seed mean; C3 substrate PASS 3/3 all arms;
  MECH-341 weakens -> non_contributory + 614d corrected re-run flag);
  **V3-EXQ-623 PASS 2026-06-01T15:20Z** on MECH-104 phasic-spike volatility
  interrupt + de-commitment ablation (ON arm n_decommit 24/31 vs ABL 0;
  clean discriminative; supports MECH-104; supersedes 126; second 07:07
  empty run marked superseded by canonical 15:20); **V3-EXQ-626 FAIL
  2026-06-01T15:27Z** on goal-pipeline developmental-window 4-arm dissociation
  (Class-1 HARNESS BUG: bespoke _run_episode never called agent.update_z_goal
  so z_goal stayed zero-init across all 4 arms; C2/C3 vacuously "true" at
  0<ceiling; NOT a substrate formation regression -- 622 S0 PASS + 582a
  refute regression; 626a harness-fix queued same day with P0 positive
  control as adjudicating bit; 626 superseded by 626a); **V3-EXQ-620b
  PASS 2026-06-01T19:01Z** on the SD-037 axis-a Phase 1 consumer-input
  distributions with affective harm stream ON (supersedes V3-EXQ-620
  vacuous-zero artifact; pooled p70 z_harm_a_norm ~0.33 vs old identical
  0.0; BLA/CeA now receive signal; governance flag: reassess whether
  SD-037 axis-b env-curriculum work was necessary); **V3-EXQ-592f FAIL
  2026-06-01T19:43Z** on the MECH-090 commitment-state transition authority
  probe (minimal controlled state-machine probe via real REEAgent.select_action
  + stubbed E3 SelectionResult forcing committed=True with controlled score
  margins; expected diagnostic FAIL_NO_RELEASE_AUTHORITY confirmed from
  592e autopsy -- nav blocks move, beta/e3 occupancy remains 1.0; no claim
  weighting). **Pending review queue (regenerated 2026-06-01T18:09Z) reads
  0 items** -- all walked via two /governance cycles (16:58 + 17:56;
  603d intentionally left for /failure-autopsy then marked reviewed
  after disposition; 625 left for /diagnose-errors then marked reviewed
  after 625b/620b queued). **Currently queued (`experiment_queue.json`
  items[]): 3 items, all claimed** -- V3-EXQ-610c INV-074 / MECH-333 /
  MECH-334 post-Phase-3-enrichment crystallization-necessity retest
  (claimed ree-cloud-3 2026-06-01T18:54Z; supersedes 610b; tests
  prediction that ARM_0 control NOW collapses under IGW-023 substrate
  amend), V3-EXQ-626a goal-pipeline harness-fix re-run (claimed
  ree-cloud-1 2026-06-01T16:57Z; supersedes 626; P0 positive control is
  the harness-bug-vs-object-binding adjudicating bit), V3-EXQ-624 ARC-068
  / MECH-320 Niv-vs-Salamone opportunity-cost-vs-effort-cost dissociation
  (claimed DLAPTOP-4.local 2026-06-01T19:58Z; affinity flipped DLAPTOP-4
  -> any earlier in day for cloud parallelism, then DLAPTOP-4 claimed
  after laptop yielded to cloud on 610c/626a). Substrate / governance
  landings since the 2026-06-01T01:10Z snapshot: (1) **Two /governance
  cycles same day** -- 16:58Z walked 8 pending experiments (614c
  MECH-341 -> non_contributory instrumentation_defect; 514k SD-049/SD-015
  -> non_contributory GAP-2/object-binding confound; 610b INV-074/MECH-333/
  MECH-334 -> non_contributory + substrate_ceiling + pending_retest;
  626 -> superseded by 626a; 623 -> supports MECH-104 supersedes 126);
  17:56Z walked 2 pending (603d -> non_contributory + substrate_ceiling +
  pending_retest_after_substrate; 625/620 -> superseded by 625b/620b);
  pending_review=0 both cycles; substrate_queue
  test_bed_enrichment_crystallization_necessity created priority 3 and
  scaffolded_sd054_onboarding amend_pending priority 1 (upstream of 622
  anneal-rate amend). (2) **Phase 3 telemetry sync_daemon sole-writer
  hardening** (ree-v3 main) -- documented hub architecture (sync_daemon.
  phase3_heartbeat_writer owns runner_heartbeats/ + runner_status/ on
  GitHub; hub runner POST-only via PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1);
  added deploy/shadow.conf.hub.example; sync_daemon now auto-reverts
  exclusive telemetry dirt before refusing ticks (fixes result-writer wedge
  when only heartbeat paths dirty); CLAUDE.md + FLEET_CHECKLIST updated.
  (3) **Cloud-scaler hub deploy + cloud-4 surge threshold 3->2** (ree-v3
  main 75b23f9) -- cloud-scaler.{service,timer} units + hcloud CLI +
  HCLOUD_TOKEN installed on hub (`ree-cloud-1`); hub timer is now
  authoritative 5-min cadence with GHA 6-hourly backstop; threshold
  lowered to surge ree-cloud-4 on 2-deep backlog. (4) **Cloud-1 hub
  runner re-enabled under Phase 3 co-tenancy** -- appended
  PHASE3_DISABLE_RUNNER_HEARTBEAT_WRITE=1 to hub shadow.conf;
  ree-runner service enabled on hub; journal confirms FILE WRITES skipped
  gate; coordinator POST path live (`/api/machines` ree-cloud-1 fresh).
  (5) **Post-result checkout alignment hardening** (ree-v3 +
  REE_assembly, local-only) -- experiment_runner prepull stash of
  untracked flat manifests + runner signals; `_report_result_and_align`
  runs POST /result then immediate + delayed (45s/120s) `_sync_pull_tick`;
  serve.py replaces ff-only auto-pull with `align_ree_assembly_checkout` at
  startup + every 5min (respects TASK_CLAIMS evidence/ skip); contracts
  7/7 PASS. (6) **Multiple failure autopsies landed** -- V3-EXQ-614c
  (MECH-341 within-class temperature instrumentation defect) + V3-EXQ-610a
  (INV-074 / MECH-333 / MECH-334 / MECH-341 substrate-ceiling re-read on
  the recovered manifest; D2 control did NOT collapse -> environment
  test-bed ceiling) + V3-EXQ-626 (Class-1 harness bug, z_goal never
  driven) + V3-EXQ-603d (Class-1 harness/wiring artifact LIVING IN THE
  SUBSTRATE MODULE; scaffolded_sd054_onboarding._train_episode never
  calls update_z_goal) + V3-EXQ-592e (MECH-090 beta-gate readiness-failure
  release path missing; no readiness-failure path releases an already-
  elevated beta latch or clears E3 committed state). (7) **Goal /
  wanting / liking stream repair intake** -- docs/thoughts/
  2026-06-01_goal_wanting_liking_stream_repair.md + goal_stream_repair_
  diagnostic_ladder_2026-06-01.md + claim_gap_2026-06-01.md + lit-pull
  evidence/literature/literature_synthesis_object_bound_incentive_salience
  + contracts/test_goalstate_forced_seed_positive_control.py (6/6 PASS);
  authored as proposals not claims; identifies missing object-bound
  incentive-salience layer as the suspected real abstraction gap behind
  514k. (8) **Developmental-window memo** (REE_assembly master
  12d2a48310) -- evidence/planning/goal_pipeline_developmental_window_
  diagnostic_memo_2026-06-01.md + docs/thoughts/2026-06-01_plasticity_
  window_neuromodulators.md persisted as plan-of-record for V3-EXQ-626/626a
  + project memory project_plasticity_window_neuromodulators.md
  registering ACh/PV-interneuron/BDNF/state-dependent plasticity-window
  as long-horizon V4-or-late-V3 territory. Bottleneck (updated framing):
  the dominant blocker is now the **scaffolded_sd054_onboarding harness /
  wiring gap** identified by V3-EXQ-603d autopsy + V3-EXQ-626 autopsy
  (same epistemic class, different scope). The substrate scheduler
  `_train_episode` never calls `agent.update_z_goal(benefit, drive)` so
  z_goal stays at zero-init across every step of every arm; the C4
  SUBSTRATE_FAILURE classification in 603d's manifest is a Class-1
  harness/wiring artifact, NOT a substrate-ceiling falsification.
  /implement-substrate AMEND on scaffolded_sd054_onboarding (wire
  update_z_goal into scheduler + Stage-0 positive-control gate) is
  strictly UPSTREAM of the 622 anneal-rate amend and the V3-EXQ-603e
  re-issue at restored budget. Until that amend lands, the substrate-
  uniform z_goal-zero pattern across 603 lineage / 626 / 540a-e / 590a /
  591 / 598 / 598b remains active. **V3-EXQ-626a P0 positive-control
  gate is the adjudicating bit between harness-bug and object-binding
  abstraction-gap hypotheses** (PASS confirms harness-bug -> object-
  binding ladder Stage 1+; FAIL contradicts 622 S0 -> /failure-autopsy
  on GoalState.update). **V3-EXQ-610c on ree-cloud-3** (INV-074 / MECH-333
  / MECH-334 post-Phase-3-enrichment crystallization-necessity retest)
  and **V3-EXQ-624 on DLAPTOP-4** (ARC-068 / MECH-320 Niv-vs-Salamone)
  are the next two scientific reads. **Governance flag carried forward:**
  once V3-EXQ-620b axis-a PASS is governance-marked superseder-of-620,
  reassess whether the SD-037 axis-b env-curriculum work was necessary
  (the original 620 zero artifact underpinned that plan-of-record). The
  V3-EXQ-543k disposition gap (carried forward from 2026-05-21) remains
  outstanding.
- **2026-06-01T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **786 cumulative
  completions (+13 since 2026-05-31T01:10Z read)** (192 PASS / 314 FAIL
  / 87 ERROR / 193 UNKNOWN; deltas: PASS +7, FAIL +3, ERROR +3, UNKNOWN
  +0); last_updated 2026-06-01T01:11:28Z -- live at this read.
  Today's ten returns: **V3-EXQ-569d PASS 2026-05-31T05:36Z** on the
  ARC-065 / MECH-341 floor-recalibrated falsifier successor; **V3-EXQ-519b
  PASS 2026-05-31T06:59Z** on the SD-051 / MECH-304 conditioned-safety-
  store readiness retest after the MECH-302 gate (c) was lifted;
  **V3-EXQ-615 PASS 2026-05-31T09:31Z** on the ARC-065 Rung-1 matched-
  entropy control (ARM_2 ALL_ON 1.111 nats vs single-class collapse on
  ARM_0 BASE_OFF and ARM_1 MATCHED_NOISE -- clean architectural-necessity
  discrimination); **V3-EXQ-617 PASS 2026-05-31T11:31Z** on SD-056 multi-
  step rollout stability amend substrate-readiness; **V3-EXQ-616 FAIL
  2026-05-31T14:15Z** on Q-054 entropy_bias_scale sweep (bit-identical
  per-seed results across scales {1.0, 2.0, 4.0, 8.0} -- mathematical
  proof MECH-341 isolation structurally not reachable via the score-layer
  scale lever); **V3-EXQ-618 PASS 2026-05-31T17:59Z** on SD-049 Phase 3
  SD-032 consumer cascade substrate-readiness (3-arm Phase A end-to-end
  smoke + Phase B direct-API probes); **V3-EXQ-614b FAIL 2026-05-31T18:20Z**
  on MECH-341 P3 behavioural falsifier under SD-056-amended substrate
  (substrate-coupling FAIL -- ARM_2 ALL_ON highest absolute entropy of any
  614 run at 0.800 nats but MECH-341 marginal contribution shrank to 0.087
  vs 0.100 C2 threshold because upstream cluster now does more diversity
  work; NOT a falsification per failure-autopsy verdict); **V3-EXQ-620
  PASS 2026-05-31T18:36Z** on the SD-037 axis (a) Phase 1 consumer-input
  distributions diagnostic (per-step BLA/CeA/PAG/dACC consumer-input
  distributions logged over fishtank baseline; closes plan Phase 1);
  **V3-EXQ-621 ERROR 2026-05-31T20:23Z** on the scaffolded_sd054_onboarding
  substrate-readiness diagnostic (runner sentinel misclassification +
  missing emit_outcome; manifest recovered from ree-cloud-3 and superseded
  by V3-EXQ-621a); **V3-EXQ-621a PASS 2026-05-31T23:09Z** on the
  scaffolded_sd054_onboarding substrate-readiness diagnostic with
  emit_outcome + P1 survival diagnostics + per-cell p1_episode_lengths.
  **Pending review queue (regenerated 2026-05-31T19:25Z) reads 0 items**
  -- all walked via the active `/governance` cycle (opened 2026-05-31T19:08Z).
  **Currently queued (`experiment_queue.json` items[]): 0 items** at this
  read (the most recent additions V3-EXQ-621a + V3-EXQ-622 were claimed by
  runners and removed from the items[] snapshot under the Phase 3
  coordinator-authoritative queueing path; check the coordinator DB
  + `runner_status` for in-flight state). The recently-queued V3-EXQ-622
  staged goal-stream S0-S3 (decomposes the 621 z_goal failure into a
  four-stage curriculum to localise where the substrate-uniform
  monostrategy collapse re-emerges) is the next-up validation. Substrate /
  governance landings since the 2026-05-31T01:10Z snapshot: (1)
  **scaffolded_sd054_onboarding substrate** (ree-v3 main 28ebd3d) --
  closes behavioral_diversity_isolation:GAP-C prereq (2) substrate
  landing; new experiment-harness scheduler at
  `experiments/scaffolded_sd054_onboarding.py` (NEW), env kwarg
  `reef_bipartite_agent_spawn_in_reef_half` (default False) on
  CausalGridWorldV2, master switch
  `use_scaffolded_sd054_onboarding_scheduler` (default False), 14
  phase-config knobs all default to memo-suggested values; three-phase
  P0/P1/P2 curriculum (frozen goal pipeline + reef-half spawn ->
  linear-anneal hazard / proximity_harm / drive-to-fire / z_beta-threshold
  -> target env frozen-policy measurement); 645/645 contracts + 17 new
  contracts PASS; bit-identical OFF guarantee verified. (2) **SD-056
  multi-step rollout stability amend** (ree-v3 d327b89) -- multi-step
  contrastive horizon h=5 + per-step output norm clamp ratio=2.0 prevents
  multi-step rollout drift; V3-EXQ-617 substrate-readiness PASS confirms
  amend stability + V3-EXQ-614b ARM_2 ALL_ON zero NaN/Inf across 162k
  steps. (3) **InfantCurriculumScheduler Phase 0->1 H_pos floor
  recalibration** -- closes
  behavioral_diversity_isolation:GAP-C prereq (3). (4) **MECH-341
  cluster autopsy** (V3-EXQ-614b + V3-EXQ-615 + V3-EXQ-616 convergent
  reading) -- ARC-065 supports (clean architectural-necessity
  discrimination; promotion-eligibility candidate -> provisional surfaced
  for next governance walk); MECH-341 non_contributory (claim correctly
  characterised as score-layer preserver of upstream-supplied candidate-
  pool diversity, not in-isolation diversity generator); Q-054 mixed
  (definitive negative answer to scale-lever framing). (5) **Multi-round
  flat-vs-runs propagation-failure mirroring** -- 29 mirrors across three
  rounds; 11 claims dropped from the active conflicts table (95 -> 84:
  ARC-045 / INV-010 / MECH-166 / MECH-261 / MECH-302 / MECH-320 / SD-017
  / SD-033a / Q-045 / MECH-313 / MECH-260; partial-clear on ARC-065
  ratio 0.231 -> 0.043). (6) **cloud-scaler GHA -> hub systemd migration
  prepared** (ree-v3 d641419) -- coordinator/deploy/cloud-scaler.{py,
  service,timer,md} adds hub-resident OnCalendar=\*:0/5 oneshot replacing
  the every-15-min GHA workflow whose actual gaps had grown to 60-273
  minutes under GHA load; HUB_NAME guard / HELD_BY_SELF veto / surge mode
  / HEARTBEAT_FRESH_MIN floor all preserved (deploy operator-driven; not
  deployed this session). (7) **Multiple smaller items** -- V3-EXQ-618
  SD-049 Phase 3 SD-032 consumer cascade substrate-readiness; V3-EXQ-620
  SD-037 axis (a) Phase 1 consumer-input distributions diagnostic;
  IGW-038..042 cohort proposal-truth-up; z_goal collapse triage; SD-037
  axis (a) consumer-input recalibration plan landed; failure-autopsy
  artifacts for V3-EXQ-614b single-target + MECH-341 cluster + V3-EXQ-615
  dual-manifest supersession. Bottleneck (updated framing): the **goal-
  pipeline / training-regime substrate enrichment** is now the load-
  bearing constraint behind the substrate-uniform monomodal-V_s
  monostrategy tail across 483c / 524a / 603 lineage / 540a-e / 590a /
  591 / 598 / 598b. The Layer-A E2 forward-model collapse fix (SD-056
  + multi-step amend) is substrate-readiness validated; the
  **scaffolded_sd054_onboarding** substrate is now landed and V3-EXQ-621a
  PASS provides the first substrate-readiness signal. The next-cycle
  bottleneck is the **V3-EXQ-622 staged goal-stream S0-S3** outcome
  watch -- decomposes the 621 z_goal failure into a four-stage curriculum
  to localise where the substrate-uniform monostrategy collapse re-emerges
  under the new scaffolded onboarding. The V3-EXQ-543k disposition gap
  (carried forward from 2026-05-21) remains outstanding.
- **2026-05-31T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **773 cumulative
  completions (+5 since 2026-05-30T01:10Z read)** (185 PASS / 311 FAIL
  / 84 ERROR / 193 UNKNOWN; deltas: PASS +2, FAIL +3, ERROR +0, UNKNOWN
  +0); last_updated 2026-05-30T18:47:00.431932Z -- ~6.5h fresh at this
  read. Today's three returns: **V3-EXQ-517c PASS 2026-05-30T12:45Z**
  on the SD-022 scheduled-injection curriculum (2/3 ARM_A seeds; 3/3
  ARM_B seeds zero events) -- cleared MECH-302 + MECH-303 v3_pending
  gates and lifted gate (c) for the MECH-304 / V3-EXQ-519 conditioned-
  inhibition experiment; **V3-EXQ-569c FAIL 2026-05-30T06:00Z** on the
  SD-056 + ARC-065 / MECH-341 matched-entropy FP-2 falsifier (C1
  borderline at 0.041-0.046 pairwise_dist vs 0.05 floor but C3 entropy
  lift 0.833-0.951 vs matched-noise 0.414 ~2.4x above) -- autopsy
  recommended per-claim direction upgrade to supports (manifest weakens
  superseded) and routed to V3-EXQ-569d floor-recalibrated falsifier +
  V3-EXQ-569e mechanism probe; **V3-EXQ-490i FAIL 2026-05-30T14:31Z**
  on MECH-295 GAP-4 Tier-1 (bridge sign-test PASSED 3/3 seeds with
  approach_commit_rate=1.0 on ARM_1 vs 0.0 on ARM_0; C3_lift_vs_baseline
  FAIL = metric-design contamination from non-zero z_goal_enabled in
  ARM_0 + SD-032b dACC consumer-pathway wiring gap) -- autopsy
  recommended MECH-295 narrow_supports and routed to V3-EXQ-490j
  severed-bridge baseline + separate /diagnose-errors on the dACC
  bundle -> E3 score_bias adapter. **Pending review queue
  (regenerated 2026-05-30T19:42Z) reads 1 item** -- V3-EXQ-490i FAIL,
  already autopsied per /failure-autopsy session 19:48Z and awaiting
  governance application of the recommended per-claim direction shift.
  **Currently queued (`experiment_queue.json`): 3 items** -- V3-EXQ-490j
  (MECH-295 successor, severed-bridge baseline with z_goal_enabled=False
  ARM_0 + direct bridge-magnitude probe replacing contaminated
  goal_norm_peak delta; priority 350), V3-EXQ-519b (SD-051 / MECH-304
  conditioned-safety-store readiness; priority 340), V3-EXQ-569d
  (ARC-065 / MECH-341 floor-recalibrated falsifier; priority 310).
  V3-EXQ-569e MECH-immediate Pathway-A-vs-B mechanism probe was queued
  17:13Z (priority 305) and runner-claimed earlier this evening so it
  no longer appears in `items[]` at this read. Substrate / governance
  landings since the 2026-05-30T01:10Z snapshot: (1) **SD-022
  scheduled-injection extension** (ree-v3 main) -- env-side curriculum
  that injects damage directly into `self.limb_damage` independent of
  agent action or hazard contact, supplying detectable damage->heal
  trajectories the MECH-302 comparator needs regardless of a trained
  avoidance policy; five new env-only kwargs (NOT surfaced through
  REEConfig.from_dims); 565/565 contracts + 7/7 preflight PASS;
  V3-EXQ-517c PASS validated. (2) **SD-037 consumer-cascade
  (MECH-281 motor-coupling axis amend)** -- four additional override-
  signal consumer sites wired (LateralPFCAnalog eff_eta scaling +
  BLAAnalog encoding_gain scaling + CeAAnalog mode_prior + fast_prime
  amplification + BetaGate urgency_interrupt threshold attenuation);
  all four scalar gains default 0.0 (bit-identical OFF); 556/556
  contracts + 13 new MECH-281 contracts PASS; V3-EXQ-483e queued for
  validation. (3) **MECH-302 + MECH-303 v3_pending cleared** (IGW-021
  at 17:16Z) on V3-EXQ-517c PASS; substrate_queue MECH-302 status
  implemented -> validated; gate (c) for MECH-304 V3-EXQ-519 lifted.
  (4) **Runner stack hardening** -- runner SIGTERM phantom-completion
  fix (`_transient_exit_codes` set extended to {137, -9, -11, -15,
  143} so cloud-scaler shutdowns intercept as infra-crash instead of
  silently writing phantom completion rows; ree-v3 main c8288f1),
  runner sentinel-detection fix (`RE_BARE_OUTCOME` regex added so
  scripts emitting bare `outcome: PASS/FAIL` lines are correctly
  classified; ree-v3 main 9c187d0), Phase 3 coordinator `/claim`
  endpoint claim_log INSERT landed (ree-v3 main 0128cdc; first
  claim_log row written since 2026-05-21), and heartbeat write-gate
  scope correction (gate now scopes to local file write only; the
  coordinator POST always fires; COORDINATOR_TIMEOUT raised 3 ->
  10s for WireGuard-payload safety; ree-v3 main d82af98). (5) **IGW
  housekeeping batch** -- 4 wrong-route proposals flipped to
  status=gated (EXP-0003 MECH-334 / EXP-0051 ARC-045 / EXP-0062 Q-045
  / EXP-0064 MECH-166); IGW auto-spawn loop root-caused at
  REE_assembly/scripts/generate_inter_governance_workset.py
  `_substrate_resolved` and patched (workset 53 -> 48 items after
  generator fix; status carry-forward extended to gated metadata);
  /inter-governance-brief regenerated workset (53 items; 20 ready;
  3 in_flight). Bottleneck (unchanged framing): the **upstream E2
  world-forward per-candidate z_world collapse** identified by the
  2026-05-25 V3-EXQ-571 root-cause investigation; the SD-056
  contrastive-next-state landing (2026-05-29) is the architectural
  fix and V3-EXQ-569b/c/d/e plus the **scaffolded_sd054_onboarding**
  substrate-queue entry are the load-bearing follow-ons. Today's two
  amends (SD-022 scheduled-injection + SD-037 consumer-cascade) closed
  separate substrate-ceiling pockets (MECH-302 + MECH-281) but neither
  removes the Layer-A (E2 forward-model collapse) cause flagged as
  the root structural blocker. The V3-EXQ-543k disposition gap
  (carried forward from 2026-05-21) remains outstanding.
- **2026-05-30T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **768 cumulative
  completions (+7 since 2026-05-29T01:10Z read)** (183 PASS / 308 FAIL
  / 84 ERROR / 193 UNKNOWN; deltas: PASS +5, FAIL +1, ERROR +1, UNKNOWN
  +0); last_updated 2026-05-29T20:17:10.848721Z -- ~5h fresh at this
  read. The Phase-3 coordinator -> central-index merge stayed caught up
  over the day. **Pending review queue (regenerated 2026-05-29T23:36Z;
  last review 2026-05-29T21:35Z) reads 2 items** -- V3-EXQ-483d FAIL
  (Tier-1 library rebuild successor to 483c; C1+C2+C3+C4 PASS but
  C3_lift_vs_baseline FAIL 1/2 arms cleared; manifest tags SD-037=weakens;
  flagged for /failure-autopsy per user-confirmed routing) and
  V3-EXQ-612b ERROR (Phase 3 cutover cloud-2 smoke, no-sentinel; queued
  for /diagnose-errors). **Currently queued (`experiment_queue.json`):
  1 item** -- V3-EXQ-592c (MECH-090 R-c commit-readiness gate validation
  on ree-cloud-3, claimed 2026-05-29T22:46Z; supersedes V3-EXQ-592b which
  silent-dropped on FAIL on DLAPTOP-4 2026-05-29T08:32Z and motivated the
  runner-side FAIL/ERROR-branch manifest-persistence fix). Substrate /
  governance landings since the 2026-05-29T01:10Z snapshot: (1)
  **SD-056 E2 action-conditional divergence preservation** (ree-v3 main
  041a974) -- contrastive next-state InfoNCE auxiliary on world_forward;
  resolves V3-EXQ-571 root-cause finding (per-candidate cand_world_pairwise_dist
  collapsed to 0.0); two new E2FastPredictor helpers (cand_world_pairwise_dist
  + world_forward_contrastive_loss); bit-identical OFF default; 539/539
  contracts + 7/7 preflight PASS. V3-EXQ-613 substrate-readiness PASS;
  V3-EXQ-569a behavioural-validation falsifier queued same day but crashed
  twice with NaN in torch.multinomial (self-anchored InfoNCE targets caused
  E2 weights to diverge); /diagnose-errors superseded 569a with V3-EXQ-569b
  using observation-anchored targets from a rolling buffer. (2) **MECH-090
  R-c continuation (nav_competence axis)** -- pass 2 of 2 for
  commitment_closure:GAP-4; CommitReadiness module landed; both R-c axes
  AND-compose at both BetaGate elevate sites; V3-EXQ-592b grid extended to
  4 arms for orthogonal-axis falsifier (ARM_0 baseline / ARM_2 GATED_NAV_COMP_ON
  / ARM_3 GATED_BOTH_ON / ARM_4 BOTH_GATES_OFF_HARNESS_FORCES_READY).
  (3) **MECH-341 parameter retune** (ree-v3 a45ca7f) -- e3_diversity_entropy_lambda
  0.05 -> 0.5 (10x); e3_diversity_entropy_bias_scale 0.1 -> 1.0 for headroom;
  triggered by V3-EXQ-611c PASS interpretation (C1=True stratified fires,
  C2=False lambda too small, C3=True diversity produced, R2c=True all ready);
  routed to V3-EXQ-614 behavioural successor per behavioral_diversity_isolation_plan.md.
  (4) **runner FAIL/ERROR manifest-persistence fix** (ree-v3 41c3411) --
  experiment_runner.py FAIL branch (2267-2305) and ERROR branch (2229-2265)
  now invoke the three-call sequence (_result_manifest_exists +
  git_push_results + coordinator_client.report_result) the PASS branch
  retrofitted on 2026-05-08; FAIL/ERROR with claimed-but-disk-missing
  manifest now WARN + release_active_claim + _pass_skip (leaves queue
  entry) instead of silently removing; 4 new source-inspection contract
  tests; 543/543 contracts PASS. (5) **/governance cycle 2026-05-29 evening**
  (REE_assembly master) -- 13 evidence_quality_note appends + 7
  pending_retest_after_substrate flags + 1 SD-033a narrow_supports_flag;
  substrate_queue net: 1 CREATE (scaffolded_sd054_onboarding) + 1 amend
  ARC-046 + 1 amend MECH-341 (no-op) + 1 amend scaffolded_sd054_onboarding
  cluster; 6 failure-autopsies applied (598/606/596-602/603-followon-604-605/
  490g-cohort/MECH-341-cluster). (6) **Phase 3 writer auto-recovery +
  coordinator.db committed_at backfill** (ree-v3 d3d3c7a) -- opt-in
  PHASE3_AUTO_RESET_ON_REBASE_CONFLICT self-heal on sync_daemon; one-shot
  backfill script resolved 8/8 NULL committed_at rows against manifests
  on origin/master. (7) **GAP-4 Tier-1 library rebuild + V3-EXQ-483d/490h
  queued** (ree-v3 3eb2601) -- experiments/_lib/goal_pipeline_tier1.py:
  cfg.use_dacc=True now unconditional; C3_lift_vs_baseline default metric
  switched from approach_commit_rate (saturated 1.0 in OFF_OFF) to
  goal_norm_peak delta vs baseline (cross-claim-comparable). Bottleneck:
  the **goal-pipeline / training-regime substrate enrichment** identified
  by the 2026-05-29 V3-EXQ-490g cohort autopsy (Cluster B disposition;
  583-uniform monomodal-V_s monostrategy tail signature across 483c / 524a
  / 603 / 603a / 603b / 603c / 604 / 605 / 540a-e / 590a / 591 / 598 /
  598b) is the load-bearing constraint behind the diversity-cluster
  non_contributory chain; the new substrate_queue entry
  **scaffolded_sd054_onboarding** (priority 1, unblocks 9 claims:
  Q-045/MECH-313/MECH-260/MECH-295/MECH-307/MECH-117/SD-049-Phase-2/ARC-030/Q-040;
  design_doc evidence/planning/sd_054_scaffolded_onboarding_substrate_design.md)
  is the next substrate-implementation session-of-record. The V3-EXQ-543k
  disposition gap (carried forward from 2026-05-21) remains outstanding.
- **2026-05-29T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **761 cumulative
  completions (+5 since 2026-05-28T01:10Z read)** (178 PASS / 307 FAIL
  / 83 ERROR / 193 UNKNOWN; deltas: PASS +0, FAIL +3, ERROR +2, UNKNOWN
  +0); last_updated 2026-05-28T17:26:40.076023Z -- ~7h45m fresh at this
  read. The Phase-2 coordinator -> central-index merge remains caught up
  from yesterday's improvement. **Pending review queue (regenerated
  2026-05-27T17:40:25Z; last review 2026-05-27T17:35Z) reads 1 item
  unchanged from yesterday** -- V3-EXQ-598b (commitment_closure GAP-1
  SD-033a bias-head trainable ablation, claim_ids=[MECH-262, SD-033a]),
  FAIL completed 2026-05-27T12:03Z, evidence_direction=does_not_support.
  No fresh failure-autopsy yet against 598b; queued for the next
  governance pending walk. **Currently queued (`experiment_queue.json`):
  3 items, all Phase-3 cutover smoke** -- V3-EXQ-612 (DLAPTOP-4.local
  claimed 2026-05-28T17:24Z), V3-EXQ-612c (ree-cloud-2 pending,
  supersedes 612b which lacked the `verdict: PASS` stdout sentinel),
  V3-EXQ-612d (ree-cloud-3 pending, supersedes 612c after emit_outcome
  wiring fix). The MECH-090 R-c readiness conjunction validation
  V3-EXQ-592b and the MECH-341 retune validation V3-EXQ-611b that the
  2026-05-28 substrate-landing sessions queued do NOT appear in
  `items[]` at this read -- both were runner-claimed earlier in the day
  (611b on DLAPTOP-4.local @17:26:40Z per the MECH-341 retune session
  close note; 592b similarly claimed for execution by the active
  MECH-090 R-c session). Substrate landings since the 2026-05-28T01:10Z
  snapshot: (1) **MECH-090 R-c commit-entry readiness conjunction**
  (per ree-v3 CLAUDE.md MECH-090 section, landed 2026-05-28 in the
  active `implement-substrate-mech090-rc-conjunction` session) --
  BetaGate.should_admit_elevation gate at the two beta_gate.elevate()
  call sites in REEAgent.select_action; reading R-c single-gate
  conjunction strongest per the 28-entry MECH-090 lit synthesis (commit
  9e68c5ca8a) with R-b Tandetnik 2021 retained as fallback; bit-
  identical OFF default; V3-EXQ-592b 2-arm GATED / GATED_FORCED_READY
  validation queued. (2) **MECH-341 retune** (ree-v3 e02e77f) --
  stratified_select call-site expanded from committed-only to BOTH
  committed and uncommitted branches in `ree_core/predictors/e3_selector.py`;
  resolves V3-EXQ-611 ARM_2 n_stratified_fired=0 zero-fires failure
  (committed branch was never entered during the validation episodes);
  V3-EXQ-611b 6-arm factorial parameter sweep queued (3 option groups x
  2 entropy_bias_scale values 1.0/2.0). (3) **coord-env runner-start
  fix** (REE_assembly fc08812b62 + ree-v3 9fc0e02) -- serve.py
  start_runner() default-injects shadow env (COORDINATION_MODE +
  COORDINATOR_URL + COORDINATOR_TOKEN) from coordinator.env when
  extra_env is None and env file is configured; runner_remote_control.py
  write_heartbeat surfaces coordination_mode field auto-read from
  os.environ for cross-machine status visibility without SSH audit;
  /queue-experiment skill (both .claude/ and .agents/ mirrors) +
  cloud_workers.md gain a verification step warning if any active
  runner is in git mode or missing the field. (4) **E2 action-conditional
  divergence substrate-design memo** (REE_assembly 7cb1200332) -- lever
  B contrastive next-state (InfoNCE-style auxiliary on E2.world_forward
  with K-1 in-batch negatives drawn from sibling CEM candidates with
  different first-actions) chosen over PLSM (lever A) and SWIRL (lever
  C) per the 2026-05-28 lit-pull SYNTHESIS verdict (REE_assembly
  04bc1f3727; 6 entries balanced across ML world-model + biology
  forward-model literatures, lit_conf 0.78); decision deferred to a
  separate /implement-substrate session. (5) **E2 action-conditional
  divergence lit-pull** (REE_assembly 04bc1f3727) -- 6-entry SYNTHESIS
  on the V3-EXQ-571 root-cause finding (E2 world-forward per-candidate
  signal collapse, `cand_world_pairwise_dist=0.0`); verdict: option (ii)
  fix E2 is the architecturally faithful target; option (i) GAP-B
  one-hot bypass is a tactical alternative; methodological gap surfaced
  (no published paper reports per-action pairwise distance between
  predicted latents as headline metric -- REE could publish the
  `cand_world_pairwise_dist` diagnostic as a standalone contribution).
  (6) **IGW housekeeping batch** -- IGW-008 GAP-A plan resync
  (behavioral_diversity_isolation_plan.md row 1 status partial ->
  blocked_pending_substrate); IGW-010 GAP-C plan-doc refresh + workset
  regen (row 3 in_progress -> blocked_pending_substrate per V3-EXQ-603c
  FAIL + V3-EXQ-611 FAIL cluster-absorbed into V3-EXQ-591 autopsy);
  IGW-011 GAP-D doc-sync + R4.b flag (row 4 in_progress ->
  pending_governance_stamp). Bottleneck: the **upstream E2 world-forward
  per-candidate z_world collapse** identified in the 2026-05-25
  V3-EXQ-571 root-cause investigation remains the structural root cause
  of the score_bias-chain flatness; the E2 action-divergence design memo
  (2026-05-28) makes lever B (contrastive next-state via InfoNCE) the
  plan-of-record fix; landing it is the next /implement-substrate
  session-of-record after V3-EXQ-611b and V3-EXQ-592b return validation
  signal. **MECH-341 substrate retune** and **MECH-090 R-c readiness
  conjunction** today both address Layer-B (post-CEM scoring) and the
  commit-entry predicate respectively, but neither removes the Layer-A
  (E2 forward-model collapse) cause flagged as the root structural
  blocker by the V3-EXQ-571 investigation. The V3-EXQ-543k disposition
  gap (carried forward from 2026-05-21) remains outstanding.
- **2026-05-28T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **756 cumulative
  completions (+3 since 2026-05-27T01:10Z read)** (178 PASS / 304 FAIL
  / 81 ERROR / 193 UNKNOWN); last_updated 2026-05-27T18:05:04Z -- the
  central file is now ~7h fresh at this read (a sharp improvement over
  the prior ~76h stall; the Phase-2 coordinator -> central-index merge
  has caught up after the 2026-05-27 governance cycle staging). +3
  central completions resolve to V3-EXQ-543j / 588b / 524a / 543k /
  490g (the cluster the governance pending-walk dispositioned this
  morning is now reflected). Cross-machine per-host aggregate at this
  read: DLAPTOP-4.local 595 (+3 since yesterday; latest V3-EXQ-611
  2026-05-27T13:02Z) + ree-cloud-1 243 (stale) + ree-cloud-2 184
  (stale) + ree-cloud-3 142 (latest V3-EXQ-609 2026-05-26T07:24Z,
  unchanged) + ree-cloud-4 141 (stale) + ree-worker-3 133 (stale) +
  EWIN-PC 77 (stale) + Daniel-PC 28 (stale) = **1543 cumulative across
  hosts** (+3 since yesterday's 1540). The fleet contraction trend
  noted yesterday continues: only DLAPTOP-4.local wrote within the last
  24h; ree-cloud-3 is at ~42h stale and the other six hosts at days /
  weeks. **Pending review queue (regenerated 2026-05-27T17:40:25Z;
  last review 2026-05-27T17:35Z) reads 1 item** -- V3-EXQ-598b
  (commitment_closure GAP-1 SD-033a bias-head trainable ablation,
  claim_ids=[MECH-262, SD-033a]), FAIL completed 2026-05-27T12:03Z,
  evidence_direction=does_not_support; the runner finished the
  ARM_0 frozen / ARM_1 trainable comparison and the permissive gate
  (manifest exists + outcome in {PASS, FAIL}) holds, so the disposition
  routes through the next governance pending walk rather than via
  `/failure-autopsy`. **Currently queued
  (`experiment_queue.json`): 1 item** -- V3-EXQ-610
  `v3_exq_610_inv074_crystallization_necessity.py` (INV-074, priority
  28, machine_affinity=any, estimated_minutes=180); this is the
  IGW-20260527-027 INV-074 retest-after-substrate (MECH-341 flags
  active per the 2026-05-27 IGW-027 close) -- V3-EXQ-611 (MECH-341
  4-arm substrate-readiness) does NOT appear in `items[]` at this
  read despite being queued at priority 260 in the MECH-341 session
  close note; worth a manual re-check whether 611 was drained or
  was never persisted, since priority 260 should run before priority
  28. Substrate landings since the 2026-05-27T01:10Z snapshot
  (single major substrate-side claim landed today):
  (1) **MECH-341 e3_scoring_preserves_trajectory_class_diversity**
  (ree-v3 547faa3) -- Layer-B post-CEM diversity-preservation
  substrate; togglable entropy-bonus + stratified-select sub-flavours;
  pure-arithmetic regulator sibling to MECH-313/314/320; bit-identical
  OFF default; 506/506 contracts + 7/7 preflight PASS; design doc
  REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
  + claims.yaml implementation_note + behavioral_diversity_isolation_plan.md
  status-table update + V3-EXQ-611 4-arm validation queued. (2)
  **Governance cycle 2026-05-27** (REE_assembly 4856a3dcdb +
  correction ac56ba507b) -- 6-pending walk: V3-EXQ-543l per-claim
  4-split governance-stamped (ARC-062 weakens narrow_supports_flag /
  MECH-309 supports first trained-policy entry / INV-074 + MECH-334
  non_contributory missing-prerequisite); V3-EXQ-591 manifest
  does_not_support -> non_contributory cluster-uniform 4th member
  (ARC-046 NOT weakened); V3-EXQ-603c cluster-absorbed into 591
  autopsy; Q-045 / MECH-313 / MECH-260 routed substrate_ceiling V3
  (NOT substrate_conditional V4 -- user-flagged correction). substrate_queue
  extended with ARC-046 (InfantCurriculumScheduler Phase-0 exit-gate
  fix per V3-EXQ-591 autopsy) + MECH-341 (entropy_bonus_scale retune
  + stratified trigger condition revision) entries. (3) Closure-drift
  /governance step + lint script (REE_assembly 01e5f79e7d) + brain-map
  prefix mapping + 4 new regions + validator tightening (REE_assembly
  4039a0dbaa, governance.sh Step 3d). (4) Failure-autopsy artifacts
  V3-EXQ-543l (REE_assembly 72bab05c93) + V3-EXQ-591 (REE_assembly
  cfedfd1353).
  Bottleneck: the **upstream E2 world-forward per-candidate z_world
  collapse** identified in the 2026-05-25 V3-EXQ-571 root-cause
  investigation remains the structural root cause of the score_bias-chain
  flatness; the MECH-341 substrate landing today addresses the
  Layer-B (E3 scoring) symptom but does not remove the Layer-A
  (E2 forward-model collapse) cause. The 2026-05-27 governance cycle's
  reclassification of Q-045 / MECH-313 / MECH-260 to substrate_ceiling
  V3 (vs the initial substrate_conditional V4 stamping the user
  corrected mid-cycle) is the most consequential governance act of
  the cycle: it keeps the diversity cluster as a V3-scoped substrate-
  enrichment problem rather than punting it to V4. **ARC-062 / MECH-309**
  picked up its first contributory trained-policy entry today
  (V3-EXQ-543l per-claim split: MECH-309 supports; ARC-062 weakens
  narrow_supports_flag), and V3-EXQ-598b's FAIL pending review will
  resolve whether the next-action substrate-enrichment pass is the
  ARC-046 InfantCurriculumScheduler Phase-0 exit gate or the
  MECH-341 entropy retune. The V3-EXQ-543k disposition gap (carried
  forward from the 2026-05-21 drained-without-manifest BLOCK) remains
  outstanding; not closed by either failure-autopsy today.
- **2026-05-27T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **753 cumulative
  completions UNCHANGED** (176 PASS / 303 FAIL / 81 ERROR / 193 UNKNOWN);
  last_updated 2026-05-23T21:06:24Z -- the central file has now been
  static for **~76h** (~3.2 days; the Phase-2 coordinator -> central-index
  merge is wedged again, longest stall of the cycle). Per-machine
  `runner_status/<hostname>.json` files carry the live writes;
  cross-machine aggregate at this read: DLAPTOP-4.local 592 +
  ree-cloud-1 243 + ree-cloud-2 184 + ree-cloud-3 142 +
  ree-cloud-4 141 + ree-worker-3 133 + EWIN-PC 77 + Daniel-PC 28 =
  **1540 cumulative across hosts** (+0 reported by central). DLAPTOP-4.local
  is the only host with a fresh heartbeat (2026-05-27T01:11Z); ree-cloud-3
  last wrote 2026-05-26T07:24Z (~18h ago); the other 6 hosts last wrote
  between 2026-04-10 (Daniel-PC) and 2026-05-21 (ree-cloud-1/2/4) --
  effectively only Mac + cloud-3 are still in active production.
  **Pending review queue (pending_review.md regenerated 2026-05-25T09:06Z;
  last review 2026-05-25T08:56Z) reads 0 items** but is now **stale** --
  ~5 new V3 manifests have landed in the intervening 40h (V3-EXQ-543l
  PASS 2026-05-26T02:30Z evidence_direction=mixed; V3-EXQ-608 PASS
  2026-05-26T02:58Z interpretation_label=R2a_e3_collapse_confirmed_large_gap
  on all 3 seeds; V3-EXQ-603/603a/603b retest manifests; V3-EXQ-609
  diagnostic 2026-05-26T07:24Z claim_ids=[]) that the active
  `governance-20260526T230808Z` cycle (started 2026-05-26T23:08Z, still
  listed `status: active` in TASK_CLAIMS.json at this read) will walk.
  **Currently queued (`experiment_queue.json`): 0 items (`items: []`).**
  The entire 6-entry fix-and-retest cohort queued 2026-05-25 (V3-EXQ-603a,
  V3-EXQ-543l, V3-EXQ-608, V3-EXQ-588b, V3-EXQ-598b, V3-EXQ-483d) has
  drained; V3-EXQ-543l + 608 wrote manifests; the others' fates will be
  resolved by the active governance walk. V3-EXQ-590a / 591 (the
  EXQ-ISEF-004/005 cohort that has been the load-bearing bottleneck for
  the prior three snapshots) **also no longer appear in the queue** --
  the explicit-rescue checkpoint-resumable V3-EXQ-590a single-host pin
  appears to have been pruned without a fresh manifest at this read,
  warranting a separate diagnostic pass.
  Substrate edits in ree-v3 since the 2026-05-25 snapshot (no new
  SD/MECH/ARC/Q claim landings; instrumentation + dead-branch cleanup
  only): (1) **MECH-111 dead-branch deletion** (ree-v3 099743e) in
  `ree_core/predictors/e3_selector.py:606-613` -- pure-cleanup of the
  uniform-scalar-shift broadcast novelty branch confirmed argmin-invariant
  by the 2026-05-25 MECH-314a propagation root-cause investigation
  (driver verified bit-identical pre/post behaviour); (2) **EXQ-571
  decomp per-channel `std_across_K` + `bias_range_mean` instrumentation
  keys** (ree-v3 5c84a4d) on `ree_core/agent.py` -- additive instrumentation
  surfacing per-candidate spread that the EXQ-571 mean-collapse metric
  obscured (read by V3-EXQ-609); the same 2026-05-25 investigation
  identified the **upstream E2 world-forward per-candidate z_world collapse**
  (`cand_world_pairwise_dist=0.0000` across K=32) as the structural
  blocker on MECH-314a / MECH-320 / MECH-295 / SD-033a / SD-033b
  bias-channel diversity, captured in
  `REE_assembly/evidence/planning/v3_exq_571_root_cause_2026-05-25.md`.
  (3) **Governance dispositions landed against the V3-EXQ-571 finding**:
  ARC-065 + MECH-314 `failure_record[0].metric` corrected in
  `substrate_queue.json` (REE_assembly 683c252158); Q-044 `blocked_by:
  [e2_world_forward_first_action_preservation]` + dated blocker
  appendix (REE_assembly 2eb5252f3f); MECH-314a Phase-2 novelty-source
  design doc + design_question entry in substrate_queue (REE_assembly
  5ec31e39c8 + 039e195637).
  Bottleneck: the **upstream E2 world-forward per-candidate z_world
  collapse** is now the structural root cause of the score_bias-chain
  flatness that has held ARC-065 (MECH-314a/b/c) + ARC-066 (MECH-320)
  + ARC-062 GAP-B (gated_policy heads) + MECH-295 / SD-033a/b at
  non_contributory across the diversity cluster; the V3-EXQ-571 finding
  doc enumerates four forward paths (rolling z_world visitation buffer
  Option A as recommended Phase-1 caveat, first-action one-hot bypass
  per existing GAP-B fix, candidate-pool relative rank, hybrid harm +
  visitation -- with action-object-identity at proposer stage surfaced
  as Option F during evaluation). **ARC-062 / MECH-309** now has its
  first contributory-direction manifest in the cluster: **V3-EXQ-543l**
  PASS 2026-05-26T02:30Z with `evidence_direction=mixed` and
  `claim_ids=[ARC-062, MECH-309, INV-074, MECH-334]` (governance
  disposition deferred to the in-flight cycle). **V3-EXQ-608 P2 PASS
  2026-05-26T02:58Z** confirmed the R2.a "e3_collapse_confirmed_large_gap"
  interpretation across all 3 seeds (frac_pre_ge2=1.0,
  frac_e3_collapse_above_eps=0.858, mean_top2_class_gap=0.378) and
  triggered the MECH-341 Phase-3 substrate-design phase per the
  `behavioral_diversity_isolation_plan.md` decision rules. **V3-EXQ-543k
  disposition recording** (the 2026-05-21 drained-without-manifest BLOCK
  flagged by the governance verification gate) **remains outstanding**
  -- 543l does not formally close 543k's audit record. Historical
  context preserved below.
- **2026-05-25T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **753 cumulative
  completions UNCHANGED** (176 PASS / 303 FAIL / 81 ERROR / 193 UNKNOWN);
  last_updated 2026-05-23T21:06:24Z -- **no change from yesterday's
  central index** (the file has been static for ~52h at this read;
  the Phase-2 coordinator -> central-index merge that caught up
  briefly yesterday has stalled again; per-machine
  `runner_status/<hostname>.json` files carry the live writes).
  **Pending review queue (regenerated 2026-05-24T19:20:42Z; last review
  2026-05-24T18:16:00Z) reads 0 items** -- the 6 items from yesterday's
  snapshot (4 FAIL: V3-EXQ-483c / V3-EXQ-597b / V3-EXQ-603 x2; 2
  ERROR: V3-EXQ-606a / V3-EXQ-598) have all been governance-cleared
  via the day's failure-autopsy / diagnose-errors sessions. **Currently
  queued (`experiment_queue.json`): 2 items unchanged from yesterday**
  -- V3-EXQ-590a (EXQ-ISEF-004 novelty-bonus Goldilocks calibration,
  MECH-314, checkpoint-resumable, pinned ree-cloud-3, priority 100;
  partial 1/15 runs saved; unclaimed at this read) + V3-EXQ-591
  (EXQ-ISEF-005 4-phase infant curriculum vs flat baselines, ARC-046,
  claim record DLAPTOP-4.local 2026-05-23T21:06:24Z unchanged for ~28h
  -- worth verifying the claim has not gone stale).
  2026-05-24 governance / queue-staging wave (no SD/MECH landings;
  scientific-disposition + queue-staging only): (1) **All 6 pending
  items closed as non_contributory**: V3-EXQ-603 interactive gate
  confirmed all three claims (MECH-260/MECH-313/Q-045) NC (MECH-260
  "weakens" is a call-path measurement artefact -- act_with_split_obs
  bypasses select_action() where dacc.record_action() fires; FIFO
  permanently empty; ARM_2==ARM_0 to 6 d.p.); V3-EXQ-483c SD-037 GAP-4
  tier-1 NC (use_dacc=True omitted from all 4 arm configs; agent.dacc
  is None; _dacc_bias_norm returns 0.0); V3-EXQ-597b MECH-258 NC
  (dacc_suppression_weight=4.0 dominated pre-clip bias, diluting PE
  signal; post-clip constant at 2.0 eliminated behavioural contrast);
  V3-EXQ-606a + V3-EXQ-598 ERROR root-caused to a git-sync gap on
  ree-cloud-2 (no code bugs). (2) **Six fix-and-retest queue-experiment
  sessions launched** (still active at the time of this read): V3-EXQ-603a
  (Q-045/MECH-313/MECH-260 call-path fix -- select_action+obs_harm_a+
  affective stream + FIFO warmup 75); V3-EXQ-483d (SD-037 broadcast
  GAP-4 with PAG/override_signal C2 + goal_norm_peak C3); V3-EXQ-543l
  (ARC-062 GAP-B mode-separation-floor successor with floor 0.25 -> 0.5
  and P1_W_DEVIATION_AUX_WEIGHT 0.1 -> 0.3); V3-EXQ-608 (MECH-319 GAP-K
  simulation-mode rule-write-gate falsifier with explicit dream replay
  loop); V3-EXQ-588b (goal-seeding pipeline diagnostic post-588 autopsy);
  V3-EXQ-598b (commitment_closure GAP-1 SD-033a bias-head trainable
  ablation gated on V3-EXQ-543l). V3-EXQ-606b held off the queue per
  user (no V3-EXQ-543k contributory successor exists). (3) **Closure-map
  audit + plan-frontmatter sync** -- serve.py CLOSURE_KNOWN_PLANS +2
  (arc_062_rule_apprehension, infant_substrate); plan frontmatter synced
  across arc_062 GAP-B/C/D/H/K, commitment_closure GAP-1, sleep_substrate
  GAP-2, goal_pipeline GAP-4.
  No new SD / MECH / ARC / Q landings since the 2026-05-22T01:10Z
  snapshot -- the substrate state is unchanged for the third nightly
  read in a row.
  Bottleneck: **EXQ-ISEF-004/005 (V3-EXQ-590a + 591)** remain the
  load-bearing developmental warm-start gate for the ARC-065 diversity
  narrative and the deferred Q-043/044/045 + INV-049 retests, with
  V3-EXQ-591's day-old unchanged claim record worth a manual check.
  **ARC-062 / MECH-309** stays substrate_ceiling-framed; V3-EXQ-543l
  (the new force-escalated GAP-B falsifier successor) is the active
  test. The previous-snapshot V3-EXQ-543k drained-without-manifest BLOCK
  is now superseded by the 543l queue-staging (the governance verification
  gate will still flag 543k as a historical incident until its disposition
  is formally recorded). Historical context preserved below.
- **2026-05-24T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **753 cumulative
  completions** (176 PASS / 303 FAIL / 81 ERROR / 193 UNKNOWN);
  last_updated 2026-05-23T21:06:24Z -- the central file is now ~4h
  fresh at this read (a sharp improvement over yesterday's ~35h stale;
  the central-index merge is catching up post Phase-2 coordinator
  cutover). +4 cumulative completions since the 2026-05-23T01:10Z
  nightly read (+0 PASS / +2 FAIL / +2 ERROR / +0 UNKNOWN). **Pending
  review queue (regenerated 2026-05-23T22:03:10Z; last review
  2026-05-23T21:57:44Z) reads 6 items** -- 0 PASS, 4 FAIL (V3-EXQ-483c
  SD-037 broadcast-override GAP-4 tier-1, V3-EXQ-597b MECH-258 PE-vs-raw
  post-SP-CEM, V3-EXQ-603 x2 Q-045 MECH-313/MECH-260 collapse falsifier),
  2 runner-only ERROR (V3-EXQ-606a ARC-064 GAP-I, V3-EXQ-598 SD-033a
  bias-head ablation). The 5 substrate PASSes from yesterday's snapshot
  (V3-EXQ-601 x2 MECH-269b-followup-A staleness gate, V3-EXQ-599a
  MECH-286, V3-EXQ-600a MECH-282, V3-EXQ-607 MECH-340) have been moved
  into `reviewed_run_ids` -- the 2026-05-21 substrate-validation
  diagnostic battery is now fully governance-cleared. **Currently
  queued (`experiment_queue.json`): 2 items unchanged from yesterday**
  -- V3-EXQ-590a (EXQ-ISEF-004 novelty-bonus Goldilocks calibration,
  MECH-314, checkpoint-resumable, pinned ree-cloud-3, priority 100;
  partial 1/15 runs saved; unclaimed at this read) + V3-EXQ-591
  (EXQ-ISEF-005 4-phase infant curriculum vs flat baselines, ARC-046,
  claimed DLAPTOP-4.local 2026-05-23T21:06Z).
  2026-05-23 governance / infra wave (no SD/MECH landings; tooling
  only): (1) **Governance verification gate** (REE_assembly
  `scripts/verify_governance_cycle.py` + `generate_governance_handoff.py`
  + `evidence/verification/` + `evidence/handoffs/` + docs landed
  2026-05-23T20:38Z, master a686bfbc66) -- additive safety layer with
  7 checks, JSON report + Markdown handoff. Current gate result: FAIL
  with 1 BLOCK (V3-EXQ-543k drained without manifest) and 2 WARN (the
  now-resolved 54h central-runner_status staleness; ree-cloud-1
  heartbeat divergence). (2) **Shadow coordinator stale-machine TTL
  fix** in serve.py (2026-05-23T21:15Z, master 418c79ca4f) -- 6h TTL
  filter on `read_machines()` + `read_shadow_status()` so stale
  machines are excluded from the dashboards. (3) **Failure-autopsy
  sessions opened** 2026-05-23T22:14-22:36Z for V3-EXQ-597b
  (MECH-258 measurement gap), V3-EXQ-483c (SD-037 / MECH-280 /
  MECH-281 dACC measurement gap), V3-EXQ-603 (Q-045 / MECH-313 /
  MECH-260 measurement gap), and `/diagnose-errors` for V3-EXQ-606a +
  V3-EXQ-598 (re-queue as 606b / 598a). The night's pending_review is
  being actively triaged. No new SD / MECH / ARC / Q landings since
  the 2026-05-22T01:10Z snapshot -- the substrate state is unchanged.
  Bottleneck: **EXQ-ISEF-004/005 (V3-EXQ-590a + 591)** remain the
  load-bearing developmental warm-start gate for the ARC-065 diversity
  narrative and the deferred Q-043/044/045 + INV-049 retests;
  **ARC-062 / MECH-309** stays substrate_ceiling-framed with the
  V3-EXQ-543k drained-without-manifest BLOCK flagged by the new
  governance verification gate. Historical context preserved below.
- **2026-05-23T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **749 cumulative
  completions** (176 PASS / 301 FAIL / 79 ERROR / 193 UNKNOWN);
  last_updated 2026-05-21T14:26:59Z -- the central file is now ~35h
  stale (Phase-2 coordinator cutover; per-machine `runner_status/<hostname>.json`
  files carry the live writes). **No change in the central totals since
  the 2026-05-22T01:10Z nightly read** -- the runs that completed in the
  intervening 24h appear in `pending_review.md` and on per-machine status
  files but have not yet been merged into the central index. **Pending
  review queue (regenerated 2026-05-22T05:31:18Z; last review
  2026-05-22T05:30:55Z) reads 10 items** -- 5 PASS (V3-EXQ-601 x2
  MECH-269b-followup-A staleness gate, V3-EXQ-599a MECH-286 sleep-onset
  gate, V3-EXQ-600a MECH-282 LPB interoceptive routing, V3-EXQ-607
  MECH-340 persistence/efficacy gate -- the 2026-05-21 substrate
  diagnostics PASSing in turn), 3 FAIL (V3-EXQ-597b MECH-258 PE-vs-raw
  post-SP-CEM, V3-EXQ-603 x2 Q-045 MECH-313/MECH-260 collapse falsifier),
  2 runner-only ERROR (V3-EXQ-606a ARC-064 GAP-I, V3-EXQ-598 SD-033a
  bias-head ablation). The active governance cycle `governance-20260522T032251Z`
  has not yet walked these. **Currently queued (`experiment_queue.json`):
  2 items** -- V3-EXQ-590a (EXQ-ISEF-004 novelty-bonus Goldilocks
  calibration, MECH-314, checkpoint-resumable, pinned ree-cloud-3,
  priority 100; partial 1/15 runs saved) + V3-EXQ-591 (EXQ-ISEF-005
  4-phase infant curriculum vs flat baselines, ARC-046, claimed
  DLAPTOP-4.local). V3-EXQ-543k drained out of the queue this window
  -- the 2026-05-21T14:13Z force_rerun re-queue is no longer present
  but no fresh 543k manifest appears in `pending_review.md` either
  (its remote fate is opaque from the central indices pending the
  next governance walk).
  2026-05-19 -> 2026-05-22 substrate / governance wave: (1) **MECH-282
  LPB interoceptive routing**, **MECH-286 override-gated sleep onset**,
  and **MECH-340 persistence/efficacy gate** (+ Q-053 agent-side
  control-efficacy / goal-unattainability appraisal wiring) landed
  2026-05-21, all bit-identical OFF; validation V3-EXQ-599 / 600 / 607
  queued. (2) **ARC-062 GAP-B mode-separation floor** landed 2026-05-20;
  V3-EXQ-543k re-queued with force_rerun after a ree-cloud-4 FAIL with
  no central manifest (failure autopsy: 543i manifest mis-filed under
  the 543k slot). (3) **Q-043/044/045 EXQs 603/604/605** ran off-queue
  (604/605 FAIL); **V3-EXQ-597b** MECH-258 C2-telemetry revalidation and
  **V3-EXQ-598/606a** GAP-B-gated experiments queued. (4) **Coordinator
  Phase-2 cutover** 2026-05-21 -- hub + Mac + ree-cloud-1..4 flipped to
  `COORDINATION_MODE=coordinator`; **Phase-3 cutover substrate**
  (PHASE3_CUTOVER.md + preflight/verify + sync_daemon scaffold) designed
  the same day; fleet pause + runner suspend/resume landed. (5)
  Governance walks 2026-05-21 cleared 13 pending (543j/543i/595/597/598/
  604/605 non_contributory; 599/600 ERROR ack); pending_review driven to
  0 twice. Bottleneck: **EXQ-ISEF-004/005 (V3-EXQ-590a + 591)** remain
  the load-bearing developmental warm-start gate for the ARC-065
  diversity narrative and the deferred Q-043/044/045 + INV-049 retests;
  **ARC-062 / MECH-309** stays substrate_ceiling-framed with the
  GAP-B mode-separation-floor falsifier V3-EXQ-543k still in flight.
  Historical context preserved below.
- **2026-05-19T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **723 cumulative
  completions** (166 PASS / 287 FAIL / 77 ERROR / 193 UNKNOWN);
  last_updated 2026-05-18T15:44:20Z -- the central file is ~9.5h stale
  at this read (the multi-machine runners write per-machine
  `runner_status/<hostname>.json`). +3 central completions since the
  2026-05-18T01:10Z nightly read (+2 PASS / +1 FAIL). **Pending review
  queue (regenerated 2026-05-18T16:32:44Z; last review
  2026-05-18T16:30:27Z) is 1 item** -- the lone pending is
  `v3_exq_543i_arc062_differential_heads_falsifier_20260518T063711Z_v3`
  (ARC-062 / INV-074 / MECH-309 / MECH-334), FLAGGED for /failure-autopsy
  by the 2026-05-18T16:30Z governance cycle (diff_on_escape=true substrate
  fix works, diff_off non-reproduction) and deliberately left pending
  pending the cross-machine confirmation. **Currently queued
  (`experiment_queue.json`): 6 items** -- V3-EXQ-590 (ISEF-004 novelty
  Goldilocks weight sweep, MECH-314, claimed), 591 (ISEF-005
  curriculum-vs-flat baselines, ARC-046), 481b (MECH-090 V_s
  commit-release re-issue), 582a (goal_pipeline:GAP-3 Option 2
  drive_floor sweep, supersedes 582), 592 (commitment_closure:GAP-11
  committed-mode curriculum pilot, MECH-090), **543j** (ARC-062
  differential-heads byte-identical cross-machine confirmation of 543i,
  pinned ree-cloud-4). 2026-05-18 -> 2026-05-19 substrate / governance
  wave: (1) **ARC-062 GatedPolicy differential-heads robustness fix
  landed** 2026-05-18 (ree-v3) -- norm-pinned base+/-delta_hat
  reparameterization making head_0==head_1 collapse a non-equilibrium;
  validation **V3-EXQ-543i (supersedes 543g+543h) FAILed branch e**
  (MECH-309 supports / ARC-062 weakens; all 4 diff-ON gated arms 3/3
  inert) on a SINGLE machine (Mac). (2) **2026-05-18T16:30Z governance
  cycle** -- confirmed failure_autopsy_V3-EXQ-543h applied to the
  ARC-062 / MECH-309 crystallization-falsifier cluster (543f x4 / 543g /
  543h x2 set evidence_direction=superseded by 543i,
  epistemic_category=substrate_ceiling, pending_retest_after_substrate);
  MECH-332 / MECH-334 hold_pending_v3_substrate; claims.json rebuilt
  (645). (3) **retrieval-cue reframe** (interpretive only --
  ARC-060 / MECH-292 / MECH-293 / SD-039 recast as content-addressed
  cued retrieval; status unchanged) + **ARC-078** (unresolved-goal bank
  = content-addressed cue-addressed retrieval system, parent) +
  **MECH-339** (composite retrieval cue + outshining gate) registered
  2026-05-19; the C3 abandon mechanism deliberately NOT registered --
  gated behind a goal-disengagement biology-before lit-pull (active).
  claims.yaml 648 -> **650 claims**. (4) **GAP-L socially-scaffolded
  rule-population lit-pull discharged** -- ARC-077 / MECH-337 / MECH-338
  lit_conf parallel signal (NOT blended into exp_conf; all remain
  candidate; the caregiver/teacher-agent substrate hard gate stays open).
  (5) **infra:** heartbeat-autostash governance-regen recovery (639-file
  regen restored from dangling commit), Phase 0-1 shadow
  experiment-coordinator built (`ree-v3/coordinator/`, shadow-only, no
  cutover, git stays authoritative), all runners gracefully drained +
  cloud-2 force-stopped for a multi-machine coordination update.
  Bottleneck (shift): the **ARC-062 / MECH-309 rule-apprehension
  cross-machine bistability** is now the load-bearing gate --
  **V3-EXQ-543j** byte-identical cross-machine confirmation of the
  543i 19:10Z run (pinned ree-cloud-4, a participant in the divergent
  INERT basin) must land before ARC-062 demotion + ARC-063 / V4 strong-
  reading governance can proceed (branch e CONFIRMS-543i -> clears the
  single-machine caveat; branch a CONTRADICTS -> 543i was itself a
  single-machine basin artifact). The **ARC-065 behavioural-diversity
  developmental warm-start failure** (EXQ-ISEF cohort, V3-EXQ-587..591)
  remains the parallel dominant blocker. Historical context preserved
  below.
- **2026-05-18T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **720 cumulative
  completions** (164 PASS / 286 FAIL / 77 ERROR / 193 UNKNOWN);
  last_updated 2026-05-17T13:11:02Z -- the central file is now ~12h
  stale (the multi-machine runners write per-machine
  `runner_status/<hostname>.json`; cross-machine aggregate as of this
  read is **1103 cumulative** -- DLAPTOP-4.local 591 + ree-cloud-1 237
  + ree-cloud-2 170 + EWIN-PC 77 + Daniel-PC 28; 274 PASS / 486 FAIL /
  107 ERROR / 236 UNKNOWN). +2 central completions since the
  2026-05-17T01:11Z nightly read (V3-EXQ-582 FAIL SD-012 EMA-sweep
  substrate-ceiling + V3-EXQ-583 PASS SP-CEM main-path default-wiring
  equivalence). **Pending review queue (regenerated
  2026-05-17T12:59:48Z; last review 2026-05-17T10:24:31Z) is 0
  items.** **Currently queued (`experiment_queue.json`): 11 items**
  (0 -> 11) -- the infant_substrate GAP / EXQ-ISEF cohort:
  V3-EXQ-584 (GAP-7 traj_cosine), 586 (GAP-9 curriculum scheduler),
  587 (EXQ-ISEF-001 harm-gradient curriculum, claimed ree-cloud-4),
  588 (ISEF-002 transient-benefit z_goal seeding, claimed
  ree-cloud-2), 589 (ISEF-003 microhabitat latent diversity, claimed
  DLAPTOP-4.local), 590 (ISEF-004 novelty Goldilocks, claimed
  ree-cloud-3), 591 (ISEF-005 curriculum-vs-flat) + V3-EXQ-481b
  (MECH-090 V_s commit-release), 582a (goal_pipeline:GAP-3 Option 2
  drive_floor sweep), 592 (commitment_closure:GAP-11 committed-mode
  curriculum pilot). Cloud capacity scaled: ree-worker-3 (CX43) +
  ree-worker-4 provisioned. 2026-05-17 substrate / claim wave:
  (1) **plasticity-crystallization cluster registered** -- INV-074 +
  MECH-333 + MECH-334 + ARC-075 + Q-052 (developmental critical-period
  crystallization), INV-075 (signal-structure temporal decoupling;
  parent of INV-074 -- a self-extinguishing load-bearing signal
  necessarily requires lock OR handoff), ARC-076 + MECH-335 + MECH-336
  (developmental commitment-loop calibration window with critical-period
  lock = personality; INV-075 LOCK-arm instance) with an
  implementation_prerequisites HARD ORDERING GATE; claims.yaml now
  **645 claims**. (2) **INV-074/MECH-333/334 Phase-3 crystallization
  substrate landed** in ree-v3 (f8b93e3) -- GatedPolicy.crystallize()
  plasticity injection + ResidueField EWC residue write-protect +
  InfantCurriculumScheduler on_phase3_entry hook;
  crystallize_at_phase3 default OFF; 484/484 contracts PASS.
  (3) **ARC-062 GAP-B/C/D landed** -- head-input first-action one-hot
  augmentation (bypasses E2 world-forward compression) + discriminator
  -> SD-033a rule_state source + trainable rule_bias_head; all
  bit-identical OFF. (4) **ARC-065 SP-CEM flipped to main-path
  default** (intentional non-no-op; legacy collapsing CEM was the
  monostrategy root cause) -- V3-EXQ-583 default-wiring equivalence
  PASS. (5) **SD-012 sustained-drive amendment** -- Option 1
  drive_ema_alpha trace + Option 2 drive_floor for goal_pipeline:GAP-3
  (V3-EXQ-582 FAIL escalated to the 582a drive_floor sweep).
  (6) **commitment_closure GAP-3 + GAP-11** -- CausalGridWorldV2 env
  extensions primitives 1-3 + committed_mode_curriculum.py harness
  helper (unblocks the OCD battery V3-EXQ-460b/463b/464b/466b/467b/468b).
  Bottleneck (unchanged framing): the **ARC-065 behavioural-diversity
  developmental warm-start failure** remains the dominant scientific
  blocker, now under active test -- **EXQ-ISEF-001 (V3-EXQ-587
  harm-gradient curriculum) is Rank 1**, claimed by ree-cloud-4 and
  running; ISEF-002..005 (V3-EXQ-588..591) queued/claimed across cloud
  workers. The freshly-landed INV-074/MECH-334 crystallization substrate
  has its falsifier **V3-EXQ-543h** (2x2x2 ARC-062 GAP-B x
  crystallize_at_phase3, supersedes 543g) queued at priority 10 on
  ree-cloud-4. Historical context preserved below.
- **2026-05-17T01:11Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **718 cumulative
  completions** (163 PASS / 285 FAIL / 77 ERROR / 193 UNKNOWN);
  last_updated 2026-05-17T01:11:14Z. +37 cumulative completions since
  the 2026-05-13T01:10Z nightly read. **Pending review queue
  (regenerated 2026-05-16T20:52:59Z; last review 2026-05-16T18:22:30Z)
  is 0 items.** **Currently queued (`experiment_queue.json`): 0 items
  (`items: []`).** 2026-05-13 -> 2026-05-17 substrate / governance
  wave: (1) **sleep_substrate GAP-6 + GAP-8 complete** -- StepHarness
  write-path audit (e1_input scaled by anchor_weight in
  run_sws_schema_pass) + MECH-272 anchor-channel consumer
  (mean_anchor threaded through SleepLoopManager._run_cycle);
  V3-EXQ-565 GAP-8 routing-consumer full-runner PASS 2026-05-15T18:03Z
  (arm0_applied_mean=1.0, arm1~0.6, C1/C2/C3 True). (2)
  **sleep_substrate GAP-3 unified master flag** --
  `REEConfig.use_sleep_aggregation_cluster` resolves the eight Phase
  A-E sub-flags consistently (OR-only; MECH-204 + anchor-set / e2_harm_s
  prereqs deliberately not bundled); V3-EXQ-581 owner-EXQ dry-run 6/6
  PASS (all four phases fire end-to-end under one flag); the
  2026-05-16 GAP-4-entry GAP-8/GAP-3 conflation corrected. (3)
  **infant_substrate GAP-2/3/5/6 env substrates landed** --
  microhabitat Voronoi zones (V3-EXQ-577 FAIL, C2 diagnosed as a
  test-design false-negative and autopsied -- substrate functionally
  validated by C1/C3/C4; V3-EXQ-577a corrected C2 routed),
  transient-benefit patches (V3-EXQ-578 PASS), pos/zone telemetry
  (GAP-5), and residue-coverage telemetry (V3-EXQ-580 PASS 3/3 seeds).
  (4) **ARC-067 / ARC-068 child-MECH design completed 2026-05-16** --
  two-child split for ARC-067 (MECH-330 acute-restlessness accumulator
  + MECH-331 chronic-anhedonic-flatness substrate); ARC-068 collapses
  into MECH-320 per the ARC-068 lit-pull R3 verdict (Niv 2007
  mathematical symmetry); 635 claims; the biology-before-formal-
  definitions gate is now fully clear for the non_deficit_action_drives
  family. (5) **calibration-debt diversity sprint** -- V3-EXQ-569 FAIL
  (all arms entropy ~0.496, zero diversity lift), V3-EXQ-570 PASS (E2
  not the bottleneck, rollout ratio 52.1), V3-EXQ-571 PASS (F /
  forward-model term dominates 88-89% of E3 temporal variance; ALL
  MECH-313/314/320 + dACC / lateral_pfc / ofc / gated_policy / mech295
  bias components contribute ~0), V3-EXQ-573 NULL (ARC-065 bias-scale
  5-10x sweep -- all 10 arms bit-for-bit identical). (6) **governance
  Proposal G2** -- `scripts/check_backward_traceability.py` wired into
  governance.sh Step 4b as a hard gate (developmental-keyword claims
  must trace to developmental_needs_register.md). Two governance
  cycles in the window (2026-05-15T18:55Z applied 9
  hold_pending_v3_substrate on ARC-066/067/068, ARC-070/071, MECH-320,
  Q-043/044/045; 0 indexed pending confirmed). Bottleneck: the
  **ARC-065 behavioural-diversity developmental warm-start failure**
  is now the dominant scientific blocker, superseding the pure
  V_s-monostrategy framing. V3-EXQ-573's bit-for-bit-identical
  10-arm 5-10x sweep diagnosed the MECH-314a / MECH-320 diversity
  biases as **literally zero at cold start** (not miscalibrated);
  the right response is a developmental warm-start curriculum
  (DEV-NEED-029, PROPOSED), not more bias-scale sweeps.
  `docs/architecture/developmental_experiment_priorities.md` (created
  2026-05-16) ranks EXQ-ISEF-001 (harm-gradient curriculum) Rank 1,
  gating everything downstream; Q-043/044/045 and INV-049 retests are
  deferred until the warm-start gate is established via
  EXQ-ISEF-001..006. Historical context preserved below.
- **2026-05-13T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **681 cumulative
  completions** (134 PASS / 271 FAIL / 76 ERROR / 193 UNKNOWN); V3
  subset is 652 runs (125 PASS / 258 FAIL / 76 ERROR / 193 UNKNOWN).
  +7 cumulative completions since the 2026-05-12T01:10Z nightly read.
  Pending review queue regenerated 2026-05-12T18:15Z is **11 items**
  (1 PASS / 6 FAIL / 4 runner-only); the index is stale on the
  V3-EXQ-552 / 555 / 557 monostrategy diagnostics whose manifests have
  not yet propagated through `build_experiment_indexes.py`, and
  V3-EXQ-540c is held as ERROR with the re-run already PASSed under
  the V3-EXQ-540d queue ID. **Currently queued
  (`experiment_queue.json`): 0 items** (`items: []`); today's wave of
  monostrategy diagnostics (V3-EXQ-554a / 555 / 556 / 557 / 558) and
  the MECH-307 default-fix validation V3-EXQ-540e were all pre-claimed
  by runners before the nightly read so the central queue file is
  empty at this snapshot. 2026-05-12 substrate / runner / governance
  wave: (1) **MECH-307 default-value recalibration landed**
  (`mech295_min_drive_to_fire` 0.1 -> 0.01 and
  `mech307_conjunction_z_beta_threshold` 0.6 -> 0.3) after V3-EXQ-540d
  diagnosed the V3-EXQ-540b conj_fire_rate=0 cohort as a
  drive-floor-never-crossed + z_beta-ceiling-below-threshold
  substrate-ceiling pattern (drive_level max=0.030 / mean=0.016
  vs 0.1 floor; z_beta_arousal max=0.545 vs 0.6 floor); 314/314
  contracts + 7/7 preflight PASS with new defaults; V3-EXQ-540e
  behavioural validation queued (DLAPTOP-4.local, 90 min,
  3 seeds x 3 conditions x 70 ep, supersedes V3-EXQ-540d); dry-run
  smoke at 6 ep / 1 seed produced ARM_2_full `conj_fire_rate=0.155`
  (first time the conjunction has fired since substrate landed
  2026-05-11). (2) **V3-EXQ-461 EXP-0157 delayed-reward persistence
  substrate-readiness PASS** 2026-05-12T18:18Z on DLAPTOP-4.local
  closing commitment_closure_plan.md GAP-2 at substrate-readiness
  level; full behavioural delayed-reward arm remains blocked on GAP-3
  CausalGridWorldV2 env extensions. (3) **V_s-monostrategy diagnostic
  cohort queued** spanning agent-init / env-init factorisation
  (V3-EXQ-555 2x2 seed factorial; V3-EXQ-557 30-cell agent-seed sweep;
  V3-EXQ-556 8-arm module-init swap; V3-EXQ-558 seed-pair
  readout/rank diagnostic) plus the proposer-stage pipeline
  localisation V3-EXQ-554a (re-queue of V3-EXQ-554 killed by the
  06:10Z systemctl restart that deployed the runner_remote_control
  `_rrc` NameError fix). (4) **runner_remote_control _rrc NameError
  fix landed** 2026-05-12T06:14Z (hoisted import to module top-level
  so the `_push_remote_heartbeat` closure can resolve `_rrc` from
  module globals rather than `main()` scope; preflight 7/7 PASS;
  ree-cloud-1 and ree-cloud-2 systemd services restarted clean via
  SSH). (5) **runner manifest-leak in conflict-recovery fix landed
  2026-05-09** (capture pre-reset HEAD SHA, restore each
  `result_files` path via `git checkout <pre_reset_sha> -- <rel>`
  after the reset+pop; `git_push_results` now passes `result_files`
  through to `_git_push_with_retry` so the recovery branch sees the
  expected files rather than falling into broad-add fallback;
  contract suite test_runner_manifest_survives_conflict_recovery.py
  added). Bottleneck: **V_s-monostrategy substrate ceiling** remains
  the dominant scientific bottleneck across at least four open
  threads (SD-029 self-attribution; SD-032b dACC arbitration;
  ARC-062 rule apprehension; MECH-307 consumer-conjunction read);
  V3-EXQ-555/556/557/558 narrow the basin geometry of the agent-init
  axis discovered by V3-EXQ-552 (entropy~0.68 at seed=7 vs 0.0 at
  seeds 42/17 under bit-identical code); V3-EXQ-553 orthogonal CEM
  seeding is the proposer-stage substrate-side fix under test;
  V3-EXQ-540e is the immediate MECH-307 consumer-side validation;
  V3-EXQ-543d remains the ARC-062 Phase 3 wiring 2x2 factorial.
  Historical context preserved below.
- **2026-05-12T01:10Z nightly read.** Central
  `evidence/experiments/runner_status.json` reports **674 cumulative
  completions** (136 PASS / 270 FAIL / 75 ERROR / 193 UNKNOWN); V3 subset
  is 639 runs (116 PASS / 257 FAIL / 73 ERROR / 193 UNKNOWN). Pending
  review queue regenerated 2026-05-11T20:14Z is **0 items** (all
  experiments reviewed including V3-EXQ-550 and V3-EXQ-543c FAILs walked
  through the 2026-05-11T20:10Z `review-exq550-543c` session).
  **Currently queued (`experiment_queue.json`): 1 item -- V3-EXQ-540b**
  (MECH-307 consumer-conjunction threshold sweep, claimed by DLAPTOP-4.local
  2026-05-12T01:07Z; 4 arms varying only the wanting/liking/z_beta consumer
  thresholds with the MECH-307 substrate fully ON in every arm; supersedes
  V3-EXQ-540a FAIL). 2026-05-11 wave landed: (1) **MECH-307 Option-b Gap-1
  substrate landing** (VALENCE_DIM 4 -> 6 with split positive/negative
  surprise channels; closes goal_pipeline:GAP-1); (2) **SD-054 bipartite
  layout extension** (reef/forage spawn partition forcing categorically
  opposite first-action argmaxes); (3) **MECH-323 + MECH-324 ARC-071
  child claims registered** (formation accumulator + maintenance operator
  for chunk lifecycle; substrate implementation pending); (4) governance
  cycle 2026-05-11T17:13Z applied 16 `hold_pending_v3_substrate`
  recommendations and reclassified the V3-EXQ-433/433a/433b/470/433d/433f/
  537/537a/523b cohort as non_contributory under the V_s monostrategy
  reading; (5) self_attribution_plan GAP-1 status `open -> blocked`
  (forensic read showed V3-EXQ-445h was two-arm not three-arm, and
  three-arm 445/445b runs produced floating-point-identical metrics
  across architectural arms -- monostrategy substrate ceiling, not
  arbitration data); (6) **V3-EXQ-550 z_goal monostrategy falsifier
  FAIL** 2026-05-11T19:10Z (action_class_entropy=0.0 in BOTH arms;
  z_goal_update_calls=1200 with z_goal_norm_peak=0.0 across all ARM_ON
  seeds -- supports MECH-269 V_s substrate-level reading at no-training
  depth; trained-z_goal follow-on recorded as natural successor); (7)
  **V3-EXQ-551 + V3-EXQ-551a pipeline-entropy diagnostic PASS** 2026-05-11
  20:16Z / 20:36Z (localised the entropy=0.0 cliff to the CEM proposer
  stage); (8) **V3-EXQ-552 forced-exploration warmup + V3-EXQ-553
  orthogonal CEM seeding queued** as the two complementary substrate-
  side fixes; (9) three new lit-pulls (Q-045 LC-NE-vs-dACC + ARC-064
  Pull-2 follow-on covering MECH-316/317/318 child mechanisms, ~14
  lit entries total) lifting literature totals to 1345. Bottleneck:
  V_s-monostrategy substrate ceiling (the dominant scientific
  bottleneck; pipeline-entropy diagnostic localised the entropy=0.0
  cliff to the proposer; V3-EXQ-553 orthogonal CEM seeding is the
  immediate substrate-side test) plus the MECH-307 consumer-side
  threshold-sweep (V3-EXQ-540b currently running) and ARC-062 Phase 3
  wiring (V3-EXQ-543d 2x2 factorial currently running) -- three
  concurrent threads sharing the same monostrategy floor. Historical
  context preserved below.
- **1045 runner-side completions across all five machines** (per per-machine
  `runner_status/<host>.json` 2026-05-11T01:10Z read: 555 DLAPTOP-4.local +
  28 Daniel-PC + 77 EWIN-PC + 222 ree-cloud-1 + 163 ree-cloud-2; cumulative
  cross-machine breakdown 241 PASS / 463 FAIL / 105 ERROR / 236 UNKNOWN;
  v3 subset still dominates the post-2026-02-27 epoch). The single-file
  `runner_status.json` cited in prior nightlies no longer exists; per the
  multi-machine coordination policy each machine writes to
  `evidence/experiments/runner_status/<hostname>.json`. The 665 figure in
  the 2026-05-10T01:10Z snapshot was an under-count and has been replaced
  here with the cross-machine aggregate. **+10 cumulative-by-machine
  completions since the 2026-05-10T01:10Z nightly read** (deduped: 8
  unique experiments because V3-EXQ-545 + V3-EXQ-546 each ran twice via
  the multi-machine claim race). Today's wave covers the four remaining
  ARC-064/ARC-065/ARC-066 child substrate landings (MECH-313 stochastic
  noise floor + MECH-314 structured curiosity bonus + MECH-319 simulation-
  mode rule-write gate + MECH-320 tonic vigor coupling -- all
  substrate-readiness 5-6 sub-test PASS; closes arc_062 GAP-K and
  ARC-066 first child), the ARC-062 Phase 3 falsifier V3-EXQ-543b on
  Mac (~4h, FAIL -- Phase 3 wiring design needs refinement before
  another attempt), the SD-049 Phase 2 behavioural validation cohort
  V3-EXQ-514h / 514i on cloud-1 (both PASS), the V3-EXQ-141c MECH-111
  novelty-drive RNG-desync FAIL on cloud-2 (re-route to /diagnose-errors
  per the per-seed-distribution diagnostic interpretation grid the
  2026-05-09 wave's V3-EXQ-141d successor was already deferred to), and
  the false-ERROR-stdout-sentinel diagnose-errors session that landed
  the canonical `verdict:` print fix across V3-EXQ-542 / 544 / 545 (and
  the sibling MECH-319 implement-substrate session applied the same fix
  to V3-EXQ-546 before queueing). **Pending review queue regenerated
  2026-05-10T13:02Z is 0 items** -- the 2026-05-10T12:24Z governance
  cycle walked the 3 PASS pending experiments (V3-EXQ-500a + V3-EXQ-543
  + V3-EXQ-503a) and applied 13 pending_user recommendations via
  decision_log.v1.jsonl appends (10 hold_pending_v3_substrate on
  ARC-062, ARC-064, ARC-065, MECH-309, MECH-312, MECH-313, MECH-316,
  MECH-317, MECH-318, SD-054 newly registered v3_pending claims; 3
  hold_candidate_resolve_conflict on ARC-045, MECH-166, MECH-204).
  substrate_queue.json grew from 69 -> 78 entries with the new
  ARC-064 / ARC-065 / MECH-313 / MECH-314 / MECH-316 / MECH-317 /
  MECH-318 / MECH-319 / SD-054 entries reflecting today's substrate
  landings + cluster registrations. Earlier 2026-05-09/10 narrative
  context preserved below. Historical context: the 2026-05-10T01:10Z
  nightly read reported 665 completions (131 PASS / 266 FAIL / 73 ERROR /
  195 UNKNOWN) +11 vs the 2026-05-09T01:10Z nightly read -- the
  2026-05-09 wave covered the ARC-062 Phase 1 substrate landing
  (V3-EXQ-542 substrate-readiness 5/5 manifest PASS UC1-UC5 with runner
  outcome ERROR per the substrate-readiness pattern), the ARC-062 Phase 2
  monomodal-collapse falsifier (V3-EXQ-543 PASS on Mac in ~50min ARM_0
  use_gated_policy=False vs ARM_1c full 3-stream discriminator at
  ARM_1_med density on SD-054 reef), the MECH-204 sleep-substrate Phase 1
  step-size sweep cohort (V3-EXQ-541a/541b/541c all PASS on DLAPTOP-4.local /
  Mac), and the sleep_substrate_plan.md GAP-2 Tier-1 successor cohort
  landings (V3-EXQ-265a PASS 2026-05-09T20:12Z + V3-EXQ-500a PASS 20:41Z +
  V3-EXQ-503a PASS 21:46Z + V3-EXQ-436a FAIL 21:52Z + V3-EXQ-418l FAIL
  21:53Z, all on Mac after the user-initiated runner restart at ~20:00Z;
  4 of 5 Tier-1 successors clear with two FAILs that route to the
  /diagnose-errors per-seed-distribution-diagnostic interpretation grid
  rather than to substrate retraction). Pending review queue
  regenerated 2026-05-09T20:18Z is **2 items** (both FAIL, both deferred to
  /diagnose-errors per the bit-identical-arms measurement-validity pattern):
  V3-EXQ-530c (ARC-016 precision-commit StepHarness retest, carried over
  from the 2026-05-08T22:34Z governance cycle) and V3-EXQ-141d (MECH-111
  novelty-drive RNG-desync). Earlier wave context (2026-05-09T01:10Z
  snapshot): +23 vs the 2026-05-08T01:11Z read -- the
  2026-05-08 wave covered the bug-fix retest cohort landings
  (V3-EXQ-433f UNKNOWN/FAIL on ree-cloud-1; V3-EXQ-483b on ree-cloud-2;
  V3-EXQ-514f sleep-on cohort), MECH-307 conjunction architecture validation
  (V3-EXQ-539 FAIL with MECH-307 hold_pending_v3_substrate decision applied
  and Q-040 narrowed via Q-040.a/b/c sub-question decomposition), MECH-204
  precision-recalibration consumer Phase 1 validation (V3-EXQ-541 FAIL at
  runner level with manifest verdict PASS pending governance walk),
  V3-EXQ-514g StepHarness wider-seed sweep PASS, V3-EXQ-244a stale-ERROR ->
  PASS reconciliation, plus the seven plan-of-record registrations
  (sleep_substrate_plan / commitment_closure_plan / self_attribution_plan /
  goal_pipeline_plan + queue-completeness back-fills MECH-267 / MECH-268 /
  SD-018 + rule-apprehension cluster registration MECH-309 / ARC-062 /
  ARC-063 / SD-054 rename + runner env-isolation fix + active-claim
  evidence/ guard broadening). Pending review queue regenerated
  2026-05-08T22:38Z is **1 item** (V3-EXQ-530c ARC-016 precision-commit
  StepHarness retest; deferred to /diagnose-errors per the bit-identical-arms
  measurement-validity pattern). The 2026-05-08T22:34Z governance cycle
  walked 10 indexed pending FAILs (4 superseded predecessors EXQ-537/537b/537c/141c
  flipped to superseded; 3 already-triaged accepted as-tagged; 530c held;
  537d + 539 accepted as-tagged) and applied 2 pending_user decisions:
  MECH-307 hold_pending_v3_substrate + Q-040 narrow_open_question
  (decomposed into Q-040.a factorial 2x2 of {MECH-269b OFF/ON} x
  {SD-032b OFF/ON} on EXQ-483a retest, Q-040.b alternative-hypothesis
  isolator MECH-295 cross-witness probe, Q-040.c mechanism-quantification
  dACC weight delta). Historical-context (2026-05-08T01:11Z snapshot):
  120 PASS / 255 FAIL / 72 ERROR / 184 UNKNOWN; +3 vs the 2026-05-07T01:12Z
  nightly read -- a quiet day
  on the runner side (Mac claimed V3-EXQ-535a SD-029 P1 eval-fix, ree-cloud-1
  claimed V3-EXQ-433f, ree-cloud-2 claimed V3-EXQ-483b, plus V3-EXQ-530b
  rename-rerun -- all three new completions surface as UNKNOWN result codes
  pending the next governance walk; the heavy activity was in /diagnose-errors
  + /governance + multi-sense audit + bug-fix retest queue construction, not
  runner throughput). +20 vs the 2026-05-06T01:10Z nightly read covering the
  2026-05-06 in-flight wave: V3-EXQ-244a / 525 / 526 / 527 / 528 / 529 / 530 /
  531 / 532 / 533 / 534 / 535 + the EXQ-418k canonical run_id restoration +
  the EXQ-433/452a manifest cleanups + the Q-019 disconfirming-balance lit pull
  (4 weakens entries pulled the corpus from 11s/5m/0w to 11s/5m/4w; lit_conf
  0.884 -> 0.776). The runner queue is now **empty** (`items: []` 2026-05-07T01:12Z)
  -- the 9 items in flight at the previous nightly snapshot have all completed
  through the day, and the active governance cycle (claim
  governance-2026-05-06T2156Z) is now walking the 32-item pending_review
  accumulation (see "Currently queued" section below for the empty-queue note
  and the pending_review breakdown). Earlier 2026-05-04/05/06 reef-enrichment
  substrate wave context preserved below for continuity. +23 vs the 2026-05-04 nightly read covering the
  2026-05-04/05/06 reef-enrichment substrate wave (SD-050 reef substrate landed in
  CausalGridWorldV2 with V3-EXQ-521 PASS 7/7 + V3-EXQ-522 PASS zone_transitions=48.9),
  the SD-019a / SD-051 / SD-052 substrate landings (V3-EXQ-518/519/520 dry-run PASS;
  V3-EXQ-518 / V3-EXQ-520 in flight), the V3-EXQ-485a SD-033b multi-mode landing PASS,
  the V3-EXQ-503 SD-017 sleep-phase discriminative pair PASS carry-over from 2026-05-01,
  and a wave of reef-superseding EXQs that re-test monostrategy-blocked predecessors
  (V3-EXQ-433e / 445e/f/g / 325f / 452a / 454a / 514c / 526 / 527 against SD-029,
  SD-032b, SD-032c, MECH-257, ARC-016, Q-034, MECH-112, SD-049 Phase 2). The
  reef-enabled supersession was the strategic lever decided 2026-05-05 after the
  monostrategy-audit-2026-05-05T0712Z full-scan: rather than attempt env-tuning per
  EXQ, swap in the reef substrate as the env-entropy precondition under all
  affected predecessors. Pending review queue (per pending_review.md regenerated
  2026-05-05T22:12Z) is **4 items** (3 FAIL + 1 ERROR): V3-EXQ-454a (ARC-016
  adaptive commitment under reef), V3-EXQ-452a (ARC-033 / MECH-257 / SD-013 dual-
  function E2 under reef), V3-EXQ-525 (SD-003 attribution anchor on post-SD-011
  substrate), and V3-EXQ-418j ERROR (SD-016 reef env-entropy fix; runner-only).
  Build-indexer fix landed 2026-05-05: backlog literature-evidence epoch filter
  brought into parity with the matrix builder (one-line change at
  `build_experiment_indexes.py:3002` to mirror the matrix's `Literature entries
  are not epoch-filtered` policy); MECH-057 lit_count 0->7 / lit_conf 0.0->0.827;
  MECH-062 dropped from backlog (confirmed_established); missing_literature_evidence
  reasons across backlog 3 -> 1 (only EVB-0131 onboarding phantom remains).
  Lit-pull supplements 2026-05-05: INV-054 depressive maintenance loop +3 entries
  (lit_conf 0.762 -> 0.858 -- Jacobson 1996 BA RCT, van de Leemput 2014 PNAS
  bistable mood, Tang 1999 sudden gains). 2026-05-04T22:01Z governance walk:
  pending_review cleared after walking ARC-026 / MECH-093 promotions and the
  V3-EXQ-485a multi-mode SD-033b PASS. **2026-05-04 narrative continues, retained
  for context:** +11 vs the 2026-05-03 nightly read -- the SD-047 / SD-048 / SD-049-Phase-1 substrate
  validation runs (V3-EXQ-509 / 511 / 513) executed on Mac 2026-05-03, plus the MECH-302
  comparator-logic substrate validation run V3-EXQ-515 PASS 7/7 on Mac 2026-05-04, plus
  V3-EXQ-495 ERROR on ree-cloud-1 (SIGTERM at 4h of ~40h MECH-163 run -- infrastructure
  kill, governance-deferred until Q-040b cluster-successor lands). Substrate-validation
  runs are typically labelled UNKNOWN by the runner result code while passing in the
  manifest-level acceptance interpretation -- the +9 UNKNOWN delta covers the four new
  substrate runs above plus carry-overs surfaced by the indexer dedup sweep.
  Most recent runner completions remain V3-EXQ-490e MECH-295 seeding-strengthening successor
  (Q-040b SD-039/MECH-295 floor-relaxation arm) **FAIL on Mac 2026-05-01T03:19Z**, and
  V3-EXQ-503 / EXP-0171a SD-017 sleep-phase **discriminative pair** PASS 3/3 seeds on
  Mac 2026-05-01T20:15Z (~0.32s; runner result code UNKNOWN but manifest verdict PASS).
  V3-EXQ-503 closes the SD-017 evidence gap left by V3-EXQ-500 -- the substrate-readiness
  probe was diagnostic_probe and excluded from scoring, so SD-017 sat at exp_conf=0.000 /
  plausible_unproven despite lit_conf=0.901; V3-EXQ-503 forces 4-phase sleep ON vs OFF over
  matched buffer state and measures three substrate metrics (M1 cumulative_sws_writes;
  M2 ctxmem_state_change Frobenius norm with magnitude rather than direction so the metric is
  agnostic to slot-diversity reduction during real prototype clustering; M3 cumulative_rem_rollouts).
  ARM_A: sws_writes=24, ctxmem_delta=4.50-5.03, rem_rollouts=18 across all seeds; ARM_B:
  zeros across the board. **Governance impact:** SD-017 quadrant flipped plausible_unproven
  -> confirmed_established; exp_conf 0.000 -> 0.775; **all four EXP-0170/171/171a/172/173
  Phase 2 standard-gating cohort claims (MECH-094, SD-017, SD-035, MECH-062) are now
  confirmed_established**. Earlier in the same window (+4 vs the 2026-04-29T11:34Z read):
  V3-EXQ-499 MECH-094 hypothesis-tag write-gate discriminative pair PASS
  (3/3 seeds, MECH-094 quadrant flipped plausible_unproven -> confirmed_established;
  exp_conf 0.000 -> 0.775); V3-EXQ-500 SD-017 sleep-phase substrate-readiness diagnostic PASS;
  V3-EXQ-501 SD-035 amygdala analog vs binary PASS; V3-EXQ-502 MECH-062 tri-loop coordination PASS.
  All four executed within minutes of queueing on DLAPTOP-4.local (the local runner picked up
  EXP-0170/0171/0172/0173 immediately after Phase 2 reckoning landed earlier the same day:
  claim-type evidence gating distinguishes substrate_coherence / answer_state / standard
  gating; impl_no_exp 15 -> 4 after gating, four genuinely-testable MECH/SD claims surfaced
  + queued + executed inside one session). +1 vs the 2026-04-29T01:10Z read covering
  V3-EXQ-490c completion (FAIL on Mac 2026-04-29T08:34Z; MECH-269b V_s gating + MECH-295
  liking-bridge factorial; Q-040b behavioural sufficiency arm; pending discussion as of
  this read). +6 vs the 2026-04-27 read covering the 2026-04-28 diagnostic wave (V3-EXQ-498 OCD Layer 1
  closure-threshold sweep -- FAIL/non_contributory in governance, escalates to Layer 2/3;
  V3-EXQ-418f SD-016 attention-uniformity probe -- diagnostic; V3-EXQ-418g SD-016 selectivity-first
  4-arm -- C1+C2+C3 PASS but C4+C5 FAIL with action_class_entropy~1.1e-10 across all four arms,
  reclassified non_contributory in 2026-04-28 governance after env-entropy precondition gap
  diagnosed; V3-EXQ-418h env-entropy precondition probe -- FAIL, routes to broader
  env-enrichment scoping; V3-EXQ-490b MECH-269b VsRolloutGate substrate-readiness probe
  -- FAIL outcome under UNKNOWN result code, governance-classified inconclusive,
  Q-040b points at MECH-295 bridge as remaining blocker now that the bridge has landed).
  The indexer-vs-runner gap is the historical pre-runner_status
  archive plus the per-seed manifests the runner records as a single queue entry.
  Spanning SD-003 through SD-023 validation, heartbeat architecture (SD-006),
  reafference (SD-007), encoder fixes (SD-008/009), harm stream separation (SD-010),
  dual nociceptive streams (SD-011/SD-022), homeostatic drive (SD-012),
  self-attribution counterfactuals (SD-013/ARC-033), valence vector recording (SD-014),
  resource encoder (SD-015), frontal cue integration (SD-016), sleep infrastructure
  (SD-017), surprise-gated replay (MECH-205), E1 predictive wanting (MECH-216),
  wanting/liking dissociation (MECH-112/229/117), goal conditioning
  (MECH-116/163/ARC-032), context memory (MECH-153/ARC-042), EXQ-223 minimal vertebrate
  ablation milestone, the SD-032 cingulate cluster (a/b/c/d/e) validated inline
  2026-04-19, SD-033a lateral-PFC-analog landing (V3-EXQ-456 PASS), SD-034 governance
  closure operator + MECH-267 + MECH-268 landing smokes, the SD-035 amygdala-analog
  landings (V3-EXQ-473 SD-035 CeA mode-prior PASS, V3-EXQ-474 SD-035 BLA encoding+remap
  PASS) plus V3-EXQ-455 SD-032a behavioural coordinator PASS, and the 2026-04-22 V_s
  invalidation runtime substrate wave (SD-036 GABAergic cross-stream decay + MECH-279
  PAG freeze gate; MECH-269 Phase 1 per-stream V_s; MECH-288 event segmenter; MECH-287
  invalidation trigger; MECH-269 Phase 2 ii AnchorSet; MECH-269 Phase 2 iii T4
  per-region V_s) all landing-diagnostic PASS via contract tests (85/85) and activation
  smokes. The 2026-04-24 wave (MECH-284 Phase 3 staleness accumulator + MECH-269 online
  hysteresis swap + MECH-290 backward credit sweep) extended the contract suite
  to 91/91 PASS with all flags OFF. The 2026-04-25 wave (SD-037 broadcast-override
  regulator + Sleep Aggregation Cluster Phases A/B/C/D/E covering SleepLoopManager
  scaffolding + MECH-285 SleepReplaySampler + MECH-272 RoutingGate + MECH-275
  BayesianAggregator + MECH-273 SelfModelAggregator + StalenessAccumulator.partial_decay
  + SD-016 Path 1 ContextMemory diversification loss) extends the contract suite to
  150/150 PASS (143 contracts + 7 preflight) with all flags OFF on the 2026-04-25
  read; the 2026-04-26 wave (SD-039 substrate + SD-033b OFC-analog + MECH-269b
  symmetric V_s gating + MECH-295 weak-reading liking bridge) extended the suite
  to 164/164 contracts + 7/7 preflight PASS; the 2026-04-27 wave (SD-039 module-
  level write-site population layer + MECH-292 ranked ghost-goal bank +
  MECH-293 waking ghost-goal probe search) extends the suite further to
  **183/183 contracts + 7/7 preflight PASS** with all flags OFF, preserving
  the bit-identical-when-OFF guarantee.
- **Currently queued (2026-05-11T01:10Z): 0 items (empty `items: []`).** The
  2026-05-10 substrate wave drained the queue again. Today's queue churn:
  V3-EXQ-543b (ARC-062 Phase 3 falsifier on Mac, FAIL after ~4h),
  V3-EXQ-544 / 545 / 546 / 547 (substrate-readiness diagnostics for
  MECH-313 / MECH-314 / MECH-319 / MECH-320, all 5-6 sub-test PASS), plus
  V3-EXQ-514h / 514i SD-049 Phase 2 behavioural successors on cloud-1
  (both PASS) and V3-EXQ-141c MECH-111 novelty-drive RNG-desync on
  cloud-2 (FAIL, re-route to /diagnose-errors). Next-up substrate work
  in `evidence/planning/substrate_queue.json` (now 78 entries, was 69
  before today's governance cycle pushed the new ARC-064/ARC-065/
  MECH-313/314/316/317/318/319/SD-054 entries) is the ARC-062 Phase 3
  wiring pass that closes commitment_closure_plan.md GAP-1 (still
  blocked after V3-EXQ-543b FAIL pending design refinement), the
  SD-049 Phase 2 z_resource encoder follow-on (ready=True priority=2),
  and the ARC-066/067/068 + ARC-069/070/071 family child-MECH design
  cycles that flow from today's two cluster registrations. ARC-070's
  first child mechanism MECH-321 (policy.decomposition_via_event_segmenter)
  was registered candidate / v3_pending today; substrate work waits on
  MECH-288 event_segmenter.py input_stream label extension per the
  R2 bidirectional-substrate verdict.

  Historical-context (2026-05-10T01:10Z snapshot): 0 items.
  Historical-context (2026-05-09T01:10Z snapshot): 1 item. V3-EXQ-540
  (MECH-307 3-arm gap decomposition + Path B consumer conjunction read,
  priority=5, machine_affinity=DLAPTOP-4.local, 70 episodes x 3 seeds x
  3 conditions, ~90 min, claimed by Mac at 2026-05-09T00:00:27Z). 3-arm
  gap decomposition: ARM_0_off / ARM_1_signed_pe (Gap 1 only) /
  ARM_2_full (all 4 gaps). Bridge consumer-conjunction-read ON in ALL
  arms so ARM_2 has the downstream consumer that V3-EXQ-539 lacked.
  ARM_3 SD-014 6-channel fallback DEFERRED -- if the experiment
  PARTIAL-PASSes (C1+C2 only), ARM_3 becomes the natural follow-on.
  Acceptance: C1 substrate-readiness counter dissociation, C2 conjunction-fire
  rate >= 0.10 in ARM_2 only, C3 approach_commit_rate ARM_2 >= ARM_0 + 0.10
  in 2/3 seeds AND ARM_2 > ARM_1. The other Tier-1 follow-on candidates
  (V3-EXQ-514g wider-seed StepHarness sweep, V3-EXQ-530b ARC-016
  rename-rerun, V3-EXQ-433f reef SD-029 comparator on cloud) all completed
  through the 2026-05-08 day cycle and are now reflected in the +23
  delta above. V3-EXQ-541 MECH-204 sleep-substrate Phase 1 validation
  also completed 2026-05-08T23:43Z (FAIL outcome at runner level; manifest
  verdict PASS pending governance walk). Historical-context (2026-05-08T01:11Z
  snapshot): 7 items. The queue went from
  `items: []` at the previous nightly to a small but full slate: the
  `bugfix-requeue-433f-483b-476c-490f-445h-514f-523b` 2026-05-07T21:30Z session
  wrote and queued seven lettered-iteration corrected scripts fixing two
  ree_core bugs (Bug 1 BreathOscillator disabled with breath_period=0 in all
  prior runs; Bug 2 `_committed_step_idx` saturation at H-1=29 never resetting
  between E3 commits in the non-bistable path); 476c also required an
  agent.act() -> sense()/select_action() API migration. Three of the seven
  are claimed and running (V3-EXQ-433f SD-029 reef comparator on ree-cloud-1
  since 2026-05-07T21:39Z; V3-EXQ-483b SD-037 broadcast override 4-arm on
  ree-cloud-2 since 2026-05-07T21:39Z; V3-EXQ-514f SD-049 Phase 2 reef
  behavioural validation on DLAPTOP-4.local since 2026-05-08T01:04Z).
  V3-EXQ-523b (SD-029 reef-unblocked comparator graduation test, Bug 1+2 fix)
  is pending. Two MECH-295 follow-ons from the
  `goal-seeding-diagnostic-followup-2026-05-07T2255Z` session are pending:
  V3-EXQ-536a (per-step instrumentation diagnostic dispatching three candidate
  root causes for EXQ-536's z_goal_active_fraction=0.0 -- H_a drive-collapse
  on contact, H_b benefit threshold never crossed, H_c update fires but
  z_goal does not grow) and V3-EXQ-536b (z_goal_inject force-arm bypassing
  persistent-state seeding via MECH-188 hook to isolate upstream-vs-downstream
  blocker). V3-EXQ-537 (SD-029/MECH-256 single-pass comparator residual)
  is pending and supersedes V3-EXQ-535a after the
  `triage-7-weighting-multi-sense-2026-05-08T0015Z` audit reclassified
  EXQ-535a non_contributory for both SD-029 and MECH-256 (the script
  computes a two-pass cf_gap, not the single-pass residual the SD-029 /
  MECH-256 spec requires). 2026-05-07 morning preceded all of this with the
  `register-threshold-supervisor-2026-05-07T2225Z` session registering Q-041
  (unified meta-level threshold supervisor research direction; default-toward-Q
  approach with EXP-0170 exploratory probe gated on V_s-monostrategy substrate
  clearing) and the `fix-update-z-goal-bug-2026-05-07T2330Z` /
  `fix-update-z-goal-bug-phase0-2026-05-07T2335Z` two-step substrate +
  manifest-supersession pass that landed `ree-v3/experiments/_harness.py`
  StepHarness (canonical sense/update_z_goal/update_residue sequence;
  kwargs-only call shape; no bare-except wrappers) + `_metrics.py` canonical
  extractors, marked 4 manifests superseded (483b/490c/490b/483a) and 7
  trace-only across the update_z_goal positional/kwarg TypeError + bare-except
  silent-swallow cohort, and added evidence_quality_note to 8 affected claims
  (Q-040, MECH-269b, MECH-295, SD-037, MECH-280, MECH-281, SD-036, MECH-279).

  Pending review queue regenerated 2026-05-08T00:43Z is **0 items** -- the
  active 2026-05-06T21:56Z governance cycle finished walking the 32-item
  accumulation overnight (decisions land downstream of this nightly; the
  cycle's claim entry lists `status: "done"` per WORKSPACE_STATE.md
  Recent Work; the multi-sense audit on top of that cleared an additional 7
  weighting scripts via supersede-only / supersede+requeue / quality_note).

  Historical-context (2026-05-07T01:12Z snapshot): the queue was empty;
  pending_review was at **32 items**. The 2026-05-06 wave that flowed into the queue and back out:
  V3-EXQ-244a (MECH-165 replay diversity, ERROR -- SIGTERM cloud kill),
  V3-EXQ-525 (SD-003 attribution anchor, FAIL then PASS on a re-run),
  V3-EXQ-526 (Q-034 reef threshold sweep, FAIL),
  V3-EXQ-527 (MECH-112 goal-directed reef + identity encoder, FAIL twice),
  V3-EXQ-528 (SD-029 comparator trained, PASS),
  V3-EXQ-529 (MECH-098 reafference selectivity, FAIL),
  V3-EXQ-530 (ARC-016 precision-commit circuit, ERROR -- SIGTERM cloud kill),
  V3-EXQ-531 (SD-015 ResourceEncoder ablation, FAIL),
  V3-EXQ-532 (SD-005 latent domain selectivity, FAIL),
  V3-EXQ-533 (MECH-102 harm-stream ablation, PASS),
  V3-EXQ-534 (SD-016 cue terrain training, PASS),
  V3-EXQ-535 (SD-029 P1 target fix; max(hazard_field_view) instead of harm_signal,
  ceiling 0.798 > lowered gate 0.65; runner claimed immediately on
  DLAPTOP-4.local at 21:48Z, FAIL).
  Pending review queue regenerated 2026-05-06T21:57Z is **32 items** (7 PASS / 14 FAIL / 11 runner-only ERROR/UNKNOWN/smoke);
  the active governance cycle (claim governance-2026-05-06T2156Z) is walking
  the queue. The 2026-05-06T11:31Z afternoon-snapshot context for the
  reef-enrichment supersession wave was preserved above; the actionable
  decisions from this nightly are downstream of the governance walk.

  Historical-context (2026-05-06T01:10Z snapshot): the queue at that read was 9 items.
  - **V3-EXQ-524** (priority=5, any) -- reef fishtank showcase reef-aware
    episode log for fishtank_viz reef rendering (reef_cells, in_reef per step,
    shelter mode, reef_entry/exit transitions; ARM_1_reef_food config from
    V3-EXQ-522).
  - **V3-EXQ-514c** (priority=8, claimed by ree-cloud-2 2026-05-05T18:48Z) --
    SD-049 Phase 2 reef-environment behavioural validation (supersedes 514b,
    monostrategy fix; reef_enabled=True + hazard_food_attraction=0.7 +
    10x10 grid + 3 hazards on reef arms; targets C2b probe_acc_neighborhood
    >= 0.6 that 514b's 8x8/0-hazard env failed at 0.483).
  - **V3-EXQ-514d** (priority=7, claimed by DLAPTOP-4.local 2026-05-05T23:34Z)
    -- SD-049 BG gating diagnostic; tracks rv_final + committed_frac_last_third
    per seed per episode under default 514b config (no bistable, no dACC, no
    urgency_weight). Smoke confirmed rv_final=0.5 all 3 seeds under random
    placement; full 90-episode training will show whether the variance gap
    ever closes.
  - **V3-EXQ-514e** (priority=7, DLAPTOP-4.local) -- SD-049 BG gating seaweed
    diagnostic; ARM_A=random vs ARM_B=seaweed (2 landmarks, landmark_b_resource_bias=1.0,
    world_obs_dim=375); supersedes 514d; PASS condition committed_frac_last_third_b
    > 0.05 AND rv_final_b < 0.40 AND > arm_a + 0.02 (E1-learnable spatial schema
    enables BG gate commitment; MECH-216 schema-readout is the architectural lever).
  - **V3-EXQ-517b** (priority=7, any) -- MECH-302 relief-completion discriminative
    pair with longer episodes (steps_per_episode 150 -> 300; supersedes 517a,
    which produced mean_events=0.33/seed because 150 steps was geometrically
    too short for heal_rate=0.002/step). Conservative scientific-validity-
    preserving fix.
  - **V3-EXQ-523a** (priority=7, any) -- SD-029 reef comparator with adaptive
    graduation gate (rolling r2>=0.85 for 5 consecutive 200-step windows up to
    8000 steps) + degraded-measurement INCONCLUSIVE_UNDERTRAINED outcome path
    + per-claim evidence_direction overrides. Supersedes 523, which was
    undertrained at fixed 2000 steps (r2=0.57) and hardcoded
    evidence_direction='supports'.
  - **V3-EXQ-525** (priority=6, any) -- SD-003 attribution anchor (corrected
    output schema superseding EXQ-205, which had outcome:None pre-format-shift).
    Fresh run with current harm stream (post-SD-019/020/021 landings) + 4 seeds
    x 1 cond x 300 episodes. *(Note: completed FAIL 2026-05-05T22:04Z and is
    on the pending_review queue; still showing in the queue file as `pending`
    pending the runner-status sweep.)*
  - **V3-EXQ-526** (priority=6, any) -- Q-034 reef-enabled threshold sweep
    (supersedes 451, which FAILed C1/C2 on monostrategy action_entropy=0.0).
    Reef substrate creates two behavioral attractors (flee-to-reef vs forage);
    grid 8x8 -> 10x10; ISO timestamp + verdict-from-outcome + global episode
    counter all corrected; 2 seeds x 15 conditions x 60 episodes.
  - **V3-EXQ-527** (priority=6, any) -- MECH-112 goal-directed reef + SD-049
    Phase 2 identity encoder (fresh angle over EXQ-189; identity_classifier
    on food/water types + z_goal seeded from z_resource; GOAL_PRESENT vs
    GOAL_ABSENT 2 conds x 3 seeds x phased P0 100 warmup + P1 150
    freeze+train + 50 eval; 10x10 grid + 3 hazards + 8 resources +
    reef_enabled=True; world_obs_dim=325).

  The queue cleared after the SD-047 / SD-048 / SD-049-Phase-1 substrate-
  readiness sweep (V3-EXQ-509/511/513 on 2026-05-03) and the MECH-302
  comparator landing (V3-EXQ-515 PASS 7/7 on 2026-05-04), and was repopulated
  2026-05-05 with the reef-enrichment supersession wave. *(2026-05-04 narrative continues:)* next-up substrate work is the SD-049
  Phase 2 follow-on (z_resource encoder identity expansion + SD-032 consumer
  cascade + V3-EXQ-514 behavioural validation; tracked in
  `evidence/planning/substrate_queue.json` as SD-049-PHASE-2, ready=True,
  priority=2). The Q-040b combined-cluster successor design (V3-EXQ-490d
  staleness-into-gate factorial) and the V3-EXQ-495 MECH-163 V3-full-completion
  gate (next attempt after the 2026-05-03T23:56Z infrastructure-kill ERROR)
  remain drafted-but-unqueued pending design refinement. **Post-governance
  state (2026-05-02 walk on top of the 2026-04-30T20:54Z baseline):**
  SD-011 (`harm_stream.dual_nociceptive_streams`) promoted **provisional ->
  stable** (exp_conf=0.871, lit_conf=0.871, 7 exp + 23 lit, conflict_ratio=0.148
  under the 0.20 stable gate; 25 supports / 2 weakens / 3 mixed). SD-012
  (`environment.homeostatic_drive`) promoted **candidate -> provisional**
  (exp_conf=0.714, lit_conf=0.874, 5 exp + 16 lit, conflict_ratio=0.20;
  18 supports / 2 weakens / 1 mixed; substrate already implemented since
  2026-04-02 -- promotion is the registry catching up). Pre-fix indexer
  recommendations function had been counting scoring_excluded entries
  (diagnostic_probe / non_contributory / superseded) into the gate's exp_conf
  computation; bug fix committed 81de5101c yesterday surfaced spurious
  promotion recommendations for 5 claims with exp_conf=0 in the matrix.
  Post-fix + 2 promotions: pending_user 17 -> 15 -> 13. The four Phase 2
  cohort PASSes (V3-EXQ-499/500/501/502) plus V3-EXQ-490c FAIL remain
  reviewed (review_tracker.json reviewed_run_ids 1934 -> 1939 across the
  full 2026-04-30 / 2026-05-01 / 2026-05-02 review pulses). The 2026-04-28T23:04Z governance walk
  remains the prior anchor (10 decisions: 4 promotions MECH-266/267/268/SD-034
  candidate->provisional, SD-033b v3_pending true->false on the V3-EXQ-485
  substrate-landing PASS, 2 holds preserved on MECH-057b/MECH-263, 6 narrow-open
  Q-claim evidence_quality_note refreshes Q-025/26/27/28/31/40, V3-EXQ-485 (4)
  manifests with MECH-263 supports->non_contributory per claim_ids accuracy
  rule, replica supersession across EXQ-484/485/493 to avoid 5x/4x/3x
  over-weighting). Recently completed and out of queue:
  - **V3-EXQ-490c** (MECH-269b V_s gating + MECH-295 liking-bridge factorial;
    Q-040b behavioural sufficiency arm; supersedes V3-EXQ-490b's blocker
    diagnosis) **completed FAIL on Mac 2026-04-29T08:34Z (~2.6h)**, pending
    discussion as of this read. PRELIMINARY READING: the Q-040b strong
    reading (MECH-269b V_s gating ON + MECH-295 bridge ON jointly recover
    approach_commit) is NOT supported under matched smoke-threshold
    overrides. Successor V3-EXQ-490d planned to drop the smoke-threshold
    override and exercise the 2026-04-29T11:00Z MECH-284 staleness-into-gate
    wiring (use_vs_gate_staleness_lookup ON vs OFF at matched 0.4 thresholds)
    as the falsifiable test of the strong reading. Successor design
    in-progress; not yet queued.
  - **V3-EXQ-490b** (MECH-269b VsRolloutGate substrate-readiness probe;
    Q-040a precondition; supersedes V3-EXQ-490a) completed
    2026-04-28T21:09Z UNKNOWN/FAIL; governance 2026-04-28T23:04Z classified
    the run inconclusive (Q-040a effectively PASS at smoke; Q-040b FAIL
    points at MECH-295 bridge as remaining blocker -- which is now refuted
    or weakened by the V3-EXQ-490c FAIL).
  - **V3-EXQ-495** (MECH-163 V3 full-completion gate -- VTA / hippocampally-
    planned arm) **drafted but not yet queued**. THE discriminative test
    for the planned arm of MECH-163 dual goal-directed systems. All
    substrate prerequisites landed 2026-04-27: SD-039 anchor goal-snapshot
    payload (V3-EXQ-494 PASS), MECH-292 ranked ghost-goal bank
    (V3-EXQ-496 PASS), MECH-293 waking ghost-goal probe search
    (V3-EXQ-497 PASS). 3 conditions (HABIT value-flat proposals;
    PLANNED ghost-seeded proposals via MECH-293; ABLATED no goal anywhere)
    x 2 paradigms (A_DETOUR mid-episode blockage; B_NOVEL_CONTEXT
    cross-episode env swap) x 7 seeds. Acceptance C2 = PLANNED-HABIT
    benefit-post-block gap >= 0.30 in detour, >= 4/7 seeds. Estimated
    25h on Mac / 40h on ree-cloud-1; machine_affinity=any. Queueing
    decision pending; the 2026-04-28 governance cycle deferred queueing
    until the EXQ-490b/MECH-295 successor lands.
  Recently completed and out of queue:
  - **V3-EXQ-503 / EXP-0171a** (SD-017 sleep-phase **discriminative**
    pair; closes the SD-017 evidence gap left by V3-EXQ-500's
    diagnostic_probe scoring exclusion) **PASS 3/3 seeds on Mac
    2026-05-01T20:15Z (~0.32s)**, **reviewed 2026-05-01T20:40Z** (Phase 3
    cutover session). Three pre-registered metrics M1 cumulative_sws_writes /
    M2 ctxmem_state_change Frobenius norm / M3 cumulative_rem_rollouts; ARM_A
    full 4-phase sleep ON: sws_writes=24, ctxmem_delta=4.50-5.03,
    rem_rollouts=18 across seeds; ARM_B sws+rem off: zeros across. Governance
    impact: SD-017 quadrant flipped plausible_unproven -> confirmed_established;
    exp_conf 0.000 -> 0.775; **all four Phase 2 standard-gating cohort claims
    (MECH-094, SD-017, SD-035, MECH-062) are now confirmed_established**.
  - **V3-EXQ-490e** (MECH-295 seeding-strengthening successor; Q-040b
    BASELINE-vs-RELAXED on activation-floor + drive_to_liking_gain with
    MECH-295 bridge ON in both arms) **FAIL on Mac 2026-05-01T03:19Z (~6h)**,
    **reviewed 2026-04-30T20:54Z** as a forward-looking successor to V3-EXQ-490c.
    Floor-relaxation alone does not recover approach_commit; combined with the
    V3-EXQ-490c FAIL, the Q-040b strong reading remains weakened.
  - **V3-EXQ-499** (MECH-094 hypothesis-tag write-gate discriminative pair --
    EXP-0170 from Phase 2 claim-type gating cohort) **PASS 3/3 seeds on Mac
    2026-04-29T18:47Z (~0.08s)**, **reviewed 2026-04-30T20:54Z**. ARM_A:
    contam=0.000, confab=0.000, MI=0.693 (perfect log(2)). ARM_B:
    contam=1.000, confab=0.640, MI=0.000. First standard-gating
    experimental evidence for MECH-094 -- prior 9 entries excluded
    (V3-EXQ-465 family diagnostic_probe; V3-EXQ-140 non_contributory).
    MECH-094 quadrant flipped plausible_unproven -> confirmed_established.
  - **V3-EXQ-500** (SD-017 sleep-phase substrate-readiness diagnostic --
    EXP-0171) **PASS on Mac 2026-04-29T19:27Z**, **reviewed 2026-04-30T20:54Z**
    (diagnostic_probe scoring exclusion -- evidence gap closed by V3-EXQ-503).
    Substrate-readiness probe; nine prior FAIL/non_contributory entries
    on SD-017 led to a fresh substrate-readiness gate before any
    behavioural retest.
  - **V3-EXQ-501** (SD-035 amygdala analog vs binary -- EXP-0172) **PASS
    on Mac 2026-04-29T19:27Z**, **reviewed 2026-04-30T20:54Z**. Discriminates
    SD-035's amygdala-analog substrate from a degenerate binary toggle.
  - **V3-EXQ-502** (MECH-062 tri-loop gate coordination -- EXP-0173)
    **PASS on Mac 2026-04-29T19:27Z**, **reviewed 2026-04-30T20:54Z**. Truly
    fresh-start MECH-062 evidence (zero priors); first tri-loop
    coordination test on the V3 substrate.
  - **V3-EXQ-498** (OCD Layer 1 closure-threshold sweep) -- FAIL outcome
    under UNKNOWN result code 2026-04-28T20:29Z, governance reclassified
    non_contributory; rules out Layer 1, licenses Layer 2 (MECH-290
    ablation) or Layer 3 (SD-046 multi-slot GoalState pull-forward).
  - **V3-EXQ-418f / 418g / 418h** -- SD-016 diagnostic probe + selectivity-
    first 4-arm + env-entropy precondition probe. 418f localised the
    failure to query selectivity (not slot content); 418g substrate-side
    fixes (learnable temperature + entropy regulariser) work as designed
    (C1+C2+C3 PASS), but C4+C5 FAIL with action_class_entropy~1.1e-10
    IDENTICALLY across all four arms because z_world is near-constant
    across the batch (cos~0.998 in 418f). 418h env-entropy probe
    confirmed: SD-023 landmarks-on alone does NOT supply enough
    cross-context z_world variance under the current env. SD-016
    parked pending env-enrichment precondition (substrate_queue.json
    status parked_pending_env_entropy_precondition); validation_experiment
    rerouted to EXQ-418h family pending broader env scoping.
- **Current bottleneck (2026-05-07):** the **32-item pending_review accumulation
  + active governance walk** is the immediate gate -- the 2026-05-06 in-flight
  wave (V3-EXQ-244a/525/526/527/528-535) completed faster than the prior cycle
  could review, and the active governance cycle (claim
  governance-2026-05-06T2156Z, ~3h old at this nightly read) holds the
  REE_assembly governance files. Until the walk lands its decisions on
  claims.yaml + review_tracker.json, the registry view of the recent
  experimental wave is fluid. Underneath that, the **monostrategy / reef-
  recovery thread** remains the dominant scientific bottleneck -- whether
  reef substrate alone is sufficient to recover behavioural acceptance under
  the SD-029 / SD-032b / SD-032c / Q-034 / MECH-112 / ARC-016 / MECH-257 /
  SD-049 Phase 2 cluster, or whether downstream calibration is also required.
  The 2026-05-06 wave produced two PASSes that probably flip MECH-102 and
  SD-016 cue_terrain quadrants and resolve a chunk of the SD-029 retry
  ladder, plus several FAILs that point recovery effort at SD-005 / SD-015 /
  Q-034 / MECH-112 / MECH-098 specifically; final accounting waits on the
  walk.

- **Current bottleneck (2026-05-11):** the **ARC-062 Phase 3 wiring pass
  + ARC-064/ARC-065/ARC-066 child-MECH cluster behavioural validation**
  is the immediate gate. Phase 3 wiring (gated_policy bias-head into E3
  optimizer + discriminator output threaded into SD-033a
  `LateralPFCAnalog.update()` source vector) was attempted in V3-EXQ-543b
  on Mac (~4h, FAIL) -- design needs refinement before another attempt;
  commitment_closure_plan.md GAP-1 remains blocked. The four ARC-064/
  ARC-065/ARC-066 child substrates (MECH-313 noise floor + MECH-314
  structured curiosity bonus + MECH-319 simulation-mode rule-write gate
  + MECH-320 tonic vigor coupling) all landed substrate-readiness PASS
  today; the next move is the Q-043 / Q-044 / Q-045 cross-cohort
  behavioural ablation (relative-weight calibration + sub-flavour
  independence + MECH-313 vs MECH-260 collapse falsifier) on the
  V3-EXQ-543c-successor admit_writes=True falsifier substrate.
  Underneath those, the **ARC-070/071 R6 SAFETY-CRITICAL governance
  decision** (MECH-094 hypothesis_tag strict-vs-relaxed for chunking
  write path, escalated by the ARC-071 lit-pull synthesis) and the
  **non_deficit_action_drives + policy_primitive_granularity child-MECH
  design cycles** are the architectural-side gates that flow from
  today's two cluster registrations. The monostrategy / reef-recovery
  thread continues to be the underlying scientific bottleneck pending
  ARC-062 Phase 3 + the rule-apprehension cluster's behavioural
  validation cohort.

- **Current bottleneck (2026-05-09):** the **sleep-substrate Phase 1
  validation + MECH-307 conjunction-architecture cluster** is the immediate
  gate. Phase 1 of `sleep_substrate_plan.md` (GAP-1 MECH-204 precision
  recalibration consumer) landed end-to-end 2026-05-08 with full contract
  coverage; V3-EXQ-541 ran 2026-05-08T23:43Z and produced a runner-level
  FAIL outcome with the result_summary recording "verdict: PASS" -- the
  next governance walk needs to reconcile the runner sentinel vs the
  in-script verdict and decide whether the linear-interpolation step=0.1
  default is calibrated correctly for the C1/C2/C3 acceptance grid (or
  whether EXP-0171 step-size sweep needs to fire ahead of the C5
  behavioural arm). The MECH-307 cluster's substrate side (signed
  VALENCE_SURPRISE + MECH-216 z_beta coupling + anticipatory
  VALENCE_LIKING write + write-at-predicted-location) cleared via
  V3-EXQ-539 substrate counters but the behavioural recovery FAILed
  because the legacy MECH-295 cue path doesn't read the conjunction
  signal; V3-EXQ-540 3-arm gap decomposition (ARM_0_off / ARM_1_signed_pe /
  ARM_2_full) is in flight on Mac with the consumer-conjunction Path B
  bridge ON in all arms so ARM_2 has the downstream consumer EXQ-539
  lacked. Underneath those, the **monostrategy / reef-recovery thread**
  remains the dominant scientific bottleneck, now reframed by the
  2026-05-08 rule-apprehension cluster registration (MECH-309
  monomodal-collapse-as-equilibrium-without-rule-apprehender; ARC-062
  V3 weak rule-apprehension reading; ARC-063 V4 strong reading) --
  whether the missing primitive is a gated-policy architectural slot
  (V3 weak reading testable via ARM_0 baseline / ARM_1 reef-only /
  ARM_2 reef + gated-policy stub three-arm discriminative pair, EXP-0171)
  or a distributed CandidateRule field with tolerance gate (V4 strong
  reading). Plan-of-record sequencing: sleep_substrate_plan (GAP-1
  in-progress) + commitment_closure_plan (GAP-1 SD-033a bias-head
  training next) + self_attribution_plan (GAP-1 V3-EXQ-445h forensic
  read independent of substrate) + goal_pipeline_plan (GAP-1 MECH-307
  conjunction architecture, gates Phase 2-6) all share the StepHarness
  audit follow-up.

- **Current bottleneck (2026-05-08):** the **bug-fix retest cohort + MECH-295
  goal-seeding diagnostic** is the immediate gate. Three reef-superseding
  bug-fix retests (V3-EXQ-433f / 483b / 514f) are running concurrently
  across Mac / ree-cloud-1 / ree-cloud-2; whether the BreathOscillator +
  saturating-step-idx fix recovers behavioural acceptance under SD-029 /
  SD-037 / SD-049 Phase 2 is the live question. Underneath that, the
  EXQ-536 z_goal_active_fraction=0.0 finding ("goal seeding is
  upstream-blocked OR commit chain is inert even with seeded z_goal") is
  the load-bearing diagnostic queued ahead -- V3-EXQ-536a / 536b will
  dispatch root cause as soon as a runner picks them up. The
  V3-EXQ-535a SD-029 P3 eval-fix attempt was reclassified non_contributory
  in the 2026-05-08T00:15Z multi-sense audit (computes two-pass cf_gap,
  not the single-pass residual the SD-029 / MECH-256 spec requires);
  V3-EXQ-537 single-pass-residual successor is queued. The pending_review
  queue closed at 0 items going into the next governance cycle. The
  monostrategy / reef-recovery thread remains the dominant underlying
  scientific bottleneck.

  **Historical 2026-05-06 narrative:** **monostrategy** was the dominant active
  blocker across the SD-029 / SD-032b / SD-032c / SD-049-Phase-2 / Q-034 /
  MECH-112 / MECH-257 / ARC-016 / SD-016 cluster -- under monomodal policy in
  legacy 8x8/0-hazard env, behavioural acceptance criteria that depend on
  diverse self-vs-env event distributions cannot be measured. The
  **2026-05-04 reef enrichment substrate (SD-050 reef)** is the first-line
  unblocker: corner-adjacent safe zones + food-attracted hazard drift create
  two behavioral attractors (flee-to-reef vs forage), restoring env-entropy
  precondition. V3-EXQ-522 PASS confirmed zone_transitions=48.9 between
  attractors. Reef-superseding versions of every monostrategy-blocked
  predecessor have been queued (V3-EXQ-433e / 445e/f/g / 325f / 452a / 454a /
  514c / 526 / 527); the next governance walks gate on whether reef recovers
  the affected behavioural acceptance criteria. Three FAILs already on
  pending_review (V3-EXQ-454a ARC-016 reef, V3-EXQ-452a ARC-033/MECH-257 reef,
  V3-EXQ-525 SD-003 attribution anchor) -- whether reef is sufficient or
  whether downstream calibration is required is the active question. The
  Q-040b combined-cluster successor design remains
  the open thread on the substrate-recovery side (V3-EXQ-490c + V3-EXQ-490e
  FAILs ruled out the matched-smoke-threshold and floor-relaxation paths;
  V3-EXQ-490d staleness-into-gate factorial drafted but unqueued pending
  design refinement). The headline V3-full-completion-gate run V3-EXQ-495
  (MECH-163 hippocampally-planned arm) suffered an infrastructure SIGTERM
  on ree-cloud-1 at ~4h of ~40h on 2026-05-03 (governance-deferred until
  the Q-040b cluster-successor lands; ree-cloud-1 capacity / supervised
  re-run plan TBD). **2026-05-03 substrate-enrichment wave landed
  three V3 H/M-priority substrates in one day:** SD-047
  (`environment.multi_source_dynamics`) implemented end-to-end and V3-EXQ-509
  PASS 7/7 (substrate-ceiling unblock for MECH-095 TPJ agency-detection);
  SD-048 (`body.interoceptive_noise_dynamics`) implemented as Level 2
  counterpart at the body-state layer with V3-EXQ-511 6/7 (C1b sub-threshold
  at scale=0.25 -- governance-accepted as calibration-off-but-architecture-
  holds per SD-doc interpretation grid; ARC-058 / ARC-033 comparator-gap
  behavioural successor V3-EXQ-512 deferred); SD-049 Phase 1
  (`environment.multi_resource_heterogeneity`) implemented env-only with
  V3-EXQ-513 PASS 13/13 incl curriculum gates (Phase 2 z_resource encoder
  identity expansion + V3-EXQ-514 behavioural validation tracked in
  substrate_queue.json as ready=True / priority=2). **2026-05-03 lit-pull
  wave** added MECH-302 + MECH-303 candidate registration (relief-completion
  reuses goal-completion pipeline + safety-cue parallel predictive substrate;
  hybrid-leaning-Model-1 verdict from 8-entry pre-registration lit-pull) and
  the commit-boundary-belief-lock pre-registration brief (7 entries; verdict
  REGISTER WITH MODIFICATION as two-mechanism cluster MECH-X commit_boundary
  + MECH-Y attribution_rigidity_setpoint; awaiting user signoff). **2026-05-04
  substrate landing:** SD-050 / MECH-302 suffering-derivative comparator
  implemented as the first downstream consumer of the relief-completion
  lit-pull verdict (SufferingDerivativeComparator on z_harm_a stream; reuses
  MECH-057a beta-gate release + MECH-094 categorical VALENCE_LIKING tag write;
  V3-EXQ-515 comparator logic PASS 7/7; V3-EXQ-516 agent-loop integration
  diagnostic queued). **2026-05-03 governance walks:** the 2026-05-03T02:38Z
  walk promoted ARC-026 candidate -> provisional (two PASS pillars EXQ-232 +
  EXQ-507) and MECH-093 candidate -> provisional (three PASS pillars
  097b/396b/505 substrate-level dissociation); the 2026-05-03T23:56Z walk
  cleared the three new substrate-readiness diagnostics (EXQ-509/511/513) +
  deferred V3-EXQ-495 ERROR -- pending_user / pending_review both 0 after
  both walks. **Phase 3 wave 2 governance schema landed 2026-05-02:** the
  `epistemic_category` field is now formalised on `claims.yaml` with 7
  canonical values (`standard`, `substrate_coherence`, `answer_state`,
  `substrate_ceiling`, `substrate_conditional`, `derivational`, `out_of_domain`);
  resolver in `build_experiment_indexes.py` (`_resolve_epistemic_category`)
  with explicit-wins-over-inferred semantics; `_recommendation_for_claim`
  dispatches via the resolver so only `standard` runs exp_conf gates and
  `narrow_open_question` fires only for `answer_state`; warn-only validator
  in `scripts/validate_claims.py`. 13 annotated claims backfilled with
  explicit categories (MECH-095 / MECH-102 -> substrate_ceiling; Q-025/26/27
  -> derivational; Q-028/029 -> substrate_ceiling; Q-030 -> standard;
  Q-031/032 -> out_of_domain; Q-037/038/039 -> substrate_conditional);
  pending decision queue dropped 16 -> 4 items (the remaining 4 are
  pre-existing V3-pending holds + SD-023 conflict alert). Two new
  planning artifacts landed alongside: `docs/architecture/substrate_roadmap.md`
  (10 in-flight V3 enrichments + 7 outstanding -- 3 H-priority: foreclosure
  primitives, multi-resource heterogeneity, long-horizon dynamics; 2
  M-priority; 2 L-priority -- mapped to the claims each would unblock and
  the SD candidate that would register it) and `docs/architecture/v4_spec.md`
  (4 V4 primitive additions V4-1 multi-agent ecology, V4-2 self-model
  integration DR-10..DR-14, V4-3 long-horizon + persistent identity, V4-4
  richer action repertoire; ~12 V4-bound claims mapped; Phase A draft only,
  Phase B onwards gated on V3 full completion (MECH-163 PASS) + governance
  authorization). The 2026-05-02 weekend lit-pull added 9 PubMed-sourced
  entries across Q-037 (psychosis substrate dissociability between
  MECH-094 / MECH-222 / MECH-065+223; lit_conf 0.0 -> 0.823), Q-038 (D_V
  temporal-depth representational status; lit_conf 0.0 -> 0.795), Q-039
  (neuromodulators / control-plane vars regulating TCL integration window;
  lit_conf 0.0 -> 0.84) -- literature_entries 1113 -> 1122. The
  2026-05-02T18:06Z `/diagnose-errors` walk cleared 3 obsolete unaddressed-
  error entries (V3-EXQ-008 obsolete March SD-003 era, V3-ONBOARD-smoke-
  EWIN-PC + V3-ONBOARD-smoke-ree-cloud-1 contributor onboarding for
  inactive machines), confirming the remaining 2 NotImplementedError
  sentinels (V3-EXQ-455a and V3-EXQ-449c) are intentional cascade-gate
  waits and not bugs. The next-up falsification of the Q-040b strong reading is the
  staleness-into-gate test (use_vs_gate_staleness_lookup OFF vs ON at
  matched 0.4 thresholds via the 2026-04-29T11:00Z MECH-284 wiring) --
  V3-EXQ-490d remains drafted but unqueued pending design refinement
  in light of the 490e FAIL. V3-EXQ-495 (V3 full-completion gate
  / MECH-163 hippocampally-planned arm) remains the headline first-paper-gate
  run -- all three substrate prerequisites cleared 2026-04-27, but queueing
  decision deferred until the Q-040b cluster-successor outcome lands. C2
  PLANNED-HABIT benefit-post-block gap is the V3-full-completion gate metric.
  The EXQ-483 wired-but-inert pattern remains the open behavioural-recovery
  thread for the SD-037 / MECH-269b / MECH-295 cluster: V3-EXQ-484/485/493
  all cleared as substrate-readiness PASSes (post run_id naming-bug fix),
  validating SD-033a / SD-033b / MECH-295 substrate landings; behavioural
  recovery of approach_commit awaits the combined-cluster successor EXQ.
  SD-016 is parked pending env-entropy precondition (EXQ-418f/g/h
  established the cue-context machinery works as designed but the env
  doesn't supply cross-context z_world variance; substrate_queue.json
  status parked_pending_env_entropy_precondition). OCD Layer 1
  hypothesis disconfirmed (V3-EXQ-498 FAIL reclassified non_contributory
  in 2026-04-28 governance), escalating to Layer 2 (MECH-290 ablation
  diagnostic) or Layer 3 (SD-046 multi-slot GoalState pull-forward). Open promotion blockers
  documented in claims.yaml: MECH-294 within-cycle-vs-cross-cycle binding
  (Kay 2020 challenge); MECH-295 strong-vs-weak liking-bridge necessity (weak
  reading committed provisionally). SD-032 cluster behavioural
  follow-through remains the
  primary first-paper-gate blocker for the cingulate track -- V3-EXQ-445a
  (full-pipeline fix for the monostrategy + terrain-prior inversion observed in
  V3-EXQ-445), V3-EXQ-445b (epsilon-greedy exploration variant), and V3-EXQ-445c
  (14x14 env variant) have all since FAILed. V3-EXQ-325d FAILed with zero
  between-arm gradient, leaving the SD-032c AIC-analog descending-modulation
  falsification signature open. V3-EXQ-454 FAILed on ARC-016 adaptive
  commitment_threshold under the 2026-04-20 harness. The SD-003 successor track
  was re-opened by V3-EXQ-433c (curriculum-ON + scripted agent-caused
  elicitation, superseded V3-EXQ-433b after the agent_caused_hazard r2=0.0
  trials-shortage failure was diagnosed as a sufficiency issue rather than a
  MECH-256 architectural failure); V3-EXQ-433c reviewed in the 2026-04-24
  cowork-a wave. ARC-007 path-memory track has cleared its most recent claim.
  SD-033a substrate landed 2026-04-20 (V3-EXQ-456 PASS on all five sub-tests); SD-034
  governance closure-operator / MECH-267 mode-conditioned hippocampal / MECH-268 dACC
  conflict-saturation all landed 2026-04-20/21 with landing-diagnostic smokes PASS
  (V3-EXQ-460 / 462 / 463 / 465 / 466 / 468). SD-035 amygdala BLA + CeA peer modules
  landed 2026-04-21 with V3-EXQ-473 (CeA mode-prior) and V3-EXQ-474 (BLA encoding+remap)
  PASS. MECH-266 asymmetric per-mode hysteresis on SalienceCoordinator landed
  2026-04-21 with the V3-EXQ-464 / V3-EXQ-467 landing diagnostics smoke-PASS. MECH-269
  / MECH-270 / MECH-271 / MECH-272 / MECH-273 / MECH-274 and MECH-275 / MECH-276 /
  MECH-277 / MECH-278 / ARC-059 registered 2026-04-21 (anchor-vs-probe +
  sleep/waking state-gated routing + scientist-agent developmental ordering cluster);
  MECH-272 / MECH-273 / MECH-275 / MECH-285 substrate landings completed 2026-04-25
  via the Sleep Aggregation Cluster (Phases A-E -- scaffolding, replay sampler,
  routing gate, Bayesian aggregator, self-model writeback). MECH-270 / MECH-271 /
  MECH-274 / MECH-276 / MECH-277 / MECH-278 / ARC-059 remain v3_pending.
  SD-016 cue_action_proj forward-path is now under Path 1 (auxiliary
  diversification loss) -- V3-EXQ-418d FAILed across all 4 write-path arms with
  attn_entropy=ln(16) regardless of write mode, confirming read+write gradients
  alone cannot break ContextMemory slot symmetry. SD-016 Path 1 substrate change
  landed 2026-04-25 (mean squared off-diagonal cosine similarity over normalized
  slot vectors); V3-EXQ-418e queued as the 4-arm verification. SD-037 broadcast
  override regulator (orexin-analog) landed 2026-04-25 as the third regulatory
  layer; V3-EXQ-483a is the in-flight 4-arm validation. The three-layer
  regression suite (preflight / contracts / deferred changed) reached **150/150
  PASS (143 contracts + 7 preflight)** with all flags OFF after the 2026-04-25
  wave added contracts for SD-037, Sleep Phase A scaffolding (8 tests),
  MECH-285 SleepReplaySampler Phase B (10 tests), MECH-272 RoutingGate Phase C
  (10 tests), MECH-275 BayesianAggregator Phase D (10 tests), and MECH-273
  SelfModelAggregator Phase E (10 tests). The 2026-04-26 wave (SD-039 substrate
  + SD-033b OFC + MECH-269b + MECH-295) extended the contracts to **164/164 +
  7/7 preflight PASS** with all flags OFF; the 2026-04-27 wave (SD-039 module-
  level write-site population layer + MECH-292 ranked ghost-goal bank +
  MECH-293 waking ghost-goal probe search) extended the suite further to
  **183/183 contracts + 7/7 preflight PASS**; and the 2026-04-29 MECH-269b +
  MECH-284 staleness-into-gate wiring added 8 new contracts for
  use_vs_gate_staleness_lookup, bringing the suite to **191/191 preflight +
  contracts PASS** with all flags OFF -- bit-identical-when-OFF guarantee
  preserved across the entire wave. Explorer preflight badge + pre-commit
  contracts hook (PR 5) remain live. **Pending review queue (per
  pending_review.md regenerated 2026-05-04T01:17Z) lists 2 items, both
  PASS:** V3-EXQ-512 (SD-048 ARC-058 / ARC-033 comparator-gap behavioural
  successor; deferred at the 2026-05-03T23:56Z governance walk but ran
  on Mac 2026-05-04T00:57Z and PASSed) and V3-EXQ-515 (MECH-302
  suffering-derivative comparator substrate readiness; PASS
  indexer-surfaced 2026-05-04T01:17Z). The 2026-05-03 governance walks
  cleared the substrate-readiness diagnostics V3-EXQ-509/511/513
  (SD-047/SD-048/SD-049 Phase 1) and the V3-EXQ-490e FAIL on Q-040;
  the four Phase 2 cohort PASSes (V3-EXQ-499/500/501/502) and the
  V3-EXQ-490c FAIL were reviewed in the 2026-04-30T20:54Z walk;
  V3-EXQ-503 / EXP-0171a SD-017 discriminative pair PASS was reviewed
  in the 2026-05-01T20:40Z Phase 3 cutover session. Phase 2 cohort is now complete on both the
  experimental-evidence and governance sides: all four standard-gating
  MECH/SD claims (MECH-094, SD-017, SD-035, MECH-062) are
  confirmed_established under the new Phase 3 production gates.
  **Lit/Exp Decoupling (Option E) Phase 3 cutover landed 2026-05-01:**
  promotion / demotion gate logic now reads `experimental_confidence`
  directly; `decision_criteria.v1.yaml` thresholds renamed
  (`min_overall_confidence` -> `min_exp_conf`, `max_overall_confidence` ->
  `max_exp_conf`; legacy keys still accepted for one cycle via the
  `_t(d, new_key, legacy_key, default)` helper); `planning_criteria.v1.yaml`
  retired `low_overall_confidence` for `low_exp_conf: 0.55` +
  `lit_only_above_cap: 0.50`; claim-type evidence gating brought
  INTO production (`substrate_coherence` for ARC + universal invariant /
  `answer_state` for open_question / `standard` for everything else);
  diff against pre-cutover snapshot of `promotion_demotion_recommendations.md`:
  +2 actionable demotion recommendations surfaced (MECH-095 + MECH-102,
  both `mechanism_hypothesis` whose lit_conf was masking insufficient
  exp_conf under the legacy blend), 0 prior recommendations lost.
  Quadrant distribution after cutover: 194 plausible_unproven, 68
  confirmed_established, 3 speculative, 1 novel_discovery. **Duplicate-
  manifest sweep landed 2026-05-01:** Phase 1 (8 Tier-1 clusters span <2h)
  + Phase 2 (10 Tier-2 clusters span 2-24h, 6 auto-superseded + 4 flagged
  for manual review at `evidence/experiments/dedup_review/phase2_manual_review.md`)
  + Phase 3 (10 Tier-3 clusters traced to 2026-03-27..30 runner regex
  bug fixed in commit 071f1fc; oldest emission kept canonical, later
  replays superseded) + per-run manifest mirroring for the indexer
  (the indexer reads `runs/<run_id>/manifest.json` not the flat-JSON
  files where the Phase 1/2/3 edits originally landed; 31 per-run
  manifests re-edited; 5 of 31 EXQ-232 ARC-026 manifests required
  re-application because their flat JSONs were wiped by runner
  auto-sync) + Phase 5 in-memory dedup guard added to
  `build_experiment_indexes.py:_detect_and_mark_duplicate_emissions()`
  with back-off when manual supersessions present. **Cumulative cleanup:
  31 phantom evidence entries across 25 clusters.** Smoke test 4s,
  31 dups caught across 13 experiment_types, with 4 v3 clusters worth
  manual review (074f, 133, 223, 484). The 2026-04-28T23:04Z governance
  cycle remains the prior anchor (10 user-approved decisions including
  4 promotions MECH-266/267/268/SD-034 candidate->provisional; SD-033b
  v3_pending true->false; 2 holds preserved on MECH-057b + MECH-263;
  6 narrow-open Q-claim evidence_quality_note refreshes for
  Q-025/26/27/28/31/40). Pipeline (post-2026-05-01 cutover):
  validate_claims --strict OK 68 invariants; claims.json rebuilt 571;
  index 920 runs / 487 types; pending_review 1+0 (V3-EXQ-490e FAIL).
  Next governance cycle gates on the V3-EXQ-490e FAIL discussion +
  the Q-040b cluster-successor (V3-EXQ-490d staleness-into-gate)
  + V3-EXQ-495 outcomes.**

### V3 / V4 Scope Boundary

**Sleep rescope 2026-04-20:** all sleep-related substrates are V3 in-scope. V4 is reserved
for social systems ("sharing joys and sorrows"). See ree-v3/CLAUDE.md "V3 scope (full
sleep mechanisms)" for the authoritative list.

**V3 scope (waking mechanisms):**
- Volatility interrupt / LC-NE analog (MECH-104)
- BG hysteresis and outcome-valence modulation (MECH-106)
- Hippocampal->BG completion coupling (MECH-105, ARC-028) — IMPLEMENTED 2026-04-04
- Beta gate committed->uncommitted dynamics (MECH-090)
- Trajectory completion signal from HippocampalModule (ARC-028) — IMPLEMENTED 2026-04-04
- Valence vector node recording: 4-component V=[wanting, liking, harm_discriminative,
  surprise] in RBFLayer + ResidueField (SD-014) — IMPLEMENTED 2026-04-04

**V3 scope (sleep mechanisms, rescoped from V4 2026-04-20):**
- Full SWR consolidation pipeline (MECH-121)
- Slow-wave sleep prediction error baseline reset
- Sleep-dependent recalibration of commit thresholds (full SR-3 / SR-4)
- Theta-gamma coupling during offline replay for memory formation
- Lansink et al. (2009) hippocampus-leads-striatum replay -- V3 evidence
- Phase boundary triggers (SR-4: sws_consolidation_complete -> REM transition)
- MECH-261 predicate enrichment on the SD-032a registry (carrier-rhythm function ->
  multi-factor admission conjunction)
- Per-mode write-gate weight refinement as new mode-gating literature lands
- MECH-272 state-gated anchor/probe routing (waking=anchor-dominant, sleep=probe-dominant)
  -- IMPLEMENTED 2026-04-25 as Sleep Aggregation Cluster Phase C (RoutingGate)
- MECH-273 sleep-dependent aggregation of SD-003 single-episode self-attribution into
  stable self-model -- IMPLEMENTED 2026-04-25 as Sleep Aggregation Cluster Phase E
  (SelfModelAggregator + StalenessAccumulator.partial_decay)
- MECH-275 sleep-phase general Bayesian aggregation -- IMPLEMENTED 2026-04-25 as
  Sleep Aggregation Cluster Phase D (BayesianAggregator with conjugate Gaussian
  posterior + probe-channel-gated update + snapshot+decay contract)
- MECH-285 sleep replay sampler -- IMPLEMENTED 2026-04-25 as Sleep Aggregation
  Cluster Phase B (SleepReplaySampler reading frozen StalenessAccumulator snapshot
  with softmax-prioritised seeds from AnchorSet.all_with_dual_trace())
- MECH-276 scientist-agent counterfactual-backed attribution, MECH-277
  motor-experimentation action-space discovery, MECH-278 experimental-action
  object-schema formation, ARC-059 three-stage developmental ordering
  self->objects->others (refines ARC-019) -- all registered 2026-04-21 (v3_pending)
- MECH-269 hippocampal replay start-state selection (anchor vs probe) -- IMPLEMENTED
  via Phase 1 + Phase 2 ii AnchorSet + Phase 2 iii T4 per-region V_s + Phase 3
  online hysteresis swap; consumed by Sleep Phase B replay sampler.
- MECH-270 ephaptic-field substrate for per-stream verisimilitude readout, MECH-271
  MECH-094 as routing signature (anchored->PFC/E1, probe->BLA/NAc) -- registered
  2026-04-21 (v3_pending)

**V4 scope (social systems, rescoped 2026-04-20):**
- Representing other agents, their z_self / z_harm_a, and trajectories that affect
  another agent's state over time. Gated on V3 full-completion-gate (MECH-163
  hippocampal multi-step trajectory planning).
- Self-navigation via z_self hippocampal trajectories (ARC-031): gated on EXQ-075 and
  EXQ-076 PASS results and Q-022 dissociation test before any implementation
- MECH-274 other-attribution sleep-dependent aggregation (ARC-010 empathy /
  mirror-modelling; implementation_phase: v4)
- SD-033e frontopolar parallel-goal deliberation (V4-reserved stub landed in
  ree_core/pfc/ 2026-04-21 with NotImplementedError guard)

**MECH-124 diagnostic (V4 risk indicator):** When reviewing wanting/liking and E1 goal-conditioning
results (EXQ-074 series, EXQ-076 series), check whether z_goal salience is competitive with harm
salience. If not, this is an early risk indicator for consolidation-mediated option-space
contraction in V4.

### Q-020 Decision

Q-020 adjudicated 2026-03-16: **ARC-007 strict.** HippocampalModule generates value-flat
proposals. Terrain sensitivity is a consequence of navigating residue-shaped z_world, not a
separate hippocampal value computation.

---

## 1. Purpose

V2 proved four core architectural separation claims (MECH-059, -056, -060, -061 all PASS) and ran a complete SD-003 self-attribution experiment series (EXQ-027, EXQ-028 both FAIL), revealing a precise architectural gap: `z_gamma` conflates the agent's own body state with its world footprint. This means:

- SD-003 causal attribution requires a split latent (SD-005)
- Hippocampal planning horizon is bounded by raw z_gamma dimensionality (SD-004)
- The three BG loops cannot be cleanly separated in a single shared latent (SD-006 + SD-005)
- Seven claims cannot be tested until V3 substrate exists (ARC-007, ARC-016, ARC-018, MECH-025, MECH-033, Q-007 — all `hold_pending_v3_substrate` in governance)

ree-v3 implements the three co-designed substrate changes (SD-004, SD-005, SD-006) needed to open those claims to genuine experimental testing.

---

## 2. What V2 Got Right (Preserve These)

These V2 results are genuine and must not regress:

| Claim | V2 result | What it means for V3 |
|---|---|---|
| MECH-059 | PASS | E1 precision and E3 confidence are structurally independent — preserve two-optimizer design |
| MECH-056 | PASS | Residue accumulates along trajectory, not only at endpoint — preserve incremental residue updates |
| MECH-060 | PASS | Write-locus separation between pre/post-commit channels — preserve commit boundary logic |
| MECH-061 | PASS | Commit boundary correctly separates error channels — preserve error routing at commit |
| SD-003 prereq | PASS | CausalGridWorld provides valid ground truth for `transition_type` — reuse and extend |

The V2 module tree (`ree_core/latent/`, `ree_core/predictors/`, `ree_core/hippocampal/`, `ree_core/trajectory/`, `ree_core/residue/`, `ree_core/environment/`) is the right organisational shape. V3 restructures internals, not the package layout.

---

## 3. Three Core Design Decisions

### SD-004 — Action Objects as Hippocampal Map Backbone

**Problem:** HippocampalModule currently navigates raw `z_gamma` state space via CEM. This caps the planning horizon — CEM must operate at full latent dimensionality, and the map has no compressed representation of action consequences.

**Change:** E2 additionally produces *action objects*: `o_t = E2.action_object(z_world_t, a_t)` — a compressed representation of the world-effect of action `a_t` from state `z_world_t`. The hippocampal map is built in action-object space `O`, not raw `z_world` space.

**Interface contract:**
```python
# E2 forward pass produces TWO outputs:
z_self_next = E2(z_self_t, a_t)           # motor-sensory prediction (SD-005)
o_t = E2.action_object(z_world_t, a_t)    # world-effect action object (SD-004)
```

**HippocampalModule** then proposes trajectories as sequences of action objects `[o_t, o_{t+1}, …]`, navigating the compressed world-effect manifold. Planning horizon extends because action-object space is lower-dimensional and semantically grounded.

**Co-dependency with SD-005:** action objects encode `z_world_t → z_world_{t+1}` under `a_t`, which requires `z_world` to exist as a separate channel.

---

### SD-005 — Self/World Latent Split

**Problem:** `z_gamma` conflates proprioceptive/interoceptive self-state (`z_self`) with exteroceptive world-state (`z_world`). This prevents:
- Clean moral attribution (residue should track world-delta, not self-delta)
- Genuine MECH-069 incommensurability (signals are partially correlated in z_gamma)
- Correct SD-003 V3 attribution (`world_delta` requires z_world to exist)

**Change:** Split the latent encoder into two streams:

```
observation → encoder → {
    body-state channels (proprioception, interoception) → z_self  [E2 domain]
    world-state channels (exteroception, env observations) → z_world  [E3/Hippocampus/ResidueField domain]
}
```

**Module responsibilities after split:**

| Module | Latent domain | Error signal | Heartbeat rate |
|---|---|---|---|
| E1 | z_self + z_world (read-only sensory prior) | Sensory prediction error | E1 rate (fast, every frame) |
| E2 | z_self | Motor-sensory prediction error | E2 rate (motor command rate) |
| E3 complex | z_world | Harm + goal error | E3 rate (deliberation rate, slowest) |
| HippocampalModule | action-object space O (indexed over z_world) | Map consolidation (replay) | E3 rate |
| ResidueField | z_world | — (accumulates world_delta) | E3 rate |

**SD-003 V3 attribution pipeline:**
```
# Step 1: E2 provides z_world dynamics
z_world_actual = E2.world_forward(z_world_t, a_actual)
z_world_cf     = E2.world_forward(z_world_t, a_cf)

# Step 2: E3 evaluates harm of each projected world-state
harm_actual = E3.harm_eval(z_world_actual)
harm_cf     = E3.harm_eval(z_world_cf)

# Step 3: causal signature
causal_signature = harm_actual - harm_cf
world_delta      = ||z_world_actual - z_world_cf||
```

Residue accumulates on `world_delta`, not on `causal_delta(z_gamma)` as in V2.

**Note:** `z_beta` (affective latent, arousal/valence) remains shared and continues to modulate E3 precision and (per MECH-093) E3 heartbeat rate. It is NOT split — affective state integrates self and world signals.

---

### SD-006 — Asynchronous Multi-Rate Loop Execution

**Problem:** V1 and V2 use synchronous single-timestep updates — all loops update on the same discrete clock tick. This means ARC-023 (thalamic heartbeat) and its dependent claims (MECH-089–093) cannot be tested; any experiment testing them produces null results by construction.

**Change:** Implement Hierarchical Temporal Abstraction (HTA). Each loop operates at its own temporal grain:

```
E1 (sensorium loop): updates every env step  →  produces z_self, z_world raw estimates
E2 (action-enacting loop): updates every N_e2 env steps (motor command rate)  →  consumes z_self
E3 (planning-gates loop): updates every N_e3 env steps (deliberation rate, N_e3 >> N_e2)  →  consumes theta-cycle summaries of z_world
```

**Cross-rate integration (MECH-089):** E3 does NOT receive raw E1/E2 output. It receives temporally-abstracted summaries:
```python
# After each E1 step, update rolling theta-cycle buffer
theta_buffer.update(z_world_estimate, z_self_estimate)

# At E3 heartbeat tick (every N_e3 steps): E3 samples the buffer summary
z_world_for_e3 = theta_buffer.summary()   # theta-cycle-averaged world state
```

**Beta-gated policy propagation (MECH-090):** E3 continues updating its internal model at each E3 heartbeat. Beta state controls whether that update propagates to action selection:
```python
if not beta_elevated:   # completion event or stop-change signal
    action_selector.update(e3.current_policy_state)
# else: E3 updates internally but output is held
```

**Phase reset (MECH-091):** Salient events (completion, unexpected harm, commitment crossing) call `e3_clock.phase_reset()`, synchronising the next E3 heartbeat to the event.

**SWR replay (MECH-092):** When E3 heartbeat fires with no pending salient event (quiescent cycle), trigger `hippocampal.replay(theta_buffer.recent)` for viability map consolidation.

**Implementation recommendation:** Use time-multiplexed execution with explicit rate parameters as V3 phase 1 (simpler, testable), design toward full HTA as phase 2. Avoid threading for now (Python GIL complications).

```python
# Config parameters added to V3
e1_steps_per_tick: int = 1     # E1 updates every step
e2_steps_per_tick: int = 3     # E2 updates every 3 steps
e3_steps_per_tick: int = 10    # E3 updates every 10 steps
theta_buffer_size: int = 10    # how many E1 steps per theta summary
```

---

## 3a. Additional Substrate Decisions (Post-Launch)

These SDs were registered and implemented during V3 experimentation. They extend §3 and are
equally binding on V3 implementation.

### SD-007 — Perspective-Corrected World Latent

**Problem:** z_world encoder conflates environmental change with self-caused change because it
has no access to the agent's own motor command. Objects entering the field of view look identical
to the agent moving toward objects — both produce z_world change.

**Solution:** ReafferencePredictor in `ree_core/latent/stack.py`. Applied in `LatentStack.encode()`:
```python
z_world_corrected = z_world_raw - reafference_predictor(z_world_raw_prev, a_prev)
```
Input is `z_world_raw_prev` (NOT z_self_prev — z_self cannot predict visual cell content changes).
Biological basis: MSTd receives visual optic flow plus efference copy (Shenoy et al. 2002).
See MECH-098, MECH-101.

### SD-008 — alpha_world >= 0.9

**Problem:** EMA alpha=0.3 in `LatentStack.encode()` makes z_world a ~3-step weighted average,
suppressing event responses to ~30% peak amplitude and making E3 precision invariant to environmental
drift (stuck at ~188 regardless of hazard rate).

**Solution:** `LatentStackConfig.alpha_world` must be >= 0.9 (default 0.3 kept for compatibility;
always set explicitly). `alpha_self` may remain low (body state is genuinely autocorrelated).
ThetaBuffer (SD-006) provides the temporal integration E3 needs — alpha=0.3 was double-smoothing.

### SD-009 — Event-Contrastive Auxiliary Loss

**Problem:** Reconstruction and E1-prediction losses are invariant to harm-relevance; z_world
learns to reconstruct the grid but does not distinguish hazard cells from empty cells.

**Solution:** Event-type cross-entropy auxiliary loss during z_world encoder training. Only
supervised event discrimination forces z_world to carry hazard-vs-empty-cell information.
EXQ-020 PASS: selectivity_margin=0.882, event_classification_acc=0.692.

### SD-010 — Harm Stream Separation

**Problem:** z_world cannot simultaneously represent (a) world-state for trajectory planning and
residue accumulation and (b) harm signal for E3 commit gating. Fused z_world causes E3 harm
evaluation to contaminate the residue field with harm-correlated, not causal-footprint-correlated,
trace geometry.

**Solution:** Dedicated harm pathway: `CausalGridWorldV2` emits `harm_obs` alongside world
observations. `HarmEncoder(harm_obs → z_harm)` trains on proximity labels. E3 takes z_harm as
primary input to `harm_eval()`. ReafferencePredictor does NOT apply to z_harm (harm proximity is
inherently agent-relative — the action is the relevant context, not optical flow correction).
EXQ-056c/058b/059c all PASS.

### SD-011 — Dual Nociceptive Streams

**Status: Implemented 2026-03-30. Validated EXQ-178b PASS.**

**Problem (original):** Single `z_harm` conflated two neurobiologically incommensurable nociceptive
pathways (Melzack & Casey 1968; Rainville et al. 1997 Science gold-standard dissociation). A
single stream cannot simultaneously serve SD-003 counterfactual attribution (requires
action-predictable sensory component) and E3 commit gating (requires accumulated motivational
urgency). EXQ-093/094 confirmed `HarmBridge(z_world → z_harm)` has `bridge_r2=0` — architecturally
infeasible because z_world ⊥ z_harm by SD-010 design.

**Implementation:**
- `CausalGridWorldV2` emits `harm_obs_a` (EMA harm accumulator, tau~20 steps) alongside `harm_obs`
- `HarmEncoder(harm_obs → z_harm)` — unchanged; sensory-discriminative, Adelta-pathway analog
  (immediate proximity/intensity, forward-predictable from action). Lateral spinothalamic → S1/S2.
- `AffectiveHarmEncoder(harm_obs_a → z_harm_a)` — new; affective-motivational, C-fiber/
  paleospinothalamic analog (accumulated homeostatic deviation, NOT forward-predictable). Medial
  pathway → CM/PF → ACC/insula/amygdala. Feeds E3 commit gating directly as motivational urgency.
- `LatentState.z_harm_a` field added (optional `[batch, z_harm_a_dim]`)
- `ResidualHarmForward` promoted to `ree_core/latent/stack.py` 2026-04-02 (supersedes `HarmForwardModel`,
  deprecated 2026-04-02 — identity collapse on autocorrelated signals; see EXQ-166b/c/d)

**SD-003 redesigned pipeline (post-SD-011):**
```python
z_harm_s_cf = ResidualHarmForward(z_harm_s, a_cf)   # predicts delta, adds to input
causal_sig = E3.harm_eval_z_harm(z_harm_s_actual) - E3.harm_eval_z_harm(z_harm_s_cf)
```
Do NOT attempt HarmBridge counterfactuals — bridge_r2=0 is architectural.

**Still open (EXQ-195 queued):** Full validation of `ResidualHarmForward` + E3 dual-stream
counterfactual pipeline (ARC-033). The architecture is in place; the experiment is queued.

### SD-012 — Homeostatic Drive Modulation

**Status: Implemented 2026-04-02.**

**Problem (original):** `GoalState.update()` did not use `drive_level`. `benefit_exposure` (EMA
alpha=0.1 of raw benefit signals) never reliably crossed `benefit_threshold` during random-walk
warmup: a single resource contact produced `benefit_exposure ~0.03`, which decayed before the
next contact. EXQ-085 through EXQ-085d all failed at the goal-seeding bottleneck
(`z_goal_norm < 0.1` in every run).

**Implementation:** Drive-scaled benefit signals: `effective_benefit = benefit_exposure * (1.0 + drive_weight * drive_level)`.
- `GoalConfig.drive_weight` default changed from 0.0 to 2.0
- `drive_weight` added to `REEConfig.from_dims()` parameter list (overridable per experiment)
- With `drive_level=1.0` (fully depleted) and `drive_weight=2.0`: `effective_benefit = 0.04 * 3.0 = 0.12`,
  which exceeds `benefit_threshold=0.1`
- Set `drive_weight=0.0` explicitly for ablation baselines
- `drive_level = 1.0 - agent_energy` computable from `obs_body[3]`

See MECH-112, MECH-113 for the broader homeostatic architecture.

### SD-014 — Hippocampal Valence Vector Node Recording

**Status: Implemented 2026-04-04.**

**Problem (original):** RBF nodes in HippocampalModule accumulated only spatial/temporal visit
statistics. There was no per-node record of the affective significance of each location, making
it impossible to implement drive-state-dependent prioritisation of replay or navigation targets.

**Implementation:** 4-component valence vector `V = [wanting, liking, harm_discriminative, surprise]`
added to `RBFLayer` and `ResidueField` (`ree_core/residue/field.py`).
- `RBFLayer.valence_vecs` buffer: shape `[num_centers, 4]`, updated incrementally per visit
- New methods: `RBFLayer.evaluate_valence(z) -> [batch, 4]`; `ResidueField.update_valence()`,
  `evaluate_valence()`, `get_valence_priority(z_world, drive_state)`
- Constants defined at module level: `VALENCE_WANTING=0`, `VALENCE_LIKING=1`,
  `VALENCE_HARM_DISCRIMINATIVE=2`, `VALENCE_SURPRISE=3`
- `ResidueConfig.valence_enabled` (default `True`; set `False` for ablation)
- MECH-094 gate applies: `hypothesis_tag=True` blocks valence updates during replay/simulation

Prerequisite for ARC-036 (multidimensional valence map) and replay prioritisation via drive state.

### ARC-028 + MECH-105 — Hippocampal-BetaGate Completion Coupling

**Status: Implemented 2026-04-04.**

**Problem (original):** HippocampalModule and BetaGate operated independently. There was no
mechanism for hippocampal trajectory quality to influence beta oscillation state — the Lisman &
Grace 2005 subiculum -> NAc -> VP -> VTA dopamine loop was architecturally absent.

**Implementation:** Two new methods wired the loop:
- `HippocampalModule.compute_completion_signal(trajectories) -> float`: scores all proposed
  trajectories via `_score_trajectory()`, maps best score to a sigmoid dopamine-analog value
  in `[0.5, 1.0)`. Caches as `self._last_completion_signal`.
- `BetaGate.receive_hippocampal_completion(signal) -> bool`: if beta is elevated and
  `signal >= completion_release_threshold` (default 0.75), calls `self.release()` and returns
  `True`.

Implements the biological loop: high hippocampal completion quality → dopamine signal → beta
drops → E3 state propagates to action selection. `get_state()` and `reset()` updated.
Return type of `propose_trajectories()` unchanged.

---

## 4. Design Gate: Q-020 — Adjudicated 2026-03-16

**Resolution: ARC-007 strict.** HippocampalModule generates value-flat proposals. Terrain
sensitivity is a consequence of navigating residue-shaped z_world, not a separate hippocampal
value computation. MECH-073 reframed as a consequence of ARC-013 applied to z_world.

Q-020 asked whether rollout proposals from HippocampalModule arrive at E3 pre-weighted by map
geometry (MECH-073) or value-neutral (ARC-007 strict). The resolution followed the SD-005
dissolution hypothesis: once z_gamma is split into z_self and z_world, the hippocampal map
operates over z_world (the residue field's domain). Valence lives in z_world structure (residue
field curvature), not in a separate hippocampal computation. ARC-007 is vindicated; Q-020
dissolved as an artefact of the unsplit z_gamma.

MECH-074 (amygdala write interface) remains valid but is not a HippocampalModule prerequisite.
See ree-v3/CLAUDE.md Q-020 Decision section for the operative policy.

---

## 5. V3 Module Architecture

### 5.1 LatentStack changes

```python
@dataclass
class LatentState:
    # Core streams (required)
    z_self: torch.Tensor    # [batch, self_dim]   — proprioceptive + interoceptive  (E2 domain)
    z_world: torch.Tensor   # [batch, world_dim]  — exteroceptive world model        (E3 domain)
    z_beta: torch.Tensor    # [batch, beta_dim]   — affective latent                 (shared)
    z_theta: torch.Tensor   # [batch, theta_dim]  — sequence context                 (shared)
    z_delta: torch.Tensor   # [batch, delta_dim]  — regime/motivation                (shared)
    precision: Dict[str, torch.Tensor]
    timestamp: Optional[int] = None
    hypothesis_tag: bool = False  # MECH-094: True = replay/simulation, blocks residue accumulation

    # Harm streams (optional — present when SD-010/011/019a enabled)
    z_harm: Optional[torch.Tensor] = None     # SD-010: sensory-discriminative harm [batch, harm_dim]
                                               #   HarmEncoder(harm_obs) — Adelta-pathway analog
    z_harm_a: Optional[torch.Tensor] = None   # SD-011: affective-motivational harm [batch, z_harm_a_dim]
                                               #   AffectiveHarmEncoder(harm_obs_a) — C-fiber analog
    z_harm_un: Optional[torch.Tensor] = None  # SD-019a: harm_unpleasantness_channel [batch, harm_dim]
                                               #   EMA(z_harm_s) at alpha=0.2; ~5-step rise; medium timescale
                                               #   AIC + E3 short-horizon urgency redirect when use_harm_un
                                               #   NOT attenuated by SD-021 descending modulation
                                               #   (controllability parity per Loffler 2018)

    # Diagnostic fields (optional)
    z_world_raw: Optional[torch.Tensor] = None   # SD-007: raw z_world before reafference correction
    event_logits: Optional[torch.Tensor] = None  # SD-009: [batch, 3] for event-contrastive CE loss
```

`z_gamma` is removed. Encoder is split into `self_encoder` and `world_encoder`. All downstream
modules consume the appropriate stream. Optional fields default to `None` for compatibility with
experiments that do not enable the corresponding SD.

### 5.2 E1 (deep predictor)

Unchanged in function: slow world model, LSTM, trains on sensory prediction error. In V3: produces predictions over both `z_self` and `z_world` channels (associative prior). Provides E1 prior into HippocampalModule (SD-002, already wired in V2).

### 5.3 E2 (fast transition model)

**Expanded interface:**
```python
class E2FastPredictor:
    def forward(self, z_self: Tensor, a: Tensor) -> Tensor:
        """Motor-sensory prediction: z_self_t + a → z_self_{t+1}"""

    def world_forward(self, z_world: Tensor, a: Tensor) -> Tensor:
        """World-state prediction: z_world_t + a → z_world_{t+1}"""
        # Used for SD-003 V3 attribution only

    def action_object(self, z_world: Tensor, a: Tensor) -> Tensor:
        """Produce compressed world-effect action object o_t (SD-004)"""
```

E2 trains on motor-sensory prediction error over `z_self` (primary). `world_forward` and `action_object` may share weights or be lightweight heads.

**Not a harm predictor.** Remove `predict_harm` head from V2 — it belongs to E3.

### 5.4 E3 complex

**Harm evaluation methods (implemented):**
```python
class E3TrajectorySelector:
    def harm_eval(self, z_world: Tensor) -> Tensor:
        """Evaluate harm of a world-state via z_world. SD-003 original pipeline."""

    def harm_eval_z_harm(self, z_harm: Tensor) -> Tensor:
        """Evaluate harm via dedicated z_harm stream (SD-010/011 pipeline).
        Used in post-SD-011 counterfactual: E3(z_harm_s_actual) - E3(z_harm_s_cf)."""

    def benefit_eval(self, z_world: Tensor) -> Tensor:
        """Evaluate benefit/goal proximity from z_world (ARC-030 Go channel).
        D1/Go symmetric to harm_eval's D2/NoGo role."""
```

E3 trains on harm + goal error over `z_world`. E3's harm evaluator is the correct locus for harm
prediction — not E2. Post-SD-011: the counterfactual attribution pipeline operates on `z_harm_s`
(sensory-discriminative stream) via `harm_eval_z_harm`, not on `z_world` directly.

Precision is E3-derived (from E3 prediction error variance), not hardcoded (required for ARC-016).

**Dynamic precision (ARC-016 — implemented):**
```python
# Commitment fires when running_variance < commit_threshold (variance space, not precision space)
# Fixed 2026-03-18: prior precision_to_threshold() was on wrong scale, causing always-committed state
commit_threshold = variance_commit_threshold(config.commitment_threshold)
committed = e3._running_variance < commit_threshold
```

### 5.5 HippocampalModule

- Navigates action-object space `O` (from SD-004), not raw `z_world`
- ResidueField operates over `z_world` (from SD-005)
- E1 associative prior wired in (SD-002, V2 resolved — preserve)
- Performs replay during quiescent E3 heartbeat cycles (MECH-092)
- Q-020 adjudication determines whether proposals are value-flat or pre-weighted at E3 input

### 5.6 ResidueField

Operates over `z_world`, not `z_gamma`. Accumulates `world_delta` from SD-003 attribution pipeline. Self-change (`z_self_delta`) does not drive residue accumulation.

### 5.7 Environment: CausalGridWorld V3

Extend to provide explicit self/world observation channels:
```python
# Observation dict (replacing flat vector)
{
    "body_state": [...],       # proprioceptive channels → z_self encoder
    "world_state": [...],      # exteroceptive channels → z_world encoder
    "contamination_view": [...] # 5×5 float grid (world channel)
}
```

Ground truth `transition_type ∈ {agent_caused_hazard, env_caused_hazard, resource, none}` preserved.

---

## 6. Heartbeat Architecture (SD-006 implementation targets)

These claims are V3-scoped. Implement and test in order:

| Claim | What to implement |
|---|---|
| ARC-023 | Three characteristic update rates: `e1_steps_per_tick`, `e2_steps_per_tick`, `e3_steps_per_tick` |
| MECH-089 | `ThetaBuffer` — rolling E1 summary consumed by E3; E3 never sees raw E1 output |
| MECH-090 | `beta_state` flag gates E3 policy propagation to action selection (not E3 internal updating) |
| MECH-091 | `e3_clock.phase_reset()` on salient events; synchronises next E3 tick to event |
| MECH-092 | `hippocampal.replay(theta_buffer.recent)` during quiescent E3 heartbeat cycles |
| MECH-093 | `e3_steps_per_tick` varies with `z_beta` magnitude (arousal → faster E3 rate) |

**Hypothesis tag (MECH-094):** All internally-generated content (replay, DMN/simulation) carries `hypothesis_tag = True`. This categorically blocks the post-commit error channel and prevents residue accumulation from simulated content. Implement as a flag on the LatentState that is checked before any residue write.

---

## 7. V3 Experiment Queue (Historical — Launch Plan)

> **This section is historical.** These were the first 10 experiments planned at V3 launch
> (2026-03-16). All have been run or superseded. The active experiment queue is in
> `ree-v3/experiment_queue.json`; completed runs are in `ree-v3/runner_status.json`.

These were the first experiments designed after substrate was built:

| ID | Claim | Depends on | Gate |
|---|---|---|---|
| V3-EXQ-001 | SD-005 channel separation validation | SD-005 | First — validates substrate |
| V3-EXQ-002 | Full SD-003 self-attribution (E2+E3 joint) | SD-005, EXQ-018 | SD-005 done |
| V3-EXQ-003 | Action-object planning horizon extension | SD-004 | SD-004 done |
| V3-EXQ-004 | Three-loop incommensurability (ARC-021 full) | SD-005 | V3-EXQ-001 PASS |
| V3-EXQ-005 | World-delta residue accuracy (MECH-072 V3) | SD-005, V3-EXQ-002 | V3-EXQ-002 done |
| V3-EXQ-006 | Intrinsic map valence vs external comparator (Q-020) | SD-005, Q-020 adjudicated | After Q-020 resolved |
| V3-EXQ-007 | Amygdala write operations affect map geometry (MECH-074) | SD-005, Q-020 | After V3-EXQ-006 |
| V3-EXQ-008 | SD-005 dissolves Q-020 (z_world = residue domain) | SD-005 | Pair with V3-EXQ-006 |
| V3-EXQ-009 | Path memory ablation with proper HippocampalModule (ARC-007) | SD-004 | SD-004 done |
| V3-EXQ-010 | Dynamic precision behavioral distinction (ARC-016) | E3-derived precision, wired commit→behavior | ARC-016 circuit complete |

**First priority order: V3-EXQ-001 → V3-EXQ-002 → V3-EXQ-003, V3-EXQ-004 (parallel)**

---

## 8. Governance Integration Requirements

All V3 experiments must produce run packs compatible with REE_assembly governance:

```
REE_assembly/evidence/experiments/claim_probe_{claim_id}/runs/{run_id}_v3/
    manifest.json    # architecture_epoch: "ree_hybrid_guardrails_v1"; run_id ends "_v3"
    metrics.json     # includes fatal_error_count: 0.0
    summary.md
```

Key fields in manifest.json:
```json
{
    "architecture_epoch": "ree_hybrid_guardrails_v1",
    "run_id": "20260401T120000_z_self_world_separation_v3",
    "status": "PASS" | "FAIL"
}
```

`sync_v2_results.py` covers V2. A `sync_v3_results.py` should be written when V3 experiments run (same pattern as sync_v2_results.py but reading from `ree-v3/evidence/experiments/`). Alternatively, V3 experiments can write run packs directly to REE_assembly if the runner is extended.

---

## 9. Repo Structure

```
ree-v3/
├── ree_core/
│   ├── __init__.py
│   ├── agent.py                    # REEAgent — updated for split latent
│   ├── latent/
│   │   ├── stack.py                # LatentState with z_self, z_world
│   │   └── theta_buffer.py         # ThetaBuffer for cross-rate integration (SD-006)
│   ├── predictors/
│   │   ├── e1_deep.py              # Unchanged in function; reads z_self + z_world
│   │   ├── e2_fast.py              # Extended: world_forward + action_object (SD-004/005)
│   │   └── e3_selector.py          # Extended: harm_eval + dynamic precision (ARC-016)
│   ├── hippocampal/
│   │   └── module.py               # Action-object space navigation (SD-004)
│   ├── residue/
│   │   └── field.py                # Operates over z_world (SD-005)
│   ├── heartbeat/
│   │   ├── clock.py                # Multi-rate clock, phase reset (SD-006, ARC-023, MECH-091)
│   │   └── beta_gate.py            # Beta-gated policy propagation (MECH-090)
│   ├── environment/
│   │   └── causal_grid_world.py    # Extended with self/world obs channels
│   └── utils/
│       └── config.py               # Extended with rate params
├── experiments/
│   ├── run.py                      # Experiment runner (inherits V2 pattern)
│   ├── pack_writer.py              # Writes governance run packs
│   ├── v3_exq_001_z_separation.py
│   ├── v3_exq_002_sd003_joint.py
│   └── ...
├── evidence/
│   └── experiments/               # V3 flat JSON results (for sync_v3_results.py)
├── tests/
├── scripts/
│   └── sync_v3_results.py         # Bridges V3 flat JSON → REE_assembly run packs
└── CLAUDE.md                       # Single-branch policy: main; python /opt/local/bin/python3
```

---

## 10. V3 CLAUDE.md Content (repo-level instructions)

```markdown
# ree-v3

## Python
Use /opt/local/bin/python3 for all execution (has torch 2.10.0).
Use sys.executable for subprocesses within experiment runners.

## Branch Policy
No feature branches. All work to `main` directly.
Push: `git push origin HEAD:main`

## Governance
Run packs go to REE_assembly/evidence/experiments/.
run_id must end _v3. architecture_epoch must be "ree_hybrid_guardrails_v1".
After experiments complete: run sync_v3_results.py then build_experiment_indexes.py.

## Key Architecture Constraints
- E2 trains on motor-sensory error (z_self). NOT harm/goal error.
- E3 is the harm evaluator. harm_eval() belongs on E3Selector.
- ResidueField accumulates world_delta (z_world). NOT z_gamma.
- HippocampalModule navigates action-object space O. NOT raw z_world.
- All replay/simulation content must carry hypothesis_tag=True (MECH-094).
- Precision is E3-derived (E3 prediction error variance). NOT hardcoded.
```

---

## 11. Build Order (Historical — COMPLETED 2026-03-16 to 2026-03-18)

> **This section is historical.** The 12-step build order was completed at V3 launch (2026-03-16 to 2026-03-18).
> Current implementation status is in §0 (SD table) and `ree-v3/CLAUDE.md`.
> All SDs listed in §0 as "Implemented" are now complete, including SD-011 (2026-03-30),
> SD-012 (2026-04-02), SD-014 (2026-04-04), ARC-028+MECH-105 (2026-04-04), and
> SD-015 (completed 2026-04-10).

1. **Q-020 adjudication** ✓ — ARC-007 strict decided 2026-03-16
2. **Latent split (SD-005)** ✓ — LatentState z_self/z_world, split encoder
3. **E2 extensions (SD-004)** ✓ — `world_forward` + `action_object` heads
4. **E3 extensions** ✓ — `harm_eval` method, E3-derived dynamic precision
5. **ResidueField update** ✓ — z_world substrate, world_delta accumulation
6. **HippocampalModule update** ✓ — action-object space navigation
7. **CausalGridWorld extension** ✓ — split observation channels
8. **Multi-rate clock (SD-006 phase 1)** ✓ — time-multiplexed, explicit rate params
9. **ThetaBuffer + cross-rate integration** ✓ — E3 consumes theta summaries
10. **Beta gate + phase reset** ✓ — MECH-090, MECH-091
11. **Replay** ✓ — MECH-092
12. **V3-EXQ-001** ✓ — substrate validated; experimentation continues through EXQ-096a+

---

## 12. What V3 Does NOT Need to Implement

> Updated 2026-04-20: sleep mechanisms were rescoped from V4 into V3. V4 is now
> reserved for social systems. The list below reflects the post-rescope boundary.

- Full DMN (ARC-014 at minutes-to-hours timescale) — MECH-092 is the micro-DMN; full DMN is later
- Multi-agent / multi-instance scenarios (V4: "sharing joys and sorrows")
- Production deployment concerns
- z_self hippocampal navigation / ARC-031 (self-navigation) — V4, gated on EXQ-075/076 PASS
- Representing other agents and trajectories that affect another agent's z_harm_a or
  benefit_exposure over time (V4 social systems; gated on V3 full-completion-gate MECH-163)
- MECH-274 other-attribution sleep-dependent aggregation (ARC-010 empathy / mirror
  modelling, V4 scope)
- SD-030 / SD-031 per-stream SD-003 comparators (V4-deferred per 2026-04-18 SD-003
  supersession)
- SD-033e frontopolar parallel-goal deliberation behavioural implementation (V4-reserved;
  the ree_core/pfc/ stub landed 2026-04-21 raises NotImplementedError when enabled)

---

## Source Documents

All decisions in this spec derive from:

- `REE_assembly/docs/architecture/v2_v3_transition_roadmap.md` — V2 results, V3 targets, transition criteria
- `REE_assembly/docs/architecture/control_plane_heartbeat.md` — ARC-023, MECH-089–093, SD-006
- `REE_assembly/docs/thoughts/2026-03-14_self_world_latent_split_sd003_limitation.md` — SD-005 motivation
- `REE_assembly/docs/architecture/sd_003_experiment_design.md` — SD-003 V2/V3 design, EXQ-027/028 post-mortem
- `REE_assembly/docs/claims/claims.yaml` — claim statuses, v3_pending flags
- `REE_assembly/evidence/experiments/promotion_demotion_recommendations.md` — governance decision queue
