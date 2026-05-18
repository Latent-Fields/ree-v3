# ree-v3 Repository Specification

**Created:** 2026-03-16
**Last updated:** 2026-05-18
**Status:** Living specification — launch doc updated with current V3 state
**Repo name:** `ree-v3`
**Governance epoch:** `ree_hybrid_guardrails_v1` (same as V2 — epoch is per-architecture not per-repo)
**Run ID suffix for governance:** `_v3`

---

## 0. Current V3 State (2026-05-18)

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
