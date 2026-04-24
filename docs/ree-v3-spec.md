# ree-v3 Repository Specification

**Created:** 2026-03-16
**Last updated:** 2026-04-24
**Status:** Living specification — launch doc updated with current V3 state
**Repo name:** `ree-v3`
**Governance epoch:** `ree_hybrid_guardrails_v1` (same as V2 — epoch is per-architecture not per-repo)
**Run ID suffix for governance:** `_v3`

---

## 0. Current V3 State (2026-04-24)

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

- **844 total completions** (indexer rebuilt 2026-04-23 after PM lit-pull session; covers
  SD-036 / MECH-279 / MECH-269 Phase 1/2 ii/2 iii T4 / MECH-288 / MECH-287 landing
  diagnostics now indexed): across EXQ-001 through
  EXQ-474 plus lettered iterations and per-seed runs. Spanning SD-003 through SD-023
  validation, heartbeat architecture (SD-006), reafference (SD-007), encoder fixes
  (SD-008/009), harm stream separation (SD-010), dual nociceptive streams
  (SD-011/SD-022), homeostatic drive (SD-012), self-attribution counterfactuals
  (SD-013/ARC-033), valence vector recording (SD-014), resource encoder (SD-015),
  frontal cue integration (SD-016), sleep infrastructure (SD-017), surprise-gated
  replay (MECH-205), E1 predictive wanting (MECH-216), wanting/liking dissociation
  (MECH-112/229/117), goal conditioning (MECH-116/163/ARC-032), context memory
  (MECH-153/ARC-042), EXQ-223 minimal vertebrate ablation milestone, the SD-032
  cingulate cluster (a/b/c/d/e) validated inline 2026-04-19, SD-033a lateral-PFC-analog
  landing (V3-EXQ-456 PASS), SD-034 governance closure operator + MECH-267 + MECH-268
  landing smokes, the SD-035 amygdala-analog landings (V3-EXQ-473 SD-035 CeA mode-prior
  PASS, V3-EXQ-474 SD-035 BLA encoding+remap PASS) plus V3-EXQ-455 SD-032a behavioural
  coordinator PASS, and the 2026-04-22 V_s invalidation runtime substrate wave
  (SD-036 GABAergic cross-stream decay + MECH-279 PAG freeze gate; MECH-269 Phase 1
  per-stream V_s; MECH-288 event segmenter; MECH-287 invalidation trigger;
  MECH-269 Phase 2 ii AnchorSet; MECH-269 Phase 2 iii T4 per-region V_s) all
  landing-diagnostic PASS via contract tests (85/85) and activation smokes. A fresh
  indexer rebuild is pending after this cycle's wave of results.
- **Currently queued (2026-04-24, 6 items -- 1 claimed, 5 pending):**
  - **V3-EXQ-476** (pending, priority 70, MECH-269 V_s validation entropy probe --
    cascade gate; baseline agent + V_s flags ON vs OFF, action_class_entropy measure;
    PASS = ON entropy > OFF entropy by >=0.1 in >=2/2 seeds; 30 min; `diagnostic`).
    This is the cascade gate for the V_s-gated cascade track (EXQ-445d / EXQ-449c /
    EXQ-455a); FAIL/INCONCLUSIVE means MECH-284 Phase 3 consumer must land before
    downstream cascade can run. Queued 2026-04-24 -- previously listed as "planned
    but not yet queued" in the 2026-04-23 snapshot.
  - **V3-EXQ-449c** (pending, priority 50, `depends_on: V3-EXQ-445d`, MECH-074b
    BLA retrieval bias V_s-gated; BLAAnalog retrieval-bias ON vs OFF; PASS =
    action_class_entropy ON - OFF >= 0.1 AND harm_rate reduced in >=2/3 seeds;
    `evidence`; 150 min).
  - **V3-EXQ-433c** (**claimed DLAPTOP-4.local 2026-04-23T23:23:48Z**, priority 55,
    SD-029 event-conditioned MECH-256 comparator with curriculum ON + scripted
    agent-caused elicitation; supersedes V3-EXQ-433b). Fix to the agent_caused_hazard
    trials-collected shortage: SD-029 curriculum enabled in P0 / P1 / eval;
    deterministic move onto adjacent hazard when agent-caused trial count short;
    C0 sufficiency gate (n_agent / n_env >= 20 in >=3/4 seeds) -- if C0 fails,
    outcome=FAIL but per-claim `evidence_direction='inconclusive_insufficient_events'`
    (not 'weakens') so governance scores are not corrupted by a trials-shortage
    run. 4 seeds x 400 eps; `evidence`; 90 min.
  - **V3-EXQ-449b** (pending, priority 52, SD-016 cue_action_proj consumer fix
    verification; supersedes V3-EXQ-449a). Verifies the SD-016 cue_action_proj
    consumer fix (predictors/e1_deep.py: cue_action_proj input changed from
    cue_context alone to `[cue_context, z_world]` concat to break the
    uniform-attention collapse localised by EXQ-449a). Three-regime protocol
    (g1 supervised-active, g2 frozen, g3 detach-bypassed); primary P1 =
    g2 action_bias_per_channel_std > 1e-3 (was 2.7e-8). Smoke-test 2026-04-23
    dry-run g2 per-channel std = 2.957e-3, primary_pass=True. 1 seed,
    75 eps/run; `diagnostic` (excluded from governance scoring); 30 min.
    Unblocks V3-EXQ-418c.
  - **V3-EXQ-418c** (pending, priority 50, SD-016+SD-017 context-conditioned
    action with cue_action_proj consumer fix; supersedes V3-EXQ-418b).
    Re-run of EXQ-418a now that the SD-016 cue_action_proj consumer fix has
    landed -- the substrate is enabled via `sd016_enabled=True` (which 418a
    already sets), so the same script is reused with the upstream fix active.
    3 seeds x 200 eps; `evidence`; 60 min.
  - **V3-EXQ-137** (pending, priority 40, MECH-097 PPS commit locus:
    PPS_LOCUS_ON vs ABLATED). Backlog-queue item EVB-0137; instrumentation
    fixed 2026-04-24 (verdict print, outcome field, timestamp_utc,
    EXPERIMENT_PURPOSE); smoke-test PASS. 1 seed x 400 eps; `evidence`;
    180 min.
  The 2026-04-22 V_s invalidation runtime wave (SD-036, MECH-279, MECH-269
  Phase 1/2 ii/2 iii T4, MECH-288, MECH-287) landed via contract tests +
  activation smokes only; V3-EXQ-476 is the combined-cluster end-to-end
  validation and is now queued. The heavy queue drain since the 2026-04-21
  snapshot reflects resolution of the SD-034 / MECH-267 / MECH-268 landing
  smokes, V3-EXQ-456 (SD-033a landing), SD-035 EXQ-473/474, and several
  SD-032 cluster variants (V3-EXQ-445b/c, V3-EXQ-449a, V3-EXQ-452/453/454,
  V3-EXQ-325d) landing as PASS/FAIL entries in runner_status.json. The
  three claimed experiments in the 2026-04-23 snapshot (V3-EXQ-447 /
  V3-EXQ-451 / V3-EXQ-445a) have all since completed or been cleared and
  no longer appear in the queue.
- **Current bottleneck:** V_s invalidation runtime end-to-end validation is now
  the next gate -- V3-EXQ-476 (cascade-gate entropy probe) has been queued 2026-04-24
  at priority 70; PASS unlocks V3-EXQ-449c and the downstream V_s-gated cascade
  (EXQ-445d / 455a) track. SD-032 cluster behavioural follow-through remains the
  primary first-paper-gate blocker for the cingulate track -- V3-EXQ-445a
  (full-pipeline fix for the monostrategy + terrain-prior inversion observed in
  V3-EXQ-445), V3-EXQ-445b (epsilon-greedy exploration variant), and V3-EXQ-445c
  (14x14 env variant) have all since FAILed. V3-EXQ-325d FAILed with zero
  between-arm gradient, leaving the SD-032c AIC-analog descending-modulation
  falsification signature open. V3-EXQ-454 FAILed on ARC-016 adaptive
  commitment_threshold under the 2026-04-20 harness. The SD-003 successor track
  is re-opened by V3-EXQ-433c (claimed DLAPTOP-4.local 2026-04-23T23:23:48Z,
  curriculum-ON + scripted agent-caused elicitation, supersedes V3-EXQ-433b
  after the agent_caused_hazard r2=0.0 trials-shortage failure was diagnosed
  as a sufficiency issue rather than a MECH-256 architectural failure). ARC-007
  path-memory track has cleared its most recent claim.
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
  all v3_pending, substrate work not yet started. SD-016 cue_action_proj
  forward-path blocker is diagnosed and a verification run is now queued:
  V3-EXQ-449a localised the collapse to a uniform-attention bottleneck inside
  extract_cue_context (g2 action_bias_per_channel_std ~= 2.7e-8 with ContextMemory
  frozen); the fix routes cue_action_proj input from `cue_context` alone (latent_dim=64)
  to `[cue_context, z_world]` concat (latent_dim+world_dim=96), and the 2026-04-23
  dry-run shows g2 per-channel std = 2.957e-3 (primary_pass=True). V3-EXQ-449b is
  queued as the verification (supersedes V3-EXQ-449a); V3-EXQ-418c is queued as the
  downstream SD-016+SD-017 context-conditioned action re-run that had FAILed three
  times on action_bias_divergence=0.0 under the broken substrate. The three-layer
  regression suite (preflight / contracts / deferred changed) with contracts C1-C8
  covering the SD-032 cluster was at 31/31 PASS after the SD-033e stub contract
  additions (2026-04-21) and grew to 85/85 after the V_s invalidation runtime
  wave added contracts for MECH-269 Phase 1 (5 tests), MECH-288 event segmenter
  (7 tests), MECH-287 invalidation trigger (5 tests), MECH-269 AnchorSet Phase 2 ii
  (9 tests incl. 2 integration smokes), and MECH-269 per-region V_s Phase 2 iii T4
  (6 tests incl. 1 integration smoke) all passing 2026-04-22; explorer preflight
  badge + pre-commit contracts hook (PR 5) remain live. **Pending review queue was
  last generated 2026-04-23T17:49:07Z and lists 25 items (24 PASS, 0 FAIL, 1 UNKNOWN
  for V3-EXQ-471; next indexer rebuild clears the UNKNOWN). PASS queue is dominated
  by the SD-033 cluster landings (EXQ-456, 460, 462-468) across multiple timestamps.
  Governance-cycle pass is pending for the SD-032 behavioural FAILs, the SD-035 /
  MECH-266 landings, and the V_s invalidation runtime substrate landings -- the
  behavioural end-to-end validation (V3-EXQ-476 cascade-gate entropy probe) is
  now queued and is the next-up decision point.**

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
- MECH-272 state-gated anchor/probe routing (waking=anchor-dominant, sleep=probe-dominant),
  MECH-273 sleep-dependent aggregation of SD-003 single-episode self-attribution into
  stable self-model -- registered 2026-04-21 (v3_pending, substrate not yet started)
- MECH-275 sleep-phase general Bayesian aggregation, MECH-276 scientist-agent
  counterfactual-backed attribution, MECH-277 motor-experimentation action-space
  discovery, MECH-278 experimental-action object-schema formation, ARC-059 three-stage
  developmental ordering self->objects->others (refines ARC-019) -- all registered
  2026-04-21 (v3_pending)
- MECH-269 hippocampal replay start-state selection (anchor vs probe), MECH-270
  ephaptic-field substrate for per-stream verisimilitude readout, MECH-271 MECH-094 as
  routing signature (anchored->PFC/E1, probe->BLA/NAc) -- all registered 2026-04-21
  (v3_pending)

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

    # Harm streams (optional — present when SD-010/011 enabled)
    z_harm: Optional[torch.Tensor] = None     # SD-010: sensory-discriminative harm [batch, harm_dim]
                                               #   HarmEncoder(harm_obs) — Adelta-pathway analog
    z_harm_a: Optional[torch.Tensor] = None   # SD-011: affective-motivational harm [batch, z_harm_a_dim]
                                               #   AffectiveHarmEncoder(harm_obs_a) — C-fiber analog

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
