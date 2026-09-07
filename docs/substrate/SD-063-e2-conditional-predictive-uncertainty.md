## SD-063: E2 Conditional Predictive-Uncertainty Head (z_world quantile) (2026-07-05)
- SD-063: e2.conditional_predictive_uncertainty_head -- IMPLEMENTED 2026-07-05
  (substrate; v3_pending until a validation experiment shows the head's per-point
  predictive variance improves E3 commitment gating over the running-variance EMA
  AND the SD-031 agency residual is preserved under joint training. PROMOTES
  NOTHING). The concrete realization of the MECH-059 confidence channel: a
  distribution-free quantile/pinball head over (z_world, action) that emits a
  per-input predictive spread tracking realized error, feeding E3's commit gate
  in place of the state-blind, global running-variance EMA. Winner of the
  V3-EXQ-712 diagnostic (quantile CRPS 0.00486 vs point 0.00514;
  precision_error_corr 0.379 vs the EMA null 0.0; Gaussian-family heads -- hetero,
  mixture -- did WORSE than the point baseline, so the distribution-free form is
  load-bearing).
  Module: ree_core/predictors/e2_world_uncertainty.py (E2WorldUncertaintyHead +
  E2WorldUncertaintyConfig; QUANTILE_LEVELS = 9 levels 0.1..0.9, the 712 winner;
  IQR_TO_STD_10_90 = 2.5631). Trunk = 2-layer MLP(ReLU) -> Linear(D*Q) -> [B,D,Q]
  (matches the 712 QuantileHead). compute_loss = pinball; predictive_variance /
  predictive_std = monotone-rearranged (torch.sort, anti-crossing) [q0.1,q0.9] IQR
  -> Gaussian-reference variance, meaned over dims, per batch item, under no_grad.
  Data flow: z_world_t.detach() + a_onehot -> E2WorldUncertaintyHead -> pinball
  (P1) / predictive_variance (read) -> E3.select(conditional_predictive_variance=)
  commit gate.
  Config: LatentStackConfig.use_e2_world_uncertainty (bool, default False;
  bit-identical OFF) + e2_world_uncertainty_hidden_dim (128) +
  e2_world_uncertainty_lr (1e-3); E3Config.use_conditional_precision_gate (bool,
  default False). Both surfaced by REEConfig.from_dims. Like use_e2_world_forward,
  the flag signals intent; the head is instantiated at the experiment/agent level
  -- LatentStack.encode() is UNTOUCHED, so no new LatentState field and OFF is
  byte-identical. z_world_dim is REQUIRED at construction (no literal default);
  unlike E2WorldForward there is NO world_dim>=128 assert (this is a predictive-
  spread readout, not the SD-031 discriminative comparator; the 712 diagnostic ran
  at world_dim=32).
  E3 consumer (e3_selector.py select()): new kwarg conditional_predictive_variance
  (default None). When E3Config.use_conditional_precision_gate is True AND a value
  is supplied, the ARC-016 commit decision compares that per-input variance against
  effective_threshold INSTEAD of self._running_variance; None or flag-off -> EMA
  fallback (byte-identical). Does not touch the use_harm_variance_commit path.
  SD-031 AGENCY-RESIDUAL GUARD (the load-bearing caveat): the head is a SEPARATE
  nn.Module sharing NO parameters with E2WorldForward or the encoder, and its P1
  loss reads DETACHED z_world inputs AND a DETACHED z_world_next target. Its
  gradients therefore never reach the forward model that produces the SD-031
  agency residual -> it cannot explain the residual away by construction. The
  validation experiment must still confirm this empirically under joint training.
  Phased training (validation, not the substrate): P0 z_world encoder warmup
  (SD-009 + SD-018); P1 head on frozen z_world (detach inputs + target); P2 eval
  CRPS + precision_error_corr + agency-residual preservation.
  MECH-094: DOES NOT APPLY -- waking online forward-model read for commitment
  gating; no memory write, no simulation/replay.
  Backward compatible: both switches False by default; agent hot path unchanged;
  1381/1385 suite PASS (the 4 fails are pre-existing, unrelated: E1 SD-016 proj-dim,
  control-vector bit-identical, 2x runner fail-branch -- all fail on the clean base
  tree). 15/15 new contracts in tests/contracts/test_sd063_conditional_uncertainty_head.py
  (config no-op + from_dims surface / head shapes + dim-required + level validation /
  pinball-trains + heteroscedastic conditional variance / SD-031 param-disjoint +
  detach-blocks-encoder-grad / E3 gate OFF-ignores + ON-overrides-both-directions +
  ON-no-value-EMA-fallback).
  Validation experiment: V3-EXQ-716 queued (see below) -- diagnostic falsifier,
  PROMOTES NOTHING.
  Design doc: REE_assembly/docs/architecture/sd_063_e2_conditional_uncertainty_head.md
  See MECH-059 (confidence channel; instantiated), SD-031 / E2WorldForward (the
  agency residual this must not disturb; dep), MECH-256 (comparator family),
  V3-EXQ-712 (motivating diagnostic), ARC-016 (dynamic-precision commit gate this
  feeds).

- SD-065: environment.conditioned_safety_cue_channel -- IMPLEMENTED 2026-07-14.
  Module: ree_core/environment/causal_grid_world.py (env-only; NOT surfaced through
  REEConfig.from_dims). The controllable, observable Pavlovian safety CS the SD-051
  ConditionedSafetyStore needs so MECH-304's promote-to-active BEHAVIOURAL falsifier
  can run (classical conditioned inhibition). Ambient (uniform-when-active) 25-dim
  field view appended LAST to world_state (feeds z_world, which the store keys on)
  + obs_dict["safety_cue_field_view"] (absent when disabled). Ambient not spatial so
  the cue's z_world contribution is position-independent -- reliably present at every
  relief tick and reproducible at test (removes a navigation confound). Adds NO
  entity type (NUM_ENTITY_TYPES stays 7; 5x5x7 local_view width fleet-invariant).
  Config (env kwargs): safety_cue_enabled (bool, default False -- master; grows
  world_obs_dim by exactly 25 when on) + safety_cue_scale (1.0, clipped [0,1]) +
  safety_cue_on_relief (False) + safety_cue_heal_floor (0.05). Two activation paths:
  (1) safety_cue_on_relief auto-activates the cue across the SD-022 damage->heal
  (MECH-302 relief) window -- active exactly when sum(limb_damage) > heal_floor; the
  store still writes its prototype ONLY on the real event_fired tick inside it, so
  the pairing is real, not synthesized; (2) set_safety_cue(active) tri-state manual
  override (test-phase API; takes precedence, persists across reset) to present
  {threat + cue} concurrently in one tick. Preconditions (loud-not-silent, mirroring
  SD-022): safety_cue_enabled requires use_proxy_fields; safety_cue_on_relief
  requires limb_damage_enabled (both raise ValueError). Info sentinels always present
  (safety_cue_enabled / safety_cue_active / safety_cue_event_count; 0/False OFF).
  Data flow: step() computes _safety_cue_active (manual override > relief window >
  inactive) -> world_state 25-dim view -> z_world -> ConditionedSafetyStore.update
  (SD-051) -> select_action beta_gate.release when signal > safety_store_threshold.
  MECH-094: DOES NOT APPLY at the env level (the MECH-094 sim-gate lives in the
  store.update sim_mode path, unchanged). Phased training: NOT REQUIRED (no new
  encoder head; env channel only). Backward compatible: safety_cue_enabled=False by
  default -- no channel, world_obs_dim unchanged, zero RNG draws, bit-identical OFF.
  10/10 new contracts in tests/contracts/test_scheduled_safety_cue_curriculum.py
  (off-inert + dim-unchanged / preconditions / geometry +25 / manual-override + scale
  clip / relief-pairing tracks damage window / reset clears dynamics + override
  persists / RNG isolation).
  Validation experiment: V3-EXQ-763 queued (MECH-304 promote-to-active behavioural
  falsifier; see below).
  Design doc: REE_assembly/docs/architecture/sd_065_conditioned_safety_cue_channel.md
  See SD-051 / MECH-304 (the consumer + the claim this makes testable), MECH-302 /
  SD-050 (relief teaching signal), MECH-303 (sister contextual-safety pathway; DV2
  sparing control), SD-022 (build-pattern precedent + the relief generator).

- SD-066: safety_prediction.common_mode_invariant_conditioned_safety_readout --
  IMPLEMENTED 2026-07-15. Module: ree_core/safety/conditioned_safety_store.py
  (ConditionedSafetyStore, non-parametric; NOT the V4 trainable-contrastive head).
  Opt-in centered readout that lifts the z_world common-mode CEILING blocking
  MECH-304's behavioural gate: under SD-008 z_world under-differentiation every
  z_world sits at cosine ~0.99, so the store's raw-cosine gate (sigmoid(gain*cos) >
  0.5 == cos>0) fires UNCONDITIONALLY once the prototype is non-empty -- the cue
  (~0.006 cosine) is unresolvable behaviourally (diagnosed 2026-07-15: raw store
  arm A(cue) == arm C(nocue) release, sig(cue)~sig(nocue)). Fix: the store keeps a
  slow EMA `baseline` of z_world (the common-mode) and does BOTH prototype
  accumulation and querying on the CENTERED residual z_world - baseline, so the cue
  residual dominates the cosine. Config: REEConfig.safety_store_centered (bool,
  default False -> bit-identical raw cosine) + safety_store_baseline_alpha (0.02),
  surfaced by from_dims; agent.py passes them to the store constructor. Baseline
  advances inside store.update() (already called every tick in sense()) -> NO agent
  hot-path/wiring change. sim_mode (MECH-094) does not advance the baseline. Data
  flow unchanged except the internal cosine is centered. Backward compatible:
  centered=False byte-identical (contract C1 re-implements the raw arithmetic and
  matches every update() return). Phased training: NOT REQUIRED (non-parametric).
  5/5 new contracts in tests/contracts/test_sd066_centered_safety_readout.py (raw
  bit-identical + never-touches-baseline / baseline lifecycle + reset / common-mode
  separation / sim_mode gate / config default); full pytest tests/ 1463 passed.
  VALIDATED end-to-end: with centered ON the MECH-304 behavioural gate becomes
  cue-specific (arm A cue release 1.0 / sig 0.7-0.94 vs arm C nocue release ~0 /
  sig 0.17-0.48; A-C ~1.0) -- V3-EXQ-763 PASS.
  Validation experiment: V3-EXQ-763 (MECH-304 promote-to-active behavioural falsifier).
  Design doc: REE_assembly/docs/architecture/sd_066_centered_conditioned_safety_readout.md
  See SD-051 (the store this extends), SD-065 (the cue channel it resolves), SD-008
  (the z_world under-differentiation it works around), MECH-304 (the gate it unblocks).

- SD-068: sleep.consolidation_pipeline_lesion_harness -- IMPLEMENTED 2026-07-17.
  EXPERIMENT-LAYER harness (NO ree_core change):
  ree-v3/experiments/_lib/consolidation_lesion_harness.py. Per-phase-damageable +
  per-phase-functional-readout instrumentation on the MECH-120 (SWS denoising) ->
  MECH-121 (NREM slot-filling) -> MECH-123 (REM precision) offline-consolidation
  pipeline, so the MECH-168/INV-047/MECH-169 staged-decline-under-uniform-damage
  falsifier is buildable. Config: none added -- builds agents via existing no-op
  flags (shy_enabled/sws_enabled/rem_enabled/use_sleep_aggregation_cluster). Data
  flow: inject known clean content onto each phase's operative substrate
  (context_memory slots / consolidation params / E3 precision reference; V3-EXQ-702
  injected-content precedent, sidesteps the failure_autopsy_V3-EXQ-538a
  encoding-starvation ceiling) -> apply one UNIFORM diffuse-damage sigma
  (RMS-scaled Gaussian, identical per phase) -> read denoising-SNR (MECH-120) /
  transfer-fidelity (MECH-121) / precision-calibration-error (MECH-123) against the
  injection. NON-VACUITY: per-phase errors normalised to fractional-of-own-range
  degradation and ranked by damage-TOLERANCE (crossing sigma) for staging order,
  plus a REM passthrough-nudge-vs-generative-pass contrast (the amplify/attenuate
  indicator). SUBSTRATE FINDING: the three phases operate on DISJOINT state, so a
  faithful cross-phase content-propagation pipe is not directly instrumentable --
  non-vacuity is carried by staging-order + the REM contrast, not a fake pipe.
  Backward compatible: pure instrumentation, no existing behaviour changes. MECH-094:
  no new simulate-then-commit surface. MECH-121 hold RESPECTED (not lifted): the
  validation run is EXPERIMENT_PURPOSE=diagnostic and does NOT tag MECH-121 as
  promotion evidence; NREM leg is substrate-plumbing-fidelity only. Glymphatic/
  amyloid structural half of MECH-169 has NO V3 analog -- OUT OF SCOPE. Phased
  training: N/A (no encoder head trained on moving latents). Smoke: run_staged_sweep
  deterministic; observed tolerance order (nrem, rem, sws) stable across seeds
  42/7 (partial match to reverse-dependency prediction; REM generative sensitivity
  null). Validation experiment: V3-EXQ-778 (diagnostic staged-damage sweep, queued 2026-07-17).
  Design doc: REE_assembly/docs/architecture/sd_068_consolidation_lesion_harness.md
  See MECH-120, MECH-121 (held), MECH-123, MECH-168, INV-047, MECH-169, SD-017.
  >> SWS READOUT REBUILT 2026-07-18 (ree-v3 main 8b18338). V3-EXQ-778c's null control
  measured null_slope_ratio 1.0000 (sd 2.7e-8) on 8/8 seeds for the sws leg: `_shy` is
  AFFINE, so shy(clean+n) - shy(clean) = shy_centred(n) independent of `clean`, making
  `noise_power` content-free and the content term a constant offset that differentiates
  away. `denoising_snr_db` therefore NEVER measured content fidelity. The scored series
  is now `_sws_pattern_completion`: cosine retrieval margin of the post-SHY store
  against the injected prototypes, probed with the UNSCALED base so the null arm
  receives a real arm-identical probe that is simply not planted (Bar et al. 2020
  "same odour, no prior pairing") rather than a 0/0-degenerate zero -- the failure mode
  the rem leg already exhibits. Same repair pattern as rem_terrain_variance ->
  rem_generative_fidelity (da873a1). Backward compatible: denoising_snr_db /
  signal_power / noise_power still emitted as TELEMETRY and the error_propagation_gain
  driver keys are unchanged, so V3-EXQ-778/778a drivers still run. NOTE their staging
  numbers are NOT reproducible across this change -- tolerance_sigma_sws flows through
  the repaired series. That is intentional; those numbers were retracted as staging
  evidence by the 778c autopsy. Local smoke ONLY (seeds 42/7/123): null_slope_ratio_sws
  0.116/0.171/0.152 vs the 0.25 ceiling, injected slope 0.32-0.36, confounded_phases
  now ['rem'] alone. CAVEAT on that pass: the readout is cosine-based and therefore
  scale-invariant, so its null arm is flat in sigma partly BY CONSTRUCTION -- a
  content-scale ladder (0.0/0.25/0.5/1.0) was run as the independent check and shows
  zero response at content_scale=0 and a large amplitude-dependent response above it.
  NOT YET VALIDATED at seed scale: V3-EXQ-778g queued 2026-07-18 re-runs the null
  control on the repaired readout at the full 8-seed 778a set. Until 778g reports, do
  NOT treat the sws leg as a validated instrument.

- SD-069: control_plane.phasic_surprise_burst -- IMPLEMENTED 2026-07-17.
  LC-NE PHASIC complement to MECH-313 noise_floor (tonic) on the SAME E3 softmax
  temperature channel -- the substrate that makes MECH-063 sub-claim (ii)
  (tonic/phasic split) behaviourally testable. Module:
  ree_core/regulators/phasic_surprise_burst.py (PhasicSurpriseBurst; pure-arithmetic
  regulator matching noise_floor.py MECH-313 / broadcast_override.py SD-037 -- no
  nn.Module, no learned params). Config: REEConfig.use_phasic_burst (default False;
  set True to enable) + phasic_burst_surprise_ema_decay (0.1), _trigger_ratio (1.5),
  _trigger_floor (1e-6), _temp_delta (-0.5 = phasic sharpening), _decay (0.5),
  _min_temperature (0.1), and _signal_source ("running_variance" default; set
  "instantaneous_pe" for the sharp source -- see below). Data flow: <surprise source>
  (per-tick PE surprise) ->
  PhasicSurpriseBurst.tick (event iff surprise >= trigger_ratio x EMA baseline;
  inject drive -> envelope decays geometrically over a few ticks) -> burst_level
  [0,1] -> temperature_delta = temp_delta x burst_level -> combined_T =
  max(tonic_effective_T + delta, min_temperature) -> e3.select(candidates,
  combined_T) at agent.py select_action(). Tonic (MECH-313) applies a SUSTAINED
  every-tick lift; SD-069 adds an EVENT-LOCKED transient on top -> comparable
  readouts, independently toggleable. IMPORTANT: reuses the MECH-104
  volatility-surprise lit basis (Aston-Jones & Cohen 2005 phasic mode) but routes to
  the softmax, NOT the ARC-016 commit gate -- so it does NOT implement the MECH-104
  claim (control_plane.volatility_interrupt, v3_exq_365), which is the same surprise
  routing to the de-commit gate. Diagnostic: phasic_burst_level / _temp_delta in
  _last_score_bias_decomp; phasic_burst block in _last_control_vector (tonic lift kept
  uncontaminated for the dissociation readout). Backward compatible: agent does not
  instantiate the regulator when use_phasic_burst=False; combined T == tonic T; OFF
  action stream bit-identical (contract C8, n_events==0). MECH-094: simulation_mode
  returns cached burst, no state advance; no memory write surface -> phased training
  N/A, hypothesis_tag N/A. Smoke: 17 SD-069 contracts pass; full tests/contracts
  1462 pass (0 regressions); injected surprise spike fires through select_action ->
  pre-commit softmax entropy drops on the event (0.80->0.06) and decays over the tail
  (event-locked transient), bit-identical before the event.
  SHARP-SURPRISE SOURCE (2026-07-17): the event-detector source is now selectable via
  REEConfig.phasic_burst_signal_source. "running_variance" (default, no-op) reads the
  SMOOTHED e3._running_variance EMA -- which decays monotonically for an untrained
  forward model (0.475 -> 0.004) and washes out real per-tick spikes, so the lever
  fires 0 natural events even under env volatility (background_drift_enabled). This
  meant a no-training 777-style probe could only exercise the lever with a synthetic
  poke to _running_variance -- exactly what MECH-063 (ii) must avoid. "instantaneous_pe"
  reads a new no-op read-only signal e3.last_instantaneous_pe = the RAW per-tick PE-MSE
  (error_var in e3.update_running_variance) captured BEFORE the running-variance EMA
  smoothing folds it in, so genuine surprise spikes survive (Aston-Jones & Cohen 2005
  phasic mode fires on SHARP/instantaneous salience, not a smoothed average).
  Validated (untrained rollout, CausalGridWorldV2 drift on, NO synthetic poke): the
  phasic burst is ACTIVE across the run under "instantaneous_pe" (burst_ticks up to
  209/232 at trigger_ratio 1.3) but NEVER active under "running_variance" (0/217) --
  a clean load-bearing contrast. Note: the regulator's own n_events counter resets per
  episode (agent.reset -> phasic_burst.reset), so a readiness gate must sum events
  across episodes rather than read end-of-run get_state. Invalid source strings raise
  ValueError at agent construction (no silent fallback). Backward compatible: default
  "running_variance" reads the identical scalar as before -> existing experiments
  bit-identical; e3.last_instantaneous_pe is written unconditionally but read by nothing
  at the default source.
  Validation experiment: V3-EXQ-779 queued 2026-07-17 (MECH-063 sub-claim ii
  tonic-vs-phasic dissociation, 777-harness pattern, phasic_burst_signal_source=
  "instantaneous_pe" on PHASIC-ON arms; P0 readiness gate requires PHASIC-ON arms to
  fire >= MIN_EVENTS real surprise events -- see Step 8).
  Design doc: REE_assembly/docs/architecture/sd_069_phasic_surprise_burst.md
  DEFECT FOUND + REPAIRED BY SD-075 (2026-07-19): the per-episode cold EMA reset
  documented above makes n_event_ticks a function of episode LENGTH rather than
  surprise on short-episode seeds, because the first tick of an episode only seeds
  the baseline and the ~10-tick time constant exceeds the whole episode. If you are
  reading this to design a phasic experiment, read the SD-075 entry FIRST and set
  phasic_burst_baseline_continuity="carry".
  See MECH-063 (enables sub-claim ii), MECH-313 (tonic counterpart), MECH-104
  (shared lit basis, different consumer), SD-075 (baseline-continuity repair),
  ARC-005.

- SD-024 LIVE-PATH PRODUCER: residue.benefit_terrain_live_producer --
  IMPLEMENTED 2026-07-20. Closes a four-day gap in the SD-024 landing of
  2026-07-16, which built the benefit terrain and its READ but never wired a
  WRITER into the agent loop.
  THE DEFECT: `ResidueField.accumulate_benefit` had NO CALLER anywhere in
  `ree_core/`. Its only two write sites into `benefit_rbf_field`
  (`residue/field.py:673`, `:682`) sit inside that one method, and agent.py
  called `update_valence` / `accumulate` / `accumulate_safety` /
  `evaluate_safety` -- never `accumulate_benefit`. Measured consequence on a
  real warmup_train loop (darwin-arm64, curiosity_weight=0.5), with BOTH
  `benefit_terrain_enabled` and `use_da_modulated_rbf_density` True:
  `benefit_rbf_field.active_mask.sum() == 0`, `num_benefit_events == 0.0`;
  `RBFLayer.compute_local_density` early-returns zeros on an empty active mask
  (`field.py:273`) so `compute_representational_density` returned exactly 0.0;
  therefore `HippocampalModule._curiosity_bonus` (`hippocampal/module.py:870-885`)
  computed `novelty = density * (1 - familiarity) = 0` and returned 0.0 on all
  14432 live calls. The `use_curiosity_familiarity` True/False ablation was
  BIT-IDENTICAL, confirming familiarity was not the binding constraint. Net: the
  SD-025 curiosity drive contributed EXACTLY ZERO to CEM trajectory scoring in
  every live agent run.
  Modified: `REEAgent.update_z_goal` in `ree_core/agent.py` (producer block).
  Config: `ResidueConfig.benefit_terrain_live_producer` (default False; set True
  to enable) + `ResidueConfig.benefit_live_producer_threshold` (default 0.1,
  the consummatory-contact gate, mirroring `liking_threshold`).
  Data flow: reward contact -> `update_z_goal(benefit_exposure, drive_level)` ->
  `ResidueField.accumulate_benefit(z_world, benefit_magnitude=benefit_exposure,
  dopamine_signal=benefit_exposure * drive_level)` -> `benefit_rbf_field` ->
  `compute_benefit_density` -> `compute_representational_density` ->
  SD-025 `_curiosity_bonus` -> CEM scoring.
  SEPARATE FLAG, NOT `benefit_terrain_enabled` -- deliberate: V3-EXQ-767/767a set
  `benefit_terrain_enabled=True` and populate the terrain THEMSELVES via direct
  `rf.accumulate_benefit()` calls (767a lines 236, 238, 305). Gating the live
  producer on that existing flag would silently double-populate those in-vitro
  designs on re-run. 767/767a remain valid in-vitro validations of the drive
  mechanism; they simply never established live-path efficacy.
  PLACEMENT -- read this before moving the block: it sits ABOVE
  `update_z_goal`'s `if self.goal_state is None ... return` guard, not below.
  The default config has `goal_state = None`, so a block after the guard never
  runs; the benefit terrain is a ResidueField concern and SD-024 has no
  GoalState dependency, so gating it there would ship a SECOND instance of the
  same no-producer defect. Not hypothetical -- the block was first written
  after the guard and contracts C2/C4/C5 failed at 0 active centers.
  `dopamine_signal` uses BASE `drive_level`, not `pacc.effective_drive` nor the
  SD-037 override-amplified value: those are goal-SEEDING gains, whereas the
  SD-012 phasic signal `accumulate_benefit` documents is
  `benefit_magnitude * drive_level`.
  ALSO IN THIS LANDING: `HippocampalConfig.familiarity_bandwidth` 1.0 -> 0.20.
  `FamiliarityTracker.query` (`hippocampal/curiosity.py:72-100`) is a CLAMPED SUM
  over anchors, not a normalised average, and the same constant is the
  association threshold in `update()` (`thresh_sq = bw*bw`, `:115`), so at 1.0
  only ~3 anchors go active and their near-unit weights pin the clamp. This did
  not bite while density was identically zero, but becomes load-bearing the
  moment it is not. V3-EXQ-786a swept it on real z_world: 0.05 -> +0.019,
  0.10 -> +0.103, 0.20 -> +0.171, 0.30 -> +0.065, 0.50 -> -0.053 (INVERTED),
  1.00 -> +0.000 -- the old default was the one value measured at exactly zero
  effect. Same root cause as SD-067's dedicated safety bandwidth (the shared
  1.0 is ~15x too wide for the z_world residual scale). Consulted only when
  `curiosity_weight > 0` (default 0.0), so no-op for every default config.
  Backward compatible: producer disabled by default; full suite 2065 passed.
  No phased training (no new encoder head).
  MECH-094: the producer reads `hypothesis_tag` off the live latent rather than
  hardcoding False, so a future replay/DMN caller cannot build benefit terrain
  by accident; `accumulate_benefit` applies the gate internally too.
  Contracts: `tests/contracts/test_sd024_benefit_terrain_live_producer.py`
  (7, C1-C6), which assert through the real REEAgent API against a real
  CausalGridWorldV2 episode loop and NEVER call `accumulate_benefit` directly --
  the 13 pre-existing SD-024 contracts all populate the field themselves, which
  is precisely why a missing producer was invisible to them.
  LANDING NOTE: the substrate edits (agent.py, config.py) were swept into
  commit `8ac193d` by a concurrent MECH-204 session's whole-file write; the
  contract test landed separately in `7f16f25ceb`. Content is complete and
  verbatim -- see CLAUDE.md "Read-modify-write contamination" remedy (a)/(a2).
  Validation experiment: V3-EXQ-795 queued 2026-07-20 (diagnostic; producer ON vs
  OFF at identical seeds, four legs: producer / density / DRIVE / selection
  authority). It never calls accumulate_benefit except in its P0 readiness
  control, so each arm's terrain can only come from the agent's own reward
  contacts. Smoke: ARM_OFF reproduces the defect signature exactly (0 centers,
  density 0.0, bonus 0.0); ARM_ON 1 center / density 0.997 / bonus 0.064.
  See SD-024, SD-025, MECH-232, ARC-030, ARC-057, MECH-117.

- MECH-204 Phase 7 / Option B: sleep.accuracy_anchored_broadcast_recalibration --
  IMPLEMENTED 2026-07-20. Ungated from V4 deferral by the confirmed
  failure_autopsy_V3-EXQ-774_2026-07-17 (adjudicated substrate_ceiling,
  "F1 alone insufficient": precision saturates during waking before the
  per-cycle WRITEBACK lever gains headroom; effect on 1/3 seeds only).
  Method: E3TrajectorySelector.broadcast_precision_pull(target_precision, gain)
  in ree_core/predictors/e3_selector.py; called at the TOP of
  REEAgent.select_action (ree_core/agent.py) so the anchored rv is what this
  tick's commit gate and current_precision consumers see.
  Config: REEConfig.use_rem_precision_broadcast (default False) +
  REEConfig.rem_precision_broadcast_gain (default 0.0). Both also on the
  from_dims factory path. Per-STEP gain -- keep well below
  rem_precision_recalibration_step, which fires once per sleep CYCLE.
  Data flow: serotonin._persistent_zero_point -> compute_recalibration_target()
  -> select_action broadcast read -> broadcast_precision_pull ->
  E3._running_variance. Reads the F1 cumulative reference per lit choice (a)
  (targeted_review_rem_precision_recalibration_timing SYNTHESIS: Hobson-Hong-
  Friston 2014 + Walker-Stickgold 2006). Runs ALONGSIDE F1, not replacing it
  (Q-042 dual-arm pattern).
  WRITE-SITE CORRECTION -- read this before "restoring" the spec: the
  2026-05-09 spec said "additive bias on E3 score". That site is PROVABLY
  SELECTION-INERT for a broadcast. A broadcast is ONE scalar for all K
  candidates; e3_selector applies score_bias as `scores = scores + bias_tensor`
  (a uniform shift, invariant under argmax AND softmax), and every downstream
  consumer is relative (raw_scores.max() - raw_scores[i], raw_score_range,
  topk, cutoff/envelope) with several reading raw_scores, which score_bias
  never touches. It would register a nonzero modulatory channel while changing
  no behaviour -- the exact shape the inert_arm_knob lint exists to catch. The
  lit-pull adjudicated WHAT TO READ, never WHERE TO WRITE. Precision space is
  non-inert: rv feeds the ABSOLUTE commit threshold and 1/(rv + 1e-6), and is
  where V3-EXQ-774's own DV lives. Decision-log entry 2026-07-20 in
  evidence/planning/sleep_substrate_plan.md.
  Backward compatible: disabled by default; existing experiments unaffected.
  Phased training required: no (no new head, no learned parameters).
  MECH-094: not applicable (no simulation/replay content written to memory).
  PAIRED WITH SD-076 -- the broadcast corrects drift, SD-076 creates it. Phase 7
  alone cannot lift the 774 ceiling, because without a drift source the DV is a
  tautology. Do not run the Phase-7 arm without considering SD-076.
  Smoke: 6/6 PASS (UC1 bit-identical OFF by explicit float equality, UC4 pull
  arithmetic, UC4b gain=0 no-op, UC5 no-REM sentinel no-op, UC6 defaults).
  Validation experiment: see sleep_substrate_plan.md Phase 7 status row.
  See MECH-204, MECH-173, Q-042, SD-076.

- SD-076: precision.waking_confidence_inflation -- IMPLEMENTED 2026-07-20.
  Design doc: REE_assembly/docs/architecture/sd_076_waking_confidence_inflation.md.
  Modified: E3TrajectorySelector.update_running_variance in
  ree_core/predictors/e3_selector.py (asymmetric EMA).
  Config: E3Config.use_waking_confidence_inflation (default False),
  E3Config.waking_confidence_inflation_asymmetry (default 0.0, range [0,1)),
  E3Config.waking_confidence_rv_floor (default 0.01).
  PROBLEM: the symmetric EMA makes _running_variance a faithful tracker of true
  prediction error, so rv ~= true error BY CONSTRUCTION and the MECH-173
  overconfidence_index is pinned near zero no matter what is ablated (V3-EXQ-774
  measured -0.000148 / -0.000918 on the suppressed arms). MECH-204's corrective
  function presupposes a daytime drift source V3 did not have, so a null on any
  MECH-204 consumer was a TAUTOLOGY, not evidence.
  MECHANISM: good news incorporated fast (alpha * (1 + asym) when error
  improves), bad news slowly (alpha * (1 - asym) when it worsens), so rv settles
  BELOW the true error mean = genuine, directional, correctable overconfidence.
  BIT-IDENTICAL OFF BY CONSTRUCTION, NOT BY ARITHMETIC: the OFF branch evaluates
  the original symmetric expression unchanged rather than re-deriving it at
  asymmetry 0. Pinned by explicit float equality over an 80-step trace, not an
  approximate comparison.
  THE FLOOR IS LOAD-BEARING, NOT HYGIENE: rv feeds an ABSOLUTE commit threshold
  (running_variance < commit_threshold, ARC-016) and current_precision =
  1/(rv + 1e-6). Unbounded downward drift would pin the agent permanently
  committed AND explode precision. Applied only on the inflation path.
  Biological basis: optimism / positive-outcome bias in waking belief updating
  -- the drift the sleep-recalibration literature behind MECH-204 presupposes.
  Functional translation, not a neuromodulator-specific mechanism claim.
  Backward compatible: disabled by default; existing experiments unaffected.
  Phased training required: no. MECH-094: not applicable.
  Smoke: 6/6 PASS. At asymmetry=0.6 on true error mean 0.05,
  overconfidence_index moves -0.164 (OFF, UNDERconfident) -> +0.273 (ON). The
  OFF value reproduces the sign and rough magnitude of 774's measured
  ARM_FULL_SLEEP = -0.2097, which is direct evidence the autopsy diagnosis
  was right.
  SD-069 unaffected: last_instantaneous_pe is captured BEFORE this smoothing.
  HEADROOM REPAIR 2026-07-22 -- READ THIS BEFORE USING THE LEVER. The absolute
  floor above is the WRONG KIND OF QUANTITY, and 0.01 is the wrong value for this
  substrate. V3-EXQ-794 measured the un-inflated operating point at rv = 0.005420
  (ARM_OFF_OFF) and arm_true_error_ref at ~0.0037, so 0.01 sits 1.8x ABOVE the
  operating point: max(0.01, rv) clamps on the first tick inflation bites and
  never releases. rv_final was EXACTLY 0.010000 on all four inflation arms and
  overconfidence_score was bit-identical to 15 significant figures
  (-1.004111904519277) at asymmetry 0.6 AND 0.8. Two doses giving one value is a
  SATURATION signature, not a null -- SD-076 and MECH-204 both went untested, and
  SD-076's recorded does_not_support was withdrawn to non_contributory by
  failure_autopsy_V3-EXQ-794_2026-07-22.
  WHY THE 2026-07-20 SMOKE PASSED 6/6 ANYWAY: it used true error mean 0.05, ~13x
  the substrate's real scale, where 0.01 IS a floor with headroom. An absolute
  constant validated at one scale silently became a clamp at another. THE SIGN IS
  NOT THE BUG (the autopsy's candidate cause (b) is ruled out): inflation drove rv
  DOWN correctly, into a floor sitting ABOVE the OFF arm, so the run's
  inflation_lowers_rv precondition saw rv rise.
  New config, both no-op at default so the ON path stays bit-identical until set:
  E3Config.waking_confidence_rv_floor_relative_frac (default 0.0 = use the
  absolute floor; >0 makes the bound that FRACTION of _wci_symmetric_rv_ref, the
  counterfactual un-inflated rv, so it scales with the substrate's own error
  scale and caps overconfidence at 1 - frac),
  E3Config.waking_confidence_rv_floor_mode ("hard" default = the original clip |
  "soft" = softplus approach) and E3Config.waking_confidence_rv_floor_softness
  (0.25, knee width as a fraction of the effective floor; inert while "hard").
  WHY SOFT MATTERS beyond biology (waking confidence drift is softly bounded):
  the softplus is STRICTLY MONOTONIC, so a residual saturation can only SHRINK a
  dose separation, never collapse it to an exact tie -- a mis-set floor then
  degrades to a small LO/HI gap the dose_saturation lint can see, instead of the
  bit-identical arms that made 794 unadjudicable without an autopsy.
  New state: E3TrajectorySelector._wci_symmetric_rv_ref (exposed as
  .wci_symmetric_rv_ref), a symmetric EMA of the same error at the same alpha,
  advanced ONLY on the inflation path. rv minus it IS the inflation produced.
  Smoke at the MEASURED 794 scale (true error 0.0037), 14/14 PASS: the OLD config
  reproduces the defect exactly (LO == HI == 0.01); the repaired config gives
  rv_final 0.0025377 (LO) vs 0.0021031 (HI), dose-ordered, both genuinely
  overconfident (+0.314 / +0.432). Contracts:
  tests/contracts/test_sd076_rv_floor_headroom.py (17), which pins the DEFECT too
  so a "simplification" back to an absolute hard floor fails there, not in a run.
  Validation experiment: V3-EXQ-794a (same-question re-run of the identical 2x2).
  See MECH-173, MECH-204, ARC-016, Q-042, SD-069.

- SD-075: phasic.ema_episode_continuity -- IMPLEMENTED 2026-07-19.
  Module: ree_core/regulators/phasic_surprise_burst.py (extends SD-069, same file).
  Config: `phasic_burst_baseline_continuity` ("reset" default | "carry") and
  `phasic_burst_warmup_ticks` (0 default = OFF, -1 = DERIVE as
  ceil(3 / surprise_ema_decay) = 30 at the default decay, positive = verbatim;
  anything below -1 RAISES so a typo cannot silently mean OFF).
  BIT-IDENTICAL OFF: both defaults are no-op, so every existing consumer of
  phasic_surprise_burst is unchanged (pinned by tests/test_flag_inertness.py and
  SD-075 D1/D5b).
  WHAT IT FIXES: reset() cleared the surprise-EMA cold at every episode boundary,
  and the first waking tick of an episode cannot fire an event (it SEEDS the
  baseline). With surprise_ema_decay 0.1 the baseline needs ~10 ticks, so a seed
  whose episodes are shorter than that never runs against a converged baseline and
  n_event_ticks measures episode LENGTH rather than surprise. V3-EXQ-779b: seed 23
  ran ~6.9-step episodes vs seeds 29/37 at 300 (43x); raising its budget 835 -> 2400
  env steps bought 345 MORE short episodes and phasic_fires_real_events did not move
  (6 vs 10). The MIN just migrated to another short-episode cell despite ~2.85x
  exposure. NO STEP-BUDGET INCREASE CAN REACH THIS -- the binding axis is episode
  length. Capability is RULED OUT, not unproven: burst_level_max = 1.00 in every
  PHASIC-ON cell including both seed-23 cells.
  LEG (a) "carry": reset() preserves the surprise-EMA while STILL clearing the
  envelope, cached delta, and per-episode diagnostics, so no in-flight burst leaks
  across a boundary. This is the biologically faithful setting (LC baseline
  adaptation is continuous across behavioural episodes; a per-episode reset has no
  biological counterpart). "reset" stays the DEFAULT only for backward compatibility
  with runs already recorded against SD-069 -- NOT because it is the better model.
  Declare "carry" deliberately in new work.
  LEG (b) warmup_ticks is ACCOUNTING ONLY -- IT DOES NOT SUPPRESS THE BURST. During
  warmup the regulator still fires and still perturbs the softmax temperature; only
  the counts split into n_events_prewarmup / n_events_converged. Suppression was
  REJECTED (user-confirmed 2026-07-19) because it would change behaviour in the first
  ticks of a lifetime and confound the MECH-063 (ii) retest with a second mechanism
  change. Do not "tighten" it into a suppressor.
  CONSUMERS: read `n_events_converged` (NOT the per-episode `n_events`) over
  `n_converged_ticks`, and declare a cell UNINFORMATIVE when n_converged_ticks is too
  small -- a MIN-across-cells precondition otherwise treats a starved cell as a real
  measurement, which is exactly how 779b was withheld.
  RETEST BRAKE: MECH-063 sub-claim (ii) is pending_retest_after_substrate against
  this build, BUT the re-derive brake fired both halves on MECH-063 (3rd autopsy) and
  refuses another lettered tonic/phasic iteration against the current regulator.
  DO NOT QUEUE V3-EXQ-779c. A new-number redesign of a DIFFERENT mechanism is allowed.
  LIVE SMOKE + A TRAP FOR THE RETEST AUTHOR (25 eps x 7 steps, seed 23, untrained,
  instantaneous_pe): reset/gate-off reproduces the failed precondition EXACTLY at 6
  events; carry/gate-off lifts the SAME exposure to 12, clearing the threshold of 10.
  But with the gate ON, n_events_converged is 0 in BOTH modes -- every event lands
  inside the first 30 ticks and an untrained agent's PE stream then settles. So the
  legs DISAGREE about this cell: (a) alone reads "12, passes", (a)+(b) reads "0
  converged, UNINFORMATIVE". A retest that enables "carry" and reads the RAW count is
  reporting 12 events the gate says are all warmup-era. Read n_events_converged, and
  expect an untrained short-episode cell may honestly have nothing to report -- a far
  more useful failure than "6 vs 10". SD-074 probe_warmup may be complementary here (a
  trained stream may keep firing past tick 30) but that is an UNTESTED hypothesis.
  Smoke: 22 SD-075 contracts pass; SD-069's 8 contracts + flag-inertness unchanged;
  full suite 1772 passed.
  Design doc: REE_assembly/docs/architecture/sd_075_phasic_ema_episode_continuity.md
  Routed by REE_assembly evidence/planning/failure_autopsy_V3-EXQ-779b_2026-07-19.json
  targets[0] (priority 1). See SD-069 (the module it extends), MECH-063.

- SD-074: probe.trained_enough_agent_warmup -- IMPLEMENTED 2026-07-18.
  Module: experiments/_lib/probe_warmup.py (WarmupRecipe / WarmupOutcome / warm_agent /
  measure_action_mass / saturation_summary / assert_state_dict_shareable /
  assert_any_informative / reapply_candidate_capture).
  HARNESS-LEVEL ONLY -- nothing under ree_core/ is touched, no config default moves,
  and no existing script imports it, so backward compatibility is TOTAL BY
  CONSTRUCTION (there is no "disabled by default" flag because there is no ree_core
  surface to disable). Training-regime substrate enrichment for the probe harness,
  the same class as the V3-EXQ-603c precedent.
  WHAT IT FIXES: the 2x2 read-only telemetry-probe family measures gain/bias
  regulators (MECH-320 tonic_vigor score-bias, MECH-313 noise_floor temperature) on
  the E3 pre-commit softmax while running an UNTRAINED agent. A regulator that
  MODULATES a distribution is unobservable when that distribution has no dynamic
  range. Measured by V3-EXQ-777a: D_action_mass_mean pinned at ceiling on 7 of 14
  seeds and at floor on 2 more, informative-seed yield 4 of 14 (28.6%), and
  corr(distance of D from saturation, norm_v_score) = 0.884. That caps power at ~51
  informative seeds needed => ~177 raw seeds / ~31 h, which is infeasible. NOT a
  sampling defect: 777a's sample-driven stopping worked perfectly (all 56 cells
  reached 250 fresh E3 selections, zero starved) and the saturation rate barely moved
  from its starved predecessor.
  Data flow: seed -> caller builds agent+env -> warm_agent() [cache lookup; MISS ->
  goal_pipeline_tier1.warmup_train -> store; HIT -> load_state_dict + restore
  non-buffer E3 scalars] -> read-only de-saturation probe via sample_driven_rollout ->
  WarmupOutcome -> consumer loads the per-seed checkpoint into each of its arms.
  Composes landed modules rather than writing a fourth training loop:
  goal_pipeline_tier1.warmup_train, the baselines/maturation_curriculum cache
  discipline (atomic os.replace, key re-verified on load, agent rebuilt on BOTH paths
  for RNG parity), and sample_driven_rollout for the read.
  Gate policy (user-confirmed): RECORD, do not abort. A still-saturated seed comes
  back saturated=True with its realised mean and the CONSUMER decides; only
  assert_any_informative() raises, and only when ZERO seeds de-saturated. Regime
  (user-confirmed): target_env only, num_episodes the swept parameter;
  regime="curriculum" RAISES rather than silently substituting a second env.
  THREE HAZARDS FOUND AND DEFENDED -- read these before touching the module:
  (H1) e3._running_variance is a PLAIN PYTHON FLOAT (ree_core/predictors/e3_selector.py:291),
  NOT a register_buffer, so it is absent from state_dict -- and it feeds commit_variance
  (:2703), directly upstream of the very probs distribution D_action_mass measures.
  Measured: fresh 0.500 vs warmed 0.0092, a ~54x drift a state_dict-only cache would
  have silently discarded, making cache-HIT and cache-MISS agents commit differently.
  Carried explicitly in the blob with an asserted round trip.
  (H2) All four 777a arms VERIFIED to share an identical 194-key state_dict, because
  TonicVigor and NoiseFloor are zero-parameter non-nn.Module (policy/tonic_vigor.py:225,
  policy/noise_floor.py:129). So ONE warmed checkpoint per seed serves all four arms and
  they differ ONLY in regulator scalars at the e3.select() call site -- which is what
  makes the 2x2 clean. assert_state_dict_shareable() ASSERTS this rather than assuming
  it, so a future arm flipping a flag that DOES construct a module fails loudly instead
  of silently leaving a module at random init.
  (H3) The consumer's generate_trajectories capture is an INSTANCE attribute and is NOT
  in state_dict. If it is lost after load_state_dict, every observe() returns None, the
  cell yields zero samples, and the run self-routes to sample_starvation_requeue -- i.e.
  a lost patch MASQUERADES AS A SAMPLING BUG, the exact misdiagnosis this lineage has
  already made twice. Call reapply_candidate_capture() after every load.
  NON-DESTRUCTIVE MEASUREMENT: torch.no_grad() + agent.eval() stop GRADIENTS but not the
  two other ways state moves while merely stepping -- plain-Python accumulators
  (_running_variance drifted 0.001839 -> 0.001855 over a 25-selection read) and
  REGISTERED PLASTICITY/ELIGIBILITY BUFFERS (e3_selector.py:373-462, updated in place
  under no_grad; two agents identical at load diverged by max|dw| = 2.5e-01 after reads
  of different lengths). measure_action_mass snapshots and restores BOTH, so the read
  leaves the warmed agent bit-identical (verified max|dw| = 0.000e+00).
  D_SAT_LOW 0.05 / D_SAT_HIGH 0.95 are deliberately IDENTICAL to V3-EXQ-777a's constants
  (script:246-247) so success is denominated in the same units as the failure record.
  Do NOT retune them to make a warmup look better.
  Phased training required: yes -- consumer becomes P0 (warmup) -> P2 (frozen read-only
  telemetry). No P1 head-training stage, so the EXQ-166b/c/d joint-training collapse mode
  does not arise, but the frozen-measurement boundary is mandatory.
  MECH-094: N/A (waking-only gradient training; no simulation, no replay, no memory write).
  Smoke test PASS 2026-07-18 (backward compat, H1 absence confirmed, H2 cross-arm
  sharing, cache MISS->HIT exact round trip, de-saturation read collects real samples).
  Validation experiment: PENDING (diagnostic, de-saturation only; see below).
  NOTE the re-derive brake: a V3-EXQ-777b or any re-test of MECH-063 sub-claim (i)
  against an UNTRAINED agent is REFUSED (failure_autopsy_MECH-063-777a-779a-cluster_2026-07-18,
  user-confirmed). The 779 lineage / sub-claim (ii) is EXEMPT and NOT blocked on this.
  CROSS-ARM CONTAMINATION REPAIR (SD-PROBE-WARMUP, three layers, two commits).
  Confirmed by failure_autopsy_V3-EXQ-963_2026-08-30: `_warmup_key` excluded the
  caller's arm-conditional flags, so a 2x2 sharing one (seed, recipe, env) hashed
  IDENTICALLY across all four arms; the first arm to run always minted, and
  `_restore_cached_surface` then stamped that mint's regulator presence over every
  later arm's own, already-correct construction. V3-EXQ-963's T0P0
  (use_noise_floor=False) minted `agent.noise_floor = None` onto T1P0/T1P1, whose
  __init__ had just built a real NoiseFloor -- silently zeroing the whole TONIC
  axis (noise_floor_temp_lift_mean 0.0 on all 20 cells, including all 10
  use_noise_floor=True cells). Nothing raised and nothing logged it:
  assert_state_dict_shareable passes because these regulators are zero-parameter
  and non-nn.Module, and the restore's missing-module logging covers only cached
  paths ABSENT here, never the reverse. ree_core is healthy; this is a harness
  defect throughout.
    (a) PREVENTION, cache key -- IMPLEMENTED 2026-08-30 (d614a9c). `_warmup_key`
        accepts `arm_key` and folds it into the hashed payload; schema v2 -> v3, so
        every pre-fix blob MISSes regardless of caller opt-in.
    (b) PREVENTION, restore -- IMPLEMENTED 2026-08-30 (d614a9c). A cached attribute
        is applied only when its TYPE matches the live HIT agent's own value for
        that name (type(None) is a type, so None-vs-None still matches). SYMMETRIC:
        a cached None never overwrites a live regulator, and a cached regulator is
        never installed over a live None. PRIMARY defence -- it holds even for a
        caller that never passes arm_key.
    (c) DETECTION, post-warmup assertion -- IMPLEMENTED 2026-09-01.
        `assert_arm_regulators_live(agent, arm_key)`, raising `ArmRegulatorMismatch`;
        called by warm_agent itself via `assert_arm_regulators: bool = True`
        (keyword-only), AFTER the HIT/MISS branches converge so both paths are
        covered. Checks each `use_<attr>` flag against `agent.<attr>` in BOTH
        directions (flag set + regulator None; flag clear + regulator present).
        Not redundant with (a)/(b): (a) only separates arms for a caller that
        actually passes arm_key -- no driver in the tree does yet -- and (b)
        protects only attributes ALREADY LIVE on the HIT agent, since
        `_restore_cached_surface` Case 1 restores a never-set attribute verbatim
        with no type check by design. (c) reads the agent warmup actually produced
        and asks whether it is still the arm the caller declared, so it fails on
        any route to the corruption, including routes that do not exist yet.
        The failure it stops is a run that COMPLETES and writes a claim-tagged
        manifest that looks valid while the manipulation is silently absent.
        Backward compatible: a STRICT no-op when arm_key is None (no declared
        flags -> nothing checked, nothing raised), which is what every pre-fix
        caller passes -- pinned as a contract, on a deliberately-corrupted agent.
        Adds NO REEConfig field, so the from_dims three-site rule does not apply.
        Skips, rather than raises on, a flag not named `use_<attr>` and a
        `use_<attr>` with no such agent attribute; both are counted in the returned
        report so an empty check cannot read as a pass.
        Contracts: tests/contracts/test_probe_warmup_arm_regulator_assertion.py
        (11) alongside test_probe_warmup_cache_key_restore.py (9, for (a)+(b)).
  STILL OWED, driver-side, NOT done here (belongs to a NEW EXQ letter via
  /queue-experiment -- V3-EXQ-963 is a burned id and its script must not be edited
  in place): the 963-lineage driver must (i) pass arm_key= to warm_agent, (ii)
  extend its `_fresh_regulator` post-warmup reinstall beyond `agent.phasic_burst`
  to `agent.noise_floor` -- that asymmetry is exactly why the phasic axis survived
  the restore and the tonic axis did not -- and (iii) record NoiseFloor.get_state()'s
  n_waking_calls / last_n_simulation_skips, which the substrate was already
  computing and which separate "never called" from "called under simulation_mode".
  See SD-074, SD-PROBE-WARMUP, MECH-063, MECH-320, MECH-313, ARC-066, and
  REE_assembly/docs/architecture/sd_074_probe_warmup_trained_enough_agent.md.

- SD-070 ADOPTION in the _train_all_on_agent driver family -- IMPLEMENTED 2026-07-20.
  Module: experiments/_lib/zworld_p0_warmup.py (run_zworld_p0 + resource_prox_target).
  THE DEFECT THIS CLOSES. SD-070 shipped 2026-07-18 as a trainer, but NO DRIVER CALLED IT.
  The P0 warmup shared by x728/x734/x737/x742 builds three optimizer groups -- e2, the
  lateral-PFC bias head, the OFC devaluation head -- and NONE covers a single latent_stack
  parameter, so split_encoder.world_encoder was never stepped and z_world stayed a FROZEN
  RANDOM PROJECTION for entire campaigns, with no error and no warning. Measured on two
  INDEPENDENT drivers: V3-EXQ-737a 0 of 61 latent_stack tensors changed (world_encoder 0 of
  4) at p0_episodes=200; V3-EXQ-728 the same signature on its OWN _train_all_on_agent copy,
  3 of 3 seeds -- so the defect was per-copy, not confined to the shared path.
  THE FIX IS NOT "ENABLE PRESCRIBED P0" -- that is refuted in-corpus (SD-009 CE + SD-018 MSE
  online at batch=1 COLLAPSES z_world to PR ~1.06). Per the V3-EXQ-783 adjudication the fix
  needs (a) a gradient path reaching latent_stack and (b) a target the world channel
  determines; SD-070 supplies both, and this landing wires it in.
  Config: `_train_all_on_agent(..., zworld_p0_episodes=N, zworld_p0_env=..., 
  zworld_p0_dry_run=...)`. Default 0 = EXACTLY the prior behaviour, bit-identical: no extra
  tensor, no optimizer group, no env construction, no RNG draw. Drivers set
  ZWORLD_P0_EPISODES=60 (SD-070's validated operating point,
  exq783_zworld_granularity.OFF_P0_ENCODER_EPISODES).
  Data flow: world_obs -> [P0a buffer] -> ZWorldP0Trainer -> world_encoder +
  world_precision_logit -> P0b e2 warmup (now over a MEANINGFUL z_world) -> P1.
  ORDERING IS LOAD-BEARING: e2 regresses on z_world, so the encoder trains BEFORE the e2
  warmup -- training it after would leave e2 fitted to the random projection, i.e. the same
  defect one phase later.
  RNG NEUTRALITY IS LOAD-BEARING, NOT HYGIENE. ZWorldP0Trainer seeds its own Generator for
  shuffling but builds its auxiliary heads with nn.Linear, which draws from the GLOBAL torch
  RNG. Unguarded, merely turning P0a on would shift every subsequent draw, confounding "the
  encoder is now trained" with "the RNG stream moved". run_zworld_p0 snapshots and restores
  the global torch + numpy streams, and the rollout runs on a DEDICATED env instance so the
  training env's layout sequence is untouched.
  FINGERPRINTS UPDATED (reuse correctness): `zworld_p0_episodes` was added to x734's and
  x728's config slices and to exq742_mech457_bias_head_baseline.off_path_config_slice. An
  arm warmed with SD-070 is a DIFFERENT arm from a frozen-random-projection arm; without
  this a pre-fix banked arm would falsely cache-HIT a post-fix consumer and silently compare
  a trained-encoder treatment against an untrained control. CONSEQUENCE: the banked
  V3-EXQ-742-m bias_head_baseline mint no longer matches the 742 consumer and must be
  RE-MINTED at zworld_p0_episodes=60.
  Scope: both _train_all_on_agent definition sites (x734:332 shared by 737/742/fanout/
  baseline; x728:522 own copy), plus _lib/mech457_fanout.warmup_zworld(zworld_p0=...) and
  _lib/baselines/exq742_mech457_bias_head_baseline.run_off_cell(zworld_p0_episodes=...).
  In 742 the bias_head_baseline OFF control carries the SAME P0a setting as the AC arms --
  otherwise the ON/OFF contrast confounds the actor-critic treatment with encoder training.
  VERIFIED both definition sites, 3 P0a episodes: OFF reproduces the defect exactly
  (latent_stack 0/61, world_encoder 0/4, max_delta 0.0, guard REFUSES); ON trains
  world_encoder 4/4 (7/61 latent_stack = 4 encoder + world_precision_logit + 2 prox-head,
  z_self UNTOUCHED per SD-070's C5 contract), max_delta 6.03e-03, guard PASSES. Driver
  dry-runs: 734 guard green on all 4 rungs (was red); 737 readiness_met True (was False);
  742 clean; 728 arm_green=True seeds_failed=0 (was 3 of 3 failed) and outcome PASS.
  Phased training: P0a -> P0b -> P1 -> P2 unchanged and still mandatory. MECH-094 N/A
  (trains on live observations, writes nothing to memory in any non-waking state).
  Validation experiment: V3-EXQ-787a (see the queue entry).
  Source: REE_assembly/evidence/planning/substrate_queue.json -> sd_zworld_warmup_optimizer_group,
  failure_autopsy_V3-EXQ-737a_2026-07-20.json,
  REE_assembly/evidence/planning/zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md section 6c/6d.
  See SD-070 (the recipe), _lib/zworld_encoder_guard.py (the detector this remedies),
  MECH-457, INV-088, Q-002.

- SD-070: latent.zworld_p0_anticollapse_recipe -- IMPLEMENTED 2026-07-18.
  Module: ree_core/latent/zworld_p0.py (ZWorldP0Config + ZWorldP0Trainer, plus the
  pure functions scene_structure_targets / variance_covariance_penalty /
  balanced_class_weights / entity_presence_mask / chebyshev_offsets).
  WHAT IT REPLACES: the P0 the substrate prescribed for training the z_world encoder
  (SD-009 event-contrastive CE + SD-018 resource-proximity MSE, online at batch=1;
  named in substrate_queue.json:971 and ree_core/predictors/e2_world.py:42-54) does
  NOT produce a trained encoder -- it COLLAPSES z_world. Measured 2026-07-18 at
  world_dim=128, seed 42, 40 eval episodes: untrained participation_ratio 9.21 /
  contrast_ratio 0.1222, vs 1.06 / 0.0726 after the prescribed P0. PR ~1 = collapsed
  onto one effective dimension, so every downstream comparator built on it is vacuous
  (the MECH-353 / V3-EXQ-642 lesson).
  THREE MEASURED FAULTS. (1) The SD-009 target is unlearnable from the channel its
  loss reads: transition_type is a property of the TRANSITION (t-1 -> t) while z_world
  is a static single-frame encoding, and an MLP-128 probe on RAW world_obs with no
  encoder in the path scores AT OR BELOW CHANCE (lift -0.014 3-class, -0.060 on a
  repaired 6-class map), while the same label probes at +0.240 / +0.427 from the BODY
  delta that SD-005 routes to z_self. So this is a WIRING fault, not a labelling one --
  class rebalancing cannot recover absent information. (2) Nothing penalises collapse:
  a ~95% class-0-saturated CE plus one scalar are both served by a 1-D code. NOTE the
  collapse is NOT the multiplicative precision gate (the obvious suspect): its sigmoid
  moves only 0.4966-0.5074 and the final layer stays full-rank (sv PR ~55) -- the
  encoder's FUNCTION collapses while its weights look healthy. (3) The loop is online
  at batch=1, leaving variance/covariance statistics undefined.
  RECIPE: static scene-structure grounding targets derived from world_obs ALONE
  (hazard/resource presence + bucketed nearest-Chebyshev distance; probed 0.943-0.965
  balanced accuracy from raw world_obs, vs chance 0.5/0.5/0.333/0.333) + class-balanced
  CE + a VICReg variance/covariance penalty + an optional world_obs reconstruction head
  + mini-batching over a rollout buffer. The COVARIANCE term is the participation-ratio
  lever; the variance hinge alone cannot raise PR (correlated dims at unit std still
  occupy one effective dimension), which is why covariance_weight defaults to 50 rather
  than VICReg's published ratio (measured w_cov=0.04 -> PR 1.80, w_cov=50 -> PR 4.02).
  Data flow: world_obs -> [buffer] -> world_encoder -> *sigmoid(world_precision_logit)
  -> {grounding heads, SD-018 prox head, recon head, var/cov penalty}. Trains exactly
  world_encoder + world_precision_logit (the set V3-EXQ-783's weight-delta readiness
  check watches); topdown + alpha_world smoothing are applied at sense() time and carry
  no P0 gradient.
  BIT-IDENTICAL OFF BY CONSTRUCTION, NOT BY FLAG: SD-070 adds NO field to
  LatentStackConfig, NO head to SplitEncoder and NO method to REEAgent. It operates on
  an existing LatentStack from outside and the auxiliary heads belong to the trainer,
  so nothing runs unless an experiment explicitly constructs a ZWorldP0Trainer and
  there is NO flag that can be left in the wrong state (pinned by contracts C6).
  Contract-pinned NOT to touch the z_self path (training it would confound every
  downstream self-stream result run after a P0).
  Phased training: this is P0 only; P1 (E2WorldForward) still trains on stop-gradient
  z_world with the encoder optimiser NOT stepped, P2 measures. MECH-094 N/A (trains on
  live observations, writes nothing to memory in any non-waking state).
  VALIDATION at config defaults, world_dim=128, 3 seeds: PR 9.21->5.19 (s42),
  6.63->5.41 (s43), 8.56->4.64 (s44). Against the V3-EXQ-783 anti-collapse gate computed
  as that harness computes it: retained_fraction 5.079/8.132 = 0.625 (needs >=0.50) and
  absolute 5.079 (needs >=2.0), both PASS; contrast ratio raised 3/3 (mean 0.137->0.238,
  clearing the untrained 0.13-0.15 band); world-path tensors changed 4/4 on every trained
  arm. DISCRIMINATIVE, not vacuously un-collapsed: held-out balanced-accuracy lift +0.23
  to +0.47 on all four grounding heads across all three seeds. The trainer ALWAYS reports
  this readout, because an anti-collapse gate can otherwise be satisfied by a regulariser
  that holds PR up while learning nothing. Smoke: 31 SD-070 contracts pass; full suite
  1590 pass (0 regressions).
  SD-009 IS ROUTED AROUND, NOT ADJUDICATED. The target/channel mismatch measured here
  concerns SD-009's own validity and is recorded for a future /governance cycle in
  REE_assembly/evidence/planning/sd009_event_contrastive_channel_mismatch_2026-07-18.md.
  NO SD-009 status, confidence, or evidence weighting was changed. Note claims.yaml:7319
  records EXQ-020 PASS on the SD-009 mechanism (selectivity_margin=0.882) -- a cosine
  separation statistic, NOT the decodability statistic measured here; reconciling the two
  is the governance question, and the reconciliation offered in that artifact is a
  hypothesis, not a finding.
  Validation experiment: V3-EXQ-783 (see Step 8).
  Design doc: REE_assembly/docs/architecture/sd_070_zworld_p0_anticollapse_recipe.md
  See SD-005 (the encoder it trains), SD-018 (retained leg), SD-031 / E2WorldForward
  (the consumer requiring a trained encoder), SD-009 (routed around), MECH-353, Q-002.

- SD-024: hippocampal_module.da_modulated_rbf_density -- IMPLEMENTED 2026-07-16.
  Built as the DIAGNOSTIC instrument that RESOLVES MECH-232 (DA representational
  expansion), NOT a feature gated behind it. Modules: ree_core/residue/field.py
  (RBFLayer, ResidueField) + ree_core/hippocampal/module.py (HippocampalModule).
  DA at reward encounters allocates a local CLUSTER of RBF centers (representational
  expansion) + optional finer per-center bandwidth on the BENEFIT terrain -- higher
  information density / sharper place fields -- WITHOUT writing an explicit
  positive-valence gradient. Modulates ONLY benefit_rbf_field; harm/safety fields keep
  single-center standard-bandwidth allocation (MECH-233 asymmetry preserved).
  RBFLayer: per_center_bandwidth buffer (default off -> byte-identical scalar reads);
  add_residue_cluster() [n = 1 + int(da_signal * allocation_scale) jittered centers,
  per-center bandwidth floored at 0.5*base]; compute_local_density() [WEIGHT-INDEPENDENT
  proximity-weighted active-center count]. ResidueField: benefit field built per-center
  + optional da_benefit_num_centers capacity when master on; accumulate_benefit(...,
  dopamine_signal=) routes to the cluster path; compute_benefit_density() wrapper.
  HippocampalModule.compute_representational_density() read-through (SD-025 hook).
  Config: ResidueConfig.use_da_modulated_rbf_density (master, default False),
  da_allocation_scale (0.0), da_jitter_radius (0.1), da_bandwidth_narrowing (0.0),
  da_benefit_num_centers (None). Data flow: reward contact -> accumulate_benefit(
  dopamine_signal=benefit*drive) [MECH-094 hypothesis_tag gate] -> add_residue_cluster
  -> benefit_rbf_field density up -> compute_local_density (read) -> [SD-025 curiosity
  drive, downstream] -> approach. THE MECH-232 DISCRIMINATOR: density is weight-INDEPENDENT,
  so DA raises density even when evaluate_benefit (weight sum) is held flat -> approach
  from representational QUALITY alone, not a valence tag. Backward compatible: all
  defaults no-op -> bit-identical OFF (full pytest tests/ 1475 passed). Phased training:
  NOT REQUIRED (no encoder head; allocation + read only). MECH-094: DA expansion inherits
  accumulate_benefit's hypothesis_tag gate (replay cannot expand).
  13 contracts in tests/contracts/test_sd024_da_modulated_rbf_density.py.
  Validation experiment: V3-EXQ-766 (MECH-232 DA-ON vs DA-OFF: leg-1 representational
  expansion at reward locations; leg-2 CRUX approach-without-explicit-gradient). PASS
  promotes MECH-232 candidate->provisional; FAIL refutes.
  Design doc: REE_assembly/docs/architecture/sd_024_da_modulated_rbf_density.md
  See MECH-232 (the mechanism it resolves), MECH-233 (asymmetry preserved), ARC-057
  (the approach-emergence architecture), SD-025 (curiosity_drive it unblocks downstream).

- SD-025: hippocampal_module.curiosity_drive -- IMPLEMENTED 2026-07-16.
  The SECOND component of ARC-057 (approach-emergence): an information-seeking bias in
  hippocampal CEM trajectory scoring that favours regions of higher REPRESENTATIONAL
  DENSITY in the SD-024 benefit RBF map. Modules: ree_core/hippocampal/curiosity.py
  (FamiliarityTracker) + ree_core/hippocampal/module.py (HippocampalModule).
  novelty(z) = density(z) * (1 - familiarity(z)); _score_trajectory subtracts
  curiosity_weight * mean_over_trajectory(novelty) from the terrain score (CEM
  minimises -> lower = better, same convention as wanting_weight). density =
  compute_representational_density (the SD-024 WEIGHT-INDEPENDENT active-center count ->
  the drive follows representational QUALITY, not a positive-valence gradient).
  familiarity = FamiliarityTracker, a proximity-weighted visit-count EMA (soft
  visitation KDE, clamped [0,1]) that rises on revisit so novelty decays there
  (anti-perseveration); it is instantiated ONLY when curiosity_weight > 0.
  Config (HippocampalConfig): curiosity_weight (master, default 0.0 -> tracker never
  built, scoring untouched, update_familiarity a no-op -> bit-identical OFF),
  familiarity_ema_alpha (0.01), use_curiosity_familiarity (True; set False for the
  density-only ablation), familiarity_bandwidth (1.0). Data flow: benefit RBF density
  (SD-024) -> compute_representational_density [read-only] -> novelty=density*(1-familiarity)
  -> _score_trajectory terrain_score -= curiosity_weight*novelty -> CEM elite selection.
  Waking visit -> agent.sense() -> update_familiarity(z_world, is_waking=not hypothesis_tag).
  Backward compatible: curiosity_weight=0.0 -> bit-identical OFF (full pytest tests/ 1488
  passed). Phased training: NOT REQUIRED (no encoder head; read + EMA state only).
  MECH-094: familiarity is real memory state -> update_familiarity advances on WAKING
  visits ONLY (agent gates on hypothesis_tag, identical to the MECH-314a visitation
  buffer); the density read during CEM scoring writes no memory.
  7 contracts in tests/contracts/test_sd025_curiosity_drive.py.
  SCOPE NOTE: the SUBSTRATE is buildable now on SD-024, but the full ARC-057 ecological
  approach-emergence claim (SD-024 x SD-025 interaction) is ENV-CONSTRAINED -- the
  CausalGridWorld cannot faithfully test it (a cell is a cell; nothing more to discover
  at higher resolution; see claims.yaml ARC-057 SUBSTRATE CONSTRAINT). The validation is
  scoped to the DRIVE MECHANISM (does curiosity propagate into CEM selection toward
  higher-density regions? -- the propagation MECH-111's broken broadcast-novelty->E3 path
  could not achieve, EXQ-141b/590a), NOT the interaction claim.
  Validation experiment: V3-EXQ-767 (curiosity ON vs OFF; leg-1 propagation: CEM
  selection biased toward the dense cluster only when ON; leg-2 anti-perseveration:
  familiarity discount attenuates density-following on revisit).
  Design doc: REE_assembly/docs/architecture/sd_024_da_modulated_rbf_density.md#curiosity-drive
  See SD-024 (the density substrate it reads), ARC-057 (the approach architecture, env-
  constrained), MECH-111 (curiosity/novelty-drive grounding; distinct broken path).
