## SD-036 + MECH-279: GABAergic Cross-Stream Decay + PAG Freeze-Gate (2026-04-22)
- SD-036: regulators.gabaergic_cross_stream_decay -- IMPLEMENTED 2026-04-22.
  Module: ree_core/regulators/gabaergic_decay.py (GABAergicDecayRegulator,
  GABAergicDecayConfig, StreamRegistration). Regulator-layer substrate
  (NOT per-stream update rule): a single broadly-projecting tonic GABAergic
  decay applied across multiple registered latent streams in parallel.
  Decay formula:
    z_s(t+1) = z_s(t) * exp(-tau_s * gaba_tone(t))
  with per-stream baseline tau and a global gaba_tone multiplier in
  [0, 2] (default 1.0). gaba_tone > 1.0 = benzo-analog (faster decay,
  easier exit from committed states); gaba_tone < 1.0 = withdrawal /
  chronic-stress analog (slower decay); gaba_tone = 0.0 = decay
  suspended.
  Default coverage (tau values from design doc):
    z_harm   tau=0.05  (~20-step half-life)  -- SD-010 sensory harm
    z_harm_a tau=0.02  (~50-step half-life)  -- SD-011 affective harm
    z_beta   tau=0.03  (~30-step half-life)  -- MECH-090 precision/affective
  Drive accumulator (SD-012) intentionally NOT covered -- the homeostatic
  override mechanism (separate, V4-or-late-V3) provides drive dynamics.
  Suspend-on-input gate: per-stream input_threshold; when |z(t)-z(t-1)|
  exceeds threshold, decay is skipped for that tick (the input drives
  the update). Default 0.0 = always decay.
  Decay is OUT-OF-PLACE (detach + scalar multiply + setattr): an in-place
  mul_() on encoder outputs breaks autograd version tracking when those
  outputs are concurrently consumed by SD-018 resource_proximity_head /
  SD-011 harm_accum_head aux losses. Out-of-place is required for the
  EXQ-471 training pipeline.
  Config: REEConfig.use_gabaergic_decay (bool, default False). 14 sub-
  knobs in REEConfig.from_dims: gaba_tone (1.0), gaba_tone_min (0.0),
  gaba_tone_max (2.0), per-stream tau (gaba_tau_z_harm_s/a/beta),
  per-stream coverage flags (gaba_decay_z_harm_s/a/beta), per-stream
  input thresholds (gaba_input_threshold_z_harm_s/a/beta).
  Agent wiring: instantiated in REEAgent.__init__ when master switch is
  on; register_default_streams() called immediately. tick() invoked in
  agent.sense() right after LatentStack.encode() and BEFORE AIC, BLA/CeA,
  salience coordinator, etc. -- so all downstream consumers see the
  decayed latent state on the same tick (no one-step lag). reset() called
  from REEAgent.reset() per-episode.
  Backward compatible: use_gabaergic_decay=False by default; agent.gabaergic_decay
  is None and tick wiring is a no-op. Existing experiments unaffected.
  No trainable parameters. No phased training needed.
  Biological basis: GABAergic system as broadly-projecting tonic
  inhibitory neuromodulator (Vogt 2005, Sohal & Rubenstein 2019).
  Decay-as-regulator-layer (not per-stream update) is the architectural
  commitment: a single GABA tonic value modulates many cortical and
  subcortical sites in parallel. SD-036 implements this commitment.
  MECH-094: simulation_mode=True path returns input unchanged and does
  not advance counters (replay / DMN content not subject to waking decay).
  Validation experiment: V3-EXQ-475 queued (matched re-run of EXQ-471
  with use_gabaergic_decay=True + use_pag_freeze_gate=True; not a
  supersede; EXQ-471 retained as no-decay baseline).
  Design doc: REE_assembly/docs/architecture/sd_036_gabaergic_decay_regulator.md
  See SD-036, MECH-279, MECH-094, SD-010, SD-011, MECH-090, SD-012.

- SD-036 harm-stream decay RECURRENCE -- IMPLEMENTED 2026-07-31.
  THE DECAY ABOVE HAD NO TEMPORAL AUTHORITY OVER THE TWO HARM STREAMS UNTIL
  THIS LANDED, AND THAT SILENTLY VACUATED THE ENTIRE PRE-REGISTERED SD-036
  EXPERIMENT PROGRAMME. Read this before designing any SD-036 experiment.
  Defect: LatentStack.encode() produced z_harm (SD-010) and z_harm_a (SD-011)
  as PURE FEEDFORWARD encodes of the current observation. The regulator ticks
  in sense() AFTER that encode and the agent stores the decayed value as
  _current_latent -- but the next encode() overwrote it from the encoder
  instead of reading it back. So for those two streams z_s(t+1) was not a
  function of z_s(t) AT ALL: the regulator degenerated to a one-step constant
  rescale, discarded every tick. z_beta was never affected -- it already
  blends with prev_state (stack.py, alpha_shared) -- and that contrast is
  what localised the defect.
  Measured on the pre-fix substrate (one identical recorded observation tape
  replayed into agents differing ONLY in gaba_tone; 60 steps, seed 0,
  471-lineage env), peak-normalised trajectory max-deviation vs tone=1.0:
    z_harm    <= 2.0e-07 (shape BIT-IDENTICAL); raw ratio at tone=0.0
              1.05127107 == exp(0.05) to 8 significant figures
    z_harm_a  <= 1.4e-07 (shape BIT-IDENTICAL); raw ratio 1.02020133
              == exp(0.02)
    z_beta    2.0e-02 .. 2.9e-02 -- genuinely compounds, does NOT match a
              single-tick exp(-tau * delta_tone)
  Post-fix the same measurement gives z_harm 1.3e-02 .. 1.8e-02 and z_harm_a
  5.0e-02 .. 7.1e-02, with z_beta unchanged to the digit.
  WHY THIS MATTERED SCIENTIFICALLY: any SCALE-FREE DV is exactly invariant to
  a constant rescale. harm_norm_sustain_ratio (= mean/peak, in
  experiments/_lib/goal_pipeline_tier1.py) is the DV of the registered SD-036
  dose-response falsifier, and its spread across the whole
  gaba_tone {0.3, 0.5, 1.0, 1.5, 2.0} sweep was 8.6e-08 -- structurally
  vacuous, not merely underpowered. A fixed-ABSOLUTE-threshold DV would
  instead have shown a clean monotone dose-response, i.e. a confident-but-
  wrong confirmation of the trivial rescale. Post-fix the spread is 1.4e-03
  (z_harm) and 1.0e-01 (z_harm_a, monotone decreasing in tone).
  Fix: a prev_state blend in encode(), matching the substrate's established
  idiom for a stateful stream:
    z_s(t) = alpha_s * encode(obs_t) + (1 - alpha_s) * z_s_decayed(t-1)
  Composed with the regulator's end-of-tick rescale the stream is a leaky
  integrator with pole (1 - alpha_s) * exp(-tau_s * gaba_tone), so gaba_tone
  moves BOTH the relaxation time constant and the steady-state gain -- the
  trajectory SHAPE, not just its scale.
  THE DECAY ARITHMETIC STAYS IN THE REGULATOR. encode() supplies only the
  recurrence, so SD-036's architectural commitment ("multi-target regulation
  lives at the regulator, not in each target") is preserved -- do not move
  exp(-tau*tone) into the stack.
  Applied to the LatentState FIELD, not to a specific encoder output, so it
  also covers the MECH-099 lateral head that SD-010 overrides -- the same
  stream-name keying the regulator uses. harm_accum_pred (the SD-011 aux head
  target) is a separate encoder output and is NOT blended, so
  compute_harm_accum_loss keeps its full gradient path.
  WHY NOT the design doc's literal "input drives OR decay" switch: it
  presumes an EVENT-like input. harm_obs is a dense continuously-present
  field view (hazard field + resource field + exposure). Measured over 60
  steps, ||harm_obs|| ranged 2.32 .. 5.21 and was NEVER zero, so there is no
  "absence of input" tick to switch on; a hard switch would either never fire
  or permanently starve the stream. The regulator's existing suspend-on-input
  gate (gaba_input_threshold_*) remains the soft form of that mechanism.
  Related measurement worth knowing: harm_encoder(zeros) has norm 0.462 and
  affective_harm_encoder(zeros, zeros) 0.332. So the "~0.7 pinned z_harm_norm"
  in the V3-EXQ-471 origin exemplar is substantially an ENCODER FLOOR
  response to the ambient hazard field, not pure temporal persistence -- the
  design doc's root-cause story is at best incomplete. Flagged for governance;
  NOT acted on here.
  Config: LatentStackConfig.gaba_harm_state_recurrence (bool, default False)
  is MIRRORED from REEConfig.use_gabaergic_decay by REEAgent.__init__ -- do
  not set it directly. Mirroring at the agent (rather than in from_dims) is
  deliberate: an experiment that flips config.use_gabaergic_decay AFTER
  construction (the idiom V3-EXQ-475 uses for several fields) would otherwise
  get the regulator without the recurrence, i.e. silently back to the vacuous
  rescale. Four from_dims knobs, all LatentStackConfig-scoped:
  gaba_recurrence_z_harm_s / _a (bool, both True -- set one False for the
  decay-without-recurrence control arm) and gaba_state_alpha_z_harm_s (0.5) /
  gaba_state_alpha_z_harm_a (0.2). alpha 1.0 = legacy feedforward;
  0.0 = pure autoregression (input never enters).
  Alpha rationale: sensory-discriminative harm (SD-010, spinothalamic
  analogue) tracks stimulus intensity so it stays input-led; affective-
  motivational harm (SD-011, spinoparabrachial analogue) integrates, matching
  the design doc's "emotional residue persists longer" rationale for its
  slower tau. Measured tradeoff -- lower alpha gives stronger decay authority;
  at alpha_s=0.3 the z_harm sustain-ratio response goes non-monotone (spread
  3.7e-04, noise-level), at 0.5 and 0.7 it is monotone. 0.5/0.2 keeps all
  three registered streams in the same responsiveness band (z_harm 1.8e-02,
  z_beta 2.9e-02, z_harm_a 7.1e-02), which is what observable #3
  (multi-stream cluster) needs in order to discriminate regulator-layer decay
  from per-stream decay.
  BACKWARD COMPATIBLE, verified differentially rather than asserted: with
  use_gabaergic_decay=False the full latent trajectory is BIT-IDENTICAL to
  the pre-change substrate (max|diff| = 0.000e+00 on z_harm, z_harm_a and
  z_beta over the 60-step tape, run against a detached worktree at the
  pre-change commit).
  NOT backward compatible for gaba_tone=0.0 WITH the master switch ON, by
  design: the regulator suspends decay at tone 0 but the recurrence is still
  active, so the stream is a leaky integrator with pole (1 - alpha) rather
  than the legacy feedforward one. tone=0 is therefore no longer the
  "legacy" control arm -- use use_gabaergic_decay=False for legacy, or
  gaba_recurrence_z_harm_s/_a=False to isolate decay from recurrence.
  Contracts: tests/contracts/test_sd_036_gabaergic_decay.py C11-C16. These
  step a REAL REEAgent through sense() over a fixed observation tape -- the
  thing no SD-036 test did before, and precisely why the defect survived.
  C1-C10 all exercised the regulator against a synthetic _Latent() stand-in
  re-ticked IN PLACE, where compounding holds trivially. C13 is the
  differential control: ablating one stream's recurrence must restore that
  stream's pre-fix vacuity (< 1e-05) while the other stays live, which is
  what stops C11/C12 passing for unrelated reasons. Verified to FAIL against
  the pre-change substrate (C11, C12, C14 all fail there).
  Validation experiment: NOT queued by this landing -- see the SD-036 entry
  above and the design doc's "Predicted observables" 1-3, all three of which
  were unbuildable before this and are now buildable.
  Design doc: REE_assembly/docs/architecture/sd_036_gabaergic_decay_regulator.md
  See SD-036, SD-010, SD-011, MECH-090, MECH-099, MECH-279, MECH-094.

- MECH-279: pag.freeze_gate -- IMPLEMENTED 2026-04-22.
  Module: ree_core/pag/freeze_gate.py (PAGFreezeGate, PAGFreezeGateConfig,
  PAGFreezeGateOutput). Periaqueductal-gray-analog committed-freeze gate.
  Freeze is a *committed* behavioural state -- sustained motor immobility
  plus elevated autonomic arousal -- with its own duration and exit
  criterion. Biologically PAG-gated; freeze-promoting cells are themselves
  GABAergic (so SD-036 gates BOTH entry and exit).
  Logic:
    duration_above_threshold(t) -- ticks since z_harm_a first crossed
      duration_input_threshold (defaults 0.4); resets when z drops below
      that threshold OR on release. Increments only while gate inactive
      (per-cycle "fresh accumulation" semantic; each commit requires a
      new run-up).
    freeze_commit(t) = (z_harm_a(t) * duration_above_threshold(t))
                       > theta_freeze (default 2.0); strict-greater so
      e.g. z=1.0 sustained at duration=2 (product=2.0) does NOT commit.
    exit_threshold(t) = theta_freeze * gaba_tone(t)
    freeze_release    = active AND z < exit_threshold AND
                        ticks_in_freeze >= min_freeze_duration; OR
                        ticks_in_freeze >= max_freeze_duration (cap).
  Action constraint: when freeze_active, REEAgent.select_action()
  replaces the chosen action with a no-op one-hot (action class
  noop_class=0 by convention; matches action shape/dtype/device).
  Tick wired AFTER beta_gate.propagate() and BEFORE _last_action assignment
  so subsequent record_transition / E2_harm_a forward steps see the no-op.
  Config: REEConfig.use_pag_freeze_gate (bool, default False). 4 sub-
  knobs: pag_theta_freeze (2.0), pag_duration_input_threshold (0.4),
  pag_min_freeze_duration (0 -- no minimum), pag_max_freeze_duration
  (0 -- no cap; set positive for forced-release safety in smoke tests).
  Backward compatible: use_pag_freeze_gate=False by default; agent.pag_freeze_gate
  is None. Existing experiments unaffected.
  No trainable parameters. Pure arithmetic over scalars + small counters.
  No phased training needed.
  Biological basis: descending inputs from amygdala / hypothalamus /
  medial PFC converge on PAG freeze-promoting cells; freeze termination
  requires GABAergic inhibition to wane. Same neurotransmitter system
  gates BOTH entry (PAG freeze-cell commitment) and exit (SD-036 decay
  returning z_harm_a below exit_threshold). Architectural prediction:
  GABA agonists treat freeze catatonia (clinical observation as
  architectural consequence, not empirical add-on).
  MECH-094: simulation_mode=True path returns zeroed PAGFreezeGateOutput
  without updating internal state (replay / DMN content must not commit
  the agent into behavioural freeze).
  Validation experiment: V3-EXQ-475 (combined SD-036 + MECH-279
  diagnostic; under default theta_freeze=2.0 the gate is expected to be
  silent on EXQ-471 dynamics, but is wired so the substrate is exercised
  end-to-end).
  See MECH-279, SD-036, SD-011, MECH-090, MECH-094.
