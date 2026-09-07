## SD-047: Multi-Source Environmental Dynamics (2026-05-03)
- SD-047: environment.multi_source_dynamics -- IMPLEMENTED 2026-05-03.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorld /
  CausalGridWorldV2). Three concurrent stochastic event sources at distinct
  spatial / temporal scales, each agent-independent, layered onto the
  existing SD-022 / SD-029 hazard substrate. Substrate-ceiling unblock for
  MECH-095 TPJ agency-detection comparator (V3-EXQ-506 C4-only-PASS pattern,
  2026-05-03). Lit-anchor: 18 PubMed entries (Asai 2016 non-monotonic agency
  S/N; Sawtell 2010 cerebellar cancellation; Pitcher & Ungerleider 2021
  lateral cortex network; Woo/Spelke 2023 falsifier; passivity cluster
  Blakemore/Frith 2000, Synofzik 2008, Stirling 2001, Gallagher 2004,
  Shamanna 2023, Brandt 2017, Ganos 2015, Lyndon 2026, Seth & Friston 2016,
  Jeganathan & Breakspear 2021, Nassar 2021, Jardri & Deneve 2013, Ward 2010).
  SD-047 lit_conf=0.841.
  Three sources:
    Source 1 (weather field): AR(1) coarse-grid additive perturbation on
      hazard_field. Continuous, smooth, autocorrelated, agent-independent.
      Per-cell signature for cerebellar-style cancellation tests (MECH-098).
      Stationary AR(1) form: x_{t+1} = alpha*x_t + sqrt(1-alpha^2)*sigma*N(0,1).
      Variance bounded at sigma^2 across long episodes.
    Source 2 (transient events): Poisson appear / disappear of transient
      hazard cells. Discrete, spatially pointwise, short-lived,
      agent-independent. Tracked separately from self.hazards
      (self._transient_hazards) for bookkeeping; underlying cell still
      registered in self.hazards so proximity field treats it as a hazard.
    Source 3 (background drift): n_drift_sources mobile single-cell hazard-
      analog objects with random_walk / linear_drift / levy_walk dynamics.
      Discrete, mobile, autocorrelated, agent-independent. Tracked in
      self._drift_sources; same dual-list bookkeeping as transients.
  Config (CausalGridWorld __init__ kwargs, env-only -- not surfaced through
  REEConfig.from_dims, matching SD-023 / SD-029 precedent for env-only SDs):
    multi_source_dynamics_enabled (bool, default False) -- master switch.
    multi_source_intensity_scale (float, default 1.0) -- 4-arm noise-sweep
      lever (OFF / 0.25 / 1.0 / 4.0); scales weather sigma, transient
      p_appear, and drift move probability uniformly.
    weather_field_enabled (bool, default False) -- per-source switch.
    weather_super_cells (int, default 4) -- coarse AR(1) grid resolution.
    weather_alpha_ar1 (float, default 0.95) -- temporal autocorrelation.
    weather_sigma (float, default 0.05) -- per-cell perturbation magnitude.
    transient_events_enabled (bool, default False) -- per-source switch.
    transient_p_appear (float, default 1e-3) -- per-tick per-cell appearance.
    transient_p_disappear (float, default 0.1) -- mean lifespan ~10 ticks.
    transient_intensity (float, default 1.0) -- reserved for harm scaling.
    background_drift_enabled (bool, default False) -- per-source switch.
    n_drift_sources (int, default 1) -- count.
    drift_policy (str, default "random_walk") -- one of random_walk /
      linear_drift / levy_walk.
  Data flow (step()): existing logic (move, harm, contamination, SD-022
    limb damage) -> SD-029 _inject_external_hazard -> _drift_hazards
    (legacy env_drift) -> [if multi_source_dynamics_enabled] _step_weather_field
    -> _step_transient_events -> _step_background_drift -> _compute_proximity_fields
    if any source perturbed hazard layout / weather. Existing SD-029 path is
    untouched; SD-047 is purely additive.
  info dict tags (always present, 0 / False when disabled):
    multi_source_dynamics_enabled, multi_source_intensity_scale,
    multi_source_weather_step_delta, multi_source_n_transient_appear,
    multi_source_n_transient_disappear, multi_source_n_transient_active,
    multi_source_n_drift_moved, multi_source_n_drift_active,
    multi_source_n_env_events (cumulative env-caused per tick),
    multi_source_n_agent_events (cumulative agent-caused per tick).
    The ratio multi_source_n_env_events / multi_source_n_agent_events is the
    calibration target signal for the validation experiment (target 1:1-2:1).
  Backward compatible: multi_source_dynamics_enabled=False by default;
    _init_multi_source_state not called; _step_* not called; proximity
    field path unchanged; info tags zero / False. RNG draws guarded inside
    `if multi_source_dynamics_enabled:` so seed sequences for existing
    experiments are bit-identical when disabled. 7/7 preflight + 184/184
    contracts PASS with master OFF (smoke 2026-05-03).
  Activation smoke (2026-05-03, 4-arm sweep, 200 ticks each):
    ARM_0 (OFF): env_events=2 (background drift only), agent_events=193,
      bit-identical to legacy.
    ARM_1 (scale=0.25 ON): env_events=77, agent_events=185.
    ARM_2 (scale=1.0 ON): env_events=330, agent_events=169.
      Calibration ratio 1.95:1 -- matches SD doc target 1:1-2:1.
    ARM_3 (scale=4.0 ON): env_events=354, agent_events=175.
      Saturation at high noise -- transient-pool bound.
    Weather AR(1) firing (super_field abs-mean 0.053 with sigma=0.1);
    drift sources moving 170 / 240 attempts (~0.71 effective move rate);
    transients churning at ~5e-3 per cell per tick.
  Implementation choices (deviations / clarifications from SD doc):
    - Flat kwargs on CausalGridWorld.__init__ rather than nested dataclass
      (MultiSourceDynamicsConfig). Matches SD-022 / SD-023 / SD-029 precedent
      for env-only SDs; nothing else uses dataclass-config for env params.
    - Transient + drift hazards live in self.hazards (with parallel
      bookkeeping lists for movement / disappearance) rather than getting
      a new ENTITY_TYPES entry. Adding a new entity type would change
      NUM_ENTITY_TYPES from 7 to 8 and break local_view shape (5x5x7=175 ->
      5x5x8=200), violating backward compat.
    - Per-source bit-identical OFF preserved: each source's RNG draws are
      gated by its own switch and the master switch, so single-source
      ablation studies are clean.
  No trainable parameters. Pure env-side stochastic substrate enrichment.
  No phased training needed.
  MECH-094: not applicable (env observation stream, not replay / simulation
    content). Validation experiments call sense() / step() in waking mode;
    simulation paths do not invoke env.step.
  Validation experiment: V3-EXQ-509 queued (4-arm noise sweep
    {OFF / 0.25x / 1.0x / 4.0x}, V3-EXQ-506-equivalent metrics). Pre-
    registered prediction per SD doc: ARM_0 replicates V3-EXQ-506 C1-C3
    FAIL; C1, C2, C3 pass-rate forms inverted U across ARM_1 -> ARM_2 ->
    ARM_3 with ARM_2 peak (Asai 2016 non-monotonic). Five-row interpretation
    grid handles validation / calibration miscalibration / Woo & Spelke
    falsifier branch (re-route MECH-095 to substrate_conditional V4) /
    opposite-direction artefact / standard validation.
  Design doc: REE_assembly/docs/architecture/sd_047_multi_source_dynamics.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_sd_047/
    + targeted_review_connectome_mech_095/ (passivity cluster).
  See SD-047, MECH-095 (substrate-ceiling unblock), MECH-098 (reafference
    cancellation, secondary unblock), MECH-099 (downstream agency
    attribution, secondary unblock), SD-022 (body-damage substrate, layered
    on top of), SD-029 (scheduled hazard curriculum, layered on top of),
    ARC-033 (E2_harm_s forward, indirect benefit on cf_gap_ratio), MECH-094
    (call-site scoping; not applicable).

### SD-047 amendment: non-saturating self/world transition_type tag (2026-07-11)
- environment.multi_source_dynamics -- AMENDED 2026-07-11 (agency_comparator_testbed_sd047).
  Module: ree_core/environment/causal_grid_world.py.
  Config: CausalGridWorld/V2 flat kwarg `tag_env_caused_multisource_ttype`
    (default False; bit-identical OFF -- verified: with the flag off the
    transition_type histogram is identical, and with SD-047 ON the flag only
    reassigns residual "none" steps, leaving every agent-caused and
    env_caused_hazard count unchanged).
  Data flow: step() -- after ALL transition_type assignment (agent-caused +
    scheduled injections) and just before the `_last_transition_type` cache,
    if the flag is set AND transition_type is still "none" AND
    multi_source_n_env_events > 0, set transition_type = "env_caused_multisource".
    It only FILLS a residual "none", never overrides an agent-caused label.
  Why: the 047-lineage MECH-095 retests (V3-EXQ-047l/047m) both self-routed
    FAIL but were adjudicated non_contributory -- the `env_events>0` additive
    label fold is degenerate at intensity 1.0 (drift moves ~every tick), so
    any world/self label folding env_events saturates (047l probe, 047m
    training label). This tag gives a NON-saturating self/world ground-truth
    from transition_type directly (is_world = prev_ttype in {env_caused_hazard,
    env_caused_multisource}); at intensity 1.0 is_world ~0.15 (a real contrast),
    not the saturated ~0.93 the fold produced.
  Backward compatible: default False; existing experiments bit-identical.
  No trainable parameters; no phased training; MECH-094 not applicable.
  Test-bed (VALID MECH-095 retest, realizes /implement-substrate substrate_queue
    sd_id agency_comparator_testbed_sd047, re-points MECH-095's owed retest off
    "SD-047 build"): V3-EXQ-741 (v3_exq_741_mech095_agency_comparator_testbed_sd047).
    4-arm SD-047 intensity sweep x 4 seeds; per cell THREE conditions --
    BASELINE (z_world), ROUTED_A (gradient BCE head on the non-saturating label,
    reshapes z_world), ROUTED_B (query-time read-out: efference-copy f_eff(z_self,a)
    vs observed z_self, agency_residual concatenated to z_world, NEVER backprops
    into z_world). Two guards per arm x seed: probe-partition (047m) + NEW
    self-world balance (the training-label guard 047m lacked). Re-derive brake
    was FIRED (2 MECH-095 non_contributory autopsies); this CLEARS it -- new EXQ
    number, different operationalisation + the pre-registered sweep, enabling
    substrate now BUILT. See failure_autopsy_V3-EXQ-047m_2026-07-11.
