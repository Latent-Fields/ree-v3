## SD-048: Interoceptive Noise Dynamics (2026-05-03)
- SD-048: body.interoceptive_noise_dynamics -- IMPLEMENTED 2026-05-03.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
  Three concurrent agent-independent stochastic body-state noise sources at
  distinct temporal scales applied to harm_obs_a readout, layered onto the
  existing SD-022 limb-damage / SD-011 EMA substrate. Level 2 counterpart
  to SD-047 at the body-state layer; substrate-ceiling unblock for ARC-058
  (and ARC-033 arbitration) per the V3-EXQ-506-equivalent C4-only-PASS
  signature on z_harm_a comparator testing.
  Three sources:
    Source 1 (autonomic noise): per-element i.i.d. Gaussian additive noise
      on harm_obs_a at every tick. Fast, continuous, low-amplitude
      (HRV / sympathetic-fluctuation analog). default sigma=0.02 normalised
      harm units.
    Source 2 (sensitisation spikes): Poisson onset of transient
      multiplicative amplification on harm_obs_a, exponentially decaying.
      Inflammatory sensitisation analog. default rate=0.008/step,
      magnitude=1.8x (0.8x amplification contribution), halflife=15 steps.
      Cumulative amplification capped at 5.0x to bound long Poisson tails.
    Source 3 (fatigue drift): slow AR(1) latent fatigue state additively
      contributing to harm_obs_a across the episode. Allostatic-load /
      sleep-pressure analog. default ar_coeff=0.995 (very slow decay),
      noise_scale=0.005, contribution_weight=0.15. Resets per episode.
  Order: fatigue (additive baseline shift) -> sensitisation (multiplicative
  gain modulation) -> autonomic (read-out noise floor). Fatigue and
  sensitisation modulate baseline / gain; autonomic noise is true read-out
  noise on top.
  Config (CausalGridWorldV2 __init__ kwargs, env-only -- not surfaced through
  REEConfig.from_dims, matching SD-047 / SD-022 / SD-029 precedent for
  env-only SDs):
    interoceptive_noise_enabled (bool, default False) -- master switch.
    interoceptive_noise_scale (float, default 1.0) -- 4-arm sweep lever
      (OFF / 0.25 / 1.0 / 4.0); scales autonomic_noise_scale,
      sensitisation_rate, and fatigue_noise_scale uniformly.
    autonomic_noise_enabled (bool, default True) -- per-source switch.
    autonomic_noise_scale (float, default 0.02) -- per-step Gaussian SD.
    sensitisation_enabled (bool, default True) -- per-source switch.
    sensitisation_rate (float, default 0.008) -- Poisson onset/step.
    sensitisation_magnitude (float, default 1.8) -- multiplicative amplifier.
    sensitisation_halflife (int, default 15) -- exponential decay half-life.
    fatigue_enabled (bool, default True) -- per-source switch.
    fatigue_ar_coeff (float, default 0.995) -- AR(1) persistence.
    fatigue_noise_scale (float, default 0.005) -- per-step innovation SD.
    fatigue_contribution_weight (float, default 0.15) -- additive weight on
      harm_obs_a.
    interoceptive_change_threshold (float, default 0.01) -- |delta_harm_obs_a|
      floor used to count body-noise vs agent-caused harm-state-change
      events (calibration-target denominator).
  Data flow (_get_observation_dict): existing SD-022 limb-damage build OR
    legacy 50-dim EMA build -> harm_obs_a numpy array ->
    [if interoceptive_noise_enabled] _apply_interoceptive_noise: fatigue
    AR(1) update / additive contribution -> sensitisation Poisson onset and
    multiplicative amplification -> autonomic per-element Gaussian ->
    delta-event classification vs cached previous tick -> torch.from_numpy.
  info dict tags (always present, 0 / False when disabled):
    interoceptive_noise_enabled, interoceptive_noise_scale,
    interoceptive_n_autonomic_events, interoceptive_n_sensitisation_events,
    interoceptive_n_fatigue_events, interoceptive_n_body_noise_events,
    interoceptive_n_agent_caused_harm_events, interoceptive_fatigue_state,
    interoceptive_sensitisation_amplification.
    The ratio interoceptive_n_body_noise_events / interoceptive_n_agent_caused_harm_events
    is the calibration target signal for the validation experiment
    (target 1:1-2:1 at ARM_2). Classification of |delta_harm_obs_a| events
    uses _last_transition_type cached from step() to attribute agent-caused
    transitions; everything else counts as body-noise-caused.
  Backward compatible: interoceptive_noise_enabled=False by default;
    _apply_interoceptive_noise short-circuits; no RNG draws; no state
    advance; harm_obs_a returned unchanged. RNG draws gated inside
    `if interoceptive_noise_enabled:` AND per-source switches so seed
    sequences for existing experiments are bit-identical when disabled.
    Default vs explicit-False bit-identical OFF guarantee verified
    2026-05-03 (50-tick parity check).
  Activation smoke (2026-05-03, 4-arm sweep, 200 ticks each, 8x8 env,
  random policy):
    ARM_0 (OFF, scale=1.0): all SD-048 counters zero.
    ARM_1 (ON, scale=0.25): autonomic 0 (sigma 0.005 below threshold 0.01),
      n_body_noise=7. Sensitisation / fatigue rare at this scale.
    ARM_2 (ON, scale=1.0): n_aut=176, n_sens=1, fatigue state evolves
      ~+/-0.04, max sensitisation amp=0.8 (one event x 0.8x contribution).
    ARM_3 (ON, scale=4.0): n_aut=200 saturated, n_sens=5.
    All three sources fire end-to-end. Body_noise:agent_caused ratio is
    dominated by random-policy contact rate at small env (164/200
    agent-caused events); validation experiment will use a larger env or
    sparser policy to land the calibration band per the SD doc's
    recalibration interpretation row.
  Implementation choices (deviations / clarifications from SD doc):
    - Flat kwargs on CausalGridWorld.__init__ rather than nested dataclass.
      Matches SD-047 / SD-022 / SD-023 / SD-029 precedent for env-only SDs.
    - Output-side perturbation on harm_obs_a readout, NOT modification of
      upstream limb_damage state. Biologically correct (interoceptive
      reporting noise vs change to underlying body) and avoids interaction
      with SD-022's failure_prob_scale movement-failure mechanism.
    - Cumulative sensitisation amplification capped at 5.0x to defend
      against long Poisson tails (numerical stability; not in SD doc).
    - Per-source diagnostic counters distinguish autonomic / sensitisation /
      fatigue source firing (always-on per-source RNG draws cross threshold)
      from the calibration-target body_noise / agent_caused classification
      (steps where readout |delta| > threshold, attributed by
      _last_transition_type).
  No trainable parameters. Pure stochastic readout enrichment. No phased
  training needed (env-only substrate; no new encoder, no new training
  target).
  MECH-094: not applicable (env observation stream, not replay / simulation
    content). Validation experiments call sense() / step() in waking mode;
    simulation paths do not invoke env.step.
  Validation experiment: V3-EXQ-511 queued (4-arm noise sweep
    {OFF / 0.25x / 1.0x / 4.0x}, 3 seeds, larger env + sparser policy to
    land the 1:1-2:1 calibration band; ARC-058-equivalent C1-C4 metrics
    deferred to the comparator-gap behavioural successor V3-EXQ-512).
    Pre-registered prediction per SD doc: ARM_0 reproduces current
    substrate-ceiling pattern (no body-noise events); ARM_2 produces
    calibrated mixed substrate; ARM_1 / ARM_3 quantify under- and
    over-noise regimes per Asai 2016 non-monotonic.
  Design doc: REE_assembly/docs/architecture/sd_048_interoceptive_noise_dynamics.md
  See SD-048, SD-011 (z_harm_a stream prerequisite), SD-022 (agent-caused
    body-state variance prerequisite), SD-047 (Level 1 environmental
    counterpart, parallel architectural logic), ARC-058 (primary Level 2
    comparator unblock), ARC-033 (competing arbitration, same substrate
    enrichment), ARC-061 (reafference comparator family Level 2
    contribution), MECH-094 (call-site scoping; not applicable).
