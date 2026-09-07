## MECH-313 (ARC-065 child): Stochastic Noise Floor (LC-NE tonic / SAC analog) (2026-05-10)
- MECH-313: policy.stochastic_noise_floor_lc_ne_tonic_analog -- IMPLEMENTED 2026-05-10.
  Module: ree_core/policy/noise_floor.py (NoiseFloor + NoiseFloorConfig). First of
  four ARC-065 child substrates (sibling MECH-314 / MECH-318 / MECH-319 are
  separate spawned tasks). Pure-arithmetic regulator (no learned parameters; no
  nn.Module inheritance); matches the SD-035 / SD-036 / SD-037 regulator pattern.
  State-independent softmax-temperature lift -- the LC-NE tonic complement to
  MECH-104 phasic spike. Distinct from MECH-260 dACC anti-recency (state-dependent);
  Q-045 falsifies whether they collapse into a single substrate.
  Algorithm at the e3.select() call site in REEAgent.select_action():
    effective_T = max(baseline_T + noise_floor_alpha, noise_floor_min_temperature)
  alpha is the SAC-entropy-bonus analog (Haarnoja 2018) -- additive lift on the
  softmax temperature; min_temperature is a hard floor preventing argmax collapse
  under annealing schedules that drive the baseline below 1.0.
  Config: REEConfig.use_noise_floor (bool, default False; bit-identical OFF) +
    noise_floor_alpha (float, default 0.1; modest +10%% of E3 baseline 1.0;
    Q-043 calibrates) + noise_floor_min_temperature (float, default 1.0;
    matches existing E3 baseline so well-formed callers clear the floor).
    All wired through REEConfig.from_dims().
  Agent wiring (REEAgent.__init__): instantiate NoiseFloor when use_noise_floor=True;
    REEAgent.select_action() reads noise_floor.compute_effective_temperature(
    baseline_temperature=temperature, simulation_mode=False) BEFORE calling
    e3.select(...). Bit-identical when noise_floor is None (the else branch
    passes the unmodified temperature kwarg). reset() clears diagnostic counters.
  MECH-094: simulation_mode=True returns baseline temperature unchanged and
    increments only the simulation-skip counter; replay / DMN consumers (none
    today; reserved for forward-compat) cannot inherit waking-tonic noise floor.
    Match the SD-035 / MECH-279 / gated_policy simulation_mode pattern.
  Phase-1 instantiation choice (NOT a settled architectural commitment):
    a separate NoiseFloor module at the e3.select() call site, rather than
    per-head temperature inside GatedPolicy as the original notes-field hint
    suggested. Phase-1 reasoning: MECH-313 is state-independent and currently
    must fire on baseline E3 selection too (which the per-head approach inside
    GatedPolicy would miss with GatedPolicy disabled). Whether the policy-layer
    regulators ultimately consolidate into one module is OPEN pending
    MECH-314 / MECH-318 / MECH-319 implementations -- those substrates may
    make different placement choices that motivate revisiting MECH-313's
    placement (MECH-314 structured-curiosity in particular may fit naturally
    inside GatedPolicy as a per-head bonus, in which case MECH-313 may want
    to co-locate). Re-evaluate at the point Q-045's 4-arm ablation is queued.
    Phase-1 module surface + config knobs are stable; what could move is the
    file location and call site.
  Backward compatible: use_noise_floor=False by default; agent.noise_floor is None
    and select_action passes the temperature kwarg unchanged. 253/253 contracts +
    7/7 preflight PASS with flag OFF (regression-clean).
  Lit-pull verdicts (resolved defaults; see Pull 1 SYNTHESIS at
    REE_assembly/evidence/literature/targeted_review_arc_065_behavioral_diversity_generation/
    SYNTHESIS.md, 9 entries, lit_conf 0.78-0.82, supports-direction):
    R1 BOTH-CHANNELS-NEEDED (conf 0.85): noise floor (this) AND structured
      curiosity (MECH-314) both required. Wilson 2014 Horizon task + Faisal/
      Selen/Wolpert 2008 noise-substrate irreducibility + Friston 2015
      complementary terms.
    R2 LC-NE tonic LOAD-BEARING (conf 0.84): Aston-Jones & Cohen 2005 adaptive-
      gain model. MECH-104 covers phasic spike (substrate-landed); MECH-313 is
      the tonic complement.
    R4 continuous, every tick (conf 0.80): non-zero softmax temperature on E3
      every waking tick, regardless of context. Aston-Jones & Cohen 2005 +
      Friston 2015. NOT triggered.
    Magnitudes intentionally NOT pinned by the lit-pull (Q-043 calibration
    sweep is the empirical route).
  Phased training: not applicable (pure scalar regulator; no learned parameters;
    no gradient flow).
  Validation experiment: V3-EXQ-544 substrate-readiness diagnostic (5 sub-tests
    UC1-UC5 covering instantiation, master-OFF backward-compat, lift-arithmetic
    sweep across alpha/min_temperature, select_action wiring contract via
    act_with_split_obs, MECH-094 simulation gate). Smoke 5/5 PASS 2026-05-10
    (manifests scrubbed; runner will write the canonical PASS manifest from the
    queued entry). 11 contract tests in tests/contracts/test_mech_313_noise_floor.py
    PASS. Behavioural validation deferred to Q-045 4-arm ablation (MECH-313 OFF /
    313 only / 260 only / both ON) on V3-EXQ-543b/c successors AFTER MECH-314
    also lands.
  Design doc: REE_assembly/docs/architecture/mech_313_stochastic_noise_floor.md.
  See MECH-313 (this claim), ARC-065 (parent architectural commitment),
    MECH-260 (related but distinct mechanism; Q-045 falsifies collapse),
    MECH-104 (LC-NE phasic complement; substrate-landed),
    MECH-314 (sibling structured-curiosity bonus under ARC-065; separate substrate),
    Q-043 (relative weight calibration -- parametric sweep),
    Q-044 (MECH-314a/b/c sub-flavour independence),
    Q-045 (MECH-313 vs MECH-260 collapse falsifier -- 4-arm ablation),
    MECH-094 (simulation_mode argument; call-site scoping for waking-only effects).
