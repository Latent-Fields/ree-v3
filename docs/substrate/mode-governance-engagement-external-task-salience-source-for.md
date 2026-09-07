## mode-governance-engagement: external_task salience source for SalienceCoordinator (2026-06-13)
- mode-governance-engagement -- IMPLEMENTED 2026-06-13 (substrate; MECH-266 stays
  provisional / SD-032a stays stable -- PROMOTES NOTHING until the 464d/467d retest
  runs). The external_task salience SOURCE the SD-032a SalienceCoordinator lacked on the
  603n foraging substrate. Routed by the confirmed
  failure_autopsy_SD-034-closure-cluster-ext_2026-06-12 (sub-cluster B: V3-EXQ-464c +
  467c) via the substrate_queue mode-governance-engagement entry minted by the 2026-06-13
  AM governance cycle.
  ROOT CAUSE (code-confirmed in salience_coordinator.py): external_task gets only
  external_task_bias (1.0) + drive_level (affinity weight 1.0), while dacc_pe /
  dacc_foraging / dacc_difficulty all push internal_planning. On the foraging substrate
  drive_level ~ 0.016 (540c probe), so on tick 1 the argmax flips to internal_planning and
  the agent settles there for the episode -> fraction_in_external_task = 0.0 on both arms /
  all seeds, and the 464c/467c eval loops count one episode-initial settle per episode
  (n_switches == n_episodes), so MECH-266's exit-rail had no contested mode to bind and the
  n_switches>=1 non-vacuity gate passed VACUOUSLY.
  THE FIX (no-op-default; bit-identical OFF; mirrors the SD-035 CeA / SD-037 override
  signal-injection pattern exactly -- the SalienceCoordinator class is UNCHANGED, it
  already accepts arbitrary named signals):
    Module: ree_core/agent.py (registration at __init__ + injection at the salience tick
      site in select_action), ree_core/utils/config.py (6 no-op-default flags + from_dims).
    Registration (REEAgent.__init__, gated on use_external_task_drive + salience present):
      affinity_weights["external_task_drive"] = {"external_task": external_task_drive_affinity_weight}
      salience_weights["external_task_drive"] = external_task_drive_salience_weight
      -- registered in BOTH so external_task can win the mode argmax (affinity) AND a switch
      INTO external_task can fire the MECH-259 trigger (salience aggregate).
    Injection (select_action, BEFORE coord.tick(), alongside the aic/cea/override injections):
      engagement = goal_active ? clip(commit_w*float(beta_gate.is_elevated)
                                      + prox_w*float(goal_state.goal_proximity(z_world)), 0, 1) : 0
      coord.update_signal("external_task_drive", engagement)
    The engagement is DYNAMIC by design (gated on an active goal, graded by committed
    pursuit x proximity), so it RELEASES toward internal_planning during deliberation /
    between-goals / just-consumed -- producing GENUINE mode competition, NOT the 464b
    "100% external_task, 0 switches" saturation degeneracy (the opposite failure the
    2026-06-04 MECH-266 evidence_quality_note recorded).
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
    use_external_task_drive (False, master), external_task_drive_affinity_weight (1.0),
    external_task_drive_salience_weight (1.0), external_task_drive_commit_weight (1.0),
    external_task_drive_proximity_weight (1.0), external_task_drive_require_goal_active (True).
  Backward compatible: use_external_task_drive=False by default -> no slot registered, no
    injection, "external_task_drive" never enters the coordinator's _input_signals (tick
    reads 0) -> bit-identical. 7/7 preflight + 1031 contracts (1026 prior + 5 new in
    tests/contracts/test_mech266_external_task_drive.py: C1 OFF no-slot + bit-identical
    action stream / C2 ON registers BOTH affinity+salience slots / C3 coordinator math --
    drive raises external_task probability AND salience_aggregate over an
    internal_planning-pushed baseline / C4 agent injects engagement>0 on a goal-active
    agent + monotone (drive never reduces external_task occupancy) / C5 goal-inactive ->
    injected engagement 0, the release path) PASS. v3_exq_464c --dry-run unchanged
    (drive OFF -> reproduces the prior sym_frac=0.0 / asym_frac=0.0 substrate-ceiling
    signature).
  Phased training: N/A (non-trainable arithmetic signal injection; no learned parameters).
    MECH-094: waking-only by call-site scoping (select_action), as with the neighbouring
    AIC / CeA / override injections. Evidence-staleness (Step 8.5): NOT triggered --
    no-op-default flag; every existing experiment uses the default (drive off), so no
    dependent claim's measured mechanism changed. KEEP all evidence.
  depends_on (unresolved at landing): scaffolded_sd054_onboarding nav-competence (Stage-H).
    The substrate + contract tests land regardless (user-directed); the VALIDATION may
    self-route substrate_not_ready if the agent does not survive/forage long enough, which
    the 603n contact guard + the restated occupancy gate handle cleanly.
  Validation experiments: V3-EXQ-464d (competing-goals) + V3-EXQ-467d (mode-stickiness
    dose-response) -- successors (NEW letter, NOT supersede) of 464c/467c with
    use_external_task_drive=True AND the readiness gate RE-STATED as
    min_across_arms(fraction_in_external_task) > floor (~0.1) replacing the n_switches>=1
    non-vacuity gate, so the asymmetric exit-rail finally has a contested mode to bind.
    claim_ids=[MECH-266, SD-032a]; experiment_purpose=evidence. Queued via /queue-experiment.
    GOVERNANCE: MECH-266 stays provisional / SD-032a stays stable; claims.yaml carries only
    an implementation_note + the pending_retest_after_substrate flag added this cycle (no
    flag/confidence/promotion change). substrate_queue mode-governance-engagement ready
    STAYS false until the retest clears the occupancy gate.
  Design doc: REE_assembly/docs/architecture/sd_032_cingulate_integration_substrate.md
    (mode-governance-engagement section). Substrate_queue:
    REE_assembly/evidence/planning/substrate_queue.json (mode-governance-engagement).
    Autopsy: REE_assembly/evidence/planning/failure_autopsy_SD-034-closure-cluster-ext_2026-06-12.{md,json}.
  See MECH-266 (asymmetric mode hysteresis -- the exit-rail this unblocks), SD-032a
    (SalienceCoordinator -- the mode register the drive feeds), MECH-259 (switch threshold),
    SD-035 CeA / SD-037 override (the affinity+salience injection pattern this mirrors),
    SD-012 drive_level (the inadequate external_task driver this complements), MECH-295
    (goal-pursuit / approach bridge -- adjacent goal machinery), V3-EXQ-464c/467c (the FAILs
    this addresses), V3-EXQ-464d/467d (validation), MECH-094 (call-site scoping).

- SD-032c: cingulate.aic_analog_salience_urgency -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/aic_analog.py (AICAnalog, AICConfig).
  Anterior-insula-analog interoceptive-salience / urgency-interrupt module.
  NOT the affective-pain consumer (that is SD-032b); this is the mode-switch
  trigger source AND the descending pain-modulation gate. Subsumes SD-021:
  the raw beta_gate.is_elevated check in agent.sense() is replaced by a
  drive-aware, operating-mode-aware gain function.
  Inputs (per sense() tick):
    z_harm_a_norm  (SD-011 affective stream)
    drive_level    (SD-012 GoalState._last_drive_level)
    beta_gate_elevated (MECH-090 committed-state signal)
    operating_mode (SD-032a coordinator, previous tick; None -> treat
                    p_external_task=1.0, preserves SD-032c function even
                    without coordinator)
    extra_salient  (optional; unexpected z_goal drop, reward-surprise,
                    irreversibility; default no-op via aic_extra_weight=0)
  Outputs (stored on the module, cached in agent._aic_last_tick):
    aic_salience   -- fed to SalienceCoordinator.update_signal("aic_salience",
                      ...) BEFORE coordinator.tick() each select_action cycle
                      (drives MECH-259 urgency-trigger).
    harm_s_gain    -- multiplier on z_harm in sense(), replacing the raw
                      SD-021 beta_gate check when use_aic_analog=True.
                      harm_s_gain < 1.0 only when committed AND the agent is
                      not depleted (drive_protect=1.0 default).
    urgency_signal -- diagnostic threshold crossing on aic_salience.
  Computation:
    baseline <- (1-alpha)*baseline + alpha * z_harm_a_norm  (EMA interoceptive
                                                             baseline)
    urgency  = max(0, (z_harm_a_norm - baseline) / (baseline + eps))
    aic_salience = urgency * (1 + drive_coupling * drive_level)
                 + aic_extra_weight * sum(extra_salient)
    drive_protect = max(0, 1 - drive_protect_weight * drive_level)
    harm_s_gain = clip_[0,1] ( 1 - base_attenuation * p_external *
                               float(beta_gate_elevated) * drive_protect )
  Config: REEConfig.use_aic_analog (bool, default False).
    Sub-knobs: aic_baseline_alpha (0.02, ~50-step window),
    aic_drive_coupling (1.0 -- MUST be non-zero for falsification signature),
    aic_urgency_threshold (1.0, diagnostic only),
    aic_base_attenuation (0.5, matches legacy descending_attenuation_factor),
    aic_drive_protect_weight (1.0; alterable-configuration knob flagged by
                              SD-032c spec: +1 preserve depleted signal,
                              0 drive-independent, -1 opposite-sign),
    aic_extra_weight (0.0, reserved for extra salient-event signals).
  Falsification signature (spec): same z_harm_a -> different mode-switch
    behaviour in depleted vs well-resourced agents. Both aic_salience AND
    harm_s_gain depend on drive_level -- this is the ONLY V3 substrate that
    makes the dependence structural. EXQ-325a FAIL (DESCENDING ==
    CONTROL bit-identical under raw beta_gate check) resolves when the AIC
    path replaces the raw check -- the descending branch becomes a
    genuinely different function of state.
  Data flow: encode() -> z_harm_a, z_harm -> aic.tick(z_harm_a_norm,
    drive_level, beta_gate_elevated, operating_mode_prev) -> aic_salience
    cached + harm_s_gain applied to z_harm if harm_descending_mod_enabled.
    select_action() injects aic_salience into coordinator via
    update_signal("aic_salience", ...) BEFORE coordinator.tick() so MECH-259
    trigger sees it on the current cycle. One-step lag on operating_mode
    read is biologically plausible (AIC->dACC->SAL is a circuit).
  Backward compatible: use_aic_analog=False by default. Legacy SD-021 raw
    beta_gate check retained behind the same harm_descending_mod_enabled
    flag -- selected only when use_aic_analog=False. With both flags off,
    existing experiments unchanged. The old descending_attenuation_factor
    config is still consumed by the legacy path.
  Biological basis: Craig 2009 AIC as interoceptive-salience hub with
    autonomic and motor efferents; Menon & Uddin 2010 salience-network
    coupling; Basbaum 1984 + Keltner 2006 ACC/AIC -> PAG descending
    inhibitory pathway.
  MECH-094: not applicable (waking observation stream, not replay content).
  Phased training: not applicable (non-trainable arithmetic, single EMA).
  Validation experiment: V3-EXQ-325b queued (3-condition x 2-drive-regime
    retest of EXQ-325a; supersedes EXQ-325a; acceptance criteria include
    drive-dependence contrast which the prior metric could not measure).
  See SD-032c, SD-032a, SD-032b, SD-021, MECH-259, MECH-261, SD-032 parent.

- SD-032d: cingulate.pcc_analog_attention_partition -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/pcc_analog.py (PCCAnalog, PCCConfig).
  Posterior-cingulate-analog metastability scalar in [0, 1] that modulates
  the SD-032a MECH-259 effective_threshold. High pcc_stability -> coordinator
  resists mode transitions; low stability (depleted / no recent rest /
  failing task outcomes) -> transitions happen at lower salience. Does NOT
  trigger mode switches directly (that is SD-032c's job). Non-trainable
  arithmetic; no gradient flow.
  Inputs (per select_action tick):
    drive_level (SD-012 fatigue, [0, 1])
    success_ema (EMA over caller-supplied task-outcome scalars, neutral 0.5
                 baseline; experiments opt in via agent.note_task_outcome())
    steps_since_offline (cross-episode counter; reset only by
                         note_offline_entry() called from
                         agent.enter_offline_mode())
  Computation:
    offline_recency = min(1.0, steps_since_offline / window)
    stability = baseline + success_weight * (success_ema - 0.5)
              - fatigue_weight * drive_level
              - offline_weight * offline_recency
    stability = clip_[0,1](stability)
  Config: REEConfig.use_pcc_analog (bool, default False).
    Sub-knobs: pcc_success_alpha (0.02, ~50-step EMA window),
    pcc_success_weight (0.5; centred contribution from success_ema),
    pcc_fatigue_weight (0.5; subtractive from drive_level),
    pcc_offline_recency_window (500 steps; saturation),
    pcc_offline_weight (0.3; subtractive from offline_recency),
    pcc_stability_baseline (0.5; additive baseline before clipping).
  Falsification signature (spec): ablating SD-032d makes the SalienceCoordinator
    effective_threshold insensitive to fatigue / time-since-offline. Agent
    over-commits to external_task without rest-driven relaxation. PCC-ON ->
    drive_level rises -> stability falls -> effective_threshold falls ->
    mode_switch_trigger rate rises under matched salience input.
  Data flow: select_action() -> pcc.tick(drive_level=sal_drive) ->
    salience.update_signal("pcc_stability", pcc.pcc_stability) BEFORE
    coordinator.tick() -> coordinator.effective_threshold modulated.
    enter_offline_mode() -> pcc.note_offline_entry() (single integration
    point shared by MECH-092 within-session quiescence and INV-049
    cross-session sleep). reset() -> pcc.reset() (per-episode; preserves
    cross-episode _steps_since_offline). agent.note_task_outcome(value) ->
    pcc.note_task_outcome(value) feeds success EMA.
  Backward compatible: use_pcc_analog=False by default. Existing experiments
    unaffected. note_task_outcome() is a no-op when pcc is None.
  Biological basis: Leech & Sharp 2013 ("Arousal, Balance, Breadth") --
    PCC tracks the global stability of the current cognitive set vs the
    need to broaden attentional sampling. Treated conservatively: a
    [0, 1] metastability index that biases the threshold for any mode
    change without committing to attention-partition geometry. Frankland
    & Bontempi 2005 systems-consolidation framing: stability falls with
    time-since-last-offline, biasing the system toward requesting offline
    consolidation when held externally too long.
  MECH-094: not applicable (waking arithmetic, no replay content authored).
  Phased training: not applicable (non-trainable arithmetic).
  Validation experiment: V3-EXQ-447 queued (PCC-OFF vs PCC-ON x rest /
    no-rest contrast; acceptance criterion: with PCC-ON and matched dACC
    salience injection, mode-switch trigger rate is monotone in
    drive_level and time-since-offline; PCC-OFF rate is invariant).
  See SD-032d, SD-032a, MECH-259, MECH-261, INV-049, MECH-092, SD-032 parent.

- SD-032d AMENDMENT: cingulate.mu_kappa_mode_prior_overlays (MECH-048) --
  IMPLEMENTED 2026-07-21.
  Module: ree_core/cingulate/salience_coordinator.py (SalienceCoordinatorConfig,
    SalienceCoordinator.tick).
  Config: REEConfig.salience_use_stability_temperature (default False; set True
    to enable). Sub-knobs salience_temperature_mu_alpha (1.0),
    salience_temperature_kappa_alpha (0.0),
    salience_temperature_exponent_clip (4.0).
  Data flow: pcc.tick() -> pcc_stability [0,1] ->
    salience.update_signal("pcc_stability") -> coordinator.tick() ->
    effective_temperature = softmax_temperature *
      exp(alpha_kappa*aic_salience - alpha_mu*pcc_stability)
    -> _softmax(logits, effective_temperature) -> operating_mode
    -> mode entropy + mode_switch_trigger + write_gates.
  Backward compatible: disabled by default; when off, effective_temperature is
    softmax_temperature exactly and exp() is never evaluated, so every tick is
    bit-identical. Existing experiments unaffected. Full contract suite green
    (2019 passed) with the change in place.
  WHY. The 2026-04-19 SD-032d landing gave pcc_stability (the mu-analogue)
    authority over the switch THRESHOLD only; affinity_weights["pcc_stability"]
    was left empty. So H(operating_mode) was EXACTLY invariant under mu --
    0.167605 at pcc_stability 0.0, 1.0 AND 3.0, identical to 6 dp, measured by
    live execution on a constructed agent, while kappa (aic_salience) moved it
    freely via its own affinity weight. MECH-048 asserts the overlays shape
    BOTH mode-prior sharpness (entropy) AND switching inertia, so its entropy
    half was untestable BY CONSTRUCTION on every substrate and every seed: any
    experiment measuring mode entropy against mu would have reported an
    arithmetic zero rather than a measurement. That is the DV-symmetry vacuity
    class (failure_autopsy_V3-EXQ-604c_2026-07-20 section 3) and is why the
    2026-07-21T13:39Z audit HELD V3-EXQ-683 rather than repairing it.
  Form is the claim's own source, not an invention:
    docs/thoughts/2026-02-11_some_control_plane_maths_hypotheses.md:63 gives
    tau = tau_0 * exp(alpha_kappa*kappa - alpha_mu*mu) literally, and :104
    gives the threshold half theta = theta_0 + rho_mu*mu - rho_kappa*kappa,
    which is the pre-existing stability_scaling multiplier (unchanged). So mu
    now reaches both legs, as the claim asserts.
  WHY NOT an affinity_weights["pcc_stability"] entry (the rival mechanism):
    a per-mode logit bias makes mu a mode PREFERENCE, and
    control_plane.md#mech-048 states the overlays "are not scalar reward
    signals; they act as stability and entropy modulators". Ruled out by claim
    text, not preference. The empty dict is now commented as intentional.
  alpha_kappa defaults to 0.0 so that flipping the master switch isolates the
    mu leg (single-lever ablation) -- kappa already reaches the mode logits via
    its affinity weight, so enabling both at once confounds two paths.
  Exponent clipped at +/-4.0 because aic_salience is unbounded above
    (urgency_scaled + extra_sum); keeps the temperature factor in [0.018, 54.6].
    The multiplicative-exponential form also guarantees tau > 0, so the
    coordinator's temperature <= 0 guard is unreachable.
  tick() additionally returns effective_temperature and mode_entropy (nat
    Shannon entropy of operating_mode) so consumers read the MECH-048 DV
    rather than recomputing it inconsistently.
  MECH-094: not applicable (no memory content authored).
  Phased training: not applicable (non-trainable arithmetic, no gradient flow).
  Contracts: tests/contracts/test_mech048_stability_temperature.py (7
    contracts). All assertions are on continuous readouts -- entropy,
    temperature, threshold -- and none on a sampled discrete mode, which is not
    reproducible across machine classes.
  EXPERIMENT-DESIGN CAUTION, read before queuing anything against MECH-048:
    with mu injected DIRECTLY, "entropy falls as mu rises" is an arithmetic
    identity of the coupling. The contracts assert it as substrate wiring; it is
    NOT evidence for MECH-048. A real experiment must drive mu from upstream
    environmental state (safety / coherence / task success, via PCCAnalog) and
    read a downstream behavioural DV, or it reproduces the exact vacuity this
    build exists to remove.
  Validation experiment: NOT yet queued -- chipped as separate
    /queue-experiment work (see WORKSPACE_STATE 2026-07-21).
  STILL OPEN (separate build, deliberately not bundled): MECH-048's
    switching-pressure half stays empirically unmeasurable because the
    contested modes are never occupied -- V3-EXQ-464b/464c/467d/464d ALL FAIL
    with fraction_in_external_task = 0.0 at every seed with
    use_external_task_drive=True across a 20x hysteresis sweep; mean_dwell is
    an episode-length artefact and n_switches == n_episodes. 467d self-routed
    substrate_not_ready_requeue / external_task_mode_not_occupied.
  See MECH-048, SD-032d, SD-032a, SD-032c, MECH-259.

- SD-032e: cingulate.pacc_autonomic_coupling -- IMPLEMENTED 2026-04-19.
  Module: ree_core/cingulate/pacc_analog.py (PACCAnalog, PACCConfig).
  Perigenual / subgenual cingulate-analog slow-EMA autonomic write-back.
  Accumulates tanh-normalised z_harm_a magnitude into a bounded drive_bias
  that shifts the effective drive_level passed into GoalState.update(),
  SalienceCoordinator.tick(), SD-032c AICAnalog, SD-032d PCCAnalog, and
  dACC bundle composition. Architectural path for chronic-pain-like
  sensitisation (Baliki 2012) compressed into the V3 drive_level proxy.
  Non-trainable arithmetic; no gradient flow.
  Scoping (see REE_assembly/evidence/literature/
  targeted_review_pacc_autonomic_coupling_write_target/synthesis.md):
    (1) Write target: drive_level as first-pass proxy. Biologically
        tighter targets (valence-signed mood setpoint, fast autonomic
        effectors) do not have V3 substrates; documented simplification.
    (2) Timescale: slow EMA, default alpha=0.002 (pacc_drive_ema=0.998;
        half-life ~347 steps). Scoping synthesis called alpha>=0.005
        "fast end of biological plausibility" -- default is inside the
        envelope; long-horizon sensitisation studies should use
        alpha<=0.0005. Compresses two biological steps (Guo 2018 ACC
        mGluR5 LTP + ACC downstream influence) into one accumulator.
    (3) Offline decay: DEFAULT 0.0 (no decay). Non-zero instantiates a
        DISTINCT sleep-recalibration claim that would need its own
        literature pull -- hook exists so a future claim can wire in
        without another implementation pass.
  Inputs (per select_action tick):
    z_harm_a_norm  (SD-011 affective stream, current latent)
    write_gate     (SalienceCoordinator.write_gate("autonomic") from
                    previous tick; one-step lag, pACC->autonomic->
                    sensitisation is slow. Defaults to 1.0 when
                    salience coordinator is disabled so drift remains
                    observable under ablation.)
    hypothesis_tag (MECH-094 gate; select_action passes False --
                    waking write. Simulation/replay paths that call
                    pacc.tick with True are skipped.)
  Computation:
    if hypothesis_tag: skip
    elif z_harm_a_norm <= z_harm_a_min: target = 0  (Guo 2018 rest relaxation)
    else: target = tanh(z_harm_a_norm) * drive_scale
    drive_bias = (1 - alpha*gate) * drive_bias + alpha*gate*target
    drive_bias = clip(drive_bias, -cap, +cap)
  Read path: effective_drive(base) = clip_[0,1](base + drive_bias).
  Consumers (all in agent.py select_action / sense / update_z_goal):
    - dACC bundle drive_level input (SD-032b)
    - SalienceCoordinator.tick drive_level (SD-032a)
    - AICAnalog.tick drive_level input (SD-032c; one-step lag via next sense)
    - PCCAnalog.tick drive_level input (SD-032d)
    - GoalState.update drive_level (SD-012 wanting-gain scaling)
  Convention: goal_state._last_drive_level stores the BASE drive_level;
  SD-032 consumers apply pacc.effective_drive() themselves to avoid
  double-counting the bias.
  Per-episode reset() clears diagnostics cache only -- drive_bias is
  cross-episode by architectural intent. enter_offline_mode() calls
  note_offline_entry() (default no-op at offline_decay=0.0).
  Config: REEConfig.use_pacc_analog (bool, default False).
    Sub-knobs: pacc_drive_alpha (0.002, ~347-step half-life),
    pacc_drive_scale (1.0), pacc_drive_bias_cap (0.5, absolute cap
    on |drive_bias|), pacc_z_harm_a_min (0.0, threshold below which
    target is 0 -- reversibility under quiescence),
    pacc_offline_decay (0.0, distinct sleep-recalibration claim if
    set non-zero).
  Falsification signature (spec): sustained z_harm_a exposure produces
    drift in drive_level, which modulates SD-032c switch threshold and
    GoalState wanting gain. With SD-032e OFF, same sustained z_harm_a
    leaves drive_level untouched (only obs_body[3] energy depletion
    moves it) -- no chronic-pain-sensitisation signature possible.
  Backward compatible: use_pacc_analog=False by default; agent.pacc is
    None and every integration site is a no-op. Existing experiments
    unaffected.
  Biological basis: Vogt 2005 ACC subdivisions (perigenual/subgenual
    as autonomic/affective-output hub); Mayberg 2005 sgACC
    depression-baseline setpoint (cited for valence-setpoint role the
    current implementation does NOT directly instantiate -- shape
    mismatch documented); Critchley 2003 ACC-autonomic coupling;
    Gianaros 2011 ACC-PAG-medulla fast-effector route (out of V3
    scope; future SD-032f); Guo 2018 ACC mGluR5 LTP days-timescale
    plasticity (primary grounding for slow-EMA default); Baliki 2012
    corticostriatal chronic-pain drift (falsification-signature
    behaviour the substrate targets).
  MECH-094: handled by hypothesis_tag skip in tick(); waking
    select_action writes are valid (tag=False).
  Phased training: not applicable (non-trainable arithmetic).
  Validation experiment: V3-EXQ-448 queued (4-arm ablation:
    pACC-OFF / pACC-ON-normal-z_harm_a / pACC-ON-sustained-z_harm_a /
    pACC-ON-hypothesis-tag-only; acceptance: drive_bias monotone in
    sustained exposure magnitude, MECH-094 skip suppresses accumulation,
    bias bounded by cap, downstream effective_drive shifts AIC
    harm_s_gain and coordinator effective_threshold in expected
    directions).
  See SD-032e, SD-032a, SD-032c, SD-032d, SD-012, SD-011, MECH-261,
  MECH-094, SD-032 parent.

- ARC-058: harm_stream.shared_forward_trunk -- REGISTERED 2026-04-19,
  COMPETES WITH ARC-033.
  Module: ree_core/latent/stack.py (HarmForwardTrunk, HarmForwardHead
  -- pre-existing substrate classes). Selection via shared_trunk
  constructor arg on E2HarmSForward / E2HarmAForward (see MECH-258).
  ARC-033 claim: independent per-stream forward models (separate
  ResidualHarmForward per stream). Biological reading: dorsal posterior
  insula (sensory PE) + anterior insula (affective PE) as separate
  learned substrates.
  ARC-058 claim (competing): shared HarmForwardTrunk (unsigned,
  modality-independent PE substrate) + stream-specific HarmForwardHead
  (signed, per-modality readout). Biological reading: Horing & Buchel
  2022 anterior insula encodes modality-independent unsigned PE shared
  across aversive modalities; dorsal posterior insula encodes
  modality-specific signed PE. Trunk ~ unsigned; head ~ signed.
  Same nn.Module topology, different wiring. Constructor switch arbitrates.
  Falsifiable: V3-EXQ-445 three-arm ablation measures per-stream
  forward_r2 for z_harm_s and z_harm_a + downstream dACC bundle
  usefulness under each path. If shared-trunk matches or beats
  independent with fewer parameters AND produces a useful unsigned
  PE signal, ARC-058 wins and ARC-033 is narrowed. If independence
  wins, ARC-058 is retired.
  See ARC-058, ARC-033, MECH-258, MECH-257, SD-032b.
