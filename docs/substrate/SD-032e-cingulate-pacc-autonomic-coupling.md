## SD-032e: cingulate.pacc_autonomic_coupling -- IMPLEMENTED (2026-04-19)
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
