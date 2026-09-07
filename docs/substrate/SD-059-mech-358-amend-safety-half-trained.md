## SD-059 / MECH-358 AMEND: safety-half trained threat-absence wiring (V3-EXQ-603i secondary gap, 2026-06-09)
- SD-059 / MECH-358 safety-half wiring amend -- IMPLEMENTED 2026-06-09. Wires the
  trained MECH-303 (contextual safety terrain) / MECH-304 (conditioned safety
  store) threat-absence prediction into the EscapeAffordanceBridge safety-credit
  path so the SAFETY half can credit non-vacuously. Routed by the SECONDARY gap of
  failure_autopsy_V3-EXQ-603i_2026-06-08 (Section 4 Prerequisites (b) + Section 6
  Learning #2): on V3-EXQ-603i the safety half credited 0/3 in EVERY arm (relief
  half credited 2/3, functional) because the raw safety condition
  (threat_scale(z_now) <= 0, i.e. z_harm_a norm below threat_floor) almost never
  fires under Stage-H -- the threat never goes fully absent after a single directed
  action. The half was wired STRUCTURALLY but STARVED: no trained threat-absence
  predictor fed it (symbol-of-mechanism-without-functional-input). REE already owns
  the trained predictors -- MECH-303 (SD-052, ResidueField.evaluate_safety RBF
  terrain) + MECH-304 (SD-051, ConditionedSafetyStore EMA-prototype cosine->sigmoid)
  -- but they were unwired to the bridge.
  THE FIX (no-op default; bit-identical OFF; bridge stays OFF by default):
    Module: ree_core/pfc/escape_affordance_bridge.py
      (EscapeAffordanceBridgeConfig + EscapeAffordanceBridge.update). Config gains
      use_trained_safety_signal (False) + safety_signal_threshold (0.5). update()
      gains safety_signal: Optional[float]=None. The SAFETY credit now fires when
      raw threat-absence (threat_scale(z_now) <= 0) OR -- when use_trained_safety_signal
      -- the supplied trained safety_signal >= safety_signal_threshold. OR-composed,
      so the original raw mechanism is retained as a fallback; the trained path
      stays INSIDE the existing under-threat (prev > threat_floor) + directed-action
      gate, so it credits genuine response-produced safety (MECH-303/304-consistent),
      not generic safe-context. New diagnostic counter _n_safety_credit_trained
      (surfaced as mech358_n_safety_credit_trained in get_state) attributes credits
      to the trained predictor specifically (the non-vacuity readout for the
      validation manifest).
    Module: ree_core/safety/conditioned_safety_store.py -- new read-only
      ConditionedSafetyStore.predict(z_world) -> float (the cosine->sigmoid query
      WITHOUT decay/EMA mutation). Additive; never called by existing paths -> zero
      behaviour change. Lets the agent read the MECH-304 cue-specific safety
      prediction for the CURRENT post-action state at the bridge-update site (the
      store's own update() runs LATER in the same sense(), so the cached
      _conditioned_safety_signal is one tick stale).
    Module: ree_core/agent.py -- at the bridge.update call site in sense() (after
      the MECH-357 eligibility update), when escape_use_trained_safety_signal is on
      and not simulation, compute _eab_safety_signal = max over enabled trained
      predictors: MECH-304 conditioned_safety_store.predict(z_world) (if the store
      is built) and MECH-303 residue_field.evaluate_safety(z_world).mean() (if
      use_contextual_safety_terrain). Both are pure reads. Passed as
      bridge.update(safety_signal=...). None when the flag is off -> bit-identical.
      Bridge build site forwards the two new config fields into the bridge config.
    Config: REEConfig.escape_use_trained_safety_signal (False) +
      escape_safety_signal_threshold (0.5), both wired through REEConfig.from_dims.
  Backward compatible: escape_use_trained_safety_signal=False by default; the
    safety_signal kwarg defaults None and the OR-branch is never taken, so the
    bridge (and the V3-EXQ-603i path) is byte-identical. ConditionedSafetyStore.predict
    is unused by existing paths. 7/7 preflight + 951 contracts PASS (incl 2 new C9/C10
    in tests/contracts/test_sd_059_escape_affordance_bridge.py + the prior 8). Bridge
    stays bit-identical OFF (default vs explicit-False action stream, C10). Activation
    smoke 2026-06-09 (bridge + MECH-304 + flag ON, affective harm stream on, retained
    threat): the agent feeds a real trained safety signal (~0.375 on an untrained
    encoder), the under-threat gate opens (z_harm_a_prev ~0.79 > floor), and the safety
    half credits 5x via the trained signal (mech358_n_safety_credit_trained=5,
    safety_affordance_max~0.40); flag OFF reproduces the 603i 0/0 starvation.
  Phased training: N/A at the substrate level (pure-arithmetic regulator + read-only
    accessor; no learned parameters). BUT the MECH-303/304 predictors are themselves
    populated from MECH-302 relief events / low-harm contexts, so the validation
    experiment must enable at least one trained predictor AND keep the SD-056 e2
    contrastive warmup (and a non-vacuity readiness gate on mech358_n_safety_credit_trained)
    -- without trained predictors the safety half re-starves.
  MECH-094: preserved. The bridge.update safety-signal path is gated by the existing
    simulation_mode no-op (replay/DMN never credit); ConditionedSafetyStore.predict is
    a pure read; the agent computes the signal only on the waking sense() path (not under
    hypothesis_tag). Evidence-staleness: NOT triggered -- no-op-default flag; no dependent
    claim's measured mechanism changed (the bridge is OFF in every existing experiment).
  GOVERNANCE: SD-059 / MECH-358 NEITHER validated NOR weakened by this amend; they stay
    candidate / v3_pending / pending_retest_after_substrate. V3-EXQ-603i script / queue /
    governance state UNTOUCHED. claims.yaml NOT modified (this amend resolves no
    dependency; the bridge stays v3_pending pending the retest).
  Validation experiment: V3-EXQ-603j claim-free safety-half-credit readiness
    microdiagnostic (ablation: escape_use_trained_safety_signal OFF reproduces the 603i
    safety_credit~0; ON -> mech358_n_safety_credit_trained > 0 on >=2/3 seeds). This is
    the focused SECONDARY-gap gate ONLY. The full 4-arm G_H behavioural bridge retest
    stays gated on the PRIMARY nav/survival-competence ceiling
    (scaffolded_sd054_onboarding Stage-H leg / separate chip) -- a retest can only score
    G_H once nav competence clears too. Does NOT validate/weaken SD-059/MECH-358.
  Design doc: REE_assembly/docs/architecture/sd_059_escape_affordance_bridge.md
    (safety-half trained-signal amend section).
  Autopsy: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-603i_2026-06-08.{md,json}.
  See SD-059 / MECH-358 (parent; arithmetic bridge landed 2026-06-08), MECH-303 / SD-052
    (contextual safety terrain), MECH-304 / SD-051 (conditioned safety store), MECH-302 /
    SD-050 (relief; the functional half), SD-058 / MECH-357 (parent gate), SD-056
    (e2 action-conditional divergence; predictor prerequisite), V3-EXQ-603i (the FAIL
    this amend's SECONDARY gap addresses), V3-EXQ-603j (validation), MECH-094 (call-site
    scoping; preserved).
