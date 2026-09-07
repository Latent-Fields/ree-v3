## MECH-319 (arc_062 GAP-K): Simulation-Mode Rule-Write Gate (Categorical Replay Tag) (2026-05-10)
- MECH-319: policy.arbitration.simulation_mode_write_gating_substrate_ree_novel_function
  -- IMPLEMENTED 2026-05-10. Module: ree_core/regulators/simulation_mode_rule_gate.py
  (SimulationModeRuleGate + SimulationModeRuleGateConfig + SimulationModeRuleGateDiagnostics).
  Substrate-level instantiation of MECH-094 at the rule-arbitration layer. Pure-arithmetic
  regulator (no nn.Module inheritance, no learned parameters); sibling to
  GABAergicDecayRegulator (SD-036) and BroadcastOverrideRegulator (SD-037) in the
  regulators package. Resolves arc_062 GAP-K substrate gap registered in the 2026-05-10
  cluster-registration session.
  Single primitive: gate.effective_simulation_mode(simulation_mode, site) -> bool.
  Truth table (master_on, admit_writes, caller_sim) -> output:
    OFF, *,     *     -> caller_sim (identity; bit-identical pre-MECH-319)
    ON,  False, False -> False     (admit waking write)
    ON,  False, True  -> True      (block simulation write -- MECH-319 normal)
    ON,  True,  False -> False     (admit waking; flag has no effect)
    ON,  True,  True  -> False     (admit simulation write -- V3-EXQ-543c falsifier)
  The gate is idempotent for waking calls (always returns False), so wiring it into
  existing waking call sites is bit-identical regardless of admit_writes. The falsifier-
  control asymmetry surfaces only when caller_sim=True (replay paths, ghost-goal probes,
  DMN passes -- not currently exercised by select_action).
  Diagnostic counters (per-call + per-site): n_calls_total, n_waking_admitted,
  n_simulation_blocked, n_simulation_admitted (the falsifier path), plus per_site_*
  dicts keyed on canonical labels SITE_GATED_POLICY, SITE_LATERAL_PFC, SITE_DEFAULT.
  New consumer call sites can pass arbitrary site strings.
  Config: REEConfig.use_simulation_mode_rule_gate (bool, default False; bit-identical OFF
  master) + REEConfig.simulation_mode_rule_gate_admit_writes (bool, default False;
  V3-EXQ-543c artificial-write-channel-routing falsifier flag). All wired through
  REEConfig.from_dims(). Construction raises ValueError when admit_writes=True without
  master ON (loud-not-silent guard against mis-configuration -- admit_writes is meaningless
  without the substrate to gate).
  Agent wiring (REEAgent.__init__): instantiate simulation_mode_rule_gate when master ON;
  None otherwise. Two existing arbitration-write call sites in REEAgent.select_action()
  consult the gate when it is instantiated:
    GatedPolicy block: replace literal simulation_mode=False with
      gate.effective_simulation_mode(False, site=SITE_GATED_POLICY) and pass to
      gated_policy.forward(...). Bit-identical for waking; seam exposed for V3-EXQ-543c.
    LateralPFCAnalog block: consult gate via
      eff_sim = gate.effective_simulation_mode(False, site=SITE_LATERAL_PFC); if eff_sim:
      skip lateral_pfc.update(...) else proceed with the existing MECH-261 mode-
      conditioned EMA. compute_bias still runs (arbitration RECEIVES the bias even
      during simulation; it just does not write back into rule_state on simulation
      ticks). Bit-identical for waking; falsifier-routed simulation writes to
      lateral_pfc would be skipped under default MECH-319 ON, admitted under
      admit_writes=True.
  Per-episode reset() clears diagnostic counters (gate has no persistent state across
  ticks beyond counters).
  MECH-094 INVARIANCE: this substrate does NOT modify MECH-094, GatedPolicy.forward's
  simulation_mode argument semantics, or LateralPFCAnalog.update. Pull 3 SYNTHESIS R1
  GENUINE-NOVELTY-CONFIRMED conf 0.72 + Pull 4 R3 KEEP-AS-IS verdict. The gate is a
  pre-call coordinator that wraps the simulation_mode argument that callers ALREADY
  pass. With MECH-319 disabled, every arbitration-write call site behaves bit-identically
  to its pre-MECH-319 form.
  RELATION TO MECH-261: MECH-261 (mode-conditioned write-gate registry on SD-032a
  SalienceCoordinator) is a complementary, continuous gate. MECH-261 returns a per-mode
  weight in [0, 1] that scales the magnitude of the EMA update on
  LateralPFCAnalog.rule_state. MECH-319 is a categorical (binary admit/block) pre-gate
  keyed to the simulation tag of the caller, not the operating mode of the agent.
  The two gates compose: caller waking + MECH-319 admits -> MECH-261 modulates EMA
  strength based on operating mode. Caller simulation + MECH-319 normal -> MECH-319
  blocks the entire update() call, MECH-261 never consulted. Caller simulation +
  MECH-319 falsifier -> MECH-319 admits -> MECH-261 modulates as if waking.
  Backward compatible: use_simulation_mode_rule_gate=False by default;
  agent.simulation_mode_rule_gate is None and both call sites take the legacy literal
  path. 288/288 contract + preflight tests PASS with master OFF (regression-clean;
  was 273 + 15 new MECH-319 contracts).
  Lit-pull verdicts (resolved defaults; see Pull 3 SYNTHESIS at
  REE_assembly/evidence/literature/targeted_review_mech_312_arbitration_divergences/
  synthesis.md):
    R1 GENUINE-NOVELTY-CONFIRMED (conf 0.72): substrate-availability premise well-
      anchored (Joo & Frank 2018 SWR review + Foster & Wilson 2006 reverse replay
      discriminable signature); the categorical write-gate FUNCTION at the arbitration
      layer is REE-novel. The literature provides only the substrate; downstream
      regions are not yet documented as exploiting the discriminable replay signature
      as a write-gate.
    Pull 4 R3 KEEP-AS-IS: MECH-094 stays as-is (architectural principle); MECH-319
      is registered as the substrate-level instantiation. The two claims are NOT
      redundant.
  Phased training: not applicable (pure boolean / counter arithmetic; no learned
  parameters; no gradient flow).
  Validation experiment: V3-EXQ-546 substrate-readiness diagnostic (6 sub-tests
  UC1-UC5 + UC3b precondition: instantiation + diagnostic keys; master-OFF backward-
  compat; truth-table coverage across the 6 valid (master, admit_writes, caller_sim)
  combinations; precondition raises on admit_writes=True without master ON;
  select_action wiring contract -- gate sees waking calls from both gated_policy and
  lateral_pfc sites after one act_with_split_obs tick, n_simulation_* counters remain
  zero on the waking path; MECH-094 invariance -- master-OFF and master-ON-with-waking-
  caller produce bit-identical wiring outputs, asymmetry surfaces only at
  caller_sim=True). Smoke 6/6 PASS 2026-05-10 (manifest scrubbed; runner will write
  the canonical PASS manifest from the queued entry). 15 contract tests in
  tests/contracts/test_mech_319_simulation_mode_rule_gate.py PASS. Behavioural
  validation -- the V3-EXQ-543c-successor falsifier with the admit_writes=True arm
  and a replay-driven invocation path -- is queued separately AFTER MECH-313 /
  MECH-314 / MECH-318 sibling substrates have landed.
  Design doc: REE_assembly/docs/architecture/mech_319_simulation_mode_rule_gate.md.
  See MECH-319 (this claim), MECH-094 (architectural principle this substrate
    instantiates; KEEP-AS-IS per Pull 3 R1 + Pull 4 R3 verdicts -- not modified),
    MECH-312 (parent arbitration layer whose write-gate this implements),
    MECH-312a/b/c/d (sub-mechanisms gated by the simulation tag),
    MECH-261 (mode-conditioned continuous write gate on SalienceCoordinator;
      complementary to MECH-319's categorical pre-gate),
    ARC-062 (rule apprehension cluster; GAP-K hosted this landing),
    MECH-313 / MECH-314 / MECH-318 (sibling ARC-065 / ARC-064 substrates landed
      in separate spawned tasks the same day),
    MECH-293 (ghost-goal probes; future call site that will pass simulation_mode=
      True through the gate),
    SD-033a (LateralPFCAnalog; one of the two wired arbitration-write call sites),
    GatedPolicy (ARC-062 Phase 1; the other wired arbitration-write call site),
    MECH-309 (logical-necessity claim motivating the broader ARC-062 cluster).
