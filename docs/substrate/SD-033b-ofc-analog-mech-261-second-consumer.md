## SD-033b: OFC-analog / MECH-261 Second Consumer (2026-04-26)
- SD-033b: pfc.ofc_analog -- IMPLEMENTED 2026-04-26.
  Module: ree_core/pfc/ofc_analog.py (OFCAnalog, OFCConfig). Second
  subdivision of SD-033 (PFC subdivision cluster) and second consumer
  of MECH-261's write-gate registry on SD-032a (SalienceCoordinator).
  Substrate for MECH-263 functional signatures (devaluation sensitivity,
  same-sensory / different-task-role discrimination); the behavioural
  signatures themselves are deferred to environment-extension EXQs.
  Config: REEConfig.use_ofc_analog (bool, default False).
  Sub-knobs: ofc_state_dim (16), ofc_update_eta (0.05),
  ofc_outcome_pool_weight (0.5), ofc_bias_scale (0.1),
  ofc_hidden_dim (32), ofc_harm_dim (0). harm_dim=0 (default) builds the
  state_code from z_world only; setting harm_dim to the SD-011 z_harm
  dim turns on outcome_proj so harm-magnitude information enters the
  outcome-state code (the architectural shape MECH-263 devaluation
  sensitivity probes).
  State: state_code buffer [1, state_dim], persistent across ticks
  within episode, reset on agent.reset(). Cross-episode carry-over not
  implemented (V3 simplification, parallel to SD-033a).
  Update rule: state_code <- (1 - eff_eta) * state_code + eff_eta * source
    where eff_eta = update_eta * gate, gate = write_gate("sd_033b"),
    source = world_proj(z_world).mean(0)
           + outcome_pool_weight * outcome_proj(z_harm).mean(0) (if harm_dim>0).
    Gate near 0 (internal_replay weight 0.05) -> state_code near-frozen.
    Gate near 1 (external_task weight 1.0) -> fast update. internal_planning
    weight 0.5 (vs 1.0 for sd_033a) reflects partial replanning during
    planning rollouts.
  Bias head: frozen-random with last nn.Linear weights and bias ZEROED at
    init so initial bias output is EXACTLY zero. Head takes concat(
    [state_code, per-candidate z_world summary]) -> scalar per candidate
    -> clamp [-bias_scale, +bias_scale] -> [K] bias vector added to
    dacc_score_bias before E3.select(). Training-dependent emergence
    deferred along with MECH-263 behavioural signatures (env extension
    required: outcome relabelling, task-role-distinct state pairs).
  Data flow: select_action() -> gate = salience.write_gate("sd_033b") (or
    1.0 if coord disabled) -> ofc.update(z_world, z_harm-if-harm_dim>0,
    gate) -> per-candidate summaries (reused from lateral_pfc tick block
    when SD-033a also on; built fresh otherwise) -> ofc.compute_bias(
    summaries) -> add to dacc_score_bias -> e3.select(score_bias=...).
  Backward compatible: use_ofc_analog=False by default. When True with
    the zeroed-last-layer head, initial bias output is exactly zero --
    agent runs bit-identical to baseline until the head is deliberately
    trained. 143/143 contracts PASS with substrate landed.
  Biological basis: MECH-263 OFC functional signatures (devaluation
    sensitivity, same-sensory / different-task-role discrimination).
    MECH-261 per-mode weights for sd_033b (from spec): external_task=1.0,
    internal_planning=0.5, internal_replay=0.05, offline_consolidation=0.3.
  MECH-094: handled via MECH-261 generalisation. write_gate("sd_033b")=
    0.05 in internal_replay means replay content cannot meaningfully
    update state_code. The gate IS the tag.
  Smoke test (2026-04-26): module instantiates; gate=1.0 state_code delta
    ~0.27 on single tick; gate=0.0 state_code delta exactly 0.0 (freeze);
    initial bias vector max-abs exactly zero; reset() zeroes state_code.
    EXQ-485 5/5 sub-tests PASS.
  Oracle path (MECH-263, 2026-05-04): OFCConfig.use_outcome_oracle (bool,
    default False). When True, OFCAnalog.query_outcome(z_harm_s, action,
    e2_harm_s) delegates to E2HarmSForward.forward() with no_grad + detach;
    raises AssertionError when oracle is disabled. REEAgent stores
    _ofc_oracle_predictions (per-candidate oracle list, cleared on reset).
    REEConfig.use_ofc_outcome_oracle (default False) wired through from_dims().
    Bit-identical OFF: the oracle block in select_action() is entirely gated
    by oracle_is_ready (False when use_outcome_oracle=False) so all existing
    experiments are unaffected. 7/7 preflight + 184/184 contracts PASS with
    oracle OFF.
  Validation experiment: V3-EXQ-485 queued (diagnostic -- five sub-tests:
    instantiation + state_code shape, gate=1 vs gate=0 update modulation,
    bias zero at init, backward compat, reset clears state_code). Smoke
    PASS 2026-04-26. V3-EXQ-485a queued (6-sub-test oracle round-trip
    extension; UC6 new: A query_outcome() matches e2_harm_s.forward() <1e-6
    diff; B AssertionError when oracle disabled; C reset() clears prediction
    caches; D oracle_is_ready=False by default; E get_state() exposes oracle
    diagnostics). Behavioural validation (MECH-263 devaluation + task-role
    discrimination) deferred to env-extension EXQs.
  Phased training: deferred along with MECH-263 behavioural signatures.
  Design doc: REE_assembly/docs/architecture/sd_033b_ofc_analog.md
  See SD-033, SD-033a (sibling consumer; additive E3 bias composition),
    SD-033b, MECH-261, MECH-263, SD-032a, SD-032b, MECH-094.
