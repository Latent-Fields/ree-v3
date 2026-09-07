## SD-033a: Lateral-PFC-analog / MECH-261 Primary Consumer (2026-04-20)
- SD-033a: pfc.lateral_pfc_analog -- IMPLEMENTED 2026-04-20.
  Module: ree_core/pfc/lateral_pfc_analog.py (LateralPFCAnalog,
  LateralPFCConfig). First subdivision of SD-033 (PFC subdivision cluster)
  and primary consumer of MECH-261's write-gate registry on SD-032a
  (SalienceCoordinator). Instantiates MECH-262 (rule-selective persistence).
  Config: REEConfig.use_lateral_pfc_analog (bool, default False).
  Sub-knobs: lateral_pfc_rule_dim (16), lateral_pfc_update_eta (0.05),
  lateral_pfc_world_pool_weight (0.5), lateral_pfc_bias_scale (0.1),
  lateral_pfc_hidden_dim (32).
  State: rule_state buffer [1, rule_dim], persistent across ticks within
  episode, reset on agent.reset(). Cross-episode carry-over is NOT
  implemented (V3 simplification; V4 extension if required).
  Update rule: rule_state <- (1 - eff_eta) * rule_state + eff_eta * source
    where eff_eta = update_eta * gate, gate = write_gate("sd_033a"),
    source = delta_proj(z_delta) + world_pool_weight * world_proj(z_world).
    Gate near 0 (internal_replay weight 0.05) -> rule-state near-frozen
    (distractor resistance). Gate near 1 (external_task / internal_planning)
    -> fast update.
  Bias head: frozen-random with last nn.Linear weights and bias ZEROED at
    init so initial bias output is EXACTLY zero. Head takes concat(
    [rule_state, per-candidate z_world summary]) -> scalar per candidate
    -> clamp [-bias_scale, +bias_scale] -> [K] bias vector added to
    dacc_score_bias before E3.select(). Training-dependent emergence
    (SD-033 signature iv) deferred: phased-training protocol not wired.
  Data flow: select_action() -> gate = salience.write_gate("sd_033a") (or
    1.0 if coord disabled) -> lateral_pfc.update(z_delta, z_world, gate)
    -> per-candidate summaries from trajectory.world_states[:, 0, :] ->
    lateral_pfc.compute_bias(summaries) -> add to dacc_score_bias ->
    e3.select(score_bias=...).
  Backward compatible: use_lateral_pfc_analog=False by default. When
    True with the zeroed-last-layer head, initial bias output is exactly
    zero -- agent runs bit-identical to baseline until the head is
    deliberately trained (deferred).
  Biological basis: Miller & Cohen 2001 (rule-as-top-down-bias), Badre
    & Nee 2018 (mid-lateral rule-hierarchy), Mansouri 2020 (rule-selective
    persistence). MECH-261 per-mode weights for sd_033a (from spec):
    external_task=1.0, internal_planning=1.0, internal_replay=0.05,
    offline_consolidation=0.3.
  MECH-094: rule persistence is gated by the MECH-261 registry (not by
    a separate hypothesis_tag check). MECH-261 generalises MECH-094:
    write_gate("sd_033a") = 0.05 in internal_replay mode means replay
    content cannot meaningfully update rule_state. The gate IS the tag.
  DESIGN ALTERNATIVES (documented in design doc, lit-pulls queued in
    task_inbox.md): A1 per-candidate vs uniform bias; A2 frozen-random
    head with zeroed last Linear vs trained head via phased protocol;
    A3 gate-modulated EMA vs recurrent GRU / synaptic-hold persistence.
  Smoke test (2026-04-20): module instantiates; gate=1.0 rule_state delta
    ~0.1 on single tick; gate=0.0 rule_state delta < 1e-6 (freeze); initial
    bias vector exactly zero; reset() zeroes rule_state. E2E five-tick
    loop with SD-033a ON hits the same pre-existing multinomial-on-
    untrained-E3-scores edge case that SD-033a-OFF also hits; confirmed
    not caused by this SD.
  Validation experiment: V3-EXQ-456 queued (diagnostic -- five sub-tests:
    instantiation, gate modulates update rate, bias reaches E3 with
    zero-init contract, backward compat, reset clears rule_state).
  Phased training: deferred until A2 alternative is considered.
  Design doc: REE_assembly/docs/architecture/sd_033a_lateral_pfc_analog.md
  See SD-033, SD-033a, MECH-261, MECH-262, SD-032a, SD-032b, MECH-094.
