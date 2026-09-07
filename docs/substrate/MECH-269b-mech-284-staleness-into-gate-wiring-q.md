## MECH-269b + MECH-284 Staleness-into-Gate Wiring (Q-040b strong reading, 2026-04-29)
- MECH-269b + MECH-284: cortical_world_model.regional_verisimilitude_rollout_gating
  STALENESS-WIRING IMPLEMENTED 2026-04-29.
  Modules: ree_core/regulators/vs_rollout_gate.py (VsRolloutGateConfig.use_staleness_lookup,
  gate / gate_stream / _gate_value extended with per_stream_staleness kwarg);
  ree_core/hippocampal/module.py (HippocampalModule.compute_per_stream_staleness);
  ree_core/agent.py (REEAgent._refresh_vs_gate_staleness, cached per-tick;
  precondition raises on missing use_staleness_accumulator / use_anchor_sets).
  The 2026-04-26 substrate compared raw per_stream_vs[s] to a fixed threshold
  (default 0.4); EXQ-490/490b/490c all had to override that threshold to smoke
  values (0.85/0.85/0.95) to make the gate fire at realistic V_s readings.
  This update wires MECH-284 region staleness into the comparison:
    effective_vs = raw_vs - per_stream_staleness[s]
  with per_stream_staleness aggregated as
    staleness[s] = max over active anchors a where s in a.stream_mixture
                   of staleness_accumulator.lookup_by_anchor_key(a.key)
  (max captures worst-case region staleness the stream is exposed to). Cached
  once per waking tick; reused by all gate / gate_stream call sites in the
  same tick.
  Config: REEConfig.from_dims(use_vs_gate_staleness_lookup=False default).
  When True, agent build raises ValueError unless use_staleness_accumulator
  AND use_anchor_sets are also True.
  Backward compatible: flag-OFF gate behaviour bit-identical to legacy raw-V_s
  path. 191/191 preflight + contracts PASS with the wiring landed.
  Q-040b strong reading: V3-EXQ-490c successor (490d) can drop the smoke
  threshold override and verify the hold path fires only when sustained
  MECH-287 broadcasts have accumulated region staleness. C4 severance arm
  (use_vs_gate_staleness_lookup=False vs True at matched thresholds) becomes
  the falsifiable test of the strong reading.
  MECH-094: aggregator + refresh helper are call-site-scoped to waking
  paths (_e1_tick); replay / simulation paths do not invoke them.
  Contract tests: tests/contracts/test_mech_269b_vs_rollout_gate_staleness.py
    C1: VsRolloutGateConfig.use_staleness_lookup defaults False;
        HippocampalConfig.use_vs_gate_staleness_lookup defaults False.
    C2: flag OFF -- supplied per_stream_staleness ignored (raw-V_s path).
    C3: flag ON without dict -- falls back to raw V_s (staleness=0 default).
    C4: flag ON with dict pushing effective_vs below threshold -- hold fires
        and snapshot is substituted.
    C5: per-stream isolation -- staleness on z_world does not affect z_self.
    C6: HippocampalModule.compute_per_stream_staleness max-over-anchors with
        stream_mixture overlap.
    C7: diagnostics (vs_gate_staleness_lookup_calls,
        vs_gate_max_staleness_<stream>) populated and cleared by reset.
    C8: agent precondition raises on missing accumulator / anchor substrates.
  Validation experiment: V3-EXQ-601 PASS 2026-05-21 (MECH-269b-followup-A
  severance at default 0.4/0.5 thresholds; evidence_direction=supports).
  Design doc: REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md
  (new section "MECH-284 staleness wiring (Q-040b strong reading,
  2026-04-29)").
  See MECH-269b, MECH-284, MECH-269 (Phase 1 / Phase 2 ii / Phase 2 iii),
  MECH-287 (broadcast trigger -- staleness source), MECH-094 (call-site
  scoping), Q-040, Q-040b.
