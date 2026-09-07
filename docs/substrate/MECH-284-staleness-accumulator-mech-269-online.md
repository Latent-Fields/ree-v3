## MECH-284 Staleness Accumulator + MECH-269 Online Hysteresis -- Phase 3 (2026-04-24)
- MECH-284: hippocampal.staleness_accumulator -- IMPLEMENTED 2026-04-24.
  Module: ree_core/hippocampal/staleness_accumulator.py (StalenessAccumulator,
  StalenessAccumulatorConfig, RegionKey). Phase 3 of the V_s invalidation
  runtime (architecture doc: REE_assembly/docs/architecture/
  v_s_invalidation_runtime.md). Region-indexed residual schema-staleness
  accumulator. Integrates MECH-287 BroadcastEvents against the currently
  active MECH-269 anchor set with an attribution weight, decays per tick,
  and exposes a getter consumed by MECH-269 online anchor-reset
  hysteresis (the online arm of the dual-readout; MECH-285 offline
  sleep-priority arm is deferred).
  Region key: (scale, segment_id) -- stream_mixture dropped to match the
  Phase 2 (iii, T4) per_region_vs partition. One (scale, segment_id)
  region reachable by multiple stream_mixture families has its staleness
  merged on the region bucket.
  Operational definition (per claims.yaml refinement 2026-04-22):
    for each schema region r in active_anchor_set(t):
      if MECH-287 trigger(t):
        staleness[r] += attribution_weight(r, source_streams) * magnitude
      staleness[r] *= leak_factor
  Attribution modes (config.attribution_mode):
    "equal"          -- 1/N uniform credit across N active anchors.
    "stream_overlap" -- |source_sources & stream_mixture| /
                        max(|source_sources|, 1) per anchor; cheap
                        cosine-similarity surrogate over stream-name
                        sets. Anchor with zero overlap gets zero credit.
  Staleness is clipped at config.staleness_clip (default 1.0) so
  V_s_anchor = V_s(r) - staleness[r] stays in [-1, 1] whether the
  Phase 2 proxy or Phase 3 lookup drives hysteresis.
  Config: HippocampalConfig.use_staleness_accumulator (bool, default
  False); HippocampalConfig.staleness_accumulator (StalenessAccumulatorConfig,
  default factory). StalenessAccumulatorConfig: leak_factor=0.995,
  attribution_mode="equal", staleness_clip=1.0, drop_epsilon=1e-6.
  MECH-269 online hysteresis swap:
    HippocampalConfig.use_mech284_hysteresis (bool, default False).
    When both use_staleness_accumulator AND use_mech284_hysteresis are
    True, AnchorSet.tick_hysteresis() receives a staleness_lookup
    callable pointing at StalenessAccumulator.lookup_by_anchor_key.
    V_s_anchor = V_s(r) - staleness_lookup(anchor_key). With
    use_staleness_accumulator ON but use_mech284_hysteresis OFF, the
    accumulator is populated as a diagnostic only; hysteresis continues
    to use the Phase 2 internal proxy ((tick - last_accessed) *
    staleness_rate).
  Integration site (HippocampalModule.tick_anchor_set):
    consume_boundary_events (MECH-269 Phase 2 ii) -> integrate broadcasts
    against active anchors (peek, not drain; MECH-287 consumers that
    run after tick_anchor_set still see the queue) -> tick_leak ->
    tick_hysteresis (with staleness_lookup if MECH-284 hysteresis is on).
    This ordering preserves the "this-tick broadcasts affect this-tick
    V_s_anchor check" semantic.
  HippocampalModule public API additions:
    integrate_staleness(broadcasts) -- explicit credit path for code
      that wants to integrate outside the tick_anchor_set cycle; no-op
      when accumulator is disabled. Applies leak after integration.
    reset_staleness_accumulator() -- per-episode reset of region map +
      diagnostic counters.
  Agent wiring (REEAgent):
    reset() -- after reset_anchor_set: if use_staleness_accumulator is
      on, hippocampal.reset_staleness_accumulator().
    sense() -- no additional call-site required: the existing
      tick_anchor_set call handles integration internally via a peek of
      the _broadcast_event_queue populated earlier in the same sense()
      tick by MECH-287.
  Backward compatible: use_staleness_accumulator=False by default;
    staleness_accumulator is None; tick_anchor_set follows the legacy
    Phase 2 path (no integration, no leak, no staleness_lookup). 91/91
    preflight + contracts PASS with flag OFF (bit-identical to pre-
    Phase-3 HEAD, 2026-04-24).
  Activation smoke (2026-04-24, two ARMs):
    ARM1 (use_staleness_accumulator=True, use_mech284_hysteresis=False):
      Two active anchors on (fast, 0.1) and (fast, 0.2) with distinct
      stream_mixtures; one synthetic BroadcastEvent with strength=1.0
      injected; tick_anchor_set called -> snapshot:
        (fast, 0.1): 0.4975, (fast, 0.2): 0.4975
      (0.5 equal credit * leak 0.995); stats: n_integrations=1,
      n_leak_ticks=1, n_regions=2, max_staleness=0.4975. Reset clears
      map + counters. PASS.
    ARM2 (use_staleness_accumulator=True, use_mech284_hysteresis=True):
      staleness_rate=0.0 (passive proxy off), hysteresis_k=3,
      reset_threshold=0.5, per_stream_vs held at 1.0. Inject staleness=0.9
      on region key each tick; tick_anchor_set ticks 3 times -> anchor
      marked inactive on tick 3 (below_threshold_streak=3). Confirms
      staleness_lookup path is exercised under the swap. PASS.
  No trainable parameters. Pure float arithmetic + dict state. No phased
  training needed.
  MECH-094: integrate() is invoked only from HippocampalModule.integrate_staleness
    and HippocampalModule.tick_anchor_set, both of which are called from
    REEAgent.sense() (waking observation stream). Simulation / replay
    paths must not route through these; hypothesis_tag gating is achieved
    by call-site scoping (same pattern as MECH-269 Phase 1 / Phase 2 ii
    / Phase 2 iii, MECH-288, MECH-287).
  Validation experiment: V3-EXQ-478 queued (Phase 3 diagnostic ablation:
    OFF vs ON x 2 seeds; metrics freeze_recommit_count, anchor_reset_count,
    mean_staleness_peak, action_class_entropy). Also unblocks previously
    gated combined-cluster validations (V3-EXQ-445d, V3-EXQ-449c,
    V3-EXQ-455a, V3-EXQ-476/475 re-run).
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-284, MECH-269 (Phase 1 + 2 ii + 2 iii; online-arm consumer),
    MECH-287 (broadcast event source), MECH-288 (boundary segmenter),
    MECH-285 (offline sleep-priority readout, deferred), MECH-272
    (Phase 3 state-gated routing), MECH-094 (call-site scoping).
