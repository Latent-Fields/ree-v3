## MECH-287 Invalidation Trigger -- Phase 2 iv (2026-04-22)
- MECH-287: regulators.invalidation_trigger -- IMPLEMENTED 2026-04-22.
  Module: ree_core/regulators/invalidation_trigger.py (InvalidationTrigger
  + BroadcastEvent dataclass). Phase 2 iv of the V_s invalidation
  runtime (architecture doc: REE_assembly/docs/architecture/
  v_s_invalidation_runtime.md). Subscribes to MECH-288 BoundaryEvents
  emitted in agent.sense() and re-emits them as graded BroadcastEvent
  objects. Graded output: broadcast_strength = posterior * gain (NO
  binary thresholding of strength). Downstream consumers (MECH-269
  anchor-reset -- T3; MECH-284 staleness accumulator -- Phase 3) drain
  via HippocampalModule.drain_broadcast_events().

  VERDICT-3 ARCHITECTURAL COMMITMENT (option c, V_s foundation lit-pull
  SYNTHESIS verdict 3): the trigger is a BoundaryEvent subscriber, NOT
  an independent comparator. The upstream CA1/CA3 mismatch comparator
  stage (Vinogradova 2001; O'Mara 2009; Lisman & Grace 2005) -- per
  MECH-287's dual-component biological substrate in the claim entry --
  is collapsed HERE to a subscription on the MECH-288 boundary queue.
  The biological-substrate text in claims.yaml remains accurate (biology
  IS a two-stage loop); the implementation collapses ComparatorStage to
  a subscriber. Whether to refactor MECH-287's claim text to make this
  explicit is a downstream governance decision -- NOT resolved in this
  commit.

  Phasic/tonic guardrail (Aston-Jones & Cohen 2005; Clewett 2025 failure
  signature 2): rolling-mean tonic estimate over config.tonic_window
  past-tick aggregated posteriors. If the estimate (measured BEFORE the
  current tick) exceeds config.tonic_threshold, the whole tick's phasic
  broadcast is suppressed (each suppressed BoundaryEvent increments
  n_suppressed). Passive decay via rolling window: once high-frequency
  boundary activity stops, the estimate falls below threshold in
  tonic_window+1 quiet ticks and broadcast resumes.

  Config: InvalidationTriggerConfig (ree_core/utils/config.py) --
  gain=1.0, targets=("mech_269_anchor_set",), tonic_threshold=0.5,
  tonic_window=50. HippocampalConfig.use_invalidation_trigger (default
  False); HippocampalConfig.invalidation_trigger (default factory).

  BroadcastEvent payload: t, strength (posterior * gain), posterior
  (inherited from BoundaryEvent, in [0, 1]), targets (list from config),
  source_scale, source_segment_id_old, source_segment_id_new,
  source_sources (original BoundaryEvent.sources).

  HippocampalModule: instantiates invalidation_trigger when flag is on;
  exposes _broadcast_event_queue (List[BroadcastEvent]),
  drain_broadcast_events() -> List[BroadcastEvent] (list + clear),
  reset_invalidation_trigger() (per-episode reset of tonic history /
  counters / queue).

  Agent wiring:
    REEAgent.sense() -- immediately AFTER the event_segmenter.step()
      call and the _boundary_event_queue extend (so this tick's
      BoundaryEvents are visible). If use_invalidation_trigger is on
      AND the segmenter produced events, the trigger is ticked with
      them and the resulting BroadcastEvents are appended to
      hippocampal._broadcast_event_queue. If use_invalidation_trigger
      is on but use_event_segmenter is OFF, the trigger is ticked with
      an empty boundary list so its tonic history advances in lockstep
      with the clock -- no broadcasts can fire (C5 dissociation).
    REEAgent.reset() -- after reset_event_segmenter:
      if use_invalidation_trigger: hippocampal.reset_invalidation_trigger().

  Backward compatible: use_invalidation_trigger=False by default;
  invalidation_trigger is None; _broadcast_event_queue stays empty;
  drain_broadcast_events() returns []. Regression: 70/70 contracts +
  7/7 preflight PASS with flag OFF (bit-identical to pre-MECH-287 HEAD).
  Activation smoke (2026-04-22): default agent constructed with
  use_invalidation_trigger=True + use_event_segmenter=True instantiates
  InvalidationTrigger; reset clears tonic_estimate to 0.0; broadcast
  queue empty on construction.

  No trainable parameters. Pure arithmetic (rolling mean on boundary
  posteriors). No phased training needed.

  MECH-094: hypothesis_tag is NOT checked inside the trigger. The
  segmenter feeds only from REEAgent.sense() (waking observation
  stream). Forced BoundaryEvents via EventSegmenter.force_boundary()
  would flow through the trigger as real broadcasts -- intentional
  (caller is responsible for the MECH-094 gate at the force-boundary
  call site).

  Contract tests: tests/contracts/test_mech_287_invalidation_trigger.py
    C1: default config backward-compat; invalidation_trigger is None
        when flag is off; drain queue empty.
    C2: BoundaryEvent arrival fires BroadcastEvent with strength =
        posterior * gain; source payload preserved.
    C3: graded posterior -> graded broadcast across [0.01 .. 1.0]
        (NO binary threshold).
    C4: tonic guardrail suppresses next phasic under sustained high-
        activity period; reopens after tonic_window+1 quiet ticks.
    C5: verdict-3 dissociation -- with event_segmenter lesioned (no
        BoundaryEvents queued), trigger never fires a broadcast
        regardless of internal state (including synthetically elevated
        tonic history). This is the falsifiable tertiary prediction
        for MECH-288 and validates the option-c implementation choice.

  Validation experiment: deferred to Phase 3 (T3 wires the MECH-269
  anchor-reset consumer). V3-EXQ-476 (re-run of EXQ-475 with full V_s
  invalidation circuit on) remains the combined-cluster end-to-end
  validation experiment.
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-287, MECH-288 (upstream BoundaryEvent emitter), MECH-269
  (Phase 3 anchor-reset consumer), MECH-284 (Phase 3 staleness
  accumulator consumer), MECH-272 (Phase 3 state-gated routing).
