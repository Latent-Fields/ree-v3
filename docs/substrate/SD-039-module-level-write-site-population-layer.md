## SD-039 Module-Level Write-Site Population Layer (2026-04-27)
- SD-039 population: hippocampal.anchor_goal_payload_population -- IMPLEMENTED 2026-04-27.
  Modules: ree_core/hippocampal/module.py (HippocampalModule.build_goal_payload,
  tick_anchor_set goal_payload kwarg, apply_invalidation_broadcasts_to_regions
  goal_payload kwarg); ree_core/agent.py (REEAgent.sense() builds payload once
  per tick and threads it through both write/remap and broadcast-invalidate
  call sites); ree_core/utils/config.py (REEConfig.from_dims accepts
  use_sd039_anchor_payload, propagates to AnchorSetConfig).
  Wires the deferred follow-on to the SD-039 substrate (landed 2026-04-26):
  REEAgent / HippocampalModule now populate AnchorGoalPayload from the
  current waking-stream signals at every anchor write / remap / invalidate
  site so that MECH-292 / MECH-293 consumers see live motivational state on
  both halves of the dual trace.
  Sourcing (build_goal_payload):
    z_goal_snapshot     <- goal_state.z_goal.detach().clone() when
                          goal_state.is_active(); None otherwise.
    wanting_strength    <- residue_field.evaluate_valence(z_world)[..., VALENCE_WANTING].mean()
                          when residue + valence_enabled; 0.0 otherwise.
    arousal_tag         <- bla_output.arousal_tag when supplied; 0.0 otherwise.
    last_vs             <- mean(self.per_stream_vs.values()) when non-empty;
                          None otherwise. Phase 2 ii proxy for the parent
                          (scale, stream_mixture) family V_s -- the payload
                          is shared across all anchors written this tick.
    staleness_at_write  <- max(staleness_accumulator.snapshot().values())
                          when MECH-284 accumulator is enabled; None otherwise.
                          Region-keyed; max-across-regions is the most
                          informative scalar for downstream MECH-292 ranking.
    payload_written_step <- agent._step_count (anchor_set._tick fallback).
  build_goal_payload returns None (skipping population entirely) when:
    - the AnchorSet substrate is disabled (anchor_set is None),
    - AnchorSetConfig.use_sd039_anchor_payload is False (master flag OFF),
    - simulation_mode=True (MECH-094 gate; replay/DMN paths must not
      populate payloads from waking signals).
  Wiring sites in agent.sense():
    1. After update_per_stream_vs (Phase 1 V_s populated): build payload.
    2. tick_anchor_set(latent, events, goal_payload=...): boundary-event
       write/remap path; consume_boundary_events forwards the payload to
       each per-event write_anchor (the dual-trace remap path internally
       writes the payload onto BOTH outgoing inactive trace and the new
       active anchor when same family is replaced).
    3. apply_invalidation_broadcasts_to_regions(broadcasts, goal_payload=...):
       MECH-287 broadcast-driven mark_inactive path; payload is refreshed on
       the outgoing anchor at the moment of broadcast invalidation.
    Hysteresis-fired mark_inactive (inside tick_hysteresis) does NOT refresh
    payload -- the prior payload is preserved as the cause-of-blockage trace
    per dual-trace semantics.
  Config: REEConfig.from_dims(use_sd039_anchor_payload=False) propagates to
  config.hippocampal.anchor_set.use_sd039_anchor_payload. Backward
  compatible: master flag default False; agent.sense build_goal_payload
  returns None; tick_anchor_set / apply_invalidation_broadcasts_to_regions
  receive goal_payload=None and behaviour is bit-identical to pre-SD-039.
  170/171 preflight + contracts PASS with population layer landed; the
  remaining failure is unrelated queue-housekeeping (V3-EXQ-418e / 490
  completion-record duplication).
  No trainable parameters. No phased training needed. ASCII-safe (no
  print() output added).
  MECH-094: build_goal_payload accepts simulation_mode argument; sense()
  passes simulation_mode=False (waking observation stream). Hysteresis
  invalidation has no fresh-state context and intentionally leaves the
  prior payload preserved.
  Validation experiment: V3-EXQ-494 6/6 PASS 2026-04-27 (UC1 module
  importable; UC2 master OFF no-op; UC3 population_fires 7/7 anchors with
  populated payloads, max_goal_match 0.9999; UC4 dual-trace preservation
  6 inactive + 1 active all carry payloads; UC5 falsifiable signature
  Phase A mean=0.0 vs Phase B mean=0.998 with 3/3 above 0.3; UC6 MECH-094
  simulation gate -- replay path produces zero anchors with populated
  payload). Validation script extends _step_episode helper to force
  MECH-288 fast-scale boundary events every 8 ticks via
  event_segmenter.force_boundary so the SD-039 contract is exercised
  without depending on stochastic boundary firing within the test window.
  Design doc: REE_assembly/docs/architecture/sd_039_anchor_goal_payload.md
  See SD-039 (parent claim), MECH-269 Phase 2 (ii) anchor substrate,
  MECH-287 broadcast trigger (invalidation site), MECH-284 staleness
  accumulator (staleness_at_write source), MECH-216 predictive wanting
  (wanting_strength source), MECH-230 z_goal structure (z_goal_snapshot
  source), MECH-292 / MECH-293 / ARC-060 (downstream ghost-goal
  consumers), MECH-094 (simulation gate).
