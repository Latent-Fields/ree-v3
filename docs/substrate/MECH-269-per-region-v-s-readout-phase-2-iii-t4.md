## MECH-269 Per-Region V_s Readout -- Phase 2 (iii, T4) (2026-04-22)
- MECH-269 Phase 2 (iii, T4): hippocampal.per_region_verisimilitude --
  IMPLEMENTED 2026-04-22. Module: ree_core/hippocampal/module.py
  (HippocampalModule.update_per_region_vs,
  HippocampalModule.apply_invalidation_broadcasts_to_regions,
  HippocampalModule.reset_per_stream_vs extension).
  Promotes the flat per_stream_vs[stream] -> float readout to a
  per-region dict per_region_vs[(scale, segment_id)][stream] -> float
  keyed on AnchorSet (Phase 2 ii) active anchor keys. V_s foundation
  lit-pull verdict 3: per-stream V_s is the projection-readout of the
  integrated mixed-selectivity code; per-region keying provides the
  scale/segment partition so downstream consumers (MECH-284 staleness
  accumulator Phase 3; replay prioritisation; BG / E3 policy
  modulation) can query V_s for a specific region without collapsing
  across all active regions.
  Computation (Phase 1 identity-proxy parity, scoped per region):
    For each active anchor a on (scale, segment_id, stream_mixture):
      region_key = (scale, segment_id)  # stream_mixture dropped for readout
      for stream_name in config.per_stream_vs_streams:
        z_curr = LatentState[stream_name] (or GoalState.z_goal)
        z_prev = self._prev_region_stream_values[region_key][stream_name]
        if z_prev is None: V_s[region_key][stream_name] = 1.0 (seed)
        else:
          err = ||z_curr - z_prev|| / (||z_curr|| + 1e-6)
          score = clip_[0,1](1 - err)
          V_s[region_key][stream_name] = (1-tau)*prev_vs + tau*score
        z_prev <- z_curr
    Regions whose active anchor has disappeared since the previous tick
    (hysteresis mark_inactive from tick_hysteresis, FIFO cap eviction,
    or an earlier apply_invalidation_broadcasts_to_regions call this tick)
    are pruned from per_region_vs and _prev_region_stream_values.
  Invalidation broadcast reset path (MECH-287 consumer):
    apply_invalidation_broadcasts_to_regions(broadcasts) iterates
    BroadcastEvents; for each bcast on (source_scale, source_segment_id_old),
    drops per_region_vs[(scale, segment_id_old)] and mark_inactive's the
    matching active anchor. This is the T3 hysteresis-shortcut reset
    path described in the design doc: k=5 hysteresis is the passive
    path; broadcasts are the explicit-reset path. Idempotent: a
    second broadcast on an already-reset region returns [] and is
    otherwise a no-op.
  Config: HippocampalConfig.use_per_region_vs (bool, default False).
    Orthogonal to use_per_stream_vs -- per-region is a refinement,
    not a replacement; both can be on simultaneously. Requires
    use_anchor_sets=True to do anything (no-op without an anchor set
    to query). Per-stream tau shared with flat path via
    per_stream_vs_tau. Per-stream set shared via per_stream_vs_streams.
  State: per_region_vs: Dict[Tuple[str,str], Dict[str,float]] and
    _prev_region_stream_values: Dict[Tuple[str,str], Dict[str,Tensor]]
    on HippocampalModule. Both cleared by reset_per_stream_vs() on
    episode boundaries (extended in this pass).
  Agent wiring: REEAgent.sense(), immediately after tick_anchor_set
  (which consumes BoundaryEvents and advances hysteresis against
  per_stream_vs):
    if use_per_region_vs:
      broadcasts = list(hippocampal._broadcast_event_queue)  # peek, not drain
      if broadcasts: apply_invalidation_broadcasts_to_regions(broadcasts)
      update_per_region_vs(new_latent, goal_state=self.goal_state)
  Peek-not-drain on the broadcast queue: downstream Phase 3 consumers
  (MECH-284 staleness accumulator) still see the events after this
  tick. The dual consumption (tick_anchor_set's consume_boundary_events
  AND apply_invalidation_broadcasts_to_regions) is intentional: the
  first is the dual-trace remap path keyed on (scale, stream_mixture);
  the second is the explicit safety net keyed on
  (source_scale, source_segment_id_old).
  Backward compatible: use_per_region_vs=False by default. With flag
  OFF, per_region_vs stays empty, update_per_region_vs / apply_invalidation_broadcasts_to_regions
  are no-ops, reset_per_stream_vs extension is inert. 85/85 preflight
  + contracts PASS unchanged (bit-identical to pre-T4 HEAD).
  Activation smoke (2026-04-22, full MECH-269 stack ON + force_boundary):
    per_stream_vs populated as before (Phase 1);
    per_region_vs keys: [('fast', '0.1')] after one forced fast
    boundary; region V_s values non-trivial (0.89 / 0.97 / 0.96 under
    mild latent drift); active anchors reflect the new region.
    Flag OFF: per_region_vs stays empty across multiple sense() ticks.
  No trainable parameters. Pure arithmetic over latent norms + dict
  membership. No phased training needed.
  MECH-094: update_per_region_vs / apply_invalidation_broadcasts_to_regions
    are invoked only from REEAgent.sense() (waking observation stream).
    Simulation / replay paths must not route through sense(), so the
    hypothesis_tag gate is achieved by call-site scoping (same pattern
    as MECH-269 Phase 1 / Phase 2 ii, MECH-288, MECH-287).
  Contract tests: tests/contracts/test_mech_269_per_region_vs.py
    C1: default flag False; with flag OFF update_per_region_vs is a
        no-op even when anchors are present; flat per_stream_vs path
        continues to work.
    C2: per_region_vs populates on BoundaryEvent-installed anchor;
        (scale, segment_id_new) key present; streams seeded at 1.0.
    C3: cross-region isolation -- two active anchors on distinct
        (scale, segment_id) keys; marking one inactive prunes only
        that region's entry; the other region's cached V_s untouched.
    C4: MECH-287 broadcast on (source_scale, source_segment_id_old)
        drops only that region's entry AND mark_inactives the matching
        anchor; other region remains active. Idempotent.
    C5: hysteresis_k=5 honoured -- 5 consecutive below-threshold
        tick_hysteresis calls fire mark_inactive; subsequent
        update_per_region_vs prunes the per_region_vs entry.
    Plus 1 integration smoke test for reset_per_stream_vs clearing
    both flat and per-region state.
  Validation experiment: deferred to V3-EXQ-476 (combined cluster
    end-to-end validation with the full V_s invalidation circuit on;
    tests MECH-288 falsifiable prediction secondary -- z_goal / z_world
    broadcast events should preferentially reset their home-region V_s
    entries rather than peer regions). No standalone T4 EXQ queued in
    this pass (follow-up task per user spec).
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-269, MECH-288 (BoundaryEvent source via tick_anchor_set),
    MECH-287 (broadcast reset path consumer), MECH-284 (Phase 3
    staleness accumulator successor; reads per_region_vs), MECH-272
    (Phase 3 state-gated routing), MECH-094 (call-site scoping).
