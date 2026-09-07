## MECH-269 Anchor Sets -- Phase 2 (ii) (2026-04-22)
- MECH-269 Phase 2 (ii): hippocampal.anchor_sets -- IMPLEMENTED 2026-04-22.
  Module: ree_core/hippocampal/anchor_set.py (AnchorSet, Anchor, AnchorKey).
  Phase 2 (ii) of the V_s invalidation runtime. Scale-tagged hippocampal
  anchor store with dual-trace preservation (Bouton 2004) and k-consecutive
  hysteresis on V_s_anchor crossings. Consumes MECH-288 BoundaryEvents
  (via HippocampalModule) to install / remap anchors keyed on
  (scale, segment_id, stream_mixture).
  Key schema:
    AnchorKey = (scale: str, segment_id: str, stream_mixture: tuple[str, ...])
  Phase 2 stand-in for stream_mixture: tuple(sorted(per_stream_vs.keys()))
  at anchor-creation tick. Learned attribution head deferred to Phase 3
  (MECH-284); this gives a deterministic, observable stream-membership
  signature sufficient for the first end-to-end validation.
  Dual-trace routing (Bouton 2004): on remap, the outgoing active anchor
  on (scale, stream_mixture) is marked INACTIVE (not erased) and retained
  in all_anchors() for retrieval / replay consumers; excluded from
  active_anchors(). Erase is never the resolution path.
  Hysteresis: per-anchor below_threshold_streak counter on
  V_s_anchor = avg(V_s over mixture) - staleness (staleness monotonic in
  (tick - last_accessed) * staleness_rate, clipped at staleness_clip).
  Streak increments when V_s_anchor < reset_threshold; resets to 0 on any
  tick at-or-above threshold. At hysteresis_k consecutive below-threshold
  ticks (default 5), the active anchor is marked inactive and returned.
  Config: HippocampalConfig.use_anchor_sets (bool, default False);
  HippocampalConfig.anchor_set (AnchorSetConfig, default factory).
  AnchorSetConfig: scales=("fast","slow"), reset_threshold=0.3,
  hysteresis_k=5, staleness_rate=0.005, staleness_clip=1.0,
  max_anchors_per_scale=128, subscribe_to_boundary_events=True.
  FIFO soft-cap: when active_per_scale exceeds max_anchors_per_scale, the
  oldest (smallest created_at) active anchor in that scale is marked
  inactive. Inactive anchors are preserved.
  HippocampalModule: instantiates anchor_set when flag is on; exposes
  tick_anchor_set(latent_state, events) and reset_anchor_set(). Stream
  mixture is built as tuple(sorted(self.per_stream_vs.keys())) at tick
  time (populated earlier in the same sense() tick by MECH-269 Phase 1).
  Agent wiring:
    REEAgent.sense() -- after per_stream_vs update, with the current
      tick's events list (local var from the event_segmenter branch,
      empty if segmenter is off or fired nothing): if use_anchor_sets
      is on, hippocampal.tick_anchor_set(new_latent, events) is called.
      tick_anchor_set consumes the events (write_anchor per registered
      scale, dual-trace remap internally) then advances
      tick_hysteresis(per_stream_vs).
    REEAgent.reset() -- after reset_invalidation_trigger:
      if use_anchor_sets: hippocampal.reset_anchor_set().
  Public API: write_anchor(scale, segment_id, stream_mixture, z_world)
  -> Anchor; get_anchor(...) -> Optional[Anchor] (refreshes last_accessed);
  mark_inactive(scale, stream_mixture) -> Optional[Anchor];
  reset_region(scale, stream_mixture, new_segment_id, z_world) -> Anchor
  (dual-trace remap; mark_inactive + write_anchor in one call);
  tick_hysteresis(per_stream_vs) -> List[Anchor] (fired this tick);
  consume_boundary_events(events, z_world, stream_mixture) -> List[Anchor]
  (skips scales not in config.scales; skips when z_world is None);
  active_anchors(scale=None) -> List[Anchor]; all_anchors(scale=None)
  -> List[Anchor]; reset() (per-episode: clears active + inactive +
  tick counter).
  Backward compatible: use_anchor_sets=False by default; HippocampalModule.
  anchor_set is None; tick_anchor_set / reset_anchor_set are no-ops.
  85/85 preflight + contracts PASS with flag OFF (bit-identical to
  pre-anchor-set HEAD). Contract tests all pass with flag ON.
  No trainable parameters. Pure arithmetic over latent norms + tick
  counters + detached z_world clones. No phased training needed.
  MECH-094: write_anchor is invoked only from HippocampalModule.tick_anchor_set,
  which is called from REEAgent.sense() (waking observation stream).
  Simulation / replay paths must not route through tick_anchor_set.
  hypothesis_tag gating is therefore achieved by call-site scoping, not
  by an inline tag check (same pattern as MECH-269 Phase 1, MECH-288,
  MECH-287).
  Contract tests: tests/contracts/test_mech_269_anchor_set.py
    C1: default config backward-compat; use_anchor_sets defaults False;
        HippocampalModule.anchor_set is None; tick/reset hooks no-op.
    C2: BoundaryEvent on registered scale installs active anchor with
        correct (scale, segment_id_new, stream_mixture) key; unregistered
        scale ignored.
    C3: second BoundaryEvent on same (scale, stream_mixture) family
        marks prior anchor INACTIVE (not erased); prior retained in
        all_anchors(); exactly one active anchor on the family.
    C4a: k-1 below-threshold ticks then at-or-above resets streak;
         anchor stays active.
    C4b: k consecutive below-threshold ticks fire the reset on the k-th
         tick; inactive anchor retained (dual-trace).
    C5: reset_region marks current active inactive and installs new
        active; both retained in all_anchors().
    C6: per-episode reset() clears active + inactive anchor stores and
        resets the internal tick counter.
    Plus 2 integration smoke tests verifying agent-level flag OFF is
    no-op and flag ON installs anchors via tick_anchor_set with
    stream_mixture drawn from sorted per_stream_vs keys.
  Validation experiment: deferred to V3-EXQ-476 (combined cluster end-
  to-end validation with the full V_s invalidation circuit on). No
  standalone Phase 2 (ii) EXQ is queued -- approved by user 2026-04-22
  in favour of the combined-cluster validation.
  Design doc: REE_assembly/docs/architecture/hippocampal_anchor_selection.md
  See MECH-269, MECH-288 (BoundaryEvent source), MECH-287 (Phase 3
  broadcast consumer for remap), MECH-284 (Phase 3 staleness accumulator
  successor to the local proxy), MECH-272 (Phase 3 state-gated routing),
  MECH-094 (waking-stream call-site scoping).
