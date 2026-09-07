## MECH-288 Event Segmenter -- Phase 2 (2026-04-22)
- MECH-288: hippocampal.event_segmenter -- IMPLEMENTED 2026-04-22.
  Module: ree_core/hippocampal/event_segmenter.py (EventSegmenter,
  BoundaryEvent, Scale, _PEThresholdDetector, _BOCPDGaussianDetector).
  Phase 2 of the V_s invalidation runtime (architecture doc:
  REE_assembly/docs/architecture/v_s_invalidation_runtime.md). Emits
  BoundaryEvent objects with nested outer.inner segment IDs at
  event-scale transitions. Downstream consumers are MECH-287
  (broadcast invalidation trigger) and MECH-269 anchor-reset
  hysteresis; the module queues BoundaryEvents on HippocampalModule
  for those consumers to drain.
  Canonical two-scale config (EventSegmenterConfig defaults):
    fast: pe_threshold on (z_world, z_self); pe_window_length=200,
          pe_threshold=0.65, tau=1, min_segment_length=2.
    slow: bocpd_gaussian on (z_goal,); hazard=1/40,
          posterior_threshold=0.5, bocpd_top_k=20, bocpd_prior_var=1.0,
          tau=40, min_segment_length=15.
  BoundaryEvent payload: segment_id_old, segment_id_new (both
  "outer.inner" strings), scale, posterior, sources (list[str]), t.
  Hierarchical rule: slow fire forces outer+=1, inner=0 and suppresses
  a same-tick fast event (slow owns the inner reset). Fast fire
  increments inner only. force_boundary(scale, reason) bypasses
  min_segment_length (supervised / scripted API hook).
  BOCPD implementation: Adams & MacKay 2007 recursion with Welford
  online variance per run. Top-k pruning keeps the posterior O(1).
  Underflow-robust: if every existing run-hypothesis assigns
  negligible log-probability (max(pred_log) < -20) to the observation,
  the regime is treated as a decisive change-point -- fire with
  posterior=1.0 and reseed the posterior. Mirrors the literal
  total<=0 underflow path.
  Config: HippocampalConfig.use_event_segmenter (bool, default False),
  HippocampalConfig.event_segmenter (EventSegmenterConfig; default
  canonical two-scale above). EventSegmenterConfig.scales is a list
  of EventSegmenterScaleConfig entries; emit_to defaults to
  ["mech_287_broadcast", "mech_269_anchor_set"]; scale_id_format
  "{outer}.{inner}"; slow_scale_name "slow".
  HippocampalModule: instantiates event_segmenter when flag is on;
  exposes _boundary_event_queue (List[BoundaryEvent]),
  drain_boundary_events() -> List[BoundaryEvent] (list + clear),
  reset_event_segmenter() (per-episode reset).
  Agent wiring:
    REEAgent.sense() -- after z_harm_a_prev cache, before per-stream
      V_s (MECH-269 Phase 1) update: if hippocampal.config.use_event_segmenter
      and hippocampal.event_segmenter is not None, builds a latent_dict
      over (z_world, z_self, z_harm, z_harm_s, z_harm_a, z_beta,
      z_goal) and calls event_segmenter.step(latent_dict, pe_dict=None,
      t=self._step_count). Emitted events appended to
      hippocampal._boundary_event_queue.
    REEAgent.reset() -- after MECH-269 per_stream_vs reset:
      if use_event_segmenter: hippocampal.reset_event_segmenter().
  Backward compatible: use_event_segmenter=False by default;
  event_segmenter is None; drain_boundary_events() returns []; all
  existing experiments unaffected. 65/65 contracts + 7/7 preflight
  PASS with flag OFF (bit-identical to legacy).

  **input_stream extension -- IMPLEMENTED 2026-07-24.** Substrate
  prerequisite for ARC-070/MECH-321 (policy_decomposition_on_prediction_failure),
  per the 2026-05-10 ARC-070 lit-pull R2 bidirectional-consumer commitment
  (claims.yaml MECH-288 notes). step(), force_boundary(), and a new
  boundary_on() query entrypoint all take input_stream: str = "observation"
  ("observation" | "rollout"). Detector instances (_PEThresholdDetector /
  _BOCPDGaussianDetector) and the outer/inner/last-fire-tick counters are
  now keyed per-stream, built for both streams at construction time -- a
  rollout-stream tick cannot share so much as a sliding-window mean or
  BOCPD run-length posterior with the observation stream, and cannot
  advance observation's segment_id, by construction (different dict keys),
  not by caller convention. This is the MECH-094 enforcement mechanism.
  Data flow: (agent.py sense() OR a future ARC-070/MECH-321 rollout-side
  caller) -> EventSegmenter.step(..., input_stream=...) -> per-stream
  detectors/counters -> BoundaryEvent(input_stream=...) -> observation-
  stream events still queue to hippocampal._boundary_event_queue for
  MECH-287/MECH-269 exactly as before; rollout-stream events do NOT --
  boundary_on()'s BoundaryQueryResult (fired: bool, posterior: float,
  events: List[BoundaryEvent]) is returned directly to the caller, and
  routing it anywhere is ARC-070/MECH-321's own future wiring, not this
  substrate's.
  boundary_on(stream, latent, pe=None, t=None) matches the call shape
  MECH-321's spec names verbatim (claims.yaml MECH-321
  functional_restatement: `MECH-288.boundary_on(stream=rollout,
  latent=p.latent_signature, pe=p.pe_signature)`); it auto-increments a
  per-stream tick counter when t is omitted (a rollout evaluates a
  hypothetical trajectory with no natural relationship to agent._step_count)
  and collapses step()'s List[BoundaryEvent] into a single fired/posterior
  summary.
  Config: none added. input_stream is a per-call parameter, not a
  persistent switch -- activation is still gated by the pre-existing
  HippocampalConfig.use_event_segmenter.
  Backward compatible: default input_stream="observation" is bit-identical
  to the pre-extension single-stream call shape for every existing caller;
  the one live call site (agent.py sense()) is unchanged and untouched.
  reset() now resets BOTH streams (was a no-op behavioural change for
  every current caller, since rollout state was previously nonexistent).
  MECH-094: applies, and is this extension's whole purpose -- see above.
  Phased training: N/A, no encoder or trainable head.
  61/61 downstream-consumer contracts PASS (test_mech_288_event_segmenter,
  test_mech_269_anchor_set, test_mech_269_per_region_vs,
  test_mech_287_invalidation_trigger, test_q081_landmark_removal) plus an
  inline smoke test (stream isolation under hammering, boundary_on firing
  with posterior on an injected PE spike, force_boundary isolation, dual
  reset, unknown-stream ValueError).
  Does not itself change MECH-288's promote-to-active status or clear
  MECH-321's v3_pending (ARC-070 itself and live MECH-094-gated usage
  remain unbuilt). Validation experiment: queued via /queue-experiment
  (diagnostic purpose -- see queue entry for the EXQ id).
  See MECH-288, MECH-321, ARC-070 in claims.yaml;
  REE_assembly/docs/architecture/event_segmenter.md.
  Activation smoke (2026-04-22): default agent constructed with
  use_event_segmenter=True instantiates both scales; fresh
  current_segment_id() == "0.0"; boundary queue drains to [] on
  empty tick.
  No trainable parameters. Pure arithmetic (sliding z-score + BOCPD
  recursion). No phased training needed.
  MECH-094: hypothesis_tag is NOT checked inside the segmenter; the
  segmenter is invoked only from REEAgent.sense() (waking observation
  stream, never replay/simulation). MECH-094 gating for replay-driven
  segmentation is deferred to the Phase 3 consumer wiring.
  Contract tests: tests/contracts/test_mech_288_event_segmenter.py
    C1: default config backward-compat; event_segmenter is None when
        flag is off; drain queue empty.
    C2: pe_threshold silent on constant baseline, fires on 10x
        sustained spike.
    C3: bocpd_gaussian silent on stationary z_goal, fires on 10x
        regime shift.
    C4: hierarchical outer.inner correctness (slow -> outer+1,
        inner=0; fast -> inner+1).
    C5: force_boundary bypasses min_segment_length, posterior=1.0,
        source tagged "force:<reason>"; unknown scale -> ValueError.
    C6: BoundaryEvent payload invariants (posterior in [0,1], sources
        populated, t within window, segment_id_new != segment_id_old,
        both contain ".").
    C7: min_segment_length suppresses immediate re-fire
        (min_segment_length=5 caps fires at <=2 over 10 ticks).
  Validation experiment: deferred to Phase 3 -- Phase 2 is the
  substrate that emits boundary events. End-to-end validation
  (MECH-287 broadcast consumption + MECH-269 anchor-reset hysteresis)
  is scheduled with the Phase 3 wiring pass; V3-EXQ-476 (re-run of
  EXQ-475 with full V_s invalidation circuit on) remains the
  end-to-end validation experiment for the combined cluster.
  Design doc: REE_assembly/docs/architecture/v_s_invalidation_runtime.md
  See MECH-288, MECH-269, MECH-287 (broadcast trigger -- Phase 3
  consumer), MECH-284 (staleness accumulator -- Phase 3 consumer),
  MECH-272 (state-gated routing -- Phase 3), MECH-094 (hypothesis_tag
  -- Phase 3 when replay consumes segments).
