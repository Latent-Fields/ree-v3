## SD Design Decisions Validated (V3) — 2026-03-18
- SD-003: self_attribution.counterfactual_e2_pipeline — **SUPERSEDED 2026-04-18** after
  28 accumulated FAILs across the two-pass counterfactual architecture. Successor layer:
  MECH-256 (general single-pass forward-model comparator, stream-agnostic) + SD-029
  (concrete z_harm_s instantiation; event-conditioned test queued as V3-EXQ-433) + MECH-257
  (dual-function single-substrate E2: comparator vs evaluator, controller-gated). Per-stream
  successors SD-030 (z_self) and SD-031 (z_world) are V4-deferred. Architecture doc:
  `REE_assembly/docs/architecture/self_attribution_per_stream.md`. The EXQ-030b world-pipeline
  PASS (world_forward_r2=0.947, attribution_gap=0.035) is preserved as historical evidence but
  does not transfer to the z_harm_s topology.
  EXQ-030b pipeline: z_world_actual = E2.world_forward(z_world, a_actual),
  z_world_cf = E2.world_forward(z_world, a_cf), causal_sig = E3(z_world_actual) - E3(z_world_cf).
  Results: world_forward_r2=0.947, attribution_gap=0.035, correct sign structure.
  NOTE: EXQ-030b validated the counterfactual ARCHITECTURE before SD-010 wired E3 to
  take z_harm. Now that E3 operates on z_harm, the counterfactual must operate on the
  harm stream. EXQ-093/094 confirmed HarmBridge(z_world->z_harm) has bridge_r2=0
  (infeasible: z_world perp z_harm by SD-010 design). Redesigned pipeline (post SD-011):
    z_harm_s_cf = E2_harm_s(z_harm_s, a_cf)
    causal_sig = E3(z_harm_s_actual) - E3(z_harm_s_cf)
  E2_harm_s is a learnable forward model on the sensory-discriminative harm stream (ARC-033).
  DO NOT attempt HarmBridge counterfactuals -- bridge_r2=0 is architectural, not a bug.
