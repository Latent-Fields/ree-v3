## MECH-276: scientist-agent counterfactual-backed attribution feedstock (waking-phase mechanism feeding the MECH-275 sleep aggregator) (2026-06-23)
- MECH-276: agent.scientist_intervention -- IMPLEMENTED 2026-06-23 (substrate;
  MECH-276 stays candidate / v3_pending, and MECH-275 stays candidate /
  substrate_conditional / v3_pending -- this PROMOTES NOTHING. The substrate-readiness
  diagnostic is queued; the MECH-275 sleep-aggregation promotion run is a SEPARATE
  later session gated on the readiness PASS). The waking-phase feedstock the MECH-275
  BayesianAggregator (ree_core/sleep/bayesian_aggregator.py) was structurally blocked
  on: MECH-275 is "only coherent given a waking-phase feedstock of counterfactual-backed
  attributions produced by MECH-276 (aggregating arbitrary correlations would produce
  noise-fit)" (claims.yaml MECH-275 functional_restatement + governance_2026_06_10).
  Because MECH-276 was unbuilt, the sleep_substrate:GAP-3b promotion run V3-EXQ-702
  (2026-06-23) had to DROP MECH-275 from its tag set; this build is the owed upstream.
  Module: ree_core/attribution/scientist_attribution_buffer.py (NEW package
  ree_core/attribution/; ScientistAttributionBuffer + ScientistAttributionConfig).
  Pure-arithmetic buffer (no nn.Module, no learned parameters, no gradient flow);
  sibling to the SD-057 IncentiveTokenBank / MECH-353 BlockedAgency regulator pattern.
  COMPOSES the existing single-pass comparators (NO new comparator, ARC-106-style
  reuse-before-duplicate): SD-031 E2WorldForward (z_world causal-footprint, domain
  "place") and ARC-033 E2HarmSForward (z_harm_s / SD-003, domain "self").
  Mechanism (the counterfactual-backed attribution, the falsifiable core of the
  MECH-275 claim): on each waking tick the agent runs the comparator on
  (z_prev, observed, a_actual) with a deterministic discriminating counterfactual
  a_cf (argmax-shifted action class):
    attribution           = ||z_observed - E2(z_prev, a_actual)||   (agency residual)
    counterfactual_contrast = ||E2(z_prev, a_actual) - E2(z_prev, a_cf)||
    is_counterfactual_backed = counterfactual_contrast >= cf_margin
  A discriminating action (the agent's choice mattered, contrast >= margin) BACKS the
  attribution with a counterfactual; a near-zero-contrast action did not discriminate
  outcomes, so its attribution is mere correlation. Under only_counterfactual_backed
  (default True) correlational records are SKIPPED -> the feedstock the aggregator reads
  is structured. Setting it False is the correlational-control arm (feed everything ->
  the predicted noise-fit, no schema revision -- the MECH-275 falsifiable secondary).
  Per-(domain, region) EMA keyed on (scale, segment_id) = the MECH-284/MECH-269 anchor
  RegionKey the sleep loop routes on (read from hippocampal.anchor_set.active_anchors()
  most-recently-created anchor; falls back to a GLOBAL_REGION sentinel when no anchor
  substrate is active). evidence_snapshot() merges across domains into a region ->
  attribution map (the v1 single-snapshot integration; per-domain snapshots are a
  documented follow-on) + carries a GLOBAL_REGION sentinel holding the global mean for
  sleep-loop fallback lookups of regions not visited that waking phase.
  Agent wiring (ree_core/agent.py): self.scientist_attribution_buffer built when
  use_scientist_attribution is on (PRECONDITION, loud ValueError: requires
  use_e2_world_forward OR use_e2_harm_s_forward -- no feedstock source otherwise).
  _update_scientist_attribution(new_latent) called from sense() after
  _update_blocked_agency, using prev-latent caches (_sci_prev_z_world / _sci_prev_z_harm_s,
  one-tick lag, the MECH-353 comparator-cache pattern). The buffer PERSISTS across
  episodes (the MECH-275 aggregator runs cross-episode); agent.reset() clears only the
  prev-latent caches, not the buffer.
  Sleep-loop hook (ree_core/sleep/phase_manager.py): _build_evidence_snapshot sources
  the MECH-276 feedstock REPLACING the MECH-284 staleness scalar when the buffer is
  present (user-confirmed: the MECH-275 claim aggregates counterfactual-backed
  attribution, not staleness); _lookup_evidence honors the GLOBAL_REGION sentinel
  fallback (no-op for staleness snapshots -> legacy path bit-identical); decay_cycle()
  + mech276_* metrics merge at cycle end.
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
  use_scientist_attribution (False, master) + scientist_attribution_cf_margin (0.05) +
  scientist_attribution_only_counterfactual_backed (True, the correlational-control
  lever) + scientist_attribution_ema_alpha (0.3) + scientist_attribution_decay (1.0).
  E2_world "place" recording requires world_dim>=128 (E2WorldForward.attribution_ready,
  the SD-031 discriminative-granularity guard from failure_autopsy_zworld-integration-
  cluster_2026-06-06); below threshold the comparator returns zeros and nothing is
  buffered (fails safe, NOT a vacuous zero-gap).
  Backward compatible: use_scientist_attribution=False by default ->
  agent.scientist_attribution_buffer is None and the sleep loop sources the MECH-284
  staleness scalar exactly as before -> bit-identical. 9/9 new contracts in
  tests/contracts/test_mech276_scientist_attribution.py (C1 default-OFF no-op +
  bit-identical action stream [ON-with-comparator-but-not-attribution-ready emits no
  bias] / C2 precondition raises without a comparator / C3 records cf-backed + skips
  correlational + MECH-094 sim no-op / C4 region-merged evidence_snapshot + GLOBAL
  sentinel + lookup fallback / C5 correlational-control arm buffers everything / C6
  config validation / C7 agent activation @ dim128 records + buffer persists across
  reset / C8 sleep-loop source-select + sentinel fallback + legacy staleness
  bit-identical / C9 decay_cycle) + 8/8 preflight + 179 sleep/config/boot/aggregator
  contracts PASS unchanged. Activation smoke 2026-06-23 (e2_world @ world_dim=128,
  25 steps): 24 counterfactual-backed attributions recorded (mean cf_contrast 0.35 >
  margin 0.05), buffer feeds the evidence snapshot; default OFF buffer None.
  Phased training: N/A at the substrate level (the buffer is pure arithmetic over the
  already-built comparators; no new learned parameters). BUT the comparator
  discrimination DEPENDS ON A TRAINED, action-conditional forward model (SD-056) + a
  trained encoder -- the readiness diagnostic trains them in P0 (the MECH-353 / SD-031
  lesson: an untrained world_forward floors the comparator to a vacuous zero). A
  trained-substrate failure to discriminate counterfactual-backed-vs-correlational is a
  substrate-ceiling finding, NOT a falsification of MECH-276.
  MECH-094: record() is a no-op under simulation_mode (a replay/DMN tick must not write
  attribution feedstock -- the attribution must come from real consequential action);
  doubly enforced (the comparators' comparator_residual/forward reads the agent feeds
  are themselves MECH-094-gated upstream). Evidence-staleness (Step 8.5): NOT triggered
  -- no-op-default flag; every existing experiment uses the default (buffer off), so no
  dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-276 stays candidate / v3_pending; MECH-275 stays
  candidate / substrate_conditional / v3_pending (this build lands its upstream feedstock
  but the promotion run + readiness PASS are still owed). claims.yaml carries only
  implementation_notes (no flag/confidence/status change).
  Validation experiment: V3-EXQ-703 substrate-readiness diagnostic (claim_ids=[];
  counterfactual-backed feedstock vs correlational-control arm
  [only_counterfactual_backed True vs False], posterior-movement / discrimination at
  world_dim=128 with a trained encoder + SD-056-trained world_forward in P0). PASS is the
  gate that unblocks the SEPARATE MECH-275 sleep-aggregation promotion run (a sibling of
  v3_exq_702_gap3b_sleep_cluster_promotion.py extended with a MECH-275 cross-episode
  posterior-aggregation discriminative metric over the now-real feedstock) + re-adds
  MECH-275 to sleep_substrate:GAP-3b unblocks_claims.
  Design doc: REE_assembly/docs/architecture/scientist_agent_developmental_ordering.md
  (MECH-276 section, V3 substrate status IMPLEMENTED). Plan node:
  REE_assembly/evidence/planning/sleep_substrate_plan.md (sleep_substrate:GAP-3b).
  See MECH-276 (this claim), MECH-275 (the consumer; sleep-phase Bayesian aggregation
  this feeds -- stays substrate_conditional pending the readiness PASS + promotion run),
  MECH-273 (self-model aggregation specialisation; its offline_gradient_pass is the
  z_harm_s sibling), SD-031 E2WorldForward (place comparator) / ARC-033 E2HarmSForward
  (self comparator) / MECH-256 (the general single-pass comparator mechanism these
  instantiate), MECH-269 (probe channel hypothesis generator; anchor RegionKey source),
  MECH-272 (state-gated routing), MECH-284/MECH-285 (staleness + replay sampler; the
  evidence/routing this composes with), SD-056 (action-conditional world_forward; the
  comparator-discrimination prerequisite trained in P0), MECH-277/MECH-278 (stage-1/2
  specialisations of MECH-276; V4-leaning developmental tests), SD-003 (stage-1 closure;
  superseded by MECH-256/SD-029), V3-EXQ-702 (the GAP-3b run that DROPPED MECH-275
  pending this build), V3-EXQ-703 (validation), MECH-094 (simulation gate).
