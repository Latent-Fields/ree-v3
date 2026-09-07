## MECH-269b Symmetric V_s Gating on E1/E2 Cortical Rollouts (2026-04-26)
- MECH-269b: cortical_world_model.regional_verisimilitude_rollout_gating -- IMPLEMENTED 2026-04-26.
  Module: ree_core/regulators/vs_rollout_gate.py (VsRolloutGate, VsRolloutGateConfig).
  Read-side consumer of MECH-269 Phase 1 hippocampal.per_stream_vs at the cortical
  forward-prediction call sites. Two integration sites in agent.py:
    (a) _e1_tick: gate latent_state for E1 side BEFORE total_state cat, BEFORE
        e1(...) call AND BEFORE extract_cue_context(). Held streams substitute
        snapshot for current value into z_self / z_world (and z_goal via
        gate_stream when e1_goal_conditioned).
    (b) select_action E2_harm_a forward block: gate _harm_a_prev for E2 side
        BEFORE the per-tick e2_harm_a forward call. Held substitution prevents
        E2_harm_a from rolling forward off a stale-but-confident-looking
        affective stream.
  Snapshot semantics: refresh per-stream snapshot to latent[s].detach().clone()
  in agent.sense() (after update_per_stream_vs / update_per_region_vs) when
  V_s[s] >= vs_gate_snapshot_refresh_threshold (default 0.5). Hold (substitute
  snapshot) at the rollout call sites when V_s[s] < per-side threshold (default
  0.4 on both sides). 0.4-0.5 dead-band gives lightweight Schmitt-trigger
  hysteresis without a streak counter.
  Config: HippocampalConfig.use_vs_rollout_gating (master, default False);
  vs_gate_snapshot_refresh_threshold (0.5), vs_gate_e1_threshold (0.4),
  vs_gate_e2_threshold (0.4), vs_gate_streams (("z_world","z_self","z_harm_s",
  "z_harm_a","z_goal","z_beta")), vs_gate_unknown_stream_passes (True). All
  wired through REEConfig.from_dims. Per-stream override dicts
  (e1_threshold_per_stream / e2_threshold_per_stream) live on
  VsRolloutGateConfig and are not surfaced via from_dims (set on the gate
  config directly when needed for asymmetric per-stream tuning).
  Precondition: agent.__init__ raises ValueError if use_vs_rollout_gating=True
  but use_per_stream_vs=False (the gate has no V_s to read).
  Diagnostics on VsRolloutGate: per-stream held counts (e1, e2), per-stream
  refresh counts, snapshot store, last-tick held flags. Surfaced via
  get_diagnostics() for inclusion in experiment manifests; the V3-EXQ-490
  acceptance criteria read these counters directly (C1).
  Backward compatible: use_vs_rollout_gating=False by default; agent.vs_rollout_gate
  is None and every integration site is no-op. With flag ON but V_s seeded at
  1.0 the gate fires zero times -- bit-identical to flag-OFF in the well-aligned
  regime. Substrate-validation smoke 2026-04-26: 7/7 preflight + 143/143 contracts
  PASS with flag OFF; with flag ON and V_s seeded at 1.0, 5-tick run produced
  zero held substitutions and 5 snapshots. Forced low V_s (per_stream_vs[s]=0.1)
  correctly triggered held substitution on the E1 side.
  No trainable parameters. Pure dataclass-replace + scalar arithmetic. No phased
  training needed.
  Biological basis (lit-pull SYNTHESIS, evidence/literature/
  targeted_review_mech269b_vs_rollout_gating/): Bastos 2012 + Feldman & Friston
  2010 + Kanai 2015 (cortex-side per-stream precision-weighted PE gating);
  Ernst & Banks 2002 (psychophysical foundation for per-stream reliability-
  weighted integration); Adams 2013 + Lawson 2014 (aberrant-precision wired-
  but-inert clinical phenotype). Symmetric application of one V_s vector to
  both proposer and cortical forward predictors is genuinely novel
  architectural ground; no paper in the anchor list demonstrates the symmetric
  claim biologically (see evidence_quality_note in claims.yaml).
  MECH-094: handled by call-site scoping. Gate invoked only from waking paths
  (sense, _e1_tick, select_action). No hypothesis_tag check inside the gate
  primitive; same pattern as MECH-269 Phase 1 / Phase 2 ii / 2 iii, MECH-288,
  MECH-287, MECH-284.
  Substrate validation: V3-EXQ-601 PASS 2026-05-21 (staleness lookup at default
  0.4/0.5 thresholds; supersedes smoke-only V3-EXQ-490b C1 path). Q-040
  behavioural factorial (V3-EXQ-490/490c cohort) remains deferred to StepHarness
  + MECH-307 substrate; C2/C3 not proven by 601.
  Design doc: REE_assembly/docs/architecture/mech_269b_vs_rollout_gating.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_mech269b_vs_rollout_gating/
  See MECH-269b, MECH-269 (parent V_s primitive), MECH-284 (online staleness arm),
  MECH-098 (reafference cancellation, one V_s signal source), Q-040 (factorial),
  MECH-295 (complementary candidate cause), SD-032b (dACC adaptive control,
  downstream consumer), SD-037 (broadcast override, fires correctly already),
  ARC-033 (E2_harm_s forward, future gate consumer), MECH-258 (E2_harm_a
  forward, current gate consumer), SD-016 (cue_action_proj reads gated z_world).
