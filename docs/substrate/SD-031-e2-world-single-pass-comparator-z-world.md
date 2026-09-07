## SD-031: E2_world Single-Pass Comparator (z_world agency) (2026-06-06)
- SD-031: self_attribution.comparator_z_world -- IMPLEMENTED 2026-06-06
  (substrate; v3_pending until the discriminative/attribution arm PASSes a
  gated validation experiment). The z_world instantiation of the MECH-256
  single-pass forward-model comparator (sibling to ARC-033 on z_harm_s; the
  z_self instantiation SD-030 stays V4). Rescoped v4 -> v3 the same day:
  SD-031 is a named dependency of the V3 z_world discriminative-granularity
  retest (failure_autopsy_zworld-integration-cluster_2026-06-06, V3-EXQ-177's
  attribution arm), so it is V3-scoped by definition (implementation_phase is
  a prediction, not a permission gate).
  Module: ree_core/predictors/e2_world.py (E2WorldForward + E2WorldConfig +
  MIN_DISCRIMINATIVE_WORLD_DIM=128). Reuses ResidualHarmForward (the generic
  residual-delta module in stack.py -- not harm-specific). Single forward pass,
  NO counterfactual:
    predicted_z_world = E2WorldForward(z_world_prev, a_actual)
    residual_world    = z_world_observed - predicted_z_world   # agency signal
  forward() is the unrestricted evaluator/rollout read (MECH-257);
  comparator_residual() is the MECH-094-gated retrospective comparator read
  (returns zeros under simulation_mode).
  Config: LatentStackConfig.use_e2_world_forward (bool, default False;
  bit-identical OFF) wired through REEConfig.from_dims. z_world_dim is read
  from config.latent.world_dim at agent construction -- NEVER a literal.
  Agent wiring (agent.py): self.e2_world built when use_e2_world_forward is on,
  mirroring the e2_harm_s block (agent-level module, no new LatentState field).
  CARRY-FORWARD GUARD (the load-bearing piece): E2WorldForward.__init__
  HARD-ASSERTS world_dim >= 128 (REEConfig.large). The 2026-06-06 cluster
  autopsy established z_world at world_dim=32 is a competent BULK predictor
  (world_forward_r2 0.72-0.94) but lacks discriminative granularity -> a dim=32
  comparator emits a vacuous zero attribution gap. The agent constructor
  therefore RAISES if use_e2_world_forward=True at the default world_dim (32).
  An explicit allow_subthreshold_dim=True escape hatch (default False) permits
  bulk-only/ablation construction below threshold but reports
  attribution_ready=False and returns a zeroed sentinel residual (never a
  misleadingly-zero gap).
  Phased training (validation experiment, not the substrate): P0 train the
  z_world encoder (SD-009 event-contrastive + SD-018 resource proximity) so
  z_world is discriminative BEFORE the forward model -- a random encoder yields
  a near-position-invariant z_world and a trivially-identity world-forward (the
  MECH-353 / V3-EXQ-642 vacuous-comparator lesson; dimensionality is necessary,
  not sufficient). P1 train E2WorldForward on FROZEN z_world (z_world_next.detach()).
  P2 attribution eval at world_dim >= 128 with ARC-065 diversity active.
  SD-013 analogue: compute_interventional_loss (margin loss pushing predictions
  apart across actions) for the strong ambient world-state correlations.
  Backward compatible: use_e2_world_forward=False by default -> agent.e2_world
  is None; bit-identical OFF (default == explicit-False action stream verified).
  845/845 contracts + 7/7 preflight PASS; 9/9 new contracts in
  tests/contracts/test_e2_world_forward.py (C1 bit-identical OFF / C2 shape +
  action-response / C3 dim-guard raises at 32 + None + subthreshold-opt-in /
  C4 delta-not-identity on autocorrelated z / C5 MECH-094 sim gate / C6
  agent-path guard raises at default dim, constructs at 128).
  Activation smoke 2026-06-06 (world_dim=128): world_forward_r2 0.969;
  single-pass comparator self-caused residual ~2.0 vs externally-caused ~22.6
  (correct attribution gap).
  MECH-094: forward() unrestricted (evaluator mode); comparator_residual()
  returns zeros under simulation_mode (replay/DMN cannot generate a spurious
  agency signal). Phased training: yes for the validation experiment (P0/P1/P2);
  the substrate landing itself adds no trained parameters beyond the forward MLP.
  Validation experiment: NOT queued (out of scope this pass). GATED -- do not
  run until world_dim >= 128 AND ARC-065 balanced-event diversity are both in
  place (ARC-065 currently phase_1 only); else it reproduces the dim=32 +
  monostrategy confound. Hand to /queue-experiment.
  Design doc: REE_assembly/docs/architecture/sd_031_e2_world_forward_model.md
  See SD-031, MECH-256 (parent comparator mechanism), ARC-033 / e2_harm_s
  (the z_harm_s sibling template), SD-029 (z_harm_s instantiation), SD-030
  (z_self instantiation; stays V4), SD-005 (z_world split; dep, implemented),
  SD-010 (harm-stream separation; dep), ARC-065 (behavioral diversity; the
  co-gate on validation), MECH-353 / V3-EXQ-642 (vacuous-untrained-encoder
  lesson), MECH-094 (call-site scoping).
