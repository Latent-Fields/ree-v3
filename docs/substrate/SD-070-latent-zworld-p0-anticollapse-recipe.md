## SD-070: latent.zworld_p0_anticollapse_recipe -- IMPLEMENTED (2026-07-18)
- SD-070: latent.zworld_p0_anticollapse_recipe -- IMPLEMENTED 2026-07-18.
  Module: ree_core/latent/zworld_p0.py (ZWorldP0Config + ZWorldP0Trainer, plus the
  pure functions scene_structure_targets / variance_covariance_penalty /
  balanced_class_weights / entity_presence_mask / chebyshev_offsets).
  WHAT IT REPLACES: the P0 the substrate prescribed for training the z_world encoder
  (SD-009 event-contrastive CE + SD-018 resource-proximity MSE, online at batch=1;
  named in substrate_queue.json:971 and ree_core/predictors/e2_world.py:42-54) does
  NOT produce a trained encoder -- it COLLAPSES z_world. Measured 2026-07-18 at
  world_dim=128, seed 42, 40 eval episodes: untrained participation_ratio 9.21 /
  contrast_ratio 0.1222, vs 1.06 / 0.0726 after the prescribed P0. PR ~1 = collapsed
  onto one effective dimension, so every downstream comparator built on it is vacuous
  (the MECH-353 / V3-EXQ-642 lesson).
  THREE MEASURED FAULTS. (1) The SD-009 target is unlearnable from the channel its
  loss reads: transition_type is a property of the TRANSITION (t-1 -> t) while z_world
  is a static single-frame encoding, and an MLP-128 probe on RAW world_obs with no
  encoder in the path scores AT OR BELOW CHANCE (lift -0.014 3-class, -0.060 on a
  repaired 6-class map), while the same label probes at +0.240 / +0.427 from the BODY
  delta that SD-005 routes to z_self. So this is a WIRING fault, not a labelling one --
  class rebalancing cannot recover absent information. (2) Nothing penalises collapse:
  a ~95% class-0-saturated CE plus one scalar are both served by a 1-D code. NOTE the
  collapse is NOT the multiplicative precision gate (the obvious suspect): its sigmoid
  moves only 0.4966-0.5074 and the final layer stays full-rank (sv PR ~55) -- the
  encoder's FUNCTION collapses while its weights look healthy. (3) The loop is online
  at batch=1, leaving variance/covariance statistics undefined.
  RECIPE: static scene-structure grounding targets derived from world_obs ALONE
  (hazard/resource presence + bucketed nearest-Chebyshev distance; probed 0.943-0.965
  balanced accuracy from raw world_obs, vs chance 0.5/0.5/0.333/0.333) + class-balanced
  CE + a VICReg variance/covariance penalty + an optional world_obs reconstruction head
  + mini-batching over a rollout buffer. The COVARIANCE term is the participation-ratio
  lever; the variance hinge alone cannot raise PR (correlated dims at unit std still
  occupy one effective dimension), which is why covariance_weight defaults to 50 rather
  than VICReg's published ratio (measured w_cov=0.04 -> PR 1.80, w_cov=50 -> PR 4.02).
  Data flow: world_obs -> [buffer] -> world_encoder -> *sigmoid(world_precision_logit)
  -> {grounding heads, SD-018 prox head, recon head, var/cov penalty}. Trains exactly
  world_encoder + world_precision_logit (the set V3-EXQ-783's weight-delta readiness
  check watches); topdown + alpha_world smoothing are applied at sense() time and carry
  no P0 gradient.
  BIT-IDENTICAL OFF BY CONSTRUCTION, NOT BY FLAG: SD-070 adds NO field to
  LatentStackConfig, NO head to SplitEncoder and NO method to REEAgent. It operates on
  an existing LatentStack from outside and the auxiliary heads belong to the trainer,
  so nothing runs unless an experiment explicitly constructs a ZWorldP0Trainer and
  there is NO flag that can be left in the wrong state (pinned by contracts C6).
  Contract-pinned NOT to touch the z_self path (training it would confound every
  downstream self-stream result run after a P0).
  Phased training: this is P0 only; P1 (E2WorldForward) still trains on stop-gradient
  z_world with the encoder optimiser NOT stepped, P2 measures. MECH-094 N/A (trains on
  live observations, writes nothing to memory in any non-waking state).
  VALIDATION at config defaults, world_dim=128, 3 seeds: PR 9.21->5.19 (s42),
  6.63->5.41 (s43), 8.56->4.64 (s44). Against the V3-EXQ-783 anti-collapse gate computed
  as that harness computes it: retained_fraction 5.079/8.132 = 0.625 (needs >=0.50) and
  absolute 5.079 (needs >=2.0), both PASS; contrast ratio raised 3/3 (mean 0.137->0.238,
  clearing the untrained 0.13-0.15 band); world-path tensors changed 4/4 on every trained
  arm. DISCRIMINATIVE, not vacuously un-collapsed: held-out balanced-accuracy lift +0.23
  to +0.47 on all four grounding heads across all three seeds. The trainer ALWAYS reports
  this readout, because an anti-collapse gate can otherwise be satisfied by a regulariser
  that holds PR up while learning nothing. Smoke: 31 SD-070 contracts pass; full suite
  1590 pass (0 regressions).
  SD-009 IS ROUTED AROUND, NOT ADJUDICATED. The target/channel mismatch measured here
  concerns SD-009's own validity and is recorded for a future /governance cycle in
  REE_assembly/evidence/planning/sd009_event_contrastive_channel_mismatch_2026-07-18.md.
  NO SD-009 status, confidence, or evidence weighting was changed. Note claims.yaml:7319
  records EXQ-020 PASS on the SD-009 mechanism (selectivity_margin=0.882) -- a cosine
  separation statistic, NOT the decodability statistic measured here; reconciling the two
  is the governance question, and the reconciliation offered in that artifact is a
  hypothesis, not a finding.
  Validation experiment: V3-EXQ-783 (see Step 8).
  Design doc: REE_assembly/docs/architecture/sd_070_zworld_p0_anticollapse_recipe.md
  See SD-005 (the encoder it trains), SD-018 (retained leg), SD-031 / E2WorldForward
  (the consumer requiring a trained encoder), SD-009 (routed around), MECH-353, Q-002.
