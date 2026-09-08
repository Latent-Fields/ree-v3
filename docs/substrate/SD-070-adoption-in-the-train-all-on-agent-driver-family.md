## SD-070 ADOPTION in the _train_all_on_agent driver family -- IMPLEMENTED (2026-07-20)
- SD-070 ADOPTION in the _train_all_on_agent driver family -- IMPLEMENTED 2026-07-20.
  Module: experiments/_lib/zworld_p0_warmup.py (run_zworld_p0 + resource_prox_target).
  THE DEFECT THIS CLOSES. SD-070 shipped 2026-07-18 as a trainer, but NO DRIVER CALLED IT.
  The P0 warmup shared by x728/x734/x737/x742 builds three optimizer groups -- e2, the
  lateral-PFC bias head, the OFC devaluation head -- and NONE covers a single latent_stack
  parameter, so split_encoder.world_encoder was never stepped and z_world stayed a FROZEN
  RANDOM PROJECTION for entire campaigns, with no error and no warning. Measured on two
  INDEPENDENT drivers: V3-EXQ-737a 0 of 61 latent_stack tensors changed (world_encoder 0 of
  4) at p0_episodes=200; V3-EXQ-728 the same signature on its OWN _train_all_on_agent copy,
  3 of 3 seeds -- so the defect was per-copy, not confined to the shared path.
  THE FIX IS NOT "ENABLE PRESCRIBED P0" -- that is refuted in-corpus (SD-009 CE + SD-018 MSE
  online at batch=1 COLLAPSES z_world to PR ~1.06). Per the V3-EXQ-783 adjudication the fix
  needs (a) a gradient path reaching latent_stack and (b) a target the world channel
  determines; SD-070 supplies both, and this landing wires it in.
  Config: `_train_all_on_agent(..., zworld_p0_episodes=N, zworld_p0_env=..., 
  zworld_p0_dry_run=...)`. Default 0 = EXACTLY the prior behaviour, bit-identical: no extra
  tensor, no optimizer group, no env construction, no RNG draw. Drivers set
  ZWORLD_P0_EPISODES=60 (SD-070's validated operating point,
  exq783_zworld_granularity.OFF_P0_ENCODER_EPISODES).
  Data flow: world_obs -> [P0a buffer] -> ZWorldP0Trainer -> world_encoder +
  world_precision_logit -> P0b e2 warmup (now over a MEANINGFUL z_world) -> P1.
  ORDERING IS LOAD-BEARING: e2 regresses on z_world, so the encoder trains BEFORE the e2
  warmup -- training it after would leave e2 fitted to the random projection, i.e. the same
  defect one phase later.
  RNG NEUTRALITY IS LOAD-BEARING, NOT HYGIENE. ZWorldP0Trainer seeds its own Generator for
  shuffling but builds its auxiliary heads with nn.Linear, which draws from the GLOBAL torch
  RNG. Unguarded, merely turning P0a on would shift every subsequent draw, confounding "the
  encoder is now trained" with "the RNG stream moved". run_zworld_p0 snapshots and restores
  the global torch + numpy streams, and the rollout runs on a DEDICATED env instance so the
  training env's layout sequence is untouched.
  FINGERPRINTS UPDATED (reuse correctness): `zworld_p0_episodes` was added to x734's and
  x728's config slices and to exq742_mech457_bias_head_baseline.off_path_config_slice. An
  arm warmed with SD-070 is a DIFFERENT arm from a frozen-random-projection arm; without
  this a pre-fix banked arm would falsely cache-HIT a post-fix consumer and silently compare
  a trained-encoder treatment against an untrained control. CONSEQUENCE: the banked
  V3-EXQ-742-m bias_head_baseline mint no longer matches the 742 consumer and must be
  RE-MINTED at zworld_p0_episodes=60.
  Scope: both _train_all_on_agent definition sites (x734:332 shared by 737/742/fanout/
  baseline; x728:522 own copy), plus _lib/mech457_fanout.warmup_zworld(zworld_p0=...) and
  _lib/baselines/exq742_mech457_bias_head_baseline.run_off_cell(zworld_p0_episodes=...).
  In 742 the bias_head_baseline OFF control carries the SAME P0a setting as the AC arms --
  otherwise the ON/OFF contrast confounds the actor-critic treatment with encoder training.
  VERIFIED both definition sites, 3 P0a episodes: OFF reproduces the defect exactly
  (latent_stack 0/61, world_encoder 0/4, max_delta 0.0, guard REFUSES); ON trains
  world_encoder 4/4 (7/61 latent_stack = 4 encoder + world_precision_logit + 2 prox-head,
  z_self UNTOUCHED per SD-070's C5 contract), max_delta 6.03e-03, guard PASSES. Driver
  dry-runs: 734 guard green on all 4 rungs (was red); 737 readiness_met True (was False);
  742 clean; 728 arm_green=True seeds_failed=0 (was 3 of 3 failed) and outcome PASS.
  Phased training: P0a -> P0b -> P1 -> P2 unchanged and still mandatory. MECH-094 N/A
  (trains on live observations, writes nothing to memory in any non-waking state).
  Validation experiment: V3-EXQ-787a (see the queue entry).
  Source: REE_assembly/evidence/planning/substrate_queue.json -> sd_zworld_warmup_optimizer_group,
  failure_autopsy_V3-EXQ-737a_2026-07-20.json,
  REE_assembly/evidence/planning/zworld_bc_install_failure_V3-EXQ-780_2026-07-19.md section 6c/6d.
  See SD-070 (the recipe), _lib/zworld_encoder_guard.py (the detector this remedies),
  MECH-457, INV-088, Q-002.
