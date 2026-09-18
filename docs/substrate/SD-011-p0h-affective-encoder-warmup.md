## SD-011 P0h Affective-Harm-Encoder Warmup (2026-09-18)

- SD-011: harm_stream.affective_encoder_p0h_warmup -- IMPLEMENTED 2026-09-18, DEFAULT OFF.
  New shared module `experiments/_lib/zharm_a_p0_warmup.py` (`run_zharm_a_p0`), wired into
  `experiments/_lib/allon_training.py::_train_all_on_agent`.
  Config: `_train_all_on_agent(zharm_a_p0_episodes=0, zharm_a_p0_env=None,
  zharm_a_p0_dry_run=False, zharm_a_p0_config=None)` -- set `zharm_a_p0_episodes > 0` plus a
  DEDICATED warmup env to enable. Objective settings in `ZHarmAP0Config`
  (lr 5e-4, epochs 4, grad-norm clip 1.0, holdout_episode_frac 0.2).
  Data flow: warmup env rollout -> (harm_obs_a, harm_history, accumulated_harm) buffer ->
  `AffectiveHarmEncoder` -> `harm_accum_pred` -> `agent.compute_harm_accum_loss` -> Adam over
  `latent_stack.affective_harm_encoder.parameters()`.
  Backward compatible: `zharm_a_p0_episodes=0` is bit-identical prior behaviour -- no optimizer,
  no tensor, and no RNG draw (pinned by contract C2).
  Phased training required: yes. P0h runs AFTER the SD-070 z_world P0a and BEFORE the P0b e2
  contrastive warmup, because `z_harm_a` feeds E3 commit gating on every tick of P0b and P1.
  MECH-094: not applicable (waking observation stream, not replay content).
  Validation experiment: NONE queued (the commissioning chip forbade it); the owed validation is
  an opted-in arm. See `REE_assembly/evidence/planning/substrate_queue.json` ->
  `sd_zharm_a_warmup_optimizer_group`.
  See SD-011, SD-020, MECH-258, ARC-033, SD-086.

### Why it exists

The three optimizer groups `_train_all_on_agent` builds (e2, lateral-PFC bias head, OFC
devaluation head) cover NO `AffectiveHarmEncoder` parameter, so `z_harm_a` was a frozen random
projection in every all-ON experiment, with no error and no warning. Measured 2026-09-18 by
parameter-identity intersection: `latent_stack` 53 tensors, `affective_harm_encoder` 4 tensors,
0 covered. This is the exact sibling of the V3-EXQ-780 `z_world` defect recorded at
`allon_training.py:475-476`, on the same file, and it is remedied the same way.
Diagnosis: `REE_assembly/evidence/planning/sd086_zharma_readout_precondition_staged_20260918.md`
section 4b.

### The training signal was specified, not chosen

`agent.compute_harm_accum_loss` is used unchanged. Two architecture sites agree that this is the
supervision for this encoder: `LatentStackConfig.harm_history_len` / `z_harm_a_aux_loss_weight`
(SD-011 "second source"), and `ree_core/predictors/e2_harm_a.py`'s "Phased training required"
block, which names the missing stage verbatim -- "P0: AffectiveHarmEncoder warmup (z_harm_a
encoder trains on accumulated-harm / harm-surprise supervision per SD-020)". SD-020's
precision-weighted PE target is the SAME loss behind the caller's existing
`REEConfig.harm_surprise_pe_enabled`, so SD-011-EMA and SD-020-PE are one loss under a config
the caller already owns, not two rival recipes this module had to arbitrate between.

### Measured: mechanically live, informationally not (READ THIS BEFORE OPTING IN)

At 6 warmup episodes, OFF moves no parameter and draws no RNG; ON moves 6 of 6 affective-encoder
tensors (max_abs_delta 6.4e-02 -- 6 not 4 because `harm_history_len > 0` adds the
`harm_accum_head`), epoch mean loss 6.5e-03 -> 6.4e-04, held-out loss 2.11e-02 -> 9.08e-05.
Downstream liveness on a real fixed-seed 120-tick rollout: mean `||z_harm_a||` 0.874 OFF ->
9.459 ON, aux-readout MAE 0.469 -> 0.033.

**And yet `p0h_readiness_met` reads FALSE on the default target, which is the finding rather
than a build defect.** `accumulated_harm` is near-constant on this substrate (mean 0.0276, std
0.0041 over a 600-tick random rollout) and `harm_accum_head` ends in a Sigmoid starting near 0.5,
so the steep loss drop is the head learning that OFFSET. Scored against a constant-mean predictor
fitted on the train split, the trained head is WORSE on held-out episodes: lift
`1 - mse_head/mse_const` = -58.9. `p0h_holdout_vs_constant` and `p0h_readiness_met` exist so that
vacuous fit cannot be read as a trained encoder; readiness is the conjunction of "the gradient
path reached the encoder" (weight delta > 0) AND "it beat a constant" (lift > 0), because either
alone is satisfiable by a distinct failure mode.

Corroborating, and the reason the substrate_queue severity is `degrading` rather than
`corrupting`: `corr(||z_harm_a||, hazard_EMA)` is 0.987 UNTRAINED and 0.989 TRAINED, so the
NORM-reading consumers -- which are most of them (E3 urgency = `||z_harm_a|| * urgency_weight`)
-- barely move either way. The exposure is to consumers that read the 16-d VECTOR.

Upstream of all of it, `harm_obs_a` has exact rank 2 (one scalar into 25 dims, a second into the
other 25; `causal_grid_world.py:3037-3038`), so no encoder training gives `z_harm_a` more than 2
independent d.o.f. about the world. Training changes WHICH 2, not how many.

The open question -- which signal to train on before SD-086 option C runs on a "trained" encoder
-- is a user decision, raised as chip
`chip-20260918-zharm-a-training-signal-degenerate-target`.

### Contracts

`tests/contracts/test_zharm_a_p0_warmup.py`, 19 tests. C0 pins the DEFECT (the three legacy
optimizer groups stay disjoint from the encoder) so a later change covering it elsewhere cannot
silently make this record wrong; C1 params-covered-when-ON; C2 bit-identical-when-OFF; C3
weights-move, RNG neutrality, `_harm_obs_ema` restoration, holdout-vs-constant, readiness
conjunction; C4 loud refusals for every half-configured shape (the aux loss returns a ZERO loss
when `harm_history_len <= 0`, which is correct for its per-tick callers and silently fatal for a
warmup); C5/C6 wiring, ordering, and shared RNG neutrality with the z_world stage.

### Not done, deliberately

`experiments/v3_exq_728b_trained_allon_capability_point.py` defines its OWN `_train_all_on_agent`
copy and was NOT touched: it passes `include_driver_script_in_hash=True`, so editing it would
invalidate the banked arms of a live PASS run -- the documented DO-NOT-COLLAPSE reason on
`sd_zworld_warmup_optimizer_group`. Wiring P0h there is owed only if a driver on that copy needs
to opt in. No driver opts in anywhere yet, so no existing or future run is affected until one
does.
