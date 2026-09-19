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

---

## Option B re-measure: the SD-020 PE target does NOT clear either (2026-09-18)

User decision 2026-09-18T22:54Z, on decision chip
`chip-20260918-zharm-a-training-signal-degenerate-target`: enable `harm_surprise_pe_enabled` and
re-measure readiness against the SD-011 baseline. No queue entry, no new recipe, env untouched.
No new config flag was needed either -- the target is already selected by the existing
default-off `REEConfig.harm_surprise_pe_enabled`.

One instrument change was needed and is the reason this is a measurement rather than a
tautology: the first cut returned `lift: None` whenever `harm_surprise_pe_enabled` was set,
which made `p0h_readiness_met` False **by construction** on exactly the path under test.
`recover_effective_target` now recovers the scalar `compute_harm_accum_loss` is actually
regressing, from the agent's own loss (force `harm_accum_pred` to zero: the loss is
`weight * target**2`, so `target = sqrt(loss/weight)`; exact because both targets are
non-negative by construction). Both targets are therefore scored on the same footing -- same
estimator, same split, same constant-mean baseline -- differing only in what the head was asked
to predict. Pinned by contracts C3h/C3i/C3j.

### Result: 3 seeds, 12 warmup episodes x 25 steps

| target | tensors moved | max_abs_delta | holdout target mean / std | head MSE | const MSE | lift per seed | readiness |
|---|---|---|---|---|---|---|---|
| SD-011 `accumulated_harm` EMA | 6 / 6 | 6.7e-02 .. 8.7e-02 | 0.0279 .. 0.0343 / 2.4e-04 .. 4.2e-04 | 7.8e-04 .. 1.2e-03 | 2.9e-07 .. 2.3e-05 | -2896.8, -377.2, -51.2 | False, False, False |
| SD-020 precision-weighted PE | 6 / 6 | 1.04e-01 .. 1.22e-01 | ~2e-06 / ~1e-06 | 3.6e-11 .. 6.7e-11 | 3.2e-11 .. 5.3e-11 | -0.278, +0.164, -0.107 | False, True, False |

**Verdict: the PE target does not clear the constant-mean baseline.** Mean lift -0.074, and the
per-seed values STRADDLE zero on a target whose absolute scale is ~1e-6 -- so the single
`readiness_met=True` at seed 1 is numerical noise, not a signal, and must not be read as a
one-in-three pass. It is a large improvement on SD-011's lift (-0.07 vs -1108 mean), but
"catastrophically worse than a constant" becoming "indistinguishable from a constant" is not
clearing it.

### Why each target fails -- two DIFFERENT, individually fixable reasons

Decomposition over 1200 ticks / 65 episodes (seed 0), measuring the two factors separately:

```
env harm_exposure (raw)          : mean +0.023856  std 0.016942  frac>0 0.9275
accumulated_harm (SD-011 target) : mean  0.028167  std 0.002394  -> CV 0.085
|accum - ema|  (PE, pre-scale)   : mean  0.00034820 std 0.00129691 -> CV 3.72
e3.current_precision             : 1.999996      (= 1/(running_variance 0.5 + 1e-6))
precision_norm                   : 0.00399999    (= min(current_precision/500, 3.0))
PE * precision_norm (SD-020 tgt) : mean 1.3928e-06 std 5.1876e-06
```

1. **SD-011 is structurally flat.** `accumulated_harm` is a *cumulative episode mean* of a
   clipped-at-zero exposure scalar (`causal_grid_world.py:4170`), so it converges and flattens by
   construction: CV 0.085. A regressor has almost nothing to fit beyond the offset. This is
   independent of precision and would not be fixed by any scaling.

2. **SD-020's dispersion is FINE; its SCALE is crushed.** The PE's coefficient of variation is
   3.72 -- **44x** the SD-011 target's -- so the signal is there. What removes it is
   `precision_norm = min(current_precision/500, 3.0)`, which at P0h equals **0.004**, because
   `current_precision` is sitting at its INIT value of 2.0. The `/500` divisor presupposes
   precision on the order of hundreds; `e3_selector.py`'s own comments reference "precision space
   (~100)" and "current_precision ~95", which are TRAINED-agent values. That is a ~250x
   attenuation at warmup time.

   **This is a phase-ordering contradiction in the architecture, not a tuning miss.** SD-020's
   target is precision-coupled by design (ARC-016), but the P0 warmup that `e2_harm_a.py`'s
   "Phased training required" block says must train this encoder runs BEFORE the agent has any
   precision to couple to -- so at P0 the coupling necessarily multiplies by its floor.

### The methodological point worth keeping

**A scale-crushed target is invisible to a weight-delta check, because Adam is scale-invariant.**
The PE arm moved the encoder *more* than the SD-011 arm (max_abs_delta 1.0-1.2e-01 vs
6.7-8.7e-02) while regressing a target of magnitude ~1e-06: Adam divides by `sqrt(v)`, so a
gradient 250x too small still produces full-size steps -- it just takes them toward numerical
noise. Every "did the encoder train?" check built on weight movement alone would have passed
this arm. The constant-mean lift is what does not.

Raised for decision as `chip-20260918-sd011-sd020-harm-target-respecification`.

---

## Option H two-arm diagnostic: arm E clears, arm F does not (2026-09-19)

User decision 2026-09-19T00:49Z, on `chip-20260918-sd011-sd020-harm-target-respecification`:
run E and F as ONE short two-arm diagnostic and let the measurement choose. Both arms landed as
default-off `ZHarmAP0Config` levers (contracts C3k-C3n); no queue entry, no env change.

- **Arm E** -- `p0_precision_norm`: pin SD-020's ARC-016 factor `min(current_precision/500, 3.0)`
  for the duration of the P0h stage only (solve for E3's running variance, restore on exit).
- **Arm F** -- `target_source="harm_exposure"`: regress the PER-TICK scalar the env already emits
  at `harm_obs[-1]` instead of SD-011's cumulative episode mean.

### Four arms x 3 seeds, same env / seeds / episodes / optimiser

| arm | target | precision_norm | holdout target mean | holdout target CV | lift per seed | mean lift | readiness |
|---|---|---|---|---|---|---|---|
| B1 SD-011 baseline | `accumulated_harm` | 0.004 | 2.9e-02 .. 3.4e-02 | 0.008 .. 0.013 | -2896.8, -377.2, -51.2 | -1108.4 | 0/3 |
| B2 SD-020 baseline | PE | 0.004 | 5.7e-07 .. 2.3e-06 | 0.505 .. 0.928 | -0.278, +0.164, -0.107 | -0.074 | 1/3 |
| **E** SD-020 decoupled | PE | **1.000** | 1.4e-04 .. 5.7e-04 | 0.505 .. 0.928 | **+0.990, +0.973, +0.777** | **+0.913** | **3/3** |
| F SD-011 re-specified | `harm_exposure` | 0.004 | 2.2e-02 .. 2.8e-02 | 0.558 .. 0.850 | -3.121, -1.559, -1.364 | -2.015 | 0/3 |

B1 and B2 reproduce the 2026-09-18 numbers exactly, which is what licenses reading E and F
against them.

**ARM E CLEARS the constant-mean baseline on every seed. ARM F DOES NOT.**

### Why E is a single-cause result, not a lucky knob

E and B2 have **identical holdout target CVs** (0.505 / 0.928 / 0.640) -- as they must, because
pinning `precision_norm` is a pure rescale and a rescale cannot change a coefficient of
variation. The information content of the target is therefore provably unchanged between the two
arms. The only thing that differs is absolute scale (target mean 5.7e-07 -> 1.4e-04, 250x), and
the lift moves from -0.074 to +0.913. So the SD-020 PE signal was learnable all along, and what
destroyed it was the ARC-016 scaling at P0 -- exactly the phase-ordering hypothesis, now
confirmed rather than inferred.

A sweep over the pin makes the threshold concrete (3 seeds each; CV constant at 0.691 throughout,
so every difference below is scale):

| pin | 0.004 | 0.020 | 0.040 | 0.100 | 0.400 | 1.000 | 3.000 |
|---|---|---|---|---|---|---|---|
| target mean | 1.29e-06 | 6.43e-06 | 1.29e-05 | 3.22e-05 | 1.29e-04 | 3.22e-04 | 9.65e-04 |
| mean lift | -0.074 | +0.958 | +0.950 | +0.930 | +0.916 | +0.913 | +0.912 |
| readiness | 1/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |

There is a **cliff between 0.004 and 0.02** -- a single 5x rescale -- and a flat plateau
(~+0.92) for two further orders of magnitude above it. That is the signature of a numerical
FLOOR, not of a tuned optimum: the failing target has magnitude ~1e-06, so its gradients are
comparable to Adam's `eps` (1e-8) and the update direction is dominated by epsilon rather than
by the loss. It also means the repair does not need `precision_norm = 1.0` specifically; **any
pin at or above ~0.02 clears**, and the plateau's mild downward slope says larger is not better.

### Arm F: better target, still no lift -- and the reason is informative

Arm F's target has far more relative dispersion than SD-011's as specified (CV 0.558-0.850 vs
0.008-0.013), and it improves the lift enormously (-2.0 vs -1108). It still does not beat its own
mean. The likely reason is structural rather than a tuning miss: the encoder's input `harm_obs_a`
is an **EMA** of the harm field, so it carries the smoothed signal; asking it to predict the
INSTANTANEOUS scalar is asking a low-pass-filtered input to recover what the filter removed.

This is the good outcome for SD-011's claim text: because F did not carry, the claim-level cost
that option F would have incurred -- changing what `z_harm_a` MEANS, collapsing the
accumulated-vs-instantaneous distinction from `z_harm_s` and re-opening the EXQ-241 D3 redundancy
the second source was built to fix -- **is not incurred**. SD-011's target quantity does not need
re-specifying.

### What this implies for the claims (NOT applied here -- raised for decision)

SD-020 as written couples the affective-PE target to precision at ALL times. The measurement says
that coupling is correct as a RUNTIME property and wrong as a TRAINING-TIME one, because P0
necessarily runs before the agent has any precision to couple to. Amending SD-020 to say so is a
claim-level change and was NOT made here; raised as
`chip-20260919-sd020-precision-coupling-runtime-property`.
