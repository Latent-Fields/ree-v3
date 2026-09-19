## MECH-018 Residue integration: the gradient step + the sleep WRITEBACK call site (2026-09-19)

- MECH-018: `sleep.residue_integration` -- IMPLEMENTED 2026-09-19 (parts 1 and 2 of 3).
  `ree_core/residue/field.py` (`ResidueField.integrate`),
  `ree_core/sleep/phase_manager.py` (`SleepLoopManager._run_cycle`, WRITEBACK phase),
  `ree_core/agent.py` (wiring), `ree_core/utils/config.py` (flags).
  Config: `ResidueConfig.use_offline_integration_gradient_step` (default `False`; set `True` to
  enable the gradient step) and `REEConfig.use_sleep_residue_integration` (default
  `False`; set `True` to add the sleep call site), with
  `REEConfig.sleep_residue_integration_steps` (default 10). All three are reachable
  through `REEConfig.from_dims`.
  Data flow: `_harm_history` (detached z_world clones) -> jittered `sample_points` ->
  `neural_field` prediction vs `no_grad` `rbf_field` target -> Adam over
  `neural_field.parameters()` only -> `evaluate()` = `rbf_value + neural_value * 0.1`.
  Backward compatible: both flags default off; OFF is bit-identical including RNG
  consumption and the returned metrics dict's key set.
  Phased training required: no (this is an offline approximator fit, not an encoder head).
  Validation experiment: EXP-0755 / EVB-1391, still `blocked_substrate` on part (3).
  See MECH-018, EXP-0755, GFLAG-0306, GFLAG-0356.

### The defect

`integrate()` computed `F.mse_loss(neural_field(sample_points), rbf_field(sample_points))`
in a `num_steps` loop and accumulated `.item()` -- but never called `loss.backward()`, and
the module declared no optimizer anywhere in its 1259 lines. It was a **metric loop wearing
the shape of a training loop**: the `integration_loss` it returned read as training progress
while nothing trained. A runtime probe (6 harm events, `num_steps=25`) recorded rbf weights
unchanged, `active_mask` unchanged, `active_centers` 6 -> 6, neural_field params unchanged,
`evaluate()` at the recorded harm locations unchanged, and no param carrying a `.grad`.
V3-EXQ-996's confirmed autopsy (ISEF-005, 2026-09-04) recorded the same finding but left
`recommended_substrate_queue_entry` null, so it sat unowned for two weeks.

MECH-018 lists *"the operation is geometrically inert"* as an explicit **FALSIFYING**
outcome. So every MECH-018 run made before this landed would have returned a **confident
false falsification**, attributable entirely to a missing optimizer step, and it would have
landed in `claim_evidence` as a genuine `weakens`.

### Why both flags default OFF

Not timidity. `residue_field.integrate()` has live experiment callers today
(`v3_exq_214`, `v3_exq_240`, `v3_exq_240a`, `v3_exq_246`), and `evaluate()` reads
`rbf_value + neural_value * 0.1`. An unconditional gradient step would silently change the
numerics of every one of those runs on re-run and invalidate their recorded evidence.

The two flags are separate because the parts are separable, and that creates one trap worth
naming: **call site ON with the gradient step OFF fires an inert integration** -- exactly
the false-falsification shape above, arriving for a substrate reason rather than a
scientific one. The call site therefore always emits `mech018_residue_trains` (1.0/0.0)
alongside `mech018_residue_integration_fired`, so a run that fired an inert integration is
identifiable from the manifest and must not be scored against MECH-018.

### No erasure, and why the obvious contract test is vacuous

`self.rbf_field` and `self.neural_field` are **disjoint submodules**, and `targets` is
computed under `torch.no_grad()`. The optimizer is built over `neural_field.parameters()`
only, so no gradient path to the recorded residue weights exists at all -- the "cannot be
erased" invariant holds here **by construction**, not by a clamp.

That is precisely why EXP-0755's `release_condition`, read literally, asks for a test that
cannot fail: *"rbf_field weights at the recorded `_harm_history` locations DO NOT fall below
the floor"* is an arithmetic identity under an optimizer that provably cannot reach them.
`tests/contracts/test_mech018_residue_integrate_gradient.py` therefore asserts two things
that **can** fail instead:

- **C2** asserts the isolation *explicitly* -- the optimizer's param set is exactly
  `neural_field`'s and contains no `rbf_field` parameter. A future change that widens the
  optimizer trips here rather than passing vacuously.
- **C5** asserts the no-erasure floor on the path where erasure **is** reachable:
  `discharge_domain()`'s in-place `.data` write with `MIN_FLOOR = 1e-6` and its
  sign-preserving clamp, under an aggressive repeated decay.

**If the optimizer is ever widened to cover `rbf_field.weights`, that `MIN_FLOOR` clamp MUST
be re-applied after every `step()`** -- autograd writes bypass the `.data` clamp entirely.

Two further hazards, recorded so they are not "simplified" away:

- `sample_points` is rebuilt **every iteration** from detached history. Do not hoist it out
  of the loop to save compute -- that turns `num_steps` independent graphs into one
  accumulated graph.
- MECH-094 does **not** apply to the WRITEBACK call: `integrate()` writes no residue. It
  never calls `accumulate()`, never touches `rbf_field.weights` or `active_mask`, and only
  trains the approximator toward the already-recorded field. Sleep content cannot become
  residue through this path. It is deliberately a *separate* gated step rather than an
  extension of the cross-module consolidator, whose contract is explicitly "no residue /
  memory writes".

### Measured behaviour, and a correction to the pre-build prediction

Liveness, 5 seeds, 6 harm events, `num_steps=25`: `integration_loss` falls 0.45-0.61 ->
<= 0.003; `neural_param_delta_norm` 0.0 -> ~4.9; rbf weights, `active_mask` and
`active_centers` bit-identical in **both** branches; `rbf_weight_abs_sum_delta` exactly 0.0.
End-to-end through a real `force_cycle`: `mech018_residue_integration_fired=1.0`,
`mech018_residue_trains=1.0`, param delta 5.49.

**The EXP-0755 pre-flight predicted that a successful optimisation would drive `evaluate()`
toward ~1.1x rbf -- a monotone rescaling UP. Measured, it does the opposite.** Training
against rbf targets at jittered sample points (noise std = `kernel_bandwidth` = 1.0) drives
the untrained Softplus head's roughly-constant ~0.7 pedestal to ~0, so `evaluate()` at the
recorded harm locations converges **downward onto the pure rbf core**: 0.167 -> 0.100 in a
live sleep cycle, and a drop of 0.069-0.078 in each of 5 seeds, landing on `rbf_core` to 4
decimal places every time.

Two consequences for whoever queues the MECH-018 experiment:

1. **Readout (i) PRESERVATION.** Residue at harm locations *does* fall (~37% of its
   pre-integration value at this scale). A floor registered against pre-integration
   `evaluate()` would FAIL for a reason that has nothing to do with erasure -- the
   un-erasable rbf core is bit-identical. **Register the floor against the rbf core.**
2. **Readout (ii) SPECIFICITY RATIO.** The drop is a near-*constant* pedestal removal, and
   subtracting a constant `c` from both terms of a ratio `h/n` with `h > n > c` **raises**
   that ratio arithmetically. The specificity ratio will therefore rise mechanically from
   pedestal removal alone, with no contextualisation at all -- and the naive-decay
   comparator the claim already requires does **not** control for it, since uniform
   multiplicative decay is a different operation from constant subtraction. A
   pedestal-matched control is needed, or (ii) is confounded.

### Part (3) is NOT done -- EXP-0755 stays blocked_substrate

Readout (iii) COMPRESSION asks that `active_centers` fall. `integrate()` approximates; it
never merges centres, and no module in `ree_core/` provides a merge op (`active_centers`
measured 6 -> 6 after the build). EXP-0755's `release_condition` part (3) offers an
either/or: implement a merge op, **or** narrow (iii) to the "lower `integration_loss` at
fewer parameters" disjunct the claim already offers and record that narrowing on the claim.
Both halves are out of scope for a build session -- `claims.yaml` is governance-only, and
inventing a merge semantics is a new mechanism rather than an unblock -- so this is routed
to `/governance` as **GFLAG-0356** (`contested_disposition`, recommending the narrowing
branch, which is pre-registered in MECH-018's own `what_would_answer`). EXP-0755 remains
`blocked_substrate` until that narrowing is recorded, per its own "ALL THREE land" condition.
