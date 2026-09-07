## SD-100 / ARC-032 / MECH-089: Phase-Aware ThetaBuffer Summary -- IMPLEMENTED (2026-08-10)
- SD-100: hippocampal.theta_buffer_phase_aware_summary -- IMPLEMENTED 2026-08-10. Full design:
  `REE_assembly/docs/architecture/sd_100_theta_buffer_phase_aware_summary.md`. Built from
  `failure_autopsy_V3-EXQ-228c_2026-08-10.md`'s routing: V3-EXQ-228c (the first fair,
  E3-tick-restricted, direct-DV test of ARC-032) failed both primary criteria with a REVERSED
  trend (theta ACTIVE noisier than bypassed, `noise_delta_mean=-0.0671`; no persistence benefit,
  `persist_delta_mean=4.47e-07`), diagnosed `implementation: partial` -- a biology-divergence
  gap, not a claim falsification -- converging with MECH-089's own confirmed EXQ-066 (batched E3
  error 2.28x worse than raw) and EXQ-122 (harm_auc delta=-0.135, adverse).
  `ree_core/latent/theta_buffer.py` `ThetaBuffer.summary()` (E1 `z_world` -> E3, the exact site
  the failure record implicates) previously computed a flat, unweighted mean over the buffered
  window -- permutation-invariant, so two theta cycles holding the same z_world values in a
  different order summarized identically. Dragoi & Buzsaki 2006 (theta-sequence compression) /
  Colgin et al. 2016 (theta-gamma phase nesting) attribute theta's functional payload
  specifically to within-cycle ordinal/phase structure, which a flat mean cannot represent by
  construction.
  Fix: a fixed (non-trained) forward-sweep phase-weighted sum. Each buffer position `t` (T
  entries, oldest->newest) gets phase `phase_t = pi*t/(T-1)` (forward HALF cycle only -- not a
  full `2*pi` wraparound, since Dragoi's compression sweep runs once, forward, per cycle and a
  full circle would place oldest/newest adjacent in phase, which is wrong for a finite window).
  Weight is a von-Mises-style kernel centred on the read-out phase (cycle completion / "now"):
  `w = softmax(-kappa * cos(phase_t))`. `-cos` is strictly monotonically increasing over
  `[0, pi]`, so weight strictly increases with recency -- a legitimate phase-readout kernel, not
  an arbitrary bias -- and degrades gracefully: at `kappa=0`, `softmax` returns uniform `1/T`,
  mathematically identical to the flat mean (the OFF path still runs the original `.mean(dim=0)`
  directly, so disabled behaviour stays bit-identical rather than merely equivalent up to
  softmax's floating-point rounding).
  Config: `HeartbeatConfig.use_theta_phase_weighted_summary` (default `False`, bit-identical --
  `ThetaBuffer.summary()` takes the original `.mean(dim=0)` branch) +
  `HeartbeatConfig.theta_phase_concentration` (von-Mises kappa, default `4.0`, only read when the
  switch is True). Both present at all three sites (`HeartbeatConfig` dataclass field,
  `REEConfig.from_dims()` parameter, `from_dims()` assignment to `config.heartbeat.*` -- mirrors
  the existing `breath_period`-style HeartbeatConfig exposure, not the top-level
  `REEConfig.use_X` pattern, since `theta_buffer_size` itself already lives in `HeartbeatConfig`
  rather than at top level; see `[memory] reference-reeconfig-from-dims-silent-kwargs`).
  `ThetaBuffer.__init__()` takes the two new kwargs directly (`use_phase_weighted_summary`,
  `phase_concentration`); `agent.py`'s `ThetaBuffer(...)` construction site reads them via
  `getattr(config.heartbeat, ..., False)` / `getattr(..., 4.0)`, matching
  `use_defensive_orienting`'s (SD-099) no-op-when-absent discipline.
  Scope: `summary()` only (the exact E1->E3 site the failure record implicates).
  `self_summary()` (E1 `z_self` -> E2) and `consolidation_summary()` (MECH-122 Phase-3 offline
  cx->hip packaging, already order-sensitive via its own separate linear-decay weighting) are
  untouched -- no failure record entry names them; extending the same treatment is plausible
  follow-on but out of this SD's scope.
  Not a learning module: no `nn.Module`, no trainable parameters -- the engineering problem
  ("make an order-invariant aggregator order-sensitive") is solved by folding a fixed
  phase-indexed kernel into the pooling weights (closest ML parallel: fixed sinusoidal
  positional encoding, Vaswani et al. 2017 -- adapted here as a weighting kernel rather than a
  concatenated feature, so the `[batch, world_dim]` output shape E3's config assumes is
  preserved) rather than training a head. Phased training (Step 3e) and MECH-094 (Step 3f) do
  not apply.
  Backward compatible: disabled by default. Confirmed via `experiments/v3_exq_228c_arc032_theta_bypass_readout.py
  --dry-run` (same per-seed PASS pattern / pre-existing short-run aggregate FAIL as before this
  change -- see SD-099's identical dry-run-backward-compat note above) and a direct `REEAgent`
  construction smoke test (OFF: `theta_buffer.use_phase_weighted_summary is False`; ON via
  `from_dims(use_theta_phase_weighted_summary=True, theta_phase_concentration=6.0)`: flag +
  kappa wired through correctly, `summary()` still returns `[batch, world_dim]`).
  New contract suite: `tests/contracts/test_theta_buffer_phase_aware_summary.py` (7 tests, all
  green) -- C1 default-OFF bit-identical (empty buffer, filled buffer, explicit-vs-default
  kwarg); C2 (the core fix) two windows holding the SAME z_world multiset in a DIFFERENT order
  produce the SAME flat mean but a DIFFERENT phase-weighted summary; C3 `kappa=0` matches the
  flat mean up to softmax rounding; C4 single-entry buffer (T=1) does not divide by zero; C5
  weights strictly increase with recency (the property that guarantees C2 for an arbitrary
  reordering, not just the one tested).
  Validation experiment: recommended (not queued by this session, per chip instruction) --
  `/queue-experiment` a V3-EXQ-228d re-test of ARC-032 with
  `use_theta_phase_weighted_summary=True`, reusing V3-EXQ-228c's E3-tick-restricted
  persistence/proximity-noise readout, since the biology-divergence gap 228c identified (not the
  frontal-hippocampal-synchrony hypothesis) is what this SD addresses.
  See ARC-032, MECH-089 (both claims.yaml-updated with this implementation note), MECH-116 (E1
  goal maintenance, the upstream signal theta-packages), MECH-122 (`consolidation_summary()`,
  the sibling cx->hip weighting, unmodified).
