## SD-RESIDUE-VALENCE-BOUND: residue.field.RBFLayer.update_valence.accumulator_bound -- IMPLEMENTED (2026-08-11)
- SD-RESIDUE-VALENCE-BOUND: residue.field.RBFLayer.update_valence.accumulator_bound --
  IMPLEMENTED 2026-08-11. Full design: `REE_assembly/docs/architecture/sd_residue_valence_bound.md`.
  Built from `failure_autopsy_V3-EXQ-906a_894b_2026-08-09.md` (original finding, `excite` only)
  + `failure_autopsy_906b-906c-911-cluster_2026-08-10.md` (scope widened to all 6 valence
  components), routed via `substrate_queue.json` (`sd_id: SD-RESIDUE-VALENCE-BOUND`,
  `severity: degrading`, `node_class: complicated (buildable)`).
  Fixes `RBFLayer.update_valence()` (`ree_core/residue/field.py`) -- the single write path
  behind all 6 SD-014/ARC-036 valence components (wanting/liking/harm_discriminative/surprise/
  positive_surprise/negative_surprise, MECH-307 Gap-1 Option-b) -- which was a raw unclamped
  `self.valence_vecs[center_idx, component] += value` with no decay. Only 32 RBF centers exist
  by default, so a long-lived agent revisiting the same regions drove the same centers'
  components unboundedly: `z_world_norm` ~150-320 vs ~0.5-0.7 at smoke scale (906a); `liking`
  mean=19.88 vs a 0.39 smoke ceiling, `dread` rising 40-110x across episodes (906b/906c cluster).
  Fix: a leaky-integrator decay + hard clamp (`v <- v*(1-decay_rate) + value`, then
  `clamp(v, -clamp_abs, clamp_abs)`), gated behind a NEW master switch rather than an
  unconditional default change -- contrast SD-ORIENTING-DECISION-SCALE immediately above, whose
  buggy path lived behind an off-by-default switch only one experiment ever exercised.
  `valence_enabled` defaults `True` and this write path is exercised by dozens of historical
  experiments with real `claims.yaml` evidence, so an unconditional fix would be a mechanism
  change requiring the `/implement-substrate` Step-8.5 evidence-staleness audit across all of
  them -- disproportionate to a plumbing bound-fix. `ResidueField.update_valence()` and
  `ResidueField.update_wanting_sensitized()` (the two callers, covering all 13 `agent.py` call
  sites plus the SD-014 incentive-sensitized WANTING path) resolve both parameters from config
  through one shared helper, `_valence_bound_params()`, so the fix applies from a single point.
  Config: `ResidueConfig.valence_bounding_enabled` (default `False` -- master switch; `False` ->
  `update_valence()` is the exact pre-fix `+=`, bit-identical) + `valence_decay_rate` (default
  `0.02`, mirrors this class's existing `integration_rate` timescale convention) +
  `valence_clamp_abs` (default `5.0`, headroom above the smoke-scale ceiling while staying far
  below the observed contamination). All three NEW, on `ResidueConfig` (not `REEConfig`
  top-level) -- unlike `REEConfig`-level knobs, `ResidueConfig` fields are not threaded through
  `REEConfig.from_dims()` as named kwargs (`valence_enabled`/`safety_terrain_enabled` are not
  either); set by direct post-construction assignment
  (`cfg.residue.valence_bounding_enabled = True`), matching the existing sibling-flag
  convention. No `from_dims()` signature change was needed or made.
  Backward compatible: confirmed by direct unit test (200 successive `+1.0` writes to the same
  component/center produce exactly `200.0` under default config; bounded at `valence_clamp_abs`
  under `valence_bounding_enabled=True`; `update_wanting_sensitized` tested identically). A full
  `--dry-run` of `v3_exq_887_sd014_node_valence_representational_functional.py` (an existing
  experiment exercising this exact write path at default config) ran end-to-end with no error.
  Phased training / MECH-094: not applicable -- `valence_vecs` is a `register_buffer`, not an
  `nn.Parameter`; every write already executes under `torch.no_grad()`. No gradient flow, no
  trainable parameters. The existing `hypothesis_tag` gate sits upstream of the new decay/clamp
  logic and is untouched (confirmed by a dedicated unit test: with `hypothesis_tag=True`
  throughout, the tracked value never leaves its 0.0 init, under BOTH bounding conditions).
  Validation experiment: `V3-EXQ-918` queued (`v3_exq_918_sd_residue_valence_bound_validation.py`,
  `experiment_purpose=diagnostic`, `claim_ids=[]`) -- unit-level ON/OFF ablation directly against
  `ResidueField`/`RBFLayer` (no `CausalGridWorld`/`REEAgent` needed, matching the `V3-EXQ-520`
  Part-1 precedent), reproducing sustained same-center writes across
  `positive_surprise`/`negative_surprise` (MECH-307 split-surprise) and `wanting` (via
  `update_wanting_sensitized`). Smoke PASS: OFF arm reproduces unbounded growth (200.0/395.5
  after 200 writes), ON arm plateaus exactly at `clamp_abs=5.0`.
  Descoped: the same failure-autopsy cluster also flagged `update_benefit_salience()`/
  `update_schema_wanting()` (the `residue_wanting`/`VALENCE_WANTING` writer methods) as "never
  called from the 906-family agent step loop." Traced directly against source: `REEAgent` has no
  internal step loop at all -- every experiment driver calls these explicitly itself (confirmed
  via `experiments/_harness.py`'s `StepHarness` and ~20 other scripts that already call them
  correctly). The 906-family driver simply omits these calls -- an experiment-script gap, not a
  `ree_core/` substrate gap, so left as a reported follow-on rather than folded into this change
  (this file's mandatory skill-path rule restricts `experiments/` edits to `/queue-experiment` or
  `/diagnose-errors`).
  No `claims.yaml` claim gates on this SD (`unblocks_claims: []` in `substrate_queue.json` --
  all three motivating runs, 906a/906c/the cluster, are `claim_ids: []` diagnostics), so no
  `v3_pending` flip was made.
  Registry-drift side-fix (same session, user-directed): `tests/test_flag_inertness.py`'s
  `test_flag_registry_is_current` previously scanned only `REEConfig`'s own top-level fields,
  silently missing every `use_*`/`*_enabled` flag on one of `REEConfig`'s 13 nested config
  classes (85 flags total, including this SD's own new `valence_bounding_enabled`, which would
  itself have landed uncategorized under the old scan -- the gap that surfaced this fix).
  Widened to scan every dataclass in `ree_core/utils/config.py` (`_current_nested_flags()`);
  `valence_bounding_enabled` is individually probed
  (`test_sd_residue_valence_bound_bounds_the_accumulator`), the other 84 pre-existing nested
  flags are bulk-seeded into `KNOWN_UNPROBED_NESTED` with a generic placeholder reason (a real
  per-flag audit is out of scope for this session -- tracked as a follow-on chip). Also caught
  and categorized one genuinely pre-existing top-level gap the widening exposed as a side
  effect: `use_defensive_orienting` (SD-ORIENTING-DECISION-SCALE / SD-099, landed
  2026-08-08/2026-08-10, unrelated to this session) was already missing from the registry before
  this widening.
  See MECH-307 (the split-surprise mechanism whose excite/dread writes are the most-affected
  components), SD-014 / ARC-036 (the valence-vector mechanism this accumulator belongs to).
