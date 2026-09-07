## SD-MECH267-CEM-SELECTION-FIX: Mode-Content Wash-Out Fix (H2 value term + H3 persistent breadth) -- IMPLEMENTED (2026-08-14)
- SD-MECH267-CEM-SELECTION-FIX: hippocampal.cem_mode_selection -- IMPLEMENTED 2026-08-14.
  `ree_core/hippocampal/module.py`. Adds MECH-267's THIRD and FOURTH facets, both targeting the
  mode-content WASH-OUT that V3-EXQ-869/869a/923 established: mode-conditioned proposal content is
  present at `num_cem_iterations=1` but collapses to a mode-independent optimum by production
  iters=2-3, because (a) the CEM refit re-derives `ao_std` purely from the mode-BLIND elite set
  every iteration (discarding the one-time `mode_noise_scale` seed applied before the loop) and
  (b) `_score_trajectory`'s ranking criterion is mode-INDEPENDENT (ARC-007 terrain + optional
  wanting/curiosity, none mode-dependent). Neither the noise-scale nor the horizon-depth facet,
  alone or together (869a), survives this. Two INDEPENDENT, deliberately-orthogonal fixes, each a
  no-op at its default, each gated by the SAME `mode_conditioning_enabled` switch. Locus was
  deliberately left unscoped (H2 vs H3) on 2026-08-12; user chose to build BOTH and re-measure.
  H2 -- `HippocampalConfig.mode_value_weight` (Dict[str, List[float]], default {}): a
  mode-dependent term in the CEM elite-selection value function. `_score_trajectory` gained an
  optional `operating_mode: Optional[Dict[str, float]] = None` param; when mode conditioning is
  on, operating_mode is supplied, and the map is non-empty, the blended weight
  `w = sum_m operating_mode[m] * mode_value_weight[m]` is dotted with the trajectory's mean
  z_world and SUBTRACTED from the terrain score (lower = better; same sign as wanting_weight).
  Keyed on mean z_world, NOT the residue valence components, on purpose: the V3-EXQ-869/923
  wash-out regime builds a FRESH ResidueField whose valence head is identically zero (verified),
  so a valence-keyed term would be inert in the exact test; z_world is always non-trivial and is
  the same space the terrain score already ranks over. Weight vectors shorter/longer than
  world_dim are padded/truncated. Active INDEPENDENTLY of wanting_weight/curiosity_weight (both
  0.0 in the wash-out experiments). Keeps the RANKING mode-differentiated on every refit
  iteration.
  H3 -- `HippocampalConfig.mode_partitioned_cem` (bool, default False): re-applies the
  mode-conditioned noise scale to the freshly-refit `ao_std` once per CEM iteration (in BOTH the
  legacy argsort-refit and the SD-055 differentiable-refit branches), so each mode-conditioned
  proposal keeps its own persistent BREADTH instead of converging to the mode-blind elite spread.
  Because `ao_std` is recomputed from scratch from the elites each iteration, the re-scale is
  applied once per iteration and does NOT compound. Gated on `mode_scale is not None` (mode
  conditioning enabled + operating_mode supplied). For a single mode-conditioned proposer call
  (the C1 measurement setup) this is equivalent to a per-mode candidate pool whose elites never
  mix across modes.
  Backward compatible at landing (2026-08-14): `_score_trajectory`'s new `operating_mode` param
  defaults None (its five other call sites pass nothing -> bit-identical); `mode_value_weight`
  empty and `mode_partitioned_cem` False were the defaults, so both new branches were skipped.
  Neither facet is wired through `REEConfig.from_dims` -- like `mode_conditioning_enabled` /
  `mode_noise_scale` / `mode_horizon_scale`, they are set directly on `HippocampalConfig` by the
  experiment (mirroring V3-EXQ-462's `_make_hippocampal`), so no from_dims silent-kwargs hazard.
  Diagnostics: `_last_mode_value_weight_active` (bool) and `_last_mode_partitioned_cem` (bool),
  both False when disabled/no-op, mirroring the `_last_mode_noise_scale` convention.
  Smoke (untrained field, single seed): defaults inert; H2 shifts a trajectory score
  0.355 -> -0.268; H3 lifts the tight(0.3)-vs-broad(1.3) mode raw_std gap from 2.1e-06 (OFF, the
  washed-out control -- matches the autopsy's ~6e-06) to 0.0172 (ON, clearing the 0.01
  FLOOR_PRODUCTION). H2's z_world-term raw_std effect is the open question the formal validation
  answers.
  Not a learning module -- no new nn.Module, no parameters, no phased training. MECH-094 N/A
  (only the CEM ranking criterion / proposal breadth for an already-computed rollout changes; no
  new simulation/replay content is written to memory).
  Validation experiment: V3-EXQ-928 queued (OFF control / H2-only / H3-only / BOTH arms,
  re-measuring V3-EXQ-869 C1 at num_cem_iterations=3; acceptance = mean pairwise raw_std mode gap
  >= 0.01 per arm; ARM OFF expected to FAIL as the washed-out control). Dry-run (3 seeds,
  OFF+H3): OFF gap ~-0.0004 (washed), H3 ~0.0246 (clears floor).

  **`mode_partitioned_cem` DEFAULT FLIPPED False -> True 2026-08-26**
  (chip-20260825-mech267-cem-flip-default), per the V3-EXQ-927/928
  validation this SD's own doc queued: H2 (`mode_value_weight`) is a confirmed NULL on the C1
  wash-out target (paired +0.0015, t=+0.48) and stays at its default `{}` -- deliberately left
  dormant rather than retired, since a null on the C1 metric is not evidence the term is inert
  everywhere, and deleting it would drop its own contract coverage
  (`test_cem_modulatory_authority.py`, `test_exp0155_action_bias_no_scoring_authority.py`) for
  no behavioural gain. H3 (`mode_partitioned_cem`) rescues the extreme-pair (tight-vs-broad)
  breadth contrast (H3-OFF control +0.0167, t=+4.34, 24/30 seeds positive) and is now the
  production default. Still gated on `mode_conditioning_enabled` AND `operating_mode` being
  supplied, so this is bit-identical for every caller that does not already enable mode
  conditioning with an operating mode.
  **RESIDUAL, not closed by this flip**: V3-EXQ-928 found the ORDERED four-mode gradient still
  NOT restored (`per_arm_all_adjacent_gaps_clear_floor` false in all four arms; adjacent gaps
  +0.00729 / +0.00750 / -0.00116, the last INVERTED). Only the broad-minus-tight extreme-pair
  contrast is rescued -- do not read this landing as full closure of MECH-267 mode-conditioning.
  See MECH-267, SD-MECH267-HORIZON-DEPTH, SD-055, `failure_autopsy_V3-EXQ-923_2026-08-12`,
  `failure_autopsy_V3-EXQ-869_2026-08-02`,
  `REE_assembly/docs/architecture/sd_mech267_cem_selection_fix.md`.
