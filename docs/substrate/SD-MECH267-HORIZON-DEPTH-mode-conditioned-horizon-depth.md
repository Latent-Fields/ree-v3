## SD-MECH267-HORIZON-DEPTH: Mode-Conditioned Horizon-Depth Modulation -- IMPLEMENTED (2026-08-02)
- SD-MECH267-HORIZON-DEPTH: hippocampal.mode_conditioned_horizon_depth -- IMPLEMENTED 2026-08-02.
  `ree_core/hippocampal/module.py`. Adds MECH-267's SECOND named mechanism (the claim's own
  2026-04-20 `implementation_note` and the original 2026-04-27 lit-pull recommendation both name
  horizon-depth modulation; only noise-scale was built at the time, with horizon-depth deferred
  to "V4 reconsideration"). Config: `HippocampalConfig.mode_horizon_scale` (Dict[str, float],
  default `{"external_task": 0.5, "internal_planning": 1.0, "internal_replay": 0.7,
  "offline_consolidation": 1.0}`), gated by the SAME `mode_conditioning_enabled` switch as the
  existing `mode_noise_scale` (both are facets of one MECH-267 mechanism, not separate features).
  Data flow: `operating_mode` -> `HippocampalModule._compute_mode_horizon_scale()` (mirrors
  `_compute_mode_noise_scale`'s per-mode weighted-average reduction) -> `effective_horizon =
  clamp(round(config.horizon * horizon_frac), 1, config.horizon)` -> `_score_trajectory(traj,
  max_horizon=effective_horizon)`, which truncates the world-state sequence used for CEM
  elite-selection scoring to `max_horizon + 1` states (initial + max_horizon steps) before the
  ARC-007 terrain-only score (and any wanting/curiosity extension) is computed. Does NOT change
  the physical rollout length -- `config.horizon` stays a fixed structural network dimension
  (`terrain_prior`'s output width, set at construction time); this SD modulates the CEM
  elite-selection SCORING WINDOW ("look-ahead depth used for candidate evaluation"), not the
  rollout itself. Elite refit still targets the full H-step action-object sequence; only the
  ranking criterion is windowed.
  Backward compatible: `_score_trajectory` gained an optional `max_horizon: Optional[int] = None`
  parameter; its four pre-existing call sites pass no argument and are bit-identical. The one
  new call site (per-candidate CEM scoring inside `propose_trajectories`) passes `None` whenever
  `mode_conditioning_enabled` is False or `operating_mode` is not supplied, so the truncation
  branch is never entered in that case -- verified by smoke test (deterministic same-seed score
  match with the mechanism disabled, and with it enabled but `operating_mode=None`).
  Diagnostics: `_last_mode_horizon_scale` (the resolved fraction) and `_last_effective_horizon`
  (the resolved int step count), both `None` when disabled/no-op, mirroring the existing
  `_last_operating_mode`/`_last_mode_noise_scale` convention.
  Biological grounding: Wikenheiser & Redish 2015 (deliberative/VTE theta sequences extend
  further ahead than fast-locomotion theta sequences during active task engagement); also
  Pfeiffer & Foster 2013, Olafsdottir et al. 2018, Mattar & Daw 2018.
  Motivated directly by V3-EXQ-869 (2026-08-02, 30 seeds, confirmed
  `failure_autopsy_V3-EXQ-869_2026-08-02`, USER-CONFIRMED): noise-scale-only modulation cleanly
  demonstrates mode-differentiated proposal content at `num_cem_iterations=1` (C0 PASS) but
  washes out to noise under production `num_cem_iterations=3` (C1 FAIL, 0/30 seeds show the
  predicted ordering) -- consistent with an initial-distribution-only perturbation not surviving
  CEM's iterative elite-refit toward a mode-independent value optimum, not with the claim being
  biologically wrong.
  Not a learning module -- no new `nn.Module`, no parameters, no phased training. MECH-094 N/A
  (no new simulation/replay content is written to memory; only the CEM ranking criterion for an
  already-computed rollout changes).
  Evidence staleness: V3-EXQ-869's run manifest
  (`evidence/experiments/v3_exq_869_mech267_mode_conditioning_content_persistence/runs/
  v3_exq_869_mech267_mode_conditioning_content_persistence_20260802T035422Z_v3/manifest.json`)
  marked `pending_retest_after_substrate: true` / `superseded_by_substrate:
  SD-MECH267-HORIZON-DEPTH@2026-08-02` (propagating claims.yaml's existing flag to the nested
  manifest the indexer reads). Landing recorded in
  `REE_assembly/evidence/planning/substrate_dependencies.json`.
  Validation experiment: NOT queued this session -- explicitly out of scope per this IGW item's
  instructions (substrate-build only). Follow-on: `/queue-experiment` a same-question re-queue of
  V3-EXQ-869's C1 condition (production `num_cem_iterations=3`) with BOTH mechanisms active,
  flagged via a follow-up chip rather than left silently undone.
  See MECH-267, SD-032a, MECH-261, MECH-092,
  `REE_assembly/docs/architecture/sd_mech267_horizon_depth_modulation.md`,
  `failure_autopsy_V3-EXQ-869_2026-08-02`.
