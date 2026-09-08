## SD-hazard-aware-policy-decomposition: policy.decomposition_via_event_segmenter.harm_aware_selection -- IMPLEMENTED 2026-08-01

V3-EXQ-844 autopsy successor (`failure_autopsy_V3-EXQ-844_2026-08-01`): MECH-321's mid-execution
abort mechanism engages correctly informationally (C2 PASS: aborting a stale macro lowers
post-abort forward-PE) but does not reduce task harm (C1 FAIL: -0.003262) because
`_apply_policy_decomposition` / `PolicyDecomposition.evaluate()`/`decompose_sequence()` read no
harm-valence signal and perform no ranked selection among a withheld chunk's own candidate
re-tilings -- every leaf tile is additively recombined unweighted. Fixed per a targeted lit-pull
(`evidence/literature/targeted_review_threat_modulated_defensive_path_selection/SYNTHESIS.md`, 9
entries -- Fanselow PIC, Mobbs 2007/2020, Evans 2018, Cooper 2016, Blanchard & Blanchard 1989)
Form B recommendation: a two-stage threat-modulated selection rule.

- **Stage 1 (graded, always active).** `PolicyDecomposition.harm_bias(harm_penalty,
  z_harm_a_norm)`: `w(h) * harm_penalty`, gain-scaled and clamped
  (`harm_bias_gain`/`harm_bias_scale`), where `w(h) = harm_threat_scale(z_harm_a_norm)` is a
  linear ramp reused verbatim from `InstrumentalAvoidanceGate.threat_scale` /
  `EscapeAffordanceBridge.threat_scale`. `harm_penalty` is read per candidate leaf tile as the
  mean `VALENCE_HARM_DISCRIMINATIVE` residue-field channel (SD-014) over the leaf's OWN predicted
  `world_states` (`HippocampalModule._decomposition_harm_penalty`, module.py) -- the same
  per-location valence-read pattern `build_goal_payload` already uses for `VALENCE_WANTING`
  (SD-039), applied to each decomposition candidate's own rollout rather than the agent's current
  position. Tagged onto each leaf `Trajectory.metadata["decomposition_harm_bias"]`; a new block in
  `REEAgent.select_action` (agent.py, composed alongside the `InstrumentalAvoidanceGate` /
  `EscapeAffordanceBridge` blocks) gathers it into the same additive `score_bias` chain E3 folds
  into its own scoring -- ARC-007 value-flatness intact, `PolicyDecomposition` never scores a
  trajectory itself.
- **Stage 2 (categorical, threshold-gated).** `PolicyDecomposition.select_harm_aware_leaves`: at
  `w(h) >= harm_override_w_threshold`, restricts a withheld chunk's OWN leaf tiles to the single
  lowest-harm-penalty one (Mobbs 2007 / Evans 2018 categorical regime shift), overriding the
  harm-blind additive default; below threshold, all leaves are kept unchanged (Evans 2018
  freeze-as-fallback). Realised as a pool-ADMISSION decision in `_apply_policy_decomposition`
  (module.py) -- the same authority the function already exercises when excluding a depth-capped
  candidate -- not an oversized score_bias, which would violate this codebase's existing
  "no single channel dominates the score_bias chain" discipline.
  Config: `PolicyDecompositionConfig.use_harm_aware_selection` (default `False`; mirrored from
  `REEConfig.decomposition_use_harm_aware_selection`) + `harm_bias_gain` (0.1) /
  `harm_bias_scale` (0.1) / `harm_threat_floor` (0.1) / `harm_threat_ref` (0.5) /
  `harm_override_w_threshold` (0.9). Three-site wiring (dataclass field + `from_dims` signature +
  assignment), no hippocampal sub-config mirror needed -- same shape as `use_policy_decomposition`
  itself.
  Data flow: `z_harm_a` (SD-011) -> `REEAgent._e3_tick` -> `HippocampalModule.propose_trajectories`
  -> `_apply_policy_decomposition` -> `_decomposition_harm_penalty` (residue field
  `VALENCE_HARM_DISCRIMINATIVE` read on each leaf's predicted `world_states`) ->
  `PolicyDecomposition.harm_bias` / `.select_harm_aware_leaves` -> leaf `Trajectory.metadata` ->
  `REEAgent.select_action` score_bias composition -> E3.
  Backward compatible: disabled by default; `use_harm_aware_selection=False` means
  `_apply_policy_decomposition` never reads the residue field for this purpose and never tags
  metadata -- bit-identical to pre-existing MECH-321 behaviour. Full pre-existing
  `test_arc070_policy_decomposition.py` contract suite (32 tests) passes unmodified.
  Biological basis: see the lit-pull SYNTHESIS.md above; deliberately deferred (not built here,
  per the lit-pull's own explicit scoping): escapability (Cooper 2016), predictability/certainty
  (Fanselow 2022 BST; Blanchard & Blanchard 1989), rate-of-change triggering (Fanselow 2019), and
  the sustained-threat disengagement failure mode as a validation negative control.
  Phased training required: no -- pure arithmetic, no learned parameters (mirrors
  `ChunkAccumulator`/`InstrumentalAvoidanceGate`).
  Validation experiment: none queued this session (substrate build only, per scope discipline --
  see the SD doc "What This SD Enables" for the follow-on behavioural-effect experiment this
  unblocks). SD doc:
  `REE_assembly/docs/architecture/sd_hazard_aware_policy_decomposition.md`.
  Contracts: `tests/contracts/test_arc070_policy_decomposition.py` C18-C24 (11 new tests: config
  wiring, `harm_threat_scale` ramp + degenerate-ramp safety, `harm_bias` clamp + off-safety,
  `select_harm_aware_leaves` threshold behaviour + stable-argmin tie-break, live-agent pool
  admission below/at threshold, `select_action` score_bias composition).
  See MECH-321, ARC-070, SD-014, SD-039, SD-011, MECH-357/SD-058, MECH-358/SD-059.
