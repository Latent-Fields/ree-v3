## SD-032b AMENDMENT: dACC candidate effort proxy (harm-forward rollout cost) (2026-09-25)

- SD-032b amendment: cingulate.dacc_candidate_effort_proxy -- IMPLEMENTED 2026-09-25
  (substrate_queue `sd032b-candidate-effort-proxy`, IGW-20260923-222).
  `REEAgent._dacc_candidate_effort()` in `ree_core/agent.py`, called at the dACC
  bundle build in `select_action()`.
  Config: `REEConfig.dacc_candidate_effort_source` (default `"horizon"`, no-op;
  set `"harm_a_forward"` to enable) and `REEConfig.dacc_effort_rollout_steps`
  (default 0 = the candidate's full action horizon). Both threaded through
  `from_dims()`; an unknown source raises `ValueError`.
  Data flow: `_current_latent.z_harm_a` -> `E2HarmAForward` rolled over each
  candidate's OWN action sequence, batched over K, no_grad ->
  `effort_k = mean_t ||z_harm_a_pred_t||_2` -> `candidate_effort[K]` ->
  `DACCAdaptiveControl.forward` (Shenhav `mode_ev = payoff - control_required *
  effort`; Croxson `harm_interaction`) -> `DACCtoE3Adapter` score_bias -> E3.
  Diagnostics: `agent._dacc_last_effort` ([K]) and
  `agent._dacc_last_effort_source` (`"horizon"` | `"harm_a_forward"`).
  Backward compatible: default `"horizon"` is the exact legacy build
  (`c.actions.shape[1]`); default and explicit-horizon runs are
  bit-identical. Falls back to `"horizon"` with a one-time RuntimeWarning when
  `use_e2_harm_a=False` or no current z_harm_a.
  Why: the legacy proxy is the physical rollout horizon, identical across every
  candidate, so `control_required * effort` was a uniform (argmin-invariant) shift
  and `harm_interaction` identically zero -- nothing pe-dependent (incl. MECH-268
  f_sat) could move E3 selection
  (REE_assembly `docs/architecture/mech_268_dacc_saturation_form.md`, Consumer A).
  Deliberately NOT routed through the MECH-269b VsRolloutGate: `gate_stream()` /
  `_gate_value()` advance diagnostic counters experiments read; all K candidates
  share one start state, so gating would move every rollout origin together.
  MECH-094: not implicated (waking select_action path, no_grad, no memory write).
  Phased training required: YES for E2_harm_a (P0 forward-model warmup, as in
  V3-EXQ-445g / 450) -- the readout itself has no parameters, but an untrained
  E2_harm_a yields a cross-candidate spread that is noise.
  Liveness (measured 2026-09-25, untrained agent, CausalGridWorldV2 8x8 / 8 hazards,
  use_dacc + affective harm + salience + lateral PFC + saturation, dacc_weight=1.0,
  600 env steps). The dACC block runs only on E3 ticks (it sits after select_action's
  `not ticks["e3_tick"]` early return), so counts are FRESH dACC evaluations with the
  `_dacc_last_*` latch cleared before every call: 61 fresh evaluations in 600 steps;
  effort spread > 0 on 61/61 (legacy "horizon": 0/61).
  SCALE CAVEAT (INERT at default cost): at `dacc_effort_cost=0.1` the effort-term
  range (~0.02) is ~1000x below the E3 payoff range (~25), so committed actions are
  unchanged (0/600 steps differ from the horizon arm); at `dacc_effort_cost=10`,
  40/600 steps differ. Control: under `"horizon"`, raising cost 0.1 -> 10 changes
  0/600 actions (uniform shift). A validation run must therefore set
  `dacc_effort_cost` (or `dacc_bias_max_abs`) so the effort term is on the payoff's
  scale, and train E2_harm_a first.
  CORRECTION (same day): the first version of this record said "80/80 ticks" and
  "10/60 ticks"; those counted latched `_dacc_last_*` re-reads on non-E3 steps
  (~10x pseudo-replication), not fresh evaluations. The direction was unaffected.
  E2_harm_a quality caveat: SD-PP-B9 (open, degrading) measured E2_harm_a BELOW the
  z(t-1) persistence predictor on V3-EXQ-1062a, so the cross-candidate effort
  spread may be dominated by model error -- liveness here is not validity; see the
  validation experiment.
  Contract: `tests/contracts/test_sd032b_candidate_effort_proxy.py` (9 tests; all 9
  FAIL against the pre-amendment tree -- effort_term uniform at 97.35 over K=32).
  Validation experiment: not yet queued (see WORKSPACE_STATE / claim note).
  See SD-032b, MECH-258, MECH-268, MECH-267.
