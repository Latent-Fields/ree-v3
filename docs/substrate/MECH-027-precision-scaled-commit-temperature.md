## MECH-027 precision-scaled commit temperature -- graded consumer for current_precision (2026-09-02)
- MECH-027 (pathological control-plane regimes) needed current_precision / running_variance to
  have a GRADED behavioral effect for its hypervigilance falsifier (elevated gain/precision +
  shortened prediction horizon + suppressed replay). A red-team review of the drafted V3-EXQ-979
  falsifier found this BLOCKED: the ONLY existing consumer of precision was the binary ARC-016
  commit gate (committed = commit_variance < effective_threshold), which SATURATES in a trained
  baseline (running_variance empirically ~125x below threshold) -- once committed, further
  lowering variance (a hypervigilance push) had no further observable effect anywhere downstream.
  use_conditional_precision_gate (SD-063) does not fix this: it only changes WHICH variance feeds
  the gate (EMA vs a per-input predictive variance), not the gate's binary shape.
  FIX: use_precision_scaled_commit_temperature applies the SAME softening shape as MECH-439
  Factor B (gap-scaled commit temperature) with the F-gap quantity replaced by a PRECISION-margin
  quantity: precision_margin_norm = clamp(1 - commit_variance/effective_threshold, 0, 1) -- 0 at
  the threshold (barely committed), 1 as commit_variance -> 0 (maximally confident). When on, the
  committed argmin becomes multinomial(softmax(-q / T_eff)) with T_eff = base_temperature +
  precision_scaled_commit_entropy_alpha * (1 - precision_margin_norm): a barely-committed tick
  commits HOTTER (softer argmax, more exploratory); a maximally-confident tick commits COLD
  (T_eff -> base, recovers the hard argmin -- rigid, narrow selection, the hypervigilance
  direction). q is restricted to an F-eligibility envelope (candidates within
  precision_scaled_commit_harm_floor * raw_score_range of the best raw score), reusing Factor B's
  safety-gate shape, so a hot commit-T can never softmax-promote a clearly-harmful candidate.
  Module: ree_core/predictors/e3_selector.py (E3TrajectorySelector.select computes
  precision_margin_norm right after the commit decision + _precision_scaled_commit_pick helper +
  4 diagnostics keys), ree_core/utils/config.py (E3Config 3 fields + REEConfig.from_dims).
  SCOPE: wired ONLY at the standalone committed branch (no active Factor-A modulatory shortlist,
  use_loop_segregation off) -- the shortlist-then-modulate and loop-segregation committed
  branches do not yet carry this lever (future extension if a falsifier needs it there).
  PRECEDENCE: when use_gap_scaled_commit_temperature is ALSO on, gap-scaling takes precedence
  (checked first) -- preserves MECH-439's existing composition semantics exactly for anyone
  already using Factor B; the two levers are not combined/blended. Default False -> the hard
  argmin path is bit-identical (verified).
  Backward compatible: no trainable parameters, no gradient flow, no new LatentState field.
  MECH-094: N/A (waking committed-selection path; no replay/memory write surface).
  Contracts: tests/contracts/test_mech027_precision_scaled_commit_temperature.py (9: OFF
  bit-identical hard argmin + default flags; precision_margin_norm always populated in
  world-variance-commit mode; T_eff monotone + precision-scaling load-bearing; cold/confident
  recovers the decisive argmin; hot/barely-committed spreads across candidates over seeds
  (direct pick-fn unit coverage); end-to-end spread through select() with a wide envelope;
  standalone harm-floor safety gate excludes a harmful candidate; gap-scaled commit temperature
  takes precedence when both levers are enabled). Registered in tests/test_flag_inertness.py
  PROBED.
  GOVERNANCE: PROMOTES NOTHING. MECH-027 stays provisional; this is substrate-only (no
  claims.yaml disposition change). Unblocks the graded-precision half of MECH-027's
  non-degeneracy precondition for a future V3-EXQ-979-style falsifier (the sleep-cycle
  interleave-in-eval half is a separate, independent gap -- see the sleep_substrate_plan.md /
  chip-20260902-mech027-precision-replay-eval-substrate discussion).
  See MECH-027 (claims.yaml; the claim this unblocks), ARC-016 (E3 dynamic precision; the
  binary gate this softens), MECH-439 (the gap-scaled commit-T lever this mirrors and defers
  to under composition), SD-063 (use_conditional_precision_gate; a DIFFERENT precision lever --
  which variance feeds the gate, not the gate's binary shape), MECH-025 (shares MECH-027's
  ARC-016/ARC-005 substrate dependency; three prior false-negative instrument defects on
  exactly this kind of read -- confirm this fix carries the same non-degeneracy discipline).
