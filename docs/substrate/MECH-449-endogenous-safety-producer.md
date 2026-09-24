## MECH-449 Endogenous Safety Producer: harm-pathway safety axis for the Go/No-Go gate, running-scale calibrated (2026-09-24)

- MECH-449 (ARC-107): e3.go_nogo.endogenous_safety_producer -- IMPLEMENTED 2026-09-24.
  `ree_core/agent.py` `REEAgent._endogenous_gng_safety` (called from the `_gng_signals`
  assembly in `select_action`, next to the MECH-260 perseveration reuse), plus
  `reset_gng_safety_state()` / `gng_safety_diagnostics()`. Chip:
  `chip-20260918-mech449-endogenous-safety-veto-producer`.
  Config: `E3Config.use_gng_endogenous_safety` (default False; set True to enable),
  `gng_safety_z_threshold` (2.0), `gng_safety_ema_decay` (0.999), `gng_safety_sd_floor`
  (0.01), `gng_safety_warmup_samples` (200). All five are plumbed through
  `REEConfig.from_dims` (field + signature + re-apply; contract-checked).
  Data flow: candidate `world_states[1:]` -> `E3.harm_eval_head` (under `torch.no_grad()`)
  -> per-candidate mean harm h_k -> z_k against the per-agent running EMA scale (read
  BEFORE this tick's values are folded in; sd floored at `gng_safety_sd_floor`) ->
  `safety_k = clamp(gng_safety_floor * z_k / gng_safety_z_threshold, 0, 1)` ->
  `go_nogo_signals["safety"]` -> `E3._go_nogo_eligibility_gate` (fail-open-immune No-Go).
  So `safety_k >= gng_safety_floor` exactly when `z_k >= gng_safety_z_threshold`.
  An injected safety vector (`set_injected_go_nogo_signals`) still overrides it.
  Calibration choice (orchestrator decision `DECIDED Q-MECH449 -> A`, orchestrate-20260924-0808):
  raw sigmoid harm never or always crosses the absolute 0.5 floor (cross-state spread
  0.045 mean / 0.133 max, V3-EXQ-603k), and per-bank min-max normalisation fires every
  tick by construction; the running per-seed scale is the absolute reference. The sd
  floor is the specificity guard: when the running sd is small, pure spread would cross
  +2 SD at a fixed tail rate (the contract measures this blind spot). Its default 0.01 is
  derived (decision Q-MECH449-FLOOR -> A): the 0.02 harm-range precondition / the 2 SD
  threshold, so a veto needs harm >= running mean + 0.02. An untrained head is not flat in
  candidate space (V3-EXQ-1090 dry run: running raw sd ~0.005-0.007), so 0.005 would not bind.
  The running scale PERSISTS across `reset()` (per-seed, not per-episode).
  Arming chain (constructing the objects does NOT arm the gate): `use_go_nogo_constitution`
  AND (`use_f_eligibility_demotion` OR `use_modulatory_shortlist_then_modulate`) AND a live
  modulatory accumulator (any score_bias / MECH-341 / route term) AND K >= 2, plus this flag.
  Diagnostics: `gng_safety_diagnostics()["n_safety_nogo_applied"]` sums E3's per-tick
  `go_nogo_n_safety_nogo` (safety-vetoed candidates INSIDE the F-built eligible set on
  ticks where the gate ran -- the EVB-1409 release (b) statistic); `n_signal_fired` counts
  producer fires over all K candidates; `last` holds the latest tick's harm / z / fired.
  Backward compatible: disabled by default; the producer never runs; bit-identical OFF.
  Phased training required: no (reads an existing head; trains nothing). MECH-094: not
  applicable (waking selection read, no replay/memory write).
  Premise correction: the harm pathway is NOT reliably discriminative in the 603k regime
  (1/3 seeds); the 603q/866b regime (harm-pathway stabilisation + 603j safety signal) is
  (3/3, harm_eval_range 0.115-0.470). A validation run must gate per seed on
  harm_eval_range >= 0.02.
  Contract: `tests/contracts/test_mech449_endogenous_safety_producer.py`.
  Validation experiment: V3-EXQ-1090 (603q-base trained agent, gate armed at eval,
  harm-ON vs harm-OFF control, ground-truth counterfactual specificity check).
  See MECH-449 / ARC-107 (`MECH-449-arc-107-go-no-go-eligibility.md`), MECH-049 / EVB-1409.
