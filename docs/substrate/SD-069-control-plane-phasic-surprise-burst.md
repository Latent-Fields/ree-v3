## SD-069: control_plane.phasic_surprise_burst -- IMPLEMENTED (2026-07-17)
- SD-069: control_plane.phasic_surprise_burst -- IMPLEMENTED 2026-07-17.
  LC-NE PHASIC complement to MECH-313 noise_floor (tonic) on the SAME E3 softmax
  temperature channel -- the substrate that makes MECH-063 sub-claim (ii)
  (tonic/phasic split) behaviourally testable. Module:
  ree_core/regulators/phasic_surprise_burst.py (PhasicSurpriseBurst; pure-arithmetic
  regulator matching noise_floor.py MECH-313 / broadcast_override.py SD-037 -- no
  nn.Module, no learned params). Config: REEConfig.use_phasic_burst (default False;
  set True to enable) + phasic_burst_surprise_ema_decay (0.1), _trigger_ratio (1.5),
  _trigger_floor (1e-6), _temp_delta (-0.5 = phasic sharpening), _decay (0.5),
  _min_temperature (0.1), and _signal_source ("running_variance" default; set
  "instantaneous_pe" for the sharp source -- see below). Data flow: <surprise source>
  (per-tick PE surprise) ->
  PhasicSurpriseBurst.tick (event iff surprise >= trigger_ratio x EMA baseline;
  inject drive -> envelope decays geometrically over a few ticks) -> burst_level
  [0,1] -> temperature_delta = temp_delta x burst_level -> combined_T =
  max(tonic_effective_T + delta, min_temperature) -> e3.select(candidates,
  combined_T) at agent.py select_action(). Tonic (MECH-313) applies a SUSTAINED
  every-tick lift; SD-069 adds an EVENT-LOCKED transient on top -> comparable
  readouts, independently toggleable. IMPORTANT: reuses the MECH-104
  volatility-surprise lit basis (Aston-Jones & Cohen 2005 phasic mode) but routes to
  the softmax, NOT the ARC-016 commit gate -- so it does NOT implement the MECH-104
  claim (control_plane.volatility_interrupt, v3_exq_365), which is the same surprise
  routing to the de-commit gate. Diagnostic: phasic_burst_level / _temp_delta in
  _last_score_bias_decomp; phasic_burst block in _last_control_vector (tonic lift kept
  uncontaminated for the dissociation readout). Backward compatible: agent does not
  instantiate the regulator when use_phasic_burst=False; combined T == tonic T; OFF
  action stream bit-identical (contract C8, n_events==0). MECH-094: simulation_mode
  returns cached burst, no state advance; no memory write surface -> phased training
  N/A, hypothesis_tag N/A. Smoke: 17 SD-069 contracts pass; full tests/contracts
  1462 pass (0 regressions); injected surprise spike fires through select_action ->
  pre-commit softmax entropy drops on the event (0.80->0.06) and decays over the tail
  (event-locked transient), bit-identical before the event.
  SHARP-SURPRISE SOURCE (2026-07-17): the event-detector source is now selectable via
  REEConfig.phasic_burst_signal_source. "running_variance" (default, no-op) reads the
  SMOOTHED e3._running_variance EMA -- which decays monotonically for an untrained
  forward model (0.475 -> 0.004) and washes out real per-tick spikes, so the lever
  fires 0 natural events even under env volatility (background_drift_enabled). This
  meant a no-training 777-style probe could only exercise the lever with a synthetic
  poke to _running_variance -- exactly what MECH-063 (ii) must avoid. "instantaneous_pe"
  reads a new no-op read-only signal e3.last_instantaneous_pe = the RAW per-tick PE-MSE
  (error_var in e3.update_running_variance) captured BEFORE the running-variance EMA
  smoothing folds it in, so genuine surprise spikes survive (Aston-Jones & Cohen 2005
  phasic mode fires on SHARP/instantaneous salience, not a smoothed average).
  Validated (untrained rollout, CausalGridWorldV2 drift on, NO synthetic poke): the
  phasic burst is ACTIVE across the run under "instantaneous_pe" (burst_ticks up to
  209/232 at trigger_ratio 1.3) but NEVER active under "running_variance" (0/217) --
  a clean load-bearing contrast. Note: the regulator's own n_events counter resets per
  episode (agent.reset -> phasic_burst.reset), so a readiness gate must sum events
  across episodes rather than read end-of-run get_state. Invalid source strings raise
  ValueError at agent construction (no silent fallback). Backward compatible: default
  "running_variance" reads the identical scalar as before -> existing experiments
  bit-identical; e3.last_instantaneous_pe is written unconditionally but read by nothing
  at the default source.
  Validation experiment: V3-EXQ-779 queued 2026-07-17 (MECH-063 sub-claim ii
  tonic-vs-phasic dissociation, 777-harness pattern, phasic_burst_signal_source=
  "instantaneous_pe" on PHASIC-ON arms; P0 readiness gate requires PHASIC-ON arms to
  fire >= MIN_EVENTS real surprise events -- see Step 8).
  Design doc: REE_assembly/docs/architecture/sd_069_phasic_surprise_burst.md
  DEFECT FOUND + REPAIRED BY SD-075 (2026-07-19): the per-episode cold EMA reset
  documented above makes n_event_ticks a function of episode LENGTH rather than
  surprise on short-episode seeds, because the first tick of an episode only seeds
  the baseline and the ~10-tick time constant exceeds the whole episode. If you are
  reading this to design a phasic experiment, read the SD-075 entry FIRST and set
  phasic_burst_baseline_continuity="carry".
  See MECH-063 (enables sub-claim ii), MECH-313 (tonic counterpart), MECH-104
  (shared lit basis, different consumer), SD-075 (baseline-continuity repair),
  ARC-005.
