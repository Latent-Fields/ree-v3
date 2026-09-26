## MECH-288 magnitude-relative slow-scale BOCPD trigger (2026-09-26)

- MECH-288 slow scale: hippocampal.event_segmenter (bocpd_gaussian, relative trigger) -- IMPLEMENTED 2026-09-26.
  `ree_core/hippocampal/event_segmenter.py` (`_BOCPDGaussianDetector` relative mode, `Scale` fields,
  per-stream `_build_detector`), pass-through in `ree_core/hippocampal/module.py`.
  substrate_queue row `MECH288-SLOW-SCALE-BOCPD-RAIL-UNREACHABLE` (IGW-20260925-220), user decision
  rec-20260925-deedce42 / GFLAG-0474 option (a).
  **Why:** the canonical slow scale could not fire on the in-agent z_goal stream (0 slow fires on a live
  stream, 3 seeds). Three independent defects: fresh runs predict with an ABSOLUTE `prior_var=1.0`; the
  implausibility cutoff is -20 nats of log-density (scale-dependent, needs a one-tick move > ~6.18); and
  `P(r_t=0) == hazard` on every tick (Adams-MacKay likelihoods cancel), so the `p0` readout never clears 0.5.
  **What:** `scale_mode="relative"` -- fresh-run prior sd = `prior_scale_k` (6) x an EMA (`scale_alpha` 0.05)
  of the stream's own |per-tick displacement|, run-variance floor 1e-12, implausibility backstop on the
  standardised residual (`implausible_z` 6). `readout="short_run_mass"` -- fire on `P(r_t <= readout_lag)`
  (lag 3), held off for lag+1 observations after start and after every reseed. Same stream (||z_goal||),
  same Gaussian model, same hazard/top-k/min_segment_length.
  Config: `EventSegmenterScaleConfig.bocpd_scale_mode` (default `"absolute"`), `bocpd_readout` (default
  `"p0"`), plus `bocpd_prior_scale_k`, `bocpd_scale_alpha`, `bocpd_rel_floor`, `bocpd_implausible_z`,
  `bocpd_readout_lag`; `REEConfig.from_dims(event_segmenter_slow_relative_trigger=True)` sets both modes on the
  slow scale.
  **Observation stream only:** the rollout stream always gets the canonical detector -- the MECH-321
  scale-resolved probe feeds the same z_goal up to 8x per candidate per tick, which collapses a relative
  detector's run variance (1332 fires / 21624 ticks in the red-team replay).
  Backward compatible: defaults run the original code path; bit-identical against pre-change ree-v3 7f08512
  (events, posteriors, run-length probabilities, both input streams).
  Measured (offline replay of live in-agent traces, EXQ-830 live-z_goal config, 3 seeds): 21/22 z_goal
  write-burst onsets detected, 0 fires outside a write burst, latency 0-1 tick; canonical 0/22.
  **Validation-config requirement:** inert unless BOTH the relative trigger AND a live z_goal
  (`z_goal_enabled=True`, `benefit_threshold=0.05`, explicit `agent.update_z_goal(...)` each step) are on.
  The segmenter resets every episode (`agent.reset()`), so episodes must be long enough to hold several
  slow segments (write-onset gaps measured 28-130 ticks).
  Phased training required: no. MECH-094: n/a (observation stream only; rollout stream unchanged).
  Contract: `tests/contracts/test_mech288_relative_bocpd.py` (R1-R9).
  Validation experiment: pending (see /queue-experiment; not yet queued).
  Design + red-team: `REE_assembly/evidence/planning/mech288_slow_scale_rail_redesign_20260926.md`.
  See MECH-288, MECH-287.
