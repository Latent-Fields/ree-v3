## scaffolded_sd054_onboarding AMEND: n_cue_recall_fires aggregation fix (2026-06-04c)
- scaffolded_sd054_onboarding cue-fires aggregation fix -- IMPLEMENTED 2026-06-04.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). The clean underlying fix for the V3-EXQ-638
  measurement gap surfaced 2026-06-04 while validating V3-EXQ-638a.
  ROOT CAUSE (code-confirmed): _eval_episode RETURNS a per-episode
  n_cue_recall_fires and _train_episode ACCUMULATES cue fires into
  goal_write_diag["n_cue_recall_fires"], but run_p2 never aggregated the
  per-episode value onto the returned P2OnboardingMetrics and run_p1 never
  surfaced a total on P1OnboardingResult -- so P2OnboardingMetrics had NO
  n_cue_recall_fires field at all. Any consumer doing
  getattr(p2, "n_cue_recall_fires", 0) therefore silently read 0 EVEN WHEN THE
  CUE FIRED (directly observed: V3-EXQ-638a smoke fired the cue 30x in P2 --
  visible in P2OnboardingMetrics.cue_diag["n_cue_recall_fires"]==30 -- while
  getattr returned 0). The original V3-EXQ-638 script reads
  getattr(p2, "n_cue_recall_fires", 0) for its C1 gate -> 638's C1 would report
  cue_fires=0 even if the cue fired. 638a already works around it by sourcing
  p2.cue_diag["n_cue_recall_fires"]; this is the clean fix so future consumers
  do not trip on it.
  THE FIX (no-op-default; bit-identical when the cue bridge is off -> 0):
    (1) New aggregated field n_cue_recall_fires: int = 0 on BOTH P2OnboardingMetrics
        and P1OnboardingResult.
    (2) run_p2 sums the per-episode ep_metrics.get("n_cue_recall_fires", 0) across
        episodes (mirroring how total_contact / num_contact_events are already
        aggregated there) and sets it on the constructor.
    (3) run_p1 surfaces goal_write_diag.get("n_cue_recall_fires", 0) (already
        accumulated in _train_episode) onto P1OnboardingResult.
  CONTRACT: the new top-level field EQUALS cue_diag["n_cue_recall_fires"] (the
  per-episode returns / goal_write_diag accumulator and the shared cue_diag both
  count the same fires); 0 when the bridge is off. cue_diag is unchanged (it
  already carries the authoritative count).
  Backward compatible: cue bridge off -> both fields default 0; cue_diag empty;
  bit-identical. 65/65 scaffold contracts (62 prior + 3 new C9) + 7/7 preflight PASS.
  Phased training: N/A (pure readout aggregation; no learned parameters).
  MECH-094: N/A (waking-stream readout; no simulation/replay write surface).
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C9 cont.
  (c9_p2_aggregates_cue_recall_fires_equals_cue_diag -- bridge-on nonzero total ==
  cue_diag + getattr no longer reads a silent 0; c9_p2_cue_recall_fires_zero_when_bridge_off
  -- bridge-off 0 == cue_diag; c9_p1_p2_result_field_defaults_zero_on_master_off --
  field exists, defaults 0, master-off short-circuit carries 0 on both results).
  See scaffolded_sd054_onboarding cue-recall FORMATION fix entry above (the
  diagnostics this completes), SD-057 / MECH-347 (L6 cue-recall claim; ree_core
  unchanged), V3-EXQ-638 (the cue-silent FAIL whose C1 read getattr), V3-EXQ-638a
  (the validation that surfaced the gap and already sources cue_diag), MECH-094 (N/A).
