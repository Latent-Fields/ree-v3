## Natural-commit LATCH-HOLD amend: establish the sustained-hold OFF baseline (V3-EXQ-460i gate amend) (2026-06-21)
- control_plane.natural_commit_latch_hold -- IMPLEMENTED 2026-06-21 (substrate;
  PROMOTES NOTHING; claims.yaml NOT modified; MECH-446/445 stay candidate /
  v3_pending / pending_retest_after_substrate). The HOLD that establishes a sustained
  natural-commit beta-latch occupancy for the rung-6 RELEASE
  (natural_commit_occupancy_release, landed 2026-06-20) to act on. Routed by the
  user-adjudicated failure_autopsy_V3-EXQ-460i_2026-06-21 (Option B "make the OFF
  baseline actually sustain"). Pairs with the V3-EXQ-460j gate-3 sustained-hold
  redesign (the experiment side).
  ROOT CAUSE (V3-EXQ-460i, confirmed): the rung-6 release self-routed
  substrate_not_ready_requeue at readiness gate 3 (lever_did_not_shorten_occupancy).
  The lever was correctly ARMED (lever_present=true on all 3 armed arms; _clone_arm
  set use_natural_commit_urgency_release + modes + gap_sensitivity) and its arm-site
  note_commit_entry was reached on NATURAL result.committed commits, but it fired
  ZERO releases because the 460h sustained ~2400-step monolithic natural-commit hold
  DID NOT REPRODUCE -- the active SD-034 de-commit control-plane (closure->beta
  coupling re-toggle + the Leg-B committed-run-scaled refractory + closure releases)
  fragmented the beta latch to ~1-tick blips EVEN WITH THE RELEASE OFF (ARM_LEVER_OFF
  total_beta_elevated ~= beta_release_events, 415/405 seed 43; ~35 re-commits/episode),
  so there was no sustained occupancy to shorten and the urgency accumulator (reset
  per fresh entry, ~0.01-0.02/tick) could not reach release_bound over ~1 tick.
  Readiness gate 3's mean_beta_elevated_steps proxy is BLIND to sustained-vs-fragmented
  and green-lit the fragmented regime. The release lever is sound; the REGIME was missing.
  THE FIX (no-op default; bit-identical OFF): a latch-HOLD SEPARATE from (and
  independent of) the release lever -- so it arms in the ARM_LEVER_OFF baseline too
  (where natural_commit_urgency is None). A fresh NATURAL commit (result.committed)
  ARMS the hold; while armed AND the committed trajectory persists, the beta latch is
  RE-ASSERTED each tick (kept elevated against the de-commit churn) so the
  natural-commit occupancy sustains BY CONSTRUCTION -- the sustained reference the
  rung-6 release shortens and the gate-3 sustained-hold proxy certifies. With the hold
  keeping beta elevated, note_commit_entry fires ONCE (the bistable elevate block is
  skipped while already elevated), so the rung-6 urgency accumulates monotonically over
  the held duration and FIRES -- exactly what 460i lacked. The hold YIELDS to (disarms
  on) the three PRINCIPLED releases so it never papers over them: (a) an SD-034 closure
  de-commit (beta_gate.refractory_remaining > 0 -> the latch is held DOWN by the
  closure; the hold does not fight it, preserving the MECH-446 within-arm occupancy-drop
  DV), (b) the MECH-091 genuine-threat urgency interrupt (safety -- NEVER overridden),
  (c) the rung-6 NaturalCommitUrgencyRelease's own duration release (the lever
  shortening the held commit -- the whole point; the hold disarms so the occupancy
  stays shortened). Also disarms when the committed trajectory ends or the optional
  max-ticks safety cap is reached.
  Modules: ree_core/utils/config.py (REEConfig.use_natural_commit_latch_hold [False,
  master, INDEPENDENT of use_natural_commit_urgency_release] + natural_commit_latch_hold_max_ticks
  [0 = unbounded safety cap] + from_dims passthrough); ree_core/agent.py (hold state
  _ncl_hold_active/_ncl_hold_ticks/_ncl_hold_reassert_count + per-tick principled-release
  flags _ncl_mech091_fired/_ncl_lever_fired in __init__ + reset; ARM at the bistable
  natural-commit elevate site on result.committed; per-tick RE-ASSERTION after all
  release sites + before the between-tick branch; flag-set at the MECH-091 + rung-6
  release sites). natural_commit_urgency.py UNCHANGED (the hold is agent-level, since it
  must work without the release lever instantiated). Data flow: arm-site -> _ncl_hold_active
  True; end-of-tick re-assertion -> if armed AND committed trajectory persists AND
  refractory_remaining==0 AND no MECH-091/rung-6 release this tick AND under max-ticks ->
  beta_gate.elevate() (keep elevated); else disarm.
  Backward compatible: use_natural_commit_latch_hold=False by default -> _ncl_hold_active
  stays False, no arm, no re-assert; the per-tick flags are no-op bool writes ->
  bit-identical OFF. 7/7 new contracts (tests/contracts/test_natural_commit_latch_hold.py:
  C1 defaults + master-off no-op / C2 arm-site / C3 re-assert-against-churn LOAD-BEARING
  [hold ON sustains where hold OFF drops a non-re-committing latch] / C4 yield to closure
  refractory / C5 yield when the commit ends / C6 max-ticks cap / C7 bit-identical OFF) +
  8/8 preflight + full contract suite 1176 passed + 39 subtests (the 2
  test_runner_fail_branch failures are the documented pre-existing local-git-env runner
  flakes, zero overlap with this change). Activation = the C3 contract (the OFF latch
  drops under churn-without-recommit; the ON hold re-asserts it sustained).
  Phased training: N/A (control-state wiring + bool flags; no learned parameters).
  MECH-094: the hold re-assertion runs on the waking select_action path; it never
  overrides the MECH-091 safety interrupt and writes no memory content.
  Evidence-staleness (Step 8.5): NOT triggered -- no-op-default flag; every existing
  experiment uses the default (hold off), so no dependent claim's measured mechanism
  changed. KEEP all evidence.
  Validation experiment: V3-EXQ-460j (supersedes V3-EXQ-460i; queued via /queue-experiment)
  -- arms the hold in ALL arms + redesigns readiness gate 3 to a SUSTAINED-HOLD proxy
  (longest consecutive beta-elevated run + mean per-commit hold length
  total_beta_elevated/max(1,beta_release_events)) above a floor on >=2/3 OFF-arm guard
  seeds AND ncur_n_releases_total>0 with a >= LEVER_OCC_DROP_FRAC occupancy drop vs OFF on
  ARM_GAP_SCALED, BEFORE the CO_OCCURRENCE DV. Do NOT re-author 460d/e/f/g/h/i.
  MECH-446/445 stay candidate/v3_pending/pending_retest_after_substrate until 460j scores
  a contributory result.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  (natural-commit LATCH-HOLD amend section). Autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-460i_2026-06-21.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling rung-6 implementation_log).
  See natural_commit_occupancy_release (the rung-6 RELEASE this HOLD establishes the
  baseline for), SD-034 / MECH-090 / MECH-342 / MECH-445 / MECH-446 (the commitment-
  closure-control-plane cluster whose de-commit machinery the hold yields to; all
  candidate, unweakened), MECH-091 (genuine-threat interrupt the hold never overrides),
  MECH-439 / MECH-448 (F-dominance front), V3-EXQ-460i (the FAIL this addresses),
  V3-EXQ-460j (validation), MECH-094 (N/A).
