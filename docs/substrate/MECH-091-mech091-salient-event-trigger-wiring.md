## MECH091-SALIENT-EVENT-TRIGGER-WIRING / MECH-091: Salient-Event Trigger Wiring -- IMPLEMENTED (2026-08-17)
- MECH-091 names THREE salient events that phase-reset the E3 heartbeat clock: task completion,
  unexpected harm, commitment-boundary crossing. Before this change, `HeartbeatClock.phase_reset()`
  (`ree_core/heartbeat/clock.py:182`, built and already the DOMINANT driver of E3 tick cadence in a
  real 53,063-step rollout per `evidence/planning/diagnostic_arc071_e3_reselection_probe_2026-08-01.md`)
  had exactly ONE call site (the `harm_signal < 0` path in `update_residue()`), so only the
  unexpected-harm trigger fired -- completion and commitment-boundary crossing were unwired for four
  months despite both events already existing and firing elsewhere in the substrate. Governance
  un-parked this claim to V3 2026-08-16 (GFLAG-0037 SPLIT decision, substrate_queue.json sd_id
  `MECH091-SALIENT-EVENT-TRIGGER-WIRING`) after confirming the SD-006-phase-2 deferral that had held
  it since 2026-05-08 was mis-scoped -- wiring two already-existing events to an already-built,
  already-firing function, not new mechanism.
  Landed ree-v3 `main` `6293b2395248` (2026-08-17T07:07:23+0100). Four `agent.py` call sites added to
  the pre-existing one: task completion via the ARC-028/MECH-105 hippocampal-completion-driven
  BetaGate release (`_e3_tick`, the `if released:` block, alongside the existing
  `_committed_anchor_keys` clear); commitment-boundary crossing (entry) at all three genuine
  not-elevated -> elevated `beta_gate.elevate()` transitions in `select_action()` (the NCL-hold
  reassert path, the readiness-conjunction admission path, and the MECH-342 pressure-reset path --
  the third site is gated on the same genuine-transition pre-check already used there, since it
  re-invokes `elevate()` on every committed tick in a legacy mode and an unconditional
  `phase_reset()` would have fired every tick, not just the boundary crossing).
  Contracts: `tests/contracts/test_mech091_salient_event_triggers.py` (5 new, all pass) -- C1/C2
  commitment-entry fires exactly once on the transition tick and never again while elevated; C3/C4
  completion-release fires `phase_reset()` iff the gate actually releases (negative control:
  below-threshold signal does neither); C5 the pre-existing harm trigger, including its MECH-094
  `simulation_mode` no-op guard, is unaffected.
  Reference consumer: `experiments/_lib/baselines/mech091_phase_reset.py` `resolve_trigger_sites()` /
  `assert_trigger_wiring()`, which resolve all three trigger classes from `agent.py`'s live source at
  runtime -- the mechanism V3-EXQ-944/944a/944b's driver scripts check before running.
  Validation status: BUILT, not yet validated. V3-EXQ-944 ran 2026-08-22 and did not validate --
  `failure_autopsy_V3-EXQ-944_2026-08-22` found all four acceptance criteria passed non-degenerately
  but the run voided at readiness precondition P3 (the rate-matched control degenerated into
  NO_RESET); the autopsy routes a `/queue-experiment` successor with three driver-side guard fixes,
  not further substrate work. `substrate_queue.json` `status_phase: validation_pending`.
  See MECH-090 (the beta-gate boundary this wiring reads), MECH-105/ARC-028 (hippocampal completion),
  MECH-342 (the pressure-reset transition site), SD-006 (the phase-2 deferral this un-parked from).
