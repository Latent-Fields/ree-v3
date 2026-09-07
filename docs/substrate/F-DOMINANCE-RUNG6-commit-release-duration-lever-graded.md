## Commit/release-DURATION lever: graded natural-commit-occupancy release (rung-6 of f_dominance_conversion_ceiling; PARALLEL to MECH-448) (2026-06-20)
- control_plane.natural_commit_occupancy_release -- IMPLEMENTED 2026-06-20
  (substrate; PROMOTES NOTHING. claims.yaml NOT modified; the 460i-successor
  de-commit falsifier is the SEQUENCED NEXT step, NOT queued here). The
  commit/release-DURATION face of the F-dominance front -- PARALLEL to, not an
  escalation of, the selection-face levers (MECH-439 conflict-grade; MECH-448
  rank-preserving F->eligibility demotion). NOT blocked on GAP-I / V3-EXQ-689c
  (a dead-end selection-face parametric retest); the duration face is its own
  lever per the 460h governance note (behavioral_diversity_isolation:GAP-I
  governance_2026_06_20).
  PROBLEM (V3-EXQ-460h): on strong (F-decisive) seeds the bistable beta latch
  elevates once and HOLDS ~2400-2600 steps because nothing releases it (decisive
  F-gap = good options -> MECH-342 maintenance-release silent; no closure ->
  SD-034 silent). That monolithic natural-commit occupancy SWAMPS the SD-034
  closure de-commit -> MECH-445 commit-intent (375 on weak seed 44) and MECH-446
  de-commit occupancy-drop (only where committed_steps weak) never co-occur on
  the same seed (the 460h disjoint-certifier problem). This lever makes the
  F-driven natural commit LESS MONOLITHIC so weak-natural-commit is the norm
  across seeds, dissolving the 460h problem.
  BG-3 SYNTHESIS divergence D1 (load-bearing): biology does NOT set commitment
  DURATION with a fixed refractory clock -- it times the hold with a GRADED
  BG/pallidal urgency (Thura/Cisek 2022) and/or makes maintenance co-extensive
  with the executing action (Jin 2014). REE's committed-run-scaled beta-gate
  refractory is the "tuned, not bio-sourced" divergence. THEREFORE the lever is
  a GRADED release, never another fixed refractory constant.
  Module: ree_core/policy/natural_commit_urgency.py (NaturalCommitUrgencyRelease
  + NaturalCommitUrgencyReleaseConfig). Pure-arithmetic regulator (no nn.Module,
  no learned params, no gradient flow); sibling to commit_maintenance_release.py
  (MECH-342). REUSES BetaGate.committed_run_length (the MECH-090 commit-gate
  machinery) -- NO parallel latch module (ARC-106 guardrail G2). Two D1-faithful
  release modes, both togglable under one master flag (the sequenced 460i-
  successor falsifier discriminates which lifts):
    (1) URGENCY (Thura/Cisek): per natural-commit tick,
        decisiveness_scale = 1 + gap_entry_sensitivity * gap_norm_at_entry;
        urgency += urgency_rate * decisiveness_scale; fire at >= release_bound.
        gap_norm_at_entry in [0,1] = normalised top-F decisiveness at entry
        (1 = decisive F-gap = the commit that monopolises). The gap-scaling is
        LOAD-BEARING: an F-decisive commit accrues urgency FASTER -> the
        strongest-F (most monopolising) holds are shortened most -> attacks
        F-dominance in the duration domain AND folds in the "gap-scaled
        commit-entry threshold" impl_hint candidate. gap_entry_sensitivity=0
        reduces to a flat fixed-rate timeout (the contrasted "fixed refractory"
        control the D1 falsifier compares against).
    (2) ACTION-EXTENT (Jin): release when the committed trajectory's executed
        action sequence completes (_committed_step_idx >= horizon) rather than
        repeating the last action indefinitely. Renders "maintenance
        co-extensive with the executing action" + the "natural-commit run-length
        cap" candidate as a BEHAVIOURALLY-grounded cap (the trajectory horizon),
        NOT a tuned constant. Fires regardless of urgency on sequence completion.
  DISTINCT from siblings: MECH-342 fires on DEGRADED readiness -> silent on the
  healthy-but-prolonged decisive commit that actually monopolises (why strong
  seeds hold ~2400 steps); SD-034 Leg-B refractory (MECH-446) holds the latch
  DOWN post-closure (this shortens the natural commit's occupancy UP, it does
  NOT install a refractory); MECH-091 is z_harm urgency (this is duration
  urgency, no harm input); ARC-028/MECH-105 releases on a HIGH completion signal
  (this releases on held-duration urgency / sequence completion regardless of
  plan quality).
  Config (REEConfig + from_dims, all no-op default -> bit-identical OFF):
  use_natural_commit_urgency_release (False, master) +
  natural_commit_release_urgency_mode (True) +
  natural_commit_release_action_extent_mode (True) + natural_commit_urgency_rate
  (0.01) + natural_commit_urgency_release_bound (1.0) + natural_commit_urgency_cap
  (1.5) + natural_commit_gap_entry_sensitivity (1.0, the load-bearing gap-scale;
  0.0 = flat control) + natural_commit_urgency_onset_ticks (0).
  Agent wiring (ree_core/agent.py): instantiate self.natural_commit_urgency when
  the master flag is on; ARM at the bistable elevate site on a fresh NATURAL
  commit (result.committed) via note_commit_entry(gap_norm) (gap_norm computed
  from result.scores; a purely closure-coupled elevation is NOT armed -> its
  occupancy stays governed by SD-034); TICK at the MECH-342 release region
  (committed_run_length=beta_gate.committed_run_length, action_sequence_complete
  =_committed_step_idx>=horizon) -> on fire release the latch (beta_gate.release()
  + _committed_step_idx=0 + _committed_anchor_keys=None + e3._committed_trajectory
  =None, mirroring the MECH-342 block); reset() per episode. NOT touched:
  e3_selector.py (clean separation from MECH-448), beta_gate.py (reuses the
  existing committed_run_length property), claims.yaml (PROMOTES NOTHING).
  Diagnostics (get_state): ncur_last_occupancy_at_release (latch-occupancy length
  at release) / ncur_n_urgency_releases + ncur_n_action_extent_releases +
  ncur_n_releases_total (release-event counts) / urgency + last_decisiveness_scale
  (graded-release magnitude) / gap_norm_at_entry / natural_commit_armed /
  ncur_n_simulation_skips.
  MECH-094: tick(simulation_mode=True) is a no-op (a replay/DMN tick must not
  abort a committed motor program); matches the SD-035/MECH-279/MECH-313/
  MECH-320/MECH-342 pattern. Phased training: N/A (pure-arithmetic regulator).
  Backward compatible: use_natural_commit_urgency_release=False by default ->
  agent.natural_commit_urgency is None; arm site + release block skipped ->
  bit-identical. 10/10 new contracts (tests/contracts/test_natural_commit_urgency.py:
  C1 config defaults + master-off None / C2 gap-scaled rate LOAD-BEARING
  [decisive releases sooner than near-tie; sensitivity=0 gap-independent] /
  C3 action-extent fires on sequence completion / C4 unarmed no-op /
  C5 MECH-094 sim no-op / C6 config validation / C7 agent release wiring +
  bounded occupancy [OFF holds ~12-tick run; ON releases, occupancy << OFF] /
  C8 agent bit-identical OFF [ON-inert == OFF action stream] / C9 agent arm-site
  [is_armed + gap_norm captured] / C10 release-only safety) + 7/7 preflight +
  full contract suite 1169 passed (the 2 test_runner_fail_branch failures are
  the documented pre-existing local-git-env runner-conflict flakes, CONFIRMED
  failing identically on a clean stash). Activation smoke = the C7/C8 agent
  contracts (occupancy bounded ON vs held OFF; ON-inert bit-identical).
  Validation experiment: NOT queued -- the 460i-successor de-commit falsifier
  (MECH-445/446 on this lever) is the SEQUENCED NEXT step.
  Design doc: REE_assembly/docs/architecture/natural_commit_occupancy_release.md
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
  (f_dominance_conversion_ceiling rung 6 implementation_log).
  See SD-034 / MECH-090 / MECH-342 / MECH-445 / MECH-446 (the commit/release-
  duration cluster this serves; all candidate, unweakened), MECH-448 (selection-
  face sibling lever; PARALLEL), MECH-439 (F-dominance root), BG-3 SYNTHESIS
  (targeted_review_commit_release_duration_latch; divergence D1), ARC-106
  (grounding framework), V3-EXQ-460h (the FAIL this addresses), MECH-094 (N/A).
