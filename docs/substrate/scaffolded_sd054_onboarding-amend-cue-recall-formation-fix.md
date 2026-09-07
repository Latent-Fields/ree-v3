## scaffolded_sd054_onboarding AMEND: cue-recall FORMATION fix + diagnostics (2026-06-04b)
- scaffolded_sd054_onboarding cue-recall formation amend -- IMPLEMENTED 2026-06-04.
  Module: experiments/scaffolded_sd054_onboarding.py (harness layer; NO ree_core /
  goal.py / claims.yaml change). Routed by the V3-EXQ-638 cue-silent autopsy
  (manifest v3_exq_638_scaffold_cue_recall_contact_ablation_20260604T142524Z_v3.json:
  C1_cue_fires_on=false, C3_contact_lift=false, FAIL/non_contributory).
  ROOT CAUSE (code-confirmed): the IncentiveTokenBank is EMPTY entering P1/P2, so
  agent.cue_recall_wanting (agent.py:5134) returns 0 at `k not in bank._base_value`
  -> cue_fires=0. Stage-0 forced feeding DOES pass resource_type into update_z_goal,
  but rt=_contacted_resource_type(obs) is almost always None because forced feeding
  is decoupled from standing on a typed cell, so the L2 bank.update bind (gated
  resource_type>0) is never reached. Tokens otherwise only bind on real P1/P2 typed
  contact -- the very GAP-2 failure-to-thrive the cue was meant to bootstrap
  (chicken-and-egg). Compounded by a bare `except: pass` in _maybe_cue_recall that
  made cue_fires=0 undiagnosable.
  TWO no-op-default changes (both bit-identical OFF):
    (A) INSTRUMENTATION. _maybe_cue_recall gains an optional cue_diag accumulator
      (_new_cue_diag()): every non-fire is attributed to a reason
      (no_token / resource_field_absent / proximity_below_threshold / bank_none /
      amp_zero_or_zobject_none / exception:<Type>) and the substrate quantities are
      recorded (n_external_cues_seen, n_cue_recall_attempts/fires, n_token_matches,
      best_prox_peak, drive_peak, token_bank_size, matched_token_strength_peak;
      n_interoceptive_need_cues + n_joint_cues reserved 0 for the next layer). The
      `except: pass` is replaced with an `exception:<Type>` reason -- thrown errors
      are now VISIBLE (still best-effort; never breaks the episode loop). Surfaced
      on P1OnboardingResult.cue_diag + P2OnboardingMetrics.cue_diag +
      Stage0NurseryResult.token_bank_size_end.
    (B) FORMATION FIX. New flag scaffold_stage0_bind_incentive_token (default False).
      When on (and the bridge is enabled), Stage-0 forced feeding binds the token to
      the STRONGEST-PERCEIVED resource type each step (new shared helper
      _strongest_perceived_type, factored from the cue logic so formation and recall
      use IDENTICAL perception -- user-confirmed binding choice 2026-06-04) instead of
      the near-always-None contacted type. Net: the bank is non-empty entering P1/P2,
      so the wild cue can match a token.
  Backward compatible: scaffold_stage0_bind_incentive_token default False -> Stage-0
  rt = _contacted_resource_type, bit-identical; cue_diag empty when no accumulator
  passed. 62/62 scaffold contracts (55 prior + 7 new C9) + 7/7 preflight PASS.
  Activation smoke 2026-06-04 (bridge + bind ON, real REEAgent, 2-ep Stage-0 -> 2-ep
  P1): Stage-0 token_bank_size_end=2 (was 0); P1 cue fires 34x (n_token_matches=34,
  empty nonfire reasons) -- cue_fires=0 -> 34 purely from populating the bank.
  Notable layer-2 signal: P1 drive_peak=0.037 (agent well-fed) -- the cue fires but
  with modest amplitude, which is where interoceptive need-gating (the NEXT pass,
  NOT this one) becomes the right test.
  Phased training: N/A (harness wiring + perception; no learned parameters). MECH-094:
  cue_recall_wanting / update_z_goal carry simulation_mode; the scheduler is a waking
  training stream.
  Contracts: tests/contracts/test_scaffolded_sd054_onboarding.py C9 cont.
  (stage0_bind flag default-noop; _strongest_perceived_type helper; cue_diag no_token
  reason -- the 638 root cause as a unit; resource_field_absent reason; fire+strength
  recorded; exception path attributed not swallowed; Stage-0 binding populates bank).
  Validation experiment: V3-EXQ-638a re-issue (638 with scaffold_stage0_bind_incentive_token
  =True) queued via /queue-experiment -- confirms C1 cue fires + bank populated +
  whether C3 contact lifts in the full 3-seed run. The interoceptive need-gating layer
  + V3-EXQ-638b (OFF / EXTERNAL_ONLY / INTEROCEPTIVE+EXTERNAL arms) is a SEPARATE later
  pass, pending 638a confirming the bank-empty reason dominated.
  See scaffolded_sd054_onboarding cue-recall bridge entry above (the 638 integration
  this fixes), SD-057 / MECH-347 (L6 cue-recall claim; ree_core substrate unchanged),
  goal.py IncentiveTokenBank (the bank this populates), MECH-295 (downstream approach
  bridge), goal_pipeline:GAP-2 (foraging-contact ceiling), V3-EXQ-638 (the cue-silent
  FAIL), V3-EXQ-638a (validation), MECH-094 (N/A).
