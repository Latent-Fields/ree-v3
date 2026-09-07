## SD Design Decisions Implemented
- SD-QUEUE-SEED-ENFORCEMENT: validate_queue.seed_enforcement_lint -- IMPLEMENTED 2026-08-13.
  experiment_queue.json's "seeds": N field was consumed ONLY by experiment_runner.py's
  _run_axis_count for progress-bar/ETA denominators -- NEVER translated into a --seeds CLI
  arg, so a driver's own argparse default was the sole source of truth for how many seeds
  actually ran. Confirmed twice within two days on different drivers: V3-EXQ-912 (queued
  seeds=2, ran seeds=[0] -- n_segments_total=60 not the designed 120, driving a FAIL on an
  under-powered run) and V3-EXQ-920 (queued seeds=8, ran 1 seed, manifest ALSO self-routed a
  flatly incorrect censoring label on top of it). Design:
  REE_assembly/docs/architecture/sd_queue_seed_enforcement.md; source autopsy:
  REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-912-913-fishtank-cluster_2026-08-11.json.
  Module: validate_queue.py -- seed_enforcement_lint / _script_seeds_default_count /
  _module_list_constants / _args_list / _declared_seed_count (validate_queue.py).
  Fires as a blocking ERROR only on the fully-conjunctive, statically-verified case: declared
  seeds > 1, no explicit --seeds override in the item's 'args' (shlex-split the same way
  experiment_runner.run_experiment parses it), AND the script's own --seeds argparse default
  is AST-resolvable to fewer seed values than declared. Resolves inline list/tuple literals,
  module-level NAME references, and list(NAME)/tuple(NAME) wrapping a module-level literal.
  Everything else (default=None -- the largest single corpus pattern; a type=str comma-string
  contract; a computed expression; no --seeds arg at all) is left unresolved and silent --
  fail-soft by design, never a guess.
  Data flow: validate() (already called from BOTH main(), the PreToolUse commit-blocking
  hook, AND experiment_runner.load_queue() at every runner's startup on every machine,
  sys.exit(1) on any error) -> seed_enforcement_lint per item -> blocking error. One check
  therefore gives both commit-time AND runner-startup (fleet-wide, including cloud workers
  that pull main directly and never see the commit hook) enforcement for free.
  Deliberately does NOT synthesize --seeds on the runner side (the task's other proposed
  option): the corpus's --seeds contracts are heterogeneous (literal seed-value lists vs a
  single int count vs a comma-string), so a synthesized 0..N-1 list would risk silently
  overwriting an author's deliberately-chosen seed values (breaking arm-reuse fingerprint
  matching) or guessing values the author never specified -- a loud refusal is safer.
  Backward compatible: fires only on the narrow conjunctive case; every ambiguous shape is
  silent (swept the resolver over the full 1351-script experiments/ corpus with zero crashes,
  ~11% confidently resolved, rest deferred).
  Phased training required: no. MECH-094: N/A (no simulation/replay).
  Validation: tests/contracts/test_validate_queue_seed_enforcement.py (17 tests, retroactively
  confirmed against the real V3-EXQ-912/920 scripts on disk).
- SD-E3-SCORER-COMPLETION: e3_selector.untrained_fallback_scorers -- IMPLEMENTED 2026-08-09.
  Two of E3TrajectorySelector.score_trajectory's cost sub-components read UNTRAINED
  nn.Sequential heads -- reality_scorer (the "viability" term in compute_reality_cost / F,
  present on EVERY scoring path) and harm_cost_fallback_scorer (the subtracted term in
  compute_harm_cost_fallback / the default fallback M path). Exhaustive grep confirmed NEITHER
  head is touched by any loss anywhere in ree_core, so they added random-init noise to every
  trajectory score, in both conditions, on the LIVE selection path (select() -> score_trajectory
  -> REEAgent.select_action) -- contaminating MECH-022's V3-EXQ-190a test (C3 collapsed, sign
  flip on repeated seed 123). Diagnosis: REE_assembly
  evidence/planning/failure_autopsy_V3-EXQ-190a_2026-08-09.md; design:
  REE_assembly/docs/architecture/sd_e3_scorer_completion.md.
  Module: E3TrajectorySelector.compute_reality_cost / compute_harm_cost_fallback (e3_selector.py).
  Config: E3Config.e3_include_untrained_fallback_scorers.
  DEFAULT IS THE FIX (False), NOT A NO-OP -- deliberately inverted from the usual E3Config
  "False = bit-identical legacy" convention, because the legacy behaviour IS the defect. With
  False, compute_reality_cost returns the parameter-free coherence (smoothness) proxy alone and
  compute_harm_cost_fallback returns the TRAINED harm_eval_head sum alone; the nn.Sequential
  heads stay instantiated so state_dict/checkpoint keys are unchanged (only their CONTRIBUTION
  is gated -- the same idiom benefit_eval_head already uses). Set True per-arm
  (cfg.e3.e3_include_untrained_fallback_scorers = True) ONLY to reproduce a pre-fix run
  bit-identically; NOT wired through REEConfig.from_dims() (follows f_weight's precedent).
  Data flow: score_trajectory -> compute_reality_cost (coherence only) + compute_harm_cost_fallback
  (harm_eval_head sum only) -> J -> select() committed argmin.
  Backward compatible: NO (default changes scoring on the fallback path) -- this is the intended
  contamination fix, not an additive feature. Legacy behaviour available behind the flag.
  Phased training required: no (nothing new is trained). MECH-094: N/A (no simulation/replay).
  Validation: tests/contracts/test_e3_scorer_completion.py (5 tests; deterministic property, no
  stochastic experiment). MECH-022 full retest is governance-gated (pending_retest_after_substrate)
  and must preserve eval condition-dependence (raise eval_episodes, NO nav_bias in the eval loop).
  See MECH-022, ARC-007, ARC-016.
- SD-084: e3.persistent_committed_program_handle -- IMPLEMENTED 2026-07-29.
  Makes MECH-321's R4 MID-EXECUTION hook REACHABLE. That hook (agent.py select_action)
  gates on a committed trajectory surviving from a PREVIOUS tick, but the LAST statement of
  E3Selector.post_action_update is an unconditional `self._committed_trajectory = None` and
  every driver calls update_residue each step -- so the hook had NEVER executed in any
  experiment (V3-EXQ-830: decomp_n_evaluated_midexec = 0 in all 10 cells against
  decomp_n_evaluated_precommit 1862-2618). Diagnosis: REE_assembly
  evidence/planning/failure_autopsy_V3-EXQ-830_2026-07-29.md section 3.
  Module: E3Selector._persistent_committed_trajectory (e3_selector.py), mirroring the
  existing _closure_committed_trajectory precedent built for the same reason.
  Config: REEConfig.use_persistent_committed_program_handle (default False; set True to
  enable). THREE wiring sites (field + from_dims signature + from_dims assignment); no
  sub-config mirror -- the consumer is REEAgent, which reads top-level REEConfig.
  Data flow: E3Selector.select `if committed:` -> _persistent_committed_trajectory
  (SET, unconditional -- output-neutral, per the _fp_*_world_endpoint precedent) ->
  survives post_action_update -> agent.py gate (4) reads the UNION -> PolicyDecomposition
  .evaluate() on the remaining unexecuted actions.
  LIVENESS IS AN INVARIANT, NOT A SITE LIST: select_action reaps the handle whenever
  beta is not elevated. agent.py has TEN beta_gate.release() sites and only FIVE clear
  _committed_trajectory (the rest are backstopped by post_action_update, a backstop this
  handle removes), so clearing only at the documented de-commit sites would strand a stale
  trajectory and fire the hook against an already-released program. The six explicit
  clears are kept for same-tick observability and are redundant with the reap.
  Backward compatible: disabled by default; OFF path is bit-identical (two attribute
  writes nothing reads) and reproduces the V3-EXQ-830 structural zero exactly.
  NOT A PURE DIAGNOSTIC WHEN ON: a reachable mid-execution hook can newly reach
  boundary.fired (feeding MECH-321's R1 OR trigger), and a mid-execution fire RELEASES
  THE COMMIT LATCH, ABORTING THE REMAINING MACRO -- action sequences change.
  MECH-094: N/A (waking path, no replay/simulation/memory-write surface).
  Phased training: N/A (no encoder head, no new parameters).
  Also makes use_decomposition_scale_resolved_probe_midexec (aaf5caac26) REACHABLE rather
  than merely unexercised -- it must not be read as validated by V3-EXQ-830.
  Contract: tests/contracts/test_mech321_midexec_natural_reachability.py -- asserts the
  hook fires in a REAL rollout with NOTHING injected, plus an OFF negative control that
  still commits multi-action programs (so its zero is the teardown, not an empty arm).
  This is the assertion test_mech321_scale_resolved_boundary.py could not make: it reaches
  the hook only by setting fake_traj.metadata, _committed_step_idx=1 and beta_gate.elevate()
  directly -- reachability-in-principle, which is exactly why the defect survived.
  VALIDATION EXPERIMENT: V3-EXQ-839 (experiments/v3_exq_839_sd084_midexec_reachability.py,
  queued 2026-07-29, b1a896fb1e). DIAGNOSTIC, claim_ids=[] -- validates the BUILD, weights
  no claim. Acceptance criterion verbatim from the autopsy's failure_record_entry.target:
  decomp_n_evaluated_midexec > 0 on a standard select_action -> update_residue loop with no
  hand-injected preconditions. 2 arms x 6 seeds in TWO pre-registered tiers -- ATTRIBUTABLE
  (3,47,71,89) commit multi-action programs and fire the hook when ON; NEGATIVE CONTROL
  (23,53) commit none while pre-commit decomposition stays live, so the DV must be zero in
  BOTH arms and the arms bit-identical. The control tier is the point: it makes a zero DV
  ATTRIBUTABLE, so "SD-084 failed" can never be confused with "this seed never committed a
  multi-action program" (which self-routes substrate_not_ready_requeue, never a verdict).
  Baseline module experiments/_lib/baselines/sd084_midexec_reachability.py; OFF cells minted
  reuse-eligible (include_driver_script_in_hash=False).
  TWO THINGS AUTHORING THAT EXPERIMENT ESTABLISHED, worth knowing before touching this area:
  (1) SEED-DEPENDENCE IS REAL AND MEASURED. Gate (6) is len(remaining)>1, so the hook only
      fires after a MULTI-ACTION commit, and E3 commits a multi-action ARC-071 chunk only when
      it beats the CEM-optimised candidates on raw score -- there is NO chunk-selection-bias
      knob. --seed-scan over ten seeds (darwin-arm64, seeded env, 2ep x 60 steps): OFF midexec
      = 0 on ALL TEN including seed 47 which commits 30 multi-action programs; ON midexec
      14-29 on exactly the four attributable seeds. Seed 47 is the cleanest cell -- OFF and ON
      identical in multi-action commits (30), total commits (72) and precommit (343), so the
      only difference is whether the hook could SEE the program.
  (2) THE CONTRACT'S ENV IS UNSEEDED, AND reset_all_rng(seed) DOES NOT FIX THAT. The contract
      builds CausalGridWorldV2() with seed=None, which is fine for its existential
      anti-vacuity assertions but NOT for any paired/quantitative measurement. Same-arm
      replicate (OFF twice at one seed): 3-11 of 24 actions differed, precommit 40 vs 84. With
      seed= passed to the env: 0/24 and identical counts. So pass an explicit env seed in any
      experiment here -- otherwise a paired behavioural delta measures RNG noise, and an
      arm-fingerprint emitted reuse_eligible is a LIE (it promises purity in
      substrate+config+seed).
  See MECH-321, ARC-070, ARC-071, MECH-288 (no claim status changed).
- SD-004: E2 action objects; HippocampalModule navigates action-object space O
- SD-005: z_gamma split into z_self (E2 domain) + z_world (E3/Hippocampal/ResidueField domain)
- SD-006: Asynchronous multi-rate loop execution (phase 1: time-multiplexed)
- SD-083: consolidation.offline_policy_window -- IMPLEMENTED 2026-07-29. TESTBED locus (the
  mech457 bootstrap-explorer stack under experiments/_lib/, alongside its sibling PolicyKLAnchor
  / MECH-475), NOT ree_core/ -- it is a MECH-476 falsifier instrument, not a cognifold faculty.
  Module: experiments/_lib/mech457_offline_consolidation.py (OfflineEWCAnchor +
  consolidate_offline_window). An OFFLINE, trace-selective (Fisher-weighted EWC, Kirkpatrick
  2017), interval-accumulated (capture c(N)=capture_max*(1-exp(-N/tau))), novelty-gated (Moncada
  2007; the lineage's own RNDModule) policy-consolidation window run BETWEEN install_bc_prior and
  the RL refinement. The window BUILDS PROTECTION and does NOT retrain (theta unchanged -> post_bc
  invariant to the interval), which is what makes the INTERVAL axis orthogonal to the DOSE axis.
  Config: BootstrapExplorerConfig.use_offline_consolidation (default False; set True + N>0 +
  offline_ewc_max_coef>0 to enable) -- NOT REEConfig; this lineage has its own config object.
  Data flow: install_bc_prior -> consolidate_offline_window (theta*, Fisher F, capture c) ->
  OfflineEWCAnchor -> train_a2c per-update penalty coef*sum_i F_i (theta_i-theta*_i)^2. Backward
  compatible: disabled by default; existing retention arms (788/789/792/836) unaffected.
  DISTINCT from MECH-475 PolicyKLAnchor (online / global / no-interval) -- that distinction is the
  MECH-476 content (Walker 2003 divergence). MECH-094: N/A (no simulated memory writes). Phased
  training: N/A (raw_view, no encoder head; window takes no optimiser step on theta).
  Contract: tests/contracts/test_sd083_offline_consolidation.py (12/12). Validation: the 12/12
  contract set + V3-EXQ-836b's N=0 control arm (the embedded ON/OFF window ablation). Evidence
  arms queued: V3-EXQ-836b (INTERVAL) + V3-EXQ-836c (NOVELTY).
  SD doc: REE_assembly/docs/architecture/sd_083_offline_policy_consolidation_window.md.
  Cognifold-port follow-on (IF MECH-476 supported): port into the ONE SD-017 sleep loop, unifying
  with MECH-441 novelty + MECH-204. See MECH-476 / MECH-475.
