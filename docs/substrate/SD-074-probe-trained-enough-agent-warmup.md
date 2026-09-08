## SD-074: probe.trained_enough_agent_warmup -- IMPLEMENTED (2026-07-18)
- SD-074: probe.trained_enough_agent_warmup -- IMPLEMENTED 2026-07-18.
  Module: experiments/_lib/probe_warmup.py (WarmupRecipe / WarmupOutcome / warm_agent /
  measure_action_mass / saturation_summary / assert_state_dict_shareable /
  assert_any_informative / reapply_candidate_capture).
  HARNESS-LEVEL ONLY -- nothing under ree_core/ is touched, no config default moves,
  and no existing script imports it, so backward compatibility is TOTAL BY
  CONSTRUCTION (there is no "disabled by default" flag because there is no ree_core
  surface to disable). Training-regime substrate enrichment for the probe harness,
  the same class as the V3-EXQ-603c precedent.
  WHAT IT FIXES: the 2x2 read-only telemetry-probe family measures gain/bias
  regulators (MECH-320 tonic_vigor score-bias, MECH-313 noise_floor temperature) on
  the E3 pre-commit softmax while running an UNTRAINED agent. A regulator that
  MODULATES a distribution is unobservable when that distribution has no dynamic
  range. Measured by V3-EXQ-777a: D_action_mass_mean pinned at ceiling on 7 of 14
  seeds and at floor on 2 more, informative-seed yield 4 of 14 (28.6%), and
  corr(distance of D from saturation, norm_v_score) = 0.884. That caps power at ~51
  informative seeds needed => ~177 raw seeds / ~31 h, which is infeasible. NOT a
  sampling defect: 777a's sample-driven stopping worked perfectly (all 56 cells
  reached 250 fresh E3 selections, zero starved) and the saturation rate barely moved
  from its starved predecessor.
  Data flow: seed -> caller builds agent+env -> warm_agent() [cache lookup; MISS ->
  goal_pipeline_tier1.warmup_train -> store; HIT -> load_state_dict + restore
  non-buffer E3 scalars] -> read-only de-saturation probe via sample_driven_rollout ->
  WarmupOutcome -> consumer loads the per-seed checkpoint into each of its arms.
  Composes landed modules rather than writing a fourth training loop:
  goal_pipeline_tier1.warmup_train, the baselines/maturation_curriculum cache
  discipline (atomic os.replace, key re-verified on load, agent rebuilt on BOTH paths
  for RNG parity), and sample_driven_rollout for the read.
  Gate policy (user-confirmed): RECORD, do not abort. A still-saturated seed comes
  back saturated=True with its realised mean and the CONSUMER decides; only
  assert_any_informative() raises, and only when ZERO seeds de-saturated. Regime
  (user-confirmed): target_env only, num_episodes the swept parameter;
  regime="curriculum" RAISES rather than silently substituting a second env.
  THREE HAZARDS FOUND AND DEFENDED -- read these before touching the module:
  (H1) e3._running_variance is a PLAIN PYTHON FLOAT (ree_core/predictors/e3_selector.py:291),
  NOT a register_buffer, so it is absent from state_dict -- and it feeds commit_variance
  (:2703), directly upstream of the very probs distribution D_action_mass measures.
  Measured: fresh 0.500 vs warmed 0.0092, a ~54x drift a state_dict-only cache would
  have silently discarded, making cache-HIT and cache-MISS agents commit differently.
  Carried explicitly in the blob with an asserted round trip.
  (H2) All four 777a arms VERIFIED to share an identical 194-key state_dict, because
  TonicVigor and NoiseFloor are zero-parameter non-nn.Module (policy/tonic_vigor.py:225,
  policy/noise_floor.py:129). So ONE warmed checkpoint per seed serves all four arms and
  they differ ONLY in regulator scalars at the e3.select() call site -- which is what
  makes the 2x2 clean. assert_state_dict_shareable() ASSERTS this rather than assuming
  it, so a future arm flipping a flag that DOES construct a module fails loudly instead
  of silently leaving a module at random init.
  (H3) The consumer's generate_trajectories capture is an INSTANCE attribute and is NOT
  in state_dict. If it is lost after load_state_dict, every observe() returns None, the
  cell yields zero samples, and the run self-routes to sample_starvation_requeue -- i.e.
  a lost patch MASQUERADES AS A SAMPLING BUG, the exact misdiagnosis this lineage has
  already made twice. Call reapply_candidate_capture() after every load.
  NON-DESTRUCTIVE MEASUREMENT: torch.no_grad() + agent.eval() stop GRADIENTS but not the
  two other ways state moves while merely stepping -- plain-Python accumulators
  (_running_variance drifted 0.001839 -> 0.001855 over a 25-selection read) and
  REGISTERED PLASTICITY/ELIGIBILITY BUFFERS (e3_selector.py:373-462, updated in place
  under no_grad; two agents identical at load diverged by max|dw| = 2.5e-01 after reads
  of different lengths). measure_action_mass snapshots and restores BOTH, so the read
  leaves the warmed agent bit-identical (verified max|dw| = 0.000e+00).
  D_SAT_LOW 0.05 / D_SAT_HIGH 0.95 are deliberately IDENTICAL to V3-EXQ-777a's constants
  (script:246-247) so success is denominated in the same units as the failure record.
  Do NOT retune them to make a warmup look better.
  Phased training required: yes -- consumer becomes P0 (warmup) -> P2 (frozen read-only
  telemetry). No P1 head-training stage, so the EXQ-166b/c/d joint-training collapse mode
  does not arise, but the frozen-measurement boundary is mandatory.
  MECH-094: N/A (waking-only gradient training; no simulation, no replay, no memory write).
  Smoke test PASS 2026-07-18 (backward compat, H1 absence confirmed, H2 cross-arm
  sharing, cache MISS->HIT exact round trip, de-saturation read collects real samples).
  Validation experiment: PENDING (diagnostic, de-saturation only; see below).
  NOTE the re-derive brake: a V3-EXQ-777b or any re-test of MECH-063 sub-claim (i)
  against an UNTRAINED agent is REFUSED (failure_autopsy_MECH-063-777a-779a-cluster_2026-07-18,
  user-confirmed). The 779 lineage / sub-claim (ii) is EXEMPT and NOT blocked on this.
  CROSS-ARM CONTAMINATION REPAIR (SD-PROBE-WARMUP, three layers, two commits).
  Confirmed by failure_autopsy_V3-EXQ-963_2026-08-30: `_warmup_key` excluded the
  caller's arm-conditional flags, so a 2x2 sharing one (seed, recipe, env) hashed
  IDENTICALLY across all four arms; the first arm to run always minted, and
  `_restore_cached_surface` then stamped that mint's regulator presence over every
  later arm's own, already-correct construction. V3-EXQ-963's T0P0
  (use_noise_floor=False) minted `agent.noise_floor = None` onto T1P0/T1P1, whose
  __init__ had just built a real NoiseFloor -- silently zeroing the whole TONIC
  axis (noise_floor_temp_lift_mean 0.0 on all 20 cells, including all 10
  use_noise_floor=True cells). Nothing raised and nothing logged it:
  assert_state_dict_shareable passes because these regulators are zero-parameter
  and non-nn.Module, and the restore's missing-module logging covers only cached
  paths ABSENT here, never the reverse. ree_core is healthy; this is a harness
  defect throughout.
    (a) PREVENTION, cache key -- IMPLEMENTED 2026-08-30 (d614a9c). `_warmup_key`
        accepts `arm_key` and folds it into the hashed payload; schema v2 -> v3, so
        every pre-fix blob MISSes regardless of caller opt-in.
    (b) PREVENTION, restore -- IMPLEMENTED 2026-08-30 (d614a9c). A cached attribute
        is applied only when its TYPE matches the live HIT agent's own value for
        that name (type(None) is a type, so None-vs-None still matches). SYMMETRIC:
        a cached None never overwrites a live regulator, and a cached regulator is
        never installed over a live None. PRIMARY defence -- it holds even for a
        caller that never passes arm_key.
    (c) DETECTION, post-warmup assertion -- IMPLEMENTED 2026-09-01.
        `assert_arm_regulators_live(agent, arm_key)`, raising `ArmRegulatorMismatch`;
        called by warm_agent itself via `assert_arm_regulators: bool = True`
        (keyword-only), AFTER the HIT/MISS branches converge so both paths are
        covered. Checks each `use_<attr>` flag against `agent.<attr>` in BOTH
        directions (flag set + regulator None; flag clear + regulator present).
        Not redundant with (a)/(b): (a) only separates arms for a caller that
        actually passes arm_key -- no driver in the tree does yet -- and (b)
        protects only attributes ALREADY LIVE on the HIT agent, since
        `_restore_cached_surface` Case 1 restores a never-set attribute verbatim
        with no type check by design. (c) reads the agent warmup actually produced
        and asks whether it is still the arm the caller declared, so it fails on
        any route to the corruption, including routes that do not exist yet.
        The failure it stops is a run that COMPLETES and writes a claim-tagged
        manifest that looks valid while the manipulation is silently absent.
        Backward compatible: a STRICT no-op when arm_key is None (no declared
        flags -> nothing checked, nothing raised), which is what every pre-fix
        caller passes -- pinned as a contract, on a deliberately-corrupted agent.
        Adds NO REEConfig field, so the from_dims three-site rule does not apply.
        Skips, rather than raises on, a flag not named `use_<attr>` and a
        `use_<attr>` with no such agent attribute; both are counted in the returned
        report so an empty check cannot read as a pass.
        Contracts: tests/contracts/test_probe_warmup_arm_regulator_assertion.py
        (11) alongside test_probe_warmup_cache_key_restore.py (9, for (a)+(b)).
  STILL OWED, driver-side, NOT done here (belongs to a NEW EXQ letter via
  /queue-experiment -- V3-EXQ-963 is a burned id and its script must not be edited
  in place): the 963-lineage driver must (i) pass arm_key= to warm_agent, (ii)
  extend its `_fresh_regulator` post-warmup reinstall beyond `agent.phasic_burst`
  to `agent.noise_floor` -- that asymmetry is exactly why the phasic axis survived
  the restore and the tonic axis did not -- and (iii) record NoiseFloor.get_state()'s
  n_waking_calls / last_n_simulation_skips, which the substrate was already
  computing and which separate "never called" from "called under simulation_mode".
  See SD-074, SD-PROBE-WARMUP, MECH-063, MECH-320, MECH-313, ARC-066, and
  REE_assembly/docs/architecture/sd_074_probe_warmup_trained_enough_agent.md.
