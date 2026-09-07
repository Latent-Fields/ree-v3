## SD-022 scheduled-injection extension (MECH-302 unblock, 2026-05-30)
- SD-022 scheduled-injection: environment.scheduled_limb_damage_curriculum
  -- IMPLEMENTED 2026-05-30. Module: ree_core/environment/causal_grid_world.py
  (CausalGridWorldV2). Env-side curriculum that periodically injects damage
  directly into self.limb_damage independent of agent action or hazard
  contact, supplying detectable damage->heal trajectories so the MECH-302
  SufferingDerivativeComparator (SD-050) has reliable suffering signals
  regardless of a trained avoidance policy.
  Triggered by failure_autopsy_V3-EXQ-517b_2026-05-30 (REE_assembly/evidence/
  planning/): three FAIL discriminative-pair attempts (V3-EXQ-517 / 517a /
  517b, 2026-05-04 / 04 / 06) ruled out parameter tuning; 517a (window=30 /
  thresh=0.005 / min_norm=0.01, 150 steps/ep) got 0.33 events/seed; 517b
  (same params, 300 steps/ep) got 0.00 events/seed. Trained avoidance policy
  filters out hazard-contact -> heal trajectories the comparator needs.
  Architectural distinction (this entry vs SD-029 scheduled_external_hazard):
    SD-029 relocates a hazard adjacent to the agent -- still requires agent
      contact to inflict damage; depends on the agent stepping onto the
      relocated hazard for the SD-022 contact-driven damage_increment path
      to fire.
    SD-022 scheduled-injection bypasses contact entirely: damage is added
      directly to self.limb_damage via the curriculum gate, regardless of
      what action the agent takes or what cell it occupies. Mimics
      allostatic / externally-imposed tissue insult (clinical analog:
      scheduled chemo cycle, sleep deprivation pulse, chronic-disease flare).
    The two curricula are orthogonal axes: both can be enabled
      simultaneously without interaction.
  Five new __init__ kwargs on CausalGridWorldV2 (env-only -- NOT surfaced
  through REEConfig.from_dims; matches SD-022 / SD-023 / SD-029 / SD-047 /
  SD-048 / SD-049 / SD-054 precedent for env-only SDs):
    scheduled_limb_damage_enabled (bool, default False) -- master switch.
    scheduled_limb_damage_interval (int, default 50) -- period in steps
      between injection attempts.
    scheduled_limb_damage_prob (float, default 0.5) -- probability per attempt.
    scheduled_limb_damage_magnitude (float, default 0.4) -- damage added to
      selected limb on fire; clamped to [0, 1] by the same min(1.0, ...) bound
      the existing SD-022 contact-damage path uses.
    scheduled_limb_damage_limb_selection (str, default "random") -- "random"
      (one limb uniformly at random) or "all" (uniform across all 4 limbs).
  Defaults rationale: at magnitude=0.4 and SD-022 default heal_rate=0.002/step,
  one injection takes ~200 steps to heal -- comfortably within a 300-step
  episode. interval=50 x prob=0.5 yields ~3 injections per 300-step episode
  in expectation -> multiple detectable healing trajectories per seed-episode.
  Preconditions (constructor raises ValueError; loud-not-silent matching the
  MECH-269b / MECH-293 / SD-049 precedent):
    scheduled_limb_damage_enabled=True requires limb_damage_enabled=True
      (without the SD-022 substrate active, limb_damage stays at zero and the
      curriculum is a silent no-op rather than the user's intent).
    scheduled_limb_damage_limb_selection must be "random" or "all".
  Data flow (env.step()): agent move + SD-022 damage accumulation + healing
  -> [if scheduled_limb_damage_enabled and steps>0 and steps%interval==0
     and rng<prob] -> _inject_scheduled_limb_damage() -> selects limb(s)
     per scheme and adds magnitude with clamp -> increment counter, set info
     tags -> existing SD-029 scheduled_external_hazard block (UNCHANGED;
     orthogonal axis) -> existing SD-054 / SD-049 / SD-047 / SD-048 blocks
     (UNCHANGED).
  Info dict tags (always present; 0 / False when disabled):
    scheduled_limb_damage_enabled, scheduled_limb_damage_injected_this_step,
    scheduled_limb_damage_event_count, scheduled_limb_damage_last_limb_idx
    (-1 when not injected or all-mode), scheduled_limb_damage_last_magnitude
    (0.0 when not injected).
  Per-episode reset: counter + last_step + last_limb_idx + last_magnitude +
  injected_this_step all cleared in reset() and in the scripted-eval
  reset_to() bypass path (same pattern as SD-029 / SD-047 / SD-048 per-episode
  resets so scripted-eval comparator harnesses start from zero state).
  Backward compatible: all five kwargs default False / inert; RNG draws gated
  inside `if self.scheduled_limb_damage_enabled:` so existing experiment
  seeds remain bit-identical when disabled. 565/565 contracts + 7/7 preflight
  PASS with master OFF (regression-clean 2026-05-30; was 556 + 9 new MECH-302-
  curriculum contracts in tests/contracts/test_scheduled_limb_damage_curriculum.py).
  Activation smoke (2026-05-30): 5 fires across 67 steps at interval=10/
  prob=1.0/magnitude=0.4 (~1 fire per 13 steps, close to expected 1/10);
  bit-identical OFF vs default-constructed control across 30 steps; random-
  mode adds magnitude to exactly one limb; all-mode adds to all four;
  clamp at 1.0 holds across multiple 0.6 injections; per-episode reset clears
  state; info dict tags always present.
  Biological grounding: clinical relief-completion literature (Tanimoto &
    Heisenberg 2004 Drosophila relief-as-reinforcer; Roesch / Calu /
    Schoenbaum 2007 ventral-striatal relief firing at experimenter-imposed
    aversive-state offset; Andreatta 2012 Learn Mem fear/relief double
    dissociation; Navratilova 2012 PNAS pain-relief activates VTA-DA +
    NAc-shell DA + DA-antagonist-blockable place preference). All cited
    studies use experimenter-imposed discrete onset/offset suffering events
    that the animal cannot avoid -- the curriculum reproduces that
    prerequisite at the env layer.
  ML/AI engineering note: scheduled-perturbation curricula (Bengio 2009
    automated curriculum learning) carry a standard failure mode where
    schedule predictability creates a degenerate solution (agent learns to
    time the perturbation rather than handle its consequence). Mitigation
    is built in: stochastic gate (prob<1.0 default) + random limb selection
    ensures the agent cannot predict exact timing or location.
  MECH-094: not applicable (env observation stream, not replay / simulation
    content). Same scope decision as SD-022 / SD-029 / SD-047 / SD-048 /
    SD-049 / SD-054; env-side substrates do not interact with MECH-094.
  No trainable parameters. No phased training needed. ASCII-safe (no
    print() output added).
  Validation experiment: V3-EXQ-517c queued 2026-05-30 (priority 280; supersedes
    V3-EXQ-517b; same 2-arm discriminative-pair structure as 517a/b with
    curriculum knobs identical on both arms; only use_suffering_derivative_comparator
    and valence_liking_enabled differ between arms). Curriculum config:
    interval=50, prob=0.5, magnitude=0.4, limb_selection=random. Acceptance
    criteria unchanged from 517b: C1 ARM_A events>=3/seed; C2 ARM_A writes>=2/seed;
    C3 ARM_B events==0; C4 ARM_B writes==0; PASS_FRACTION_REQUIRED=2/3.
    PASS clears MECH-302 v3_pending gate AND lifts gate (c) for the MECH-304
    conditioned-inhibition experiment.
  See SD-022 (parent claim -- limb damage substrate this curriculum extends),
    SD-050 / MECH-302 (downstream comparator the curriculum unblocks),
    SD-051 / MECH-304 + SD-052 / MECH-303 (sister safety-prediction
    substrates also blocked by MECH-302 v3_pending),
    SD-029 (parallel curriculum on hazard-relocation axis; orthogonal),
    SD-023 / SD-047 / SD-048 / SD-049 / SD-054 (parallel env-only
    substrate-enrichment kwargs precedent for not surfacing through
    REEConfig.from_dims),
    failure_autopsy_V3-EXQ-517b_2026-05-30 (the autopsy that routed this
    implement-substrate amend session),
    MECH-094 (call-site scoping; not applicable -- env observation stream).
