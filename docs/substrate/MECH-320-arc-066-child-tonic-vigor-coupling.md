## MECH-320 (ARC-066 child): Tonic Vigor Coupling Score Bias (mesolimbic-DA-vigor / avg-reward-rate) (2026-05-10)
- MECH-320: action.tonic_vigor_coupling_score_bias -- IMPLEMENTED 2026-05-10.
  Module: ree_core/policy/tonic_vigor.py (TonicVigor + TonicVigorConfig +
  TonicVigorOutput). First child mechanism for ARC-066 (the
  non_deficit_action_drives architectural family). Pure-arithmetic regulator
  (no learned parameters; no nn.Module inheritance); sister to MECH-313
  NoiseFloor and MECH-314 StructuredCuriosity in the ree_core.policy package.
  Adds an additive (or multiplicative-gain falsifiable secondary) bias to E3
  trajectory scoring such that action-trajectories receive a NEGATIVE bias
  (REE lower-is-better favours action) and no-op-trajectories receive a
  POSITIVE bias (penalises passivity), proportional to a slow EWMA over the
  realised E3-score-receipt stream gated by secondary internal-state
  modulators (energy / drive / recent PE). TARGET-FREE: bias applies
  regardless of whether any z_goal is currently active -- closes the
  "well-fed-safe-familiar agent has no positive gradient to act" gap that
  ARC-066 registered.
  Algorithm at the e3.select() call site in REEAgent.select_action():
    update_score_receipt: v_raw <- (1-alpha)*v_raw + alpha*(-score_realised)
      (REE-low-is-better internally negated so v_raw climbs in reward-rich
      regimes; alpha = 1 - 0.5**(1/half_life))
    compute_score_bias: v_t = max(0, v_raw) * gate_energy * gate_drive * gate_pe
      bias[i] = -w_action * v_t   if action_classes[i] != noop_class
                +w_passive * v_t  if action_classes[i] == noop_class    (additive)
      bias[i] = (-w_action * v_t * |scores[i]|) on action /
                (+w_passive * v_t * |scores[i]|) on noop                 (multiplicative)
      bias = clamp(bias, [-bias_scale, +bias_scale])
  Composed AFTER MECH-314 curiosity (orthogonal axis: curiosity rewards
  novelty / uncertainty / LP at the candidate level; vigor biases on the
  action-vs-no-op axis) and BEFORE MECH-313 noise_floor (which lifts softmax
  temperature, not scores -- orthogonal to bias).
  Composition order at e3.select call site:
    dacc_score_bias  +=  lateral_pfc_bias       (SD-033a)
                      +  ofc_bias               (SD-033b)
                      +  mech295_liking_bias    (MECH-295)
                      +  curiosity_bias         (MECH-314)
                      +  tonic_vigor_bias       (MECH-320; -action / +noop)
    [then MECH-313 lifts effective_temperature for the softmax]
  Config: REEConfig.use_tonic_vigor (default False; bit-identical OFF) +
    tonic_vigor_half_life (100.0; long-window EWMA per R4 verdict --
    Niv 2007 long-run avg-reward-rate, NOT short-window) + tonic_vigor_w_action
    (0.1) + tonic_vigor_w_passive (0.1) + tonic_vigor_bias_scale (0.1; mirrors
    lateral_pfc / curiosity bias_scale so MECH-320 cannot dominate the
    score-bias chain at extreme reward histories) + tonic_vigor_gate_energy_min
    (0.2) + tonic_vigor_gate_drive_max (0.7) + tonic_vigor_gate_pe_max (1.0) +
    tonic_vigor_form ("additive" | "multiplicative"; validated at construction)
    + tonic_vigor_noop_class (0; matches MECH-279 PAG freeze-gate convention).
    All wired through REEConfig.from_dims().
  Agent wiring (REEAgent.__init__): instantiate TonicVigor when
    use_tonic_vigor=True (parallel to self.noise_floor / self.curiosity).
    REEAgent.select_action() reads tonic_vigor.compute_score_bias(...)
    AFTER MECH-314 curiosity block and composes additively into
    dacc_score_bias; AFTER e3.select() returns, feeds the SELECTED
    candidate's E3 score back into tonic_vigor.update_score_receipt(...) to
    advance the EWMA for the next tick. reset() clears EWMA + diagnostic
    counters.
  Energy proxy: 1 - effective_drive_level (post-pACC). Drive: post-pACC
    effective_drive_level (matches AIC / PCC / SalienceCoordinator reads).
    Recent PE: e3._running_variance (same signal MECH-314c learning-progress
    consumes).
  MECH-094: simulation_mode=True on either compute_score_bias or
    update_score_receipt returns zeros (or skips state advance) and
    increments only the simulation-skip counter. Match the SD-035 / MECH-279 /
    gated_policy / MECH-313 / MECH-314 simulation_mode pattern.
  Lit-pull verdicts (resolved defaults; see ARC-066 SYNTHESIS at
    REE_assembly/evidence/literature/targeted_review_arc_066_tonic_vigor/
    synthesis.md, lit_conf 0.789, supports-direction, 7 entries):
    R1 -- mesolimbic DA-vigor LOAD-BEARING (Niv 2007 formalism + Salamone &
          Correa 2012 substrate identity + Beierholm 2013 human L-DOPA
          causal test). LC-NE-direction REJECTED -- LC-NE tonic mode is
          one mechanism (noise = MECH-313), per Kane et al. 2017 DREADD
          test by the original Aston-Jones / Cohen authorship group.
    R3 -- ADDITIVE form is primary (Niv 2007 opportunity-cost derivation
          is naturally additive). MULTIPLICATIVE GAIN is the falsifiable
          secondary; both implementable via tonic_vigor_form. The
          discriminative-pair behavioural validation (queued separately)
          chooses which lands as primary.
    R4 -- SLOW EWMA over realised E3-score-receipt is the primary scalar
          (Niv 2007 average-reward-rate formalism + Beierholm 2013
          empirical confirmation). Internal-state proxies (energy, drive,
          recent PE) enter as SECONDARY MODULATORS. The slot's
          registration-time pre-guess of "high energy AND low recent PE
          AND low drive" composite is REFINED to the literature-attributed
          history-average primary + internal-state secondaries.
  Phase-1 instantiation choice (mirrors MECH-313 / MECH-314): a SEPARATE
    TonicVigor module at the e3.select() call site, parallel to NoiseFloor
    and StructuredCuriosity. Whether the policy-layer regulators ultimately
    consolidate into one module is OPEN pending future refactor passes.
  Distinct-from contracts: orthogonal to MECH-313 (noise on choice vs
    direction on score axis); target-free vs target-conditioned MECH-216;
    capacity-keyed vs deficit-keyed inverse SD-012; mathematical complement
    of ARC-068 (opportunity-cost no-op penalty -- collapse-vs-separate
    decision deferred to ARC-068 lit-pull); state-INDEPENDENT vs
    state-dependent recency-keyed MECH-260.
  Backward compatible: use_tonic_vigor=False by default; agent.tonic_vigor
    is None and the entire select_action MECH-320 block is skipped.
    309/309 contracts (281 prior + 28 new MECH-320) + 7/7 preflight PASS
    with master OFF (regression-clean, 2026-05-10).
  Phased training: not applicable (pure-arithmetic regulator; no learned
    parameters; no gradient flow).
  Validation experiment: V3-EXQ-547 substrate-readiness diagnostic (6
    sub-tests UC1-UC6 covering instantiation, master-OFF backward-compat,
    EWMA convergence + half-life + sign convention, select_action wiring
    contract via act_with_split_obs (compute_score_bias pre-select +
    update_score_receipt post-select both fire), MECH-094 simulation gate,
    R3 form discriminability additive-vs-multiplicative). Smoke 6/6 PASS
    2026-05-10 (manifest scrubbed; runner will write the canonical PASS
    manifest from the queued entry). 28 contract tests in
    tests/contracts/test_mech_320_tonic_vigor.py PASS. Behavioural
    validation -- the 3-arm discriminative pair (baseline / additive /
    multiplicative on a well-fed-safe-familiar environment substrate) --
    is queued separately AFTER V3-EXQ-547 PASSes via the runner.
  Design doc: REE_assembly/docs/architecture/non_deficit_action_drives.md
  Lit-pull: REE_assembly/evidence/literature/targeted_review_arc_066_tonic_vigor/
  See MECH-320 (this claim), ARC-066 (parent architectural commitment),
    MECH-313 (sibling LC-NE noise floor; orthogonal axis -- substrate-landed
    earlier the same day; lit-pull R2 verdict establishes LC-NE tonic mode
    is fully covered by MECH-313 with no remaining ARC-066 LC-NE function),
    MECH-314 (sibling structured curiosity bonus; substrate-landed earlier
    the same day; orthogonal axis at the candidate-feature level),
    MECH-216 (target-conditioned predictive wanting; ARC-066 is target-free),
    SD-012 (deficit-keyed homeostatic drive; ARC-066 is capacity-keyed inverse),
    MECH-260 (state-dependent dACC anti-recency; orthogonal),
    MECH-295 (sibling score-bias contributor composed before MECH-320),
    ARC-068 (opportunity_cost_no_op_penalty; mathematical complement;
      collapse-vs-separate decision deferred to ARC-068 lit-pull),
    SD-037 (broadcast override; deficit-recruited; ARC-066 is surplus-
      recruited; opposite corners of state space),
    MECH-094 (simulation_mode argument; call-site scoping for waking-only
      effects).
