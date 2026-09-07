## SD-034: Governance Closure Operator (2026-04-20)
- SD-034: governance.closure_operator -- IMPLEMENTED 2026-04-20.
  Module: ree_core/governance/closure_operator.py (ClosureOperator,
  ClosureOperatorConfig, ClosureEvent). First substrate in SD-033
  governance cluster; first consumer of MECH-261 write-gate registry's
  mode-conditioning predicate. Coordinates a five-part "done" token
  emitted at rule-completion events:
    (a) MECH-090 beta_gate.release() -- commitment latch drops.
    (b) MECH-260 dacc.inject_nogo(action_class, count) -- targeted
        No-Go FIFO injection on the just-completed action class
        (semantically distinct from execution record; same mechanism).
    (c) ResidueField.discharge_domain(z_world, factor, radius) --
        rule-domain multiplicative decay on RBF weights; hard 1e-6
        floor preserves the "residue cannot be erased" invariant.
    (d) SalienceCoordinator.update_signal("closure_event", value) --
        re-biases affinity toward internal_planning via registered
        affinity_weights (default internal_planning=0.5).
    (e) dacc.reset_episode_pe() + optional dacc_pe_cap install --
        MECH-268 pe saturation/reset.
  Config: REEConfig.use_closure_operator (bool, default False). Sub-
  knobs: closure_rule_delta_threshold (0.001), closure_stable_ticks
  (3), closure_require_beta_elevated (True), closure_min_sd033a_gate
  (0.5), closure_nogo_injection_count (3), closure_residue_discharge_
  factor (0.5), closure_residue_discharge_radius (1.5),
  closure_signal_value (1.0), closure_reset_pe_ema (True),
  closure_pe_cap_after (None), closure_signal_affinity_internal_
  planning (0.5).
  Completion detector (tick path): rule_state delta < threshold for
  N consecutive ticks AND beta elevated AND current_mode in
  allowed_closure_modes AND sd_033a write_gate >= min. Rule-state
  norm guard prevents firing on unset rule_state. Explicit
  emit_closure() path is the experiment hook (bypass_mode_
  conditioning for controlled ablations).
  Mode conditioning is the falsifiability predicate: if MECH-090 +
  MECH-260 + MECH-094 tuning WITHOUT closure produces the signature
  in follow-up behavioural variants, SD-034 is over-specification.
  ResidueField.discharge_domain API added in same pass: multiplicative
  decay + sign-aware 1e-6 floor + radius-scoped in-domain selection
  via squared distance vs (radius * bandwidth)^2; valence_vecs NOT
  modified (4-component valence preserved so replay prioritisation
  remains faithful). DACCAdaptiveControl extended with dacc_pe_cap,
  inject_nogo(), reset_episode_pe() (distinct from full reset() --
  preserves _action_history where the just-injected No-Go lives).
  Agent wiring: REEAgent.__init__ instantiates closure_operator when
  enabled (requires use_lateral_pfc_analog=True, use_dacc=True;
  salience coordinator optional). select_action() calls tick() after
  action emission with current z_world + argmax action_class +
  operating_mode + sd_033a gate. reset() calls closure_operator.reset().
  register_on_coordinator() wires closure_event into
  salience.config.affinity_weights at init.
  Backward compatible: use_closure_operator=False by default ->
  agent.closure_operator is None; every integration site is a no-op.
  Existing experiments unaffected. Bit-identical with
  closure_signal_affinity_internal_planning=0.0 and the master switch
  off.
  Biological basis: Rich & Shapiro 2009 (OFC sequence-completion
  cells); Collins & Frank 2014 (task-set disengagement); Schuck 2016
  (mPFC task-stage encoding). Five-part signal collocates multiple
  biologically-observed end-of-sequence signatures; EXP-0156 and
  EXP-0162 probe whether the collocation is a single substrate or
  co-occurring independent processes.
  MECH-094: mode-conditioning on operating_mode generalises the
  MECH-094 hypothesis-tag -- internal_replay / offline_consolidation
  modes block closure firing via allowed_closure_modes and via
  sd_033a gate floor (write_gate("sd_033a")=0.05 in internal_replay).
  Validation experiments:
    V3-EXQ-460 (EXP-0156, ocd4 verified-but-not-released) -- landing
      diagnostic (6 sub-tests: backward compat, wiring, beta release,
      No-Go, pe reset, mode conditioning). PASS on smoke.
    V3-EXQ-466 (EXP-0162, ocd4 satisficing / No-Go thresholding) --
      residue-discharge landing diagnostic (5 sub-tests: near
      attenuation, far spared, invariant preserved, closure->discharge
      end-to-end, distant-z spares). PASS on smoke.
  Behavioral variants with full E3 task loop + tolerance-band
  completion env are deferred: they depend on phased rule_state
  training and an env variant not yet on any roadmap item.
  Anchor doc: REE_assembly/evidence/planning/sd033_governance_plan.md
  Source: docs/thoughts/2026-04-20_ocd4.md
  See SD-034, MECH-090, MECH-260, MECH-261, MECH-262, MECH-094,
  MECH-268, SD-032a, SD-033a.
