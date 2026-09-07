## SD-034 AMEND: commitment-closure-control-plane (env-completion hook + de-commit hold) (2026-06-12)
- commitment-closure-control-plane -- IMPLEMENTED 2026-06-12. The behavioural-
  authority amend the SD-034 ClosureOperator lacked on the 603n foraging-competent
  substrate. Routed by the confirmed failure_autopsy_SD-034-closure-cluster_2026-06-12
  (V3-EXQ-460c + V3-EXQ-468c: closure wired at unit level but no behavioural
  authority -- 460c n_closures=0 on 3/3 seeds despite env sequence_completions=2/5/6;
  468c closure-coupled beta release fires but committed_frac cap-pins ~39 both arms).
  TWO no-op-default legs (bit-identical OFF):
    Leg A -- explicit env-completion hook seam (closes 460c n_closures=0):
      ree_core/agent.py new REEAgent.notify_env_completion(action_class, z_world=None,
      bypass_mode_conditioning=False, simulation_mode=False) -> Optional[ClosureEvent].
      When use_closure_env_completion_hook=True AND closure_operator is not None AND
      not simulation, routes the env's transition_type=="sequence_complete" signal
      into closure_operator.emit_closure(action_class, z_world or _current_latent.z_world,
      ...). Returns the ClosureEvent so the harness counts fires / No-Go installs;
      None (no-op) when the flag is off / no operator / simulation. The experiment
      harness calls it post-env.step on a completion tick (the *d retest does the
      call). Fixes the *c-cohort gap: the env emitted completions but nothing routed
      them into emit_closure -- the agent relied solely on the automatic
      rule_state-stability detector, whose conjunction (delta<0.001 x3 + meaningful
      magnitude + sd_033a gate>=0.5 + allowed mode) was unmet on the
      untrained/zeroed rule_bias_head + SP-CEM-perturbed agent.
    Leg B -- de-commitment hold / refractory (closes 468c committed_frac cap-pin):
      ree_core/heartbeat/beta_gate.py gains _refractory_remaining + apply_refractory(n)
      + refractory_remaining property; elevate() is a NO-OP while the window is active
      (increments _n_elevation_refractory_blocked); propagate() decrements the window
      once per tick (propagate runs every select_action); reset() + get_state()
      extended (sd034_refractory_remaining / sd034_n_elevation_refractory_blocked).
      ree_core/governance/closure_operator.py: ClosureOperatorConfig.decommit_hold_ticks
      (default 0); _fire() installs beta_gate.apply_refractory(decommit_hold_ticks) on
      any closure fire (recorded as ClosureEvent.decommit_refractory_applied) so the
      closure-driven release survives >1 tick -> measurable latch-occupancy drop
      instead of immediate re-commit.
    Leg C (experiment-side, NOT substrate): the *d retests set the landed GAP-D
      lateral_pfc_train_rule_bias_head so the automatic detector has a
      magnitude-bearing rule_state, gate readiness on n_closures>0 reachable, and
      read de-commitment on a non-cap-pinned statistic (post-completion uncommitted
      fraction / committed-run-length delta).
  Config (REEConfig + from_dims; both no-op default, bit-identical OFF):
    use_closure_env_completion_hook (False), closure_decommit_hold_ticks (0). The
    decommit_hold_ticks is wired into the ClosureOperatorConfig build in
    REEAgent.__init__ via getattr fallback (absent flat attr -> bit-identical).
  Backward compatible: both default no-op -> notify_env_completion returns None,
    _fire never calls apply_refractory, elevate() bit-identical. 1014 contracts
    (1008 prior + 6 new in tests/contracts/test_sd034_decommit_hold_and_env_hook.py:
    C1 BetaGate refractory default-OFF bit-identical / C2 refractory blocks-then-
    expires + max-not-truncate + n<=0 no-op / C3 ClosureOperator installs refractory
    on fire + default-0 no-op / C4 agent hook OFF no-op / C5 hook ON fires emit_closure
    + installs hold + blocks re-commit / C6 MECH-094 simulation gate) + 7/7 preflight
    PASS. v3_exq_460c --dry-run unchanged (hook OFF -> reproduces the prior FAIL
    signature). Agent smoke 2026-06-12: hook ON routes a completion -> n_closures 0->1
    + nogo_pushed=3 (the 460c n_closures=0/nogo=0 defect closed on the positive
    control) + refractory_applied=5 blocks re-commit (the 468c immediate-re-commit
    defect addressed).
  Phased training: N/A (pure wiring + arithmetic; no learned parameters). MECH-094:
    the env hook is waking-only + simulation_mode/hypothesis_tag-gated (a replay/DMN
    completion cannot emit a waking done-token); the refractory is a control-state
    transition, not memory content. Evidence-staleness (Step 8.5): NOT triggered --
    no-op-default; every existing experiment uses the defaults (hook off, hold 0), so
    no dependent claim's measured mechanism changed. KEEP all evidence
    (SD-034/MECH-260/MECH-268/MECH-090).
  Validation experiments: V3-EXQ-460d (supersedes 460c) + V3-EXQ-468d (supersedes
    468c), queued via /queue-experiment. Retest gate (per the autopsy failure_record):
    n_closures>=1 reachable on the positive control AND nogo_installed>=1 on >=2/3
    seeds after the env->emit_closure wiring; and a non-cap-pinned de-commitment DV
    showing ON<OFF on >=2/3 seeds. substrate_queue commitment-closure-control-plane
    ready stays FALSE until both clear.
  GOVERNANCE: SD-034 / MECH-260 / MECH-268 / MECH-090 NEITHER validated NOR weakened;
    SD-034 provisional holds, MECH-261 stable untouched, MECH-260 stays candidate.
    claims.yaml carries only an implementation_note (no flag/confidence change).
  Design doc: REE_assembly/docs/architecture/sd_034_governance_closure_operator.md
    (commitment-closure-control-plane amend section). Autopsy:
    REE_assembly/evidence/planning/failure_autopsy_SD-034-closure-cluster_2026-06-12.{md,json}.
  Substrate_queue: REE_assembly/evidence/planning/substrate_queue.json
    (commitment-closure-control-plane).
  See SD-034 (parent), MECH-090 (BetaGate -- the refractory host), MECH-260 (No-Go;
    strictly downstream of a closure fire -- its 460c zero was a positive-negative),
    MECH-268 (dACC PE saturation; coupled in 468c), MECH-261 (mode-conditioning;
    stable, not exercised), SD-033a GAP-D lateral_pfc_train_rule_bias_head (the *d
    rule_state lever), V3-EXQ-460c/468c (the FAILs this amend addresses), V3-EXQ-460d/468d
    (validation), MECH-094 (call-site scoping + simulation gate).
