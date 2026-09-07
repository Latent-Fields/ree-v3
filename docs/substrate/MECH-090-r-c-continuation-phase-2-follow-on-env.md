## MECH-090 R-c continuation Phase-2 follow-on: env-emitted readiness-outcome source (2026-06-02)
- MECH-090 R-c continuation Phase-2: control_plane.beta_gate.commit_entry_readiness_
  conjunction.nav_competence.env_source -- IMPLEMENTED 2026-06-02. The named
  Phase-2 follow-on to the 2026-05-29 landing (the R-c continuation block above:
  "Phase 2 follow-on (separate /implement-substrate pass): wire an env-emitted
  'mech090_readiness_outcome' key reading in agent.sense() so the substrate
  advances readiness automatically without harness involvement"). Closes a real
  gap: the 2026-05-29 pass wired the CONSUMER (commit_readiness.is_above_floor
  AND-composed at both beta_gate.elevate() sites) + the notify_outcome seam, but
  left NO automatic SOURCE -- and no code path anywhere ever called notify_outcome
  (committed_mode_curriculum.py computes nav_competence but does NOT push it via
  the seam; grep-verified zero notify_outcome callers in the repo). So in any
  ecological run the readiness EMA sat pinned at its fail-open initial 1.0 and the
  across-tick nav_competence axis added no signal. This is exactly why V3-EXQ-063a
  (ARC-029 committed-mode revalidation) deliberately left the across-tick axis OFF.
  Two-part wiring (both default-OFF, bit-identical):
    (a) ENV SOURCE -- ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
      New env-only kwarg mech090_readiness_outcome_enabled (default False; NOT
      surfaced through REEConfig.from_dims -- matches SD-022 / SD-029 / SD-047 /
      SD-048 / SD-049 / SD-054 env-only precedent). When True, step() emits
      info["mech090_readiness_outcome"] = clip(1.0 - mean(limb_damage), 0, 1) --
      a [0,1] motor-program-readiness / nav-competence scalar that degrades as
      limb damage accumulates (SD-022 hazard contact + scheduled injection) and
      recovers as damage heals. Requires limb_damage_enabled=True to be
      informative; with limb damage off the outcome is a constant 1.0 (fail-open).
      ABSENT-WHEN-DISABLED (not an always-present sentinel): when the kwarg is
      False the info key is simply not added, so agent.sense reads None and the
      EMA is not advanced (true bit-identical OFF, no RNG draws, no dynamics
      change). Biological reading: Cisek & Kalaska 2010 affordance-preparation
      (can the prepared motor program be executed) + Roesch/Calu/Schoenbaum 2007
      dopaminergic readiness -- the across-tick axis the within-tick score-margin
      gate cannot see.
    (b) AGENT SINK -- ree_core/agent.py REEAgent.sense() gains
      mech090_readiness_outcome: Optional[float] = None. The caller forwards the
      env-emitted info value to the NEXT sense() call (one-tick lag, biologically
      plausible -- readiness reflects the just-experienced motor outcome). When
      self.commit_readiness is not None AND the value is non-None, sense() calls
      commit_readiness.update(outcome_signal=..., simulation_mode=hypothesis_tag)
      near the end of the method, advancing the readiness EMA automatically.
      No-op when commit_readiness is None (master flags off) OR the value is None
      (key absent; CommitReadiness.update's None-sentinel returns readiness
      unchanged).
  No new module, no new config dataclass field on the agent side, no
  REEConfig.from_dims change (the env kwarg is env-only; the agent param is a
  sense() argument). The CommitReadiness module itself is UNCHANGED -- its update()
  already supported the None-sentinel and simulation_mode gate; this pass only
  wires a real source into it.
  MECH-094: sense() passes simulation_mode=hypothesis_tag into update(); a
  simulation/replay latent does not advance the EMA (the same standard pattern as
  every other regulator). Env step() is a waking observation stream.
  Phased training: N/A (pure wiring + arithmetic; no learned parameters, no
  gradient flow, no encoder head).
  Backward compatible: 719/719 contracts (700 prior + ~13 new in
  tests/contracts/test_mech090_readiness_outcome_wiring.py + adjacent) + 7/7
  preflight PASS. Env default-OFF emits no key; agent commit_readiness=None when
  master flags off; existing experiment dry-run (V3-EXQ-063a standard path)
  unchanged.
  Activation smoke (2026-06-02): aggressive all-limb scheduled-injection env
  (magnitude 0.5, heal_rate 0.001) drives mean(limb_damage) -> 1.0 ->
  mech090_readiness_outcome -> 0.0 -> readiness EMA -> 0.001 (well below the
  default floor 0.3); the env signal advances the EMA off the fail-open 1.0 and
  healing recovers it. The across-tick nav_competence axis is now exercisable in
  an ecological run for the first time.
  Contract tests: tests/contracts/test_mech090_readiness_outcome_wiring.py (C1
  env-OFF key-absent / C2 env-ON exact 1-mean(damage) + monotone + heal-recovery +
  limb-off-constant-1 / C3 sense(None) no-op / C4 sustained-low drives below floor
  + sustained-high stays above + recover-after-degrade / C5 master-OFF no-op +
  kwarg-omitted==None bit-identical / C6 MECH-094 simulation no-op).
  Validation experiment: V3-EXQ-630 (queued via /queue-experiment) -- the
  ecological across-tick ARC-029 successor to V3-EXQ-063a that exercises this
  source (592b ARM_2 GATED_NAV_COMP_ON + ARM_3 GATED_BOTH_ON arms ecologically),
  measuring that the nav_competence gate suppresses commitment as readiness
  degrades and admits as it recovers.
  Design doc: REE_assembly/docs/architecture/mech_090_commit_entry_predicate.md
  (R-c amendment continued / "Phase-2 env-source follow-on" note 2026-06-02).
  See MECH-090 (parent), commit_readiness.py (the consumer this feeds; UNCHANGED),
  the 2026-05-29 MECH-090 R-c continuation block above (the pass that wired the
  consumer + seam this completes), MECH-342 (release-side sibling; reads the same
  readiness signal via commit_readiness.get_readiness()), SD-022 (limb-damage
  substrate the env outcome reads), MECH-094 (simulation_mode standard pattern),
  ARC-029 (the committed-mode claim the validation EXQ re-evaluates).
