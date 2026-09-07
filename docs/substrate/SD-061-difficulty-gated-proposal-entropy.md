## SD-061: difficulty-gated proposal-entropy regulator (stuck-state detector + transient CEM proposal-widening; MECH-343 blocker part 2 / Q-056) (2026-06-19)
- SD-061: control_plane.difficulty_gated_proposal_entropy -- IMPLEMENTED 2026-06-19
  (substrate; MECH-343 stays candidate / substrate_conditional / v3_pending -- this
  PROMOTES NOTHING, it builds the missing substrate the mechanism is blocked on). Routed
  by the Q-054/Q-055/Q-056 buildability triage (REE_assembly/evidence/planning/
  q054_q055_q056_buildability_triage_2026-06-19.md): MECH-343's evidence_quality_note
  names two blockers -- (1) modulatory-bias-selection-authority (NOW implemented, 569i
  top-k) and (2) "a difficulty-gated proposal-entropy regulator (stuck-state detector +
  transient CEM temperature/candidate-count gain + decay) not yet designed." SD-061 is (2).
  Two coupled no-op-default modules (OFF = bit-identical):
    (1) ree_core/cingulate/stuck_state_detector.py (StuckStateDetector +
      StuckStateDetectorConfig) -- integrates goal-progress stall (GoalState.goal_proximity
      window) + E3 first-action score margin + committed-action-class lock-in (window) +
      dACC choice_difficulty (inverted: small EV spread = hard) into a graded stuck_score
      in [0,1] + binary is_stuck, GUARDED by goal salience (no goal -> 0; the
      stuck-WITH-goal distinction). Present-axis deficits combine by mean|max; asymmetric
      EMA (ema_alpha_rise >> ema_alpha_fall) -> fast rise, slow decay (the MECH-343
      "entropy narrows once a workable candidate is found" hysteresis).
    (2) ree_core/policy/difficulty_gated_proposal_entropy.py (DifficultyGatedProposalEntropy
      + Config) -- maps stuck_score to a transient PROPOSAL-layer gain:
      extra_candidates = round(candidate_widen_max * s); temperature_gain = 1 +
      temperature_gain_max * s. Identity at s=0.
  Pure-arithmetic regulators (no nn.Module, no learned params, no gradient flow); sibling
  to MECH-313 NoiseFloor / MECH-320 TonicVigor / MECH-342 CommitMaintenanceRelease.
  Wiring (ree_core/agent.py): both built in __init__ when
  use_difficulty_gated_proposal_entropy=True (else None); self._last_stuck_score lag seam.
  _e3_tick applies the gain to HippocampalModule.propose_trajectories (num_candidates +=
  extra; differentiable_cem_temperature *= gain, transient, restored in finally).
  select_action updates the detector EVERY tick (after the maintenance_release block, not
  gated on beta elevation) from e3.last_scores margin + _dacc_last_bundle choice_difficulty
  + goal_state proximity/norm + e3._committed_trajectory first-action class ->
  _last_stuck_score (one-tick lag the next _e3_tick reads). reset() clears both + the lag.
  Scoring / commitment (MECH-090/342) / selection authority (569i top-k / MECH-341) are
  UNTOUCHED -- a hard problem widens proposals, not behaviour.
  Config (REEConfig + from_dims, all no-op default): use_difficulty_gated_proposal_entropy
  (False) + stuck_progress_window (8) + stuck_progress_stall_eps (0.01) +
  stuck_score_margin_floor (0.05) + stuck_committed_diversity_window (8) +
  stuck_committed_diversity_floor (0.34) + stuck_choice_difficulty_ref (0.05) +
  stuck_goal_salience_floor (0.05) + stuck_ema_alpha_rise (0.3) + stuck_ema_alpha_fall
  (0.05) + stuck_threshold (0.5) + stuck_combine_mode ("mean") + dgpe_candidate_widen_max
  (8) + dgpe_temperature_gain_max (1.0).
  Backward compatible: use_difficulty_gated_proposal_entropy=False by default -> both
  modules None; the _e3_tick gain + select_action detector-update blocks skipped ->
  bit-identical (verified: default == explicit-False action stream). preflight 8/8 + 8 new
  contracts (tests/contracts/test_sd_061_difficulty_gated_proposal_entropy.py: C1
  default-OFF bit-identical / C2 rises under impasse-with-goal / C3 goal-salience guard /
  C4 hysteretic decay / C5 regulator gain mapping + clamp / C6 MECH-094 sim no-op / C7
  agent build+tick+reset / C8 from_dims surfaces knobs). Activation smoke 2026-06-19:
  detector rises to 0.90 under sustained impasse-with-goal, decays to 0.12 after relief,
  stays 0.0 with no goal; regulator s=1 -> (8 extra, 2.0x temp); agent OFF/ON both run
  end-to-end (ON: detector ticks 30, regulator called 5).
  Phased training: N/A (pure-arithmetic; no learned parameters). MECH-094: both modules'
  state-advancing methods no-op under simulation_mode (replay must not accumulate waking
  impasse or widen an imagined proposal). Evidence-staleness (Step 8.5): NOT triggered --
  no-op-default flag; every existing experiment uses the default (regulator off), so no
  dependent claim's measured mechanism changed. KEEP all evidence.
  GOVERNANCE: PROMOTES NOTHING. MECH-343 stays candidate / substrate_conditional /
  v3_pending; claims.yaml carries only the new SD-061 registration + an implementation_note
  (no MECH-343 status/flag change).
  Validation experiment: a substrate-readiness diagnostic (claim_ids=[]; regulator OFF vs
  ON under an induced stuck-state, confirming candidate_first_action_entropy rises under
  stuck + decays after) queued via /queue-experiment. The Q-056 3-arm governance falsifier
  (off / stuck-gated / always-high, matched easy/hard controls) is a SEPARATE later session
  once this readiness check PASSes.
  Design doc: REE_assembly/docs/architecture/sd_061_difficulty_gated_proposal_entropy.md
  Triage: REE_assembly/evidence/planning/q054_q055_q056_buildability_triage_2026-06-19.md
  See MECH-343 (parent mechanism; the substrate_conditional blocker part 2 this builds),
  modulatory-bias-selection-authority (blocker part 1; implemented 569i top-k), ARC-018
  (HippocampalModule proposal locus the gain widens), MECH-341 / ARC-062 (selection-side
  diversity; untouched), MECH-090 / MECH-342 (commitment predicates; untouched), SD-032b
  dACC choice_difficulty (a detector input), MECH-313 (state-independent action-selection
  noise floor; DISTINCT), Q-056 (the falsifier), MECH-094 (call-site scoping).
