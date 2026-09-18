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

### 2026-09-18 AMENDMENT (a)+(b): the temperature half was inert, and two detector axes never arrive (GFLAG-0352)

- **AMENDED 2026-09-18** -- user-decided route after V3-EXQ-1056 (the Q-056 upstream leg)
  was REFUSED at `/queue-experiment` Step 2.5a. Substrate only; **PROMOTES NOTHING**.
  MECH-343 stays candidate / substrate_conditional / v3_pending. Full measurement record:
  `REE_assembly/evidence/planning/exq1056_mech343_q056_upstream_leg_design_refusal_20260918.md`.

- **(a) THE TEMPERATURE HALF OF THE GAIN HAD NO LIVE CONSUMER.** The paragraph above says
  `_e3_tick` applies `differentiable_cem_temperature *= gain`. It does -- but that value is
  READ at exactly ONE place in `ree_core`: `HippocampalModule`'s CEM refit
  (`hippocampal/module.py`), inside SD-055's `if getattr(self.config,
  "use_differentiable_cem", False):`, which defaults **False** and is named nowhere in the
  Config list above, in this record, in `claims.yaml`, or in V3-EXQ-694's driver. So at
  every configuration SD-061's own design record named, the temperature mutation wrote a
  value nothing read and the regulator's effective manipulation was **candidate-COUNT
  widening alone**.
  **This misled the record:** V3-EXQ-694's C2 "regulator load-bearing" PASS certified the
  count half only, while SD-061's `what_would_answer` criterion (2) reads as certifying
  that the regulator "lifts `differentiable_cem_temperature` transiently". The Activation
  smoke line above ("regulator s=1 -> (8 extra, 2.0x temp)") records the GAIN the regulator
  RETURNED, not a temperature any consumer applied.
  **Fix:** `REEConfig.dgpe_enable_differentiable_cem` (default **False**, `from_dims`
  plumbed). True alongside the master flag -> `REEAgent.__init__` sets
  `hippocampal.use_differentiable_cem = True`. Default-off is bit-identical and V3-EXQ-694
  still reproduces exactly.
  **Visibility, which is the durable half of the fix:**
  `DifficultyGatedProposalEntropy.get_state()` now reports
  `sd061_temperature_lever_consumer_live` and `sd061_temperature_half_inert`, so any
  manifest carrying the regulator state says whether the half acted. A run reporting
  `sd061_temperature_half_inert: true` measured count-widening, whatever its docstring says.

- **(a2) MEASURED SCOPE OF THE COUPLED LEVER -- do not over-read the fix.** With the count
  lever off (`dgpe_candidate_widen_max=0`) so the temperature is isolated:
  *uncoupled*, the proposed candidate set is **bit-identical** at `stuck_score` 1.0 vs 0.0
  (max abs diff exactly 0.0) -- the direct proof of (a);
  *coupled*, action-OBJECT content changes, but by only ~1.2e-5, and the candidate
  **first-action-CLASS** distribution is **unchanged** (identical classes, identical
  entropy). That class is a coarse argmax of the sampled trajectory, and a perturbation
  that small essentially never flips it.
  So the coupling makes the lever LIVE; it does **not** make it able to move
  `candidate_first_action_entropy` -- the DV that this record's own validation sentence,
  SD-061 `what_would_answer` criterion (3), and MECH-343's upstream leg (a) all name. In
  the same measurement the only thing that moved that DV was the COUNT lever, and it moved
  it in BOTH directions across proposals (0.377 -> 0.439 on one, 0.311 -> 0.199 on the
  next) -- consistent with V3-EXQ-694's "count-widening is silent-to-adverse on entropy".
  Pinned by contract `C13b`; if it ever fails, re-derive the finding rather than relax it.

- **(b) TWO OF THE FOUR DECLARED DETECTOR INPUTS NEVER ARRIVE.** Measured over an
  ecological loop (`REEConfig.goal_stream` + `CausalGridWorldV2`, agent selecting its own
  actions): `goal_proximity` 100/100 ticks, `goal_salience` 100/100, `score_margin` 99/100,
  `committed_action_class` **0/100**, `choice_difficulty` **0/100**.
  `choice_difficulty` (the SD-032b axis) needs **four** conditions, not one --
  `agent.select_action` writes `_dacc_last_bundle` only inside
  `if self.dacc is not None and z_harm_a is not None:`
  1. `use_dacc=True` -- constructs `agent.dacc`;
  2. `use_affective_harm_stream=True` -- constructs the `AffectiveHarmEncoder` that
     produces `z_harm_a` (`latent/stack.py`);
  3. the environment must emit `harm_obs_a` (`CausalGridWorldV2` does);
  4. **the driver must forward it**: `agent.sense(..., obs_harm_a=...)`.
     `act_with_split_obs` calls `sense(obs_body, obs_world)` with no harm channel, so a
     driver on that convenience interface can **never** populate the axis, at any config.
     `experiments/_harness.py`, `experiments/_lib/allon_training.py` and the
     `_lib/baselines/*` modules all forward it correctly; a hand-rolled
     `act_with_split_obs` loop does not.
  `use_dacc=True` was therefore **not** made sufficient -- doing so would mean forcing an
  `nn.Module` encoder on, which changes the parameter set and the RNG stream and is not
  bit-identical. The requirement is recorded instead, per the user's decision.
  `committed_action_class` needs a commitment to have occurred (`e3._committed_trajectory`,
  i.e. a beta elevation), which runs into MECH-342's registered open failure ("no natural
  commit when score margins are flat", V3-EXQ-629).

- **(b2) WHY THIS MATTERS ARITHMETICALLY.** The combine is a mean over **PRESENT** axes
  only, so which axes arrive sets the ATTAINABLE MAXIMUM of `stuck_score`, not merely its
  value. With the progress axis saturated at 1.0 and the margin axis at 0.0, the evidence
  is exactly `mean(1.0, 0.0) = 0.5` -- identical to the default `stuck_threshold`,
  approached from below by the EMA, so `is_stuck` **never fires** (measured duty cycle
  0.000 in every arm). And `last_deficit_*` is 0.0 for BOTH an absent axis and a
  present-but-zero one, so a null was previously unattributable between "not stuck" and
  "the axes that would have said so never arrived".
  **Fix (diagnostics only, `update()` arithmetic untouched):** `get_state()` now reports
  `sd061_last_present_{progress,margin,diversity,difficulty}`,
  `sd061_n_present_*`, and `sd061_n_axes_present_last`.

- **NOT DECIDED HERE, deliberately (c).** What the detector SHOULD do when axes are absent
  -- an axis mask, a minimum-present-axes requirement, or a threshold recalibration -- and
  WHICH axes ought to carry the firing, are open design decisions that determine what
  "stuck" MEANS. They are raised as a decision chip, not taken by this build.
  **Do not queue Q-056 until (c) is answered.**

- Config added: `dgpe_enable_differentiable_cem` (False). Contracts: C9-C16 added to
  `tests/contracts/test_sd_061_difficulty_gated_proposal_entropy.py` (17 pass).
  Status stays `implemented_pending_validation`.
