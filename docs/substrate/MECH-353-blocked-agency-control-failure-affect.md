## MECH-353: blocked-agency / control-failure affect stream (z_block) (2026-06-06)
- MECH-353: affect.blocked_agency_control_failure_stream -- IMPLEMENTED 2026-06-06
  (substrate; v3_pending until the discriminative experiment PASSes). The
  energised "assert / restore" pole REE lacks: REE encodes only the
  capacity-collapsed WITHDRAW pole (SD-019b z_harm_a + Q-036). z_block is the
  capacity-RETAINED ASSERT pole that rises when an intended, predicted-to-succeed
  action is repeatedly BLOCKED while goal + capacity-belief are retained.
  Registered 2026-06-05 from the blocked-agency lit-pull (chip task_64c2e558,
  Davis & Montag 2019 RAGE / Papini 2024 frustrative non-reward / Bertsch 2020
  reactive-aggression assert channel / Carruthers 2012 comparator). Stream A only
  (single-agent); Stream B coercion/injustice is V4-social, NOT built.
  Modules:
    ree_core/affect/blocked_agency.py (NEW package + BlockedAgency +
      BlockedAgencyConfig + BlockedAgencyOutput). Pure-arithmetic regulator (no
      nn.Module, no learned params), mirrors MECH-313 / MECH-320 / MECH-342.
    ree_core/latent/stack.py (LatentState.z_block [batch,1] + detach()).
    ree_core/agent.py (import; self.blocked_agency + _ba_prev_z_world/_z_self
      caches in __init__; reset(); _update_blocked_agency() called from sense()
      after the MECH-095 TPJ update; ASSERT score-bias in select_action after the
      tonic_vigor block; ARC-016-gated DECOMMIT release beside the MECH-342 site).
    ree_core/utils/config.py (REEConfig + from_dims: 20 no-op-default knobs).
    ree_core/environment/causal_grid_world.py (scheduled_action_block_enabled /
      _interval / _prob env-only kwargs; step() cancels the move on a block with
      no damage / no layout change, transition_type="action_blocked"; info tags;
      reset of per-episode counters).
  Detector (in _update_blocked_agency, waking path):
    Action-outcome comparator (SD-029 on z_world; Carruthers 2012):
      zw_pred = E2.world_forward(z_world_prev, last_action);
      pred_mag = ||zw_pred - z_world_prev||;
      outcome_mismatch = ||zw_pred - z_world_now|| / (pred_mag + eps), gated by a
      predicted-effect floor (pred_mag >= blocked_agency_predicted_effect_floor;
      else 0 -> nothing to be blocked from + fails safe untrained). Calibration
      (trained world_forward): blocked -> ~1.0; success -> ~0.0. FNR's
      expected-minus-obtained generalised to action-effect-omission (Papini 2024).
    Motor-agency / external-attribution comparator (z_self channel):
      motor_agency = 1/(1 + ||E2.predict_next_self(z_self_prev, last_action) -
      z_self_now||). High = motor executed as predicted -> external block, not own
      motor error (predict_next_self is E2's trained objective -> reliable gate).
    Expectation (MECH-112): goal_state.is_active() -> there is an intended outcome.
  Integration (BlockedAgency.update): z_block accumulates outcome_mismatch over a
    window ONLY when motor_agency >= attribution_motor_floor AND outcome_mismatch
    >= outcome_mismatch_floor AND goal retained; leaks on success. Capacity split:
    z_block_assert = z_block * capacity_belief; withdraw_handoff = z_block *
    (1 - capacity_belief); capacity_belief = clip(1 - w*||z_harm_a||, 0, 1).
  Consumers (select_action):
    ASSERT: self-emitted per-candidate score-bias (negative on action / positive
      on no-op / extra positive on the blocked action class) composed into
      dacc_score_bias after the tonic_vigor block (the "raise MECH-320 vigor +
      alternative-action search" effect; zero when no asserting block this tick).
    DECOMMIT: update() emits decommit_signal after z_block_assert sustains above
      decommit_bound for decommit_consecutive_ticks; consumed beside the MECH-342
      site -> beta_gate.release(), gated by ARC-016 (e3.current_precision <=
      blocked_agency_decommit_arc016_precision_max; <=0 disables the gate).
    HANDOFF: assert decays as capacity falls; withdraw_handoff surfaced; existing
      SD-019b / z_harm_a / Q-036 withdraw machinery takes over (no forced write).
  Config: REEConfig.use_blocked_agency (default False; bit-identical OFF -> agent.
    blocked_agency is None, LatentState.z_block stays None, no consumer fires) +
    19 sub-knobs (all no-op default). Env: scheduled_action_block_* (env-only,
    not in from_dims; bit-identical OFF, no RNG draws).
  Backward compatible: 803/803 ree-v3 contracts + bit-identical action stream with
    master OFF AND with use_blocked_agency=True but no env block (regulator uses no
    torch RNG; zero bias when z_block~0). 9/9 new contracts in
    tests/contracts/test_mech_353_blocked_agency.py PASS.
  Phased training: N/A (pure-arithmetic regulator; no learned parameters).
  MECH-094: update() no-op under simulation_mode / hypothesis_tag (replay must not
    accumulate blocked-agency on imagined outcomes); z_block left None on tagged
    latents.
  DETECTOR DISCRIMINATION DEPENDS ON A TRAINED SUBSTRATE: at smoke / untrained
    scale z_world deltas do not track single-cell moves (random encoder), so the
    action-outcome comparator is uninformative (expected). The validation EXQ
    trains the encoder + action-conditional world_forward (SD-056) in P0 and
    measures block-vs-control discrimination in P1. A trained-substrate failure to
    discriminate is a substrate-ceiling finding (encoder/world_forward enrichment),
    NOT a falsification of the affective claim.
  OUTCOME-MISMATCH-FLOOR CALIBRATION AMENDMENT -- IMPLEMENTED 2026-08-30
    (ree-v3 d49db86f3e64670eb59d05eaccdfcda091ede52e; IGW-20260830-226).
    V3-EXQ-642a cleared the trained-comparator readiness gate but made C1
    unsatisfiable: the legacy fixed 0.1 floor sat below the measured free-step
    mismatch baseline (~0.38-0.50), so ordinary world-model error accumulated
    z_block and both arms pinned at z_block_cap=1.5. The new opt-in
    blocked_agency_outcome_mismatch_floor_mode="baseline_relative" compares
    against max(baseline_min_floor, free_step_baseline_ema * floor_ratio), with
    defaults alpha=0.02, ratio=1.5, minimum=0.02. The initial no-history sense
    tick does not advance calibration; the first real action-outcome observation
    bootstraps the baseline without classifying itself, and later free ticks alone
    advance it, so a sustained external block cannot raise its own floor.
    "absolute" remains the default and is bit-identical to the pre-amendment path.
    Contracts C10-C15 pin default-mode
    compatibility, calibrated discrimination, direct reproduction/removal of the
    642a saturation signature, from_dims wiring, and loud config validation.
    V3-EXQ-642b is the opt-in post-build discriminative validation; MECH-353 stays
    v3_pending until that run PASSes.
  Validation experiment: V3-EXQ blocked-action discriminative diagnostic
    (claim_ids=[]; env repeatedly blocks an intended predicted-to-succeed action,
    harm + goal-value held constant; measure z_block rise + assert-vs-withdraw +
    dissociation from z_harm_a under matched controllability). Queued via
    /queue-experiment. PASS clears MECH-353 v3_pending.
  Design doc: REE_assembly/docs/architecture/mech_353_blocked_agency_zblock.md;
    affect-register row: REE_assembly/docs/architecture/affect_primitives.md
    (Extension Register, blocked_agency).
  See MECH-353, SD-029 (comparator detector), MECH-112 (z_goal expectation),
    MECH-320 (assert/vigor pole), MECH-342 (decommit actuator), ARC-016 (decommit
    gate), SD-011 (harm, differentiated-from), SD-019b/Q-036 (suffering withdraw
    pole, opposite controllability), MECH-056 (residue, differentiated-from),
    MECH-090 (commitment-hold, differentiated-from), MECH-095 (TPJ z_self agency
    sibling), SD-056 (action-conditional world_forward; detector prerequisite),
    MECH-094 (simulation gate).
