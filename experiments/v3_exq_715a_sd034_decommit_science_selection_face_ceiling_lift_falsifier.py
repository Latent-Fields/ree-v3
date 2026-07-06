"""
V3-EXQ-715a: SD-034 DE-COMMIT SCIENCE falsifier WITH the SELECTION-FACE CEILING-LIFT ENABLED
(Move M1 of claim_synthesis_MECH-445-446_2026-07-06.md) -- MECH-445 (closure->beta commit-INTENT
engagement) + MECH-446 (de-commit-AUTHORITY magnitude / within-arm around-closure occupancy-drop)
CO-OCCURRENCE, re-run on the F-independent commit-ENTRY substrate (use_closure_commit_entry,
VALIDATED 460o/460p PASS) but this time with the ALREADY-BUILT selection-face conversion-ceiling
levers TURNED ON: MECH-448 (rank-preserving F->eligibility demotion, use_f_eligibility_demotion;
VALIDATED V3-EXQ-689d PASS, promoted provisional) + MECH-449 (Go/No-Go eligibility constitution,
use_go_nogo_constitution; VALIDATED V3-EXQ-689g PASS, promoted provisional).

SUPERSEDES V3-EXQ-715 (the levers-OFF de-commit-science falsifier that self-routed
substrate_not_ready: gate (a) armed on only 1/3 seeds -- seed 44 -- while the two strong-natural-
commit seeds 42/43, ~181/274 F-commits, never armed; and the MECH-446 around-closure window was
vacuous even on the arming seed). 715a is the corrected iteration: it does NOT re-run 715's
levers-OFF config -- it turns on a VALIDATED substrate capability that was OFF in every prior
iteration (460h/i/j/k/l/715).

WHY THIS IS BRAKE-SAFE (re-derive brake RELEASED). MECH-445/446 each carry 6 substrate_ceiling
autopsies (460h/i/j/k/l/715) -- the re-derive brake (threshold 2) FIRED and the 715 autopsy
REFUSED a same-claim de-commit-falsifier re-queue. The brake is RELEASED here on the Step-2.5b
release condition (the named upstream substrate is now BUILT+VALIDATED): the 715 autopsy routed
to /implement-substrate on f_dominance_conversion_ceiling asking for "an arming-under-moderate-F
lever on the de-commit-release face"; the /claim-synthesis refusal record
(claim_synthesis_MECH-445-446_2026-07-06.md, the higher governance adjudication directed as the
715 autopsy's secondary route) then ruled Move M1 explicitly "Brake-safe: a NEW substrate
CONFIGURATION, not a same-selector re-derive" (S5/S6), because the selection-face levers MECH-448
(689d PASS) + MECH-449 (689g PASS) are already built+validated+promoted-provisional and were OFF
in 715. This experiment IS Move M1: it tests whether that already-paid-for selection-face lift
DOUBLES AS the arming-under-moderate-F lever the autopsy asked for -- with zero new substrate
build. The brake stays FIRED for any levers-OFF same-selector de-commit-MAGNITUDE re-run.

THE HYPOTHESIS (Move M1). The 715 failure root is the F-dominance conversion ceiling (MECH-439):
the F-driven natural commit monopolises selection, so the closure-entry latch only arms when F
collapses (weak-F regime, seed 44 only) and the sustained occupancy MECH-446 must de-commit FROM
only exists in the strong-F regime -- the two children's certifiers land on anti-correlated seeds
(empty intersection). MECH-448 removes F from the final committed argmin (uses it ONLY as a graded
rank-preserving eligibility envelope), so turning it ON should create a MODERATE-F regime that is
the norm across seeds, so (a) MECH-445 arms on >= 2/3 seeds AND (b) MECH-446 gets a closure-driven
sustained occupancy to de-commit from, co-located with the SD-034 closure fires -- potentially
unblocking BOTH children with zero new build. If it instead OVER-suppresses F and destroys the
occupancy MECH-446 needs, that is itself the diagnostic that the de-commit-release face needs a
DISTINCT lever (routed back to /implement-substrate via a specific route_reason).

DESIGN -- 2x2 arm grid (use_closure_commit_entry {OFF,ON} x selection-face lift {OFF,ON}), all
four arms cloned from the SAME trained substrate (the entry latch + demotion envelope + Go/No-Go
gate carry NO trainable parameters, so they are toggled per-arm at eval, mirroring 715's
per-arm entry toggle):
  ARM_LIFT_OFF_ENTRY_OFF -- 715 baseline (no lift, no entry); the no-arm non-degeneracy floor.
  ARM_LIFT_OFF_ENTRY_ON  -- reproduces 715's science arm IN-RUN (entry ON, lift OFF): expected to
                            arm only on the weak-F seed, reproducing the 1/3 failure signature.
  ARM_LIFT_ON_ENTRY_OFF  -- lift ON, entry OFF: the lift-on no-arm non-degeneracy contrast +
                            F-regime characterization (does the lift moderate F without the entry
                            latch?).
  ARM_LIFT_ON_ENTRY_ON   -- THE M1 SCIENCE ARM (lift + entry): does the moderate-F regime arm on
                            >= 2/3 seeds AND produce a scorable de-commit window?
The in-run LIFT_OFF vs LIFT_ON contrast makes the causal read airtight (same seeds, same env, same
trained weights) rather than depending on a cross-run comparison to the 715 manifest.

WHAT IS TESTED (on ARM_LIFT_ON_ENTRY_ON; identical DVs to 715):
  MECH-445 (closure->beta commit-intent, child A): the refractory-independent commit-intent counter
    beta_gate.sd034_n_closure_commit_intent (counted BEFORE the elevate/refractory gate, so the
    MECH-446 magnitude lever cannot zero it) is >= MIN_COMMIT_INTENT on >= 2/3 seeds.
  MECH-446 (de-commit-authority magnitude, child B): within-arm mean post-closure latch occupancy
    is below the mean pre-closure occupancy by >= DECOMMIT_MIN_DROP_FRAC (relative), over
    >= C2_MIN_WINDOW_EVENTS scored around-closure windows whose pre-occupancy cleared
    WITHIN_PRE_OCC_FLOOR, on >= 2/3 seeds.

GATE ORDER:
  (a) READINESS (self-route substrate_not_ready_requeue, NEVER a weakens):
      1. foraging contact guard (per-seed P2 contact_rate + z_goal, unchanged from 715).
      2. a goal-active rule-directed commitment forms on the science arm (the SET precondition).
      3. LIFT ENGAGEMENT (NEW, the load-bearing M1 readiness gate; guards the 485i silent
         all-admit trap): on ARM_LIFT_ON_ENTRY_ON the MECH-448 demotion actually ENGAGED --
         f_eligibility_demotion_active True AND f_eligibility_excluded_count > 0 on some ticks --
         on >= 2/3 seeds. If the demotion silently all-admits (excluded_count == 0), the "lift"
         is inert and the run self-routes substrate_not_ready (route_reason=lift_did_not_engage):
         you cannot conclude the moderate-F regime helped or hurt from a no-op lift.
      4. gate (a) armed_and_sustained on ARM_LIFT_ON_ENTRY_ON (ncl_hold_closure_armed_total > 0
         AND max_consecutive_beta_run >= SUSTAIN_MIN_TICKS) AND ARM_LIFT_ON_ENTRY_OFF does NOT
         arm (non-degeneracy: the occupancy is entry-driven).
      5. the MECH-446 around-closure window is non-vacuous on ARM_LIFT_ON_ENTRY_ON.
  (b)+(c) SCIENCE (falsifiable only once readiness clears): the SD-034 closure de-commit shortens
      the sustained occupancy (MECH-446 within-arm drop) AND MECH-445 commit-intent CO-OCCURS with
      it on the SAME >= 2/3 seeds. PASS = co-occurrence. A readiness-cleared child-gate failure is
      a genuine WEAKENS.

M1 DIAGNOSTIC route_reasons (the payload back to the f_dominance_conversion_ceiling
/implement-substrate owner -- answers "does the selection-face lift double as the de-commit-release
substrate?"):
  lift_did_not_engage                         -- demotion silently all-admitted; lift inert on this
                                                 bank (needs envelope-floor calibration, per 485i).
  lift_engaged_but_arming_still_regime_scoped -- lift moderated F but arming still < 2/3 -> the
                                                 de-commit-release face needs a DISTINCT lever.
  lift_engaged_arming_ok_but_window_starved   -- armed >= 2/3 but MECH-446 window vacuous -> closure
                                                 fires still don't co-locate; distinct lever needed.
  (readiness clears) -> genuine PASS / weakens on the science.

F-REGIME readouts (characterize the mechanism, not a hard gate): per-arm n_f_commits, and the
moderate-F delta (does the lift reduce F-commits on the previously-strong seeds 42/43 into the
arming band?). Reported for interpretation; the load-bearing readiness gate is lift engagement +
gate (a), not a pre-registered F-commit band.

PROMOTES NOTHING -- MECH-445/446 stay candidate / v3_pending / pending_retest_after_substrate
until this scores; a PASS/weaken is applied by governance, not by this script.

MECH-094: the closure-entry latch SET (e3._closure_committed_active) is a WAKING control-state
transition -- no replay / no memory-write surface -- so hypothesis_tag does NOT apply
(agent.update_residue is called with hypothesis_tag=False in the eval loop). Ethics preflight:
all-false / decision allow (V3 pre-ethical instrumentation; SENT-0 boundary).

DESIGN docs: REE_assembly/evidence/planning/claim_synthesis_MECH-445-446_2026-07-06.md (Move M1),
docs/architecture/natural_commit_occupancy_release.md (commit-ENTRY amend), the
f_dominance_conversion_ceiling substrate_queue entry (MECH-448/449 selection-face levers).
Substrate harness mirrors VALIDATED V3-EXQ-460o/715; the selection-face lever config mirrors the
VALIDATED V3-EXQ-689d (MECH-448) / 714 fullstack (MECH-449 auto-wired perseveration axis).

experiment_purpose: evidence. claim_ids: [MECH-445, MECH-446].
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.heartbeat.beta_gate import BetaGate  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _benefit_and_drive,
    _contacted_resource_type,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)

EXPERIMENT_TYPE = "v3_exq_715a_sd034_decommit_science_selection_face_ceiling_lift_falsifier"
QUEUE_ID = "V3-EXQ-715a"
CLAIM_IDS: List[str] = ["MECH-445", "MECH-446"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-715"  # supersedes the levers-OFF 715 de-commit-science falsifier (see docstring)

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_DECOMMIT_SCIENCE_SELECTION_FACE_LIFT_2x2_ENTRY_x_LIFT"

# --- Goal-pipeline / encoder dims (mirror 460o exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C; mirror 460o) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- closure-plane commit-ENTRY primitive (REEConfig defaults; entry ON arms) ---
CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR = 0.01  # rule_state norm floor for the SET predicate

# --- Selection-face conversion-ceiling LIFT (Move M1): MECH-448 demotion + MECH-449 Go/No-Go.
#     Values mirror the VALIDATED V3-EXQ-689d (MECH-448 lead lever) and V3-EXQ-714 fullstack
#     (MECH-449 constitution). Toggled per-arm at eval by the "lift" flag; carries NO trainable
#     parameters (selection-time envelope + gate over candidate scores) so the same trained
#     weights serve all four arms. MECH-449 runs on its AUTO-wired perseveration axis (MECH-260
#     dACC suppression, supplied by agent.select_action); no OFC viability injection here (that
#     is the 714 fullstack valuation face, out of scope for the selection-face M1 probe). ---
F_ELIGIBILITY_ENVELOPE_FLOOR = 0.30  # absolute DN merit-share floor (689d; GAP-A-tuned)
F_ELIGIBILITY_DN_SIGMA = 0.0         # DN semi-saturation (0 = no envelope narrowing)
GNG_SAFETY_FLOOR = 0.5               # No-Go if safety-undesirability >= floor (714)
GNG_STALENESS_FLOOR = 0.5            # No-Go if staleness >= floor (714)
GNG_PERSEVERATION_FLOOR = 0.5        # No-Go if MECH-260 recency-share >= floor (714; auto-wired)
GNG_VIABILITY_FLOOR = 0.1            # No-Go if viability < floor (714; not injected here)
GNG_PROTECT_MIN_ELIGIBLE = 1         # fail-open guard: soft No-Go never empties the eligible set

# --- Within-arm around-closure window DV (MECH-446, mirror 460h/460o) ---
CLOSURE_WINDOW = 10
WINDOW_MIN_TICKS = 3

# --- Curriculum budgets (mirror 460o exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
CLOSURE_EVAL_EPISODES = 15  # per arm (x2 arms)
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- Pre-registered acceptance thresholds ---
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0
ARM_PASS_FRACTION = 2.0 / 3.0       # gate (a) + science gates: on >= 2/3 seeds
SUSTAIN_MIN_TICKS = 2               # gate (a): multi-tick closure-formed occupancy
RULE_DIRECTED_MIN_TICKS = 1         # non-vacuity: >= 1 goal-active rule-directed commit tick
# MECH-445 (commit-intent, child A): refractory-independent counter >= this on the ON arm.
MIN_COMMIT_INTENT = 1
# MECH-446 (within-arm around-closure drop, child B): mean post-closure occupancy at least this
# RELATIVE fraction below mean pre-closure occupancy (paired across closures). Mirror 460h.
DECOMMIT_MIN_DROP_FRAC = 0.10
C2_MIN_WINDOW_EVENTS = 2            # minimum scored around-closure windows on the ON arm
WITHIN_PRE_OCC_FLOOR = 0.1         # pre-closure occupancy must be non-trivial (was committed)
# M1 lift-engagement readiness (load-bearing): the MECH-448 demotion must actually exclude a
# non-empty F-eligible tail on at least this many ticks of the science arm (guards 485i all-admit).
LIFT_ENGAGE_MIN_TICKS = 1

# --- Eval-arm definitions: 2x2 grid (use_closure_commit_entry {OFF,ON} x selection-face lift
#     {OFF,ON}). Closure-exclusive de-commit eval ON in every arm; rung-6 + ARC-108 driver OFF in
#     every arm. The lift (MECH-448 demotion + MECH-449 Go/No-Go) is toggled per-arm at eval. ---
ARM_LIFT_OFF_ENTRY_OFF = "ARM_LIFT_OFF_ENTRY_OFF"  # 715 baseline: no-arm non-degeneracy floor
ARM_LIFT_OFF_ENTRY_ON = "ARM_LIFT_OFF_ENTRY_ON"    # reproduces 715's 1/3-arming science arm
ARM_LIFT_ON_ENTRY_OFF = "ARM_LIFT_ON_ENTRY_OFF"    # lift-on no-arm contrast + F-regime readout
ARM_LIFT_ON_ENTRY_ON = "ARM_LIFT_ON_ENTRY_ON"      # THE M1 SCIENCE ARM (lift + entry)
ARMS: List[Dict[str, Any]] = [
    {"key": ARM_LIFT_OFF_ENTRY_OFF, "entry": False, "lift": False},
    {"key": ARM_LIFT_OFF_ENTRY_ON, "entry": True, "lift": False},
    {"key": ARM_LIFT_ON_ENTRY_OFF, "entry": False, "lift": True},
    {"key": ARM_LIFT_ON_ENTRY_ON, "entry": True, "lift": True},
]
# The science arm + its non-degeneracy contrast (lift-on pair) and the 715-reproduction arm.
SCIENCE_ARM = ARM_LIFT_ON_ENTRY_ON
NONDEGEN_CONTRAST_ARM = ARM_LIFT_ON_ENTRY_OFF  # lift-on, entry-off: must NOT arm
REPRO_715_ARM = ARM_LIFT_OFF_ENTRY_ON          # lift-off, entry-on: reproduces 715's 1/3 signature


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        scaffold_train_rule_bias_head=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """460o-validated foraging substrate + commitment control-plane + commitment-closure-
    control-plane amend Legs A/B/C + beta-engagement coupling + the closure-exclusive de-commit
    eval mode + the natural-commit latch-hold. use_closure_commit_entry is LEFT OFF on the
    trained base (armed per-arm at eval by _clone_arm; the latch + hold carry no trainable
    parameters). The rung-6 NaturalCommitUrgencyRelease and the ARC-108 JOB-2 driver pair are
    OFF in every arm (NOT under test -- this is the intrinsic-SD-034-de-commit science)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        use_dacc=True,
        use_salience_coordinator=True,
        use_lateral_pfc_analog=True,
        use_closure_operator=True,
        # SD-034 commitment-closure-control-plane amend (Legs A/B/C + coupling):
        use_closure_env_completion_hook=True,          # Leg A
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B base (MECH-446 refractory)
        closure_decommit_hold_scale_with_run=CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        closure_decommit_hold_max_ticks=CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        lateral_pfc_train_rule_bias_head=True,         # Leg C un-zero (GAP-D)
        use_closure_commit_beta_coupling=True,         # beta-engagement coupling (MECH-445 path)
        # rung-6 natural-commit-occupancy-release lever: OFF in every arm (parked line).
        use_natural_commit_urgency_release=False,
        # The natural-commit LATCH-HOLD is ARMED on the base config (carried into every arm via
        # _clone_arm's deepcopy). Precondition of use_closure_commit_entry + closure_exclusive_-
        # decommit_eval.
        use_natural_commit_latch_hold=True,
        # The CLOSURE-EXCLUSIVE DE-COMMIT EVAL mode (ree-v3 e52158d): beta elevation closure-
        # EXCLUSIVE + the latch-hold arms on _closure_commit_active. ON in EVERY arm.
        closure_exclusive_decommit_eval=True,
        # The F-INDEPENDENT commit-ENTRY primitive under test: OFF on the trained base; armed
        # per-arm at eval by _clone_arm. Preconditions (loud ValueError): requires
        # use_closure_commit_beta_coupling AND use_natural_commit_latch_hold -- both set above.
        use_closure_commit_entry=False,
        closure_commit_entry_rule_norm_floor=CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
        # Selection-face conversion-ceiling LIFT (MECH-448 demotion + MECH-449 Go/No-Go): OFF on
        # the trained base; armed per-arm at eval by _clone_arm on the "lift" arms. Both operate
        # at selection time over candidate scores (no trainable parameters), so the same trained
        # weights serve the lift-OFF and lift-ON arms. The floors are carried on the base config
        # so a per-arm flip of use_f_eligibility_demotion / use_go_nogo_constitution suffices.
        use_f_eligibility_demotion=False,
        f_eligibility_envelope_floor=F_ELIGIBILITY_ENVELOPE_FLOOR,
        f_eligibility_dn_sigma=F_ELIGIBILITY_DN_SIGMA,
        use_go_nogo_constitution=False,
        gng_safety_floor=GNG_SAFETY_FLOOR,
        gng_staleness_floor=GNG_STALENESS_FLOOR,
        gng_perseveration_floor=GNG_PERSEVERATION_FLOOR,
        gng_viability_floor=GNG_VIABILITY_FLOOR,
        gng_protect_min_eligible=GNG_PROTECT_MIN_ELIGIBLE,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _arm_config_slice(arm: Dict[str, Any]) -> Dict[str, Any]:
    """The full per-arm config dict for the arm_fingerprint. The inter-arm variables are the
    2x2 grid (use_closure_commit_entry x the selection-face lift = use_f_eligibility_demotion +
    use_go_nogo_constitution); the rest is shared substrate config."""
    return {
        "arm_key": arm["key"],
        "use_closure_commit_entry": bool(arm["entry"]),
        "closure_commit_entry_rule_norm_floor": CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
        # Selection-face lift (MECH-448 + MECH-449), the second grid axis:
        "use_f_eligibility_demotion": bool(arm["lift"]),
        "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
        "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
        "use_go_nogo_constitution": bool(arm["lift"]),
        "gng_safety_floor": GNG_SAFETY_FLOOR,
        "gng_staleness_floor": GNG_STALENESS_FLOOR,
        "gng_perseveration_floor": GNG_PERSEVERATION_FLOOR,
        "gng_viability_floor": GNG_VIABILITY_FLOOR,
        "gng_protect_min_eligible": GNG_PROTECT_MIN_ELIGIBLE,
        "closure_exclusive_decommit_eval": True,
        "use_closure_commit_beta_coupling": True,
        "use_natural_commit_latch_hold": True,
        "use_natural_commit_urgency_release": False,
        "beta_gate_bistable": True,
        "use_lateral_pfc_analog": True,
        "use_closure_operator": True,
        "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
        "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        "world_dim": WORLD_DIM,
        "z_goal_enabled": True,
        "drive_weight": DRIVE_WEIGHT,
        "scaffold_train_rule_bias_head": True,
        "closure_eval_episodes": CLOSURE_EVAL_EPISODES,
    }


def _build_closure_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint tolerance-band
    completion so the SD-034 closure operator has completions to fire on (mirror 460o)."""
    p2_hfa = (
        scaffold_cfg.scaffold_p2_hazard_food_attraction_guard
        if scaffold_cfg.scaffold_p2_hazard_food_attraction_guard >= 0.0
        else scaffold_cfg.scaffold_p2_hazard_food_attraction
    )
    return CausalGridWorldV2(
        size=scaffold_cfg.scaffold_env_size,
        num_hazards=scaffold_cfg.scaffold_p2_num_hazards,
        num_resources=scaffold_cfg.scaffold_p2_num_resources,
        hazard_food_attraction=p2_hfa,
        proximity_harm_scale=scaffold_cfg.scaffold_p2_proximity_harm_scale,
        limb_damage_enabled=True,
        reef_enabled=True,
        reef_bipartite_layout=True,
        reef_bipartite_axis=scaffold_cfg.scaffold_reef_bipartite_axis,
        reef_bipartite_agent_band_radius=scaffold_cfg.scaffold_reef_bipartite_agent_band_radius,
        reef_bipartite_agent_spawn_in_reef_half=False,
        subgoal_mode=True,
        num_waypoints=2,
        completion_tolerance_enabled=True,
        completion_tolerance_frac=0.25,
        completion_tolerance_metric="chebyshev",
        completion_tolerance_targets="waypoint",
        **_sd049_kwargs(scaffold_cfg),
    )


def _clone_arm(trained_agent: REEAgent, device: torch.device, arm: Dict[str, Any]) -> REEAgent:
    """Clone the SAME trained weights into an agent built with this arm's 2x2 config
    (use_closure_commit_entry x the selection-face lift). The closure-exclusive eval + latch-hold
    + closure operator stay ON in every arm. The entry latch, the MECH-448 demotion envelope, and
    the MECH-449 Go/No-Go gate all carry NO trainable parameters (selection-time ops), so the
    state_dict loads cleanly onto every arm (mirrors 460o/715's _clone_arm)."""
    cfg = copy.deepcopy(trained_agent.config)
    cfg.use_closure_commit_entry = bool(arm["entry"])
    cfg.closure_commit_entry_rule_norm_floor = CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR
    # Selection-face LIFT (MECH-448 demotion + MECH-449 Go/No-Go) toggled per-arm (on cfg.e3).
    cfg.e3.use_f_eligibility_demotion = bool(arm["lift"])
    cfg.e3.f_eligibility_envelope_floor = F_ELIGIBILITY_ENVELOPE_FLOOR
    cfg.e3.f_eligibility_dn_sigma = F_ELIGIBILITY_DN_SIGMA
    cfg.e3.use_go_nogo_constitution = bool(arm["lift"])
    cfg.e3.gng_safety_floor = GNG_SAFETY_FLOOR
    cfg.e3.gng_staleness_floor = GNG_STALENESS_FLOOR
    cfg.e3.gng_perseveration_floor = GNG_PERSEVERATION_FLOOR
    cfg.e3.gng_viability_floor = GNG_VIABILITY_FLOOR
    cfg.e3.gng_protect_min_eligible = GNG_PROTECT_MIN_ELIGIBLE
    # rung-6 + ARC-108 JOB-2 driver levers stay OFF in every arm.
    cfg.use_natural_commit_urgency_release = False
    cfg.use_closure_operator = True
    cfg.heartbeat.beta_gate_bistable = True
    agent = REEAgent(cfg).to(device)
    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent.load_state_dict(state)
    except RuntimeError:
        agent.load_state_dict(state, strict=False)
    agent.e3._running_variance = float(trained_agent.e3._running_variance)
    agent.beta_gate = BetaGate(completion_release_threshold=2.0)
    # HARNESS FIX (failure_autopsy_V3-EXQ-460m-460n_2026-06-23; carried from 460o): copy the
    # trained substrate's LIVE goal-seeding calibration onto the clone. The scaffold writes
    # z_goal_seeding_gain / benefit_threshold / drive_floor onto agent.goal_state.config (the
    # live GoalConfig), NOT onto REEConfig, so the deepcopy above does NOT carry them -> without
    # this the clone keeps default GoalConfig (benefit_threshold=0.1) and wild contact (~0.03)
    # would not clear the seeding gate, leaving goal_state inactive.
    _src_gc = getattr(getattr(trained_agent, "goal_state", None), "config", None)
    _dst_gc = getattr(getattr(agent, "goal_state", None), "config", None)
    if _src_gc is not None and _dst_gc is not None:
        for _attr in ("z_goal_seeding_gain", "benefit_threshold", "drive_floor"):
            if hasattr(_src_gc, _attr):
                setattr(_dst_gc, _attr, getattr(_src_gc, _attr))
    return agent


def _around_closure_windows(
    beta_history: List[bool], fire_ticks: List[int]
) -> List[Dict[str, float]]:
    """For each closure fire at tick t, the beta-latch occupancy FRACTION in the pre-closure
    window [t-W, t) and the post-closure window (t, t+W] (the paired within-arm de-commit datum;
    mirror 460h/460o -- the MECH-446 DV)."""
    n = len(beta_history)
    events: List[Dict[str, float]] = []
    for t in fire_ticks:
        pre_lo = max(0, t - CLOSURE_WINDOW)
        pre = beta_history[pre_lo:t]
        post_hi = min(n, t + 1 + CLOSURE_WINDOW)
        post = beta_history[t + 1:post_hi]
        if len(pre) < WINDOW_MIN_TICKS or len(post) < WINDOW_MIN_TICKS:
            continue
        pre_occ = sum(1 for b in pre if b) / float(len(pre))
        post_occ = sum(1 for b in post if b) / float(len(post))
        events.append({"pre_occ": pre_occ, "post_occ": post_occ})
    return events


def _max_consecutive_true(seq: List[bool]) -> int:
    """Longest run of consecutive True (beta-elevated) ticks -- the sustained-occupancy proxy."""
    best = 0
    cur = 0
    for b in seq:
        if b:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def _rule_state_norm(agent: REEAgent) -> float:
    """||lateral_pfc.rule_state|| (the SET predicate's 'rule is being followed' magnitude);
    0.0 when lateral_pfc absent / rule_state None."""
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None:
        return 0.0
    rs = getattr(lpfc, "rule_state", None)
    if rs is None:
        return 0.0
    try:
        return float(rs.norm().item())
    except Exception:
        return 0.0


def _eval_arm_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Eval instrumented for both SD-034 de-commit-pipeline children. Reads the closure-armed
    counter (agent._ncl_hold_closure_armed_count), the longest consecutive beta-elevated run,
    the F-driven natural commits (n_f_commits, expected ~0 -- F-independence), the SD-034 closure
    fire count, the refractory-independent commit-intent counter (MECH-445), the around-closure
    occupancy windows (MECH-446), and the non-vacuity readout n_rule_directed_commit_ticks. Calls
    agent.update_residue(hypothesis_tag=False) each tick so the waking post-action path runs
    (mirror 460o)."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_dacc = getattr(agent, "dacc", None) is not None
    hook_enabled = bool(getattr(agent.config, "use_closure_env_completion_hook", False))
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream
    rule_floor = float(
        getattr(agent.config, "closure_commit_entry_rule_norm_floor",
                CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR)
    )
    # HARNESS FIX (carried from 460o): seed z_goal each eval step (consumption-gated) so
    # goal_state.is_active() can be True during the closure-eval -- the F-independent SET
    # predicate's precondition. The raw-benefit seeding gate mirrors _reconciled_gating_threshold:
    #   floor = benefit_threshold / (z_goal_seeding_gain * (1 + drive_weight * drive_floor)).
    bridge_on = bool(getattr(agent.config, "use_cue_recall", False)) or bool(
        getattr(agent.config, "use_incentive_token_bank", False)
    )
    _gc = getattr(getattr(agent, "goal_state", None), "config", None)
    if _gc is not None:
        _gain = float(getattr(_gc, "z_goal_seeding_gain", 1.0))
        _thr = float(getattr(_gc, "benefit_threshold", 0.1))
        _dw = float(getattr(_gc, "drive_weight", 0.0))
        _df = float(getattr(_gc, "drive_floor", 0.0))
        _denom = _gain * (1.0 + _dw * _df)
        seed_gate = (_thr / _denom) if _denom > 1e-12 else 0.0
    else:
        seed_gate = 0.0

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    beta_release_events = 0
    nogo_installed_total = 0
    total_committed_steps = 0
    total_beta_elevated = 0
    n_sequence_completions = 0
    n_hook_fires = 0
    n_closure_commit_intent = 0
    n_closure_coupled_elevations = 0
    around_events: List[Dict[str, float]] = []
    max_consecutive_beta_run = 0
    ncl_hold_reassert_total = 0
    ncl_hold_closure_armed_total = 0
    # F-driven natural commit count (expected ~0 on this substrate -- the F-independence regime).
    n_f_commits = 0
    # NON-VACUITY readout: goal active AND rule_state norm >= floor (the SET precondition).
    n_rule_directed_commit_ticks = 0
    rule_state_norm_peak = 0.0
    committed_class_counts: Counter = Counter()
    # Selection-face LIFT engagement diagnostics (read off agent.e3.last_score_diagnostics after
    # each select_action). MECH-448 demotion: demotion_active_ticks = flag ON; demotion_engaged_
    # ticks = excluded_count > 0 (a non-empty F-tail was actually demoted -- guards the 485i
    # silent all-admit). MECH-449 Go/No-Go: gonogo_active_ticks + soft-applied count (auto-wired
    # perseveration axis). On the lift-OFF arms these stay 0 (the flags are off).
    n_select_ticks = 0
    demotion_active_ticks = 0
    demotion_engaged_ticks = 0
    demotion_excluded_total = 0
    demotion_winner_neq_f_argmin_ticks = 0
    gonogo_active_ticks = 0
    gonogo_soft_applied_total = 0

    for _ in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        prev_beta = bool(agent.beta_gate.is_elevated)
        beta_history: List[bool] = []
        fire_ticks: List[int] = []

        for tick_idx in range(steps_per_episode):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = _sense_with_optional_harm(
                agent, obs_body, obs_world, obs_dict, device, feed_harm
            )

            n_closures_before = (
                int(agent.closure_operator._n_closures) if has_closure else 0
            )
            dacc_hist_before = len(agent.dacc._action_history) if has_dacc else 0

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            # Selection-face lift engagement (LIVE at this select tick; 0-remains on lift-OFF arms).
            diags = getattr(agent.e3, "last_score_diagnostics", None) or {}
            n_select_ticks += 1
            if bool(diags.get("f_eligibility_demotion_active", False)):
                demotion_active_ticks += 1
                _exc = int(diags.get("f_eligibility_excluded_count", 0) or 0)
                if _exc > 0:
                    demotion_engaged_ticks += 1
                    demotion_excluded_total += _exc
                if bool(diags.get("f_eligibility_winner_neq_f_argmin", False)):
                    demotion_winner_neq_f_argmin_ticks += 1
            if bool(diags.get("go_nogo_constitution_active", False)):
                gonogo_active_ticks += 1
                gonogo_soft_applied_total += int(diags.get("go_nogo_n_soft_applied", 0) or 0)

            if has_closure:
                fired_now = int(agent.closure_operator._n_closures) - n_closures_before
                if fired_now > 0 and has_dacc:
                    nogo_installed_total += (
                        len(agent.dacc._action_history) - dacc_hist_before
                    )

            cur_beta = bool(agent.beta_gate.is_elevated)
            beta_history.append(cur_beta)
            committed_now = agent.e3._committed_trajectory is not None
            if committed_now:
                total_committed_steps += 1
                n_f_commits += 1  # an F-driven natural commit was live AT this read point
                committed_class_counts[action_idx] += 1
            if cur_beta:
                total_beta_elevated += 1
            if prev_beta and not cur_beta:
                beta_release_events += 1
            prev_beta = cur_beta

            # NON-VACUITY: did the SET predicate have a goal-active rule-directed commitment to
            # latch on? (goal_state.is_active() AND rule_state norm >= floor).
            gs = getattr(agent, "goal_state", None)
            goal_active = bool(gs is not None and gs.is_active())
            rs_norm = _rule_state_norm(agent)
            if rs_norm > rule_state_norm_peak:
                rule_state_norm_peak = rs_norm
            if goal_active and rs_norm >= rule_floor:
                n_rule_directed_commit_ticks += 1

            _, _harm, done, info, obs_dict = env.step(action_idx)

            # Drive the waking post-action path (mirror 460o). MECH-094: hypothesis_tag=False --
            # a WAKING control-state transition, no replay / no memory-write surface.
            agent.update_residue(harm_signal=float(_harm), hypothesis_tag=False)

            # HARNESS FIX (carried from 460o): seed z_goal from the POST-step body-state so
            # goal_state.is_active() can be True during the closure-eval. Contact-gated: a
            # sub-seeding whiff is SKIPPED (not decay-only updated) so the consolidated trace is
            # protected from washout (the V3-EXQ-634b lesson); a genuine contact step seeds.
            _benefit_seed, _drive_seed = _benefit_and_drive(obs_dict["body_state"].to(device))
            _rt_seed = _contacted_resource_type(obs_dict) if bridge_on else None
            if _benefit_seed > seed_gate:
                agent.update_z_goal(
                    benefit_exposure=_benefit_seed,
                    drive_level=_drive_seed,
                    resource_type=_rt_seed,
                )

            if info.get("transition_type") == "sequence_complete":
                n_sequence_completions += 1
                if has_closure and hook_enabled:
                    ev = agent.notify_env_completion(action_class=action_idx)
                    if ev is not None and getattr(ev, "fired", False):
                        n_hook_fires += 1
                        nogo_installed_total += int(getattr(ev, "nogo_pushed", 0))

            if has_closure and int(agent.closure_operator._n_closures) > n_closures_before:
                fire_ticks.append(tick_idx)
            if done:
                break

        around_events.extend(_around_closure_windows(beta_history, fire_ticks))
        _ep_run = _max_consecutive_true(beta_history)
        if _ep_run > max_consecutive_beta_run:
            max_consecutive_beta_run = _ep_run
        # Read per-episode counters BEFORE the next agent.reset() wipes them.
        ncl_hold_reassert_total += int(getattr(agent, "_ncl_hold_reassert_count", 0))
        ncl_hold_closure_armed_total += int(
            getattr(agent, "_ncl_hold_closure_armed_count", 0)
        )
        _bstate = agent.beta_gate.get_state()
        n_closure_commit_intent += int(_bstate.get("sd034_n_closure_commit_intent", 0))
        n_closure_coupled_elevations += int(
            _bstate.get("sd034_n_closure_coupled_elevations", 0)
        )

    n_closures = (
        int(agent.closure_operator._n_closures) - closures_pre if has_closure else 0
    )
    n_window_events = len(around_events)
    mean_pre_occ = (
        sum(e["pre_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    mean_post_occ = (
        sum(e["post_occ"] for e in around_events) / n_window_events
        if n_window_events else 0.0
    )
    return {
        "n_closures": n_closures,
        "n_automatic_fires": max(0, n_closures - n_hook_fires),
        "n_hook_fires": n_hook_fires,
        "sd034_n_closure_commit_intent": n_closure_commit_intent,
        "sd034_n_closure_coupled_elevations": n_closure_coupled_elevations,
        "beta_release_events": beta_release_events,
        "nogo_installed_total": nogo_installed_total,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "mean_beta_elevated_steps": total_beta_elevated / max(1, n_eps),
        "mean_per_commit_hold": total_beta_elevated / max(1, beta_release_events),
        "max_consecutive_beta_run": max_consecutive_beta_run,
        "ncl_hold_reassert_total": ncl_hold_reassert_total,
        "ncl_hold_closure_armed_total": ncl_hold_closure_armed_total,
        "n_f_commits": n_f_commits,
        "n_rule_directed_commit_ticks": n_rule_directed_commit_ticks,
        "rule_state_norm_peak": rule_state_norm_peak,
        "n_sequence_completions": n_sequence_completions,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
        "env_hook_enabled": hook_enabled,
        "n_window_events": n_window_events,
        "mean_pre_closure_occ": mean_pre_occ,
        "mean_post_closure_occ": mean_post_occ,
        "committed_class_entropy_n_classes": len(committed_class_counts),
        # Selection-face lift engagement (0 on lift-OFF arms).
        "n_select_ticks": n_select_ticks,
        "demotion_active_ticks": demotion_active_ticks,
        "demotion_engaged_ticks": demotion_engaged_ticks,
        "demotion_excluded_total": demotion_excluded_total,
        "demotion_winner_neq_f_argmin_ticks": demotion_winner_neq_f_argmin_ticks,
        "gonogo_active_ticks": gonogo_active_ticks,
        "gonogo_soft_applied_total": gonogo_soft_applied_total,
    }


def _empty_arm() -> Dict[str, Any]:
    return {
        "n_closures": 0, "n_automatic_fires": 0, "n_hook_fires": 0,
        "sd034_n_closure_commit_intent": 0, "sd034_n_closure_coupled_elevations": 0,
        "beta_release_events": 0, "nogo_installed_total": 0, "total_committed_steps": 0,
        "total_beta_elevated": 0, "mean_beta_elevated_steps": 0.0,
        "mean_per_commit_hold": 0.0, "max_consecutive_beta_run": 0,
        "ncl_hold_reassert_total": 0, "ncl_hold_closure_armed_total": 0,
        "n_f_commits": 0, "n_rule_directed_commit_ticks": 0, "rule_state_norm_peak": 0.0,
        "n_sequence_completions": 0, "n_eval_episodes": 0, "closure_present": False,
        "env_hook_enabled": False, "n_window_events": 0, "mean_pre_closure_occ": 0.0,
        "mean_post_closure_occ": 0.0, "committed_class_entropy_n_classes": 0,
        "n_select_ticks": 0, "demotion_active_ticks": 0, "demotion_engaged_ticks": 0,
        "demotion_excluded_total": 0, "demotion_winner_neq_f_argmin_ticks": 0,
        "gonogo_active_ticks": 0, "gonogo_soft_applied_total": 0,
    }


def _arm_armed_and_sustained(arm: Dict[str, Any]) -> bool:
    """gate (a) per-seed predicate (ARM_ENTRY_ON): the latch armed
    (ncl_hold_closure_armed_total > 0) AND the occupancy sustained
    (max_consecutive_beta_run >= SUSTAIN_MIN_TICKS)."""
    return bool(
        int(arm.get("ncl_hold_closure_armed_total", 0)) > 0
        and int(arm.get("max_consecutive_beta_run", 0)) >= SUSTAIN_MIN_TICKS
    )


def _mech445_commit_intent_met(arm_on: Dict[str, Any]) -> bool:
    """MECH-445 (child A): the refractory-independent closure-plane commit-intent counter fired
    on the ON arm (>= MIN_COMMIT_INTENT) AND a sequence completed (closure had an opportunity).
    Keys on sd034_n_closure_commit_intent -- counted BEFORE the elevate/refractory gate -- so the
    MECH-446 magnitude lever cannot zero it (the 460g self-defeat)."""
    return bool(
        int(arm_on.get("sd034_n_closure_commit_intent", 0)) >= MIN_COMMIT_INTENT
        and int(arm_on.get("n_sequence_completions", 0)) > 0
    )


def _mech446_within_arm_drop_met(arm_on: Dict[str, Any]) -> bool:
    """MECH-446 (child B, load-bearing within-arm DV): on the ON arm, mean post-closure occupancy
    < mean pre-closure occupancy with a >= DECOMMIT_MIN_DROP_FRAC relative drop, over
    >= C2_MIN_WINDOW_EVENTS scored windows whose pre-occupancy cleared WITHIN_PRE_OCC_FLOOR
    (there was a sustained occupancy to de-commit). Mirror 460h C2."""
    n_ev = int(arm_on.get("n_window_events", 0))
    pre = float(arm_on.get("mean_pre_closure_occ", 0.0))
    post = float(arm_on.get("mean_post_closure_occ", 0.0))
    if n_ev < C2_MIN_WINDOW_EVENTS or pre <= WITHIN_PRE_OCC_FLOOR:
        return False
    return bool(post < pre and (pre - post) >= DECOMMIT_MIN_DROP_FRAC * pre)


def _within_window_nonvacuous(arm_on: Dict[str, Any]) -> bool:
    """Readiness: the ON arm produced enough scored around-closure windows with a non-trivial
    pre-closure occupancy for the MECH-446 within-arm DV to be interpretable."""
    return bool(
        int(arm_on.get("n_window_events", 0)) >= C2_MIN_WINDOW_EVENTS
        and float(arm_on.get("mean_pre_closure_occ", 0.0)) > WITHIN_PRE_OCC_FLOOR
    )


def _lift_engaged(science_arm: Dict[str, Any]) -> bool:
    """M1 readiness (load-bearing): the MECH-448 demotion actually ENGAGED on the science arm --
    it was active AND excluded a non-empty F-eligible tail on >= LIFT_ENGAGE_MIN_TICKS ticks
    (demotion_engaged_ticks). Guards the 485i silent all-admit: if the demotion never excludes
    anyone (excluded_count == 0 every tick), the 'lift' is inert and the run cannot conclude the
    moderate-F regime helped or hurt -> self-route substrate_not_ready (route lift_did_not_engage)."""
    return bool(int(science_arm.get("demotion_engaged_ticks", 0)) >= LIFT_ENGAGE_MIN_TICKS)


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "arms": {a["key"]: _empty_arm() for a in ARMS},
        "arm_results": [
            {"seed": seed, "arm": a["key"], "aborted": True, **_empty_arm()}
            for a in ARMS
        ],
        "on_armed_and_sustained": False,
        "off_did_not_arm": False,
        "on_zero_f_commits": False,
        "lift_engaged": False,
        "on_rule_directed_met": False,
        "within_window_nonvacuous": False,
        "mech445_commit_intent_met": False,
        "mech446_within_arm_drop_met": False,
        "cooccur_met": False,
        "science_arm_n_f_commits": 0,
        "repro715_arm_n_f_commits": 0,
        "liftoff_entryoff_n_f_commits": 0,
        "lifton_entryoff_n_f_commits": 0,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")
    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    eval_eps = 2 if dry_run else CLOSURE_EVAL_EPISODES

    probe_env = _build_closure_env(scaffold_cfg)
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
          f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
          f" retention={s0b.retention_ratio:.3f}"
          f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" median_last={hz.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_seed_record(seed, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    # Eval both arms on the SAME trained substrate (clone per arm; the commit-ENTRY flag
    # toggled). Each (seed x arm) cell is wrapped in arm_cell (resets RNG on enter, stamps the
    # fingerprint) -- the multi-arm arm_fingerprint obligation (mint-as-you-go: the ENTRY_OFF
    # baseline cell is emitted reuse-eligible by default).
    arms_out: Dict[str, Any] = {}
    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        with arm_cell(
            seed,
            config_slice=_arm_config_slice(arm),
            script_path=Path(__file__),
        ) as cell:
            closure_env = _build_closure_env(scaffold_cfg)
            closure_env.reset()
            print(f"Seed {seed} Condition {arm['key']}", flush=True)
            agent_arm = _clone_arm(agent, device, arm)
            agent_arm.e3._running_variance = float(agent.e3._running_variance)
            metrics = _eval_arm_behaviour(
                agent_arm, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
            )
            done += eval_eps
            arms_out[arm["key"]] = metrics
            row = {"seed": seed, "arm": arm["key"], "aborted": False, **metrics}
            cell.stamp(row)
            arm_results.append(row)

    # 2x2 grid readout. The M1 science arm is LIFT_ON_ENTRY_ON; its non-degeneracy contrast is
    # LIFT_ON_ENTRY_OFF (must NOT arm); REPRO_715 (LIFT_OFF_ENTRY_ON) reproduces 715's signature.
    science = arms_out[SCIENCE_ARM]
    contrast = arms_out[NONDEGEN_CONTRAST_ARM]
    repro715 = arms_out[REPRO_715_ARM]
    baseline = arms_out[ARM_LIFT_OFF_ENTRY_OFF]

    # Readiness (gate a) legs -- on the science arm (LIFT_ON_ENTRY_ON).
    on_armed_and_sustained = _arm_armed_and_sustained(science)
    # Non-degeneracy: the lift-on ENTRY-OFF arm does NOT arm (occupancy is entry-driven, not F).
    off_did_not_arm = bool(int(contrast.get("ncl_hold_closure_armed_total", 0)) == 0)
    # F-independence context (REPORTED, not a hard gate -- M1 expects MODERATE F, not zero F).
    on_zero_f_commits = bool(int(science.get("n_f_commits", 0)) == 0)
    # NEW load-bearing M1 readiness leg: the MECH-448 demotion actually engaged on the science arm.
    lift_engaged = _lift_engaged(science)
    on_rule_directed_met = bool(
        int(science.get("n_rule_directed_commit_ticks", 0)) >= RULE_DIRECTED_MIN_TICKS
    )
    within_window_nonvacuous = _within_window_nonvacuous(science)

    # Science (gates b/c) -- the two SD-034 de-commit-pipeline children + their co-occurrence.
    mech445_commit_intent_met = _mech445_commit_intent_met(science)
    mech446_within_arm_drop_met = _mech446_within_arm_drop_met(science)
    cooccur_met = bool(mech445_commit_intent_met and mech446_within_arm_drop_met)

    print(f"  [eval] arm_eval seed={seed} ep {done}/{total_eps}"
          f" | REPRO715(liftOFF,entryON) armed={repro715['ncl_hold_closure_armed_total']}"
          f" f_commits={repro715['n_f_commits']}"
          f" | SCIENCE(liftON,entryON) armed={science['ncl_hold_closure_armed_total']}"
          f" run={science['max_consecutive_beta_run']} f_commits={science['n_f_commits']}"
          f" demote_eng={science['demotion_engaged_ticks']}/{science['demotion_active_ticks']}"
          f" excl={science['demotion_excluded_total']} gng={science['gonogo_soft_applied_total']}"
          f" rdir={science['n_rule_directed_commit_ticks']}"
          f" intent={science['sd034_n_closure_commit_intent']}"
          f" pre_occ={science['mean_pre_closure_occ']:.3f} post_occ={science['mean_post_closure_occ']:.3f}"
          f" win={science['n_window_events']}"
          f" | CONTRAST(liftON,entryOFF) armed={contrast['ncl_hold_closure_armed_total']}"
          f" f_commits={contrast['n_f_commits']}"
          f" | armed_sustained={on_armed_and_sustained} contrast_no_arm={off_did_not_arm}"
          f" lift_engaged={lift_engaged} within_win={within_window_nonvacuous}"
          f" | m445={mech445_commit_intent_met} m446={mech446_within_arm_drop_met}"
          f" cooccur={cooccur_met}", flush=True)
    # Per-seed verdict for the runner: guard + readiness legs + co-occurrence. (Governance
    # scoring is aggregate, in run_experiment; this line drives the progress bar only.)
    seed_ready = bool(
        guard_pass and lift_engaged and on_armed_and_sustained and off_did_not_arm
        and on_rule_directed_met and within_window_nonvacuous
    )
    seed_pass = bool(seed_ready and cooccur_met)
    print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} lift_engaged={lift_engaged} ready={seed_ready} cooccur={cooccur_met}"
          f" (contact_rate={p2.contact_rate:.4f} z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f})",
          flush=True)

    return {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "guard_pass": guard_pass,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "arms": arms_out,
        "arm_results": arm_results,
        "on_armed_and_sustained": on_armed_and_sustained,
        "off_did_not_arm": off_did_not_arm,
        "on_zero_f_commits": on_zero_f_commits,
        "lift_engaged": lift_engaged,
        "on_rule_directed_met": on_rule_directed_met,
        "within_window_nonvacuous": within_window_nonvacuous,
        "mech445_commit_intent_met": mech445_commit_intent_met,
        "mech446_within_arm_drop_met": mech446_within_arm_drop_met,
        "cooccur_met": cooccur_met,
        # F-regime readouts (moderate-F hypothesis characterization; not a hard gate).
        "science_arm_n_f_commits": int(science.get("n_f_commits", 0)),
        "repro715_arm_n_f_commits": int(repro715.get("n_f_commits", 0)),
        "liftoff_entryoff_n_f_commits": int(baseline.get("n_f_commits", 0)),
        "lifton_entryoff_n_f_commits": int(contrast.get("n_f_commits", 0)),
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    n_arms = len(ARMS)
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2 + n_arms * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET + n_arms * CLOSURE_EVAL_EPISODES
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    n = len(per_seed)
    guard_flags = [r["guard_pass"] for r in per_seed]
    guard_frac = _frac(guard_flags)
    guard_passing = [r for r in per_seed if r["guard_pass"]]
    contact_non_vacuity_met = bool(guard_frac >= MIN_FRACTION)

    # --- Readiness (gate a) on the guard-passing seeds ---
    def _gp_frac(key: str) -> float:
        return _frac([bool(r.get(key, False)) for r in guard_passing])

    rule_directed_frac = _gp_frac("on_rule_directed_met")
    rule_directed_met = bool(rule_directed_frac >= ARM_PASS_FRACTION)
    armed_frac = _gp_frac("on_armed_and_sustained")
    armed_and_sustained_met = bool(armed_frac >= ARM_PASS_FRACTION)
    off_did_not_arm_frac = _gp_frac("off_did_not_arm")
    # Non-degeneracy: on the lift-on pair the ENTRY-OFF (NONDEGEN_CONTRAST) arm's latch-hold
    # reduces to the F path (no entry primitive), so if the (now demotion-moderated) F-commit were
    # still strong enough to form an occupancy that arm WOULD arm. off_did_not_arm==True therefore
    # certifies the science-arm occupancy is ENTRY-driven, not F-driven -- the M1 non-degeneracy
    # leg. n_f_commits is REPORTED (moderate-F characterization), NOT a separate hard gate.
    off_did_not_arm_met = bool(off_did_not_arm_frac >= ARM_PASS_FRACTION)
    zero_f_frac = _gp_frac("on_zero_f_commits")
    # LIFT ENGAGEMENT (load-bearing M1 readiness): the MECH-448 demotion actually excluded a
    # non-empty F-eligible tail on the science arm (guards the 485i silent all-admit no-op).
    lift_engaged_frac = _gp_frac("lift_engaged")
    lift_engaged_met = bool(lift_engaged_frac >= ARM_PASS_FRACTION)
    within_window_frac = _gp_frac("within_window_nonvacuous")
    within_window_met = bool(within_window_frac >= ARM_PASS_FRACTION)

    # Non-degeneracy of gate (a): on the lift-on pair the SCIENCE arm arms (armed>0) while the
    # NONDEGEN_CONTRAST arm does not (armed==0) -- the two arms are not the same random variable.
    contrast_seeds = [
        r for r in guard_passing
        if int(r["arms"][NONDEGEN_CONTRAST_ARM].get("ncl_hold_closure_armed_total", 0)) == 0
        and int(r["arms"][SCIENCE_ARM].get("ncl_hold_closure_armed_total", 0)) > 0
    ]
    gate_a_non_degenerate = bool(len(contrast_seeds) > 0)

    # --- Science (gates b/c) on the guard-passing seeds ---
    mech445_frac = _gp_frac("mech445_commit_intent_met")
    mech445_supported = bool(mech445_frac >= ARM_PASS_FRACTION)
    mech446_frac = _gp_frac("mech446_within_arm_drop_met")
    mech446_supported = bool(mech446_frac >= ARM_PASS_FRACTION)
    cooccur_frac = _gp_frac("cooccur_met")
    cooccur_met = bool(cooccur_frac >= ARM_PASS_FRACTION)

    # F-regime characterization (moderate-F hypothesis; reported, not a hard gate): did the lift
    # reduce F-commits at matched entry-setting? Mean over guard-passing seeds.
    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else 0.0
    mean_repro715_f = _mean([float(r.get("repro715_arm_n_f_commits", 0)) for r in guard_passing])
    mean_science_f = _mean([float(r.get("science_arm_n_f_commits", 0)) for r in guard_passing])
    mean_baseline_f = _mean([float(r.get("liftoff_entryoff_n_f_commits", 0)) for r in guard_passing])
    mean_contrast_f = _mean([float(r.get("lifton_entryoff_n_f_commits", 0)) for r in guard_passing])
    # Positive = lift moderated F at matched entry setting (entry-ON: science vs repro715).
    moderate_f_delta_entry_on = mean_repro715_f - mean_science_f

    # Full readiness = every leg of gate (a) clears (self-route legs; NEVER a false weakens).
    # Adds the M1 lift-engagement leg. F-independence via off_did_not_arm_met + gate_a_non_degenerate.
    readiness_met = bool(
        contact_non_vacuity_met and rule_directed_met and lift_engaged_met
        and armed_and_sustained_met and off_did_not_arm_met and gate_a_non_degenerate
        and within_window_met
    )

    # --- Routing (M1 diagnostic route_reasons -- the hand-off to the f_dominance_conversion_
    #     ceiling /implement-substrate owner). Readiness unmet -> substrate_not_ready_requeue
    #     (scoring-excluded; NEVER a weakens). ---
    if not contact_non_vacuity_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "contact_guard_unmet"
    elif not rule_directed_met:
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "closure_rule_directed_commit_not_formed"
    elif not lift_engaged_met:
        # The MECH-448 demotion silently all-admitted (excluded_count==0) -- the "lift" is inert
        # on this bank, so nothing can be concluded about the moderate-F hypothesis. Needs the
        # 485i envelope-floor calibration, not a de-commit re-run.
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "lift_did_not_engage"
    elif not (armed_and_sustained_met and off_did_not_arm_met and gate_a_non_degenerate):
        # The lift ENGAGED but the science arm still fails to arm on >= 2/3 seeds -> the moderate-F
        # regime did NOT deliver arming. DIAGNOSTIC: the de-commit-release face needs a DISTINCT
        # lever (the selection-face lift does NOT double as the de-commit-release substrate).
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "lift_engaged_but_arming_still_regime_scoped"
    elif not within_window_met:
        # Armed OK but the MECH-446 around-closure window is vacuous -> SD-034 closure fires still
        # do not co-locate with the latch-armed occupancy. DIAGNOSTIC: the de-commit-release face
        # needs a DISTINCT (co-location) lever; the lift moderated F but did not fix co-registration.
        outcome, label = "FAIL", "substrate_not_ready_requeue"
        route_reason = "lift_engaged_arming_ok_but_window_starved"
    else:
        # Readiness (gate a) CLEARED with the lift ON -- the science is now scorable (a genuine
        # PASS or weaken). A PASS here means the selection-face lift DOUBLES AS the de-commit-
        # release substrate (both children unblocked with zero new build).
        if cooccur_met:
            outcome, label = "PASS", "decommit_science_children_cooccur_with_lift"
            route_reason = "lift_engaged_mech445_and_mech446_cooccur_2of3_selection_face_doubles_as_decommit_release"
        else:
            outcome, label = "FAIL", "decommit_science_children_did_not_cooccur"
            route_reason = "readiness_met_but_child_gate(s)_failed_or_disjoint"

    # Per-claim evidence direction. Readiness unmet -> unknown + scoring-excluded (degenerate).
    if not readiness_met:
        evidence_direction = "unknown"
        evidence_direction_per_claim = {"MECH-445": "unknown", "MECH-446": "unknown"}
        non_degenerate = False
        non_degenerate_per_claim = {"MECH-445": False, "MECH-446": False}
        degeneracy_reason = f"substrate_not_ready: {route_reason}"
    else:
        m445_dir = "supports" if mech445_supported else "weakens"
        m446_dir = "supports" if mech446_supported else "weakens"
        evidence_direction_per_claim = {"MECH-445": m445_dir, "MECH-446": m446_dir}
        if mech445_supported and mech446_supported:
            evidence_direction = "supports"
        elif mech445_supported or mech446_supported:
            evidence_direction = "mixed"
        else:
            evidence_direction = "weakens"
        non_degenerate = True
        non_degenerate_per_claim = {"MECH-445": True, "MECH-446": True}
        degeneracy_reason = ""

    print(f"[{EXPERIMENT_TYPE}] contact={contact_non_vacuity_met} (guard {sum(guard_flags)}/{n})"
          f" rule_directed={rule_directed_met} lift_engaged={lift_engaged_met}(f={lift_engaged_frac:.3f})"
          f" armed_sustained={armed_and_sustained_met}"
          f" contrast_no_arm={off_did_not_arm_met}(f={off_did_not_arm_frac:.3f}) zero_f_ctx={zero_f_frac:.3f}"
          f" within_win={within_window_met} non_degen={gate_a_non_degenerate}"
          f" | F-regime: repro715_f={mean_repro715_f:.1f} science_f={mean_science_f:.1f}"
          f" moderate_f_delta={moderate_f_delta_entry_on:.1f}"
          f" | READINESS={readiness_met}"
          f" | m445={mech445_supported}(f={mech445_frac:.3f})"
          f" m446={mech446_supported}(f={mech446_frac:.3f})"
          f" cooccur={cooccur_met}(f={cooccur_frac:.3f})"
          f" -> outcome={outcome} label={label} dir={evidence_direction} route={route_reason}", flush=True)

    overall_pass = bool(outcome == "PASS")

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "rule_directed_met": rule_directed_met,
        "rule_directed_fraction": rule_directed_frac,
        "lift_engaged_met": lift_engaged_met,
        "lift_engaged_fraction": lift_engaged_frac,
        "armed_and_sustained_met": armed_and_sustained_met,
        "armed_and_sustained_fraction": armed_frac,
        "off_did_not_arm_met": off_did_not_arm_met,
        "off_did_not_arm_fraction": off_did_not_arm_frac,
        "gate_a_non_degenerate": gate_a_non_degenerate,
        "zero_f_commits_fraction_context": zero_f_frac,
        "within_window_nonvacuous_met": within_window_met,
        "within_window_nonvacuous_fraction": within_window_frac,
        "readiness_met": readiness_met,
        "mech445_commit_intent_supported": mech445_supported,
        "mech445_commit_intent_fraction": mech445_frac,
        "mech446_within_arm_drop_supported": mech446_supported,
        "mech446_within_arm_drop_fraction": mech446_frac,
        "cooccurrence_met": cooccur_met,
        "cooccurrence_fraction": cooccur_frac,
        "overall_pass": overall_pass,
        "route_reason": route_reason,
        # F-regime characterization (moderate-F hypothesis; means over guard-passing seeds).
        "f_regime": {
            "mean_repro715_arm_n_f_commits": mean_repro715_f,
            "mean_science_arm_n_f_commits": mean_science_f,
            "mean_liftoff_entryoff_n_f_commits": mean_baseline_f,
            "mean_lifton_entryoff_n_f_commits": mean_contrast_f,
            "moderate_f_delta_entry_on": moderate_f_delta_entry_on,
            "moderate_f_delta_note": "positive = the lift reduced F-commits at matched entry-ON "
                                     "setting (repro715 lift-OFF vs science lift-ON)",
        },
        "per_seed_guard_pass": guard_flags,
        "per_seed_lift_engaged": [bool(r.get("lift_engaged", False)) for r in per_seed],
        "per_seed_mech445": [bool(r.get("mech445_commit_intent_met", False)) for r in per_seed],
        "per_seed_mech446": [bool(r.get("mech446_within_arm_drop_met", False)) for r in per_seed],
        "per_seed_cooccur": [bool(r.get("cooccur_met", False)) for r in per_seed],
    }

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "non_degenerate": non_degenerate,
        "non_degenerate_per_claim": non_degenerate_per_claim,
        "degeneracy_reason": degeneracy_reason,
        "acceptance": acceptance,
        "interpretation": {
            "label": label,
            "preconditions": [
                {
                    "name": "foraging_contact_guard",
                    "description": "603n G2+G3: per-seed P2 contact_rate > 0 AND "
                                   "z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout",
                },
                {
                    "name": "closure_rule_directed_commit_formed",
                    "description": "ARM_LIFT_ON_ENTRY_ON forms a goal-active rule-directed commitment "
                                   "the latch can set on (>= RULE_DIRECTED_MIN_TICKS) on >= 2/3 seeds.",
                    "measured": rule_directed_frac,
                    "threshold": ARM_PASS_FRACTION,
                    "met": rule_directed_met,
                    "control": "ARM_LIFT_ON_ENTRY_ON eval loop",
                },
                {
                    "name": "lift_engaged_mech448_demotion_excluded_nonempty_tail",
                    "description": "M1 load-bearing readiness: on ARM_LIFT_ON_ENTRY_ON the MECH-448 "
                                   "demotion was ACTIVE and excluded a non-empty F-eligible tail "
                                   "(demotion_engaged_ticks >= LIFT_ENGAGE_MIN_TICKS) on >= 2/3 "
                                   "seeds. Guards the 485i silent all-admit: below floor the 'lift' "
                                   "is inert (excluded_count==0) so the moderate-F hypothesis is "
                                   "untestable -> substrate_not_ready_requeue (route lift_did_not_"
                                   "engage), NEVER a weakens.",
                    "measured": lift_engaged_frac,
                    "threshold": ARM_PASS_FRACTION,
                    "met": lift_engaged_met,
                    "control": "ARM_LIFT_ON_ENTRY_ON f_eligibility diagnostics vs ARM_LIFT_OFF (0)",
                },
                {
                    "name": "gate_a_armed_and_sustained_entry_driven",
                    "description": "ARM_LIFT_ON_ENTRY_ON arms (ncl_hold_closure_armed_total > 0) AND "
                                   "sustains (max_consecutive_beta_run >= SUSTAIN_MIN_TICKS), AND "
                                   "ARM_LIFT_ON_ENTRY_OFF does NOT arm (non-degenerate: occupancy is "
                                   "entry-driven, not F-driven) -- the 460o gate (a) under the "
                                   "moderate-F regime. Below floor with the lift ENGAGED -> "
                                   "route lift_engaged_but_arming_still_regime_scoped "
                                   "(the de-commit-release face needs a DISTINCT lever); "
                                   "substrate_not_ready_requeue, NEVER a weakens.",
                    "measured": armed_frac,
                    "threshold": ARM_PASS_FRACTION,
                    "met": bool(armed_and_sustained_met and off_did_not_arm_met and gate_a_non_degenerate),
                    "control": "ARM_LIFT_ON_ENTRY_ON vs ARM_LIFT_ON_ENTRY_OFF contrast",
                },
                {
                    "name": "mech446_around_closure_window_nonvacuous",
                    "description": "ARM_LIFT_ON_ENTRY_ON produced >= C2_MIN_WINDOW_EVENTS scored "
                                   "around-closure windows with mean_pre_closure_occ > "
                                   "WITHIN_PRE_OCC_FLOOR on >= 2/3 seeds. Below floor with arming OK "
                                   "-> route lift_engaged_arming_ok_but_window_starved (closure "
                                   "fires still don't co-locate; distinct lever needed) -> "
                                   "substrate_not_ready_requeue (starved, not falsified).",
                    "measured": within_window_frac,
                    "threshold": ARM_PASS_FRACTION,
                    "met": within_window_met,
                    "control": "ARM_LIFT_ON_ENTRY_ON around-closure windows",
                },
            ],
            "criteria_non_degenerate": {
                "gate_a_armed_sustained_contrast": gate_a_non_degenerate,
                "lift_actually_engaged": lift_engaged_met,
            },
            "criteria": [
                {
                    "name": "mech445_commit_intent_2of3",
                    "load_bearing": True,
                    "passed": mech445_supported,
                },
                {
                    "name": "mech446_within_arm_drop_2of3",
                    "load_bearing": True,
                    "passed": mech446_supported,
                },
                {
                    "name": "mech445_mech446_cooccurrence_2of3",
                    "load_bearing": True,
                    "passed": cooccur_met,
                },
            ],
            "science_note": "Move M1 -- re-run of the 715 de-commit science WITH the selection-face "
                            "lift ON. MECH-445 (child A, closure->beta commit-intent): the "
                            "refractory-independent sd034_n_closure_commit_intent counter (counted "
                            "BEFORE the elevate/refractory gate, so the MECH-446 magnitude lever "
                            "cannot zero it) is >= MIN_COMMIT_INTENT on the ARM_LIFT_ON_ENTRY_ON "
                            "arm. MECH-446 (child B): the within-arm mean post-closure occupancy "
                            "drops below the pre-closure occupancy by >= DECOMMIT_MIN_DROP_FRAC. "
                            "Gate (c): the two co-occur on the SAME >= 2/3 seeds. The M1 hypothesis "
                            "is that the MECH-448 demotion (removing F from the committed argmin) "
                            "creates a MODERATE-F regime -- the norm across seeds, not only the "
                            "weak-F seed 44 -- so arming reaches >= 2/3 AND the MECH-446 window "
                            "becomes scorable, unblocking BOTH children with zero new build. A PASS "
                            "means the selection-face lift DOUBLES AS the de-commit-release "
                            "substrate; a lift-engaged readiness abort means it does NOT and a "
                            "distinct de-commit-release lever is owed (see route_reason).",
            "levers_off_note": "The rung-6 NaturalCommitUrgencyRelease (parked 460k DURATION lever) "
                               "and the ARC-108 JOB-2 dopaminergic DRIVER pair (parked 460l) are OFF "
                               "in every arm. The de-commit under test is the INTRINSIC SD-034 "
                               "closure de-commit. The NEW lever vs 715 is the SELECTION-face "
                               "conversion-ceiling lift (MECH-448 + MECH-449), toggled per-arm.",
            "levers_under_test": {
                "grid": "2x2 (use_closure_commit_entry {OFF,ON} x selection-face lift {OFF,ON})",
                "science_arm": SCIENCE_ARM,
                "nondegen_contrast_arm": NONDEGEN_CONTRAST_ARM,
                "repro_715_arm": REPRO_715_ARM,
                "mech448_flag": "use_f_eligibility_demotion (VALIDATED V3-EXQ-689d PASS)",
                "mech448_engagement_diag": "e3.last_score_diagnostics.f_eligibility_excluded_count",
                "mech449_flag": "use_go_nogo_constitution (VALIDATED V3-EXQ-689g PASS; "
                                "auto-wired MECH-260 perseveration axis, no OFC viability injection)",
                "entry_flag": "use_closure_commit_entry (VALIDATED 460o/p PASS)",
                "mech445_counter": "beta_gate.sd034_n_closure_commit_intent (refractory-independent)",
                "mech446_dv": "within-arm mean_pre_closure_occ vs mean_post_closure_occ",
                "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
                "rule_norm_floor": CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
            },
        },
        "per_seed": per_seed,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "supersedes_note": "supersedes V3-EXQ-715 (the levers-OFF de-commit-science falsifier that "
                           "self-routed substrate_not_ready at 1/3 arming). 715a is Move M1 of "
                           "claim_synthesis_MECH-445-446_2026-07-06.md: the SAME de-commit science, "
                           "re-run with the ALREADY-BUILT selection-face conversion-ceiling lift "
                           "(MECH-448 demotion + MECH-449 Go/No-Go) turned ON. Brake-safe (a NEW "
                           "substrate CONFIGURATION, not a same-selector re-derive; the levers were "
                           "OFF in 715 and all of 460h..460l). Do NOT re-author 460d..460l / 715.",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "non_degenerate": result["non_degenerate"],
        "non_degenerate_per_claim": result["non_degenerate_per_claim"],
        "degeneracy_reason": result["degeneracy_reason"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "ethics_preflight": {
            "involves_negative_valence": False,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
            "note": "V3 pre-ethical instrumentation (SENT-0). MECH-094: the closure-entry latch "
                    "SET is a WAKING control-state transition (no replay / no memory-write "
                    "surface); agent.update_residue called with hypothesis_tag=False. "
                    "hypothesis_tag does not apply.",
        },
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> P0 -> "
                     "Stage-H -> P1 -> P2; 460o config) + commitment control-plane (bistable "
                     "BetaGate + SD-034 ClosureOperator + SD-033a LateralPFC + SD-032 "
                     "dACC/salience) + subgoal_mode waypoint tolerance-band completion + "
                     "commitment-closure-control-plane Legs A/B/C + beta-engagement coupling + the "
                     "natural-commit LATCH-HOLD + the CLOSURE-EXCLUSIVE DE-COMMIT EVAL mode "
                     "(closure_exclusive_decommit_eval) ARMED in ALL arms. 2x2 eval grid: the "
                     "F-independent commit-ENTRY primitive (use_closure_commit_entry, VALIDATED "
                     "460o/p) x the SELECTION-FACE conversion-ceiling LIFT (MECH-448 "
                     "use_f_eligibility_demotion VALIDATED 689d + MECH-449 use_go_nogo_constitution "
                     "VALIDATED 689g, auto-wired perseveration axis) toggled per arm. rung-6 "
                     "NaturalCommitUrgencyRelease + ARC-108 JOB-2 driver pair OFF in all arms.",
        "condition": CONDITION_LABEL,
        "method_note": "Move M1 de-commit SCIENCE falsifier WITH the selection-face ceiling-lift ON. "
                       "On the F-independent use_closure_commit_entry substrate, with MECH-448/449 "
                       "ON, do the two SD-034 de-commit-pipeline children CO-OCCUR -- MECH-445 "
                       "(refractory-independent closure->beta commit-intent >= MIN_COMMIT_INTENT) "
                       "AND MECH-446 (within-arm around-closure occupancy drop >= "
                       "DECOMMIT_MIN_DROP_FRAC) on the SAME >= 2/3 seeds -- on the "
                       "ARM_LIFT_ON_ENTRY_ON science arm? Gate (a) READINESS (self-route, NEVER a "
                       "weakens): the MECH-448 demotion ENGAGED (excluded a non-empty F-tail; guards "
                       "485i all-admit) AND ARM_LIFT_ON_ENTRY_ON arms + sustains (armed>0, run>= "
                       "SUSTAIN_MIN_TICKS) AND ARM_LIFT_ON_ENTRY_OFF does not arm (non-degenerate) "
                       "AND a rule-directed commitment formed AND enough around-closure windows "
                       "exist. Any readiness leg unmet -> substrate_not_ready_requeue "
                       "(non_degenerate=false) with an M1 diagnostic route_reason (lift_did_not_"
                       "engage / lift_engaged_but_arming_still_regime_scoped / lift_engaged_arming_"
                       "ok_but_window_starved) telling implement-substrate whether the selection-"
                       "face lift doubles as the de-commit-release substrate. Gates (b)+(c) SCIENCE "
                       "(falsifiable only once readiness clears): the SD-034 closure de-commit "
                       "shortens the sustained occupancy (MECH-446) AND MECH-445 commit-intent "
                       "co-occurs with it -> PASS. PROMOTES NOTHING.",
        "arm_note": "ARMS = " + ", ".join(a["key"] for a in ARMS) + " (2x2: use_closure_commit_"
                    "entry {OFF,ON} x selection-face lift {OFF,ON}). closure-exclusive eval + "
                    "beta-engagement coupling + natural-commit latch-hold ON in all arms. "
                    "ARM_LIFT_OFF_ENTRY_OFF = 715 no-arm baseline; ARM_LIFT_OFF_ENTRY_ON reproduces "
                    "715's 1/3-arming science arm IN-RUN; ARM_LIFT_ON_ENTRY_OFF = lift-on no-arm "
                    "non-degeneracy contrast + F-regime readout; ARM_LIFT_ON_ENTRY_ON = the M1 "
                    "SCIENCE arm (MECH-445 commit-intent + MECH-446 within-arm occupancy-drop both "
                    "measured under the moderate-F regime the lift creates).",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "arm_pass_fraction": ARM_PASS_FRACTION,
            "sustain_min_ticks": SUSTAIN_MIN_TICKS,
            "rule_directed_min_ticks": RULE_DIRECTED_MIN_TICKS,
            "min_commit_intent": MIN_COMMIT_INTENT,
            "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
            "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
            "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
            "closure_window": CLOSURE_WINDOW,
            "window_min_ticks": WINDOW_MIN_TICKS,
            "use_natural_commit_latch_hold": True,
            "closure_exclusive_decommit_eval": True,
            "use_closure_commit_beta_coupling": True,
            "use_natural_commit_urgency_release": False,
            "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
            "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
            "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
            "closure_commit_entry_rule_norm_floor": CLOSURE_COMMIT_ENTRY_RULE_NORM_FLOOR,
            # Selection-face lift (MECH-448 + MECH-449) thresholds (Move M1):
            "lift_engage_min_ticks": LIFT_ENGAGE_MIN_TICKS,
            "f_eligibility_envelope_floor": F_ELIGIBILITY_ENVELOPE_FLOOR,
            "f_eligibility_dn_sigma": F_ELIGIBILITY_DN_SIGMA,
            "gng_safety_floor": GNG_SAFETY_FLOOR,
            "gng_staleness_floor": GNG_STALENESS_FLOOR,
            "gng_perseveration_floor": GNG_PERSEVERATION_FLOOR,
            "gng_viability_floor": GNG_VIABILITY_FLOOR,
            "gng_protect_min_eligible": GNG_PROTECT_MIN_ELIGIBLE,
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "closure_eval_episodes_per_arm": CLOSURE_EVAL_EPISODES,
            "n_eval_arms": len(ARMS),
            "train_steps": TRAIN_STEPS,
            "n_resource_types": N_RESOURCE_TYPES,
            "scaffold_train_harm_pathway": True,
            "scaffold_train_rule_bias_head": True,
            "config_basis": "closure-exclusive de-commit eval (ree-v3 main e52158d) + F-independent "
                            "commit-ENTRY primitive (use_closure_commit_entry, VALIDATED "
                            "V3-EXQ-460o/460p PASS) + selection-face conversion-ceiling lift "
                            "(MECH-448 use_f_eligibility_demotion VALIDATED V3-EXQ-689d PASS; "
                            "MECH-449 use_go_nogo_constitution VALIDATED V3-EXQ-689g PASS)",
        },
        "stage_plan": stage_plan(),
    }
    manifest.update(result)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    _outcome_raw = str(_res["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=_res.get("manifest_path"),
        dry_run=args.dry_run,
    )
