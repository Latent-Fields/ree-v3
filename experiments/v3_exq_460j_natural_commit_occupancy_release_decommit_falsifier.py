"""
V3-EXQ-460j (SUPERSEDES V3-EXQ-460i): SD-034 closure de-commit falsifier on the rung-6
COMMIT/RELEASE-DURATION lever -- the graded natural-commit-occupancy release
(ree_core/policy/natural_commit_urgency.py, NaturalCommitUrgencyRelease) -- now with the
natural-commit LATCH-HOLD substrate amend (use_natural_commit_latch_hold) ARMED in ALL
arms so the OFF baseline sustains a natural-commit beta latch BY CONSTRUCTION.

WHY 460j SUPERSEDES 460i (the gate amend; failure_autopsy_V3-EXQ-460i_2026-06-21,
user-adjudicated Option B "make the OFF baseline actually sustain"):
  460i self-routed substrate_not_ready_requeue at readiness gate 3
  (lever_did_not_shorten_occupancy). The rung-6 lever was correctly ARMED and its arm-site
  reached on NATURAL commits, but it fired ZERO releases because the 460h sustained
  ~2400-step monolithic natural-commit hold DID NOT REPRODUCE -- the active SD-034 de-commit
  control-plane fragmented the beta latch to ~1-tick blips EVEN WITH THE LEVER OFF
  (ARM_LEVER_OFF total_beta_elevated ~= beta_release_events, 415/405 seed 43), so there was
  no sustained occupancy to shorten. AND readiness gate 3's mean_beta_elevated_steps proxy
  is BLIND to sustained-vs-fragmented (it cleared its floor on ~35 fragmented 1-tick
  commits). TWO coupled amends, both landed for 460j:
    (1) SUBSTRATE: the natural-commit LATCH-HOLD (use_natural_commit_latch_hold, agent.py +
        config.py) -- a natural commit arms a hold that RE-ASSERTS the beta latch each tick
        against the de-commit churn, so the OFF baseline sustains by construction. It YIELDS
        to the three principled releases (SD-034 closure de-commit / MECH-091 threat / the
        rung-6 duration release), so the MECH-446 occupancy-drop DV stays measurable.
    (2) READINESS GATE redesign: replace the sustained-blind mean_beta_elevated_steps proxy
        with a SUSTAINED-HOLD proxy -- mean per-commit hold length
        total_beta_elevated/max(1,beta_release_events) (+ longest consecutive beta-elevated
        run, reported) -- above a floor on ARM_LEVER_OFF on >= 2/3 guard seeds, CERTIFYING a
        sustained natural-commit hold IS present (the new readiness gate 3) before the
        lever-shortened gate (4) + the CO_OCCURRENCE DV are allowed.

(Original rung-6 lever, landed ree-v3 main ab2c1a9 2026-06-20.)

WHY THIS IS A NEW EXPERIMENT (not a re-author of 460d/e/f/g/h): 460d-460h iterated the
SELECTION-side / coupling / de-commit-MAGNITUDE machinery of the SD-034
commitment-closure-control-plane. 460h (failure_autopsy lineage) established the
load-bearing finding this run addresses -- the 460h DISJOINT-CERTIFIER problem:

    on STRONG (F-decisive) seeds the bistable beta latch elevates once on a NATURAL
    commit and HOLDS ~2400-2600 steps because nothing releases it (a decisive F-gap =
    "good options" so MECH-342 maintenance-release is silent on the healthy commit, and
    no closure fires so SD-034 is silent). That monolithic natural-commit occupancy
    SWAMPS the SD-034 closure de-commit -> MECH-445 commit-intent
    (sd034_n_closure_commit_intent) fires broadly only where the natural commit is WEAK,
    and MECH-446 de-commit occupancy-drop is measurable only there too, so the two
    certifiers never co-occur on the SAME seed (strong seeds: commit_intent ~0 +
    occupancy never drops; weak seeds: both, but degenerate).

The rung-6 lever (NaturalCommitUrgencyRelease) shortens the F-driven natural-commit
latch occupancy so WEAK-natural-commit becomes the norm ACROSS seeds, dissolving the
disjoint-certifier problem: with the lever ON, MECH-445 commit-intent AND MECH-446
within-arm post-closure occupancy drop should CO-OCCUR on the SAME seeds (>= 2/3).

BG-3 SYNTHESIS divergence D1 (load-bearing): biology does NOT set commitment DURATION
with a fixed refractory clock -- it times the hold with a GRADED BG/pallidal urgency
(Thura/Cisek 2022) and/or makes maintenance co-extensive with the executing action
(Jin 2014). The lever is therefore a GRADED release, NEVER another fixed refractory.
The gap-scaling (gap_entry_sensitivity > 0) is the load-bearing piece: an F-decisive
natural commit accrues release-urgency FASTER, so the strongest-F holds -- the ones
that swamp the de-commit -- are shortened MOST. gap_entry_sensitivity = 0 reduces the
urgency to a flat fixed-rate timeout = the contrasted "another fixed refractory" control
the D1 falsifier compares the gap-scaled lever against.

SUBSTRATE under test (the full 460h/460g lineage stack, IDENTICAL config, on the 603n
foraging-competent substrate) PLUS the rung-6 lever toggled per arm:
  Leg A  env-completion hook (use_closure_env_completion_hook) -> emit_closure.
  Leg B  de-commit refractory + the committed-run-scaled MAGNITUDE lever.
  Leg C  scaffold_train_rule_bias_head (598b REINFORCE in P1) -- trained rule_state.
  beta-engagement coupling (use_closure_commit_beta_coupling).
  rung-6 NaturalCommitUrgencyRelease (use_natural_commit_urgency_release) -- the lever
         under test; arms only on a NATURAL commit (result.committed), NOT on a purely
         closure-coupled elevation.

ARMS (one curriculum build per seed -- lever OFF during training -- then FOUR eval arms,
each a clone of the SAME trained weights re-configured with the arm's lever config; the
lever carries no trainable parameters so the clone is exact, mirroring 460h's
closure-OFF clone). Closure stays ON in EVERY arm; the variable is the natural-commit
lever config:
  ARM_LEVER_OFF     -- lever OFF  = the 460h regime baseline (the disjoint-certifier
                       problem; the sustained-occupancy reference for the readiness gate).
  ARM_GAP_SCALED    -- lever ON, URGENCY mode, gap_entry_sensitivity=1.0. THE PRIMARY
                       load-bearing arm (the D1-faithful graded lever). Co-occurrence
                       scored here.
  ARM_FLAT_RATE     -- lever ON, URGENCY mode, gap_entry_sensitivity=0.0. The D1
                       fixed-refractory control ("graded urgency beats a fixed
                       refractory" is testable as GAP_SCALED vs FLAT_RATE).
  ARM_ACTION_EXTENT -- lever ON, ACTION-EXTENT mode only (urgency mode off). The second
                       D1 rendering (Jin maintenance-co-extensive release) so the
                       falsifier discriminates WHICH rendering lifts.

READINESS / NON-VACUITY (all must clear before co-occurrence is scored; any unmet
self-routes substrate_not_ready_requeue -- NEVER a false weakens):
  (1) 603n foraging contact guard: per-seed P2 contact_rate > 0 AND
      z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.
  (2) rule_bias_head trained (anti-460d-bug gate): P1 rule_bias_pathway_enabled AND mean
      per-candidate |bias| > floor on >= 2/3 seeds.
  (3) OFF-BASELINE SUSTAINED NATURAL-COMMIT HOLD (the V3-EXQ-460j gate-3 REDESIGN,
      replacing 460i's sustained-blind mean_beta_elevated_steps proxy): on ARM_LEVER_OFF
      the mean per-commit hold length total_beta_elevated/max(1,beta_release_events) >=
      SUSTAINED_HOLD_MEAN_FLOOR on >= 2/3 guard seeds (the latch-hold amend established
      the 460h monolithic-hold regime; 460i OFF was ~1.0 and FAILS this -> substrate_not_
      ready_requeue, as intended). max_consecutive_beta_run + ncl_hold_reassert_total
      reported. SAME statistic the 460i proxy was blind to.
  (4) LEVER ACTUALLY SHORTENED OCCUPANCY (the rung-6 non-vacuity gate): on ARM_GAP_SCALED
      the lever fired (ncur_n_releases_total > 0) AND mean beta-latch occupancy dropped
      vs ARM_LEVER_OFF by >= LEVER_OCC_DROP_FRAC (OFF occupancy non-trivial) on >= 2/3
      guard seeds. If the lever did not shorten the (now sustained) occupancy there is
      nothing to test -> substrate_not_ready_requeue.
  (5) closure-coupling non-vacuity (MECH-445, refractory-INDEPENDENT): on ARM_GAP_SCALED
      sd034_n_closure_commit_intent > 0 AND n_sequence_completions > 0 on >= 2/3 guard
      seeds (counted BEFORE the elevate/refractory gate, immune to the MECH-446 magnitude
      lever -- the 460g 36->0 self-defeat the 460h certifier fixed).
  (6) closure-trigger available: ARM_GAP_SCALED n_closures > 0 on >= 2/3 guard seeds.
  (7) within-arm window non-vacuity: ARM_GAP_SCALED produced >= C2_MIN_WINDOW_EVENTS
      scored around-closure windows with mean_pre_occ > WITHIN_PRE_OCC_FLOOR on >= 2/3
      guard seeds (something committed to de-commit).

PRE-REGISTERED ACCEPTANCE (constants; scored only once all seven readiness gates clear):
  CO-OCCURRENCE (load-bearing; MECH-446 scored, MECH-445 precondition): per guard seed,
    on ARM_GAP_SCALED, sd034_n_closure_commit_intent > 0 AND the within-arm around-closure
    occupancy DROP (mean post-closure occupancy < mean pre-closure occupancy with a >=
    DECOMMIT_MIN_DROP_FRAC relative drop over >= C2_MIN_WINDOW_EVENTS windows) BOTH hold
    on the SAME seed. overall PASS = co-occurrence on >= 2/3 guard seeds (dissolves the
    460h disjoint-certifier problem). A fairly-tested no-drop (readiness met, no
    occupancy drop) is a genuine MECH-446 weakens.

D1 / RENDERING CONTRASTS (SECONDARY, REPORTED -- they do NOT gate MECH-446; they make
the D1 mechanism falsifiable):
  - GAP_SCALED vs FLAT_RATE: "graded urgency beats a fixed refractory" -- GAP_SCALED
    co-occurrence seed count >= FLAT_RATE count AND GAP_SCALED mean last_decisiveness_scale
    > 1.0 (gap-scaling active) while FLAT_RATE == 1.0 (flat).
  - urgency (GAP_SCALED / FLAT_RATE) vs action-extent (ARM_ACTION_EXTENT): which D1
    rendering lifts the co-occurrence (reported per-arm).

claim_ids: MECH-446 (scored, de-commit-authority magnitude / occupancy drop), MECH-445
  (closure->beta coupling engagement; the commit-intent non-vacuity precondition,
  readiness gate 4 = its what_would_answer). Re-evaluated from scratch (NOT inherited
  from 460h): MECH-260/MECH-261 NOT re-tagged (No-Go is already a narrow supports; the
  Leg-A hook bypasses mode-conditioning -> MECH-261 unexercised; n_automatic_fires
  reported as a diagnostic).
experiment_purpose: evidence.
supersedes: none (460h ran and stands; this is the NEXT step on the rung-6 lever).

SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.heartbeat.beta_gate import BetaGate  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _sd049_kwargs,
    _sense_with_optional_harm,
    stage_plan,
)

EXPERIMENT_TYPE = "v3_exq_460j_natural_commit_occupancy_release_decommit_falsifier"
QUEUE_ID = "V3-EXQ-460j"
# MECH-446 (de-commit-authority magnitude / occupancy drop) is the SCORED claim (the
# load-bearing within-arm occupancy-drop DV); MECH-445 (closure->beta coupling
# engagement) is the commit-intent non-vacuity precondition (readiness gate 4 = its
# what_would_answer). MECH-260/MECH-261 are intentionally NOT tagged (see docstring).
CLAIM_IDS: List[str] = ["MECH-446", "MECH-445"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "v3_exq_460i_natural_commit_occupancy_release_decommit_falsifier"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "CURRICULUM_BUILT_NATURAL_COMMIT_RELEASE_DECOMMIT_CO_OCCURRENCE"

# --- Goal-pipeline / encoder dims (mirror 603n / 460h exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- SD-034 commitment-closure-control-plane amend knobs (Legs A/B/C; mirror 460h) ---
CLOSURE_DECOMMIT_HOLD_TICKS = 5
CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN = 0.1
CLOSURE_DECOMMIT_HOLD_MAX_TICKS = 60

# --- rung-6 natural-commit-occupancy-release lever (the registered operating point;
#     ree_core/policy/natural_commit_urgency.py defaults) ---
NCUR_RATE = 0.01           # per-tick base urgency increment
NCUR_BOUND = 1.0           # urgency-mode release threshold
NCUR_CAP = 1.5             # hard clamp on urgency (>= bound)
NCUR_ONSET = 0             # grace ticks before urgency accrues
NCUR_GAP_SENSITIVITY = 1.0  # load-bearing gap-scaling for the gap-scaled / action-extent arms

# --- Within-arm around-closure window DV (part b; mirror 460h) ---
CLOSURE_WINDOW = 10
WINDOW_MIN_TICKS = 3
C2_MIN_WINDOW_EVENTS = 2
WITHIN_PRE_OCC_FLOOR = 0.1

# --- rung-6 lever non-vacuity (the occupancy-shortening readiness gate) ---
LEVER_OFF_OCC_FLOOR = 0.5   # OFF mean beta-occupancy must be non-trivial (something to shorten)
LEVER_OCC_DROP_FRAC = 0.15  # gap-scaled mean occupancy must be >= this relative drop below OFF
# --- sustained-hold proxy (the V3-EXQ-460j gate-3 redesign; replaces the
#     sustained-blind mean_beta_elevated_steps proxy of 460i). The natural-commit
#     LATCH-HOLD must establish a SUSTAINED beta-latch occupancy in ARM_LEVER_OFF
#     (mean per-commit hold length = total_beta_elevated/max(1,beta_release_events))
#     well above the 460i fragmented ~1.0-tick regime, certifying the 460h
#     monolithic-hold regime IS present before the co-occurrence DV is allowed. ---
SUSTAINED_HOLD_MEAN_FLOOR = 5.0   # OFF mean per-commit hold length floor (460i OFF was ~1.0)

# --- Curriculum budgets (mirror 603n / 460h exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
CLOSURE_EVAL_EPISODES = 15  # per arm (x4 arms)
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
C1_MIN_CLOSURES = 1
C3_MIN_NOGO = 1
DECOMMIT_MIN_DROP_FRAC = 0.10
RULE_BIAS_MEAN_FLOOR = 0.005

# --- Eval-arm definitions (lever config; closure stays ON in every arm) ---
PRIMARY_ARM = "ARM_GAP_SCALED"
ARMS: List[Dict[str, Any]] = [
    {"key": "ARM_LEVER_OFF",
     "lever": {"on": False, "urgency": True, "action_extent": True, "gap_sensitivity": NCUR_GAP_SENSITIVITY}},
    {"key": "ARM_GAP_SCALED",
     "lever": {"on": True, "urgency": True, "action_extent": False, "gap_sensitivity": NCUR_GAP_SENSITIVITY}},
    {"key": "ARM_FLAT_RATE",
     "lever": {"on": True, "urgency": True, "action_extent": False, "gap_sensitivity": 0.0}},
    {"key": "ARM_ACTION_EXTENT",
     "lever": {"on": True, "urgency": False, "action_extent": True, "gap_sensitivity": NCUR_GAP_SENSITIVITY}},
]


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
    """603n-validated foraging substrate (mirror 460h) + the commitment control-plane +
    the commitment-closure-control-plane amend Legs A/B/C + beta-engagement coupling +
    the DE-COMMIT-AUTHORITY MAGNITUDE lever. The rung-6 natural-commit-occupancy-release
    lever is LEFT OFF here (the trained-base config); it is armed per-arm at eval by
    _clone_arm so all four arms share one trained substrate (the lever carries no
    trainable parameters)."""
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
        closure_decommit_hold_ticks=CLOSURE_DECOMMIT_HOLD_TICKS,  # Leg B base
        closure_decommit_hold_scale_with_run=CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
        closure_decommit_hold_max_ticks=CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
        lateral_pfc_train_rule_bias_head=True,         # Leg C un-zero (GAP-D)
        use_closure_commit_beta_coupling=True,         # beta-engagement coupling
        # rung-6 natural-commit-occupancy-release lever: OFF on the trained base; armed
        # per-arm at eval (the lever has no trainable parameters -> clone is exact).
        use_natural_commit_urgency_release=False,
        # V3-EXQ-460j amend: the natural-commit LATCH-HOLD is ARMED on the base config
        # (so _clone_arm's deepcopy carries it into EVERY arm, incl ARM_LEVER_OFF) ->
        # a natural commit re-asserts the beta latch against the de-commit churn so the
        # OFF baseline sustains by construction (the sustained reference the rung-6
        # release shortens + the gate-3 sustained-hold proxy certifies). The hold yields
        # to the closure de-commit / MECH-091 / the rung-6 release (no trainable params).
        use_natural_commit_latch_hold=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.heartbeat.beta_gate_bistable = True
    return cfg


def _build_closure_env(scaffold_cfg: ScaffoldedSD054OnboardingConfig) -> CausalGridWorldV2:
    """P2-config foraging env (world_obs_dim parity) WITH subgoal_mode + waypoint
    tolerance-band completion so the SD-034 closure operator has completions to fire on
    (mirror 460h)."""
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
    """Clone the SAME trained weights into an agent built with this arm's natural-commit
    lever config. Closure stays ON in every arm (the lever -- not closure -- is the
    variable). The lever + closure carry no trainable parameters, so the state_dict
    loads cleanly (mirrors 460h's _clone_closure_off)."""
    cfg = copy.deepcopy(trained_agent.config)
    lv = arm["lever"]
    cfg.use_natural_commit_urgency_release = bool(lv["on"])
    cfg.natural_commit_release_urgency_mode = bool(lv["urgency"])
    cfg.natural_commit_release_action_extent_mode = bool(lv["action_extent"])
    cfg.natural_commit_gap_entry_sensitivity = float(lv["gap_sensitivity"])
    cfg.natural_commit_urgency_rate = NCUR_RATE
    cfg.natural_commit_urgency_release_bound = NCUR_BOUND
    cfg.natural_commit_urgency_cap = NCUR_CAP
    cfg.natural_commit_urgency_onset_ticks = NCUR_ONSET
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
    return agent


def _around_closure_windows(
    beta_history: List[bool], fire_ticks: List[int]
) -> List[Dict[str, float]]:
    """For each closure fire at tick t, compute the beta-latch occupancy FRACTION in the
    pre-closure window [t-W, t) and the post-closure window (t, t+W], requiring at least
    WINDOW_MIN_TICKS available ticks on each side. Returns one {pre_occ, post_occ} dict
    per scored window (the paired within-arm de-commit datum; mirror 460h)."""
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
    """Longest run of consecutive True (beta-elevated) ticks -- the V3-EXQ-460j
    sustained-hold proxy component, distinguishing a sustained latch from the 460i
    fragmented ~1-tick blips (mean_beta_elevated_steps was blind to this)."""
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


def _eval_arm_behaviour(
    agent: REEAgent,
    env: CausalGridWorldV2,
    scaffold_cfg: ScaffoldedSD054OnboardingConfig,
    device: torch.device,
    n_eps: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    """Frozen-policy eval instrumented for SD-034 closure behaviour (mirror 460h) PLUS
    the rung-6 natural-commit-occupancy-release lever diagnostics (releases, occupancy at
    release, decisiveness scale). Per-episode lever counters are read from
    agent.natural_commit_urgency.get_state() BEFORE the next agent.reset() wipes them."""
    agent.eval()
    world_dim = agent.config.latent.world_dim
    has_closure = agent.closure_operator is not None
    has_dacc = getattr(agent, "dacc", None) is not None
    has_lever = getattr(agent, "natural_commit_urgency", None) is not None
    hook_enabled = bool(getattr(agent.config, "use_closure_env_completion_hook", False))
    feed_harm = scaffold_cfg.scaffold_feed_harm_stream

    closures_pre = int(agent.closure_operator._n_closures) if has_closure else 0
    beta_release_events = 0
    nogo_installed_total = 0
    total_committed_steps = 0
    total_beta_elevated = 0
    n_sequence_completions = 0
    n_hook_fires = 0
    n_closure_coupled_elevations = 0
    n_closure_commit_intent = 0
    around_events: List[Dict[str, float]] = []
    # rung-6 lever accumulators
    ncur_releases_total = 0
    ncur_urgency_releases = 0
    ncur_action_extent_releases = 0
    occ_at_release: List[float] = []
    decisiveness_scales: List[float] = []
    gap_norms: List[float] = []
    # V3-EXQ-460j sustained-hold accumulators
    max_consecutive_beta_run = 0      # longest consecutive elevated run across episodes
    ncl_hold_reassert_total = 0       # natural-commit latch-hold re-assert count

    with torch.no_grad():
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

                if has_closure:
                    fired_now = int(agent.closure_operator._n_closures) - n_closures_before
                    if fired_now > 0 and has_dacc:
                        nogo_installed_total += (
                            len(agent.dacc._action_history) - dacc_hist_before
                        )

                cur_beta = bool(agent.beta_gate.is_elevated)
                beta_history.append(cur_beta)
                if agent.e3._committed_trajectory is not None:
                    total_committed_steps += 1
                if cur_beta:
                    total_beta_elevated += 1
                if prev_beta and not cur_beta:
                    beta_release_events += 1
                prev_beta = cur_beta

                _, _harm, done, info, obs_dict = env.step(action_idx)
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
            # V3-EXQ-460j sustained-hold proxy: longest consecutive elevated run this
            # episode (maxed across episodes) + the natural-commit latch-hold re-assert
            # count (read BEFORE the next agent.reset() wipes it, like the lever state).
            _ep_run = _max_consecutive_true(beta_history)
            if _ep_run > max_consecutive_beta_run:
                max_consecutive_beta_run = _ep_run
            ncl_hold_reassert_total += int(getattr(agent, "_ncl_hold_reassert_count", 0))
            _bstate = agent.beta_gate.get_state()
            n_closure_commit_intent += int(
                _bstate.get("sd034_n_closure_commit_intent", 0)
            )
            n_closure_coupled_elevations += int(
                _bstate.get("sd034_n_closure_coupled_elevations", 0)
            )
            if has_lever:
                lstate = agent.natural_commit_urgency.get_state()
                ep_rel = int(lstate.get("ncur_n_releases_total", 0))
                ncur_releases_total += ep_rel
                ncur_urgency_releases += int(lstate.get("ncur_n_urgency_releases", 0))
                ncur_action_extent_releases += int(
                    lstate.get("ncur_n_action_extent_releases", 0)
                )
                if ep_rel > 0:
                    occ_at_release.append(
                        float(lstate.get("ncur_last_occupancy_at_release", 0.0))
                    )
                    decisiveness_scales.append(
                        float(lstate.get("last_decisiveness_scale", 0.0))
                    )
                    gap_norms.append(float(lstate.get("gap_norm_at_entry", 0.0)))

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
        "sd034_n_closure_commit_intent": n_closure_commit_intent,
        "sd034_n_closure_coupled_elevations": n_closure_coupled_elevations,
        "n_hook_fires": n_hook_fires,
        "n_automatic_fires": max(0, n_closures - n_hook_fires),
        "beta_release_events": beta_release_events,
        "nogo_installed_total": nogo_installed_total,
        "total_committed_steps": total_committed_steps,
        "total_beta_elevated": total_beta_elevated,
        "mean_beta_elevated_steps": total_beta_elevated / max(1, n_eps),
        # V3-EXQ-460j SUSTAINED-HOLD proxy (replaces mean_beta_elevated_steps as the
        # gate-3 occupancy proxy): mean per-commit hold length is BLIND-FREE to
        # sustained-vs-fragmented (460i OFF was ~1.0; the latch-hold lifts it).
        "mean_per_commit_hold": total_beta_elevated / max(1, beta_release_events),
        "max_consecutive_beta_run": max_consecutive_beta_run,
        "ncl_hold_reassert_total": ncl_hold_reassert_total,
        "n_sequence_completions": n_sequence_completions,
        "n_eval_episodes": n_eps,
        "closure_present": has_closure,
        "env_hook_enabled": hook_enabled,
        "n_window_events": n_window_events,
        "mean_pre_closure_occ": mean_pre_occ,
        "mean_post_closure_occ": mean_post_occ,
        # rung-6 lever diagnostics
        "lever_present": has_lever,
        "ncur_n_releases_total": ncur_releases_total,
        "ncur_n_urgency_releases": ncur_urgency_releases,
        "ncur_n_action_extent_releases": ncur_action_extent_releases,
        "ncur_mean_occ_at_release": (
            sum(occ_at_release) / len(occ_at_release) if occ_at_release else 0.0
        ),
        "ncur_mean_decisiveness_scale": (
            sum(decisiveness_scales) / len(decisiveness_scales) if decisiveness_scales else 0.0
        ),
        "ncur_mean_gap_norm_at_entry": (
            sum(gap_norms) / len(gap_norms) if gap_norms else 0.0
        ),
    }


def _within_arm_decommit_drop(arm: Dict[str, Any]) -> bool:
    """MECH-446 within-arm around-closure de-commit DV: mean post-closure occupancy
    fraction < mean pre-closure occupancy fraction with a >= DECOMMIT_MIN_DROP_FRAC
    relative drop, over >= C2_MIN_WINDOW_EVENTS scored windows whose pre-occupancy cleared
    WITHIN_PRE_OCC_FLOOR (mirror 460h)."""
    n_ev = int(arm.get("n_window_events", 0))
    pre = float(arm.get("mean_pre_closure_occ", 0.0))
    post = float(arm.get("mean_post_closure_occ", 0.0))
    if n_ev < C2_MIN_WINDOW_EVENTS or pre <= WITHIN_PRE_OCC_FLOOR:
        return False
    return bool(post < pre and (pre - post) >= DECOMMIT_MIN_DROP_FRAC * pre)


def _within_arm_window_nonvacuous(arm: Dict[str, Any]) -> bool:
    return bool(
        int(arm.get("n_window_events", 0)) >= C2_MIN_WINDOW_EVENTS
        and float(arm.get("mean_pre_closure_occ", 0.0)) > WITHIN_PRE_OCC_FLOOR
    )


def _co_occurs(arm: Dict[str, Any]) -> bool:
    """The dissolution of the 460h disjoint-certifier problem on ONE arm/seed: MECH-445
    commit-intent (sd034_n_closure_commit_intent > 0) AND MECH-446 within-arm post-closure
    occupancy drop BOTH hold."""
    return bool(
        int(arm.get("sd034_n_closure_commit_intent", 0)) > 0
        and _within_arm_decommit_drop(arm)
    )


def _sustained_hold_certified(arm_off: Dict[str, Any]) -> bool:
    """V3-EXQ-460j readiness gate 3 (REDESIGNED from 460i's sustained-blind
    mean_beta_elevated_steps proxy): the ARM_LEVER_OFF baseline must sustain a
    natural-commit beta latch -- mean per-commit hold length
    total_beta_elevated/max(1,beta_release_events) >= SUSTAINED_HOLD_MEAN_FLOOR --
    so the 460h monolithic-hold regime IS certified present (the latch-hold amend
    established it) BEFORE the lever-shortened gate + the co-occurrence DV run. The
    460i fragmented regime had mean per-commit hold ~1.0 and would FAIL this gate
    (self-route substrate_not_ready_requeue), exactly as intended."""
    return bool(
        float(arm_off.get("mean_per_commit_hold", 0.0)) >= SUSTAINED_HOLD_MEAN_FLOOR
    )


def _lever_shortened_occupancy(arm_on: Dict[str, Any], arm_off: Dict[str, Any]) -> bool:
    """rung-6 non-vacuity gate (3): the lever fired on the ON (gap-scaled) arm AND mean
    beta-latch occupancy dropped vs the lever-OFF arm by >= LEVER_OCC_DROP_FRAC, with the
    OFF occupancy non-trivial (there was a sustained natural commit to shorten)."""
    fired = int(arm_on.get("ncur_n_releases_total", 0)) > 0
    on_occ = float(arm_on.get("mean_beta_elevated_steps", 0.0))
    off_occ = float(arm_off.get("mean_beta_elevated_steps", 0.0))
    if off_occ <= LEVER_OFF_OCC_FLOOR:
        return False
    dropped = bool(on_occ < off_occ and (off_occ - on_occ) >= LEVER_OCC_DROP_FRAC * off_occ)
    return bool(fired and dropped)


def _rule_bias_mean(p1) -> float:
    diag = getattr(p1, "rule_bias_diag", None) or {}
    n = int(diag.get("n_bias_samples", 0))
    s = float(diag.get("sum_bias_abs_mean", 0.0))
    return s / n if n > 0 else 0.0


def _empty_arm() -> Dict[str, Any]:
    return {
        "n_closures": 0, "sd034_n_closure_commit_intent": 0,
        "sd034_n_closure_coupled_elevations": 0, "n_hook_fires": 0,
        "n_automatic_fires": 0, "beta_release_events": 0, "nogo_installed_total": 0,
        "total_committed_steps": 0, "total_beta_elevated": 0,
        "mean_beta_elevated_steps": 0.0, "mean_per_commit_hold": 0.0,
        "max_consecutive_beta_run": 0, "ncl_hold_reassert_total": 0,
        "n_sequence_completions": 0,
        "n_eval_episodes": 0, "closure_present": False, "env_hook_enabled": False,
        "n_window_events": 0, "mean_pre_closure_occ": 0.0, "mean_post_closure_occ": 0.0,
        "lever_present": False, "ncur_n_releases_total": 0,
        "ncur_n_urgency_releases": 0, "ncur_n_action_extent_releases": 0,
        "ncur_mean_occ_at_release": 0.0, "ncur_mean_decisiveness_scale": 0.0,
        "ncur_mean_gap_norm_at_entry": 0.0,
    }


def _aborted_seed_record(seed: int, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_num_contact_events": 0,
        "rule_bias_pathway_enabled": False,
        "rule_bias_mean_abs": 0.0,
        "rule_bias_n_train_steps": 0,
        "rule_bias_trained": False,
        "arms": {a["key"]: _empty_arm() for a in ARMS},
        "sustained_hold_certified": False,
        "lever_shortened_occupancy": False,
        "coupling_nonvacuous": False,
        "closure_trigger_available": False,
        "within_window_nonvacuous": False,
        "co_occurs_primary": False,
        "pass": False,
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
    rule_bias_enabled = bool(getattr(p1, "rule_bias_pathway_enabled", False))
    rule_bias_mean = _rule_bias_mean(p1)
    rule_bias_steps = int((getattr(p1, "rule_bias_diag", None) or {}).get("n_train_steps", 0))
    rule_bias_trained = bool(rule_bias_enabled and rule_bias_mean > RULE_BIAS_MEAN_FLOOR)
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" median_last={p1.median_last_window_episode_length:.1f}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
          f" rule_bias_enabled={rule_bias_enabled} rule_bias_mean={rule_bias_mean:.4f}"
          f" rule_bias_steps={rule_bias_steps} rule_bias_trained={rule_bias_trained}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    # Eval all four arms on the SAME trained substrate (clone per arm; lever toggled).
    arms_out: Dict[str, Any] = {}
    for arm in ARMS:
        closure_env = _build_closure_env(scaffold_cfg)
        closure_env.reset()
        print(f"Seed {seed} Condition {arm['key']}", flush=True)
        agent_arm = _clone_arm(agent, device, arm)
        agent_arm.e3._running_variance = float(agent.e3._running_variance)
        arms_out[arm["key"]] = _eval_arm_behaviour(
            agent_arm, closure_env, scaffold_cfg, device, eval_eps, steps_per_ep
        )
        done += eval_eps

    primary = arms_out[PRIMARY_ARM]
    arm_off = arms_out["ARM_LEVER_OFF"]

    # Per-seed readiness / co-occurrence (all keyed on the PRIMARY gap-scaled arm).
    # V3-EXQ-460j gate 3 (NEW): the OFF baseline must sustain a natural-commit hold.
    sustained_hold_certified = _sustained_hold_certified(arm_off)
    lever_shortened = _lever_shortened_occupancy(primary, arm_off)
    coupling_nonvacuous = bool(
        int(primary.get("sd034_n_closure_commit_intent", 0)) > 0
        and int(primary.get("n_sequence_completions", 0)) > 0
    )
    closure_trigger_available = bool(int(primary.get("n_closures", 0)) > 0)
    within_window_nonvacuous = _within_arm_window_nonvacuous(primary)
    co_occurs_primary = _co_occurs(primary)
    seed_pass = bool(co_occurs_primary)

    # SECONDARY D1 / rendering contrasts (reported only).
    co_flat = _co_occurs(arms_out["ARM_FLAT_RATE"])
    co_action = _co_occurs(arms_out["ARM_ACTION_EXTENT"])

    print(f"  [train] arm_eval seed={seed} ep {done}/{total_eps}"
          f" gap_scaled: intent={primary['sd034_n_closure_commit_intent']}"
          f" releases={primary['ncur_n_releases_total']}"
          f" occ={primary['mean_beta_elevated_steps']:.2f} (off_occ={arm_off['mean_beta_elevated_steps']:.2f})"
          f" pre_occ={primary['mean_pre_closure_occ']:.3f} post_occ={primary['mean_post_closure_occ']:.3f}"
          f" win={primary['n_window_events']} decisiveness={primary['ncur_mean_decisiveness_scale']:.3f}"
          f" | off_hold={arm_off['mean_per_commit_hold']:.2f} (run={arm_off['max_consecutive_beta_run']}"
          f" reassert={arm_off['ncl_hold_reassert_total']}) sustained={sustained_hold_certified}"
          f" lever_shortened={lever_shortened} co_occur={co_occurs_primary}"
          f" | flat_co={co_flat} action_co={co_action}", flush=True)
    print(f"verdict: {'PASS' if (guard_pass and seed_pass) else 'FAIL'} seed={seed}"
          f" guard_pass={guard_pass} sustained_hold={sustained_hold_certified}"
          f" lever_shortened={lever_shortened}"
          f" coupling_nonvacuous={coupling_nonvacuous} closure_trigger={closure_trigger_available}"
          f" within_window={within_window_nonvacuous} rule_bias_trained={rule_bias_trained}"
          f" co_occurs={co_occurs_primary}"
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
        "rule_bias_pathway_enabled": rule_bias_enabled,
        "rule_bias_mean_abs": float(rule_bias_mean),
        "rule_bias_n_train_steps": rule_bias_steps,
        "rule_bias_trained": rule_bias_trained,
        "arms": arms_out,
        "sustained_hold_certified": sustained_hold_certified,
        "lever_shortened_occupancy": lever_shortened,
        "coupling_nonvacuous": coupling_nonvacuous,
        "closure_trigger_available": closure_trigger_available,
        "within_window_nonvacuous": within_window_nonvacuous,
        "co_occurs_primary": co_occurs_primary,
        "co_occurs_flat_rate": co_flat,
        "co_occurs_action_extent": co_action,
        "pass": seed_pass,
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

    # Readiness gate (2): rule_bias_head trained.
    rb_flags = [bool(r.get("rule_bias_trained", False)) for r in guard_passing]
    rb_frac = _frac(rb_flags)
    rule_bias_trained_met = bool(rb_frac >= MIN_FRACTION)

    # Readiness gate (3, REDESIGNED for 460j): the ARM_LEVER_OFF baseline SUSTAINS a
    # natural-commit hold (sustained-hold proxy above floor) -- replaces 460i's
    # sustained-blind mean_beta_elevated_steps gate. Certifies the 460h monolithic-hold
    # regime IS present (the latch-hold amend established it) before anything is scored.
    sh_flags = [bool(r.get("sustained_hold_certified", False)) for r in guard_passing]
    sh_frac = _frac(sh_flags)
    sustained_hold_met = bool(sh_frac >= MIN_FRACTION)

    # Readiness gate (4): the rung-6 lever actually shortened the (now sustained) occupancy.
    ls_flags = [bool(r.get("lever_shortened_occupancy", False)) for r in guard_passing]
    ls_frac = _frac(ls_flags)
    lever_shortened_met = bool(ls_frac >= MIN_FRACTION)

    # Readiness gate (4): closure-coupling non-vacuity (MECH-445 commit-intent).
    cp_flags = [bool(r.get("coupling_nonvacuous", False)) for r in guard_passing]
    cp_frac = _frac(cp_flags)
    coupling_nonvacuity_met = bool(cp_frac >= MIN_FRACTION)

    # Readiness gate (5): closure-trigger available.
    ct_flags = [bool(r.get("closure_trigger_available", False)) for r in guard_passing]
    ct_frac = _frac(ct_flags)
    closure_trigger_available_met = bool(ct_frac >= MIN_FRACTION)

    # Readiness gate (6): within-arm window non-vacuity.
    ww_flags = [bool(r.get("within_window_nonvacuous", False)) for r in guard_passing]
    ww_frac = _frac(ww_flags)
    within_window_met = bool(ww_frac >= MIN_FRACTION)

    co_flags = [bool(r.get("co_occurs_primary", False)) for r in guard_passing]
    n_pass = sum(1 for f in co_flags if f)
    pass_frac = _frac(co_flags)
    overall_criteria_pass = bool(pass_frac >= MIN_FRACTION)

    # SECONDARY D1 contrasts (reported only): does the gap-scaled (graded) lever beat the
    # flat-rate (fixed-refractory) control? And which D1 rendering lifts the co-occurrence?
    co_flat_flags = [bool(r.get("co_occurs_flat_rate", False)) for r in guard_passing]
    co_action_flags = [bool(r.get("co_occurs_action_extent", False)) for r in guard_passing]
    n_co_gap = n_pass
    n_co_flat = sum(1 for f in co_flat_flags if f)
    n_co_action = sum(1 for f in co_action_flags if f)
    gap_decisiveness = [
        float(r.get("arms", {}).get(PRIMARY_ARM, {}).get("ncur_mean_decisiveness_scale", 0.0))
        for r in guard_passing
    ]
    flat_decisiveness = [
        float(r.get("arms", {}).get("ARM_FLAT_RATE", {}).get("ncur_mean_decisiveness_scale", 0.0))
        for r in guard_passing
    ]
    mean_gap_decisiveness = (sum(gap_decisiveness) / len(gap_decisiveness)) if gap_decisiveness else 0.0
    mean_flat_decisiveness = (sum(flat_decisiveness) / len(flat_decisiveness)) if flat_decisiveness else 0.0
    d1_graded_beats_fixed = bool(
        n_co_gap >= n_co_flat and mean_gap_decisiveness > 1.0 and mean_flat_decisiveness <= 1.0001
    )

    readiness_all_met = bool(
        contact_non_vacuity_met and rule_bias_trained_met and sustained_hold_met
        and lever_shortened_met and coupling_nonvacuity_met
        and closure_trigger_available_met and within_window_met
    )

    if not contact_non_vacuity_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "contact_guard_unmet"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not rule_bias_trained_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "rule_bias_head_untrained"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not sustained_hold_met:
        # The ARM_LEVER_OFF baseline did NOT sustain a natural-commit hold (mean
        # per-commit hold below floor = the 460i fragmented regime persists) -> the
        # latch-hold amend did not establish the 460h monolithic-hold regime, so there
        # is nothing for the rung-6 lever to shorten = substrate not ready (NEVER a
        # false weakens). The redesigned gate-3 that the 460i autopsy prescribed.
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "off_baseline_not_sustained"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not lever_shortened_met:
        # The rung-6 lever did not shorten the natural-commit occupancy on the gap-scaled
        # arm -> the disjoint-certifier regime is unchanged, nothing to test = substrate
        # not ready (NEVER a false weakens).
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "lever_did_not_shorten_occupancy"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not coupling_nonvacuity_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "closure_coupling_not_engaged"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not closure_trigger_available_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "closure_trigger_unavailable"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    elif not within_window_met:
        outcome, readiness_route, route_reason = "FAIL", "substrate_not_ready_requeue", "within_arm_windows_vacuous"
        direction_map = {cid: "non_contributory" for cid in CLAIM_IDS}
        overall_direction = "non_contributory"
    else:
        # All six readiness gates clear -> the co-occurrence DV is interpretable.
        outcome = "PASS" if overall_criteria_pass else "FAIL"
        readiness_route = ("natural_commit_release_dissolves_disjoint_certifier"
                           if overall_criteria_pass else "decommit_co_occurrence_open")
        route_reason = ("mech445_446_co_occur_majority_met" if overall_criteria_pass
                        else "co_occurrence_dv_unmet_genuine_weakens")
        direction_map = {
            # MECH-446 SCORED on the load-bearing within-arm occupancy-drop component of
            # the co-occurrence (a fairly-tested no-drop is its own falsifier = weakens).
            "MECH-446": "supports" if overall_criteria_pass else "weakens",
            # MECH-445 supports: reaching the scoring branch REQUIRES readiness gate (4)
            # -- sd034_n_closure_commit_intent > 0 on >= 2/3 -- which IS its
            # what_would_answer PASS condition. NEVER weakens here (a coupling-gate failure
            # self-routes substrate_not_ready_requeue above).
            "MECH-445": "supports",
        }
        overall_direction = "supports" if overall_criteria_pass else "mixed"

    print(f"[{EXPERIMENT_TYPE}] contact={contact_non_vacuity_met}"
          f" (guard {sum(guard_flags)}/{n}) rule_bias_trained={rule_bias_trained_met} (frac={rb_frac:.3f})"
          f" sustained_hold={sustained_hold_met} (frac={sh_frac:.3f})"
          f" lever_shortened={lever_shortened_met} (frac={ls_frac:.3f})"
          f" coupling={coupling_nonvacuity_met} (frac={cp_frac:.3f})"
          f" closure_trigger={closure_trigger_available_met} (frac={ct_frac:.3f})"
          f" within_window={within_window_met} (frac={ww_frac:.3f})"
          f" | co_occur gap={n_co_gap} flat={n_co_flat} action={n_co_action}"
          f" d1_graded_beats_fixed={d1_graded_beats_fixed}"
          f" criteria_pass={overall_criteria_pass} ({n_pass}/{len(guard_passing)})"
          f" -> outcome={outcome} route={readiness_route}", flush=True)
    for cid in CLAIM_IDS:
        print(f"[{EXPERIMENT_TYPE}] per_claim {cid}={direction_map[cid]}", flush=True)

    acceptance = {
        "contact_non_vacuity_met": contact_non_vacuity_met,
        "guard_fraction": guard_frac,
        "n_guard_passing_seeds": len(guard_passing),
        "rule_bias_trained_met": rule_bias_trained_met,
        "rule_bias_trained_fraction": rb_frac,
        "sustained_hold_met": sustained_hold_met,
        "sustained_hold_fraction": sh_frac,
        "lever_shortened_occupancy_met": lever_shortened_met,
        "lever_shortened_occupancy_fraction": ls_frac,
        "coupling_nonvacuity_met": coupling_nonvacuity_met,
        "coupling_nonvacuity_fraction": cp_frac,
        "closure_trigger_available_met": closure_trigger_available_met,
        "closure_trigger_fraction": ct_frac,
        "within_window_met": within_window_met,
        "within_window_fraction": ww_frac,
        "co_occurrence_pass_fraction": pass_frac,
        "n_seeds_co_occur_gap_scaled": n_co_gap,
        "n_seeds_co_occur_flat_rate": n_co_flat,
        "n_seeds_co_occur_action_extent": n_co_action,
        "mean_gap_scaled_decisiveness": mean_gap_decisiveness,
        "mean_flat_rate_decisiveness": mean_flat_decisiveness,
        "d1_graded_beats_fixed": d1_graded_beats_fixed,
        "overall_pass": bool(readiness_all_met and overall_criteria_pass),
        "per_seed_guard_pass": guard_flags,
        "per_seed_co_occur_gap_scaled": [bool(r.get("co_occurs_primary", False)) for r in per_seed],
        "route_reason": route_reason,
    }

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": direction_map,
        "acceptance": acceptance,
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "preconditions": [
                {
                    "name": "foraging_contact_guard",
                    "description": "603n G2+G3: per-seed P2 contact_rate > 0 AND "
                                   "z_goal_norm_at_contact_peak > 0.4 on >= 2/3 seeds.",
                    "control": "scaffolded_sd054_onboarding run_p2 consumption-gated readout.",
                    "measured": guard_frac,
                    "threshold": MIN_FRACTION,
                    "met": contact_non_vacuity_met,
                },
                {
                    "name": "rule_bias_head_trained",
                    "description": "Leg C readiness (anti-460d-bug gate): P1 "
                                   "rule_bias_pathway_enabled AND mean per-candidate |bias| "
                                   "> floor on >= 2/3 seeds. Below floor -> substrate_not_"
                                   "ready_requeue, NEVER a weakens.",
                    "control": "P1OnboardingResult.rule_bias_diag mean |bias|.",
                    "measured": rb_frac,
                    "threshold": MIN_FRACTION,
                    "met": rule_bias_trained_met,
                },
                {
                    "name": "off_baseline_sustained_natural_commit_hold",
                    "description": "V3-EXQ-460j gate 3 (REDESIGNED from 460i's "
                                   "sustained-blind mean_beta_elevated_steps proxy): the "
                                   "ARM_LEVER_OFF baseline must SUSTAIN a natural-commit "
                                   "beta latch -- mean per-commit hold length "
                                   "total_beta_elevated/max(1,beta_release_events) >= "
                                   "SUSTAINED_HOLD_MEAN_FLOOR (460i OFF was ~1.0) -- on "
                                   ">= 2/3 guard seeds, certifying the latch-hold amend "
                                   "established the 460h monolithic-hold regime BEFORE "
                                   "the lever-shortened gate + co-occurrence DV. The 460i "
                                   "fragmented regime FAILS this (self-routes "
                                   "substrate_not_ready_requeue), as intended. SAME "
                                   "statistic the 460i mean_beta_elevated_steps proxy was "
                                   "blind to (mean per-commit hold, not per-episode mean).",
                    "control": "ARM_LEVER_OFF mean_per_commit_hold (+ max_consecutive_beta_run "
                               "+ ncl_hold_reassert_total reported).",
                    "measured": sh_frac,
                    "threshold": MIN_FRACTION,
                    "met": sustained_hold_met,
                },
                {
                    "name": "natural_commit_lever_shortened_occupancy",
                    "description": "rung-6 non-vacuity (the lever must DO something): on "
                                   "ARM_GAP_SCALED ncur_n_releases_total > 0 AND mean "
                                   "beta-latch occupancy dropped vs ARM_LEVER_OFF by >= "
                                   "LEVER_OCC_DROP_FRAC (OFF occupancy > LEVER_OFF_OCC_FLOOR) "
                                   "on >= 2/3 guard seeds. If the lever did not shorten the "
                                   "natural-commit occupancy the 460h disjoint-certifier "
                                   "regime is unchanged -> substrate_not_ready_requeue, NEVER "
                                   "a false weakens.",
                    "control": "ARM_GAP_SCALED ncur_n_releases_total + mean_beta_elevated_steps "
                               "vs ARM_LEVER_OFF mean_beta_elevated_steps.",
                    "measured": ls_frac,
                    "threshold": MIN_FRACTION,
                    "met": lever_shortened_met,
                },
                {
                    "name": "closure_coupling_nonvacuous_refractory_independent",
                    "description": "MECH-445 coupling engagement (refractory-INDEPENDENT): "
                                   "ARM_GAP_SCALED sd034_n_closure_commit_intent > 0 AND "
                                   "n_sequence_completions > 0 on >= 2/3 guard seeds (counted "
                                   "BEFORE the elevate/refractory gate; immune to the MECH-446 "
                                   "magnitude lever). Below floor -> substrate_not_ready_"
                                   "requeue, NEVER a false weakens.",
                    "control": "ARM_GAP_SCALED sd034_n_closure_commit_intent.",
                    "measured": cp_frac,
                    "threshold": MIN_FRACTION,
                    "met": coupling_nonvacuity_met,
                },
                {
                    "name": "closure_trigger_available_count",
                    "description": "ARM_GAP_SCALED n_closures > 0 on >= 2/3 guard seeds. "
                                   "Below floor -> substrate_not_ready_requeue.",
                    "control": "ARM_GAP_SCALED n_closures > 0 (Leg-A hook + trained head).",
                    "measured": ct_frac,
                    "threshold": MIN_FRACTION,
                    "met": closure_trigger_available_met,
                },
                {
                    "name": "within_arm_window_nonvacuous",
                    "description": "ARM_GAP_SCALED produced >= C2_MIN_WINDOW_EVENTS scored "
                                   "around-closure windows with mean_pre_occ > "
                                   "WITHIN_PRE_OCC_FLOOR on >= 2/3 guard seeds. Below floor "
                                   "-> substrate_not_ready_requeue (nothing to de-commit).",
                    "control": "ARM_GAP_SCALED n_window_events + mean_pre_closure_occ.",
                    "measured": ww_frac,
                    "threshold": MIN_FRACTION,
                    "met": within_window_met,
                },
            ],
            "criteria": [
                {"name": "CO_OCCURRENCE_gap_scaled_mech445_and_mech446",
                 "load_bearing": True, "passed": overall_criteria_pass},
                {"name": "D1_graded_beats_fixed_refractory",
                 "load_bearing": False, "passed": d1_graded_beats_fixed},
            ],
            "criteria_non_degenerate": {
                # The co-occurrence DV is non-degenerate iff all six readiness gates cleared
                # (contact, trained head, lever shortened occupancy, coupling engaged,
                # closure fired, scored windows with committed pre-occupancy) -- otherwise
                # the occupancy delta is structurally uninterpretable.
                "CO_OCCURRENCE_gap_scaled_mech445_and_mech446": readiness_all_met,
                "D1_graded_beats_fixed_refractory": readiness_all_met,
            },
            "co_occurrence_dv": {
                "definition": "Per guard seed on ARM_GAP_SCALED: sd034_n_closure_commit_intent "
                              "> 0 (MECH-445) AND within-arm around-closure occupancy DROP "
                              "(MECH-446: mean post-closure occupancy < mean pre-closure "
                              "occupancy with a >= DECOMMIT_MIN_DROP_FRAC relative drop over "
                              ">= C2_MIN_WINDOW_EVENTS windows with mean_pre_occ > "
                              "WITHIN_PRE_OCC_FLOOR) BOTH hold on the SAME seed. overall PASS "
                              "= co-occurrence on >= 2/3 guard seeds = dissolution of the 460h "
                              "disjoint-certifier problem.",
                "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
                "closure_window": CLOSURE_WINDOW,
                "window_min_ticks": WINDOW_MIN_TICKS,
                "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
                "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
                "lever_off_occ_floor": LEVER_OFF_OCC_FLOOR,
                "lever_occ_drop_frac": LEVER_OCC_DROP_FRAC,
            },
            "rung6_lever": {
                "module": "ree_core/policy/natural_commit_urgency.py NaturalCommitUrgencyRelease",
                "natural_commit_urgency_rate": NCUR_RATE,
                "natural_commit_urgency_release_bound": NCUR_BOUND,
                "natural_commit_urgency_cap": NCUR_CAP,
                "natural_commit_gap_entry_sensitivity_gap_scaled": NCUR_GAP_SENSITIVITY,
                "natural_commit_urgency_onset_ticks": NCUR_ONSET,
                "note": "Arms on a NATURAL commit (result.committed) only; a purely "
                        "closure-coupled elevation is NOT armed. The gap-scaling "
                        "(gap_entry_sensitivity > 0) is the load-bearing D1 piece; "
                        "ARM_FLAT_RATE (sensitivity 0) is the fixed-refractory control.",
            },
            "d1_contrast": {
                "n_co_occur_gap_scaled": n_co_gap,
                "n_co_occur_flat_rate": n_co_flat,
                "n_co_occur_action_extent": n_co_action,
                "mean_gap_scaled_decisiveness": mean_gap_decisiveness,
                "mean_flat_rate_decisiveness": mean_flat_decisiveness,
                "graded_beats_fixed": d1_graded_beats_fixed,
                "note": "SECONDARY / REPORTED -- does NOT gate MECH-446. 'graded urgency "
                        "beats a fixed refractory' = ARM_GAP_SCALED co-occurrence count >= "
                        "ARM_FLAT_RATE count AND gap-scaled mean decisiveness_scale > 1.0 "
                        "(gap-scaling active) while flat == 1.0. ARM_ACTION_EXTENT reports "
                        "the second D1 rendering (Jin maintenance-co-extensive release).",
            },
            "amend_legs_under_test": {
                "leg_a_env_completion_hook": "REEAgent.notify_env_completion -> emit_closure.",
                "leg_b_decommit_hold_magnitude": "committed-run-scaled refractory.",
                "leg_c_trained_rule_bias_head": "scaffold_train_rule_bias_head (598b REINFORCE in P1).",
                "beta_engagement_coupling": "use_closure_commit_beta_coupling.",
                "rung6_natural_commit_release": "use_natural_commit_urgency_release (per-arm).",
            },
            "mech261_note": "MECH-261 (mode-conditioning) NOT tagged -- the Leg-A env-completion "
                            "hook bypasses mode-conditioning (n_automatic_fires reported as a "
                            "diagnostic). MECH-260 (No-Go) NOT re-tagged (already narrow "
                            "supports; nogo_installed reported only).",
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
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> "
                     "P0 -> Stage-H -> P1 -> P2; 603n config) + commitment control-plane "
                     "(bistable BetaGate + SD-034 ClosureOperator + SD-033a LateralPFC + "
                     "SD-032 dACC/salience) + subgoal_mode waypoint tolerance-band completion "
                     "+ commitment-closure-control-plane Legs A/B/C + beta-engagement coupling "
                     "+ the DE-COMMIT-AUTHORITY MAGNITUDE lever + the REFRACTORY-INDEPENDENT "
                     "commit-intent certifier + the rung-6 NaturalCommitUrgencyRelease lever "
                     "(use_natural_commit_urgency_release) toggled per arm + the V3-EXQ-460j "
                     "natural-commit LATCH-HOLD (use_natural_commit_latch_hold) ARMED in ALL "
                     "arms so the OFF baseline sustains a natural-commit beta latch by "
                     "construction (the gate-3 sustained-hold redesign).",
        "condition": CONDITION_LABEL,
        "method_note": "460h-successor (NOT a supersede; 460h ran and stands). 460h "
                       "established the disjoint-certifier problem: on STRONG (F-decisive) "
                       "seeds the bistable beta latch holds ~2400-2600 steps (nothing releases "
                       "the healthy natural commit), so MECH-445 commit-intent and MECH-446 "
                       "de-commit occupancy-drop never co-occur on the SAME seed. This run arms "
                       "the rung-6 graded natural-commit-occupancy release (BG-3 D1: graded "
                       "BG/pallidal urgency, NOT a fixed refractory) on top of the full 460h "
                       "substrate and tests whether the lever shortens the F-driven natural "
                       "commit so MECH-445 + MECH-446 CO-OCCUR on >= 2/3 seeds. Four eval arms "
                       "(one trained substrate per seed, lever toggled at eval via clone): "
                       "ARM_LEVER_OFF (460h baseline) / ARM_GAP_SCALED (graded urgency, PRIMARY) "
                       "/ ARM_FLAT_RATE (gap_entry_sensitivity=0, the D1 fixed-refractory "
                       "control) / ARM_ACTION_EXTENT (Jin rendering). Six readiness gates "
                       "self-route substrate_not_ready_requeue when unmet -- never a false "
                       "weakens (incl. the new lever-shortened-occupancy gate). claim_ids: "
                       "MECH-446 scored (within-arm occupancy-drop component), MECH-445 the "
                       "commit-intent precondition; MECH-260/MECH-261 NOT re-tagged.",
        "arm_note": "ARMS = " + ", ".join(a["key"] for a in ARMS) + ". Closure ON in every "
                    "arm; the variable is the natural-commit lever config. ARM_GAP_SCALED is "
                    "the load-bearing primary; ARM_LEVER_OFF is the occupancy reference + the "
                    "460h baseline; ARM_FLAT_RATE / ARM_ACTION_EXTENT are the D1 contrasts.",
        "pre_registered_thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE,
            "contact_gate": CONTACT_GATE,
            "min_fraction": MIN_FRACTION,
            "c1_min_closures": C1_MIN_CLOSURES,
            "c3_min_nogo": C3_MIN_NOGO,
            "decommit_min_drop_frac": DECOMMIT_MIN_DROP_FRAC,
            "closure_window": CLOSURE_WINDOW,
            "window_min_ticks": WINDOW_MIN_TICKS,
            "c2_min_window_events": C2_MIN_WINDOW_EVENTS,
            "within_pre_occ_floor": WITHIN_PRE_OCC_FLOOR,
            "rule_bias_mean_floor": RULE_BIAS_MEAN_FLOOR,
            "lever_off_occ_floor": LEVER_OFF_OCC_FLOOR,
            "lever_occ_drop_frac": LEVER_OCC_DROP_FRAC,
            "sustained_hold_mean_floor": SUSTAINED_HOLD_MEAN_FLOOR,
            "use_natural_commit_latch_hold": True,
            "closure_decommit_hold_ticks": CLOSURE_DECOMMIT_HOLD_TICKS,
            "closure_decommit_hold_scale_with_run": CLOSURE_DECOMMIT_HOLD_SCALE_WITH_RUN,
            "closure_decommit_hold_max_ticks": CLOSURE_DECOMMIT_HOLD_MAX_TICKS,
            "ncur_rate": NCUR_RATE,
            "ncur_bound": NCUR_BOUND,
            "ncur_cap": NCUR_CAP,
            "ncur_gap_sensitivity_gap_scaled": NCUR_GAP_SENSITIVITY,
            "ncur_onset_ticks": NCUR_ONSET,
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
            "config_basis": "V3-EXQ-460h substrate + rung-6 natural-commit-occupancy-release "
                            "lever (ree-v3 main ab2c1a9, 2026-06-20)",
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
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
        )
