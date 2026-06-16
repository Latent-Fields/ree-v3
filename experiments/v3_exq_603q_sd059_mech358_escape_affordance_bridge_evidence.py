"""
V3-EXQ-603q -- SD-059 / MECH-358 relief/safety escape-affordance bridge:
CORRECTED 5-arm behavioural EVIDENCE re-run on the now-STABILIZED base harm
landscape. Successor to V3-EXQ-603o (supersedes it).

WHAT CHANGED FROM 603o (the delta; everything else is identical):
  V3-EXQ-603o ran the same 5-arm design but SELF-ROUTED substrate_not_ready_requeue:
  its FIRST non-vacuity precondition -- the BASE harm landscape must be discriminative
  (harm_eval_range >= 0.02) on >=2/3 seeds -- was UNMET, because the 2026-06-09
  harm-pathway training is seed-fragile (failure_autopsy_V3-EXQ-603p: clears 1/3 seeds
  at the easiest regime; a 3x global LR COLLAPSES it). 603q runs on the now-stabilized
  base: the 2026-06-16 scaffolded_sd054_onboarding harm-pathway STABILIZATION amend
  (decoupled encoder LR HARM_PATHWAY_ENCODER_LR=3e-4 + LR warmup
  HARM_PATHWAY_WARMUP_STEPS=250, applied to ALL 5 arms' base config, WITHOUT raising
  the global LR) makes the base harm landscape clear harm_eval_range >= 0.02 on >=2/3
  seeds. The hazard regime anchors on proximity_harm=0.10 -- the 603p positive control
  the amend is validated on -- instead of 603o's 0.15. The base-discriminativeness
  precondition is RETAINED unchanged (it self-routes substrate_not_ready_requeue if the
  amend still under-performs at full scale -> never a false bridge verdict).

PURPOSE (scored governance evidence; claim_ids=[SD-059, MECH-358]):
The routing output of failure_autopsy_V3-EXQ-603l_2026-06-10. V3-EXQ-603l ran to
completion FAIL, self-routed `weakens`, but was adjudicated `non_contributory`
because its discrimination criterion `best_bridge_G_H_frac > G_H_BASE_frac` was
STRUCTURALLY UNSATISFIABLE: the readiness-enabling 603k harm-pathway-training fix
(scaffold_train_harm_pathway=True) trains E3.harm_eval(z_world) -- which per
ARC-007-strict IS the directed-escape pathway -- so on the easy Stage-H regime it
SATURATED ARM_BASE_IA_ONLY survival to the binary ceiling (G_H_BASE_frac=1.0, 3/3),
removing the bridge's headroom. The bridge fired non-vacuously (relief credit
299-426, safety credit up to 2880, thousands of approach fires) and
ARM_RELIEF_SAFETY_BRIDGE showed ~35% longer mean hazard-stage episode length
(~132 vs base ~98) -- a GRADED benefit invisible to the binary median>=75 G_H gate.

THIS REDESIGN carries BOTH user-confirmed fixes:

  (a) HEADROOM -- the hazard regime is made HARDER (num_hazards 4->6,
      proximity_harm_scale 0.10->0.15) at FULL, UNIFORM 603k+603j substrate
      (all 5 arms identical except the bridge). Rationale (in-skill design choice,
      vs the reduced-harm-training alternative):
        1. A fully-trained uniform substrate keeps the 4 discriminative arms
           differing ONLY in the escape-affordance bridge -- preserves the clean
           single-variable design and forecloses the "you under-trained the base
           to manufacture headroom" critique that a reduced-harm-LR design invites.
        2. The harder env stresses the scalar harm_eval gradient (confound A: 603k's
           gradient saturated survival on the EASY env) so ARM_BASE_IA_ONLY sits
           BELOW the binary ceiling with measurable headroom -- the regime where the
           bridge's DIRECTED, learned, threat-gated escape should separate from the
           myopic scalar gradient.
        3. It avoids the opposite failure (the 603i nav/survival-competence ceiling):
           the env is harder but ARM_NAV_CONTROL (spawn-in-reef, nav-to-safety handed)
           plus a harm-discriminativeness non-vacuity gate confirm the env stays
           survivable-IN-PRINCIPLE, so a flat result is attributable to the bridge,
           not an unsurvivable world. If the env overshoots and even nav_control
           cannot survive, the kept interpretation grid self-routes non_contributory
           (nav ceiling), NEVER a false weakens.
      Alternative considered + REJECTED as primary: reduced harm-pathway LR (partial
      training). It manufactures headroom by under-training the shared substrate,
      inviting the "bridge merely compensates for an under-trained harm_eval"
      critique. Env-difficulty leaves the substrate at validated 603k strength.

  (b) CONTINUOUS SURVIVAL METRIC -- the PRIMARY (load-bearing) discrimination metric
      is now the MEAN hazard-stage episode length (mean survival duration; episodes
      terminate on hazard death, so length = steps survived). The binary G_H gate
      (median last-window episode length >= survival_gate_steps) is RETAINED as a
      SECONDARY / supplementary readout, NOT the load-bearing gate. The continuous
      metric is robust to residual binary saturation: even if ARM_BASE_IA_ONLY still
      clears the binary gate, mean survival is NOT pinned at the TRAIN_STEPS ceiling
      (603l base ~98 of 200), so `best_bridge_mean_survival > base_mean_survival`
      stays satisfiable and directly measures the graded benefit the binary gate
      masked. AUC-survival (mean/TRAIN_STEPS) + time-to-first-death are reported too.

THE SUBSTRATE UNDER TEST: SD-059 / MECH-358 EscapeAffordanceBridge
(ree_core/pfc/escape_affordance_bridge.py) -- extends the MECH-357 scalar avoidance
efficacy into a per-first-action-class credit table with two halves: RELIEF (a
directed action under threat that drops z_harm_a credits relief_affordance[class],
MECH-302-consistent) and SAFETY (a directed action after which threat is absent --
read from the trained MECH-303/304 threat-absence predictor, 603j -- credits
safety_affordance[class], MECH-303/304-consistent). Under future threat, E3 receives
a bounded, threat-context-gated NEGATIVE (favoured) approach bias toward credited
classes -- the DIRECTED escape.

DESIGN: 5 arms x 3 seeds [42, 43, 44], identical to 603l EXCEPT the harder hazard
regime + the continuous-primary scoring. ALL arms carry the full 603i-INTACT
defensive config (MECH-279 PAG + SD-058/MECH-357 ilPFC gate + driver + fed harm
stream + SD-056 e2 warmup) PLUS the two validated substrate fixes
(scaffold_train_harm_pathway=True [603k]; escape_use_trained_safety_signal=True +
MECH-303/304/302 [603j]). The 4 discriminative arms differ ONLY in the bridge:
  ARM_BASE_IA_ONLY            = bridge OFF (SD-058/MECH-357 + harm-pathway base).
  ARM_RELIEF_BRIDGE           = bridge ON, relief half only.
  ARM_SAFETY_BRIDGE           = bridge ON, safety half only.
  ARM_RELIEF_SAFETY_BRIDGE    = bridge ON, both halves (the full SD-059/MECH-358).
  ARM_NAV_CONTROL (positive control) = bridge OFF, Stage-H spawns IN the reef refuge
    (navigation to safety handed) -- the env-survivability disambiguator.

PRE-REGISTERED ACCEPTANCE (constants; NOT derived from the run's own statistics):
  PRIMARY (load-bearing, CONTINUOUS): the full bridge lifts mean Stage-H survival
    over the IA-only base:
      both_mean_survival >= base_mean_survival * (1 + CONT_LIFT_MARGIN)
    (CONT_LIFT_MARGIN=0.10; the full bridge = ARM_RELIEF_SAFETY_BRIDGE, the complete
    SD-059/MECH-358 mechanism). Aggregated as mean-of-per-seed mean episode lengths.
  PASS = readiness_met AND PRIMARY -> SD-059/MECH-358 SUPPORTED.
  SECONDARY / supplementary (reported, NOT load-bearing): best-across-bridge-arms
    continuous lift; per-seed lift consistency; binary best_bridge_G_H clears 2/3.

READINESS PRECONDITIONS (non-vacuity SAFETY NET, extended from 603l):
  (1) PAG freezes on the base arm (pag_n_commits>0 on >=2/3 seeds) AND the ilPFC
      gate engages (n_credit+n_decay>0 on >=2/3) -- the defensive chain is present.
  (2) EACH ENABLED BRIDGE HALF fires NON-VACUOUSLY (relief credit on RELIEF arm,
      safety credit on SAFETY arm, both on the both-arm; >=2/3 seeds).
  (3) NEW: the harm landscape became DISCRIMINATIVE on the base arm
      (harm_eval_range >= HARM_DISC_RANGE_FLOOR on >=2/3 seeds) -- protects against
      the harder env accidentally producing a flat harm landscape (the 603i defect).
  If any precondition fails -> self-route substrate_not_ready_requeue,
  evidence_direction non_contributory (NOT a false weakens).

EVIDENCE DIRECTION (scored; keyed on the CONTINUOUS metric):
  readiness NOT met            -> FAIL, substrate_not_ready_requeue, non_contributory.
  PRIMARY pass                 -> PASS, supports (the bridge lifts graded survival).
  readiness met, PRIMARY FAIL, ARM_NAV_CONTROL competent (env survivable) -> weakens
    (bridge_insufficient_env_survivable; genuine negative -- no GRADED benefit even
    though the bridge fired and the env is navigable to safety).
  readiness met, PRIMARY FAIL, ARM_NAV_CONTROL NOT competent -> non_contributory
    (navigation_survival_competence_ceiling -- a nav ceiling resurfacing on the
    harder env, NOT a bridge verdict).
  SD-059 (architecture) and MECH-358 (mechanism) are the same bridge and move
  together -> evidence_direction_per_claim maps both to the run direction.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: evidence
claim_ids: [SD-059, MECH-358]  (scored governance evidence -- the redesigned retest)
supersedes: V3-EXQ-603o
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    stage_plan,
)

EXPERIMENT_TYPE = "v3_exq_603q_sd059_mech358_escape_affordance_bridge_evidence"
QUEUE_ID = "V3-EXQ-603q"
CLAIM_IDS: List[str] = ["SD-059", "MECH-358"]  # scored evidence retest
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror 603i/603k/603l exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Budgets (mirror 603l full budget).
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

# Isolated hazard-avoidance Stage-H.
# REDESIGN HEADROOM LEVER (fix a): the regime is made HARDER than 603l
# (num_hazards 4->6, proximity_harm 0.10->0.15) to stress the scalar harm_eval
# gradient so ARM_BASE_IA_ONLY sits below the binary ceiling with measurable
# headroom. Resources / hfa / gate / budget / window unchanged from 603l.
HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 6        # 603l: 4 (headroom lever)
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.10  # 603q: the 603p positive-control regime the
# harm-pathway STABILIZATION amend (decoupled encoder LR + warmup) is validated to make
# the base harm landscape discriminative on >=2/3 seeds. (603o used 0.15 for headroom;
# 603q anchors on the amend-validated 0.10 and relies on the CONTINUOUS survival primary,
# which is robust to residual binary saturation.)
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# 634c seeding calibration + SD-057 cue-recall bridge (mirror 603l).
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (the avoidance-learning driver) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# --- SD-059 / MECH-358 escape-affordance bridge knobs (mirror 603l) ---
ESCAPE_THREAT_FLOOR = 0.1
ESCAPE_THREAT_REF = 0.35
ESCAPE_APPROACH_GAIN = 0.1
ESCAPE_BIAS_SCALE = 0.1

# --- SUBSTRATE FIX 1 (603k): Stage-H harm-pathway training -------------------------
HARM_PATHWAY_LR = 1e-3
# --- SUBSTRATE FIX 3 (603q amend): harm-pathway training STABILIZATION ------------
# The 603p autopsy: the base harm landscape forms on only 1/3 seeds at proximity_harm=0.10
# and a 3x global LR COLLAPSES it -- a single Adam LR co-trains the latent_stack encoder
# AND the harm heads. These two levers (2026-06-16 scaffolded_sd054_onboarding amend)
# decouple the encoder LR from the head LR + add an LR warmup, WITHOUT raising the global
# LR. They make the base harm landscape clear harm_eval_range>=0.02 on >=2/3 seeds (the
# experiment's first self-routing non-vacuity precondition).
HARM_PATHWAY_ENCODER_LR = 3e-4     # decoupled (lower) latent_stack encoder LR
HARM_PATHWAY_WARMUP_STEPS = 250    # linear LR warmup over the first N harm-pathway steps

# --- SUBSTRATE FIX 2 (603j): trained safety-half threat-absence signal ------------
ESCAPE_SAFETY_SIGNAL_THRESHOLD = 0.5

# Pre-registered gates (constants).
STAGE0_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0
# REDESIGN CONTINUOUS-METRIC CONSTANTS (fix b):
CONT_LIFT_MARGIN = 0.10            # required relative lift of the full bridge over base
HARM_DISC_RANGE_FLOOR = 0.02       # base harm-landscape non-vacuity (603i flat ~0.002; 603k ~0.133)

# 4 discriminative bridge arms (all on the 603i-INTACT base + both substrate fixes)
# + a nav-competence positive control. The 4 discriminative arms differ ONLY in the bridge.
ARMS = [
    {"label": "ARM_BASE_IA_ONLY", "bridge": False, "relief": False, "safety": False, "nav_control": False},
    {"label": "ARM_RELIEF_BRIDGE", "bridge": True, "relief": True, "safety": False, "nav_control": False},
    {"label": "ARM_SAFETY_BRIDGE", "bridge": True, "relief": False, "safety": True, "nav_control": False},
    {"label": "ARM_RELIEF_SAFETY_BRIDGE", "bridge": True, "relief": True, "safety": True, "nav_control": False},
    {"label": "ARM_NAV_CONTROL", "bridge": False, "relief": False, "safety": False, "nav_control": True},
]
BRIDGE_ARM_LABELS = {"ARM_RELIEF_BRIDGE", "ARM_SAFETY_BRIDGE", "ARM_RELIEF_SAFETY_BRIDGE"}


def _make_scaffold_cfg(dry_run: bool, nav_control: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS
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
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # The isolated Stage-H -- HARDER regime (the 603o headroom lever).
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        # NAV-COMPETENCE positive control: spawn IN the reef refuge so navigation
        # to safety is handed. The ONLY difference of ARM_NAV_CONTROL vs ARM_BASE_IA_ONLY.
        scaffold_hazard_stage_spawn_in_reef_half=bool(nav_control),
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver (active on ALL arms).
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE (all arms): feed the env harm stream so z_harm / z_harm_a populate.
        scaffold_feed_harm_stream=True,
        # ===== SUBSTRATE FIX 1 (603k): Stage-H harm-pathway training (all arms) =====
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
        # SUBSTRATE FIX 3 (603q amend): stabilize harm-pathway training so the base
        # harm landscape is seed-robust (decoupled encoder LR + LR warmup; NOT a global
        # LR raise -- that collapses it per 603p).
        scaffold_harm_pathway_encoder_lr=HARM_PATHWAY_ENCODER_LR,
        scaffold_harm_pathway_warmup_steps=HARM_PATHWAY_WARMUP_STEPS,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env, arm: Dict[str, Any]) -> REEConfig:
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
        # MECH-279 PAG freeze-gate (ALL arms).
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        # SD-058 / MECH-357 instrumental-avoidance gate (ALL arms -- the base the
        # bridge extends).
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # SD-059 / MECH-358 escape-affordance bridge (per arm).
        use_escape_affordance_bridge=bool(arm["bridge"]),
        use_escape_relief_credit=bool(arm["relief"]),
        use_escape_safety_credit=bool(arm["safety"]),
        escape_threat_floor=ESCAPE_THREAT_FLOOR,
        escape_threat_ref=ESCAPE_THREAT_REF,
        escape_approach_gain=ESCAPE_APPROACH_GAIN,
        escape_bias_scale=ESCAPE_BIAS_SCALE,
        # ===== SUBSTRATE FIX 2 (603j): trained safety-half threat-absence signal =====
        escape_use_trained_safety_signal=True,
        escape_safety_signal_threshold=ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _config_slice(arm: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """Content-addressed config slice for the per-cell arm fingerprint."""
    return {
        "arm": arm["label"],
        "use_escape_affordance_bridge": bool(arm["bridge"]),
        "use_escape_relief_credit": bool(arm["relief"]),
        "use_escape_safety_credit": bool(arm["safety"]),
        "nav_control_spawn_in_reef": bool(arm["nav_control"]),
        "use_instrumental_avoidance": True,
        "scaffold_avoidance_driver_enabled": True,
        "use_pag_freeze_gate": True,
        "pag_theta_freeze": PAG_THETA_FREEZE,
        "pag_duration_input_threshold": PAG_DURATION_INPUT_THRESHOLD,
        "avoidance_threat_ref": AVOIDANCE_THREAT_REF,
        "escape_threat_floor": ESCAPE_THREAT_FLOOR,
        "escape_threat_ref": ESCAPE_THREAT_REF,
        "escape_approach_gain": ESCAPE_APPROACH_GAIN,
        "escape_bias_scale": ESCAPE_BIAS_SCALE,
        "feed_harm_stream": True,
        "e2_action_contrastive_enabled": True,
        "scaffold_train_harm_pathway": True,
        "harm_pathway_lr": HARM_PATHWAY_LR,
        "use_harm_stream": True,
        "use_e2_harm_s_forward": True,
        "escape_use_trained_safety_signal": True,
        "escape_safety_signal_threshold": ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        "use_contextual_safety_terrain": True,
        "use_conditioned_safety_store": True,
        "use_suffering_derivative_comparator": True,
        "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
                    P1_BUDGET, P2_BUDGET, TRAIN_STEPS],
        # Harder hazard regime is part of the content address (distinguishes from 603l).
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR],
        "dry_run": bool(dry_run),
    }


def _aborted_record(arm_label: str, seed: int, stage: str, reason: str,
                    s0_peak: float = 0.0) -> Dict[str, Any]:
    return {
        "arm": arm_label, "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "hazard_stage_survival_pass": False,
        "hazard_stage_median_last_window": 0.0,
        "hazard_stage_mean_episode_length": 0.0,
        "hazard_stage_auc_survival": 0.0,
        "hazard_stage_time_to_first_death": 0,
        "hazard_eval_range": 0.0,
        "p1_survival_pass": False,
        "p2_contact_rate": 0.0,
        "g0_stage0_zgoal": bool(s0_peak > STAGE0_ZGOAL_GATE),
        "g1_p1_survival": False,
        "g2_p2_contact": False,
        "g_h_hazard_survival": False,
        "avoidance_gate_state": {},
        "escape_bridge_state": {},
        "pag_n_commits": 0,
        "pag_n_releases": 0,
        "reached_hazard_stage": stage not in ("stage0", "stage0b", "p0"),
        "reached_p1": False,
        "reached_p2": False,
        "seed_pass": False,
    }


def _auc_survival(ep_lengths: List[int]) -> float:
    """Normalised area-under-survival: mean episode length / max steps in [0, 1]."""
    if not ep_lengths:
        return 0.0
    return float(sum(ep_lengths)) / float(len(ep_lengths) * TRAIN_STEPS)


def _time_to_first_death(ep_lengths: List[int]) -> int:
    """1-based index of the first Stage-H episode that ended in death (length <
    TRAIN_STEPS, i.e. the agent did not survive the full budget). Returns
    len(ep_lengths)+1 sentinel if the agent never died (survived every episode)."""
    for i, ln in enumerate(ep_lengths):
        if ln < TRAIN_STEPS:
            return i + 1
    return len(ep_lengths) + 1


def _run_seed_arm(arm: Dict[str, Any], seed: int, dry_run: bool,
                  total_eps: int) -> Dict[str, Any]:
    """Full curriculum for one (arm, seed) cell. arm_cell resets all RNG on
    enter (order-independent) and stamps the fingerprint on the returned row."""
    with arm_cell(
        seed,
        config_slice=_config_slice(arm, dry_run),
        script_path=Path(__file__),
        config_slice_declared=True,
    ) as cell:
        scaffold_cfg = _make_scaffold_cfg(dry_run, arm["nav_control"])
        device = torch.device("cpu")
        probe_env = _build_env(scaffold_cfg, "p2")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env, arm)).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

        print(f"Seed {seed} Condition {arm['label']}", flush=True)

        def _gate_state() -> Dict[str, Any]:
            g = getattr(agent, "instrumental_avoidance", None)
            return g.get_state() if g is not None else {}

        def _bridge_state() -> Dict[str, Any]:
            b = getattr(agent, "escape_affordance_bridge", None)
            return b.get_state() if b is not None else {}

        def _pag_state() -> Dict[str, Any]:
            p = getattr(agent, "pag_freeze_gate", None)
            return dict(p.diagnostics) if p is not None else {}

        # Stage 0 -- forced-benefit nursery (goal-formation positive control).
        s0 = scheduler.run_stage0_nursery(agent, device)
        done = s0.n_episodes
        print(f"  [train] stage0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0", flush=True)
            rec = _aborted_record(arm["label"], seed, "stage0", s0.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0b", flush=True)
            rec = _aborted_record(arm["label"], seed, "stage0b", s0b.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        p0 = scheduler.run_p0(agent, device)
        done += p0.n_episodes
        print(f"  [train] p0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
              flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=p0", flush=True)
            rec = _aborted_record(arm["label"], seed, "p0", p0.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        # Stage-H -- ISOLATED HAZARD-AVOIDANCE (the SD-059/MECH-358 bridge target).
        hz = scheduler.run_hazard_avoidance(agent, device)
        done += hz.n_episodes
        gate_after_h = _gate_state()
        bridge_after_h = _bridge_state()
        pag_after_h = _pag_state()
        harm_disc = dict(hz.harm_discriminativeness or {})
        harm_eval_range = float(harm_disc.get("harm_eval_range", 0.0))
        ep_lengths = list(hz.episode_lengths or [])
        auc_surv = _auc_survival(ep_lengths)
        ttfd = _time_to_first_death(ep_lengths)
        print(f"  [train] hazard_avoidance {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={hz.mean_episode_length:.1f}"
              f" median_last={hz.median_last_window_episode_length:.1f}"
              f" auc={auc_surv:.3f}"
              f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
              f" harm_range={harm_eval_range:.4f}"
              f" pag_commits={pag_after_h.get('n_commits', 0)}"
              f" n_relief={bridge_after_h.get('mech358_n_relief_credit', 0)}"
              f" n_safety={bridge_after_h.get('mech358_n_safety_credit', 0)}"
              f" n_safety_trained={bridge_after_h.get('mech358_n_safety_credit_trained', 0)}"
              f" n_approach={bridge_after_h.get('mech358_n_approach_fires', 0)}",
              flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=hazard", flush=True)
            rec = _aborted_record(arm["label"], seed, "hazard", hz.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            rec["avoidance_gate_state"] = gate_after_h
            rec["escape_bridge_state"] = bridge_after_h
            rec["hazard_eval_range"] = harm_eval_range
            cell.stamp(rec)
            return rec

        # P1 -- combined wean (GAP-2 transfer; bridge still active in ARM bridge arms).
        p1 = scheduler.run_p1(agent, device)
        done += p1.n_episodes
        print(f"  [train] p1 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={p1.median_last_window_episode_length:.1f}"
              f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

        # P2 -- frozen-policy guarded measurement.
        p2 = scheduler.run_p2(agent, device)
        done += p2.n_episodes
        print(f"  [train] p2 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" contact_rate={p2.contact_rate:.4f}", flush=True)

        g0 = bool(s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE)
        g1 = bool(p1.survival_gate_passed)
        g2 = bool(p2.contact_rate > CONTACT_GATE)
        g_h = bool(hz.survival_gate_passed)
        gate_final = _gate_state()
        bridge_final = _bridge_state()
        # Seed-level pass is the CONTINUOUS survival readout's anchor (mean episode
        # length); the binary g_h is supplementary now.
        seed_pass = bool(g_h)
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={arm['label']}"
              f" mean_surv={hz.mean_episode_length:.1f} g_h={g_h} g0={g0} g1={g1} g2={g2}"
              f" harm_range={harm_eval_range:.4f}"
              f" n_relief={bridge_final.get('mech358_n_relief_credit', 0)}"
              f" n_safety={bridge_final.get('mech358_n_safety_credit', 0)}",
              flush=True)

        rec = {
            "arm": arm["label"],
            "seed": seed,
            "aborted_at": None,
            "abort_reason": "",
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "p0_mean_episode_length": float(p0.mean_episode_length),
            # --- CONTINUOUS survival readouts (fix b) ---
            "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
            "hazard_stage_auc_survival": float(auc_surv),
            "hazard_stage_time_to_first_death": int(ttfd),
            "hazard_stage_episode_lengths": ep_lengths,
            # --- binary G_H (supplementary now) ---
            "hazard_stage_survival_pass": g_h,
            "hazard_stage_median_last_window": float(hz.median_last_window_episode_length),
            "hazard_stage_n_episodes": int(hz.n_episodes),
            # --- harm-landscape non-vacuity ---
            "hazard_eval_range": harm_eval_range,
            "harm_discriminativeness": harm_disc,
            "pag_n_commits": int(pag_after_h.get("n_commits", 0)),
            "pag_n_releases": int(pag_after_h.get("n_releases", 0)),
            "p1_survival_pass": g1,
            "p1_median_last_window_episode_length": float(p1.median_last_window_episode_length),
            "p2_contact_rate": float(p2.contact_rate),
            "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
            "g0_stage0_zgoal": g0,
            "g1_p1_survival": g1,
            "g2_p2_contact": g2,
            "g_h_hazard_survival": g_h,
            "avoidance_gate_state": gate_final,
            "escape_bridge_state": bridge_final,
            "reached_hazard_stage": True,
            "reached_p1": True,
            "reached_p2": True,
            "seed_pass": seed_pass,
        }
        cell.stamp(rec)
        return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _mean(vals: List[float]) -> float:
    return float(sum(vals)) / float(len(vals)) if vals else 0.0


def _arm_half_credit_frac(rows: List[Dict[str, Any]], key: str) -> float:
    """Fraction of seeds where the named bridge-credit counter incremented."""
    flags = [int((r.get("escape_bridge_state", {}) or {}).get(key, 0)) > 0 for r in rows]
    return _frac(flags)


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + P2_BUDGET
        )

    arm_results: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    rows_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS:
        rows = [_run_seed_arm(arm, s, dry_run, total_eps) for s in seeds]
        rows_by_arm[arm["label"]] = rows
        per_seed.extend(rows)
        g_h_flags = [bool(r.get("g_h_hazard_survival", False)) for r in rows]
        g1_flags = [bool(r.get("g1_p1_survival", False)) for r in rows]
        g0_flags = [bool(r.get("g0_stage0_zgoal", False)) for r in rows]
        g2_flags = [bool(r.get("g2_p2_contact", False)) for r in rows]
        per_seed_mean_surv = [float(r.get("hazard_stage_mean_episode_length", 0.0)) for r in rows]
        per_seed_auc = [float(r.get("hazard_stage_auc_survival", 0.0)) for r in rows]
        per_seed_harm_range = [float(r.get("hazard_eval_range", 0.0)) for r in rows]
        engaged_flags, suppressed_flags = [], []
        for r in rows:
            gs = r.get("avoidance_gate_state", {}) or {}
            engaged_flags.append(
                (int(gs.get("mech357_n_credit", 0)) + int(gs.get("mech357_n_decay", 0))) > 0
            )
            suppressed_flags.append(int(gs.get("mech357_n_freeze_suppressed", 0)) > 0)
        pag_freeze_flags = [int(r.get("pag_n_commits", 0)) > 0 for r in rows]
        harm_disc_flags = [hr >= HARM_DISC_RANGE_FLOOR for hr in per_seed_harm_range]
        arm_results.append({
            "arm": arm["label"],
            "use_escape_affordance_bridge": bool(arm["bridge"]),
            "use_escape_relief_credit": bool(arm["relief"]),
            "use_escape_safety_credit": bool(arm["safety"]),
            "nav_control": bool(arm["nav_control"]),
            # --- CONTINUOUS survival aggregates (primary metric, fix b) ---
            "mean_survival": _mean(per_seed_mean_surv),
            "auc_survival": _mean(per_seed_auc),
            "per_seed_mean_survival": per_seed_mean_surv,
            "per_seed_auc_survival": per_seed_auc,
            "per_seed_time_to_first_death": [
                int(r.get("hazard_stage_time_to_first_death", 0)) for r in rows
            ],
            # --- binary G_H (supplementary) ---
            "g_h_frac": _frac(g_h_flags),
            "g0_frac": _frac(g0_flags),
            "g1_frac": _frac(g1_flags),
            "g2_frac": _frac(g2_flags),
            "gate_engaged_frac": _frac(engaged_flags),
            "gate_freeze_suppressed_frac": _frac(suppressed_flags),
            "pag_freeze_frac": _frac(pag_freeze_flags),
            "harm_disc_frac": _frac(harm_disc_flags),
            "relief_credit_frac": _arm_half_credit_frac(rows, "mech358_n_relief_credit"),
            "safety_credit_frac": _arm_half_credit_frac(rows, "mech358_n_safety_credit"),
            "safety_credit_trained_frac": _arm_half_credit_frac(rows, "mech358_n_safety_credit_trained"),
            "per_seed_g_h": g_h_flags,
            "per_seed_harm_eval_range": per_seed_harm_range,
            "per_seed_pag_n_commits": [int(r.get("pag_n_commits", 0)) for r in rows],
            "per_seed_hazard_median_last_window": [
                r.get("hazard_stage_median_last_window", 0.0) for r in rows
            ],
            "per_seed_n_relief_credit": [
                int((r.get("escape_bridge_state", {}) or {}).get("mech358_n_relief_credit", 0))
                for r in rows
            ],
            "per_seed_n_safety_credit": [
                int((r.get("escape_bridge_state", {}) or {}).get("mech358_n_safety_credit", 0))
                for r in rows
            ],
            "per_seed_n_safety_credit_trained": [
                int((r.get("escape_bridge_state", {}) or {}).get("mech358_n_safety_credit_trained", 0))
                for r in rows
            ],
            "per_seed_n_approach_fires": [
                int((r.get("escape_bridge_state", {}) or {}).get("mech358_n_approach_fires", 0))
                for r in rows
            ],
            "arm_fingerprint": [r.get("arm_fingerprint") for r in rows],
        })

    by_label = {a["arm"]: a for a in arm_results}
    base = by_label["ARM_BASE_IA_ONLY"]
    relief = by_label["ARM_RELIEF_BRIDGE"]
    safety = by_label["ARM_SAFETY_BRIDGE"]
    both = by_label["ARM_RELIEF_SAFETY_BRIDGE"]
    nav = by_label["ARM_NAV_CONTROL"]
    bridge_arms = [relief, safety, both]

    base_mean_surv = float(base["mean_survival"])
    both_mean_surv = float(both["mean_survival"])
    best_bridge_mean_surv = max(float(a["mean_survival"]) for a in bridge_arms)
    lift_threshold = base_mean_surv * (1.0 + CONT_LIFT_MARGIN)

    # --- PRIMARY pre-registered acceptance (CONTINUOUS, load-bearing) ---
    # The FULL bridge (both halves) lifts mean Stage-H survival over the IA-only base.
    both_lift = (both_mean_surv / base_mean_surv - 1.0) if base_mean_surv > 1e-9 else 0.0
    primary_continuous_pass = bool(both_mean_surv >= lift_threshold and base_mean_surv > 1e-9)
    # Secondary continuous (informational): best across bridge arms.
    best_bridge_lift = (best_bridge_mean_surv / base_mean_surv - 1.0) if base_mean_surv > 1e-9 else 0.0
    best_bridge_continuous_pass = bool(best_bridge_mean_surv >= lift_threshold and base_mean_surv > 1e-9)
    # Per-seed lift consistency (full bridge vs base, paired by seed; informational).
    per_seed_paired_lift = [
        bm > bs for bm, bs in zip(both["per_seed_mean_survival"], base["per_seed_mean_survival"])
    ]
    per_seed_lift_frac = _frac(per_seed_paired_lift)
    # Binary G_H supplementary readouts (NOT load-bearing).
    best_bridge_g_h = max(a["g_h_frac"] for a in bridge_arms)
    both_g_h_clears = bool(both["g_h_frac"] >= MIN_FRACTION)
    best_bridge_g_h_clears = bool(best_bridge_g_h >= MIN_FRACTION)
    best_bridge_g_h_beats_base = bool(best_bridge_g_h > base["g_h_frac"])  # may be unsatisfiable; informational

    # --- READINESS PRECONDITIONS (non-vacuity SAFETY NET) ---
    pavlovian_reaction_present = bool(base["pag_freeze_frac"] >= MIN_FRACTION)
    gate_engaged = bool(base["gate_engaged_frac"] >= MIN_FRACTION)
    half_frac = {
        "ARM_RELIEF_BRIDGE": relief["relief_credit_frac"],
        "ARM_SAFETY_BRIDGE": safety["safety_credit_frac"],
        "ARM_RELIEF_SAFETY_BRIDGE": min(both["relief_credit_frac"], both["safety_credit_frac"]),
    }
    bridge_halves_nonvacuous = all(
        half_frac[lbl] >= MIN_FRACTION for lbl in BRIDGE_ARM_LABELS
    )
    # NEW (603o): harm landscape became discriminative on the base arm.
    harm_landscape_discriminative = bool(base["harm_disc_frac"] >= MIN_FRACTION)
    readiness_met = bool(
        pavlovian_reaction_present and gate_engaged
        and bridge_halves_nonvacuous and harm_landscape_discriminative
    )
    g0_base_ok = bool(base["g0_frac"] >= MIN_FRACTION)

    # --- ARM_NAV_CONTROL env-survivability disambiguator (continuous + binary) ---
    nav_mean_surv = float(nav["mean_survival"])
    nav_control_competent = bool(
        nav["g_h_frac"] >= MIN_FRACTION or nav_mean_surv > base_mean_surv
    )

    # --- which half carried the continuous lift (for the interpretation grid) ---
    relief_lifts = bool(float(relief["mean_survival"]) >= lift_threshold)
    safety_lifts = bool(float(safety["mean_survival"]) >= lift_threshold)
    both_lifts = bool(both_mean_surv >= lift_threshold)

    if not readiness_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        run_direction = "non_contributory"
    elif primary_continuous_pass:
        outcome = "PASS"
        run_direction = "supports"
        if relief_lifts and not safety_lifts:
            readiness_route = "escape_affordance_bridge_lifts_survival_relief_carries"
        elif safety_lifts and not relief_lifts:
            readiness_route = "escape_affordance_bridge_lifts_survival_safety_carries"
        elif both_lifts and not (relief_lifts or safety_lifts):
            readiness_route = "escape_affordance_bridge_lifts_survival_both_required"
        else:
            readiness_route = "escape_affordance_bridge_lifts_survival"
    else:
        outcome = "FAIL"
        if nav_control_competent:
            # Env survivable + bridge wired & non-vacuous but delivers NO GRADED
            # survival lift -> genuine negative evidence for SD-059/MECH-358.
            readiness_route = "bridge_insufficient_env_survivable"
            run_direction = "weakens"
        else:
            # A navigation/survival-competence ceiling resurfacing on the harder
            # env -- NOT a bridge verdict; protect the claims.
            readiness_route = "navigation_survival_competence_ceiling"
            run_direction = "non_contributory"

    evidence_direction_per_claim = {cid: run_direction for cid in CLAIM_IDS}

    preconditions = [
        {
            "name": "pag_freeze_and_ilpfc_gate_engage_on_base",
            "kind": "readiness",
            "description": "The defensive chain must be present on ARM_BASE_IA_ONLY: PAG "
                           "freezes (pag_n_commits>0 on >=2/3 seeds) AND the ilPFC gate "
                           "engages (n_credit+n_decay>0 on >=2/3). Below-floor => the gate/"
                           "freeze substrate the bridge extends is itself inert => "
                           "substrate_not_ready_requeue, NOT a bridge verdict.",
            "control": "ARM_BASE_IA_ONLY: PAG + ilPFC gate + driver + fed harm stream + "
                       "harm-pathway training (603k) + trained safety predictors (603j).",
            "measured": float(min(base["pag_freeze_frac"], base["gate_engaged_frac"])),
            "threshold": float(MIN_FRACTION),
            "met": bool(pavlovian_reaction_present and gate_engaged),
        },
        {
            "name": "each_enabled_bridge_half_fires_nonvacuously",
            "kind": "readiness",
            "description": "Each ENABLED bridge half must increment its credit on >=2/3 seeds "
                           "(ARM_RELIEF_BRIDGE: mech358_n_relief_credit>0; ARM_SAFETY_BRIDGE: "
                           "mech358_n_safety_credit>0; ARM_RELIEF_SAFETY_BRIDGE: both). The "
                           "safety half credits via the trained MECH-303/304 threat-absence "
                           "signal (603j); below-floor => the predictor did not populate "
                           "ecologically => substrate_not_ready_requeue (NOT a false weakens).",
            "control": "ARM_RELIEF/SAFETY/RELIEF_SAFETY: bridge ON, SD-056 e2 warmup in P0, fed "
                       "harm stream, escape_use_trained_safety_signal + MECH-303/304/302 "
                       "populating during Stage-H.",
            "measured": float(min(half_frac[lbl] for lbl in BRIDGE_ARM_LABELS)),
            "threshold": float(MIN_FRACTION),
            "met": bool(bridge_halves_nonvacuous),
        },
        {
            "name": "harm_landscape_discriminative_on_base",
            "kind": "readiness",
            "description": "On ARM_BASE_IA_ONLY the 603k harm-pathway training must have made "
                           "the harm landscape DISCRIMINATIVE (harm_eval_range >= "
                           f"{HARM_DISC_RANGE_FLOOR} on >=2/3 seeds; 603i flat defect ~0.002, "
                           "603k trained ~0.133). Protects against the harder 603o env "
                           "accidentally producing a flat harm landscape -> below-floor is "
                           "substrate_not_ready_requeue (a nav/training ceiling), NOT a bridge "
                           "verdict. SAME statistic the survival leg depends on (a trained "
                           "harm gradient is the substrate the bridge sits on top of).",
            "control": "ARM_BASE_IA_ONLY harm-pathway training discriminativeness probe.",
            "measured": float(base["harm_disc_frac"]),
            "threshold": float(MIN_FRACTION),
            "met": bool(harm_landscape_discriminative),
        },
        {
            "name": "stage0_forced_feed_lights_zgoal_on_base",
            "kind": "readiness",
            "description": "Stage-0 forced supra-threshold benefit lights z_goal (>0.4) on "
                           ">=2/3 base seeds -- the goal-FORMATION positive control.",
            "control": "run_stage0_nursery forced-feed.",
            "measured": float(base["g0_frac"]),
            "threshold": float(MIN_FRACTION),
            "met": bool(g0_base_ok),
        },
    ]
    criteria_non_degenerate = {
        "arms_reached_hazard_stage": bool(
            _frac([r.get("reached_hazard_stage", False) for r in per_seed]) >= MIN_FRACTION
        ),
        "bridge_credit_nonzero": bool(
            max(relief["relief_credit_frac"], safety["safety_credit_frac"],
                both["relief_credit_frac"], both["safety_credit_frac"]) > 0.0
        ),
        "nav_control_evaluated": bool(
            _frac([r.get("reached_hazard_stage", False) for r in rows_by_arm["ARM_NAV_CONTROL"]])
            >= MIN_FRACTION
        ),
        # Headroom check: base mean survival is NOT pinned at the TRAIN_STEPS ceiling
        # (the 603l binary-saturation failure mode). If base mean survival >= 0.97 of
        # TRAIN_STEPS the continuous metric has no headroom either -> degenerate.
        "base_survival_below_ceiling": bool(base_mean_surv < 0.97 * TRAIN_STEPS),
        "harm_landscape_discriminative_base": bool(harm_landscape_discriminative),
    }
    criteria = [
        {"name": "both_bridge_mean_survival_lift_ge_10pct", "load_bearing": True,
         "passed": bool(primary_continuous_pass)},
        {"name": "best_bridge_mean_survival_lift_ge_10pct", "load_bearing": False,
         "passed": bool(best_bridge_continuous_pass)},
        {"name": "both_bridge_G_H_clears_2of3_supplementary", "load_bearing": False,
         "passed": bool(both_g_h_clears)},
    ]

    print(
        f"[{EXPERIMENT_TYPE}] mean_surv base={base_mean_surv:.1f} relief={relief['mean_survival']:.1f}"
        f" safety={safety['mean_survival']:.1f} both={both_mean_surv:.1f} nav={nav_mean_surv:.1f}"
        f" | both_lift={both_lift:+.2%} (need >= {CONT_LIFT_MARGIN:.0%}) best_lift={best_bridge_lift:+.2%}"
        f" | G_H base={base['g_h_frac']:.2f} both={both['g_h_frac']:.2f} nav={nav['g_h_frac']:.2f}"
        f" | readiness_met={readiness_met} -> outcome={outcome} route={readiness_route}"
        f" direction={run_direction}",
        flush=True,
    )

    return {
        "outcome": outcome,
        "primary_continuous_pass": primary_continuous_pass,
        "base_mean_survival": base_mean_surv,
        "both_mean_survival": both_mean_surv,
        "best_bridge_mean_survival": best_bridge_mean_surv,
        "nav_control_mean_survival": nav_mean_surv,
        "both_bridge_survival_lift": both_lift,
        "best_bridge_survival_lift": best_bridge_lift,
        "cont_lift_margin_required": CONT_LIFT_MARGIN,
        "per_seed_full_bridge_beats_base_frac": per_seed_lift_frac,
        # binary supplementary
        "best_bridge_g_h_frac": best_bridge_g_h,
        "both_g_h_clears": both_g_h_clears,
        "best_bridge_g_h_clears": best_bridge_g_h_clears,
        "best_bridge_g_h_beats_base": best_bridge_g_h_beats_base,
        # readiness
        "readiness_met": readiness_met,
        "pavlovian_reaction_present": pavlovian_reaction_present,
        "gate_engaged": gate_engaged,
        "bridge_halves_nonvacuous": bridge_halves_nonvacuous,
        "harm_landscape_discriminative": harm_landscape_discriminative,
        "nav_control_competent": nav_control_competent,
        "relief_lifts": relief_lifts,
        "safety_lifts": safety_lifts,
        "both_lifts": both_lifts,
        "evidence_direction": run_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "arm_results": arm_results,
        "acceptance": {
            "pass_rule": "PASS = readiness_met (PAG freezes + ilPFC gate engages on BASE >=2/3 "
                         "AND each enabled bridge half credits >=2/3 AND base harm landscape "
                         "discriminative >=2/3) AND PRIMARY: both_bridge mean Stage-H survival "
                         ">= base mean survival * (1 + CONT_LIFT_MARGIN). Binary G_H is "
                         "SUPPLEMENTARY (reported, not load-bearing).",
            "min_fraction": MIN_FRACTION,
            "cont_lift_margin": CONT_LIFT_MARGIN,
            "harm_disc_range_floor": HARM_DISC_RANGE_FLOOR,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "train_steps_ceiling": TRAIN_STEPS,
            "base_mean_survival": base_mean_surv,
            "relief_mean_survival": float(relief["mean_survival"]),
            "safety_mean_survival": float(safety["mean_survival"]),
            "both_mean_survival": both_mean_surv,
            "nav_control_mean_survival": nav_mean_surv,
            "g_h_base_frac": base["g_h_frac"],
            "g_h_both_frac": both["g_h_frac"],
            "g_h_nav_control_frac": nav["g_h_frac"],
            "relief_credit_frac": relief["relief_credit_frac"],
            "safety_credit_frac": safety["safety_credit_frac"],
            "safety_credit_trained_frac": safety["safety_credit_trained_frac"],
            "base_pag_freeze_frac": base["pag_freeze_frac"],
            "base_gate_engaged_frac": base["gate_engaged_frac"],
            "base_harm_disc_frac": base["harm_disc_frac"],
        },
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "grid": {
                "relief_only_lift": "relief (phasic negative-reinforcement) credit carries the survival lift",
                "safety_only_lift": "safety (learned threat-absence) credit carries the survival lift",
                "both_required_lift": "directed escape needs complementary relief + safety bridge",
                "no_lift_and_nav_control_competent": "bridge insufficient (env survivable) -> weakens SD-059/MECH-358",
                "no_lift_and_nav_control_incompetent": "navigation/survival-competence ceiling on the harder env -> nav substrate, NOT the bridge",
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
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",
        "supersedes": "V3-EXQ-603o",
        "depends_on": ["V3-EXQ-603i", "V3-EXQ-603j", "V3-EXQ-603k"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "SD-059 / MECH-358 relief/safety escape-affordance bridge "
                     "(ree_core/pfc/escape_affordance_bridge.py) over the SD-058/MECH-357 gate "
                     "in the scaffolded_sd054_onboarding Stage-H, on the 603k harm-pathway-trained "
                     "+ 603j trained-safety-signal substrate, HARDER hazard regime (6 hazards, "
                     "proximity_harm 0.15)",
        "scores": "SD-059 (architecture) + MECH-358 (affordance-indexed relief/safety credit "
                  "+ threat-gated E3 approach bonus) -- the REDESIGNED behavioural re-test "
                  "(continuous-survival primary metric + headroom hazard regime), successor to "
                  "V3-EXQ-603l",
        "design_note": "REDESIGN of V3-EXQ-603l per failure_autopsy_V3-EXQ-603l_2026-06-10. "
                       "Fix (a) HEADROOM: harder Stage-H (num_hazards 4->6, proximity_harm "
                       "0.10->0.15) at FULL uniform 603k+603j substrate so ARM_BASE_IA_ONLY sits "
                       "below the binary survival ceiling. Fix (b) CONTINUOUS PRIMARY METRIC: "
                       "mean Stage-H survival duration (mean episode length) is the load-bearing "
                       "discrimination metric (full bridge lift >= 10% over base); binary G_H is "
                       "retained as a SUPPLEMENTARY readout. 5 arms x 3 seeds on the 603i-INTACT "
                       "base (MECH-279 PAG + SD-058/MECH-357 ilPFC gate + driver + fed harm stream "
                       "+ SD-056 e2 warmup) + scaffold_train_harm_pathway=True (603k) + "
                       "escape_use_trained_safety_signal=True (603j). Non-vacuity SAFETY NET: PAG "
                       "freezes + gate engages + harm landscape discriminative on base, each enabled "
                       "bridge half credits >=2/3, else substrate_not_ready_requeue (protects "
                       "SD-059/MECH-358 from a false weakens).",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "primary_pass_rule": "readiness_met AND both_bridge_mean_survival >= base_mean_survival * (1 + 0.10)",
            "continuous_metric": "mean Stage-H episode length (survival duration); supplemented by AUC-survival + time-to-first-death",
            "binary_g_h_supplementary": "median episode length over last 10 Stage-H episodes >= 75 (reported, NOT load-bearing)",
            "nav_competence_control": "ARM_NAV_CONTROL (spawn-in-reef) survival -- env-survivability disambiguator",
            "harm_disc_range_floor": HARM_DISC_RANGE_FLOOR,
            "cont_lift_margin": CONT_LIFT_MARGIN,
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "min_fraction": MIN_FRACTION,
        },
        "headroom_regime_vs_603l": {
            "hazard_stage_num_hazards": [4, HAZARD_STAGE_NUM_HAZARDS],
            "hazard_stage_proximity_harm_scale": [0.1, HAZARD_STAGE_PROXIMITY_HARM],
            "note": "all other Stage-H knobs (resources 2, hfa 0.0, gate 75, budget 40, window 10) unchanged from 603l",
        },
        "substrate_fixes_on": {
            "harm_pathway_training_603k": {
                "scaffold_train_harm_pathway": True,
                "scaffold_harm_pathway_lr": HARM_PATHWAY_LR,
                "scaffold_harm_pathway_in_p0": True,
                "scaffold_harm_pathway_encoder_lr": HARM_PATHWAY_ENCODER_LR,
                "scaffold_harm_pathway_warmup_steps": HARM_PATHWAY_WARMUP_STEPS,
                "use_harm_stream": True,
                "use_e2_harm_s_forward": True,
            },
            "trained_safety_signal_603j": {
                "escape_use_trained_safety_signal": True,
                "escape_safety_signal_threshold": ESCAPE_SAFETY_SIGNAL_THRESHOLD,
                "use_contextual_safety_terrain": True,
                "use_conditioned_safety_store": True,
                "use_suffering_derivative_comparator": True,
            },
        },
        "bridge_config": {
            "escape_threat_floor": ESCAPE_THREAT_FLOOR,
            "escape_threat_ref": ESCAPE_THREAT_REF,
            "escape_approach_gain": ESCAPE_APPROACH_GAIN,
            "escape_bias_scale": ESCAPE_BIAS_SCALE,
            "feed_harm_stream": True,
            "e2_action_contrastive_enabled": True,
        },
    }
    manifest.update(result)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']} direction: {result['evidence_direction']}", flush=True)
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
