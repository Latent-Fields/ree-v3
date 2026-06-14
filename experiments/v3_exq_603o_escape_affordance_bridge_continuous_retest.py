"""
V3-EXQ-603o -- SD-059 / MECH-358 relief/safety escape-affordance bridge:
the REDESIGNED 4-arm (+ nav-control) BEHAVIOURAL re-test (scored evidence).

PURPOSE (scored governance evidence; claim_ids=[SD-059, MECH-358]):
This is the test-design REDESIGN mandated by the confirmed
failure_autopsy_V3-EXQ-603l_2026-06-10 (status confirmed, user-adjudicated),
which reclassified V3-EXQ-603l from `weakens` to `non_contributory`. 603l FAILed
for a TEST-DESIGN reason, NOT a mechanism reason:

  (CONFOUND A -- removed deficit) The readiness-enabling 603k harm-pathway-training
  fix OVER-corrected. It trains E3.harm_eval(z_world), which by ARC-007-strict is
  what PICKS the escape direction, so ARM_BASE_IA_ONLY rose to the BINARY survival
  ceiling (G_H = 1.0, 3/3). The primary discrimination criterion
  best_bridge_G_H > G_H_BASE was therefore STRUCTURALLY UNSATISFIABLE -- a bridge
  arm can at best TIE.
  (CONFOUND B -- saturated metric) The binary median-episode-length >= 75 gate
  saturated for both base and the both-arm. The bridge fired NON-VACUOUSLY (relief
  299-426 / safety up to 2880 credit + thousands of approach-bonus fires) and
  ARM_RELIEF_SAFETY_BRIDGE showed ~35% longer MEAN hazard-stage episode length
  than base (~132 vs ~98) -- a GRADED benefit invisible to the binary gate.

TWO FIXES (autopsy Section 7; user choice "Both"):
  (a) HEADROOM -- a HARDER Stage-H hazard regime (more hazards + higher
      proximity_harm) so ARM_BASE_IA_ONLY sits at G_H ~0.33-0.67 with measurable
      headroom rather than at the 1.0 ceiling. The 603k harm-pathway training stays
      ON (nav/survival competence is a prerequisite -- without it the 603i
      navigation ceiling resurfaces) but the env now denies the base arm the binary
      ceiling, so the base carries the deficit the bridge is meant to close.
  (b) CONTINUOUS METRIC -- the PRIMARY discrimination metric is now MEAN hazard-stage
      episode length (with median, AUC-survival, and time-to-first-death as
      supplementary continuous readouts). The binary G_H gate is SUPPLEMENTARY only.

THE SUBSTRATE UNDER TEST: SD-059 / MECH-358 EscapeAffordanceBridge
(ree_core/pfc/escape_affordance_bridge.py) -- extends the MECH-357 scalar avoidance
efficacy into a per-first-action-class credit table with two halves: RELIEF (a
directed action under threat that drops z_harm_a credits relief_affordance[class],
MECH-302-consistent) and SAFETY (a directed action after which threat is absent --
read from the trained MECH-303/304 threat-absence predictor -- credits
safety_affordance[class], MECH-303/304-consistent). Under future threat, E3 receives
a bounded, threat-context-gated NEGATIVE (favoured) approach bias toward credited
classes -- the DIRECTED escape.

DESIGN: same 4 discriminative arms as 603l + the nav-competence positive control,
5 arms x 3 seeds [42, 43, 44], all on the 603i-INTACT defensive base
(MECH-279 PAG + SD-058/MECH-357 ilPFC gate + driver + fed harm stream + SD-056 e2
warmup) PLUS the two validated substrate fixes (603k harm-pathway training; 603j
trained safety-signal). The 4 discriminative arms differ ONLY in the bridge:
  ARM_BASE_IA_ONLY            = bridge OFF (SD-058/MECH-357 + harm-pathway base).
  ARM_RELIEF_BRIDGE           = bridge ON, relief half only.
  ARM_SAFETY_BRIDGE           = bridge ON, safety half only.
  ARM_RELIEF_SAFETY_BRIDGE    = bridge ON, both halves.
  ARM_NAV_CONTROL (positive control) = bridge OFF, Stage-H spawns IN the reef refuge
    (navigation to safety handed). Reef-refuge-reachability nav-competence control.

PRE-REGISTERED ACCEPTANCE (constants; NOT derived from the run's own statistics):
  PRIMARY (continuous): pick the bridge arm with the highest MEAN hazard-stage
    episode length (best_bridge_arm); compute per-seed delta = best_bridge mean -
    base mean. PASS requires:
      mean(delta) > margin  AND  delta > 0 on >= 2/3 seeds
    where margin = max(CONTINUOUS_EFFECT_SD_MULT * pstdev(delta), CONTINUOUS_ABS_FLOOR).
    The SD-of-DELTA term (NOT SD of the baseline LEVEL) scales noise; the ABSOLUTE
    FLOOR is load-bearing because a consistent delta collapses SD-of-delta to ~0
    (per feedback_effect_size_pass_gate_margin: MECH-423 V3-EXQ-680a false PASS).
  PASS = readiness_met AND non-degenerate AND PRIMARY -> SD-059/MECH-358 SUPPORTED.

NON-DEGENERACY / NON-VACUITY GUARD (pre-registered; the 603l-class self-route):
  base-at-ceiling: if ARM_BASE_IA_ONLY is at the survival ceiling (binary G_H_frac
    >= 1.0 OR mean episode length >= SURVIVAL_CEILING_FRAC * steps on >= 2/3 seeds),
    the regime is TOO EASY -- the bridge has no headroom (the literal 603l failure).
    Self-route substrate_not_ready_requeue -> non_contributory (re-tune HARDER), set
    non_degenerate=False. NEVER a false weakens.
  vacuous-DV: if cross-arm variance of the per-arm mean continuous metric ~ 0, OR
    every arm collapses (mean episode length <= COLLAPSE_FLOOR_FRAC * steps -- regime
    TOO HARD), the discrimination metric cannot fire -> substrate_not_ready_requeue ->
    non_contributory, set non_degenerate=False. (This fleet hits vacuous-DV /
    zero-cross-arm-variance degeneracy often; guard it explicitly.)
  bridge-halves-fire: each ENABLED bridge half must credit on >= 2/3 seeds (relief
    on ARM_RELIEF_BRIDGE; safety on ARM_SAFETY_BRIDGE; both on the both-arm). If an
    enabled half never credits, its G_H/continuous comparison is uninformative ->
    substrate_not_ready_requeue -> non_contributory (the trained MECH-303/304 safety
    predictor must populate ecologically before scoring).
  defensive-chain-present: PAG freezes + ilPFC gate engages on the base arm
    (>= 2/3 seeds) -- the substrate the bridge extends is live.

INTERPRETATION GRID (one row per outcome -> next action):
  | readiness NOT met                                  -> non_contributory, substrate_not_ready_requeue: fix the gate/predictor, re-queue (NOT a bridge verdict).
  | base AT survival ceiling (regime too easy)         -> non_contributory, substrate_not_ready_requeue: HARDER hazard regime (more hazards / proximity_harm) OR partial harm-pathway training; re-queue. (The literal 603l failure -- prevents the structurally-unsatisfiable comparison recurring.)
  | vacuous DV (all arms collapse / zero cross-arm var)-> non_contributory, substrate_not_ready_requeue: EASIER hazard regime / longer budget so survival has dynamic range; re-queue.
  | PASS (continuous primary holds, non-degenerate)    -> supports SD-059/MECH-358: the bridge delivers GRADED sustained-survival benefit; promote on the evidence.
  | continuous FAIL, nav_control clears (env survivable)-> weakens SD-059/MECH-358 (bridge_insufficient_env_survivable): genuine negative evidence; the bridge fired but added no graded survival on a survivable env.
  | continuous FAIL, nav_control also fails            -> non_contributory (navigation_survival_competence_ceiling): a nav ceiling resurfacing, NOT a bridge verdict; route to a navigation/competence substrate.

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: evidence
claim_ids: [SD-059, MECH-358]  (scored governance evidence -- the redesigned bridge retest)
"""

from __future__ import annotations

import argparse
import json
import statistics
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

EXPERIMENT_TYPE = "v3_exq_603o_escape_affordance_bridge_continuous_retest"
QUEUE_ID = "V3-EXQ-603o"
CLAIM_IDS: List[str] = ["SD-059", "MECH-358"]  # scored evidence retest
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror 603l/603k exactly).
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

# Isolated hazard-avoidance Stage-H -- HARDER than 603l (FIX (a) HEADROOM).
# 603l used num_hazards=4, proximity_harm=0.1 and ARM_BASE_IA_ONLY hit the binary
# ceiling (G_H=1.0, 3/3). A harder regime (more hazards + higher proximity_harm)
# pulls the base arm into the deficit band G_H ~0.33-0.67 so the bridge has
# headroom. hazard_food_attraction stays 0.0 (clean avoidance signal -- foraging
# does NOT raise hazard exposure). If the base still saturates OR all arms
# collapse, the non-degeneracy guard self-routes substrate_not_ready_requeue.
HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 6          # 603l: 4  -> harder
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.2     # 603l: 0.1 -> harder
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

# --- SD-059 / MECH-358 escape-affordance bridge knobs (threat envelope matched
# to the same z_harm_a magnitude as the gate so the bridge engages under threat) ---
ESCAPE_THREAT_FLOOR = 0.1
ESCAPE_THREAT_REF = 0.35
ESCAPE_APPROACH_GAIN = 0.1
ESCAPE_BIAS_SCALE = 0.1

# --- SUBSTRATE FIX 1 (603k): Stage-H harm-pathway training -----------------------
HARM_PATHWAY_LR = 1e-3

# --- SUBSTRATE FIX 2 (603j): trained safety-half threat-absence signal ------------
ESCAPE_SAFETY_SIGNAL_THRESHOLD = 0.5

# Pre-registered gates (constants).
STAGE0_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0

# --- Continuous-metric PASS criterion (FIX (b)) ---
# margin = max(CONTINUOUS_EFFECT_SD_MULT * pstdev(delta), CONTINUOUS_ABS_FLOOR).
# SD-of-DELTA (not SD of the baseline level) + an absolute floor: a consistent
# delta collapses SD-of-delta to ~0 so the floor is load-bearing
# (feedback_effect_size_pass_gate_margin).
CONTINUOUS_EFFECT_SD_MULT = 0.5
CONTINUOUS_ABS_FLOOR = 10.0   # episode-length steps (~10% of the 603l base ~98)

# --- Non-degeneracy guard thresholds (FIX (a) + vacuity guard) ---
SURVIVAL_CEILING_FRAC = 0.9   # base at ceiling if mean >= 0.9 * steps_per_episode
COLLAPSE_FLOOR_FRAC = 0.15    # arm collapsed if mean <= 0.15 * steps_per_episode
CROSS_ARM_VAR_FLOOR = 1.0     # pstdev of per-arm mean continuous metric must exceed this

# 4 discriminative bridge arms (all on the 603i-INTACT base + both substrate fixes)
# + a nav-competence positive control. The 4 discriminative arms differ ONLY in
# the bridge.
ARMS = [
    {"label": "ARM_BASE_IA_ONLY", "bridge": False, "relief": False, "safety": False, "nav_control": False},
    {"label": "ARM_RELIEF_BRIDGE", "bridge": True, "relief": True, "safety": False, "nav_control": False},
    {"label": "ARM_SAFETY_BRIDGE", "bridge": True, "relief": False, "safety": True, "nav_control": False},
    {"label": "ARM_RELIEF_SAFETY_BRIDGE", "bridge": True, "relief": True, "safety": True, "nav_control": False},
    {"label": "ARM_NAV_CONTROL", "bridge": False, "relief": False, "safety": False, "nav_control": True},
]
BRIDGE_ARM_LABELS = {"ARM_RELIEF_BRIDGE", "ARM_SAFETY_BRIDGE", "ARM_RELIEF_SAFETY_BRIDGE"}


def _steps_per_episode(dry_run: bool) -> int:
    return 30 if dry_run else TRAIN_STEPS


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
        # The isolated Stage-H (603g amend) -- HARDER regime for headroom.
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        # NAV-COMPETENCE positive control: spawn IN the reef refuge so navigation
        # to safety is handed (reef-refuge-reachability). The ONLY difference of
        # ARM_NAV_CONTROL vs ARM_BASE_IA_ONLY.
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
        # Trains the harm-avoidance VALUATION pathway so nav/survival competence
        # holds. KEPT ON: the harder hazard regime (not removing this fix) is what
        # denies the base arm the binary ceiling -- see the module docstring.
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
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
        # Sensory z_harm stream (SD-010) so harm-pathway terms 2 + 4 engage, plus
        # the affective stream (SD-011) the base defensive chain keys on.
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        # E2_harm_s forward model (ARC-033) so harm-pathway term 4 engages.
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
        # SD-056 e2 contrastive warmup so the relief detector reads a trained-enough
        # world-forward (without it the relief credit re-starves like 603h).
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
        # Substrate fixes ON in every arm (603k + 603j).
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
        # HARDER hazard regime is part of the fingerprint (distinguishes 603o from 603l).
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, HAZARD_STAGE_PROXIMITY_HARM,
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR],
        "dry_run": bool(dry_run),
    }


def _continuous_from_lengths(lengths: List[int], steps_per_ep: int) -> Dict[str, float]:
    """Continuous survival readouts from a hazard-stage episode-length list."""
    if not lengths:
        return {"mean": 0.0, "median": 0.0, "auc": 0.0, "time_to_first_death": 0.0}
    n = len(lengths)
    mean = float(sum(lengths)) / float(n)
    median = float(statistics.median(lengths))
    # AUC-survival: normalised area under the per-episode survival curve in [0, 1].
    auc = float(sum(lengths)) / float(max(1, n * steps_per_ep))
    # time-to-first-death: first episode index (1-based) that ended before the cap.
    ttfd = float(n)  # never died within the stage -> full horizon
    for i, ln in enumerate(lengths):
        if ln < steps_per_ep:
            ttfd = float(i + 1)
            break
    return {"mean": mean, "median": median, "auc": auc, "time_to_first_death": ttfd}


def _aborted_record(arm_label: str, seed: int, stage: str, reason: str,
                    s0_peak: float = 0.0) -> Dict[str, Any]:
    return {
        "arm": arm_label, "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "hazard_stage_survival_pass": False,
        "hazard_stage_median_last_window": 0.0,
        "hazard_stage_mean_episode_length": 0.0,
        "hazard_stage_auc_survival": 0.0,
        "hazard_stage_time_to_first_death": 0.0,
        "hazard_stage_episode_lengths": [],
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


def _run_seed_arm(arm: Dict[str, Any], seed: int, dry_run: bool,
                  total_eps: int) -> Dict[str, Any]:
    """Full curriculum for one (arm, seed) cell. arm_cell resets all RNG on
    enter (order-independent) and stamps the fingerprint on the returned row."""
    steps_per_ep = _steps_per_episode(dry_run)
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
        hz_cont = _continuous_from_lengths(list(hz.episode_lengths), steps_per_ep)
        print(f"  [train] hazard_avoidance {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={hz.mean_episode_length:.1f} median_last={hz.median_last_window_episode_length:.1f}"
              f" auc={hz_cont['auc']:.3f} ttfd={hz_cont['time_to_first_death']:.0f}"
              f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
              f" pag_commits={pag_after_h.get('n_commits', 0)}"
              f" n_relief={bridge_after_h.get('mech358_n_relief_credit', 0)}"
              f" n_safety={bridge_after_h.get('mech358_n_safety_credit', 0)}"
              f" n_approach={bridge_after_h.get('mech358_n_approach_fires', 0)}",
              flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=hazard", flush=True)
            rec = _aborted_record(arm["label"], seed, "hazard", hz.abort_reason,
                                  s0_peak=s0.z_goal_norm_peak)
            rec["avoidance_gate_state"] = gate_after_h
            rec["escape_bridge_state"] = bridge_after_h
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
        seed_pass = bool(g_h)
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={arm['label']}"
              f" g_h={g_h} mean_len={hz.mean_episode_length:.1f} g0={g0} g1={g1} g2={g2}"
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
            "hazard_stage_survival_pass": g_h,
            "hazard_stage_median_last_window": float(hz.median_last_window_episode_length),
            "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
            "hazard_stage_auc_survival": float(hz_cont["auc"]),
            "hazard_stage_time_to_first_death": float(hz_cont["time_to_first_death"]),
            "hazard_stage_episode_lengths": [int(x) for x in hz.episode_lengths],
            "hazard_stage_n_episodes": int(hz.n_episodes),
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


def _arm_half_credit_frac(rows: List[Dict[str, Any]], key: str) -> float:
    """Fraction of seeds where the named bridge-credit counter incremented."""
    flags = [int((r.get("escape_bridge_state", {}) or {}).get(key, 0)) > 0 for r in rows]
    return _frac(flags)


def _arm_mean_continuous(rows: List[Dict[str, Any]]) -> float:
    vals = [float(r.get("hazard_stage_mean_episode_length", 0.0)) for r in rows]
    return float(sum(vals)) / float(len(vals)) if vals else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    steps_per_ep = _steps_per_episode(dry_run)
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
        engaged_flags, suppressed_flags = [], []
        for r in rows:
            gs = r.get("avoidance_gate_state", {}) or {}
            engaged_flags.append(
                (int(gs.get("mech357_n_credit", 0)) + int(gs.get("mech357_n_decay", 0))) > 0
            )
            suppressed_flags.append(int(gs.get("mech357_n_freeze_suppressed", 0)) > 0)
        pag_freeze_flags = [int(r.get("pag_n_commits", 0)) > 0 for r in rows]
        arm_results.append({
            "arm": arm["label"],
            "use_escape_affordance_bridge": bool(arm["bridge"]),
            "use_escape_relief_credit": bool(arm["relief"]),
            "use_escape_safety_credit": bool(arm["safety"]),
            "nav_control": bool(arm["nav_control"]),
            "g_h_frac": _frac(g_h_flags),
            "g0_frac": _frac(g0_flags),
            "g1_frac": _frac(g1_flags),
            "g2_frac": _frac(g2_flags),
            "gate_engaged_frac": _frac(engaged_flags),
            "gate_freeze_suppressed_frac": _frac(suppressed_flags),
            "pag_freeze_frac": _frac(pag_freeze_flags),
            "relief_credit_frac": _arm_half_credit_frac(rows, "mech358_n_relief_credit"),
            "safety_credit_frac": _arm_half_credit_frac(rows, "mech358_n_safety_credit"),
            "safety_credit_trained_frac": _arm_half_credit_frac(rows, "mech358_n_safety_credit_trained"),
            "mean_hazard_episode_length_arm": _arm_mean_continuous(rows),
            "per_seed_g_h": g_h_flags,
            "per_seed_g1": g1_flags,
            "per_seed_pag_n_commits": [int(r.get("pag_n_commits", 0)) for r in rows],
            "per_seed_hazard_mean_episode_length": [
                float(r.get("hazard_stage_mean_episode_length", 0.0)) for r in rows
            ],
            "per_seed_hazard_median_last_window": [
                r.get("hazard_stage_median_last_window", 0.0) for r in rows
            ],
            "per_seed_hazard_auc_survival": [
                float(r.get("hazard_stage_auc_survival", 0.0)) for r in rows
            ],
            "per_seed_hazard_time_to_first_death": [
                float(r.get("hazard_stage_time_to_first_death", 0.0)) for r in rows
            ],
            "per_seed_n_relief_credit": [
                int((r.get("escape_bridge_state", {}) or {}).get("mech358_n_relief_credit", 0))
                for r in rows
            ],
            "per_seed_n_safety_credit": [
                int((r.get("escape_bridge_state", {}) or {}).get("mech358_n_safety_credit", 0))
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

    base_means = base["per_seed_hazard_mean_episode_length"]

    # --- PRIMARY continuous acceptance (load-bearing) ---
    # Pick the bridge arm with the highest mean continuous metric across seeds, then
    # test that winning arm per-seed against base with an SD-of-delta + abs-floor margin.
    best_bridge_arm = max(bridge_arms, key=lambda a: a["mean_hazard_episode_length_arm"])
    best_bridge_label = best_bridge_arm["arm"]
    best_means = best_bridge_arm["per_seed_hazard_mean_episode_length"]
    deltas = [float(bm - bs) for bm, bs in zip(best_means, base_means)]
    mean_delta = float(sum(deltas)) / float(len(deltas)) if deltas else 0.0
    delta_sd = float(statistics.pstdev(deltas)) if len(deltas) > 1 else 0.0
    continuous_margin = max(CONTINUOUS_EFFECT_SD_MULT * delta_sd, CONTINUOUS_ABS_FLOOR)
    n_seeds_positive = sum(1 for d in deltas if d > continuous_margin)
    continuous_pass = bool(
        mean_delta > continuous_margin
        and (n_seeds_positive / float(len(deltas) if deltas else 1)) >= MIN_FRACTION
    )

    # --- READINESS PRECONDITIONS (non-vacuity SAFETY NET) ---
    pavlovian_reaction_present = bool(base["pag_freeze_frac"] >= MIN_FRACTION)
    gate_engaged = bool(base["gate_engaged_frac"] >= MIN_FRACTION)
    half_frac = {}
    half_frac["ARM_RELIEF_BRIDGE"] = relief["relief_credit_frac"]
    half_frac["ARM_SAFETY_BRIDGE"] = safety["safety_credit_frac"]
    half_frac["ARM_RELIEF_SAFETY_BRIDGE"] = min(
        both["relief_credit_frac"], both["safety_credit_frac"]
    )
    bridge_halves_nonvacuous = all(
        half_frac[lbl] >= MIN_FRACTION for lbl in BRIDGE_ARM_LABELS
    )
    readiness_met = bool(
        pavlovian_reaction_present and gate_engaged and bridge_halves_nonvacuous
    )
    g0_base_ok = bool(base["g0_frac"] >= MIN_FRACTION)

    # --- NON-DEGENERACY GUARDS (pre-registered) ---
    survival_ceiling = SURVIVAL_CEILING_FRAC * float(steps_per_ep)
    collapse_floor = COLLAPSE_FLOOR_FRAC * float(steps_per_ep)
    # base-at-ceiling: binary G_H 3/3 OR mean ep len at the survival ceiling on >=2/3.
    base_ceiling_seed_flags = [bm >= survival_ceiling for bm in base_means]
    base_at_ceiling = bool(
        base["g_h_frac"] >= 1.0 or _frac(base_ceiling_seed_flags) >= MIN_FRACTION
    )
    # vacuous-DV: zero cross-arm variance of the per-arm mean OR every arm collapsed.
    arm_means = [a["mean_hazard_episode_length_arm"] for a in arm_results]
    cross_arm_sd = float(statistics.pstdev(arm_means)) if len(arm_means) > 1 else 0.0
    all_arms_collapsed = all(m <= collapse_floor for m in arm_means)
    vacuous_dv = bool(cross_arm_sd < CROSS_ARM_VAR_FLOOR or all_arms_collapsed)
    base_has_deficit = bool(not base_at_ceiling)
    non_degenerate_ok = bool(base_has_deficit and not vacuous_dv)

    # nav-control clears: env survivable when nav-to-safety is handed (binary OR
    # continuous: nav mean exceeds base mean by the abs floor).
    nav_mean = nav["mean_hazard_episode_length_arm"]
    base_mean_arm = base["mean_hazard_episode_length_arm"]
    nav_control_clears = bool(
        nav["g_h_frac"] >= MIN_FRACTION or nav_mean > (base_mean_arm + CONTINUOUS_ABS_FLOOR)
    )

    # which bridge arm carried the continuous lift (interpretation grid)
    relief_lift = float(relief["mean_hazard_episode_length_arm"] - base_mean_arm)
    safety_lift = float(safety["mean_hazard_episode_length_arm"] - base_mean_arm)
    both_lift = float(both["mean_hazard_episode_length_arm"] - base_mean_arm)

    non_degenerate = True
    degeneracy_reason = ""

    if not readiness_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        run_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = ("readiness preconditions unmet (PAG freeze / ilPFC gate / "
                             "bridge-half credit below 2/3) -- comparison uninformative")
    elif base_at_ceiling:
        # The literal 603l failure: base at the survival ceiling -> bridge has no
        # headroom -> structurally unsatisfiable. Re-tune harder, do NOT score.
        outcome = "FAIL"
        readiness_route = "base_at_survival_ceiling_regime_too_easy"
        run_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = ("ARM_BASE_IA_ONLY at survival ceiling (G_H 3/3 or mean ep "
                             "len >= 0.9*steps on >=2/3 seeds) -- no bridge headroom; "
                             "harden the hazard regime or use partial harm-pathway training")
    elif vacuous_dv:
        outcome = "FAIL"
        readiness_route = "vacuous_dv_no_cross_arm_dynamic_range"
        run_direction = "non_contributory"
        non_degenerate = False
        degeneracy_reason = ("continuous metric has no dynamic range (cross-arm SD < floor "
                             "or all arms collapsed) -- regime too hard / budget too short")
    elif continuous_pass:
        outcome = "PASS"
        run_direction = "supports"
        if relief_lift > 0 and safety_lift <= 0 and best_bridge_label == "ARM_RELIEF_BRIDGE":
            readiness_route = "escape_affordance_bridge_lifts_survival_relief_only"
        elif safety_lift > 0 and relief_lift <= 0 and best_bridge_label == "ARM_SAFETY_BRIDGE":
            readiness_route = "escape_affordance_bridge_lifts_survival_safety_only"
        elif best_bridge_label == "ARM_RELIEF_SAFETY_BRIDGE":
            readiness_route = "escape_affordance_bridge_lifts_survival_both_required"
        else:
            readiness_route = "escape_affordance_bridge_lifts_survival"
    else:
        outcome = "FAIL"
        if nav_control_clears:
            readiness_route = "bridge_insufficient_env_survivable"
            run_direction = "weakens"
        else:
            readiness_route = "navigation_survival_competence_ceiling"
            run_direction = "non_contributory"

    # SD-059 (architecture) and MECH-358 (mechanism) are the same bridge and move
    # together with the run direction.
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
                           "(ARM_RELIEF_BRIDGE relief; ARM_SAFETY_BRIDGE safety; both-arm both). "
                           "Below-floor => the safety predictor did not populate ecologically "
                           "=> substrate_not_ready_requeue (NOT a false weakens).",
            "control": "ARM_RELIEF/SAFETY/RELIEF_SAFETY: bridge ON, SD-056 e2 warmup in P0, fed "
                       "harm stream, escape_use_trained_safety_signal + MECH-303/304/302 "
                       "predictors populating during Stage-H.",
            "measured": float(min(half_frac[lbl] for lbl in BRIDGE_ARM_LABELS)),
            "threshold": float(MIN_FRACTION),
            "met": bool(bridge_halves_nonvacuous),
        },
        {
            "name": "base_carries_deficit_not_at_survival_ceiling",
            "kind": "readiness",
            "description": "ARM_BASE_IA_ONLY must NOT be at the binary survival ceiling -- the "
                           "603l confound. Measured = base mean hazard-stage episode length; "
                           "ABOVE the ceiling (>= 0.9*steps on >=2/3 seeds, or G_H 3/3) means "
                           "the regime is too easy and the bridge has no headroom => "
                           "substrate_not_ready_requeue (HARDER regime). This is a CEILING "
                           "precondition: met = measured BELOW the ceiling.",
            "control": "harder Stage-H hazard regime (num_hazards=6, proximity_harm=0.2).",
            "measured": float(base_mean_arm),
            "threshold": float(survival_ceiling),
            "direction": "upper",
            "met": bool(base_has_deficit),
        },
        {
            "name": "continuous_metric_has_cross_arm_dynamic_range",
            "kind": "readiness",
            "description": "The per-arm mean hazard-stage episode length must vary across arms "
                           "(cross-arm SD >= floor) and not all collapse -- else the continuous "
                           "discrimination metric is vacuous => substrate_not_ready_requeue.",
            "control": "5 arms x 3 seeds on the harder Stage-H regime.",
            "measured": float(cross_arm_sd),
            "threshold": float(CROSS_ARM_VAR_FLOOR),
            "met": bool(not vacuous_dv),
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
        "base_not_at_survival_ceiling": bool(base_has_deficit),
        "cross_arm_continuous_dynamic_range": bool(not vacuous_dv),
        "nav_control_evaluated": bool(
            _frac([r.get("reached_hazard_stage", False) for r in rows_by_arm["ARM_NAV_CONTROL"]])
            >= MIN_FRACTION
        ),
    }
    criteria = [
        {"name": "continuous_best_bridge_mean_delta_clears_margin", "load_bearing": True,
         "passed": bool(mean_delta > continuous_margin)},
        {"name": "continuous_best_bridge_positive_on_2of3_seeds", "load_bearing": True,
         "passed": bool((n_seeds_positive / float(len(deltas) if deltas else 1)) >= MIN_FRACTION)},
    ]

    print(
        f"[{EXPERIMENT_TYPE}] mean_ep_len base={base_mean_arm:.1f} relief={relief['mean_hazard_episode_length_arm']:.1f}"
        f" safety={safety['mean_hazard_episode_length_arm']:.1f} both={both['mean_hazard_episode_length_arm']:.1f}"
        f" nav={nav_mean:.1f} | best_bridge={best_bridge_label} mean_delta={mean_delta:.2f}"
        f" margin={continuous_margin:.2f} n_pos={n_seeds_positive}/{len(deltas)}"
        f" | base_at_ceiling={base_at_ceiling} vacuous_dv={vacuous_dv} readiness={readiness_met}"
        f" -> outcome={outcome} route={readiness_route} direction={run_direction}",
        flush=True,
    )

    result: Dict[str, Any] = {
        "outcome": outcome,
        "continuous_pass": continuous_pass,
        "best_bridge_label": best_bridge_label,
        "continuous_mean_delta": mean_delta,
        "continuous_delta_per_seed": deltas,
        "continuous_delta_sd": delta_sd,
        "continuous_margin": continuous_margin,
        "continuous_n_seeds_positive": int(n_seeds_positive),
        "readiness_met": readiness_met,
        "pavlovian_reaction_present": pavlovian_reaction_present,
        "gate_engaged": gate_engaged,
        "bridge_halves_nonvacuous": bridge_halves_nonvacuous,
        "base_at_ceiling": base_at_ceiling,
        "vacuous_dv": vacuous_dv,
        "non_degenerate_ok": non_degenerate_ok,
        "cross_arm_sd": cross_arm_sd,
        "survival_ceiling": survival_ceiling,
        "collapse_floor": collapse_floor,
        "nav_control_clears": nav_control_clears,
        "relief_lift": relief_lift,
        "safety_lift": safety_lift,
        "both_lift": both_lift,
        "evidence_direction": run_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "non_degenerate": non_degenerate,
        "arm_results": arm_results,
        "acceptance": {
            "pass_rule": "PASS = readiness_met (PAG freezes + ilPFC gate engages on BASE >=2/3 "
                         "AND each enabled bridge half credits >=2/3) AND non-degenerate (base "
                         "NOT at survival ceiling AND cross-arm continuous dynamic range) AND "
                         "best_bridge mean hazard-stage episode-length delta > "
                         "max(0.5*pstdev(delta), 10.0) AND positive on >=2/3 seeds",
            "primary_metric": "mean_hazard_stage_episode_length (continuous)",
            "supplementary_metrics": ["median_last_window", "auc_survival",
                                      "time_to_first_death", "binary_G_H"],
            "min_fraction": MIN_FRACTION,
            "continuous_effect_sd_mult": CONTINUOUS_EFFECT_SD_MULT,
            "continuous_abs_floor": CONTINUOUS_ABS_FLOOR,
            "survival_ceiling_frac": SURVIVAL_CEILING_FRAC,
            "collapse_floor_frac": COLLAPSE_FLOOR_FRAC,
            "cross_arm_var_floor": CROSS_ARM_VAR_FLOOR,
            "steps_per_episode": steps_per_ep,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "mean_ep_len_base": base_mean_arm,
            "mean_ep_len_relief": relief["mean_hazard_episode_length_arm"],
            "mean_ep_len_safety": safety["mean_hazard_episode_length_arm"],
            "mean_ep_len_both": both["mean_hazard_episode_length_arm"],
            "mean_ep_len_nav_control": nav_mean,
            "g_h_base_frac": base["g_h_frac"],
            "g_h_relief_frac": relief["g_h_frac"],
            "g_h_safety_frac": safety["g_h_frac"],
            "g_h_both_frac": both["g_h_frac"],
            "g_h_nav_control_frac": nav["g_h_frac"],
            "relief_credit_frac": relief["relief_credit_frac"],
            "safety_credit_frac": safety["safety_credit_frac"],
            "safety_credit_trained_frac": safety["safety_credit_trained_frac"],
            "base_pag_freeze_frac": base["pag_freeze_frac"],
            "base_gate_engaged_frac": base["gate_engaged_frac"],
        },
        "interpretation": {
            "label": readiness_route,
            "readiness_route": readiness_route,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "grid": {
                "readiness_not_met": "substrate_not_ready_requeue -> non_contributory (fix gate/predictor)",
                "base_at_ceiling": "substrate_not_ready_requeue -> non_contributory (HARDER regime; the 603l confound)",
                "vacuous_dv": "substrate_not_ready_requeue -> non_contributory (easier regime / longer budget)",
                "continuous_pass": "supports SD-059/MECH-358 (graded sustained-survival benefit)",
                "continuous_fail_nav_control_clears": "weakens SD-059/MECH-358 (env survivable, bridge insufficient)",
                "continuous_fail_nav_control_fails": "navigation_survival_competence_ceiling -> non_contributory",
            },
        },
        "per_seed": per_seed,
    }
    if not non_degenerate:
        result["degeneracy_reason"] = degeneracy_reason
        # SD-059/MECH-358 move together: drop both from confidence/conflict scoring.
        result["non_degenerate_per_claim"] = {cid: False for cid in CLAIM_IDS}
    return result


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
        # evidence_direction + evidence_direction_per_claim set dynamically below.
        "evidence_direction": "non_contributory",
        "supersedes": "v3_exq_603l_escape_affordance_bridge_behavioural_retest",
        "depends_on": ["V3-EXQ-603i", "V3-EXQ-603j", "V3-EXQ-603k", "V3-EXQ-603l"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "SD-059 / MECH-358 relief/safety escape-affordance bridge "
                     "(ree_core/pfc/escape_affordance_bridge.py) over the SD-058/MECH-357 gate "
                     "in the scaffolded_sd054_onboarding Stage-H, on the 603k harm-pathway-trained "
                     "+ 603j trained-safety-signal substrate, with a HARDER hazard regime",
        "scores": "SD-059 (architecture) + MECH-358 (affordance-indexed relief/safety credit "
                  "+ threat-gated E3 approach bonus) -- behavioural re-test REDESIGN (continuous "
                  "survival metric + headroom), NOT a readiness diagnostic",
        "design_note": "603l autopsy-mandated REDESIGN carrying BOTH Section-7 fixes: (a) HEADROOM "
                       "-- harder Stage-H hazard regime (num_hazards 4->6, proximity_harm 0.1->0.2) "
                       "so ARM_BASE_IA_ONLY sits below the binary survival ceiling (603l hit G_H=1.0, "
                       "structurally unsatisfiable), keeping 603k harm-pathway training ON for nav "
                       "competence; (b) CONTINUOUS METRIC -- PRIMARY discrimination is mean hazard-stage "
                       "episode length with SD-of-delta + abs-floor margin (binary G_H supplementary). "
                       "5 arms x 3 seeds. Pre-registered non-degeneracy guards: base-at-ceiling and "
                       "vacuous-DV both self-route substrate_not_ready_requeue -> non_contributory + "
                       "non_degenerate=False, so a saturated or collapsed regime is never a false weakens.",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "primary_continuous_pass_rule": "readiness_met AND non_degenerate AND best_bridge mean "
                                            "hazard-stage episode-length delta > max(0.5*pstdev(delta), "
                                            "10.0) AND positive on >=2/3 seeds",
            "primary_metric": "mean_hazard_stage_episode_length",
            "binary_g_h_supplementary": "median episode length over last 10 Stage-H episodes >= 75",
            "nav_competence_control": "ARM_NAV_CONTROL (spawn-in-reef) -- the reef-refuge-reachability ceiling",
            "base_at_ceiling_guard": "base G_H 3/3 OR mean ep len >= 0.9*steps on >=2/3 -> substrate_not_ready_requeue",
            "vacuous_dv_guard": "cross-arm mean SD < 1.0 OR all arms collapse (mean <= 0.15*steps) -> substrate_not_ready_requeue",
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "min_fraction": MIN_FRACTION,
        },
        "substrate_fixes_on": {
            "harm_pathway_training_603k": {
                "scaffold_train_harm_pathway": True,
                "scaffold_harm_pathway_lr": HARM_PATHWAY_LR,
                "scaffold_harm_pathway_in_p0": True,
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
        "hazard_regime": {
            "num_hazards": HAZARD_STAGE_NUM_HAZARDS,
            "proximity_harm_scale": HAZARD_STAGE_PROXIMITY_HARM,
            "hazard_food_attraction": HAZARD_STAGE_HFA,
            "num_resources": HAZARD_STAGE_NUM_RESOURCES,
            "survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "note": "harder than 603l (num_hazards 4->6, proximity_harm 0.1->0.2) for base-arm headroom",
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
