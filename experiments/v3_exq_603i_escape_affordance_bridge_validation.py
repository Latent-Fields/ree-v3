"""
V3-EXQ-603i -- SD-059 / MECH-358 relief/safety escape-affordance bridge:
4-arm discriminative Stage-H validation + nav-competence positive control.

PURPOSE (substrate readiness, NOT governance evidence; claim_ids=[]):
The validation gate for the SD-059 / MECH-358 escape-affordance bridge landed
2026-06-08. V3-EXQ-603h (the SD-058/MECH-357 instrumental-avoidance Stage-H
validation) FAILed engaged-but-insufficient: the ilPFC gate suppressed the
MECH-279 PAG freeze on all INTACT seeds (readiness met) but G_H_INTACT was 0/3
and did not beat LESION; seed-43 reached scalar avoidance_efficacy 0.633 and
survived WORST (11.0). Diagnosis (failure_autopsy_V3-EXQ-603h_2026-06-08):
MECH-357's avoidance_efficacy is a GLOBAL SCALAR that only penalises the no-op
class -- compute_action_bias by design "does NOT compute the escape direction".
The agent un-froze without acquiring a DIRECTED escape. Moscarello & LeDoux
2013: active avoidance needs the LA/BA->NAcc relief/safety action-credit half,
not only the freeze-suppression half. REE owns relief (MECH-302) + safety
(MECH-303/304) but they were UNWIRED to avoidance. SD-059/MECH-358 wires them.

THE SUBSTRATE THIS VALIDATES: SD-059 / MECH-358 EscapeAffordanceBridge
(ree_core/pfc/escape_affordance_bridge.py) -- extends the MECH-357 scalar
efficacy into a per-first-action-class credit table with two halves: RELIEF (a
directed action under threat that drops z_harm_a credits relief_affordance[class],
MECH-302-consistent) and SAFETY (a directed action after which threat is absent
credits safety_affordance[class], MECH-303/304-consistent). Under future threat,
E3 receives a bounded, threat-context-gated NEGATIVE (favoured) approach bias
toward credited classes -- the DIRECTED escape. Three guards: bias_scale clamp,
threat-context gate (zero when safe), per-tick leak.

DESIGN: thought-intake Section 5 4-arm + the autopsy's nav-competence co-branch.
5 arms x 3 seeds [42, 43, 44]. ALL arms carry the full 603h-INTACT config
(MECH-279 PAG + SD-058/MECH-357 ilPFC gate + driver + fed harm stream); the
4 discriminative arms differ ONLY in the escape-affordance bridge:
  ARM_BASE_IA_ONLY            = bridge OFF (SD-058/MECH-357 exactly as 603h INTACT).
  ARM_RELIEF_BRIDGE           = bridge ON, relief half only.
  ARM_SAFETY_BRIDGE           = bridge ON, safety half only.
  ARM_RELIEF_SAFETY_BRIDGE    = bridge ON, both halves.
  ARM_NAV_CONTROL (positive control) = bridge OFF, Stage-H spawns IN the reef
    refuge (navigation to safety handed). This is the reef-refuge-reachability
    nav-competence control: if even the handed-safety arm fails G_H, the env's
    survival ceiling is unreachable at this budget (a navigation/survival-
    competence gap), so a flat G_H across the bridge arms is NOT attributable to
    the bridge. If ARM_NAV_CONTROL clears G_H but the bridge arms do not, the env
    is survivable and the bridge is the locus.

THE LOAD-BEARING PREREQUISITE (shared with 603h): scaffold_feed_harm_stream=True
so z_harm_a is populated (~0.34 in Stage-H) -- without it MECH-279/SD-035/the
gate/the bridge are all inert. SD-056 e2 contrastive warmup stays in P0 so the
relief detector (per-action d(z_harm_a)/dt) reads a trained-enough world-forward;
without it the relief credit re-starves the way 603h's n_credit did (6/0 on 2/3).

PRE-REGISTERED ACCEPTANCE (constants; NOT derived from the run's own statistics):
  PRIMARY (does the escape-affordance bridge train the survival leg?):
    best_bridge_G_H >= 2/3 (some bridge arm clears the Stage-H survival gate --
            median last-window episode length over Stage-H >= 75), AND
    best_bridge_G_H_frac > G_H_BASE_frac (the bridge lifts survival over the
            SD-058/MECH-357-only baseline).
  PASS = readiness_met AND both PRIMARY conditions hold.

READINESS PRECONDITIONS (non-vacuity; the discriminative design must instantiate):
  (1) PAG freezes on the base arm (pag_n_commits>0 on >=2/3 seeds) AND the ilPFC
      gate engages (n_credit+n_decay>0 on >=2/3) -- the 603h chain is present.
  (2) EACH ENABLED BRIDGE HALF fires NON-VACUOUSLY: on ARM_RELIEF_BRIDGE the
      relief credit increments (mech358_n_relief_credit>0 on >=2/3 seeds); on
      ARM_SAFETY_BRIDGE the safety credit increments (mech358_n_safety_credit>0);
      on ARM_RELIEF_SAFETY_BRIDGE both. If an enabled half never credits, a G_H
      comparison on that arm is UNINFORMATIVE -> self-route
      substrate_not_ready_requeue (the relief detector needs a trained encoder/
      world-forward; it re-starved on 603h), NEVER a substrate verdict.

INTERPRETATION ON OUTCOME (this run weights no claim; diagnostic in every case):
  Readiness unmet -> substrate_not_ready_requeue.
  PASS -> escape_affordance_bridge_trains_survival_leg (relief_only / safety_only /
    both_required per which arm cleared); follow-on /governance + /queue-experiment
    runs the full 603g-style readiness gate to flip substrate_queue ready (NOT here).
  No bridge arm lifts G_H, BUT ARM_NAV_CONTROL clears G_H -> bridge_insufficient
    (env survivable; the bridge is wired+non-vacuous but does not deliver Stage-H
    survival at this budget/config) -> /failure-autopsy (tune/redesign bridge).
  No bridge arm lifts G_H AND ARM_NAV_CONTROL also fails G_H ->
    navigation_survival_competence_ceiling (the deeper hazard-navigation gap the
    user kept as a co-equal branch; route to a navigation/competence substrate,
    NOT the bridge).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: diagnostic
claim_ids: []  (substrate readiness; validates SD-059/MECH-358 wiring, weights no claim)
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
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_603i_escape_affordance_bridge_validation"
QUEUE_ID = "V3-EXQ-603i"
CLAIM_IDS: List[str] = []  # substrate readiness; tags no claim
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror 603h exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Budgets (mirror 603h full budget).
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

# Isolated hazard-avoidance Stage-H (the 603g curriculum-decomposition amend).
HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# 634c seeding calibration + SD-057 cue-recall bridge (mirror 603h).
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (the avoidance-learning driver) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
# Calibrated to the Stage-H z_harm_a magnitude ~0.34 (mirror 603h).
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

# --- SD-059 / MECH-358 escape-affordance bridge knobs (threat envelope matched
# to the same z_harm_a magnitude as the gate so the bridge engages under threat) ---
ESCAPE_THREAT_FLOOR = 0.1
ESCAPE_THREAT_REF = 0.35
ESCAPE_APPROACH_GAIN = 0.1
ESCAPE_BIAS_SCALE = 0.1

# Pre-registered gates (constants).
STAGE0_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0

# 4 discriminative bridge arms (all on the 603h-INTACT base) + a nav-competence
# positive control. use_ia / driver are True everywhere (the SD-058/MECH-357 gate
# is the base both the bridge extends and the nav control runs on top of).
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
        # The isolated Stage-H (603g amend).
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        # NAV-COMPETENCE positive control: spawn IN the reef refuge so navigation
        # to safety is handed (reef-refuge-reachability). All other arms spawn
        # midline (must navigate the hazard band). This is the ONLY difference of
        # ARM_NAV_CONTROL vs ARM_BASE_IA_ONLY.
        scaffold_hazard_stage_spawn_in_reef_half=bool(nav_control),
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver (active on ALL arms -- the
        # bridge extends the gate; the nav control runs the gate too).
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE (all arms): feed the env harm stream so z_harm_a is populated.
        scaffold_feed_harm_stream=True,
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
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
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
        "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
                    P1_BUDGET, P2_BUDGET, TRAIN_STEPS],
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
        print(f"  [train] hazard_avoidance {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" median_last={hz.median_last_window_episode_length:.1f}"
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
              f" g_h={g_h} g0={g0} g1={g1} g2={g2}"
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
            "per_seed_g_h": g_h_flags,
            "per_seed_g1": g1_flags,
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

    # --- PRIMARY pre-registered acceptance (load-bearing) ---
    best_bridge_g_h = max(a["g_h_frac"] for a in bridge_arms)
    best_bridge_clears = bool(best_bridge_g_h >= MIN_FRACTION)
    best_bridge_beats_base = bool(best_bridge_g_h > base["g_h_frac"])
    primary_pass = bool(best_bridge_clears and best_bridge_beats_base)

    # --- READINESS PRECONDITIONS (non-vacuity) ---
    # (1) The 603h chain is present on the base arm: PAG freezes + ilPFC gate engages.
    pavlovian_reaction_present = bool(base["pag_freeze_frac"] >= MIN_FRACTION)
    gate_engaged = bool(base["gate_engaged_frac"] >= MIN_FRACTION)
    # (2) Each ENABLED bridge half fires non-vacuously on its arm. Per-arm the
    # measured frac is the MIN over enabled halves (both halves must fire on the
    # both-arm); a bridge arm with no enabled half is N/A (skipped).
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
    nav_control_clears = bool(nav["g_h_frac"] >= MIN_FRACTION)

    # --- which half carried the lift (for the interpretation grid) ---
    relief_clears = bool(relief["g_h_frac"] >= MIN_FRACTION and relief["g_h_frac"] > base["g_h_frac"])
    safety_clears = bool(safety["g_h_frac"] >= MIN_FRACTION and safety["g_h_frac"] > base["g_h_frac"])
    both_clears = bool(both["g_h_frac"] >= MIN_FRACTION and both["g_h_frac"] > base["g_h_frac"])

    if not readiness_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
    elif primary_pass:
        outcome = "PASS"
        if relief_clears and not safety_clears:
            readiness_route = "escape_affordance_bridge_trains_survival_leg_relief_only"
        elif safety_clears and not relief_clears:
            readiness_route = "escape_affordance_bridge_trains_survival_leg_safety_only"
        elif both_clears and not (relief_clears or safety_clears):
            readiness_route = "escape_affordance_bridge_trains_survival_leg_both_required"
        else:
            readiness_route = "escape_affordance_bridge_trains_survival_leg"
    else:
        outcome = "FAIL"
        if nav_control_clears:
            readiness_route = "bridge_insufficient_env_survivable"
        else:
            readiness_route = "navigation_survival_competence_ceiling"

    # Diagnostic adjudication structures.
    preconditions = [
        {
            "name": "pag_freeze_and_ilpfc_gate_engage_on_base",
            "kind": "readiness",
            "description": "The 603h chain must be present on ARM_BASE_IA_ONLY: PAG freezes "
                           "(pag_n_commits>0 on >=2/3 seeds) AND the ilPFC gate engages "
                           "(n_credit+n_decay>0 on >=2/3). Below-floor => the gate/freeze "
                           "substrate the bridge extends is itself inert => "
                           "substrate_not_ready_requeue, NOT a bridge verdict.",
            "control": "ARM_BASE_IA_ONLY: PAG + ilPFC gate + driver + fed harm stream (= 603h INTACT).",
            "measured": float(min(base["pag_freeze_frac"], base["gate_engaged_frac"])),
            "threshold": float(MIN_FRACTION),
            "met": bool(pavlovian_reaction_present and gate_engaged),
        },
        {
            "name": "each_enabled_bridge_half_fires_nonvacuously",
            "kind": "readiness",
            "description": "Each ENABLED bridge half must increment its credit on >=2/3 seeds "
                           "(ARM_RELIEF_BRIDGE: mech358_n_relief_credit>0; ARM_SAFETY_BRIDGE: "
                           "mech358_n_safety_credit>0; ARM_RELIEF_SAFETY_BRIDGE: both). This is "
                           "the SAME mechanism the PRIMARY G_H lift routes on -- the bridge "
                           "cannot lift survival if it never credited an escape affordance. "
                           "Below-floor => the relief detector starved (needs a trained encoder/"
                           "world-forward; it re-starved on 603h n_credit 6/0) => "
                           "substrate_not_ready_requeue, NOT a bridge verdict.",
            "control": "ARM_RELIEF/SAFETY/RELIEF_SAFETY: bridge ON, SD-056 e2 warmup in P0, fed "
                       "harm stream so a directed action under threat that drops z_harm_a / "
                       "terminates the threat actually credits.",
            "measured": float(min(half_frac[lbl] for lbl in BRIDGE_ARM_LABELS)),
            "threshold": float(MIN_FRACTION),
            "met": bool(bridge_halves_nonvacuous),
        },
        {
            "name": "stage0_forced_feed_lights_zgoal_on_base",
            "kind": "readiness",
            "description": "Stage-0 forced supra-threshold benefit lights z_goal (>0.4) on "
                           ">=2/3 base seeds -- the goal-FORMATION positive control (shared "
                           "from 603g/603h). Confirms the curriculum is intact.",
            "control": "run_stage0_nursery forced-feed.",
            "measured": float(base["g0_frac"]),
            "threshold": float(MIN_FRACTION),
            "met": bool(g0_base_ok),
        },
    ]
    criteria_non_degenerate = {
        # All arms reached Stage-H so the G_H comparison is non-degenerate.
        "arms_reached_hazard_stage": bool(
            _frac([r.get("reached_hazard_stage", False) for r in per_seed]) >= MIN_FRACTION
        ),
        # The bridge genuinely fired (some bridge arm credited an affordance).
        "bridge_credit_nonzero": bool(
            max(relief["relief_credit_frac"], safety["safety_credit_frac"],
                both["relief_credit_frac"], both["safety_credit_frac"]) > 0.0
        ),
        # The nav-competence control is informative (it ran to Stage-H).
        "nav_control_evaluated": bool(
            _frac([r.get("reached_hazard_stage", False) for r in rows_by_arm["ARM_NAV_CONTROL"]])
            >= MIN_FRACTION
        ),
    }
    criteria = [
        {"name": "best_bridge_G_H_clears_2of3", "load_bearing": True, "passed": bool(best_bridge_clears)},
        {"name": "best_bridge_G_H_beats_BASE", "load_bearing": True, "passed": bool(best_bridge_beats_base)},
    ]

    print(
        f"[{EXPERIMENT_TYPE}] G_H base={base['g_h_frac']:.2f} relief={relief['g_h_frac']:.2f}"
        f" safety={safety['g_h_frac']:.2f} both={both['g_h_frac']:.2f} nav_control={nav['g_h_frac']:.2f}"
        f" | relief_credit={relief['relief_credit_frac']:.2f} safety_credit={safety['safety_credit_frac']:.2f}"
        f" | readiness_met={readiness_met} -> outcome={outcome} route={readiness_route}",
        flush=True,
    )

    return {
        "outcome": outcome,
        "primary_pass": primary_pass,
        "best_bridge_g_h_frac": best_bridge_g_h,
        "best_bridge_clears": best_bridge_clears,
        "best_bridge_beats_base": best_bridge_beats_base,
        "readiness_met": readiness_met,
        "pavlovian_reaction_present": pavlovian_reaction_present,
        "gate_engaged": gate_engaged,
        "bridge_halves_nonvacuous": bridge_halves_nonvacuous,
        "nav_control_clears": nav_control_clears,
        "relief_clears": relief_clears,
        "safety_clears": safety_clears,
        "both_clears": both_clears,
        "arm_results": arm_results,
        "acceptance": {
            "pass_rule": "PASS = readiness_met (PAG freezes + ilPFC gate engages on BASE >=2/3 "
                         "AND each enabled bridge half credits >=2/3) AND best_bridge_G_H >= 2/3 "
                         "AND best_bridge_G_H_frac > G_H_BASE_frac",
            "min_fraction": MIN_FRACTION,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "g_h_base_frac": base["g_h_frac"],
            "g_h_relief_frac": relief["g_h_frac"],
            "g_h_safety_frac": safety["g_h_frac"],
            "g_h_both_frac": both["g_h_frac"],
            "g_h_nav_control_frac": nav["g_h_frac"],
            "best_bridge_g_h_frac": best_bridge_g_h,
            "relief_credit_frac": relief["relief_credit_frac"],
            "safety_credit_frac": safety["safety_credit_frac"],
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
                "relief_only_pass": "missing phasic negative-reinforcement (relief) credit",
                "safety_only_pass": "missing learned threat-absence (safety) predictor",
                "both_required_pass": "avoidance needs complementary relief + safety bridge",
                "neither_and_nav_control_passes": "bridge insufficient (env survivable) -> /failure-autopsy",
                "neither_and_nav_control_fails": "navigation/survival-competence ceiling -> nav substrate, NOT the bridge",
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
        "evidence_direction": "non_contributory",  # diagnostic; tags no claim
        "supersedes": None,  # 603h FAIL stays in the record; 603i validates the new bridge
        "depends_on": "V3-EXQ-603h",
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "SD-059 / MECH-358 relief/safety escape-affordance bridge "
                     "(ree_core/pfc/escape_affordance_bridge.py) over the SD-058/MECH-357 gate "
                     "in the scaffolded_sd054_onboarding Stage-H",
        "validates": "SD-059 (architecture) + MECH-358 (affordance-indexed relief/safety credit "
                     "+ threat-gated E3 approach bonus)",
        "design_note": "thought-intake Section 5 4-arm + the failure_autopsy_V3-EXQ-603h_2026-06-08 "
                       "nav-competence co-branch. 5 arms x 3 seeds, all on the 603h-INTACT base "
                       "(MECH-279 PAG + SD-058/MECH-357 ilPFC gate + driver + fed harm stream + "
                       "SD-056 e2 warmup): ARM_BASE_IA_ONLY (bridge OFF), ARM_RELIEF_BRIDGE, "
                       "ARM_SAFETY_BRIDGE, ARM_RELIEF_SAFETY_BRIDGE (bridge halves), and "
                       "ARM_NAV_CONTROL (bridge OFF, Stage-H spawns IN the reef refuge -- the "
                       "reef-refuge-reachability nav-competence positive control). Non-vacuity: "
                       "each enabled bridge half must credit >=2/3 seeds before G_H is scored "
                       "(the relief detector reads z_harm_a/world-forward; it re-starves without "
                       "a trained encoder, the 603h n_credit 6/0 lesson). Routed by "
                       "failure_autopsy_V3-EXQ-603h_2026-06-08.",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "primary_pass_rule": "readiness_met AND best_bridge_G_H >= 2/3 AND best_bridge_G_H_frac > G_H_BASE_frac",
            "g_h_hazard_stage_survival": "median episode length over last 10 Stage-H episodes >= 75",
            "nav_competence_control": "ARM_NAV_CONTROL (spawn-in-reef) G_H -- the reef-refuge-reachability ceiling",
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "min_fraction": MIN_FRACTION,
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
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
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
