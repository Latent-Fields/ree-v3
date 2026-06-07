"""
V3-EXQ-603g -- scaffolded_sd054_onboarding substrate-readiness run on the
2026-06-07 CURRICULUM-DECOMPOSITION amend (isolated hazard-avoidance Stage-H).

PURPOSE (substrate readiness, NOT governance evidence; claim_ids=[]):
The re-issued readiness gate for the GAP-2 cohort after the curriculum-
decomposition amend. V3-EXQ-603f (full budget, all prior foraging-competence
levers) PROVED the goal-formation + ecological-seeding chain is SOUND -- seed 44
foraged (P2 contact_rate 0.393, 85 events) AND seeded z_goal ecologically
(z_goal_norm_at_contact_peak 0.450 > 0.4) -- yet still FAILed because the P1
SURVIVAL / hazard-avoidance leg collapsed (G1 0/3; median episode len
12.5/38.0/28.5 vs gate 75; even the foraging seed 44 died at 28.5). The 603f
autopsy (failure_autopsy_V3-EXQ-603f_2026-06-07) diagnosed the cause: P1 couples
TWO competencies at once (goal-pipeline unfreeze + wean into the hazard band) and
the agent cannot acquire both simultaneously; P0 trains only in the safe reef
refuge, so survival competence never develops before P1.

THE AMEND THIS VALIDATES (the SINGLE new thing vs 603f): a separately-trained
ISOLATED HAZARD-AVOIDANCE STAGE (Stage-H) inserted between P0 and P1. Goal
pipeline FROZEN (seed_goal=False -> z_goal untouched -- the isolation), hazards
present (num_hazards 4, proximity_harm 0.1), foraging minimal (num_resources 2),
hazard_food_attraction=0 (so foraging does NOT raise hazard exposure -- clean
avoidance signal), midline spawn (the agent must navigate the hazard band; the
reef refuge stays available as the flee-to-safety attractor). P1 is then entered
by an already-survival-AND-goal-competent policy.

Curriculum: Stage-0 (forced feed) -> Stage-0b (consolidate) -> P0 (encoder
warm-up, goal frozen) -> Stage-H (isolated hazard avoidance, goal frozen) ->
P1 (combined wean) -> P2 (measure). Every other lever is identical to 603f.

SINGLE-ARM, all-levers-ON (readiness gate, not a discriminative ablation). 3
seeds [42, 43, 44].

PRE-REGISTERED SUBSTRATE-READINESS GATES (UNCHANGED from 603f; each >= 2/3 seeds;
do NOT retune):
  G0 stage0_positive_control : Stage-0 forced-feed z_goal_norm_peak > 0.4 (the
                               goal stream lights when fed -- ALSO the built-in
                               POSITIVE CONTROL; below-floor self-routes
                               substrate_not_ready_requeue, NOT a foraging verdict).
  G1 p1_survival             : P1 survival/foraging gate passed (median episode
                               length over last P1_STABILITY_WINDOW=10 episodes
                               >= P1_SURVIVAL_GATE_STEPS=75).
  G2 p2_contact              : P2 foraging contact_rate > 0.
EXPERIMENT PASS = G0 AND G1 AND G2 (each >= 2/3 seeds). The NEW Stage-H survival
readout G_H (hazard_stage_survival_pass) is reported as a DIAGNOSTIC only and is
NOT in the PASS rule -- it lets the manifest confirm the isolated stage achieved
avoidance before P1, which is the mechanism the amend bets on.

CANONICAL READINESS (also reported, NOT the PASS driver): substrate_readiness_from_results()
adds G3 = consumption-event-gated P2 z_goal (z_goal_norm_at_contact_peak > 0.4).

INTERPRETATION ON OUTCOME (this run weights no claim; in every case diagnostic):
  PASS (G0+G1+G2 >=2/3) AND canonical G3 also clears -> substrate ready: a
        follow-on /governance + /queue-experiment action (NOT automatic, NOT in
        this script) flips substrate_queue scaffolded_sd054_onboarding ready=true
        and resumes the GAP-2 downstream cohorts.
  G0 fails (forced-feed cannot light z_goal) -> substrate_not_ready_requeue: the
        positive control itself failed; the goal-formation substrate (not the
        foraging scaffold) is the problem -> re-route /implement-substrate.
  G0 passes but G1/G2 < 2/3 -> foraging-competence still open: the
        curriculum-decomposition amend did not lift survival+contact to >=2/3 ->
        re-route /failure-autopsy (the isolated hazard-avoidance stage needs
        strengthening, or a deeper survival-substrate gap remains). G_H tells
        whether the isolated stage itself achieved avoidance (if G_H passes but
        G1 still fails, the survival competence did not TRANSFER into the combined
        P1 -- catastrophic interference; if G_H also fails, the isolated stage
        itself could not learn avoidance at this budget).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: diagnostic
claim_ids: []  (substrate readiness; gates downstream cohorts, weights no claim)
supersedes: V3-EXQ-603f (same readiness gate; the curriculum-decomposition amend
  inserts Stage-H, the single change vs 603f).
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
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    classify_interpretation_branch,
    evaluate_substrate_gate,
    stage_plan,
    substrate_readiness_from_results,
)

EXPERIMENT_TYPE = "v3_exq_603g_scaffolded_sd054_substrate_readiness"
QUEUE_ID = "V3-EXQ-603g"
CLAIM_IDS: List[str] = []  # substrate readiness; tags no claim
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "v3_exq_603f_scaffolded_sd054_substrate_readiness"

SEEDS = [42, 43, 44]
CONDITION_LABEL = "ALL_LEVERS_ON_PLUS_HAZARD_STAGE"

# Goal-pipeline / encoder dims (mirror 603f exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0  # SD-012 amplification (two-part-fix precondition)

# Restored budget P0/P1 = 100/50 (matches 603f). Stage0/Stage0b/P2 as in 603f.
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3   # holds the hazard/food-attraction anneal low (early P1)
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3       # P2 measurement guard (admits contact)

# Foraging-competence amend levers (as 603f).
P1_REEF_SPAWN_HOLD_FRACTION = 0.4   # graded reef-spawn weaning (survival lever)

# --- THE CURRICULUM-DECOMPOSITION AMEND (the single new thing vs 603f) ---
# Isolated hazard-avoidance Stage-H, inserted between P0 and P1.
HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0            # hazards drift randomly (decoupled from foraging)
HAZARD_STAGE_PROXIMITY_HARM = 0.1  # target-level harm so avoidance is incentivised
HAZARD_STAGE_SPAWN_IN_REEF = False  # midline spawn -> must navigate the hazard band
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# 634c seeding calibration.
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9

# SD-057 cue-recall bridge.
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

# Pre-registered gates (constants; NOT derived from the run's own statistics).
STAGE0_ZGOAL_GATE = 0.4
STAGE0B_RETENTION_GATE = 0.75
P2_ZGOAL_GATE = 0.4
CONTACT_GATE = 0.0
MIN_FRACTION = 2.0 / 3.0


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
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
        # --- developmental-window / consolidation amend (2026-06-03b) ---
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        # --- 634c seeding calibration (2026-06-03c) ---
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        # --- foraging-competence residual amend (2026-06-05) ---
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        # --- SD-057 cue-recall bridge (wean-to-wild contact lever) ---
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        # --- THE CURRICULUM-DECOMPOSITION AMEND (2026-06-07): isolated Stage-H ---
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
    )
    # Dry-run: scale the survival gates so short episodes can clear them.
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
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
    )
    cfg.latent.use_resource_encoder = True  # SD-015 (direct, not via from_dims)
    return cfg


def _aborted_seed_record(seed: int, stage: str, reason: str,
                         s0_peak: float = 0.0, s0b_pass: bool = False) -> Dict[str, Any]:
    return {
        "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "stage0b_retention_gate_passed": bool(s0b_pass),
        "p0_mean_episode_length": 0.0,
        "hazard_stage_survival_pass": False,
        "hazard_stage_median_last_window": 0.0,
        "p1_survival_pass": False,
        "p1_median_last_window_episode_length": 0.0,
        "p2_contact_rate": 0.0,
        "p2_contact_steps": 0,
        "p2_num_contact_events": 0,
        "p2_z_goal_norm_peak": 0.0,
        "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_n_cue_recall_fires": 0,
        "g0_stage0_zgoal": False,
        "g1_p1_survival": False,
        "g2_p2_contact": False,
        "reached_hazard_stage": False,
        "reached_p1": False,
        "reached_p2": False,
        "seed_pass": False,
    }


def _run_seed(seed: int, dry_run: bool, total_eps: int):
    """Returns (per_seed_dict, stage0_result, p1_result, p2_metrics) so the
    caller can feed the scheduler's own substrate_readiness_from_results()."""
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")

    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {CONDITION_LABEL}", flush=True)

    # Stage 0 -- forced-benefit nursery (the z_goal-formation positive control).
    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    token_bank = int(getattr(s0, "token_bank_size_end", 0))
    print(
        f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}"
        f" z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}"
        f" token_bank_size_end={token_bank}",
        flush=True,
    )
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return (_aborted_seed_record(seed, "stage0", s0.abort_reason,
                                     s0_peak=s0.z_goal_norm_peak), None, None, None)

    # Stage 0b -- PROTECTED consolidation.
    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(
        f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}"
        f" retention={s0b.retention_ratio:.3f}"
        f" gate={'pass' if s0b.retention_gate_passed else 'FAIL'}",
        flush=True,
    )
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return (_aborted_seed_record(seed, "stage0b", s0b.abort_reason,
                                     s0_peak=s0.z_goal_norm_peak,
                                     s0b_pass=s0b.retention_gate_passed), None, None, None)

    # Stage 1 -- guided low-conflict warm-up (run_p0, goal frozen, reef refuge).
    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(
        f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
        f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
        flush=True,
    )
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        rec = _aborted_seed_record(seed, "p0", p0.abort_reason,
                                   s0_peak=s0.z_goal_norm_peak,
                                   s0b_pass=s0b.retention_gate_passed)
        rec["p0_mean_episode_length"] = float(p0.mean_episode_length)
        return (rec, s0, None, None)

    # Stage-H -- ISOLATED HAZARD-AVOIDANCE (the curriculum-decomposition amend).
    # Goal pipeline FROZEN inside run_hazard_avoidance; survival learned alone.
    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(
        f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
        f" mean_len={hz.mean_episode_length:.1f}"
        f" median_last={hz.median_last_window_episode_length:.1f}"
        f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}"
        f" rv={hz.final_running_variance:.5f}",
        flush=True,
    )
    if hz.aborted:
        print(f"verdict: FAIL seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        rec = _aborted_seed_record(seed, "hazard", hz.abort_reason,
                                   s0_peak=s0.z_goal_norm_peak,
                                   s0b_pass=s0b.retention_gate_passed)
        rec["p0_mean_episode_length"] = float(p0.mean_episode_length)
        return (rec, s0, None, None)

    # Stage 2+3 -- easy->guarded foraging (run_p1; goal re-unfrozen). P1 is now
    # entered by an already-survival-competent policy (the bet the amend makes).
    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(
        f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
        f" median_last={p1.median_last_window_episode_length:.1f}"
        f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}"
        f" reef_spawn_eps={getattr(p1, 'n_reef_spawn_episodes', 0)}"
        f" refresh={p1.n_contact_refresh_updates}"
        f" decay_only={p1.n_decay_only_updates}",
        flush=True,
    )

    # Stage 4 -- frozen-policy guarded measurement (run_p2).
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    p2_cue_diag = dict(getattr(p2, "cue_diag", {}) or {})
    cue_fires = int(p2_cue_diag.get("n_cue_recall_fires", getattr(p2, "n_cue_recall_fires", 0)))
    print(
        f"  [train] p2_measure seed={seed} ep {done}/{total_eps}"
        f" contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events}"
        f" z_goal_frozen={p2.z_goal_norm_peak_max:.4f}"
        f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}"
        f" cue_fires={cue_fires} hfa_used={p2.hazard_food_attraction_used:.2f}",
        flush=True,
    )

    # Per-seed task gates (G0/G1/G2). G3 (consumption-gated) reported, not in PASS.
    # G_H (Stage-H survival) is a DIAGNOSTIC readout, NOT in the PASS rule.
    g0 = bool(s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE)
    g1 = bool(p1.survival_gate_passed)
    g2 = bool(p2.contact_rate > CONTACT_GATE)
    seed_pass = bool(g0 and g1 and g2)
    print(
        f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed}"
        f" g0={g0} g1={g1} g2={g2} g_h={bool(hz.survival_gate_passed)}",
        flush=True,
    )

    rec = {
        "seed": seed,
        "aborted_at": None,
        "abort_reason": "",
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "stage0_z_goal_formed": bool(s0.z_goal_formed),
        "stage0_token_bank_size_end": token_bank,
        "stage0b_retention_ratio": float(s0b.retention_ratio),
        "stage0b_retention_gate_passed": bool(s0b.retention_gate_passed),
        "p0_mean_episode_length": float(p0.mean_episode_length),
        # NEW: isolated hazard-avoidance Stage-H survival diagnostic (G_H).
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "hazard_stage_median_last_window": float(hz.median_last_window_episode_length),
        "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
        "hazard_stage_n_episodes": int(hz.n_episodes),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "p1_median_last_window_episode_length": float(p1.median_last_window_episode_length),
        "p1_n_reef_spawn_episodes": int(getattr(p1, "n_reef_spawn_episodes", 0)),
        "p1_n_contact_refresh_updates": int(p1.n_contact_refresh_updates),
        "p1_n_decay_only_updates": int(p1.n_decay_only_updates),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_contact_steps": int(p2.contact_steps),
        "p2_num_contact_events": int(p2.num_contact_events),
        "p2_z_goal_norm_peak": float(p2.z_goal_norm_peak_max),                  # frozen (diagnostic)
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),  # consumption-gated (G3)
        "p2_n_cue_recall_fires": cue_fires,
        "p2_hazard_food_attraction_used": float(p2.hazard_food_attraction_used),
        "g0_stage0_zgoal": g0,
        "g1_p1_survival": g1,
        "g2_p2_contact": g2,
        "reached_hazard_stage": True,
        "reached_p1": True,
        "reached_p2": True,
        "seed_pass": seed_pass,
    }
    return (rec, s0, p1, p2)


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


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

    per_seed: List[Dict[str, Any]] = []
    stage0_results = []
    p1_results = []
    p2_metrics = []
    for s in seeds:
        rec, s0, p1, p2 = _run_seed(s, dry_run, total_eps)
        per_seed.append(rec)
        if s0 is not None:
            stage0_results.append(s0)
        if p1 is not None:
            p1_results.append(p1)
        if p2 is not None:
            p2_metrics.append(p2)

    n = len(per_seed)
    # --- The task's three pre-registered gates (each >= 2/3 seeds) -- UNCHANGED ---
    g0_pass = _frac([r["g0_stage0_zgoal"] for r in per_seed]) >= MIN_FRACTION
    g1_pass = _frac([r["g1_p1_survival"] for r in per_seed]) >= MIN_FRACTION
    g2_pass = _frac([r["g2_p2_contact"] for r in per_seed]) >= MIN_FRACTION
    overall_pass = bool(g0_pass and g1_pass and g2_pass)
    outcome = "PASS" if overall_pass else "FAIL"

    # --- NEW diagnostic: Stage-H isolated-avoidance gate (G_H). Reported only;
    # NOT in the PASS rule. Tells whether the isolated stage achieved survival
    # competence before P1 (the mechanism the amend bets on). ---
    g_h_frac = _frac([r.get("hazard_stage_survival_pass", False) for r in per_seed])
    hazard_stage_gate = {
        "g_h_hazard_stage_survival": bool(g_h_frac >= MIN_FRACTION),
        "g_h_fraction": float(g_h_frac),
        "min_fraction": MIN_FRACTION,
        "per_seed_g_h": [r.get("hazard_stage_survival_pass", False) for r in per_seed],
        "per_seed_hazard_median_last_window": [
            r.get("hazard_stage_median_last_window", 0.0) for r in per_seed
        ],
        "note": "diagnostic only -- NOT in the PASS rule (PASS = G0 AND G1 AND G2). "
                "G_H pass + G1 fail => survival did not TRANSFER into combined P1 "
                "(catastrophic interference); G_H fail => the isolated stage itself "
                "could not learn avoidance at this budget.",
    }

    task_gate = {
        "g0_stage0_positive_control": bool(g0_pass),
        "g1_p1_survival": bool(g1_pass),
        "g2_p2_contact": bool(g2_pass),
        "overall_pass": overall_pass,
        "min_fraction": MIN_FRACTION,
        "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
        "contact_gate": CONTACT_GATE,
        "per_seed_g0": [r["g0_stage0_zgoal"] for r in per_seed],
        "per_seed_g1": [r["g1_p1_survival"] for r in per_seed],
        "per_seed_g2": [r["g2_p2_contact"] for r in per_seed],
    }

    # --- Canonical design-doc readiness (consumption-event-gated G3); reported,
    # NOT the PASS driver. Only meaningful when seeds reached P2. ---
    if stage0_results and p1_results and p2_metrics:
        canonical = substrate_readiness_from_results(
            stage0_results, p1_results, p2_metrics,
            z_goal_gate=STAGE0_ZGOAL_GATE,
            contact_gate=CONTACT_GATE,
            min_fraction=MIN_FRACTION,
            use_consumption_gated_g3=True,
        )
    else:
        canonical = evaluate_substrate_gate(
            [r["stage0_z_goal_norm_peak"] for r in per_seed],
            [r["p1_survival_pass"] for r in per_seed],
            [r["p2_z_goal_norm_at_contact_peak"] for r in per_seed],
            [r["p2_contact_rate"] for r in per_seed],
            z_goal_gate=STAGE0_ZGOAL_GATE, contact_gate=CONTACT_GATE, min_fraction=MIN_FRACTION,
        )
        canonical["g3_source"] = "z_goal_norm_at_contact_peak"

    branch = classify_interpretation_branch(canonical)

    # --- Diagnostic adjudication structures (interpretation gate) ---
    # G0 (Stage-0 forced-feed z_goal>0.4) is the built-in POSITIVE CONTROL for
    # z_goal formation. Readiness precondition: the measured statistic is the
    # SAME the load-bearing G0 routes on (the >=2/3-seed fraction); the indexer
    # recomputes met = measured>=threshold.
    g0_measured = _frac([r["g0_stage0_zgoal"] for r in per_seed])
    preconditions = [
        {
            "name": "stage0_forced_feed_lights_zgoal",
            "kind": "readiness",
            "description": "Stage-0 forced supra-threshold benefit must light z_goal "
                           "(>0.4 on >=2/3 seeds). Positive control that the goal-FORMATION "
                           "substrate works, decoupled from foraging competence. Below-floor "
                           "=> substrate_not_ready_requeue (goal-formation broken), NOT a "
                           "foraging-competence verdict.",
            "control": "Stage-0 nursery feeds a forced supra-threshold benefit every step "
                       "regardless of contact (run_stage0_nursery).",
            "measured": float(g0_measured),
            "threshold": float(MIN_FRACTION),
            "met": bool(g0_pass),
        },
    ]
    frac_reached_p1 = _frac([r.get("reached_p1", False) for r in per_seed])
    frac_reached_p2 = _frac([r.get("reached_p2", False) for r in per_seed])
    frac_reached_hazard = _frac([r.get("reached_hazard_stage", False) for r in per_seed])
    criteria_non_degenerate = {
        "G0_stage0": bool(n >= 2),
        "G1_survival": bool(frac_reached_p1 >= MIN_FRACTION),
        "G2_contact": bool(frac_reached_p2 >= MIN_FRACTION),
        "G_H_hazard_stage": bool(frac_reached_hazard >= MIN_FRACTION),
    }
    criteria = [
        {"name": "G0_stage0_positive_control", "load_bearing": True, "passed": bool(g0_pass)},
        {"name": "G1_p1_survival", "load_bearing": True, "passed": bool(g1_pass)},
        {"name": "G2_p2_contact", "load_bearing": True, "passed": bool(g2_pass)},
    ]

    # Readiness route: a failed positive control (G0) is "substrate not ready".
    if not g0_pass:
        readiness_route = "substrate_not_ready_requeue"
    elif overall_pass and canonical.get("substrate_gate_passed"):
        readiness_route = "substrate_ready_flip_scaffolded_sd054_onboarding"
    elif overall_pass and not canonical.get("substrate_gate_passed"):
        readiness_route = "task_gate_clear_canonical_g3_open"
    else:
        readiness_route = "foraging_competence_open"

    print(
        f"[{EXPERIMENT_TYPE}] task_gate G0={g0_pass} G1={g1_pass} G2={g2_pass}"
        f" (G_H_diag={hazard_stage_gate['g_h_hazard_stage_survival']}) -> outcome={outcome}",
        flush=True,
    )
    print(
        f"[{EXPERIMENT_TYPE}] canonical_4gate(incl consumption-gated G3)="
        f"{canonical.get('substrate_gate_passed')} branch={branch}"
        f" readiness_route={readiness_route}",
        flush=True,
    )

    return {
        "outcome": outcome,
        "substrate_gate_passed_task_three": overall_pass,
        "task_gate": task_gate,
        "hazard_stage_gate": hazard_stage_gate,
        "canonical_readiness_gate": canonical,
        "interpretation": {
            "label": branch,
            "readiness_route": readiness_route,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
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
        "evidence_direction": "non_contributory",  # diagnostic; tags no claim
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (curriculum-decomposition amend: isolated "
                     "hazard-avoidance Stage-H, 2026-06-07)",
        "condition": CONDITION_LABEL,
        "amend_note": "The SINGLE change vs V3-EXQ-603f is the isolated hazard-avoidance "
                      "Stage-H inserted between P0 and P1 (scaffold_hazard_stage_enabled=True + "
                      "run_hazard_avoidance called between run_p0 and run_p1). Goal pipeline "
                      "frozen during Stage-H (z_goal untouched); hazards present, foraging "
                      "minimal, hfa=0, midline spawn. Routed by failure_autopsy_V3-EXQ-603f_2026-06-07.",
        "predecessor": "V3-EXQ-603f (full budget, all prior foraging-competence levers; "
                       "G0 2/3, G1 0/3, G2 1/3 -- survival leg the sole blocker).",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "g0_stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
            "g1_p1_survival": "median episode length over last 10 P1 episodes >= 75",
            "g2_contact_gate": CONTACT_GATE,
            "g3_canonical_consumption_gated": "z_goal_norm_at_contact_peak > 0.4 (reported; "
                                              "canonical design-doc gate, NOT the task PASS driver)",
            "g_h_hazard_stage_survival": "median episode length over last 10 Stage-H episodes "
                                         ">= 75 (DIAGNOSTIC; NOT in the PASS rule)",
            "min_fraction": MIN_FRACTION,
            "pass_rule": "PASS = G0 AND G1 AND G2 (each >= 2/3 seeds)",
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET,
            "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "train_steps": TRAIN_STEPS, "p1_anneal_hold_fraction": P1_HOLD_FRACTION,
            "p0_num_hazards": P0_NUM_HAZARDS, "p2_hfa_guard": P2_HFA_GUARD,
            "p1_reef_spawn_hold_fraction": P1_REEF_SPAWN_HOLD_FRACTION,
            "hazard_stage_num_hazards": HAZARD_STAGE_NUM_HAZARDS,
            "hazard_stage_num_resources": HAZARD_STAGE_NUM_RESOURCES,
            "hazard_stage_hazard_food_attraction": HAZARD_STAGE_HFA,
            "hazard_stage_proximity_harm_scale": HAZARD_STAGE_PROXIMITY_HARM,
            "hazard_stage_spawn_in_reef_half": HAZARD_STAGE_SPAWN_IN_REEF,
            "hazard_stage_survival_gate_steps": HAZARD_STAGE_SURVIVAL_GATE_STEPS,
            "auto_reconcile_gating_to_seeding": True,
            "seeding_gain": SEED_GAIN, "seeding_benefit_threshold": SEED_BENEFIT_THRESHOLD,
            "seeding_drive_floor": SEED_DRIVE_FLOOR,
            "cue_recall_bridge_enabled": True, "cue_n_resource_types": N_RESOURCE_TYPES,
            "stage0_bind_incentive_token": True, "cue_recall_gain": CUE_RECALL_GAIN,
            "developmental_window_enabled": True, "stage0b_enabled": True,
            "contact_gated_goal_updates": True,
            "z_goal_enabled": True, "drive_weight": DRIVE_WEIGHT,
        },
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
