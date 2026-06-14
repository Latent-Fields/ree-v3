"""
V3-EXQ-603p -- BASE harm-landscape discriminativeness diagnostic (diagnose-first
for the SD-059/MECH-358 escape-affordance-bridge retest re-queue).

PURPOSE (claim-free diagnostic; claim_ids=[]):
The 2026-06-11 V3-EXQ-603o redesign (the headroom + continuous-survival-metric
retest of the bridge, supersedes V3-EXQ-603l) self-routed substrate_not_ready_requeue.
Its continuous metric ALREADY showed a strong bridge lift (ARM_RELIEF_SAFETY_BRIDGE
mean Stage-H survival ~50.3 vs ARM_BASE_IA_ONLY ~18.65), and every readiness gate
passed EXCEPT one: `harm_landscape_discriminative_on_base` -- the base arm's trained
E3 harm landscape was discriminative (harm_eval_range >= 0.02) on only 1/3 seeds at
the harder Stage-H regime (num_hazards=6, proximity_harm=0.15). Without a discriminative
base harm landscape on >=2/3 seeds the base-vs-bridge comparison is confounded (is the
bridge ADDING value, or COMPENSATING for a base whose E3 harm gradient never trained?),
so 603o correctly refused to score.

This diagnostic LOCATES the cause BEFORE committing a 603p/603q evidence re-design:
is the base-harm-landscape failure (a) a HAZARD-REGIME-DIFFICULTY effect (the harder
regime kills the agent so fast the harm pathway gets too few harm-relevant samples to
train), or (b) a HARM-TRAINING-STRENGTH effect (the pathway just needs a stronger
gradient signal to converge at the hard regime)? The answer dictates the corrected
evidence design: ease proximity_harm to the hardest regime that still trains, vs
strengthen harm-pathway training at proximity_harm=0.15.

DESIGN: BASE arm only (bridge OFF), 4 cells x 3 seeds, run through Stage-H only (the
discriminativeness readout is computed at the end of run_hazard_avoidance; P1/P2 are
irrelevant to the harm-landscape question and skipped to save compute). All cells carry
the full 603i-INTACT defensive base + the 603k harm-pathway training + 603j trained
safety-signal fixes; they differ ONLY in the hazard proximity_harm and the
harm-pathway learning rate:
  ARM_REGIME_0p10        proximity_harm=0.10, harm_lr=1e-3  (POSITIVE CONTROL -- easiest
                         regime; 603k trained a discriminative harm landscape here, so
                         it should clear harm_eval_range>=0.02 on >=2/3. If even this
                         fails, the harm-training pathway or the readiness metric is
                         broken -- a substrate_not_ready_requeue, NOT a regime story.)
  ARM_REGIME_0p12        proximity_harm=0.12, harm_lr=1e-3
  ARM_REGIME_0p15        proximity_harm=0.15, harm_lr=1e-3  (reproduces the 603o base
                         regime that cleared on only 1/3 seeds)
  ARM_HARMTRAIN_3X_0p15  proximity_harm=0.15, harm_lr=3e-3  (training-strength rescue:
                         does a 3x harm-pathway LR restore discriminativeness at the
                         hardest regime, holding the regime fixed?)

PRIMARY readout (the SAME statistic the 603o verdict routed on -- same-statistic rule):
per cell, harm_disc_frac = fraction of seeds with harm_eval_range >= HARM_DISC_RANGE_FLOOR
(0.02). Supplementary: mean harm_eval_range, mean harm_eval_prox_corr, base Stage-H mean
survival.

INTERPRETATION GRID (one row per outcome -> next action):
  | POSITIVE CONTROL (0p10) does NOT clear harm_disc on >=2/3            -> substrate_not_ready_requeue: harm-pathway training or the harm_eval_range readiness metric is broken even at the easy regime; diagnose the harm-pathway optimizer/loss before any regime tuning (do NOT read this as a regime-difficulty result).
  | positive control clears AND a hard-regime cell (0p12/0p15) clears    -> regime_difficulty_threshold_located: the corrected 603p evidence design uses the HARDEST proximity_harm that still clears on >=2/3 (max base headroom that keeps the harm landscape trainable).
  | positive control clears, 0p15 fails, BUT ARM_HARMTRAIN_3X_0p15 clears-> harm_train_strength_rescues_hard_regime: keep proximity_harm=0.15 in 603p and raise the harm-pathway LR (3x) so the base harm landscape trains; max headroom preserved.
  | positive control clears, only the eased regimes clear, no train rescue-> hard_regime_breaks_harm_landscape_use_eased_regime: ease proximity_harm to the hardest clearing cell for 603p (training strength alone does not rescue 0.15).
  | NOTHING clears (incl positive control)                              -> substrate_not_ready_requeue: deeper harm-pathway substrate issue; escalate to /failure-autopsy on the harm-pathway training, do not queue a 603p evidence run.

This is a DIAGNOSTIC: it weights no claim. Its output is the regime / training-strength
parameter for the corrected SD-059/MECH-358 evidence re-queue (603q).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler).

experiment_purpose: diagnostic
claim_ids: []  (claim-free -- locates the harm-landscape parameter for the evidence re-queue)
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

EXPERIMENT_TYPE = "v3_exq_603p_base_harm_landscape_discriminativeness_diagnostic"
QUEUE_ID = "V3-EXQ-603p"
CLAIM_IDS: List[str] = []  # claim-free diagnostic
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]

# Goal-pipeline / encoder dims (mirror 603l/603o exactly).
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Budgets through Stage-H (P1/P2 NOT run -- diagnostic stops after Stage-H).
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
TRAIN_STEPS = 200
P0_NUM_HAZARDS = 1
P1_HOLD_FRACTION = 0.3
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

# Isolated hazard-avoidance Stage-H. num_hazards fixed at the 603o value (6);
# proximity_harm + harm_pathway_lr are the per-cell diagnostic axes.
HAZARD_STAGE_BUDGET = 40
HAZARD_STAGE_NUM_HAZARDS = 6
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# 634c seeding calibration + SD-057 cue-recall bridge (mirror 603l/603o).
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

# --- SUBSTRATE FIX 1 (603k): Stage-H harm-pathway training (harm_lr per cell) -----
HARM_PATHWAY_LR_STANDARD = 1e-3
HARM_PATHWAY_LR_STRONG = 3e-3

# --- SUBSTRATE FIX 2 (603j): trained safety-half threat-absence signal ------------
ESCAPE_SAFETY_SIGNAL_THRESHOLD = 0.5

# Pre-registered gates (constants).
STAGE0_ZGOAL_GATE = 0.4
MIN_FRACTION = 2.0 / 3.0
# Same statistic + floor the 603o readiness gate routed on (603i flat ~0.002; 603k ~0.133).
HARM_DISC_RANGE_FLOOR = 0.02

# BASE arm only (bridge OFF). Diagnostic axes: proximity_harm + harm_pathway_lr.
ARMS = [
    {"label": "ARM_REGIME_0p10", "proximity_harm": 0.10, "harm_lr": HARM_PATHWAY_LR_STANDARD},
    {"label": "ARM_REGIME_0p12", "proximity_harm": 0.12, "harm_lr": HARM_PATHWAY_LR_STANDARD},
    {"label": "ARM_REGIME_0p15", "proximity_harm": 0.15, "harm_lr": HARM_PATHWAY_LR_STANDARD},
    {"label": "ARM_HARMTRAIN_3X_0p15", "proximity_harm": 0.15, "harm_lr": HARM_PATHWAY_LR_STRONG},
]
POSITIVE_CONTROL = "ARM_REGIME_0p10"


def _make_scaffold_cfg(dry_run: bool, proximity_harm: float, harm_lr: float
                       ) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, steps = 2, 2, 5, 5, 30
    else:
        stage0, stage0b, p0, hazard, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET, TRAIN_STEPS
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        # P1/P2 budgets present in the dataclass but those stages are NOT called here.
        scaffold_p1_episode_budget=5,
        scaffold_p2_episode_budget=2,
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
        # Isolated Stage-H -- per-cell proximity_harm; num_hazards fixed at 6.
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=float(proximity_harm),
        scaffold_hazard_stage_spawn_in_reef_half=False,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver (active on ALL cells).
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE: feed the env harm stream so z_harm / z_harm_a populate.
        scaffold_feed_harm_stream=True,
        # SUBSTRATE FIX 1 (603k): Stage-H harm-pathway training -- LR per cell.
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=float(harm_lr),
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """BASE config -- bridge OFF (the harm-landscape question is about the base arm)."""
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
        # Bridge OFF (base arm).
        use_escape_affordance_bridge=False,
        use_escape_relief_credit=False,
        use_escape_safety_credit=False,
        # 603j trained safety-signal substrate present (no-op without the bridge, but
        # the predictors populate; matches the 603o base config exactly).
        escape_use_trained_safety_signal=True,
        escape_safety_signal_threshold=ESCAPE_SAFETY_SIGNAL_THRESHOLD,
        use_contextual_safety_terrain=True,
        use_conditioned_safety_store=True,
        use_suffering_derivative_comparator=True,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


def _config_slice(arm: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    return {
        "arm": arm["label"],
        "bridge": False,
        "proximity_harm": float(arm["proximity_harm"]),
        "harm_pathway_lr": float(arm["harm_lr"]),
        "num_hazards": HAZARD_STAGE_NUM_HAZARDS,
        "use_instrumental_avoidance": True,
        "scaffold_avoidance_driver_enabled": True,
        "use_pag_freeze_gate": True,
        "pag_theta_freeze": PAG_THETA_FREEZE,
        "pag_duration_input_threshold": PAG_DURATION_INPUT_THRESHOLD,
        "avoidance_threat_ref": AVOIDANCE_THREAT_REF,
        "feed_harm_stream": True,
        "e2_action_contrastive_enabled": True,
        "scaffold_train_harm_pathway": True,
        "use_harm_stream": True,
        "use_e2_harm_s_forward": True,
        "escape_use_trained_safety_signal": True,
        "use_contextual_safety_terrain": True,
        "use_conditioned_safety_store": True,
        "use_suffering_derivative_comparator": True,
        "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
        "budgets": [STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET, TRAIN_STEPS],
        "hazard_stage": [HAZARD_STAGE_NUM_HAZARDS, HAZARD_STAGE_NUM_RESOURCES,
                         HAZARD_STAGE_HFA, float(arm["proximity_harm"]),
                         HAZARD_STAGE_SURVIVAL_GATE_STEPS],
        "seeding": [SEED_GAIN, SEED_BENEFIT_THRESHOLD, SEED_DRIVE_FLOOR],
        "dry_run": bool(dry_run),
    }


def _aborted_record(arm_label: str, seed: int, stage: str, reason: str,
                    s0_peak: float = 0.0) -> Dict[str, Any]:
    return {
        "arm": arm_label, "seed": seed, "aborted_at": stage, "abort_reason": reason,
        "stage0_z_goal_norm_peak": float(s0_peak),
        "harm_eval_range": 0.0,
        "harm_eval_prox_corr": float("nan"),
        "harm_disc_pass": False,
        "hazard_stage_mean_episode_length": 0.0,
        "hazard_stage_survival_pass": False,
        "g0_stage0_zgoal": bool(s0_peak > STAGE0_ZGOAL_GATE),
        "pag_n_commits": 0,
        "gate_engaged": False,
        "reached_hazard_stage": stage not in ("stage0", "stage0b", "p0"),
        "seed_pass": False,
    }


def _run_seed_arm(arm: Dict[str, Any], seed: int, dry_run: bool,
                  total_eps: int) -> Dict[str, Any]:
    with arm_cell(
        seed,
        config_slice=_config_slice(arm, dry_run),
        script_path=Path(__file__),
        config_slice_declared=True,
    ) as cell:
        scaffold_cfg = _make_scaffold_cfg(dry_run, arm["proximity_harm"], arm["harm_lr"])
        device = torch.device("cpu")
        probe_env = _build_env(scaffold_cfg, "hazard")
        probe_env.reset()
        agent = REEAgent(_make_config(probe_env)).to(device)
        scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

        print(f"Seed {seed} Condition {arm['label']}", flush=True)

        def _pag_commits() -> int:
            p = getattr(agent, "pag_freeze_gate", None)
            return int(dict(p.diagnostics).get("n_commits", 0)) if p is not None else 0

        def _gate_engaged() -> bool:
            g = getattr(agent, "instrumental_avoidance", None)
            if g is None:
                return False
            gs = g.get_state()
            return (int(gs.get("mech357_n_credit", 0)) + int(gs.get("mech357_n_decay", 0))) > 0

        s0 = scheduler.run_stage0_nursery(agent, device)
        done = s0.n_episodes
        print(f"  [train] stage0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" z_goal_peak={s0.z_goal_norm_peak:.4f}", flush=True)
        if s0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0", flush=True)
            rec = _aborted_record(arm["label"], seed, "stage0", s0.abort_reason, s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        s0b = scheduler.run_stage0b_consolidation(
            agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
        done += s0b.n_episodes
        if s0b.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=stage0b", flush=True)
            rec = _aborted_record(arm["label"], seed, "stage0b", s0b.abort_reason, s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        p0 = scheduler.run_p0(agent, device)
        done += p0.n_episodes
        print(f"  [train] p0 {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={p0.mean_episode_length:.1f} rv={p0.final_running_variance:.5f}",
              flush=True)
        if p0.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=p0", flush=True)
            rec = _aborted_record(arm["label"], seed, "p0", p0.abort_reason, s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        # Stage-H -- isolated hazard-avoidance; harm_discriminativeness measured here.
        hz = scheduler.run_hazard_avoidance(agent, device)
        done += hz.n_episodes
        harm_disc = dict(hz.harm_discriminativeness or {})
        harm_eval_range = float(harm_disc.get("harm_eval_range", 0.0))
        harm_eval_prox_corr = harm_disc.get("harm_eval_prox_corr", float("nan"))
        try:
            harm_eval_prox_corr = float(harm_eval_prox_corr)
        except (TypeError, ValueError):
            harm_eval_prox_corr = float("nan")
        harm_disc_pass = bool(harm_eval_range >= HARM_DISC_RANGE_FLOOR)
        pag_commits = _pag_commits()
        gate_eng = _gate_engaged()
        print(f"  [train] hazard_avoidance {arm['label']} seed={seed} ep {done}/{total_eps}"
              f" mean_len={hz.mean_episode_length:.1f}"
              f" harm_range={harm_eval_range:.4f} (floor {HARM_DISC_RANGE_FLOOR}: "
              f"{'PASS' if harm_disc_pass else 'FAIL'})"
              f" harm_corr={harm_eval_prox_corr:.3f} pag_commits={pag_commits}"
              f" gate_engaged={gate_eng}", flush=True)
        if hz.aborted:
            print(f"verdict: FAIL seed={seed} arm={arm['label']} aborted_at=hazard", flush=True)
            rec = _aborted_record(arm["label"], seed, "hazard", hz.abort_reason, s0.z_goal_norm_peak)
            cell.stamp(rec)
            return rec

        seed_pass = harm_disc_pass
        print(f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} arm={arm['label']}"
              f" harm_range={harm_eval_range:.4f} mean_len={hz.mean_episode_length:.1f}", flush=True)

        rec = {
            "arm": arm["label"],
            "seed": seed,
            "aborted_at": None,
            "abort_reason": "",
            "proximity_harm": float(arm["proximity_harm"]),
            "harm_pathway_lr": float(arm["harm_lr"]),
            "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
            "harm_eval_range": harm_eval_range,
            "harm_eval_prox_corr": harm_eval_prox_corr,
            "harm_disc_pass": harm_disc_pass,
            "hazard_stage_mean_episode_length": float(hz.mean_episode_length),
            "hazard_stage_median_last_window": float(hz.median_last_window_episode_length),
            "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
            "g0_stage0_zgoal": bool(s0.z_goal_norm_peak > STAGE0_ZGOAL_GATE),
            "pag_n_commits": pag_commits,
            "gate_engaged": gate_eng,
            "harm_discriminativeness": harm_disc,
            "reached_hazard_stage": True,
            "seed_pass": seed_pass,
        }
        cell.stamp(rec)
        return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _mean(vals: List[float]) -> float:
    finite = [v for v in vals if v == v]  # drop NaN
    return float(sum(finite)) / float(len(finite)) if finite else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5
    else:
        total_eps = STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET

    arm_results: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    rows_by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS:
        rows = [_run_seed_arm(arm, s, dry_run, total_eps) for s in seeds]
        rows_by_arm[arm["label"]] = rows
        per_seed.extend(rows)
        disc_flags = [bool(r.get("harm_disc_pass", False)) for r in rows]
        g0_flags = [bool(r.get("g0_stage0_zgoal", False)) for r in rows]
        pag_flags = [int(r.get("pag_n_commits", 0)) > 0 for r in rows]
        gate_flags = [bool(r.get("gate_engaged", False)) for r in rows]
        arm_results.append({
            "arm": arm["label"],
            "proximity_harm": float(arm["proximity_harm"]),
            "harm_pathway_lr": float(arm["harm_lr"]),
            "harm_disc_frac": _frac(disc_flags),
            "mean_harm_eval_range": _mean([float(r.get("harm_eval_range", 0.0)) for r in rows]),
            "mean_harm_eval_prox_corr": _mean([r.get("harm_eval_prox_corr", float("nan")) for r in rows]),
            "mean_hazard_episode_length": _mean(
                [float(r.get("hazard_stage_mean_episode_length", 0.0)) for r in rows]),
            "g0_frac": _frac(g0_flags),
            "pag_freeze_frac": _frac(pag_flags),
            "gate_engaged_frac": _frac(gate_flags),
            "per_seed_harm_eval_range": [float(r.get("harm_eval_range", 0.0)) for r in rows],
            "per_seed_harm_disc_pass": disc_flags,
            "per_seed_hazard_mean_episode_length": [
                float(r.get("hazard_stage_mean_episode_length", 0.0)) for r in rows],
            "arm_fingerprint": [r.get("arm_fingerprint") for r in rows],
        })

    by_label = {a["arm"]: a for a in arm_results}
    pos = by_label[POSITIVE_CONTROL]
    pos_control_clears = bool(pos["harm_disc_frac"] >= MIN_FRACTION)

    # Which cells clear the harm-landscape floor on >=2/3 seeds.
    clearing = [a for a in arm_results if a["harm_disc_frac"] >= MIN_FRACTION]
    # Hardest regime (max proximity_harm) at STANDARD lr that clears.
    std_clearing = [a for a in clearing if a["harm_pathway_lr"] == HARM_PATHWAY_LR_STANDARD]
    hardest_std_clearing = (
        max(std_clearing, key=lambda a: a["proximity_harm"]) if std_clearing else None
    )
    arm_0p15 = by_label["ARM_REGIME_0p15"]
    arm_strong = by_label["ARM_HARMTRAIN_3X_0p15"]
    train_rescue = bool(
        arm_strong["harm_disc_frac"] >= MIN_FRACTION
        and arm_0p15["harm_disc_frac"] < MIN_FRACTION
    )
    any_actionable = bool(len(clearing) > 0)

    # cross-arm dynamic range over harm_eval_range (non-vacuity).
    arm_ranges = [a["mean_harm_eval_range"] for a in arm_results]
    cross_arm_range_spread = float(max(arm_ranges) - min(arm_ranges)) if arm_ranges else 0.0

    if not pos_control_clears:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        recommendation = ("harm-pathway training or the harm_eval_range readiness metric is "
                          "broken even at the easy regime (positive control 0p10 below floor on "
                          ">=2/3 seeds); diagnose the harm-pathway optimizer/loss before any "
                          "regime tuning -- do NOT read this as a regime-difficulty result")
    elif train_rescue:
        outcome = "PASS"
        label = "harm_train_strength_rescues_hard_regime"
        recommendation = ("keep proximity_harm=0.15 in the 603q evidence re-queue and raise the "
                           "harm-pathway LR to 3e-3 -- training strength restores base harm-landscape "
                           "discriminativeness at the hardest regime (max headroom preserved)")
    elif hardest_std_clearing is not None:
        ph = hardest_std_clearing["proximity_harm"]
        outcome = "PASS"
        if hardest_std_clearing["arm"] in ("ARM_REGIME_0p12", "ARM_REGIME_0p15"):
            label = "regime_difficulty_threshold_located"
        else:
            label = "hard_regime_breaks_harm_landscape_use_eased_regime"
        recommendation = (f"the 603q evidence re-queue uses proximity_harm={ph:.2f} (the hardest "
                          f"standard-LR regime that keeps the base harm landscape discriminative on "
                          f">=2/3 seeds -- max base headroom that still trains)")
    else:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        recommendation = ("nothing beyond the easy positive control clears; deeper harm-pathway "
                          "substrate issue -- escalate to /failure-autopsy on harm-pathway training, "
                          "do NOT queue a 603q evidence run")

    print(
        f"[{EXPERIMENT_TYPE}] harm_disc_frac "
        + " ".join(f"{a['arm'].replace('ARM_','')}={a['harm_disc_frac']:.2f}(range~{a['mean_harm_eval_range']:.4f})"
                   for a in arm_results)
        + f" | pos_ctrl_clears={pos_control_clears} train_rescue={train_rescue}"
        + f" -> outcome={outcome} label={label}",
        flush=True,
    )
    print(f"[{EXPERIMENT_TYPE}] RECOMMENDATION: {recommendation}", flush=True)

    preconditions = [
        {
            "name": "positive_control_harm_landscape_discriminative",
            "kind": "readiness",
            "description": "ARM_REGIME_0p10 (easiest regime; 603k trained a discriminative harm "
                           "landscape here) must clear harm_eval_range>=0.02 on >=2/3 seeds -- the "
                           "SAME statistic the diagnostic routes on, measured on the known-positive "
                           "control. Below-floor => harm-pathway training or the readiness metric is "
                           "broken (substrate_not_ready_requeue), NOT a regime-difficulty verdict.",
            "control": "ARM_REGIME_0p10: bridge OFF, full 603k harm-pathway training, easiest "
                       "proximity_harm=0.10.",
            "measured": float(pos["harm_disc_frac"]),
            "threshold": float(MIN_FRACTION),
            "met": bool(pos_control_clears),
        },
        {
            "name": "harm_eval_range_has_cross_arm_dynamic_range",
            "kind": "readiness",
            "description": "mean harm_eval_range must vary across the difficulty/training cells "
                           "(spread >= floor) -- else the readout is pinned and the diagnostic is "
                           "vacuous.",
            "control": "4 cells (proximity_harm 0.10/0.12/0.15 + 3x-LR at 0.15).",
            "measured": float(cross_arm_range_spread),
            "threshold": float(HARM_DISC_RANGE_FLOOR),
            "met": bool(cross_arm_range_spread >= HARM_DISC_RANGE_FLOOR),
        },
        {
            "name": "stage0_forced_feed_lights_zgoal_on_positive_control",
            "kind": "readiness",
            "description": "Stage-0 forced supra-threshold benefit lights z_goal (>0.4) on >=2/3 "
                           "positive-control seeds -- the curriculum is intact.",
            "control": "run_stage0_nursery forced-feed.",
            "measured": float(pos["g0_frac"]),
            "threshold": float(MIN_FRACTION),
            "met": bool(pos["g0_frac"] >= MIN_FRACTION),
        },
    ]
    criteria_non_degenerate = {
        "cells_reached_hazard_stage": bool(
            _frac([r.get("reached_hazard_stage", False) for r in per_seed]) >= MIN_FRACTION
        ),
        "positive_control_clears": bool(pos_control_clears),
        "cross_arm_harm_range_dynamic_range": bool(cross_arm_range_spread >= HARM_DISC_RANGE_FLOOR),
        "pag_freeze_present": bool(pos["pag_freeze_frac"] >= MIN_FRACTION),
    }
    criteria = [
        {"name": "positive_control_harm_landscape_discriminative", "load_bearing": True,
         "passed": bool(pos_control_clears)},
        {"name": "actionable_regime_or_training_recommendation_found", "load_bearing": True,
         "passed": bool(any_actionable and pos_control_clears)},
    ]

    return {
        "outcome": outcome,
        "evidence_direction": "non_contributory",  # diagnostic: weights no claim
        "diagnostic_label": label,
        "recommendation": recommendation,
        "positive_control_clears": pos_control_clears,
        "train_rescue": train_rescue,
        "hardest_standard_clearing_proximity_harm": (
            hardest_std_clearing["proximity_harm"] if hardest_std_clearing is not None else None
        ),
        "clearing_arms": [a["arm"] for a in clearing],
        "cross_arm_range_spread": cross_arm_range_spread,
        "harm_disc_range_floor": HARM_DISC_RANGE_FLOOR,
        "arm_results": arm_results,
        "acceptance": {
            "primary_metric": "harm_disc_frac (fraction of seeds with harm_eval_range >= 0.02)",
            "min_fraction": MIN_FRACTION,
            "harm_disc_range_floor": HARM_DISC_RANGE_FLOOR,
            "positive_control_arm": POSITIVE_CONTROL,
            "per_arm_harm_disc_frac": {a["arm"]: a["harm_disc_frac"] for a in arm_results},
            "per_arm_mean_harm_eval_range": {a["arm"]: a["mean_harm_eval_range"] for a in arm_results},
            "per_arm_mean_survival": {a["arm"]: a["mean_hazard_episode_length"] for a in arm_results},
        },
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "grid": {
                "positive_control_fails": "substrate_not_ready_requeue: harm-pathway training/metric broken at easy regime; diagnose harm-pathway optimizer/loss",
                "hard_regime_cell_clears": "regime_difficulty_threshold_located: 603q uses the hardest clearing proximity_harm",
                "train_rescue": "harm_train_strength_rescues_hard_regime: 603q keeps 0.15 + 3x harm-pathway LR",
                "only_eased_clears": "hard_regime_breaks_harm_landscape_use_eased_regime: 603q eases proximity_harm to the hardest clearing cell",
                "nothing_clears": "substrate_not_ready_requeue: escalate to /failure-autopsy on harm-pathway training",
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
        "depends_on": ["V3-EXQ-603k", "V3-EXQ-603l", "V3-EXQ-603o"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding Stage-H base arm (bridge OFF) on the 603k "
                     "harm-pathway-trained + 603j trained-safety-signal substrate",
        "scores": "claim-free diagnostic: locates the base harm-landscape discriminativeness "
                  "parameter (hazard proximity_harm vs harm-pathway LR) for the SD-059/MECH-358 "
                  "bridge-retest re-queue (603q). Weights no claim.",
        "design_note": "diagnose-first for the 603o substrate_not_ready_requeue. 603o's only failing "
                       "readiness gate was harm_landscape_discriminative_on_base (harm_eval_range>=0.02 "
                       "on 1/3 seeds at proximity_harm=0.15); the continuous bridge lift was already "
                       "strong (both ~50.3 vs base ~18.65). 4 cells x 3 seeds, BASE arm only, run "
                       "through Stage-H only: proximity_harm sweep {0.10 (positive control),0.12,0.15} "
                       "at harm_lr=1e-3 + a {0.15, harm_lr=3e-3} training-strength rescue cell. Locates "
                       "whether the base-harm-landscape failure is regime-difficulty (find the hardest "
                       "trainable proximity_harm) or harm-training-strength (3x LR rescue at 0.15). "
                       "Positive-control readiness gate: if even 0.10 fails to clear on >=2/3 the result "
                       "self-routes substrate_not_ready_requeue (harm-pathway/metric broken), never a "
                       "regime verdict.",
        "stage_plan": stage_plan(),
        "pre_registered_gates": {
            "primary_metric": "harm_disc_frac (fraction of seeds with harm_eval_range >= 0.02)",
            "harm_disc_range_floor": HARM_DISC_RANGE_FLOOR,
            "positive_control": POSITIVE_CONTROL + " must clear on >=2/3 (else substrate_not_ready_requeue)",
            "min_fraction": MIN_FRACTION,
            "stage0_z_goal_gate": STAGE0_ZGOAL_GATE,
        },
        "hazard_regime": {
            "num_hazards": HAZARD_STAGE_NUM_HAZARDS,
            "proximity_harm_sweep": [a["proximity_harm"] for a in ARMS],
            "harm_pathway_lr_axis": [HARM_PATHWAY_LR_STANDARD, HARM_PATHWAY_LR_STRONG],
            "hazard_food_attraction": HAZARD_STAGE_HFA,
            "note": "base arm only (bridge OFF); run through Stage-H only (P1/P2 skipped)",
        },
    }
    manifest.update(result)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']} label: {result['diagnostic_label']}", flush=True)
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
