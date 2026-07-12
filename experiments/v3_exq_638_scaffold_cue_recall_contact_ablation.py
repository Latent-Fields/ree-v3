"""
V3-EXQ-638: SD-057 cue-recall bridge -- foraging contact-rate ablation (GAP-2).

Scientific question: does the SD-057 L6 cue-recall bridge (a perceived resource
cue retrieves its incentive token and pulls z_goal toward it -> MECH-295 approach
bias -> first contact) RAISE the agent's ecological foraging CONTACT RATE in the
scaffolded_sd054_onboarding nursery curriculum? This is the "wean-to-wild"
hypothesis: the nursery forced-feed already builds per-object tokens; cue-recall
turns a perceived-but-uncontacted resource into approach, breaking the foraging
contact ceiling that gates goal_pipeline:GAP-2 (and the GAP-7 L9 retest).

SELF-CONTAINED + DECOUPLED FROM the stalled V3-EXQ-634c RUN: both arms set the
landed 634c seeding-calibration knobs (drive_floor=0.9 + benefit_threshold=0.02,
the ARM_3 regime) directly, so wild contact actually seeds z_goal. The ONLY
difference between arms is the cue-recall bridge. So the contact-rate delta
isolates cue-recall's effect.

ARMS (3 seeds 42/43/44 each; identical seeding regime; same curriculum as 634c):
  ARM_OFF      cue-recall bridge OFF (legacy scaffold). Baseline contact rate.
  ARM_CUE_ON   cue-recall bridge ON: agent built with use_incentive_token_bank +
               use_cue_recall + use_resource_encoder; scaffold envs SD-049-enabled;
               cue_recall_wanting fires on the strongest-perceived resource each
               P1/P2 step.

Pre-registered acceptance criteria:
  C1 (cue fires ON):   ARM_CUE_ON P2 n_cue_recall_fires > 0 on >= 2/3 seeds.
  C2 (cue silent OFF):  ARM_OFF P2 n_cue_recall_fires == 0 on ALL seeds (no bank).
  C3 (contact lift):    ARM_CUE_ON P2 contact_rate > ARM_OFF P2 contact_rate
                        (per matched seed) on >= 2/3 seeds -- the load-bearing
                        foraging-contact lever test.
  C4 (no survival regression): ARM_CUE_ON P1 survival pass-rate >= ARM_OFF
                        survival pass-rate (cue-recall must not trade contact for
                        deaths net-negative). Reported; informational guard.
  overall_pass = C1 AND C2 AND C3 (C4 reported, not gating -- survival is a
  separate axis the curriculum owns).

Interpretation grid:
  PASS                 -> cue-recall is a working foraging-contact lever; route to
                          a combined scaffold readiness run (cue-recall + seeding)
                          and then the GAP-7 L9 retest.
  C3 FAIL, C1 PASS     -> cue fires but does not lift contact (approach not the
                          bottleneck, or hazard offsets it) -> /failure-autopsy;
                          re-examine survival axis.
  C1 FAIL              -> cue never fires ecologically (tokens not built / cue not
                          perceived) -> /diagnose-errors on the bridge wiring.
  C2 FAIL              -> OFF arm leaking cue behaviour (flag-gating bug).

claim_ids = []  (substrate-readiness diagnostic; NOT governance evidence).
experiment_purpose = "diagnostic".

References:
- ree-v3/experiments/scaffolded_sd054_onboarding.py (cue-recall bridge amend)
- REE_assembly/docs/architecture/sd_057_object_bound_incentive_salience.md
- V3-EXQ-634c (seeding-calibration regime this reuses), V3-EXQ-636/637 (SD-057 v1+phase2)
- REE_assembly/evidence/planning/goal_pipeline_plan.md (GAP-2 / GAP-7)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from experiments.scaffolded_sd054_onboarding import (
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
)
from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_638_scaffold_cue_recall_contact_ablation"
QUEUE_ID = "V3-EXQ-638"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SEEDS = [42, 43, 44]

WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# Curriculum budgets (mirror 634c).
STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, P1_BUDGET, P2_BUDGET = 20, 10, 120, 70, 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
STAGE0_ZGOAL_GATE = 0.4
STAGE0B_RETENTION_GATE = 0.75
MIN_FRACTION = 2.0 / 3.0
N_RESOURCE_TYPES = 3

# Shared ARM_3 seeding regime (so wild contact seeds in BOTH arms).
SEED_DRIVE_FLOOR = 0.9
SEED_BENEFIT_THRESHOLD = 0.02

ARMS: List[Dict[str, Any]] = [
    {"name": "ARM_OFF", "cue": False},
    {"name": "ARM_CUE_ON", "cue": True},
]


def _seeding_floor() -> float:
    # b* = benefit_threshold / (gain(1.0) * (1 + drive_weight * drive_floor))
    return SEED_BENEFIT_THRESHOLD / (1.0 * (1.0 + DRIVE_WEIGHT * SEED_DRIVE_FLOOR))


def _make_scaffold_cfg(dry_run: bool, cue_on: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, p1, p2, steps = 2, 2, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, P1_BUDGET, P2_BUDGET, TRAIN_STEPS
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
        # Shared ARM_3 seeding regime (both arms).
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_contact_gating_benefit_threshold=_seeding_floor(),
    )
    if cue_on:
        cfg.scaffold_cue_recall_bridge_enabled = True
        cfg.scaffold_cue_n_resource_types = N_RESOURCE_TYPES
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env, cue_on: bool) -> REEConfig:
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
        # SD-057: cue-recall bridge agent flags (ON arm only).
        use_incentive_token_bank=cue_on,
        use_cue_recall=cue_on,
        cue_recall_gain=0.2,
    )
    if cue_on:
        cfg.latent.use_resource_encoder = True  # SD-015 (direct, not via from_dims)
    return cfg


def _run_seed(seed: int, dry_run: bool, arm: Dict[str, Any]) -> Dict[str, Any]:
    arm_name, cue_on = arm["name"], bool(arm["cue"])
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run, cue_on)
    device = torch.device("cpu")

    # Probe env via the scheduler's own builder so the agent dims match the
    # phase envs EXACTLY (ON arm = SD-049-enabled => larger world_obs_dim).
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env, cue_on)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition {arm_name}", flush=True)
    total_eps = (2 + 2 + 5 + 5 + 2) if dry_run else (
        STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + P1_BUDGET + P2_BUDGET)

    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    print(f"  [train] stage0 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"z_goal_peak={s0.z_goal_norm_peak:.4f} formed={s0.z_goal_formed}", flush=True)
    if s0.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=stage0", flush=True)
        return _abort(seed, arm_name, "stage0")

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"retention={s0b.retention_ratio:.3f} gate={'pass' if s0b.retention_gate_passed else 'FAIL'}", flush=True)
    if s0b.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=stage0b", flush=True)
        return _abort(seed, arm_name, "stage0b")

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"mean_len={p0.mean_episode_length:.1f}", flush=True)
    if p0.aborted:
        print(f"verdict: FAIL seed={seed} arm={arm_name} aborted=p0", flush=True)
        return _abort(seed, arm_name, "p0")

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"survival={'pass' if p1.survival_gate_passed else 'FAIL'} "
          f"refresh={p1.n_contact_refresh_updates}", flush=True)

    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    cue_fires = int(getattr(p2, "n_cue_recall_fires", 0))
    print(f"  [train] p2 seed={seed} arm={arm_name} ep {done}/{total_eps} "
          f"contact_rate={p2.contact_rate:.4f} contact_events={p2.num_contact_events} "
          f"cue_fires={cue_fires} z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    print(f"verdict: seed={seed} arm={arm_name} contact_rate={p2.contact_rate:.4f} cue_fires={cue_fires}", flush=True)
    return {
        "seed": seed, "arm": arm_name, "aborted_at": None,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "stage0b_retention_gate_passed": bool(s0b.retention_gate_passed),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_contact_steps": int(p2.contact_steps),
        "p2_num_contact_events": int(p2.num_contact_events),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_n_cue_recall_fires": cue_fires,
    }


def _abort(seed: int, arm_name: str, stage: str) -> Dict[str, Any]:
    return {
        "seed": seed, "arm": arm_name, "aborted_at": stage,
        "stage0_z_goal_norm_peak": 0.0, "stage0b_retention_gate_passed": False,
        "p1_survival_pass": False, "p2_contact_rate": 0.0, "p2_contact_steps": 0,
        "p2_num_contact_events": 0, "p2_z_goal_norm_at_contact_peak": 0.0,
        "p2_n_cue_recall_fires": 0,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for arm in ARMS:
        by_arm[arm["name"]] = [_run_seed(s, dry_run, arm) for s in seeds]

    off = by_arm["ARM_OFF"]
    on = by_arm["ARM_CUE_ON"]

    c1 = _frac([r["p2_n_cue_recall_fires"] > 0 for r in on]) >= MIN_FRACTION
    c2 = all(r["p2_n_cue_recall_fires"] == 0 for r in off)
    # C3: per matched seed, cue-ON contact_rate > OFF contact_rate.
    off_by_seed = {r["seed"]: r["p2_contact_rate"] for r in off}
    lift_flags = [r["p2_contact_rate"] > off_by_seed.get(r["seed"], 0.0) for r in on]
    c3 = _frac(lift_flags) >= MIN_FRACTION
    # C4 (informational): survival not regressed.
    c4 = _frac([r["p1_survival_pass"] for r in on]) >= _frac([r["p1_survival_pass"] for r in off])

    overall = bool(c1 and c2 and c3)
    outcome = "PASS" if overall else "FAIL"
    print(f"[{EXPERIMENT_TYPE}] C1_cue_fires_on={c1} C2_cue_silent_off={c2} "
          f"C3_contact_lift={c3} C4_survival_ok={c4} outcome={outcome}", flush=True)

    return {
        "outcome": outcome,
        "acceptance": {
            "C1_cue_fires_on": c1, "C2_cue_silent_off": c2,
            "C3_contact_lift": c3, "C4_survival_not_regressed": c4,
            "overall_pass": overall,
        },
        "on_contact_rates": [r["p2_contact_rate"] for r in on],
        "off_contact_rates": [r["p2_contact_rate"] for r in off],
        "on_cue_fires": [r["p2_n_cue_recall_fires"] for r in on],
        "per_arm": by_arm,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id, "experiment_type": EXPERIMENT_TYPE, "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS, "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1", "timestamp_utc": ts,
        "outcome": result["outcome"], "evidence_direction": "non_contributory",
        "substrate": "scaffolded_sd054_onboarding (SD-057 cue-recall bridge amend, 2026-06-04)",
        "decoupling_note": (
            "Self-contained: both arms set the landed 634c ARM_3 seeding regime "
            "(drive_floor=0.9 + benefit_threshold=0.02) directly, so wild contact "
            "seeds in both. The ONLY difference is the SD-057 cue-recall bridge -> "
            "the P2 contact-rate delta isolates cue-recall's foraging-contact effect. "
            "Does not depend on the stalled V3-EXQ-634c run, only its landed code."
        ),
        "seeding_regime": {"drive_floor": SEED_DRIVE_FLOOR,
                           "benefit_threshold": SEED_BENEFIT_THRESHOLD,
                           "seeding_floor": _seeding_floor()},
        "pre_registered_gates": {
            "C1_cue_fires_on": "ARM_CUE_ON P2 n_cue_recall_fires > 0 on >= 2/3 seeds",
            "C2_cue_silent_off": "ARM_OFF P2 n_cue_recall_fires == 0 on ALL seeds",
            "C3_contact_lift": "ARM_CUE_ON P2 contact_rate > ARM_OFF per matched seed on >= 2/3",
            "min_fraction": MIN_FRACTION,
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
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    if _res.get("manifest_path"):
        _o = str(_res["outcome"]).upper()
        emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                     manifest_path=_res["manifest_path"])
    sys.exit(0 if _res["outcome"] == "PASS" else 1)
