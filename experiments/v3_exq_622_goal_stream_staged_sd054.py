"""
V3-EXQ-622: staged goal-stream curriculum on SD-054 (successor diagnostic to 621/621a).

Decomposes the V3-EXQ-621 substrate-readiness failure (z_goal=0 everywhere; C1
survival gate failures) into four explicit goal-stream stages. Unlike 621,
this script enables z_goal_enabled=True and calls update_z_goal() every step
(621 omitted both -- likely root cause of zero z_goal measurements).

Stages (sequential per seed; weights carry S0->S3):
  S0  goal-only / minimal hazard
      Rich resources, transient benefit patches, reef refuge spawn,
      drive_floor=0.9 goal feeding. PASS: z_goal_norm_peak >= 0.1.
  S1  mild hazards, hazard_food_attraction=0
      Anneal drive_floor 0.9 -> 0.2. PASS: z_goal persists + survivable episodes.
  S2  hazards + food attraction, slow anneal
      Anneal HFA 0->0.7 + bridge gates. PASS: z_goal + bridge + survival coexist.
  S3  full SD-054 target env (frozen policy eval)
      PASS: survival + (approach_commit OR bridge OR dacc bias) under conflict.

1 condition x 3 seeds = 3 cells. Overall PASS = S0 AND S1 AND S2 AND S3 all pass
(progressive ladder; any stage FAIL localises the bottleneck for 621 follow-up).

claim_ids = [] (diagnostic; prereq (2) goal-pipeline z_goal probe per
z_goal_collapse_triage_2026-05-31.md). experiment_purpose = diagnostic.

Related: V3-EXQ-621 / 621a (scaffolded_sd054_onboarding 4-arm readiness).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

from experiment_protocol import emit_outcome

from experiments.goal_stream_stages_sd054 import (
    GoalStreamStagesConfig,
    GoalStreamStagesRunner,
    StageMetrics,
)

EXPERIMENT_TYPE = "v3_exq_622_goal_stream_staged_sd054"
QUEUE_ID = "V3-EXQ-622"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
SEEDS = [42, 43, 44]
CONDITION = "STAGED_GOAL_STREAM"

# Pre-registered thresholds (mirrored on GoalStreamStagesConfig defaults)
S0_Z_GOAL_PEAK_MIN = 0.1
S1_Z_GOAL_MEDIAN_MIN = 0.05
S1_MEDIAN_EP_LEN_MIN = 40.0
S2_Z_GOAL_MEDIAN_MIN = 0.04
S2_MEDIAN_EP_LEN_MIN = 50.0
S2_BRIDGE_PER_EP_MIN = 0.5
S3_MEDIAN_EP_LEN_MIN = 60.0
S3_APPROACH_COMMIT_MIN = 0.01
S3_BRIDGE_PER_EP_MIN = 1.0
S3_DACC_PER_EP_MIN = 0.5


def build_agent(seed: int, device: torch.device, world_obs_dim: int) -> REEAgent:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = REEConfig.from_dims(
        world_obs_dim=world_obs_dim,
        body_obs_dim=17,
        harm_obs_dim=50,
        action_dim=5,
        z_goal_enabled=True,
        drive_weight=2.0,
        drive_ema_alpha=1.0,
        drive_floor=0.9,  # S0 goal feed; runner adjusts per stage
        use_mech295_liking_bridge=True,
        use_mech307_conjunction=True,
    )
    cfg.latent.alpha_world = 0.9
    agent = REEAgent(cfg).to(device)
    return agent


def make_stage_cfg(
    s0_eps: int,
    s1_eps: int,
    s2_eps: int,
    s3_eps: int,
    steps_per_ep: int,
) -> GoalStreamStagesConfig:
    return GoalStreamStagesConfig(
        steps_per_episode=steps_per_ep,
        s0_episode_budget=s0_eps,
        s1_episode_budget=s1_eps,
        s2_episode_budget=s2_eps,
        s3_episode_budget=s3_eps,
        s0_drive_floor=0.9,
        s0_z_goal_peak_min=S0_Z_GOAL_PEAK_MIN,
        s1_z_goal_median_min=S1_Z_GOAL_MEDIAN_MIN,
        s1_median_ep_len_min=S1_MEDIAN_EP_LEN_MIN,
        s2_z_goal_median_min=S2_Z_GOAL_MEDIAN_MIN,
        s2_median_ep_len_min=S2_MEDIAN_EP_LEN_MIN,
        s2_bridge_per_ep_min=S2_BRIDGE_PER_EP_MIN,
        s3_median_ep_len_min=S3_MEDIAN_EP_LEN_MIN,
        s3_approach_commit_min=S3_APPROACH_COMMIT_MIN,
        s3_bridge_per_ep_min=S3_BRIDGE_PER_EP_MIN,
        s3_dacc_per_ep_min=S3_DACC_PER_EP_MIN,
    )


def total_training_episodes(cfg: GoalStreamStagesConfig) -> int:
    return (
        cfg.s0_episode_budget
        + cfg.s1_episode_budget
        + cfg.s2_episode_budget
        + cfg.s3_episode_budget
    )


def evaluate_overall(stages: List[StageMetrics]) -> Dict[str, Any]:
    by_stage = {s.stage: s for s in stages}
    s0 = by_stage.get("S0")
    s1 = by_stage.get("S1")
    s2 = by_stage.get("S2")
    s3 = by_stage.get("S3")
    overall = bool(
        s0 and s0.stage_pass
        and s1 and s1.stage_pass
        and s2 and s2.stage_pass
        and s3 and s3.stage_pass
    )
    first_fail = ""
    for st in ("S0", "S1", "S2", "S3"):
        m = by_stage.get(st)
        if m is None or not m.stage_pass:
            first_fail = st
            break
    return {
        "overall_pass": overall,
        "S0_pass": bool(s0 and s0.stage_pass),
        "S1_pass": bool(s1 and s1.stage_pass),
        "S2_pass": bool(s2 and s2.stage_pass),
        "S3_pass": bool(s3 and s3.stage_pass),
        "first_failing_stage": first_fail,
    }


def run_seed(
    seed: int,
    device: torch.device,
    world_obs_dim: int,
    cfg: GoalStreamStagesConfig,
    total_eps: int,
) -> Tuple[List[StageMetrics], Dict[str, Any]]:
    print(f"Seed {seed} Condition {CONDITION}")
    agent = build_agent(seed, device, world_obs_dim)
    runner = GoalStreamStagesRunner(cfg)
    ep_done = 0
    stages: List[StageMetrics] = []

    for stage_name, budget in (
        ("S0", cfg.s0_episode_budget),
        ("S1", cfg.s1_episode_budget),
        ("S2", cfg.s2_episode_budget),
        ("S3", cfg.s3_episode_budget),
    ):
        print(f"  [stage] seed={seed} entering {stage_name} budget={budget}")
        if stage_name == "S3":
            sm = runner._run_eval_stage(agent, device, "S3")
        else:
            sm = runner._run_training_stage(agent, device, stage_name)
        stages.append(sm)
        ep_done += sm.n_episodes
        print(
            f"  [train] seed={seed} ep {ep_done}/{total_eps} "
            f"stage={stage_name} pass={sm.stage_pass} "
            f"z_peak={sm.z_goal_norm_peak_max:.4f} "
            f"reason={sm.stage_pass_reason}",
            flush=True,
        )

    acceptance = evaluate_overall(stages)
    passed = acceptance["overall_pass"]
    print(
        f"verdict: {'PASS' if passed else 'FAIL'} seed={seed} "
        f"S0={acceptance['S0_pass']} S1={acceptance['S1_pass']} "
        f"S2={acceptance['S2_pass']} S3={acceptance['S3_pass']}",
        flush=True,
    )
    return stages, acceptance


def emit_manifest(
    all_cells: List[Dict[str, Any]],
    acceptance_summary: Dict[str, Any],
    out_dir: Path,
    dry_run: bool,
) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"v3_exq_622_goal_stream_staged_sd054_{ts}_v3"
    outcome = "PASS" if acceptance_summary["all_seeds_pass"] else "FAIL"
    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Staged goal-stream diagnostic decomposing 621/621a z_goal=0 failure. "
            "Enables z_goal + per-step update_z_goal (absent in 621). "
            "PASS on S0 alone supports prereq (2) trainability; full ladder PASS "
            "suggests 621 failure was measurement/training wiring not scaffold design. "
            "FAIL at S0 -> goal feed insufficient; FAIL at S1+ -> persistence/risk; "
            "FAIL at S3 -> arbitration under full SD-054 conflict."
        ),
        "scoring_excluded": "diagnostic_probe",
        "timestamp_utc": ts,
        "dry_run": dry_run,
        "related_queue_ids": ["V3-EXQ-621", "V3-EXQ-621a"],
        "acceptance": acceptance_summary,
        "cells": all_cells,
        "seeds": SEEDS,
        "condition": CONDITION,
        "thresholds": {
            "S0_z_goal_peak_min": S0_Z_GOAL_PEAK_MIN,
            "S1_z_goal_median_min": S1_Z_GOAL_MEDIAN_MIN,
            "S1_median_ep_len_min": S1_MEDIAN_EP_LEN_MIN,
            "S2_z_goal_median_min": S2_Z_GOAL_MEDIAN_MIN,
            "S2_median_ep_len_min": S2_MEDIAN_EP_LEN_MIN,
            "S2_bridge_per_ep_min": S2_BRIDGE_PER_EP_MIN,
            "S3_median_ep_len_min": S3_MEDIAN_EP_LEN_MIN,
            "S3_approach_commit_min": S3_APPROACH_COMMIT_MIN,
            "S3_bridge_per_ep_min": S3_BRIDGE_PER_EP_MIN,
            "S3_dacc_per_ep_min": S3_DACC_PER_EP_MIN,
        },
        "triage_memo": "REE_assembly/evidence/planning/z_goal_collapse_triage_2026-05-31.md",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"manifest written: {out_path}")
    return out_path


def main(args: argparse.Namespace) -> Tuple[str, str | None]:
    device = torch.device("cpu")
    if args.dry_run:
        s0, s1, s2, s3, steps = 3, 3, 2, 2, 20
    else:
        s0, s1, s2, s3, steps = 40, 30, 30, 30, 200

    cfg = make_stage_cfg(s0, s1, s2, s3, steps)
    total_eps = total_training_episodes(cfg)

    probe = CausalGridWorldV2(
        size=12,
        num_hazards=2,
        num_resources=3,
        reef_enabled=True,
        reef_bipartite_layout=True,
        limb_damage_enabled=True,
        seed=0,
    )
    probe.reset()
    world_obs_dim = probe.world_obs_dim

    all_cells: List[Dict[str, Any]] = []
    n_pass_seeds = 0
    for seed in SEEDS:
        stages, acc = run_seed(seed, device, world_obs_dim, cfg, total_eps)
        if acc["overall_pass"]:
            n_pass_seeds += 1
        all_cells.append(
            {
                "seed": seed,
                "condition": CONDITION,
                "overall_pass": acc["overall_pass"],
                "stages": [asdict(s) for s in stages],
                "acceptance": acc,
            }
        )

    acceptance_summary = {
        "all_seeds_pass": n_pass_seeds == len(SEEDS),
        "n_seeds_pass": n_pass_seeds,
        "n_seeds": len(SEEDS),
        "progressive_ladder": "S0 AND S1 AND S2 AND S3 per seed",
    }

    if args.dry_run:
        outcome = "PASS" if acceptance_summary["all_seeds_pass"] else "FAIL"
        print(f"verdict: dry_run overall={outcome}")
        return outcome, None

    out_dir = Path(args.output_dir)
    out_path = emit_manifest(all_cells, acceptance_summary, out_dir, dry_run=False)
    outcome = "PASS" if acceptance_summary["all_seeds_pass"] else "FAIL"
    print(f"Overall: {outcome} ({n_pass_seeds}/{len(SEEDS)} seeds)")
    return outcome, str(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(
            REPO_ROOT.parent
            / "REE_assembly"
            / "evidence"
            / "experiments"
            / EXPERIMENT_TYPE
        ),
    )
    args = parser.parse_args()
    outcome, manifest_path = main(args)
    if not args.dry_run and manifest_path:
        emit_outcome(outcome=outcome, manifest_path=manifest_path)
    sys.exit(0 if outcome == "PASS" else 1)
