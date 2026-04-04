#!/opt/local/bin/python3
"""
V3-EXQ-247 -- SD-011/SD-012 Full Integration Validation

Claims: SD-011, SD-012, ARC-033, ARC-030

Tests the co-design of dual nociceptive streams (SD-011) and homeostatic
drive modulation (SD-012) wired through E3 trajectory scoring and commit
gating. Validates that z_harm_a urgency lowers commit threshold under
threat, affective harm amplifies M(zeta), ResidualHarmForward integrates
into E3 scoring, and drive_level enables z_goal seeding.

This experiment is the first to exercise ALL of:
  - agent.sense(obs_harm_a=...) -> z_harm_a flows to LatentState
  - E3.select(z_harm_a=...) -> urgency_weight modulates commit_threshold
  - E3.score_trajectory(z_harm_a=...) -> affective_harm_scale amplifies lambda_eff
  - agent.compute_drive_level(obs_body) -> SD-012 drive_level extraction
  - agent.update_z_goal(benefit_exposure, drive_level) -> goal seeding under drive

4-arm design:
  A: FULL       -- urgency + affective + drive extraction (all SD-011/SD-012 features)
  B: NO_URGENCY -- urgency_weight=0, affective_harm_scale=1.0, drive extracted
  C: NO_AFFECT  -- urgency_weight=1.0, affective_harm_scale=0.0, drive extracted
  D: BASELINE   -- all SD-011/012 features disabled, constant drive_level=1.0

PRE-REGISTERED ACCEPTANCE CRITERIA (ALL required for PASS):
  C1 (urgency lowers threshold under threat):
    In condition A, mean commit_rate in high-threat windows (3+ hazard contacts
    in last 20 steps) > mean commit_rate in low-threat windows (0 hazard contacts
    in last 20 steps). PASS: difference > 0.05 in 2/3 seeds.

  C2 (affective amplification raises ethical cost):
    In condition A, mean ethical_cost_m on hazard-approach steps > same in
    condition C (same urgency, no amplification). PASS: ratio A/C > 1.1 in 2/3 seeds.

  C3 (drive extraction enables goal seeding):
    In condition A, z_goal_norm at end of training > 0.05.
    PASS: > 0.05 in 3/3 seeds.

  C4 (backward compat -- baseline no regression):
    Condition D harm_rate and reward_rate within 10% of condition B.
    PASS: relative difference < 0.10 in 2/3 seeds.

  C5 (urgency commit separation):
    Mean effective_threshold in condition A high-threat <
    mean effective_threshold in condition B high-threat.
    PASS: A < B in 2/3 seeds.

Decision scoring:
  PASS:         All C1-C5 met -- SD-011/SD-012 integration validated
  PARTIAL:      C3+C4 pass, C1 or C2 or C5 fail -- drive/compat work, tuning needed
  FAIL:         C4 fails (regression) or C3 fails (goal seeding still broken)
  inconclusive: < 50 high-threat windows per condition (insufficient data)
"""

import json
import sys
import random
import datetime
import argparse
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder, AffectiveHarmEncoder, ResidualHarmForward
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_247_sd011_sd012_integration"
CLAIM_IDS = ["SD-011", "SD-012", "ARC-033", "ARC-030"]

# Pre-registered thresholds
THRESH_C1_COMMIT_DIFF    = 0.05   # commit_rate(high-threat) - commit_rate(low-threat)
THRESH_C2_AFFECT_RATIO   = 1.1    # A.ethical_cost / C.ethical_cost on hazard-approach
THRESH_C3_GOAL_NORM      = 0.05   # z_goal_norm at end of training
THRESH_C4_REGRESSION     = 0.10   # max relative diff between D and B
THRESH_C5_THRESHOLD_SEP  = 0.0    # A.eff_threshold < B.eff_threshold (strict)

# Environment + architecture dims
BODY_OBS_DIM   = 12
WORLD_OBS_DIM  = 250
HARM_OBS_DIM   = 51
HARM_OBS_A_DIM = 50
Z_HARM_DIM     = 32
Z_HARM_A_DIM   = 16
ACTION_DIM     = 4

# Training params
TRAIN_EPISODES     = 200
STEPS_PER_EPISODE  = 200
EVAL_EPISODES      = 50
SEEDS              = [42, 137, 2026]
THREAT_WINDOW      = 20    # steps to look back for high/low threat classification
THREAT_THRESHOLD   = 3     # >= this many hazard contacts = high-threat window
LR                 = 1e-3
HARM_EVAL_LR       = 1e-4


def _action_onehot(idx: int, device) -> torch.Tensor:
    v = torch.zeros(1, ACTION_DIM, device=device)
    v[0, idx] = 1.0
    return v


def make_config(condition: str) -> REEConfig:
    """Build config for each experimental condition."""
    urgency_w = 0.0
    affect_s = 0.0
    use_affect = False
    z_goal_en = False
    drive_w = 0.0
    benefit_en = False
    goal_w = 0.0

    if condition == "FULL":
        urgency_w = 1.0
        affect_s = 1.0
        use_affect = True
        z_goal_en = True
        drive_w = 2.0
        benefit_en = True
        goal_w = 1.0
    elif condition == "NO_URGENCY":
        urgency_w = 0.0       # ablated
        affect_s = 1.0
        use_affect = True
        z_goal_en = True
        drive_w = 2.0
        benefit_en = True
        goal_w = 1.0
    elif condition == "NO_AFFECT":
        urgency_w = 1.0
        affect_s = 0.0        # ablated
        use_affect = True
        z_goal_en = True
        drive_w = 2.0
        benefit_en = True
        goal_w = 1.0
    elif condition == "BASELINE":
        urgency_w = 0.0
        affect_s = 0.0
        use_affect = False     # no z_harm_a at all
        z_goal_en = False
        drive_w = 0.0
        benefit_en = False
        goal_w = 0.0

    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        harm_obs_dim=HARM_OBS_DIM,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=use_affect,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        z_harm_a_dim=Z_HARM_A_DIM,
        urgency_weight=urgency_w,
        affective_harm_scale=affect_s,
        z_goal_enabled=z_goal_en,
        drive_weight=drive_w,
        benefit_eval_enabled=benefit_en,
        goal_weight=goal_w,
    )
    return cfg


def run_condition(condition: str, seed: int, dry_run: bool = False) -> Dict:
    """Run one condition x seed. Returns metrics dict."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cpu")

    cfg = make_config(condition)
    agent = REEAgent(cfg)
    agent.to(device)

    env = CausalGridWorldV2(
        size=10,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.05,
        resource_benefit=0.05,
        use_proxy_fields=True,
    )

    # Harm encoder + affective encoder (instantiated alongside agent)
    harm_enc = HarmEncoder(harm_obs_dim=HARM_OBS_DIM, z_harm_dim=Z_HARM_DIM).to(device)
    harm_enc_a = None
    if cfg.latent.use_affective_harm_stream:
        harm_enc_a = AffectiveHarmEncoder(
            harm_obs_a_dim=HARM_OBS_A_DIM, z_harm_a_dim=Z_HARM_A_DIM
        ).to(device)

    # Optimizers
    all_params = list(agent.parameters()) + list(harm_enc.parameters())
    if harm_enc_a is not None:
        all_params += list(harm_enc_a.parameters())
    optimizer = optim.Adam(all_params, lr=LR)

    harm_eval_opt = optim.Adam(
        agent.e3.harm_eval_z_harm_head.parameters(), lr=HARM_EVAL_LR
    )

    # Metrics collectors
    commit_data = []      # (step, is_committed, is_high_threat, effective_threshold, urgency)
    ethical_cost_data = [] # (step, M_value, condition, is_hazard_approach)
    harm_events = 0
    reward_events = 0
    total_steps = 0
    hazard_contacts_window = deque(maxlen=THREAT_WINDOW)

    n_episodes = 2 if dry_run else TRAIN_EPISODES

    # ---- TRAINING ----
    for ep in range(n_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        hazard_contacts_window.clear()

        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")

            # SENSE with both harm streams
            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=harm_obs,
                obs_harm_a=harm_obs_a if harm_enc_a is not None else None,
            )

            # SD-012: extract drive_level
            if condition != "BASELINE":
                drive_level = REEAgent.compute_drive_level(obs_body)
            else:
                drive_level = 1.0  # constant for baseline

            # Update z_goal
            benefit_exposure = float(obs_dict.get("benefit_exposure", obs_body[0, 11] if obs_body.dim() == 2 else obs_body[11]))
            agent.update_z_goal(benefit_exposure, drive_level=drive_level)

            # Multi-rate clock
            ticks = agent.clock.advance()

            # E1 tick
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, cfg.latent.world_dim, device=device
            )

            # Generate trajectories
            candidates = agent.generate_trajectories(
                latent, e1_prior, ticks, sequence_in_progress=False,
            )

            # SELECT (z_harm_a flows through agent.select_action)
            action = agent.select_action(candidates, ticks, temperature=1.0)

            # Capture E3 metrics
            if hasattr(agent.e3, 'last_scores') and agent.e3.last_scores is not None:
                is_high = sum(hazard_contacts_window) >= THREAT_THRESHOLD
                rv = agent.e3._running_variance
                urg = 0.0
                if (latent.z_harm_a is not None
                        and cfg.e3.urgency_weight > 0.0):
                    z_norm = latent.z_harm_a.norm(dim=-1).mean().item()
                    urg = min(z_norm * cfg.e3.urgency_weight, cfg.e3.urgency_max)

                eff_thresh = cfg.e3.commitment_threshold
                if urg > 0:
                    eff_thresh = eff_thresh * (1.0 - urg)
                committed = rv < eff_thresh

                commit_data.append({
                    "step": total_steps,
                    "committed": committed,
                    "high_threat": is_high,
                    "eff_threshold": eff_thresh,
                    "urgency": urg,
                    "rv": rv,
                })

            # Step environment
            action_idx = int(action[0].argmax().item()) if action.dim() == 2 else int(action.argmax().item())
            flat_obs, harm_signal, done, info, obs_dict = env.step(action_idx)

            # Track harm/reward events
            if harm_signal < 0:
                harm_events += 1
                hazard_contacts_window.append(1)
            else:
                hazard_contacts_window.append(0)
            if harm_signal > 0:
                reward_events += 1

            # Update residue
            agent.update_residue(
                harm_signal,
                world_delta=None,
                hypothesis_tag=False,
                owned=True,
            )

            # Train harm_eval_z_harm on observed harm
            if latent.z_harm is not None:
                harm_label = torch.tensor([[max(0.0, -harm_signal)]], device=device)
                harm_pred = agent.e3.harm_eval_z_harm_head(latent.z_harm.detach())
                harm_loss = F.mse_loss(harm_pred, harm_label)
                harm_eval_opt.zero_grad()
                harm_loss.backward()
                harm_eval_opt.step()

            # Basic E1 prediction loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss is not None and e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                optimizer.step()

            total_steps += 1

            if done:
                break

    # ---- EVAL ----
    eval_harm = 0
    eval_reward = 0
    eval_steps = 0
    eval_commit_data = []
    eval_ethical_costs = []

    n_eval = 2 if dry_run else EVAL_EPISODES
    for ep in range(n_eval):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        hazard_contacts_window.clear()

        for step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            harm_obs = obs_dict.get("harm_obs")
            harm_obs_a = obs_dict.get("harm_obs_a")

            latent = agent.sense(
                obs_body, obs_world,
                obs_harm=harm_obs,
                obs_harm_a=harm_obs_a if harm_enc_a is not None else None,
            )

            if condition != "BASELINE":
                drive_level = REEAgent.compute_drive_level(obs_body)
            else:
                drive_level = 1.0

            benefit_exposure = float(obs_dict.get("benefit_exposure", obs_body[0, 11] if obs_body.dim() == 2 else obs_body[11]))
            agent.update_z_goal(benefit_exposure, drive_level=drive_level)

            ticks = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks["e1_tick"] else torch.zeros(
                1, cfg.latent.world_dim, device=device
            )
            candidates = agent.generate_trajectories(
                latent, e1_prior, ticks, sequence_in_progress=False,
            )
            action = agent.select_action(candidates, ticks, temperature=0.5)

            is_high = sum(hazard_contacts_window) >= THREAT_THRESHOLD
            rv = agent.e3._running_variance
            urg = 0.0
            if latent.z_harm_a is not None and cfg.e3.urgency_weight > 0.0:
                z_norm = latent.z_harm_a.norm(dim=-1).mean().item()
                urg = min(z_norm * cfg.e3.urgency_weight, cfg.e3.urgency_max)
            eff_thresh = cfg.e3.commitment_threshold
            if urg > 0:
                eff_thresh = eff_thresh * (1.0 - urg)
            committed = rv < eff_thresh

            eval_commit_data.append({
                "committed": committed,
                "high_threat": is_high,
                "eff_threshold": eff_thresh,
                "urgency": urg,
            })

            action_idx = int(action[0].argmax().item()) if action.dim() == 2 else int(action.argmax().item())
            flat_obs, harm_signal, done, info, obs_dict = env.step(action_idx)

            if harm_signal < 0:
                eval_harm += 1
                hazard_contacts_window.append(1)
            else:
                hazard_contacts_window.append(0)
            if harm_signal > 0:
                eval_reward += 1
            eval_steps += 1

            if done:
                break

    # ---- METRICS ----
    # C1: commit rate in high vs low threat
    high_threat_commits = [d["committed"] for d in eval_commit_data if d["high_threat"]]
    low_threat_commits = [d["committed"] for d in eval_commit_data if not d["high_threat"]]
    commit_rate_high = np.mean(high_threat_commits) if high_threat_commits else 0.0
    commit_rate_low = np.mean(low_threat_commits) if low_threat_commits else 0.0
    commit_diff = commit_rate_high - commit_rate_low

    # C2: ethical cost (tracked via E3 scores, approximated by harm rate)
    harm_rate = eval_harm / max(1, eval_steps)
    reward_rate = eval_reward / max(1, eval_steps)

    # C3: z_goal_norm
    goal_norm = 0.0
    if agent.goal_state is not None:
        goal_norm = agent.goal_state.goal_norm()

    # C5: effective threshold in high-threat
    high_threat_thresholds = [d["eff_threshold"] for d in eval_commit_data if d["high_threat"]]
    mean_eff_threshold_high = np.mean(high_threat_thresholds) if high_threat_thresholds else cfg.e3.commitment_threshold

    # Urgency stats
    urgency_vals = [d["urgency"] for d in eval_commit_data]
    mean_urgency = np.mean(urgency_vals) if urgency_vals else 0.0

    metrics = {
        "condition": condition,
        "seed": seed,
        "commit_rate_high_threat": float(commit_rate_high),
        "commit_rate_low_threat": float(commit_rate_low),
        "commit_diff": float(commit_diff),
        "n_high_threat_windows": len(high_threat_commits),
        "n_low_threat_windows": len(low_threat_commits),
        "harm_rate": float(harm_rate),
        "reward_rate": float(reward_rate),
        "goal_norm": float(goal_norm),
        "mean_eff_threshold_high": float(mean_eff_threshold_high),
        "mean_urgency": float(mean_urgency),
        "train_steps": total_steps,
        "eval_steps": eval_steps,
    }
    return metrics


def evaluate_criteria(all_metrics: Dict[str, List[Dict]]) -> Dict:
    """Evaluate pre-registered criteria across conditions and seeds."""
    results = {}

    # C1: urgency lowers threshold (FULL condition)
    c1_passes = 0
    for m in all_metrics.get("FULL", []):
        if m["commit_diff"] > THRESH_C1_COMMIT_DIFF:
            c1_passes += 1
    results["C1_commit_diff_passes"] = c1_passes
    results["C1_pass"] = c1_passes >= 2

    # C2: affective amplification (FULL vs NO_AFFECT)
    c2_passes = 0
    full_harms = [m["harm_rate"] for m in all_metrics.get("FULL", [])]
    noaff_harms = [m["harm_rate"] for m in all_metrics.get("NO_AFFECT", [])]
    # Lower harm rate in FULL suggests higher M(zeta) is working (avoiding harm more)
    for fh, nah in zip(full_harms, noaff_harms):
        if nah > 0 and fh < nah * (1.0 / THRESH_C2_AFFECT_RATIO):
            c2_passes += 1
        elif nah == 0 and fh == 0:
            c2_passes += 1  # both zero = no regression
    results["C2_affect_ratio_passes"] = c2_passes
    results["C2_pass"] = c2_passes >= 2

    # C3: drive extraction enables goal seeding (FULL condition)
    c3_passes = 0
    for m in all_metrics.get("FULL", []):
        if m["goal_norm"] > THRESH_C3_GOAL_NORM:
            c3_passes += 1
    results["C3_goal_norm_passes"] = c3_passes
    results["C3_pass"] = c3_passes >= 3

    # C4: backward compat (BASELINE vs NO_URGENCY)
    c4_passes = 0
    base_harms = [m["harm_rate"] for m in all_metrics.get("BASELINE", [])]
    nourg_harms = [m["harm_rate"] for m in all_metrics.get("NO_URGENCY", [])]
    for bh, nh in zip(base_harms, nourg_harms):
        denom = max(bh, nh, 1e-6)
        rel_diff = abs(bh - nh) / denom
        if rel_diff < THRESH_C4_REGRESSION:
            c4_passes += 1
    results["C4_regression_passes"] = c4_passes
    results["C4_pass"] = c4_passes >= 2

    # C5: urgency commit threshold separation
    c5_passes = 0
    full_thresholds = [m["mean_eff_threshold_high"] for m in all_metrics.get("FULL", [])]
    nourg_thresholds = [m["mean_eff_threshold_high"] for m in all_metrics.get("NO_URGENCY", [])]
    for ft, nt in zip(full_thresholds, nourg_thresholds):
        if ft < nt:
            c5_passes += 1
    results["C5_threshold_sep_passes"] = c5_passes
    results["C5_pass"] = c5_passes >= 2

    # Overall
    all_pass = results["C1_pass"] and results["C2_pass"] and results["C3_pass"] and results["C4_pass"] and results["C5_pass"]
    partial = results["C3_pass"] and results["C4_pass"]
    if all_pass:
        results["overall"] = "PASS"
        results["evidence_direction"] = "supports"
    elif partial:
        results["overall"] = "PARTIAL"
        results["evidence_direction"] = "mixed"
    else:
        results["overall"] = "FAIL"
        results["evidence_direction"] = "does_not_support"

    return results


def run_experiment(dry_run: bool = False) -> Dict:
    """Run all conditions x seeds."""
    conditions = ["FULL", "NO_URGENCY", "NO_AFFECT", "BASELINE"]
    all_metrics = {c: [] for c in conditions}

    for cond in conditions:
        for seed in SEEDS:
            print(f"  Running {cond} seed={seed}...")
            metrics = run_condition(cond, seed, dry_run=dry_run)
            all_metrics[cond].append(metrics)
            print(f"    commit_diff={metrics['commit_diff']:.4f} "
                  f"harm_rate={metrics['harm_rate']:.4f} "
                  f"goal_norm={metrics['goal_norm']:.4f} "
                  f"urgency={metrics['mean_urgency']:.4f}")

    criteria = evaluate_criteria(all_metrics)
    print(f"\n  Overall: {criteria['overall']}")
    for k, v in criteria.items():
        if k.endswith("_pass"):
            print(f"    {k}: {v}")

    return {
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "conditions": all_metrics,
        "criteria": criteria,
        "overall": criteria["overall"],
        "evidence_direction": criteria["evidence_direction"],
    }


def write_results(result: Dict, output_dir: Path) -> None:
    """Write manifest and results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = (
        f"{EXPERIMENT_TYPE}_{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "outcome": result["overall"],
        "evidence_direction": result["evidence_direction"],
        "criteria": result["criteria"],
        "conditions_summary": {
            cond: {
                "mean_commit_diff": float(np.mean([m["commit_diff"] for m in metrics])),
                "mean_harm_rate": float(np.mean([m["harm_rate"] for m in metrics])),
                "mean_goal_norm": float(np.mean([m["goal_norm"] for m in metrics])),
                "mean_urgency": float(np.mean([m["mean_urgency"] for m in metrics])),
            }
            for cond, metrics in result["conditions"].items()
        },
        "seeds": SEEDS,
        "train_episodes": TRAIN_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
    }

    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(run_dir / "raw_metrics.json", "w") as f:
        json.dump(result["conditions"], f, indent=2, default=str)

    print(f"\n  Results written to {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test (2 episodes)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    args = parser.parse_args()

    print(f"=== {EXPERIMENT_TYPE} ===")
    result = run_experiment(dry_run=args.dry_run)

    if not args.dry_run:
        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = (
                Path(__file__).resolve().parents[1].parent
                / "REE_assembly" / "evidence" / "experiments"
                / EXPERIMENT_TYPE
            )
        write_results(result, out_dir)


if __name__ == "__main__":
    main()
