#!/opt/local/bin/python3
"""V3-EXQ-530: ARC-016 Precision-to-Commitment Circuit

Tests the claim: E3-derived precision (1/running_variance) modulates commit threshold.

Two arms, both reef-enabled:
  ARM_0: use_dacc=False -- no precision-commit circuit
  ARM_1: use_dacc=True  -- dACC uses e3.current_precision in score_bias computation

Protocol:
  Training: random policy, compute_prediction_loss(), Adam lr=5e-4
  Eval: record (precision, beta_elevated) pairs at each step

Metrics (eval phase, per arm per seed):
  commit_rate:              fraction of steps where beta_elevated=True
  high_precision_commit_rate: commit_rate when precision > median
  low_precision_commit_rate:  commit_rate when precision <= median
  precision_commit_ratio:     high / (low + 1e-8)
  mean_precision:             average e3.current_precision

Acceptance criteria:
  C1: ARM_1 precision_commit_ratio > ARM_0 precision_commit_ratio
  C2: ARM_1 precision_commit_ratio > 1.0
  C3: ARM_1 commit_rate > 0.01 (non-degenerate)
  Overall PASS = C1 AND C2 AND C3

claim_ids: ["ARC-016"]
experiment_purpose: evidence
queue_id: V3-EXQ-530
"""

import os
import sys
import json
import time
import argparse
import statistics
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.utils.config import REEConfig
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from pathlib import Path  # noqa: E402

QUEUE_ID = "V3-EXQ-530"
EXPERIMENT_TYPE = "v3_exq_530_arc016_precision_commit"
CLAIM_IDS = ["ARC-016"]

# Full run parameters
N_TRAIN_EPS = 30
N_EVAL_EPS = 20
N_SEEDS = 3
GRID_SIZE = 12
STEPS_PER_EP = 200
LR = 5e-4

# Reef environment parameters
REEF_KWARGS = dict(
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
)

ARMS = [
    {"arm_id": 0, "use_dacc": False, "label": "ARM_0_no_dacc"},
    {"arm_id": 1, "use_dacc": True,  "label": "ARM_1_dacc"},
]


def _make_env(seed, dry_run=False):
    size = GRID_SIZE if not dry_run else 8
    return CausalGridWorldV2(
        seed=seed,
        size=size,
        num_hazards=3,
        num_resources=3,
        hazard_harm=0.04,
        proximity_harm_scale=0.12,
        **REEF_KWARGS,
    )


def _make_agent(env, use_dacc, dry_run=False):
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_dacc=use_dacc,
        dacc_weight=0.5 if use_dacc else 0.0,
        dacc_precision_scale=500.0,
    )
    # no dry-run config changes -- keep full CEM budget to avoid degenerate scores
    return REEAgent(cfg)


def _obs_tensors(obs_dict):
    body = obs_dict["body_state"].float().unsqueeze(0)
    world = obs_dict["world_state"].float().unsqueeze(0)
    return body, world


def _run_arm_seed(arm_cfg, seed, n_train_eps, n_eval_eps, steps_per_ep, verbose=False, dry_run=False):
    """Run one arm for one seed. Returns dict of metrics."""
    use_dacc = arm_cfg["use_dacc"]
    arm_id = arm_cfg["arm_id"]

    torch.manual_seed(seed)
    env = _make_env(seed, dry_run=dry_run)
    agent = _make_agent(env, use_dacc, dry_run=dry_run)
    optimizer = optim.Adam(agent.parameters(), lr=LR)

    # --------------------------------------------------------
    # Training phase: random policy + prediction loss
    # --------------------------------------------------------
    for ep in range(n_train_eps):
        agent.reset()
        _, obs_dict = env.reset()

        for _step in range(steps_per_ep):
            body, world = _obs_tensors(obs_dict)

            agent.sense(obs_body=body, obs_world=world)
            ticks = agent.clock.advance()

            world_dim = agent.config.latent.world_dim
            e1_prior = (
                agent._e1_tick(agent._current_latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(agent._current_latent, e1_prior, ticks)
            action_tensor = agent.select_action(candidates, ticks)

            action_int = int(action_tensor.argmax(dim=-1).item())
            _, _harm, done, _info, obs_dict = env.step(action_int)

            # Training step
            optimizer.zero_grad()
            loss = agent.compute_prediction_loss()
            if loss.requires_grad:
                loss.backward()
                optimizer.step()

            if done:
                break

    if verbose:
        print(
            f"  arm={arm_id} seed={seed} training complete "
            f"precision={agent.e3.current_precision:.4f}",
            flush=True,
        )

    # --------------------------------------------------------
    # Eval phase: collect (precision, beta_elevated) per step
    # --------------------------------------------------------
    precision_vals = []
    beta_vals = []

    for _ep in range(n_eval_eps):
        agent.reset()
        _, obs_dict = env.reset()

        for _step in range(steps_per_ep):
            body, world = _obs_tensors(obs_dict)

            agent.sense(obs_body=body, obs_world=world)
            ticks = agent.clock.advance()

            world_dim = agent.config.latent.world_dim
            e1_prior = (
                agent._e1_tick(agent._current_latent) if ticks["e1_tick"]
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(agent._current_latent, e1_prior, ticks)
            action_tensor = agent.select_action(candidates, ticks)

            # Record at each step
            prec = float(agent.e3.current_precision)
            beta = bool(agent.beta_gate.is_elevated)
            precision_vals.append(prec)
            beta_vals.append(beta)

            action_int = int(action_tensor.argmax(dim=-1).item())
            _, _harm, done, _info, obs_dict = env.step(action_int)

            if done:
                break

    # --------------------------------------------------------
    # Compute metrics
    # --------------------------------------------------------
    n_steps = len(precision_vals)
    if n_steps == 0:
        return {
            "arm_id": arm_id, "seed": seed,
            "commit_rate": 0.0,
            "high_precision_commit_rate": 0.0,
            "low_precision_commit_rate": 0.0,
            "precision_commit_ratio": 1.0,
            "mean_precision": 0.0,
            "n_steps": 0,
        }

    commit_rate = sum(1 for b in beta_vals if b) / n_steps
    mean_precision = sum(precision_vals) / n_steps

    # Split at median precision
    sorted_prec = sorted(precision_vals)
    median_prec = sorted_prec[n_steps // 2]

    high_steps = [(p, b) for p, b in zip(precision_vals, beta_vals) if p > median_prec]
    low_steps  = [(p, b) for p, b in zip(precision_vals, beta_vals) if p <= median_prec]

    high_commit = (sum(1 for _, b in high_steps if b) / len(high_steps)) if high_steps else 0.0
    low_commit  = (sum(1 for _, b in low_steps  if b) / len(low_steps))  if low_steps  else 0.0

    precision_commit_ratio = high_commit / (low_commit + 1e-8)

    metrics = {
        "arm_id": arm_id,
        "seed": seed,
        "commit_rate": float(commit_rate),
        "high_precision_commit_rate": float(high_commit),
        "low_precision_commit_rate": float(low_commit),
        "precision_commit_ratio": float(precision_commit_ratio),
        "mean_precision": float(mean_precision),
        "median_precision": float(median_prec),
        "n_steps": n_steps,
        "n_high_steps": len(high_steps),
        "n_low_steps": len(low_steps),
    }

    if verbose:
        print(
            f"  arm={arm_id} seed={seed} "
            f"commit_rate={commit_rate:.3f} "
            f"hi_commit={high_commit:.3f} lo_commit={low_commit:.3f} "
            f"ratio={precision_commit_ratio:.3f} "
            f"mean_prec={mean_precision:.4f}",
            flush=True,
        )

    return metrics


def run_experiment(dry_run=False):
    """Run the full experiment. Returns dict with metrics and overall_pass."""
    if dry_run:
        n_train = 1
        n_eval = 1
        n_seeds = 1
        steps = 20
    else:
        n_train = N_TRAIN_EPS
        n_eval = N_EVAL_EPS
        n_seeds = N_SEEDS
        steps = STEPS_PER_EP

    rng = np.random.default_rng(42)
    seeds = [int(rng.integers(1000, 9999)) for _ in range(n_seeds)]

    all_results = []
    t_start = time.time()

    for arm_cfg in ARMS:
        arm_id = arm_cfg["arm_id"]
        print(f"Running arm {arm_id} ({arm_cfg['label']}) ...", flush=True)
        for si, seed in enumerate(seeds):
            print(f"  seed {si+1}/{n_seeds} (seed={seed})", flush=True)
            res = _run_arm_seed(
                arm_cfg, seed, n_train, n_eval, steps, verbose=dry_run,
                dry_run=dry_run,
            )
            all_results.append(res)

    elapsed = time.time() - t_start
    print(f"Total elapsed: {elapsed:.1f}s", flush=True)

    # Aggregate by arm
    arm0 = [r for r in all_results if r["arm_id"] == 0]
    arm1 = [r for r in all_results if r["arm_id"] == 1]

    def mean_metric(results, key):
        vals = [r[key] for r in results]
        return float(sum(vals) / len(vals)) if vals else 0.0

    arm0_ratio   = mean_metric(arm0, "precision_commit_ratio")
    arm1_ratio   = mean_metric(arm1, "precision_commit_ratio")
    arm1_commit  = mean_metric(arm1, "commit_rate")
    arm0_commit  = mean_metric(arm0, "commit_rate")

    # Acceptance criteria
    c1 = arm1_ratio > arm0_ratio
    c2 = arm1_ratio > 1.0
    c3 = arm1_commit > 0.01

    overall_pass = c1 and c2 and c3
    outcome = "PASS" if overall_pass else "FAIL"

    print(f"\nResults:", flush=True)
    print(f"  ARM_0 precision_commit_ratio: {arm0_ratio:.4f}", flush=True)
    print(f"  ARM_1 precision_commit_ratio: {arm1_ratio:.4f}", flush=True)
    print(f"  ARM_0 commit_rate: {arm0_commit:.4f}", flush=True)
    print(f"  ARM_1 commit_rate: {arm1_commit:.4f}", flush=True)
    print(f"  C1 (arm1_ratio > arm0_ratio): {c1}", flush=True)
    print(f"  C2 (arm1_ratio > 1.0): {c2}", flush=True)
    print(f"  C3 (arm1_commit > 0.01): {c3}", flush=True)
    print(f"  Outcome: {outcome}", flush=True)

    metrics = {
        "arm0_mean_precision_commit_ratio": arm0_ratio,
        "arm1_mean_precision_commit_ratio": arm1_ratio,
        "arm0_mean_commit_rate": arm0_commit,
        "arm1_mean_commit_rate": arm1_commit,
        "arm0_mean_precision": mean_metric(arm0, "mean_precision"),
        "arm1_mean_precision": mean_metric(arm1, "mean_precision"),
        "c1_arm1_ratio_gt_arm0": c1,
        "c2_arm1_ratio_gt_1": c2,
        "c3_arm1_commit_positive": c3,
        "per_seed_results": all_results,
        "elapsed_seconds": float(elapsed),
        "n_seeds": n_seeds,
        "n_train_eps": n_train,
        "n_eval_eps": n_eval,
    }

    return {"metrics": metrics, "overall_pass": overall_pass, "outcome": outcome}


def write_result(result, run_id):
    """Write experiment manifest to REE_assembly/evidence/experiments/."""
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    outcome = result["outcome"]
    metrics = result["metrics"]

    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if outcome == "PASS" else "weakens",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": ts,
        "outcome": outcome,
        "metrics": metrics,
        "config": {
            "n_train_eps": metrics["n_train_eps"],
            "n_eval_eps": metrics["n_eval_eps"],
            "n_seeds": metrics["n_seeds"],
            "grid_size": GRID_SIZE,
            "steps_per_ep": STEPS_PER_EP,
            "lr": LR,
            "arms": ARMS,
            "reef_kwargs": REEF_KWARGS,
        },
    }

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(
        script_dir, "..", "..", "REE_assembly", "evidence", "experiments",
        EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=None,
        script_path=Path(__file__),
    )
    print(f"Result written to: {out_path}", flush=True)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test with minimal episodes")
    args = parser.parse_args()

    dry_run = args.dry_run
    if dry_run:
        print("Dry run mode: n_train=1, n_eval=1, n_seeds=1, steps=20", flush=True)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    result = run_experiment(dry_run=dry_run)

    if not dry_run:
        write_result(result, run_id)
    else:
        print("Dry run complete. Outcome:", result["outcome"], flush=True)
        print("Metrics:", json.dumps(result["metrics"], indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
